#!/usr/bin/env python

"""GUI recorder for bimanual SO101 teleoperation with RGB and RealSense depth cameras.

Run from the LeRobot conda environment:
    conda run -n lerobot python mycode/gui_record_so101_bimanual.py
"""

from __future__ import annotations

import queue
import json
import shutil
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

try:
    from mycode.gui_view_lerobot_dataset import read_nested_parquets, rewrite_dataset, video_keys_from_info
except ModuleNotFoundError:
    from gui_view_lerobot_dataset import read_nested_parquets, rewrite_dataset, video_keys_from_info

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.utils import init_logging


FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PREVIEW_SIZE = (1280, 720)
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "calibration"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "datanew"
TASK_CHOICES = ("cube1", "cube2", "cube3")


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    valid = depth[depth > 0]
    if valid.size == 0:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        near = np.percentile(valid, 2)
        far = np.percentile(valid, 98)
        if far <= near:
            far = near + 1
        normalized = np.clip((depth - near) * 255.0 / (far - near), 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


class RealSenseDepthView:
    """A camera-like view that exposes colorized depth from an existing RealSense RGB camera."""

    def __init__(self, parent: RealSenseCamera | FlexibleRealSenseCamera) -> None:
        self.parent = parent

    @property
    def is_connected(self) -> bool:
        return self.parent.is_connected

    def connect(self) -> None:
        if not self.parent.is_connected:
            self.parent.connect()

    def disconnect(self) -> None:
        pass

    def async_read(self, timeout_ms: int = 200) -> np.ndarray:
        del timeout_ms
        with self.parent.frame_lock:
            depth = getattr(self.parent, "latest_depth_preview_frame", None)
            if depth is None:
                depth = self.parent.latest_depth_frame
        if depth is None:
            depth = self.parent.read_depth()
        return depth_to_rgb(depth)


class FlexibleRealSenseCamera:
    """RealSense camera wrapper that starts the SDK with default stream profiles."""

    def __init__(self, serial_number: str, width: int, height: int) -> None:
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.frame_lock = threading.Lock()
        self.latest_depth_frame: np.ndarray | None = None
        self.latest_depth_preview_frame: np.ndarray | None = None
        self.latest_color_frame: np.ndarray | None = None
        self.rs_pipeline = None
        self.rs_profile = None
        self._rs = None

    @property
    def is_connected(self) -> bool:
        return self.rs_pipeline is not None and self.rs_profile is not None

    def connect(self) -> None:
        if self.is_connected:
            return
        import pyrealsense2 as rs

        self._rs = rs
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device(config, self.serial_number)
        config.enable_stream(rs.stream.color)
        config.enable_stream(rs.stream.depth)
        self.rs_profile = pipeline.start(config)
        self.rs_pipeline = pipeline

        # Warm up a few frames; RealSense often returns incomplete early frames.
        for _ in range(10):
            self._read_frames()

    def disconnect(self) -> None:
        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
        self.rs_pipeline = None
        self.rs_profile = None

    def async_read(self, timeout_ms: int = 200) -> np.ndarray:
        del timeout_ms
        color, _depth = self._read_frames()
        return color

    def read_depth(self, timeout_ms: int = 200) -> np.ndarray:
        del timeout_ms
        _color, depth = self._read_frames()
        return depth

    def _read_frames(self) -> tuple[np.ndarray, np.ndarray]:
        if self.rs_pipeline is None:
            raise RuntimeError("RealSense pipeline is not connected.")

        frames = self.rs_pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense frame set did not include both color and depth frames.")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        color = self._resize_color(color)
        depth_preview = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        with self.frame_lock:
            self.latest_color_frame = color
            self.latest_depth_frame = depth
            self.latest_depth_preview_frame = depth_preview
        return color, depth_preview

    def _resize_color(self, color: np.ndarray) -> np.ndarray:
        if color.ndim == 2:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
        elif color.shape[2] == 4:
            color = color[:, :, :3]
        if color.shape[1] != self.width or color.shape[0] != self.height:
            color = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(color)


class LocalBimanualSOFollower:
    name = "bi_so_follower"

    def __init__(self, left_arm: SO101Follower, right_arm: SO101Follower) -> None:
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.cameras = {**left_arm.cameras, **right_arm.cameras}

    @property
    def robot_type(self) -> str:
        return self.name

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.observation_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.observation_features.items()},
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self.left_arm.action_features.items()},
            **{f"right_{key}": value for key, value in self.right_arm.action_features.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self) -> None:
        self.left_arm.connect()
        self.right_arm.connect()

    def get_observation(self) -> dict[str, Any]:
        left_obs = self.left_arm.get_observation()
        right_obs = self.right_arm.get_observation()
        return {
            **{f"left_{key}": value for key, value in left_obs.items()},
            **{f"right_{key}": value for key, value in right_obs.items()},
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        left_action = {key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")}
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    def disconnect(self) -> None:
        self._disconnect_arm_safely("left follower", self.left_arm)
        self._disconnect_arm_safely("right follower", self.right_arm)

    @staticmethod
    def _disconnect_arm_safely(label: str, arm: SO101Follower) -> None:
        try:
            arm.disconnect()
            return
        except Exception as exc:
            print(f"[GUI recorder] Warning: normal disconnect failed for {label}: {exc}", flush=True)

        try:
            arm.bus.disconnect(disable_torque=False)
        except Exception as exc:
            print(f"[GUI recorder] Warning: force-closing bus failed for {label}: {exc}", flush=True)

        for name, cam in getattr(arm, "cameras", {}).items():
            try:
                if getattr(cam, "is_connected", False):
                    cam.disconnect()
            except Exception as exc:
                print(f"[GUI recorder] Warning: disconnecting camera {label}/{name} failed: {exc}", flush=True)


class LocalBimanualSOLeader:
    name = "bi_so_leader"

    def __init__(self, left_arm: SO101Leader, right_arm: SO101Leader) -> None:
        self.left_arm = left_arm
        self.right_arm = right_arm

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self) -> None:
        self.left_arm.connect()
        self.right_arm.connect()

    def get_action(self) -> dict[str, float]:
        left_action = self.left_arm.get_action()
        right_action = self.right_arm.get_action()
        return {
            **{f"left_{key}": value for key, value in left_action.items()},
            **{f"right_{key}": value for key, value in right_action.items()},
        }

    def disconnect(self) -> None:
        self._disconnect_arm_safely("left leader", self.left_arm)
        self._disconnect_arm_safely("right leader", self.right_arm)

    @staticmethod
    def _disconnect_arm_safely(label: str, arm: SO101Leader) -> None:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"[GUI recorder] Warning: disconnect failed for {label}: {exc}", flush=True)


@dataclass(frozen=True)
class RecorderSettings:
    left_follower_port: str
    right_follower_port: str
    left_leader_port: str
    right_leader_port: str
    left_follower_id: str
    right_follower_id: str
    left_leader_id: str
    right_leader_id: str
    opencv_front: str
    opencv_side: str
    enable_realsense: bool
    realsense_serial: str
    repo_id: str
    dataset_root: str
    calibration_dir: str
    task: str
    resume: bool
    push_to_hub: bool


@dataclass(frozen=True)
class DatasetHealth:
    info_episodes: int
    metadata_episodes: int
    valid_episode_indices: list[int]
    total_frames: int
    needs_rewrite: bool
    reason: str


@dataclass(frozen=True)
class EpisodeBrowserRef:
    episode_index: int
    row: Any


class BimanualRecorder:
    def __init__(
        self,
        settings: RecorderSettings,
        status_queue: queue.Queue[str],
        preview_queue: queue.Queue[dict[str, np.ndarray]],
        state_queue: queue.Queue[dict[str, float]],
    ) -> None:
        self.settings = settings
        self.status_queue = status_queue
        self.preview_queue = preview_queue
        self.state_queue = state_queue
        self.command_queue: queue.Queue[Any] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.recording = False
        self.ready = False
        self.failed = False
        self.episode_count = 0
        self.current_episode_task = settings.task
        self.dataset: LeRobotDataset | None = None
        self.raw_depth_frames: list[np.ndarray] = []
        self.raw_depth_metadata: dict[str, np.ndarray] = {}

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def request_record(self, task: str) -> None:
        if not self.ready:
            self._put_status("Not ready yet. Wait until status says preview and teleoperation are running.")
            return
        self.command_queue.put(("record", task))

    def request_end_episode(self) -> None:
        self.command_queue.put("end")

    def request_discard_episode(self) -> None:
        self.command_queue.put("discard")

    def stop(self) -> None:
        self.command_queue.put("stop")
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _put_status(self, message: str) -> None:
        print(f"[GUI recorder] {message}", flush=True)
        self.status_queue.put(message)

    def _put_progress(self, current: int, total: int, message: str) -> None:
        self.status_queue.put(f"PROGRESS|{current}|{total}|{message}")

    def _put_episode(self, episode_index: int, recording: bool = False) -> None:
        self.status_queue.put(f"EPISODE|{episode_index}|{int(recording)}")

    def _run(self) -> None:
        robot = None
        teleop = None
        video_manager = None
        try:
            self._put_status("Building robot and teleoperator configs...")
            robot = self._make_robot()
            teleop = self._make_teleop()
            teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

            self._put_status("Computing LeRobot dataset features...")
            dataset_features = combine_feature_dicts(
                aggregate_pipeline_dataset_features(
                    pipeline=teleop_action_processor,
                    initial_features=create_initial_features(action=robot.action_features),
                    use_videos=True,
                ),
                aggregate_pipeline_dataset_features(
                    pipeline=robot_observation_processor,
                    initial_features=create_initial_features(observation=robot.observation_features),
                    use_videos=True,
                ),
            )

            root = Path(self.settings.dataset_root).expanduser() if self.settings.dataset_root else None
            self._put_status("Creating/opening LeRobot dataset...")
            self.dataset = self._open_or_create_dataset(root, robot, dataset_features)
            self.episode_count = self.dataset.num_episodes
            self._put_episode(self.episode_count)
            self._put_status(f"Dataset root: {self.dataset.root}")
            self._put_status(f"Existing episodes: {self.episode_count}.")

            self._put_status("Connecting follower arms and cameras...")
            robot.connect()
            self._put_status("Follower arms and cameras connected. Connecting leader arms...")
            teleop.connect()
            self._put_status("Leader arms connected.")

            self._put_status("Starting video encoding manager...")
            video_manager = VideoEncodingManager(self.dataset)
            video_manager.__enter__()
            self.ready = True
            self._put_status("Ready. Preview and teleoperation are running. You can click Record.")

            while not self.stop_event.is_set():
                self._process_commands()
                start_loop_t = time.perf_counter()

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                action = teleop_action_processor((teleop.get_action(), obs))
                action_to_send = robot_action_processor((action, obs))
                _ = robot.send_action(action_to_send)

                self._publish_preview(obs_processed)
                self._publish_state(obs_processed, action)

                if self.recording:
                    observation_frame = build_dataset_frame(self.dataset.features, obs_processed, prefix=OBS_STR)
                    action_frame = build_dataset_frame(self.dataset.features, action, prefix=ACTION)
                    frame = {**observation_frame, **action_frame, "task": self.current_episode_task}
                    self.dataset.add_frame(frame)
                    self._store_raw_depth_frame(robot)

                dt_s = time.perf_counter() - start_loop_t
                time.sleep(max(1 / FPS - dt_s, 0.0))

            self._end_episode_if_needed()
        except Exception as exc:
            self.failed = True
            self._put_status(f"ERROR: {exc}")
        finally:
            self.ready = False
            try:
                if video_manager is not None:
                    video_manager.__exit__(None, None, None)
            finally:
                if self.dataset is not None:
                    self.dataset.finalize()
                    if self.settings.push_to_hub:
                        self._put_status("Uploading dataset to Hugging Face Hub...")
                        self.dataset.push_to_hub()
                if robot is not None:
                    robot.disconnect()
                if teleop is not None:
                    teleop.disconnect()
                self._put_status("Recorder stopped.")

    def _open_or_create_dataset(
        self, root: Path | None, robot: LocalBimanualSOFollower, dataset_features: dict[str, dict]
    ) -> LeRobotDataset:
        image_writer_threads = max(4, 4 * len(robot.cameras))
        repo_id = self.settings.repo_id
        target_root = self._dataset_root(root, repo_id)

        if root is not None:
            self._put_status(f"Selected dataset folder: {target_root}")

        if target_root.exists() and self._has_dataset_metadata(target_root):
            health = self._inspect_dataset_health(target_root)
            self._put_status(
                f"Dataset check: info={health.info_episodes}, metadata={health.metadata_episodes}, valid={len(health.valid_episode_indices)} episode(s)."
            )
            if health.needs_rewrite:
                if health.valid_episode_indices:
                    raise RuntimeError(
                        "Dataset needs cleanup before recording. "
                        f"{health.reason} Use the GUI cleanup button or choose another folder."
                    )
                if health.info_episodes > 0 or health.metadata_episodes > 0:
                    raise RuntimeError(
                        "Dataset has no valid episode to keep. "
                        f"{health.reason} Choose another folder or create a new dataset folder."
                    )

            episodes = len(health.valid_episode_indices)
            if episodes > 0:
                self._put_status(f"Dataset folder already has {episodes} clean episode(s). Appending from episode {episodes}.")
                return self._load_existing_dataset(repo_id, root, image_writer_threads)

            self._put_status("Selected dataset has 0 episodes. Reusing this folder and recording from episode 0.")
            self._reset_dataset_folder(target_root)
            return LeRobotDataset.create(
                repo_id,
                FPS,
                root=root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=image_writer_threads,
                vcodec="h264",
            )

        if self.settings.resume and target_root.exists():
            self._put_status("Resume enabled. Loading existing dataset.")
            return self._load_existing_dataset(repo_id, root, image_writer_threads)

        if target_root.exists():
            if self._is_plain_empty_dir(target_root):
                target_root.rmdir()
                self._put_status(f"Using empty dataset folder: {target_root}")
            else:
                root, repo_id, target_root = self._make_new_dataset_location(root, repo_id, target_root)
                self._put_status(
                    f"Selected folder is not an empty LeRobot dataset, so a new local dataset will be created at: {target_root}"
                )

        try:
            return LeRobotDataset.create(
                repo_id,
                FPS,
                root=root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=image_writer_threads,
                vcodec="h264",
            )
        except FileExistsError:
            self._put_status("Dataset folder already exists. Loading it in resume mode.")
            try:
                return self._load_existing_dataset(repo_id, root, image_writer_threads)
            except Exception as exc:
                root, repo_id, target_root = self._make_new_dataset_location(root, repo_id, target_root)
                self._put_status(
                    f"Could not load existing dataset locally ({exc}). Creating a new local dataset at: {target_root}"
                )
                return LeRobotDataset.create(
                    repo_id,
                    FPS,
                    root=root,
                    robot_type=robot.name,
                    features=dataset_features,
                    use_videos=True,
                    image_writer_processes=0,
                    image_writer_threads=image_writer_threads,
                    vcodec="h264",
                )

    def _load_existing_dataset(self, repo_id: str, root: Path | None, image_writer_threads: int) -> LeRobotDataset:
        dataset = LeRobotDataset(repo_id, root=root, revision=None, download_videos=False, vcodec="h264")
        if image_writer_threads:
            dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)
        return dataset

    @staticmethod
    def _dataset_root(root: Path | None, repo_id: str) -> Path:
        return root if root is not None else HF_LEROBOT_HOME / repo_id

    @staticmethod
    def _has_dataset_metadata(dataset_root: Path) -> bool:
        return (dataset_root / "meta" / "info.json").exists()

    @staticmethod
    def _dataset_episode_count(dataset_root: Path) -> int:
        info_path = dataset_root / "meta" / "info.json"
        try:
            with info_path.open() as f:
                info = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return 0
        return int(info.get("total_episodes", 0))

    @classmethod
    def _inspect_dataset_health(cls, dataset_root: Path) -> DatasetHealth:
        info_path = dataset_root / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text())
            info_episodes = int(info.get("total_episodes", 0))
            total_frames = int(info.get("total_frames", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
            return DatasetHealth(0, 0, [], 0, True, f"metadata unreadable: {exc}")

        try:
            episodes_df = read_nested_parquets(dataset_root / "meta" / "episodes").sort_values("episode_index")
        except Exception as exc:
            if info_episodes == 0:
                return DatasetHealth(info_episodes, 0, [], total_frames, False, "empty dataset")
            return DatasetHealth(info_episodes, 0, [], total_frames, True, f"episode metadata unreadable: {exc}")

        try:
            frames_df = read_nested_parquets(dataset_root / "data", skip_bad=True)
        except Exception as exc:
            return DatasetHealth(info_episodes, len(episodes_df), [], total_frames, True, f"frame data unreadable: {exc}")

        video_keys = video_keys_from_info(info)
        valid: list[int] = []
        valid_frame_count = 0
        invalid_reasons: list[str] = []
        for _, row in episodes_df.iterrows():
            try:
                ep_idx = int(row["episode_index"])
                ep_frames = frames_df[frames_df["episode_index"] == ep_idx]
                if ep_frames.empty:
                    invalid_reasons.append(f"episode {ep_idx} has no frame rows")
                    continue
                expected_length = int(row.get("length", len(ep_frames)))
                if expected_length != len(ep_frames):
                    invalid_reasons.append(
                        f"episode {ep_idx} length says {expected_length}, data has {len(ep_frames)}"
                    )
                    continue
                missing_video = False
                for video_key in video_keys:
                    video_path = dataset_root / info["video_path"].format(
                        video_key=video_key,
                        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
                        file_index=int(row[f"videos/{video_key}/file_index"]),
                    )
                    if not video_path.exists() or video_path.stat().st_size == 0:
                        invalid_reasons.append(f"episode {ep_idx} missing video {video_key}")
                        missing_video = True
                        break
                if not missing_video:
                    valid.append(ep_idx)
                    valid_frame_count += len(ep_frames)
            except Exception as exc:
                invalid_reasons.append(f"episode row unreadable: {exc}")

        expected_indices = list(range(len(valid)))
        metadata_indices = [int(row["episode_index"]) for _, row in episodes_df.iterrows()]
        needs_rewrite = (
            info_episodes != len(valid)
            or total_frames != valid_frame_count
            or len(episodes_df) != len(valid)
            or valid != expected_indices
            or metadata_indices[: len(valid)] != expected_indices
            or bool(invalid_reasons)
        )
        reason_parts: list[str] = []
        if info_episodes != len(valid):
            reason_parts.append(f"info.json says {info_episodes}, valid is {len(valid)}")
        if total_frames != valid_frame_count:
            reason_parts.append(f"info.json frames {total_frames}, valid frames {valid_frame_count}")
        if len(episodes_df) != len(valid):
            reason_parts.append(f"metadata rows {len(episodes_df)}, valid is {len(valid)}")
        if valid != expected_indices:
            reason_parts.append("episode indices are not clean 0..N-1")
        reason_parts.extend(invalid_reasons[:3])
        reason = "; ".join(reason_parts) if reason_parts else "dataset is clean"
        return DatasetHealth(info_episodes, len(episodes_df), valid, total_frames, needs_rewrite, reason)

    @classmethod
    def _is_empty_dataset_root(cls, dataset_root: Path) -> bool:
        if not cls._has_dataset_metadata(dataset_root):
            return True
        return cls._dataset_episode_count(dataset_root) == 0

    @classmethod
    def _reset_zero_episode_dataset(cls, dataset_root: Path) -> None:
        if not cls._has_dataset_metadata(dataset_root) or cls._dataset_episode_count(dataset_root) != 0:
            raise ValueError(f"Refusing to reset non-empty dataset folder: {dataset_root}")
        cls._reset_dataset_folder(dataset_root)

    @staticmethod
    def _reset_dataset_folder(dataset_root: Path) -> None:
        for child in dataset_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        dataset_root.rmdir()

    @staticmethod
    def _is_plain_empty_dir(path: Path) -> bool:
        return path.is_dir() and not any(path.iterdir())

    @staticmethod
    def _make_new_repo_id(repo_id: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{repo_id}_{timestamp}"

    @staticmethod
    def _make_new_dataset_location(
        root: Path | None, repo_id: str, target_root: Path
    ) -> tuple[Path | None, str, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if root is None:
            repo_id = f"{repo_id}_{timestamp}"
            return None, repo_id, HF_LEROBOT_HOME / repo_id
        new_root = target_root.parent / f"{target_root.name}_{timestamp}"
        return new_root, f"{repo_id}_{timestamp}", new_root

    def _calibration_dirs(self) -> tuple[Path | None, Path | None]:
        raw_path = self.settings.calibration_dir.strip()
        if not raw_path:
            return None, None

        def existing_or_default(parent: Path, candidates: tuple[str, ...]) -> Path:
            for candidate in candidates:
                path = parent / candidate
                if path.exists():
                    return path
            return parent / candidates[0]

        path = Path(raw_path).expanduser()
        if path.name == "calibration":
            return (
                existing_or_default(path / "robots", ("so_follower", "so101_follower")),
                existing_or_default(path / "teleoperators", ("so_leader", "so101_leader")),
            )
        if path.name == "robots":
            calibration_root = path.parent
            return (
                existing_or_default(path, ("so_follower", "so101_follower")),
                existing_or_default(calibration_root / "teleoperators", ("so_leader", "so101_leader")),
            )
        if path.name in {"so_follower", "so101_follower"} and path.parent.name == "robots":
            calibration_root = path.parent.parent
            return path, existing_or_default(calibration_root / "teleoperators", ("so_leader", "so101_leader"))
        return path, path

    def _make_robot(self) -> LocalBimanualSOFollower:
        cameras = {}
        serial = self._resolve_realsense_serial()
        if serial:
            cameras["front"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
            self._put_status("RealSense will be connected before OpenCV cameras.")

        front_path = self.settings.opencv_front.strip()
        if front_path and not self.settings.enable_realsense:
            cameras["front"] = OpenCVCameraConfig(
                index_or_path=front_path,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                fourcc="MJPG",
            )
        elif front_path and self.settings.enable_realsense:
            self._put_status(
                f"RealSense is enabled as front; ignoring OpenCV front path {front_path}."
            )

        side_path = self._resolve_opencv_side_path(self.settings.opencv_side.strip())
        if side_path:
            side_description = self._describe_video_path(side_path)
            if side_description:
                self._put_status(f"OpenCV side RGB {side_path} is {side_description}.")
            cameras["side"] = OpenCVCameraConfig(
                index_or_path=side_path,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                fourcc="MJPG",
            )
        cameras = {
            **cameras,
        }

        robot_calibration_dir, _ = self._calibration_dirs()
        if robot_calibration_dir is not None:
            self._put_status(f"Robot calibration dir: {robot_calibration_dir}")
        left_arm = SO101Follower(
            SOFollowerRobotConfig(
                id=self.settings.left_follower_id,
                calibration_dir=robot_calibration_dir,
                port=self.settings.left_follower_port,
                cameras=cameras,
            )
        )
        right_arm = SO101Follower(
            SOFollowerRobotConfig(
                id=self.settings.right_follower_id,
                calibration_dir=robot_calibration_dir,
                port=self.settings.right_follower_port,
            )
        )
        robot = LocalBimanualSOFollower(left_arm, right_arm)

        if serial and "front" in robot.left_arm.cameras:
            robot.left_arm.cameras["front"] = FlexibleRealSenseCamera(serial, CAMERA_WIDTH, CAMERA_HEIGHT)
            robot.cameras["front"] = robot.left_arm.cameras["front"]
            self._put_status("Using flexible RealSense default-profile wrapper for front.")

        if "front" in robot.left_arm.cameras and isinstance(
            robot.left_arm.cameras["front"], (RealSenseCamera, FlexibleRealSenseCamera)
        ):
            rs_camera = robot.left_arm.cameras["front"]
            robot.left_arm.cameras["front_depth"] = RealSenseDepthView(rs_camera)
            robot.left_arm.config.cameras["front_depth"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
            robot.cameras["front_depth"] = robot.left_arm.cameras["front_depth"]

        return robot

    def _resolve_opencv_side_path(self, requested_path: str) -> str:
        if not requested_path:
            self._put_status("OpenCV side RGB is blank; recording without side RGB.")
            return ""

        if self.settings.enable_realsense and self._is_realsense_video_path(requested_path):
            self._put_status(
                f"OpenCV side RGB {requested_path} points to a RealSense video node; ignoring it because RealSense front/depth is enabled."
            )
            requested_path = ""

        try:
            available = OpenCVCamera.find_cameras()
        except Exception as exc:
            if requested_path and Path(requested_path).exists():
                self._put_status(
                    f"OpenCV camera scan failed ({exc}); trying requested side RGB {requested_path}."
                )
                return requested_path
            self._put_status(f"OpenCV camera scan failed ({exc}); recording without side RGB.")
            return ""

        available_paths = [str(camera.get("id")) for camera in available]
        if requested_path and requested_path in available_paths:
            return requested_path

        if requested_path:
            if Path(requested_path).exists():
                self._put_status(
                    f"OpenCV side RGB {requested_path} exists but did not open during camera scan."
                )
            else:
                self._put_status(f"OpenCV side RGB {requested_path} does not exist.")

        for candidate in available_paths:
            if self.settings.enable_realsense and self._is_realsense_video_path(candidate):
                continue
            self._put_status(f"Using detected OpenCV side RGB {candidate}.")
            return candidate

        if available_paths and self.settings.enable_realsense:
            self._put_status(
                "Only RealSense OpenCV video nodes were detected; recording without side RGB to avoid opening the active RealSense twice."
            )
        else:
            self._put_status("No usable OpenCV side RGB camera detected; recording without side RGB.")
        return ""

    @staticmethod
    def _describe_video_path(video_path: str) -> str:
        path = Path(video_path)
        if not path.exists():
            return ""
        try:
            resolved = path.resolve()
        except OSError:
            return ""
        by_id_root = Path("/dev/v4l/by-id")
        if not by_id_root.exists():
            return BimanualRecorder._video_device_name(path)
        for link in sorted(by_id_root.iterdir()):
            try:
                if link.resolve() == resolved:
                    return link.name
            except OSError:
                continue
        return BimanualRecorder._video_device_name(path)

    @staticmethod
    def _is_realsense_video_path(video_path: str) -> bool:
        description = BimanualRecorder._describe_video_path(video_path).lower()
        return "realsense" in description or "intel(r) real sense" in description

    @staticmethod
    def _video_device_name(path: Path) -> str:
        if not path.name.startswith("video"):
            return ""
        name_path = Path("/sys/class/video4linux") / path.name / "name"
        try:
            return name_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _make_teleop(self) -> LocalBimanualSOLeader:
        _, teleop_calibration_dir = self._calibration_dirs()
        if teleop_calibration_dir is not None:
            self._put_status(f"Teleop calibration dir: {teleop_calibration_dir}")
        left_arm = SO101Leader(
            SOLeaderTeleopConfig(
                id=self.settings.left_leader_id,
                calibration_dir=teleop_calibration_dir,
                port=self.settings.left_leader_port,
            )
        )
        right_arm = SO101Leader(
            SOLeaderTeleopConfig(
                id=self.settings.right_leader_id,
                calibration_dir=teleop_calibration_dir,
                port=self.settings.right_leader_port,
            )
        )
        return LocalBimanualSOLeader(left_arm, right_arm)

    def _resolve_realsense_serial(self) -> str:
        if not self.settings.enable_realsense:
            self._put_status("RealSense disabled. Using OpenCV cameras only.")
            return ""
        requested_serial = self.settings.realsense_serial.strip()
        try:
            cameras = RealSenseCamera.find_cameras()
        except Exception as exc:
            if requested_serial:
                self._put_status(
                    f"RealSense auto-detect failed ({exc}); trying manually entered serial {requested_serial}."
                )
                return requested_serial
            self._put_status(f"RealSense auto-detect failed, skipping RealSense: {exc}")
            return ""
        if not cameras:
            self._put_status("No RealSense found, skipping RealSense RGB/depth.")
            return ""
        serials = [str(camera["id"]) for camera in cameras]
        self._put_status(f"RealSense SDK sees serial(s): {serials}")
        if requested_serial:
            if requested_serial in serials:
                self._put_status(f"Using requested RealSense serial {requested_serial}.")
                return requested_serial
            self._put_status(
                f"Requested RealSense serial {requested_serial} was not found by SDK; using {serials[0]} instead."
            )
            return serials[0]
        serial = str(cameras[0]["id"])
        self._put_status(f"Using RealSense serial {serial}.")
        return serial

    def _process_commands(self) -> None:
        while True:
            try:
                command = self.command_queue.get_nowait()
            except queue.Empty:
                break

            command_name = command[0] if isinstance(command, tuple) else command

            if command_name == "record":
                if not self.recording:
                    requested_task = command[1] if isinstance(command, tuple) and len(command) > 1 else self.settings.task
                    self.current_episode_task = str(requested_task or self.settings.task)
                    self.recording = True
                    self.raw_depth_frames = []
                    self.raw_depth_metadata = {}
                    self._put_episode(self.episode_count, recording=True)
                    self._put_progress(0, 100, f"Recording episode {self.episode_count}")
                    self._put_status(f"Recording episode {self.episode_count} with task '{self.current_episode_task}'...")
            elif command_name == "end":
                self._end_episode_if_needed()
            elif command_name == "discard":
                self._discard_episode_if_needed()
            elif command_name == "stop":
                self.stop_event.set()
                self._end_episode_if_needed()

    def _end_episode_if_needed(self) -> None:
        if not self.recording or self.dataset is None:
            return
        self.recording = False
        saved_episode_index = self.episode_count
        frame_count = int(self.dataset.episode_buffer["size"]) if self.dataset.episode_buffer else 0
        self._put_episode(saved_episode_index, recording=False)
        self._put_progress(5, 100, f"Saving episode {saved_episode_index} ({frame_count} frames)...")
        self._put_status(f"Saving episode {saved_episode_index} ({frame_count} frames)...")
        try:
            self._put_progress(20, 100, "Writing parquet data and encoding videos...")
            self.dataset.save_episode()
            self._put_progress(85, 100, "Saving raw depth sidecar...")
            self._save_raw_depth_sidecar(saved_episode_index)
            self.episode_count = self.dataset.num_episodes
            if self.episode_count != saved_episode_index + 1:
                raise RuntimeError(
                    f"saved episode count mismatch: expected {saved_episode_index + 1}, dataset reports {self.episode_count}"
                )
            self.raw_depth_frames = []
            self.raw_depth_metadata = {}
            self._put_episode(self.episode_count)
            self._put_progress(100, 100, f"Episode {saved_episode_index} saved successfully.")
            self._put_status(
                f"Saved episode {saved_episode_index} successfully. Next episode is {self.episode_count}."
            )
        except Exception as exc:
            self.failed = True
            self._put_progress(0, 100, f"Save failed for episode {saved_episode_index}.")
            self._put_status(f"ERROR: failed to save episode {saved_episode_index}: {exc}")
            raise

    def _discard_episode_if_needed(self) -> None:
        if not self.recording or self.dataset is None:
            self._put_status("No active recording to discard.")
            return
        self.recording = False
        self.dataset.clear_episode_buffer(delete_images=True)
        discarded_frames = len(self.raw_depth_frames)
        self.raw_depth_frames = []
        self.raw_depth_metadata = {}
        self._put_episode(self.episode_count)
        self._put_progress(0, 100, f"Episode {self.episode_count} discarded.")
        self._put_status(
            f"Discarded episode {self.episode_count} ({discarded_frames} raw depth frame(s) dropped). Click Record to retry."
        )

    def _store_raw_depth_frame(self, robot: LocalBimanualSOFollower) -> None:
        rs_camera = robot.left_arm.cameras.get("front")
        if not isinstance(rs_camera, (RealSenseCamera, FlexibleRealSenseCamera)):
            return
        with rs_camera.frame_lock:
            depth = rs_camera.latest_depth_frame
        if depth is not None:
            self.raw_depth_frames.append(np.asarray(depth, dtype=np.uint16).copy())
            if not self.raw_depth_metadata:
                self.raw_depth_metadata = self._get_realsense_depth_metadata(rs_camera)

    def _save_raw_depth_sidecar(self, episode_index: int) -> None:
        if self.dataset is None or not self.raw_depth_frames:
            return
        depth_dir = self.dataset.root / "sidecar_depth" / "left_front_depth_mm"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = depth_dir / f"episode_{episode_index:06d}.npz"
        depth_stack = np.stack(self.raw_depth_frames, axis=0)
        np.savez_compressed(
            depth_path,
            depth_mm=depth_stack,
            fps=np.array(FPS, dtype=np.int32),
            episode_index=np.array(episode_index, dtype=np.int64),
            **self.raw_depth_metadata,
        )
        self._put_status(f"Saved raw depth sidecar: {depth_path}")

    def _get_realsense_depth_metadata(self, rs_camera: RealSenseCamera | FlexibleRealSenseCamera) -> dict[str, np.ndarray]:
        metadata: dict[str, np.ndarray] = {}
        try:
            import pyrealsense2 as rs
        except Exception:
            return metadata

        try:
            if rs_camera.rs_profile is None:
                return metadata
            depth_stream = rs_camera.rs_profile.get_stream(rs.stream.depth).as_video_stream_profile()
            color_stream = rs_camera.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
            depth_intr = depth_stream.get_intrinsics()
            color_intr = color_stream.get_intrinsics()
            metadata["depth_intrinsics"] = np.array(
                [depth_intr.width, depth_intr.height, depth_intr.fx, depth_intr.fy, depth_intr.ppx, depth_intr.ppy],
                dtype=np.float32,
            )
            metadata["color_intrinsics"] = np.array(
                [color_intr.width, color_intr.height, color_intr.fx, color_intr.fy, color_intr.ppx, color_intr.ppy],
                dtype=np.float32,
            )
            device = rs_camera.rs_profile.get_device()
            depth_sensor = device.first_depth_sensor()
            metadata["depth_scale_m"] = np.array(depth_sensor.get_depth_scale(), dtype=np.float32)
        except Exception:
            return metadata
        return metadata

    def _publish_preview(self, obs: dict[str, Any]) -> None:
        frames = {key: value for key, value in obs.items() if isinstance(value, np.ndarray) and value.ndim == 3}
        if not frames:
            return
        self._put_latest(self.preview_queue, frames)

    def _publish_state(self, obs: dict[str, Any], action: dict[str, Any]) -> None:
        values: dict[str, float] = {}
        for prefix, data in (("obs", obs), ("act", action)):
            for key, value in data.items():
                if isinstance(value, (int, float, np.floating)):
                    values[f"{prefix}.{key}"] = float(value)
        self._put_latest(self.state_queue, values)

    @staticmethod
    def _put_latest(q: queue.Queue, item: Any) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)


class RecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LeRobot SO101 Bimanual Recorder")
        self.root.geometry("1360x940")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda _event: self.set_fullscreen(False))
        self._configure_style()

        self.status_queue: queue.Queue[str] = queue.Queue()
        self.preview_queue: queue.Queue[dict[str, np.ndarray]] = queue.Queue(maxsize=1)
        self.state_queue: queue.Queue[dict[str, float]] = queue.Queue(maxsize=1)
        self.recorder: BimanualRecorder | None = None
        self.cleanup_running = False
        self.pending_cleanup_root: Path | None = None
        self.pending_cleanup_episodes: list[int] = []
        self.current_image: ImageTk.PhotoImage | None = None
        self.dataset_image: ImageTk.PhotoImage | None = None
        self.latest_frames: dict[str, np.ndarray] = {}
        self.video_mode = "dataset"
        self.dataset_info: dict[str, Any] = {}
        self.dataset_episodes: dict[int, EpisodeBrowserRef] = {}
        self.dataset_video_keys: list[str] = []
        self.current_dataset_episode: EpisodeBrowserRef | None = None
        self.dataset_captures: dict[str, cv2.VideoCapture] = {}
        self.playing_dataset = False
        self.play_offset_s = 0.0
        self.play_started_at = 0.0

        self.vars = {
            "left_follower_port": tk.StringVar(value="/dev/ttyACM0"),
            "right_follower_port": tk.StringVar(value="/dev/ttyACM1"),
            "left_leader_port": tk.StringVar(value="/dev/ttyACM2"),
            "right_leader_port": tk.StringVar(value="/dev/ttyACM3"),
            "left_follower_id": tk.StringVar(value="my_awesome_follower_arm"),
            "right_follower_id": tk.StringVar(value="my_awesome_follower_arm_r"),
            "left_leader_id": tk.StringVar(value="my_awesome_leader_arm"),
            "right_leader_id": tk.StringVar(value="my_awesome_leader_arm_r"),
            "opencv_front": tk.StringVar(value=""),
            "opencv_side": tk.StringVar(value="/dev/video4"),
            "realsense_serial": tk.StringVar(value=""),
            "repo_id": tk.StringVar(value=self._infer_repo_id(DEFAULT_DATASET_ROOT)),
            "dataset_root": tk.StringVar(value=str(DEFAULT_DATASET_ROOT)),
            "calibration_dir": tk.StringVar(value=str(DEFAULT_CALIBRATION_PATH / "robots")),
            "task": tk.StringVar(value=TASK_CHOICES[0]),
            "edit_task": tk.StringVar(value=TASK_CHOICES[0]),
            "preview_key": tk.StringVar(value=""),
        }
        self.resume = tk.BooleanVar(value=False)
        self.push_to_hub = tk.BooleanVar(value=False)
        self.enable_realsense = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Configure ports, then click Connect.")
        self.state_text = tk.StringVar(value="")
        self.episode_text = tk.StringVar(value="Episode: not connected")
        self.progress_text = tk.StringVar(value="Idle")
        self.progress_value = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.root.after(30, self._poll_queues)

    def _configure_style(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=12)
        self.root.option_add("*Font", default_font)

        style = ttk.Style(self.root)
        style.configure("TButton", font=("TkDefaultFont", 12), padding=(14, 8))
        style.configure("TLabel", font=("TkDefaultFont", 12))
        style.configure("TCheckbutton", font=("TkDefaultFont", 12), padding=(4, 4))
        style.configure("TEntry", font=("TkTextFont", 12), padding=(4, 4))
        style.configure("TCombobox", font=("TkTextFont", 12), padding=(4, 4))
        style.configure("Episode.Treeview", font=("TkTextFont", 11), rowheight=30)
        style.configure("Episode.Treeview.Heading", font=("TkDefaultFont", 11, "bold"))

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=14)
        controls.pack(fill=tk.X)

        grid = ttk.Frame(controls)
        grid.pack(fill=tk.X)
        fields = [
            ("Follower L", "left_follower_port"),
            ("Follower R", "right_follower_port"),
            ("Leader L", "left_leader_port"),
            ("Leader R", "right_leader_port"),
            ("OpenCV front", "opencv_front"),
            ("OpenCV side RGB", "opencv_side"),
            ("Front RealSense SN", "realsense_serial"),
            ("Dataset name/id", "repo_id"),
            ("Dataset folder", "dataset_root"),
            ("Calibration path", "calibration_dir"),
            ("Task", "task"),
        ]
        for i, (label, key) in enumerate(fields):
            ttk.Label(grid, text=label).grid(row=i // 2, column=(i % 2) * 2, sticky=tk.W, padx=(0, 8), pady=5)
            if key == "dataset_root":
                dataset_frame = ttk.Frame(grid)
                dataset_frame.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky=tk.EW, padx=(0, 16), pady=5)
                dataset_frame.columnconfigure(0, weight=1)
                ttk.Entry(dataset_frame, textvariable=self.vars[key], width=34).grid(
                    row=0, column=0, sticky=tk.EW
                )
                ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset_folder).grid(
                    row=0, column=1, padx=(8, 0)
                )
                ttk.Button(dataset_frame, text="New", command=self.new_dataset_folder).grid(
                    row=0, column=2, padx=(8, 0)
                )
                ttk.Button(dataset_frame, text="Clean", command=self.clean_dataset).grid(
                    row=0, column=3, padx=(8, 0)
                )
            elif key == "task":
                ttk.Combobox(
                    grid,
                    textvariable=self.vars[key],
                    values=TASK_CHOICES,
                    state="readonly",
                    width=46,
                ).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky=tk.EW, padx=(0, 16), pady=5)
            else:
                ttk.Entry(grid, textvariable=self.vars[key], width=48).grid(
                    row=i // 2, column=(i % 2) * 2 + 1, sticky=tk.EW, padx=(0, 16), pady=5
                )
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        options = ttk.Frame(controls)
        options.pack(fill=tk.X, pady=(8, 0))
        ttk.Checkbutton(options, text="Resume dataset", variable=self.resume).pack(side=tk.LEFT)
        ttk.Checkbutton(options, text="Push to Hub on close", variable=self.push_to_hub).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(options, text="Enable RealSense", variable=self.enable_realsense).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Label(options, text="Preview").pack(side=tk.LEFT, padx=(24, 6))
        self.preview_combo = ttk.Combobox(
            options,
            textvariable=self.vars["preview_key"],
            state="readonly",
            width=34,
        )
        self.preview_combo.pack(side=tk.LEFT)

        buttons = ttk.Frame(controls)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="Connect", command=self.connect).pack(side=tk.LEFT, ipady=4)
        ttk.Button(buttons, text="Record", command=self.record).pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="End", command=self.end_episode).pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="Discard", command=self.discard_episode).pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="Fullscreen", command=self.toggle_fullscreen).pack(
            side=tk.LEFT, padx=(10, 0), ipady=4
        )
        ttk.Button(buttons, text="Stop", command=self.stop_recorder).pack(side=tk.LEFT, padx=(10, 0), ipady=4)

        content = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(content)
        left_panel.grid(row=0, column=0, sticky=tk.NS, padx=(0, 10))
        left_panel.rowconfigure(2, weight=1)
        left_panel.columnconfigure(0, weight=1)
        ttk.Label(left_panel, text="Episodes").pack(anchor=tk.W)
        edit_row = ttk.Frame(left_panel)
        edit_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Combobox(
            edit_row,
            textvariable=self.vars["edit_task"],
            values=TASK_CHOICES,
            state="readonly",
            width=10,
        ).pack(side=tk.LEFT)
        ttk.Button(edit_row, text="Set Task", command=self.set_selected_episode_task).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        columns = ("id", "length", "task")
        tree_frame = ttk.Frame(left_panel)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.episode_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=18,
            style="Episode.Treeview",
        )
        for col, width in (("id", 56), ("length", 76), ("task", 220)):
            self.episode_tree.heading(col, text=col)
            self.episode_tree.column(col, width=width, minwidth=48, anchor=tk.W, stretch=col == "task")
        episode_y_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.episode_tree.yview)
        episode_x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.episode_tree.xview)
        self.episode_tree.configure(yscrollcommand=episode_y_scroll.set, xscrollcommand=episode_x_scroll.set)
        self.episode_tree.grid(row=0, column=0, sticky=tk.NSEW)
        episode_y_scroll.grid(row=0, column=1, sticky=tk.NS)
        episode_x_scroll.grid(row=1, column=0, sticky=tk.EW)
        self.episode_tree.bind("<<TreeviewSelect>>", self.on_episode_selected)

        right_panel = ttk.Frame(content)
        right_panel.grid(row=0, column=1, sticky=tk.NSEW)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        self.video_title = tk.StringVar(value="Select a dataset folder or click Connect for live preview.")
        title_label = ttk.Label(right_panel, textvariable=self.video_title, anchor=tk.W, font=("TkDefaultFont", 10))
        title_label.grid(row=0, column=0, sticky=tk.EW)
        self.preview_label = ttk.Label(right_panel, anchor=tk.CENTER)
        self.preview_label.grid(row=1, column=0, sticky=tk.NSEW, pady=(6, 0))
        playback_row = ttk.Frame(right_panel)
        playback_row.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))
        self.play_button_text = tk.StringVar(value="Play")
        ttk.Button(playback_row, textvariable=self.play_button_text, command=self.toggle_dataset_playback).pack(
            side=tk.LEFT
        )
        ttk.Button(playback_row, text="Stop Playback", command=self.stop_dataset_playback).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        progress_row = ttk.Frame(bottom)
        progress_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(progress_row, textvariable=self.episode_text, width=26, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Progressbar(progress_row, variable=self.progress_value, maximum=100).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8)
        )
        ttk.Label(progress_row, textvariable=self.progress_text, width=48, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Label(bottom, textvariable=self.status, anchor=tk.W).pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.state_text, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

    def browse_dataset_folder(self) -> None:
        folder = self._ask_directory("Select an existing LeRobot dataset folder")
        if not folder:
            return
        self._set_dataset_folder(Path(folder))

    def new_dataset_folder(self) -> None:
        parent = self._ask_directory("Select parent folder for the new dataset")
        if not parent:
            return
        name = simpledialog.askstring("New dataset folder", "Folder name:", parent=self.root)
        if not name:
            return
        folder = Path(parent).expanduser() / name
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status.set(f"Could not create dataset folder {folder}: {exc}")
            return
        self._set_dataset_folder(folder, normalize_existing=False)
        self.status.set(f"New dataset folder created: {folder}")

    def _ask_directory(self, title: str) -> str:
        current = Path(self.vars["dataset_root"].get() or Path.home()).expanduser()
        if not current.exists():
            current = current.parent if current.parent.exists() else Path.home()

        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("920x620")
        dialog.minsize(760, 440)
        dialog.transient(self.root)
        dialog.grab_set()

        result = {"path": ""}
        path_var = tk.StringVar(value=str(current))

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        ttk.Label(frame, text=title).grid(row=0, column=0, sticky=tk.W)
        entry = ttk.Entry(frame, textvariable=path_var)
        entry.grid(row=1, column=0, sticky=tk.EW, pady=(8, 8))

        list_frame = ttk.Frame(frame)
        list_frame.grid(row=2, column=0, sticky=tk.NSEW)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        listbox = tk.Listbox(list_frame, height=10, font=tkfont.nametofont("TkTextFont"))
        listbox.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        listbox.configure(yscrollcommand=scrollbar.set)

        def refresh(path: Path | None = None) -> None:
            nonlocal current
            if path is not None:
                current = path.expanduser()
            if not current.exists() or not current.is_dir():
                return
            path_var.set(str(current))
            listbox.delete(0, tk.END)
            try:
                dirs = [p for p in current.iterdir() if p.is_dir()]
            except OSError:
                dirs = []
            for child in sorted(dirs, key=lambda p: p.name.lower()):
                listbox.insert(tk.END, child.name)

        def selected_child() -> Path | None:
            selection = listbox.curselection()
            if not selection:
                return None
            return current / listbox.get(selection[0])

        def open_selected(_event: tk.Event | None = None) -> None:
            child = selected_child()
            typed = Path(path_var.get()).expanduser()
            refresh(child if child is not None else typed)

        def choose() -> None:
            typed = Path(path_var.get()).expanduser()
            result["path"] = str(typed if typed.is_dir() else current)
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        button_row = ttk.Frame(frame)
        button_row.grid(row=3, column=0, sticky=tk.EW, pady=(10, 0))
        button_row.columnconfigure(4, weight=1)
        for col, (text, command) in enumerate(
            (
                ("Up", lambda: refresh(current.parent)),
                ("Home", lambda: refresh(Path.home())),
                ("Open", open_selected),
                ("Refresh", lambda: refresh(Path(path_var.get()))),
            )
        ):
            ttk.Button(button_row, text=text, command=command).grid(row=0, column=col, sticky=tk.W, padx=(0, 8))
        ttk.Button(button_row, text="Select", command=choose).grid(row=0, column=5, sticky=tk.E, padx=(8, 8))
        ttk.Button(button_row, text="Cancel", command=cancel).grid(row=0, column=6, sticky=tk.E)

        listbox.bind("<Double-Button-1>", open_selected)
        entry.bind("<Return>", lambda _event: refresh(Path(path_var.get())))
        refresh(current)
        dialog.wait_window()
        return result["path"]

    def _set_dataset_folder(self, folder: Path, normalize_existing: bool = True) -> None:
        if self.cleanup_running:
            self.status.set("Dataset cleanup is still running. Wait for it to finish before changing folders.")
            return
        self.stop_dataset_playback()
        self.close_dataset_captures()
        selected = folder.expanduser()
        dataset_root = self._find_dataset_root(selected) if normalize_existing else selected
        if dataset_root != selected:
            self.status.set(f"Selected a subfolder; using dataset root: {dataset_root}")
        self.vars["dataset_root"].set(str(dataset_root))
        self.vars["repo_id"].set(self._infer_repo_id(dataset_root))
        health = self._show_dataset_folder_summary(dataset_root)
        self._remember_cleanup_choice(dataset_root, health)
        self._load_episode_browser(dataset_root)

    def _load_episode_browser(self, dataset_root: Path) -> None:
        self.stop_dataset_playback()
        self.close_dataset_captures()
        self.dataset_episodes.clear()
        self.dataset_info = {}
        self.dataset_video_keys = []
        self.current_dataset_episode = None
        self.episode_tree.delete(*self.episode_tree.get_children())
        self.video_mode = "dataset"
        self.video_title.set("No playable dataset selected.")
        self.preview_label.configure(image="")
        self.dataset_image = None

        info_path = dataset_root / "meta" / "info.json"
        if not info_path.exists():
            self.video_title.set("New dataset folder. Connect to start live preview.")
            return
        try:
            self.dataset_info = json.loads(info_path.read_text())
            episodes_df = read_nested_parquets(dataset_root / "meta" / "episodes").sort_values("episode_index")
        except Exception as exc:
            self.status.set(f"Could not load episode browser: {exc}")
            self.video_title.set("Dataset metadata could not be loaded.")
            return

        self.dataset_video_keys = video_keys_from_info(self.dataset_info)
        if self.dataset_video_keys and not (self.recorder is not None and self.recorder.is_alive()):
            self.preview_combo["values"] = self.dataset_video_keys
            self.vars["preview_key"].set(self.dataset_video_keys[0])
        for _, row in episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            ep = EpisodeBrowserRef(ep_idx, row)
            self.dataset_episodes[ep_idx] = ep
            self.episode_tree.insert(
                "",
                tk.END,
                iid=str(ep_idx),
                values=(ep_idx, int(row.get("length", 0)), self._task_text(row)),
            )
        if self.dataset_episodes:
            self.video_title.set(
                f"Loaded {len(self.dataset_episodes)} episode(s). Select one on the left to play."
            )
        else:
            self.video_title.set("Dataset has no episode metadata yet.")

    @staticmethod
    def _task_text(row: Any) -> str:
        tasks = row.get("tasks", [])
        if isinstance(tasks, np.ndarray):
            tasks = tasks.tolist()
        if isinstance(tasks, list):
            return ", ".join(str(task) for task in tasks)
        return str(tasks)

    def on_episode_selected(self, _event: tk.Event) -> None:
        selection = self.episode_tree.selection()
        if not selection:
            return
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Live recording/preview is active. Stop recorder before browsing saved episodes.")
            return
        episode = self.dataset_episodes.get(int(selection[0]))
        if episode is None:
            return
        task_text = self._task_text(episode.row)
        if task_text in TASK_CHOICES:
            self.vars["edit_task"].set(task_text)
        self.select_dataset_episode(episode, autoplay=True)

    def set_selected_episode_task(self) -> None:
        selection = self.episode_tree.selection()
        if not selection:
            self.status.set("Select an episode before setting its task.")
            return
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Stop recorder before editing saved episode task labels.")
            return
        episode_index = int(selection[0])
        new_task = self.vars["edit_task"].get()
        if new_task not in TASK_CHOICES:
            self.status.set(f"Choose one of: {', '.join(TASK_CHOICES)}")
            return
        dataset_root = Path(self.vars["dataset_root"].get()).expanduser()
        try:
            self._write_episode_task(dataset_root, episode_index, new_task)
        except Exception as exc:
            self.status.set(f"ERROR: failed to update episode {episode_index} task: {exc}")
            return
        self.status.set(f"Updated episode {episode_index} task to {new_task}.")
        self._load_episode_browser(dataset_root)
        if str(episode_index) in self.episode_tree.get_children():
            self.episode_tree.selection_set(str(episode_index))

    def _write_episode_task(self, dataset_root: Path, episode_index: int, new_task: str) -> None:
        info_path = dataset_root / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        task_index = self._ensure_task_index(dataset_root, info, new_task)

        episodes_df = read_nested_parquets(dataset_root / "meta" / "episodes").sort_values("episode_index")
        matching = episodes_df[episodes_df["episode_index"] == episode_index]
        if matching.empty:
            raise ValueError(f"episode {episode_index} was not found")
        row = matching.iloc[0]

        episode_meta_path = dataset_root / f"meta/episodes/chunk-{int(row['meta/episodes/chunk_index']):03d}/file-{int(row['meta/episodes/file_index']):03d}.parquet"
        episode_file_df = pd.read_parquet(episode_meta_path)
        episode_mask = episode_file_df["episode_index"] == episode_index
        if not episode_mask.any():
            raise ValueError(f"episode {episode_index} was not found in {episode_meta_path}")
        episode_file_df.loc[episode_mask, "tasks"] = pd.Series([[new_task]] * int(episode_mask.sum()), index=episode_file_df.index[episode_mask])
        episode_file_df.to_parquet(episode_meta_path, index=False)

        data_path = dataset_root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )
        data_df = pd.read_parquet(data_path)
        data_mask = data_df["episode_index"] == episode_index
        if not data_mask.any():
            raise ValueError(f"episode {episode_index} has no frame rows in {data_path}")
        data_df.loc[data_mask, "task_index"] = int(task_index)
        data_df.to_parquet(data_path, index=False)

    @staticmethod
    def _ensure_task_index(dataset_root: Path, info: dict[str, Any], task: str) -> int:
        tasks_path = dataset_root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            tasks_df = pd.read_parquet(tasks_path)
        else:
            tasks_df = pd.DataFrame({"task_index": []})
        if task in tasks_df.index:
            return int(tasks_df.loc[task, "task_index"])
        next_index = int(tasks_df["task_index"].max()) + 1 if len(tasks_df) else 0
        tasks_df.loc[task, "task_index"] = next_index
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_df.to_parquet(tasks_path)
        info["total_tasks"] = int(len(tasks_df))
        (dataset_root / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
        return next_index

    def select_dataset_episode(self, episode: EpisodeBrowserRef, autoplay: bool = False) -> None:
        self.stop_dataset_playback()
        self.close_dataset_captures()
        self.video_mode = "dataset"
        self.current_dataset_episode = episode
        row = episode.row
        length = int(row.get("length", 0))
        fps = int(self.dataset_info.get("fps", FPS))
        self.video_title.set(
            f"Episode {episode.episode_index} | {length} frames | {length / max(1, fps):.1f}s | {self._task_text(row)}"
        )
        self.seek_dataset_episode(0.0)
        if autoplay:
            self.start_dataset_playback()

    def dataset_video_path_for(self, row: Any, video_key: str) -> Path:
        dataset_root = Path(self.vars["dataset_root"].get()).expanduser()
        return dataset_root / self.dataset_info["video_path"].format(
            video_key=video_key,
            chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
            file_index=int(row[f"videos/{video_key}/file_index"]),
        )

    def open_dataset_capture(self, video_key: str) -> cv2.VideoCapture | None:
        if self.current_dataset_episode is None:
            return None
        if video_key in self.dataset_captures:
            return self.dataset_captures[video_key]
        path = self.dataset_video_path_for(self.current_dataset_episode.row, video_key)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.status.set(f"Failed to open video: {path}")
            return None
        self.dataset_captures[video_key] = cap
        return cap

    def active_dataset_video_key(self) -> str:
        selected = self.vars["preview_key"].get()
        if selected in self.dataset_video_keys:
            return selected
        return self.dataset_video_keys[0] if self.dataset_video_keys else ""

    def seek_dataset_episode(self, rel_s: float) -> None:
        if self.current_dataset_episode is None or not self.dataset_video_keys:
            return
        video_key = self.active_dataset_video_key()
        if not video_key:
            return
        cap = self.open_dataset_capture(video_key)
        if cap is None:
            return
        row = self.current_dataset_episode.row
        fps = int(self.dataset_info.get("fps", FPS))
        start_s = float(row[f"videos/{video_key}/from_timestamp"])
        end_s = float(row[f"videos/{video_key}/to_timestamp"])
        self.play_offset_s = max(0.0, rel_s)
        target_s = min(start_s + self.play_offset_s, max(start_s, end_s - 1.0 / max(1, fps)))
        cap.set(cv2.CAP_PROP_POS_MSEC, target_s * 1000.0)
        ok, frame = cap.read()
        if ok:
            self._show_bgr_frame(frame)

    def start_dataset_playback(self) -> None:
        if self.current_dataset_episode is None:
            return
        self.video_mode = "dataset"
        self.playing_dataset = True
        self.play_started_at = time.monotonic() - self.play_offset_s
        self.play_button_text.set("Pause")

    def stop_dataset_playback(self) -> None:
        self.playing_dataset = False
        if hasattr(self, "play_button_text"):
            self.play_button_text.set("Play")

    def toggle_dataset_playback(self) -> None:
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Live preview is active. Stop recorder before playing saved episodes.")
            return
        if self.playing_dataset:
            self.stop_dataset_playback()
        else:
            self.start_dataset_playback()

    def close_dataset_captures(self) -> None:
        for cap in self.dataset_captures.values():
            cap.release()
        self.dataset_captures.clear()

    @staticmethod
    def _find_dataset_root(path: Path) -> Path:
        path = path.expanduser()
        if (path / "meta" / "info.json").exists():
            return path
        if not path.exists():
            return path
        if path.is_dir() and not any(path.iterdir()):
            return path
        for candidate in path.parents:
            if not (candidate / "meta" / "info.json").exists():
                continue
            try:
                relative_parts = path.relative_to(candidate).parts
            except ValueError:
                continue
            if relative_parts and relative_parts[0] in {"meta", "data", "videos"}:
                return candidate
        return path

    @staticmethod
    def _infer_repo_id(dataset_root: Path) -> str:
        root = dataset_root.expanduser()
        try:
            return root.relative_to(HF_LEROBOT_HOME).as_posix()
        except ValueError:
            pass
        return root.name or "local_dataset"

    def _show_dataset_folder_summary(self, folder: Path) -> DatasetHealth | None:
        info_path = folder / "meta" / "info.json"
        if not info_path.exists():
            self.status.set(f"Dataset folder selected: {folder} (new dataset will be created)")
            return None
        try:
            with info_path.open() as f:
                info = json.load(f)
            episodes = int(info.get("total_episodes", 0))
            frames = int(info.get("total_frames", 0))
        except (json.JSONDecodeError, ValueError):
            self.status.set(f"Dataset folder selected: {folder} (metadata is not readable)")
            return None
        health = BimanualRecorder._inspect_dataset_health(folder)
        if health.needs_rewrite:
            self.status.set(
                f"Dataset issue detected: {health.reason}. Click Clean to repair, or Browse/New to choose another folder."
            )
        else:
            self.status.set(f"Dataset folder selected: {folder} ({episodes} clean episodes, {frames} frames)")
        return health

    def _remember_cleanup_choice(self, dataset_root: Path, health: DatasetHealth | None) -> None:
        if health is not None and health.needs_rewrite and health.valid_episode_indices:
            self.pending_cleanup_root = dataset_root
            self.pending_cleanup_episodes = health.valid_episode_indices
            self.progress_value.set(0.0)
            self.progress_text.set(f"Needs cleanup: keep {len(health.valid_episode_indices)} valid episode(s)")
            return
        self.pending_cleanup_root = None
        self.pending_cleanup_episodes = []

    def clean_dataset(self) -> None:
        dataset_root = Path(self.vars["dataset_root"].get()).expanduser()
        health = self._show_dataset_folder_summary(dataset_root)
        self._remember_cleanup_choice(dataset_root, health)
        if health is None:
            self.status.set("No existing LeRobot dataset metadata found here. Choose a dataset folder first.")
            return
        if not health.needs_rewrite:
            self.status.set("Dataset is already clean. No cleanup needed.")
            return
        if not health.valid_episode_indices:
            self.status.set("No valid episode can be kept. Choose another folder or create a new dataset.")
            return
        if not messagebox.askyesno(
            "Clean dataset?",
            (
                f"Repair this dataset and keep {len(health.valid_episode_indices)} valid episode(s)?\n\n"
                "A timestamped backup folder will be created next to the dataset before replacement."
            ),
            parent=self.root,
        ):
            self.status.set("Cleanup cancelled. You can choose another folder.")
            return
        self._start_dataset_cleanup(dataset_root, health.valid_episode_indices)

    def _start_dataset_cleanup(self, dataset_root: Path, keep_episode_indices: list[int]) -> None:
        if self.cleanup_running:
            self.status.set("Dataset cleanup is already running. Wait for it to finish before selecting another folder.")
            return
        self.cleanup_running = True
        self.status_queue.put(f"PROGRESS|0|100|Cleaning dataset...")
        self.status_queue.put(f"Cleaning selected dataset and keeping {len(keep_episode_indices)} valid episode(s)...")

        def worker() -> None:
            try:
                backup = rewrite_dataset(dataset_root, keep_episode_indices, self.status_queue)
                self.status_queue.put(f"CLEAN_DONE|{dataset_root}|{backup}")
            except Exception as exc:
                self.status_queue.put(f"CLEAN_ERROR|{dataset_root}|{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def connect(self) -> None:
        if self.cleanup_running:
            self.status.set("Dataset cleanup is still running. Connect after the progress bar says complete.")
            return
        if self.recorder is not None:
            if self.recorder.ready:
                self.status.set("Recorder is already connected and ready.")
                return
            if self.recorder.is_alive():
                self.status.set("Recorder is still connecting. Watch this status line or the terminal.")
                return
            self.status.set("Previous recorder stopped or failed. Creating a new one...")
            self.recorder = None
        dataset_root = self.vars["dataset_root"].get().strip()
        if not dataset_root:
            self.status.set("Choose a dataset folder before Connect.")
            return
        normalized_root = self._find_dataset_root(Path(dataset_root).expanduser())
        if str(normalized_root) != dataset_root:
            self.vars["dataset_root"].set(str(normalized_root))
            self.vars["repo_id"].set(self._infer_repo_id(normalized_root))
        health = self._show_dataset_folder_summary(normalized_root)
        self._remember_cleanup_choice(normalized_root, health)
        if health is not None and health.needs_rewrite:
            if health.valid_episode_indices:
                self.status.set(
                    f"Dataset issue detected: {health.reason}. Click Clean before Connect, or choose another folder."
                )
            else:
                self.status.set(
                    f"Dataset issue detected: {health.reason}. No valid episode can be kept; choose another folder or New."
                )
            return
        settings = RecorderSettings(
            left_follower_port=self.vars["left_follower_port"].get(),
            right_follower_port=self.vars["right_follower_port"].get(),
            left_leader_port=self.vars["left_leader_port"].get(),
            right_leader_port=self.vars["right_leader_port"].get(),
            left_follower_id=self.vars["left_follower_id"].get(),
            right_follower_id=self.vars["right_follower_id"].get(),
            left_leader_id=self.vars["left_leader_id"].get(),
            right_leader_id=self.vars["right_leader_id"].get(),
            opencv_front=self.vars["opencv_front"].get(),
            opencv_side=self.vars["opencv_side"].get(),
            enable_realsense=self.enable_realsense.get(),
            realsense_serial=self.vars["realsense_serial"].get(),
            repo_id=self.vars["repo_id"].get(),
            dataset_root=self.vars["dataset_root"].get().strip(),
            calibration_dir=self.vars["calibration_dir"].get(),
            task=self.vars["task"].get(),
            resume=self.resume.get(),
            push_to_hub=self.push_to_hub.get(),
        )
        self.status.set("Connect clicked. Building devices and dataset...")
        self.root.update_idletasks()
        self.stop_dataset_playback()
        self.close_dataset_captures()
        self.video_mode = "live"
        self.video_title.set("Live preview starting...")
        self.recorder = BimanualRecorder(settings, self.status_queue, self.preview_queue, self.state_queue)
        self.recorder.start()

    def record(self) -> None:
        if self.recorder is None:
            self.status.set("Click Connect first.")
            return
        if not self.recorder.ready:
            self.status.set("Not ready yet. Wait for: Ready. Preview and teleoperation are running.")
            return
        task = self.vars["task"].get()
        if task not in TASK_CHOICES:
            self.status.set(f"Choose a valid task before recording: {', '.join(TASK_CHOICES)}")
            return
        self.recorder.request_record(task)

    def end_episode(self) -> None:
        if self.recorder is not None:
            self.recorder.request_end_episode()

    def discard_episode(self) -> None:
        if self.recorder is None:
            self.status.set("Click Connect first.")
            return
        self.recorder.request_discard_episode()

    def stop_recorder(self) -> None:
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None
            self.video_mode = "dataset"
            self._load_episode_browser(Path(self.vars["dataset_root"].get()).expanduser())

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not bool(self.root.attributes("-fullscreen")))

    def set_fullscreen(self, enabled: bool) -> None:
        self.root.attributes("-fullscreen", enabled)

    def _poll_queues(self) -> None:
        while True:
            try:
                self._handle_status_message(self.status_queue.get_nowait())
            except queue.Empty:
                break

        if self.recorder is not None and not self.recorder.is_alive() and not self.recorder.ready:
            if self.recorder.failed:
                self.status.set(f"{self.status.get()} You can fix the setting and click Connect again.")
            self.recorder = None

        try:
            self.latest_frames = self.preview_queue.get_nowait()
            keys = sorted(self.latest_frames)
            self.preview_combo["values"] = keys
            if (not self.vars["preview_key"].get() or self.vars["preview_key"].get() not in keys) and keys:
                self.vars["preview_key"].set(keys[0])
            self.video_mode = "live"
        except queue.Empty:
            pass

        key = self.vars["preview_key"].get()
        if self.video_mode == "live" and key in self.latest_frames:
            self.video_title.set(f"Live preview: {key}")
            self._show_frame(self.latest_frames[key])

        if self.playing_dataset and self.current_dataset_episode is not None:
            length_s = float(self.current_dataset_episode.row.get("length", 0)) / int(
                self.dataset_info.get("fps", FPS)
            )
            rel_s = time.monotonic() - self.play_started_at
            if rel_s >= length_s:
                self.stop_dataset_playback()
                rel_s = 0.0
            self.seek_dataset_episode(rel_s)

        try:
            state = self.state_queue.get_nowait()
            self._show_state(state)
        except queue.Empty:
            pass

        self.root.after(30, self._poll_queues)

    def _handle_status_message(self, message: str) -> None:
        if message.startswith("PROGRESS|"):
            try:
                _kind, current, total, detail = message.split("|", 3)
                current_i = int(current)
                total_i = max(1, int(total))
                self.progress_value.set(current_i * 100.0 / total_i)
                self.progress_text.set(f"{current_i}/{total_i} {detail}")
            except ValueError:
                self.status.set(message)
            return
        if message.startswith("EPISODE|"):
            try:
                _kind, episode, recording = message.split("|", 2)
                label = "Recording" if recording == "1" else "Next episode"
                self.episode_text.set(f"{label}: {episode}")
            except ValueError:
                self.status.set(message)
            return
        if message.startswith("CLEAN_DONE|"):
            try:
                _kind, dataset_root, backup = message.split("|", 2)
                self.cleanup_running = False
                self.pending_cleanup_root = None
                self.pending_cleanup_episodes = []
                self.progress_value.set(100.0)
                self.progress_text.set("Cleanup complete")
                health = self._show_dataset_folder_summary(Path(dataset_root))
                self._remember_cleanup_choice(Path(dataset_root), health)
                self._load_episode_browser(Path(dataset_root))
                self.status.set(f"Dataset cleaned successfully. Backup: {backup}")
            except ValueError:
                self.status.set(message)
            return
        if message.startswith("CLEAN_ERROR|"):
            try:
                _kind, _dataset_root, error = message.split("|", 2)
                self.cleanup_running = False
                self.progress_value.set(0.0)
                self.progress_text.set("Cleanup failed")
                self.status.set(f"ERROR: dataset cleanup failed: {error}")
            except ValueError:
                self.status.set(message)
            return
        self.status.set(message)

    def _show_frame(self, frame: np.ndarray) -> None:
        image = Image.fromarray(np.ascontiguousarray(frame))
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.current_image)

    def _show_bgr_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.ascontiguousarray(frame_rgb))
        image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.dataset_image = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.dataset_image)

    def _show_state(self, state: dict[str, float]) -> None:
        items = sorted(state.items())[:18]
        text = " | ".join(f"{key}: {value:.1f}" for key, value in items)
        self.state_text.set(text)

    def close(self) -> None:
        if self.cleanup_running:
            self.status.set("Dataset cleanup is still running. Wait until it finishes before closing.")
            return
        self.stop_dataset_playback()
        self.close_dataset_captures()
        self.stop_recorder()
        self.root.destroy()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except AttributeError:
            pass
    init_logging()
    root = tk.Tk()
    RecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
