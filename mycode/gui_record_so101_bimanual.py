#!/usr/bin/env python

"""GUI recorder for bimanual SO101 teleoperation with RGB and RealSense depth cameras.

Run from the LeRobot conda environment:
    conda run -n lerobot python mycode/gui_record_so101_bimanual.py
"""

from __future__ import annotations

import concurrent.futures
import os
import queue
import json
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk
from types import MethodType
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

CURRENT_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(CURRENT_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_SRC))

try:
    from mycode.gui_view_lerobot_dataset import (
        read_nested_parquets,
        rewrite_dataset,
        stats_from_episode_row,
        video_keys_from_info,
    )
except ModuleNotFoundError:
    from gui_view_lerobot_dataset import read_nested_parquets, rewrite_dataset, stats_from_episode_row, video_keys_from_info

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts, write_stats
from lerobot.datasets.video_utils import VideoEncodingManager, get_video_duration_in_s
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.utils import init_logging


FPS = 30
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
LOW_CAMERA_WIDTH = 640
LOW_CAMERA_HEIGHT = 360
PREVIEW_SIZE = (1280, 720)
SIDE_PREVIEW_SIZE = (640, 360)
DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "calibration"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "bettersetup"
DEFAULT_LEFT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E122511-if00"
DEFAULT_LEFT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E118729-if00"
DEFAULT_RIGHT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E119029-if00"
DEFAULT_RIGHT_LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E121504-if00"
DEFAULT_FRONT_CAMERA_ID = "v4l2-serial://239222303378/rgb"
DEFAULT_SIDE_CAMERA_ID = "v4l2-serial://202412231836/rgb"
V4L2_SERIAL_PREFIX = "v4l2-serial://"
TASK_CHOICES = ("cube", "screw", "paperball", "cube_r", "screw_r", "paperball_r")
EPISODE_FILE_SIZE_MB = 1e-6


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
    """Camera-like view that exposes colorized depth for preview/training video features."""

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
        if isinstance(self.parent, RealSenseCamera):
            depth = self.parent.read_depth(timeout_ms=timeout_ms)
        else:
            with self.parent.frame_lock:
                depth = self.parent.latest_depth_preview_frame
            if depth is None:
                depth = self.parent.read_depth(timeout_ms=timeout_ms)
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
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, FPS)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, FPS)
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
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="bimanual-follower-io"
        )

    @property
    def global_camera_keys(self) -> set[str]:
        return set(self.cameras)

    @property
    def robot_type(self) -> str:
        return self.name

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {
            **{
                key if key in self.global_camera_keys else f"left_{key}": value
                for key, value in self.left_arm.observation_features.items()
            },
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
        futures: dict[concurrent.futures.Future, tuple[str, str]] = {}
        futures[self._io_executor.submit(self._read_arm_motors, "left follower", self.left_arm)] = (
            "left_motors",
            "",
        )
        futures[self._io_executor.submit(self._read_arm_motors, "right follower", self.right_arm)] = (
            "right_motors",
            "",
        )

        delayed_depth_views: dict[str, RealSenseDepthView] = {}
        for cam_key, cam in self.cameras.items():
            if isinstance(cam, RealSenseDepthView):
                delayed_depth_views[cam_key] = cam
            else:
                futures[self._io_executor.submit(cam.async_read)] = ("camera", cam_key)

        left_obs: dict[str, Any] = {}
        right_obs: dict[str, Any] = {}
        camera_obs: dict[str, Any] = {}
        for future in concurrent.futures.as_completed(futures):
            kind, key = futures[future]
            if kind == "left_motors":
                left_obs = future.result()
            elif kind == "right_motors":
                right_obs = future.result()
            else:
                camera_obs[key] = future.result()

        for cam_key, cam in delayed_depth_views.items():
            camera_obs[cam_key] = cam.async_read()

        return {
            **{f"left_{key}": value for key, value in left_obs.items()},
            **{f"right_{key}": value for key, value in right_obs.items()},
            **camera_obs,
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        left_action = {key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")}
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }
        left_future = self._io_executor.submit(self._send_arm_action, "left follower", self.left_arm, left_action)
        right_future = self._io_executor.submit(self._send_arm_action, "right follower", self.right_arm, right_action)
        sent_left = left_future.result()
        sent_right = right_future.result()
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    @staticmethod
    def _read_arm_motors(label: str, arm: SO101Follower) -> dict[str, float]:
        try:
            obs_dict = arm.bus.sync_read("Present_Position")
        except Exception as exc:
            raise RuntimeError(f"{label} failed to read Present_Position: {exc}") from exc
        return {f"{motor}.pos": val for motor, val in obs_dict.items()}

    @staticmethod
    def _send_arm_action(label: str, arm: SO101Follower, action: dict[str, float]) -> dict[str, float]:
        try:
            return arm.send_action(action)
        except Exception as exc:
            raise RuntimeError(f"{label} failed to send action: {exc}") from exc

    def disconnect(self) -> None:
        self._disconnect_arm_safely("left follower", self.left_arm)
        self._disconnect_arm_safely("right follower", self.right_arm)
        self._io_executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _disconnect_arm_safely(label: str, arm: SO101Follower) -> None:
        try:
            arm.disconnect()
            return
        except Exception as exc:
            print(f"[GUI recorder] Warning: normal disconnect failed for {label}: {exc}", flush=True)

        LocalBimanualSOFollower._disable_torque_best_effort(label, arm)

        try:
            arm.bus.disconnect(disable_torque=False)
        except Exception as exc:
            print(f"[GUI recorder] Warning: force-closing bus failed for {label}: {exc}", flush=True)
            try:
                arm.bus.port_handler.closePort()
            except Exception as close_exc:
                print(f"[GUI recorder] Warning: closing bus port failed for {label}: {close_exc}", flush=True)

        for name, cam in getattr(arm, "cameras", {}).items():
            try:
                if getattr(cam, "is_connected", False):
                    cam.disconnect()
            except Exception as exc:
                print(f"[GUI recorder] Warning: disconnecting camera {label}/{name} failed: {exc}", flush=True)

    @staticmethod
    def _disable_torque_best_effort(label: str, arm: SO101Follower) -> None:
        bus = arm.bus
        if not getattr(bus, "is_connected", False):
            return
        for motor in bus.motors:
            try:
                bus.disable_torque(motor, num_retry=2)
            except Exception as exc:
                print(f"[GUI recorder] Warning: disabling torque failed for {label}/{motor}: {exc}", flush=True)


class LocalBimanualSOLeader:
    name = "bi_so_leader"

    def __init__(self, left_arm: SO101Leader, right_arm: SO101Leader) -> None:
        self.left_arm = left_arm
        self.right_arm = right_arm
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="bimanual-leader-io"
        )

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self) -> None:
        self.left_arm.connect()
        self.right_arm.connect()

    def get_action(self) -> dict[str, float]:
        left_future = self._io_executor.submit(self._get_arm_action, "left leader", self.left_arm)
        right_future = self._io_executor.submit(self._get_arm_action, "right leader", self.right_arm)
        left_action = left_future.result()
        right_action = right_future.result()
        return {
            **{f"left_{key}": value for key, value in left_action.items()},
            **{f"right_{key}": value for key, value in right_action.items()},
        }

    @staticmethod
    def _get_arm_action(label: str, arm: SO101Leader) -> dict[str, float]:
        try:
            return arm.get_action()
        except Exception as exc:
            raise RuntimeError(f"{label} failed to read action: {exc}") from exc

    def disconnect(self) -> None:
        self._disconnect_arm_safely("left leader", self.left_arm)
        self._disconnect_arm_safely("right leader", self.right_arm)
        self._io_executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _disconnect_arm_safely(label: str, arm: SO101Leader) -> None:
        try:
            arm.disconnect()
        except Exception as exc:
            print(f"[GUI recorder] Warning: disconnect failed for {label}: {exc}", flush=True)
            LocalBimanualSOLeader._disable_torque_best_effort(label, arm)
            try:
                arm.bus.port_handler.closePort()
            except Exception as close_exc:
                print(f"[GUI recorder] Warning: closing bus port failed for {label}: {close_exc}", flush=True)

    @staticmethod
    def _disable_torque_best_effort(label: str, arm: SO101Leader) -> None:
        bus = arm.bus
        if not getattr(bus, "is_connected", False):
            return
        for motor in bus.motors:
            try:
                bus.disable_torque(motor, num_retry=2)
            except Exception as exc:
                print(f"[GUI recorder] Warning: disabling torque failed for {label}/{motor}: {exc}", flush=True)


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
    record_opencv_front: bool
    record_opencv_side: bool
    enable_realsense: bool
    capture_depth_sidecar: bool
    low_resolution: bool
    disable_gripper: bool
    realsense_serial: str
    repo_id: str
    dataset_root: str
    calibration_dir: str
    calibration_match: str
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


@dataclass(frozen=True)
class DatasetRewriteOutcome:
    backup_root: Path
    mode: str


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp_{os.getpid()}_{time.monotonic_ns()}")
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp_{os.getpid()}_{time.monotonic_ns()}")
    try:
        tmp_path.write_text(json.dumps(data, indent=4) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _hardlink_backup_tree(root: Path) -> Path:
    backup_root = root.parent / f"{root.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if backup_root.exists():
        raise FileExistsError(f"Backup already exists: {backup_root}")
    try:
        shutil.copytree(root, backup_root, copy_function=os.link)
    except Exception:
        if backup_root.exists():
            shutil.rmtree(backup_root)
        shutil.copytree(root, backup_root)
    return backup_root


def _episode_data_path(root: Path, info: dict[str, Any], row: Any) -> Path:
    return root / info["data_path"].format(
        chunk_index=int(row["data/chunk_index"]),
        file_index=int(row["data/file_index"]),
    )


def _episode_meta_path(root: Path, row: Any) -> Path:
    return root / f"meta/episodes/chunk-{int(row['meta/episodes/chunk_index']):03d}/file-{int(row['meta/episodes/file_index']):03d}.parquet"


def _episode_video_path(root: Path, info: dict[str, Any], row: Any, video_key: str) -> Path:
    return root / info["video_path"].format(
        video_key=video_key,
        chunk_index=int(row[f"videos/{video_key}/chunk_index"]),
        file_index=int(row[f"videos/{video_key}/file_index"]),
    )


def _expected_data_path(root: Path, info: dict[str, Any], episode_index: int, chunks_size: int) -> Path:
    return root / info["data_path"].format(
        chunk_index=episode_index // chunks_size,
        file_index=episode_index % chunks_size,
    )


def _expected_meta_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    return root / f"meta/episodes/chunk-{episode_index // chunks_size:03d}/file-{episode_index % chunks_size:03d}.parquet"


def _expected_video_path(
    root: Path, info: dict[str, Any], video_key: str, episode_index: int, chunks_size: int
) -> Path:
    return root / info["video_path"].format(
        video_key=video_key,
        chunk_index=episode_index // chunks_size,
        file_index=episode_index % chunks_size,
    )


def _sidecar_episode_paths(root: Path, episode_index: int) -> list[Path]:
    sidecar_root = root / "sidecar_depth"
    if not sidecar_root.exists():
        return []
    return sorted(sidecar_root.glob(f"**/episode_{episode_index:06d}.*"))


def _can_fast_reindex(
    root: Path,
    info: dict[str, Any],
    episodes_df: pd.DataFrame,
    keep_episode_indices: list[int],
    video_keys: list[str],
) -> tuple[bool, str]:
    chunks_size = int(info.get("chunks_size", 1000))
    all_indices = [int(ep) for ep in episodes_df["episode_index"].tolist()]
    if all_indices != list(range(len(all_indices))):
        return False, "episode indices are not contiguous 0..N-1"
    keep_set = set(keep_episode_indices)
    if keep_episode_indices != sorted(keep_set):
        return False, "keep episode list is not sorted and unique"
    if not keep_set or not keep_set.issubset(set(all_indices)):
        return False, "keep episode list is empty or contains missing episodes"
    if keep_episode_indices == all_indices:
        return False, "no episode deletion was requested"
    if info.get("data_path") != "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet":
        return False, "data path format is not the expected episode-per-file layout"
    if info.get("video_path") != "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4":
        return False, "video path format is not the expected episode-per-file layout"

    seen_data_paths: set[Path] = set()
    seen_meta_paths: set[Path] = set()
    seen_video_paths: set[Path] = set()
    fps = int(info["fps"])
    for _, row in episodes_df.iterrows():
        ep_idx = int(row["episode_index"])
        expected_chunk = ep_idx // chunks_size
        expected_file = ep_idx % chunks_size
        if int(row["data/chunk_index"]) != expected_chunk or int(row["data/file_index"]) != expected_file:
            return False, f"episode {ep_idx} data file does not match episode-per-file index"
        if (
            int(row["meta/episodes/chunk_index"]) != expected_chunk
            or int(row["meta/episodes/file_index"]) != expected_file
        ):
            return False, f"episode {ep_idx} metadata file does not match episode-per-file index"

        data_path = _episode_data_path(root, info, row)
        meta_path = _episode_meta_path(root, row)
        if data_path in seen_data_paths or meta_path in seen_meta_paths:
            return False, f"episode {ep_idx} shares data or metadata file with another episode"
        seen_data_paths.add(data_path)
        seen_meta_paths.add(meta_path)
        if not data_path.exists() or not meta_path.exists():
            return False, f"episode {ep_idx} data or metadata file is missing"

        try:
            data_df = pd.read_parquet(data_path)
            meta_df = pd.read_parquet(meta_path)
        except Exception as exc:
            return False, f"episode {ep_idx} parquet is unreadable: {exc}"
        if set(data_df["episode_index"].astype(int).unique()) != {ep_idx}:
            return False, f"episode {ep_idx} data file contains another episode"
        if len(data_df) != int(row["length"]):
            return False, f"episode {ep_idx} data length does not match metadata"
        if set(meta_df["episode_index"].astype(int).unique()) != {ep_idx} or len(meta_df) != 1:
            return False, f"episode {ep_idx} metadata file is not one row for this episode"

        for video_key in video_keys:
            video_path = _episode_video_path(root, info, row, video_key)
            expected_video_path = _expected_video_path(root, info, video_key, ep_idx, chunks_size)
            if video_path != expected_video_path:
                return False, f"episode {ep_idx} video file does not match episode-per-file index"
            if video_path in seen_video_paths:
                return False, f"episode {ep_idx} shares video file with another episode"
            seen_video_paths.add(video_path)
            if not video_path.exists():
                return False, f"episode {ep_idx} video is missing: {video_path}"
            if abs(float(row[f"videos/{video_key}/from_timestamp"])) > 1e-6:
                return False, f"episode {ep_idx} video {video_key} does not start at timestamp 0"
            expected_end = int(row["length"]) / fps
            actual_end = float(row[f"videos/{video_key}/to_timestamp"])
            if abs(actual_end - expected_end) > max(0.05, 2.0 / fps):
                return False, f"episode {ep_idx} video {video_key} duration metadata is not episode-local"
    return True, "episode-per-file layout confirmed"


def _move_sidecar_episode_files(root: Path, old_ep: int, new_ep: int) -> None:
    for old_path in _sidecar_episode_paths(root, old_ep):
        new_path = old_path.with_name(f"episode_{new_ep:06d}{old_path.suffix}")
        if old_path == new_path:
            continue
        if new_path.exists():
            new_path.unlink()
        old_path.rename(new_path)


def _delete_episode_files(root: Path, info: dict[str, Any], row: Any, video_keys: list[str]) -> None:
    for path in [_episode_data_path(root, info, row), _episode_meta_path(root, row)]:
        if path.exists():
            path.unlink()
    for video_key in video_keys:
        path = _episode_video_path(root, info, row, video_key)
        if path.exists():
            path.unlink()
    for path in _sidecar_episode_paths(root, int(row["episode_index"])):
        path.unlink()


def _validate_fast_reindexed_dataset(root: Path, info: dict[str, Any]) -> None:
    health = BimanualRecorder._inspect_dataset_health(root)
    if health.needs_rewrite:
        raise RuntimeError(f"fast reindex validation failed: {health.reason}")
    episodes_df = read_nested_parquets(root / "meta" / "episodes").sort_values("episode_index")
    if int(info.get("total_episodes", -1)) != len(episodes_df):
        raise RuntimeError("fast reindex validation failed: info.json episode count mismatch")


def fast_delete_or_rewrite_dataset(
    root: Path, keep_episode_indices: list[int], progress: queue.Queue[str]
) -> DatasetRewriteOutcome:
    root = root.resolve()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    episodes_df = read_nested_parquets(root / "meta" / "episodes").sort_values("episode_index").reset_index(drop=True)
    video_keys = video_keys_from_info(info)
    can_fast, reason = _can_fast_reindex(root, info, episodes_df, keep_episode_indices, video_keys)
    if not can_fast:
        progress.put(f"Fast delete skipped: {reason}. Falling back to full rewrite.")
        backup = rewrite_dataset(root, keep_episode_indices, progress)
        return DatasetRewriteOutcome(backup, "full rewrite")

    chunks_size = int(info.get("chunks_size", 1000))
    fps = int(info["fps"])
    keep_set = set(keep_episode_indices)
    rows_by_old = {int(row["episode_index"]): row for _, row in episodes_df.iterrows()}
    deleted_indices = [ep for ep in rows_by_old if ep not in keep_set]
    old_to_new = {old_ep: new_ep for new_ep, old_ep in enumerate(keep_episode_indices)}
    min_changed = min(deleted_indices)
    affected = [old_ep for old_ep in keep_episode_indices if old_to_new[old_ep] != old_ep]
    total_steps = len(deleted_indices) + len(affected) * 3 + 4
    step = 0

    progress.put(f"PROGRESS|{step}|{total_steps}|Fast delete verified: {reason}")
    backup_root = _hardlink_backup_tree(root)
    step += 1
    progress.put(f"PROGRESS|{step}|{total_steps}|Fast backup created: {backup_root.name}")

    for old_ep in deleted_indices:
        _delete_episode_files(root, info, rows_by_old[old_ep], video_keys)
        step += 1
        progress.put(f"PROGRESS|{step}|{total_steps}|Deleted episode {old_ep} files.")

    new_episode_rows: list[pd.Series] = []
    all_stats: list[dict[str, dict[str, np.ndarray]]] = []
    global_index = 0
    for old_ep in keep_episode_indices:
        old_row = rows_by_old[old_ep]
        new_ep = old_to_new[old_ep]
        ep_len = int(old_row["length"])
        new_row = old_row.copy()
        new_row["episode_index"] = new_ep
        new_row["length"] = ep_len
        new_row["data/chunk_index"] = new_ep // chunks_size
        new_row["data/file_index"] = new_ep % chunks_size
        new_row["dataset_from_index"] = global_index
        new_row["dataset_to_index"] = global_index + ep_len
        new_row["meta/episodes/chunk_index"] = new_ep // chunks_size
        new_row["meta/episodes/file_index"] = new_ep % chunks_size
        for video_key in video_keys:
            new_row[f"videos/{video_key}/chunk_index"] = new_ep // chunks_size
            new_row[f"videos/{video_key}/file_index"] = new_ep % chunks_size
            new_row[f"videos/{video_key}/from_timestamp"] = 0.0
            new_row[f"videos/{video_key}/to_timestamp"] = ep_len / fps

        if old_ep < min_changed:
            new_episode_rows.append(new_row)
            all_stats.append(stats_from_episode_row(new_row, info["features"]))
            global_index += ep_len
            continue

        old_data_path = _episode_data_path(root, info, old_row)
        new_data_path = _expected_data_path(root, info, new_ep, chunks_size)
        data_df = pd.read_parquet(old_data_path).copy().reset_index(drop=True)
        data_df["episode_index"] = new_ep
        data_df["frame_index"] = np.arange(ep_len, dtype=np.int64)
        data_df["index"] = np.arange(global_index, global_index + ep_len, dtype=np.int64)
        if "timestamp" in data_df:
            data_df["timestamp"] = data_df["frame_index"].astype(np.float32) / fps
        _atomic_write_parquet(data_df, new_data_path)
        if old_data_path != new_data_path and old_data_path.exists():
            old_data_path.unlink()
        step += 1
        progress.put(f"PROGRESS|{step}|{total_steps}|Episode {old_ep} -> {new_ep}: data updated.")

        for video_key in video_keys:
            old_video_path = _episode_video_path(root, info, old_row, video_key)
            new_video_path = _expected_video_path(root, info, video_key, new_ep, chunks_size)
            if old_video_path != new_video_path:
                new_video_path.parent.mkdir(parents=True, exist_ok=True)
                if new_video_path.exists():
                    raise RuntimeError(f"fast reindex target video already exists: {new_video_path}")
                old_video_path.rename(new_video_path)
        _move_sidecar_episode_files(root, old_ep, new_ep)
        step += 1
        progress.put(f"PROGRESS|{step}|{total_steps}|Episode {old_ep} -> {new_ep}: media moved.")

        old_meta_path = _episode_meta_path(root, old_row)
        new_meta_path = _expected_meta_path(root, new_ep, chunks_size)
        _atomic_write_parquet(pd.DataFrame([new_row]), new_meta_path)
        if old_meta_path != new_meta_path and old_meta_path.exists():
            old_meta_path.unlink()
        step += 1
        progress.put(f"PROGRESS|{step}|{total_steps}|Episode {old_ep} -> {new_ep}: metadata updated.")

        new_episode_rows.append(new_row)
        all_stats.append(stats_from_episode_row(new_row, info["features"]))
        global_index += ep_len

    info["total_episodes"] = len(new_episode_rows)
    info["total_frames"] = int(global_index)
    info["splits"] = {"train": f"0:{len(new_episode_rows)}"}
    _atomic_write_json(info, info_path)
    step += 1
    progress.put(f"PROGRESS|{step}|{total_steps}|Updated info.json.")

    if all_stats:
        write_stats(aggregate_stats(all_stats), root)
    step += 1
    progress.put(f"PROGRESS|{step}|{total_steps}|Updated stats.json.")

    _validate_fast_reindexed_dataset(root, info)
    progress.put(f"PROGRESS|{total_steps}|{total_steps}|Fast delete + reindex complete.")
    return DatasetRewriteOutcome(backup_root, "fast suffix reindex")


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
        self.saving_episode = False
        self.pending_save_confirmation = False
        self.save_thread: threading.Thread | None = None
        self.episode_count = 0
        self.current_episode_task = settings.task
        self.dataset: LeRobotDataset | None = None
        self.raw_depth_frames: list[np.ndarray] = []
        self.raw_depth_metadata: dict[str, np.ndarray] = {}
        self.sidecar_depth_camera: FlexibleRealSenseCamera | None = None
        self.consecutive_control_io_errors = 0
        self.last_control_io_error_status_t = 0.0
        self.skipped_recording_frames = 0
        self.control_io_warning_reasons: list[str] = []

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def request_record(self, task: str) -> None:
        if not self.ready:
            self._put_status("Not ready yet. Wait until status says preview and teleoperation are running.")
            return
        if self.saving_episode:
            self._put_status("Episode is still saving. Wait until save completes before recording again.")
            return
        if self.pending_save_confirmation:
            self._put_status("Episode is waiting for save/discard confirmation.")
            return
        self.command_queue.put(("record", task))

    def request_end_episode(self) -> None:
        self.command_queue.put("end")

    def request_confirm_save_episode(self) -> None:
        self.command_queue.put("confirm_save")

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

    def _put_save_state(self, saving: bool) -> None:
        self.status_queue.put(f"SAVE_STATE|{int(saving)}")

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
            self._configure_episode_file_storage(self.dataset)
            self._install_episode_video_saver(self.dataset)
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

                try:
                    obs = robot.get_observation()
                except Exception as exc:
                    self._handle_control_io_error(f"observation read failed: {exc}")
                    time.sleep(1 / FPS)
                    continue
                obs_processed = robot_observation_processor(obs)
                try:
                    raw_leader_action = teleop.get_action()
                except Exception as exc:
                    self._handle_control_io_error(f"leader action read failed: {exc}", obs_processed)
                    time.sleep(1 / FPS)
                    continue
                leader_action = teleop_action_processor((raw_leader_action, obs))
                action_to_send = robot_action_processor((leader_action, obs))
                try:
                    sent_action = robot.send_action(action_to_send)
                except Exception as exc:
                    self._handle_control_io_error(f"follower action send failed: {exc}", obs_processed)
                    time.sleep(1 / FPS)
                    continue
                if self.consecutive_control_io_errors:
                    self._put_status(
                        f"Control I/O recovered after {self.consecutive_control_io_errors} failed frame(s)."
                    )
                    self.consecutive_control_io_errors = 0

                self._publish_preview(obs_processed)
                self._publish_state(obs_processed, sent_action)

                if self.recording:
                    observation_frame = build_dataset_frame(self.dataset.features, obs_processed, prefix=OBS_STR)
                    action_frame = build_dataset_frame(self.dataset.features, sent_action, prefix=ACTION)
                    frame = {**observation_frame, **action_frame, "task": self.current_episode_task}
                    self.dataset.add_frame(frame)
                    self._store_raw_depth_frame(robot)

                dt_s = time.perf_counter() - start_loop_t
                time.sleep(max(1 / FPS - dt_s, 0.0))

            self._end_episode_if_needed()
            self._wait_for_save_if_needed()
        except Exception as exc:
            self.failed = True
            self._put_status(f"ERROR: {exc}")
        finally:
            self.ready = False
            self._wait_for_save_if_needed()
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
                self._disconnect_sidecar_depth_camera()
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
                raise RuntimeError(
                    f"Selected folder is not empty and is not a LeRobot dataset: {target_root}. "
                    "Choose an empty folder, click New with a new name, or choose an existing LeRobot dataset."
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
            raise RuntimeError(
                f"Dataset folder already exists and cannot be created as new: {target_root}. "
                "Choose an empty folder, click New with a new name, or enable Resume for an existing dataset."
            )

    def _configure_episode_file_storage(self, dataset: LeRobotDataset) -> None:
        dataset.meta.update_chunk_settings(
            data_files_size_in_mb=EPISODE_FILE_SIZE_MB,
            video_files_size_in_mb=EPISODE_FILE_SIZE_MB,
        )
        self._put_status(
            "Dataset storage set to episode-per-file mode "
            f"(data/video file limit {EPISODE_FILE_SIZE_MB} MB)."
        )

    def _install_episode_video_saver(self, dataset: LeRobotDataset) -> None:
        def save_episode_video_per_file(
            dataset_self: LeRobotDataset,
            video_key: str,
            episode_index: int,
            temp_path: Path | None = None,
        ) -> dict[str, float | int]:
            ep_path = (
                dataset_self._encode_temporary_episode_video(video_key, episode_index)
                if temp_path is None
                else temp_path
            )
            ep_duration_in_s = get_video_duration_in_s(ep_path)
            chunk_idx = int(episode_index) // dataset_self.meta.chunks_size
            file_idx = int(episode_index) % dataset_self.meta.chunks_size
            new_path = dataset_self.root / dataset_self.meta.video_path.format(
                video_key=video_key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if new_path.exists():
                new_path.unlink()
            shutil.move(str(ep_path), str(new_path))
            shutil.rmtree(str(ep_path.parent), ignore_errors=True)
            return {
                f"videos/{video_key}/chunk_index": chunk_idx,
                f"videos/{video_key}/file_index": file_idx,
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": ep_duration_in_s,
            }

        dataset._save_episode_video = MethodType(save_episode_video_per_file, dataset)
        self._put_status("Video storage set to strict episode-per-file mode.")

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

    @staticmethod
    def _serial_from_by_id_name(name: str) -> str:
        marker = "USB_Single_Serial_"
        if marker in name:
            return name.split(marker, 1)[1].split("-if", 1)[0]
        if "_Serial_" in name:
            return name.rsplit("_Serial_", 1)[1].split("-if", 1)[0]
        if "-if" in name and "_" in name:
            return name.rsplit("_", 1)[1].split("-if", 1)[0]
        return ""

    @staticmethod
    def _board_serial_from_port(port: str) -> str:
        port = port.strip()
        if not port:
            return ""

        serial = BimanualRecorder._serial_from_by_id_name(Path(port).name)
        if serial:
            return serial

        try:
            target = Path(port).resolve()
        except OSError:
            return ""

        by_id_root = Path("/dev/serial/by-id")
        if not by_id_root.exists():
            return ""

        for link in sorted(by_id_root.iterdir()):
            try:
                if link.resolve() == target:
                    return BimanualRecorder._serial_from_by_id_name(link.name)
            except OSError:
                continue
        return ""

    def _calibration_id_for_port(
        self, label: str, port: str, legacy_id: str, calibration_dir: Path | None
    ) -> str:
        if self.settings.calibration_match == "Arm id":
            self._put_status(f"{label} calibration id: arm id {legacy_id}")
            return legacy_id

        serial = self._board_serial_from_port(port)
        if not serial:
            self._put_status(f"{label} uses legacy calibration id '{legacy_id}' because no board serial was found.")
            return legacy_id

        if calibration_dir is not None:
            serial_path = calibration_dir / f"{serial}.json"
            legacy_path = calibration_dir / f"{legacy_id}.json"
            if not serial_path.exists() and legacy_path.exists():
                shutil.copy2(legacy_path, serial_path)
                self._put_status(f"{label} calibration migrated: {legacy_path.name} -> {serial_path.name}")
        self._put_status(f"{label} calibration id: board serial {serial}")
        return serial

    def _disable_gripper_motor(self, label: str, arm: SO101Follower | SO101Leader) -> None:
        bus = arm.bus
        if "gripper" not in bus.motors:
            return

        bus.motors.pop("gripper", None)
        bus.calibration.pop("gripper", None)
        bus._id_to_model_dict = {m.id: m.model for m in bus.motors.values()}
        bus._id_to_name_dict = {m.id: motor for motor, m in bus.motors.items()}
        for attr in ("ids", "models", "_has_different_ctrl_tables"):
            bus.__dict__.pop(attr, None)
        for attr in ("action_features", "observation_features"):
            arm.__dict__.pop(attr, None)
        self._put_status(f"{label}: gripper motor disabled for this session.")

    def _make_robot(self) -> LocalBimanualSOFollower:
        cameras = {}
        camera_width, camera_height = self._camera_size()
        serial = self._resolve_realsense_serial() if self.settings.capture_depth_sidecar else ""
        if self.settings.enable_realsense and not self.settings.capture_depth_sidecar:
            self._put_status("RealSense enabled but no depth option is selected; skipping RealSense.")
        if serial and not self.settings.record_opencv_front:
            cameras["front"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=camera_width,
                height=camera_height,
                fps=FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
            self._put_status("Using RealSense RGB/depth as front because OpenCV front recording is disabled.")

        requested_front_path = self.settings.opencv_front.strip() if self.settings.record_opencv_front else ""
        front_path = self._resolve_opencv_identifier(requested_front_path) if requested_front_path else ""
        if self.settings.opencv_front.strip() and not self.settings.record_opencv_front:
            self._put_status("OpenCV front path is set but not selected for recording; skipping front RGB.")
        front_is_realsense_video = bool(front_path and serial and self._is_realsense_video_path(front_path))
        if front_is_realsense_video:
            cameras["front"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=camera_width,
                height=camera_height,
                fps=FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
            self._put_status(
                f"OpenCV front {front_path} is a RealSense RGB node; using the RealSense SDK for synchronized RGB/depth."
            )
        elif front_path:
            cameras["front"] = OpenCVCameraConfig(
                index_or_path=self._opencv_value(front_path),
                width=camera_width,
                height=camera_height,
                fps=FPS,
                fourcc="MJPG",
            )
            if requested_front_path != front_path:
                self._put_status(f"OpenCV front {requested_front_path} resolved to {front_path}.")
            if serial:
                self._put_status(f"Using OpenCV front RGB {front_path}; RealSense is reserved for depth.")

        side_path = (
            self._resolve_opencv_side_path(self.settings.opencv_side.strip())
            if self.settings.record_opencv_side
            else ""
        )
        if self.settings.opencv_side.strip() and not self.settings.record_opencv_side:
            self._put_status("OpenCV side path is set but not selected for recording; skipping side RGB.")
        if side_path:
            side_description = self._describe_video_path(side_path)
            if side_description:
                self._put_status(f"OpenCV side RGB {side_path} is {side_description}.")
            cameras["side"] = OpenCVCameraConfig(
                index_or_path=self._opencv_value(side_path),
                width=camera_width,
                height=camera_height,
                fps=FPS,
                fourcc="MJPG",
            )
        cameras = {
            **cameras,
        }

        robot_calibration_dir, _ = self._calibration_dirs()
        if robot_calibration_dir is not None:
            self._put_status(f"Robot calibration dir: {robot_calibration_dir}")
        left_follower_id = self._calibration_id_for_port(
            "Left follower", self.settings.left_follower_port, self.settings.left_follower_id, robot_calibration_dir
        )
        right_follower_id = self._calibration_id_for_port(
            "Right follower",
            self.settings.right_follower_port,
            self.settings.right_follower_id,
            robot_calibration_dir,
        )
        left_arm = SO101Follower(
            SOFollowerRobotConfig(
                id=left_follower_id,
                calibration_dir=robot_calibration_dir,
                port=self.settings.left_follower_port,
                disable_torque_on_disconnect=True,
                cameras=cameras,
            )
        )
        right_arm = SO101Follower(
            SOFollowerRobotConfig(
                id=right_follower_id,
                calibration_dir=robot_calibration_dir,
                port=self.settings.right_follower_port,
                disable_torque_on_disconnect=True,
            )
        )
        if self.settings.disable_gripper:
            self._disable_gripper_motor("Left follower", left_arm)
            self._disable_gripper_motor("Right follower", right_arm)
        robot = LocalBimanualSOFollower(left_arm, right_arm)

        if serial and self.settings.capture_depth_sidecar:
            front_camera = robot.left_arm.cameras.get("front")
            if isinstance(front_camera, RealSenseCamera):
                front_camera = FlexibleRealSenseCamera(serial, camera_width, camera_height)
                robot.left_arm.cameras["front"] = front_camera
                robot.cameras["front"] = front_camera
                self._put_status("Using flexible RealSense wrapper for synchronized front RGB/depth.")
            if isinstance(front_camera, (RealSenseCamera, FlexibleRealSenseCamera)):
                rs_camera = front_camera
                self._put_status("RealSense raw depth will be saved from the front RealSense camera.")
            else:
                rs_camera = FlexibleRealSenseCamera(serial, camera_width, camera_height)
                self.sidecar_depth_camera = rs_camera
                self._put_status("RealSense raw depth sidecar camera prepared.")
            robot.left_arm.cameras["front_depth"] = RealSenseDepthView(rs_camera)
            robot.left_arm.config.cameras["front_depth"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=camera_width,
                height=camera_height,
                fps=FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
            robot.cameras["front_depth"] = robot.left_arm.cameras["front_depth"]
            self._put_status("RealSense depth will be saved as dataset input 'front_depth' and raw sidecar depth_mm.")

        return robot

    def _camera_size(self) -> tuple[int, int]:
        if self.settings.low_resolution:
            self._put_status(f"Low resolution enabled: cameras will record {LOW_CAMERA_WIDTH}x{LOW_CAMERA_HEIGHT}.")
            return LOW_CAMERA_WIDTH, LOW_CAMERA_HEIGHT
        return CAMERA_WIDTH, CAMERA_HEIGHT

    @staticmethod
    def _resolve_opencv_identifier(identifier: str) -> str:
        if not identifier.startswith(V4L2_SERIAL_PREFIX):
            return identifier

        selector = identifier.removeprefix(V4L2_SERIAL_PREFIX)
        serial, separator, role = selector.partition("/")
        if not serial or (separator and role != "rgb"):
            raise ValueError(
                f"Invalid V4L2 camera identifier '{identifier}'. "
                f"Expected {V4L2_SERIAL_PREFIX}<serial>/rgb."
            )

        candidates: list[tuple[int, Path]] = []
        serial_nodes: list[Path] = []
        format_scores = {"MJPG": 4, "YUYV": 3, "RGB3": 2, "BGR3": 2}
        for path in sorted(Path("/dev").glob("video*")):
            try:
                properties_result = subprocess.run(
                    ["udevadm", "info", "--query=property", f"--name={path}"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            properties = dict(
                line.split("=", 1)
                for line in properties_result.stdout.splitlines()
                if "=" in line
            )
            device_serials = (
                properties.get("ID_SERIAL_SHORT", ""),
                properties.get("ID_SERIAL", ""),
            )
            if not any(serial == value or serial in value for value in device_serials if value):
                continue
            serial_nodes.append(path)

        if not serial_nodes:
            serial_nodes.extend(BimanualRecorder._realsense_v4l2_nodes(serial))

        for path in serial_nodes:
            try:
                formats_result = subprocess.run(
                    ["v4l2-ctl", "-d", str(path), "--list-formats"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            formats = formats_result.stdout
            score = max(
                (
                    preference
                    for pixel_format, preference in format_scores.items()
                    if f"'{pixel_format}'" in formats
                ),
                default=0,
            )
            if score:
                candidates.append((score, path))

        if not candidates:
            matched = ", ".join(map(str, serial_nodes)) if serial_nodes else "none"
            raise ValueError(
                f"Could not find an RGB V4L2 node for camera serial {serial}. "
                f"Matching video nodes: {matched}."
            )
        candidates.sort(key=lambda item: (-item[0], item[1].name))
        return str(candidates[0][1])

    @staticmethod
    def _realsense_v4l2_nodes(serial: str) -> list[Path]:
        try:
            import pyrealsense2 as rs
        except ImportError:
            return []

        try:
            device = next(
                device
                for device in rs.context().query_devices()
                if device.get_info(rs.camera_info.serial_number) == serial
            )
            physical_port = Path(device.get_info(rs.camera_info.physical_port)).resolve()
            usb_device_root = physical_port.parents[2]
        except (OSError, RuntimeError, StopIteration, ValueError):
            return []

        nodes: list[Path] = []
        for path in sorted(Path("/dev").glob("video*")):
            try:
                device_path = (Path("/sys/class/video4linux") / path.name / "device").resolve()
            except OSError:
                continue
            if usb_device_root == device_path or usb_device_root in device_path.parents:
                nodes.append(path)
        return nodes

    @staticmethod
    def _stable_opencv_identifier(identifier: str) -> str:
        if identifier.startswith(V4L2_SERIAL_PREFIX):
            return identifier
        path = Path(identifier)
        if not path.name.startswith("video"):
            return identifier
        try:
            properties_result = subprocess.run(
                ["udevadm", "info", "--query=property", f"--name={path}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return identifier
        properties = dict(
            line.split("=", 1)
            for line in properties_result.stdout.splitlines()
            if "=" in line
        )
        serial = properties.get("ID_SERIAL_SHORT", "")
        if not serial:
            serial_id = properties.get("ID_SERIAL", "")
            serial = serial_id.rsplit("_", 1)[-1] if serial_id else ""
        if not serial:
            return identifier
        return f"{V4L2_SERIAL_PREFIX}{serial}/rgb"

    def _resolve_opencv_side_path(self, requested_path: str) -> str:
        if not requested_path:
            self._put_status("OpenCV side RGB is blank; recording without side RGB.")
            return ""
        requested_path = self._resolve_opencv_identifier(requested_path)

        realsense_active = self.settings.enable_realsense or self.settings.capture_depth_sidecar
        if realsense_active and self._is_realsense_video_path(requested_path):
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
            if realsense_active and self._is_realsense_video_path(candidate):
                continue
            self._put_status(f"Using detected OpenCV side RGB {candidate}.")
            return candidate

        if available_paths and realsense_active:
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

    @staticmethod
    def _opencv_value(index_or_path: str) -> int | Path:
        return int(index_or_path) if index_or_path.isdigit() else Path(index_or_path)

    def _make_teleop(self) -> LocalBimanualSOLeader:
        _, teleop_calibration_dir = self._calibration_dirs()
        if teleop_calibration_dir is not None:
            self._put_status(f"Teleop calibration dir: {teleop_calibration_dir}")
        left_leader_id = self._calibration_id_for_port(
            "Left leader", self.settings.left_leader_port, self.settings.left_leader_id, teleop_calibration_dir
        )
        right_leader_id = self._calibration_id_for_port(
            "Right leader", self.settings.right_leader_port, self.settings.right_leader_id, teleop_calibration_dir
        )
        left_arm = SO101Leader(
            SOLeaderTeleopConfig(
                id=left_leader_id,
                calibration_dir=teleop_calibration_dir,
                port=self.settings.left_leader_port,
            )
        )
        right_arm = SO101Leader(
            SOLeaderTeleopConfig(
                id=right_leader_id,
                calibration_dir=teleop_calibration_dir,
                port=self.settings.right_leader_port,
            )
        )
        if self.settings.disable_gripper:
            self._disable_gripper_motor("Left leader", left_arm)
            self._disable_gripper_motor("Right leader", right_arm)
        return LocalBimanualSOLeader(left_arm, right_arm)

    def _resolve_realsense_serial(self) -> str:
        if not self.settings.enable_realsense:
            self._put_status("RealSense depth requested; enabling RealSense for depth capture.")
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
                if self.saving_episode:
                    self._put_status("Episode is still saving. Wait until save completes before recording again.")
                elif self.pending_save_confirmation:
                    self._put_status("Episode is waiting for save/discard confirmation.")
                elif not self.recording:
                    requested_task = command[1] if isinstance(command, tuple) and len(command) > 1 else self.settings.task
                    self.current_episode_task = str(requested_task or self.settings.task)
                    self._delete_temporary_episode_images(self.episode_count)
                    self.recording = True
                    self.raw_depth_frames = []
                    self.raw_depth_metadata = {}
                    self.skipped_recording_frames = 0
                    self.control_io_warning_reasons = []
                    self._put_episode(self.episode_count, recording=True)
                    self._put_progress(0, 100, f"Recording episode {self.episode_count}")
                    self._put_status(f"Recording episode {self.episode_count} with task '{self.current_episode_task}'...")
            elif command_name == "end":
                self._end_episode_if_needed()
            elif command_name == "confirm_save":
                self._save_pending_episode_if_needed()
            elif command_name == "discard":
                self._discard_episode_if_needed()
            elif command_name == "stop":
                self.stop_event.set()
                self._end_episode_if_needed()

    def _end_episode_if_needed(self) -> None:
        if not self.recording or self.dataset is None:
            if self.saving_episode:
                self._put_status("Episode is already saving. Teleoperation can continue, but Record is disabled.")
            elif self.pending_save_confirmation:
                self._put_status("Episode is waiting for save/discard confirmation.")
            return
        self.recording = False
        if self.skipped_recording_frames > 0:
            self.pending_save_confirmation = True
            frame_count = int(self.dataset.episode_buffer["size"]) if self.dataset.episode_buffer else 0
            payload = {
                "episode_index": self.episode_count,
                "frame_count": frame_count,
                "skipped_control_io_frames": self.skipped_recording_frames,
                "warnings": list(self.control_io_warning_reasons),
            }
            self._put_episode(self.episode_count, recording=False)
            self._put_progress(0, 100, f"Episode {self.episode_count} waiting for save confirmation.")
            self.status_queue.put(f"CONFIRM_SAVE_SKIPPED|{json.dumps(payload, ensure_ascii=False)}")
            return
        self._begin_episode_save()

    def _save_pending_episode_if_needed(self) -> None:
        if not self.pending_save_confirmation:
            self._put_status("No episode is waiting for save confirmation.")
            return
        self._begin_episode_save()

    def _begin_episode_save(self) -> None:
        if self.dataset is None:
            self._put_status("No dataset is available for saving.")
            return
        self.pending_save_confirmation = False
        saved_episode_index = self.episode_count
        frame_count = int(self.dataset.episode_buffer["size"]) if self.dataset.episode_buffer else 0
        raw_depth_frames = self.raw_depth_frames
        raw_depth_metadata = self.raw_depth_metadata
        skipped_recording_frames = self.skipped_recording_frames
        warning_reasons = list(self.control_io_warning_reasons)
        self.raw_depth_frames = []
        self.raw_depth_metadata = {}
        self.skipped_recording_frames = 0
        self.control_io_warning_reasons = []
        self.saving_episode = True
        self._put_save_state(True)
        self._put_episode(saved_episode_index, recording=False)
        self._put_progress(5, 100, f"Saving episode {saved_episode_index} ({frame_count} frames)...")
        self._put_status(
            f"Saving episode {saved_episode_index} ({frame_count} frames, "
            f"{skipped_recording_frames} skipped bad frame(s)) in background. Teleoperation remains active."
        )
        self.save_thread = threading.Thread(
            target=self._save_episode_worker,
            args=(
                saved_episode_index,
                frame_count,
                raw_depth_frames,
                raw_depth_metadata,
                skipped_recording_frames,
                warning_reasons,
            ),
            daemon=True,
        )
        self.save_thread.start()

    def _save_episode_worker(
        self,
        saved_episode_index: int,
        frame_count: int,
        raw_depth_frames: list[np.ndarray],
        raw_depth_metadata: dict[str, np.ndarray],
        skipped_recording_frames: int,
        warning_reasons: list[str],
    ) -> None:
        try:
            if self.dataset is None:
                raise RuntimeError("dataset is not available while saving episode")
            if self.settings.capture_depth_sidecar and len(raw_depth_frames) != frame_count:
                raise RuntimeError(
                    f"raw depth frame count mismatch: depth={len(raw_depth_frames)}, dataset={frame_count}"
                )
            self._put_progress(20, 100, "Writing parquet data and encoding videos...")
            self._validate_temporary_video_frames(saved_episode_index, frame_count)
            self.dataset.save_episode()
            self._put_progress(85, 100, "Saving raw depth sidecar...")
            self._save_raw_depth_sidecar(saved_episode_index, raw_depth_frames, raw_depth_metadata)
            self._append_record_warning_sidecar(
                saved_episode_index,
                frame_count,
                skipped_recording_frames,
                warning_reasons,
            )
            self.episode_count = self.dataset.num_episodes
            if self.episode_count != saved_episode_index + 1:
                raise RuntimeError(
                    f"saved episode count mismatch: expected {saved_episode_index + 1}, dataset reports {self.episode_count}"
                )
            self._put_episode(self.episode_count)
            self._put_progress(100, 100, f"Episode {saved_episode_index} saved successfully.")
            self._put_status(
                f"Saved episode {saved_episode_index} successfully "
                f"({skipped_recording_frames} skipped bad frame(s)). Next episode is {self.episode_count}."
            )
        except Exception as exc:
            self.failed = True
            self.stop_event.set()
            self._put_progress(0, 100, f"Save failed for episode {saved_episode_index}.")
            self._put_status(f"ERROR: failed to save episode {saved_episode_index}: {exc}")
        finally:
            self.saving_episode = False
            self._put_save_state(False)

    def _append_record_warning_sidecar(
        self,
        episode_index: int,
        frame_count: int,
        skipped_control_io_frames: int,
        warning_reasons: list[str],
    ) -> None:
        if skipped_control_io_frames <= 0 and not warning_reasons:
            return
        if self.dataset is None:
            return
        warnings_path = self.dataset.root / "meta" / "record_warnings.jsonl"
        warnings_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "episode_index": int(episode_index),
            "frame_count": int(frame_count),
            "skipped_control_io_frames": int(skipped_control_io_frames),
            "warning_count": len(warning_reasons),
            "warnings": warning_reasons,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with warnings_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _wait_for_save_if_needed(self) -> None:
        if self.save_thread is not None and self.save_thread.is_alive():
            self._put_status("Waiting for background episode save to finish...")
            self.save_thread.join()

    def _discard_episode_if_needed(self) -> None:
        if (not self.recording and not self.pending_save_confirmation) or self.dataset is None:
            self._put_status("No active recording to discard.")
            return
        self.recording = False
        self.pending_save_confirmation = False
        self._delete_temporary_episode_images(self.episode_count)
        self.dataset.clear_episode_buffer(delete_images=False)
        discarded_frames = len(self.raw_depth_frames)
        self.raw_depth_frames = []
        self.raw_depth_metadata = {}
        self.skipped_recording_frames = 0
        self.control_io_warning_reasons = []
        self._put_episode(self.episode_count)
        self._put_progress(0, 100, f"Episode {self.episode_count} discarded.")
        self._put_status(
            f"Discarded episode {self.episode_count} ({discarded_frames} raw depth frame(s) dropped). Click Record to retry."
        )

    def _delete_temporary_episode_images(self, episode_index: int) -> None:
        if self.dataset is None:
            return
        if self.dataset.image_writer is not None:
            self.dataset._wait_image_writer()
        camera_keys = dict.fromkeys([*self.dataset.meta.image_keys, *self.dataset.meta.video_keys])
        for camera_key in camera_keys:
            image_dir = self.dataset._get_image_file_dir(episode_index, camera_key)
            if image_dir.is_dir():
                shutil.rmtree(image_dir)

    def _validate_temporary_video_frames(self, episode_index: int, frame_count: int) -> None:
        if self.dataset is None:
            raise RuntimeError("dataset is not available while validating temporary video frames")
        if self.dataset.image_writer is not None:
            self.dataset._wait_image_writer()
        expected_names = {f"frame-{frame_index:06d}.png" for frame_index in range(frame_count)}
        for video_key in self.dataset.meta.video_keys:
            image_dir = self.dataset._get_image_file_dir(episode_index, video_key)
            actual_names = {path.name for path in image_dir.glob("frame-*.png")}
            if actual_names == expected_names:
                continue
            missing = sorted(expected_names - actual_names)[:5]
            extra = sorted(actual_names - expected_names)[:5]
            raise RuntimeError(
                f"temporary video frame mismatch for {video_key}: expected={frame_count}, "
                f"actual={len(actual_names)}, missing={missing}, extra={extra}"
            )

    def _handle_control_io_error(self, reason: str, preview_obs: dict[str, Any] | None = None) -> None:
        self.consecutive_control_io_errors += 1
        if self.recording:
            self.skipped_recording_frames += 1
            if len(self.control_io_warning_reasons) < 20:
                self.control_io_warning_reasons.append(reason)
        if preview_obs is not None:
            self._publish_preview(preview_obs)

        now = time.monotonic()
        should_report = (
            self.consecutive_control_io_errors == 1
            or self.consecutive_control_io_errors in {5, 15, 30, 60}
            or now - self.last_control_io_error_status_t >= 2.0
        )
        if should_report:
            self.last_control_io_error_status_t = now
            self._put_status(
                f"CONTROL_IO_WARNING: {reason}. "
                f"Keeping devices connected"
                f"{'; skipping this recording frame' if self.recording else ''} "
                f"(consecutive failed frames: {self.consecutive_control_io_errors}, "
                f"episode skipped frames: {self.skipped_recording_frames})."
            )

    def _store_raw_depth_frame(self, robot: LocalBimanualSOFollower) -> None:
        if not self.settings.capture_depth_sidecar:
            return
        rs_camera = self.sidecar_depth_camera
        if rs_camera is None:
            front_camera = robot.left_arm.cameras.get("front")
            if isinstance(front_camera, (RealSenseCamera, FlexibleRealSenseCamera)):
                rs_camera = front_camera
        if not isinstance(rs_camera, (RealSenseCamera, FlexibleRealSenseCamera)):
            raise RuntimeError("RealSense depth sidecar camera is not available.")
        try:
            if not rs_camera.is_connected:
                rs_camera.connect()
            depth = None
            if isinstance(rs_camera, FlexibleRealSenseCamera):
                with rs_camera.frame_lock:
                    depth = rs_camera.latest_depth_frame
            if depth is None:
                depth = rs_camera.read_depth()
        except Exception as exc:
            raise RuntimeError(f"RealSense depth sidecar read failed: {exc}") from exc
        self.raw_depth_frames.append(np.asarray(depth, dtype=np.uint16).copy())
        if not self.raw_depth_metadata:
            self.raw_depth_metadata = self._get_realsense_depth_metadata(rs_camera)

    def _disconnect_sidecar_depth_camera(self) -> None:
        if self.sidecar_depth_camera is None:
            return
        try:
            if self.sidecar_depth_camera.is_connected:
                self.sidecar_depth_camera.disconnect()
        except Exception as exc:
            self._put_status(f"WARNING: RealSense depth sidecar disconnect failed: {exc}")
        finally:
            self.sidecar_depth_camera = None

    def _save_raw_depth_sidecar(
        self,
        episode_index: int,
        raw_depth_frames: list[np.ndarray],
        raw_depth_metadata: dict[str, np.ndarray],
    ) -> None:
        if self.dataset is None or not raw_depth_frames:
            return
        depth_dir = self.dataset.root / "sidecar_depth" / "front_depth_mm"
        depth_dir.mkdir(parents=True, exist_ok=True)
        depth_path = depth_dir / f"episode_{episode_index:06d}.npz"
        depth_stack = np.stack(raw_depth_frames, axis=0)
        np.savez_compressed(
            depth_path,
            depth_mm=depth_stack,
            fps=np.array(FPS, dtype=np.int32),
            episode_index=np.array(episode_index, dtype=np.int64),
            frame_index=np.arange(depth_stack.shape[0], dtype=np.int64),
            **raw_depth_metadata,
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
        frames = {
            key: value
            for key, value in obs.items()
            if key != "realsense_depth_source" and isinstance(value, np.ndarray) and value.ndim == 3
        }
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


class CameraOnlyPreview:
    def __init__(
        self,
        settings: RecorderSettings,
        status_queue: queue.Queue[str],
        preview_queue: queue.Queue[dict[str, np.ndarray]],
    ) -> None:
        self.settings = settings
        self.status_queue = status_queue
        self.preview_queue = preview_queue
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _put_status(self, message: str) -> None:
        print(f"[GUI recorder] {message}", flush=True)
        self.status_queue.put(message)

    def _run(self) -> None:
        captures: dict[str, cv2.VideoCapture] = {}
        try:
            camera_width = LOW_CAMERA_WIDTH if self.settings.low_resolution else CAMERA_WIDTH
            camera_height = LOW_CAMERA_HEIGHT if self.settings.low_resolution else CAMERA_HEIGHT
            candidates = {
                "front": self.settings.opencv_front.strip() if self.settings.record_opencv_front else "",
                "side": self.settings.opencv_side.strip() if self.settings.record_opencv_side else "",
            }
            for name, identifier in candidates.items():
                if not identifier:
                    continue
                try:
                    path = BimanualRecorder._resolve_opencv_identifier(identifier)
                except ValueError as exc:
                    self._put_status(f"Camera-only preview skipped {name}: {exc}")
                    continue
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    self._put_status(f"Camera-only preview failed to open {name}: {identifier} -> {path}")
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                cap.set(cv2.CAP_PROP_FPS, FPS)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                captures[name] = cap
                self._put_status(f"Camera-only preview using {name}: {identifier} -> {path}")

            if not captures:
                self._put_status("Camera-only preview could not open any OpenCV camera.")
                return

            self._put_status("Camera-only preview running. Robot/teleop is not connected; recording is disabled.")
            while not self.stop_event.is_set():
                start_t = time.perf_counter()
                frames: dict[str, np.ndarray] = {}
                for name, cap in list(captures.items()):
                    ok, frame_bgr = cap.read()
                    if not ok:
                        continue
                    frames[name] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frames:
                    BimanualRecorder._put_latest(self.preview_queue, frames)
                time.sleep(max(1 / FPS - (time.perf_counter() - start_t), 0.0))
        finally:
            for cap in captures.values():
                cap.release()
            self._put_status("Camera-only preview stopped.")


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
        self.camera_preview: CameraOnlyPreview | None = None
        self.last_connect_settings: RecorderSettings | None = None
        self.cleanup_running = False
        self.pending_cleanup_root: Path | None = None
        self.pending_cleanup_episodes: list[int] = []
        self.current_image: ImageTk.PhotoImage | None = None
        self.side_preview_image: ImageTk.PhotoImage | None = None
        self.dataset_image: ImageTk.PhotoImage | None = None
        self.latest_frames: dict[str, np.ndarray] = {}
        self.video_mode = "dataset"
        self.dataset_info: dict[str, Any] = {}
        self.dataset_episodes: dict[int, EpisodeBrowserRef] = {}
        self.deleted_episode_indices: set[int] = set()
        self.dataset_video_keys: list[str] = []
        self.current_dataset_episode: EpisodeBrowserRef | None = None
        self.dataset_captures: dict[str, cv2.VideoCapture] = {}
        self.playing_dataset = False
        self.play_offset_s = 0.0
        self.play_started_at = 0.0
        self.preview_zoom = tk.DoubleVar(value=1.0)
        self.live_frame_stats: dict[str, tuple[int, int, float]] = {}
        self.live_frame_times: dict[str, float] = {}
        self.active_dataset_root: Path | None = None
        self.last_warned_ineffective_path = ""
        self.dataset_root_warning_after: str | None = None
        self.episode_save_active = False
        self.vars = {
            "left_follower_port": tk.StringVar(value=DEFAULT_LEFT_FOLLOWER_PORT),
            "right_follower_port": tk.StringVar(value=DEFAULT_RIGHT_FOLLOWER_PORT),
            "left_leader_port": tk.StringVar(value=DEFAULT_LEFT_LEADER_PORT),
            "right_leader_port": tk.StringVar(value=DEFAULT_RIGHT_LEADER_PORT),
            "left_follower_id": tk.StringVar(value="my_awesome_follower_arm"),
            "right_follower_id": tk.StringVar(value="my_awesome_follower_arm_r"),
            "left_leader_id": tk.StringVar(value="my_awesome_leader_arm"),
            "right_leader_id": tk.StringVar(value="my_awesome_leader_arm_r"),
            "opencv_front": tk.StringVar(value=DEFAULT_FRONT_CAMERA_ID),
            "opencv_side": tk.StringVar(value=DEFAULT_SIDE_CAMERA_ID),
            "realsense_serial": tk.StringVar(value=""),
            "repo_id": tk.StringVar(value=""),
            "dataset_root": tk.StringVar(value=str(DEFAULT_DATASET_ROOT)),
            "calibration_dir": tk.StringVar(value=str(DEFAULT_CALIBRATION_PATH)),
            "calibration_match": tk.StringVar(value="Board serial"),
            "task": tk.StringVar(value=TASK_CHOICES[0]),
            "edit_task": tk.StringVar(value=TASK_CHOICES[0]),
            "preview_key": tk.StringVar(value=""),
        }
        self.resume = tk.BooleanVar(value=False)
        self.push_to_hub = tk.BooleanVar(value=False)
        self.enable_realsense = tk.BooleanVar(value=False)
        self.record_opencv_front = tk.BooleanVar(value=True)
        self.record_opencv_side = tk.BooleanVar(value=True)
        self.capture_depth_sidecar = tk.BooleanVar(value=False)
        self.low_resolution = tk.BooleanVar(value=True)
        self.disable_gripper = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Configure ports, then click Connect.")
        self.state_text = tk.StringVar(value="")
        self.episode_text = tk.StringVar(value="Episode: not connected")
        self.progress_text = tk.StringVar(value="Idle")
        self.progress_value = tk.DoubleVar(value=0.0)
        self.selected_save_text = tk.StringVar()
        self.active_save_text = tk.StringVar(value="Active save folder: not connected")
        self.vars["dataset_root"].trace_add("write", self._on_dataset_root_changed)
        self._refresh_selected_save_text()

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
            ("OpenCV frontview", "opencv_front"),
            ("OpenCV side RGB", "opencv_side"),
            ("Front RealSense SN", "realsense_serial"),
            ("Dataset name/id", "repo_id"),
            ("Dataset folder", "dataset_root"),
            ("Calibration path", "calibration_dir"),
            ("Calibration match", "calibration_match"),
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
            elif key == "calibration_match":
                ttk.Combobox(
                    grid,
                    textvariable=self.vars[key],
                    values=("Board serial", "Arm id"),
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
        ttk.Checkbutton(options, text="Save RealSense depth", variable=self.capture_depth_sidecar).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Checkbutton(options, text="Record frontview", variable=self.record_opencv_front).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Checkbutton(options, text="Record side RGB", variable=self.record_opencv_side).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Checkbutton(options, text="Low res cameras", variable=self.low_resolution).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        ttk.Checkbutton(options, text="Disable gripper", variable=self.disable_gripper).pack(
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
        self.preview_combo.bind("<<ComboboxSelected>>", self.on_preview_key_selected)

        buttons = ttk.Frame(controls)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="Preview Cameras", command=self.preview_cameras).pack(side=tk.LEFT, ipady=4)
        ttk.Button(buttons, text="Calibrate Arms", command=self.open_calibration_dialog).pack(
            side=tk.LEFT, padx=(10, 0), ipady=4
        )
        ttk.Button(buttons, text="Connect", command=self.connect).pack(side=tk.LEFT, ipady=4)
        self.record_button = ttk.Button(buttons, text="Record", command=self.record)
        self.record_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="End", command=self.end_episode).pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="Discard", command=self.discard_episode).pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="Fullscreen", command=self.toggle_fullscreen).pack(
            side=tk.LEFT, padx=(10, 0), ipady=4
        )
        ttk.Button(buttons, text="Stop", command=self.stop_recorder).pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="Disconnect", command=self.disconnect_recorder).pack(
            side=tk.LEFT, padx=(10, 0), ipady=4
        )
        ttk.Label(buttons, textvariable=self.episode_text, width=18, anchor=tk.W).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Progressbar(buttons, variable=self.progress_value, maximum=100, length=180).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(buttons, textvariable=self.progress_text, width=30, anchor=tk.W).pack(side=tk.LEFT, padx=(8, 0))

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
        delete_row = ttk.Frame(left_panel)
        delete_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(delete_row, text="Mark Delete", command=self.toggle_delete_selected_episode).pack(side=tk.LEFT)
        ttk.Button(delete_row, text="Apply Delete + Reindex", command=self.apply_episode_deletions).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        columns = ("id", "status", "length", "task")
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
        for col, width in (("id", 48), ("status", 70), ("length", 70), ("task", 180)):
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
        right_panel.rowconfigure(1, weight=3)
        right_panel.rowconfigure(3, weight=1)
        self.video_title = tk.StringVar(value="Select a dataset folder or click Connect for live preview.")
        title_label = ttk.Label(right_panel, textvariable=self.video_title, anchor=tk.W, font=("TkDefaultFont", 10))
        title_label.grid(row=0, column=0, sticky=tk.EW)
        self.preview_label = ttk.Label(right_panel, anchor=tk.CENTER)
        self.preview_label.grid(row=1, column=0, sticky=tk.NSEW, pady=(6, 0))
        self.side_video_title = tk.StringVar(value="Aux preview: not connected")
        side_title_label = ttk.Label(
            right_panel, textvariable=self.side_video_title, anchor=tk.W, font=("TkDefaultFont", 10)
        )
        side_title_label.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))
        self.side_preview_label = ttk.Label(right_panel, anchor=tk.CENTER)
        self.side_preview_label.grid(row=3, column=0, sticky=tk.NSEW, pady=(4, 0))
        playback_row = ttk.Frame(right_panel)
        playback_row.grid(row=4, column=0, sticky=tk.EW, pady=(8, 0))
        self.play_button_text = tk.StringVar(value="Play")
        ttk.Button(playback_row, textvariable=self.play_button_text, command=self.toggle_dataset_playback).pack(
            side=tk.LEFT
        )
        ttk.Button(playback_row, text="Stop Playback", command=self.stop_dataset_playback).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(playback_row, text="Zoom").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Button(playback_row, text="-", width=3, command=lambda: self.adjust_preview_zoom(0.8)).pack(side=tk.LEFT)
        ttk.Label(playback_row, textvariable=self.preview_zoom, width=4, anchor=tk.CENTER).pack(
            side=tk.LEFT, padx=(4, 4)
        )
        ttk.Button(playback_row, text="+", width=3, command=lambda: self.adjust_preview_zoom(1.25)).pack(side=tk.LEFT)
        ttk.Button(playback_row, text="Reset", command=self.reset_preview_zoom).pack(side=tk.LEFT, padx=(8, 0))

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.selected_save_text, anchor=tk.W).pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.active_save_text, anchor=tk.W).pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status, anchor=tk.W).pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.state_text, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

    def _on_dataset_root_changed(self, *_args: Any) -> None:
        self._refresh_selected_save_text()
        if self.dataset_root_warning_after is not None:
            self.root.after_cancel(self.dataset_root_warning_after)
        self.dataset_root_warning_after = self.root.after(600, self._warn_if_dataset_change_is_inactive)

    def _refresh_selected_save_text(self) -> None:
        selected = self.vars["dataset_root"].get().strip() or "(empty)"
        self.selected_save_text.set(f"Selected save folder: {selected}")

    def _set_active_dataset_root(self, dataset_root: Path | None) -> None:
        self.active_dataset_root = dataset_root.expanduser().resolve() if dataset_root is not None else None
        if self.active_dataset_root is None:
            self.active_save_text.set("Active save folder: not connected")
        else:
            self.active_save_text.set(f"Active save folder: {self.active_dataset_root}")

    def _is_recorder_alive(self) -> bool:
        return self.recorder is not None and self.recorder.is_alive()

    def _warn_if_dataset_change_is_inactive(self) -> None:
        self.dataset_root_warning_after = None
        if not self._is_recorder_alive() or self.active_dataset_root is None:
            return
        raw_path = self.vars["dataset_root"].get().strip()
        if not raw_path:
            return
        selected = Path(raw_path).expanduser()
        try:
            selected_resolved = selected.resolve()
        except OSError:
            selected_resolved = selected
        if selected_resolved == self.active_dataset_root:
            return
        warning_key = str(selected_resolved)
        if warning_key == self.last_warned_ineffective_path:
            return
        self.last_warned_ineffective_path = warning_key
        self.active_save_text.set(f"Active save folder: {self.active_dataset_root}")
        self.status.set(
            "Dataset folder was changed after Connect. The running recorder still saves to the active folder."
        )
        messagebox.showwarning(
            "Reconnect required",
            (
                "You changed the dataset folder after Connect.\n\n"
                f"Current recording still saves to:\n{self.active_dataset_root}\n\n"
                f"Selected for next Connect:\n{selected_resolved}\n\n"
                "Stop the recorder, then Connect again to apply the new save folder."
            ),
            parent=self.root,
        )

    def preview_cameras(self) -> None:
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Stop the recorder before opening the standalone camera preview.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("OpenCV Camera Preview")
        dialog.geometry("980x760")
        dialog.minsize(780, 560)
        dialog.transient(self.root)

        state: dict[str, Any] = {
            "capture": None,
            "image": None,
            "running": True,
            "last_frame_t": 0.0,
            "fps": 0.0,
        }
        camera_var = tk.StringVar(value=self.vars["opencv_front"].get() or DEFAULT_FRONT_CAMERA_ID)
        zoom_var = tk.DoubleVar(value=1.0)
        status_var = tk.StringVar(value="Scan cameras, select one, then preview or assign it.")

        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        top = ttk.Frame(frame)
        top.grid(row=0, column=0, sticky=tk.EW)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Camera").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        camera_combo = ttk.Combobox(top, textvariable=camera_var, width=48)
        camera_combo.grid(row=0, column=1, sticky=tk.EW)

        image_label = ttk.Label(frame, anchor=tk.CENTER)
        image_label.grid(row=2, column=0, sticky=tk.NSEW, pady=(10, 8))
        ttk.Label(frame, textvariable=status_var, anchor=tk.W).grid(row=3, column=0, sticky=tk.EW)

        def describe(path: str) -> str:
            description = BimanualRecorder._describe_video_path(path)
            return f"{path}  ({description})" if description else path

        def scan() -> None:
            candidates: list[str] = [DEFAULT_FRONT_CAMERA_ID, DEFAULT_SIDE_CAMERA_ID]
            try:
                cameras = OpenCVCamera.find_cameras()
                candidates.extend(str(camera.get("id")) for camera in cameras if camera.get("id") is not None)
            except Exception as exc:
                status_var.set(f"OpenCV scan failed: {exc}. You can still type a path manually.")
            for path in sorted(Path("/dev").glob("video*")):
                candidates.append(str(path))
                candidates.append(BimanualRecorder._stable_opencv_identifier(str(path)))
            candidates = sorted(dict.fromkeys(candidates))
            camera_combo["values"] = candidates
            if candidates and camera_var.get() not in candidates:
                camera_var.set(candidates[0])
            status_var.set(
                f"Found {len(candidates)} OpenCV candidate(s)."
                if candidates
                else "No OpenCV camera found. Type a path like /dev/video4 manually."
            )

        def close_capture() -> None:
            cap = state.get("capture")
            if cap is not None:
                cap.release()
                state["capture"] = None

        def start_preview() -> None:
            close_capture()
            path = camera_var.get().strip()
            if not path:
                status_var.set("Choose or type a camera path first.")
                return
            try:
                resolved_path = BimanualRecorder._resolve_opencv_identifier(path)
            except ValueError as exc:
                status_var.set(str(exc))
                return
            cap = cv2.VideoCapture(resolved_path)
            if not cap.isOpened():
                status_var.set(f"Failed to open {path} ({resolved_path}).")
                cap.release()
                return
            camera_width, camera_height = self._selected_camera_size()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            state["capture"] = cap
            state["last_frame_t"] = 0.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            status_var.set(f"Previewing {path} -> {describe(resolved_path)} | {width}x{height} | camera fps {fps:.1f}")

        def assign_front() -> None:
            path = camera_var.get().strip()
            if not path:
                return
            path = BimanualRecorder._stable_opencv_identifier(path)
            self.vars["opencv_front"].set(path)
            self.record_opencv_front.set(True)
            status_var.set(f"Assigned {path} to OpenCV frontview.")

        def assign_side() -> None:
            path = camera_var.get().strip()
            if not path:
                return
            path = BimanualRecorder._stable_opencv_identifier(path)
            self.vars["opencv_side"].set(path)
            self.record_opencv_side.set(True)
            status_var.set(f"Assigned {path} to OpenCV side RGB and enabled Record side RGB.")

        def update_frame() -> None:
            if not state["running"]:
                return
            cap = state.get("capture")
            if cap is not None:
                ok, frame_bgr = cap.read()
                if ok:
                    now = time.monotonic()
                    previous_t = float(state.get("last_frame_t") or 0.0)
                    if previous_t > 0 and now > previous_t:
                        state["fps"] = 1.0 / (now - previous_t)
                    state["last_frame_t"] = now
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(np.ascontiguousarray(frame_rgb))
                    max_size = self._zoomed_preview_size(float(zoom_var.get()))
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    state["image"] = ImageTk.PhotoImage(image=image)
                    image_label.configure(image=state["image"])
                    try:
                        resolved_path = BimanualRecorder._resolve_opencv_identifier(camera_var.get().strip())
                    except ValueError:
                        resolved_path = camera_var.get().strip()
                    status_var.set(
                        f"Previewing {camera_var.get().strip()} -> "
                        f"{describe(resolved_path)} | "
                        f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]} | {float(state['fps']):.1f} fps | "
                        f"zoom {zoom_var.get():.2f}x"
                    )
                else:
                    status_var.set(f"No frame from {camera_var.get().strip()}.")
            dialog.after(33, update_frame)

        def on_close() -> None:
            state["running"] = False
            close_capture()
            dialog.destroy()

        row = ttk.Frame(frame)
        row.grid(row=1, column=0, sticky=tk.EW, pady=(10, 0))
        ttk.Button(row, text="Scan", command=scan).pack(side=tk.LEFT)
        ttk.Button(row, text="Start Preview", command=start_preview).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text="Use as front", command=assign_front).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text="Use as side", command=assign_side).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(row, text="Zoom").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Button(row, text="-", width=3, command=lambda: zoom_var.set(max(0.25, zoom_var.get() * 0.8))).pack(
            side=tk.LEFT
        )
        ttk.Button(row, text="+", width=3, command=lambda: zoom_var.set(min(4.0, zoom_var.get() * 1.25))).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(row, text="Reset", command=lambda: zoom_var.set(1.0)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row, text="Close", command=on_close).pack(side=tk.RIGHT)

        camera_combo.bind("<<ComboboxSelected>>", lambda _event: start_preview())
        dialog.protocol("WM_DELETE_WINDOW", on_close)
        scan()
        start_preview()
        update_frame()

    def open_calibration_dialog(self) -> None:
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Stop the recorder before recalibrating arms.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Calibrate SO101 Arms")
        dialog.geometry("760x360")
        dialog.minsize(640, 320)
        dialog.transient(self.root)

        frame = ttk.Frame(dialog, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Run one calibration at a time. Follow the prompts in the terminal that launched this GUI.").grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 12)
        )

        rows = (
            ("Left follower", "robot", "left_follower_port", "left_follower_id"),
            ("Right follower", "robot", "right_follower_port", "right_follower_id"),
            ("Left leader", "teleop", "left_leader_port", "left_leader_id"),
            ("Right leader", "teleop", "right_leader_port", "right_leader_id"),
        )
        for row_idx, (label, kind, port_key, id_key) in enumerate(rows, start=1):
            ttk.Label(frame, text=label).grid(row=row_idx, column=0, sticky=tk.W, pady=5)
            ttk.Label(frame, textvariable=self.vars[port_key]).grid(row=row_idx, column=1, sticky=tk.W, pady=5)
            ttk.Button(
                frame,
                text="Calibrate",
                command=lambda k=kind, p=port_key, i=id_key, name=label: self.start_calibration(k, p, i, name),
            ).grid(row=row_idx, column=2, sticky=tk.E, pady=5)

        ttk.Label(frame, text=f"Calibration path: {self.vars['calibration_dir'].get()}").grid(
            row=5, column=0, columnspan=3, sticky=tk.W, pady=(16, 0)
        )
        ttk.Button(frame, text="Close", command=dialog.destroy).grid(row=6, column=2, sticky=tk.E, pady=(18, 0))

    def start_calibration(self, kind: str, port_key: str, id_key: str, label: str) -> None:
        port = self.vars[port_key].get().strip()
        device_id = self.vars[id_key].get().strip()
        if not port:
            self.status.set(f"Set the port before calibrating {label}.")
            return
        if not device_id:
            self.status.set(f"Set the id before calibrating {label}.")
            return

        robot_calibration_dir, teleop_calibration_dir = self._calibration_dirs_for_gui()
        calibration_dir = robot_calibration_dir if kind == "robot" else teleop_calibration_dir
        device_id = self._calibration_id_for_gui_port(
            label, port, device_id, calibration_dir, self.vars["calibration_match"].get()
        )
        prefix = "--robot" if kind == "robot" else "--teleop"
        device_type = "so101_follower" if kind == "robot" else "so101_leader"
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_calibrate",
            f"{prefix}.type={device_type}",
            f"{prefix}.port={port}",
            f"{prefix}.id={device_id}",
        ]
        if calibration_dir is not None:
            cmd.append(f"{prefix}.calibration_dir={calibration_dir}")

        if not self.ask_yes_no_english(
            "Start calibration?",
            (
                f"Start calibration for {label} on {port}?\n\n"
                "The interactive prompts will appear in the terminal. Do not connect or record while calibration is running."
            ),
        ):
            return

        self.status.set(f"Calibrating {label}; follow terminal prompts...")
        threading.Thread(target=self._run_calibration_command, args=(cmd, label), daemon=True).start()

    def _run_calibration_command(self, cmd: list[str], label: str) -> None:
        try:
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False)
        except Exception as exc:
            self.status_queue.put(f"Calibration failed to start for {label}: {exc}")
            return
        if result.returncode == 0:
            self.status_queue.put(f"Calibration finished for {label}.")
        else:
            self.status_queue.put(f"ERROR: calibration failed for {label} with exit code {result.returncode}.")

    def _calibration_dirs_for_gui(self) -> tuple[Path | None, Path | None]:
        raw_path = self.vars["calibration_dir"].get().strip()
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

    def _calibration_id_for_gui_port(
        self, label: str, port: str, legacy_id: str, calibration_dir: Path | None, calibration_match: str
    ) -> str:
        if calibration_match == "Arm id":
            self.status.set(f"{label}: using arm id calibration {legacy_id}.")
            return legacy_id

        serial = BimanualRecorder._board_serial_from_port(port)
        if not serial:
            self.status.set(f"{label}: no board serial found; using calibration id {legacy_id}.")
            return legacy_id

        if calibration_dir is not None:
            serial_path = calibration_dir / f"{serial}.json"
            legacy_path = calibration_dir / f"{legacy_id}.json"
            if not serial_path.exists() and legacy_path.exists():
                shutil.copy2(legacy_path, serial_path)
                self.status.set(f"{label}: copied calibration {legacy_path.name} -> {serial_path.name}.")
        return serial

    def ask_yes_no_english(self, title: str, message: str) -> bool:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("560x220")
        dialog.minsize(480, 180)
        dialog.transient(self.root)
        dialog.grab_set()

        result = {"yes": False}
        frame = ttk.Frame(dialog, padding=18)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=message, justify=tk.LEFT, wraplength=520).grid(row=0, column=0, sticky=tk.NSEW)

        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, sticky=tk.E, pady=(18, 0))

        def choose(value: bool) -> None:
            result["yes"] = value
            dialog.destroy()

        ttk.Button(buttons, text="Start", command=lambda: choose(True)).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Cancel", command=lambda: choose(False)).pack(side=tk.LEFT, padx=(10, 0))
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose(False))
        dialog.wait_window()
        return result["yes"]

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
        if folder.exists() and any(folder.iterdir()):
            self.status.set(f"Folder already exists and is not empty: {folder}. Choose a new name.")
            messagebox.showwarning(
                "Folder exists",
                f"This folder already exists and is not empty:\n{folder}\n\nChoose a new dataset name.",
                parent=self.root,
            )
            return
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
        self.deleted_episode_indices.clear()
        self.vars["dataset_root"].set(str(dataset_root))
        self.vars["repo_id"].set(self._infer_repo_id(dataset_root))
        health = self._show_dataset_folder_summary(dataset_root)
        self._remember_cleanup_choice(dataset_root, health)
        self._load_episode_browser(dataset_root)

    def _load_episode_browser(self, dataset_root: Path) -> None:
        self.stop_dataset_playback()
        self.close_dataset_captures()
        self.latest_frames = {}
        self.live_frame_stats = {}
        self.live_frame_times = {}
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
        loaded_indices: set[int] = set()
        for _, row in episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            loaded_indices.add(ep_idx)
            ep = EpisodeBrowserRef(ep_idx, row)
            self.dataset_episodes[ep_idx] = ep
        self.deleted_episode_indices.intersection_update(loaded_indices)
        self._rebuild_episode_browser_list()
        if self.dataset_episodes:
            self.video_title.set(
                f"Loaded {len(self.dataset_episodes)} episode(s). Select one on the left to play."
            )
        else:
            self.video_title.set("Dataset has no episode metadata yet.")

    def _rebuild_episode_browser_list(self) -> None:
        self.episode_tree.delete(*self.episode_tree.get_children())
        for ep_idx in sorted(self.dataset_episodes):
            ep = self.dataset_episodes[ep_idx]
            row = ep.row
            status = "delete" if ep_idx in self.deleted_episode_indices else "keep"
            self.episode_tree.insert(
                "",
                tk.END,
                iid=str(ep_idx),
                values=(ep_idx, status, int(row.get("length", 0)), self._task_text(row)),
            )

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

    def toggle_delete_selected_episode(self) -> None:
        selection = self.episode_tree.selection()
        if not selection:
            self.status.set("Select an episode before marking it for deletion.")
            return
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Stop recorder before marking saved episodes for deletion.")
            return
        if self.cleanup_running:
            self.status.set("Dataset rewrite is already running. Wait for it to finish.")
            return
        ep_idx = int(selection[0])
        if ep_idx in self.deleted_episode_indices:
            self.deleted_episode_indices.remove(ep_idx)
        else:
            self.deleted_episode_indices.add(ep_idx)
        self._rebuild_episode_browser_list()
        if str(ep_idx) in self.episode_tree.get_children():
            self.episode_tree.selection_set(str(ep_idx))
        self.status.set(f"Marked {len(self.deleted_episode_indices)} episode(s) for deletion.")

    def apply_episode_deletions(self) -> None:
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Stop recorder before deleting saved episodes.")
            return
        if self.cleanup_running:
            self.status.set("Dataset rewrite is already running. Wait for it to finish.")
            return
        dataset_root_text = self.vars["dataset_root"].get().strip()
        if not dataset_root_text:
            self.status.set("Choose a dataset folder before deleting episodes.")
            return
        dataset_root = self._find_dataset_root(Path(dataset_root_text).expanduser())
        if not (dataset_root / "meta" / "info.json").exists():
            self.status.set("No existing LeRobot dataset metadata found here.")
            return
        if not self.dataset_episodes:
            self._load_episode_browser(dataset_root)
        if not self.deleted_episode_indices:
            self.status.set("No episode is marked for deletion.")
            return
        keep_episode_indices = [
            ep_idx for ep_idx in sorted(self.dataset_episodes) if ep_idx not in self.deleted_episode_indices
        ]
        if not keep_episode_indices:
            messagebox.showerror("Refused", "At least one episode must be kept.", parent=self.root)
            return
        deleted_text = ", ".join(str(ep_idx) for ep_idx in sorted(self.deleted_episode_indices))
        if not messagebox.askyesno(
            "Delete selected episodes?",
            (
                f"Delete {len(self.deleted_episode_indices)} episode(s): {deleted_text}\n\n"
                f"The dataset will be rewritten with {len(keep_episode_indices)} kept episode(s), "
                "then reindexed to 0..N-1.\n\n"
                "A timestamped backup folder will be created next to the dataset before replacement."
            ),
            parent=self.root,
        ):
            self.status.set("Episode deletion cancelled.")
            return
        self.stop_dataset_playback()
        self.close_dataset_captures()
        self._start_dataset_cleanup(
            dataset_root,
            keep_episode_indices,
            start_message=f"Deleting {len(self.deleted_episode_indices)} episode(s) and reindexing dataset...",
        )

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
        prefixed = f"observation.images.{selected}"
        if prefixed in self.dataset_video_keys:
            return prefixed
        for key in self.dataset_video_keys:
            if key.rsplit(".", 1)[-1] == selected:
                return key
        return self.dataset_video_keys[0] if self.dataset_video_keys else ""

    def on_preview_key_selected(self, _event: tk.Event | None = None) -> None:
        if self.video_mode == "dataset" and self.current_dataset_episode is not None:
            self.seek_dataset_episode(self.play_offset_s)

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
        self._show_dataset_side_frame(target_s)

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

    def _show_dataset_side_frame(self, target_s: float) -> None:
        side_key = self._side_preview_key()
        if not side_key or side_key not in self.dataset_video_keys:
            self.side_video_title.set("Aux preview: no depth/side video")
            self.side_preview_label.configure(image="")
            self.side_preview_image = None
            return
        cap = self.open_dataset_capture(side_key)
        if cap is None:
            return
        cap.set(cv2.CAP_PROP_POS_MSEC, target_s * 1000.0)
        ok, frame_bgr = cap.read()
        if not ok:
            self.side_video_title.set(f"Aux preview: no frame from {side_key}")
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.ascontiguousarray(frame_rgb))
        image.thumbnail(self._side_preview_size(self.preview_zoom.get()), Image.Resampling.LANCZOS)
        self.side_preview_image = ImageTk.PhotoImage(image=image)
        self.side_preview_label.configure(image=self.side_preview_image)
        self.side_video_title.set(f"Aux preview: {side_key} | zoom {self.preview_zoom.get():.2f}x")

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

    def _start_dataset_cleanup(
        self,
        dataset_root: Path,
        keep_episode_indices: list[int],
        start_message: str | None = None,
    ) -> None:
        if self.cleanup_running:
            self.status.set("Dataset cleanup is already running. Wait for it to finish before selecting another folder.")
            return
        self.cleanup_running = True
        self.status_queue.put(f"PROGRESS|0|100|Cleaning dataset...")
        self.status_queue.put(
            start_message or f"Cleaning selected dataset and keeping {len(keep_episode_indices)} valid episode(s)..."
        )

        def worker() -> None:
            try:
                outcome = fast_delete_or_rewrite_dataset(dataset_root, keep_episode_indices, self.status_queue)
                self.status_queue.put(f"CLEAN_DONE|{dataset_root}|{outcome.backup_root}|{outcome.mode}")
            except Exception as exc:
                self.status_queue.put(f"CLEAN_ERROR|{dataset_root}|{exc}")

        threading.Thread(target=worker, daemon=True).start()

    def connect(self) -> None:
        if self.cleanup_running:
            self.status.set("Dataset cleanup is still running. Connect after the progress bar says complete.")
            return
        self._stop_camera_preview()
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
            record_opencv_front=self.record_opencv_front.get(),
            record_opencv_side=self.record_opencv_side.get(),
            enable_realsense=self.enable_realsense.get(),
            capture_depth_sidecar=self.capture_depth_sidecar.get(),
            low_resolution=self.low_resolution.get(),
            disable_gripper=self.disable_gripper.get(),
            realsense_serial=self.vars["realsense_serial"].get(),
            repo_id=self.vars["repo_id"].get(),
            dataset_root=self.vars["dataset_root"].get().strip(),
            calibration_dir=self.vars["calibration_dir"].get(),
            calibration_match=self.vars["calibration_match"].get(),
            task=self.vars["task"].get(),
            resume=self.resume.get(),
            push_to_hub=self.push_to_hub.get(),
        )
        self.status.set("Connect clicked. Building devices and dataset...")
        self.last_warned_ineffective_path = ""
        self.episode_save_active = False
        self._refresh_record_button_state()
        self._set_active_dataset_root(normalized_root)
        self.root.update_idletasks()
        self.stop_dataset_playback()
        self.close_dataset_captures()
        self.video_mode = "live"
        self.video_title.set("Live preview starting...")
        self.recorder = BimanualRecorder(settings, self.status_queue, self.preview_queue, self.state_queue)
        self.last_connect_settings = settings
        self.recorder.start()

    def record(self) -> None:
        if self.recorder is None:
            self.status.set("Click Connect first.")
            return
        if not self.recorder.ready:
            self.status.set("Not ready yet. Wait for: Ready. Preview and teleoperation are running.")
            return
        if self.episode_save_active:
            self.status.set("Episode is still saving. Record is disabled until save completes.")
            return
        task = self.vars["task"].get()
        if task not in TASK_CHOICES:
            self.status.set(f"Choose a valid task before recording: {', '.join(TASK_CHOICES)}")
            return
        episode = self.episode_text.get().split(":", 1)[-1].strip()
        if episode.isdigit():
            self.episode_text.set(f"Recording: {episode}")
        self.progress_text.set("Record requested...")
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
        self._stop_camera_preview()
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None
        self.episode_save_active = False
        self._refresh_record_button_state()
        self._set_active_dataset_root(None)
        self.video_mode = "dataset"
        self._load_episode_browser(Path(self.vars["dataset_root"].get()).expanduser())

    def disconnect_recorder(self) -> None:
        if self.recorder is None and self.camera_preview is None:
            self.status.set("Recorder is not connected.")
            return
        self.status.set("Disconnecting robot, teleoperator, and cameras...")
        self.root.update_idletasks()
        self.stop_recorder()
        self.status.set("Disconnected.")

    def _start_camera_preview_after_failure(self) -> None:
        if self.camera_preview is not None and self.camera_preview.is_alive():
            return
        settings = self.last_connect_settings
        if settings is None:
            return
        self.camera_preview = CameraOnlyPreview(settings, self.status_queue, self.preview_queue)
        self.camera_preview.start()

    def _stop_camera_preview(self) -> None:
        if self.camera_preview is not None:
            self.camera_preview.stop()
            self.camera_preview = None

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
            failed = self.recorder.failed
            if failed:
                self.status.set(f"{self.status.get()} You can fix the setting and click Connect again.")
            self.recorder = None
            self.episode_save_active = False
            self._refresh_record_button_state()
            self._set_active_dataset_root(None)
            if failed:
                self._start_camera_preview_after_failure()

        live_source_active = (
            (self.recorder is not None and self.recorder.is_alive())
            or (self.camera_preview is not None and self.camera_preview.is_alive())
        )
        if live_source_active:
            try:
                self.latest_frames = self.preview_queue.get_nowait()
                now = time.monotonic()
                keys = sorted(self.latest_frames)
                for frame_key, frame in self.latest_frames.items():
                    if isinstance(frame, np.ndarray) and frame.ndim >= 2:
                        previous_t = self.live_frame_times.get(frame_key)
                        fps = 0.0 if previous_t is None or now <= previous_t else 1.0 / (now - previous_t)
                        self.live_frame_times[frame_key] = now
                        self.live_frame_stats[frame_key] = (int(frame.shape[1]), int(frame.shape[0]), fps)
                self.preview_combo["values"] = keys
                if (not self.vars["preview_key"].get() or self.vars["preview_key"].get() not in keys) and keys:
                    self.vars["preview_key"].set(keys[0])
                self.video_mode = "live"
            except queue.Empty:
                pass
        else:
            try:
                while True:
                    self.preview_queue.get_nowait()
            except queue.Empty:
                pass

        key = self.vars["preview_key"].get()
        if self.video_mode == "live" and key in self.latest_frames:
            width, height, fps = self.live_frame_stats.get(key, (0, 0, 0.0))
            self.video_title.set(
                f"Live preview: {key} | {width}x{height} | {fps:.1f} fps | zoom {self.preview_zoom.get():.2f}x"
            )
            self._show_frame(self.latest_frames[key])
            self._show_side_live_preview()

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
        if message.startswith("CONFIRM_SAVE_SKIPPED|"):
            try:
                payload = json.loads(message.split("|", 1)[1])
            except (json.JSONDecodeError, IndexError) as exc:
                self.status.set(f"ERROR: invalid save confirmation message: {exc}")
                return
            self._confirm_save_skipped_episode(payload)
            return
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
        if message.startswith("SAVE_STATE|"):
            try:
                _kind, saving = message.split("|", 1)
                self.episode_save_active = saving == "1"
                self._refresh_record_button_state()
            except ValueError:
                self.status.set(message)
            return
        if message.startswith("CLEAN_DONE|"):
            try:
                parts = message.split("|", 3)
                if len(parts) == 4:
                    _kind, dataset_root, backup, mode = parts
                else:
                    _kind, dataset_root, backup = message.split("|", 2)
                    mode = "full rewrite"
                self.cleanup_running = False
                self.pending_cleanup_root = None
                self.pending_cleanup_episodes = []
                self.deleted_episode_indices.clear()
                self.progress_value.set(100.0)
                self.progress_text.set("Cleanup complete")
                health = self._show_dataset_folder_summary(Path(dataset_root))
                self._remember_cleanup_choice(Path(dataset_root), health)
                self._load_episode_browser(Path(dataset_root))
                self.status.set(f"Dataset cleaned successfully by {mode}. Backup: {backup}")
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
        if message.startswith("Dataset root:"):
            raw_root = message.removeprefix("Dataset root:").strip()
            if raw_root:
                self._set_active_dataset_root(Path(raw_root))
        self.status.set(message)

    def _confirm_save_skipped_episode(self, payload: dict[str, Any]) -> None:
        episode_index = int(payload.get("episode_index", -1))
        frame_count = int(payload.get("frame_count", 0))
        skipped = int(payload.get("skipped_control_io_frames", 0))
        warnings = payload.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        preview_warnings = "\n".join(f"- {str(item)}" for item in warnings[:3])
        if len(warnings) > 3:
            preview_warnings += f"\n- ... {len(warnings) - 3} more"
        detail = (
            f"Episode {episode_index} contains {skipped} skipped control I/O frame(s).\n\n"
            f"Saved frame count if accepted: {frame_count}\n\n"
            "The skipped frames were not written to images, state, action, or raw depth. "
            "The remaining frames stay aligned, but the real-time trajectory has gap(s).\n\n"
        )
        if preview_warnings:
            detail += f"Warnings:\n{preview_warnings}\n\n"
        detail += "Save this episode?"
        save_episode = messagebox.askyesno(
            "Save episode with skipped frames?",
            detail,
            parent=self.root,
        )
        if self.recorder is None:
            self.status.set("Recorder is no longer active; save confirmation ignored.")
            return
        if save_episode:
            self.status.set(f"Saving episode {episode_index} with {skipped} skipped frame(s).")
            self.recorder.request_confirm_save_episode()
        else:
            self.status.set(f"Discarding episode {episode_index} with {skipped} skipped frame(s).")
            self.recorder.request_discard_episode()

    def _refresh_record_button_state(self) -> None:
        state = tk.DISABLED if self.episode_save_active else tk.NORMAL
        self.record_button.configure(state=state)

    def adjust_preview_zoom(self, factor: float) -> None:
        self.preview_zoom.set(min(4.0, max(0.25, self.preview_zoom.get() * factor)))

    def reset_preview_zoom(self) -> None:
        self.preview_zoom.set(1.0)

    @staticmethod
    def _zoomed_preview_size(zoom: float) -> tuple[int, int]:
        zoom = min(4.0, max(0.25, zoom))
        return (max(1, int(PREVIEW_SIZE[0] * zoom)), max(1, int(PREVIEW_SIZE[1] * zoom)))

    @staticmethod
    def _side_preview_size(zoom: float) -> tuple[int, int]:
        zoom = min(4.0, max(0.25, zoom))
        return (max(1, int(SIDE_PREVIEW_SIZE[0] * zoom)), max(1, int(SIDE_PREVIEW_SIZE[1] * zoom)))

    def _selected_camera_size(self) -> tuple[int, int]:
        if self.low_resolution.get():
            return LOW_CAMERA_WIDTH, LOW_CAMERA_HEIGHT
        return CAMERA_WIDTH, CAMERA_HEIGHT

    def _show_frame(self, frame: np.ndarray) -> None:
        image = Image.fromarray(np.ascontiguousarray(frame))
        image.thumbnail(self._zoomed_preview_size(self.preview_zoom.get()), Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.current_image)

    def _show_side_frame(self, frame: np.ndarray) -> None:
        image = Image.fromarray(np.ascontiguousarray(frame))
        image.thumbnail(self._side_preview_size(self.preview_zoom.get()), Image.Resampling.LANCZOS)
        self.side_preview_image = ImageTk.PhotoImage(image=image)
        self.side_preview_label.configure(image=self.side_preview_image)

    def _show_side_live_preview(self) -> None:
        side_key = self._side_preview_key()
        if side_key == "" or side_key not in self.latest_frames:
            self.side_video_title.set("Aux preview: no depth/side frame")
            self.side_preview_label.configure(image="")
            self.side_preview_image = None
            return

        width, height, fps = self.live_frame_stats.get(side_key, (0, 0, 0.0))
        self.side_video_title.set(
            f"Aux preview: {side_key} | {width}x{height} | {fps:.1f} fps | zoom {self.preview_zoom.get():.2f}x"
        )
        self._show_side_frame(self.latest_frames[side_key])

    def _side_preview_key(self) -> str:
        available_keys = set(self.latest_frames) if self.video_mode == "live" else set(self.dataset_video_keys)
        active_key = self.vars["preview_key"].get()
        if self.video_mode == "dataset":
            active_key = self.active_dataset_video_key()
        preferred_keys = (
            "front_depth",
            "observation.images.front_depth",
            "left_front_depth",
            "observation.images.left_front_depth",
            "left_side",
            "side",
            "observation.images.left_side",
            "observation.images.side",
        )
        for key in preferred_keys:
            if key in available_keys and key != active_key:
                return key
        for key in sorted(available_keys):
            if "side" in key and key != active_key:
                return key
        return ""

    def _show_bgr_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.ascontiguousarray(frame_rgb))
        image.thumbnail(self._zoomed_preview_size(self.preview_zoom.get()), Image.Resampling.LANCZOS)
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
