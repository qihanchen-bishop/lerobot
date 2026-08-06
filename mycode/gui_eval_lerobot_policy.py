#!/usr/bin/env python
"""GUI launcher for evaluating trained LeRobot policies on a real robot."""

from __future__ import annotations

import json
import os
import queue
import shutil
import shlex
import signal
import subprocess
import threading
import time
import traceback
import tkinter as tk
import tkinter.font as tkfont
import concurrent.futures
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from tkinter import filedialog, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "train"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "eval" / "object3color2"
DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration" / "robots" / "so_follower"
DEFAULT_POLICY_PATH = DEFAULT_OUTPUT_ROOT / "1A_3object" / "checkpoint_step_100000"
DEFAULT_POLICY_TYPE = "mask_act"
DEFAULT_DIFFUSION_PREDICTION_STEPS = 24
DEFAULT_DIFFUSION_INFERENCE_STEPS = 16
DEFAULT_DIFFUSION_REPLAN_STEPS = 8
DEFAULT_DIFFUSION_FUSION_STEPS = 4
DEFAULT_DIFFUSION_FUSION_HISTORY_WEIGHT = 0.8
DEFAULT_DIFFUSION_OBSERVATION_WARMUP_FRAMES = 4
DEFAULT_DIFFUSION_EXECUTION_MODE = "asynchronous"
DEFAULT_CAMERA_READ_MODE = "wait_new_frame"
DEFAULT_SEGMENTATION_MODEL_PATH = Path(__file__).resolve().parent / "tool" / "best.pt"
SEGMENTATION_OBJECT_PRESENT_MIN_RATIO = 0.001
SEGMENTATION_SCREW_OBJECT_PRESENT_MIN_RATIO = 0.0008
SEGMENTATION_OCCLUDER_COMPLETE_RATIO = 0.50
SEGMENTATION_VOTE_WINDOW = 7
SEGMENTATION_VOTE_REQUIRED = 4
RESULTS_FILENAME = "eval_results.jsonl"
DEFAULT_TRIALS_PER_GRID = 10
DEFAULT_LEFT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E122511-if00"
DEFAULT_RIGHT_FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3E119029-if00"

POLICY_TYPES = (
    "mask_act",
    "act",
    "diffusion",
    "smolvla",
    "pi0",
    "pi0_fast",
    "pi05",
    "vqbet",
    "tdmpc",
    "sarm",
)
RESULT_CHOICES = ("success", "failure")
TASK_CHOICES = ("cube", "paperball", "screw", "cube_r", "paperball_r", "screw_r")

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
LOW_CAMERA_WIDTH = 640
LOW_CAMERA_HEIGHT = 360
FALLBACK_CAMERA_WIDTH = 640
FALLBACK_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

SEGMENTATION_PALETTE = (
    (0, 0, 0),
    (50, 160, 255),
    (255, 80, 80),
    (60, 220, 120),
    (210, 210, 80),
)
SEGMENTATION_MEAN = (0.485, 0.456, 0.406)
SEGMENTATION_STD = (0.229, 0.224, 0.225)
PREVIEW_MIN_WIDTH = 220
PREVIEW_MAX_WIDTH = 420

# Median initial joint pose from the nine successful center-cell 4B evaluations
# recorded on 2026-06-22. Medians reduce the influence of individual reset drift.
DEFAULT_90_PERCENT_SUCCESS_POSE = {
    "left_shoulder_pan.pos": -2.137583,
    "left_shoulder_lift.pos": -89.469284,
    "left_elbow_flex.pos": 95.840866,
    "left_wrist_flex.pos": 82.700798,
    "left_wrist_roll.pos": 51.500912,
    "left_gripper.pos": 3.017544,
    "right_shoulder_pan.pos": -28.099495,
    "right_shoulder_lift.pos": -89.004578,
    "right_elbow_flex.pos": 100.0,
    "right_wrist_flex.pos": 63.317245,
    "right_wrist_roll.pos": 31.498674,
    "right_gripper.pos": 18.631178,
}


@dataclass(frozen=True)
class CheckpointOption:
    label: str
    path: Path
    policy_type: str


class LocalBimanualSOFollower:
    name = "bi_so_follower"

    def __init__(self, left_arm: Any, right_arm: Any) -> None:
        self.left_arm = left_arm
        self.right_arm = right_arm
        self.cameras = {**left_arm.cameras, **right_arm.cameras}
        self._io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="eval-bimanual-follower-io"
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
        futures[self._io_executor.submit(self._read_arm_motors, self.left_arm)] = ("left_motors", "")
        futures[self._io_executor.submit(self._read_arm_motors, self.right_arm)] = ("right_motors", "")
        for camera_key, camera in self.cameras.items():
            futures[self._io_executor.submit(camera.async_read)] = ("camera", camera_key)

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
        left_future = self._io_executor.submit(self.left_arm.send_action, left_action)
        right_future = self._io_executor.submit(self.right_arm.send_action, right_action)
        sent_left = left_future.result()
        sent_right = right_future.result()
        return {
            **{f"left_{key}": value for key, value in sent_left.items()},
            **{f"right_{key}": value for key, value in sent_right.items()},
        }

    @staticmethod
    def _read_arm_motors(arm: Any) -> dict[str, float]:
        obs_dict = arm.bus.sync_read("Present_Position")
        return {f"{motor}.pos": val for motor, val in obs_dict.items()}

    def disconnect(self) -> None:
        self._disconnect_arm_safely("left follower", self.left_arm)
        self._disconnect_arm_safely("right follower", self.right_arm)
        self._io_executor.shutdown(wait=True, cancel_futures=True)

    @staticmethod
    def _disconnect_arm_safely(label: str, arm: Any) -> None:
        try:
            arm.disconnect()
            return
        except Exception as exc:
            print(f"[GUI eval] Warning: normal disconnect failed for {label}: {exc}", flush=True)

        try:
            for motor in getattr(arm.bus, "motors", {}):
                try:
                    arm.bus.disable_torque(motor, num_retry=2)
                except Exception:
                    pass
            arm.bus.disconnect(disable_torque=False)
        except Exception as exc:
            print(f"[GUI eval] Warning: force-closing bus failed for {label}: {exc}", flush=True)


class PersistentFlexibleRealSenseCamera:
    """RealSense wrapper for persistent eval; mirrors the recorder but cleans up failed starts."""

    def __init__(self, serial_number: str, width: int, height: int) -> None:
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.frame_lock = threading.Lock()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.rs_pipeline = None
        self.rs_profile = None
        self._rs = None

    @property
    def is_connected(self) -> bool:
        return self.rs_pipeline is not None and self.rs_profile is not None

    def connect(self) -> None:
        if self.is_connected:
            return
        import cv2
        import numpy as np
        import pyrealsense2 as rs

        del cv2, np
        self._rs = rs
        last_error: Exception | None = None
        for attempt in range(1, 4):
            pipeline = rs.pipeline()
            config = rs.config()
            rs.config.enable_device(config, self.serial_number)
            config.enable_stream(rs.stream.color)
            config.enable_stream(rs.stream.depth)
            try:
                self.rs_profile = pipeline.start(config)
                self.rs_pipeline = pipeline
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    try:
                        self._read_frames(timeout_ms=10_000)
                        return
                    except RuntimeError as exc:
                        last_error = exc
                        time.sleep(0.2)
                raise RuntimeError(f"RealSense frame did not arrive during warmup attempt {attempt}.")
            except Exception as exc:
                last_error = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
                self.rs_pipeline = None
                self.rs_profile = None
                time.sleep(0.5)
        raise RuntimeError(f"Failed to connect RealSense {self.serial_number}: {last_error}")

    def disconnect(self) -> None:
        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except Exception:
                pass
        self.rs_pipeline = None
        self.rs_profile = None

    def async_read(self, timeout_ms: int = 200):
        color, _depth = self._read_frames(timeout_ms=max(timeout_ms, 2_000))
        return color

    def _read_frames(self, timeout_ms: int = 10_000):
        import cv2
        import numpy as np

        if self.rs_pipeline is None:
            raise RuntimeError("RealSense pipeline is not connected.")
        frames = self.rs_pipeline.wait_for_frames(timeout_ms=timeout_ms)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense frame set did not include both color and depth frames.")
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        if color.ndim == 2:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
        elif color.shape[2] == 4:
            color = color[:, :, :3]
        if color.shape[1] != self.width or color.shape[0] != self.height:
            color = cv2.resize(color, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if depth.shape[1] != self.width or depth.shape[0] != self.height:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        color = np.ascontiguousarray(color)
        with self.frame_lock:
            self.latest_color_frame = color
            self.latest_depth_frame = depth
        return color, depth


class EvalFlexibleOpenCVCamera:
    """OpenCV camera that returns frames resized to the configured dataset resolution."""

    def __init__(self, config: Any, fallback_width: int = FALLBACK_CAMERA_WIDTH, fallback_height: int = FALLBACK_CAMERA_HEIGHT) -> None:
        self.config = config
        self.index_or_path = config.index_or_path
        self.width = int(config.width)
        self.height = int(config.height)
        self.fps = int(config.fps or DEFAULT_CAMERA_FPS)
        self.fallback_width = fallback_width
        self.fallback_height = fallback_height
        self.capture = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp: float | None = None

    @property
    def is_connected(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def connect(self) -> None:
        if self.is_connected:
            return
        import cv2

        target = self.index_or_path
        if isinstance(target, Path):
            target = str(target)
        self.capture = cv2.VideoCapture(target)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Could not open OpenCV camera {self.index_or_path}")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.fallback_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.fallback_height)
            ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read OpenCV camera {self.index_or_path}")
        self._store_frame(frame)

    def disconnect(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None

    def async_read(self, timeout_ms: int = 200):
        del timeout_ms
        import cv2

        if self.capture is None:
            raise RuntimeError("OpenCV camera is not connected.")
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read OpenCV camera {self.index_or_path}")
        return self._store_frame(frame)

    def read_latest(self, max_age_ms: int = 1000):
        del max_age_ms
        return self.async_read()

    def _store_frame(self, frame: Any):
        import cv2
        import numpy as np

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = np.ascontiguousarray(frame)
        with self.frame_lock:
            self.latest_frame = frame
            self.latest_timestamp = time.perf_counter()
        return frame


class SegmentationPreviewModel:
    def __init__(self, checkpoint_path: str | Path, device_name: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.device_name = device_name
        self.model = None
        self.labels: list[str] = []
        self.image_size: tuple[int, int] = (320, 180)
        self.device = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SeparableConv(nn.Module):
            def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x)

        class TinyUNet(nn.Module):
            def __init__(self, num_classes: int, width: int = 32) -> None:
                super().__init__()
                c1, c2, c3, c4 = width, width * 2, width * 4, width * 6
                self.stem = nn.Sequential(
                    nn.Conv2d(3, c1, 3, padding=1, bias=False),
                    nn.BatchNorm2d(c1),
                    nn.ReLU(inplace=True),
                    SeparableConv(c1, c1),
                )
                self.down1 = SeparableConv(c1, c2, stride=2)
                self.down2 = SeparableConv(c2, c3, stride=2)
                self.down3 = SeparableConv(c3, c4, stride=2)
                self.fuse2 = SeparableConv(c4 + c3, c3)
                self.fuse1 = SeparableConv(c3 + c2, c2)
                self.fuse0 = SeparableConv(c2 + c1, c1)
                self.head = nn.Conv2d(c1, num_classes, 1)

            def forward(self, x: Any) -> Any:
                s0 = self.stem(x)
                s1 = self.down1(s0)
                s2 = self.down2(s1)
                x = self.down3(s2)
                x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
                x = self.fuse2(torch.cat([x, s2], dim=1))
                x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
                x = self.fuse1(torch.cat([x, s1], dim=1))
                x = F.interpolate(x, size=s0.shape[-2:], mode="bilinear", align_corners=False)
                x = self.fuse0(torch.cat([x, s0], dim=1))
                return self.head(x)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Segmentation checkpoint does not exist: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.labels = list(checkpoint.get("labels", ["background", "occluder", "object", "region", "tool"]))
        self.image_size = tuple(checkpoint.get("image_size", (320, 180)))
        width = int(checkpoint.get("args", {}).get("width", 32))
        requested = self.device_name.strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        model = TinyUNet(num_classes=len(self.labels), width=width)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device).eval()
        self.model = model

    @property
    def loaded_device(self) -> str:
        return str(self.device) if self.device is not None else "not-loaded"

    def predict(self, images: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        self._load()
        import cv2
        import numpy as np
        import torch

        assert self.model is not None
        assert self.device is not None
        keys: list[str] = []
        tensors = []
        originals: dict[str, Any] = {}
        mean = np.asarray(SEGMENTATION_MEAN, dtype=np.float32)
        std = np.asarray(SEGMENTATION_STD, dtype=np.float32)
        for key in ("front", "side"):
            image = images.get(key)
            if image is None or not hasattr(image, "ndim") or image.ndim != 3:
                continue
            rgb = np.asarray(image)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            resized = cv2.resize(rgb, self.image_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            resized = (resized - mean) / std
            tensors.append(torch.from_numpy(resized.transpose(2, 0, 1)))
            originals[key] = rgb
            keys.append(key)
        if not tensors:
            return {}, {}, {}

        batch = torch.stack(tensors).float().to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        palette = np.asarray(SEGMENTATION_PALETTE, dtype=np.uint8)
        overlays: dict[str, Any] = {}
        summaries: dict[str, str] = {}
        measurements: dict[str, Any] = {}
        for key, pred_small in zip(keys, preds):
            image = originals[key]
            pred = cv2.resize(pred_small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            color = palette[np.clip(pred, 0, len(palette) - 1)]
            parts = []
            total = max(pred.size, 1)
            class_ratios: dict[str, float] = {}
            for class_id, label in enumerate(self.labels[1:], start=1):
                ratio = float((pred == class_id).sum()) / total
                class_ratios[label] = ratio
                parts.append(f"{label} {100.0 * ratio:.1f}%")
            summaries[key] = ", ".join(parts)
            class_bboxes = {
                "object": self._class_bbox(pred, "object", largest_component=True),
                "occluder": self._class_bbox(pred, "occluder"),
                "region": self._class_bbox(pred, "region", largest_component=True),
            }
            overlay = np.ascontiguousarray(color)
            for label, box_color in (
                ("object", (255, 40, 40)),
                ("occluder", (50, 160, 255)),
                ("region", (60, 220, 120)),
            ):
                bbox = class_bboxes.get(label)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                cv2.rectangle(overlay, (x0, y0), (x1, y1), box_color, 2)
                cv2.putText(
                    overlay,
                    label,
                    (x0, max(y0 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )
            overlays[key] = overlay
            measurements[key] = {
                "class_ratios": class_ratios,
                "class_bboxes": class_bboxes,
                "object_in_region": self._object_bbox_inside_region(pred),
                "object_occluder_separated": self._object_bbox_separated_from_occluder(pred),
            }
        return overlays, summaries, measurements

    def _class_bbox(self, pred: Any, label: str, largest_component: bool = False) -> list[int] | None:
        import numpy as np

        label_to_id = {label_name: idx for idx, label_name in enumerate(self.labels)}
        class_id = label_to_id.get(label)
        if class_id is None:
            return None
        mask = (pred == class_id).astype(np.uint8)
        if largest_component:
            mask = self._largest_connected_component(mask)
        yx = np.argwhere(mask)
        if yx.size == 0:
            return None
        y0, x0 = yx.min(axis=0)
        y1, x1 = yx.max(axis=0)
        return [int(x0), int(y0), int(x1), int(y1)]

    @staticmethod
    def _largest_connected_component(mask: Any) -> Any:
        import cv2
        import numpy as np

        mask = np.asarray(mask, dtype=np.uint8)
        if mask.sum() == 0:
            return mask
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest_label).astype(np.uint8)

    def _object_bbox_inside_region(self, pred: Any, margin_px: int = 2) -> bool:
        object_bbox = self._class_bbox(pred, "object", largest_component=True)
        region_bbox = self._class_bbox(pred, "region", largest_component=True)
        if object_bbox is None or region_bbox is None:
            return False
        obj_x0, obj_y0, obj_x1, obj_y1 = object_bbox
        reg_x0, reg_y0, reg_x1, reg_y1 = region_bbox
        return (
            obj_x0 >= reg_x0 - margin_px
            and obj_y0 >= reg_y0 - margin_px
            and obj_x1 <= reg_x1 + margin_px
            and obj_y1 <= reg_y1 + margin_px
        )

    def _object_bbox_separated_from_occluder(self, pred: Any, margin_px: int = 2) -> bool:
        import cv2
        import numpy as np

        label_to_id = {label_name: idx for idx, label_name in enumerate(self.labels)}
        object_id = label_to_id.get("object")
        occluder_id = label_to_id.get("occluder")
        if object_id is None or occluder_id is None:
            return False
        object_mask = (pred == object_id).astype(np.uint8)
        occluder_mask = (pred == occluder_id).astype(np.uint8)
        if object_mask.sum() == 0 or occluder_mask.sum() == 0:
            return False
        if margin_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * margin_px + 1, 2 * margin_px + 1),
            )
            object_mask = cv2.dilate(object_mask, kernel)
        return not bool((object_mask.astype(bool) & occluder_mask.astype(bool)).any())


class EvalPolicyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LeRobot Policy Evaluation")
        window_width = min(1280, max(self.root.winfo_screenwidth() - 40, 800))
        window_height = min(920, max(self.root.winfo_screenheight() - 80, 600))
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()

        self.process: subprocess.Popen[str] | None = None
        self.started_at: float | None = None
        self.started_at_iso: str | None = None
        self.stop_requested = False
        self.signal_fallback_scheduled = False
        self.process_kind = ""
        self.last_command: list[str] = []
        self.last_run: dict[str, str | float | int | bool | None] = {}
        self.result_saved = True
        self.saving_result = False
        self.active_eval_metadata: dict[str, Any] = {}
        self.active_dataset_root: Path | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.preview_queue: queue.Queue[
            tuple[dict[str, Any], dict[str, Any], float, str, dict[str, Any]]
        ] = queue.Queue(maxsize=1)
        self.checkpoint_options: list[CheckpointOption] = []
        self.robot: Any | None = None
        self.robot_lock = threading.RLock()
        self.connected = False
        self.connecting = False
        self.eval_running = False
        self.stop_policy_requested = False
        self.teleop_running = False
        self.stop_teleop_requested = False
        self.reset_running = False
        self.initial_joint_action: dict[str, float] | None = None
        self.current_images: dict[str, Any] = {}
        self.current_segmentation_images: dict[str, Any] = {}
        self.current_model_mask_images: dict[str, Any] = {}
        self.preview_labels: dict[str, ttk.Label] = {}
        self.segmentation_preview_labels: dict[str, ttk.Label] = {}
        self.segmentation_summary_labels: dict[str, ttk.Label] = {}
        self.model_mask_preview_labels: dict[str, ttk.Label] = {}
        self.model_mask_summary_labels: dict[str, ttk.Label] = {}
        self.segmentation_model: SegmentationPreviewModel | None = None
        self.segmentation_loaded_logged = False
        self.segmentation_error_logged = False
        self.segmentation_eval_metrics = self._new_segmentation_eval_metrics()
        self.connected_preview_stop: threading.Event | None = None
        self.connected_preview_thread: threading.Thread | None = None
        self.camera_preview_dialog: tk.Toplevel | None = None
        self.result_dialog: tk.Toplevel | None = None
        self.camera_preview_stop: threading.Event | None = None
        self.config_preset_paths: dict[str, Path] = {}

        self.vars = {
            "config_preset": tk.StringVar(value=""),
            "policy_type": tk.StringVar(value=DEFAULT_POLICY_TYPE),
            "checkpoint": tk.StringVar(value=""),
            "checkpoint_path": tk.StringVar(value=str(DEFAULT_POLICY_PATH)),
            "conda_env": tk.StringVar(value="lerobot"),
            "robot_mode": tk.StringVar(value="bimanual"),
            "robot_type": tk.StringVar(value="bi_so_follower"),
            "robot_port": tk.StringVar(value=DEFAULT_LEFT_FOLLOWER_PORT),
            "robot_id": tk.StringVar(value="eval_bimanual_follower"),
            "left_follower_port": tk.StringVar(value=DEFAULT_LEFT_FOLLOWER_PORT),
            "right_follower_port": tk.StringVar(value=DEFAULT_RIGHT_FOLLOWER_PORT),
            "left_follower_id": tk.StringVar(value="my_awesome_follower_arm"),
            "right_follower_id": tk.StringVar(value="my_awesome_follower_arm_r"),
            "calibration_dir": tk.StringVar(value=str(DEFAULT_CALIBRATION_DIR)),
            "calibration_match": tk.StringVar(value="Board serial"),
            "realsense_serial": tk.StringVar(value=""),
            "front_camera_type": tk.StringVar(value="opencv"),
            "front_camera_choice": tk.StringVar(value=""),
            "opencv_front": tk.StringVar(value="/dev/video10"),
            "opencv_side": tk.StringVar(value="/dev/video4"),
            "camera_config": tk.StringVar(value=""),
            "dataset_repo_id": tk.StringVar(value="seeed/eval_test"),
            "dataset_root": tk.StringVar(value=str(DEFAULT_EVAL_ROOT)),
            "task": tk.StringVar(value=TASK_CHOICES[0]),
            "episode_time_s": tk.StringVar(value="25"),
            "fps": tk.StringVar(value="30"),
            "model_chunk_size": tk.StringVar(value="N/A"),
            "prediction_steps": tk.StringVar(value=""),
            "n_action_steps": tk.StringVar(value=str(DEFAULT_DIFFUSION_REPLAN_STEPS)),
            "num_inference_steps": tk.StringVar(value=str(DEFAULT_DIFFUSION_INFERENCE_STEPS)),
            "noise_scheduler_type": tk.StringVar(value="DDPM"),
            "execution_mode": tk.StringVar(value=DEFAULT_DIFFUSION_EXECUTION_MODE),
            "camera_read_mode": tk.StringVar(value=DEFAULT_CAMERA_READ_MODE),
            "segmentation_model_path": tk.StringVar(value=str(DEFAULT_SEGMENTATION_MODEL_PATH)),
            "segmentation_device": tk.StringVar(value="auto"),
            "fusion_steps": tk.StringVar(value="0"),
            "fusion_history_weight": tk.StringVar(value="0"),
            "grid_rows": tk.StringVar(value="3"),
            "grid_cols": tk.StringVar(value="4"),
            "grid_cell": tk.StringVar(value="(0,0)"),
            "reset_time_s": tk.StringVar(value="5.0"),
            "extra_args": tk.StringVar(
                value="--display_data=false --dataset.push_to_hub=false --dataset.num_episodes=1 --dataset.vcodec=h264"
            ),
            "trials_per_grid": tk.StringVar(value=str(DEFAULT_TRIALS_PER_GRID)),
            "grid_stats": tk.StringVar(value="No saved trials for the selected policy/grid."),
            "status": tk.StringVar(value="Click Check Connect first, then Start."),
            "elapsed": tk.StringVar(value="Elapsed: 00:00"),
            "result": tk.StringVar(value="failure"),
            "object_exposed": tk.StringVar(value="no"),
            "object_pushed_out": tk.StringVar(value="no"),
            "object_push_success": tk.StringVar(value="no"),
            "result_tags": tk.StringVar(value=""),
        }
        self.enable_realsense = tk.BooleanVar(value=False)
        self.include_side_camera = tk.BooleanVar(value=True)
        self.low_resolution = tk.BooleanVar(value=True)
        self.auto_reset_after_policy = tk.BooleanVar(value=True)
        self.lock_grippers = tk.BooleanVar(value=True)
        self.save_video = tk.BooleanVar(value=True)
        self.use_amp = tk.BooleanVar(value=True)
        self.enable_segmentation_preview = tk.BooleanVar(value=True)
        self.camera_options: dict[str, tuple[str, str]] = {}

        self._build_ui()
        self._refresh_camera_config_preview()
        for key in ("realsense_serial", "opencv_front", "opencv_side"):
            self.vars[key].trace_add("write", lambda *_args: self._refresh_camera_config_preview())
        for key in ("segmentation_model_path", "segmentation_device"):
            self.vars[key].trace_add("write", self._reset_segmentation_preview_model)
        self.vars["front_camera_type"].trace_add("write", self._on_front_camera_type_changed)
        self.enable_realsense.trace_add("write", lambda *_args: self._refresh_camera_config_preview())
        self.include_side_camera.trace_add("write", lambda *_args: self._refresh_camera_config_preview())
        self.low_resolution.trace_add("write", lambda *_args: self._refresh_camera_config_preview())
        for key in ("grid_rows", "grid_cols"):
            self.vars[key].trace_add("write", self._refresh_grid_cells)
        for key in ("policy_type", "task", "grid_cell", "dataset_root", "trials_per_grid"):
            self.vars[key].trace_add("write", self._refresh_grid_stats)
        self.vars["result"].trace_add("write", self._on_result_changed)
        self._refresh_grid_cells()
        self._on_result_changed()
        self.refresh_config_presets()
        self.refresh_checkpoints()
        self.root.after(200, self._tick)

    def _configure_style(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=12)
        self.root.option_add("*Font", default_font)

        style = ttk.Style(self.root)
        style.configure("TButton", font=("TkDefaultFont", 12), padding=(14, 8))
        style.configure("TLabel", font=("TkDefaultFont", 12))
        style.configure("TEntry", font=("TkTextFont", 12), padding=(4, 4))
        style.configure("TCombobox", font=("TkTextFont", 12), padding=(4, 4))

    def _confirm(
        self,
        title: str,
        message: str,
        confirm_text: str = "Confirm",
        cancel_text: str = "Cancel",
        destructive: bool = False,
    ) -> bool:
        """Show an application-owned confirmation dialog with explicit English button labels."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        result = {"confirmed": False}
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text=message, wraplength=460, justify=tk.LEFT).pack(fill=tk.X)

        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(16, 0))

        def finish(confirmed: bool) -> None:
            result["confirmed"] = confirmed
            dialog.destroy()

        confirm_button = ttk.Button(
            buttons,
            text=confirm_text,
            command=lambda: finish(True),
        )
        confirm_button.pack(side=tk.RIGHT)
        cancel_button = ttk.Button(
            buttons,
            text=cancel_text,
            command=lambda: finish(False),
        )
        cancel_button.pack(side=tk.RIGHT, padx=(0, 8))

        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(False))
        dialog.bind("<Escape>", lambda _event: finish(False))
        dialog.bind("<Return>", lambda _event: finish(True))
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max((self.root.winfo_width() - dialog.winfo_reqwidth()) // 2, 0)
        y = self.root.winfo_rooty() + max((self.root.winfo_height() - dialog.winfo_reqheight()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")
        if destructive:
            cancel_button.focus_set()
        else:
            confirm_button.focus_set()
        self.root.wait_window(dialog)
        return bool(result["confirmed"])

    def _alert(self, title: str, message: str, button_text: str = "OK") -> None:
        """Show an application-owned alert so button labels never depend on system locale."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text=message, wraplength=520, justify=tk.LEFT).pack(fill=tk.X)
        button = ttk.Button(body, text=button_text, command=dialog.destroy)
        button.pack(side=tk.RIGHT, pady=(16, 0))

        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.bind("<Return>", lambda _event: dialog.destroy())
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max((self.root.winfo_width() - dialog.winfo_reqwidth()) // 2, 0)
        y = self.root.winfo_rooty() + max((self.root.winfo_height() - dialog.winfo_reqheight()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")
        button.focus_set()
        self.root.wait_window(dialog)

    def _choose(
        self,
        title: str,
        message: str,
        choices: tuple[tuple[str, str], ...],
        cancel_text: str = "Cancel",
    ) -> str | None:
        """Show a choice dialog with explicit application-owned button labels."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        selected: dict[str, str | None] = {"value": None}
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text=message, wraplength=560, justify=tk.LEFT).pack(fill=tk.X)
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(16, 0))

        def finish(value: str | None) -> None:
            selected["value"] = value
            dialog.destroy()

        cancel_button = ttk.Button(buttons, text=cancel_text, command=lambda: finish(None))
        cancel_button.pack(side=tk.RIGHT)
        for value, label in reversed(choices):
            ttk.Button(buttons, text=label, command=lambda item=value: finish(item)).pack(
                side=tk.RIGHT, padx=(0, 8)
            )

        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        dialog.bind("<Escape>", lambda _event: finish(None))
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max((self.root.winfo_width() - dialog.winfo_reqwidth()) // 2, 0)
        y = self.root.winfo_rooty() + max((self.root.winfo_height() - dialog.winfo_reqheight()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")
        cancel_button.focus_set()
        self.root.wait_window(dialog)
        return selected["value"]

    def _ask_integer(
        self,
        title: str,
        message: str,
        *,
        minimum: int,
        maximum: int,
        initial: int,
    ) -> int | None:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        selected: dict[str, int | None] = {"value": None}
        value_var = tk.StringVar(value=str(initial))
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text=message, wraplength=560, justify=tk.LEFT).pack(fill=tk.X)
        entry = ttk.Entry(body, textvariable=value_var, width=12)
        entry.pack(anchor=tk.W, pady=(12, 0))
        error_var = tk.StringVar(value="")
        ttk.Label(body, textvariable=error_var).pack(anchor=tk.W, pady=(6, 0))
        buttons = ttk.Frame(body)
        buttons.pack(fill=tk.X, pady=(14, 0))

        def finish(value: int | None) -> None:
            selected["value"] = value
            dialog.destroy()

        def submit() -> None:
            try:
                value = int(value_var.get())
            except ValueError:
                error_var.set("Enter an integer.")
                return
            if not minimum <= value <= maximum:
                error_var.set(f"Enter a value from {minimum} to {maximum}.")
                return
            finish(value)

        ttk.Button(buttons, text="Cancel", command=lambda: finish(None)).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Continue", command=submit).pack(side=tk.RIGHT, padx=(0, 8))
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(None))
        dialog.bind("<Escape>", lambda _event: finish(None))
        dialog.bind("<Return>", lambda _event: submit())
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max((self.root.winfo_width() - dialog.winfo_reqwidth()) // 2, 0)
        y = self.root.winfo_rooty() + max((self.root.winfo_height() - dialog.winfo_reqheight()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")
        entry.focus_set()
        entry.selection_range(0, tk.END)
        self.root.wait_window(dialog)
        return selected["value"]

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.main_canvas = tk.Canvas(main_frame, highlightthickness=0)
        main_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        content = ttk.Frame(self.main_canvas)
        content_window = self.main_canvas.create_window((0, 0), window=content, anchor=tk.NW)

        def update_scroll_region(_event: tk.Event) -> None:
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        def fit_content_width(event: tk.Event) -> None:
            self.main_canvas.itemconfigure(content_window, width=event.width)

        content.bind("<Configure>", update_scroll_region)
        self.main_canvas.bind("<Configure>", fit_content_width)
        self.root.bind("<MouseWheel>", self._scroll_main_view, add="+")
        self.root.bind("<Button-4>", self._scroll_main_view, add="+")
        self.root.bind("<Button-5>", self._scroll_main_view, add="+")

        top = ttk.Frame(content, padding=14)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        row = 0
        ttk.Label(top, text="Config preset").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        self.config_preset_combo = ttk.Combobox(
            top,
            textvariable=self.vars["config_preset"],
            state="readonly",
            width=18,
        )
        self.config_preset_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        self.config_preset_combo.bind("<<ComboboxSelected>>", self.load_selected_config_preset)
        ttk.Button(top, text="Refresh Configs", command=self.refresh_config_presets).grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )
        ttk.Button(top, text="Load Config", command=self.load_selected_config_preset).grid(
            row=row, column=3, sticky=tk.W, padx=(8, 0), pady=5
        )

        row += 1
        ttk.Label(top, text="Algorithm").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["policy_type"],
            values=POLICY_TYPES,
            state="readonly",
            width=18,
        ).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Button(top, text="Refresh Weights", command=self.refresh_checkpoints).grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )

        row += 1
        ttk.Label(top, text="Checkpoint").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        self.checkpoint_combo = ttk.Combobox(top, textvariable=self.vars["checkpoint"], state="readonly")
        self.checkpoint_combo.grid(row=row, column=1, columnspan=2, sticky=tk.EW, pady=5)
        self.checkpoint_combo.bind("<<ComboboxSelected>>", self.on_checkpoint_selected)
        ttk.Button(top, text="Browse", command=self.browse_checkpoint).grid(row=row, column=3, sticky=tk.W, padx=(8, 0))

        row += 1
        ttk.Label(top, text="Weight path").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["checkpoint_path"]).grid(row=row, column=1, columnspan=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Conda env").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["conda_env"], width=18).grid(row=row, column=1, sticky=tk.W)
        ttk.Label(top, text="Robot mode").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["robot_mode"],
            values=("bimanual",),
            state="readonly",
            width=24,
        ).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Label(top, text="Robot type").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["robot_type"],
            values=("bi_so_follower",),
            state="readonly",
            width=24,
        ).grid(row=row, column=1, sticky=tk.W)
        ttk.Label(top, text="Runtime id").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["robot_id"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Left port").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["left_follower_port"]).grid(row=row, column=1, sticky=tk.EW)
        ttk.Label(top, text="Right port").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["right_follower_port"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Left calib id").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["left_follower_id"]).grid(row=row, column=1, sticky=tk.EW)
        ttk.Label(top, text="Right calib id").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["right_follower_id"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Calibration dir").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["calibration_dir"]).grid(row=row, column=1, columnspan=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Calibration match").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["calibration_match"],
            values=("Board serial", "Arm id"),
            state="readonly",
            width=18,
        ).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(top, text="Follower only; leaders are not connected for policy evaluation.").grid(
            row=row, column=2, columnspan=2, sticky=tk.W, padx=(16, 0), pady=5
        )

        row += 1
        ttk.Label(top, text="Front camera type").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["front_camera_type"],
            values=("opencv", "realsense"),
            state="readonly",
            width=18,
        ).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Button(top, text="Scan Cameras", command=self.find_cameras).grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8)
        )
        ttk.Checkbutton(top, text="Include side", variable=self.include_side_camera).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Label(top, text="Detected front").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        self.front_camera_combo = ttk.Combobox(
            top,
            textvariable=self.vars["front_camera_choice"],
            state="readonly",
        )
        self.front_camera_combo.grid(row=row, column=1, columnspan=2, sticky=tk.EW)
        self.front_camera_combo.bind("<<ComboboxSelected>>", self._on_front_camera_selected)
        ttk.Button(top, text="Preview", command=self.preview_front_camera).grid(
            row=row, column=3, sticky=tk.W, padx=(8, 0)
        )

        row += 1
        ttk.Label(top, text="OpenCV front").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["opencv_front"]).grid(row=row, column=1, sticky=tk.EW)
        ttk.Label(top, text="RealSense serial").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["realsense_serial"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="OpenCV side").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["opencv_side"]).grid(row=row, column=1, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Camera config").grid(row=row, column=0, sticky=tk.NW, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["camera_config"], state="readonly").grid(
            row=row, column=1, columnspan=3, sticky=tk.EW
        )

        row += 1
        ttk.Label(top, text="Dataset repo").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["dataset_repo_id"]).grid(row=row, column=1, sticky=tk.EW)
        ttk.Label(top, text="Save parent").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        save_frame = ttk.Frame(top)
        save_frame.grid(row=row, column=3, sticky=tk.EW)
        save_frame.columnconfigure(0, weight=1)
        ttk.Entry(save_frame, textvariable=self.vars["dataset_root"]).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(save_frame, text="Browse", command=self.browse_dataset_root).grid(row=0, column=1, padx=(8, 0))

        row += 1
        ttk.Label(top, text="Task type").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["task"],
            values=TASK_CHOICES,
            state="readonly",
        ).grid(row=row, column=1, columnspan=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Initial region grid rows×cols").grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        grid_shape_frame = ttk.Frame(top)
        grid_shape_frame.grid(row=row, column=1, sticky=tk.W)
        ttk.Entry(grid_shape_frame, textvariable=self.vars["grid_rows"], width=5).grid(row=0, column=0)
        ttk.Label(grid_shape_frame, text="×").grid(row=0, column=1, padx=4)
        ttk.Entry(grid_shape_frame, textvariable=self.vars["grid_cols"], width=5).grid(row=0, column=2)
        ttk.Label(top, text="Target cell (x,y)").grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )
        cell_frame = ttk.Frame(top)
        cell_frame.grid(row=row, column=3, sticky=tk.W)
        self.grid_cell_combo = ttk.Combobox(
            cell_frame,
            textvariable=self.vars["grid_cell"],
            state="readonly",
            width=12,
        )
        self.grid_cell_combo.grid(row=0, column=0, sticky=tk.W)
        ttk.Button(cell_frame, text="Next Open Cell", command=self.select_next_open_grid_cell).grid(
            row=0, column=1, padx=(8, 0)
        )

        row += 1
        ttk.Label(top, text="Trials per cell").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["trials_per_grid"], width=12).grid(
            row=row, column=1, sticky=tk.W
        )
        ttk.Label(top, textvariable=self.vars["grid_stats"]).grid(
            row=row, column=2, columnspan=2, sticky=tk.W, padx=(16, 0), pady=5
        )

        row += 1
        ttk.Label(top, text="Episode sec").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["episode_time_s"], width=12).grid(row=row, column=1, sticky=tk.W)
        ttk.Label(top, text="FPS").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["fps"], width=12).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Label(top, text="Model chunk size").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["model_chunk_size"], width=12, state="readonly").grid(
            row=row, column=1, sticky=tk.W
        )
        ttk.Label(top, text="Prediction steps").grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )
        ttk.Entry(top, textvariable=self.vars["prediction_steps"], width=12).grid(
            row=row, column=3, sticky=tk.W
        )

        row += 1
        ttk.Label(top, text="Replan interval (steps)").grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Entry(top, textvariable=self.vars["n_action_steps"], width=12).grid(
            row=row, column=1, sticky=tk.W
        )
        ttk.Label(top, text="Execution mode").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        execution_mode_frame = ttk.Frame(top)
        execution_mode_frame.grid(row=row, column=3, sticky=tk.W)
        ttk.Radiobutton(
            execution_mode_frame,
            text="Sync",
            variable=self.vars["execution_mode"],
            value="synchronous",
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            execution_mode_frame,
            text="Async",
            variable=self.vars["execution_mode"],
            value="asynchronous",
        ).pack(side=tk.LEFT, padx=(12, 0))

        row += 1
        ttk.Label(top, text="Diffusion inference steps").grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Entry(top, textvariable=self.vars["num_inference_steps"], width=12).grid(
            row=row, column=1, sticky=tk.W
        )
        ttk.Label(top, text="Diffusion scheduler").grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )
        ttk.Combobox(
            top,
            textvariable=self.vars["noise_scheduler_type"],
            values=("checkpoint", "DDPM", "DDIM"),
            state="readonly",
            width=12,
        ).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Checkbutton(top, text="Use AMP", variable=self.use_amp).grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Checkbutton(top, text="Low res cameras", variable=self.low_resolution).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(top, text="Camera read").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["camera_read_mode"],
            values=("wait_new_frame", "latest_nonblocking"),
            state="readonly",
            width=20,
        ).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Checkbutton(
            top,
            text="Segmentation preview",
            variable=self.enable_segmentation_preview,
        ).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["segmentation_model_path"]).grid(
            row=row, column=1, columnspan=2, sticky=tk.EW, pady=5
        )
        ttk.Entry(top, textvariable=self.vars["segmentation_device"], width=12).grid(
            row=row, column=3, sticky=tk.W, pady=5
        )

        row += 1
        ttk.Label(top, text="Custom fusion steps (0=off)").grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Entry(top, textvariable=self.vars["fusion_steps"], width=12).grid(
            row=row, column=1, sticky=tk.W
        )
        ttk.Label(top, text="Custom history weight [0,1]").grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )
        ttk.Entry(top, textvariable=self.vars["fusion_history_weight"], width=12).grid(
            row=row, column=3, sticky=tk.W
        )

        row += 1
        ttk.Checkbutton(top, text="Auto reset after run", variable=self.auto_reset_after_policy).grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Checkbutton(top, text="Lock grippers", variable=self.lock_grippers).grid(
            row=row, column=1, sticky=tk.W, pady=5
        )
        ttk.Label(top, text="Reset sec").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["reset_time_s"], width=12).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Checkbutton(top, text="Save MP4 video", variable=self.save_video).grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Label(
            top,
            text="Off: keep trajectory + image frames, skip MP4 encoding",
        ).grid(row=row, column=1, columnspan=3, sticky=tk.W, pady=5)

        row += 1
        ttk.Label(top, text="Extra args").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["extra_args"]).grid(row=row, column=1, columnspan=3, sticky=tk.EW)

        buttons = ttk.Frame(content, padding=(14, 0, 14, 10))
        buttons.pack(fill=tk.X)
        self.connect_button = ttk.Button(buttons, text="Check Connect", command=self.check_connect)
        self.connect_button.pack(side=tk.LEFT, ipady=4)
        self.start_button = ttk.Button(buttons, text="Start", command=self.start_eval)
        self.start_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.stop_button = ttk.Button(buttons, text="End / Stop", command=self.stop_eval, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.teleop_button = ttk.Button(buttons, text="Start Teleop", command=self.start_teleop, state=tk.DISABLED)
        self.stop_teleop_button = ttk.Button(
            buttons, text="Stop Teleop", command=self.stop_teleop, state=tk.DISABLED
        )
        self.capture_reset_button = ttk.Button(
            buttons, text="Capture Reset Pose", command=self.capture_reset_pose, state=tk.DISABLED
        )
        self.capture_reset_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.reset_button = ttk.Button(buttons, text="Reset Arms", command=self.reset_arms, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.disconnect_button = ttk.Button(buttons, text="Disconnect", command=self.disconnect_robot, state=tk.DISABLED)
        self.disconnect_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="Open Save Folder", command=self.open_save_folder).pack(
            side=tk.LEFT, padx=(10, 0), ipady=4
        )
        ttk.Label(buttons, textvariable=self.vars["elapsed"]).pack(side=tk.RIGHT)

        default_pose_frame = ttk.Frame(content, padding=(14, 0, 14, 10))
        default_pose_frame.pack(fill=tk.X)
        ttk.Label(default_pose_frame, text="Validated baseline: 2026-06-22 center-cell 9/10 success median pose").pack(
            side=tk.LEFT
        )
        self.default_pose_button = ttk.Button(
            default_pose_frame,
            text="Go to 90% Pose",
            command=self.go_to_default_success_pose,
            state=tk.DISABLED,
        )
        self.default_pose_button.pack(side=tk.LEFT, padx=(12, 0), ipady=4)

        result_frame = ttk.LabelFrame(content, text="Result", padding=12)
        result_frame.pack(fill=tk.X, padx=14, pady=(0, 10))
        ttk.Label(result_frame, text="Last run").pack(side=tk.LEFT)
        ttk.Combobox(
            result_frame,
            textvariable=self.vars["result"],
            values=RESULT_CHOICES,
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(result_frame, text="Review / Save Result", command=self._show_result_dialog).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        ttk.Label(result_frame, textvariable=self.vars["status"]).pack(side=tk.LEFT, padx=(16, 0))

        log_frame = ttk.Frame(content, padding=(14, 0, 14, 14))
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(1, weight=0)
        preview_frame = ttk.Frame(log_frame)
        preview_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.rowconfigure(4, weight=1)
        for col in range(3):
            preview_frame.columnconfigure(col, weight=1, minsize=PREVIEW_MIN_WIDTH)
        for view_index, key in enumerate(("front", "side")):
            header_row = view_index * 3
            image_row = header_row + 1
            summary_row = header_row + 2
            headers = (key, f"{key} U-Net", f"{key} model masks")
            for col, text in enumerate(headers):
                ttk.Label(preview_frame, text=text).grid(
                    row=header_row,
                    column=col,
                    sticky=tk.W,
                    padx=(0 if col == 0 else 8, 0),
                    pady=(8 if view_index else 0, 0),
                )

            raw_label = ttk.Label(preview_frame, anchor=tk.CENTER)
            raw_label.grid(row=image_row, column=0, sticky=tk.NSEW)
            self.preview_labels[key] = raw_label

            seg_label = ttk.Label(preview_frame, anchor=tk.CENTER)
            seg_label.grid(row=image_row, column=1, sticky=tk.NSEW, padx=(8, 0))
            self.segmentation_preview_labels[key] = seg_label

            model_mask_label = ttk.Label(preview_frame, anchor=tk.CENTER)
            model_mask_label.grid(row=image_row, column=2, sticky=tk.NSEW, padx=(8, 0))
            self.model_mask_preview_labels[key] = model_mask_label

            summary_label = ttk.Label(preview_frame, text="", wraplength=PREVIEW_MIN_WIDTH)
            summary_label.grid(row=summary_row, column=1, sticky=tk.W, padx=(8, 0))
            self.segmentation_summary_labels[key] = summary_label

            model_summary_label = ttk.Label(preview_frame, text="", wraplength=PREVIEW_MIN_WIDTH)
            model_summary_label.grid(row=summary_row, column=2, sticky=tk.W, padx=(8, 0))
            self.model_mask_summary_labels[key] = model_summary_label

        right_log = ttk.Frame(log_frame)
        right_log.grid(row=0, column=1, sticky=tk.NS)
        ttk.Label(right_log, text="Log / Action").pack(anchor=tk.W)
        self.log_text = tk.Text(right_log, wrap=tk.WORD, width=24, height=14)
        scrollbar = ttk.Scrollbar(right_log, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

    def _reset_segmentation_preview_model(self, *_args: Any) -> None:
        self.segmentation_model = None
        self.segmentation_loaded_logged = False
        self.segmentation_error_logged = False

    def _segmentation_base_task(self) -> str:
        if not hasattr(self, "vars"):
            return ""
        return self.vars["task"].get().strip().removesuffix("_r")

    def _segmentation_object_present_min_ratio(self) -> float:
        if self._segmentation_base_task() == "screw":
            return SEGMENTATION_SCREW_OBJECT_PRESENT_MIN_RATIO
        return SEGMENTATION_OBJECT_PRESENT_MIN_RATIO

    def _new_segmentation_eval_metrics(self) -> dict[str, Any]:
        return {
            "segmentation_enabled": self.enable_segmentation_preview.get() if hasattr(self, "enable_segmentation_preview") else True,
            "segmentation_frames": 0,
            "segmentation_object_present_min_ratio": self._segmentation_object_present_min_ratio(),
            "segmentation_occluder_complete_threshold": SEGMENTATION_OCCLUDER_COMPLETE_RATIO,
            "segmentation_side_fallback_enabled": False,
            "segmentation_side_diagnostic_only": True,
            "segmentation_vote_window": SEGMENTATION_VOTE_WINDOW,
            "segmentation_vote_required": SEGMENTATION_VOTE_REQUIRED,
            "recent_segmentation_votes": [],
            "recent_exposed_votes": 0,
            "recent_separated_votes": 0,
            "recent_in_region_votes": 0,
            "recent_front_coverage_votes": 0,
            "recent_success_votes": 0,
            "auto_object_exposed": False,
            "auto_object_separated": False,
            "auto_object_reached_region": False,
            "auto_final_front_coverage": False,
            "auto_task_complete": False,
            "final_object_exposed": False,
            "final_object_separated": False,
            "final_object_reached_region": False,
            "final_front_coverage": False,
            "final_task_success": False,
            "final_failure_reason": "not_started",
            "auto_object_exposed_time_s": None,
            "auto_object_separated_time_s": None,
            "auto_object_reached_region_time_s": None,
            "auto_final_front_coverage_time_s": None,
            "auto_task_complete_time_s": None,
            "max_occluder_coverage_after_region": 0.0,
            "max_front_occluder_coverage_after_region": 0.0,
            "max_occluder_coverage": 0.0,
            "last_front_occluder_coverage": 0.0,
            "last_front_object_separated": False,
            "last_side_object_separated": False,
            "last_any_object_separated": False,
            "last_front_object_in_region": False,
            "last_side_object_in_region": False,
            "last_any_object_in_region": False,
            "last_front_object_detected": False,
            "last_front_object_confident": False,
            "last_side_object_detected": False,
            "last_decision_source": "none",
            "last_failure_reason": "not_started",
            "segmentation_views": {
                "front": self._new_segmentation_view_metrics(),
                "side": self._new_segmentation_view_metrics(),
            },
        }

    @staticmethod
    def _new_segmentation_view_metrics() -> dict[str, Any]:
        return {
            "object_seen": False,
            "object_occluder_separated_seen": False,
            "object_in_region_seen": False,
            "task_complete_seen": False,
            "max_object_ratio": 0.0,
            "max_region_ratio": 0.0,
            "max_occluder_ratio": 0.0,
            "max_occluder_ratio_after_region": 0.0,
            "last_object_ratio": 0.0,
            "last_region_ratio": 0.0,
            "last_occluder_ratio": 0.0,
            "last_object_in_region": False,
            "last_object_occluder_separated": False,
            "last_bboxes": {},
        }

    def _update_segmentation_eval_metrics(self, measurements: dict[str, Any]) -> None:
        if not self.eval_running or self.started_at is None:
            return
        metrics = self.segmentation_eval_metrics
        metrics["segmentation_enabled"] = self.enable_segmentation_preview.get()
        metrics["segmentation_frames"] += 1
        elapsed_s = round(time.monotonic() - self.started_at, 3)
        view_metrics = metrics["segmentation_views"]
        object_min_ratio = float(
            metrics.get("segmentation_object_present_min_ratio", SEGMENTATION_OBJECT_PRESENT_MIN_RATIO)
        )

        frame_object_in_region = False
        frame_separated_views: set[str] = set()
        frame_max_occluder = 0.0
        for view, measurement in measurements.items():
            if view not in view_metrics:
                view_metrics[view] = self._new_segmentation_view_metrics()
            ratios = measurement.get("class_ratios", {})
            object_ratio = float(ratios.get("object", 0.0))
            region_ratio = float(ratios.get("region", 0.0))
            occluder_ratio = float(ratios.get("occluder", 0.0))
            object_seen = object_ratio >= object_min_ratio
            object_in_region = bool(measurement.get("object_in_region")) and object_seen
            object_occluder_separated = bool(measurement.get("object_occluder_separated")) and object_seen

            view_record = view_metrics[view]
            view_record["last_object_ratio"] = object_ratio
            view_record["last_region_ratio"] = region_ratio
            view_record["last_occluder_ratio"] = occluder_ratio
            view_record["last_object_in_region"] = object_in_region
            view_record["last_object_occluder_separated"] = object_occluder_separated
            view_record["last_bboxes"] = measurement.get("class_bboxes", {})
            view_record["max_object_ratio"] = max(float(view_record["max_object_ratio"]), object_ratio)
            view_record["max_region_ratio"] = max(float(view_record["max_region_ratio"]), region_ratio)
            view_record["max_occluder_ratio"] = max(float(view_record["max_occluder_ratio"]), occluder_ratio)
            view_record["object_seen"] = bool(view_record["object_seen"] or object_seen)
            view_record["object_occluder_separated_seen"] = bool(
                view_record["object_occluder_separated_seen"] or object_occluder_separated
            )
            view_record["object_in_region_seen"] = bool(view_record["object_in_region_seen"] or object_in_region)

            frame_object_in_region = frame_object_in_region or object_in_region
            if object_occluder_separated:
                frame_separated_views.add(view)
            frame_max_occluder = max(frame_max_occluder, occluder_ratio)

        any_object_separated = bool(frame_separated_views)
        any_object_in_region = frame_object_in_region
        metrics["max_occluder_coverage"] = max(float(metrics["max_occluder_coverage"]), frame_max_occluder)
        front_measurement = measurements.get("front", {})
        front_view = view_metrics.get("front", {})
        side_view = view_metrics.get("side", {})
        front_object_separated = bool(front_view.get("last_object_occluder_separated"))
        front_object_in_region = bool(front_view.get("last_object_in_region"))
        side_object_separated = bool(side_view.get("last_object_occluder_separated"))
        side_object_in_region = bool(side_view.get("last_object_in_region"))
        front_object_ratio = float(front_view.get("last_object_ratio", 0.0))
        side_object_ratio = float(side_view.get("last_object_ratio", 0.0))
        front_object_detected = front_object_ratio >= object_min_ratio
        front_object_confident = front_object_ratio >= SEGMENTATION_OBJECT_PRESENT_MIN_RATIO
        side_object_detected = side_object_ratio >= object_min_ratio
        metrics["last_front_object_separated"] = front_object_separated
        metrics["last_side_object_separated"] = side_object_separated
        metrics["last_any_object_separated"] = any_object_separated
        metrics["last_front_object_in_region"] = front_object_in_region
        metrics["last_side_object_in_region"] = side_object_in_region
        metrics["last_any_object_in_region"] = any_object_in_region
        metrics["last_front_object_detected"] = front_object_detected
        metrics["last_front_object_confident"] = front_object_confident
        metrics["last_side_object_detected"] = side_object_detected

        effective_object_exposed = False
        effective_object_separated = False
        effective_object_in_region = False
        decision_source = "front_missing"
        if "front" in measurements:
            effective_object_exposed = front_object_detected
            effective_object_separated = front_object_separated
            effective_object_in_region = front_object_in_region
            if front_object_confident:
                decision_source = "front"
            elif front_object_detected:
                decision_source = "front_low_confidence"
            else:
                decision_source = "front_not_detected"

        metrics["last_decision_source"] = decision_source
        front_occluder_ratio = float(front_measurement.get("class_ratios", {}).get("occluder", 0.0))
        metrics["last_front_occluder_coverage"] = front_occluder_ratio
        frame_front_coverage = front_occluder_ratio >= SEGMENTATION_OCCLUDER_COMPLETE_RATIO
        frame_task_success = effective_object_in_region and frame_front_coverage
        vote_history = metrics["recent_segmentation_votes"]
        vote_history.append(
            {
                "object_exposed": effective_object_exposed,
                "object_separated": effective_object_separated,
                "object_in_region": effective_object_in_region,
                "front_coverage": frame_front_coverage,
                "task_success": frame_task_success,
                "decision_source": decision_source,
            }
        )
        del vote_history[:-SEGMENTATION_VOTE_WINDOW]

        def vote_count(field: str) -> int:
            return sum(bool(vote.get(field)) for vote in vote_history)

        metrics["recent_exposed_votes"] = vote_count("object_exposed")
        metrics["recent_separated_votes"] = vote_count("object_separated")
        metrics["recent_in_region_votes"] = vote_count("object_in_region")
        metrics["recent_front_coverage_votes"] = vote_count("front_coverage")
        metrics["recent_success_votes"] = vote_count("task_success")
        metrics["final_object_exposed"] = metrics["recent_exposed_votes"] >= SEGMENTATION_VOTE_REQUIRED
        metrics["final_object_separated"] = metrics["recent_separated_votes"] >= SEGMENTATION_VOTE_REQUIRED
        metrics["final_object_reached_region"] = metrics["recent_in_region_votes"] >= SEGMENTATION_VOTE_REQUIRED
        metrics["final_front_coverage"] = (
            metrics["recent_front_coverage_votes"] >= SEGMENTATION_VOTE_REQUIRED
        )
        metrics["final_task_success"] = metrics["recent_success_votes"] >= SEGMENTATION_VOTE_REQUIRED

        if metrics["final_object_exposed"] and not metrics["auto_object_exposed"]:
            metrics["auto_object_exposed"] = True
            metrics["auto_object_exposed_time_s"] = elapsed_s
        if (
            metrics["auto_object_exposed"]
            and metrics["final_object_separated"]
            and not metrics["auto_object_separated"]
        ):
            metrics["auto_object_separated"] = True
            metrics["auto_object_separated_time_s"] = elapsed_s
        if metrics["final_object_reached_region"] and not metrics["auto_object_reached_region"]:
            metrics["auto_object_reached_region"] = True
            metrics["auto_object_reached_region_time_s"] = elapsed_s
        if metrics["auto_object_reached_region"]:
            metrics["max_front_occluder_coverage_after_region"] = max(
                float(metrics["max_front_occluder_coverage_after_region"]),
                front_occluder_ratio,
            )
            metrics["max_occluder_coverage_after_region"] = max(
                float(metrics["max_occluder_coverage_after_region"]),
                frame_max_occluder,
            )
            for view, measurement in measurements.items():
                occluder_ratio = float(measurement.get("class_ratios", {}).get("occluder", 0.0))
                view_record = view_metrics[view]
                view_record["max_occluder_ratio_after_region"] = max(
                    float(view_record["max_occluder_ratio_after_region"]),
                    occluder_ratio,
                )
                if (
                    view == "front"
                    and front_object_in_region
                    and occluder_ratio >= SEGMENTATION_OCCLUDER_COMPLETE_RATIO
                ):
                    view_record["task_complete_seen"] = True
            if metrics["final_front_coverage"] and not metrics["auto_final_front_coverage"]:
                metrics["auto_final_front_coverage"] = True
                metrics["auto_final_front_coverage_time_s"] = elapsed_s
            if metrics["final_task_success"] and not metrics["auto_task_complete"]:
                metrics["auto_task_complete"] = True
                metrics["auto_task_complete_time_s"] = elapsed_s
                self.stop_policy_requested = True
                self.log_queue.put(
                    "[SEG] Auto task complete: object reached region and front occluder coverage "
                    f"reached {100.0 * front_occluder_ratio:.1f}%; "
                    f"votes={metrics['recent_success_votes']}/{SEGMENTATION_VOTE_WINDOW}, "
                    f"source={decision_source}; stopping policy."
                )
        metrics["final_failure_reason"] = self._segmentation_final_failure_reason(metrics)
        metrics["last_failure_reason"] = metrics["final_failure_reason"]

    def _segmentation_metric_record(self) -> dict[str, Any]:
        metrics = self.segmentation_eval_metrics
        return {
            "segmentation_auto_metrics": metrics,
            "final_object_exposed": bool(metrics["final_object_exposed"]),
            "final_object_separated": bool(metrics["final_object_separated"]),
            "final_object_reached_region": bool(metrics["final_object_reached_region"]),
            "final_front_coverage": bool(metrics["final_front_coverage"]),
            "final_task_success": bool(metrics["final_task_success"]),
            "final_failure_reason": self._segmentation_final_failure_reason(metrics),
            "last_front_object_separated": bool(metrics.get("last_front_object_separated")),
            "last_side_object_separated": bool(metrics.get("last_side_object_separated")),
            "last_any_object_separated": bool(metrics.get("last_any_object_separated")),
            "last_front_object_in_region": bool(metrics.get("last_front_object_in_region")),
            "last_side_object_in_region": bool(metrics.get("last_side_object_in_region")),
            "last_any_object_in_region": bool(metrics.get("last_any_object_in_region")),
            "auto_object_exposed": bool(metrics["auto_object_exposed"]),
            "auto_object_separated": bool(metrics["auto_object_separated"]),
            "auto_object_reached_region": bool(metrics["auto_object_reached_region"]),
            "auto_final_front_coverage": bool(metrics["auto_final_front_coverage"]),
            "auto_task_complete": bool(metrics["auto_task_complete"]),
            "auto_failure_reason": self._segmentation_failure_reason(metrics),
            "auto_max_front_occluder_coverage_after_region": float(metrics["max_front_occluder_coverage_after_region"]),
            "auto_max_occluder_coverage_after_region": float(metrics["max_occluder_coverage_after_region"]),
            "segmentation_object_present_min_ratio": float(metrics["segmentation_object_present_min_ratio"]),
            "segmentation_recent_success_votes": int(metrics["recent_success_votes"]),
            "segmentation_vote_window": int(metrics["segmentation_vote_window"]),
            "segmentation_vote_required": int(metrics["segmentation_vote_required"]),
            "segmentation_decision_source": str(metrics["last_decision_source"]),
        }

    def _apply_auto_result_fields(self) -> None:
        metrics = self.segmentation_eval_metrics
        self.vars["result"].set("success" if metrics["final_task_success"] else "failure")
        self.vars["object_exposed"].set("yes" if metrics["final_object_exposed"] else "no")
        self.vars["object_pushed_out"].set("yes" if metrics["final_object_separated"] else "no")
        self.vars["object_push_success"].set("yes" if metrics["final_task_success"] else "no")

    def _auto_result_summary_text(self) -> str:
        metrics = self.segmentation_eval_metrics
        return (
            "Auto segmentation: "
            f"final exposed={'yes' if metrics['final_object_exposed'] else 'no'}, "
            f"final separated(voted)={'yes' if metrics['final_object_separated'] else 'no'}, "
            f"final in_region(voted)={'yes' if metrics['final_object_reached_region'] else 'no'}, "
            f"final front_cover={'yes' if metrics['final_front_coverage'] else 'no'} "
            f"({100.0 * float(metrics['last_front_occluder_coverage']):.1f}%), "
            f"success_votes={metrics['recent_success_votes']}/{metrics['segmentation_vote_window']}, "
            f"source={metrics['last_decision_source']}, "
            f"ever exposed={'yes' if metrics['auto_object_exposed'] else 'no'}, "
            f"ever separated={'yes' if metrics['auto_object_separated'] else 'no'}, "
            f"ever in_region={'yes' if metrics['auto_object_reached_region'] else 'no'}, "
            f"reason={self._segmentation_final_failure_reason(metrics)}"
        )

    def _auto_result_detail_rows(self) -> list[tuple[str, str, str]]:
        metrics = self.segmentation_eval_metrics

        def yes_no(value: Any) -> str:
            return "yes" if value else "no"

        def pct(value: Any) -> str:
            try:
                return f"{100.0 * float(value):.1f}%"
            except (TypeError, ValueError):
                return "n/a"

        return [
            ("Final object exposed", yes_no(metrics["final_object_exposed"]), "core"),
            ("Final separated from occlusion (voted)", yes_no(metrics["final_object_separated"]), "core"),
            ("Front separated from occlusion", yes_no(metrics.get("last_front_object_separated")), "core"),
            ("Side separated from occlusion", yes_no(metrics.get("last_side_object_separated")), "diagnostic"),
            ("Final object in region (voted)", yes_no(metrics["final_object_reached_region"]), "core"),
            ("Front object in region", yes_no(metrics.get("last_front_object_in_region")), "core"),
            ("Side object in region", yes_no(metrics.get("last_side_object_in_region")), "diagnostic"),
            (
                "Final front cover",
                f"{yes_no(metrics['final_front_coverage'])} ({pct(metrics['last_front_occluder_coverage'])})",
                "core",
            ),
            (
                "Recent success votes",
                f"{metrics['recent_success_votes']}/{metrics['segmentation_vote_window']} "
                f"(need {metrics['segmentation_vote_required']})",
                "core",
            ),
            ("Decision source", str(metrics["last_decision_source"]), "diagnostic"),
            ("Ever object exposed", yes_no(metrics["auto_object_exposed"]), "diagnostic"),
            ("Ever separated from occlusion", yes_no(metrics["auto_object_separated"]), "diagnostic"),
            ("Ever object reached region", yes_no(metrics["auto_object_reached_region"]), "diagnostic"),
            ("Ever task complete", yes_no(metrics["auto_task_complete"]), "diagnostic"),
            ("Final failure reason", self._segmentation_final_failure_reason(metrics), "diagnostic"),
            ("Ever failure reason", self._segmentation_failure_reason(metrics), "diagnostic"),
        ]

    @staticmethod
    def _segmentation_failure_reason(metrics: dict[str, Any]) -> str:
        if metrics.get("auto_task_complete"):
            return "success"
        if not metrics.get("auto_object_exposed"):
            return "timeout_no_object_exposed"
        if not metrics.get("auto_object_separated"):
            return "timeout_exposed_not_separated_in_any_view"
        if not metrics.get("auto_object_reached_region"):
            return "timeout_separated_not_reached_region"
        if not metrics.get("auto_final_front_coverage"):
            return "timeout_region_reached_front_coverage_below_50"
        return "timeout_unknown"

    @staticmethod
    def _segmentation_final_failure_reason(metrics: dict[str, Any]) -> str:
        if metrics.get("final_task_success"):
            return "success"
        if not metrics.get("final_object_exposed"):
            if metrics.get("auto_object_exposed"):
                return "final_object_not_visible_after_prior_exposure"
            return "timeout_no_object_exposed"
        if not metrics.get("final_object_separated"):
            if metrics.get("auto_object_separated"):
                return "final_not_separated_after_prior_separation"
            return "timeout_exposed_not_separated_in_any_view"
        if not metrics.get("final_object_reached_region"):
            if metrics.get("auto_object_reached_region"):
                return "final_object_left_region_after_prior_region_reached"
            return "timeout_separated_not_reached_region"
        if not metrics.get("final_front_coverage"):
            return "final_region_reached_front_coverage_below_50"
        return "timeout_unknown"

    def _scroll_main_view(self, event: tk.Event) -> str | None:
        if hasattr(self, "log_text") and event.widget is self.log_text:
            return None
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return None
            units = -1 if delta > 0 else 1
        self.main_canvas.yview_scroll(units, "units")
        return "break"

    def refresh_config_presets(self) -> None:
        config_paths = sorted(Path(__file__).resolve().parent.glob("*.json"))
        self.config_preset_paths = {path.stem: path for path in config_paths}
        values = list(self.config_preset_paths)
        if hasattr(self, "config_preset_combo"):
            self.config_preset_combo.configure(values=values)
        current = self.vars["config_preset"].get()
        if values and current not in self.config_preset_paths:
            default_name = DEFAULT_POLICY_TYPE.upper()
            self.vars["config_preset"].set(default_name if default_name in self.config_preset_paths else values[0])

    def load_selected_config_preset(self, _event: tk.Event | None = None) -> None:
        if self.eval_running or self.connecting or self.reset_running:
            self.vars["status"].set("Cannot switch config while a connection, policy, or reset task is running.")
            return
        selected = self.vars["config_preset"].get().strip()
        path = self.config_preset_paths.get(selected)
        if path is None:
            self.refresh_config_presets()
            path = self.config_preset_paths.get(selected)
        if path is None:
            self.vars["status"].set("Select a config preset first.")
            return
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._alert("Cannot load config", str(exc))
            return
        self._apply_config_preset(config)
        self.vars["status"].set(f"Loaded config preset: {path.name}")
        self._append_log(f"[CONFIG] Loaded {path}")

    def _apply_config_preset(self, config: dict[str, Any]) -> None:
        string_keys = (
            "policy_type",
            "checkpoint_path",
            "dataset_repo_id",
            "dataset_root",
            "task",
            "episode_time_s",
            "fps",
            "grid_rows",
            "grid_cols",
            "grid_cell",
            "trials_per_grid",
            "robot_mode",
            "robot_type",
            "robot_port",
            "robot_id",
            "left_follower_port",
            "right_follower_port",
            "left_follower_id",
            "right_follower_id",
            "calibration_dir",
            "calibration_match",
            "realsense_serial",
            "opencv_front",
            "opencv_side",
            "front_camera_type",
            "front_camera_choice",
            "execution_mode",
            "camera_read_mode",
            "model_chunk_size",
            "prediction_steps",
            "n_action_steps",
            "num_inference_steps",
            "noise_scheduler_type",
            "fusion_steps",
            "fusion_history_weight",
            "segmentation_model_path",
            "segmentation_device",
            "reset_time_s",
            "extra_args",
        )
        for key in string_keys:
            if key not in config or key not in self.vars:
                continue
            value = config[key]
            self.vars[key].set("" if value is None else str(value))

        bool_targets = {
            "include_side_camera": self.include_side_camera,
            "low_resolution": self.low_resolution,
            "enable_segmentation_preview": self.enable_segmentation_preview,
            "use_amp": self.use_amp,
            "auto_reset_after_policy": self.auto_reset_after_policy,
            "lock_grippers": self.lock_grippers,
            "save_video": self.save_video,
            "enable_realsense": self.enable_realsense,
        }
        for key, var in bool_targets.items():
            if key in config:
                var.set(bool(config[key]))

        self._select_checkpoint_combo_for_path(Path(self.vars["checkpoint_path"].get()).expanduser())
        self._refresh_camera_config_preview()
        self._refresh_grid_cells()
        self._refresh_grid_stats()
        self._reset_segmentation_preview_model()
        if self.connected:
            self._append_log("[CONFIG] Existing robot/camera connection is still open; changed ports or cameras apply after reconnect.")

    def _select_checkpoint_combo_for_path(self, checkpoint_path: Path) -> None:
        try:
            target = checkpoint_path.resolve()
        except OSError:
            target = checkpoint_path
        for option in self.checkpoint_options:
            try:
                option_path = option.path.resolve()
            except OSError:
                option_path = option.path
            if option_path == target:
                self.vars["checkpoint"].set(option.label)
                return
        self.vars["checkpoint"].set(f"{self.vars['policy_type'].get()}: {checkpoint_path}")

    def refresh_checkpoints(self) -> None:
        options: list[CheckpointOption] = []
        for path in sorted(DEFAULT_OUTPUT_ROOT.glob("**/pretrained_model")):
            if not path.is_dir():
                continue
            policy_type = self._infer_policy_type(path)
            try:
                rel = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = path
            options.append(CheckpointOption(f"{policy_type}: {rel}", path, policy_type))

        for state_path in sorted(DEFAULT_OUTPUT_ROOT.glob("**/checkpoint_step_*/training_state.pt")):
            checkpoint_dir = state_path.parent
            try:
                rel = checkpoint_dir.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = checkpoint_dir
            experiment = self._mask_act_experiment_from_checkpoint(checkpoint_dir)
            policy_label = f"mask_act/{experiment}" if experiment else "mask_act"
            options.append(CheckpointOption(f"{policy_label}: {rel}", checkpoint_dir, "mask_act"))

        self.checkpoint_options = options
        values = [option.label for option in options]
        self.checkpoint_combo.configure(values=values)
        if values and not self.vars["checkpoint"].get():
            default_path = DEFAULT_POLICY_PATH.resolve()
            selected_option = next(
                (option for option in options if option.path.resolve() == default_path),
                options[-1],
            )
            self.vars["checkpoint"].set(selected_option.label)
            self.on_checkpoint_selected()
        self._append_log(f"Found {len(options)} checkpoint(s) under {DEFAULT_OUTPUT_ROOT}")

    def _infer_policy_type(self, path: Path) -> str:
        if (path / "training_state.pt").is_file() or (path / "mask_act_run_config.json").is_file():
            return "mask_act"
        config_path = path / "config.json"
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text())
                for key in ("type", "policy_type", "name"):
                    value = data.get(key)
                    if isinstance(value, str) and value:
                        return value
            except (OSError, json.JSONDecodeError):
                pass

        lowered = path.as_posix().lower()
        for policy_type in POLICY_TYPES:
            if policy_type in lowered:
                return policy_type
        return self.vars["policy_type"].get() or "act"

    def on_checkpoint_selected(self, _event: tk.Event | None = None) -> None:
        selected = self.vars["checkpoint"].get()
        for option in self.checkpoint_options:
            if option.label == selected:
                self.vars["checkpoint_path"].set(str(option.path))
                self.vars["policy_type"].set(option.policy_type)
                self._refresh_action_chunk_settings(option.path)
                self._refresh_grid_stats()
                return

    def browse_checkpoint(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=str(DEFAULT_OUTPUT_ROOT),
            title="Select pretrained_model, Mask-ACT run, or checkpoint_step folder",
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        self.vars["checkpoint_path"].set(str(path))
        self.vars["policy_type"].set(self._infer_policy_type(path))
        self.vars["checkpoint"].set(f"{self.vars['policy_type'].get()}: {path}")
        self._refresh_action_chunk_settings(path)
        self._refresh_grid_stats()

    def _refresh_action_chunk_settings(self, checkpoint_path: Path) -> None:
        data: dict[str, Any] | None = None
        config_path = checkpoint_path / "config.json"
        if config_path.is_file():
            try:
                data = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
        else:
            current = checkpoint_path
            if current.is_file():
                current = current.parent
            while current != current.parent:
                run_config = current / "mask_act_run_config.json"
                if run_config.is_file():
                    try:
                        data = json.loads(run_config.read_text())
                    except (OSError, json.JSONDecodeError):
                        data = None
                    break
                current = current.parent

        chunk_size = data.get("chunk_size") if data else None
        horizon = data.get("horizon") if data else None
        n_obs_steps = data.get("n_obs_steps") if data else None
        prediction_steps = data.get("n_action_steps") if data else None
        is_diffusion = isinstance(horizon, int) and isinstance(n_obs_steps, int)
        model_chunk_size = chunk_size if isinstance(chunk_size, int) else horizon
        self.vars["model_chunk_size"].set(
            str(model_chunk_size) if isinstance(model_chunk_size, int) else "N/A"
        )
        if is_diffusion:
            usable_steps = horizon - n_obs_steps + 1
            default_prediction_steps = min(DEFAULT_DIFFUSION_PREDICTION_STEPS, usable_steps)
            replan_steps = min(DEFAULT_DIFFUSION_REPLAN_STEPS, default_prediction_steps)
            fusion_steps = min(
                DEFAULT_DIFFUSION_FUSION_STEPS,
                default_prediction_steps - replan_steps,
            )
            self.vars["prediction_steps"].set(str(default_prediction_steps))
            self.vars["n_action_steps"].set(str(replan_steps))
            self.vars["fusion_steps"].set(str(fusion_steps))
            self.vars["fusion_history_weight"].set(
                str(DEFAULT_DIFFUSION_FUSION_HISTORY_WEIGHT if fusion_steps else 0.0)
            )
        elif isinstance(prediction_steps, int):
            self.vars["prediction_steps"].set(str(prediction_steps))
            self.vars["n_action_steps"].set(str(prediction_steps))
            self.vars["fusion_steps"].set("0")
            self.vars["fusion_history_weight"].set("0")
        elif data and self.vars["policy_type"].get().strip() not in {"act", "mask_act", "diffusion"}:
            self.vars["prediction_steps"].set("1")
            self.vars["n_action_steps"].set("1")
        else:
            self.vars["prediction_steps"].set("")
            self.vars["n_action_steps"].set("")

        num_inference_steps = data.get("num_inference_steps") if data else None
        effective_inference_steps = num_inference_steps
        if is_diffusion and not isinstance(effective_inference_steps, int):
            effective_inference_steps = DEFAULT_DIFFUSION_INFERENCE_STEPS
        self.vars["num_inference_steps"].set(
            str(effective_inference_steps) if isinstance(effective_inference_steps, int) else ""
        )
        scheduler = data.get("noise_scheduler_type") if data else None
        self.vars["noise_scheduler_type"].set(
            str(scheduler) if scheduler in {"DDPM", "DDIM"} else "checkpoint"
        )
        self.use_amp.set(bool(data.get("use_amp", False)) if data else False)
        if is_diffusion:
            self.use_amp.set(True)
            self.vars["execution_mode"].set(DEFAULT_DIFFUSION_EXECUTION_MODE)
            self.vars["camera_read_mode"].set(DEFAULT_CAMERA_READ_MODE)

        if isinstance(horizon, int) and isinstance(n_obs_steps, int) and isinstance(prediction_steps, int):
            usable_steps = horizon - n_obs_steps + 1
            self._append_log(
                f"[Diffusion] horizon={horizon}, usable actions={usable_steps}, "
                f"checkpoint prediction/replan steps={prediction_steps}, "
                f"default prediction/replan={self.vars['prediction_steps'].get()}/"
                f"{self.vars['n_action_steps'].get()}, "
                f"inference_steps={num_inference_steps}, effective_inference_steps={effective_inference_steps}, "
                f"scheduler={scheduler}, fusion={self.vars['fusion_steps'].get()}@"
                f"{self.vars['fusion_history_weight'].get()}, "
                f"default_execution={DEFAULT_DIFFUSION_EXECUTION_MODE}."
            )
        elif isinstance(chunk_size, int) and isinstance(prediction_steps, int):
            self._append_log(
                f"[Policy] Model predicts {chunk_size} actions per chunk; "
                f"checkpoint execute steps={prediction_steps}."
            )

    def browse_dataset_root(self) -> None:
        current = Path(self.vars["dataset_root"].get() or DEFAULT_EVAL_ROOT).expanduser()
        initialdir = current if current.exists() and current.is_dir() else current.parent
        selected = self._choose_directory("Select result save parent", initialdir)
        if selected:
            self.vars["dataset_root"].set(str(selected))

    def _choose_directory(self, title: str, initialdir: Path) -> Path | None:
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(True, True)
        dialog.geometry("760x520")
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(2, weight=1)

        selected = {"path": None}
        start = initialdir.expanduser()
        try:
            start = start.resolve()
        except OSError:
            start = DEFAULT_EVAL_ROOT
        path_var = tk.StringVar(value=str(start))
        status_var = tk.StringVar(value="")
        directory_entries: list[Path] = []

        ttk.Label(dialog, text="Folder").grid(row=0, column=0, sticky=tk.W, padx=14, pady=(14, 4))
        path_frame = ttk.Frame(dialog)
        path_frame.grid(row=1, column=0, sticky=tk.EW, padx=14, pady=(0, 8))
        path_frame.columnconfigure(0, weight=1)
        path_entry = ttk.Entry(path_frame, textvariable=path_var)
        path_entry.grid(row=0, column=0, sticky=tk.EW)

        list_frame = ttk.Frame(dialog)
        list_frame.grid(row=2, column=0, sticky=tk.NSEW, padx=14, pady=(0, 8))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        listbox = tk.Listbox(list_frame, activestyle="dotbox")
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        ttk.Label(dialog, textvariable=status_var).grid(row=3, column=0, sticky=tk.W, padx=14, pady=(0, 8))

        buttons = ttk.Frame(dialog)
        buttons.grid(row=4, column=0, sticky=tk.E, padx=14, pady=(0, 14))

        def current_path() -> Path:
            return Path(path_var.get()).expanduser()

        def refresh() -> None:
            directory_entries.clear()
            listbox.delete(0, tk.END)
            path = current_path()
            try:
                resolved = path.resolve()
            except OSError:
                status_var.set("Path cannot be resolved.")
                return
            if not resolved.exists():
                status_var.set("Path does not exist.")
                return
            if not resolved.is_dir():
                status_var.set("Path is not a folder.")
                return
            path_var.set(str(resolved))
            status_var.set("Double-click a folder to enter it, or choose Use This Folder.")
            parent = resolved.parent
            if parent != resolved:
                directory_entries.append(parent)
                listbox.insert(tk.END, "..")
            try:
                children = sorted(
                    (child for child in resolved.iterdir() if child.is_dir()),
                    key=lambda item: item.name.lower(),
                )
            except OSError as exc:
                status_var.set(f"Cannot list folder: {exc}")
                return
            for child in children:
                directory_entries.append(child)
                listbox.insert(tk.END, child.name)

        def enter_selected(_event: tk.Event | None = None) -> None:
            selection = listbox.curselection()
            if not selection:
                return
            index = int(selection[0])
            if 0 <= index < len(directory_entries):
                path_var.set(str(directory_entries[index]))
                refresh()

        def use_current() -> None:
            path = current_path()
            if not path.exists() or not path.is_dir():
                status_var.set("Choose an existing folder.")
                return
            try:
                selected["path"] = path.resolve()
            except OSError:
                selected["path"] = path
            dialog.destroy()

        def cancel() -> None:
            selected["path"] = None
            dialog.destroy()

        ttk.Button(buttons, text="Refresh", command=refresh).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Open Selected", command=enter_selected).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Use This Folder", command=use_current).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=(8, 0))

        listbox.bind("<Double-Button-1>", enter_selected)
        path_entry.bind("<Return>", lambda _event: refresh())
        dialog.bind("<Escape>", lambda _event: cancel())
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        refresh()
        path_entry.focus_set()
        self.root.wait_window(dialog)
        return selected["path"]

    def _refresh_grid_cells(self, *_args: Any) -> None:
        try:
            grid_rows = int(self.vars["grid_rows"].get())
            grid_cols = int(self.vars["grid_cols"].get())
        except ValueError:
            grid_rows = 0
            grid_cols = 0
        cells = (
            [f"({x},{y})" for y in range(grid_rows) for x in range(grid_cols)]
            if grid_rows > 0 and grid_cols > 0
            else []
        )
        if hasattr(self, "grid_cell_combo"):
            self.grid_cell_combo.configure(values=cells)
        current = self.vars["grid_cell"].get()
        if cells and current not in cells:
            self.vars["grid_cell"].set(cells[0])
        self._refresh_grid_stats()

    def _grid_metadata(self) -> dict[str, Any]:
        grid_rows = int(self.vars["grid_rows"].get())
        grid_cols = int(self.vars["grid_cols"].get())
        cell = self.vars["grid_cell"].get().strip()
        if not (cell.startswith("(") and cell.endswith(")") and "," in cell):
            raise ValueError("Target cell must use the (x,y) format.")
        x_text, y_text = cell[1:-1].split(",", 1)
        grid_x, grid_y = int(x_text), int(y_text)
        if grid_rows < 1 or grid_cols < 1:
            raise ValueError("Initial region grid rows and columns must be greater than zero.")
        if not (0 <= grid_x < grid_cols and 0 <= grid_y < grid_rows):
            raise ValueError(f"Grid cell {cell} is outside the {grid_rows}x{grid_cols} grid.")
        return {
            "grid_size": grid_rows if grid_rows == grid_cols else None,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_cell": f"({grid_x},{grid_y})",
            "grid_origin": "top_left",
            "grid_x_direction": "right",
            "grid_y_direction": "down",
        }

    @staticmethod
    def _record_matches_grid(record: dict[str, Any], grid: dict[str, Any]) -> bool:
        if record.get("grid_cell") != grid["grid_cell"]:
            return False
        if "grid_rows" in record or "grid_cols" in record:
            return record.get("grid_rows") == grid["grid_rows"] and record.get("grid_cols") == grid["grid_cols"]
        return grid.get("grid_rows") == grid.get("grid_cols") and record.get("grid_size") == grid.get("grid_size")

    def _base_save_root(self) -> Path:
        return Path(self.vars["dataset_root"].get()).expanduser()

    @staticmethod
    def _mask_act_experiment_from_checkpoint(checkpoint_path: str | Path) -> str | None:
        current = Path(checkpoint_path).expanduser()
        if current.is_file():
            current = current.parent
        while current != current.parent:
            run_config = current / "mask_act_run_config.json"
            if run_config.is_file():
                try:
                    experiment = json.loads(run_config.read_text()).get("experiment")
                except (OSError, json.JSONDecodeError):
                    return None
                if isinstance(experiment, str) and experiment.strip():
                    return experiment.strip().upper()
                return None
            current = current.parent
        return None

    def _selected_policy_variant(self, policy_type: str | None = None) -> str | None:
        selected_policy = (policy_type or self.vars["policy_type"].get()).strip()
        if selected_policy != "mask_act":
            return None
        checkpoint = self.vars["checkpoint_path"].get().strip()
        return self._mask_act_experiment_from_checkpoint(checkpoint) if checkpoint else None

    def _policy_save_parent(
        self,
        policy_type: str | None = None,
        *,
        policy_variant: str | None = None,
        create: bool = True,
    ) -> Path:
        selected_policy = (policy_type or self.vars["policy_type"].get()).strip()
        if selected_policy not in POLICY_TYPES:
            raise ValueError(f"Unsupported policy type: {selected_policy}")
        path = self._base_save_root() / selected_policy
        selected_variant = policy_variant
        if selected_variant is None:
            selected_variant = self._selected_policy_variant(selected_policy)
        if selected_policy == "mask_act":
            if not selected_variant:
                raise ValueError(
                    "Could not determine the Mask-ACT experiment from mask_act_run_config.json."
                )
            path /= selected_variant
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except OSError:
            return []
        return records

    def _result_records(
        self,
        save_root: Path,
        policy_type: str,
        policy_variant: str | None = None,
    ) -> list[dict[str, Any]]:
        if policy_type == "mask_act" and policy_variant:
            paths = (save_root / policy_type / policy_variant / RESULTS_FILENAME,)
        else:
            paths = (
                save_root / policy_type / RESULTS_FILENAME,
                save_root / RESULTS_FILENAME,
            )
        records: list[dict[str, Any]] = []
        for path in paths:
            records.extend(
                record
                for record in self._read_jsonl_records(path)
                if record.get("policy_type") == policy_type
                and (
                    policy_type != "mask_act"
                    or not policy_variant
                    or record.get("policy_variant") == policy_variant
                    or record.get("mask_act_experiment") == policy_variant
                )
            )
        return records

    @staticmethod
    def _parse_result_tags(text: str) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for raw_tag in text.replace(";", ",").split(","):
            tag = " ".join(raw_tag.strip().split())
            key = tag.lower()
            if tag and key not in seen:
                seen.add(key)
                tags.append(tag)
        return tags

    @staticmethod
    def _tags_from_record(record: dict[str, Any]) -> list[str]:
        tags = record.get("tags")
        if isinstance(tags, list):
            return [str(tag).strip() for tag in tags if str(tag).strip()]
        tags_text = record.get("tags_text")
        if isinstance(tags_text, str):
            return EvalPolicyApp._parse_result_tags(tags_text)
        return []

    def _result_tag_candidates(self) -> list[str]:
        try:
            result_paths = sorted(self._base_save_root().rglob(RESULTS_FILENAME))
        except OSError:
            result_paths = []
        candidates: dict[str, str] = {}
        for result_path in result_paths:
            for record in self._read_jsonl_records(result_path):
                for tag in self._tags_from_record(record):
                    candidates.setdefault(tag.lower(), tag)
        return sorted(candidates.values(), key=str.lower)

    def _current_result_path(self) -> Path:
        return self._policy_save_parent(create=False) / RESULTS_FILENAME

    def _delete_latest_current_grid_result(self) -> Path:
        result_path = self._current_result_path()
        if not result_path.is_file():
            raise ValueError("No saved result exists for the selected policy and grid position.")

        grid = self._grid_metadata()
        policy_type = self.vars["policy_type"].get().strip()
        policy_variant = self._selected_policy_variant(policy_type)
        task = self.vars["task"].get().strip()
        original_text = result_path.read_text(encoding="utf-8")
        lines = original_text.splitlines(keepends=True)
        delete_index: int | None = None
        selected_record: dict[str, Any] | None = None

        for index in range(len(lines) - 1, -1, -1):
            try:
                record = json.loads(lines[index])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(record, dict)
                and record.get("policy_type") == policy_type
                and record.get("task") == task
                and self._record_matches_grid(record, grid)
                and (
                    policy_type != "mask_act"
                    or record.get("policy_variant") == policy_variant
                    or record.get("mask_act_experiment") == policy_variant
                )
            ):
                delete_index = index
                selected_record = record
                break

        if delete_index is None or selected_record is None:
            raise ValueError("No saved result exists for the selected policy and grid position.")

        dataset_root = selected_record.get("dataset_root")
        if not dataset_root:
            raise ValueError("The latest saved result does not contain a recorded-data path.")
        run_root = Path(str(dataset_root)).expanduser().resolve()
        save_parent = result_path.parent.resolve()
        if run_root.parent != save_parent:
            raise ValueError(f"Refusing to delete data outside the selected policy directory: {run_root}")
        if run_root.exists():
            if not run_root.is_dir():
                raise ValueError(f"Recorded-data path is not a directory: {run_root}")
            if not (
                (run_root / "evaluation_result.json").is_file()
                or (run_root / "evaluation_metadata.json").is_file()
                or (run_root / "meta" / "info.json").is_file()
            ):
                raise ValueError(f"Directory does not look like an evaluation run: {run_root}")

        updated_text = "".join(lines[:delete_index] + lines[delete_index + 1 :])
        temp_path = result_path.with_name(f".{result_path.name}.tmp")
        temp_path.write_text(updated_text, encoding="utf-8")
        temp_path.replace(result_path)
        try:
            if run_root.exists():
                shutil.rmtree(run_root)
        except OSError:
            result_path.write_text(original_text, encoding="utf-8")
            raise

        self._append_log(
            f"[GUI] Deleted latest saved trial for {policy_type}"
            f"{f'/{policy_variant}' if policy_variant else ''} {grid['grid_cell']}: {run_root}"
        )
        self._refresh_grid_stats()
        return run_root

    def _delete_oldest_current_grid_results(self, count: int) -> list[Path]:
        result_path = self._current_result_path()
        if not result_path.is_file():
            raise ValueError("No saved results exist for the selected policy and grid position.")
        if count <= 0:
            raise ValueError("Delete count must be greater than zero.")

        grid = self._grid_metadata()
        policy_type = self.vars["policy_type"].get().strip()
        policy_variant = self._selected_policy_variant(policy_type)
        task = self.vars["task"].get().strip()
        original_text = result_path.read_text(encoding="utf-8")
        lines = original_text.splitlines(keepends=True)

        parsed: list[dict[str, Any] | None] = []
        matching_indices: list[int] = []
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            parsed.append(record if isinstance(record, dict) else None)
            if (
                isinstance(record, dict)
                and record.get("policy_type") == policy_type
                and record.get("task") == task
                and self._record_matches_grid(record, grid)
                and (
                    policy_type != "mask_act"
                    or record.get("policy_variant") == policy_variant
                    or record.get("mask_act_experiment") == policy_variant
                )
            ):
                matching_indices.append(index)

        if count > len(matching_indices):
            raise ValueError(
                f"Cannot delete {count} trial(s); only {len(matching_indices)} matching trial(s) exist."
            )

        delete_indices = set(matching_indices[:count])
        delete_roots: list[Path] = []
        save_parent = result_path.parent.resolve()
        for index in matching_indices[:count]:
            record = parsed[index]
            dataset_root = record.get("dataset_root") if record else None
            if not dataset_root:
                raise ValueError(f"Trial at result line {index + 1} has no recorded-data path.")
            run_root = Path(str(dataset_root)).expanduser().resolve()
            if run_root.parent != save_parent:
                raise ValueError(f"Refusing to delete data outside the selected policy directory: {run_root}")
            if run_root.exists() and not (
                run_root.is_dir()
                and (
                    (run_root / "evaluation_result.json").is_file()
                    or (run_root / "evaluation_metadata.json").is_file()
                    or (run_root / "meta" / "info.json").is_file()
                )
            ):
                raise ValueError(f"Directory does not look like an evaluation run: {run_root}")
            delete_roots.append(run_root)

        remaining_matching = iter(range(1, len(matching_indices) - count + 1))
        updated_lines: list[str] = []
        per_run_updates: list[tuple[Path, str, str]] = []
        for index, line in enumerate(lines):
            if index in delete_indices:
                continue
            record = parsed[index]
            if index in matching_indices and record is not None:
                new_index = next(remaining_matching)
                updated_record = dict(record)
                updated_record["trial_index"] = new_index
                updated_record["planned_trial_index"] = new_index
                line = json.dumps(updated_record, ensure_ascii=False) + "\n"
                dataset_root = updated_record.get("dataset_root")
                if dataset_root:
                    result_file = Path(str(dataset_root)).expanduser() / "evaluation_result.json"
                    if result_file.is_file():
                        per_run_updates.append(
                            (
                                result_file,
                                result_file.read_text(encoding="utf-8"),
                                json.dumps(updated_record, ensure_ascii=False, indent=2) + "\n",
                            )
                        )
            updated_lines.append(line)

        staged_roots: list[tuple[Path, Path]] = []
        temp_path = result_path.with_name(f".{result_path.name}.tmp")
        try:
            for sequence, run_root in enumerate(delete_roots, start=1):
                if not run_root.exists():
                    continue
                staged = run_root.with_name(f".delete-{run_root.name}-{os.getpid()}-{sequence}")
                if staged.exists():
                    raise ValueError(f"Temporary deletion path already exists: {staged}")
                run_root.rename(staged)
                staged_roots.append((run_root, staged))

            temp_path.write_text("".join(updated_lines), encoding="utf-8")
            temp_path.replace(result_path)
            for result_file, _original, updated in per_run_updates:
                result_file.write_text(updated, encoding="utf-8")
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            result_path.write_text(original_text, encoding="utf-8")
            for result_file, original, _updated in per_run_updates:
                if result_file.parent.exists():
                    result_file.write_text(original, encoding="utf-8")
            for run_root, staged in reversed(staged_roots):
                if staged.exists() and not run_root.exists():
                    staged.rename(run_root)
            raise

        cleanup_errors: list[str] = []
        for _run_root, staged in staged_roots:
            try:
                shutil.rmtree(staged)
            except OSError as exc:
                cleanup_errors.append(f"{staged}: {exc}")

        self._append_log(
            f"[GUI] Deleted the oldest {count} saved trial(s) for {policy_type}"
            f"{f'/{policy_variant}' if policy_variant else ''} task={task} {grid['grid_cell']}; "
            f"renumbered {len(matching_indices) - count} remaining trial(s)."
        )
        if cleanup_errors:
            self._append_log(
                "[GUI] Warning: result records were updated, but some quarantined data directories "
                "could not be removed:\n" + "\n".join(cleanup_errors)
            )
        self._refresh_grid_stats()
        return delete_roots

    def _grid_stats_from_records(
        self,
        records: list[dict[str, Any]],
        *,
        task: str,
        grid: dict[str, Any],
        grid_cell: str,
    ) -> tuple[int, int]:
        matching = [
            record
            for record in records
            if record.get("task") == task
            and self._record_matches_grid(record, {**grid, "grid_cell": grid_cell})
        ]
        successes = sum(record.get("result") == "success" for record in matching)
        return len(matching), successes

    def _current_grid_stats(self) -> tuple[int, int]:
        grid = self._grid_metadata()
        policy_type = self.vars["policy_type"].get().strip()
        policy_variant = self._selected_policy_variant(policy_type)
        records = self._result_records(self._base_save_root(), policy_type, policy_variant)
        return self._grid_stats_from_records(
            records,
            task=self.vars["task"].get().strip(),
            grid=grid,
            grid_cell=grid["grid_cell"],
        )

    def _refresh_grid_stats(self, *_args: Any) -> None:
        try:
            grid = self._grid_metadata()
            target = int(self.vars["trials_per_grid"].get())
            if target < 1:
                raise ValueError
            policy_type = self.vars["policy_type"].get().strip()
            policy_variant = self._selected_policy_variant(policy_type)
            records = self._result_records(self._base_save_root(), policy_type, policy_variant)
            task = self.vars["task"].get().strip()
            total, successes = self._grid_stats_from_records(
                records,
                task=task,
                grid=grid,
                grid_cell=grid["grid_cell"],
            )
        except (OSError, ValueError):
            self.vars["grid_stats"].set("Enter a valid grid and trials-per-cell value.")
            return
        rate = 0.0 if total == 0 else successes / total * 100
        policy_label = f"{policy_type}/{policy_variant}" if policy_variant else policy_type
        cell = grid["grid_cell"]
        other_task_counts: dict[str, int] = {}
        if total == 0:
            for record in records:
                record_task = str(record.get("task") or "")
                if record_task and record_task != task and self._record_matches_grid(record, grid):
                    other_task_counts[record_task] = other_task_counts.get(record_task, 0) + 1
        hint = ""
        if other_task_counts:
            hint = "; other tasks here: " + ", ".join(
                f"{name}={count}" for name, count in sorted(other_task_counts.items())
            )
        self.vars["grid_stats"].set(
            f"{policy_label} task={task} {cell}: {total}/{target} trials, "
            f"{successes} success ({rate:.1f}%){hint}"
        )

    def select_next_open_grid_cell(self) -> None:
        try:
            grid = self._grid_metadata()
            target = int(self.vars["trials_per_grid"].get())
            if target < 1:
                raise ValueError("Trials per cell must be greater than zero.")
            policy_type = self.vars["policy_type"].get().strip()
            policy_variant = self._selected_policy_variant(policy_type)
            records = self._result_records(self._base_save_root(), policy_type, policy_variant)
            task = self.vars["task"].get().strip()
        except (OSError, ValueError) as exc:
            self.vars["status"].set(f"Cannot select next grid cell: {exc}")
            return

        for y in range(int(grid["grid_rows"])):
            for x in range(int(grid["grid_cols"])):
                cell = f"({x},{y})"
                total, _successes = self._grid_stats_from_records(
                    records,
                    task=task,
                    grid=grid,
                    grid_cell=cell,
                )
                if total < target:
                    self.vars["grid_cell"].set(cell)
                    self.vars["status"].set(f"Selected next open grid cell {cell}: {total}/{target} saved.")
                    return
        self.vars["status"].set(f"All {grid['grid_rows']}x{grid['grid_cols']} grid cells have at least {target} saved trials.")

    def _on_result_changed(self, *_args: Any) -> None:
        if self.vars["result"].get() != "success":
            return
        self.vars["object_exposed"].set("yes")
        self.vars["object_pushed_out"].set("yes")
        self.vars["object_push_success"].set("yes")

    def open_save_folder(self) -> None:
        path = self._policy_save_parent()
        self._alert("Save folder", str(path), button_text="Close")

    def find_cameras(self) -> None:
        lines: list[str] = []
        options: dict[str, tuple[str, str]] = {}
        try:
            from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

            cameras = OpenCVCamera.find_cameras()
            for camera in cameras:
                camera_id = str(camera.get("id"))
                profile = camera.get("default_stream_profile", {})
                label = (
                    f"OpenCV | {camera_id} | "
                    f"{profile.get('width', '?')}x{profile.get('height', '?')} @ {profile.get('fps', '?')} fps"
                )
                options[label] = ("opencv", camera_id)
            lines.append(f"OpenCV: {len(cameras)} found")
        except Exception as exc:
            lines.append(f"OpenCV scan failed: {exc}")

        try:
            from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

            cameras = RealSenseCamera.find_cameras()
            serials = [str(camera.get("id")) for camera in cameras]
            for camera, serial in zip(cameras, serials, strict=False):
                label = f"RealSense | {camera.get('name', 'camera')} | {serial}"
                options[label] = ("realsense", serial)
            lines.append(f"RealSense: {len(serials)} found")
        except Exception as exc:
            lines.append(f"RealSense scan failed: {exc}")

        self.camera_options = options
        labels = list(options)
        self.front_camera_combo.configure(values=labels)
        selected = self.vars["front_camera_choice"].get()
        if selected not in options and labels:
            preferred_type = self.vars["front_camera_type"].get()
            selected = next((label for label in labels if options[label][0] == preferred_type), labels[0])
            self.vars["front_camera_choice"].set(selected)
            self._on_front_camera_selected()
        self._append_log("[GUI] " + " | ".join(lines))
        self._refresh_camera_config_preview()

    def _on_front_camera_type_changed(self, *_args: Any) -> None:
        camera_type = self.vars["front_camera_type"].get()
        self.enable_realsense.set(camera_type == "realsense")
        self._refresh_camera_config_preview()

    def _on_front_camera_selected(self, _event: tk.Event | None = None) -> None:
        selected = self.vars["front_camera_choice"].get()
        option = self.camera_options.get(selected)
        if option is None:
            return
        camera_type, camera_id = option
        self.vars["front_camera_type"].set(camera_type)
        if camera_type == "opencv":
            self.vars["opencv_front"].set(camera_id)
        else:
            self.vars["realsense_serial"].set(camera_id)
        self._append_log(f"[GUI] Selected front camera: {selected}")

    def preview_front_camera(self) -> None:
        if self.connected or self.connecting:
            self._alert(
                "Camera is in use",
                "The camera is already in use by the robot. Disconnect the robot before opening a standalone preview.",
                button_text="Close",
            )
            return
        if self.camera_preview_dialog is not None and self.camera_preview_dialog.winfo_exists():
            self.camera_preview_dialog.lift()
            return

        camera_type = self.vars["front_camera_type"].get()
        if camera_type == "realsense":
            camera_id = self.vars["realsense_serial"].get().strip()
            if not camera_id:
                self._alert("Cannot preview", "Scan and select a RealSense camera first.")
                return
        else:
            camera_id = self.vars["opencv_front"].get().strip()
            if not camera_id:
                self._alert("Cannot preview", "Scan and select an OpenCV camera first.")
                return

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Front Camera Preview - {camera_type}: {camera_id}")
        dialog.geometry("900x700")
        dialog.minsize(640, 480)
        self.camera_preview_dialog = dialog
        stop_event = threading.Event()
        self.camera_preview_stop = stop_event
        frame_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        image_label = ttk.Label(dialog, anchor=tk.CENTER)
        image_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        status = tk.StringVar(value=f"Opening {camera_type}: {camera_id} ...")
        ttk.Label(dialog, textvariable=status, anchor=tk.W).pack(fill=tk.X, padx=12, pady=(0, 12))
        image_ref: dict[str, Any] = {"value": None}

        def put_latest(kind: str, value: Any) -> None:
            try:
                frame_queue.put_nowait((kind, value))
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait((kind, value))

        def capture_loop() -> None:
            capture = None
            camera = None
            try:
                camera_width, camera_height = self._camera_size()
                if camera_type == "opencv":
                    import cv2

                    target = int(camera_id) if camera_id.isdigit() else camera_id
                    capture = cv2.VideoCapture(target)
                    if not capture.isOpened():
                        raise RuntimeError(f"Could not open OpenCV camera {camera_id}")
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
                    capture.set(cv2.CAP_PROP_FPS, DEFAULT_CAMERA_FPS)
                    while not stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"Failed to read OpenCV camera {camera_id}")
                        if frame.shape[1] != camera_width or frame.shape[0] != camera_height:
                            frame = cv2.resize(frame, (camera_width, camera_height), interpolation=cv2.INTER_AREA)
                        put_latest("frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    camera = PersistentFlexibleRealSenseCamera(
                        camera_id,
                        camera_width,
                        camera_height,
                    )
                    camera.connect()
                    while not stop_event.is_set():
                        put_latest("frame", camera.async_read())
            except Exception as exc:
                put_latest("error", str(exc))
            finally:
                if capture is not None:
                    capture.release()
                if camera is not None:
                    camera.disconnect()

        def close_preview() -> None:
            stop_event.set()
            if dialog.winfo_exists():
                dialog.destroy()
            self.camera_preview_dialog = None
            self.camera_preview_stop = None

        def update_preview() -> None:
            if stop_event.is_set() or not dialog.winfo_exists():
                return
            latest: tuple[str, Any] | None = None
            while True:
                try:
                    latest = frame_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                kind, value = latest
                if kind == "error":
                    status.set(f"Preview error: {value}")
                else:
                    from PIL import Image, ImageTk
                    import numpy as np

                    frame = np.asarray(value)
                    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
                    image.thumbnail((860, 620))
                    image_ref["value"] = ImageTk.PhotoImage(image)
                    image_label.configure(image=image_ref["value"])
                    status.set(f"Previewing {camera_type}: {camera_id} | {frame.shape[1]}x{frame.shape[0]}")
            dialog.after(30, update_preview)

        dialog.protocol("WM_DELETE_WINDOW", close_preview)
        threading.Thread(target=capture_loop, daemon=True).start()
        dialog.after(30, update_preview)

    def _refresh_camera_config_preview(self) -> None:
        try:
            self.vars["camera_config"].set(self._build_camera_config())
        except ValueError as exc:
            self.vars["camera_config"].set(f"Invalid camera config: {exc}")

    def _resolve_realsense_serial(self) -> str:
        requested = self.vars["realsense_serial"].get().strip()
        if requested:
            return requested
        try:
            from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

            cameras = RealSenseCamera.find_cameras()
        except Exception as exc:
            raise ValueError(
                "RealSense auto-detect failed. Please click Find Cameras or fill Front RealSense SN manually. "
                f"Original error: {exc}"
            ) from exc
        if not cameras:
            raise ValueError(
                "Enable RealSense front is checked, but no RealSense was detected. "
                "Fill OpenCV front and uncheck RealSense if you want to use an OpenCV camera."
            )
        serial = str(cameras[0]["id"])
        self.vars["realsense_serial"].set(serial)
        return serial

    def _camera_size(self) -> tuple[int, int]:
        if self.low_resolution.get():
            return LOW_CAMERA_WIDTH, LOW_CAMERA_HEIGHT
        return CAMERA_WIDTH, CAMERA_HEIGHT

    def _build_camera_config(self) -> str:
        cameras: list[str] = []
        camera_width, camera_height = self._camera_size()
        front_path = self.vars["opencv_front"].get().strip()
        serial = self.vars["realsense_serial"].get().strip() or "<auto>"
        front_is_realsense_video = bool(
            self.enable_realsense.get()
            and front_path
            and self._is_realsense_video_path(front_path)
        )
        if self.enable_realsense.get() and (not front_path or front_is_realsense_video):
            if serial != "<auto>":
                cameras.append(
                    "front: {"
                    f'type: intelrealsense, serial_number_or_name: "{serial}", '
                    f"width: {camera_width}, height: {camera_height}, "
                    f"fps: {DEFAULT_CAMERA_FPS}, use_depth: true"
                    "}"
                )
            else:
                cameras.append("front: {type: intelrealsense, serial_number_or_name: <auto>}")
        elif front_path:
            cameras.append(self._opencv_camera_entry("front", front_path))

        if self.include_side_camera.get() and self.vars["opencv_side"].get().strip():
            cameras.append(self._opencv_camera_entry("side", self.vars["opencv_side"].get().strip()))

        if not cameras:
            raise ValueError("at least one front camera is required")
        return "{ " + ", ".join(cameras) + " }"

    def _opencv_camera_entry(self, name: str, index_or_path: str) -> str:
        value = index_or_path if index_or_path.isdigit() else f'"{index_or_path}"'
        width, height = self._camera_size()
        return (
            f"{name}: {{"
            f"type: opencv, index_or_path: {value}, "
            f"width: {width}, height: {height}, fps: {DEFAULT_CAMERA_FPS}, fourcc: \"MJPG\""
            "}"
        )

    def _camera_config_for_command(self) -> str:
        front_path = self.vars["opencv_front"].get().strip()
        camera_width, camera_height = self._camera_size()
        if self.enable_realsense.get() and (not front_path or self._is_realsense_video_path(front_path)):
            serial = self._resolve_realsense_serial()
            entries = [
                "front: {"
                f'type: intelrealsense, serial_number_or_name: "{serial}", '
                f"width: {camera_width}, height: {camera_height}, "
                f"fps: {DEFAULT_CAMERA_FPS}, use_depth: true"
                "}"
            ]
        else:
            if not front_path:
                raise ValueError("RealSense disabled: OpenCV front cannot be empty.")
            entries = [self._opencv_camera_entry("front", front_path)]

        if self.include_side_camera.get() and self.vars["opencv_side"].get().strip():
            entries.append(self._opencv_camera_entry("side", self.vars["opencv_side"].get().strip()))

        return "{ " + ", ".join(entries) + " }"

    def check_connect(self) -> None:
        if self.camera_preview_dialog is not None and self.camera_preview_dialog.winfo_exists():
            self.vars["status"].set("Close the camera preview before connecting the robot.")
            self._alert(
                "Close preview",
                "Close the camera preview and wait for the device to be released before clicking Check Connect.",
                button_text="Close",
            )
            return
        if self.connected:
            self.vars["status"].set("Already connected. Start can reuse this connection.")
            return
        if self.connecting:
            self.vars["status"].set("Connection check is already running.")
            return

        self.log_text.delete("1.0", tk.END)
        self._append_log("[GUI] Connecting robot and cameras. This connection will stay open.")
        self.connecting = True
        self.connect_button.configure(state=tk.DISABLED)
        self.start_button.configure(state=tk.DISABLED)
        self.vars["status"].set("Checking robot connection...")
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _check_trial_limit_before_start(self) -> bool:
        while True:
            total, _successes = self._current_grid_stats()
            target = int(self.vars["trials_per_grid"].get())
            if total < target:
                return True

            policy_type = self.vars["policy_type"].get().strip()
            policy_variant = self._selected_policy_variant(policy_type)
            policy_label = f"{policy_type}/{policy_variant}" if policy_variant else policy_type
            grid_cell = self.vars["grid_cell"].get().strip()
            choice = self._choose(
                "Trial limit reached",
                (
                    f"{policy_label} at grid position {grid_cell} already has {total} saved trial(s), "
                    f"which reaches or exceeds the configured limit of {target}.\n\n"
                    "Increase the limit to allow one more trial, or delete a manually selected number "
                    "of the oldest matching trials and their trajectory/images/video data."
                ),
                (
                    ("increase", "Increase Limit"),
                    ("delete", "Delete Oldest N"),
                ),
            )
            if choice is None:
                self.vars["status"].set("Start cancelled because the trial limit was reached.")
                return False
            if choice == "increase":
                new_target = total + 1
                self.vars["trials_per_grid"].set(str(new_target))
                self._refresh_grid_stats()
                self._append_log(
                    f"[GUI] Increased trial limit for {policy_label} {grid_cell}: {target} -> {new_target}"
                )
                return True
            minimum_delete = max(total - target + 1, 1)
            delete_count = self._ask_integer(
                "Delete oldest trials",
                (
                    f"{policy_label}, task {self.vars['task'].get().strip()}, grid {grid_cell} has "
                    f"{total} saved trials with a limit of {target}.\n\n"
                    f"Enter how many oldest trials to delete. To get below the limit, delete at least "
                    f"{minimum_delete}. The allowed range is {minimum_delete}–{total}."
                ),
                minimum=minimum_delete,
                maximum=total,
                initial=minimum_delete,
            )
            if delete_count is None:
                continue
            remaining = total - delete_count
            if remaining >= target:
                self._alert(
                    "Delete count is too small",
                    f"Deleting {delete_count} would leave {remaining} trials, which is not below limit {target}.",
                )
                continue
            if not self._confirm(
                "Delete oldest saved trials?",
                (
                    f"Permanently delete the oldest {delete_count} saved {policy_label} trial(s) for "
                    f"task {self.vars['task'].get().strip()} at {grid_cell}?\n\n"
                    "Their result entries, trajectories, images, and videos will be deleted. "
                    f"The remaining {remaining} trial(s) will keep their relative order and be renumbered."
                ),
                confirm_text=f"Delete Oldest {delete_count}",
                cancel_text="Back",
                destructive=True,
            ):
                continue
            try:
                deleted_paths = self._delete_oldest_current_grid_results(delete_count)
            except (OSError, ValueError) as exc:
                self._alert("Cannot delete oldest trials", str(exc))
                return False
            self.vars["status"].set(
                f"Deleted {len(deleted_paths)} oldest trial(s); {remaining}/{target} remain."
            )

    def start_eval(self) -> None:
        if self.eval_running:
            self.vars["status"].set("A policy run is already active.")
            return
        if self.teleop_running or self.reset_running:
            self.vars["status"].set("Stop teleop/reset before starting policy.")
            return
        if not self.connected:
            self._alert("Not connected", "Click Check Connect first; it will keep the robot connected.")
            return
        if self.last_run and not self.result_saved:
            discard = self._confirm(
                "Unsaved result",
                "The previous result is not saved. Discard it and permanently delete its recorded data before starting a new run?",
                confirm_text="Discard + Delete",
                destructive=True,
            )
            if not discard:
                self._show_result_dialog()
                return
            self.discard_result(confirm=False)

        try:
            self._validate_eval_settings()
        except ValueError as exc:
            self._alert("Cannot start", str(exc))
            return
        if not self._check_trial_limit_before_start():
            return

        self.started_at = time.monotonic()
        self.started_at_iso = datetime.now().isoformat(timespec="seconds")
        self.stop_requested = False
        self.stop_policy_requested = False
        self.last_run = {}
        self.result_saved = True
        self.segmentation_eval_metrics = self._new_segmentation_eval_metrics()
        save_root = self._base_save_root()
        policy_variant = self._selected_policy_variant()
        policy_save_parent = self._policy_save_parent()
        trial_count, _successes = self._current_grid_stats()
        self.active_eval_metadata = {
            **self._grid_metadata(),
            "policy_variant": policy_variant,
            "mask_act_experiment": policy_variant if self.vars["policy_type"].get() == "mask_act" else None,
            "save_video": self.save_video.get(),
            "prediction_steps": (
                int(self.vars["prediction_steps"].get())
                if self.vars["prediction_steps"].get().strip()
                else None
            ),
            "n_action_steps": (
                int(self.vars["n_action_steps"].get())
                if self.vars["n_action_steps"].get().strip()
                else None
            ),
            "replan_interval_steps": int(self.vars["n_action_steps"].get()),
            "num_inference_steps": (
                int(self.vars["num_inference_steps"].get())
                if self.vars["num_inference_steps"].get().strip()
                else None
            ),
            "noise_scheduler_type": self.vars["noise_scheduler_type"].get(),
            "use_amp": self.use_amp.get(),
            "execution_mode": self.vars["execution_mode"].get(),
            "camera_read_mode": self.vars["camera_read_mode"].get(),
            "locked_gripper_observation_baseline": self.lock_grippers.get(),
            "fusion_steps": int(self.vars["fusion_steps"].get()),
            "fusion_history_weight": float(self.vars["fusion_history_weight"].get()),
            "observation_warmup_frames": (
                DEFAULT_DIFFUSION_OBSERVATION_WARMUP_FRAMES
                if self.vars["policy_type"].get().strip() == "diffusion"
                else 0
            ),
            "fps": int(self.vars["fps"].get()),
            "save_root": str(save_root),
            "save_parent": str(policy_save_parent),
            "target_trials_per_grid": int(self.vars["trials_per_grid"].get()),
            "planned_trial_index": trial_count + 1,
        }
        self.connect_button.configure(state=tk.DISABLED)
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.disconnect_button.configure(state=tk.DISABLED)
        self.teleop_button.configure(state=tk.DISABLED)
        self.stop_teleop_button.configure(state=tk.DISABLED)
        self.capture_reset_button.configure(state=tk.DISABLED)
        self.reset_button.configure(state=tk.DISABLED)
        self.default_pose_button.configure(state=tk.DISABLED)
        self.eval_running = True
        self.vars["status"].set("Policy is running on the existing connection...")
        self._append_log("[GUI] Starting policy on the existing robot/camera connection.")
        threading.Thread(target=self._eval_worker, daemon=True).start()

    def _start_process(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid,
            )
        except OSError as exc:
            self.process = None
            self.process_kind = ""
            self.connect_button.configure(state=tk.NORMAL)
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.vars["status"].set(f"Failed to start: {exc}")
            return

        threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True).start()

    def _connect_worker(self) -> None:
        robot = None
        try:
            robot = self._make_robot()
            self._patch_realsense_like_recorder(robot)
            robot.connect()
            obs = robot.get_observation()
            self.initial_joint_action = self._action_from_observation(robot, obs)
            with self.robot_lock:
                self.robot = robot
                self.connected = True
            self.log_queue.put("Robot/camera connection OK.")
            self.log_queue.put(f"Raw observation keys: {sorted(obs)}")
            self.log_queue.put(f"Captured initial reset pose with {len(self.initial_joint_action)} joint target(s).")
            self.log_queue.put("__CONNECT_DONE__|0|")
        except Exception:
            error = traceback.format_exc()
            self.log_queue.put(error.rstrip())
            if robot is not None:
                self._safe_disconnect_robot(robot)
            self.log_queue.put("__CONNECT_DONE__|1|Connection failed.")

    def _patch_realsense_like_recorder(self, robot: Any) -> None:
        if not self.enable_realsense.get():
            return
        front_path = self.vars["opencv_front"].get().strip()
        if front_path and not self._is_realsense_video_path(front_path):
            return
        serial = self._resolve_realsense_serial()
        camera_width, camera_height = self._camera_size()
        replacement = PersistentFlexibleRealSenseCamera(serial, camera_width, camera_height)
        if hasattr(robot, "left_arm") and "front" in robot.left_arm.cameras:
            robot.left_arm.cameras["front"] = replacement
            robot.cameras["front"] = replacement
            self.log_queue.put("[GUI] Using recorder-compatible FlexibleRealSenseCamera for left_front.")
        elif hasattr(robot, "cameras") and "front" in robot.cameras:
            robot.cameras["front"] = replacement
            self.log_queue.put("[GUI] Using recorder-compatible FlexibleRealSenseCamera for front.")

    def _eval_worker(self) -> None:
        try:
            self._run_eval_on_connected_robot()
            self.log_queue.put("__EVAL_DONE__|0|")
        except Exception:
            self.log_queue.put(traceback.format_exc().rstrip())
            self.log_queue.put("__EVAL_DONE__|1|Policy run failed.")

    def _run_eval_on_connected_robot(self) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.utils import make_robot_action
        from lerobot.processor import make_default_processors
        from lerobot.processor.rename_processor import rename_stats
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, OBS_STR
        from copy import copy

        import torch

        from action_chunk_fusion import ActionChunkFusionPlanner
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.utils.control_utils import predict_action
        from lerobot.utils.robot_utils import precise_sleep
        from lerobot.utils.utils import get_safe_torch_device

        robot = self.robot
        if robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")

        checkpoint = self.vars["checkpoint_path"].get().strip()
        repo_id = self.vars["dataset_repo_id"].get().strip()
        task = self.vars["task"].get().strip()
        fps = int(self.vars["fps"].get())
        episode_time_s = float(self.vars["episode_time_s"].get())
        save_video = bool(self.active_eval_metadata.get("save_video", self.save_video.get()))
        save_parent = self._save_parent()
        run_dataset_root = self._make_run_dataset_root(save_parent, repo_id)
        self.active_dataset_root = run_dataset_root

        from mask_act_inference import is_mask_act_checkpoint, load_mask_act_for_inference

        mask_act_checkpoint = self.vars["policy_type"].get() == "mask_act" or is_mask_act_checkpoint(checkpoint)
        if mask_act_checkpoint:
            policy, preprocessor, postprocessor, details = load_mask_act_for_inference(
                checkpoint,
                PROJECT_ROOT,
            )
            policy_cfg = policy.config
            self.log_queue.put(
                "[Mask-ACT] "
                f"experiment={details['experiment']} rgb={details['rgb_key']} "
                f"metadata={details['metadata_root']}"
            )
        else:
            policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
            policy_cfg.pretrained_path = checkpoint
        self._apply_action_step_override(policy_cfg)
        _teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=robot_action_processor,
                initial_features=create_initial_features(action=robot.action_features),
                use_videos=True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=True,
            ),
        )
        if not save_video:
            dataset_features = self._video_features_to_images(dataset_features)
        policy_inputs = sorted(policy_cfg.input_features)
        policy_action_names = self._policy_feature_names(dataset_features, policy_cfg.output_features, ACTION)
        policy_state_names = self._policy_feature_names(
            dataset_features, policy_cfg.input_features, "observation.state"
        )
        if policy_action_names:
            dataset_features = self._features_with_names(dataset_features, ACTION, policy_action_names)
        if policy_state_names and "observation.state" in dataset_features:
            dataset_features = self._features_with_names(
                dataset_features,
                "observation.state",
                policy_state_names,
            )
        obs_features = sorted(key for key in dataset_features if key.startswith("observation."))
        self.log_queue.put(f"Raw robot observation features: {sorted(robot.observation_features)}")
        self.log_queue.put(f"Dataset observation features: {obs_features}")
        self.log_queue.put(f"Policy input features: {policy_inputs}")
        self.log_queue.put(f"Policy output features: {sorted(policy_cfg.output_features)}")
        raw_action_names = list(robot.action_features)
        if len(policy_action_names) != len(raw_action_names):
            self.log_queue.put(
                f"[GRIPPER] Policy action vector uses {len(policy_action_names)} non-gripper dim(s): "
                f"{policy_action_names}; raw robot action dim is {len(raw_action_names)}. "
                "Gripper will keep its current/initial angle because no gripper command is sent."
            )

        dataset = None
        replan_executor: ThreadPoolExecutor | None = None
        last_action_log_t = 0.0
        try:
            dataset = LeRobotDataset.create(
                repo_id,
                fps,
                root=run_dataset_root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=save_video,
                image_writer_processes=0,
                image_writer_threads=max(1, 4 * len(robot.cameras)),
                batch_encoding_size=1_000_000 if save_video else 1,
                vcodec="h264",
            )
            evaluation_metadata = {
                **self.active_eval_metadata,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "policy_type": self.vars["policy_type"].get(),
                "policy_path": checkpoint,
                "task": task,
                "dataset_repo_id": repo_id,
                "dataset_root": str(run_dataset_root),
            }
            (run_dataset_root / "evaluation_metadata.json").write_text(
                json.dumps(evaluation_metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if not mask_act_checkpoint:
                policy = make_policy(policy_cfg, ds_meta=dataset.meta)
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=policy_cfg,
                    pretrained_path=policy_cfg.pretrained_path,
                    dataset_stats=rename_stats(dataset.meta.stats, {}),
                    preprocessor_overrides={
                        "device_processor": {"device": policy_cfg.device},
                        "rename_observations_processor": {"rename_map": {}},
                    },
                )
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            fusion_steps = int(self.vars["fusion_steps"].get())
            execution_mode = self.vars["execution_mode"].get()
            predict_chunk = getattr(policy, "predict_action_chunk", None)
            is_diffusion_policy = all(
                hasattr(policy.config, attr)
                for attr in ("horizon", "n_obs_steps", "noise_scheduler_type")
            ) and hasattr(policy, "diffusion")
            predicted_steps = (
                int(policy.config.n_action_steps)
                if is_diffusion_policy
                else int(getattr(policy.config, "chunk_size", policy.config.n_action_steps))
            )
            replan_steps = int(self.vars["n_action_steps"].get())
            use_chunk_planner = (
                callable(predict_chunk)
                and (hasattr(policy.config, "chunk_size") or is_diffusion_policy)
                and (
                    execution_mode == "asynchronous"
                    or fusion_steps > 0
                    or replan_steps < predicted_steps
                )
            )
            planner = (
                ActionChunkFusionPlanner(
                    replan_steps=replan_steps,
                    fusion_steps=fusion_steps,
                    initial_history_weight=float(self.vars["fusion_history_weight"].get()),
                )
                if use_chunk_planner
                else None
            )
            if planner is not None and execution_mode == "asynchronous":
                replan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-replan")
            policy_device = get_safe_torch_device(policy.config.device)
            with self.robot_lock:
                self.initial_joint_action = self._action_from_observation(robot, robot.get_observation())
            self.log_queue.put(
                f"Captured policy-start reset pose with {len(self.initial_joint_action)} joint target(s)."
            )
            if self.lock_grippers.get():
                gripper_keys = self._gripper_keys(robot.action_features)
                state_names = dataset.features.get("observation.state", {}).get("names", [])
                if any(self._is_gripper_key(str(name)) for name in state_names):
                    self.log_queue.put(
                        f"[GRIPPER] Locked during policy and reset; commands disabled for {gripper_keys}. "
                        "Policy gripper observations use the captured/validated baseline."
                    )
                else:
                    self.log_queue.put(
                        f"[GRIPPER] None-gripper policy mode; commands disabled for {gripper_keys}. "
                        "The gripper keeps its current/initial angle."
                    )
            self.log_queue.put(
                "[RECORD] MP4 video saving enabled for front/side; encoding is deferred until Save."
                if save_video
                else "[RECORD] MP4 encoding disabled; trajectory and image frames will still be saved."
            )
            self.log_queue.put(
                f"[CONTROL] execution_mode={execution_mode}, "
                f"camera_read_mode={self.vars['camera_read_mode'].get()}, target_fps={fps}."
            )

            encoding_context = nullcontext()
            with encoding_context:
                start_t = time.perf_counter()
                timestamp = 0.0
                control_step = 0
                rate_window_started = start_t
                rate_window_steps = 0
                observation_history: deque[dict[str, Any]] = deque(
                    maxlen=int(policy.config.n_obs_steps) if is_diffusion_policy else 1
                )
                if is_diffusion_policy:
                    warmup_frames = max(
                        int(policy.config.n_obs_steps),
                        DEFAULT_DIFFUSION_OBSERVATION_WARMUP_FRAMES,
                    )
                    self.log_queue.put(
                        f"[Diffusion] Collecting {warmup_frames} observation warm-up frames; "
                        "no robot action will be sent."
                    )
                    for warmup_index in range(warmup_frames):
                        raw_warmup_obs = self._get_eval_observation(robot)
                        processed_warmup_obs = robot_observation_processor(raw_warmup_obs)
                        warmup_observation_frame = build_dataset_frame(
                            dataset.features,
                            processed_warmup_obs,
                            prefix=OBS_STR,
                        )
                        observation_history.append(
                            copy(
                                self._policy_observation_frame(
                                    warmup_observation_frame,
                                    dataset.features,
                                    policy_cfg.input_features,
                                )
                            )
                        )
                        if warmup_index + 1 < warmup_frames:
                            precise_sleep(1 / fps)
                    start_t = time.perf_counter()
                    rate_window_started = start_t
                    self.log_queue.put(
                        f"[Diffusion] Observation history ready with "
                        f"{len(observation_history)} real frame(s)."
                    )
                pending_replan: tuple[
                    Future[tuple[torch.Tensor, float]],
                    int,
                    float,
                    float,
                ] | None = None

                def observation_snapshot() -> list[dict[str, Any]]:
                    observations = [copy(item) for item in observation_history]
                    if not observations:
                        raise RuntimeError("No observation is available for action prediction.")
                    required = int(policy.config.n_obs_steps) if is_diffusion_policy else 1
                    while len(observations) < required:
                        observations.insert(0, copy(observations[0]))
                    return observations

                def predict_chunk_async(observations: list[dict[str, Any]]) -> tuple[torch.Tensor, float]:
                    if not callable(predict_chunk):
                        raise RuntimeError(f"{type(policy).__name__} does not expose predict_action_chunk().")
                    if policy_device.type == "cuda":
                        torch.cuda.synchronize(policy_device)
                    pipeline_started = time.perf_counter()
                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type=policy_device.type)
                        if policy_device.type == "cuda" and policy.config.use_amp
                        else nullcontext(),
                    ):
                        prepared_observations = []
                        for observation in observations:
                            policy_observation = prepare_observation_for_inference(
                                copy(observation),
                                policy_device,
                                task,
                                robot.robot_type,
                            )
                            prepared_observations.append(preprocessor(policy_observation))

                        if is_diffusion_policy:
                            for policy_observation in prepared_observations:
                                if policy.config.image_features:
                                    policy_observation[OBS_IMAGES] = torch.stack(
                                        [
                                            policy_observation[key]
                                            for key in policy.config.image_features
                                        ],
                                        dim=-4,
                                    )
                            temporal_batch = {
                                key: torch.stack(
                                    [policy_observation[key] for policy_observation in prepared_observations],
                                    dim=1,
                                )
                                for key in (OBS_STATE, OBS_IMAGES, OBS_ENV_STATE)
                                if key in prepared_observations[0]
                            }
                            predicted_chunk = policy.diffusion.generate_actions(temporal_batch)
                        else:
                            predicted_chunk = predict_chunk(prepared_observations[-1])
                        predicted_chunk = postprocessor(predicted_chunk)
                    if policy_device.type == "cuda":
                        torch.cuda.synchronize(policy_device)
                    return predicted_chunk, time.perf_counter() - pipeline_started

                def install_pending_replan(wait: bool = False) -> bool:
                    nonlocal pending_replan
                    if pending_replan is None:
                        return False
                    future, snapshot_step, observation_started, observation_ready = pending_replan
                    if not wait and not future.done():
                        return False
                    new_chunk, async_pipeline_s = future.result()
                    result_ready = time.perf_counter()
                    elapsed_steps = max(control_step - snapshot_step, 0)
                    chunk_steps = int(new_chunk.shape[-2])
                    if elapsed_steps >= chunk_steps:
                        raise RuntimeError(
                            "Asynchronous replan result arrived too late: "
                            f"{elapsed_steps} actions were executed while the new chunk has only "
                            f"{chunk_steps} steps."
                        )
                    aligned_chunk = new_chunk[..., elapsed_steps:, :]
                    fused_steps = planner.update(aligned_chunk) if planner is not None else 0
                    pending_replan = None
                    self.log_queue.put(
                        f"[REPLAN] observation={max(observation_ready - observation_started, 0.0) * 1000:.1f} ms, "
                        f"async_pipeline={async_pipeline_s * 1000:.1f} ms, "
                        f"end_to_end={max(result_ready - observation_started, 0.0) * 1000:.1f} ms, "
                        f"executed_old={elapsed_steps} steps, "
                        f"discarded_new={elapsed_steps} steps, "
                        f"remaining_new={aligned_chunk.shape[-2]} steps, "
                        f"fused={fused_steps}"
                    )
                    return True

                while timestamp < episode_time_s and not self.stop_policy_requested:
                    loop_t = time.perf_counter()
                    observation_started = time.perf_counter()
                    raw_obs = self._get_eval_observation(robot)
                    obs_processed = robot_observation_processor(raw_obs)
                    observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                    policy_observation_frame = self._policy_observation_frame(
                        observation_frame,
                        dataset.features,
                        policy_cfg.input_features,
                    )
                    observation_history.append(copy(policy_observation_frame))
                    observation_ready = time.perf_counter()
                    if planner is None:
                        action_values = predict_action(
                            observation=policy_observation_frame,
                            policy=policy,
                            device=policy_device,
                            preprocessor=preprocessor,
                            postprocessor=postprocessor,
                            use_amp=policy.config.use_amp,
                            task=task,
                            robot_type=robot.robot_type,
                        )
                    elif execution_mode == "synchronous":
                        if planner.needs_replan:
                            new_chunk, pipeline_s = predict_chunk_async(observation_snapshot())
                            fused_steps = planner.update(new_chunk)
                            self.log_queue.put(
                                f"[REPLAN] synchronous_pipeline={pipeline_s * 1000:.1f} ms, "
                                f"predicted={new_chunk.shape[-2]} steps, fused={fused_steps}."
                            )
                        action_values = planner.pop_action()
                    else:
                        install_pending_replan()
                        if planner.remaining_history_steps == 0:
                            if pending_replan is not None:
                                self.log_queue.put("[REPLAN] Action buffer exhausted; waiting for inference.")
                                install_pending_replan(wait=True)
                            else:
                                new_chunk, inference_s = predict_chunk_async(observation_snapshot())
                                planner.update(new_chunk)
                                self.log_queue.put(
                                    f"[REPLAN] initial inference={inference_s * 1000:.1f} ms, "
                                    f"predicted={new_chunk.shape[-2]} steps."
                                )
                        if planner.needs_replan and pending_replan is None:
                            if replan_executor is None:
                                raise RuntimeError("Asynchronous replanning executor is unavailable.")
                            pending_replan = (
                                replan_executor.submit(predict_chunk_async, observation_snapshot()),
                                control_step,
                                observation_started,
                                observation_ready,
                            )
                            self.log_queue.put(
                                f"[REPLAN] Background pipeline started from snapshot at control step {control_step}; "
                                "continuing the previous action chunk."
                            )
                        action_values = planner.pop_action()
                    action_dict = make_robot_action(
                        action_values,
                        dataset.features,
                    )
                    self._fill_missing_action_values(action_dict, dataset.features, raw_obs)
                    robot_action_to_send = robot_action_processor((action_dict, raw_obs))
                    if self.lock_grippers.get():
                        robot_action_to_send = self._without_gripper_actions(robot_action_to_send)
                        for key in self._gripper_keys(action_dict):
                            if key in raw_obs:
                                action_dict[key] = raw_obs[key]
                    robot.send_action(robot_action_to_send)
                    control_step += 1
                    rate_window_steps += 1

                    action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
                    dataset.add_frame({**observation_frame, **action_frame, "task": task})

                    loop_s = max(time.perf_counter() - loop_t, 1e-6)
                    latest_masks = getattr(policy, "latest_inference_mask_preview", None)
                    model_masks = latest_masks() if callable(latest_masks) else {}
                    self._put_preview(
                        raw_obs,
                        action_dict,
                        1.0 / loop_s,
                        ", ".join(sorted(observation_frame)),
                        model_masks,
                    )
                    now = time.perf_counter()
                    if now - last_action_log_t > 1.0:
                        last_action_log_t = now
                        compact_action = ", ".join(f"{k}={float(v):.2f}" for k, v in action_dict.items())
                        self.log_queue.put(f"[ACTION] {compact_action}")
                    rate_window_s = now - rate_window_started
                    if rate_window_s >= 1.0:
                        self.log_queue.put(
                            f"[CONTROL] actual_fps={rate_window_steps / rate_window_s:.1f}, "
                            f"target_fps={fps}, step={control_step}."
                        )
                        rate_window_started = now
                        rate_window_steps = 0

                    dt_s = time.perf_counter() - loop_t
                    precise_sleep(max(1 / fps - dt_s, 0.0))
                    timestamp = time.perf_counter() - start_t

                dataset.save_episode()
                self.log_queue.put("Policy episode data saved. Robot remains connected.")
        finally:
            if replan_executor is not None:
                replan_executor.shutdown(wait=True, cancel_futures=True)
            if dataset is not None:
                dataset.finalize()

    def _validate_eval_settings(self) -> None:
        checkpoint = self.vars["checkpoint_path"].get().strip()
        if not checkpoint:
            raise ValueError("Select a pretrained_model or Mask-ACT checkpoint directory.")
        if not Path(checkpoint).expanduser().exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint}")
        if self.vars["policy_type"].get().strip() == "mask_act" and not self._selected_policy_variant():
            raise ValueError(
                "Could not determine the Mask-ACT experiment from mask_act_run_config.json."
            )
        if not self.vars["dataset_repo_id"].get().strip():
            raise ValueError("Dataset repo cannot be empty.")
        if not self.vars["task"].get().strip():
            raise ValueError("Task cannot be empty.")
        self._grid_metadata()
        trials_per_grid = int(self.vars["trials_per_grid"].get())
        if trials_per_grid <= 0:
            raise ValueError("Trials per cell must be greater than zero.")
        float(self.vars["episode_time_s"].get())
        fps = int(self.vars["fps"].get())
        if fps <= 0:
            raise ValueError("FPS must be greater than zero.")
        if self.vars["execution_mode"].get() not in {"synchronous", "asynchronous"}:
            raise ValueError("Execution mode must be synchronous or asynchronous.")
        if self.vars["camera_read_mode"].get() not in {"wait_new_frame", "latest_nonblocking"}:
            raise ValueError("Camera read mode must be wait_new_frame or latest_nonblocking.")
        if self.enable_segmentation_preview.get():
            segmentation_path = Path(self.vars["segmentation_model_path"].get()).expanduser()
            if not segmentation_path.is_file():
                raise ValueError(f"Segmentation checkpoint path does not exist: {segmentation_path}")
        action_steps_text = self.vars["n_action_steps"].get().strip()
        if not action_steps_text or int(action_steps_text) <= 0:
            raise ValueError("Replan interval must be greater than zero.")
        prediction_steps_text = self.vars["prediction_steps"].get().strip()
        if prediction_steps_text and int(prediction_steps_text) <= 0:
            raise ValueError("Prediction steps must be greater than zero.")
        inference_steps_text = self.vars["num_inference_steps"].get().strip()
        if inference_steps_text and int(inference_steps_text) <= 0:
            raise ValueError("Diffusion inference steps must be greater than zero.")
        if self.vars["noise_scheduler_type"].get() not in {"checkpoint", "DDPM", "DDIM"}:
            raise ValueError("Diffusion scheduler must be checkpoint, DDPM, or DDIM.")
        fusion_steps = int(self.vars["fusion_steps"].get())
        if fusion_steps < 0:
            raise ValueError("Fusion steps must be zero or greater.")
        fusion_history_weight = float(self.vars["fusion_history_weight"].get())
        if not 0.0 <= fusion_history_weight <= 1.0:
            raise ValueError("Initial history weight must be between 0 and 1.")
        reset_time_s = float(self.vars["reset_time_s"].get())
        if reset_time_s <= 0:
            raise ValueError("Reset sec must be greater than zero.")

    def _apply_action_step_override(self, policy_cfg: Any) -> None:
        requested_replan = int(self.vars["n_action_steps"].get())
        fusion_steps = int(self.vars["fusion_steps"].get())
        is_diffusion = all(
            hasattr(policy_cfg, attr)
            for attr in ("horizon", "n_obs_steps", "noise_scheduler_type", "num_inference_steps")
        )

        if is_diffusion:
            horizon = int(policy_cfg.horizon)
            n_obs_steps = int(policy_cfg.n_obs_steps)
            max_prediction_steps = horizon - n_obs_steps + 1
            prediction_text = self.vars["prediction_steps"].get().strip()
            prediction_steps = (
                int(prediction_text) if prediction_text else int(policy_cfg.n_action_steps)
            )
            if prediction_steps < 1 or prediction_steps > max_prediction_steps:
                raise ValueError(
                    "Diffusion prediction steps must be between 1 and "
                    f"horizon - n_obs_steps + 1 = {max_prediction_steps}; got {prediction_steps}."
                )
            if requested_replan > prediction_steps:
                raise ValueError(
                    f"Replan interval cannot exceed prediction steps={prediction_steps}; "
                    f"got {requested_replan}."
                )
            max_fusion_steps = prediction_steps - requested_replan
            if fusion_steps > max_fusion_steps:
                raise ValueError(
                    "Fusion steps cannot exceed prediction_steps - replan_interval "
                    f"= {max_fusion_steps}; got {fusion_steps}."
                )

            inference_steps_text = self.vars["num_inference_steps"].get().strip()
            if inference_steps_text:
                inference_steps = int(inference_steps_text)
                if inference_steps > int(policy_cfg.num_train_timesteps):
                    raise ValueError(
                        "Diffusion inference steps cannot exceed num_train_timesteps="
                        f"{policy_cfg.num_train_timesteps}; got {inference_steps}."
                    )
                policy_cfg.num_inference_steps = inference_steps
            scheduler = self.vars["noise_scheduler_type"].get()
            if scheduler != "checkpoint":
                policy_cfg.noise_scheduler_type = scheduler
            policy_cfg.n_action_steps = prediction_steps
            policy_cfg.use_amp = self.use_amp.get()
            self.vars["model_chunk_size"].set(str(horizon))
            self.vars["prediction_steps"].set(str(prediction_steps))
            self.log_queue.put(
                f"[Diffusion] horizon={horizon}, prediction_steps={prediction_steps}, "
                f"replan_interval={requested_replan} "
                f"({requested_replan / max(int(self.vars['fps'].get()), 1):.3f}s), "
                f"inference_steps={policy_cfg.num_inference_steps}, "
                f"scheduler={policy_cfg.noise_scheduler_type}, AMP={policy_cfg.use_amp}, "
                f"fusion_steps={fusion_steps}."
            )
            return

        if not hasattr(policy_cfg, "chunk_size") or not hasattr(policy_cfg, "n_action_steps"):
            if int(self.vars["fusion_steps"].get()) > 0:
                raise ValueError(
                    f"{type(policy_cfg).__name__} does not expose chunk_size/n_action_steps, "
                    "so action-chunk fusion cannot be enabled for this policy."
                )
            if requested_replan != 1:
                raise ValueError(
                    f"{type(policy_cfg).__name__} does not expose action chunks; set replan interval to 1."
                )
            if hasattr(policy_cfg, "use_amp"):
                policy_cfg.use_amp = self.use_amp.get()
            self.vars["model_chunk_size"].set("N/A")
            self.vars["prediction_steps"].set("1")
            self.log_queue.put(
                f"[Policy] {type(policy_cfg).__name__} uses single-step action execution; "
                f"replan_interval=1, AMP={getattr(policy_cfg, 'use_amp', self.use_amp.get())}."
            )
            return
        chunk_size = int(policy_cfg.chunk_size)
        if requested_replan < 1 or requested_replan > chunk_size:
            raise ValueError(
                f"Replan interval must be between 1 and model chunk_size={chunk_size}; "
                f"got {requested_replan}."
            )
        max_fusion_steps = chunk_size - requested_replan
        if fusion_steps > max_fusion_steps:
            raise ValueError(
                f"Fusion steps cannot exceed the aligned historical remainder "
                f"(chunk_size - replan_steps = {max_fusion_steps}); got {fusion_steps}."
            )
        if (
            fusion_steps == 0
            and getattr(policy_cfg, "temporal_ensemble_coeff", None) is not None
            and requested_replan != 1
        ):
            raise ValueError("ACT policies using temporal ensembling require replan interval = 1.")
        policy_cfg.n_action_steps = requested_replan
        policy_cfg.use_amp = self.use_amp.get()
        self.vars["model_chunk_size"].set(str(chunk_size))
        self.vars["prediction_steps"].set(str(chunk_size))
        self.log_queue.put(
            f"[Policy] chunk_size={chunk_size}, replan_interval={requested_replan}; "
            f"replanning every {requested_replan / max(int(self.vars['fps'].get()), 1):.3f}s; "
            f"fusion_steps={fusion_steps}, "
            f"initial_history_weight={float(self.vars['fusion_history_weight'].get()):.3f}."
        )

    def _make_robot(self) -> LocalBimanualSOFollower | Any:
        from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

        if self.vars["robot_mode"].get() != "bimanual" or self.vars["robot_type"].get().strip() != "bi_so_follower":
            raise ValueError("Policy evaluation is configured to connect exactly two SO101 follower arms.")

        calibration_dir = Path(self.vars["calibration_dir"].get()).expanduser()
        if not calibration_dir.exists():
            raise ValueError(f"Calibration directory does not exist: {calibration_dir}")
        cameras = self._camera_config_objects()
        left_id = self._calibration_id_for_port(
            "Left follower",
            self.vars["left_follower_port"].get().strip(),
            self.vars["left_follower_id"].get().strip(),
            calibration_dir,
        )
        right_id = self._calibration_id_for_port(
            "Right follower",
            self.vars["right_follower_port"].get().strip(),
            self.vars["right_follower_id"].get().strip(),
            calibration_dir,
        )
        left_arm = SO101Follower(
            SOFollowerRobotConfig(
                id=left_id,
                calibration_dir=calibration_dir,
                port=self.vars["left_follower_port"].get().strip(),
                disable_torque_on_disconnect=True,
                cameras=cameras,
            )
        )
        right_arm = SO101Follower(
            SOFollowerRobotConfig(
                id=right_id,
                calibration_dir=calibration_dir,
                port=self.vars["right_follower_port"].get().strip(),
                disable_torque_on_disconnect=True,
            )
        )
        self._patch_opencv_cameras_for_eval(left_arm)
        self.log_queue.put(f"Robot calibration dir: {calibration_dir}")
        self.log_queue.put(f"Left follower calibration id: {left_id}")
        self.log_queue.put(f"Right follower calibration id: {right_id}")
        return LocalBimanualSOFollower(left_arm, right_arm)

    def _patch_opencv_cameras_for_eval(self, arm: Any) -> None:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

        for key, camera in list(getattr(arm, "cameras", {}).items()):
            if isinstance(camera, OpenCVCamera):
                arm.cameras[key] = EvalFlexibleOpenCVCamera(camera.config)
                self.log_queue.put(
                    f"OpenCV {key} will be resized to {camera.config.width}x{camera.config.height} for policy input."
                )

    def _camera_config_objects(self) -> dict[str, Any]:
        from lerobot.cameras.configs import ColorMode
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        cameras: dict[str, Any] = {}
        camera_width, camera_height = self._camera_size()
        serial = self._resolve_realsense_serial() if self.enable_realsense.get() else ""
        front_path = self.vars["opencv_front"].get().strip()
        front_is_realsense_video = bool(front_path and serial and self._is_realsense_video_path(front_path))
        if self.enable_realsense.get() and (not front_path or front_is_realsense_video):
            cameras["front"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=camera_width,
                height=camera_height,
                fps=DEFAULT_CAMERA_FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
            if front_is_realsense_video:
                self.log_queue.put(
                    f"OpenCV front {front_path} is a RealSense RGB node; using the RealSense SDK like the recorder."
                )
        else:
            if not front_path:
                raise ValueError("RealSense disabled: OpenCV front cannot be empty.")
            cameras["front"] = OpenCVCameraConfig(
                index_or_path=self._opencv_value(front_path),
                width=camera_width,
                height=camera_height,
                fps=DEFAULT_CAMERA_FPS,
                fourcc="MJPG",
            )
            if serial:
                self.log_queue.put(f"Using OpenCV front RGB {front_path}; RealSense is not used for front RGB.")

        side_path = (
            self._resolve_opencv_side_path(self.vars["opencv_side"].get().strip())
            if self.include_side_camera.get()
            else ""
        )
        if side_path:
            side_description = self._describe_video_path(side_path)
            if side_description:
                self.log_queue.put(f"OpenCV side RGB {side_path} is {side_description}.")
            cameras["side"] = OpenCVCameraConfig(
                index_or_path=self._opencv_value(side_path),
                width=camera_width,
                height=camera_height,
                fps=DEFAULT_CAMERA_FPS,
                fourcc="MJPG",
            )
        return cameras

    def _resolve_opencv_side_path(self, requested_path: str) -> str:
        if not requested_path:
            return ""
        if not self._is_realsense_video_path(requested_path):
            return requested_path

        by_id_root = Path("/dev/v4l/by-id")
        if by_id_root.exists():
            for link in sorted(by_id_root.iterdir()):
                name = link.name.lower()
                if "realsense" in name or "intel(r)_real" in name:
                    continue
                try:
                    target = link.resolve()
                except OSError:
                    continue
                if target.name.startswith("video"):
                    self.log_queue.put(
                        f"OpenCV side {requested_path} is a RealSense node; using non-RealSense side RGB {target}."
                    )
                    return str(target)
        self.log_queue.put(
            "Only RealSense OpenCV video nodes were detected for side; evaluating without side RGB."
        )
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
        if by_id_root.exists():
            for link in sorted(by_id_root.iterdir()):
                try:
                    if link.resolve() == resolved:
                        return link.name
                except OSError:
                    continue
        return EvalPolicyApp._video_device_name(path)

    @staticmethod
    def _is_realsense_video_path(video_path: str) -> bool:
        description = EvalPolicyApp._describe_video_path(video_path).lower()
        return "realsense" in description or "intel(r) real sense" in description or "intel(r)_real" in description

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
        serial = EvalPolicyApp._serial_from_by_id_name(Path(port).name)
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
                    return EvalPolicyApp._serial_from_by_id_name(link.name)
            except OSError:
                continue
        return ""

    def _calibration_id_for_port(
        self, label: str, port: str, legacy_id: str, calibration_dir: Path | None
    ) -> str:
        if self.vars["calibration_match"].get() == "Arm id":
            return legacy_id
        serial = self._board_serial_from_port(port)
        if not serial:
            self.log_queue.put(f"{label} uses legacy calibration id '{legacy_id}' because no board serial was found.")
            return legacy_id
        if calibration_dir is not None:
            serial_path = calibration_dir / f"{serial}.json"
            legacy_path = calibration_dir / f"{legacy_id}.json"
            if not serial_path.exists() and legacy_path.exists():
                shutil.copy2(legacy_path, serial_path)
                self.log_queue.put(f"{label} calibration migrated: {legacy_path.name} -> {serial_path.name}")
        return serial

    def _get_eval_observation(self, robot: Any) -> dict[str, Any]:
        """Read arm state synchronously and optionally peek the latest camera frame without waiting."""
        if self.vars["camera_read_mode"].get() != "latest_nonblocking":
            return robot.get_observation()

        def read_camera(camera: Any) -> Any:
            read_latest = getattr(camera, "read_latest", None)
            if callable(read_latest):
                max_age_ms = max(int(3000 / max(int(self.vars["fps"].get()), 1)), 100)
                return read_latest(max_age_ms=max_age_ms)
            return camera.async_read()

        def read_arm(arm: Any) -> dict[str, Any]:
            positions = arm.bus.sync_read("Present_Position")
            observation = {f"{motor}.pos": value for motor, value in positions.items()}
            for camera_key, camera in arm.cameras.items():
                observation[camera_key] = read_camera(camera)
            return observation

        if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
            left_positions = robot.left_arm.bus.sync_read("Present_Position")
            right_positions = robot.right_arm.bus.sync_read("Present_Position")
            camera_obs = {
                camera_key: read_camera(camera)
                for camera_key, camera in getattr(robot, "cameras", {}).items()
            }
            return {
                **{f"left_{motor}.pos": value for motor, value in left_positions.items()},
                **{f"right_{motor}.pos": value for motor, value in right_positions.items()},
                **camera_obs,
            }
        if hasattr(robot, "bus") and hasattr(robot, "cameras"):
            return read_arm(robot)
        return robot.get_observation()

    def _policy_observation_frame(
        self,
        observation_frame: dict[str, Any],
        dataset_features: dict[str, dict[str, Any]],
        policy_input_features: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare the observation vector expected by the policy without changing recorded sensor data."""
        if "observation.state" not in observation_frame:
            return observation_frame

        state_names = self._policy_feature_names(dataset_features, policy_input_features, "observation.state")
        dataset_state_feature = dataset_features.get("observation.state", {})
        dataset_state_names = dataset_state_feature.get("names")
        if not isinstance(dataset_state_names, list):
            return observation_frame

        policy_frame = dict(observation_frame)
        state = observation_frame["observation.state"]
        clone = getattr(state, "clone", None)
        policy_state = clone() if callable(clone) else state.copy()
        if self.lock_grippers.get():
            for index, name in enumerate(dataset_state_names):
                baseline = DEFAULT_90_PERCENT_SUCCESS_POSE.get(str(name))
                if baseline is not None and self._is_gripper_key(str(name)):
                    policy_state[index] = baseline

        if list(state_names) != list(dataset_state_names):
            keep_indices = [dataset_state_names.index(name) for name in state_names]
            policy_state = self._select_vector_indices(policy_state, keep_indices)

        policy_frame["observation.state"] = policy_state
        return policy_frame

    @classmethod
    def _policy_feature_names(
        cls,
        dataset_features: dict[str, dict[str, Any]],
        policy_features: dict[str, Any],
        feature_key: str,
    ) -> list[str]:
        dataset_feature = dataset_features.get(feature_key)
        if not isinstance(dataset_feature, dict):
            return []
        names = dataset_feature.get("names")
        if not isinstance(names, list):
            return []

        policy_feature = policy_features.get(feature_key) if isinstance(policy_features, dict) else None
        expected_dim = cls._feature_dim(policy_feature)
        if expected_dim is None or expected_dim == len(names):
            return list(names)

        non_gripper_names = [str(name) for name in names if not cls._is_gripper_key(str(name))]
        if expected_dim == len(non_gripper_names):
            return non_gripper_names

        raise ValueError(
            f"Cannot map policy feature '{feature_key}' with dim {expected_dim} onto dataset names {names}. "
            "Only full vectors and gripper-dropped vectors are currently supported."
        )

    @staticmethod
    def _feature_dim(feature: Any) -> int | None:
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if shape is None:
            return None
        if len(shape) != 1:
            return None
        return int(shape[0])

    @staticmethod
    def _select_vector_indices(vector: Any, indices: list[int]) -> Any:
        try:
            return vector[indices]
        except Exception:
            return [vector[index] for index in indices]

    @staticmethod
    def _features_with_names(
        dataset_features: dict[str, dict[str, Any]],
        feature_key: str,
        names: list[str],
    ) -> dict[str, dict[str, Any]]:
        features = dict(dataset_features)
        feature = dict(dataset_features[feature_key])
        feature["names"] = list(names)
        feature["shape"] = (len(names),)
        features[feature_key] = feature
        return features

    @staticmethod
    def _fill_missing_action_values(
        action: dict[str, Any],
        dataset_features: dict[str, dict[str, Any]],
        raw_obs: dict[str, Any],
    ) -> None:
        action_feature = dataset_features.get("action", {})
        names = action_feature.get("names")
        if not isinstance(names, list):
            return
        for name in names:
            key = str(name)
            if key in action:
                continue
            if key in raw_obs:
                action[key] = raw_obs[key]

    @staticmethod
    def _opencv_value(index_or_path: str) -> int | Path:
        return int(index_or_path) if index_or_path.isdigit() else Path(index_or_path)

    @staticmethod
    def _preview_images(raw_obs: dict[str, Any]) -> dict[str, Any]:
        images: dict[str, Any] = {}
        for key in ("front", "side", "left_front", "left_side"):
            value = raw_obs.get(key)
            if hasattr(value, "ndim") and value.ndim == 3:
                display_key = key.removeprefix("left_")
                images[display_key] = value
        if images:
            return images
        for key, value in raw_obs.items():
            if hasattr(value, "ndim") and value.ndim == 3:
                images[str(key)] = value
        return images

    def _put_preview(
        self,
        raw_obs: dict[str, Any],
        action: dict[str, Any],
        fps_hz: float,
        input_keys: str,
        model_masks: dict[str, Any] | None = None,
    ) -> None:
        item = (self._preview_images(raw_obs), action, fps_hz, input_keys, model_masks or {})
        try:
            self.preview_queue.put_nowait(item)
        except queue.Full:
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                pass
            self.preview_queue.put_nowait(item)

    def capture_reset_pose(self) -> None:
        if not self.connected or self.robot is None:
            self.vars["status"].set("Robot is not connected.")
            return
        if self.eval_running or self.reset_running:
            self.vars["status"].set("Cannot capture while policy/reset is running.")
            return
        threading.Thread(target=self._capture_reset_pose_worker, daemon=True).start()

    def _capture_reset_pose_worker(self) -> None:
        try:
            robot = self.robot
            if robot is None:
                raise RuntimeError("Robot is not connected.")
            with self.robot_lock:
                obs = robot.get_observation()
                self.initial_joint_action = self._action_from_observation(robot, obs)
            self.log_queue.put(f"__CAPTURE_RESET_DONE__|0|{len(self.initial_joint_action)}")
        except Exception:
            self.log_queue.put(traceback.format_exc().rstrip())
            self.log_queue.put("__CAPTURE_RESET_DONE__|1|0")

    def reset_arms(self) -> None:
        if not self.connected or self.robot is None:
            self.vars["status"].set("Robot is not connected.")
            return
        if self.eval_running or self.teleop_running or self.reset_running:
            self.vars["status"].set("Stop policy/teleop/reset before manual reset.")
            return
        if not self.initial_joint_action:
            self.vars["status"].set("No reset pose captured yet.")
            return
        self.reset_running = True
        self.reset_button.configure(state=tk.DISABLED)
        self.default_pose_button.configure(state=tk.DISABLED)
        self.start_button.configure(state=tk.DISABLED)
        self.teleop_button.configure(state=tk.DISABLED)
        self.capture_reset_button.configure(state=tk.DISABLED)
        self.disconnect_button.configure(state=tk.DISABLED)
        self.vars["status"].set("Resetting arms to captured pose...")
        threading.Thread(target=self._reset_arms_worker, daemon=True).start()

    def _reset_arms_worker(self) -> None:
        try:
            if not self.initial_joint_action:
                raise RuntimeError("No reset pose captured.")
            self._move_robot_to_action(self.initial_joint_action, float(self.vars["reset_time_s"].get()))
            self.log_queue.put("__RESET_DONE__|0|")
        except Exception:
            self.log_queue.put(traceback.format_exc().rstrip())
            self.log_queue.put("__RESET_DONE__|1|")

    def go_to_default_success_pose(self) -> None:
        if not self.connected or self.robot is None:
            self.vars["status"].set("Robot is not connected.")
            return
        if self.eval_running or self.teleop_running or self.reset_running:
            self.vars["status"].set("Stop policy/teleop/reset before moving to the default pose.")
            return
        if self.vars["robot_mode"].get() != "bimanual":
            self.vars["status"].set("The 90% success pose is defined for bimanual mode only.")
            return

        self.reset_running = True
        self.start_button.configure(state=tk.DISABLED)
        self.teleop_button.configure(state=tk.DISABLED)
        self.capture_reset_button.configure(state=tk.DISABLED)
        self.reset_button.configure(state=tk.DISABLED)
        self.default_pose_button.configure(state=tk.DISABLED)
        self.disconnect_button.configure(state=tk.DISABLED)
        self.vars["status"].set("Moving arms to the 90% success default pose...")
        threading.Thread(target=self._default_success_pose_worker, daemon=True).start()

    def _default_success_pose_worker(self) -> None:
        try:
            target = dict(DEFAULT_90_PERCENT_SUCCESS_POSE)
            self._move_robot_to_action(
                target,
                float(self.vars["reset_time_s"].get()),
                include_grippers=True,
            )
            # Make subsequent Reset Arms and automatic post-policy reset return to this baseline.
            self.initial_joint_action = target
            self.log_queue.put(
                f"[RESET] Installed 90% success pose as the current reset pose ({len(target)} targets)."
            )
            self.log_queue.put("__DEFAULT_POSE_DONE__|0|")
        except Exception:
            self.log_queue.put(traceback.format_exc().rstrip())
            self.log_queue.put("__DEFAULT_POSE_DONE__|1|")

    def start_teleop(self) -> None:
        self.vars["status"].set("Leader teleop is disabled; policy evaluation only connects two follower arms.")
        return
        if not self.connected or self.robot is None:
            self.vars["status"].set("Robot is not connected.")
            return
        if self.eval_running or self.teleop_running or self.reset_running:
            self.vars["status"].set("Stop policy/teleop/reset before starting teleop.")
            return
        self.stop_teleop_requested = False
        self.teleop_running = True
        self.teleop_button.configure(state=tk.DISABLED)
        self.stop_teleop_button.configure(state=tk.NORMAL)
        self.start_button.configure(state=tk.DISABLED)
        self.disconnect_button.configure(state=tk.DISABLED)
        self.reset_button.configure(state=tk.DISABLED)
        self.default_pose_button.configure(state=tk.DISABLED)
        self.vars["status"].set("Teleop is running. Move leaders to adjust follower pose.")
        threading.Thread(target=self._teleop_worker, daemon=True).start()

    def stop_teleop(self) -> None:
        if not self.teleop_running:
            self.vars["status"].set("Teleop is not running.")
            return
        self.stop_teleop_requested = True
        self.vars["status"].set("Stopping teleop...")

    def _teleop_worker(self) -> None:
        teleop = None
        try:
            from lerobot.processor import make_default_processors
            from lerobot.utils.robot_utils import precise_sleep

            robot = self.robot
            if robot is None:
                raise RuntimeError("Robot is not connected.")
            teleop = self._make_teleop()
            teleop_action_processor, robot_action_processor, _robot_observation_processor = make_default_processors()
            teleop.connect()
            self.log_queue.put("[TELEOP] Leader arms connected.")
            fps = max(int(self.vars["fps"].get()), 1)
            last_log_t = 0.0
            while not self.stop_teleop_requested:
                loop_t = time.perf_counter()
                with self.robot_lock:
                    obs = robot.get_observation()
                    action = teleop_action_processor((teleop.get_action(), obs))
                    action_to_send = robot_action_processor((action, obs))
                    robot.send_action(action_to_send)
                now = time.perf_counter()
                if now - last_log_t > 1.0:
                    last_log_t = now
                    self.log_queue.put("[TELEOP] Streaming leader actions to follower arms.")
                precise_sleep(max(1 / fps - (time.perf_counter() - loop_t), 0.0))

            with self.robot_lock:
                obs = robot.get_observation()
                self.initial_joint_action = self._action_from_observation(robot, obs)
            self.log_queue.put(
                f"[TELEOP] Captured adjusted reset pose with {len(self.initial_joint_action)} joint target(s)."
            )
            self.log_queue.put("__TELEOP_DONE__|0|")
        except Exception:
            self.log_queue.put(traceback.format_exc().rstrip())
            self.log_queue.put("__TELEOP_DONE__|1|")
        finally:
            if teleop is not None:
                try:
                    teleop.disconnect()
                except Exception as exc:
                    self.log_queue.put(f"[TELEOP] Warning: leader disconnect failed: {exc}")

    def _make_teleop(self) -> Any:
        raise RuntimeError("Leader teleop is disabled; policy evaluation only connects two follower arms.")

    def _start_post_run_reset(self) -> None:
        if not self.auto_reset_after_policy.get():
            return
        if not self.connected or self.robot is None:
            self._append_log("[RESET] Post-run reset skipped because the robot is not connected.")
            return
        if self.initial_joint_action is None:
            self._append_log("[RESET] Post-run reset skipped because no policy-start pose was captured.")
            return
        if self.reset_running:
            self._append_log("[RESET] Post-run reset skipped because reset is already running.")
            return
        try:
            reset_time_s = float(self.vars["reset_time_s"].get())
        except ValueError as exc:
            self._append_log(f"[RESET] Post-run reset skipped because reset time is invalid: {exc}")
            return
        target_action = dict(self.initial_joint_action)

        def worker() -> None:
            try:
                self._append_log("[RESET] Returning arms to captured policy-start pose after policy end.")
                self._move_robot_to_action(target_action, reset_time_s)
                self._append_log("[RESET] Arms returned to the captured policy-start pose.")
            except Exception as exc:
                self._append_log(f"[RESET] Failed to return arms after policy end: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _move_robot_to_action(
        self,
        target_action: dict[str, float],
        duration_s: float,
        include_grippers: bool = False,
    ) -> None:
        from lerobot.utils.robot_utils import precise_sleep

        robot = self.robot
        if robot is None:
            raise RuntimeError("Robot is not connected.")
        fps = min(max(int(self.vars["fps"].get()), 1), 30)
        duration_s = max(duration_s, 0.1)
        was_reset_running = self.reset_running
        self.reset_running = True
        try:
            start_action = self._current_robot_motor_action(robot)
            keys = [key for key in robot.action_features if key in target_action and key in start_action]
            if self.lock_grippers.get() and not include_grippers:
                keys = [key for key in keys if not self._is_gripper_key(key)]
            if not keys:
                raise RuntimeError("No shared action keys found for reset.")

            steps = max(int(duration_s * fps), 1)
            move_start_t = time.perf_counter()
            for step in range(1, steps + 1):
                if not self.connected:
                    break
                alpha = step / steps
                eased = alpha * alpha * (3 - 2 * alpha)
                action = {
                    key: start_action[key] + (target_action[key] - start_action[key]) * eased
                    for key in keys
                }
                loop_t = time.perf_counter()
                with self.robot_lock:
                    robot.send_action(action)
                precise_sleep(max(1 / fps - (time.perf_counter() - loop_t), 0.0))
            with self.robot_lock:
                robot.send_action({key: target_action[key] for key in keys})
            elapsed_s = max(time.perf_counter() - move_start_t, 1e-6)
            self._append_log(f"[RESET] Sent {steps} reset steps over {elapsed_s:.2f}s ({steps / elapsed_s:.1f}Hz target={fps}Hz).")
        finally:
            self.reset_running = was_reset_running

    def _current_robot_motor_action(self, robot: Any) -> dict[str, float]:
        if hasattr(robot, "left_arm") and hasattr(robot, "right_arm"):
            with self.robot_lock:
                left_future = robot._io_executor.submit(robot._read_arm_motors, robot.left_arm)
                right_future = robot._io_executor.submit(robot._read_arm_motors, robot.right_arm)
                left_obs = left_future.result()
                right_obs = right_future.result()
            return {
                **{f"left_{key}": value for key, value in left_obs.items()},
                **{f"right_{key}": value for key, value in right_obs.items()},
            }
        with self.robot_lock:
            start_obs = robot.get_observation()
        return self._action_from_observation(robot, start_obs)

    @staticmethod
    def _action_from_observation(robot: Any, observation: dict[str, Any]) -> dict[str, float]:
        action: dict[str, float] = {}
        for key in robot.action_features:
            if key not in observation:
                continue
            try:
                action[key] = float(observation[key])
            except (TypeError, ValueError):
                continue
        return action

    @staticmethod
    def _is_gripper_key(key: str) -> bool:
        return "gripper" in key.lower()

    @classmethod
    def _gripper_keys(cls, keys: Any) -> list[str]:
        return [str(key) for key in keys if cls._is_gripper_key(str(key))]

    @classmethod
    def _without_gripper_actions(cls, action: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in action.items() if not cls._is_gripper_key(key)}

    @staticmethod
    def _video_features_to_images(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            key: ({**feature, "dtype": "image"} if feature.get("dtype") == "video" else dict(feature))
            for key, feature in features.items()
        }

    def disconnect_robot(self) -> None:
        if self.eval_running:
            self.vars["status"].set("Stop the policy loop first; robot remains connected.")
            return
        if self.teleop_running:
            self.vars["status"].set("Stop teleop before disconnecting.")
            return
        if self.reset_running:
            self.vars["status"].set("Wait for reset to finish before disconnecting.")
            return
        if not self.connected or self.robot is None:
            self.vars["status"].set("Robot is not connected.")
            return
        self.disconnect_button.configure(state=tk.DISABLED)
        self.connect_button.configure(state=tk.DISABLED)
        self.start_button.configure(state=tk.DISABLED)
        self.vars["status"].set("Disconnecting robot and cameras...")
        threading.Thread(target=self._disconnect_worker, daemon=True).start()

    def _disconnect_worker(self) -> None:
        self._stop_connected_preview()
        robot = self.robot
        self._safe_disconnect_robot(robot)
        with self.robot_lock:
            self.robot = None
            self.connected = False
            self.initial_joint_action = None
        self.log_queue.put("__DISCONNECT_DONE__|0|")

    def _start_connected_preview(self) -> None:
        if self.connected_preview_thread is not None and self.connected_preview_thread.is_alive():
            return
        stop_event = threading.Event()
        self.connected_preview_stop = stop_event
        self.connected_preview_thread = threading.Thread(
            target=self._connected_preview_worker,
            args=(stop_event,),
            daemon=True,
        )
        self.connected_preview_thread.start()

    def _stop_connected_preview(self) -> None:
        if self.connected_preview_stop is not None:
            self.connected_preview_stop.set()
        self.connected_preview_stop = None
        self.connected_preview_thread = None

    def _connected_preview_worker(self, stop_event: threading.Event) -> None:
        last_error_t = 0.0
        while not stop_event.is_set():
            if not self.connected or self.robot is None:
                break
            if self.eval_running or self.reset_running:
                time.sleep(0.1)
                continue
            loop_t = time.perf_counter()
            try:
                with self.robot_lock:
                    raw_obs = self._get_eval_observation(self.robot)
                loop_s = max(time.perf_counter() - loop_t, 1e-6)
                self._put_preview(raw_obs, {}, 1.0 / loop_s, ", ".join(sorted(raw_obs)))
            except Exception as exc:
                now = time.perf_counter()
                if now - last_error_t > 2.0:
                    last_error_t = now
                    self.log_queue.put(f"[PREVIEW] Warning: {exc}")
            time.sleep(0.1)

    def _safe_disconnect_robot(self, robot: Any | None) -> None:
        if robot is None:
            return
        try:
            if robot.is_connected:
                robot.disconnect()
        except Exception as exc:
            self.log_queue.put(f"[GUI] Warning: disconnect failed: {exc}")
        for cam in self._iter_robot_cameras(robot):
            try:
                if getattr(cam, "is_connected", False):
                    cam.disconnect()
            except Exception as exc:
                self.log_queue.put(f"[GUI] Warning: camera cleanup failed: {exc}")

    @staticmethod
    def _iter_robot_cameras(robot: Any) -> list[Any]:
        cameras: list[Any] = []
        seen: set[int] = set()
        for owner in (robot, getattr(robot, "left_arm", None), getattr(robot, "right_arm", None)):
            for cam in getattr(owner, "cameras", {}).values():
                ident = id(cam)
                if ident not in seen:
                    seen.add(ident)
                    cameras.append(cam)
        return cameras

    def _build_connect_command(self) -> list[str]:
        command = self._base_python_command()
        command.extend(["mycode/check_lerobot_robot_connection.py", *self._build_robot_args()])
        return command

    def _build_command(self) -> list[str]:
        checkpoint = self.vars["checkpoint_path"].get().strip()
        if not checkpoint:
            raise ValueError("Select a pretrained_model checkpoint directory.")
        if not Path(checkpoint).expanduser().exists():
            raise ValueError(f"Checkpoint path does not exist: {checkpoint}")

        repo_id = self.vars["dataset_repo_id"].get().strip()
        if not repo_id:
            raise ValueError("Dataset repo cannot be empty, for example: seeed/eval_test123.")

        task = self.vars["task"].get().strip()
        if not task:
            raise ValueError("Task cannot be empty.")

        try:
            float(self.vars["episode_time_s"].get())
            int(self.vars["fps"].get())
        except ValueError as exc:
            raise ValueError("Episode sec must be numeric and FPS must be an integer.") from exc

        save_parent = self._save_parent()
        run_dataset_root = self._make_run_dataset_root(save_parent, repo_id)
        self.active_dataset_root = run_dataset_root

        command = self._base_record_command()
        command.extend(self._build_robot_args())
        command.extend(
            [
                f"--dataset.repo_id={repo_id}",
                f"--dataset.root={run_dataset_root}",
                f"--dataset.single_task={task}",
                f"--dataset.episode_time_s={self.vars['episode_time_s'].get().strip()}",
                f"--dataset.fps={self.vars['fps'].get().strip()}",
                f"--policy.path={checkpoint}",
            ]
        )

        extra_args = self.vars["extra_args"].get().strip()
        if extra_args:
            command.extend(shlex.split(extra_args))
        return command

    def _base_record_command(self) -> list[str]:
        command = self._base_python_command()
        command.extend(["mycode/live_eval_lerobot_policy.py"])
        return command

    def _base_python_command(self) -> list[str]:
        conda_env = self.vars["conda_env"].get().strip()
        command = ["python", "-u"]
        if conda_env and not self._already_in_conda_env(conda_env):
            command = ["conda", "run", "--no-capture-output", "-n", conda_env, *command]
        return command

    @staticmethod
    def _already_in_conda_env(conda_env: str) -> bool:
        return os.environ.get("CONDA_DEFAULT_ENV") == conda_env

    def _build_robot_args(self) -> list[str]:
        robot_mode = self.vars["robot_mode"].get()
        robot_type = self.vars["robot_type"].get().strip()
        if robot_mode == "bimanual":
            if robot_type != "bi_so_follower":
                raise ValueError("For bimanual evaluation, set Robot type to bi_so_follower.")
            runtime_calibration_dir = self._prepare_bimanual_calibration()
            return [
                "--robot.type=bi_so_follower",
                f"--robot.id={self.vars['robot_id'].get().strip()}",
                f"--robot.calibration_dir={runtime_calibration_dir}",
                f"--robot.left_arm_config.port={self.vars['left_follower_port'].get().strip()}",
                f"--robot.right_arm_config.port={self.vars['right_follower_port'].get().strip()}",
                f"--robot.left_arm_config.cameras={self._camera_config_for_command()}",
            ]

        if robot_type == "bi_so_follower":
            raise ValueError("For single-arm evaluation, use so101_follower or so100_follower.")
        return [
            f"--robot.type={robot_type}",
            f"--robot.port={self.vars['robot_port'].get().strip() or self.vars['left_follower_port'].get().strip()}",
            f"--robot.cameras={self._camera_config_for_command()}",
            f"--robot.id={self.vars['left_follower_id'].get().strip()}",
            f"--robot.calibration_dir={self.vars['calibration_dir'].get().strip()}",
        ]

    def _save_parent(self) -> Path:
        return self._policy_save_parent()

    def _make_run_dataset_root(self, save_parent: Path, repo_id: str) -> Path:
        repo_name = repo_id.rstrip("/").split("/")[-1] or "eval"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = save_parent / f"{repo_name}_{timestamp}"
        suffix = 1
        while candidate.exists():
            candidate = save_parent / f"{repo_name}_{timestamp}_{suffix:02d}"
            suffix += 1
        return candidate

    def _prepare_bimanual_calibration(self) -> Path:
        base_id = self.vars["robot_id"].get().strip()
        left_id = self.vars["left_follower_id"].get().strip()
        right_id = self.vars["right_follower_id"].get().strip()
        if not base_id:
            raise ValueError("Runtime id cannot be empty.")
        if not left_id or not right_id:
            raise ValueError("Left calib id and Right calib id cannot be empty.")

        source_dir = Path(self.vars["calibration_dir"].get()).expanduser()
        if not source_dir.exists():
            raise ValueError(f"Calibration directory does not exist: {source_dir}")

        left_src = source_dir / f"{left_id}.json"
        right_src = source_dir / f"{right_id}.json"
        missing = [str(path) for path in (left_src, right_src) if not path.exists()]
        if missing:
            raise ValueError("Could not find bimanual calibration files:\n" + "\n".join(missing))

        save_root = self._base_save_root()
        runtime_dir = save_root / "_runtime_calibration" / base_id
        runtime_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(left_src, runtime_dir / f"{base_id}_left.json")
        shutil.copy2(right_src, runtime_dir / f"{base_id}_right.json")
        return runtime_dir

    def stop_eval(self) -> None:
        if self.eval_running:
            self.stop_policy_requested = True
            self.vars["status"].set("Stopping policy loop; keeping robot connected.")
            self._append_log("[GUI] Stop requested. Policy will stop, robot stays connected.")
            return
        if self.process is None or self.process.poll() is not None:
            self.vars["status"].set("No active policy run. Robot remains connected.")
            return
        self.stop_requested = True
        self.vars["status"].set("Asking lerobot-record to finish and save...")
        if self._send_escape_key():
            self._append_log("[GUI] Sent Esc key to request a graceful stop.")
        else:
            self._append_log("[GUI] Could not send Esc key; will use SIGINT fallback.")
        if not self.signal_fallback_scheduled:
            self.signal_fallback_scheduled = True
            self.root.after(8000, self._force_stop_if_running)

    def _send_escape_key(self) -> bool:
        try:
            from pynput.keyboard import Controller, Key

            keyboard = Controller()
            keyboard.press(Key.esc)
            keyboard.release(Key.esc)
            return True
        except Exception as exc:
            self._append_log(f"[GUI] Esc key request failed: {exc}")
            return False

    def _force_stop_if_running(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.vars["status"].set("Graceful stop timed out; sending SIGINT...")
        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        except ProcessLookupError:
            pass
        except OSError as exc:
            self._append_log(f"[GUI] Failed to send SIGINT: {exc}")

    def _read_process_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            self.log_queue.put(line.rstrip())
        return_code = process.wait()
        self.log_queue.put(f"__PROCESS_DONE__|{return_code}")

    def _tick(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if message.startswith("__PROCESS_DONE__|"):
                return_code = int(message.split("|", 1)[1])
                self._handle_process_done(return_code)
            elif message.startswith("__CONNECT_DONE__|"):
                self.connecting = False
                ok = message.split("|", 2)[1] == "0"
                self.connect_button.configure(state=tk.DISABLED if ok else tk.NORMAL)
                self.start_button.configure(state=tk.NORMAL if ok else tk.DISABLED)
                self.disconnect_button.configure(state=tk.NORMAL if ok else tk.DISABLED)
                self.teleop_button.configure(state=tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if ok else tk.DISABLED)
                self.reset_button.configure(state=tk.NORMAL if ok and self.initial_joint_action else tk.DISABLED)
                self.default_pose_button.configure(state=tk.NORMAL if ok else tk.DISABLED)
                self.stop_teleop_button.configure(state=tk.DISABLED)
                self.stop_button.configure(state=tk.DISABLED)
                if ok:
                    self._start_connected_preview()
                self.vars["status"].set(
                    "Connected. Start will reuse this connection."
                    if ok
                    else "Connection failed."
                )
            elif message.startswith("__EVAL_DONE__|"):
                parts = message.split("|", 2)
                ok = parts[1] == "0"
                elapsed_s = 0.0 if self.started_at is None else time.monotonic() - self.started_at
                self.eval_running = False
                self.started_at = None
                self.start_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.stop_button.configure(state=tk.DISABLED)
                self.disconnect_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.teleop_button.configure(state=tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.reset_button.configure(
                    state=tk.NORMAL if self.connected and self.initial_joint_action else tk.DISABLED
                )
                self.default_pose_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.vars["elapsed"].set(f"Elapsed: {int(elapsed_s) // 60:02d}:{int(elapsed_s) % 60:02d}")
                self.last_run = {
                    "started_at": self.started_at_iso,
                    "ended_at": datetime.now().isoformat(timespec="seconds"),
                    "duration_s": round(elapsed_s, 3),
                    "return_code": 0 if ok else 1,
                    "manual_stop": self.stop_policy_requested,
                    "policy_type": self.vars["policy_type"].get(),
                    "policy_path": self.vars["checkpoint_path"].get(),
                    "dataset_repo_id": self.vars["dataset_repo_id"].get(),
                    "save_root": self.active_eval_metadata.get("save_root", self.vars["dataset_root"].get()),
                    "save_parent": self.active_eval_metadata.get("save_parent")
                    or str(self._policy_save_parent()),
                    "dataset_root": str(self.active_dataset_root) if self.active_dataset_root else None,
                    "task": self.vars["task"].get(),
                    "command": "in-process persistent eval",
                    **self.active_eval_metadata,
                    **self._segmentation_metric_record(),
                }
                self.result_saved = False
                self._reset_result_fields()
                self._apply_auto_result_fields()
                self._start_post_run_reset()
                if ok:
                    status = "Policy stopped; reset is running in background. Save Result will encode videos."
                else:
                    status = "Policy failed; reset is running in background. Review the auto labels, then save or discard."
                self.vars["status"].set(status)
                self._show_result_dialog()
            elif message.startswith("__DISCONNECT_DONE__|"):
                self.connect_button.configure(state=tk.NORMAL)
                self.start_button.configure(state=tk.DISABLED)
                self.disconnect_button.configure(state=tk.DISABLED)
                self.stop_button.configure(state=tk.DISABLED)
                self.teleop_button.configure(state=tk.DISABLED)
                self.stop_teleop_button.configure(state=tk.DISABLED)
                self.capture_reset_button.configure(state=tk.DISABLED)
                self.reset_button.configure(state=tk.DISABLED)
                self.default_pose_button.configure(state=tk.DISABLED)
                self.vars["status"].set("Disconnected.")
            elif message.startswith("__TELEOP_DONE__|"):
                ok = message.split("|", 2)[1] == "0"
                self.teleop_running = False
                self.stop_teleop_requested = False
                self.start_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.teleop_button.configure(state=tk.DISABLED)
                self.stop_teleop_button.configure(state=tk.DISABLED)
                self.disconnect_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.reset_button.configure(
                    state=tk.NORMAL if self.connected and self.initial_joint_action else tk.DISABLED
                )
                self.default_pose_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.vars["status"].set(
                    "Teleop stopped. Adjusted pose captured as reset pose."
                    if ok
                    else "Teleop failed/stopped with error."
                )
            elif message.startswith("__RESET_DONE__|"):
                ok = message.split("|", 2)[1] == "0"
                self.reset_running = False
                self.start_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.teleop_button.configure(state=tk.DISABLED)
                self.disconnect_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.reset_button.configure(
                    state=tk.NORMAL if self.connected and self.initial_joint_action else tk.DISABLED
                )
                self.default_pose_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.vars["status"].set("Reset complete." if ok else "Reset failed.")
            elif message.startswith("__DEFAULT_POSE_DONE__|"):
                ok = message.split("|", 2)[1] == "0"
                self.reset_running = False
                self.start_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.teleop_button.configure(state=tk.DISABLED)
                self.disconnect_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.reset_button.configure(
                    state=tk.NORMAL if self.connected and self.initial_joint_action else tk.DISABLED
                )
                self.default_pose_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.vars["status"].set(
                    "Arms reached the 90% success pose; it is now the active reset pose."
                    if ok
                    else "Moving to the 90% success pose failed."
                )
            elif message.startswith("__CAPTURE_RESET_DONE__|"):
                parts = message.split("|", 2)
                ok = parts[1] == "0"
                count = parts[2] if len(parts) > 2 else "0"
                self.reset_button.configure(
                    state=tk.NORMAL if ok and self.connected and self.initial_joint_action else tk.DISABLED
                )
                self.vars["status"].set(
                    f"Captured reset pose with {count} joint target(s)." if ok else "Capture reset pose failed."
                )
            else:
                self._append_log(message)

        while True:
            try:
                images, action, fps_hz, input_keys, model_masks = self.preview_queue.get_nowait()
            except queue.Empty:
                break
            self._update_preview(images, action, fps_hz, input_keys, model_masks)

        if self.started_at is not None and (self.eval_running or self.process is not None):
            elapsed = int(time.monotonic() - self.started_at)
            self.vars["elapsed"].set(f"Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d}")

        self.root.after(200, self._tick)

    @staticmethod
    def _fit_preview_image(image: Any, target_width: int, *, nearest: bool = False) -> Any:
        from PIL import Image

        resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
        width = min(max(int(target_width), PREVIEW_MIN_WIDTH), PREVIEW_MAX_WIDTH)
        target_size = (width, round(width * 9 / 16))
        fitted = image.copy()
        fitted.thumbnail(target_size, resampling)
        canvas = Image.new("RGB", target_size, "black")
        canvas.paste(
            fitted,
            ((target_size[0] - fitted.width) // 2, (target_size[1] - fitted.height) // 2),
        )
        return canvas

    @staticmethod
    def _render_model_mask_previews(
        model_masks: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        import numpy as np

        semantic_order = ("occluder", "object", "region", "tool")
        semantic_colors = {
            semantic: SEGMENTATION_PALETTE[idx]
            for idx, semantic in enumerate(semantic_order, start=1)
        }
        masks_by_view: dict[str, dict[str, Any]] = {"front": {}, "side": {}}
        for key, value in model_masks.items():
            leaf = str(key).rsplit(".", 1)[-1]
            view, separator, semantic = leaf.partition("_")
            if not separator or view not in masks_by_view or semantic not in semantic_colors:
                continue
            if hasattr(value, "numpy"):
                value = value.numpy()
            array = np.asarray(value)
            if array.ndim > 2:
                array = np.squeeze(array)
            if array.ndim != 2:
                continue
            masks_by_view[view][semantic] = array.astype(np.float32) / 255.0

        overlays: dict[str, Any] = {}
        summaries: dict[str, str] = {}
        for view, semantic_masks in masks_by_view.items():
            if not semantic_masks:
                continue
            available = [semantic for semantic in semantic_order if semantic in semantic_masks]
            probabilities = np.stack([semantic_masks[semantic] for semantic in available], axis=0)
            winning_index = probabilities.argmax(axis=0)
            confidence = probabilities.max(axis=0)
            overlay = np.zeros((*confidence.shape, 3), dtype=np.uint8)
            for idx, semantic in enumerate(available):
                selected = (winning_index == idx) & (confidence >= 0.5)
                overlay[selected] = semantic_colors[semantic]
            overlays[view] = np.ascontiguousarray(overlay)
            summaries[view] = ", ".join(
                f"{semantic} {100.0 * float((semantic_masks[semantic] >= 0.5).mean()):.1f}%"
                for semantic in available
            )
        return overlays, summaries

    def _update_preview(
        self,
        images: dict[str, Any],
        action: dict[str, Any],
        fps_hz: float,
        input_keys: str,
        model_masks: dict[str, Any],
    ) -> None:
        try:
            from PIL import Image, ImageTk
            import numpy as np

            for image_key, image in images.items():
                label = self.preview_labels.get(image_key)
                if label is None or image is None:
                    continue
                arr = np.asarray(image)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
                    pil = self._fit_preview_image(pil, label.winfo_width())
                    self.current_images[image_key] = ImageTk.PhotoImage(pil)
                    label.configure(image=self.current_images[image_key])
            if self.enable_segmentation_preview.get():
                if self.segmentation_model is None:
                    self.segmentation_model = SegmentationPreviewModel(
                        self.vars["segmentation_model_path"].get(),
                        self.vars["segmentation_device"].get(),
                    )
                overlays, summaries, measurements = self.segmentation_model.predict(images)
                self._update_segmentation_eval_metrics(measurements)
                for image_key, overlay in overlays.items():
                    label = self.segmentation_preview_labels.get(image_key)
                    if label is None:
                        continue
                    pil = Image.fromarray(np.asarray(overlay).astype(np.uint8), mode="RGB")
                    pil = self._fit_preview_image(pil, label.winfo_width(), nearest=True)
                    self.current_segmentation_images[image_key] = ImageTk.PhotoImage(pil)
                    label.configure(image=self.current_segmentation_images[image_key])
                    summary = self.segmentation_summary_labels.get(image_key)
                    if summary is not None:
                        summary.configure(text=summaries.get(image_key, ""))
                if not self.segmentation_loaded_logged and self.segmentation_model.loaded_device != "not-loaded":
                    self.segmentation_loaded_logged = True
                    self._append_log(
                        f"[SEG] Loaded {self.vars['segmentation_model_path'].get()} "
                        f"on {self.segmentation_model.loaded_device}."
                    )
            else:
                self.current_segmentation_images.clear()
                for label in self.segmentation_preview_labels.values():
                    label.configure(image="")
                for summary in self.segmentation_summary_labels.values():
                    summary.configure(text="")

            model_overlays, model_summaries = self._render_model_mask_previews(model_masks)
            for image_key, label in self.model_mask_preview_labels.items():
                overlay = model_overlays.get(image_key)
                summary = self.model_mask_summary_labels.get(image_key)
                if overlay is None:
                    self.current_model_mask_images.pop(image_key, None)
                    label.configure(image="")
                    if summary is not None:
                        summary.configure(text="Waiting for Mask-ACT inference")
                    continue
                pil = Image.fromarray(overlay, mode="RGB")
                pil = self._fit_preview_image(pil, label.winfo_width(), nearest=True)
                self.current_model_mask_images[image_key] = ImageTk.PhotoImage(pil)
                label.configure(image=self.current_model_mask_images[image_key])
                if summary is not None:
                    summary.configure(text=model_summaries.get(image_key, ""))
        except Exception as exc:
            if not self.segmentation_error_logged:
                self.segmentation_error_logged = True
                self._append_log(f"[GUI] Preview/segmentation update failed: {exc}")

        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, f"image_keys: {', '.join(images) or 'none'}\n")
        self.log_text.insert(tk.END, f"loop_hz: {fps_hz:.1f}\n")
        self.log_text.insert(tk.END, f"policy inputs: {input_keys}\n\n")
        auto = self.segmentation_eval_metrics
        self.log_text.insert(
            tk.END,
            "auto metrics: "
            f"final_exposed={auto['final_object_exposed']}, "
            f"final_separated={auto['final_object_separated']}, "
            f"final_in_region={auto['final_object_reached_region']}, "
            f"final_front_cover={auto['final_front_coverage']}, "
            f"final_success={auto['final_task_success']}, "
            f"ever_exposed={auto['auto_object_exposed']}, "
            f"ever_separated={auto['auto_object_separated']}, "
            f"ever_in_region={auto['auto_object_reached_region']}, "
            f"front_occ_last={100.0 * float(auto['last_front_occluder_coverage']):.1f}%, "
            f"reason={self._segmentation_final_failure_reason(auto)}\n\n",
        )
        self.log_text.insert(tk.END, "policy action:\n")
        for key, value in action.items():
            try:
                text = f"{float(value): .4f}"
            except (TypeError, ValueError):
                text = str(value)
            self.log_text.insert(tk.END, f"{key:<28} {text}\n")

    def _handle_process_done(self, return_code: int) -> None:
        elapsed_s = 0.0 if self.started_at is None else time.monotonic() - self.started_at
        ended_at = datetime.now().isoformat(timespec="seconds")
        process_kind = self.process_kind
        self.last_run = {
            "started_at": self.started_at_iso,
            "ended_at": ended_at,
            "duration_s": round(elapsed_s, 3),
            "return_code": return_code,
            "manual_stop": self.stop_requested,
            "policy_type": self.vars["policy_type"].get(),
            "policy_path": self.vars["checkpoint_path"].get(),
            "dataset_repo_id": self.vars["dataset_repo_id"].get(),
            "save_root": self.active_eval_metadata.get("save_root", self.vars["dataset_root"].get()),
            "save_parent": self.active_eval_metadata.get("save_parent") or str(self._policy_save_parent()),
            "dataset_root": str(self.active_dataset_root) if self.active_dataset_root else None,
            "task": self.vars["task"].get(),
            "command": shlex.join(self.last_command),
            **self.active_eval_metadata,
        }
        self.result_saved = False
        self._reset_result_fields()
        self._apply_auto_result_fields()
        self.process = None
        self.started_at = None
        self.process_kind = ""
        self.connect_button.configure(state=tk.NORMAL)
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.vars["elapsed"].set(f"Elapsed: {int(elapsed_s) // 60:02d}:{int(elapsed_s) % 60:02d}")
        if process_kind == "connect":
            self.vars["status"].set(
                "Connection check OK. You can click Start."
                if return_code == 0
                else f"Connection check failed with code {return_code}."
            )
            self._append_log(f"[GUI] Connection check exited with code {return_code}")
            return

        self.vars["status"].set(
            "Run finished. Choose success/failure, then Save Result."
            if return_code == 0
            else f"Run exited with code {return_code}. Choose result, then Save Result."
        )
        self._append_log(f"[GUI] Process exited with code {return_code}")
        self._start_post_run_reset()
        self._show_result_dialog()

    def _show_result_dialog(self) -> None:
        if not self.last_run:
            self.vars["status"].set("No finished run to review.")
            return
        if self.result_dialog is not None and self.result_dialog.winfo_exists():
            self.result_dialog.deiconify()
            self.result_dialog.lift()
            self.result_dialog.focus_set()
            return
        self._apply_auto_result_fields()
        dialog = tk.Toplevel(self.root)
        self.result_dialog = dialog
        dialog.title("Save evaluation result")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.columnconfigure(0, weight=0)
        dialog.columnconfigure(1, weight=1)
        ttk.Label(dialog, text="Result").grid(row=0, column=0, sticky=tk.W, padx=14, pady=(14, 8))
        ttk.Combobox(
            dialog,
            textvariable=self.vars["result"],
            values=RESULT_CHOICES,
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky=tk.EW, padx=(0, 14), pady=(14, 8))

        ttk.Label(dialog, text="Object exposed").grid(row=1, column=0, sticky=tk.W, padx=14, pady=8)
        ttk.Combobox(
            dialog,
            textvariable=self.vars["object_exposed"],
            values=("yes", "no"),
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky=tk.EW, padx=(0, 14), pady=8)

        ttk.Label(dialog, text="Object separated from occlusion").grid(
            row=2, column=0, sticky=tk.W, padx=14, pady=8
        )
        ttk.Combobox(
            dialog,
            textvariable=self.vars["object_pushed_out"],
            values=("yes", "no"),
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky=tk.EW, padx=(0, 14), pady=8)

        ttk.Label(dialog, text="Task success (region + front cover)").grid(
            row=3, column=0, sticky=tk.W, padx=14, pady=8
        )
        ttk.Combobox(
            dialog,
            textvariable=self.vars["object_push_success"],
            values=("yes", "no"),
            state="readonly",
            width=18,
        ).grid(row=3, column=1, sticky=tk.EW, padx=(0, 14), pady=8)

        details_frame = ttk.LabelFrame(dialog, text="Auto segmentation details", padding=(10, 8))
        details_frame.grid(row=4, column=0, columnspan=2, sticky=tk.EW, padx=14, pady=(4, 8))
        details_frame.columnconfigure(1, weight=1)
        detail_labels: list[tuple[str, tk.Label, tk.Label]] = []
        for detail_row, (name, value, category) in enumerate(self._auto_result_detail_rows()):
            name_label = tk.Label(details_frame, text=name, anchor="w")
            value_label = tk.Label(details_frame, text=value, anchor="w")
            name_label.grid(row=detail_row, column=0, sticky=tk.W, padx=(0, 18), pady=2)
            value_label.grid(row=detail_row, column=1, sticky=tk.W, pady=2)
            detail_labels.append((category, name_label, value_label))

        def update_detail_state(*_args: Any) -> None:
            result_success = self.vars["result"].get() == "success"
            for category, name_label, value_label in detail_labels:
                color = "#8a8a8a" if result_success and category == "diagnostic" else "#000000"
                name_label.configure(fg=color)
                value_label.configure(fg=color)

        result_trace_id = self.vars["result"].trace_add("write", update_detail_state)
        update_detail_state()

        def close_dialog() -> None:
            try:
                self.vars["result"].trace_remove("write", result_trace_id)
            except tk.TclError:
                pass
            if self.result_dialog is dialog:
                self.result_dialog = None
            dialog.destroy()

        tags_frame = ttk.LabelFrame(dialog, text="Tags", padding=(10, 8))
        tags_frame.grid(row=5, column=0, columnspan=2, sticky=tk.EW, padx=14, pady=(0, 8))
        tags_frame.columnconfigure(0, weight=1)
        ttk.Entry(tags_frame, textvariable=self.vars["result_tags"]).grid(
            row=0, column=0, columnspan=3, sticky=tk.EW
        )
        ttk.Label(tags_frame, text="Comma-separated labels saved with this trial.").grid(
            row=1, column=0, columnspan=3, sticky=tk.W, pady=(4, 0)
        )
        tag_candidate_var = tk.StringVar(value="")
        tag_candidates = self._result_tag_candidates()
        tag_combo = ttk.Combobox(
            tags_frame,
            textvariable=tag_candidate_var,
            values=tag_candidates,
            state="readonly" if tag_candidates else tk.DISABLED,
        )
        tag_combo.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))

        def add_selected_tag() -> None:
            selected_tag = tag_candidate_var.get().strip()
            if not selected_tag:
                return
            tags = self._parse_result_tags(self.vars["result_tags"].get())
            if selected_tag.lower() not in {tag.lower() for tag in tags}:
                tags.append(selected_tag)
            self.vars["result_tags"].set(", ".join(tags))

        ttk.Button(tags_frame, text="Add Tag", command=add_selected_tag).grid(
            row=2, column=1, sticky=tk.E, padx=(8, 0), pady=(8, 0)
        )
        ttk.Button(tags_frame, text="Clear", command=lambda: self.vars["result_tags"].set("")).grid(
            row=2, column=2, sticky=tk.E, padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(dialog, text=self.vars["elapsed"].get()).grid(
            row=6, column=0, columnspan=2, sticky=tk.W, padx=14, pady=(4, 8)
        )
        buttons = ttk.Frame(dialog)
        buttons.grid(row=7, column=0, columnspan=2, sticky=tk.E, padx=14, pady=(4, 14))

        save_button: ttk.Button
        later_button: ttk.Button
        discard_button: ttk.Button

        def save_and_close() -> None:
            if self.saving_result:
                return
            try:
                record, result_path = self._prepare_result_save()
            except ValueError as exc:
                self._alert("Cannot save result", str(exc))
                return
            self.saving_result = True
            self.vars["status"].set("Saving result and encoding videos...")
            for button in (save_button, later_button, discard_button):
                button.configure(state=tk.DISABLED)

            def worker() -> None:
                try:
                    self._commit_result_save(record, result_path)
                except Exception as exc:
                    self.root.after(0, lambda exc=exc: save_failed(exc))
                    return
                self.root.after(0, save_finished)

            def save_finished() -> None:
                self.saving_result = False
                self.last_run = {}
                self.result_saved = True
                self._refresh_grid_stats()
                self.vars["status"].set(f"Saved result: {result_path}")
                self._append_log(f"[GUI] Saved result metadata to {result_path}")
                close_dialog()

            def save_failed(exc: Exception) -> None:
                self.saving_result = False
                for button in (save_button, later_button, discard_button):
                    button.configure(state=tk.NORMAL)
                self.vars["status"].set("Save failed. Check the error and try again.")
                self._alert("Cannot save result", str(exc))

            threading.Thread(target=worker, daemon=True).start()

        def keep_for_later() -> None:
            self.vars["status"].set("Result kept as pending. Use Review / Save Result before the next run.")
            close_dialog()

        def discard_and_close() -> None:
            if self.discard_result(confirm=True):
                close_dialog()

        save_button = ttk.Button(buttons, text="Save", command=save_and_close)
        save_button.pack(side=tk.LEFT)
        later_button = ttk.Button(buttons, text="Later (Keep Data)", command=keep_for_later)
        later_button.pack(side=tk.LEFT, padx=(8, 0))
        discard_button = ttk.Button(
            buttons,
            text="Discard + Delete Data",
            command=discard_and_close,
        )
        discard_button.pack(side=tk.LEFT, padx=(8, 0))
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)

        def activate_dialog(attempt: int = 0) -> None:
            try:
                if not dialog.winfo_exists():
                    return
                dialog.lift()
                if not dialog.winfo_viewable():
                    if attempt < 50:
                        dialog.after(20, activate_dialog, attempt + 1)
                    return
                dialog.grab_set()
                dialog.focus_set()
            except tk.TclError:
                if attempt < 50:
                    self.root.after(20, activate_dialog, attempt + 1)

        dialog.update_idletasks()
        x = self.root.winfo_rootx() + max((self.root.winfo_width() - dialog.winfo_reqwidth()) // 2, 0)
        y = self.root.winfo_rooty() + max((self.root.winfo_height() - dialog.winfo_reqheight()) // 2, 0)
        dialog.geometry(f"+{x}+{y}")
        dialog.after_idle(activate_dialog)

    def _reset_result_fields(self) -> None:
        self.vars["result"].set("failure")
        self.vars["object_exposed"].set("no")
        self.vars["object_pushed_out"].set("no")
        self.vars["object_push_success"].set("no")
        self.vars["result_tags"].set("")

    def _result_metrics(self) -> dict[str, Any]:
        tags = self._parse_result_tags(self.vars["result_tags"].get())
        return {
            "object_exposed": self.vars["object_exposed"].get() == "yes",
            "object_separated_from_occlusion": self.vars["object_pushed_out"].get() == "yes",
            "object_pushed_out_of_occlusion": self.vars["object_pushed_out"].get() == "yes",
            "object_push_success": self.vars["object_push_success"].get() == "yes",
            "tags": tags,
            "tags_text": ", ".join(tags),
        }

    def save_result(self) -> bool:
        try:
            record, result_path = self._prepare_result_save()
            self._commit_result_save(record, result_path)
        except ValueError as exc:
            self._alert("Cannot save result", str(exc))
            return False
        except Exception as exc:
            self._alert("Cannot save result", str(exc))
            return False
        self.last_run = {}
        self.result_saved = True
        self._refresh_grid_stats()
        self.vars["status"].set(f"Saved result: {result_path}")
        self._append_log(f"[GUI] Saved result metadata to {result_path}")
        return True

    def _prepare_result_save(self) -> tuple[dict[str, Any], Path]:
        if not self.last_run:
            raise ValueError("No finished run to save yet.")
        try:
            metrics = self._result_metrics()
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        policy_type = str(self.last_run.get("policy_type") or self.vars["policy_type"].get())
        policy_variant_value = self.last_run.get("policy_variant") or self.last_run.get("mask_act_experiment")
        policy_variant = str(policy_variant_value) if policy_variant_value else None
        save_root = Path(
            str(self.last_run.get("save_root") or self.vars["dataset_root"].get())
        ).expanduser()
        default_save_parent = save_root / policy_type
        if policy_type == "mask_act" and policy_variant:
            default_save_parent /= policy_variant
        save_parent = Path(str(self.last_run.get("save_parent") or default_save_parent)).expanduser()
        save_parent.mkdir(parents=True, exist_ok=True)
        grid = {
            "grid_size": self.last_run.get("grid_size"),
            "grid_rows": self.last_run.get("grid_rows"),
            "grid_cols": self.last_run.get("grid_cols"),
            "grid_cell": self.last_run.get("grid_cell"),
        }
        if grid["grid_rows"] is None or grid["grid_cols"] is None:
            grid["grid_rows"] = grid["grid_size"]
            grid["grid_cols"] = grid["grid_size"]
        grid_cell = str(self.last_run["grid_cell"])
        task = str(self.last_run["task"])
        prior_records = self._result_records(save_root, policy_type, policy_variant)
        prior_count, _prior_successes = self._grid_stats_from_records(
            prior_records,
            task=task,
            grid=grid,
            grid_cell=grid_cell,
        )
        record = {
            **self.last_run,
            "result": self.vars["result"].get(),
            **metrics,
            "trial_index": prior_count + 1,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        result_path = save_parent / RESULTS_FILENAME
        return record, result_path

    def _commit_result_save(self, record: dict[str, Any], result_path: Path) -> None:
        self._encode_confirmed_run_videos(record)
        with result_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        dataset_root = record.get("dataset_root")
        if dataset_root:
            run_root = Path(str(dataset_root)).expanduser()
            if run_root.exists():
                (run_root / "evaluation_result.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

    def _encode_confirmed_run_videos(self, record: dict[str, Any]) -> None:
        if not record.get("save_video"):
            return
        dataset_root = record.get("dataset_root")
        repo_id = record.get("dataset_repo_id")
        if not dataset_root or not repo_id:
            raise ValueError("The pending result does not contain a valid dataset root/repo id.")
        run_root = Path(str(dataset_root)).expanduser()
        if not run_root.exists():
            raise ValueError(f"Recorded-data path does not exist: {run_root}")

        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, load_episodes

        import pandas as pd

        encoder = LeRobotDataset.__new__(LeRobotDataset)
        encoder.repo_id = str(repo_id)
        encoder.root = run_root
        encoder.meta = LeRobotDatasetMetadata(str(repo_id), run_root)
        encoder.revision = None
        encoder.image_writer = None
        encoder.batch_encoding_size = 1_000_000
        encoder.episodes_since_last_encoding = 0
        encoder.vcodec = "h264"
        encoder.episodes = None
        encoder.hf_dataset = None
        encoder.writer = None
        encoder.latest_episode = None
        encoder._current_file_start_frame = None
        encoder._lazy_loading = True
        encoder._recorded_frames = encoder.meta.total_frames
        encoder._writer_closed_for_reading = False

        video_keys = list(encoder.meta.video_keys)
        if not video_keys:
            return
        if encoder.meta.total_episodes != 1:
            raise ValueError(
                f"Expected exactly one pending episode for confirmation-time video encoding; "
                f"found {encoder.meta.total_episodes} in {run_root}."
            )

        episode_index = 0
        episode = encoder.meta.episodes[episode_index]
        if all(
            f"videos/{video_key}/chunk_index" in episode
            and (run_root / encoder.meta.get_video_file_path(episode_index, video_key)).is_file()
            for video_key in video_keys
        ):
            self._append_log("[RECORD] MP4 videos already exist for this run; skipping encoding.")
            return

        self._append_log(f"[RECORD] Encoding confirmed MP4 videos for {', '.join(video_keys)}.")
        chunk_idx = episode["data/chunk_index"]
        file_idx = episode["data/file_index"]
        episode_df_path = run_root / DEFAULT_EPISODES_PATH.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        episode_df = pd.read_parquet(episode_df_path)
        video_metadata: dict[str, Any] = {}
        original_episodes = encoder.meta.episodes
        try:
            # This run directory contains one pending episode. Temporarily hiding existing
            # episode metadata makes LeRobot initialize each camera video from file-000.
            encoder.meta.episodes = []
            for video_key in video_keys:
                video_metadata.update(encoder._save_episode_video(video_key, episode_index))
        finally:
            encoder.meta.episodes = original_episodes
        video_metadata.pop("episode_index", None)
        video_df = pd.DataFrame(video_metadata, index=[episode_index]).convert_dtypes(
            dtype_backend="pyarrow"
        )
        episode_df = episode_df.combine_first(video_df)
        episode_df.to_parquet(episode_df_path)
        encoder.meta.episodes = load_episodes(run_root)
        encoder.finalize()
        self._append_log("[RECORD] Confirmed MP4 video encoding finished.")

    def discard_result(self, confirm: bool = True) -> bool:
        if not self.last_run:
            self.vars["status"].set("No pending result to discard.")
            return False
        if confirm and not self._confirm(
            "Discard result?",
            "Discard this result and permanently delete its trajectory, images, and video files?",
            confirm_text="Delete Data",
            destructive=True,
        ):
            return False
        try:
            deleted_path = self._delete_pending_run_data()
        except (OSError, ValueError) as exc:
            self._alert("Cannot discard result", str(exc))
            return False
        self.last_run = {}
        self.result_saved = True
        self._reset_result_fields()
        self.vars["status"].set("Pending result and recorded data deleted.")
        self._append_log(f"[GUI] Discarded result and deleted recorded data: {deleted_path}")
        return True

    def _delete_pending_run_data(self) -> Path:
        dataset_root = self.last_run.get("dataset_root")
        save_parent = self.last_run.get("save_parent")
        if not dataset_root or not save_parent:
            raise ValueError("The pending result does not contain a valid recorded-data path.")

        run_root = Path(str(dataset_root)).expanduser().resolve()
        parent_root = Path(str(save_parent)).expanduser().resolve()
        if run_root.parent != parent_root:
            raise ValueError(
                f"Refusing to delete data outside the configured save directory: {run_root}"
            )
        if not run_root.exists():
            return run_root
        if not run_root.is_dir():
            raise ValueError(f"Recorded-data path is not a directory: {run_root}")
        if not ((run_root / "evaluation_metadata.json").is_file() or (run_root / "meta" / "info.json").is_file()):
            raise ValueError(f"Directory does not look like an evaluation run; refusing to delete: {run_root}")

        shutil.rmtree(run_root)
        if self.active_dataset_root is not None and self.active_dataset_root.resolve() == run_root:
            self.active_dataset_root = None
        return run_root

    def _append_log(self, text: str) -> None:
        if threading.current_thread() is not threading.main_thread():
            self.log_queue.put(text)
            return
        print(text, flush=True)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def close(self) -> None:
        if self.last_run and not self.result_saved:
            if not self._confirm(
                "Unsaved result",
                "A result kept with Later is still unsaved. Discard it and permanently delete its recorded data before closing?",
                confirm_text="Discard + Close",
                destructive=True,
            ):
                self._show_result_dialog()
                return
            self.discard_result(confirm=False)
        if self.camera_preview_stop is not None:
            self.camera_preview_stop.set()
        if self.camera_preview_dialog is not None and self.camera_preview_dialog.winfo_exists():
            self.camera_preview_dialog.destroy()
        self.camera_preview_dialog = None
        self.camera_preview_stop = None
        if self.teleop_running:
            self.stop_teleop_requested = True
            self.vars["status"].set("Stopping teleop before closing...")
            self.root.after(500, self.close)
            return
        if self.reset_running:
            self.vars["status"].set("Waiting for reset before closing...")
            self.root.after(500, self.close)
            return
        if self.eval_running:
            self.stop_policy_requested = True
            self.vars["status"].set("Stopping policy before closing...")
            self.root.after(500, self.close)
            return
        if self.process is not None and self.process.poll() is None:
            if not self._confirm(
                "Stop active run?",
                "Policy is still running. Stop it and close?",
                confirm_text="Stop + Close",
            ):
                return
            self.stop_eval()
            self.root.after(500, self.close)
            return
        if self.connected and self.robot is not None:
            self.vars["status"].set("Disconnecting before close...")
            self.root.update_idletasks()
            robot = self.robot
            self._safe_disconnect_robot(robot)
            self.robot = None
            self.connected = False
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    EvalPolicyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
