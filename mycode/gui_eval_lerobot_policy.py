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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from tkinter import filedialog, messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "train"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "eval_runs"
DEFAULT_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration" / "robots" / "so_follower"
DEFAULT_TELEOP_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration" / "teleoperators" / "so_leader"
RESULTS_FILENAME = "eval_results.jsonl"

POLICY_TYPES = (
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
TASK_CHOICES = ("cube1", "cube2", "cube3")

DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30


@dataclass(frozen=True)
class CheckpointOption:
    label: str
    path: Path
    policy_type: str


class LocalBimanualSOLeader:
    def __init__(self, left_arm: Any, right_arm: Any) -> None:
        self.left_arm = left_arm
        self.right_arm = right_arm

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
        for label, arm in (("left leader", self.left_arm), ("right leader", self.right_arm)):
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"[GUI eval] Warning: disconnect failed for {label}: {exc}", flush=True)


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


class EvalPolicyApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LeRobot Policy Evaluation")
        self.root.geometry("1280x920")
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
        self.active_dataset_root: Path | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.preview_queue: queue.Queue[tuple[str, Any, dict[str, Any], float, str]] = queue.Queue(maxsize=1)
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
        self.current_image = None

        self.vars = {
            "policy_type": tk.StringVar(value="act"),
            "checkpoint": tk.StringVar(value=""),
            "checkpoint_path": tk.StringVar(value=""),
            "conda_env": tk.StringVar(value="lerobot"),
            "robot_mode": tk.StringVar(value="bimanual"),
            "robot_type": tk.StringVar(value="bi_so_follower"),
            "robot_port": tk.StringVar(value="/dev/ttyACM0"),
            "robot_id": tk.StringVar(value="eval_bimanual_follower"),
            "left_follower_port": tk.StringVar(value="/dev/ttyACM0"),
            "right_follower_port": tk.StringVar(value="/dev/ttyACM1"),
            "left_follower_id": tk.StringVar(value="my_awesome_follower_arm"),
            "right_follower_id": tk.StringVar(value="my_awesome_follower_arm_r"),
            "calibration_dir": tk.StringVar(value=str(DEFAULT_CALIBRATION_DIR)),
            "left_leader_port": tk.StringVar(value="/dev/ttyACM2"),
            "right_leader_port": tk.StringVar(value="/dev/ttyACM3"),
            "left_leader_id": tk.StringVar(value="my_awesome_leader_arm"),
            "right_leader_id": tk.StringVar(value="my_awesome_leader_arm_r"),
            "teleop_calibration_dir": tk.StringVar(value=str(DEFAULT_TELEOP_CALIBRATION_DIR)),
            "realsense_serial": tk.StringVar(value=""),
            "opencv_front": tk.StringVar(value=""),
            "opencv_side": tk.StringVar(value="/dev/video10"),
            "camera_config": tk.StringVar(value=""),
            "dataset_repo_id": tk.StringVar(value="seeed/eval_test"),
            "dataset_root": tk.StringVar(value=str(DEFAULT_EVAL_ROOT)),
            "task": tk.StringVar(value=TASK_CHOICES[0]),
            "episode_time_s": tk.StringVar(value="60"),
            "fps": tk.StringVar(value="30"),
            "reset_time_s": tk.StringVar(value="4.0"),
            "extra_args": tk.StringVar(
                value="--display_data=false --dataset.push_to_hub=false --dataset.num_episodes=1 --dataset.vcodec=h264"
            ),
            "status": tk.StringVar(value="建议先点击 Check Connect；通过后再点击 Start。"),
            "elapsed": tk.StringVar(value="Elapsed: 00:00"),
            "result": tk.StringVar(value=RESULT_CHOICES[0]),
        }
        self.enable_realsense = tk.BooleanVar(value=True)
        self.include_side_camera = tk.BooleanVar(value=False)
        self.auto_reset_after_policy = tk.BooleanVar(value=True)

        self._build_ui()
        self._refresh_camera_config_preview()
        for key in ("realsense_serial", "opencv_front", "opencv_side"):
            self.vars[key].trace_add("write", lambda *_args: self._refresh_camera_config_preview())
        self.enable_realsense.trace_add("write", lambda *_args: self._refresh_camera_config_preview())
        self.include_side_camera.trace_add("write", lambda *_args: self._refresh_camera_config_preview())
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

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=14)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        row = 0
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
            values=("bimanual", "single"),
            state="readonly",
            width=24,
        ).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Label(top, text="Robot type").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["robot_type"],
            values=("bi_so_follower", "so101_follower", "so100_follower"),
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
        ttk.Label(top, text="Left leader").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["left_leader_port"]).grid(row=row, column=1, sticky=tk.EW)
        ttk.Label(top, text="Right leader").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["right_leader_port"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Leader calib ids").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        leader_ids = ttk.Frame(top)
        leader_ids.grid(row=row, column=1, sticky=tk.EW)
        leader_ids.columnconfigure(0, weight=1)
        leader_ids.columnconfigure(1, weight=1)
        ttk.Entry(leader_ids, textvariable=self.vars["left_leader_id"]).grid(row=0, column=0, sticky=tk.EW)
        ttk.Entry(leader_ids, textvariable=self.vars["right_leader_id"]).grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))
        ttk.Label(top, text="Leader calib dir").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["teleop_calibration_dir"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        ttk.Checkbutton(top, text="Enable RealSense front", variable=self.enable_realsense).grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Entry(top, textvariable=self.vars["realsense_serial"]).grid(row=row, column=1, sticky=tk.EW)
        ttk.Button(top, text="Find Cameras", command=self.find_cameras).grid(row=row, column=2, sticky=tk.W, padx=(16, 8))
        ttk.Checkbutton(top, text="Include side", variable=self.include_side_camera).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Label(top, text="OpenCV front").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["opencv_front"]).grid(row=row, column=1, sticky=tk.EW)
        ttk.Label(top, text="OpenCV side").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["opencv_side"]).grid(row=row, column=3, sticky=tk.EW)

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
        ttk.Label(top, text="Task").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["task"],
            values=TASK_CHOICES,
            state="readonly",
        ).grid(row=row, column=1, columnspan=3, sticky=tk.EW)

        row += 1
        ttk.Label(top, text="Episode sec").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["episode_time_s"], width=12).grid(row=row, column=1, sticky=tk.W)
        ttk.Label(top, text="FPS").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["fps"], width=12).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Checkbutton(top, text="Auto reset after policy", variable=self.auto_reset_after_policy).grid(
            row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Label(top, text="Reset sec").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["reset_time_s"], width=12).grid(row=row, column=3, sticky=tk.W)

        row += 1
        ttk.Label(top, text="Extra args").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["extra_args"]).grid(row=row, column=1, columnspan=3, sticky=tk.EW)

        buttons = ttk.Frame(self.root, padding=(14, 0, 14, 10))
        buttons.pack(fill=tk.X)
        self.connect_button = ttk.Button(buttons, text="Check Connect", command=self.check_connect)
        self.connect_button.pack(side=tk.LEFT, ipady=4)
        self.start_button = ttk.Button(buttons, text="Start", command=self.start_eval)
        self.start_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.stop_button = ttk.Button(buttons, text="End / Stop", command=self.stop_eval, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.teleop_button = ttk.Button(buttons, text="Start Teleop", command=self.start_teleop, state=tk.DISABLED)
        self.teleop_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.stop_teleop_button = ttk.Button(
            buttons, text="Stop Teleop", command=self.stop_teleop, state=tk.DISABLED
        )
        self.stop_teleop_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
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

        result_frame = ttk.LabelFrame(self.root, text="Result", padding=12)
        result_frame.pack(fill=tk.X, padx=14, pady=(0, 10))
        ttk.Label(result_frame, text="Last run").pack(side=tk.LEFT)
        ttk.Combobox(
            result_frame,
            textvariable=self.vars["result"],
            values=RESULT_CHOICES,
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(result_frame, text="Save Result", command=self.save_result).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(result_frame, textvariable=self.vars["status"]).pack(side=tk.LEFT, padx=(16, 0))

        log_frame = ttk.Frame(self.root, padding=(14, 0, 14, 14))
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(1, weight=0)
        self.preview_label = ttk.Label(log_frame, anchor=tk.CENTER)
        self.preview_label.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))
        right_log = ttk.Frame(log_frame)
        right_log.grid(row=0, column=1, sticky=tk.NS)
        ttk.Label(right_log, text="Log / Action").pack(anchor=tk.W)
        self.log_text = tk.Text(right_log, wrap=tk.WORD, height=18)
        scrollbar = ttk.Scrollbar(right_log, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

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

        self.checkpoint_options = options
        values = [option.label for option in options]
        self.checkpoint_combo.configure(values=values)
        if values and not self.vars["checkpoint"].get():
            self.vars["checkpoint"].set(values[-1])
            self.on_checkpoint_selected()
        self._append_log(f"Found {len(options)} checkpoint(s) under {DEFAULT_OUTPUT_ROOT}")

    def _infer_policy_type(self, path: Path) -> str:
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
                return

    def browse_checkpoint(self) -> None:
        selected = filedialog.askdirectory(
            initialdir=str(DEFAULT_OUTPUT_ROOT),
            title="Select pretrained_model folder",
        )
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        self.vars["checkpoint_path"].set(str(path))
        self.vars["policy_type"].set(self._infer_policy_type(path))
        self.vars["checkpoint"].set(f"{self.vars['policy_type'].get()}: {path}")

    def browse_dataset_root(self) -> None:
        current = Path(self.vars["dataset_root"].get() or DEFAULT_EVAL_ROOT).expanduser()
        selected = filedialog.askdirectory(initialdir=str(current.parent), title="Select result save parent")
        if selected:
            self.vars["dataset_root"].set(str(Path(selected).expanduser()))

    def open_save_folder(self) -> None:
        path = Path(self.vars["dataset_root"].get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        messagebox.showinfo("Save folder", str(path), parent=self.root)

    def find_cameras(self) -> None:
        lines: list[str] = []
        try:
            from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

            cameras = RealSenseCamera.find_cameras()
            serials = [str(camera.get("id")) for camera in cameras]
            if serials:
                lines.append(f"RealSense serials: {serials}")
                if not self.vars["realsense_serial"].get().strip():
                    self.vars["realsense_serial"].set(serials[0])
            else:
                lines.append("RealSense serials: none")
        except Exception as exc:
            lines.append(f"RealSense scan failed: {exc}")

        self._append_log("[GUI] " + " | ".join(lines))
        self._refresh_camera_config_preview()

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

    def _build_camera_config(self) -> str:
        cameras: list[str] = []
        if self.enable_realsense.get():
            serial = self.vars["realsense_serial"].get().strip() or "<auto>"
            if serial != "<auto>":
                cameras.append(
                    "front: {"
                    f'type: intelrealsense, serial_number_or_name: "{serial}", '
                    f"width: {DEFAULT_CAMERA_WIDTH}, height: {DEFAULT_CAMERA_HEIGHT}, "
                    f"fps: {DEFAULT_CAMERA_FPS}, use_depth: true"
                    "}"
                )
            else:
                cameras.append("front: {type: intelrealsense, serial_number_or_name: <auto>}")
        elif self.vars["opencv_front"].get().strip():
            cameras.append(self._opencv_camera_entry("front", self.vars["opencv_front"].get().strip()))

        if self.include_side_camera.get() and self.vars["opencv_side"].get().strip():
            cameras.append(self._opencv_camera_entry("side", self.vars["opencv_side"].get().strip()))

        if not cameras:
            raise ValueError("at least one front camera is required")
        return "{ " + ", ".join(cameras) + " }"

    @staticmethod
    def _opencv_camera_entry(name: str, index_or_path: str) -> str:
        value = index_or_path if index_or_path.isdigit() else f'"{index_or_path}"'
        return (
            f"{name}: {{"
            f"type: opencv, index_or_path: {value}, "
            f"width: {DEFAULT_CAMERA_WIDTH}, height: {DEFAULT_CAMERA_HEIGHT}, fps: {DEFAULT_CAMERA_FPS}, fourcc: \"MJPG\""
            "}"
        )

    def _camera_config_for_command(self) -> str:
        if self.enable_realsense.get():
            serial = self._resolve_realsense_serial()
            entries = [
                "front: {"
                f'type: intelrealsense, serial_number_or_name: "{serial}", '
                f"width: {DEFAULT_CAMERA_WIDTH}, height: {DEFAULT_CAMERA_HEIGHT}, "
                f"fps: {DEFAULT_CAMERA_FPS}, use_depth: true"
                "}"
            ]
        else:
            front_path = self.vars["opencv_front"].get().strip()
            if not front_path:
                raise ValueError("RealSense disabled: OpenCV front cannot be empty.")
            entries = [self._opencv_camera_entry("front", front_path)]

        if self.include_side_camera.get() and self.vars["opencv_side"].get().strip():
            entries.append(self._opencv_camera_entry("side", self.vars["opencv_side"].get().strip()))

        return "{ " + ", ".join(entries) + " }"

    def check_connect(self) -> None:
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

    def start_eval(self) -> None:
        if self.eval_running:
            self.vars["status"].set("A policy run is already active.")
            return
        if self.teleop_running or self.reset_running:
            self.vars["status"].set("Stop teleop/reset before starting policy.")
            return
        if not self.connected:
            messagebox.showerror("Not connected", "Click Check Connect first; it will keep the robot connected.", parent=self.root)
            return

        try:
            self._validate_eval_settings()
        except ValueError as exc:
            messagebox.showerror("Cannot start", str(exc), parent=self.root)
            return

        self.started_at = time.monotonic()
        self.started_at_iso = datetime.now().isoformat(timespec="seconds")
        self.stop_requested = False
        self.stop_policy_requested = False
        self.last_run = {}
        self.connect_button.configure(state=tk.DISABLED)
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.disconnect_button.configure(state=tk.DISABLED)
        self.teleop_button.configure(state=tk.DISABLED)
        self.stop_teleop_button.configure(state=tk.DISABLED)
        self.capture_reset_button.configure(state=tk.DISABLED)
        self.reset_button.configure(state=tk.DISABLED)
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
            from lerobot.robots import make_robot_from_config

            cfg = self._make_robot_config()
            robot = make_robot_from_config(cfg)
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
        serial = self._resolve_realsense_serial()
        replacement = PersistentFlexibleRealSenseCamera(serial, DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT)
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
        from lerobot.datasets.video_utils import VideoEncodingManager
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.policies.utils import make_robot_action
        from lerobot.processor import make_default_processors
        from lerobot.processor.rename_processor import rename_stats
        from lerobot.utils.constants import ACTION, OBS_STR
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
        save_parent = self._save_parent()
        run_dataset_root = self._make_run_dataset_root(save_parent, repo_id)
        self.active_dataset_root = run_dataset_root

        policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
        policy_cfg.pretrained_path = checkpoint
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
        policy_inputs = sorted(policy_cfg.input_features)
        obs_features = sorted(key for key in dataset_features if key.startswith("observation."))
        self.log_queue.put(f"Raw robot observation features: {sorted(robot.observation_features)}")
        self.log_queue.put(f"Dataset observation features: {obs_features}")
        self.log_queue.put(f"Policy input features: {policy_inputs}")
        self.log_queue.put(f"Policy output features: {sorted(policy_cfg.output_features)}")

        dataset = None
        last_action_log_t = 0.0
        try:
            dataset = LeRobotDataset.create(
                repo_id,
                fps,
                root=run_dataset_root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=max(1, 4 * len(robot.cameras)),
                batch_encoding_size=1,
                vcodec="h264",
            )
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
            with self.robot_lock:
                self.initial_joint_action = self._action_from_observation(robot, robot.get_observation())
            self.log_queue.put(
                f"Captured policy-start reset pose with {len(self.initial_joint_action)} joint target(s)."
            )

            with VideoEncodingManager(dataset):
                start_t = time.perf_counter()
                timestamp = 0.0
                while timestamp < episode_time_s and not self.stop_policy_requested:
                    loop_t = time.perf_counter()
                    raw_obs = robot.get_observation()
                    obs_processed = robot_observation_processor(raw_obs)
                    observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                    action_values = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=task,
                        robot_type=robot.robot_type,
                    )
                    action_dict = make_robot_action(action_values, dataset.features)
                    robot_action_to_send = robot_action_processor((action_dict, raw_obs))
                    robot.send_action(robot_action_to_send)

                    action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
                    dataset.add_frame({**observation_frame, **action_frame, "task": task})

                    image_key, image = self._select_preview_image(raw_obs)
                    loop_s = max(time.perf_counter() - loop_t, 1e-6)
                    self._put_preview(image_key, image, action_dict, 1.0 / loop_s, ", ".join(sorted(observation_frame)))
                    now = time.perf_counter()
                    if now - last_action_log_t > 1.0:
                        last_action_log_t = now
                        compact_action = ", ".join(f"{k}={float(v):.2f}" for k, v in action_dict.items())
                        self.log_queue.put(f"[ACTION] {compact_action}")

                    dt_s = time.perf_counter() - loop_t
                    precise_sleep(max(1 / fps - dt_s, 0.0))
                    timestamp = time.perf_counter() - start_t

                dataset.save_episode()
                self.log_queue.put("Policy episode saved. Robot remains connected.")
        finally:
            if dataset is not None:
                dataset.finalize()
            if self.auto_reset_after_policy.get() and self.initial_joint_action and self.connected:
                self.log_queue.put("[RESET] Returning arms to captured policy-start pose.")
                self._move_robot_to_action(self.initial_joint_action, float(self.vars["reset_time_s"].get()))

    def _validate_eval_settings(self) -> None:
        checkpoint = self.vars["checkpoint_path"].get().strip()
        if not checkpoint:
            raise ValueError("请选择一个 pretrained_model 权重目录。")
        if not Path(checkpoint).expanduser().exists():
            raise ValueError(f"权重路径不存在: {checkpoint}")
        if not self.vars["dataset_repo_id"].get().strip():
            raise ValueError("Dataset repo 不能为空。")
        if not self.vars["task"].get().strip():
            raise ValueError("Task 不能为空。")
        float(self.vars["episode_time_s"].get())
        int(self.vars["fps"].get())
        reset_time_s = float(self.vars["reset_time_s"].get())
        if reset_time_s <= 0:
            raise ValueError("Reset sec 必须大于 0。")

    def _make_robot_config(self) -> Any:
        from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
        from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig, SOFollowerRobotConfig

        robot_mode = self.vars["robot_mode"].get()
        robot_type = self.vars["robot_type"].get().strip()
        cameras = self._camera_config_objects()
        if robot_mode == "bimanual":
            if robot_type != "bi_so_follower":
                raise ValueError("双臂测试请把 Robot type 设为 bi_so_follower。")
            return BiSOFollowerConfig(
                id=self.vars["robot_id"].get().strip(),
                calibration_dir=self._prepare_bimanual_calibration(),
                left_arm_config=SOFollowerConfig(
                    port=self.vars["left_follower_port"].get().strip(),
                    cameras=cameras,
                ),
                right_arm_config=SOFollowerConfig(
                    port=self.vars["right_follower_port"].get().strip(),
                ),
            )
        return SOFollowerRobotConfig(
            id=self.vars["left_follower_id"].get().strip(),
            calibration_dir=Path(self.vars["calibration_dir"].get()).expanduser(),
            port=self.vars["robot_port"].get().strip() or self.vars["left_follower_port"].get().strip(),
            cameras=cameras,
        )

    def _camera_config_objects(self) -> dict[str, Any]:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        cameras: dict[str, Any] = {}
        if self.enable_realsense.get():
            cameras["front"] = RealSenseCameraConfig(
                serial_number_or_name=self._resolve_realsense_serial(),
                width=DEFAULT_CAMERA_WIDTH,
                height=DEFAULT_CAMERA_HEIGHT,
                fps=DEFAULT_CAMERA_FPS,
                use_depth=True,
            )
        else:
            front_path = self.vars["opencv_front"].get().strip()
            if not front_path:
                raise ValueError("RealSense disabled: OpenCV front cannot be empty.")
            cameras["front"] = OpenCVCameraConfig(
                index_or_path=self._opencv_value(front_path),
                width=DEFAULT_CAMERA_WIDTH,
                height=DEFAULT_CAMERA_HEIGHT,
                fps=DEFAULT_CAMERA_FPS,
                fourcc="MJPG",
            )
        if self.include_side_camera.get() and self.vars["opencv_side"].get().strip():
            cameras["side"] = OpenCVCameraConfig(
                index_or_path=self._opencv_value(self.vars["opencv_side"].get().strip()),
                width=DEFAULT_CAMERA_WIDTH,
                height=DEFAULT_CAMERA_HEIGHT,
                fps=DEFAULT_CAMERA_FPS,
                fourcc="MJPG",
            )
        return cameras

    @staticmethod
    def _opencv_value(index_or_path: str) -> int | Path:
        return int(index_or_path) if index_or_path.isdigit() else Path(index_or_path)

    @staticmethod
    def _select_preview_image(raw_obs: dict[str, Any]) -> tuple[str, Any | None]:
        if "left_front" in raw_obs:
            return "left_front", raw_obs["left_front"]
        for key, value in raw_obs.items():
            if hasattr(value, "ndim") and value.ndim == 3:
                return key, value
        return "none", None

    def _put_preview(self, image_key: str, image: Any, action: dict[str, Any], fps_hz: float, input_keys: str) -> None:
        item = (image_key, image, action, fps_hz, input_keys)
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
        self.start_button.configure(state=tk.DISABLED)
        self.teleop_button.configure(state=tk.DISABLED)
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

    def start_teleop(self) -> None:
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
        from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

        if self.vars["robot_mode"].get() != "bimanual":
            raise RuntimeError("Teleop adjustment is currently configured for bimanual SO leader arms.")
        calibration_dir = Path(self.vars["teleop_calibration_dir"].get()).expanduser()
        return LocalBimanualSOLeader(
            SO101Leader(
                SOLeaderTeleopConfig(
                    id=self.vars["left_leader_id"].get().strip(),
                    calibration_dir=calibration_dir,
                    port=self.vars["left_leader_port"].get().strip(),
                )
            ),
            SO101Leader(
                SOLeaderTeleopConfig(
                    id=self.vars["right_leader_id"].get().strip(),
                    calibration_dir=calibration_dir,
                    port=self.vars["right_leader_port"].get().strip(),
                )
            ),
        )

    def _move_robot_to_action(self, target_action: dict[str, float], duration_s: float) -> None:
        from lerobot.utils.robot_utils import precise_sleep

        robot = self.robot
        if robot is None:
            raise RuntimeError("Robot is not connected.")
        fps = max(int(self.vars["fps"].get()), 1)
        duration_s = max(duration_s, 0.1)
        with self.robot_lock:
            start_obs = robot.get_observation()
            start_action = self._action_from_observation(robot, start_obs)
        keys = [key for key in robot.action_features if key in target_action and key in start_action]
        if not keys:
            raise RuntimeError("No shared action keys found for reset.")

        steps = max(int(duration_s * fps), 1)
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
        robot = self.robot
        self._safe_disconnect_robot(robot)
        with self.robot_lock:
            self.robot = None
            self.connected = False
            self.initial_joint_action = None
        self.log_queue.put("__DISCONNECT_DONE__|0|")

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
            raise ValueError("请选择一个 pretrained_model 权重目录。")
        if not Path(checkpoint).expanduser().exists():
            raise ValueError(f"权重路径不存在: {checkpoint}")

        repo_id = self.vars["dataset_repo_id"].get().strip()
        if not repo_id:
            raise ValueError("Dataset repo 不能为空，例如 seeed/eval_test123。")

        task = self.vars["task"].get().strip()
        if not task:
            raise ValueError("Task 不能为空。")

        try:
            float(self.vars["episode_time_s"].get())
            int(self.vars["fps"].get())
        except ValueError as exc:
            raise ValueError("Episode sec 必须是数字，FPS 必须是整数。") from exc

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
                raise ValueError("双臂测试请把 Robot type 设为 bi_so_follower。")
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
            raise ValueError("单臂测试请把 Robot type 设为 so101_follower 或 so100_follower。")
        return [
            f"--robot.type={robot_type}",
            f"--robot.port={self.vars['robot_port'].get().strip() or self.vars['left_follower_port'].get().strip()}",
            f"--robot.cameras={self._camera_config_for_command()}",
            f"--robot.id={self.vars['left_follower_id'].get().strip()}",
            f"--robot.calibration_dir={self.vars['calibration_dir'].get().strip()}",
        ]

    def _save_parent(self) -> Path:
        save_parent = Path(self.vars["dataset_root"].get()).expanduser()
        save_parent.mkdir(parents=True, exist_ok=True)
        return save_parent

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
            raise ValueError("Runtime id 不能为空。")
        if not left_id or not right_id:
            raise ValueError("Left calib id 和 Right calib id 不能为空。")

        source_dir = Path(self.vars["calibration_dir"].get()).expanduser()
        if not source_dir.exists():
            raise ValueError(f"标定目录不存在: {source_dir}")

        left_src = source_dir / f"{left_id}.json"
        right_src = source_dir / f"{right_id}.json"
        missing = [str(path) for path in (left_src, right_src) if not path.exists()]
        if missing:
            raise ValueError("找不到双臂标定文件:\n" + "\n".join(missing))

        save_parent = self._save_parent()
        runtime_dir = save_parent / "_runtime_calibration" / base_id
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
                self.teleop_button.configure(state=tk.NORMAL if ok else tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if ok else tk.DISABLED)
                self.reset_button.configure(state=tk.NORMAL if ok and self.initial_joint_action else tk.DISABLED)
                self.stop_teleop_button.configure(state=tk.DISABLED)
                self.stop_button.configure(state=tk.DISABLED)
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
                self.teleop_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.reset_button.configure(
                    state=tk.NORMAL if self.connected and self.initial_joint_action else tk.DISABLED
                )
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
                    "save_parent": self.vars["dataset_root"].get(),
                    "dataset_root": str(self.active_dataset_root) if self.active_dataset_root else None,
                    "task": self.vars["task"].get(),
                    "command": "in-process persistent eval",
                }
                self.vars["status"].set(
                    "Policy stopped/saved. Robot remains connected; use Disconnect when ready."
                    if ok
                    else "Policy failed. Robot remains connected; use Disconnect when ready."
                )
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
                self.vars["status"].set("Disconnected.")
            elif message.startswith("__TELEOP_DONE__|"):
                ok = message.split("|", 2)[1] == "0"
                self.teleop_running = False
                self.stop_teleop_requested = False
                self.start_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.teleop_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.stop_teleop_button.configure(state=tk.DISABLED)
                self.disconnect_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.reset_button.configure(
                    state=tk.NORMAL if self.connected and self.initial_joint_action else tk.DISABLED
                )
                self.vars["status"].set(
                    "Teleop stopped. Adjusted pose captured as reset pose."
                    if ok
                    else "Teleop failed/stopped with error."
                )
            elif message.startswith("__RESET_DONE__|"):
                ok = message.split("|", 2)[1] == "0"
                self.reset_running = False
                self.start_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.teleop_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.disconnect_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.capture_reset_button.configure(state=tk.NORMAL if self.connected else tk.DISABLED)
                self.reset_button.configure(
                    state=tk.NORMAL if self.connected and self.initial_joint_action else tk.DISABLED
                )
                self.vars["status"].set("Reset complete." if ok else "Reset failed.")
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
                image_key, image, action, fps_hz, input_keys = self.preview_queue.get_nowait()
            except queue.Empty:
                break
            self._update_preview(image_key, image, action, fps_hz, input_keys)

        if self.started_at is not None and (self.eval_running or self.process is not None):
            elapsed = int(time.monotonic() - self.started_at)
            self.vars["elapsed"].set(f"Elapsed: {elapsed // 60:02d}:{elapsed % 60:02d}")

        self.root.after(200, self._tick)

    def _update_preview(self, image_key: str, image: Any, action: dict[str, Any], fps_hz: float, input_keys: str) -> None:
        try:
            from PIL import Image, ImageTk
            import numpy as np

            if image is not None:
                arr = np.asarray(image)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    pil = Image.fromarray(arr.astype(np.uint8), mode="RGB")
                    pil.thumbnail((700, 520))
                    self.current_image = ImageTk.PhotoImage(pil)
                    self.preview_label.configure(image=self.current_image)
        except Exception as exc:
            self._append_log(f"[GUI] Preview update failed: {exc}")

        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, f"image_key: {image_key}\n")
        self.log_text.insert(tk.END, f"loop_hz: {fps_hz:.1f}\n")
        self.log_text.insert(tk.END, f"policy inputs: {input_keys}\n\n")
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
            "save_parent": self.vars["dataset_root"].get(),
            "dataset_root": str(self.active_dataset_root) if self.active_dataset_root else None,
            "task": self.vars["task"].get(),
            "command": shlex.join(self.last_command),
        }
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
        self._show_result_dialog()

    def _show_result_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Save evaluation result")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        ttk.Label(dialog, text="Result").grid(row=0, column=0, sticky=tk.W, padx=14, pady=(14, 8))
        ttk.Combobox(
            dialog,
            textvariable=self.vars["result"],
            values=RESULT_CHOICES,
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky=tk.EW, padx=(0, 14), pady=(14, 8))
        ttk.Label(dialog, text=self.vars["elapsed"].get()).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=14, pady=(0, 8)
        )
        buttons = ttk.Frame(dialog)
        buttons.grid(row=2, column=0, columnspan=2, sticky=tk.E, padx=14, pady=(4, 14))
        ttk.Button(buttons, text="Save", command=lambda: (self.save_result(), dialog.destroy())).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Later", command=dialog.destroy).pack(side=tk.LEFT, padx=(8, 0))

    def save_result(self) -> None:
        if not self.last_run:
            self.vars["status"].set("No finished run to save yet.")
            return
        save_parent = Path(str(self.last_run.get("save_parent") or self.vars["dataset_root"].get())).expanduser()
        save_parent.mkdir(parents=True, exist_ok=True)
        record = {
            **self.last_run,
            "result": self.vars["result"].get(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        result_path = save_parent / RESULTS_FILENAME
        with result_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.vars["status"].set(f"Saved result: {result_path}")
        self._append_log(f"[GUI] Saved result metadata to {result_path}")

    def _append_log(self, text: str) -> None:
        print(text, flush=True)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def close(self) -> None:
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
            if not messagebox.askyesno("Stop active run?", "Policy is still running. Stop it and close?", parent=self.root):
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
