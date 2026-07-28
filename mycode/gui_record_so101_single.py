#!/usr/bin/env python

"""Visual single-arm SO101 teleoperation recorder.

This is the single-arm companion to ``gui_record_so101_bimanual.py``.  It
reuses that script's dataset, camera and raw-depth implementation while
exposing only one follower and one leader in the GUI.

Run from the LeRobot environment:
    conda run -n lerobot python mycode/gui_record_so101_single.py
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

CURRENT_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(CURRENT_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(CURRENT_REPO_SRC))

try:
    from mycode.gui_record_so101_bimanual import (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        DEFAULT_CALIBRATION_PATH,
        DEFAULT_DATASET_ROOT,
        FPS,
        PREVIEW_SIZE,
        BimanualRecorder,
        FlexibleRealSenseCamera,
        RealSenseDepthView,
        RecorderSettings,
    )
except ModuleNotFoundError:
    from gui_record_so101_bimanual import (
        CAMERA_HEIGHT,
        CAMERA_WIDTH,
        DEFAULT_CALIBRATION_PATH,
        DEFAULT_DATASET_ROOT,
        FPS,
        PREVIEW_SIZE,
        BimanualRecorder,
        FlexibleRealSenseCamera,
        RealSenseDepthView,
        RecorderSettings,
    )

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig
from lerobot.utils.utils import init_logging

try:
    from mycode.calibrate_so101_single import install_sts3215_single_turn_fix
except ModuleNotFoundError:
    from calibrate_so101_single import install_sts3215_single_turn_fix

install_sts3215_single_turn_fix()


class SafeSO101Follower(SO101Follower):
    """SO101 follower that always closes its serial port after a failed handshake."""

    def disconnect(self) -> None:
        try:
            if self.bus.is_connected:
                self.bus.disconnect(self.config.disable_torque_on_disconnect)
        except Exception as exc:
            print(f"[GUI recorder] Warning: follower disconnect failed: {exc}", flush=True)
        finally:
            # FeetechMotorsBus.disconnect can fail while disabling torque if
            # the controller is visible but the motors have no power. Its
            # normal implementation then never reaches closePort().
            if self.bus.is_connected:
                self.bus.port_handler.closePort()
            for camera in self.cameras.values():
                try:
                    if camera.is_connected:
                        camera.disconnect()
                except Exception:
                    pass


class SafeSO101Leader(SO101Leader):
    """SO101 leader with guaranteed serial cleanup on connection failures."""

    def disconnect(self) -> None:
        try:
            if self.bus.is_connected:
                self.bus.disconnect()
        except Exception as exc:
            print(f"[GUI recorder] Warning: leader disconnect failed: {exc}", flush=True)
        finally:
            if self.bus.is_connected:
                self.bus.port_handler.closePort()


class SingleArmRecorder(BimanualRecorder):
    """Single-arm hardware adapter for the shared recorder engine."""

    def _make_robot(self) -> SO101Follower:
        cameras: dict[str, Any] = {}
        serial = self._resolve_realsense_serial() if self.settings.record_realsense else ""
        if self.settings.enable_realsense and not self.settings.record_realsense:
            self._put_status("RealSense enabled but not selected for recording; skipping it.")

        if serial:
            cameras["front"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )

        front_path = self.settings.opencv_front.strip() if self.settings.record_opencv_front else ""
        if front_path and not serial:
            cameras["front"] = OpenCVCameraConfig(
                index_or_path=front_path,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                fourcc="MJPG",
            )
        elif front_path and serial:
            self._put_status(f"RealSense is the front camera; ignoring OpenCV front {front_path}.")

        side_path = (
            self._resolve_opencv_side_path(self.settings.opencv_side.strip())
            if self.settings.record_opencv_side
            else ""
        )
        if side_path:
            cameras["side"] = OpenCVCameraConfig(
                index_or_path=side_path,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                fourcc="MJPG",
            )

        robot_calibration_dir, _ = self._calibration_dirs()
        robot = SafeSO101Follower(
            SOFollowerRobotConfig(
                id=self.settings.left_follower_id,
                calibration_dir=robot_calibration_dir,
                port=self.settings.left_follower_port,
                disable_torque_on_disconnect=False,
                cameras=cameras,
            )
        )

        # The SDK wrapper accepts the camera's default stream profile, which is
        # more tolerant of RealSense models that do not expose 1280x720@30.
        if serial and "front" in robot.cameras:
            robot.cameras["front"] = FlexibleRealSenseCamera(serial, CAMERA_WIDTH, CAMERA_HEIGHT)
            self._put_status("Using flexible RealSense default-profile wrapper.")

        if "front" in robot.cameras and isinstance(
            robot.cameras["front"], (RealSenseCamera, FlexibleRealSenseCamera)
        ):
            rs_camera = robot.cameras["front"]
            robot.cameras["front_depth"] = RealSenseDepthView(rs_camera)
            robot.config.cameras["front_depth"] = RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                color_mode=ColorMode.RGB,
                use_depth=True,
            )
        return robot

    def _make_teleop(self) -> SO101Leader:
        _, teleop_calibration_dir = self._calibration_dirs()
        return SafeSO101Leader(
            SOLeaderTeleopConfig(
                id=self.settings.left_leader_id,
                calibration_dir=teleop_calibration_dir,
                port=self.settings.left_leader_port,
            )
        )

    def _store_raw_depth_frame(self, robot: SO101Follower) -> None:
        rs_camera = robot.cameras.get("front")
        if not isinstance(rs_camera, (RealSenseCamera, FlexibleRealSenseCamera)):
            return
        with rs_camera.frame_lock:
            depth = rs_camera.latest_depth_frame
        if depth is not None:
            self.raw_depth_frames.append(np.asarray(depth, dtype=np.uint16).copy())
            if not self.raw_depth_metadata:
                self.raw_depth_metadata = self._get_realsense_depth_metadata(rs_camera)

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
        np.savez_compressed(
            depth_path,
            depth_mm=np.stack(raw_depth_frames, axis=0),
            fps=np.array(FPS, dtype=np.int32),
            episode_index=np.array(episode_index, dtype=np.int64),
            **raw_depth_metadata,
        )
        self._put_status(f"Saved raw depth sidecar: {depth_path}")


class SingleArmRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LeRobot SO101 Single-Arm Recorder")
        self.root.geometry("1280x900")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))

        self.status_queue: queue.Queue[str] = queue.Queue()
        self.preview_queue: queue.Queue[dict[str, np.ndarray]] = queue.Queue(maxsize=1)
        self.state_queue: queue.Queue[dict[str, float]] = queue.Queue(maxsize=1)
        self.recorder: SingleArmRecorder | None = None
        self.latest_frames: dict[str, np.ndarray] = {}
        self.current_image: ImageTk.PhotoImage | None = None
        self.saving = False

        self.vars = {
            "follower_port": tk.StringVar(value="/dev/ttyACM0"),
            "leader_port": tk.StringVar(value="/dev/ttyACM1"),
            "follower_id": tk.StringVar(value="my_awesome_follower_arm"),
            "leader_id": tk.StringVar(value="my_awesome_leader_arm"),
            "opencv_front": tk.StringVar(value="/dev/video4"),
            "opencv_side": tk.StringVar(value=""),
            "realsense_serial": tk.StringVar(value=""),
            "repo_id": tk.StringVar(value="local/so101_single"),
            "dataset_root": tk.StringVar(value=str(DEFAULT_DATASET_ROOT / "so101_single")),
            "calibration_dir": tk.StringVar(value=str(DEFAULT_CALIBRATION_PATH / "robots")),
            "task": tk.StringVar(value="Pick and place the object"),
            "preview_key": tk.StringVar(value=""),
        }
        self.resume = tk.BooleanVar(value=False)
        self.push_to_hub = tk.BooleanVar(value=False)
        self.enable_realsense = tk.BooleanVar(value=False)
        self.record_realsense = tk.BooleanVar(value=False)
        self.record_front = tk.BooleanVar(value=True)
        self.record_side = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Set one follower and one leader, then click Connect.")
        self.episode = tk.StringVar(value="Episode: not connected")
        self.state_text = tk.StringVar(value="")

        self._configure_style()
        self._build_ui()
        self.root.after(30, self._poll_queues)

    def _configure_style(self) -> None:
        tkfont.nametofont("TkDefaultFont").configure(size=11)
        tkfont.nametofont("TkTextFont").configure(size=11)
        style = ttk.Style(self.root)
        style.configure("TButton", padding=(12, 7))
        style.configure("TEntry", padding=(4, 4))

    def _build_ui(self) -> None:
        controls = ttk.Frame(self.root, padding=12)
        controls.pack(fill=tk.X)
        grid = ttk.Frame(controls)
        grid.pack(fill=tk.X)
        fields = [
            ("Follower port", "follower_port"),
            ("Leader port", "leader_port"),
            ("Follower id", "follower_id"),
            ("Leader id", "leader_id"),
            ("OpenCV front", "opencv_front"),
            ("OpenCV side", "opencv_side"),
            ("RealSense serial", "realsense_serial"),
            ("Dataset id", "repo_id"),
            ("Dataset folder", "dataset_root"),
            ("Calibration path", "calibration_dir"),
            ("Task", "task"),
        ]
        for index, (label, key) in enumerate(fields):
            row, half = divmod(index, 2)
            column = half * 2
            ttk.Label(grid, text=label).grid(row=row, column=column, sticky=tk.W, padx=(0, 6), pady=4)
            if key == "dataset_root":
                box = ttk.Frame(grid)
                box.grid(row=row, column=column + 1, sticky=tk.EW, padx=(0, 12), pady=4)
                box.columnconfigure(0, weight=1)
                ttk.Entry(box, textvariable=self.vars[key]).grid(row=0, column=0, sticky=tk.EW)
                ttk.Button(box, text="Browse", command=self.browse_dataset).grid(row=0, column=1, padx=(6, 0))
            else:
                ttk.Entry(grid, textvariable=self.vars[key]).grid(
                    row=row, column=column + 1, sticky=tk.EW, padx=(0, 12), pady=4
                )
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        options = ttk.Frame(controls)
        options.pack(fill=tk.X, pady=(8, 0))
        for text, variable in (
            ("Resume", self.resume),
            ("Push to Hub on close", self.push_to_hub),
            ("Enable RealSense", self.enable_realsense),
            ("Record RealSense", self.record_realsense),
            ("Record front", self.record_front),
            ("Record side", self.record_side),
        ):
            ttk.Checkbutton(options, text=text, variable=variable).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(options, text="Preview").pack(side=tk.LEFT, padx=(12, 5))
        self.preview_combo = ttk.Combobox(
            options, textvariable=self.vars["preview_key"], state="readonly", width=26
        )
        self.preview_combo.pack(side=tk.LEFT)

        buttons = ttk.Frame(controls)
        buttons.pack(fill=tk.X, pady=(10, 0))
        for text, command in (
            ("Camera candidates", self.preview_cameras),
            ("Calibrate follower", lambda: self.calibrate("robot")),
            ("Calibrate leader", lambda: self.calibrate("teleop")),
            ("Connect", self.connect),
        ):
            ttk.Button(buttons, text=text, command=command).pack(side=tk.LEFT, padx=(0, 8))
        self.record_button = ttk.Button(buttons, text="Record", command=self.record)
        self.record_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="End", command=self.end_episode).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Discard", command=self.discard_episode).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Stop", command=self.stop_recorder).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(buttons, text="Fullscreen", command=self.toggle_fullscreen).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(self.root, textvariable=self.status, anchor=tk.W, padding=(12, 5)).pack(fill=tk.X)
        ttk.Label(self.root, textvariable=self.episode, anchor=tk.W, padding=(12, 3)).pack(fill=tk.X)
        ttk.Label(self.root, textvariable=self.state_text, anchor=tk.W, padding=(12, 3)).pack(fill=tk.X)
        self.preview_label = ttk.Label(self.root, anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))

    def browse_dataset(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.vars["dataset_root"].get() or str(DEFAULT_DATASET_ROOT))
        if selected:
            self.vars["dataset_root"].set(selected)

    def preview_cameras(self) -> None:
        """Scan, preview and assign OpenCV camera candidates before connecting."""
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Stop the recorder before opening the standalone camera preview.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("OpenCV Camera Candidates and Preview")
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
        camera_var = tk.StringVar(value=self.vars["opencv_front"].get() or "/dev/video0")
        zoom_var = tk.DoubleVar(value=1.0)
        status_var = tk.StringVar(value="Scan cameras, select one, then preview or assign it.")

        content = ttk.Frame(dialog, padding=12)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(2, weight=1)

        selector = ttk.Frame(content)
        selector.grid(row=0, column=0, sticky=tk.EW)
        selector.columnconfigure(1, weight=1)
        ttk.Label(selector, text="Camera candidate").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        camera_combo = ttk.Combobox(selector, textvariable=camera_var, width=48)
        camera_combo.grid(row=0, column=1, sticky=tk.EW)

        image_label = ttk.Label(content, anchor=tk.CENTER)
        image_label.grid(row=2, column=0, sticky=tk.NSEW, pady=(10, 8))
        ttk.Label(content, textvariable=status_var, anchor=tk.W).grid(row=3, column=0, sticky=tk.EW)

        def describe(path: str) -> str:
            description = SingleArmRecorder._describe_video_path(path)
            return f"{path} ({description})" if description else path

        def scan() -> None:
            candidates: list[str] = []
            errors: list[str] = []
            try:
                cameras = OpenCVCamera.find_cameras()
                candidates.extend(str(camera["id"]) for camera in cameras if camera.get("id") is not None)
            except Exception as exc:
                errors.append(str(exc))

            # Include video nodes even when probing one of their formats failed.
            candidates.extend(str(path) for path in sorted(Path("/dev").glob("video*")))
            candidates = sorted(dict.fromkeys(candidates))
            camera_combo["values"] = candidates
            if candidates and camera_var.get() not in candidates:
                camera_var.set(candidates[0])
            if candidates:
                status_var.set(f"Found {len(candidates)} OpenCV candidate(s). Select one to preview.")
            elif errors:
                status_var.set(f"Camera scan failed: {errors[0]}. A device path can still be entered manually.")
            else:
                status_var.set("No OpenCV camera found. A path such as /dev/video0 can be entered manually.")

        def close_capture() -> None:
            capture = state.get("capture")
            if capture is not None:
                capture.release()
                state["capture"] = None

        def start_preview() -> None:
            close_capture()
            path = camera_var.get().strip()
            if not path:
                status_var.set("Select or enter a camera path first.")
                return
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                capture.release()
                status_var.set(f"Failed to open {describe(path)}.")
                return
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            capture.set(cv2.CAP_PROP_FPS, FPS)
            state["capture"] = capture
            state["last_frame_t"] = 0.0
            state["fps"] = 0.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            status_var.set(f"Previewing {describe(path)} | {width}x{height} | camera fps {fps:.1f}")

        def assign_front() -> None:
            path = camera_var.get().strip()
            if not path:
                status_var.set("Select a camera before assigning it.")
                return
            self.vars["opencv_front"].set(path)
            self.record_front.set(True)
            # The front key can contain either OpenCV or RealSense, not both.
            self.enable_realsense.set(False)
            self.record_realsense.set(False)
            status_var.set(f"Assigned {describe(path)} as front and enabled front recording.")

        def assign_side() -> None:
            path = camera_var.get().strip()
            if not path:
                status_var.set("Select a camera before assigning it.")
                return
            self.vars["opencv_side"].set(path)
            self.record_side.set(True)
            status_var.set(f"Assigned {describe(path)} as side and enabled side recording.")

        def adjust_zoom(factor: float) -> None:
            zoom_var.set(min(4.0, max(0.25, zoom_var.get() * factor)))

        def update_frame() -> None:
            if not state["running"] or not dialog.winfo_exists():
                return
            capture = state.get("capture")
            if capture is not None:
                ok, frame_bgr = capture.read()
                if ok:
                    now = time.monotonic()
                    previous_t = float(state.get("last_frame_t") or 0.0)
                    if previous_t > 0 and now > previous_t:
                        state["fps"] = 1.0 / (now - previous_t)
                    state["last_frame_t"] = now
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(np.ascontiguousarray(frame_rgb))
                    zoom = float(zoom_var.get())
                    image.thumbnail(
                        (max(1, int(PREVIEW_SIZE[0] * zoom)), max(1, int(PREVIEW_SIZE[1] * zoom))),
                        Image.Resampling.LANCZOS,
                    )
                    state["image"] = ImageTk.PhotoImage(image=image)
                    image_label.configure(image=state["image"])
                    status_var.set(
                        f"Previewing {describe(camera_var.get().strip())} | "
                        f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]} | {float(state['fps']):.1f} fps | "
                        f"zoom {zoom:.2f}x"
                    )
                else:
                    status_var.set(f"No frame received from {describe(camera_var.get().strip())}.")
            dialog.after(33, update_frame)

        def on_close() -> None:
            state["running"] = False
            close_capture()
            dialog.destroy()

        actions = ttk.Frame(content)
        actions.grid(row=1, column=0, sticky=tk.EW, pady=(10, 0))
        ttk.Button(actions, text="Scan", command=scan).pack(side=tk.LEFT)
        ttk.Button(actions, text="Start preview", command=start_preview).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Use as front", command=assign_front).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Use as side", command=assign_side).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(actions, text="Zoom").pack(side=tk.LEFT, padx=(18, 6))
        ttk.Button(actions, text="-", width=3, command=lambda: adjust_zoom(0.8)).pack(side=tk.LEFT)
        ttk.Button(actions, text="+", width=3, command=lambda: adjust_zoom(1.25)).pack(
            side=tk.LEFT, padx=(4, 0)
        )
        ttk.Button(actions, text="Reset", command=lambda: zoom_var.set(1.0)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Close", command=on_close).pack(side=tk.RIGHT)

        camera_combo.bind("<<ComboboxSelected>>", lambda _event: start_preview())
        dialog.protocol("WM_DELETE_WINDOW", on_close)
        scan()
        start_preview()
        update_frame()

    def _settings(self) -> RecorderSettings:
        # The shared settings object still contains right-arm slots.  They are
        # deliberately blank and are never read by SingleArmRecorder.
        return RecorderSettings(
            left_follower_port=self.vars["follower_port"].get().strip(),
            right_follower_port="",
            left_leader_port=self.vars["leader_port"].get().strip(),
            right_leader_port="",
            left_follower_id=self.vars["follower_id"].get().strip(),
            right_follower_id="",
            left_leader_id=self.vars["leader_id"].get().strip(),
            right_leader_id="",
            opencv_front=self.vars["opencv_front"].get().strip(),
            opencv_side=self.vars["opencv_side"].get().strip(),
            record_opencv_front=self.record_front.get(),
            record_opencv_side=self.record_side.get(),
            record_realsense=self.record_realsense.get(),
            enable_realsense=self.enable_realsense.get(),
            realsense_serial=self.vars["realsense_serial"].get().strip(),
            repo_id=self.vars["repo_id"].get().strip(),
            dataset_root=self.vars["dataset_root"].get().strip(),
            calibration_dir=self.vars["calibration_dir"].get().strip(),
            task=self.vars["task"].get().strip(),
            resume=self.resume.get(),
            push_to_hub=self.push_to_hub.get(),
        )

    def connect(self) -> None:
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Recorder is already connecting or running.")
            return
        required = ("follower_port", "leader_port", "follower_id", "leader_id", "repo_id", "dataset_root", "task")
        missing = [key for key in required if not self.vars[key].get().strip()]
        if missing:
            self.status.set(f"Missing required setting(s): {', '.join(missing)}")
            return
        self.recorder = SingleArmRecorder(
            self._settings(), self.status_queue, self.preview_queue, self.state_queue
        )
        self.status.set("Connecting single follower, cameras and leader...")
        self.recorder.start()

    def record(self) -> None:
        if self.recorder is None or not self.recorder.ready:
            self.status.set("Connect first and wait for the Ready message.")
            return
        if self.saving:
            self.status.set("The previous episode is still saving.")
            return
        self.recorder.request_record(self.vars["task"].get().strip())

    def end_episode(self) -> None:
        if self.recorder is not None:
            self.recorder.request_end_episode()

    def discard_episode(self) -> None:
        if self.recorder is not None:
            self.recorder.request_discard_episode()

    def stop_recorder(self) -> None:
        if self.recorder is not None:
            self.status.set("Stopping recorder and finishing pending saves...")
            self.root.update_idletasks()
            self.recorder.stop()
            self.recorder = None
            self.saving = False
            self.record_button.configure(state=tk.NORMAL)

    def calibrate(self, kind: str) -> None:
        if self.recorder is not None and self.recorder.is_alive():
            self.status.set("Stop the recorder before calibration.")
            return
        is_robot = kind == "robot"
        port = self.vars["follower_port" if is_robot else "leader_port"].get().strip()
        device_id = self.vars["follower_id" if is_robot else "leader_id"].get().strip()
        if not port or not device_id:
            self.status.set("Set the corresponding port and id before calibration.")
            return
        if not messagebox.askyesno("Start calibration?", "Follow the interactive prompts in the terminal."):
            return
        prefix = "--robot" if is_robot else "--teleop"
        device_type = "so101_follower" if is_robot else "so101_leader"
        cmd = [
            sys.executable,
            "-m",
            "mycode.calibrate_so101_single",
            f"{prefix}.type={device_type}",
            f"{prefix}.port={port}",
            f"{prefix}.id={device_id}",
        ]
        calibration_dir = self.vars["calibration_dir"].get().strip()
        if calibration_dir:
            # The recorder resolves the conventional robots/teleoperators
            # subdirectories; calibration accepts the same root convention.
            temp = SingleArmRecorder(self._settings(), queue.Queue(), queue.Queue(), queue.Queue())
            robot_dir, teleop_dir = temp._calibration_dirs()
            selected_dir = robot_dir if is_robot else teleop_dir
            if selected_dir is not None:
                cmd.append(f"{prefix}.calibration_dir={selected_dir}")
        self.status.set("Calibration running; use the terminal for prompts.")
        threading.Thread(target=self._run_calibration, args=(cmd,), daemon=True).start()

    def _run_calibration(self, cmd: list[str]) -> None:
        try:
            result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False)
            message = "Calibration finished." if result.returncode == 0 else f"ERROR: calibration exited {result.returncode}."
        except Exception as exc:
            message = f"ERROR: calibration failed to start: {exc}"
        self.status_queue.put(message)

    def _poll_queues(self) -> None:
        while True:
            try:
                message = self.status_queue.get_nowait()
            except queue.Empty:
                break
            if message.startswith("EPISODE|"):
                _, index, recording = message.split("|", 2)
                self.episode.set(f"{'Recording' if recording == '1' else 'Next episode'}: {index}")
            elif message.startswith("SAVE_STATE|"):
                self.saving = message.endswith("|1")
                self.record_button.configure(state=tk.DISABLED if self.saving else tk.NORMAL)
            elif message.startswith("PROGRESS|"):
                self.status.set(message.split("|", 3)[-1])
            else:
                self.status.set(message)

        try:
            self.latest_frames = self.preview_queue.get_nowait()
            keys = sorted(self.latest_frames)
            self.preview_combo["values"] = keys
            if keys and self.vars["preview_key"].get() not in keys:
                self.vars["preview_key"].set(keys[0])
        except queue.Empty:
            pass

        key = self.vars["preview_key"].get()
        if key in self.latest_frames:
            image = Image.fromarray(np.ascontiguousarray(self.latest_frames[key]))
            image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
            self.current_image = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self.current_image)

        try:
            state = self.state_queue.get_nowait()
            self.state_text.set(" | ".join(f"{key}: {value:.1f}" for key, value in sorted(state.items())[:18]))
        except queue.Empty:
            pass

        if self.recorder is not None and not self.recorder.is_alive() and not self.recorder.ready:
            self.recorder = None
            self.saving = False
            self.record_button.configure(state=tk.NORMAL)
        self.root.after(30, self._poll_queues)

    def toggle_fullscreen(self) -> None:
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))

    def close(self) -> None:
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
    SingleArmRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
