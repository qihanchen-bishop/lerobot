#!/usr/bin/env python

"""Interactive camera preview for OpenCV webcams and Intel RealSense cameras."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


PREVIEW_WIDTH = 960
PREVIEW_HEIGHT = 540


@dataclass(frozen=True)
class CameraOption:
    label: str
    camera_type: str
    camera_id: str | int
    stream: str
    width: int | None = None
    height: int | None = None
    fps: int | None = None


def _profile_value(camera_info: dict[str, Any], key: str) -> Any:
    return camera_info.get("default_stream_profile", {}).get(key)


def find_camera_options() -> tuple[list[CameraOption], list[str]]:
    """Return all discoverable OpenCV and RealSense preview options."""
    options: list[CameraOption] = []
    warnings: list[str] = []

    try:
        for info in OpenCVCamera.find_cameras():
            width = _profile_value(info, "width")
            height = _profile_value(info, "height")
            fps = _profile_value(info, "fps")
            profile = f"{width}x{height}"
            if fps:
                profile += f" @ {fps:g} fps"
            options.append(
                CameraOption(
                    label=f"RGB | OpenCV | {info['id']} | {profile}",
                    camera_type="opencv",
                    camera_id=info["id"],
                    stream="rgb",
                    width=int(width) if width else None,
                    height=int(height) if height else None,
                    fps=int(round(fps)) if fps else None,
                )
            )
    except Exception as exc:
        warnings.append(f"OpenCV camera scan failed: {exc}")

    try:
        for info in RealSenseCamera.find_cameras():
            width = _profile_value(info, "width")
            height = _profile_value(info, "height")
            fps = _profile_value(info, "fps")
            profile = f"{width}x{height}"
            if fps:
                profile += f" @ {fps:g} fps"
            serial_number = str(info["id"])
            name = info.get("name", "RealSense")
            common_kwargs = {
                "camera_type": "realsense",
                "camera_id": serial_number,
                "width": None,
                "height": None,
                "fps": None,
            }
            options.append(
                CameraOption(
                    label=f"RGB | RealSense | {name} | {serial_number} | {profile}",
                    stream="rgb",
                    **common_kwargs,
                )
            )
            options.append(
                CameraOption(
                    label=f"Depth | RealSense | {name} | {serial_number} | colorized",
                    stream="depth",
                    **common_kwargs,
                )
            )
    except Exception as exc:
        warnings.append(f"RealSense camera scan failed: {exc}")

    return options, warnings


def make_camera(option: CameraOption):
    if option.camera_type == "opencv":
        config = OpenCVCameraConfig(
            index_or_path=option.camera_id,
            color_mode=ColorMode.RGB,
            width=option.width,
            height=option.height,
            fps=option.fps,
        )
        return OpenCVCamera(config)

    if option.camera_type == "realsense":
        config = RealSenseCameraConfig(
            serial_number_or_name=str(option.camera_id),
            color_mode=ColorMode.RGB,
            use_depth=option.stream == "depth",
            width=option.width,
            height=option.height,
            fps=option.fps,
        )
        return RealSenseCamera(config)

    raise ValueError(f"Unsupported camera type: {option.camera_type}")


def depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth)
    valid_depth = depth[depth > 0]
    if valid_depth.size == 0:
        normalized = np.zeros_like(depth, dtype=np.uint8)
    else:
        near = np.percentile(valid_depth, 2)
        far = np.percentile(valid_depth, 98)
        if far <= near:
            far = near + 1
        normalized = np.clip((depth - near) * 255.0 / (far - near), 0, 255).astype(np.uint8)

    colored_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


class CameraPreviewApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LeRobot Camera Preview")
        self.root.geometry("1100x760")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.options: list[CameraOption] = []
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.camera = None
        self.current_image: ImageTk.PhotoImage | None = None

        self.selected_camera = tk.StringVar()
        self.status_text = tk.StringVar(value="Scanning cameras...")

        self._build_ui()
        self.refresh_cameras()
        self.root.after(30, self._update_ui)

    def _build_ui(self) -> None:
        top_bar = ttk.Frame(self.root, padding=(12, 12, 12, 8))
        top_bar.pack(fill=tk.X)

        ttk.Label(top_bar, text="Camera").pack(side=tk.LEFT)
        self.camera_combo = ttk.Combobox(
            top_bar,
            textvariable=self.selected_camera,
            state="readonly",
            width=90,
        )
        self.camera_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 8))
        self.camera_combo.bind("<<ComboboxSelected>>", self._on_camera_selected)

        ttk.Button(top_bar, text="Refresh", command=self.refresh_cameras).pack(side=tk.LEFT)

        self.preview_label = ttk.Label(self.root, anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        status_bar = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        status_bar.pack(fill=tk.X)
        ttk.Label(status_bar, textvariable=self.status_text, anchor=tk.W).pack(fill=tk.X)

    def refresh_cameras(self) -> None:
        self._stop_stream()
        self.status_text.set("Scanning cameras...")
        self.options, warnings = find_camera_options()
        labels = [option.label for option in self.options]
        self.camera_combo["values"] = labels

        if labels:
            self.camera_combo.current(0)
            status = f"Found {len(labels)} camera stream option(s). Select one to start preview."
        else:
            self.selected_camera.set("")
            status = "No cameras found. Check USB connection and camera permissions."

        if warnings:
            print("\n".join(warnings))
            status += f" Warning: {warnings[-1]}"
        self.status_text.set(status)

    def _on_camera_selected(self, _event: tk.Event) -> None:
        index = self.camera_combo.current()
        if index < 0 or index >= len(self.options):
            return
        self._start_stream(self.options[index])

    def _start_stream(self, option: CameraOption) -> None:
        self._stop_stream()
        self._clear_queues()
        self.stop_event.clear()
        self.status_text.set(f"Opening {option.label} ...")
        self.worker = threading.Thread(target=self._capture_loop, args=(option,), daemon=True)
        self.worker.start()

    def _capture_loop(self, option: CameraOption) -> None:
        camera = None
        try:
            camera = make_camera(option)
            self.camera = camera
            camera.connect()
            self.status_queue.put(f"Streaming {option.label}")

            while not self.stop_event.is_set():
                if option.stream == "depth":
                    frame = depth_to_rgb(camera.read_depth())
                else:
                    frame = camera.read()
                self._put_latest_frame(frame)
        except Exception as exc:
            self.status_queue.put(f"Camera error: {exc}")
        finally:
            if camera is not None:
                try:
                    camera.disconnect()
                except Exception:
                    pass
            if self.camera is camera:
                self.camera = None

    def _put_latest_frame(self, frame: np.ndarray) -> None:
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(frame)

    def _update_ui(self) -> None:
        while True:
            try:
                self.status_text.set(self.status_queue.get_nowait())
            except queue.Empty:
                break

        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            self._show_frame(frame)

        self.root.after(30, self._update_ui)

    def _show_frame(self, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = frame[:, :, :3]

        image = Image.fromarray(np.ascontiguousarray(frame))
        image.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT), Image.Resampling.LANCZOS)
        self.current_image = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.current_image)

    def _stop_stream(self) -> None:
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=3)
        self.worker = None

        if self.camera is not None:
            try:
                self.camera.disconnect()
            except Exception:
                pass
            self.camera = None

    def _clear_queues(self) -> None:
        for q in (self.frame_queue, self.status_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def close(self) -> None:
        self._stop_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    CameraPreviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
