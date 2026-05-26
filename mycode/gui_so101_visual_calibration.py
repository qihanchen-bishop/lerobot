#!/usr/bin/env python
"""Visual SO101 calibration helper.

Run from the LeRobot environment:
    conda run -n lerobot python mycode/gui_so101_visual_calibration.py
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_ROOT = PROJECT_ROOT / "calibration"
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_ORDER = ("left_follower", "right_follower", "left_leader", "right_leader")
ARM_LABELS = {
    "left_follower": "Follower L",
    "right_follower": "Follower R",
    "left_leader": "Leader L",
    "right_leader": "Leader R",
}
ARM_DEFAULTS = {
    "left_follower": ("robot", "/dev/ttyACM0", "my_awesome_follower_arm"),
    "right_follower": ("robot", "/dev/ttyACM1", "my_awesome_follower_arm_r"),
    "left_leader": ("teleop", "/dev/ttyACM2", "my_awesome_leader_arm"),
    "right_leader": ("teleop", "/dev/ttyACM3", "my_awesome_leader_arm_r"),
}


def calibration_dir_for(kind: str, root: Path) -> Path:
    if kind == "robot":
        return root / "robots" / "so_follower"
    return root / "teleoperators" / "so_leader"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def save_calibration_json(path: Path, calibration: dict[str, MotorCalibration]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {motor: asdict(calibration[motor]) for motor in MOTORS if motor in calibration}
    with path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def format_calibration(data: dict[str, Any] | None) -> str:
    if not data:
        return "No calibration file.\n"
    lines = [f"{'joint':<15} {'id':>2} {'home':>6} {'min':>6} {'max':>6}"]
    for motor in MOTORS:
        item = data.get(motor)
        if not item:
            lines.append(f"{motor:<15} missing")
            continue
        lines.append(
            f"{motor:<15} {item.get('id', ''):>2} {item.get('homing_offset', ''):>6} "
            f"{item.get('range_min', ''):>6} {item.get('range_max', ''):>6}"
        )
    return "\n".join(lines) + "\n"


def make_device(kind: str, port: str, arm_id: str, calibration_root: Path):
    calib_dir = calibration_dir_for(kind, calibration_root)
    if kind == "robot":
        return SO101Follower(SOFollowerRobotConfig(id=arm_id, port=port, calibration_dir=calib_dir))
    return SO101Leader(SOLeaderTeleopConfig(id=arm_id, port=port, calibration_dir=calib_dir))


class VisualCalibrationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SO101 Visual Calibration")
        self.root.geometry("1500x920")
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_flags: dict[str, threading.Event] = {}
        self.devices: dict[str, Any] = {}
        self.bus_locks: dict[str, threading.Lock] = {}
        self.pending_files: dict[str, Path] = {}
        self.pending_data: dict[str, dict[str, Any]] = {}
        self.range_state: dict[str, dict[str, int]] = {}
        self.teleop_thread: threading.Thread | None = None
        self.monitor_thread: threading.Thread | None = None
        self.monitor_stop = threading.Event()
        self.teleop_stop = threading.Event()
        self.teleop_running = False

        self.calibration_root = tk.StringVar(value=str(DEFAULT_CALIBRATION_ROOT))
        self.selected_arm = tk.StringVar(value="left_follower")
        self.test_pair = tk.StringVar(value="left")
        self.status = tk.StringVar(value="Ready. Existing calibration is loaded from the project calibration folder.")
        self.arm_vars: dict[str, dict[str, tk.StringVar]] = {}
        for arm, (kind, port, arm_id) in ARM_DEFAULTS.items():
            self.arm_vars[arm] = {
                "kind": tk.StringVar(value=kind),
                "port": tk.StringVar(value=port),
                "id": tk.StringVar(value=arm_id),
            }

        self.previous_texts: dict[str, tk.Text] = {}
        self.current_texts: dict[str, tk.Text] = {}
        self.new_texts: dict[str, tk.Text] = {}

        self._configure_style()
        self._build_ui()
        self.refresh_calibrations()
        self.root.after(80, self._poll_events)

    def _configure_style(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=11)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=11)
        self.root.option_add("*Font", default_font)
        style = ttk.Style(self.root)
        style.configure("TButton", padding=(10, 6))
        style.configure("TEntry", padding=(4, 4))
        style.configure("TCombobox", padding=(4, 4))

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Calibration root").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Entry(top, textvariable=self.calibration_root).grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(top, text="Browse", command=self.browse_root).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(top, text="Refresh", command=self.refresh_calibrations).grid(row=0, column=3, padx=(8, 0))

        ports = ttk.LabelFrame(self.root, text="Arms", padding=10)
        ports.pack(fill=tk.X, padx=10, pady=(0, 8))
        for col, title in enumerate(("Arm", "Port", "Calibration id")):
            ttk.Label(ports, text=title).grid(row=0, column=col, sticky=tk.W, padx=(0, 10))
        for row, arm in enumerate(ARM_ORDER, start=1):
            ttk.Label(ports, text=ARM_LABELS[arm]).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3)
            ttk.Entry(ports, textvariable=self.arm_vars[arm]["port"], width=18).grid(row=row, column=1, sticky=tk.W)
            ttk.Entry(ports, textvariable=self.arm_vars[arm]["id"], width=34).grid(row=row, column=2, sticky=tk.W)

        actions = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        actions.pack(fill=tk.X)
        ttk.Label(actions, text="Recalibrate").pack(side=tk.LEFT)
        ttk.Combobox(
            actions,
            textvariable=self.selected_arm,
            values=ARM_ORDER,
            width=18,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(8, 8))
        ttk.Button(actions, text="Connect selected", command=self.connect_selected).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(actions, text="Set middle / start", command=self.set_middle_selected).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(actions, text="Start range record", command=self.start_range_selected).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(actions, text="Stop and save pending", command=self.stop_range_selected).pack(
            side=tk.LEFT, padx=(0, 16)
        )
        ttk.Button(actions, text="Connect all monitor", command=self.connect_all_monitor).pack(side=tk.LEFT)
        ttk.Button(actions, text="Disconnect all", command=self.disconnect_all).pack(side=tk.LEFT, padx=(8, 0))

        teleop = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        teleop.pack(fill=tk.X)
        ttk.Label(teleop, text="Teleop test pair").pack(side=tk.LEFT)
        ttk.Combobox(teleop, textvariable=self.test_pair, values=("left", "right"), width=8, state="readonly").pack(
            side=tk.LEFT, padx=(8, 8)
        )
        ttk.Button(teleop, text="Start teleop test", command=self.start_teleop_test).pack(side=tk.LEFT)
        ttk.Button(teleop, text="Disconnect teleop test", command=self.stop_teleop_test).pack(side=tk.LEFT, padx=(8, 0))

        panes = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        panes.pack(fill=tk.BOTH, expand=True)
        for col, title in enumerate(("Previous calibration", "Current positions", "New pending calibration")):
            frame = ttk.LabelFrame(panes, text=title, padding=8)
            frame.grid(row=0, column=col, sticky=tk.NSEW, padx=(0 if col == 0 else 8, 0))
            panes.columnconfigure(col, weight=1)
            self._build_arm_grid(frame, col)
        panes.rowconfigure(0, weight=1)

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_arm_grid(self, parent: ttk.Frame, column_kind: int) -> None:
        for index, arm in enumerate(ARM_ORDER):
            box = ttk.LabelFrame(parent, text=ARM_LABELS[arm], padding=6)
            box.grid(row=index // 2, column=index % 2, sticky=tk.NSEW, padx=4, pady=4)
            parent.columnconfigure(index % 2, weight=1)
            parent.rowconfigure(index // 2, weight=1)
            text = tk.Text(box, height=10, width=42, wrap=tk.NONE, font=("TkFixedFont", 10))
            text.pack(fill=tk.BOTH, expand=True)
            text.configure(state=tk.DISABLED)
            if column_kind == 0:
                self.previous_texts[arm] = text
            elif column_kind == 1:
                self.current_texts[arm] = text
            else:
                self.new_texts[arm] = text
                ttk.Button(box, text="Overwrite with pending", command=lambda a=arm: self.overwrite_pending(a)).pack(
                    fill=tk.X, pady=(6, 0)
                )

    def browse_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.calibration_root.get())
        if selected:
            self.calibration_root.set(selected)
            self.refresh_calibrations()

    def calibration_file(self, arm: str) -> Path:
        kind = self.arm_vars[arm]["kind"].get()
        arm_id = self.arm_vars[arm]["id"].get().strip()
        return calibration_dir_for(kind, Path(self.calibration_root.get()).expanduser()) / f"{arm_id}.json"

    def pending_root(self) -> Path:
        return Path(self.calibration_root.get()).expanduser() / "_pending"

    def backup_root(self) -> Path:
        return Path(self.calibration_root.get()).expanduser() / "_backup"

    def refresh_calibrations(self) -> None:
        for arm in ARM_ORDER:
            data = load_json(self.calibration_file(arm))
            self._set_text(self.previous_texts[arm], format_calibration(data))
            pending = self.pending_data.get(arm)
            self._set_text(self.new_texts[arm], format_calibration(pending) if pending else "No pending calibration.\n")
        self.status.set(f"Loaded calibration files from {Path(self.calibration_root.get()).expanduser()}")

    def connect_selected(self) -> None:
        self._run_thread("connect", self._connect_arm, self.selected_arm.get())

    def connect_all_monitor(self) -> None:
        self.monitor_stop.clear()
        self._run_thread("connect_all", self._connect_all_and_monitor)

    def set_middle_selected(self) -> None:
        arm = self.selected_arm.get()
        if not messagebox.askokcancel(
            "Set middle",
            f"Move {ARM_LABELS[arm]} to the middle of every joint range, then click OK.",
        ):
            return
        self._run_thread("middle", self._set_middle, arm)

    def start_range_selected(self) -> None:
        arm = self.selected_arm.get()
        self.stop_flags[arm] = threading.Event()
        self._run_thread("range", self._record_range, arm)

    def stop_range_selected(self) -> None:
        arm = self.selected_arm.get()
        flag = self.stop_flags.get(arm)
        if flag:
            flag.set()
            self.status.set(f"Stopping range recording for {ARM_LABELS[arm]}...")

    def start_teleop_test(self) -> None:
        if self.teleop_thread and self.teleop_thread.is_alive():
            self.status.set("Teleop test is already running.")
            return
        self.teleop_stop.clear()
        self.monitor_stop.set()
        self.teleop_thread = threading.Thread(target=self._teleop_loop, daemon=True)
        self.teleop_thread.start()

    def stop_teleop_test(self) -> None:
        self.teleop_stop.set()
        self.status.set("Stopping teleop test...")

    def disconnect_all(self) -> None:
        self.monitor_stop.set()
        self.teleop_stop.set()
        self._run_thread("disconnect", self._disconnect_all)

    def overwrite_pending(self, arm: str) -> None:
        pending = self.pending_files.get(arm)
        if not pending or not pending.exists():
            messagebox.showinfo("No pending calibration", f"{ARM_LABELS[arm]} has no pending calibration.")
            return
        target = self.calibration_file(arm)
        if not messagebox.askyesno("Overwrite calibration", f"Overwrite {target.name} with the pending result?"):
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = self.backup_root() / timestamp / target.relative_to(Path(self.calibration_root.get()).expanduser())
        backup.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.copy2(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pending, target)
        self.status.set(f"Overwrote {target}; previous file backed up under {self.backup_root() / timestamp}.")
        self.refresh_calibrations()

    def _run_thread(self, name: str, func, *args) -> None:
        thread = threading.Thread(target=self._thread_wrapper, args=(name, func, args), daemon=True)
        thread.start()

    def _thread_wrapper(self, name: str, func, args: tuple[Any, ...]) -> None:
        try:
            func(*args)
        except Exception as exc:
            self.events.put(("error", f"{name}: {exc}"))

    def _connect_arm(self, arm: str):
        if arm in self.devices and self._device_connected(self.devices[arm]):
            self.events.put(("status", f"{ARM_LABELS[arm]} already connected."))
            return self.devices[arm]
        kind = self.arm_vars[arm]["kind"].get()
        port = self.arm_vars[arm]["port"].get().strip()
        arm_id = self.arm_vars[arm]["id"].get().strip()
        device = make_device(kind, port, arm_id, Path(self.calibration_root.get()).expanduser())
        device.bus.connect()
        self.bus_locks.setdefault(arm, threading.Lock())
        with self.bus_locks[arm]:
            if device.calibration:
                device.bus.write_calibration(device.calibration)
            try:
                device.configure()
            except Exception:
                device.bus.disable_torque()
                for motor in device.bus.motors:
                    device.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            device.bus.disable_torque()
        self.devices[arm] = device
        self.events.put(("status", f"{ARM_LABELS[arm]} connected on {port}. Torque disabled for calibration."))
        return device

    def _connect_all_and_monitor(self) -> None:
        for arm in ARM_ORDER:
            if self.monitor_stop.is_set():
                return
            try:
                self._connect_arm(arm)
            except Exception as exc:
                self.events.put(("error", f"{ARM_LABELS[arm]} connect failed: {exc}"))
        self._monitor_positions()

    def _set_middle(self, arm: str) -> None:
        device = self._connect_arm(arm)
        with self.bus_locks[arm]:
            device.bus.disable_torque()
            homing_offsets = device.bus.set_half_turn_homings(list(MOTORS))
        self.range_state[arm] = {
            "homing_offsets": {motor: int(homing_offsets[motor]) for motor in MOTORS},
            "mins": {},
            "maxes": {},
        }
        self.events.put(("status", f"{ARM_LABELS[arm]} middle captured. Now start range recording."))

    def _record_range(self, arm: str) -> None:
        device = self._connect_arm(arm)
        state = self.range_state.setdefault(arm, {})
        if not state.get("homing_offsets"):
            self.events.put(("status", f"{ARM_LABELS[arm]} has no middle yet; capturing current pose as middle."))
            with self.bus_locks[arm]:
                homing_offsets = device.bus.set_half_turn_homings(list(MOTORS))
            state["homing_offsets"] = {motor: int(homing_offsets[motor]) for motor in MOTORS}
        with self.bus_locks[arm]:
            device.bus.disable_torque()
            start_positions = device.bus.sync_read("Present_Position", list(MOTORS), normalize=False)
        mins = {motor: int(start_positions[motor]) for motor in MOTORS}
        maxes = mins.copy()
        flag = self.stop_flags[arm]
        self.events.put(("status", f"Recording {ARM_LABELS[arm]} ranges. Move every joint widely, including wrist_roll and gripper."))
        while not flag.is_set():
            with self.bus_locks[arm]:
                positions = device.bus.sync_read("Present_Position", list(MOTORS), normalize=False)
                mins = {motor: min(int(positions[motor]), mins[motor]) for motor in MOTORS}
                maxes = {motor: max(int(positions[motor]), maxes[motor]) for motor in MOTORS}
                current_text = self._format_positions(device, positions, mins, maxes)
            self.events.put(("current", (arm, current_text)))
            time.sleep(0.08)

        calibration: dict[str, MotorCalibration] = {}
        for motor, motor_obj in device.bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=motor_obj.id,
                drive_mode=0,
                homing_offset=int(state["homing_offsets"][motor]),
                range_min=int(mins[motor]),
                range_max=int(maxes[motor]),
            )
        device.calibration = calibration
        with self.bus_locks[arm]:
            device.bus.write_calibration(calibration)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pending_path = self.pending_root() / timestamp / self.calibration_file(arm).relative_to(
            Path(self.calibration_root.get()).expanduser()
        )
        save_calibration_json(pending_path, calibration)
        self.events.put(("pending", (arm, pending_path, load_json(pending_path))))
        self.events.put(("status", f"Saved pending calibration for {ARM_LABELS[arm]} to {pending_path}."))

    def _monitor_positions(self) -> None:
        while not self.monitor_stop.is_set():
            for arm, device in list(self.devices.items()):
                if not self._device_connected(device):
                    continue
                try:
                    with self.bus_locks.setdefault(arm, threading.Lock()):
                        positions = device.bus.sync_read("Present_Position", list(MOTORS), normalize=False)
                        current_text = self._format_positions(device, positions)
                    self.events.put(("current", (arm, current_text)))
                except Exception as exc:
                    self.events.put(("current", (arm, f"Read failed: {exc}\n")))
            time.sleep(0.15)

    def _teleop_loop(self) -> None:
        pair = self.test_pair.get()
        follower_arm = "left_follower" if pair == "left" else "right_follower"
        leader_arm = "left_leader" if pair == "left" else "right_leader"
        follower = self._connect_arm(follower_arm)
        leader = self._connect_arm(leader_arm)
        self.teleop_running = True
        try:
            with self.bus_locks[follower_arm]:
                follower.bus.enable_torque()
            self.events.put(("status", f"Teleop test running: {ARM_LABELS[leader_arm]} -> {ARM_LABELS[follower_arm]}."))
            last_health_at = 0.0
            while not self.teleop_stop.is_set():
                with self.bus_locks[leader_arm]:
                    action = leader.get_action()
                with self.bus_locks[follower_arm]:
                    follower.send_action(action)
                now = time.monotonic()
                if now - last_health_at >= 1.0:
                    self._emit_health(follower_arm, follower)
                    last_health_at = now
                time.sleep(0.03)
        except Exception as exc:
            self.events.put(("error", f"Teleop test stopped after device error: {exc}"))
        finally:
            try:
                with self.bus_locks[follower_arm]:
                    follower.bus.disable_torque()
            except Exception:
                pass
            self.teleop_running = False
            self.events.put(("status", "Teleop test stopped."))

    def _emit_health(self, arm: str, device: Any) -> None:
        lines = []
        for motor in MOTORS:
            try:
                status = device.bus.read("Status", motor, normalize=False, num_retry=1)
                temp = device.bus.read("Present_Temperature", motor, normalize=False, num_retry=1)
                volt = device.bus.read("Present_Voltage", motor, normalize=False, num_retry=1)
            except Exception:
                continue
            marker = "ALARM" if int(status) else "ok"
            lines.append(f"{motor:<15} {marker:<5} status={int(status):>3} temp={int(temp):>2}C volt={int(volt)/10:.1f}V")
        if lines:
            self.events.put(("current", (arm, "Teleop health\n" + "\n".join(lines) + "\n")))

    def _disconnect_all(self) -> None:
        for arm, device in list(self.devices.items()):
            try:
                if self._device_connected(device):
                    if hasattr(device, "bus"):
                        try:
                            with self.bus_locks.setdefault(arm, threading.Lock()):
                                device.bus.disable_torque()
                        except Exception:
                            pass
                    with self.bus_locks.setdefault(arm, threading.Lock()):
                        device.disconnect() if hasattr(device, "disconnect") else device.bus.disconnect()
            except Exception as exc:
                self.events.put(("error", f"{ARM_LABELS[arm]} disconnect failed: {exc}"))
        self.devices.clear()
        self.bus_locks.clear()
        self.events.put(("status", "All connected arms disconnected."))

    @staticmethod
    def _device_connected(device: Any) -> bool:
        try:
            return bool(device.bus.is_connected)
        except Exception:
            return False

    def _format_positions(
        self,
        device: Any,
        positions: dict[str, Any],
        mins: dict[str, int] | None = None,
        maxes: dict[str, int] | None = None,
    ) -> str:
        lines = [f"{'joint':<15} {'raw':>6} {'norm':>8} {'min':>6} {'max':>6}"]
        for motor in MOTORS:
            raw = int(positions[motor])
            try:
                norm = device.bus.read("Present_Position", motor, normalize=True)
                norm_text = f"{float(norm):8.2f}"
            except Exception:
                norm_text = f"{'n/a':>8}"
            min_text = f"{mins[motor]:>6}" if mins else f"{'':>6}"
            max_text = f"{maxes[motor]:>6}" if maxes else f"{'':>6}"
            lines.append(f"{motor:<15} {raw:>6} {norm_text} {min_text} {max_text}")
        return "\n".join(lines) + "\n"

    def _poll_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "status":
                self.status.set(str(payload))
            elif event == "error":
                self.status.set(str(payload))
                messagebox.showerror("SO101 calibration", str(payload))
            elif event == "current":
                arm, text = payload
                self._set_text(self.current_texts[arm], text)
            elif event == "pending":
                arm, path, data = payload
                self.pending_files[arm] = path
                self.pending_data[arm] = data
                self._set_text(self.new_texts[arm], format_calibration(data))
        self.root.after(80, self._poll_events)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    app = VisualCalibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.disconnect_all(), root.after(200, root.destroy)))
    root.mainloop()


if __name__ == "__main__":
    main()
