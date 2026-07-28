#!/usr/bin/env python
"""Single-arm GUI for live visual policy validation.

This launcher intentionally reuses the real-robot GUI implementation from
gui_eval_lerobot_policy.py so camera scanning, camera preview, port selection,
connection checks, live image preview, and action readout stay consistent with
the bimanual validation GUI.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MYCODE_DIR = Path(__file__).resolve().parent
if str(MYCODE_DIR) not in sys.path:
    sys.path.insert(0, str(MYCODE_DIR))

from gui_eval_lerobot_policy import (
    DEFAULT_EVAL_ROOT,
    DEFAULT_OUTPUT_ROOT,
    EvalPolicyApp,
)


DEFAULT_SINGLE_POLICY = (
    DEFAULT_OUTPUT_ROOT / "so101_single_act_no_gripper/checkpoints/050000/pretrained_model"
)
DEFAULT_SINGLE_EVAL_ROOT = DEFAULT_EVAL_ROOT / "single_arm"


class SingleArmVisualValidationApp(EvalPolicyApp):
    """Preset the existing validation GUI for one SO101 follower arm."""

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root)
        self.root.title("SO101 Single-Arm Visual Policy Validation")
        self._apply_single_arm_defaults()

    def _apply_single_arm_defaults(self) -> None:
        self.vars["robot_mode"].set("single")
        self.vars["robot_type"].set("so101_follower")
        self.vars["robot_port"].set(self.vars["left_follower_port"].get() or "/dev/ttyACM0")
        self.vars["left_follower_port"].set(self.vars["robot_port"].get() or "/dev/ttyACM0")
        self.vars["robot_id"].set("so101_single_visual_validation")
        self.vars["left_follower_id"].set("so101_single_follower")
        self.vars["dataset_repo_id"].set("seeed/so101_single_visual_validation")
        self.vars["dataset_root"].set(str(DEFAULT_SINGLE_EVAL_ROOT))
        self.vars["task"].set("cube1")
        self.vars["episode_time_s"].set("30")
        self.vars["fps"].set("30")
        self.vars["opencv_front"].set("/dev/video4")
        self.vars["front_camera_type"].set("opencv")
        self.include_side_camera.set(False)
        self.auto_reset_after_policy.set(False)
        self.lock_grippers.set(True)
        self.save_video.set(True)
        self.use_amp.set(False)
        self.vars["extra_args"].set(
            "--display_data=false --dataset.push_to_hub=false "
            "--dataset.num_episodes=1 --dataset.vcodec=h264"
        )

        if DEFAULT_SINGLE_POLICY.exists():
            self.vars["checkpoint_path"].set(str(DEFAULT_SINGLE_POLICY))
            self.vars["policy_type"].set(self._infer_policy_type(DEFAULT_SINGLE_POLICY))
            for option in self.checkpoint_options:
                if option.path.resolve() == DEFAULT_SINGLE_POLICY.resolve():
                    self.vars["checkpoint"].set(option.label)
                    break
            self._refresh_action_chunk_settings(DEFAULT_SINGLE_POLICY)

        self._refresh_camera_config_preview()
        self._refresh_grid_stats()
        self.vars["status"].set("Select/check the single-arm port and camera, then Check Connect.")

    def _build_ui(self) -> None:
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.main_canvas = tk.Canvas(main_frame, highlightthickness=0)
        main_scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=main_scrollbar.set)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        content = tk.Frame(self.main_canvas)
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

        top = tk.Frame(content, padx=14, pady=14)
        top.pack(fill=tk.X)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(3, weight=1)

        row = 0
        tk.Label(top, text="Algorithm").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        self.policy_type_combo = ttk.Combobox(
            top,
            textvariable=self.vars["policy_type"],
            values=("act", "diffusion", "mask_act", "smolvla", "pi0", "pi0_fast", "pi05", "vqbet", "tdmpc", "sarm"),
            state="readonly",
            width=18,
        )
        self.policy_type_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Button(top, text="Refresh Weights", command=self.refresh_checkpoints).grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )

        row += 1
        tk.Label(top, text="Checkpoint").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        self.checkpoint_combo = ttk.Combobox(top, textvariable=self.vars["checkpoint"], state="readonly")
        self.checkpoint_combo.grid(row=row, column=1, columnspan=2, sticky=tk.EW, pady=5)
        self.checkpoint_combo.bind("<<ComboboxSelected>>", self.on_checkpoint_selected)
        ttk.Button(top, text="Browse", command=self.browse_checkpoint).grid(
            row=row, column=3, sticky=tk.W, padx=(8, 0)
        )

        row += 1
        tk.Label(top, text="Weight path").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["checkpoint_path"]).grid(
            row=row, column=1, columnspan=3, sticky=tk.EW
        )

        row += 1
        tk.Label(top, text="Conda env").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["conda_env"], width=18).grid(row=row, column=1, sticky=tk.W)
        tk.Label(top, text="Arm port").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["robot_port"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        tk.Label(top, text="Calib id").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["left_follower_id"]).grid(row=row, column=1, sticky=tk.EW)
        tk.Label(top, text="Calibration dir").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["calibration_dir"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        tk.Label(top, text="Front camera type").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
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
        ttk.Button(top, text="Preview", command=self.preview_front_camera).grid(row=row, column=3, sticky=tk.W)

        row += 1
        tk.Label(top, text="Detected front").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        self.front_camera_combo = ttk.Combobox(
            top,
            textvariable=self.vars["front_camera_choice"],
            state="readonly",
        )
        self.front_camera_combo.grid(row=row, column=1, columnspan=3, sticky=tk.EW)
        self.front_camera_combo.bind("<<ComboboxSelected>>", self._on_front_camera_selected)

        row += 1
        tk.Label(top, text="OpenCV front").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["opencv_front"]).grid(row=row, column=1, sticky=tk.EW)
        tk.Label(top, text="RealSense serial").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["realsense_serial"]).grid(row=row, column=3, sticky=tk.EW)

        row += 1
        tk.Label(top, text="Camera config").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["camera_config"], state="readonly").grid(
            row=row, column=1, columnspan=3, sticky=tk.EW
        )

        row += 1
        tk.Label(top, text="Dataset repo").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["dataset_repo_id"]).grid(row=row, column=1, sticky=tk.EW)
        tk.Label(top, text="Save parent").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        save_frame = tk.Frame(top)
        save_frame.grid(row=row, column=3, sticky=tk.EW)
        save_frame.columnconfigure(0, weight=1)
        ttk.Entry(save_frame, textvariable=self.vars["dataset_root"]).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(save_frame, text="Browse", command=self.browse_dataset_root).grid(row=0, column=1, padx=(8, 0))

        row += 1
        tk.Label(top, text="Task").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Combobox(
            top,
            textvariable=self.vars["task"],
            values=("cube1", "cube2", "cube3"),
        ).grid(row=row, column=1, sticky=tk.EW)
        tk.Label(top, text="Grid cell").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        grid_frame = tk.Frame(top)
        grid_frame.grid(row=row, column=3, sticky=tk.W)
        ttk.Entry(grid_frame, textvariable=self.vars["grid_size"], width=6).pack(side=tk.LEFT)
        self.grid_cell_combo = ttk.Combobox(grid_frame, textvariable=self.vars["grid_cell"], state="readonly", width=10)
        self.grid_cell_combo.pack(side=tk.LEFT, padx=(8, 0))

        row += 1
        tk.Label(top, text="Trials per cell").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["trials_per_grid"], width=12).grid(row=row, column=1, sticky=tk.W)
        tk.Label(top, textvariable=self.vars["grid_stats"]).grid(
            row=row, column=2, columnspan=2, sticky=tk.W, padx=(16, 0), pady=5
        )

        row += 1
        tk.Label(top, text="Episode sec").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["episode_time_s"], width=12).grid(row=row, column=1, sticky=tk.W)
        tk.Label(top, text="FPS").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["fps"], width=12).grid(row=row, column=3, sticky=tk.W)

        row += 1
        tk.Label(top, text="Model chunk").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["model_chunk_size"], width=12, state="readonly").grid(
            row=row, column=1, sticky=tk.W
        )
        tk.Label(top, text="Prediction / replan").grid(row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5)
        steps = tk.Frame(top)
        steps.grid(row=row, column=3, sticky=tk.W)
        ttk.Entry(steps, textvariable=self.vars["prediction_steps"], width=10).pack(side=tk.LEFT)
        ttk.Entry(steps, textvariable=self.vars["n_action_steps"], width=10).pack(side=tk.LEFT, padx=(8, 0))

        row += 1
        ttk.Checkbutton(top, text="None gripper / keep initial angle", variable=self.lock_grippers).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, padx=(0, 8), pady=5
        )
        ttk.Checkbutton(top, text="Save MP4 video", variable=self.save_video).grid(
            row=row, column=2, sticky=tk.W, padx=(16, 8), pady=5
        )
        ttk.Checkbutton(top, text="Use AMP", variable=self.use_amp).grid(row=row, column=3, sticky=tk.W)

        row += 1
        tk.Label(top, text="Extra args").grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=5)
        ttk.Entry(top, textvariable=self.vars["extra_args"]).grid(row=row, column=1, columnspan=3, sticky=tk.EW)

        # Keep advanced inherited settings valid without showing the full bimanual/leader UI.
        self.vars["robot_mode"].set("single")
        self.vars["robot_type"].set("so101_follower")
        self.vars["execution_mode"].set("synchronous")
        self.vars["camera_read_mode"].set("wait_new_frame")
        self.vars["fusion_steps"].set("0")
        self.vars["fusion_history_weight"].set("0")
        self.vars["num_inference_steps"].set("")
        self.vars["noise_scheduler_type"].set("checkpoint")
        self.include_side_camera.set(False)

        buttons = tk.Frame(content, padx=14, pady=10)
        buttons.pack(fill=tk.X)
        self.connect_button = ttk.Button(buttons, text="Check Connect", command=self.check_connect)
        self.connect_button.pack(side=tk.LEFT, ipady=4)
        self.start_button = ttk.Button(buttons, text="Start", command=self.start_eval, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.stop_button = ttk.Button(buttons, text="End / Stop", command=self.stop_eval, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.capture_reset_button = ttk.Button(
            buttons, text="Capture Reset Pose", command=self.capture_reset_pose, state=tk.DISABLED
        )
        self.capture_reset_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.reset_button = ttk.Button(buttons, text="Reset Arm", command=self.reset_arms, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        self.disconnect_button = ttk.Button(buttons, text="Disconnect", command=self.disconnect_robot, state=tk.DISABLED)
        self.disconnect_button.pack(side=tk.LEFT, padx=(10, 0), ipady=4)
        ttk.Button(buttons, text="Open Save Folder", command=self.open_save_folder).pack(
            side=tk.LEFT, padx=(10, 0), ipady=4
        )
        tk.Label(buttons, textvariable=self.vars["elapsed"]).pack(side=tk.RIGHT)

        # Hidden inherited buttons that base state-management code still updates.
        self.teleop_button = ttk.Button(buttons, text="Start Teleop", command=self.start_teleop, state=tk.DISABLED)
        self.stop_teleop_button = ttk.Button(buttons, text="Stop Teleop", command=self.stop_teleop, state=tk.DISABLED)
        self.default_pose_button = ttk.Button(
            buttons, text="Go to 90% Pose", command=self.go_to_default_success_pose, state=tk.DISABLED
        )

        result_frame = ttk.LabelFrame(content, text="Result", padding=12)
        result_frame.pack(fill=tk.X, padx=14, pady=(0, 10))
        tk.Label(result_frame, text="Last run").pack(side=tk.LEFT)
        ttk.Combobox(
            result_frame,
            textvariable=self.vars["result"],
            values=("success", "failure"),
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(result_frame, text="Review / Save Result", command=self._show_result_dialog).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        tk.Label(result_frame, textvariable=self.vars["status"]).pack(side=tk.LEFT, padx=(16, 0))

        log_frame = tk.Frame(content, padx=14, pady=14)
        log_frame.pack(fill=tk.BOTH, expand=True)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(1, weight=0)
        self.preview_label = ttk.Label(log_frame, anchor=tk.CENTER)
        self.preview_label.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 10))
        right_log = tk.Frame(log_frame)
        right_log.grid(row=0, column=1, sticky=tk.NS)
        tk.Label(right_log, text="Log / Action").pack(anchor=tk.W)
        self.log_text = tk.Text(right_log, wrap=tk.WORD, height=18)
        scrollbar = ttk.Scrollbar(right_log, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

    def start_teleop(self) -> None:
        self.vars["status"].set("Single-arm visual validation does not use bimanual leader teleop.")

    def go_to_default_success_pose(self) -> None:
        self.vars["status"].set("The stored 90% bimanual pose is not used for single-arm validation.")


def main() -> None:
    root = tk.Tk()
    SingleArmVisualValidationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
