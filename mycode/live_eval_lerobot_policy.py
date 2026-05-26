#!/usr/bin/env python
"""Live policy evaluation with left-front camera preview and action readout."""

from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any
import time

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import RobotConfig, bi_so_follower, make_robot_from_config, so_follower  # noqa: F401
from lerobot.scripts.lerobot_record import DatasetRecordConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging


@dataclass
class LiveEvalConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    policy: PreTrainedConfig | None = None
    display_data: bool = False
    keep_connected_after_run: bool = True

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if not policy_path:
            raise ValueError("Please provide --policy.path=/path/to/pretrained_model")

        cli_overrides = parser.get_cli_overrides("policy")
        self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
        self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class LivePolicyWindow:
    def __init__(self, image_key: str, action_names: list[str], expected_inputs: list[str]) -> None:
        self.root = tk.Tk()
        self.root.title("LeRobot Live Policy Eval")
        self.root.geometry("1040x760")
        self.stop_requested = False
        self.image_key = image_key
        self.image_ref: ImageTk.PhotoImage | None = None
        self.root.protocol("WM_DELETE_WINDOW", self.request_stop)

        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill=tk.X)
        self.status = tk.StringVar(value=f"Policy inputs: {', '.join(expected_inputs)}")
        tk.Label(top, textvariable=self.status, anchor=tk.W).pack(fill=tk.X)

        body = tk.Frame(self.root, padx=10, pady=8)
        body.pack(fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(body, bg="black")
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(body, padx=10)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Label(right, text="Policy action").pack(anchor=tk.W)
        self.action_text = tk.Text(right, width=42, height=max(16, len(action_names) + 4))
        self.action_text.pack(fill=tk.Y, expand=False)
        tk.Button(right, text="Stop", command=self.request_stop).pack(fill=tk.X, pady=(10, 0))

    def request_stop(self) -> None:
        self.stop_requested = True

    def update(self, image: np.ndarray | None, action: dict[str, Any], fps_hz: float) -> None:
        if image is not None:
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[2] == 3:
                pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
                pil.thumbnail((720, 540))
                self.image_ref = ImageTk.PhotoImage(pil)
                self.image_label.configure(image=self.image_ref)

        self.action_text.delete("1.0", tk.END)
        self.action_text.insert(tk.END, f"loop_hz: {fps_hz:.1f}\n")
        self.action_text.insert(tk.END, f"image_key: {self.image_key}\n\n")
        for key, value in action.items():
            try:
                text = f"{float(value): .4f}"
            except (TypeError, ValueError):
                text = str(value)
            self.action_text.insert(tk.END, f"{key:<28} {text}\n")
        self.root.update_idletasks()
        self.root.update()


def _select_preview_image(raw_obs: dict[str, Any]) -> tuple[str, np.ndarray | None]:
    if "left_front" in raw_obs:
        return "left_front", raw_obs["left_front"]
    for key, value in raw_obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            return key, value
    return "none", None


@parser.wrap()
def run_live_eval(cfg: LiveEvalConfig) -> LeRobotDataset:
    init_logging()
    print("Live eval config:")
    print(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=robot_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    print(f"Raw robot observation features: {sorted(robot.observation_features)}")
    print(f"Dataset/policy observation features: {sorted(k for k in dataset_features if k.startswith('observation.'))}")
    print(f"Policy input features: {sorted(cfg.policy.input_features)}")
    print(f"Policy output features: {sorted(cfg.policy.output_features)}")

    dataset = None
    listener = None
    try:
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
        )

        policy = make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
                "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
            },
        )

        robot.connect()
        listener, events = init_keyboard_listener()
        window = LivePolicyWindow(
            image_key="left_front",
            action_names=list(dataset.features[ACTION]["names"]),
            expected_inputs=sorted(cfg.policy.input_features),
        )

        with VideoEncodingManager(dataset):
            for episode_idx in range(cfg.dataset.num_episodes):
                print(f"Recording live policy episode {episode_idx}")
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()

                start_t = time.perf_counter()
                timestamp = 0.0
                while timestamp < cfg.dataset.episode_time_s and not events["stop_recording"] and not window.stop_requested:
                    loop_t = time.perf_counter()
                    if events["exit_early"]:
                        events["exit_early"] = False
                        break

                    raw_obs = robot.get_observation()
                    obs_processed = robot_observation_processor(raw_obs)
                    observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
                    policy_input_keys = sorted(k for k in observation_frame if k.startswith("observation."))

                    action_values = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=cfg.dataset.single_task,
                        robot_type=robot.robot_type,
                    )
                    action_dict = make_robot_action(action_values, dataset.features)
                    robot_action_to_send = robot_action_processor((action_dict, raw_obs))
                    robot.send_action(robot_action_to_send)

                    action_frame = build_dataset_frame(dataset.features, action_dict, prefix=ACTION)
                    frame = {**observation_frame, **action_frame, "task": cfg.dataset.single_task}
                    dataset.add_frame(frame)

                    image_key, image = _select_preview_image(raw_obs)
                    loop_s = max(time.perf_counter() - loop_t, 1e-6)
                    window.status.set(
                        f"Task: {cfg.dataset.single_task} | policy inputs: {', '.join(policy_input_keys)}"
                    )
                    window.update(image, action_dict, 1.0 / loop_s)

                    dt_s = time.perf_counter() - loop_t
                    precise_sleep(max(1 / cfg.dataset.fps - dt_s, 0.0))
                    timestamp = time.perf_counter() - start_t

                dataset.save_episode()
                if events["stop_recording"] or window.stop_requested:
                    break

        print("Live eval finished.")
        if cfg.keep_connected_after_run and not events["stop_recording"] and not window.stop_requested:
            print("Keeping robot and camera connected. Press End/Stop, Esc, or close the live window to disconnect.")
            window.status.set("Episode finished. Connected. Press End/Stop, Esc, or close this window to disconnect.")
            while not events["stop_recording"] and not window.stop_requested:
                window.root.update_idletasks()
                window.root.update()
                time.sleep(0.05)
    finally:
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"[live_eval] Warning: disconnect failed after data was saved: {exc}", flush=True)
        if listener:
            listener.stop()
    return dataset


def main() -> None:
    register_third_party_plugins()
    run_live_eval()


if __name__ == "__main__":
    main()
