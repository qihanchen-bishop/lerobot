#!/usr/bin/env python
"""Train the BI-UR3 ACT policy with follower joint deltas and a separate gripper head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_ROOT = Path("/home/qihan/data/lerobot/data/test1_full")
DEFAULT_REPO_ID = "local/test1_full"
DEFAULT_IMAGE_KEYS = ["observation.images.front", "observation.images.top"]
DEFAULT_STATE_KEYS = ["observation.state"]
DEFAULT_OUTPUT_DIR = Path("outputs/train/UR-FDeltaSeparateGrip-ACT-FT-test1-full")
DEFAULT_JOB_NAME = "UR-FDeltaSeparateGrip-ACT-FT-test1-full"
ACTION_TARGETS = (
    "follower_joint_delta_gripper_absolute",
    "follower_joint_anchor_delta_gripper_absolute",
)
EXPECTED_FEATURE_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--image-keys", nargs="+", default=DEFAULT_IMAGE_KEYS)
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--n-action-steps", type=int, default=60)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(240, 320),
        help="Aspect-preserving ACT resize-and-pad shape for every selected camera.",
    )
    parser.add_argument(
        "--act-action-target",
        choices=ACTION_TARGETS,
        default=ACTION_TARGETS[0],
    )
    parser.add_argument("--act-follower-state-key", default="observation.state")
    parser.add_argument("--act-gripper-loss-weight", type=float, default=0.2)
    parser.add_argument("--act-gripper-positive-weight", type=float, default=2.5)
    parser.add_argument(
        "--pretrained-backbone-weights",
        default="ResNet18_Weights.IMAGENET1K_V1",
    )
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--view-root", type=Path, default=None)
    parser.add_argument("--rebuild-view", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def validate_dataset(root: Path, image_keys: list[str], state_keys: list[str]) -> None:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata under {root / 'meta'}.")

    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    required = [*image_keys, *state_keys, "action"]
    missing = [key for key in required if key not in features]
    if missing:
        raise KeyError(f"Missing BI-UR3 feature(s): {missing}")
    if state_keys != ["observation.state"]:
        raise ValueError("BI-UR3 follower training requires --state-keys observation.state.")
    if len(image_keys) != len(set(image_keys)):
        raise ValueError(f"Duplicate image keys are not allowed: {image_keys}")

    for key in ["observation.state", "action"]:
        feature = features[key]
        if feature.get("shape") != [13] or feature.get("names") != EXPECTED_FEATURE_NAMES:
            raise ValueError(
                f"{key} does not match the expected 12-joint + right-gripper layout: {feature}"
            )
    for image_key in image_keys:
        if features[image_key].get("dtype") != "video":
            raise ValueError(f"Selected image feature is not a video: {image_key}")
        if not (root / "videos" / image_key).is_dir():
            raise FileNotFoundError(root / "videos" / image_key)

    stats = json.loads(stats_path.read_text())
    for image_key in image_keys:
        std = stats.get(image_key, {}).get("std")
        flattened = _flatten_numbers(std)
        if len(flattened) != 3 or min(flattened) <= 0.02:
            raise ValueError(
                f"Implausible image std for {image_key}: {std}. "
                "Run mycode/repair_lerobot_image_stats.py first."
            )


def _flatten_numbers(value) -> list[float]:
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_numbers(item))
        return result
    return [float(value)] if value is not None else []


def configure_base_args(args: argparse.Namespace) -> None:
    args.policy_type = "act"
    args.no_gripper = False
    args.act_action_representation = "absolute"
    args.act_image_size = tuple(args.image_size)
    args.act_anchor_loss_weight = 0.25
    args.act_motion_loss_weight = 0.25
    args.act_reconstruction_loss_weight = 0.5
    args.act_anchor_motion_gate_init = 0.1
    args.act_separate_gripper_head = True
    args.act_camera_embedding = False
    args.act_camera_embedding_mode = "default"
    args.act_camera_embedding_gate_init = 0.0


def main() -> None:
    args = parse_args()
    configure_base_args(args)
    validate_dataset(args.root, args.image_keys, args.state_keys)

    import train_lerobot_policy as base

    final_log_path = args.output_dir / "logs" / "train.log"
    temp_log_path = args.output_dir.parent / f".{args.output_dir.name}.train.log.tmp"
    with base.tee_output(temp_log_path):
        print("BI-UR3 follower-delta ACT training")
        print("Action heads: 12-joint regression + independent binary gripper classification")
        print(f"Dataset root: {args.root}")
        base.run_training(args, final_log_path)
    if args.output_dir.exists():
        final_log_path.parent.mkdir(parents=True, exist_ok=True)
        if final_log_path.exists():
            final_log_path.unlink()
        temp_log_path.replace(final_log_path)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
