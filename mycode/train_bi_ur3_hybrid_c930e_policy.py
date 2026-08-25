#!/usr/bin/env python
"""Train a LeRobot policy for the BI-UR3 GELLO/UR hybrid C930e dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_ROOT = Path("/home/qihan/data/lerobot/data/bi_ur3_hybrid_c930e")
DEFAULT_REPO_ID = "local/bi_ur3_hybrid_c930e"
DEFAULT_IMAGE_KEYS = ["observation.images.c930e"]
DEFAULT_STATE_KEYS = ["observation.state"]
DEFAULT_OUTPUT_DIR = Path("outputs/train/act_bi_ur3_hybrid_c930e_c930e")
DEFAULT_JOB_NAME = "act_bi_ur3_hybrid_c930e_c930e"
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
    parser.add_argument("--policy-type", default="act", help="Policy type, e.g. act or diffusion.")
    parser.add_argument("--image-keys", nargs="*", default=DEFAULT_IMAGE_KEYS)
    parser.add_argument("--state-keys", nargs="*", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default=DEFAULT_JOB_NAME)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=100, help="ACT action chunk size.")
    parser.add_argument("--n-action-steps", type=int, default=100, help="ACT actions used per policy call.")
    parser.add_argument("--diffusion-n-obs-steps", type=int, default=2, help="Diffusion observation window.")
    parser.add_argument("--diffusion-horizon", type=int, default=16, help="Diffusion action horizon.")
    parser.add_argument(
        "--diffusion-n-action-steps",
        type=int,
        default=8,
        help="Diffusion executed action steps per policy call.",
    )
    parser.add_argument(
        "--diffusion-full-frame",
        action="store_true",
        help="Disable Diffusion Policy's default 84x84 crop and send the complete image to its backbone.",
    )
    parser.add_argument(
        "--diffusion-backbone-norm",
        choices=["group", "batch", "frozen_batch"],
        default="group",
        help="Normalization inside the diffusion ResNet. Use frozen_batch with pretrained weights.",
    )
    parser.add_argument(
        "--diffusion-drop-n-last-frames",
        type=int,
        default=None,
        help="Exclude this many frames at each episode end from diffusion training sampling.",
    )
    parser.add_argument(
        "--diffusion-mask-padding-loss",
        action="store_true",
        help="Exclude copy-padded action targets from the diffusion denoising loss.",
    )
    parser.add_argument(
        "--diffusion-num-train-timesteps",
        type=int,
        default=100,
        help="Number of noise levels in the diffusion training schedule.",
    )
    parser.add_argument(
        "--diffusion-num-inference-steps",
        type=int,
        default=None,
        help="Reverse denoising steps per action chunk. Defaults to all training noise levels.",
    )
    parser.add_argument(
        "--diffusion-lr-backbone",
        type=float,
        default=None,
        help="Optional learning rate for the diffusion visual backbone; other DP parameters use 1e-4.",
    )
    parser.add_argument(
        "--pretrained-backbone-weights",
        default=None,
        help="Optional torchvision backbone weights for ACT or diffusion, e.g. ResNet18_Weights.IMAGENET1K_V1.",
    )
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=0, help="Keep 0 for real robot offline datasets.")
    parser.add_argument("--video-backend", default="pyav", help="Use pyav if torchcodec/FFmpeg is not healthy.")
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--view-root", type=Path, default=None)
    parser.add_argument("--rebuild-view", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args()


def validate_bi_ur3_hybrid_dataset(root: Path, image_keys: list[str], state_keys: list[str]) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")

    with open(info_path) as f:
        info = json.load(f)
    features = info.get("features", {})
    missing = [key for key in [*image_keys, *state_keys, "action"] if key not in features]
    if missing:
        raise KeyError(f"Missing expected BI-UR3 feature(s): {missing}")

    for key in [*state_keys, "action"]:
        names = features[key].get("names")
        if names != EXPECTED_FEATURE_NAMES:
            raise ValueError(
                f"{key} feature names do not match the BI-UR3 hybrid layout.\n"
                f"Expected: {EXPECTED_FEATURE_NAMES}\n"
                f"Found:    {names}"
            )

    gripper_names = [name for name in features["action"]["names"] if "gripper" in str(name).lower()]
    if gripper_names != ["right_gripper"]:
        raise ValueError(
            "This script expects the hybrid dataset to contain only the right gripper in the action/state vector. "
            f"Found gripper dimensions: {gripper_names}"
        )

    for image_key in image_keys:
        video_root = root / "videos" / image_key
        if not video_root.exists():
            raise FileNotFoundError(f"Missing selected C930e video directory: {video_root}")


def main() -> None:
    args = parse_args()
    args.no_gripper = False
    validate_bi_ur3_hybrid_dataset(args.root, args.image_keys, args.state_keys)

    import train_lerobot_policy as base

    final_log_path = args.output_dir / "logs" / "train.log"
    temp_log_path = args.output_dir.parent / f".{args.output_dir.name}.train.log.tmp"
    with base.tee_output(temp_log_path):
        print("BI-UR3 hybrid C930e training")
        print(f"Dataset root: {args.root}")
        print("Action/state layout: left joints 0-5, right joints 0-5, right_gripper")
        base.run_training(args, final_log_path)
    if args.output_dir.exists():
        final_log_path.parent.mkdir(parents=True, exist_ok=True)
        if final_log_path.exists():
            final_log_path.unlink()
        temp_log_path.replace(final_log_path)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
