#!/usr/bin/env python
"""Check a local LeRobot dataset before policy training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


DEFAULT_ROOT = Path("/home/romilab/.cache/huggingface/lerobot/seeedstudio123/test_20260506_153720")
DEFAULT_REPO_ID = "seeedstudio123/test_20260506_153720"
DEFAULT_IMAGE_KEYS = ["observation.images.left_front"]
DEFAULT_STATE_KEYS = ["observation.state"]
DEFAULT_ACTION_KEY = "action"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Dataset repo_id recorded in LeRobot.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Local dataset root.")
    parser.add_argument("--image-keys", nargs="*", default=DEFAULT_IMAGE_KEYS)
    parser.add_argument("--state-keys", nargs="*", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument("--num-samples", type=int, default=8, help="Number of frames to decode and inspect.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Future action chunk length to check.")
    parser.add_argument("--video-backend", default="pyav", help="LeRobot video backend.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def check_feature_exists(features: dict, key: str) -> None:
    if key not in features:
        available = "\n  ".join(features)
        fail(f"Missing feature '{key}'. Available features:\n  {available}")


def tensor_summary(value: torch.Tensor) -> str:
    value = value.float()
    return (
        f"shape={tuple(value.shape)} dtype={value.dtype} "
        f"min={value.min().item():.4g} max={value.max().item():.4g} "
        f"mean={value.mean().item():.4g}"
    )


def main() -> None:
    args = parse_args()
    required_keys = [args.action_key, *args.state_keys, *args.image_keys]

    if not args.root.exists():
        fail(f"Dataset root does not exist: {args.root}")

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    features = meta.features

    print("== Metadata ==")
    print(f"root: {args.root}")
    print(f"repo_id: {args.repo_id}")
    print(f"robot_type: {meta.robot_type}")
    print(f"fps: {meta.fps}")
    print(f"episodes: {meta.total_episodes}")
    print(f"frames: {meta.total_frames}")
    print(f"camera_keys: {meta.camera_keys}")

    for key in required_keys:
        check_feature_exists(features, key)

    print("\n== Required Features ==")
    for key in required_keys:
        print(f"{key}: {json.dumps(features[key], ensure_ascii=False)}")

    missing_stats = [key for key in required_keys if key not in meta.stats]
    if missing_stats:
        fail(f"Missing stats for required feature(s): {missing_stats}")

    print("\n== File Presence ==")
    data_files = sorted((args.root / "data").glob("*/*.parquet"))
    print(f"data parquet files: {len(data_files)}")
    if not data_files:
        fail("No parquet files found under data/*/*.parquet")

    for image_key in args.image_keys:
        video_files = sorted((args.root / "videos" / image_key).glob("*/*.mp4"))
        print(f"{image_key} videos: {len(video_files)}")
        if not video_files:
            fail(f"No mp4 files found for {image_key}")

    delta_timestamps = {
        args.action_key: [i / meta.fps for i in range(args.chunk_size)],
    }
    dataset_kwargs = {}
    if args.video_backend:
        dataset_kwargs["video_backend"] = args.video_backend
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-4,
        **dataset_kwargs,
    )

    print("\n== Sample Checks ==")
    sample_count = min(args.num_samples, len(dataset))
    if sample_count == 0:
        fail("Dataset has no samples.")

    indices = torch.linspace(0, len(dataset) - 1, steps=sample_count).long().tolist()
    for idx in indices:
        item = dataset[idx]
        print(f"sample index {idx}")
        for key in args.state_keys + [args.action_key]:
            value = item[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            if not torch.isfinite(value.float()).all():
                fail(f"Non-finite values found in {key} at dataset index {idx}")
            print(f"  {key}: {tensor_summary(value)}")

        for key in args.image_keys:
            image = item[key]
            if not torch.is_tensor(image):
                image = torch.as_tensor(image)
            if image.ndim != 3:
                fail(f"{key} should be a single RGB image tensor with 3 dims, got {tuple(image.shape)}")
            if not torch.isfinite(image.float()).all():
                fail(f"Non-finite pixels found in {key} at dataset index {idx}")
            print(f"  {key}: {tensor_summary(image)}")

    print("\n[OK] Dataset looks usable for the requested training inputs/outputs.")


if __name__ == "__main__":
    main()
