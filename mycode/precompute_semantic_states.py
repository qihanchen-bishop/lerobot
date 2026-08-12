#!/usr/bin/env python
"""Precompute structured semantic-state targets from fixed mask videos."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


SEMANTIC_CLASSES = ("occluder", "object", "region", "tool")
FEATURES_PER_VIEW = (
    "area_occluder",
    "area_object",
    "area_region",
    "area_tool",
    "x_object",
    "y_object",
    "x_region",
    "y_region",
    "x_tool",
    "y_tool",
    "contact_object_region",
    "contact_object_occluder",
    "distance_object_region",
    "distance_tool_object",
)


def _safe_key(key: str) -> str:
    return key.replace("/", "__").replace(".", "_")


def _read_binary_frame(capture: Any, threshold: float) -> tuple[bool, np.ndarray | None]:
    ok, frame = capture.read()
    if not ok:
        return False, None
    if frame.ndim == 3:
        frame = frame.astype(np.float32).mean(axis=2)
    return True, frame >= threshold


def semantic_state_from_masks(
    masks: np.ndarray,
    *,
    contact_radius_ratio: float = 0.03,
    eps: float = 1e-6,
) -> np.ndarray:
    """Match `SoftSemanticStateExtractor(include_confidence=False)` for hard masks."""
    import cv2

    if masks.ndim != 3 or masks.shape[0] != len(SEMANTIC_CLASSES):
        raise ValueError(f"Expected masks (4, H, W), got {tuple(masks.shape)}.")
    masks = masks.astype(bool, copy=False)
    _, height, width = masks.shape
    areas = masks.mean(axis=(1, 2), dtype=np.float64)

    centroids: dict[str, tuple[float, float]] = {}
    class_indices = {name: idx for idx, name in enumerate(SEMANTIC_CLASSES)}
    for name in ("object", "region", "tool"):
        mask = masks[class_indices[name]].astype(np.uint8)
        moments = cv2.moments(mask, binaryImage=True)
        mass = max(float(moments["m00"]), eps)
        centroids[name] = (
            float(moments["m10"] / mass / max(width - 1, 1)),
            float(moments["m01"] / mass / max(height - 1, 1)),
        )

    radius = round(min(height, width) * contact_radius_ratio)
    kernel = (
        np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        if radius > 0
        else None
    )
    object_mask = masks[class_indices["object"]]
    object_mass = max(float(object_mask.sum()), eps)

    def contact(other_name: str) -> float:
        other = masks[class_indices[other_name]].astype(np.uint8)
        if kernel is not None:
            other = cv2.dilate(other, kernel, iterations=1)
        return float((object_mask & other.astype(bool)).sum() / object_mass)

    def distance(first: str, second: str) -> float:
        first_x, first_y = centroids[first]
        second_x, second_y = centroids[second]
        return math.sqrt((first_x - second_x) ** 2 + (first_y - second_y) ** 2) / math.sqrt(2)

    return np.asarray(
        [
            *areas,
            *centroids["object"],
            *centroids["region"],
            *centroids["tool"],
            contact("region"),
            contact("occluder"),
            distance("object", "region"),
            distance("tool", "object"),
        ],
        dtype=np.float32,
    )


def _precompute_episode_view(
    task: tuple[
        int,
        int,
        tuple[str, ...],
        tuple[int, ...],
        int,
        float,
        float,
    ],
) -> tuple[int, int, np.ndarray, int]:
    import cv2

    cv2.setNumThreads(1)
    episode_idx, view_idx, paths, start_frames, frame_count, mask_threshold, contact_radius_ratio = task
    captures = [cv2.VideoCapture(path) for path in paths]
    for path, capture, start_frame in zip(paths, captures, start_frames, strict=True):
        if not capture.isOpened():
            raise RuntimeError(f"Could not open semantic mask video: {path}")
        if start_frame:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    states = np.empty((frame_count, len(FEATURES_PER_VIEW)), dtype=np.float32)
    overlap_pixels = 0
    try:
        for frame_idx in range(frame_count):
            masks = []
            for path, capture in zip(paths, captures, strict=True):
                ok, mask = _read_binary_frame(capture, mask_threshold)
                if not ok or mask is None:
                    raise RuntimeError(
                        f"Mask video ended early at local frame {frame_idx}/{frame_count}: {path}"
                    )
                masks.append(mask)
            stacked = np.stack(masks)
            overlap_pixels += int((stacked.sum(axis=0) > 1).sum())
            states[frame_idx] = semantic_state_from_masks(
                stacked,
                contact_radius_ratio=contact_radius_ratio,
            )
    finally:
        for capture in captures:
            capture.release()
    return episode_idx, view_idx, states, overlap_pixels


def _load_quality(
    quality_dir: Path,
    mask_key: str,
    total_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = quality_dir / f"{_safe_key(mask_key)}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask quality report: {path}")
    with np.load(path) as report:
        if "quality_score" not in report or "uncertain" not in report:
            raise KeyError(f"Quality report lacks quality_score/uncertain: {path}")
        score = report["quality_score"].astype(np.float32, copy=True)
        uncertain = report["uncertain"].astype(bool, copy=True)
    if score.shape != (total_frames,) or uncertain.shape != (total_frames,):
        raise ValueError(f"Quality report '{path}' does not cover all {total_frames} frames.")
    if not np.isfinite(score).all():
        raise ValueError(f"Quality report '{path}' is incomplete.")
    return score, uncertain


def precompute(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1.")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be at least 1.")
    if not 0.0 <= args.contact_radius_ratio <= 0.5:
        raise ValueError("--contact-radius-ratio must be in [0, 0.5].")

    root = args.root.expanduser().resolve()
    quality_dir = (args.quality_dir or root / "segmentation_quality").expanduser().resolve()
    meta = LeRobotDatasetMetadata(args.repo_id, root=root)
    episode_count = min(meta.total_episodes, args.max_episodes) if args.max_episodes else meta.total_episodes
    total_frames = meta.total_frames
    semantic_states = np.full(
        (total_frames, len(args.rgb_keys), len(FEATURES_PER_VIEW)),
        np.nan,
        dtype=np.float32,
    )
    episode_start_index = np.zeros(total_frames, dtype=np.int64)
    episode_end_index = np.zeros(total_frames, dtype=np.int64)

    tasks = []
    for episode_idx in range(episode_count):
        episode = meta.episodes[episode_idx]
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        episode_start_index[start:end] = start
        episode_end_index[start:end] = end
        for view_idx, rgb_key in enumerate(args.rgb_keys):
            mask_keys = tuple(f"{rgb_key}_{class_name}" for class_name in SEMANTIC_CLASSES)
            paths = tuple(str(root / meta.get_video_file_path(episode_idx, key)) for key in mask_keys)
            start_frames = tuple(
                round(episode[f"videos/{key}/from_timestamp"] * meta.fps) for key in mask_keys
            )
            tasks.append(
                (
                    episode_idx,
                    view_idx,
                    paths,
                    start_frames,
                    end - start,
                    args.mask_threshold,
                    args.contact_radius_ratio,
                )
            )

    executor = None
    if args.workers == 1:
        results = map(_precompute_episode_view, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(_precompute_episode_view, tasks)
    total_overlap_pixels = 0
    try:
        for completed, (episode_idx, view_idx, states, overlap_pixels) in enumerate(results, start=1):
            episode = meta.episodes[episode_idx]
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            semantic_states[start:end, view_idx] = states
            total_overlap_pixels += overlap_pixels
            if completed == 1 or completed == len(tasks) or completed % args.log_every == 0:
                print(
                    f"semantic states {completed}/{len(tasks)}: episode={episode_idx}, "
                    f"view={args.rgb_keys[view_idx]}, overlap_pixels={overlap_pixels}"
                )
    finally:
        if executor is not None:
            executor.shutdown()

    quality_score = np.full(
        (total_frames, len(args.rgb_keys), len(SEMANTIC_CLASSES)),
        np.nan,
        dtype=np.float32,
    )
    uncertain = np.ones_like(quality_score, dtype=bool)
    for view_idx, rgb_key in enumerate(args.rgb_keys):
        for class_idx, class_name in enumerate(SEMANTIC_CLASSES):
            score, class_uncertain = _load_quality(
                quality_dir,
                f"{rgb_key}_{class_name}",
                total_frames,
            )
            quality_score[:, view_idx, class_idx] = score
            uncertain[:, view_idx, class_idx] = class_uncertain

    processed = np.isfinite(semantic_states).all(axis=(1, 2))
    if episode_count == meta.total_episodes and not processed.all():
        raise RuntimeError(f"Full semantic precomputation left {(~processed).sum()} frames incomplete.")
    if total_overlap_pixels:
        raise ValueError(
            f"Semantic labels contain {total_overlap_pixels} overlapping foreground pixels; "
            "multiclass semantic targets must be mutually exclusive."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        semantic_states=semantic_states,
        quality_score=quality_score,
        uncertain=uncertain,
        episode_start_index=episode_start_index,
        episode_end_index=episode_end_index,
        processed=processed,
        rgb_keys=np.asarray(args.rgb_keys),
        class_names=np.asarray(SEMANTIC_CLASSES),
        feature_names=np.asarray(FEATURES_PER_VIEW),
    )
    summary = {
        "repo_id": args.repo_id,
        "root": str(root),
        "quality_dir": str(quality_dir),
        "output": str(args.output.resolve()),
        "total_frames": total_frames,
        "processed_frames": int(processed.sum()),
        "total_episodes": meta.total_episodes,
        "processed_episodes": episode_count,
        "fps": meta.fps,
        "rgb_keys": args.rgb_keys,
        "class_names": list(SEMANTIC_CLASSES),
        "feature_names_per_view": list(FEATURES_PER_VIEW),
        "state_shape": list(semantic_states.shape),
        "mask_threshold": args.mask_threshold,
        "contact_radius_ratio": args.contact_radius_ratio,
        "overlap_pixels": total_overlap_pixels,
        "state_min": np.nanmin(semantic_states, axis=(0, 1)).tolist(),
        "state_max": np.nanmax(semantic_states, axis=(0, 1)).tolist(),
        "quality_score_mean": np.nanmean(quality_score, axis=0).tolist(),
        "uncertain_fraction": uncertain[processed].mean(axis=0).tolist(),
        "warning": "Targets are valid only for the listed view order and semantic-state definition.",
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--quality-dir", type=Path, default=None)
    parser.add_argument("--rgb-keys", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=float, default=127.5)
    parser.add_argument("--contact-radius-ratio", type=float, default=0.03)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-episodes", type=int, default=None, help="Smoke-test only.")
    return parser.parse_args()


if __name__ == "__main__":
    precompute(parse_args())
