"""Offline quality diagnostics for SAM2-propagated binary mask videos."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MaskQualityThresholds:
    temporal_iou_min: float = 0.60
    area_log_change_max: float = 0.45
    centroid_jump_max: float = 0.08
    max_components: int = 2
    quality_score_min: float = 0.60
    min_component_area_ratio: float = 1e-4


def binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool, copy=False)
    second = second.astype(bool, copy=False)
    union = np.logical_or(first, second).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first, second).sum() / union)


def _read_binary_frame(capture: Any, threshold: int) -> tuple[bool, np.ndarray | None]:
    ok, frame = capture.read()
    if not ok:
        return False, None
    if frame.ndim == 3:
        frame = frame.max(axis=2)
    return True, frame >= threshold


def _mask_geometry(mask: np.ndarray, min_component_pixels: int) -> tuple[float, float, float, int]:
    import cv2

    height, width = mask.shape
    area_pixels = int(mask.sum())
    area_ratio = area_pixels / float(height * width)
    if area_pixels:
        moments = cv2.moments(mask.astype(np.uint8), binaryImage=True)
        centroid_x = moments["m10"] / moments["m00"] / max(width - 1, 1)
        centroid_y = moments["m01"] / moments["m00"] / max(height - 1, 1)
    else:
        centroid_x = math.nan
        centroid_y = math.nan
    labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = sum(
        int(stats[label, cv2.CC_STAT_AREA]) >= min_component_pixels for label in range(1, labels)
    )
    return area_ratio, centroid_x, centroid_y, components


def _nan_neighbor_mean(pair_iou: np.ndarray) -> np.ndarray:
    frame_count = pair_iou.shape[0] + 1
    previous = np.full(frame_count, np.nan, dtype=np.float32)
    following = np.full(frame_count, np.nan, dtype=np.float32)
    previous[1:] = pair_iou
    following[:-1] = pair_iou
    stacked = np.stack([previous, following])
    valid_count = np.isfinite(stacked).sum(axis=0)
    total = np.nansum(stacked, axis=0)
    return np.divide(total, valid_count, out=np.ones(frame_count, dtype=np.float32), where=valid_count > 0)


def score_mask_diagnostics(
    diagnostics: dict[str, np.ndarray],
    thresholds: MaskQualityThresholds,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    temporal_score = np.clip(diagnostics["temporal_iou"], 0.0, 1.0)
    area_score = np.exp(-diagnostics["area_log_change"] / thresholds.area_log_change_max)
    centroid_score = np.exp(-diagnostics["centroid_jump"] / thresholds.centroid_jump_max)
    centroid_score = np.where(np.isfinite(centroid_score), centroid_score, 1.0)
    component_score = np.minimum(1.0, thresholds.max_components / np.maximum(diagnostics["components"], 1))
    components = {
        "temporal": temporal_score.astype(np.float32),
        "area": area_score.astype(np.float32),
        "centroid": centroid_score.astype(np.float32),
        "components": component_score.astype(np.float32),
    }
    weights = {"temporal": 0.40, "area": 0.25, "centroid": 0.25, "components": 0.10}
    weighted = np.zeros_like(temporal_score, dtype=np.float32)
    available_weight = np.zeros_like(temporal_score, dtype=np.float32)
    for name, values in components.items():
        available = np.isfinite(values)
        weighted[available] += weights[name] * values[available]
        available_weight[available] += weights[name]
    quality_score = np.divide(
        weighted,
        available_weight,
        out=np.zeros_like(weighted),
        where=available_weight > 0,
    )

    failures = {
        "low_temporal_iou": diagnostics["temporal_iou"] < thresholds.temporal_iou_min,
        "area_jump": diagnostics["area_log_change"] > thresholds.area_log_change_max,
        "centroid_jump": diagnostics["centroid_jump"] > thresholds.centroid_jump_max,
        "component_anomaly": diagnostics["components"] > thresholds.max_components,
    }
    # Single-frame anomalies can be real task motion. Treat them as diagnostics
    # contributing to the aggregate score rather than standalone hard failures.
    uncertain = quality_score < thresholds.quality_score_min
    return quality_score.astype(np.float32), uncertain, failures


def thresholds_for_mask_key(base: MaskQualityThresholds, key: str) -> MaskQualityThresholds:
    """Apply conservative class-specific motion/topology expectations."""
    suffix = key.rsplit("_", 1)[-1]
    if suffix == "occluder":
        # Cloth can be split by arm occlusion and image boundaries.
        return replace(
            base,
            temporal_iou_min=min(base.temporal_iou_min, 0.50),
            max_components=max(base.max_components, 8),
        )
    if suffix == "tool":
        # The actuator moves quickly and may contain multiple disconnected parts.
        return replace(
            base,
            temporal_iou_min=min(base.temporal_iou_min, 0.40),
            centroid_jump_max=max(base.centroid_jump_max, 0.12),
            max_components=max(base.max_components, 6),
        )
    if suffix == "region":
        return replace(base, max_components=max(base.max_components, 3))
    return base


def evaluate_mask_video(
    video_path: Path,
    *,
    thresholds: MaskQualityThresholds = MaskQualityThresholds(),
    mask_threshold: int = 127,
    start_frame: int = 0,
    frame_count: int | None = None,
) -> dict[str, np.ndarray]:
    import cv2

    cv2.setNumThreads(1)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open mask video: {video_path}")
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    area_ratio: list[float] = []
    centroid_x: list[float] = []
    centroid_y: list[float] = []
    components: list[int] = []
    pair_iou: list[float] = []
    previous: np.ndarray | None = None
    try:
        while True:
            if frame_count is not None and len(area_ratio) >= frame_count:
                break
            ok, mask = _read_binary_frame(capture, mask_threshold)
            if not ok or mask is None:
                break
            min_component_pixels = max(1, round(mask.size * thresholds.min_component_area_ratio))
            area, cx, cy, count = _mask_geometry(mask, min_component_pixels)
            area_ratio.append(area)
            centroid_x.append(cx)
            centroid_y.append(cy)
            components.append(count)
            if previous is not None:
                pair_iou.append(binary_iou(previous, mask))
            previous = mask

    finally:
        capture.release()

    if not area_ratio:
        raise ValueError(f"Mask video contains no frames: {video_path}")
    area = np.asarray(area_ratio, dtype=np.float32)
    cx = np.asarray(centroid_x, dtype=np.float32)
    cy = np.asarray(centroid_y, dtype=np.float32)
    frame_count = area.shape[0]
    area_log_change = np.zeros(frame_count, dtype=np.float32)
    centroid_jump = np.zeros(frame_count, dtype=np.float32)
    if frame_count > 1:
        area_log_change[1:] = np.abs(np.log((area[1:] + 1e-8) / (area[:-1] + 1e-8)))
        centroid_jump[1:] = np.sqrt(np.square(cx[1:] - cx[:-1]) + np.square(cy[1:] - cy[:-1])) / math.sqrt(2)

    diagnostics = {
        "area_ratio": area,
        "centroid_x": cx,
        "centroid_y": cy,
        "components": np.asarray(components, dtype=np.int16),
        "temporal_iou": _nan_neighbor_mean(np.asarray(pair_iou, dtype=np.float32)),
        "area_log_change": area_log_change,
        "centroid_jump": centroid_jump,
    }
    quality_score, uncertain, failures = score_mask_diagnostics(diagnostics, thresholds)
    return {**diagnostics, "quality_score": quality_score, "uncertain": uncertain, **failures}


def _evaluate_episode_task(
    task: tuple[int, str, MaskQualityThresholds, int, int, int],
) -> tuple[int, dict[str, np.ndarray]]:
    ep_idx, primary_path, thresholds, mask_threshold, start_frame, frame_count = task
    return ep_idx, evaluate_mask_video(
        Path(primary_path),
        thresholds=thresholds,
        mask_threshold=mask_threshold,
        start_frame=start_frame,
        frame_count=frame_count,
    )


def _safe_key(key: str) -> str:
    return key.replace("/", "__").replace(".", "_")


def evaluate_dataset(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    dataset_root = args.root.expanduser().resolve()
    output_dir = args.output_dir or dataset_root / "segmentation_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    mask_keys = args.mask_keys or sorted(
        key
        for key in meta.video_keys
        if key.rsplit("_", 1)[-1] in {"occluder", "object", "region", "tool"}
    )
    if not mask_keys:
        raise ValueError("No semantic mask video keys were found; pass --mask-keys explicitly.")

    thresholds = MaskQualityThresholds(
        temporal_iou_min=args.temporal_iou_min,
        area_log_change_max=args.area_log_change_max,
        centroid_jump_max=args.centroid_jump_max,
        max_components=args.max_components,
        quality_score_min=args.quality_score_min,
        min_component_area_ratio=args.min_component_area_ratio,
    )
    total_frames = meta.total_frames
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "base_thresholds": asdict(thresholds),
        "mask_keys": mask_keys,
        "streams": {},
        "warning": "Thresholds are diagnostic until calibrated against manually reviewed labels.",
    }
    episode_count = min(meta.total_episodes, args.max_episodes) if args.max_episodes else meta.total_episodes

    for key in mask_keys:
        key_thresholds = thresholds_for_mask_key(thresholds, key)
        arrays: dict[str, np.ndarray] = {}
        tasks = []
        for ep_idx in range(episode_count):
            episode = meta.episodes[ep_idx]
            primary_path = dataset_root / meta.get_video_file_path(ep_idx, key)
            tasks.append(
                (
                    ep_idx,
                    str(primary_path),
                    key_thresholds,
                    args.mask_threshold,
                    round(episode[f"videos/{key}/from_timestamp"] * meta.fps),
                    episode["dataset_to_index"] - episode["dataset_from_index"],
                )
            )

        executor = None
        if args.workers == 1:
            results = map(_evaluate_episode_task, tasks)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            results = executor.map(_evaluate_episode_task, tasks)
        try:
            for ep_idx, result in results:
                episode = meta.episodes[ep_idx]
                start, end = episode["dataset_from_index"], episode["dataset_to_index"]
                expected = end - start
                if result["quality_score"].shape[0] != expected:
                    raise ValueError(
                        f"Frame count mismatch for episode {ep_idx}, key '{key}': "
                        f"video={result['quality_score'].shape[0]}, metadata={expected}."
                    )
                if not arrays:
                    for name, values in result.items():
                        fill = False if values.dtype == np.bool_ else np.nan
                        dtype = values.dtype if values.dtype != np.int16 else np.float32
                        arrays[name] = np.full(total_frames, fill, dtype=dtype)
                for name, values in result.items():
                    arrays[name][start:end] = values
                if ep_idx == 0 or ep_idx + 1 == episode_count or (ep_idx + 1) % args.log_every == 0:
                    print(
                        f"[{key}] episode {ep_idx + 1}/{episode_count}: "
                        f"uncertain={result['uncertain'].mean():.1%}, "
                        f"score={result['quality_score'].mean():.3f}"
                    )
        finally:
            if executor is not None:
                executor.shutdown()

        output_path = output_dir / f"{_safe_key(key)}.npz"
        np.savez_compressed(output_path, **arrays)
        processed = np.isfinite(arrays["quality_score"])
        stream_summary = {
            "output": str(output_path),
            "thresholds": asdict(key_thresholds),
            "processed_frames": int(processed.sum()),
            "quality_score_mean": float(np.nanmean(arrays["quality_score"])),
            "uncertain_fraction": float(arrays["uncertain"][processed].mean()),
            "failure_fractions": {
                name: float(arrays[name][processed].mean())
                for name in (
                    "low_temporal_iou",
                    "area_jump",
                    "centroid_jump",
                    "component_anomaly",
                )
            },
        }
        summary["streams"][key] = stream_summary
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--mask-keys", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--temporal-iou-min", type=float, default=0.60)
    parser.add_argument("--area-log-change-max", type=float, default=0.45)
    parser.add_argument("--centroid-jump-max", type=float, default=0.08)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--quality-score-min", type=float, default=0.60)
    parser.add_argument("--min-component-area-ratio", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Smoke-test only; omit for a full report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_dataset(parse_args())
