#!/usr/bin/env python
"""Learn ordered five-phase soft labels from offline semantic diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


PHASE_NAMES = ("uncover", "expose", "transport", "restore", "done")
SEMANTIC_CLASSES = ("occluder", "object", "region", "tool")


def _safe_key(key: str) -> str:
    return key.replace("/", "__").replace(".", "_")


def _parse_reliability_overrides(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Reliability override must be KEY=VALUE, got '{value}'.")
        key, raw_weight = value.rsplit("=", 1)
        weight = float(raw_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Reliability for '{key}' must be in [0, 1], got {weight}.")
        overrides[key] = weight
    return overrides


def _interpolate_finite(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    positions = np.arange(values.shape[0])
    values[~finite] = np.interp(positions[~finite], positions[finite], values[finite])
    return values


def _smooth_features(features: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return features.copy()
    window = min(window, features.shape[0])
    kernel = np.ones(window, dtype=np.float32) / window
    return np.stack(
        [np.convolve(features[:, idx], kernel, mode="same") for idx in range(features.shape[1])],
        axis=1,
    )


def _view_feature_names(view: str) -> list[str]:
    prefix = f"{view}/"
    return [
        *(prefix + f"area_{name}" for name in SEMANTIC_CLASSES),
        prefix + "x_object",
        prefix + "y_object",
        prefix + "x_region",
        prefix + "y_region",
        prefix + "x_tool",
        prefix + "y_tool",
        prefix + "distance_object_region",
        prefix + "distance_tool_object",
    ]


def _load_report(quality_dir: Path, key: str, total_frames: int) -> dict[str, np.ndarray]:
    path = quality_dir / f"{_safe_key(key)}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing semantic quality report: {path}")
    with np.load(path) as report:
        required = ("area_ratio", "centroid_x", "centroid_y", "quality_score", "uncertain")
        missing = [name for name in required if name not in report]
        if missing:
            raise KeyError(f"Quality report '{path}' is missing {missing}.")
        arrays = {name: report[name].copy() for name in required}
    if any(values.shape != (total_frames,) for values in arrays.values()):
        raise ValueError(f"Quality report '{path}' does not cover all {total_frames} frames.")
    return arrays


def build_semantic_features(
    meta: LeRobotDatasetMetadata,
    quality_dir: Path,
    rgb_keys: list[str],
    reliability_overrides: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    total_frames = meta.total_frames
    reports: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    reliability = np.ones((len(rgb_keys), len(SEMANTIC_CLASSES)), dtype=np.float32)
    for view_idx, rgb_key in enumerate(rgb_keys):
        for class_idx, class_name in enumerate(SEMANTIC_CLASSES):
            key = f"{rgb_key}_{class_name}"
            reports[(view_idx, class_name)] = _load_report(quality_dir, key, total_frames)
            reliability[view_idx, class_idx] = reliability_overrides.get(
                key,
                reliability_overrides.get(f"{rgb_key.rsplit('.', 1)[-1]}_{class_name}", 1.0),
            )

    features = np.zeros((total_frames, len(rgb_keys) * 12), dtype=np.float32)
    confidence = np.zeros_like(features)
    feature_names: list[str] = []
    for view_idx, rgb_key in enumerate(rgb_keys):
        view_name = rgb_key.rsplit(".", 1)[-1]
        feature_names.extend(_view_feature_names(view_name))
        column = view_idx * 12
        class_scores: dict[str, np.ndarray] = {}
        centroids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for class_idx, class_name in enumerate(SEMANTIC_CLASSES):
            report = reports[(view_idx, class_name)]
            gate = reliability[view_idx, class_idx]
            score = np.clip(report["quality_score"], 0.0, 1.0)
            score = np.where(report["uncertain"], 0.0, score).astype(np.float32)
            class_scores[class_name] = score
            features[:, column + class_idx] = report["area_ratio"] * gate
            confidence[:, column + class_idx] = score
            if class_name in {"object", "region", "tool"}:
                centroids[class_name] = (
                    report["centroid_x"].astype(np.float32, copy=True),
                    report["centroid_y"].astype(np.float32, copy=True),
                )

        for episode in meta.episodes:
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            for class_name in ("object", "region", "tool"):
                cx, cy = centroids[class_name]
                cx[start:end] = _interpolate_finite(cx[start:end])
                cy[start:end] = _interpolate_finite(cy[start:end])

        centroid_columns = {"object": 4, "region": 6, "tool": 8}
        class_indices = {name: idx for idx, name in enumerate(SEMANTIC_CLASSES)}
        for class_name, offset in centroid_columns.items():
            cx, cy = centroids[class_name]
            gate = reliability[view_idx, class_indices[class_name]]
            features[:, column + offset] = cx * gate
            features[:, column + offset + 1] = cy * gate
            confidence[:, column + offset : column + offset + 2] = class_scores[class_name][:, None]

        def add_distance(first: str, second: str, offset: int) -> None:
            first_x, first_y = centroids[first]
            second_x, second_y = centroids[second]
            distance = np.sqrt((first_x - second_x) ** 2 + (first_y - second_y) ** 2) / np.sqrt(2.0)
            gate = min(
                reliability[view_idx, class_indices[first]],
                reliability[view_idx, class_indices[second]],
            )
            features[:, column + offset] = distance * gate
            confidence[:, column + offset] = np.minimum(class_scores[first], class_scores[second])

        add_distance("object", "region", 10)
        add_distance("tool", "object", 11)

    return features, confidence, feature_names, reliability


def _segment_episode(
    standardized: np.ndarray,
    frame_confidence: np.ndarray,
    centroids: np.ndarray,
    min_duration: int,
    time_prior_weight: float,
) -> tuple[np.ndarray, float]:
    frame_count = standardized.shape[0]
    phase_count = centroids.shape[0]
    min_duration = min(min_duration, max(1, frame_count // phase_count))
    progress = (np.arange(frame_count, dtype=np.float32) + 0.5) / frame_count
    phase_centers = (np.arange(phase_count, dtype=np.float32) + 0.5) / phase_count
    semantic_cost = np.mean((standardized[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    time_cost = (progress[:, None] - phase_centers[None, :]) ** 2
    confidence = np.clip(frame_confidence[:, None], 0.05, 1.0)
    emission = confidence * semantic_cost + time_prior_weight * time_cost
    prefix = np.concatenate([np.zeros((1, phase_count)), np.cumsum(emission, axis=0)], axis=0)

    infinity = np.inf
    dp = np.full((phase_count + 1, frame_count + 1), infinity, dtype=np.float64)
    back = np.full((phase_count + 1, frame_count + 1), -1, dtype=np.int32)
    dp[0, 0] = 0.0
    for phase in range(1, phase_count + 1):
        min_end = phase * min_duration
        max_end = frame_count - (phase_count - phase) * min_duration
        for end in range(min_end, max_end + 1):
            starts = np.arange((phase - 1) * min_duration, end - min_duration + 1)
            previous = dp[phase - 1, starts]
            valid = np.isfinite(previous)
            if not valid.any():
                continue
            candidates = previous[valid] + prefix[end, phase - 1] - prefix[starts[valid], phase - 1]
            best_idx = int(np.argmin(candidates))
            dp[phase, end] = candidates[best_idx]
            back[phase, end] = int(starts[valid][best_idx])

    if not np.isfinite(dp[phase_count, frame_count]):
        raise RuntimeError(
            f"No valid five-phase segmentation for {frame_count} frames with min_duration={min_duration}."
        )
    boundaries = np.empty(phase_count - 1, dtype=np.int32)
    end = frame_count
    for phase in range(phase_count, 0, -1):
        start = back[phase, end]
        if phase > 1:
            boundaries[phase - 2] = start
        end = start
    return boundaries, float(dp[phase_count, frame_count] / frame_count)


def _labels_from_boundaries(frame_count: int, boundaries: np.ndarray) -> np.ndarray:
    return np.searchsorted(boundaries, np.arange(frame_count), side="right").astype(np.uint8)


def _soft_labels(frame_count: int, boundaries: np.ndarray, width: int) -> np.ndarray:
    labels = _labels_from_boundaries(frame_count, boundaries)
    probabilities = np.eye(len(PHASE_NAMES), dtype=np.float32)[labels]
    if width <= 0:
        return probabilities
    for phase, boundary in enumerate(boundaries):
        start = max(0, int(boundary) - width)
        end = min(frame_count, int(boundary) + width + 1)
        positions = np.arange(start, end, dtype=np.float32)
        right = np.clip((positions - (boundary - width)) / max(2 * width, 1), 0.0, 1.0)
        probabilities[start:end] = 0.0
        probabilities[start:end, phase] = 1.0 - right
        probabilities[start:end, phase + 1] = right
    return probabilities


def learn_phase_labels(args: argparse.Namespace) -> None:
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    quality_dir = args.quality_dir or args.root / "segmentation_quality"
    overrides = _parse_reliability_overrides(args.reliability)
    features, feature_confidence, feature_names, reliability = build_semantic_features(
        meta,
        quality_dir,
        args.rgb_keys,
        overrides,
    )

    smoothed = np.zeros_like(features)
    velocity = np.zeros_like(features)
    frame_confidence = np.zeros(meta.total_frames, dtype=np.float32)
    for episode in meta.episodes:
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        smoothed[start:end] = _smooth_features(features[start:end], args.smoothing_window)
        velocity[start + 1 : end] = smoothed[start + 1 : end] - smoothed[start : end - 1]
        frame_confidence[start:end] = feature_confidence[start:end].mean(axis=1)

    clustering_features = np.concatenate([smoothed, velocity * args.velocity_scale], axis=1)
    median = np.median(clustering_features, axis=0)
    lower, upper = np.percentile(clustering_features, [10, 90], axis=0)
    scale = np.maximum(upper - lower, 1e-4)
    standardized = np.clip((clustering_features - median) / scale, -5.0, 5.0).astype(np.float32)

    boundaries_by_episode = []
    labels = np.zeros(meta.total_frames, dtype=np.uint8)
    for episode in meta.episodes:
        length = episode["dataset_to_index"] - episode["dataset_from_index"]
        boundaries = np.rint(np.arange(1, len(PHASE_NAMES)) * length / len(PHASE_NAMES)).astype(np.int32)
        boundaries_by_episode.append(boundaries)
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        labels[start:end] = _labels_from_boundaries(length, boundaries)

    iteration_changes = []
    centroids = np.zeros((len(PHASE_NAMES), standardized.shape[1]), dtype=np.float32)
    for iteration in range(args.iterations):
        for phase in range(len(PHASE_NAMES)):
            selected = labels == phase
            weights = np.clip(frame_confidence[selected], 0.05, 1.0)
            centroids[phase] = np.average(standardized[selected], axis=0, weights=weights)

        changed = 0
        new_labels = np.zeros_like(labels)
        new_boundaries = []
        for episode_idx, episode in enumerate(meta.episodes):
            start, end = episode["dataset_from_index"], episode["dataset_to_index"]
            boundaries, _ = _segment_episode(
                standardized[start:end],
                frame_confidence[start:end],
                centroids,
                args.min_phase_frames,
                args.time_prior_weight,
            )
            changed += int(np.abs(boundaries - boundaries_by_episode[episode_idx]).sum())
            new_boundaries.append(boundaries)
            new_labels[start:end] = _labels_from_boundaries(end - start, boundaries)
        boundaries_by_episode = new_boundaries
        labels = new_labels
        iteration_changes.append(changed)
        print(f"phase EM {iteration + 1}/{args.iterations}: total boundary movement={changed} frames")
        if changed == 0:
            break

    phase_probabilities = np.zeros((meta.total_frames, len(PHASE_NAMES)), dtype=np.float32)
    phase_confidence = np.zeros(meta.total_frames, dtype=np.float32)
    episode_confidence = np.zeros(meta.total_episodes, dtype=np.float32)
    episode_cost = np.zeros(meta.total_episodes, dtype=np.float32)
    for episode_idx, episode in enumerate(meta.episodes):
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        boundaries = boundaries_by_episode[episode_idx]
        phase_probabilities[start:end] = _soft_labels(end - start, boundaries, args.soft_boundary_frames)
        _, cost = _segment_episode(
            standardized[start:end],
            frame_confidence[start:end],
            centroids,
            args.min_phase_frames,
            args.time_prior_weight,
        )
        episode_cost[episode_idx] = cost
        semantic_match = float(np.exp(-min(cost, 20.0)))
        episode_confidence[episode_idx] = np.sqrt(
            max(float(frame_confidence[start:end].mean()), 0.0) * semantic_match
        )
        phase_confidence[start:end] = frame_confidence[start:end] * episode_confidence[episode_idx]

    boundary_array = np.stack(boundaries_by_episode)
    episode_lengths = np.asarray([episode["length"] for episode in meta.episodes], dtype=np.int32)
    durations = np.diff(
        np.concatenate(
            [
                np.zeros((meta.total_episodes, 1), dtype=np.int32),
                boundary_array,
                episode_lengths[:, None],
            ],
            axis=1,
        ),
        axis=1,
    )
    minimum_duration_fraction = (durations <= args.min_phase_frames).mean(axis=0)
    degenerate_phases = np.flatnonzero(minimum_duration_fraction >= 0.8).tolist()
    if degenerate_phases and not args.allow_degenerate_phases:
        names = [PHASE_NAMES[idx] for idx in degenerate_phases]
        raise RuntimeError(
            "Ordered phase learning collapsed phases to the minimum duration in at least 80% of episodes: "
            f"{names}. Increase --time-prior-weight, improve semantic features, or pass "
            "--allow-degenerate-phases only for diagnosis."
        )
    episode_start = np.zeros(meta.total_frames, dtype=np.int64)
    for episode in meta.episodes:
        start, end = episode["dataset_from_index"], episode["dataset_to_index"]
        episode_start[start:end] = start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        phase_probabilities=phase_probabilities,
        phase_index=labels,
        phase_confidence=phase_confidence,
        semantic_features=features,
        episode_start_index=episode_start,
        episode_boundaries=boundary_array,
        episode_confidence=episode_confidence,
        feature_reliability=reliability,
    )
    boundary_fractions = boundary_array / episode_lengths.astype(np.float32)[:, None]
    summary: dict[str, Any] = {
        "repo_id": args.repo_id,
        "root": str(args.root.resolve()),
        "quality_dir": str(quality_dir.resolve()),
        "output": str(args.output.resolve()),
        "phase_names": list(PHASE_NAMES),
        "rgb_keys": args.rgb_keys,
        "feature_names": feature_names,
        "feature_reliability": reliability.tolist(),
        "reliability_overrides": overrides,
        "iterations_completed": len(iteration_changes),
        "boundary_movement_by_iteration": iteration_changes,
        "boundary_fraction_mean": boundary_fractions.mean(axis=0).tolist(),
        "boundary_fraction_std": boundary_fractions.std(axis=0).tolist(),
        "phase_frame_counts": np.bincount(labels, minlength=len(PHASE_NAMES)).tolist(),
        "minimum_duration_fraction_by_phase": minimum_duration_fraction.tolist(),
        "degenerate_phase_indices": degenerate_phases,
        "phase_confidence_mean": float(phase_confidence.mean()),
        "episode_confidence_mean": float(episode_confidence.mean()),
        "low_confidence_episode_indices": np.flatnonzero(
            episode_confidence < args.low_confidence_threshold
        ).tolist(),
        "warning": (
            "These are learned pseudo-labels, not ground truth. Review low-confidence and a stratified "
            "sample of remaining trajectories before interpreting phase names semantically."
        ),
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
    parser.add_argument(
        "--reliability",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Static view/class reliability, for example side_tool=0.25.",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--min-phase-frames", type=int, default=30)
    parser.add_argument("--soft-boundary-frames", type=int, default=15)
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--velocity-scale", type=float, default=10.0)
    parser.add_argument("--time-prior-weight", type=float, default=5.0)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.65)
    parser.add_argument("--allow-degenerate-phases", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    learn_phase_labels(parse_args())
