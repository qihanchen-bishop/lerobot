#!/usr/bin/env python
"""Generate event-driven, quality-weighted SSACT-3 stage supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stage_semantic import (
    EVENT_NAMES,
    PHASE_NAMES,
    PHASE_RELATION_MASK,
    RELATION_NAMES,
    TRANSITION_NAMES,
)


SEMANTIC_CLASSES = ("occluder", "object", "region", "tool")
STATE_FEATURES = (
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


def _quality_weight(score: np.ndarray, uncertain: np.ndarray, uncertain_scale: float) -> np.ndarray:
    return np.clip(score, 0.0, 1.0) * np.where(uncertain, uncertain_scale, 1.0)


def _weighted_smooth(values: np.ndarray, weights: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32, copy=True)
    window = min(window, values.shape[0])
    kernel = np.ones(window, dtype=np.float32)
    numerator = np.convolve(values * weights, kernel, mode="same")
    denominator = np.convolve(weights, kernel, mode="same")
    return (numerator / np.maximum(denominator, 1e-6)).astype(np.float32)


def _soft_phase_labels(indices: np.ndarray, width: int) -> np.ndarray:
    probabilities = np.eye(len(PHASE_NAMES), dtype=np.float32)[indices]
    if width <= 0:
        return probabilities
    changes = np.flatnonzero(indices[1:] != indices[:-1]) + 1
    for boundary in changes:
        previous = int(indices[boundary - 1])
        following = int(indices[boundary])
        start = max(0, boundary - width)
        end = min(indices.shape[0], boundary + width + 1)
        alpha = np.linspace(0.0, 1.0, end - start, dtype=np.float32)
        probabilities[start:end] = 0.0
        probabilities[start:end, previous] = 1.0 - alpha
        probabilities[start:end, following] = alpha
    return probabilities


def _stable_state_machine(
    event_scores: np.ndarray,
    event_confidence: np.ndarray,
    *,
    on_threshold: float | np.ndarray,
    off_threshold: float | np.ndarray,
    advance_frames: int,
    rollback_frames: int,
    minimum_transition_quality: float,
) -> np.ndarray:
    on_threshold = np.broadcast_to(np.asarray(on_threshold, dtype=np.float32), (4,))
    off_threshold = np.broadcast_to(np.asarray(off_threshold, dtype=np.float32), (4,))
    visible, separated, inside, restored = event_scores.T
    visible_q, separated_q, inside_q, restored_q = event_confidence.T
    phases = np.zeros(event_scores.shape[0], dtype=np.int64)
    phase = 0
    advance_count = 0
    rollback_count = 0

    for frame in range(event_scores.shape[0]):
        advance_ready = False
        rollback_ready = False
        if phase == 0:
            advance_ready = (
                visible[frame] >= on_threshold[0]
                and visible_q[frame] >= minimum_transition_quality
            )
        elif phase == 1:
            advance_ready = (
                separated[frame] >= on_threshold[1]
                and separated_q[frame] >= minimum_transition_quality
            )
            rollback_ready = (
                visible[frame] <= off_threshold[0]
                and visible_q[frame] >= minimum_transition_quality
            )
        elif phase == 2:
            advance_ready = (
                inside[frame] >= on_threshold[2]
                and inside_q[frame] >= minimum_transition_quality
            )
            rollback_ready = (
                separated[frame] <= off_threshold[1]
                and separated_q[frame] >= minimum_transition_quality
            )
        elif phase == 3:
            advance_ready = (
                restored[frame] >= on_threshold[3]
                and restored_q[frame] >= minimum_transition_quality
            )
            rollback_ready = (
                inside[frame] <= off_threshold[2]
                and inside_q[frame] >= minimum_transition_quality
            )

        advance_count = advance_count + 1 if advance_ready else 0
        rollback_count = rollback_count + 1 if rollback_ready else 0
        if phase < len(PHASE_NAMES) - 1 and advance_count >= advance_frames:
            phase += 1
            advance_count = 0
            rollback_count = 0
        elif phase > 0 and phase < len(PHASE_NAMES) - 1 and rollback_count >= rollback_frames:
            phase -= 1
            advance_count = 0
            rollback_count = 0
        phases[frame] = phase
    return phases


def _feature_quality(class_quality: np.ndarray) -> np.ndarray:
    cloth, obj, region, tool = (class_quality[:, idx] for idx in range(4))
    return np.stack(
        [
            cloth,
            obj,
            region,
            tool,
            obj,
            obj,
            region,
            region,
            tool,
            tool,
            np.minimum(obj, region),
            np.minimum(obj, cloth),
            np.minimum(obj, region),
            np.minimum(tool, obj),
        ],
        axis=-1,
    ).astype(np.float32)


def _parse_view_reliability(values: list[str], rgb_keys: list[str]) -> np.ndarray:
    mapping = {key.rsplit(".", 1)[-1]: 1.0 for key in rgb_keys}
    for value in values:
        if "=" not in value:
            raise ValueError(f"View reliability must be VIEW=WEIGHT, got '{value}'.")
        name, raw = value.rsplit("=", 1)
        if name not in mapping:
            raise ValueError(f"Unknown view '{name}', expected one of {sorted(mapping)}.")
        mapping[name] = float(raw)
    reliability = np.asarray([mapping[key.rsplit(".", 1)[-1]] for key in rgb_keys], dtype=np.float32)
    if np.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError("View reliability values must be in [0, 1].")
    return reliability


def _class_balance(labels: np.ndarray, classes: int, maximum: float = 5.0) -> np.ndarray:
    counts = np.bincount(labels, minlength=classes).astype(np.float32)
    factors = np.sqrt(max(labels.shape[0], 1) / np.maximum(classes * counts, 1.0))
    return np.clip(factors, 0.5, maximum)


def generate(args: argparse.Namespace) -> None:
    unit_interval = {
        "uncertain_quality_scale": args.uncertain_quality_scale,
        "primary_quality_threshold": args.primary_quality_threshold,
        "minimum_transition_quality": args.minimum_transition_quality,
        "visible_on_threshold": args.visible_on_threshold,
        "separation_on_threshold": args.separation_on_threshold,
        "inside_on_threshold": args.inside_on_threshold,
        "restore_on_threshold": args.restore_on_threshold,
        "visible_off_threshold": args.visible_off_threshold,
        "separation_off_threshold": args.separation_off_threshold,
        "inside_off_threshold": args.inside_off_threshold,
        "max_incomplete_fraction": args.max_incomplete_fraction,
    }
    invalid = {name: value for name, value in unit_interval.items() if not 0.0 <= value <= 1.0}
    if invalid:
        raise ValueError(f"Stage threshold values must be in [0, 1]: {invalid}")
    if not (
        args.visible_off_threshold < args.visible_on_threshold
        and args.separation_off_threshold < args.separation_on_threshold
        and args.inside_off_threshold < args.inside_on_threshold
    ):
        raise ValueError("Each stage event off threshold must be below its on threshold.")
    positive_ratios = {
        "object_visible_ratio": args.object_visible_ratio,
        "separation_object_ratio": args.separation_object_ratio,
        "separation_contact_ratio": args.separation_contact_ratio,
        "inside_contact_ratio": args.inside_contact_ratio,
        "inside_distance_ratio": args.inside_distance_ratio,
        "done_cover_ratio": args.done_cover_ratio,
    }
    invalid_ratios = {name: value for name, value in positive_ratios.items() if value <= 0.0}
    if invalid_ratios:
        raise ValueError(f"Stage geometry ratios must be positive: {invalid_ratios}")
    if min(
        args.advance_frames,
        args.rollback_frames,
        args.smoothing_window,
    ) <= 0 or args.soft_boundary_frames < 0:
        raise ValueError("Stage persistence/smoothing must be positive and soft boundary non-negative.")

    with np.load(args.semantic_states) as archive:
        required = {
            "semantic_states",
            "quality_score",
            "uncertain",
            "episode_start_index",
            "episode_end_index",
            "processed",
            "rgb_keys",
            "class_names",
            "feature_names",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Semantic state cache is missing {missing}.")
        states = archive["semantic_states"].astype(np.float32, copy=True)
        quality_score = archive["quality_score"].astype(np.float32, copy=True)
        uncertain = archive["uncertain"].astype(bool, copy=True)
        episode_start = archive["episode_start_index"].astype(np.int64, copy=True)
        episode_end = archive["episode_end_index"].astype(np.int64, copy=True)
        processed = archive["processed"].astype(bool, copy=True)
        rgb_keys = archive["rgb_keys"].astype(str).tolist()
        class_names = archive["class_names"].astype(str).tolist()
        feature_names = archive["feature_names"].astype(str).tolist()

    if not processed.all():
        raise ValueError("Stage supervision requires a complete semantic-state cache.")
    if class_names != list(SEMANTIC_CLASSES) or feature_names != list(STATE_FEATURES):
        raise ValueError(
            f"Unexpected semantic cache definition: classes={class_names}, features={feature_names}."
        )
    if states.ndim != 3 or states.shape[1] != len(rgb_keys) or states.shape[2] != len(STATE_FEATURES):
        raise ValueError(f"Unexpected semantic state shape {states.shape}.")
    if quality_score.shape != states.shape[:2] + (len(SEMANTIC_CLASSES),):
        raise ValueError("Semantic quality shape does not align with states.")
    if args.primary_view not in [key.rsplit(".", 1)[-1] for key in rgb_keys]:
        raise ValueError(f"Primary view '{args.primary_view}' is not present in {rgb_keys}.")

    total_frames, num_views = states.shape[:2]
    primary_idx = [key.rsplit(".", 1)[-1] for key in rgb_keys].index(args.primary_view)
    view_reliability = _parse_view_reliability(args.view_reliability, rgb_keys)
    class_quality = _quality_weight(quality_score, uncertain, args.uncertain_quality_scale)
    class_quality *= view_reliability[None, :, None]
    feature_quality = np.stack(
        [_feature_quality(class_quality[:, view]) for view in range(num_views)],
        axis=1,
    )
    view_quality = class_quality.mean(axis=-1)
    # Add one online-compatible confidence feature after each view's 14 geometry values.
    semantic_features = np.concatenate([states, view_quality[..., None]], axis=-1).reshape(total_frames, -1)

    phase_index = np.zeros(total_frames, dtype=np.int64)
    phase_probabilities = np.zeros((total_frames, len(PHASE_NAMES)), dtype=np.float32)
    phase_confidence = np.zeros(total_frames, dtype=np.float32)
    event_targets = np.zeros((total_frames, len(EVENT_NAMES)), dtype=np.float32)
    event_weights = np.zeros_like(event_targets)
    progress_target = np.zeros(total_frames, dtype=np.float32)
    progress_weight = np.zeros(total_frames, dtype=np.float32)
    transition_target = np.zeros(total_frames, dtype=np.int64)
    transition_weight = np.zeros(total_frames, dtype=np.float32)
    relation_targets = np.zeros((total_frames, len(RELATION_NAMES)), dtype=np.float32)
    relation_weights = np.zeros_like(relation_targets)
    incomplete_episodes: list[int] = []
    transition_counts_by_episode: list[dict[str, int]] = []

    starts = np.flatnonzero(np.arange(total_frames) == episode_start)
    for episode_idx, start in enumerate(starts):
        end = int(episode_end[start])
        episode_states = states[start:end]
        episode_quality = class_quality[start:end]
        length = end - start

        object_area = episode_states[:, :, 1]
        object_cover_contact = episode_states[:, :, 11]
        object_region_contact = episode_states[:, :, 10]
        object_region_distance = episode_states[:, :, 12]
        cover_area = episode_states[:, :, 0]

        peak_object = max(float(object_area[:, primary_idx].max()), args.object_visible_ratio)
        visible_threshold = max(
            args.object_visible_ratio,
            args.object_visible_peak_fraction * peak_object,
        )
        separated_object_threshold = max(
            args.separation_object_ratio,
            args.separation_object_peak_fraction * peak_object,
        )

        object_q = episode_quality[:, :, 1]
        cloth_object_q = np.minimum(episode_quality[:, :, 0], object_q)
        region_object_q = np.minimum(episode_quality[:, :, 2], object_q)

        def primary_with_fallback(metric: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            values = metric[:, primary_idx].copy()
            conf = confidence[:, primary_idx].copy()
            fallback_candidates = [idx for idx in range(num_views) if idx != primary_idx]
            if fallback_candidates:
                fallback_conf = confidence[:, fallback_candidates]
                best_local = fallback_conf.argmax(axis=1)
                best_idx = np.asarray(fallback_candidates)[best_local]
                rows = np.arange(length)
                use_fallback = conf < args.primary_quality_threshold
                values[use_fallback] = metric[rows[use_fallback], best_idx[use_fallback]]
                conf[use_fallback] = confidence[rows[use_fallback], best_idx[use_fallback]]
            return values, conf

        visible_ratio, visible_q = primary_with_fallback(object_area, object_q)
        contact_cover, separated_q = primary_with_fallback(object_cover_contact, cloth_object_q)
        separation_area, _ = primary_with_fallback(object_area, object_q)
        visible_score = np.clip(visible_ratio / max(visible_threshold, 1e-6), 0.0, 1.0)
        separation_visibility = np.clip(
            separation_area / max(separated_object_threshold, 1e-6), 0.0, 1.0
        )
        separation_gap = np.clip(
            1.0 - contact_cover / max(args.separation_contact_ratio, 1e-6), 0.0, 1.0
        )
        separated_score = np.sqrt(separation_visibility * separation_gap)

        inside_contact = np.clip(
            object_region_contact[:, primary_idx] / max(args.inside_contact_ratio, 1e-6),
            0.0,
            1.0,
        )
        inside_distance = np.clip(
            1.0 - object_region_distance[:, primary_idx] / max(args.inside_distance_ratio, 1e-6),
            0.0,
            1.0,
        )
        inside_score = np.sqrt(inside_contact * inside_distance)
        inside_q = region_object_q[:, primary_idx]
        restored_score = np.clip(
            cover_area[:, primary_idx] / max(args.done_cover_ratio, 1e-6), 0.0, 1.0
        )
        restored_q = episode_quality[:, primary_idx, 0]

        positive_scores = np.stack(
            [visible_score, separated_score, inside_score, restored_score], axis=-1
        )
        positive_q = np.stack([visible_q, separated_q, inside_q, restored_q], axis=-1)
        for column in range(positive_scores.shape[1]):
            positive_scores[:, column] = _weighted_smooth(
                positive_scores[:, column], positive_q[:, column], args.smoothing_window
            )

        local_phase = _stable_state_machine(
            positive_scores,
            positive_q,
            on_threshold=np.asarray(
                [
                    args.visible_on_threshold,
                    args.separation_on_threshold,
                    args.inside_on_threshold,
                    args.restore_on_threshold,
                ]
            ),
            off_threshold=np.asarray(
                [
                    args.visible_off_threshold,
                    args.separation_off_threshold,
                    args.inside_off_threshold,
                    0.0,
                ]
            ),
            advance_frames=args.advance_frames,
            rollback_frames=args.rollback_frames,
            minimum_transition_quality=args.minimum_transition_quality,
        )
        if local_phase[-1] != len(PHASE_NAMES) - 1:
            incomplete_episodes.append(episode_idx)

        local_phase_probabilities = _soft_phase_labels(local_phase, args.soft_boundary_frames)
        phase_index[start:end] = local_phase
        phase_probabilities[start:end] = local_phase_probabilities

        phase_required_quality = np.stack(
            [
                np.minimum(episode_quality[:, primary_idx, 0], visible_q),
                separated_q,
                np.minimum.reduce(
                    [
                        episode_quality[:, primary_idx, 1],
                        episode_quality[:, primary_idx, 2],
                        episode_quality[:, primary_idx, 3],
                    ]
                ),
                restored_q,
                restored_q,
            ],
            axis=-1,
        )
        local_phase_confidence = (
            local_phase_probabilities * phase_required_quality
        ).sum(axis=-1)
        phase_confidence[start:end] = local_phase_confidence

        local_events = np.stack(
            [
                visible_score,
                separated_score,
                inside_score,
                restored_score,
                1.0 - visible_score,
                1.0 - separated_score,
                1.0 - inside_score,
                view_quality[start:end, primary_idx],
            ],
            axis=-1,
        ).astype(np.float32)
        local_event_weights = np.stack(
            [visible_q, separated_q, inside_q, restored_q, visible_q, separated_q, inside_q, np.ones(length)],
            axis=-1,
        ).astype(np.float32)
        event_targets[start:end] = local_events
        event_weights[start:end] = local_event_weights

        phase_progress = np.stack(
            [visible_score, separated_score, inside_score, restored_score, np.ones(length)], axis=-1
        )
        local_progress = (local_phase_probabilities * phase_progress).sum(axis=-1)
        progress_target[start:end] = local_progress
        progress_weight[start:end] = local_phase_confidence

        local_transition = np.full(length, TRANSITION_NAMES.index("stay"), dtype=np.int64)
        if length > 1:
            delta = local_phase[1:] - local_phase[:-1]
            local_transition[np.flatnonzero(delta > 0)] = TRANSITION_NAMES.index("advance")
            local_transition[np.flatnonzero(delta < 0)] = TRANSITION_NAMES.index("rollback")
        semantic_valid = view_quality[start:end, primary_idx]
        local_transition[semantic_valid < args.minimum_transition_quality] = TRANSITION_NAMES.index(
            "uncertain"
        )
        transition_target[start:end] = local_transition
        local_transition_weight = np.maximum(local_phase_confidence, 0.05)
        local_transition_weight[local_transition == TRANSITION_NAMES.index("uncertain")] = np.maximum(
            1.0 - semantic_valid[local_transition == TRANSITION_NAMES.index("uncertain")],
            0.25,
        )
        transition_weight[start:end] = local_transition_weight

        primary = episode_states[:, primary_idx]
        local_relations = np.stack(
            [
                primary[:, 0],
                primary[:, 1],
                primary[:, 11],
                primary[:, 13],
                primary[:, 12],
                primary[:, 10],
                primary[:, 2],
            ],
            axis=-1,
        )
        primary_feature_q = feature_quality[start:end, primary_idx]
        relation_q = np.stack(
            [
                primary_feature_q[:, 0],
                primary_feature_q[:, 1],
                primary_feature_q[:, 11],
                primary_feature_q[:, 13],
                primary_feature_q[:, 12],
                primary_feature_q[:, 10],
                primary_feature_q[:, 2],
            ],
            axis=-1,
        )
        relation_mask = local_phase_probabilities @ PHASE_RELATION_MASK.numpy()
        relation_targets[start:end] = local_relations
        relation_weights[start:end] = relation_q * relation_mask

        transition_counts_by_episode.append(
            {
                name: int((local_transition == idx).sum())
                for idx, name in enumerate(TRANSITION_NAMES)
            }
        )

    phase_balance = _class_balance(phase_index, len(PHASE_NAMES))
    phase_confidence *= phase_balance[phase_index]
    transition_balance = _class_balance(transition_target, len(TRANSITION_NAMES))
    transition_weight *= transition_balance[transition_target]

    incomplete_fraction = len(incomplete_episodes) / max(len(starts), 1)
    if incomplete_fraction > args.max_incomplete_fraction and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(incomplete_episodes)}/{len(starts)} episodes do not reach done under semantic rules. "
            "Inspect the generated summary or adjust semantic thresholds; pass --allow-incomplete only "
            "for diagnosis."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        semantic_features=semantic_features.astype(np.float32),
        phase_index=phase_index,
        phase_probabilities=phase_probabilities,
        phase_confidence=phase_confidence.astype(np.float32),
        event_targets=event_targets,
        event_weights=event_weights,
        progress_target=progress_target,
        progress_weight=progress_weight,
        transition_target=transition_target,
        transition_weight=transition_weight,
        relation_targets=relation_targets,
        relation_weights=relation_weights,
        episode_start_index=episode_start,
        episode_end_index=episode_end,
        rgb_keys=np.asarray(rgb_keys),
        phase_names=np.asarray(PHASE_NAMES),
        event_names=np.asarray(EVENT_NAMES),
        transition_names=np.asarray(TRANSITION_NAMES),
        relation_names=np.asarray(RELATION_NAMES),
    )
    summary: dict[str, Any] = {
        "semantic_states": str(args.semantic_states.resolve()),
        "output": str(args.output.resolve()),
        "total_frames": total_frames,
        "total_episodes": len(starts),
        "rgb_keys": rgb_keys,
        "primary_view": args.primary_view,
        "view_reliability": view_reliability.tolist(),
        "phase_names": list(PHASE_NAMES),
        "event_names": list(EVENT_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "relation_names": list(RELATION_NAMES),
        "phase_frame_counts": np.bincount(phase_index, minlength=len(PHASE_NAMES)).tolist(),
        "phase_balance": phase_balance.tolist(),
        "transition_frame_counts": np.bincount(
            transition_target, minlength=len(TRANSITION_NAMES)
        ).tolist(),
        "transition_balance": transition_balance.tolist(),
        "transition_counts_by_episode": transition_counts_by_episode,
        "phase_confidence_mean": float(phase_confidence.mean()),
        "event_weight_mean": event_weights.mean(axis=0).tolist(),
        "relation_weight_mean": relation_weights.mean(axis=0).tolist(),
        "incomplete_episode_indices": incomplete_episodes,
        "thresholds": {
            "object_visible_ratio": args.object_visible_ratio,
            "object_visible_peak_fraction": args.object_visible_peak_fraction,
            "separation_object_ratio": args.separation_object_ratio,
            "separation_object_peak_fraction": args.separation_object_peak_fraction,
            "separation_contact_ratio": args.separation_contact_ratio,
            "inside_contact_ratio": args.inside_contact_ratio,
            "inside_distance_ratio": args.inside_distance_ratio,
            "done_cover_ratio": args.done_cover_ratio,
            "visible_on_threshold": args.visible_on_threshold,
            "separation_on_threshold": args.separation_on_threshold,
            "inside_on_threshold": args.inside_on_threshold,
            "restore_on_threshold": args.restore_on_threshold,
            "visible_off_threshold": args.visible_off_threshold,
            "separation_off_threshold": args.separation_off_threshold,
            "inside_off_threshold": args.inside_off_threshold,
            "advance_frames": args.advance_frames,
            "rollback_frames": args.rollback_frames,
            "minimum_transition_quality": args.minimum_transition_quality,
        },
        "warning": (
            "These labels are event-derived pseudo-labels. Quality is a continuous loss weight, not ground-truth "
            "correctness. Review incomplete episodes, transition windows, and a stratified sample before training."
        ),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
    print(f"Phase counts: {summary['phase_frame_counts']}")
    print(f"Transition counts: {summary['transition_frame_counts']}")
    print(f"Incomplete episodes: {incomplete_episodes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-view", default="front")
    parser.add_argument("--view-reliability", nargs="*", default=[])
    parser.add_argument("--uncertain-quality-scale", type=float, default=0.25)
    parser.add_argument("--primary-quality-threshold", type=float, default=0.60)
    parser.add_argument("--minimum-transition-quality", type=float, default=0.45)
    parser.add_argument("--object-visible-ratio", type=float, default=0.0003)
    parser.add_argument("--object-visible-peak-fraction", type=float, default=0.10)
    parser.add_argument("--separation-object-ratio", type=float, default=0.001)
    parser.add_argument("--separation-object-peak-fraction", type=float, default=0.30)
    parser.add_argument("--separation-contact-ratio", type=float, default=0.08)
    parser.add_argument("--inside-contact-ratio", type=float, default=0.45)
    parser.add_argument("--inside-distance-ratio", type=float, default=0.18)
    parser.add_argument("--done-cover-ratio", type=float, default=0.50)
    parser.add_argument("--visible-on-threshold", type=float, default=0.80)
    parser.add_argument("--separation-on-threshold", type=float, default=0.80)
    parser.add_argument("--inside-on-threshold", type=float, default=0.65)
    parser.add_argument("--restore-on-threshold", type=float, default=0.98)
    parser.add_argument("--visible-off-threshold", type=float, default=0.35)
    parser.add_argument("--separation-off-threshold", type=float, default=0.35)
    parser.add_argument("--inside-off-threshold", type=float, default=0.35)
    parser.add_argument("--advance-frames", type=int, default=8)
    parser.add_argument("--rollback-frames", type=int, default=12)
    parser.add_argument("--smoothing-window", type=int, default=9)
    parser.add_argument("--soft-boundary-frames", type=int, default=8)
    parser.add_argument("--max-incomplete-fraction", type=float, default=0.10)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
