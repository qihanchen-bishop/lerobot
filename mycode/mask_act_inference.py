#!/usr/bin/env python
"""Load custom Mask-ACT training checkpoints for live policy evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors

from train_mask_act_policy import (
    ACTIONSEM_EXPERIMENTS,
    FROZEN_SEMANTIC_EXPERIMENTS,
    MaskACTPolicy,
    SEMANTIC_EXPERIMENTS,
    VIEW_FUSION_EXPERIMENTS,
    VIEWFUS_FUSED_FRONT_KEY,
    VIEW_SPECIFIC_PRETRAINED_SEGMENTER_EXPERIMENTS,
    act_image_keys_for_experiment,
    make_policy,
    reshape_visual_stats_for_channel_first,
)
from train_lerobot_policy import make_filtered_dataset_view


VIEWFUS_INFERENCE_STATS_FILENAME = "viewfus_inference_stats.json"


def _load_mask_act_state_dict_compatibly(
    model: MaskACTPolicy,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    """Load learned weights strictly while allowing newly derived palette buffers."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing = set(getattr(model, "_semantic_palette_buffer_names", ()))
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    invalid_missing = missing - allowed_missing
    if invalid_missing or unexpected:
        problems = []
        if invalid_missing:
            problems.append(f"missing learned/state keys: {sorted(invalid_missing)}")
        if unexpected:
            problems.append(f"unexpected checkpoint keys: {sorted(unexpected)}")
        raise RuntimeError("Mask-ACT checkpoint is incompatible: " + "; ".join(problems))
    return sorted(missing)


def resolve_mask_act_checkpoint(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        if requested.name != "training_state.pt":
            raise ValueError(f"Expected training_state.pt, got: {requested}")
        checkpoint_dir = requested.parent
    elif (requested / "training_state.pt").is_file():
        checkpoint_dir = requested
    else:
        candidates = sorted(requested.glob("checkpoint_step_*/training_state.pt"))
        if not candidates:
            raise ValueError(
                "Mask-ACT path must be a training_state.pt file, a checkpoint_step_* directory, "
                "or a run directory containing checkpoint_step_*."
            )
        checkpoint_dir = candidates[-1].parent

    run_dir = checkpoint_dir.parent
    while run_dir != run_dir.parent and not (run_dir / "mask_act_run_config.json").is_file():
        run_dir = run_dir.parent
    if not (run_dir / "mask_act_run_config.json").is_file():
        raise ValueError(f"Could not find mask_act_run_config.json above {checkpoint_dir}")
    return checkpoint_dir, run_dir


def is_mask_act_checkpoint(path: str | Path) -> bool:
    try:
        resolve_mask_act_checkpoint(path)
    except (OSError, ValueError):
        return False
    return True


def _metadata_has_features(root: Path, required_features: set[str]) -> bool:
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    if not info_path.is_file() or not stats_path.is_file():
        return False
    try:
        features = set(json.loads(info_path.read_text()).get("features", {}))
    except (OSError, json.JSONDecodeError):
        return False
    return required_features.issubset(features)


def _find_metadata_root(run_cfg: dict[str, Any], project_root: Path) -> Path:
    rgb_keys = run_cfg.get("rgb_keys") or [run_cfg["rgb_key"]]
    required = {
        "action",
        *rgb_keys,
        *run_cfg["state_keys"],
    }
    # SEM policies predict their masks from RGB at inference. Their ACT
    # pre/post-processors only require the source RGB/state/action statistics.
    experiment = str(run_cfg.get("experiment", "")).upper()
    if experiment not in SEMANTIC_EXPERIMENTS | VIEW_FUSION_EXPERIMENTS:
        required.update(run_cfg["mask_target_keys"])
    configured = Path(run_cfg["root"]).expanduser()
    direct_candidates = [
        configured,
        project_root / configured,
        project_root / "data" / run_cfg["repo_id"],
        project_root / "data" / configured.name,
    ]
    for candidate in direct_candidates:
        if _metadata_has_features(candidate, required):
            return candidate.resolve()

    matches: list[Path] = []
    for search_root in (project_root / "data", project_root / "outputs" / "train" / "dataset_views"):
        if not search_root.exists():
            continue
        for info_path in search_root.glob("**/meta/info.json"):
            candidate = info_path.parent.parent
            if _metadata_has_features(candidate, required):
                matches.append(candidate)
    if matches:
        matches.sort(
            key=lambda path: (
                run_cfg["repo_id"] not in path.as_posix(),
                configured.name not in path.as_posix(),
                len(path.as_posix()),
            )
        )
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not locate dataset metadata required to rebuild Mask-ACT. "
        f"Need features {sorted(required)}. Original training root was {configured}."
    )


def _inject_viewfusion_inference_metadata(
    meta: LeRobotDatasetMetadata,
    run_cfg: dict[str, Any],
    run_dir: Path,
    project_root: Path,
) -> str:
    """Restore the derived ViewFus input spec when only base dataset metadata is local."""
    if VIEWFUS_FUSED_FRONT_KEY in meta.features and VIEWFUS_FUSED_FRONT_KEY in meta.stats:
        return "dataset_metadata"

    rgb_keys = list(run_cfg.get("rgb_keys") or [])
    if not rgb_keys or rgb_keys[0] not in meta.features or rgb_keys[0] not in meta.stats:
        raise KeyError("ViewFus-v1 requires front RGB feature metadata and statistics.")

    stats_candidates = (
        run_dir / VIEWFUS_INFERENCE_STATS_FILENAME,
        project_root / "mycode" / "viewfus_inference_stats" / f"{run_dir.name}.json",
    )
    stats_path = next((path for path in stats_candidates if path.is_file()), None)
    if stats_path is None:
        raise FileNotFoundError(
            f"ViewFus-v1 metadata is missing '{VIEWFUS_FUSED_FRONT_KEY}', and no calibrated "
            "inference statistics were found. Checked: "
            + ", ".join(str(path) for path in stats_candidates)
        )
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read ViewFus-v1 inference statistics: {stats_path}") from exc

    fused_stats = payload.get("stats")
    if not isinstance(fused_stats, dict):
        raise ValueError(f"ViewFus-v1 inference statistics must contain a 'stats' object: {stats_path}")
    for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
        values = fused_stats.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"ViewFus-v1 statistic '{key}' must contain three channels: {stats_path}")
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError(f"ViewFus-v1 statistic '{key}' contains a non-numeric value: {stats_path}")
    if any(float(value) <= 0 for value in fused_stats["std"]):
        raise ValueError(f"ViewFus-v1 standard deviations must be positive: {stats_path}")

    meta.info["features"][VIEWFUS_FUSED_FRONT_KEY] = deepcopy(meta.features[rgb_keys[0]])
    restored_stats = deepcopy(meta.stats[rgb_keys[0]])
    for key, values in fused_stats.items():
        restored_stats[key] = deepcopy(values)
    meta.stats[VIEWFUS_FUSED_FRONT_KEY] = restored_stats
    return str(stats_path.resolve())


def load_mask_act_for_inference(
    checkpoint_path: str | Path,
    project_root: str | Path,
) -> tuple[MaskACTPolicy, Any, Any, dict[str, Any]]:
    checkpoint_dir, run_dir = resolve_mask_act_checkpoint(checkpoint_path)
    run_cfg = json.loads((run_dir / "mask_act_run_config.json").read_text())
    project_root = Path(project_root).expanduser().resolve()
    metadata_root = _find_metadata_root(run_cfg, project_root)

    args = argparse.Namespace(**run_cfg)
    args.image_size = tuple(args.image_size) if args.image_size is not None else None
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Loading the checkpoint replaces these weights, so avoid an unnecessary network download.
    args.pretrained_backbone_weights = None
    if str(args.experiment).upper() in {
        *FROZEN_SEMANTIC_EXPERIMENTS,
        *ACTIONSEM_EXPERIMENTS,
    }:
        if str(args.experiment).upper() in VIEW_SPECIFIC_PRETRAINED_SEGMENTER_EXPERIMENTS:
            configured = [
                Path(path).expanduser()
                for path in getattr(args, "pretrained_segmentation_checkpoints", []) or []
            ]
            if len(configured) != len(args.rgb_keys) or any(not path.is_file() for path in configured):
                model_root = metadata_root / "models"
                fallback_candidates_by_view = {
                    "observation.images.front": (
                        model_root / "unet_front_v4_r1" / "best.pt",
                        project_root / "mycode" / "tool" / "unet_front_v4_r1" / "best.pt",
                    ),
                    "observation.images.side": (
                        model_root / "unet_side" / "best.pt",
                        project_root / "mycode" / "tool" / "unet_side" / "best.pt",
                    ),
                }
                configured = [
                    next(
                        (path.resolve() for path in fallback_candidates_by_view[key] if path.is_file()),
                        fallback_candidates_by_view[key][-1],
                    )
                    for key in args.rgb_keys
                ]
                missing_views = [
                    key for key, path in zip(args.rgb_keys, configured, strict=True) if not path.is_file()
                ]
                if missing_views:
                    raise FileNotFoundError(
                        "View-specific segmentation checkpoints are unavailable at the recorded "
                        "paths, dataset model package, and bundled project model package; missing views "
                        f"{missing_views}."
                    )
            args.pretrained_segmentation_checkpoints = [str(path) for path in configured]
        else:
            configured_segmenter = Path(args.pretrained_segmentation_checkpoint).expanduser()
            if configured_segmenter.is_file():
                configured_segmenter = configured_segmenter.resolve()
            else:
                segmenter_version = (
                    "seg_v3"
                    if str(args.experiment).upper() == "UNET-SEM-V3-F-NOEMB"
                    else "seg_v2"
                )
                bundled_segmenter = project_root / "mycode" / "tool" / segmenter_version / "best.pt"
                if not bundled_segmenter.is_file():
                    raise FileNotFoundError(
                        "Frozen semantic segmentation checkpoint is unavailable at both the recorded path "
                        f"({configured_segmenter}) and bundled path ({bundled_segmenter})."
                    )
                configured_segmenter = bundled_segmenter
            args.pretrained_segmentation_checkpoint = str(configured_segmenter)

    meta = LeRobotDatasetMetadata(args.repo_id, root=metadata_root)
    if run_cfg.get("no_gripper"):
        image_keys = run_cfg.get("dataset_view_image_keys") or [
            run_cfg["rgb_key"],
            *run_cfg["mask_target_keys"],
        ]
        metadata_root = make_filtered_dataset_view(
            source_root=metadata_root,
            view_root=run_dir / "inference_no_gripper_dataset_view",
            image_keys=list(image_keys),
            state_keys=list(run_cfg["state_keys"]),
            rebuild=False,
            no_gripper=True,
        )
        meta = LeRobotDatasetMetadata(args.repo_id, root=metadata_root)

    viewfusion_stats_source = None
    if str(args.experiment).upper() in VIEW_FUSION_EXPERIMENTS:
        viewfusion_stats_source = _inject_viewfusion_inference_metadata(
            meta,
            run_cfg,
            run_dir,
            project_root,
        )

    features = dataset_to_policy_features(meta.features)
    act_input_keys = [*args.state_keys, *act_image_keys_for_experiment(args)]
    act_input_features = {key: features[key] for key in act_input_keys if key in features}
    inference_stats = deepcopy(meta.stats)
    if getattr(args, "act_action_target", "dataset_action") != "dataset_action":
        action_stats = run_cfg.get("action_stats")
        if not isinstance(action_stats, dict):
            raise ValueError(
                "Follower-delta Mask-ACT checkpoint is missing derived action_stats in "
                "mask_act_run_config.json."
            )
        inference_stats["action"] = action_stats
    stats = reshape_visual_stats_for_channel_first(inference_stats, act_input_features)

    model = make_policy(args, meta, stats=stats)
    state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
    initialized_derived_buffers = _load_mask_act_state_dict_compatibly(model, state["model"])
    model.inference_image_size = args.image_size
    model.to(torch.device(model.config.device))
    model.eval()
    model.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        dataset_stats=stats,
    )
    details = {
        "checkpoint_dir": str(checkpoint_dir),
        "run_dir": str(run_dir),
        "metadata_root": str(metadata_root),
        "experiment": args.experiment,
        "rgb_key": args.rgb_key,
        "rgb_keys": list(args.rgb_keys),
        "viewfusion_stats_source": viewfusion_stats_source,
        "initialized_derived_state_buffers": initialized_derived_buffers,
    }
    return model, preprocessor, postprocessor, details
