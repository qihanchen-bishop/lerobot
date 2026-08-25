#!/usr/bin/env python
"""Load custom Mask-ACT training checkpoints for live policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors

from train_mask_act_policy import (
    MaskACTPolicy,
    SEMANTIC_EXPERIMENTS,
    act_image_keys_for_experiment,
    make_policy,
    reshape_visual_stats_for_channel_first,
)
from train_lerobot_policy import make_filtered_dataset_view


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
    if str(run_cfg.get("experiment", "")).upper() not in SEMANTIC_EXPERIMENTS:
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

    features = dataset_to_policy_features(meta.features)
    act_input_keys = [*args.state_keys, *act_image_keys_for_experiment(args)]
    act_input_features = {key: features[key] for key in act_input_keys if key in features}
    stats = reshape_visual_stats_for_channel_first(meta.stats, act_input_features)

    model = make_policy(args, meta, stats=stats)
    state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
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
    }
    return model, preprocessor, postprocessor, details
