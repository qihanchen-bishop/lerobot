#!/usr/bin/env python
"""Train a LeRobot policy on a filtered local dataset view."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path

import torch

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_policy_config
from lerobot.scripts.lerobot_train import train
from training_artifacts import plot_training_curves, tee_output


DEFAULT_ROOT = Path("/home/romilab/.cache/huggingface/lerobot/seeedstudio123/test_20260506_153720")
DEFAULT_REPO_ID = "seeedstudio123/test_20260506_153720"
DEFAULT_IMAGE_KEYS = ["observation.images.left_front"]
DEFAULT_STATE_KEYS = ["observation.state"]
DEFAULT_OUTPUT_DIR = Path("outputs/train/act_bi_so_left_front")


warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)


def ensure_device_is_usable(device: str) -> None:
    if not device.startswith("cuda"):
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA training, but torch.cuda.is_available() is False in this environment. "
            "Check that the NVIDIA driver is loaded and that this conda env has a CUDA-enabled PyTorch build."
        )

    supported_arches = set(torch.cuda.get_arch_list())
    major, minor = torch.cuda.get_device_capability()
    current_arch = f"sm_{major}{minor}"
    if current_arch not in supported_arches:
        raise RuntimeError(
            f"Current GPU compute capability '{current_arch}' is not supported by this PyTorch build. "
            f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
            f"supported_arches={sorted(supported_arches)}. "
            "Install a newer PyTorch build that supports this GPU, or rerun with --device cpu."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--policy-type", default="act", help="Policy type, e.g. act or diffusion.")
    parser.add_argument("--image-keys", nargs="*", default=DEFAULT_IMAGE_KEYS)
    parser.add_argument("--state-keys", nargs="*", default=DEFAULT_STATE_KEYS)
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Drop gripper dimensions from action and selected state features before training.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default="act_bi_so_left_front")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=100, help="ACT action chunk size.")
    parser.add_argument("--n-action-steps", type=int, default=100, help="ACT actions used per policy call.")
    parser.add_argument(
        "--act-camera-embedding",
        action="store_true",
        help="Add one learned camera identity embedding per ordered ACT image key.",
    )
    parser.add_argument(
        "--act-camera-embedding-mode",
        choices=["default", "zero", "gated"],
        default="default",
        help=(
            "Camera embedding initialization: default PyTorch initialization, all zeros, or "
            "N(0, 0.02) multiplied by a zero-initialized learnable gate."
        ),
    )
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


def symlink_tree(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(path, target)


def _drop_dim_value(value, drop_indices: list[int], expected_dim: int):
    if not isinstance(value, list):
        return value
    if len(value) == expected_dim:
        return [item for idx, item in enumerate(value) if idx not in drop_indices]
    if value and all(isinstance(row, list) and len(row) == expected_dim for row in value):
        return [[item for idx, item in enumerate(row) if idx not in drop_indices] for row in value]
    return value


def _gripper_drop_indices_from_features(features: dict, trim_keys: set[str]) -> dict[str, list[int]]:
    drop_indices: dict[str, list[int]] = {}
    for key in trim_keys:
        feature = features.get(key)
        if not isinstance(feature, dict):
            continue
        shape = feature.get("shape")
        names = feature.get("names")
        if not isinstance(shape, list) or len(shape) != 1 or shape[0] <= 1:
            continue
        if not isinstance(names, list) or len(names) != shape[0]:
            continue
        indices = [idx for idx, name in enumerate(names) if "gripper" in str(name).lower()]
        if indices:
            drop_indices[key] = indices
    return drop_indices


def _drop_dim_metadata(data: dict, drop_indices_by_key: dict[str, list[int]]) -> dict:
    features = data.get("features")
    if not isinstance(features, dict):
        return data

    original_dims_by_key: dict[str, int] = {}
    for key, drop_indices in drop_indices_by_key.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            continue
        shape = feature.get("shape")
        if not isinstance(shape, list) or len(shape) != 1 or shape[0] <= len(drop_indices):
            continue
        original_dims_by_key[key] = shape[0]
        feature["shape"] = [shape[0] - len(drop_indices)]
        names = feature.get("names")
        if isinstance(names, list) and len(names) == shape[0]:
            feature["names"] = [name for idx, name in enumerate(names) if idx not in drop_indices]

    stats = data.get("stats")
    if isinstance(stats, dict):
        for key, original_dim in original_dims_by_key.items():
            if not isinstance(stats.get(key), dict):
                continue
            for stat_name, stat_value in list(stats[key].items()):
                stats[key][stat_name] = _drop_dim_value(stat_value, drop_indices_by_key[key], original_dim)

    return data


def _drop_dim_stats(data: dict, drop_indices_by_key: dict[str, list[int]], original_dims_by_key: dict[str, int]) -> dict:
    for key, original_dim in original_dims_by_key.items():
        if not isinstance(data.get(key), dict):
            continue
        for stat_name, stat_value in list(data[key].items()):
            data[key][stat_name] = _drop_dim_value(stat_value, drop_indices_by_key[key], original_dim)
    return data


def write_filtered_json(
    src: Path,
    dst: Path,
    keep_keys: set[str],
    drop_indices_by_key: dict[str, list[int]] | None = None,
    original_dims_by_key: dict[str, int] | None = None,
) -> None:
    with open(src) as f:
        data = json.load(f)
    if "features" in data:
        data["features"] = {key: value for key, value in data["features"].items() if key in keep_keys}
        if drop_indices_by_key:
            data = _drop_dim_metadata(data, drop_indices_by_key)
    else:
        original_dims_by_key = original_dims_by_key or {
            key: len(value["mean"])
            for key, value in data.items()
            if drop_indices_by_key
            and key in drop_indices_by_key
            and isinstance(value, dict)
            and isinstance(value.get("mean"), list)
        }
        data = {key: value for key, value in data.items() if key in keep_keys}
        if drop_indices_by_key:
            data = _drop_dim_stats(data, drop_indices_by_key, original_dims_by_key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, indent=4)


def _drop_dim_cell(value, drop_indices: list[int]):
    if value is None:
        return value
    return [item for idx, item in enumerate(value) if idx not in drop_indices]


def write_trimmed_data_tree(src_dir: Path, dst_dir: Path, drop_indices_by_key: dict[str, list[int]]) -> None:
    if not drop_indices_by_key:
        symlink_tree(src_dir, dst_dir)
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "--no-gripper needs pandas/pyarrow to rewrite LeRobot parquet data files. "
            "Install the dataset dependencies in this environment before using --no-gripper."
        ) from exc

    for path in src_dir.rglob("*"):
        rel = path.relative_to(src_dir)
        target = dst_dir / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".parquet":
            if not target.exists():
                os.symlink(path, target)
            continue
        df = pd.read_parquet(path)
        for key, drop_indices in drop_indices_by_key.items():
            if key in df.columns:
                df[key] = df[key].map(lambda value, indices=drop_indices: _drop_dim_cell(value, indices))
        df.to_parquet(target, index=False)


def make_filtered_dataset_view(
    source_root: Path,
    view_root: Path,
    image_keys: list[str],
    state_keys: list[str],
    action_key: str = "action",
    rebuild: bool = False,
    no_gripper: bool = False,
) -> Path:
    keep_keys = {
        action_key,
        *state_keys,
        *image_keys,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }

    if rebuild and view_root.exists():
        shutil.rmtree(view_root)

    view_root.mkdir(parents=True, exist_ok=True)
    data_root = view_root / "data"
    no_gripper_marker = view_root / "meta" / ".no_gripper"
    if no_gripper != no_gripper_marker.exists() and data_root.exists():
        shutil.rmtree(data_root)

    drop_indices_by_key: dict[str, list[int]] = {}
    original_dims_by_key: dict[str, int] = {}
    if no_gripper:
        trim_keys = {action_key, *state_keys}
        with open(source_root / "meta" / "info.json") as f:
            source_info = json.load(f)
        features = source_info.get("features", {})
        if not isinstance(features, dict):
            raise ValueError(f"Could not read feature metadata from {source_root / 'meta' / 'info.json'}")
        drop_indices_by_key = _gripper_drop_indices_from_features(features, trim_keys)
        original_dims_by_key = {
            key: features[key]["shape"][0]
            for key in drop_indices_by_key
            if isinstance(features.get(key), dict)
            and isinstance(features[key].get("shape"), list)
            and len(features[key]["shape"]) == 1
        }
    write_filtered_json(
        source_root / "meta" / "info.json",
        view_root / "meta" / "info.json",
        keep_keys,
        drop_indices_by_key,
        original_dims_by_key,
    )
    write_filtered_json(
        source_root / "meta" / "stats.json",
        view_root / "meta" / "stats.json",
        keep_keys,
        drop_indices_by_key,
        original_dims_by_key,
    )

    if no_gripper:
        if data_root.exists() and not rebuild:
            shutil.rmtree(data_root)
        write_trimmed_data_tree(source_root / "data", data_root, drop_indices_by_key)
        no_gripper_marker.parent.mkdir(parents=True, exist_ok=True)
        no_gripper_marker.write_text(json.dumps(drop_indices_by_key, indent=4) + "\n")
    else:
        if no_gripper_marker.exists():
            no_gripper_marker.unlink()
        symlink_tree(source_root / "data", view_root / "data")
    symlink_tree(source_root / "meta" / "episodes", view_root / "meta" / "episodes")

    tasks_src = source_root / "meta" / "tasks.parquet"
    tasks_dst = view_root / "meta" / "tasks.parquet"
    if tasks_src.exists() and not tasks_dst.exists():
        tasks_dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(tasks_src, tasks_dst)

    subtasks_src = source_root / "meta" / "subtasks.parquet"
    subtasks_dst = view_root / "meta" / "subtasks.parquet"
    if subtasks_src.exists() and not subtasks_dst.exists():
        os.symlink(subtasks_src, subtasks_dst)

    for image_key in image_keys:
        src = source_root / "videos" / image_key
        if not src.exists():
            raise FileNotFoundError(f"Missing selected video key directory: {src}")
        symlink_tree(src, view_root / "videos" / image_key)

    return view_root


def build_policy_config(policy_type: str, meta: LeRobotDatasetMetadata, args: argparse.Namespace):
    features = dataset_to_policy_features(meta.features)
    input_keys = [*args.state_keys, *args.image_keys]
    missing = [key for key in input_keys + ["action"] if key not in features]
    if missing:
        raise KeyError(f"Missing required policy feature(s): {missing}")

    kwargs = {
        "input_features": {key: features[key] for key in input_keys},
        "output_features": {"action": features["action"]},
        "device": args.device,
        "push_to_hub": args.push_to_hub,
    }
    if policy_type == "act":
        kwargs["chunk_size"] = args.chunk_size
        kwargs["n_action_steps"] = args.n_action_steps
        kwargs["pretrained_backbone_weights"] = args.pretrained_backbone_weights
        if getattr(args, "act_camera_embedding", False):
            kwargs["image_camera_ids"] = list(range(len(args.image_keys)))
            kwargs["image_camera_embedding_mode"] = args.act_camera_embedding_mode
    elif policy_type == "diffusion":
        kwargs["n_obs_steps"] = args.diffusion_n_obs_steps
        kwargs["horizon"] = args.diffusion_horizon
        kwargs["n_action_steps"] = args.diffusion_n_action_steps
        kwargs["pretrained_backbone_weights"] = args.pretrained_backbone_weights
        if getattr(args, "diffusion_full_frame", False):
            kwargs["crop_shape"] = None
        backbone_norm = getattr(args, "diffusion_backbone_norm", "group")
        kwargs["use_group_norm"] = backbone_norm == "group"
        kwargs["use_frozen_batch_norm"] = backbone_norm == "frozen_batch"
        drop_n_last_frames = getattr(args, "diffusion_drop_n_last_frames", None)
        if drop_n_last_frames is not None:
            kwargs["drop_n_last_frames"] = drop_n_last_frames
        kwargs["do_mask_loss_for_padding"] = getattr(args, "diffusion_mask_padding_loss", False)
        kwargs["num_train_timesteps"] = getattr(args, "diffusion_num_train_timesteps", 100)
        num_inference_steps = getattr(args, "diffusion_num_inference_steps", None)
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        lr_backbone = getattr(args, "diffusion_lr_backbone", None)
        if lr_backbone is not None:
            kwargs["optimizer_lr_backbone"] = lr_backbone

    return make_policy_config(policy_type, **kwargs)


def main() -> None:
    args = parse_args()
    final_log_path = args.output_dir / "logs" / "train.log"
    temp_log_path = args.output_dir.parent / f".{args.output_dir.name}.train.log.tmp"
    with tee_output(temp_log_path):
        run_training(args, final_log_path)
    if args.output_dir.exists():
        final_log_path.parent.mkdir(parents=True, exist_ok=True)
        if final_log_path.exists():
            final_log_path.unlink()
        shutil.move(str(temp_log_path), final_log_path)


def run_training(args: argparse.Namespace, log_path: Path) -> None:
    ensure_device_is_usable(args.device)
    if args.act_camera_embedding and args.policy_type.lower() != "act":
        raise ValueError("--act-camera-embedding is only valid with --policy-type act.")
    if args.act_camera_embedding and not args.image_keys:
        raise ValueError("--act-camera-embedding requires at least one --image-keys entry.")
    if not args.act_camera_embedding and args.act_camera_embedding_mode != "default":
        raise ValueError(
            "--act-camera-embedding-mode zero/gated requires --act-camera-embedding."
        )
    view_root = args.view_root or (args.output_dir.parent / "dataset_views" / args.output_dir.name)

    if args.output_dir.exists() and args.overwrite_output:
        shutil.rmtree(args.output_dir)

    filtered_root = make_filtered_dataset_view(
        source_root=args.root,
        view_root=view_root,
        image_keys=args.image_keys,
        state_keys=args.state_keys,
        rebuild=args.rebuild_view,
        no_gripper=args.no_gripper,
    )

    meta = LeRobotDatasetMetadata(args.repo_id, root=filtered_root)
    policy_cfg = build_policy_config(args.policy_type, meta, args)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, root=str(filtered_root), video_backend=args.video_backend),
        policy=policy_cfg,
        output_dir=args.output_dir,
        job_name=args.job_name,
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        wandb=WandBConfig(enable=args.wandb_enable),
    )

    print(f"Training dataset view: {filtered_root}")
    print(f"Input features: {list(policy_cfg.input_features)}")
    print(f"Output features: {list(policy_cfg.output_features)}")
    if args.policy_type == "act":
        print(
            "ACT camera embedding: "
            f"ids={policy_cfg.image_camera_ids}, mode={policy_cfg.image_camera_embedding_mode}, "
            f"std={policy_cfg.image_camera_embedding_std:g}"
            if policy_cfg.image_camera_ids is not None
            else "ACT camera embedding: disabled"
        )
    if args.policy_type == "diffusion":
        print(
            "Diffusion temporal config: "
            f"obs={policy_cfg.n_obs_steps}, horizon={policy_cfg.horizon}, "
            f"execute={policy_cfg.n_action_steps}, drop_last={policy_cfg.drop_n_last_frames}"
        )
        print(
            "Diffusion visual config: "
            f"crop={policy_cfg.crop_shape or 'full frame'}, "
            f"pretrained={policy_cfg.pretrained_backbone_weights}, "
            f"group_norm={policy_cfg.use_group_norm}, "
            f"frozen_batch_norm={policy_cfg.use_frozen_batch_norm}, "
            f"backbone_lr={policy_cfg.optimizer_lr_backbone or policy_cfg.optimizer_lr:g}"
        )
        print(
            "Diffusion schedule: "
            f"train_timesteps={policy_cfg.num_train_timesteps}, "
            f"inference_steps={policy_cfg.num_inference_steps or policy_cfg.num_train_timesteps}, "
            f"mask_padding_loss={policy_cfg.do_mask_loss_for_padding}"
        )
    if args.no_gripper:
        print("No gripper: dropped gripper dimensions when present in action and selected state features.")
    print(f"Train log: {log_path}")
    print(f"Metrics JSONL: {args.output_dir / 'metrics' / 'train_metrics.jsonl'}")
    print(f"Loss curve: {args.output_dir / 'metrics' / 'train_loss_curve.png'}")
    train(cfg)
    plot_training_curves(
        args.output_dir / "metrics" / "train_metrics.jsonl",
        args.output_dir / "metrics" / "train_loss_curve.png",
        f"{args.policy_type} training losses",
    )


if __name__ == "__main__":
    main()
