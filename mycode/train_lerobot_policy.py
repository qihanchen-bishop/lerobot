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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--job-name", default="act_bi_so_left_front")
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


def write_filtered_json(src: Path, dst: Path, keep_keys: set[str]) -> None:
    with open(src) as f:
        data = json.load(f)
    if "features" in data:
        data["features"] = {key: value for key, value in data["features"].items() if key in keep_keys}
    else:
        data = {key: value for key, value in data.items() if key in keep_keys}
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(data, f, indent=4)


def make_filtered_dataset_view(
    source_root: Path,
    view_root: Path,
    image_keys: list[str],
    state_keys: list[str],
    action_key: str = "action",
    rebuild: bool = False,
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
    write_filtered_json(source_root / "meta" / "info.json", view_root / "meta" / "info.json", keep_keys)
    write_filtered_json(source_root / "meta" / "stats.json", view_root / "meta" / "stats.json", keep_keys)

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
    elif policy_type == "diffusion":
        kwargs["n_obs_steps"] = args.diffusion_n_obs_steps
        kwargs["horizon"] = args.diffusion_horizon
        kwargs["n_action_steps"] = args.diffusion_n_action_steps
        kwargs["pretrained_backbone_weights"] = args.pretrained_backbone_weights

    return make_policy_config(policy_type, **kwargs)


def main() -> None:
    args = parse_args()
    ensure_device_is_usable(args.device)
    view_root = args.view_root or (args.output_dir.parent / "dataset_views" / args.output_dir.name)

    if args.output_dir.exists() and args.overwrite_output:
        shutil.rmtree(args.output_dir)

    filtered_root = make_filtered_dataset_view(
        source_root=args.root,
        view_root=view_root,
        image_keys=args.image_keys,
        state_keys=args.state_keys,
        rebuild=args.rebuild_view,
    )

    meta = LeRobotDatasetMetadata(args.repo_id, root=filtered_root)
    policy_cfg = build_policy_config(args.policy_type, meta, args)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, root=str(filtered_root), video_backend=args.video_backend),
        policy=policy_cfg,
        output_dir=args.output_dir,
        job_name=args.job_name,
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
    train(cfg)


if __name__ == "__main__":
    main()
