"""Isolated UR ACT + hard semantics + five supervised task queries."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from train_bi_ur3_policy import validate_dataset
from train_lerobot_policy import make_filtered_dataset_view
from ur_semq import FRONT_KEY, HEAD_NAMES, LABELS, SEMANTIC_KEY, URSemanticCache, digest, query_loss

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import METRIC_PRED, ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import get_step_checkpoint_dir, save_checkpoint, update_last_checkpoint


VARIANTS = ("full", "semantic-no-query", "rgb-query")


def variant_flags(args):
    variant = getattr(args, "variant", "full")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown UR variant: {variant}")
    return variant != "rgb-query", variant != "semantic-no-query"


class URQueryDataset(Dataset):
    def __init__(self, dataset, cache, image_keys, semantic_input=True, query_supervision=True):
        self.dataset, self.cache, self.image_keys = dataset, cache, image_keys
        self.semantic_input, self.query_supervision = semantic_input, query_supervision

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if (
            int(sample["episode_index"]) != self.cache.arrays["episode_index"][index]
            or int(sample["frame_index"]) != self.cache.arrays["frame_index"][index]
        ):
            raise ValueError("Cache and training sample index mismatch.")
        result = {k: sample[k] for k in [*self.image_keys, "observation.state", "action", "action_is_pad"]}
        if self.semantic_input:
            result["semantic_ids"] = torch.from_numpy(self.cache.labels(index))
        if self.query_supervision:
            for name in ("targets", "valid", "weights"):
                result[f"query_{name}"] = torch.from_numpy(self.cache.arrays[name][index].copy())
        return result


def build_data(args):
    root = args.root.resolve()
    semantic_input, query_supervision = variant_flags(args)
    images = [FRONT_KEY] + (["observation.images.depth_preview"] if args.mode == "rgbd" else [])
    validate_dataset(root, images, ["observation.state"])
    cache = URSemanticCache(args.cache, root)
    view = args.output_dir.parent / "dataset_views" / args.output_dir.name
    make_filtered_dataset_view(
        root,
        view,
        images,
        ["observation.state"],
        action_target="follower_joint_delta_gripper_absolute",
        follower_state_key="observation.state",
    )
    meta = LeRobotDatasetMetadata("local/ur_semq_merged4", root=view)
    features = dataset_to_policy_features(meta.features)
    inputs = {k: features[k] for k in ["observation.state", *images]}
    if semantic_input:
        inputs[SEMANTIC_KEY] = copy.deepcopy(inputs[FRONT_KEY])
    config = ACTConfig(
        input_features=inputs,
        output_features={"action": features["action"]},
        device=args.device,
        push_to_hub=False,
        chunk_size=60,
        n_action_steps=60,
        action_target="follower_joint_delta_gripper_absolute",
        separate_gripper_head=True,
        gripper_loss_weight=0.2,
        gripper_positive_weight=2.5,
        metric_mode="encoder_tokens" if query_supervision else None,
        metric_dim=5,
        metric_token_output_dims=[1, 1, 1, 1, 2] if query_supervision else None,
    )
    config.validate_features()
    stats = copy.deepcopy(meta.stats)
    for key in config.image_features:
        stats[key] = {
            "mean": np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1),
            "std": np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1),
        }
    dataset = LeRobotDataset(
        "local/ur_semq_merged4",
        root=view,
        video_backend="pyav",
        delta_timestamps={"action": [i / meta.fps for i in range(60)]},
    )
    if len(dataset) != cache.manifest["shape"][0]:
        raise ValueError("Training view/cache lengths differ.")
    return URQueryDataset(dataset, cache, images, semantic_input, query_supervision), config, stats, view


def prepare_batch(batch, palette, preprocessor, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    targets, valid, weights = (batch.pop(f"query_{name}", None) for name in ("targets", "valid", "weights"))
    if "semantic_ids" in batch:
        batch[SEMANTIC_KEY] = palette[batch.pop("semantic_ids").long()].permute(0, 3, 1, 2).contiguous()
    return preprocessor(batch), targets, valid, weights


def save_bundle(checkpoint, args, cache):
    semantic_input, query_supervision = variant_flags(args)
    pretrained = checkpoint / "pretrained_model"
    source = args.root / "models/unet_front"
    destination = pretrained / "semantic_model"
    if semantic_input:
        destination.mkdir()
        for name in ("model.ts", "model_config.json", "filter_policy.json"):
            shutil.copy2(source / name, destination / name)
    metadata = {
        "experiment": "UR-SemQ-v1",
        "mode": args.mode,
        "variant": getattr(args, "variant", "full"),
        "semantic_input": semantic_input,
        "query_supervision": query_supervision,
        "labels": LABELS,
        "heads": HEAD_NAMES if query_supervision else [],
        "scales": cache.manifest["scales"],
        "query_weight": args.query_weight if query_supervision else 0.0,
        "semantic_key": SEMANTIC_KEY,
        "semantic_normalization": "imagenet",
        "semantic_model_sha256": digest(destination / "model.ts") if semantic_input else None,
        "cache_manifest_sha256": digest(args.cache / "manifest.json"),
        "region_initial_frames": 20,
        "region_scope": "per episode; labels only, never policy input",
        "action": "follower sequential joint delta; separate absolute binary gripper; decoded once by ACTPolicy",
    }
    (pretrained / "ur_semq.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    torch.set_num_threads(4)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dataset, config, stats, view = build_data(args)
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="local/ur_semq_merged4", root=str(view), video_backend="pyav"),
        policy=config,
        output_dir=args.output_dir,
        job_name=args.output_dir.name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.steps,
        seed=args.seed,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=0,
    )
    cfg.validate()
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics").mkdir()
    cfg.save_pretrained(args.output_dir)
    (args.output_dir / "run_arguments.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    policy = ACTPolicy(config, dataset_stats=stats).to(args.device)
    pre, post = make_act_pre_post_processors(config, stats)
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    loader_kwargs = {"persistent_workers": True, "prefetch_factor": 2} if args.num_workers else {}
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
        **loader_kwargs,
    )
    palette = (
        torch.tensor(dataset.cache.manifest["palette_rgb"], device=args.device, dtype=torch.float32) / 255
    )
    scales = torch.tensor(dataset.cache.manifest["scales"], device=args.device)
    sums, count, started = {}, 0, time.monotonic()
    print(
        f"UR-SemQ-v1 {args.mode} {getattr(args, 'variant', 'full')}: {len(dataset)} frames, {dataset.dataset.num_episodes} episodes; images={list(config.image_features)}",
        flush=True,
    )
    print(
        f"query_weight={args.query_weight if dataset.query_supervision else 0.0}; scales={scales.tolist()}; FP32/TF32; shared ResNet; no identity embeddings",
        flush=True,
    )
    iterator = iter(loader)
    with (args.output_dir / "metrics/train_metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            load_start = time.monotonic()
            try:
                raw = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw = next(iterator)
            batch, target, valid, weights = prepare_batch(raw, palette, pre, args.device)
            load_time = time.monotonic() - load_start
            policy.train()
            optimizer.zero_grad(set_to_none=True)
            action_loss, metrics = policy(batch)
            if dataset.query_supervision:
                prediction = metrics.pop(METRIC_PRED)
                auxiliary_loss, query_metrics = query_loss(prediction, target, valid, weights, scales)
            else:
                auxiliary_loss, query_metrics = action_loss.new_zeros(()), {}
            loss = action_loss + args.query_weight * auxiliary_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.optimizer.grad_clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            values = {
                **metrics,
                **query_metrics,
                "loss": loss.detach(),
                "action_loss": action_loss.detach(),
                "query_loss": auxiliary_loss.detach(),
                "grad_norm": grad,
                "dataloading_s": load_time,
            }
            for key, value in values.items():
                sums[key] = sums.get(key, 0.0) + (
                    value.detach() if isinstance(value, torch.Tensor) else value
                )
            count += 1
            if step % args.log_freq == 0 or step == args.steps:
                record = {key: float(value) / count for key, value in sums.items()}
                record.update(
                    step=step, elapsed_s=time.monotonic() - started, lr=optimizer.param_groups[0]["lr"]
                )
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                sums, count = {}, 0
            if step % args.save_freq == 0 or step == args.steps:
                checkpoint = get_step_checkpoint_dir(args.output_dir, args.steps, step)
                save_checkpoint(
                    checkpoint, step, cfg, policy, optimizer, preprocessor=pre, postprocessor=post
                )
                save_bundle(checkpoint, args, dataset.cache)
                update_last_checkpoint(checkpoint)
                print(f"Checkpoint saved: {checkpoint}", flush=True)
    print("End of training", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/realsense_rgbd_merged4_seg"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/cache/ur_semq_v1_merged4"))
    parser.add_argument("--mode", choices=("rgb", "rgbd"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--query-weight", type=float, default=0.1)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--save-freq", type=int, default=20000)
    args = parser.parse_args()
    if (
        min(args.steps, args.batch_size, args.log_freq, args.save_freq) < 1
        or args.num_workers < 0
        or not math.isfinite(args.query_weight)
        or args.query_weight < 0
    ):
        parser.error("Invalid training parameters.")
    return args


if __name__ == "__main__":
    train(parse_args())
