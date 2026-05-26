#!/usr/bin/env python
"""Train ACT variants with mask supervision.

Experiments:
  1A: RGB -> U-Net -> predicted masks -> ACT. Seg loss trains U-Net; action loss trains ACT only.
  1B: Same as 1A, but action loss also backpropagates through the U-Net mask path.
  2A: ACT sees predicted masks plus a pooled RGB encoder latent. Seg loss trains U-Net; action loss
      trains ACT only.
  2B: ACT sees predicted masks plus a pooled RGB encoder latent. Action loss trains the latent encoder
      path, but not the mask decoder path.
  3:  ACT sees only the pooled RGB encoder latent. U-Net mask decoder is kept as an auxiliary task.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.utils.constants import OBS_ENV_STATE

from train_lerobot_policy import ensure_device_is_usable, make_filtered_dataset_view


DEFAULT_ROOT = Path("simdata/cube1")
DEFAULT_REPO_ID = "cube1"
DEFAULT_RGB_KEY = "observation.images.camera"
DEFAULT_STATE_KEYS = ["observation.state"]
DEFAULT_MASK_KEYS = [
    "observation.images.occluder",
    "observation.images.object",
    "observation.images.region",
    "observation.images.left_arm",
    "observation.images.right_arm",
]


warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)


@dataclass
class MaskActRunConfig:
    experiment: str
    repo_id: str
    root: str
    rgb_key: str
    state_keys: list[str]
    mask_target_keys: list[str]
    output_dir: str
    steps: int
    batch_size: int
    lr: float
    lr_backbone: float
    latent_dim: int
    unet_base_channels: int
    seg_loss_weight: float
    action_loss_weight: float
    chunk_size: int
    n_action_steps: int
    pretrained_backbone_weights: str | None


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class UNetSegNet(nn.Module):
    """RGB -> U-Net encoder latent -> decoder masks, plus pooled latent for ACT."""

    def __init__(self, out_masks: int, latent_dim: int, base_channels: int):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.inc = DoubleConv(3, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.up1 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up3 = Up(c2, c1, c1)
        self.out_conv = nn.Conv2d(c1, out_masks, kernel_size=1)
        self.latent_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        x1 = self.inc(rgb)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        latent = self.latent_proj(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out_conv(x), latent


class MaskACTPolicy(nn.Module):
    def __init__(
        self,
        act_policy: ACTPolicy,
        experiment: str,
        rgb_key: str,
        mask_keys: list[str],
        stats: dict,
        latent_dim: int,
        unet_base_channels: int,
        seg_loss_weight: float,
        action_loss_weight: float,
    ):
        super().__init__()
        self.act_policy = act_policy
        self.experiment = experiment.upper()
        self.rgb_key = rgb_key
        self.mask_keys = mask_keys
        self.stats = stats
        self.latent_dim = latent_dim
        self.seg_loss_weight = seg_loss_weight
        self.action_loss_weight = action_loss_weight
        self.bce = nn.BCEWithLogitsLoss()

        if self.experiment not in {"1A", "1B", "2A", "2B", "3"}:
            raise ValueError(f"Unknown experiment '{experiment}'. Choose 1A, 1B, 2A, 2B, or 3.")
        self.seg_net = UNetSegNet(
            out_masks=len(mask_keys),
            latent_dim=latent_dim,
            base_channels=unet_base_channels,
        )

    @property
    def config(self):
        return self.act_policy.config

    def mask_action_grad_enabled(self) -> bool:
        return self.experiment == "1B"

    def latent_action_grad_enabled(self) -> bool:
        return self.experiment in {"2B", "3"}

    def act_uses_masks(self) -> bool:
        return self.experiment in {"1A", "1B", "2A", "2B"}

    def act_uses_latent(self) -> bool:
        return self.experiment in {"2A", "2B", "3"}

    def normalize_visual_like(self, image: Tensor, key: str) -> Tensor:
        key_stats = self.stats[key]
        mean = torch.as_tensor(key_stats["mean"], dtype=image.dtype, device=image.device)
        std = torch.as_tensor(key_stats["std"], dtype=image.dtype, device=image.device).clamp_min(1e-6)
        return (image - mean) / std

    def build_mask_targets(self, raw_batch: dict[str, Tensor], device: torch.device) -> Tensor:
        masks = []
        for key in self.mask_keys:
            mask = raw_batch[key].to(device=device, dtype=torch.float32)
            if mask.shape[1] == 3:
                mask = mask.mean(dim=1, keepdim=True)
            else:
                mask = mask[:, :1]
            masks.append(mask.clamp(0.0, 1.0))
        return torch.cat(masks, dim=1)

    def predict_masks_and_latent(self, raw_batch: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
        rgb = raw_batch[self.rgb_key].to(device=device, dtype=torch.float32)
        return self.seg_net(rgb)

    def forward(self, batch: dict[str, Tensor], raw_batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        device = batch["action"].device
        mask_targets = self.build_mask_targets(raw_batch, device=device)
        mask_logits, rgb_latent = self.predict_masks_and_latent(raw_batch, device=device)
        seg_loss = self.bce(mask_logits, mask_targets)

        act_batch = dict(batch)
        if self.act_uses_latent():
            if not self.latent_action_grad_enabled():
                rgb_latent = rgb_latent.detach()
            act_batch[OBS_ENV_STATE] = rgb_latent

        if self.act_uses_masks():
            mask_probs = torch.sigmoid(mask_logits)
            if not self.mask_action_grad_enabled():
                mask_probs = mask_probs.detach()

            for idx, key in enumerate(self.mask_keys):
                mask_image = mask_probs[:, idx : idx + 1].repeat(1, 3, 1, 1)
                act_batch[key] = self.normalize_visual_like(mask_image, key)

        action_loss, action_logs = self.act_policy(act_batch)
        loss = self.action_loss_weight * action_loss + self.seg_loss_weight * seg_loss
        logs = {
            "loss": float(loss.detach().cpu()),
            "action_loss": float(action_loss.detach().cpu()),
            "seg_loss": float(seg_loss.detach().cpu()),
        }
        logs.update({key: float(value) for key, value in action_logs.items() if isinstance(value, (int, float))})
        return loss, logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["1A", "1B", "2A", "2B", "3"], required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rgb-key", default=DEFAULT_RGB_KEY)
    parser.add_argument("--state-keys", nargs="*", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--mask-target-keys", nargs="+", default=DEFAULT_MASK_KEYS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/mask_act"))
    parser.add_argument("--view-root", type=Path, default=None)
    parser.add_argument("--rebuild-view", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--unet-base-channels", type=int, default=16)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    parser.add_argument("--action-loss-weight", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=100)
    parser.add_argument("--pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=10_000)
    return parser.parse_args()


def act_image_keys_for_experiment(args: argparse.Namespace) -> list[str]:
    experiment = args.experiment.upper()
    if experiment in {"1A", "1B"}:
        return list(args.mask_target_keys)
    if experiment in {"2A", "2B"}:
        return list(args.mask_target_keys)
    if experiment == "3":
        return []
    raise ValueError(experiment)


def act_uses_latent(experiment: str) -> bool:
    return experiment.upper() in {"2A", "2B", "3"}


def make_policy(args: argparse.Namespace, meta: LeRobotDatasetMetadata, stats: dict) -> MaskACTPolicy:
    features = dataset_to_policy_features(meta.features)
    act_image_keys = act_image_keys_for_experiment(args)
    input_keys = [*args.state_keys, *act_image_keys]
    missing = [key for key in [*input_keys, "action"] if key not in features]
    if missing:
        raise KeyError(f"Missing feature(s) for ACT policy: {missing}")

    input_features = {key: features[key] for key in input_keys}
    if act_uses_latent(args.experiment):
        input_features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(args.latent_dim,))

    policy_cfg = make_policy_config(
        "act",
        input_features=input_features,
        output_features={"action": features["action"]},
        device=args.device,
        push_to_hub=False,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
    )
    act_policy = ACTPolicy(policy_cfg)
    return MaskACTPolicy(
        act_policy=act_policy,
        experiment=args.experiment,
        rgb_key=args.rgb_key,
        mask_keys=list(args.mask_target_keys),
        stats=stats,
        latent_dim=args.latent_dim,
        unet_base_channels=args.unet_base_channels,
        seg_loss_weight=args.seg_loss_weight,
        action_loss_weight=args.action_loss_weight,
    )


def save_run_config(args: argparse.Namespace, image_keys_in_view: list[str]) -> None:
    run_cfg = MaskActRunConfig(
        experiment=args.experiment,
        repo_id=args.repo_id,
        root=str(args.root),
        rgb_key=args.rgb_key,
        state_keys=list(args.state_keys),
        mask_target_keys=list(args.mask_target_keys),
        output_dir=str(args.output_dir),
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        latent_dim=args.latent_dim,
        unet_base_channels=args.unet_base_channels,
        seg_loss_weight=args.seg_loss_weight,
        action_loss_weight=args.action_loss_weight,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "mask_act_run_config.json", "w") as f:
        payload = asdict(run_cfg)
        payload["dataset_view_image_keys"] = image_keys_in_view
        payload["act_image_keys"] = act_image_keys_for_experiment(args)
        json.dump(payload, f, indent=4)
        f.write("\n")


def save_checkpoint(output_dir: Path, step: int, model: MaskACTPolicy, optimizer: torch.optim.Optimizer) -> None:
    checkpoint_dir = output_dir / f"checkpoint_step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )


def main() -> None:
    args = parse_args()
    if isinstance(args.pretrained_backbone_weights, str) and args.pretrained_backbone_weights.lower() in {
        "none",
        "null",
        "",
    }:
        args.pretrained_backbone_weights = None
    ensure_device_is_usable(args.device)
    device = torch.device(args.device)

    if args.output_dir.exists() and args.overwrite_output:
        shutil.rmtree(args.output_dir)

    image_keys_in_view = sorted({args.rgb_key, *args.mask_target_keys})
    view_root = args.view_root or (args.output_dir.parent / "dataset_views" / args.output_dir.name)
    source_root = args.root.resolve()
    filtered_root = make_filtered_dataset_view(
        source_root=source_root,
        view_root=view_root,
        image_keys=image_keys_in_view,
        state_keys=args.state_keys,
        rebuild=args.rebuild_view,
    )
    save_run_config(args, image_keys_in_view)

    meta = LeRobotDatasetMetadata(args.repo_id, root=filtered_root)
    delta_timestamps = {"action": [i / meta.fps for i in range(args.chunk_size)]}
    dataset = LeRobotDataset(
        args.repo_id,
        root=filtered_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-4,
        video_backend=args.video_backend,
    )

    model = make_policy(args, meta, stats=meta.stats).to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=model.config,
        dataset_stats=meta.stats,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    dl_iter = cycle(dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Experiment: {args.experiment}")
    print(f"Dataset view: {filtered_root}")
    print(f"ACT image keys: {act_image_keys_for_experiment(args)}")
    print(f"Mask supervision keys: {args.mask_target_keys}")
    print(f"Output dir: {args.output_dir}")

    model.train()
    start_time = time.perf_counter()
    for step in range(1, args.steps + 1):
        raw_batch = next(dl_iter)
        batch = preprocessor(raw_batch)

        optimizer.zero_grad(set_to_none=True)
        loss, logs = model(batch, raw_batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if args.log_freq > 0 and step % args.log_freq == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"step={step} loss={logs['loss']:.4f} action={logs['action_loss']:.4f} "
                f"seg={logs['seg_loss']:.4f} grad={float(grad_norm):.3f} elapsed_s={elapsed:.1f}"
            )
            start_time = time.perf_counter()

        if args.save_freq > 0 and (step % args.save_freq == 0 or step == args.steps):
            save_checkpoint(args.output_dir, step, model, optimizer)


if __name__ == "__main__":
    main()
