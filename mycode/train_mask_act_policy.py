#!/usr/bin/env python
"""Train ACT variants with mask supervision.

Experiments:
  1A: RGB -> U-Net -> predicted masks -> ACT. Seg loss trains U-Net; action loss trains ACT only.
  1B: Same as 1A, but action loss also backpropagates through the U-Net mask path.
  4A: Same as 1B, with sqrt(object area ratio) and object-region centroid distance as ACT env state.
  4B: Same as 1B, with two ACT encoder metric tokens supervised to predict the two mask metrics.
  4C: Same as 1B, with the two metrics predicted by the ACT decoder and fed back chunk-autoregressively.
  5:  RGB-only inference with latent semantic distillation. During training, ACT receives an RGB main latent
      plus five mask-encoder semantic teacher latents; RGB semantic heads are aligned to those teacher latents.
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
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import cycle, dataset_to_policy_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.modeling_act import METRIC_INPUT, METRIC_PRED, METRIC_SEED
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
    semantic_loss_weight: float
    metric_loss_weight: float
    metric_eps: float
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


class MaskSemanticEncoder(nn.Module):
    """Shared encoder that maps each binary semantic mask to one latent vector."""

    def __init__(self, latent_dim: int, base_channels: int):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.net = nn.Sequential(
            DoubleConv(1, c1),
            Down(c1, c2),
            Down(c2, c3),
            Down(c3, c4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, masks: Tensor) -> Tensor:
        batch_size, num_masks = masks.shape[:2]
        latents = self.net(masks.reshape(batch_size * num_masks, 1, *masks.shape[-2:]))
        return latents.reshape(batch_size, num_masks, -1)


class SemanticLatentNet(nn.Module):
    """RGB ResNet latent plus per-class RGB semantic heads and mask semantic teachers."""

    def __init__(
        self,
        num_masks: int,
        latent_dim: int,
        mask_base_channels: int,
        pretrained_backbone_weights: str | None,
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=pretrained_backbone_weights)
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.rgb_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.rgb_main_proj = nn.Sequential(nn.Linear(resnet.fc.in_features, latent_dim), nn.LayerNorm(latent_dim))
        self.rgb_semantic_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(resnet.fc.in_features, latent_dim), nn.LayerNorm(latent_dim))
            for _ in range(num_masks)
        )
        self.mask_encoder = MaskSemanticEncoder(latent_dim=latent_dim, base_channels=mask_base_channels)

    def forward(self, rgb: Tensor, masks: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
        rgb_features = self.rgb_pool(self.rgb_encoder(rgb))
        rgb_main_latent = self.rgb_main_proj(rgb_features)
        rgb_semantic_latents = torch.stack([head(rgb_features) for head in self.rgb_semantic_heads], dim=1)
        mask_semantic_latents = self.mask_encoder(masks) if masks is not None else None
        return rgb_main_latent, rgb_semantic_latents, mask_semantic_latents


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
        semantic_loss_weight: float,
        metric_loss_weight: float,
        metric_eps: float,
        pretrained_backbone_weights: str | None,
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
        self.semantic_loss_weight = semantic_loss_weight
        self.metric_loss_weight = metric_loss_weight
        self.metric_eps = metric_eps
        self.bce = nn.BCEWithLogitsLoss()

        if self.experiment not in {"1A", "1B", "2A", "2B", "3", "4A", "4B", "4C", "5"}:
            raise ValueError(f"Unknown experiment '{experiment}'. Choose 1A, 1B, 2A, 2B, 3, 4A, 4B, 4C, or 5.")
        self.seg_net = UNetSegNet(
            out_masks=len(mask_keys),
            latent_dim=latent_dim,
            base_channels=unet_base_channels,
        )
        self.semantic_net = (
            SemanticLatentNet(
                num_masks=len(mask_keys),
                latent_dim=latent_dim,
                mask_base_channels=unet_base_channels,
                pretrained_backbone_weights=pretrained_backbone_weights,
            )
            if self.experiment == "5"
            else None
        )

    @property
    def config(self):
        return self.act_policy.config

    def mask_action_grad_enabled(self) -> bool:
        return self.experiment in {"1B", "4A", "4B", "4C"}

    def latent_action_grad_enabled(self) -> bool:
        return self.experiment in {"2B", "3"}

    def act_uses_masks(self) -> bool:
        return self.experiment in {"1A", "1B", "2A", "2B", "4A", "4B", "4C"}

    def act_uses_latent(self) -> bool:
        return self.experiment in {"2A", "2B", "3"}

    def uses_mask_metrics(self) -> bool:
        return self.experiment in {"4A", "4B", "4C"}

    def uses_semantic_latents(self) -> bool:
        return self.experiment == "5"

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

    def predict_semantic_latents(
        self,
        raw_batch: dict[str, Tensor],
        device: torch.device,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        rgb = raw_batch[self.rgb_key].to(device=device, dtype=torch.float32)
        if self.semantic_net is None:
            raise RuntimeError("Semantic latent prediction is only available for experiment 5.")
        return self.semantic_net(rgb, masks)

    def compute_mask_metrics(self, masks: Tensor) -> Tensor:
        object_idx = self.mask_keys.index("observation.images.object")
        region_idx = self.mask_keys.index("observation.images.region")
        object_mask = masks[:, object_idx]
        region_mask = masks[:, region_idx]

        object_area_ratio = object_mask.mean(dim=(-2, -1)).clamp_min(0.0)
        object_exposure = torch.sqrt(object_area_ratio.clamp_min(self.metric_eps))

        height, width = object_mask.shape[-2:]
        y_coords = torch.linspace(0.0, float(height - 1), height, device=masks.device, dtype=masks.dtype)
        x_coords = torch.linspace(0.0, float(width - 1), width, device=masks.device, dtype=masks.dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        object_sum = object_mask.sum(dim=(-2, -1)).clamp_min(self.metric_eps)
        region_sum = region_mask.sum(dim=(-2, -1)).clamp_min(self.metric_eps)
        object_x = (object_mask * xx).sum(dim=(-2, -1)) / object_sum
        object_y = (object_mask * yy).sum(dim=(-2, -1)) / object_sum
        region_x = (region_mask * xx).sum(dim=(-2, -1)) / region_sum
        region_y = (region_mask * yy).sum(dim=(-2, -1)) / region_sum

        diagonal = torch.sqrt(
            torch.as_tensor((height - 1) ** 2 + (width - 1) ** 2, device=masks.device, dtype=masks.dtype)
        ).clamp_min(self.metric_eps)
        distance = torch.sqrt((object_x - region_x).square() + (object_y - region_y).square()) / diagonal
        return torch.stack([object_exposure, distance.clamp(0.0, 1.0)], dim=-1)

    def forward(self, batch: dict[str, Tensor], raw_batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        device = batch["action"].device
        mask_targets = self.build_mask_targets(raw_batch, device=device)

        if self.uses_semantic_latents():
            rgb_main_latent, rgb_semantic_latents, mask_semantic_latents = self.predict_semantic_latents(
                raw_batch,
                device=device,
                masks=mask_targets,
            )
            if mask_semantic_latents is None:
                raise RuntimeError("Experiment 5 requires mask teacher latents during training.")
            semantic_loss = F.smooth_l1_loss(rgb_semantic_latents, mask_semantic_latents.detach())
            act_batch = dict(batch)
            teacher_semantic_latents = mask_semantic_latents.reshape(mask_semantic_latents.shape[0], -1)
            act_batch[OBS_ENV_STATE] = torch.cat([rgb_main_latent, teacher_semantic_latents], dim=-1)
            action_loss, action_logs = self.act_policy(act_batch)
            loss = self.action_loss_weight * action_loss + self.semantic_loss_weight * semantic_loss
            semantic_cosine = F.cosine_similarity(
                rgb_semantic_latents.detach().flatten(1),
                mask_semantic_latents.detach().flatten(1),
                dim=-1,
            ).mean()
            logs = {
                "loss": float(loss.detach().cpu()),
                "action_loss": float(action_loss.detach().cpu()),
                "semantic_loss": float(semantic_loss.detach().cpu()),
                "semantic_cosine": float(semantic_cosine.cpu()),
            }
            logs.update({key: float(value) for key, value in action_logs.items() if isinstance(value, (int, float))})
            return loss, logs

        mask_logits, rgb_latent = self.predict_masks_and_latent(raw_batch, device=device)
        seg_loss = self.bce(mask_logits, mask_targets)
        metric_targets = self.compute_mask_metrics(mask_targets) if self.uses_mask_metrics() else None

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

            if self.uses_mask_metrics():
                metric_inputs = self.compute_mask_metrics(mask_probs)
                if self.experiment == "4A":
                    act_batch[OBS_ENV_STATE] = metric_inputs
                elif self.experiment == "4B":
                    act_batch["metric_target"] = metric_targets
                elif self.experiment == "4C":
                    metric_chunk = metric_targets.unsqueeze(1).expand(-1, self.config.chunk_size, -1)
                    act_batch[METRIC_INPUT] = metric_chunk
                    act_batch[METRIC_SEED] = metric_inputs
                    act_batch["metric_target"] = metric_chunk

        action_loss, action_logs = self.act_policy(act_batch)
        metric_loss = None
        metric_pred = action_logs.pop(METRIC_PRED, None)
        if metric_targets is not None and metric_pred is not None:
            target = act_batch["metric_target"]
            metric_loss = F.smooth_l1_loss(metric_pred, target)

        loss = self.action_loss_weight * action_loss + self.seg_loss_weight * seg_loss
        if metric_loss is not None:
            loss = loss + self.metric_loss_weight * metric_loss
        logs = {
            "loss": float(loss.detach().cpu()),
            "action_loss": float(action_loss.detach().cpu()),
            "seg_loss": float(seg_loss.detach().cpu()),
        }
        if metric_loss is not None:
            pred_for_mae = metric_pred
            target_for_mae = act_batch["metric_target"]
            if pred_for_mae.ndim == 3:
                pred_for_mae = pred_for_mae[:, 0]
                target_for_mae = target_for_mae[:, 0]
            metric_mae = (pred_for_mae.detach() - target_for_mae.detach()).abs().mean(dim=0)
            logs["metric_loss"] = float(metric_loss.detach().cpu())
            logs["object_exposure_mae"] = float(metric_mae[0].cpu())
            logs["object_region_distance_mae"] = float(metric_mae[1].cpu())
        logs.update({key: float(value) for key, value in action_logs.items() if isinstance(value, (int, float))})
        return loss, logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["1A", "1B", "2A", "2B", "3", "4A", "4B", "4C", "5"], required=True)
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
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--metric-eps", type=float, default=1e-6)
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
    if experiment in {"4A", "4B", "4C"}:
        return list(args.mask_target_keys)
    if experiment in {"3", "5"}:
        return []
    raise ValueError(experiment)


def act_uses_latent(experiment: str) -> bool:
    return experiment.upper() in {"2A", "2B", "3"}


def act_uses_semantic_env_state(experiment: str) -> bool:
    return experiment.upper() == "5"


def act_uses_metric_env_state(experiment: str) -> bool:
    return experiment.upper() == "4A"


def act_metric_mode(experiment: str) -> str | None:
    experiment = experiment.upper()
    if experiment == "4B":
        return "encoder_tokens"
    if experiment == "4C":
        return "decoder_autoregressive"
    return None


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
    if act_uses_metric_env_state(args.experiment):
        input_features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(2,))
    if act_uses_semantic_env_state(args.experiment):
        input_features[OBS_ENV_STATE] = PolicyFeature(
            type=FeatureType.ENV,
            shape=(args.latent_dim * (1 + len(args.mask_target_keys)),),
        )

    policy_cfg = make_policy_config(
        "act",
        input_features=input_features,
        output_features={"action": features["action"]},
        device=args.device,
        push_to_hub=False,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        metric_mode=act_metric_mode(args.experiment),
        metric_dim=2,
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
        semantic_loss_weight=args.semantic_loss_weight,
        metric_loss_weight=args.metric_loss_weight,
        metric_eps=args.metric_eps,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
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
        semantic_loss_weight=args.semantic_loss_weight,
        metric_loss_weight=args.metric_loss_weight,
        metric_eps=args.metric_eps,
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


def save_checkpoint(output_dir: Path, step: int, model: MaskACTPolicy, optimizer: torch.optim.Optimizer) -> Path:
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
    return checkpoint_dir


def tensor_image_to_uint8(image: Tensor) -> Tensor:
    image = image.detach().cpu().float()
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = image.permute(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 3)
    if image.max() <= 1.5:
        image = image * 255.0
    return image.clamp(0, 255).to(torch.uint8)


def make_labeled_tile(image: Tensor, label: str, size: tuple[int, int] = (192, 144)) -> Image.Image:
    array = tensor_image_to_uint8(image).numpy()
    tile = Image.fromarray(array).resize(size, Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size[0], size[1] + 20), "white")
    canvas.paste(tile, (0, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill=(0, 0, 0))
    return canvas


@torch.no_grad()
def save_mask_preview(checkpoint_dir: Path, step: int, model: MaskACTPolicy, raw_batch: dict[str, Tensor]) -> None:
    if model.uses_semantic_latents():
        return

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    mask_targets = model.build_mask_targets(raw_batch, device=device)
    mask_logits, _ = model.predict_masks_and_latent(raw_batch, device=device)
    mask_probs = torch.sigmoid(mask_logits)

    rgb = raw_batch[model.rgb_key][0]
    tiles = [make_labeled_tile(rgb, "rgb")]
    for idx, key in enumerate(model.mask_keys):
        name = key.rsplit(".", 1)[-1]
        gt = mask_targets[0, idx : idx + 1].repeat(3, 1, 1)
        pred = mask_probs[0, idx : idx + 1].repeat(3, 1, 1)
        tiles.append(make_labeled_tile(gt, f"gt {name}"))
        tiles.append(make_labeled_tile(pred, f"pred {name}"))

    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    tile_w, tile_h = tiles[0].size
    grid = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_w
        y = (idx // columns) * tile_h
        grid.paste(tile, (x, y))
    grid.save(checkpoint_dir / f"mask_preview_step_{step:06d}.png")

    if was_training:
        model.train()


class TrainingMetricsLogger:
    """Persist scalar training metrics as JSON and TensorBoard event files."""

    def __init__(self, output_dir: Path, run_config: dict[str, Any]):
        self.output_dir = output_dir
        self.metrics_dir = output_dir / "metrics"
        self.tensorboard_dir = output_dir / "tensorboard"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.jsonl_path = self.metrics_dir / "train_metrics.jsonl"
        self.json_path = self.metrics_dir / "train_metrics.json"
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir)) if SummaryWriter is not None else None

        self.metadata = {
            "run_config": run_config,
            "jsonl_path": str(self.jsonl_path),
            "tensorboard_logdir": str(self.tensorboard_dir),
            "created_at_unix": time.time(),
        }
        self._write_json()

    def log(self, step: int, logs: dict[str, float], grad_norm: float, elapsed_s: float, lr: float) -> None:
        record = {
            "step": step,
            "wall_time_unix": time.time(),
            "elapsed_s": elapsed_s,
            "lr": lr,
            "grad_norm": grad_norm,
            **{key: float(value) for key, value in logs.items() if isinstance(value, (int, float))},
        }
        self.records.append(record)

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        self._write_json()

        if self.writer is not None:
            for key, value in record.items():
                if key in {"step", "wall_time_unix"} or not isinstance(value, (int, float)):
                    continue
                tag = self._tensorboard_tag(key)
                self.writer.add_scalar(tag, value, step)
            self.writer.flush()

    def close(self) -> None:
        self._write_json()
        if self.writer is not None:
            self.writer.close()

    def _write_json(self) -> None:
        with open(self.json_path, "w") as f:
            json.dump({"metadata": self.metadata, "records": self.records}, f, indent=4)
            f.write("\n")

    @staticmethod
    def _tensorboard_tag(key: str) -> str:
        if key in {"loss", "action_loss", "seg_loss"}:
            return f"train/{key}"
        if key in {"grad_norm", "lr"}:
            return f"optim/{key}"
        if key == "elapsed_s":
            return "time/elapsed_s_per_log_interval"
        return f"train/{key}"


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
    run_config_path = args.output_dir / "mask_act_run_config.json"
    with open(run_config_path) as f:
        metrics_run_config = json.load(f)
    metrics_logger = TrainingMetricsLogger(args.output_dir, metrics_run_config)
    print(f"Metrics JSON: {metrics_logger.json_path}")
    print(f"Metrics JSONL: {metrics_logger.jsonl_path}")
    print(f"TensorBoard logdir: {metrics_logger.tensorboard_dir}")

    model.train()
    start_time = time.perf_counter()
    try:
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
                grad_norm_float = float(grad_norm.detach().cpu() if isinstance(grad_norm, Tensor) else grad_norm)
                lr = float(optimizer.param_groups[0]["lr"])
                metrics_logger.log(step=step, logs=logs, grad_norm=grad_norm_float, elapsed_s=elapsed, lr=lr)
                extra_loss = logs.get("seg_loss", logs.get("semantic_loss", 0.0))
                extra_name = "seg" if "seg_loss" in logs else "semantic"
                print(
                    f"step={step} loss={logs['loss']:.4f} action={logs['action_loss']:.4f} "
                    f"{extra_name}={extra_loss:.4f} grad={grad_norm_float:.3f} elapsed_s={elapsed:.1f}"
                )
                start_time = time.perf_counter()

            if args.save_freq > 0 and (step % args.save_freq == 0 or step == args.steps):
                checkpoint_dir = save_checkpoint(args.output_dir, step, model, optimizer)
                save_mask_preview(checkpoint_dir, step, model, raw_batch)
    finally:
        metrics_logger.close()


if __name__ == "__main__":
    main()
