#!/usr/bin/env python
"""Train ACT variants with mask supervision.

Experiments:
  1A: RGB -> U-Net -> predicted masks -> ACT. Seg loss trains U-Net; action loss trains ACT only.
  1B: Same as 1A, but action loss also backpropagates through the U-Net mask path.
  4A: Same as 1B, with sqrt(object area ratio) and object-region centroid distance as ACT env state.
  4B: Same as 1B, with two ACT encoder metric tokens supervised to predict the two mask metrics.
  4C: Same as 1B, with the two metrics predicted by the ACT decoder and fed back chunk-autoregressively.
  All experiments use the same canonical five binary semantic masks in this order:
      occluder, object, region, left_arm, right_arm.
  5:  RGB-only inference with latent semantic distillation. During training, ACT receives an RGB main latent
      plus those five mask-encoder semantic teacher latents; RGB semantic heads are aligned to the teachers.
  2A: ACT sees predicted masks plus a pooled RGB encoder latent. Seg loss trains U-Net; action loss
      trains ACT only.
  2B: ACT sees predicted masks plus a pooled RGB encoder latent. Action loss trains the latent encoder
      path, but not the mask decoder path.
  2C: ACT sees predicted masks plus the original RGB images. Masks and RGB share ACT's image backbone,
      and action loss also backpropagates through the U-Net mask path.
  3:  ACT sees only the pooled RGB encoder latent. U-Net mask decoder is kept as an auxiliary task.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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
from tqdm.auto import tqdm

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
from training_artifacts import plot_training_curves, tee_output


DEFAULT_ROOT = Path("simdata/cube1")
DEFAULT_REPO_ID = "cube1"
DEFAULT_RGB_KEY = "observation.images.camera"
DEFAULT_STATE_KEYS = ["observation.state"]
SOARM_RGB_KEY = "observation.images.left_front"
CANONICAL_SEMANTIC_MASK_KEYS = [
    "observation.images.occluder",
    "observation.images.object",
    "observation.images.region",
    "observation.images.left_arm",
    "observation.images.right_arm",
]
DEFAULT_MASK_KEYS = CANONICAL_SEMANTIC_MASK_KEYS


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
    rgb_keys: list[str]
    state_keys: list[str]
    mask_target_keys: list[str]
    image_size: list[int] | None
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
    canonical_mask_definition: list[str]
    no_gripper: bool


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


def _mask_suffix_for_rgb(mask_key: str, rgb_key: str) -> str | None:
    prefix = f"{rgb_key}_"
    if mask_key.startswith(prefix):
        return mask_key[len(prefix) :]
    return None


def build_mask_layout(rgb_keys: list[str], mask_keys: list[str]) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Map each target mask key to (rgb view index, shared semantic output index)."""

    if len(rgb_keys) == 1:
        matched = [(_mask_suffix_for_rgb(key, rgb_keys[0]), key) for key in mask_keys]
        if not any(suffix is not None for suffix, _ in matched):
            return list(mask_keys), {key: (0, idx) for idx, key in enumerate(mask_keys)}

    suffixes: list[str] = []
    key_to_view_suffix: dict[str, tuple[int, str]] = {}
    for key in mask_keys:
        matches = [
            (view_idx, suffix)
            for view_idx, rgb_key in enumerate(rgb_keys)
            if (suffix := _mask_suffix_for_rgb(key, rgb_key)) is not None
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Could not assign mask key '{key}' to exactly one RGB key from {rgb_keys}. "
                "For multi-view training, mask keys should be named like '<rgb-key>_<semantic>', "
                "for example 'observation.images.front_object'."
            )
        view_idx, suffix = matches[0]
        key_to_view_suffix[key] = (view_idx, suffix)
        if suffix not in suffixes:
            suffixes.append(suffix)

    missing_pairs = []
    for rgb_key in rgb_keys:
        for suffix in suffixes:
            expected = f"{rgb_key}_{suffix}"
            if expected not in mask_keys:
                missing_pairs.append(expected)
    if missing_pairs:
        raise ValueError(
            "Multi-view MaskACT expects the same semantic mask suffixes for every RGB view. "
            f"Missing: {missing_pairs}"
        )

    suffix_to_idx = {suffix: idx for idx, suffix in enumerate(suffixes)}
    return suffixes, {
        key: (view_idx, suffix_to_idx[suffix]) for key, (view_idx, suffix) in key_to_view_suffix.items()
    }


def find_semantic_mask_index(mask_keys: list[str], semantic_name: str) -> int:
    suffix = f"_{semantic_name}"
    for idx, key in enumerate(mask_keys):
        if key.endswith(suffix) or key.rsplit(".", 1)[-1] == semantic_name:
            return idx
    raise ValueError(f"Could not find semantic mask '{semantic_name}' in mask keys: {mask_keys}")


class MaskACTPolicy(nn.Module):
    def __init__(
        self,
        act_policy: ACTPolicy,
        experiment: str,
        rgb_keys: list[str],
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
        self.rgb_keys = rgb_keys
        self.rgb_key = rgb_keys[0]
        self.mask_keys = mask_keys
        self.mask_suffixes, self.mask_key_map = build_mask_layout(rgb_keys, mask_keys)
        self.stats = stats
        self.latent_dim = latent_dim
        self.seg_loss_weight = seg_loss_weight
        self.action_loss_weight = action_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.metric_loss_weight = metric_loss_weight
        self.metric_eps = metric_eps
        self.bce = nn.BCEWithLogitsLoss()

        if self.experiment not in {"1A", "1B", "2A", "2B", "2C", "3", "4A", "4B", "4C", "5"}:
            raise ValueError(
                f"Unknown experiment '{experiment}'. Choose 1A, 1B, 2A, 2B, 2C, 3, 4A, 4B, 4C, or 5."
            )
        self.seg_net = UNetSegNet(
            out_masks=len(self.mask_suffixes),
            latent_dim=latent_dim,
            base_channels=unet_base_channels,
        )
        self.semantic_net = (
            SemanticLatentNet(
                num_masks=len(self.mask_suffixes),
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

    def reset(self) -> None:
        self.act_policy.reset()

    def _resize_inference_rgb(self, rgb: Tensor) -> Tensor:
        image_size = getattr(self, "inference_image_size", None)
        if image_size is None or tuple(rgb.shape[-2:]) == tuple(image_size):
            return rgb
        return F.interpolate(rgb, size=tuple(image_size), mode="bilinear", align_corners=False, antialias=True)

    def _get_rgb_inputs(self, batch: dict[str, Tensor], *, device: torch.device | None = None) -> list[Tensor]:
        rgbs = []
        missing = [key for key in self.rgb_keys if key not in batch]
        if missing:
            raise KeyError(
                f"Mask-ACT requires live RGB input(s) {self.rgb_keys}, but missing {missing}. "
                f"Received keys: {sorted(batch)}"
            )
        for key in self.rgb_keys:
            rgb = batch[key]
            if device is not None:
                rgb = rgb.to(device=device)
            rgbs.append(self._resize_inference_rgb(rgb.to(dtype=torch.float32)))
        return rgbs

    def _stack_logits_for_mask_keys(self, logits_by_view: list[Tensor]) -> Tensor:
        return torch.cat(
            [
                logits_by_view[view_idx][:, suffix_idx : suffix_idx + 1]
                for view_idx, suffix_idx in (self.mask_key_map[key] for key in self.mask_keys)
            ],
            dim=1,
        )

    @torch.no_grad()
    def _prepare_inference_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        self.eval()
        rgbs = self._get_rgb_inputs(batch)
        act_batch = dict(batch)

        if self.uses_semantic_latents():
            if self.semantic_net is None:
                raise RuntimeError("Experiment 5 semantic network is missing.")
            rgb_main_latent, rgb_semantic_latents = self.predict_inference_semantic_latents(rgbs)
            act_batch[OBS_ENV_STATE] = torch.cat(
                [rgb_main_latent, rgb_semantic_latents.reshape(rgb_semantic_latents.shape[0], -1)],
                dim=-1,
            )
        else:
            mask_logits, rgb_latent = self.predict_masks_and_latent_from_rgbs(rgbs)
            mask_probs = torch.sigmoid(mask_logits)

            if self.act_uses_latent():
                act_batch[OBS_ENV_STATE] = rgb_latent

            if self.act_uses_masks():
                for idx, key in enumerate(self.mask_keys):
                    mask_image = mask_probs[:, idx : idx + 1].repeat(1, 3, 1, 1)
                    act_batch[key] = self.normalize_visual_like(mask_image, key)

                if self.uses_mask_metrics():
                    metric_inputs = self.compute_mask_metrics(mask_probs)
                    if self.experiment == "4A":
                        act_batch[OBS_ENV_STATE] = metric_inputs
                    elif self.experiment == "4C":
                        act_batch[METRIC_SEED] = metric_inputs

        return act_batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Run RGB-only Mask-ACT inference and return one action from the ACT action queue."""
        return self.act_policy.select_action(self._prepare_inference_batch(batch))

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Run RGB-only Mask-ACT inference and return the complete predicted action chunk."""
        return self.act_policy.predict_action_chunk(self._prepare_inference_batch(batch))

    def mask_action_grad_enabled(self) -> bool:
        return self.experiment in {"1B", "2C", "4A", "4B", "4C"}

    def latent_action_grad_enabled(self) -> bool:
        return self.experiment in {"2B", "3"}

    def act_uses_masks(self) -> bool:
        return self.experiment in {"1A", "1B", "2A", "2B", "2C", "4A", "4B", "4C"}

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
        return self.predict_masks_and_latent_from_rgbs(self._get_rgb_inputs(raw_batch, device=device))

    def predict_masks_and_latent_from_rgbs(self, rgbs: list[Tensor]) -> tuple[Tensor, Tensor]:
        logits_by_view = []
        latents = []
        for rgb in rgbs:
            logits, latent = self.seg_net(rgb)
            logits_by_view.append(logits)
            latents.append(latent)
        return self._stack_logits_for_mask_keys(logits_by_view), torch.cat(latents, dim=-1)

    def predict_semantic_latents(
        self,
        raw_batch: dict[str, Tensor],
        device: torch.device,
        masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.semantic_net is None:
            raise RuntimeError("Semantic latent prediction is only available for experiment 5.")
        return self.predict_semantic_latents_from_rgbs_masks(
            self._get_rgb_inputs(raw_batch, device=device), masks
        )

    def predict_inference_semantic_latents(self, rgbs: list[Tensor]) -> tuple[Tensor, Tensor]:
        if self.semantic_net is None:
            raise RuntimeError("Semantic latent prediction is only available for experiment 5.")
        main_latents = []
        semantic_by_view = []
        for rgb in rgbs:
            rgb_main_latent, rgb_semantic_latents, _ = self.semantic_net(rgb, masks=None)
            main_latents.append(rgb_main_latent)
            semantic_by_view.append(rgb_semantic_latents)
        semantic_latents = torch.stack(
            [
                semantic_by_view[view_idx][:, suffix_idx]
                for view_idx, suffix_idx in (self.mask_key_map[key] for key in self.mask_keys)
            ],
            dim=1,
        )
        return torch.cat(main_latents, dim=-1), semantic_latents

    def predict_semantic_latents_from_rgbs_masks(
        self,
        rgbs: list[Tensor],
        masks: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if self.semantic_net is None:
            raise RuntimeError("Semantic latent prediction is only available for experiment 5.")
        main_latents = []
        rgb_semantic_by_view = []
        mask_semantic_by_view = []
        for view_idx, rgb in enumerate(rgbs):
            masks_for_view = None
            if masks is not None:
                view_masks = []
                for suffix_idx in range(len(self.mask_suffixes)):
                    matching = [
                        key_idx
                        for key_idx, key in enumerate(self.mask_keys)
                        if self.mask_key_map[key] == (view_idx, suffix_idx)
                    ]
                    if not matching:
                        raise KeyError(
                            f"No mask target found for RGB key '{self.rgb_keys[view_idx]}' "
                            f"and semantic suffix '{self.mask_suffixes[suffix_idx]}'."
                        )
                    view_masks.append(masks[:, matching[0] : matching[0] + 1])
                masks_for_view = torch.cat(view_masks, dim=1)
            rgb_main_latent, rgb_semantic_latents, mask_semantic_latents = self.semantic_net(rgb, masks_for_view)
            main_latents.append(rgb_main_latent)
            rgb_semantic_by_view.append(rgb_semantic_latents)
            if mask_semantic_latents is not None:
                mask_semantic_by_view.append(mask_semantic_latents)

        rgb_semantic_latents = torch.stack(
            [
                rgb_semantic_by_view[view_idx][:, suffix_idx]
                for view_idx, suffix_idx in (self.mask_key_map[key] for key in self.mask_keys)
            ],
            dim=1,
        )
        mask_semantic_latents = None
        if mask_semantic_by_view:
            mask_semantic_latents = torch.stack(
                [
                    mask_semantic_by_view[view_idx][:, suffix_idx]
                    for view_idx, suffix_idx in (self.mask_key_map[key] for key in self.mask_keys)
                ],
                dim=1,
            )
        return torch.cat(main_latents, dim=-1), rgb_semantic_latents, mask_semantic_latents

    def compute_mask_metrics(self, masks: Tensor) -> Tensor:
        object_idx = find_semantic_mask_index(self.mask_keys, "object")
        region_idx = find_semantic_mask_index(self.mask_keys, "region")
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
    parser.add_argument(
        "--experiment",
        choices=["1A", "1B", "2A", "2B", "2C", "3", "4A", "4B", "4C", "5"],
        required=True,
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--rgb-key", default=DEFAULT_RGB_KEY)
    parser.add_argument(
        "--rgb-keys",
        nargs="+",
        default=None,
        help="One or more RGB image keys. Overrides --rgb-key when provided.",
    )
    parser.add_argument("--state-keys", nargs="*", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--mask-target-keys", nargs="+", default=DEFAULT_MASK_KEYS)
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Drop gripper dimensions from action and selected state features before training.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Resize RGB and all mask targets to this aligned training resolution.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train/mask_act"))
    parser.add_argument("--view-root", type=Path, default=None)
    parser.add_argument("--rebuild-view", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume MaskACT training from a checkpoint training_state.pt file or checkpoint_step_* directory.",
    )
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


def validate_image_size(image_size: list[int] | tuple[int, int] | None) -> tuple[int, int] | None:
    if image_size is None:
        return None
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"--image-size values must be positive, got {height} {width}.")
    return height, width


def resize_training_visuals(
    raw_batch: dict[str, Any],
    rgb_keys: list[str],
    mask_keys: list[str],
    image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    """Resize aligned RGB/mask tensors while preserving binary mask boundaries."""

    if image_size is None:
        return raw_batch

    resized_batch = dict(raw_batch)
    visual_keys = [*rgb_keys, *mask_keys]
    for key in visual_keys:
        image = raw_batch[key]
        if not isinstance(image, Tensor) or image.ndim != 4:
            raise ValueError(f"Expected batched BCHW tensor for '{key}', got {type(image).__name__}.")
        if tuple(image.shape[-2:]) == image_size:
            continue
        if key in rgb_keys:
            resized_batch[key] = F.interpolate(image, size=image_size, mode="bilinear", align_corners=False, antialias=True)
        else:
            resized_batch[key] = F.interpolate(image, size=image_size, mode="nearest")
    return resized_batch


def load_dataset_features(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    return info.get("features", {})


def normalize_dataset_keys(args: argparse.Namespace, source_root: Path) -> None:
    features = load_dataset_features(source_root)
    available_keys = set(features)

    if args.rgb_keys is None:
        args.rgb_keys = [args.rgb_key]

    normalized_rgb_keys = []
    for rgb_key in args.rgb_keys:
        if rgb_key not in available_keys:
            if rgb_key == DEFAULT_RGB_KEY and SOARM_RGB_KEY in available_keys:
                print(f"RGB key '{DEFAULT_RGB_KEY}' not found; using '{SOARM_RGB_KEY}' from this dataset.")
                rgb_key = SOARM_RGB_KEY
            else:
                image_keys = sorted(key for key in available_keys if key.startswith("observation.images."))
                raise KeyError(
                    f"RGB key '{rgb_key}' is not in dataset features. "
                    f"Available image keys: {image_keys}"
                )
        normalized_rgb_keys.append(rgb_key)
    args.rgb_keys = normalized_rgb_keys
    args.rgb_key = args.rgb_keys[0]

    missing_mask_keys = [key for key in args.mask_target_keys if key not in available_keys]
    if missing_mask_keys:
        image_keys = sorted(key for key in available_keys if key.startswith("observation.images."))
        raise KeyError(
            f"Missing mask target key(s): {missing_mask_keys}. "
            f"Available image keys: {image_keys}"
        )

    build_mask_layout(args.rgb_keys, list(args.mask_target_keys))

    if list(args.mask_target_keys) == CANONICAL_SEMANTIC_MASK_KEYS:
        return

    canonical_available = all(key in available_keys for key in CANONICAL_SEMANTIC_MASK_KEYS)
    if canonical_available:
        print(
            "Using custom mask target keys. The canonical experiment-5-compatible definition is: "
            f"{CANONICAL_SEMANTIC_MASK_KEYS}"
        )


def reshape_visual_stats_for_channel_first(
    stats: dict[str, dict[str, Any]],
    features: dict[str, PolicyFeature],
) -> dict[str, dict[str, Any]]:
    """Make visual stats broadcast against LeRobot channel-first image tensors."""

    reshaped_stats = deepcopy(stats)
    for key, feature in features.items():
        if feature.type != FeatureType.VISUAL or key not in reshaped_stats or len(feature.shape) != 3:
            continue

        channels = feature.shape[0]
        for stat_name, value in list(reshaped_stats[key].items()):
            shape = getattr(value, "shape", None)
            if shape is not None:
                if tuple(shape) != (channels,):
                    continue
                flat_values = value.tolist()
            elif isinstance(value, list):
                if len(value) != channels:
                    continue
                if value and isinstance(value[0], list):
                    continue
                flat_values = value
            else:
                continue
            reshaped_stats[key][stat_name] = [[[channel_value]] for channel_value in flat_values]

    return reshaped_stats


def act_image_keys_for_experiment(args: argparse.Namespace) -> list[str]:
    experiment = args.experiment.upper()
    if experiment in {"1A", "1B"}:
        return list(args.mask_target_keys)
    if experiment in {"2A", "2B"}:
        return list(args.mask_target_keys)
    if experiment == "2C":
        return [*args.mask_target_keys, *args.rgb_keys]
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
    if args.experiment.upper() in {"4A", "4B", "4C"}:
        try:
            find_semantic_mask_index(args.mask_target_keys, "object")
            find_semantic_mask_index(args.mask_target_keys, "region")
        except ValueError as exc:
            raise KeyError(
                f"Experiment {args.experiment} requires object and region mask target keys for metric computation."
            ) from exc

    features = dataset_to_policy_features(meta.features)
    act_image_keys = act_image_keys_for_experiment(args)
    input_keys = [*args.state_keys, *act_image_keys]
    missing = [key for key in [*input_keys, "action"] if key not in features]
    if missing:
        raise KeyError(f"Missing feature(s) for ACT policy: {missing}")

    input_features = {key: features[key] for key in input_keys}
    if args.image_size is not None:
        height, width = args.image_size
        for key in act_image_keys:
            feature = input_features[key]
            input_features[key] = PolicyFeature(type=feature.type, shape=(feature.shape[0], height, width))
    if act_uses_latent(args.experiment):
        input_features[OBS_ENV_STATE] = PolicyFeature(
            type=FeatureType.ENV, shape=(args.latent_dim * len(args.rgb_keys),)
        )
    if act_uses_metric_env_state(args.experiment):
        input_features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(2,))
    if act_uses_semantic_env_state(args.experiment):
        input_features[OBS_ENV_STATE] = PolicyFeature(
            type=FeatureType.ENV,
            shape=(args.latent_dim * (len(args.rgb_keys) + len(args.mask_target_keys)),),
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
        rgb_keys=list(args.rgb_keys),
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
        rgb_keys=list(args.rgb_keys),
        state_keys=list(args.state_keys),
        mask_target_keys=list(args.mask_target_keys),
        image_size=list(args.image_size) if args.image_size is not None else None,
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
        canonical_mask_definition=list(CANONICAL_SEMANTIC_MASK_KEYS),
        no_gripper=args.no_gripper,
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


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "training_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path


def load_checkpoint(
    checkpoint_path: Path,
    model: MaskACTPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint model keys do not match current configuration. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)
    return int(checkpoint["step"])


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

    tiles = [
        make_labeled_tile(raw_batch[rgb_key][0], f"rgb {rgb_key.rsplit('.', 1)[-1]}")
        for rgb_key in model.rgb_keys
    ]
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

    def __init__(self, output_dir: Path, run_config: dict[str, Any], initial_step: int = 0):
        self.output_dir = output_dir
        self.initial_step = initial_step
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
            "initial_step": initial_step,
        }
        self._write_json()

    def log(
        self,
        step: int,
        total_steps: int,
        logs: dict[str, float],
        grad_norm: float,
        interval_elapsed_s: float,
        elapsed_total_s: float,
        lr: float,
    ) -> None:
        completed_this_run = max(step - self.initial_step, 0)
        steps_per_second = completed_this_run / elapsed_total_s if elapsed_total_s > 0 else 0.0
        eta_s = (total_steps - step) / steps_per_second if steps_per_second > 0 else None
        record = {
            "step": step,
            "total_steps": total_steps,
            "wall_time_unix": time.time(),
            "interval_elapsed_s": interval_elapsed_s,
            "elapsed_total_s": elapsed_total_s,
            "avg_step_time_s": elapsed_total_s / completed_this_run if completed_this_run > 0 else None,
            "steps_per_second": steps_per_second,
            "eta_s": eta_s,
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
        plot_training_curves(
            self.jsonl_path,
            self.metrics_dir / "train_loss_curve.png",
            "MaskACT training losses",
        )

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
        if key in {
            "interval_elapsed_s",
            "elapsed_total_s",
            "avg_step_time_s",
            "steps_per_second",
            "eta_s",
        }:
            return f"time/{key}"
        return f"train/{key}"


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
    args.image_size = validate_image_size(args.image_size)
    if isinstance(args.pretrained_backbone_weights, str) and args.pretrained_backbone_weights.lower() in {
        "none",
        "null",
        "",
    }:
        args.pretrained_backbone_weights = None
    ensure_device_is_usable(args.device)
    device = torch.device(args.device)
    source_root = args.root.resolve()
    normalize_dataset_keys(args, source_root)

    if args.overwrite_output and args.resume_checkpoint is not None:
        raise ValueError("--overwrite-output cannot be used together with --resume-checkpoint.")

    if args.output_dir.exists() and args.overwrite_output:
        shutil.rmtree(args.output_dir)

    image_keys_in_view = sorted({*args.rgb_keys, *args.mask_target_keys})
    view_root = args.view_root or (args.output_dir.parent / "dataset_views" / args.output_dir.name)
    filtered_root = make_filtered_dataset_view(
        source_root=source_root,
        view_root=view_root,
        image_keys=image_keys_in_view,
        state_keys=args.state_keys,
        rebuild=args.rebuild_view,
        no_gripper=args.no_gripper,
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

    act_features = dataset_to_policy_features(meta.features)
    act_input_keys = [*args.state_keys, *act_image_keys_for_experiment(args)]
    act_input_features = {key: act_features[key] for key in act_input_keys if key in act_features}
    stats = reshape_visual_stats_for_channel_first(meta.stats, act_input_features)

    model = make_policy(args, meta, stats=stats).to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=model.config,
        dataset_stats=stats,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume_checkpoint is not None:
        start_step = load_checkpoint(args.resume_checkpoint, model, optimizer, device)
        if start_step >= args.steps:
            raise ValueError(
                f"Checkpoint step ({start_step}) is already >= requested total steps ({args.steps})."
            )
        print(f"Resumed checkpoint: {resolve_checkpoint_path(args.resume_checkpoint)}")
        print(f"Resuming from step {start_step}; target total steps {args.steps}.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    dl_iter = cycle(dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Experiment: {args.experiment}")
    print(f"Dataset view: {filtered_root}")
    print(f"RGB keys: {args.rgb_keys}")
    print(f"ACT image keys: {act_image_keys_for_experiment(args)}")
    print(f"Mask supervision keys: {args.mask_target_keys}")
    print(f"Training image size: {args.image_size or 'dataset native resolution'}")
    if args.no_gripper:
        print("No gripper: dropped gripper dimensions when present in action and selected state features.")
    print(f"Output dir: {args.output_dir}")
    print(f"Train log: {log_path}")
    run_config_path = args.output_dir / "mask_act_run_config.json"
    with open(run_config_path) as f:
        metrics_run_config = json.load(f)
    metrics_logger = TrainingMetricsLogger(args.output_dir, metrics_run_config, initial_step=start_step)
    print(f"Metrics JSON: {metrics_logger.json_path}")
    print(f"Metrics JSONL: {metrics_logger.jsonl_path}")
    print(f"TensorBoard logdir: {metrics_logger.tensorboard_dir}")

    model.train()
    training_start_time = time.perf_counter()
    interval_start_time = training_start_time
    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        desc=f"train {args.experiment}",
        unit="step",
        dynamic_ncols=True,
        mininterval=1.0,
        smoothing=0.1,
        initial=start_step,
        total=args.steps,
    )
    try:
        for step in progress:
            raw_batch = next(dl_iter)
            raw_batch = resize_training_visuals(
                raw_batch,
                rgb_keys=args.rgb_keys,
                mask_keys=args.mask_target_keys,
                image_size=args.image_size,
            )
            batch = preprocessor(raw_batch)

            optimizer.zero_grad(set_to_none=True)
            loss, logs = model(batch, raw_batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            if args.log_freq > 0 and (step % args.log_freq == 0 or step == args.steps):
                now = time.perf_counter()
                interval_elapsed = now - interval_start_time
                elapsed_total = now - training_start_time
                grad_norm_float = float(grad_norm.detach().cpu() if isinstance(grad_norm, Tensor) else grad_norm)
                lr = float(optimizer.param_groups[0]["lr"])
                metrics_logger.log(
                    step=step,
                    total_steps=args.steps,
                    logs=logs,
                    grad_norm=grad_norm_float,
                    interval_elapsed_s=interval_elapsed,
                    elapsed_total_s=elapsed_total,
                    lr=lr,
                )
                extra_loss = logs.get("seg_loss", logs.get("semantic_loss", 0.0))
                extra_name = "seg" if "seg_loss" in logs else "semantic"
                progress.set_postfix(
                    loss=f"{logs['loss']:.4f}",
                    action=f"{logs['action_loss']:.4f}",
                    **{extra_name: f"{extra_loss:.4f}"},
                )
                interval_start_time = now

            if args.save_freq > 0 and (step % args.save_freq == 0 or step == args.steps):
                checkpoint_dir = save_checkpoint(args.output_dir, step, model, optimizer)
                save_mask_preview(checkpoint_dir, step, model, raw_batch)
    finally:
        progress.close()
        metrics_logger.close()


if __name__ == "__main__":
    main()
