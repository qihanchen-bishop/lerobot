#!/usr/bin/env python
"""Train ACT variants with mask supervision.

Experiments:
  1A: RGB -> U-Net -> predicted masks -> ACT. Seg loss trains U-Net; action loss trains ACT only.
  1B: Same as 1A, but action loss also backpropagates through the U-Net mask path.
  4A: Same as 1B, with sqrt(object area ratio) and object-region centroid distance as ACT env state.
  4B: Same as 1B, with two ACT encoder metric tokens supervised to predict the two mask metrics.
  4C: Same as 1B, with the two metrics predicted by the ACT decoder and fed back chunk-autoregressively.
  Legacy single-view experiments default to five binary semantic masks in this order:
      occluder, object, region, left_arm, right_arm.
  5:  RGB-only inference with latent semantic distillation. During training, ACT receives an RGB main latent
      plus those five mask-encoder semantic teacher latents; RGB semantic heads are aligned to the teachers.
  2A: ACT sees predicted masks plus a pooled RGB encoder latent. Seg loss trains U-Net; action loss
      trains ACT only.
  2B: ACT sees predicted masks plus a pooled RGB encoder latent. Action loss backpropagates through
      the U-Net mask path, while the pooled latent path is detached.
  2C: ACT sees predicted masks plus the original RGB images. Masks and RGB share ACT's image backbone,
      and action loss also backpropagates through the U-Net mask path.
  SEM-1: Each view's masks are merged into one soft semantic RGB map. ACT sees semantic maps plus
      the original RGB views through one shared image backbone; no pooled RGB latent is used.
  SEM-1-V2: Each view keeps all five soft class probabilities. A lightweight semantic adapter maps
      them to a residual for the corresponding RGB ResNet feature map, so ACT receives no extra image tokens.
  ViewFus-v1: A shared two-view U-Net predicts semantics, then a homography-aligned fusion adapter
      distills the label-derived side-assisted front map. ACT sees front RGB, side RGB, and the fused
      front semantic map through one shared ResNet. Teacher forcing protects ACT while segmentation learns.
  ASEM-1: Same inputs and losses as SEM-1, but the ACT action L1 gradient also supervises the
      segmentation network. The task gradient is warmed up, norm-limited relative to the supervised
      segmentation gradient, and projected when it conflicts with segmentation.
  SSACT-1: SEM-1 plus an expert-action-conditioned probabilistic semantic dynamics model and an
      automatically pseudo-labeled five-phase history model. Future semantic targets are loaded from
      an offline cache; predicted phase probabilities condition ACT.
  SSACT-3: SEM-1-V2 plus event-derived expose/separate/transport/restore/done supervision,
      Phase/Progress/Event/Transition/Relation heads, and phase-conditioned relation attention + FiLM.
      Pseudo-label quality continuously weights auxiliary losses; action gradients never alter segmentation
      or stage heads, and predicted semantic/phase conditioning is introduced only after warmup.
  SEM-2: Same soft semantic maps as SEM-1, but ACT sees no original RGB images and no pooled latent.
  UNET-SEM: A frozen pretrained TinyUNet produces soft semantic maps. ACT receives those maps and
      the corresponding RGB views with gated camera and modality identity embeddings.
  3:  ACT sees only the pooled RGB encoder latent. U-Net mask decoder is kept as an auxiliary task.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
from lerobot.policies.act.modeling_act import IMAGE_FEATURE_RESIDUALS, METRIC_INPUT, METRIC_PRED, METRIC_SEED
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

from train_lerobot_policy import ensure_device_is_usable, make_filtered_dataset_view
from training_artifacts import plot_training_curves, tee_output
from pretrained_semantic_segmenter import FrozenTinyUNetSegmenter
from semantic_servo import (
    ActionConditionedSemanticDynamics,
    PhaseHistoryModel,
    PhaseSemanticFeatureExtractor,
    SoftSemanticStateExtractor,
)
from stage_semantic import (
    EVENT_NAMES as STAGE_EVENT_NAMES,
    PHASE_NAMES as STAGE_PHASE_NAMES,
    RELATION_NAMES as STAGE_RELATION_NAMES,
    TRANSITION_NAMES as STAGE_TRANSITION_NAMES,
    PhaseConditionedSemanticAdapter,
    StageAwareTemporalModel,
    StageSupervisionStore,
    stage_losses,
)


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
STAGE_AWARE_EXPERIMENTS = {"SSACT-3"}
VIEW_FUSION_EXPERIMENTS = {"VIEWFUS-V1"}
FROZEN_SEMANTIC_EXPERIMENTS = {"UNET-SEM"}
VIEWFUS_FUSED_FRONT_KEY = "observation.images.front_fused_semantic"
SEMANTIC_FEATURE_FUSION_EXPERIMENTS = {"SEM-1-V2", *STAGE_AWARE_EXPERIMENTS}
SEMANTIC_EXPERIMENTS = {
    "SEM-1",
    "SEM-1-V2",
    "ASEM-1",
    "SSACT-1",
    "SSACT-3",
    "SEM-2",
    "VIEWFUS-V1",
    *FROZEN_SEMANTIC_EXPERIMENTS,
}
ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS = {"ASEM-1"}
PREDICTIVE_SEMANTIC_EXPERIMENTS = {"SSACT-1"}
PHASE_CONDITIONED_EXPERIMENTS = {"SSACT-1", *STAGE_AWARE_EXPERIMENTS}
SEMANTIC_CLASSES = ("occluder", "object", "region", "tool")
SEMANTIC_STATE_FEATURES_PER_VIEW = (
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
SEMANTIC_PALETTE_BY_CLASS = {
    "background": (0.0, 0.0, 0.0),
    "occluder": (50 / 255, 160 / 255, 1.0),
    "object": (1.0, 80 / 255, 80 / 255),
    "region": (60 / 255, 220 / 255, 120 / 255),
    "tool": (210 / 255, 210 / 255, 80 / 255),
}
SEMANTIC_CLASS_WEIGHT_BY_NAME = {
    "background": 0.5,
    "occluder": 1.0,
    "object": 2.0,
    "region": 1.0,
    "tool": 1.0,
}
IMAGENET_VISUAL_MEAN = (0.485, 0.456, 0.406)
IMAGENET_VISUAL_STD = (0.229, 0.224, 0.225)


warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)


def mask_quality_filename(mask_key: str) -> str:
    return mask_key.replace("/", "__").replace(".", "_") + ".npz"


class MaskQualityStore:
    """In-memory frame-level quality generated by sam2_mask_quality.py."""

    def __init__(
        self,
        quality_dir: Path,
        mask_keys: list[str],
        *,
        total_frames: int,
        minimum_score: float,
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(f"mask_quality_min_score must be in [0, 1], got {minimum_score}.")
        self.quality_dir = quality_dir
        self.minimum_score = minimum_score
        self.valid_by_key: dict[str, np.ndarray] = {}
        self.score_by_key: dict[str, np.ndarray] = {}
        for key in mask_keys:
            path = quality_dir / mask_quality_filename(key)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Mask quality report is missing for '{key}': {path}. "
                    "Run mycode/sam2_mask_quality.py before training."
                )
            with np.load(path) as report:
                if "quality_score" not in report or "uncertain" not in report:
                    raise KeyError(f"Mask quality report lacks quality_score/uncertain arrays: {path}")
                score = report["quality_score"].astype(np.float32, copy=True)
                uncertain = report["uncertain"].astype(bool, copy=True)
            if score.shape != (total_frames,) or uncertain.shape != (total_frames,):
                raise ValueError(
                    f"Mask quality report '{path}' has {score.shape[0]} frames; expected {total_frames}."
                )
            if not np.isfinite(score).all():
                missing_count = int((~np.isfinite(score)).sum())
                raise ValueError(
                    f"Mask quality report '{path}' is incomplete ({missing_count} frames unprocessed). "
                    "Generate the full report without --max-episodes before training."
                )
            self.score_by_key[key] = score
            self.valid_by_key[key] = np.isfinite(score) & (score >= minimum_score) & ~uncertain

    def add_batch_quality(
        self,
        raw_batch: dict[str, Any],
        *,
        mask_keys: list[str],
        mask_key_map: dict[str, tuple[int, int]],
        num_views: int,
        num_classes: int,
    ) -> dict[str, Any]:
        indices = raw_batch["index"].detach().cpu().numpy().astype(np.int64)
        current_valid = np.zeros((indices.shape[0], num_views, num_classes), dtype=bool)
        current_scores = np.zeros((indices.shape[0], num_views, num_classes), dtype=np.float32)
        for key in mask_keys:
            view_idx, class_idx = mask_key_map[key]
            current_valid[:, view_idx, class_idx] = self.valid_by_key[key][indices]
            current_scores[:, view_idx, class_idx] = self.score_by_key[key][indices]

        enriched = dict(raw_batch)
        enriched["mask_quality_current_valid"] = torch.from_numpy(current_valid)
        enriched["mask_quality_current_score"] = torch.from_numpy(current_scores)
        return enriched


def mask_quality_scores_to_weights(
    scores: Tensor,
    *,
    full_score: float,
    gamma: float,
) -> Tensor:
    """Map label-quality scores continuously to segmentation-loss weights."""
    if not 0.0 < full_score <= 1.0:
        raise ValueError(f"mask_quality_full_score must be in (0, 1], got {full_score}.")
    if gamma <= 0.0:
        raise ValueError(f"mask_quality_weight_gamma must be positive, got {gamma}.")
    normalized = scores.to(dtype=torch.float32).clamp(0.0, 1.0) / full_score
    return normalized.clamp(max=1.0).pow(gamma)


class SemanticStateStore:
    """Offline future semantic targets and label-quality gates for SSACT-1."""

    def __init__(
        self,
        path: Path,
        *,
        total_frames: int,
        rgb_keys: list[str],
        minimum_quality_score: float,
    ) -> None:
        if not 0.0 <= minimum_quality_score <= 1.0:
            raise ValueError("minimum_quality_score must be in [0, 1].")
        if not path.is_file():
            raise FileNotFoundError(
                f"Offline semantic states are missing: {path}. "
                "Run mycode/precompute_semantic_states.py before SSACT-1 training."
            )
        with np.load(path) as archive:
            required = {
                "semantic_states",
                "quality_score",
                "uncertain",
                "episode_end_index",
                "processed",
                "rgb_keys",
                "class_names",
                "feature_names",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise KeyError(f"Semantic-state file '{path}' is missing {missing}.")
            self.semantic_states = archive["semantic_states"].astype(np.float32, copy=True)
            self.quality_score = archive["quality_score"].astype(np.float32, copy=True)
            self.uncertain = archive["uncertain"].astype(bool, copy=True)
            self.episode_end_index = archive["episode_end_index"].astype(np.int64, copy=True)
            processed = archive["processed"].astype(bool, copy=True)
            stored_rgb_keys = archive["rgb_keys"].astype(str).tolist()
            stored_class_names = archive["class_names"].astype(str).tolist()
            self.feature_names = archive["feature_names"].astype(str).tolist()

        expected_state_shape = (total_frames, len(rgb_keys), len(SEMANTIC_STATE_FEATURES_PER_VIEW))
        expected_quality_shape = (total_frames, len(rgb_keys), len(SEMANTIC_CLASSES))
        if self.semantic_states.shape != expected_state_shape:
            raise ValueError(
                f"Semantic states have shape {self.semantic_states.shape}; expected {expected_state_shape}."
            )
        if self.quality_score.shape != expected_quality_shape or self.uncertain.shape != expected_quality_shape:
            raise ValueError(
                "Semantic-state quality arrays must have shape "
                f"{expected_quality_shape}; got {self.quality_score.shape} and {self.uncertain.shape}."
            )
        if self.episode_end_index.shape != (total_frames,) or processed.shape != (total_frames,):
            raise ValueError("Semantic-state episode and processed arrays must cover every dataset frame.")
        if stored_rgb_keys != rgb_keys:
            raise ValueError(
                f"Semantic-state views are {stored_rgb_keys}; training requested {rgb_keys}. "
                "Regenerate the cache for the exact ordered view list."
            )
        if stored_class_names != list(SEMANTIC_CLASSES):
            raise ValueError(
                f"Semantic-state classes are {stored_class_names}; expected {list(SEMANTIC_CLASSES)}."
            )
        if self.feature_names != list(SEMANTIC_STATE_FEATURES_PER_VIEW):
            raise ValueError(
                f"Semantic-state features are {self.feature_names}; expected "
                f"{list(SEMANTIC_STATE_FEATURES_PER_VIEW)}."
            )
        if not processed.all():
            raise ValueError(f"Semantic-state file '{path}' does not cover {(~processed).sum()} frames.")
        if not np.isfinite(self.semantic_states).all() or not np.isfinite(self.quality_score).all():
            raise ValueError(f"Semantic-state file '{path}' contains non-finite values.")
        if np.any((self.semantic_states < 0.0) | (self.semantic_states > 1.0)):
            raise ValueError(f"Semantic-state file '{path}' contains values outside [0, 1].")
        frame_indices = np.arange(total_frames, dtype=np.int64)
        if np.any((self.episode_end_index <= frame_indices) | (self.episode_end_index > total_frames)):
            raise ValueError(f"Semantic-state file '{path}' contains invalid episode end indices.")

        self.path = path
        self.minimum_quality_score = minimum_quality_score

    def add_batch_semantics(
        self,
        raw_batch: dict[str, Any],
        *,
        prediction_offsets: tuple[int, ...],
    ) -> dict[str, Any]:
        indices = raw_batch["index"].detach().cpu().numpy().astype(np.int64)
        if np.any((indices < 0) | (indices >= self.semantic_states.shape[0])):
            raise IndexError("Batch contains frame indices outside the semantic-state cache.")

        current_scores = self.quality_score[indices]
        current_valid = (
            (current_scores >= self.minimum_quality_score)
            & ~self.uncertain[indices]
        )

        offsets = np.asarray(prediction_offsets, dtype=np.int64)
        future_indices = indices[:, None] + offsets[None, :]
        within_episode = future_indices < self.episode_end_index[indices, None]
        clipped_indices = np.clip(future_indices, 0, self.semantic_states.shape[0] - 1)
        future_scores_by_class = self.quality_score[clipped_indices]
        future_uncertain = self.uncertain[clipped_indices]
        future_valid = (
            within_episode
            & (future_scores_by_class >= self.minimum_quality_score).all(axis=(2, 3))
            & ~future_uncertain.any(axis=(2, 3))
        )

        enriched = dict(raw_batch)
        enriched["mask_quality_current_valid"] = torch.from_numpy(current_valid)
        enriched["mask_quality_current_score"] = torch.from_numpy(current_scores)
        enriched["future_semantic_states"] = torch.from_numpy(
            self.semantic_states[clipped_indices].reshape(indices.shape[0], len(offsets), -1)
        )
        enriched["future_semantic_valid"] = torch.from_numpy(future_valid)
        enriched["mask_quality_future_valid"] = torch.from_numpy(future_valid)
        enriched["mask_quality_future_score"] = torch.from_numpy(
            future_scores_by_class.min(axis=(2, 3))
        )
        return enriched


class SemanticPhaseStore:
    """Frame-level soft phase labels and semantic histories generated offline."""

    def __init__(
        self,
        path: Path,
        *,
        total_frames: int,
        num_views: int,
        history_length: int,
        history_stride: int,
        minimum_confidence: float,
    ) -> None:
        if history_length <= 0 or history_stride <= 0:
            raise ValueError("Phase history length and stride must be positive.")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("phase_min_confidence must be in [0, 1].")
        if not path.is_file():
            raise FileNotFoundError(
                f"Semantic phase labels are missing: {path}. "
                "Run mycode/semantic_phase_labels.py before SSACT-1 training."
            )
        with np.load(path) as labels:
            required = {
                "phase_probabilities",
                "phase_confidence",
                "semantic_features",
                "episode_start_index",
                "feature_reliability",
            }
            missing = sorted(required.difference(labels.files))
            if missing:
                raise KeyError(f"Phase label file '{path}' is missing {missing}.")
            self.phase_probabilities = labels["phase_probabilities"].astype(np.float32, copy=True)
            self.phase_confidence = labels["phase_confidence"].astype(np.float32, copy=True)
            self.semantic_features = labels["semantic_features"].astype(np.float32, copy=True)
            self.episode_start_index = labels["episode_start_index"].astype(np.int64, copy=True)
            self.feature_reliability = labels["feature_reliability"].astype(np.float32, copy=True)
        if self.phase_probabilities.shape != (total_frames, 5):
            raise ValueError(
                f"Phase probabilities have shape {self.phase_probabilities.shape}; expected ({total_frames}, 5)."
            )
        expected_feature_dim = num_views * 12
        if self.semantic_features.shape != (total_frames, expected_feature_dim):
            raise ValueError(
                f"Phase semantic features have shape {self.semantic_features.shape}; "
                f"expected ({total_frames}, {expected_feature_dim})."
            )
        if self.phase_confidence.shape != (total_frames,) or self.episode_start_index.shape != (total_frames,):
            raise ValueError("Phase confidence and episode start arrays must cover every dataset frame.")
        if self.feature_reliability.shape != (num_views, len(SEMANTIC_CLASSES)):
            raise ValueError(
                f"Phase reliability has shape {self.feature_reliability.shape}; "
                f"expected ({num_views}, {len(SEMANTIC_CLASSES)})."
            )
        if not all(
            np.isfinite(values).all()
            for values in (self.phase_probabilities, self.phase_confidence, self.semantic_features)
        ):
            raise ValueError(f"Phase label file '{path}' contains non-finite values.")
        self.path = path
        self.history_length = history_length
        self.history_stride = history_stride
        self.minimum_confidence = minimum_confidence

    @property
    def feature_dim(self) -> int:
        return self.semantic_features.shape[1]

    def add_batch_phase(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        indices = raw_batch["index"].detach().cpu().numpy().astype(np.int64)
        history_offsets = np.arange(self.history_length - 1, -1, -1, dtype=np.int64) * self.history_stride
        history_indices = indices[:, None] - history_offsets[None, :]
        history_indices = np.maximum(history_indices, self.episode_start_index[indices, None])
        confidence = self.phase_confidence[indices]

        enriched = dict(raw_batch)
        enriched["phase_semantic_history"] = torch.from_numpy(self.semantic_features[history_indices])
        enriched["phase_target"] = torch.from_numpy(self.phase_probabilities[indices])
        enriched["phase_confidence"] = torch.from_numpy(
            np.where(confidence >= self.minimum_confidence, confidence, 0.0).astype(np.float32)
        )
        return enriched


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
    pretrained_segmentation_checkpoint: str | None
    identity_embedding_gate_init: float
    canonical_mask_definition: list[str]
    no_gripper: bool
    dice_loss_weight: float
    semantic_temperature: float
    semantic_adapter_base_channels: int
    semantic_fusion_warmup_steps: int
    semantic_fusion_ramp_steps: int
    viewfusion_homography_json: str
    viewfusion_homography: list[list[float]] | None
    viewfusion_adapter_base_channels: int
    viewfusion_loss_weight: float
    viewfusion_teacher_forcing_steps: int
    viewfusion_teacher_forcing_ramp_steps: int
    action_to_seg_grad_ratio: float
    action_to_seg_warmup_steps: int
    action_to_seg_ramp_steps: int
    action_to_seg_conflict_projection: bool
    semantic_prediction_offsets: list[int]
    semantic_dynamics_hidden_dim: int
    semantic_dynamics_loss_weight: float
    semantic_states: str | None
    mask_quality_dir: str | None
    mask_quality_min_score: float
    mask_quality_weighting: str
    mask_quality_full_score: float
    mask_quality_weight_gamma: float
    phase_labels: str | None
    phase_history_length: int
    phase_history_stride: int
    phase_hidden_dim: int
    phase_loss_weight: float
    phase_min_confidence: float
    phase_teacher_forcing_steps: int
    phase_teacher_forcing_ramp_steps: int
    phase_feature_reliability: list[list[float]] | None
    stage_supervision: str | None
    stage_conditioning_mode: str
    stage_predicted_input_warmup_steps: int
    stage_predicted_input_ramp_steps: int
    stage_phase_loss_weight: float
    stage_event_loss_weight: float
    stage_progress_loss_weight: float
    stage_transition_loss_weight: float
    stage_relation_loss_weight: float
    stage_attention_regularization_weight: float
    stage_feature_dim: int | None
    seed: int


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


class HomographyViewFusionAdapter(nn.Module):
    """Fuse soft semantics using a calibrated side-object right-bottom position prior."""

    def __init__(self, num_classes: int, homography: list[list[float]], base_channels: int = 16):
        super().__init__()
        if base_channels <= 0:
            raise ValueError(f"View-fusion base channels must be positive, got {base_channels}.")
        matrix = torch.as_tensor(homography, dtype=torch.float32)
        if matrix.shape != (3, 3) or not torch.isfinite(matrix).all():
            raise ValueError(f"Expected a finite 3x3 side-to-front homography, got {matrix}.")
        self.register_buffer("side_to_front_homography", matrix)
        self.object_class_index = 2  # background, occluder, object, region, tool
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.inc = DoubleConv(2 * num_classes + 1, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.up1 = Up(c3, c2, c2)
        self.up2 = Up(c2, c1, c1)
        self.output = nn.Conv2d(c1, 3, kernel_size=1)

    def projected_object_prior(self, side_probabilities: Tensor) -> Tensor:
        batch_size, num_classes, height, width = side_probabilities.shape
        if self.object_class_index >= num_classes:
            raise ValueError(
                f"Object class index {self.object_class_index} is invalid for {num_classes} classes."
            )
        yy, xx = torch.meshgrid(
            torch.arange(height, device=side_probabilities.device, dtype=side_probabilities.dtype),
            torch.arange(width, device=side_probabilities.device, dtype=side_probabilities.dtype),
            indexing="ij",
        )
        x_normalized = xx / max(width - 1, 1)
        y_normalized = yy / max(height - 1, 1)
        corner_bias = torch.exp(8.0 * (x_normalized + y_normalized - 2.0))
        object_probability = side_probabilities[:, self.object_class_index]
        anchor_weights = object_probability.pow(4) * corner_bias
        weight_sum = anchor_weights.sum(dim=(-2, -1)).clamp_min(1e-6)
        side_x = (anchor_weights * xx).sum(dim=(-2, -1)) / weight_sum
        side_y = (anchor_weights * yy).sum(dim=(-2, -1)) / weight_sum
        side_points = torch.stack([side_x, side_y, torch.ones_like(side_x)], dim=-1)
        front_points = side_points @ self.side_to_front_homography.to(side_probabilities).transpose(0, 1)
        denominator = front_points[:, 2]
        denominator = torch.where(
            denominator.abs() < 1e-6,
            torch.where(
                denominator < 0,
                -torch.ones_like(denominator),
                torch.ones_like(denominator),
            )
            * 1e-6,
            denominator,
        )
        front_x = front_points[:, 0] / denominator
        front_y = front_points[:, 1] / denominator
        sigma = max(height, width) * 0.025
        distance_squared = (
            (xx.unsqueeze(0) - front_x[:, None, None]).square()
            + (yy.unsqueeze(0) - front_y[:, None, None]).square()
        )
        visibility = (
            object_probability.sum(dim=(-2, -1)) / max(height * width * 0.0008, 1.0)
        ).clamp(0.0, 1.0)
        return (
            torch.exp(-0.5 * distance_squared / max(sigma * sigma, 1e-6))
            * visibility[:, None, None]
        ).unsqueeze(1)

    def forward(self, front_probabilities: Tensor, side_probabilities: Tensor) -> Tensor:
        object_prior = self.projected_object_prior(side_probabilities)
        side_context = side_probabilities.mean(dim=(-2, -1), keepdim=True).expand_as(side_probabilities)
        x1 = self.inc(torch.cat([front_probabilities, side_context, object_prior], dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        fused = self.up1(x3, x2)
        fused = self.up2(fused, x1)
        return torch.sigmoid(self.output(fused))


class SemanticAdapterDownsample(nn.Module):
    """Cheap spatial downsampling with a depthwise convolution and pointwise channel projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, probabilities: Tensor) -> Tensor:
        return self.net(probabilities)


class SemanticFeatureAdapter(nn.Module):
    """Map five class probabilities to an RGB-backbone-aligned feature residual."""

    def __init__(self, num_classes: int, output_channels: int, base_channels: int = 32):
        super().__init__()
        if base_channels <= 0:
            raise ValueError(f"semantic adapter base channels must be positive, got {base_channels}.")
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(num_classes, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, channels[0]), num_channels=channels[0]),
            nn.SiLU(inplace=True),
        )
        self.downsamples = nn.Sequential(
            SemanticAdapterDownsample(channels[0], channels[1]),
            SemanticAdapterDownsample(channels[1], channels[2]),
            SemanticAdapterDownsample(channels[2], channels[3]),
            SemanticAdapterDownsample(channels[3], channels[3]),
        )
        self.output_proj = nn.Conv2d(channels[3], output_channels, kernel_size=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, probabilities: Tensor, output_size: tuple[int, int] | None = None) -> Tensor:
        residual = self.output_proj(self.downsamples(self.stem(probabilities)))
        if output_size is not None and residual.shape[-2:] != output_size:
            residual = F.interpolate(residual, size=output_size, mode="bilinear", align_corners=False)
        return residual


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


def semantic_image_key(rgb_key: str) -> str:
    return f"{rgb_key}_semantic"


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
        dice_loss_weight: float,
        semantic_loss_weight: float,
        metric_loss_weight: float,
        metric_eps: float,
        semantic_temperature: float,
        semantic_adapter_base_channels: int,
        semantic_fusion_warmup_steps: int,
        semantic_fusion_ramp_steps: int,
        viewfusion_homography: list[list[float]] | None,
        viewfusion_adapter_base_channels: int,
        viewfusion_loss_weight: float,
        viewfusion_teacher_forcing_steps: int,
        viewfusion_teacher_forcing_ramp_steps: int,
        action_to_seg_grad_ratio: float,
        action_to_seg_warmup_steps: int,
        action_to_seg_ramp_steps: int,
        action_to_seg_conflict_projection: bool,
        semantic_prediction_offsets: list[int],
        semantic_dynamics_hidden_dim: int,
        semantic_dynamics_loss_weight: float,
        phase_feature_reliability: list[list[float]] | None,
        phase_history_length: int,
        phase_hidden_dim: int,
        phase_loss_weight: float,
        phase_teacher_forcing_steps: int,
        phase_teacher_forcing_ramp_steps: int,
        stage_feature_dim: int | None,
        stage_conditioning_mode: str,
        stage_predicted_input_warmup_steps: int,
        stage_predicted_input_ramp_steps: int,
        stage_phase_loss_weight: float,
        stage_event_loss_weight: float,
        stage_progress_loss_weight: float,
        stage_transition_loss_weight: float,
        stage_relation_loss_weight: float,
        stage_attention_regularization_weight: float,
        pretrained_backbone_weights: str | None,
        pretrained_segmentation_checkpoint: str | Path | None,
        mask_quality_weighting: str = "hard",
        mask_quality_full_score: float = 0.95,
        mask_quality_weight_gamma: float = 1.0,
        phase_history_stride: int = 1,
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
        self.dice_loss_weight = dice_loss_weight
        self.semantic_loss_weight = semantic_loss_weight
        self.metric_loss_weight = metric_loss_weight
        self.metric_eps = metric_eps
        self.semantic_temperature = semantic_temperature
        self.semantic_adapter_base_channels = semantic_adapter_base_channels
        self.semantic_fusion_warmup_steps = semantic_fusion_warmup_steps
        self.semantic_fusion_ramp_steps = semantic_fusion_ramp_steps
        self.viewfusion_loss_weight = viewfusion_loss_weight
        self.viewfusion_teacher_forcing_steps = viewfusion_teacher_forcing_steps
        self.viewfusion_teacher_forcing_ramp_steps = viewfusion_teacher_forcing_ramp_steps
        self.action_to_seg_grad_ratio = action_to_seg_grad_ratio
        self.action_to_seg_warmup_steps = action_to_seg_warmup_steps
        self.action_to_seg_ramp_steps = action_to_seg_ramp_steps
        self.action_to_seg_conflict_projection = action_to_seg_conflict_projection
        self.semantic_prediction_offsets = tuple(semantic_prediction_offsets)
        self.semantic_dynamics_loss_weight = semantic_dynamics_loss_weight
        self.phase_history_length = phase_history_length
        self.phase_history_stride = phase_history_stride
        self.phase_loss_weight = phase_loss_weight
        self.phase_teacher_forcing_steps = phase_teacher_forcing_steps
        self.phase_teacher_forcing_ramp_steps = phase_teacher_forcing_ramp_steps
        self.stage_feature_dim = stage_feature_dim
        self.stage_conditioning_mode = stage_conditioning_mode
        self.stage_predicted_input_warmup_steps = stage_predicted_input_warmup_steps
        self.stage_predicted_input_ramp_steps = stage_predicted_input_ramp_steps
        self.stage_phase_loss_weight = stage_phase_loss_weight
        self.stage_event_loss_weight = stage_event_loss_weight
        self.stage_progress_loss_weight = stage_progress_loss_weight
        self.stage_transition_loss_weight = stage_transition_loss_weight
        self.stage_relation_loss_weight = stage_relation_loss_weight
        self.stage_attention_regularization_weight = stage_attention_regularization_weight
        self.mask_quality_weighting = mask_quality_weighting
        self.mask_quality_full_score = mask_quality_full_score
        self.mask_quality_weight_gamma = mask_quality_weight_gamma
        self._training_step = 0
        self._asem_backward_losses: tuple[Tensor, Tensor, float] | None = None
        self.bce = nn.BCEWithLogitsLoss()
        self._latest_inference_mask_preview: dict[str, Tensor] = {}
        self._latest_semantic_state: Tensor | None = None
        self._latest_semantic_rollout: dict[str, Tensor] = {}
        self._phase_inference_history: Tensor | None = None
        self._latest_phase_probabilities: Tensor | None = None
        self._latest_stage_outputs: dict[str, Tensor] = {}
        self._inference_control_step: int | None = None
        self._last_phase_history_step: int | None = None
        self._ssact_runtime_controller: Any | None = None
        self._latest_ssact_control_report: dict[str, Any] = {}

        valid_experiments = {
            "1A",
            "1B",
            "2A",
            "2B",
            "2C",
            "3",
            "4A",
            "4B",
            "4C",
            "5",
            *SEMANTIC_EXPERIMENTS,
            *VIEW_FUSION_EXPERIMENTS,
        }
        if self.experiment not in valid_experiments:
            raise ValueError(
                f"Unknown experiment '{experiment}'. Choose one of {sorted(valid_experiments)}."
            )
        if self.semantic_temperature <= 0:
            raise ValueError(f"semantic_temperature must be positive, got {self.semantic_temperature}.")
        if self.semantic_fusion_warmup_steps < 0 or self.semantic_fusion_ramp_steps < 0:
            raise ValueError(
                "semantic fusion warmup and ramp steps must be non-negative, got "
                f"{self.semantic_fusion_warmup_steps} and {self.semantic_fusion_ramp_steps}."
            )
        if self.experiment in VIEW_FUSION_EXPERIMENTS:
            if viewfusion_homography is None:
                raise ValueError("ViewFus-v1 requires a side-to-front homography.")
            if self.viewfusion_loss_weight < 0:
                raise ValueError("View-fusion loss weight must be non-negative.")
            if self.viewfusion_teacher_forcing_steps < 0 or self.viewfusion_teacher_forcing_ramp_steps < 0:
                raise ValueError("View-fusion teacher-forcing step counts must be non-negative.")
        if not 0.0 <= self.action_to_seg_grad_ratio <= 1.0:
            raise ValueError(
                "action_to_seg_grad_ratio must be between 0 and 1, "
                f"got {self.action_to_seg_grad_ratio}."
            )
        if self.action_to_seg_warmup_steps < 0 or self.action_to_seg_ramp_steps < 0:
            raise ValueError(
                "action-to-seg warmup and ramp steps must be non-negative, got "
                f"{self.action_to_seg_warmup_steps} and {self.action_to_seg_ramp_steps}."
            )
        if self.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
            if (
                not self.semantic_prediction_offsets
                or tuple(sorted(set(self.semantic_prediction_offsets))) != self.semantic_prediction_offsets
                or self.semantic_prediction_offsets[0] <= 0
                or self.semantic_prediction_offsets[-1] > self.config.chunk_size
            ):
                raise ValueError(
                    "SSACT-1 semantic prediction offsets must be unique, increasing, positive, and no larger "
                    f"than chunk_size={self.config.chunk_size}; got {self.semantic_prediction_offsets}."
                )
            if self.semantic_dynamics_loss_weight < 0:
                raise ValueError("semantic_dynamics_loss_weight must be non-negative.")
        if self.experiment in PHASE_CONDITIONED_EXPERIMENTS:
            if self.experiment == "SSACT-1" and phase_feature_reliability is None:
                raise ValueError("SSACT-1 requires phase feature reliability from --phase-labels.")
            if self.phase_history_length <= 0 or phase_hidden_dim <= 0:
                raise ValueError("SSACT-1 phase history length and hidden dimension must be positive.")
            if self.phase_history_stride <= 0:
                raise ValueError("SSACT-1 phase history stride must be positive.")
            if self.phase_loss_weight < 0:
                raise ValueError("phase_loss_weight must be non-negative.")
            if self.phase_teacher_forcing_steps < 0 or self.phase_teacher_forcing_ramp_steps < 0:
                raise ValueError("Phase teacher-forcing step counts must be non-negative.")
        if getattr(self, "experiment", None) in STAGE_AWARE_EXPERIMENTS:
            if self.stage_feature_dim is None or self.stage_feature_dim <= 0:
                raise ValueError("SSACT-3 requires a positive stage feature dimension.")
            if self.stage_conditioning_mode not in PhaseConditionedSemanticAdapter.VALID_MODES:
                raise ValueError(f"Invalid stage conditioning mode '{self.stage_conditioning_mode}'.")
            if self.stage_predicted_input_warmup_steps < 0 or self.stage_predicted_input_ramp_steps < 0:
                raise ValueError("Stage predicted-input warmup and ramp must be non-negative.")
            stage_weights = (
                self.stage_phase_loss_weight,
                self.stage_event_loss_weight,
                self.stage_progress_loss_weight,
                self.stage_transition_loss_weight,
                self.stage_relation_loss_weight,
                self.stage_attention_regularization_weight,
            )
            if any(weight < 0 for weight in stage_weights):
                raise ValueError("SSACT-3 loss weights must be non-negative.")
        if self.uses_semantic_maps() and set(self.mask_suffixes) != set(SEMANTIC_CLASSES):
            raise ValueError(
                f"{self.experiment} requires exactly these semantic mask suffixes: {list(SEMANTIC_CLASSES)}; "
                f"got {self.mask_suffixes}."
            )

        semantic_output_channels = (
            len(self.mask_suffixes) + 1 if self.uses_semantic_maps() else len(self.mask_suffixes)
        )
        self.pretrained_segmenter = None
        if self.experiment in FROZEN_SEMANTIC_EXPERIMENTS:
            if pretrained_segmentation_checkpoint is None:
                raise ValueError(f"{self.experiment} requires a pretrained segmentation checkpoint.")
            self.pretrained_segmenter = FrozenTinyUNetSegmenter(pretrained_segmentation_checkpoint)
            if len(self.pretrained_segmenter.labels) != semantic_output_channels:
                raise ValueError(
                    f"{self.experiment} segmenter has {len(self.pretrained_segmenter.labels)} classes, "
                    f"but the semantic layout requires {semantic_output_channels}."
                )
            self.seg_net = None
        else:
            self.seg_net = UNetSegNet(
                out_masks=semantic_output_channels,
                latent_dim=latent_dim,
                base_channels=unet_base_channels,
            )
        self.semantic_adapter = None
        if self.uses_semantic_feature_fusion():
            if not self.config.image_features or len(self.config.image_features) != len(self.rgb_keys):
                raise ValueError(
                    "SEM-1-V2 requires exactly one ACT RGB image input per configured RGB view."
                )
            backbone_channels = self.act_policy.model.encoder_img_feat_input_proj.in_channels
            if self.experiment in STAGE_AWARE_EXPERIMENTS:
                self.semantic_adapter = PhaseConditionedSemanticAdapter(
                    output_channels=backbone_channels,
                    context_dim=phase_hidden_dim,
                    base_channels=self.semantic_adapter_base_channels,
                    mode=self.stage_conditioning_mode,
                )
            else:
                self.semantic_adapter = SemanticFeatureAdapter(
                    num_classes=semantic_output_channels,
                    output_channels=backbone_channels,
                    base_channels=self.semantic_adapter_base_channels,
                )
        self.viewfusion_adapter = (
            HomographyViewFusionAdapter(
                num_classes=semantic_output_channels,
                homography=viewfusion_homography,
                base_channels=viewfusion_adapter_base_channels,
            )
            if self.experiment in VIEW_FUSION_EXPERIMENTS
            else None
        )
        if self.uses_semantic_maps():
            palette = [SEMANTIC_PALETTE_BY_CLASS["background"]]
            palette.extend(SEMANTIC_PALETTE_BY_CLASS[suffix] for suffix in self.mask_suffixes)
            class_weights = [SEMANTIC_CLASS_WEIGHT_BY_NAME["background"]]
            class_weights.extend(SEMANTIC_CLASS_WEIGHT_BY_NAME[suffix] for suffix in self.mask_suffixes)
            self.register_buffer("semantic_palette", torch.tensor(palette, dtype=torch.float32))
            self.register_buffer("semantic_class_weights", torch.tensor(class_weights, dtype=torch.float32))
            self.register_buffer("semantic_visual_mean", torch.tensor(IMAGENET_VISUAL_MEAN).view(1, 3, 1, 1))
            self.register_buffer("semantic_visual_std", torch.tensor(IMAGENET_VISUAL_STD).view(1, 3, 1, 1))
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
        self.semantic_state_extractor = None
        self.semantic_dynamics = None
        self.phase_feature_extractor = None
        self.phase_model = None
        self.stage_model = None
        self.semantic_state_names: tuple[str, ...] = ()
        self.phase_feature_names: tuple[str, ...] = ()
        if self.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
            # Hard future annotations do not contain calibrated segmentation
            # confidence, so confidence is monitored online but not used as a
            # fake all-ones dynamics target.
            self.semantic_state_extractor = SoftSemanticStateExtractor(
                include_confidence=False,
            )
            view_names = [key.rsplit(".", 1)[-1] for key in self.rgb_keys]
            self.semantic_state_names = self.semantic_state_extractor.feature_names(view_names)
            semantic_dim = len(self.semantic_state_names)
            robot_state_dim = self.act_policy.config.robot_state_feature.shape[0]
            action_dim = self.act_policy.config.action_feature.shape[0]
            self.semantic_dynamics = ActionConditionedSemanticDynamics(
                semantic_dim=semantic_dim,
                robot_state_dim=robot_state_dim,
                action_dim=action_dim,
                prediction_offsets=self.semantic_prediction_offsets,
                hidden_dim=semantic_dynamics_hidden_dim,
            )
        if self.experiment == "SSACT-1":
            reliability = torch.tensor(phase_feature_reliability, dtype=torch.float32)
            self.phase_feature_extractor = PhaseSemanticFeatureExtractor(reliability)
            view_names = [key.rsplit(".", 1)[-1] for key in self.rgb_keys]
            self.phase_feature_names = self.phase_feature_extractor.feature_names(view_names)
            self.phase_model = PhaseHistoryModel(
                semantic_dim=self.phase_feature_extractor.output_dim,
                robot_state_dim=0,
                action_dim=0,
                hidden_dim=phase_hidden_dim,
            )
        if self.experiment in STAGE_AWARE_EXPERIMENTS:
            self.semantic_state_extractor = SoftSemanticStateExtractor(include_confidence=True)
            view_names = [key.rsplit(".", 1)[-1] for key in self.rgb_keys]
            self.semantic_state_names = self.semantic_state_extractor.feature_names(view_names)
            if len(self.semantic_state_names) != self.stage_feature_dim:
                raise ValueError(
                    f"Online SSACT-3 features have width {len(self.semantic_state_names)}, but offline "
                    f"supervision has width {self.stage_feature_dim}."
                )
            robot_state_dim = self.act_policy.config.robot_state_feature.shape[0]
            self.stage_model = StageAwareTemporalModel(
                semantic_dim=self.stage_feature_dim,
                robot_state_dim=robot_state_dim,
                hidden_dim=phase_hidden_dim,
            )

    @property
    def config(self):
        return self.act_policy.config

    def reset(self) -> None:
        self.act_policy.reset()
        self._latest_inference_mask_preview = {}
        self._latest_semantic_state = None
        self._latest_semantic_rollout = {}
        self._phase_inference_history = None
        self._latest_phase_probabilities = None
        self._latest_stage_outputs = {}
        self._inference_control_step = None
        self._last_phase_history_step = None
        self._latest_ssact_control_report = {}
        if self._ssact_runtime_controller is not None:
            self._ssact_runtime_controller.reset()

    def configure_ssact_runtime(self, config: dict[str, Any] | None) -> None:
        """Enable runtime semantic control without altering checkpoint weights."""
        if config is None:
            self._ssact_runtime_controller = None
            self._latest_ssact_control_report = {}
            return
        if self.experiment not in PHASE_CONDITIONED_EXPERIMENTS:
            raise ValueError("SSACT runtime control requires a phase-conditioned checkpoint.")
        from semantic_servo import SSACTRuntimeConfig, SSACTRuntimeController

        self._ssact_runtime_controller = SSACTRuntimeController(SSACTRuntimeConfig(**config))
        self._latest_ssact_control_report = {}

    def set_inference_control_step(self, control_step: int) -> None:
        if control_step < 0:
            raise ValueError("control_step must be non-negative.")
        self._inference_control_step = int(control_step)

    def latest_ssact_control_report(self) -> dict[str, Any]:
        return dict(self._latest_ssact_control_report)

    def ssact_runtime_requires_grad(self) -> bool:
        return self._ssact_runtime_controller is not None

    def set_training_step(self, step: int) -> None:
        self._training_step = step

    def scheduled_action_to_seg_grad_ratio(self) -> float:
        if self.experiment not in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS:
            return 0.0
        if self._training_step <= self.action_to_seg_warmup_steps:
            return 0.0
        if self.action_to_seg_ramp_steps == 0:
            return self.action_to_seg_grad_ratio
        ramp_progress = min(
            (self._training_step - self.action_to_seg_warmup_steps) / self.action_to_seg_ramp_steps,
            1.0,
        )
        return self.action_to_seg_grad_ratio * ramp_progress

    def scheduled_semantic_fusion_scale(self) -> float:
        if not self.uses_semantic_feature_fusion():
            return 0.0
        if not self.training:
            return 1.0
        if self._training_step <= self.semantic_fusion_warmup_steps:
            return 0.0
        if self.semantic_fusion_ramp_steps == 0:
            return 1.0
        return min(
            (self._training_step - self.semantic_fusion_warmup_steps)
            / self.semantic_fusion_ramp_steps,
            1.0,
        )

    def semantic_feature_residuals(
        self,
        probabilities_by_view: list[Tensor],
        *,
        scale: float,
        phase_probabilities: Tensor | None = None,
        stage_context: Tensor | None = None,
        stage_confidence: Tensor | None = None,
    ) -> list[Tensor]:
        if self.semantic_adapter is None:
            raise RuntimeError("Semantic feature residuals require the SEM-1-V2 semantic adapter.")
        # SEM-1-V2 keeps segmentation supervised only by pseudo-labels. The action loss trains
        # the adapter and ACT, but cannot distort the segmentation network.
        if getattr(self, "experiment", None) in STAGE_AWARE_EXPERIMENTS:
            if phase_probabilities is None or stage_context is None:
                raise RuntimeError("SSACT-3 semantic fusion requires phase probabilities and stage context.")
            return [
                self.semantic_adapter(
                    probabilities.detach(),
                    phase_probabilities.detach(),
                    stage_context.detach(),
                    confidence=None if stage_confidence is None else stage_confidence.detach(),
                )
                * scale
                for probabilities in probabilities_by_view
            ]
        return [self.semantic_adapter(probabilities.detach()) * scale for probabilities in probabilities_by_view]

    def scheduled_stage_predicted_input_ratio(self) -> float:
        if self.experiment not in STAGE_AWARE_EXPERIMENTS:
            return 1.0
        if not self.training:
            return 1.0
        if self._training_step <= self.stage_predicted_input_warmup_steps:
            return 0.0
        if self.stage_predicted_input_ramp_steps == 0:
            return 1.0
        return min(
            (self._training_step - self.stage_predicted_input_warmup_steps)
            / self.stage_predicted_input_ramp_steps,
            1.0,
        )

    def scheduled_phase_teacher_forcing_ratio(self) -> float:
        if self.experiment not in PHASE_CONDITIONED_EXPERIMENTS:
            return 0.0
        if self._training_step <= self.phase_teacher_forcing_steps:
            return 1.0
        if self.phase_teacher_forcing_ramp_steps == 0:
            return 0.0
        progress = min(
            (self._training_step - self.phase_teacher_forcing_steps)
            / self.phase_teacher_forcing_ramp_steps,
            1.0,
        )
        return 1.0 - progress

    def scheduled_viewfusion_teacher_forcing_ratio(self) -> float:
        if self.experiment not in VIEW_FUSION_EXPERIMENTS:
            return 0.0
        if self._training_step <= self.viewfusion_teacher_forcing_steps:
            return 1.0
        if self.viewfusion_teacher_forcing_ramp_steps == 0:
            return 0.0
        progress = min(
            (self._training_step - self.viewfusion_teacher_forcing_steps)
            / self.viewfusion_teacher_forcing_ramp_steps,
            1.0,
        )
        return 1.0 - progress

    def latest_inference_mask_preview(self) -> dict[str, Tensor]:
        """Return the latest inference masks as CPU uint8 probability maps.

        Multiclass semantic experiments also expose each view's background
        probability so consumers can reconstruct the exact five-class soft map.
        """
        return dict(self._latest_inference_mask_preview)

    def latest_semantic_rollout(self) -> dict[str, Tensor]:
        """Return the latest SSACT-1 rollout as detached CPU tensors."""
        return dict(self._latest_semantic_rollout)

    @staticmethod
    def _current_visual_frame(image: Tensor) -> Tensor:
        if image.ndim == 5:
            return image[:, 0]
        if image.ndim != 4:
            raise ValueError(f"Expected BCHW or BTCHW visual tensor, got {tuple(image.shape)}.")
        return image

    def _resize_inference_rgb(self, rgb: Tensor) -> Tensor:
        image_size = getattr(self, "inference_image_size", None)
        if image_size is None or tuple(rgb.shape[-2:]) == tuple(image_size):
            return rgb
        return F.interpolate(rgb, size=tuple(image_size), mode="bilinear", align_corners=False, antialias=True)

    def _get_rgb_inputs(
        self,
        batch: dict[str, Tensor],
        *,
        device: torch.device | None = None,
        denormalize: bool = False,
    ) -> list[Tensor]:
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
            if denormalize:
                rgb = self.denormalize_visual_like(rgb, key)
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
        rgbs = self._get_rgb_inputs(batch, denormalize=self.act_uses_raw_rgb_images())
        act_batch = dict(batch)

        if self.uses_semantic_latents():
            if self.semantic_net is None:
                raise RuntimeError("Experiment 5 semantic network is missing.")
            rgb_main_latent, rgb_semantic_latents = self.predict_inference_semantic_latents(rgbs)
            act_batch[OBS_ENV_STATE] = torch.cat(
                [rgb_main_latent, rgb_semantic_latents.reshape(rgb_semantic_latents.shape[0], -1)],
                dim=-1,
            )
        elif self.uses_semantic_maps():
            logits_by_view, _ = self.predict_view_logits_and_latent_from_rgbs(rgbs)
            probabilities_by_view = self.semantic_probabilities(logits_by_view)
            if self.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
                self._latest_semantic_state = self.semantic_states_from_probabilities(probabilities_by_view)
            stage_context = None
            stage_confidence = None
            if self.experiment == "SSACT-1":
                current_phase_features = self.phase_features_from_probabilities(probabilities_by_view)
                if (
                    self._phase_inference_history is None
                    or self._phase_inference_history.shape[0] != current_phase_features.shape[0]
                ):
                    self._phase_inference_history = current_phase_features.unsqueeze(1).expand(
                        -1, self.phase_history_length, -1
                    ).clone()
                    self._last_phase_history_step = self._inference_control_step
                elif (
                    self._inference_control_step is None
                    or self._last_phase_history_step is None
                    or self._inference_control_step - self._last_phase_history_step >= self.phase_history_stride
                ):
                    self._phase_inference_history = torch.cat(
                        [self._phase_inference_history[:, 1:], current_phase_features.unsqueeze(1)],
                        dim=1,
                    )
                    self._last_phase_history_step = self._inference_control_step
                _, phase_probabilities = self.predict_phase_from_history(self._phase_inference_history)
                self._latest_phase_probabilities = phase_probabilities
                act_batch[OBS_ENV_STATE] = phase_probabilities
            elif self.experiment in STAGE_AWARE_EXPERIMENTS:
                current_stage_features = self.semantic_states_from_probabilities(probabilities_by_view)
                if (
                    self._phase_inference_history is None
                    or self._phase_inference_history.shape[0] != current_stage_features.shape[0]
                ):
                    self._phase_inference_history = current_stage_features.unsqueeze(1).expand(
                        -1, self.phase_history_length, -1
                    ).clone()
                    self._last_phase_history_step = self._inference_control_step
                elif (
                    self._inference_control_step is None
                    or self._last_phase_history_step is None
                    or self._inference_control_step - self._last_phase_history_step >= self.phase_history_stride
                ):
                    self._phase_inference_history = torch.cat(
                        [self._phase_inference_history[:, 1:], current_stage_features.unsqueeze(1)],
                        dim=1,
                    )
                    self._last_phase_history_step = self._inference_control_step
                if self.stage_model is None:
                    raise RuntimeError("SSACT-3 stage model is missing.")
                stage_outputs = self.stage_model(self._phase_inference_history, act_batch[OBS_STATE])
                phase_probabilities = torch.softmax(stage_outputs["phase_logits"], dim=-1)
                transition_probabilities = torch.softmax(stage_outputs["transition_logits"], dim=-1)
                self._latest_phase_probabilities = phase_probabilities
                self._latest_stage_outputs = {
                    "phase_probabilities": phase_probabilities.detach().cpu(),
                    "event_probabilities": torch.sigmoid(stage_outputs["event_logits"]).detach().cpu(),
                    "progress": stage_outputs["progress"].detach().cpu(),
                    "transition_probabilities": transition_probabilities.detach().cpu(),
                    "relations": stage_outputs["relations"].detach().cpu(),
                }
                act_batch[OBS_ENV_STATE] = phase_probabilities
                stage_context = stage_outputs["context"]
                stage_confidence = phase_probabilities.amax(dim=-1) * (
                    1.0 - transition_probabilities[:, STAGE_TRANSITION_NAMES.index("uncertain")]
                )
            mask_probs = self.semantic_foreground_probs_for_mask_keys(probabilities_by_view)
            preview = mask_probs[0].detach().clamp(0.0, 1.0).mul(255).to(dtype=torch.uint8).cpu()
            self._latest_inference_mask_preview = {
                key: preview[idx] for idx, key in enumerate(self.mask_keys)
            }
            for view_idx, rgb_key in enumerate(self.rgb_keys):
                background = (
                    probabilities_by_view[view_idx][0, 0]
                    .detach()
                    .clamp(0.0, 1.0)
                    .mul(255)
                    .to(dtype=torch.uint8)
                    .cpu()
                )
                self._latest_inference_mask_preview[f"{rgb_key}_background"] = background
            probabilities_for_act = self.semantic_probabilities_for_act(probabilities_by_view)
            if self.experiment in VIEW_FUSION_EXPERIMENTS:
                if self.viewfusion_adapter is None:
                    raise RuntimeError("ViewFus-v1 fusion adapter is missing.")
                if len(probabilities_for_act) != 2:
                    raise RuntimeError("ViewFus-v1 inference requires front and side probabilities.")
                fused_front = self.viewfusion_adapter(
                    probabilities_for_act[0], probabilities_for_act[1]
                )
                act_batch[VIEWFUS_FUSED_FRONT_KEY] = self.normalize_visual_like(
                    fused_front, VIEWFUS_FUSED_FRONT_KEY
                )
            elif self.uses_semantic_feature_fusion():
                act_batch[IMAGE_FEATURE_RESIDUALS] = self.semantic_feature_residuals(
                    probabilities_for_act,
                    scale=1.0,
                    phase_probabilities=self._latest_phase_probabilities,
                    stage_context=stage_context,
                    stage_confidence=stage_confidence,
                )
            else:
                for rgb_key, semantic_rgb in zip(
                    self.rgb_keys,
                    self.semantic_rgb_maps(probabilities_for_act),
                    strict=True,
                ):
                    act_batch[semantic_image_key(rgb_key)] = self.normalize_semantic_map(semantic_rgb)
        else:
            mask_logits, rgb_latent = self.predict_masks_and_latent_from_rgbs(rgbs)
            mask_probs = torch.sigmoid(mask_logits)
            preview = mask_probs[0].detach().clamp(0.0, 1.0).mul(255).to(dtype=torch.uint8).cpu()
            self._latest_inference_mask_preview = {
                key: preview[idx] for idx, key in enumerate(self.mask_keys)
            }

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

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Run RGB-only Mask-ACT inference and return the complete predicted action chunk."""
        with torch.no_grad():
            act_batch = self._prepare_inference_batch(batch)
            actions = self.act_policy.predict_action_chunk(act_batch)
        if self.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
            if self.semantic_dynamics is None or self._latest_semantic_state is None:
                raise RuntimeError("SSACT-1 semantic dynamics state was not prepared for inference.")
            if self._ssact_runtime_controller is not None:
                if self._latest_phase_probabilities is None:
                    raise RuntimeError("SSACT phase probabilities were not prepared for runtime control.")
                actions, report = self._ssact_runtime_controller.apply(
                    semantic_dynamics=self.semantic_dynamics,
                    semantic_state=self._latest_semantic_state,
                    robot_state=act_batch[OBS_STATE],
                    nominal_actions=actions,
                    phase_probabilities=self._latest_phase_probabilities,
                    prediction_offsets=self.semantic_prediction_offsets,
                )
                self._latest_ssact_control_report = report
            with torch.no_grad():
                means, log_stds = self.semantic_dynamics(
                    self._latest_semantic_state,
                    act_batch[OBS_STATE],
                    actions,
                )
            self._latest_semantic_rollout = {
                "mean": means.detach().cpu(),
                "std": log_stds.exp().detach().cpu(),
                "offsets": torch.tensor(self.semantic_prediction_offsets),
            }
        return actions

    def latest_phase_probabilities(self) -> Tensor | None:
        if self._latest_phase_probabilities is None:
            return None
        return self._latest_phase_probabilities.detach().cpu()

    def latest_stage_outputs(self) -> dict[str, Tensor]:
        return dict(self._latest_stage_outputs)

    def mask_action_grad_enabled(self) -> bool:
        return self.experiment in {"1B", "2B", "2C", "4A", "4B", "4C"}

    def latent_action_grad_enabled(self) -> bool:
        return self.experiment == "3"

    def act_uses_masks(self) -> bool:
        return self.experiment in {"1A", "1B", "2A", "2B", "2C", "4A", "4B", "4C"}

    def act_uses_latent(self) -> bool:
        return self.experiment in {"2A", "2B", "3"}

    def uses_mask_metrics(self) -> bool:
        return self.experiment in {"4A", "4B", "4C"}

    def uses_semantic_latents(self) -> bool:
        return self.experiment == "5"

    def uses_semantic_maps(self) -> bool:
        return self.experiment in SEMANTIC_EXPERIMENTS

    def uses_semantic_feature_fusion(self) -> bool:
        return self.experiment in SEMANTIC_FEATURE_FUSION_EXPERIMENTS

    def uses_frozen_semantic_segmenter(self) -> bool:
        return self.experiment in FROZEN_SEMANTIC_EXPERIMENTS

    def act_uses_raw_rgb_images(self) -> bool:
        return self.experiment in {
            "2C",
            "SEM-1",
            "SEM-1-V2",
            "ASEM-1",
            "SSACT-1",
            "SSACT-3",
            "VIEWFUS-V1",
            "UNET-SEM",
        }

    def normalize_visual_like(self, image: Tensor, key: str) -> Tensor:
        key_stats = self.stats[key]
        mean = torch.as_tensor(key_stats["mean"], dtype=image.dtype, device=image.device)
        std = torch.as_tensor(key_stats["std"], dtype=image.dtype, device=image.device).clamp_min(1e-6)
        return (image - mean) / std

    def denormalize_visual_like(self, image: Tensor, key: str) -> Tensor:
        key_stats = self.stats[key]
        mean = torch.as_tensor(key_stats["mean"], dtype=image.dtype, device=image.device)
        std = torch.as_tensor(key_stats["std"], dtype=image.dtype, device=image.device)
        return image * std + mean

    def build_mask_targets(self, raw_batch: dict[str, Tensor], device: torch.device) -> Tensor:
        masks = []
        for key in self.mask_keys:
            mask = self._current_visual_frame(raw_batch[key]).to(device=device, dtype=torch.float32)
            if mask.shape[1] == 3:
                mask = mask.mean(dim=1, keepdim=True)
            else:
                mask = mask[:, :1]
            masks.append(mask.clamp(0.0, 1.0))
        return torch.cat(masks, dim=1)

    def build_semantic_targets(
        self,
        raw_batch: dict[str, Tensor],
        device: torch.device,
    ) -> list[Tensor]:
        targets_by_view = []
        for view_idx in range(len(self.rgb_keys)):
            masks = []
            for suffix_idx in range(len(self.mask_suffixes)):
                key = next(
                    key
                    for key, mapped_position in self.mask_key_map.items()
                    if mapped_position == (view_idx, suffix_idx)
                )
                mask = self._current_visual_frame(raw_batch[key]).to(device=device, dtype=torch.float32)
                mask = mask.mean(dim=1) if mask.shape[1] == 3 else mask[:, 0]
                masks.append(mask >= 0.5)
            stacked = torch.stack(masks, dim=1)
            overlap = stacked.sum(dim=1) > 1
            if overlap.any():
                overlap_count = int(overlap.sum().detach().cpu())
                raise ValueError(
                    f"{self.experiment} requires mutually exclusive semantic labels, but view "
                    f"'{self.rgb_keys[view_idx]}' has {overlap_count} overlapping pixels in this batch."
                )
            foreground_present = stacked.any(dim=1)
            target = stacked.to(dtype=torch.int64).argmax(dim=1) + 1
            targets_by_view.append(torch.where(foreground_present, target, torch.zeros_like(target)))
        return targets_by_view

    def semantic_states_from_probabilities(self, probabilities_by_view: list[Tensor]) -> Tensor:
        if self.semantic_state_extractor is None:
            raise RuntimeError("Structured semantic states are not enabled for this experiment.")
        state = self.semantic_state_extractor(self.semantic_control_probabilities(probabilities_by_view))
        return self.apply_semantic_state_reliability(state)

    def apply_semantic_state_reliability(self, state: Tensor) -> Tensor:
        """Apply the same fixed view/class gates to predicted and cached states."""
        if self.phase_feature_extractor is None:
            return state
        reliability = self.phase_feature_extractor.reliability
        state_gates = []
        for view_gates in reliability:
            cloth, object_gate, goal, actuator = view_gates
            state_gates.extend(
                [
                    cloth,
                    object_gate,
                    goal,
                    actuator,
                    object_gate,
                    object_gate,
                    goal,
                    goal,
                    actuator,
                    actuator,
                    torch.minimum(object_gate, goal),
                    torch.minimum(object_gate, cloth),
                    torch.minimum(object_gate, goal),
                    torch.minimum(actuator, object_gate),
                ]
            )
        gate_tensor = torch.stack(state_gates).to(device=state.device, dtype=state.dtype)
        return state * gate_tensor

    def semantic_control_probabilities(self, probabilities_by_view: list[Tensor]) -> Tensor:
        control_order = ("occluder", "object", "region", "tool")
        class_indices = [self.mask_suffixes.index(name) + 1 for name in control_order]
        return torch.stack(
            [probabilities[:, class_indices] for probabilities in probabilities_by_view],
            dim=1,
        )

    def phase_features_from_probabilities(self, probabilities_by_view: list[Tensor]) -> Tensor:
        if self.phase_feature_extractor is None:
            raise RuntimeError("Phase semantic features are only enabled for SSACT-1.")
        return self.phase_feature_extractor(self.semantic_control_probabilities(probabilities_by_view))

    def predict_phase_from_history(self, semantic_history: Tensor) -> tuple[Tensor, Tensor]:
        if self.phase_model is None:
            raise RuntimeError("Phase model is only enabled for SSACT-1.")
        empty = semantic_history.new_zeros((*semantic_history.shape[:2], 0))
        logits = self.phase_model(semantic_history, empty, empty)
        return logits, torch.softmax(logits, dim=-1)

    def build_future_semantic_states(
        self,
        raw_batch: dict[str, Tensor],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Load cached future hard-label semantic targets for SSACT-1."""
        if self.semantic_state_extractor is None:
            raise RuntimeError("Future semantic states are only available for SSACT-1.")
        required = ("future_semantic_states", "future_semantic_valid")
        missing = [key for key in required if key not in raw_batch]
        if missing:
            raise KeyError(f"SSACT-1 batch is missing offline semantic targets: {missing}.")
        targets = raw_batch["future_semantic_states"].to(device=device, dtype=torch.float32)
        valid = raw_batch["future_semantic_valid"].to(device=device, dtype=torch.bool)
        expected_shape = (
            targets.shape[0],
            len(self.semantic_prediction_offsets),
            len(self.semantic_state_names),
        )
        if targets.shape != expected_shape:
            raise ValueError(
                f"Offline semantic targets have shape {tuple(targets.shape)}; expected {expected_shape}."
            )
        if valid.shape != expected_shape[:2]:
            raise ValueError(
                f"Offline semantic validity has shape {tuple(valid.shape)}; expected {expected_shape[:2]}."
            )
        return self.apply_semantic_state_reliability(targets), valid

    def semantic_probabilities(self, logits_by_view: list[Tensor]) -> list[Tensor]:
        return [
            torch.softmax(logits / self.semantic_temperature, dim=1)
            for logits in logits_by_view
        ]

    def semantic_rgb_maps(self, probabilities_by_view: list[Tensor]) -> list[Tensor]:
        return [
            torch.einsum("bkhw,kc->bchw", probabilities, self.semantic_palette)
            for probabilities in probabilities_by_view
        ]

    def semantic_probabilities_for_act(self, probabilities_by_view: list[Tensor]) -> list[Tensor]:
        """Apply static view/class reliability without changing segmentation supervision."""
        if self.phase_feature_extractor is None:
            return probabilities_by_view
        reliability = self.phase_feature_extractor.reliability
        gated_probabilities = []
        for view_idx, probabilities in enumerate(probabilities_by_view):
            gates = reliability[view_idx].to(device=probabilities.device, dtype=probabilities.dtype)
            foreground = probabilities[:, 1:] * gates.view(1, -1, 1, 1)
            removed = (probabilities[:, 1:] - foreground).sum(dim=1, keepdim=True)
            gated_probabilities.append(torch.cat([probabilities[:, :1] + removed, foreground], dim=1))
        return gated_probabilities

    def normalize_semantic_map(self, semantic_rgb: Tensor) -> Tensor:
        mean = self.semantic_visual_mean.to(dtype=semantic_rgb.dtype)
        std = self.semantic_visual_std.to(dtype=semantic_rgb.dtype)
        return (semantic_rgb - mean) / std

    def semantic_foreground_probs_for_mask_keys(self, probabilities_by_view: list[Tensor]) -> Tensor:
        return torch.cat(
            [
                probabilities_by_view[view_idx][:, suffix_idx + 1 : suffix_idx + 2]
                for view_idx, suffix_idx in (self.mask_key_map[key] for key in self.mask_keys)
            ],
            dim=1,
        )

    def semantic_segmentation_loss(
        self,
        logits_by_view: list[Tensor],
        targets_by_view: list[Tensor],
        class_quality_by_view: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
        ce_losses = []
        dice_by_class = []
        dice_valid_by_class = []
        if class_quality_by_view is None:
            class_quality_by_view = [None] * len(logits_by_view)
        for logits, target, foreground_quality in zip(
            logits_by_view,
            targets_by_view,
            class_quality_by_view,
            strict=True,
        ):
            if foreground_quality is None:
                ce_losses.append(F.cross_entropy(logits, target, weight=self.semantic_class_weights))
                sample_class_quality = torch.ones(
                    logits.shape[:2],
                    dtype=logits.dtype,
                    device=logits.device,
                )
            else:
                foreground_quality = foreground_quality.to(device=logits.device, dtype=logits.dtype)
                if foreground_quality.shape != (logits.shape[0], logits.shape[1] - 1):
                    raise ValueError(
                        "Mask quality class weights must have shape "
                        f"{(logits.shape[0], logits.shape[1] - 1)}, got {tuple(foreground_quality.shape)}."
                    )
                foreground_quality = foreground_quality.clamp(0.0, 1.0)
                background_quality = foreground_quality.amin(dim=1, keepdim=True)
                sample_class_quality = torch.cat([background_quality, foreground_quality], dim=1)
                pixel_quality = sample_class_quality.gather(1, target.flatten(1)).reshape_as(target)
                pixel_class_weight = self.semantic_class_weights[target]
                ce_per_pixel = F.cross_entropy(
                    logits,
                    target,
                    weight=self.semantic_class_weights,
                    reduction="none",
                )
                ce_losses.append(
                    (ce_per_pixel * pixel_quality).sum()
                    / (pixel_class_weight * pixel_quality).sum().clamp_min(1e-6)
                )
            probabilities = torch.softmax(logits, dim=1)
            one_hot = (
                F.one_hot(target, num_classes=logits.shape[1])
                .permute(0, 3, 1, 2)
                .to(logits.dtype)
            )
            dims = (0, 2, 3)
            dice_weights = sample_class_quality.unsqueeze(-1).unsqueeze(-1)
            intersection = (probabilities * one_hot * dice_weights).sum(dim=dims)
            denominator = ((probabilities + one_hot) * dice_weights).sum(dim=dims)
            dice_by_class.append((2 * intersection + 1e-6) / (denominator + 1e-6))
            dice_valid_by_class.append((sample_class_quality > 0).any(dim=0))

        ce_loss = torch.stack(ce_losses).mean()
        dice_stack = torch.stack(dice_by_class)
        dice_valid_stack = torch.stack(dice_valid_by_class)
        valid_counts = dice_valid_stack.sum(dim=0)
        mean_dice = (dice_stack * dice_valid_stack).sum(dim=0) / valid_counts.clamp_min(1)
        valid_foreground = valid_counts[1:] > 0
        if valid_foreground.any():
            dice_loss = 1 - (
                (mean_dice[1:] * valid_foreground).sum() / valid_foreground.sum()
            )
        else:
            dice_loss = dice_stack.sum() * 0.0
        seg_loss = ce_loss + self.dice_loss_weight * dice_loss
        dice_logs = {
            f"dice_{suffix}": float(mean_dice[idx + 1].detach().cpu())
            for idx, suffix in enumerate(self.mask_suffixes)
        }
        return seg_loss, ce_loss, dice_loss, dice_logs

    def predict_masks_and_latent(self, raw_batch: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
        return self.predict_masks_and_latent_from_rgbs(self._get_rgb_inputs(raw_batch, device=device))

    def predict_view_logits_and_latent_from_rgbs(self, rgbs: list[Tensor]) -> tuple[list[Tensor], Tensor]:
        if self.uses_frozen_semantic_segmenter():
            if self.pretrained_segmenter is None:
                raise RuntimeError(f"{self.experiment} pretrained segmenter is missing.")
            probabilities_by_view = [self.pretrained_segmenter(rgb) for rgb in rgbs]
            logits_by_view = [probabilities.clamp_min(1e-8).log() for probabilities in probabilities_by_view]
            empty_latent = rgbs[0].new_zeros((rgbs[0].shape[0], 0))
            return logits_by_view, empty_latent
        if self.seg_net is None:
            raise RuntimeError(f"{self.experiment} trainable segmentation network is missing.")
        logits_by_view = []
        latents = []
        for rgb in rgbs:
            logits, latent = self.seg_net(rgb)
            logits_by_view.append(logits)
            latents.append(latent)
        return logits_by_view, torch.cat(latents, dim=-1)

    def predict_masks_and_latent_from_rgbs(self, rgbs: list[Tensor]) -> tuple[Tensor, Tensor]:
        logits_by_view, latent = self.predict_view_logits_and_latent_from_rgbs(rgbs)
        if self.uses_semantic_maps():
            probabilities_by_view = self.semantic_probabilities(logits_by_view)
            foreground_probs_by_view = [probabilities[:, 1:] for probabilities in probabilities_by_view]
            logits_by_view = [
                torch.logit(probabilities.clamp(1e-6, 1 - 1e-6)) for probabilities in foreground_probs_by_view
            ]
        return self._stack_logits_for_mask_keys(logits_by_view), latent

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
        mask_targets = None
        if not self.uses_frozen_semantic_segmenter():
            mask_targets = self.build_mask_targets(raw_batch, device=device)

        if self.uses_semantic_latents():
            if mask_targets is None:
                raise RuntimeError("Semantic latent training requires mask targets.")
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

        if self.uses_semantic_maps():
            rgbs = self._get_rgb_inputs(raw_batch, device=device)
            logits_by_view, _ = self.predict_view_logits_and_latent_from_rgbs(rgbs)
            semantic_targets = None
            if not self.uses_frozen_semantic_segmenter():
                semantic_targets = self.build_semantic_targets(raw_batch, device=device)
            quality_validity = raw_batch.get("mask_quality_current_valid")
            quality_scores = raw_batch.get("mask_quality_current_score")
            class_quality_by_view = None
            fusion_sample_quality = None
            quality_logs = {}
            if self.mask_quality_weighting == "soft" and quality_scores is not None:
                quality_scores = quality_scores.to(device=device, dtype=torch.float32)
                quality_weights = mask_quality_scores_to_weights(
                    quality_scores,
                    full_score=self.mask_quality_full_score,
                    gamma=self.mask_quality_weight_gamma,
                )
                class_quality_by_view = [
                    quality_weights[:, view_idx] for view_idx in range(len(self.rgb_keys))
                ]
                fusion_sample_quality = quality_weights.mean(dim=(1, 2))
                quality_logs = {
                    "mask_quality_current_score": float(quality_scores.mean().cpu()),
                    "mask_quality_current_weight": float(quality_weights.mean().cpu()),
                    "mask_quality_current_full_weight_ratio": float(
                        (quality_weights >= 1.0).float().mean().cpu()
                    ),
                }
            elif quality_validity is not None:
                quality_validity = quality_validity.to(device=device, dtype=torch.bool)
                class_quality_by_view = [
                    quality_validity[:, view_idx].to(dtype=torch.float32)
                    for view_idx in range(len(self.rgb_keys))
                ]
                fusion_sample_quality = quality_validity.to(dtype=torch.float32).mean(dim=(1, 2))
                quality_logs = {
                    "mask_quality_current_valid_ratio": float(quality_validity.float().mean().cpu()),
                    "mask_quality_current_score": float(raw_batch["mask_quality_current_score"].float().mean()),
                }
            if self.uses_frozen_semantic_segmenter():
                seg_loss = torch.zeros((), device=device, dtype=torch.float32)
                ce_loss = seg_loss
                dice_loss = seg_loss
                dice_logs = {}
            else:
                if semantic_targets is None:
                    raise RuntimeError("Trainable semantic experiments require semantic targets.")
                seg_loss, ce_loss, dice_loss, dice_logs = self.semantic_segmentation_loss(
                    logits_by_view,
                    semantic_targets,
                    class_quality_by_view,
                )
            probabilities_by_view = self.semantic_probabilities(logits_by_view)
            action_to_seg_ratio = self.scheduled_action_to_seg_grad_ratio()
            act_batch = dict(batch)
            stage_total_loss = None
            stage_logs = {}
            stage_phase_condition = None
            stage_context = None
            stage_confidence = None
            if self.experiment in STAGE_AWARE_EXPERIMENTS:
                required_stage_keys = (
                    "stage_semantic_history",
                    "stage_phase_target",
                    "stage_phase_weight",
                    "stage_event_target",
                    "stage_event_weight",
                    "stage_progress_target",
                    "stage_progress_weight",
                    "stage_transition_target",
                    "stage_transition_weight",
                    "stage_relation_target",
                    "stage_relation_weight",
                )
                missing_stage_keys = [key for key in required_stage_keys if key not in raw_batch]
                if missing_stage_keys:
                    raise KeyError(f"SSACT-3 batch is missing stage data: {missing_stage_keys}.")
                stage_history = raw_batch["stage_semantic_history"].to(
                    device=device, dtype=torch.float32
                )
                expected_history_shape = (
                    stage_history.shape[0],
                    self.phase_history_length,
                    self.stage_feature_dim,
                )
                if stage_history.shape != expected_history_shape:
                    raise ValueError(
                        f"SSACT-3 stage history has shape {tuple(stage_history.shape)}; "
                        f"expected {expected_history_shape}."
                    )
                predicted_features = self.semantic_states_from_probabilities(
                    probabilities_by_view
                ).detach()
                predicted_input_ratio = self.scheduled_stage_predicted_input_ratio()
                stage_history = stage_history.clone()
                stage_history[:, -1] = (
                    (1.0 - predicted_input_ratio) * stage_history[:, -1]
                    + predicted_input_ratio * predicted_features
                )
                if self.stage_model is None:
                    raise RuntimeError("SSACT-3 stage model is missing.")
                stage_outputs = self.stage_model(stage_history, batch[OBS_STATE])
                stage_loss_terms, stage_metric_logs = stage_losses(stage_outputs, raw_batch)
                stage_total_loss = (
                    self.stage_phase_loss_weight * stage_loss_terms["phase"]
                    + self.stage_event_loss_weight * stage_loss_terms["event"]
                    + self.stage_progress_loss_weight * stage_loss_terms["progress"]
                    + self.stage_transition_loss_weight * stage_loss_terms["transition"]
                    + self.stage_relation_loss_weight * stage_loss_terms["relation"]
                )
                attention_regularization = self.semantic_adapter.attention_regularization()
                stage_total_loss = (
                    stage_total_loss
                    + self.stage_attention_regularization_weight * attention_regularization
                )
                predicted_phase = torch.softmax(stage_outputs["phase_logits"], dim=-1)
                predicted_transition = torch.softmax(stage_outputs["transition_logits"], dim=-1)
                stage_phase_target = raw_batch["stage_phase_target"].to(
                    device=device, dtype=torch.float32
                )
                teacher_ratio = self.scheduled_phase_teacher_forcing_ratio()
                stage_phase_condition = (
                    teacher_ratio * stage_phase_target
                    + (1.0 - teacher_ratio) * predicted_phase.detach()
                )
                act_batch[OBS_ENV_STATE] = stage_phase_condition.detach()
                stage_context = stage_outputs["context"].detach()
                stage_confidence = stage_phase_condition.amax(dim=-1) * (
                    1.0
                    - predicted_transition.detach()[:, STAGE_TRANSITION_NAMES.index("uncertain")]
                )
                stage_logs = {
                    "stage_total_loss": float(stage_total_loss.detach().cpu()),
                    "stage_phase_loss": float(stage_loss_terms["phase"].detach().cpu()),
                    "stage_event_loss": float(stage_loss_terms["event"].detach().cpu()),
                    "stage_progress_loss": float(stage_loss_terms["progress"].detach().cpu()),
                    "stage_transition_loss": float(stage_loss_terms["transition"].detach().cpu()),
                    "stage_relation_loss": float(stage_loss_terms["relation"].detach().cpu()),
                    "stage_attention_regularization": float(attention_regularization.detach().cpu()),
                    "stage_predicted_input_ratio": predicted_input_ratio,
                    "phase_teacher_forcing_ratio": teacher_ratio,
                    "stage_condition_confidence": float(stage_confidence.mean().detach().cpu()),
                    **stage_metric_logs,
                }
            semantic_fusion_logs = {}
            probabilities_for_act = self.semantic_probabilities_for_act(probabilities_by_view)
            viewfusion_loss = None
            if self.experiment in VIEW_FUSION_EXPERIMENTS:
                if self.viewfusion_adapter is None or len(probabilities_for_act) != 2:
                    raise RuntimeError("ViewFus-v1 requires a two-view fusion adapter.")
                teacher_ratio = self.scheduled_viewfusion_teacher_forcing_ratio()
                if semantic_targets is None:
                    raise RuntimeError("ViewFus-v1 requires semantic targets for teacher forcing.")
                teacher_probabilities = [
                    F.one_hot(target, num_classes=len(self.mask_suffixes) + 1)
                    .permute(0, 3, 1, 2)
                    .to(dtype=torch.float32)
                    for target in semantic_targets
                ]
                fusion_inputs = [
                    teacher_ratio * teacher + (1.0 - teacher_ratio) * predicted.detach()
                    for teacher, predicted in zip(
                        teacher_probabilities, probabilities_for_act, strict=True
                    )
                ]
                predicted_fused_front = self.viewfusion_adapter(*fusion_inputs)
                fused_front_target = self._current_visual_frame(
                    raw_batch[VIEWFUS_FUSED_FRONT_KEY]
                ).to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
                fusion_error = F.smooth_l1_loss(
                    predicted_fused_front, fused_front_target, reduction="none"
                ).mean(dim=1)
                foreground_weight = 1.0 + 4.0 * fused_front_target.amax(dim=1).gt(0.02)
                fusion_error = (fusion_error * foreground_weight).mean(dim=(-2, -1))
                if fusion_sample_quality is None:
                    viewfusion_loss = fusion_error.mean()
                else:
                    viewfusion_loss = (
                        fusion_error * fusion_sample_quality
                    ).sum() / fusion_sample_quality.sum().clamp_min(1e-6)

                normalized_prediction = self.normalize_visual_like(
                    predicted_fused_front.detach(), VIEWFUS_FUSED_FRONT_KEY
                )
                act_batch[VIEWFUS_FUSED_FRONT_KEY] = (
                    teacher_ratio * batch[VIEWFUS_FUSED_FRONT_KEY]
                    + (1.0 - teacher_ratio) * normalized_prediction
                )
                semantic_fusion_logs = {
                    "viewfusion_loss": float(viewfusion_loss.detach().cpu()),
                    "viewfusion_teacher_forcing_ratio": teacher_ratio,
                    "viewfusion_predicted_input_ratio": 1.0 - teacher_ratio,
                }
            elif self.uses_semantic_feature_fusion():
                semantic_fusion_scale = self.scheduled_semantic_fusion_scale()
                semantic_residuals = self.semantic_feature_residuals(
                    probabilities_for_act,
                    scale=semantic_fusion_scale,
                    phase_probabilities=stage_phase_condition,
                    stage_context=stage_context,
                    stage_confidence=stage_confidence,
                )
                act_batch[IMAGE_FEATURE_RESIDUALS] = semantic_residuals
                residual_rms = torch.stack(
                    [residual.detach().square().mean().sqrt() for residual in semantic_residuals]
                ).mean()
                semantic_fusion_logs = {
                    "semantic_fusion_scale": semantic_fusion_scale,
                    "semantic_residual_rms": float(residual_rms.cpu()),
                }
            else:
                semantic_maps = self.semantic_rgb_maps(probabilities_for_act)
                if (
                    self.experiment not in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS
                    or action_to_seg_ratio <= 0.0
                ):
                    semantic_maps = [semantic_map.detach() for semantic_map in semantic_maps]
                for rgb_key, semantic_map in zip(self.rgb_keys, semantic_maps, strict=True):
                    act_batch[semantic_image_key(rgb_key)] = self.normalize_semantic_map(semantic_map)

            phase_loss = None
            phase_logs = {}
            if self.experiment == "SSACT-1":
                required_phase_keys = ("phase_semantic_history", "phase_target", "phase_confidence")
                missing_phase_keys = [key for key in required_phase_keys if key not in raw_batch]
                if missing_phase_keys:
                    raise KeyError(f"SSACT-1 batch is missing phase data: {missing_phase_keys}.")
                phase_history = raw_batch["phase_semantic_history"].to(device=device, dtype=torch.float32)
                if phase_history.shape[1:] != (self.phase_history_length, len(self.phase_feature_names)):
                    raise ValueError(
                        "SSACT-1 phase history has shape "
                        f"{tuple(phase_history.shape)}; expected "
                        f"(B, {self.phase_history_length}, {len(self.phase_feature_names)})."
                    )
                phase_history = phase_history.clone()
                phase_history[:, -1] = self.phase_features_from_probabilities(probabilities_by_view).detach()
                phase_logits, predicted_phase_probabilities = self.predict_phase_from_history(phase_history)
                phase_target = raw_batch["phase_target"].to(device=device, dtype=torch.float32)
                phase_confidence = raw_batch["phase_confidence"].to(device=device, dtype=torch.float32)
                per_sample_phase_loss = -(
                    phase_target * torch.log_softmax(phase_logits, dim=-1)
                ).sum(dim=-1)
                phase_loss = (
                    per_sample_phase_loss * phase_confidence
                ).sum() / phase_confidence.sum().clamp_min(1e-6)
                teacher_ratio = self.scheduled_phase_teacher_forcing_ratio()
                phase_condition = (
                    teacher_ratio * phase_target
                    + (1.0 - teacher_ratio) * predicted_phase_probabilities.detach()
                )
                act_batch[OBS_ENV_STATE] = phase_condition
                phase_accuracy = (
                    predicted_phase_probabilities.detach().argmax(dim=-1)
                    == phase_target.argmax(dim=-1)
                ).to(dtype=torch.float32)
                valid_phase = phase_confidence > 0
                phase_logs = {
                    "phase_loss": float(phase_loss.detach().cpu()),
                    "phase_accuracy": float(
                        (phase_accuracy * valid_phase).sum().cpu() / valid_phase.sum().clamp_min(1).cpu()
                    ),
                    "phase_valid_ratio": float(valid_phase.float().mean().cpu()),
                    "phase_confidence": float(phase_confidence.mean().cpu()),
                    "phase_teacher_forcing_ratio": teacher_ratio,
                }

            action_loss, action_logs = self.act_policy(act_batch)
            loss = self.action_loss_weight * action_loss + self.seg_loss_weight * seg_loss
            if viewfusion_loss is not None:
                loss = loss + self.viewfusion_loss_weight * viewfusion_loss
            if phase_loss is not None:
                loss = loss + self.phase_loss_weight * phase_loss
            if stage_total_loss is not None:
                loss = loss + stage_total_loss
            dynamics_logs = {}
            if self.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
                if self.semantic_dynamics is None:
                    raise RuntimeError("SSACT-1 semantic dynamics module is missing.")
                current_semantic_state = self.semantic_states_from_probabilities(probabilities_by_view).detach()
                future_semantic_states, valid_future_steps = self.build_future_semantic_states(
                    raw_batch,
                    device=device,
                )
                quality_future_valid = raw_batch.get("mask_quality_future_valid")
                if quality_future_valid is not None:
                    quality_future_valid = quality_future_valid.to(device=device, dtype=torch.bool)
                    valid_future_steps = valid_future_steps & quality_future_valid
                    quality_logs["mask_quality_future_valid_ratio"] = float(
                        quality_future_valid.float().mean().cpu()
                    )
                    quality_logs["mask_quality_future_score"] = float(
                        raw_batch["mask_quality_future_score"].float().mean()
                    )
                predicted_mean, predicted_log_std = self.semantic_dynamics(
                    current_semantic_state,
                    batch[OBS_STATE],
                    batch["action"],
                )
                dynamics_loss = self.semantic_dynamics.gaussian_nll(
                    predicted_mean,
                    predicted_log_std,
                    future_semantic_states,
                    valid_future_steps,
                )
                loss = loss + self.semantic_dynamics_loss_weight * dynamics_loss
                valid_weights = valid_future_steps.to(dtype=predicted_mean.dtype).unsqueeze(-1)
                dynamics_mae = (
                    (predicted_mean.detach() - future_semantic_states).abs() * valid_weights
                ).sum() / (valid_weights.sum() * predicted_mean.shape[-1]).clamp_min(1.0)
                dynamics_logs = {
                    "semantic_dynamics_loss": float(dynamics_loss.detach().cpu()),
                    "semantic_dynamics_mae": float(dynamics_mae.cpu()),
                    "semantic_dynamics_std": float(predicted_log_std.detach().exp().mean().cpu()),
                    "semantic_dynamics_valid_ratio": float(valid_future_steps.float().mean().cpu()),
                }
            if self.experiment in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS:
                self._asem_backward_losses = (action_loss, seg_loss, action_to_seg_ratio)
            logs = {
                "loss": float(loss.detach().cpu()),
                "action_loss": float(action_loss.detach().cpu()),
                "seg_loss": float(seg_loss.detach().cpu()),
                "seg_ce_loss": float(ce_loss.detach().cpu()),
                "seg_dice_loss": float(dice_loss.detach().cpu()),
                **dice_logs,
                **dynamics_logs,
                **phase_logs,
                **stage_logs,
                **quality_logs,
                **semantic_fusion_logs,
            }
            if self.experiment in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS:
                logs["action_to_seg_target_grad_ratio"] = action_to_seg_ratio
            logs.update({key: float(value) for key, value in action_logs.items() if isinstance(value, (int, float))})
            return loss, logs

        if mask_targets is None:
            raise RuntimeError(f"{self.experiment} requires mask targets.")
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

    def backward_training_loss(self, loss: Tensor, logs: dict[str, float]) -> None:
        """Backpropagate, with guarded multi-task gradients for ASEM-1."""

        if self.experiment not in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS:
            loss.backward()
            return
        if self._asem_backward_losses is None:
            raise RuntimeError("ASEM-1 backward state is missing. Call forward before backward_training_loss.")

        action_loss, seg_loss, target_ratio = self._asem_backward_losses
        self._asem_backward_losses = None
        seg_parameters = [parameter for parameter in self.seg_net.parameters() if parameter.requires_grad]

        weighted_action_loss = self.action_loss_weight * action_loss
        weighted_action_loss.backward(retain_graph=target_ratio > 0.0)
        action_grads = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in seg_parameters
        ]
        for parameter in seg_parameters:
            parameter.grad = None

        weighted_seg_loss = self.seg_loss_weight * seg_loss
        weighted_seg_loss.backward()
        seg_grads = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in seg_parameters
        ]

        device = action_loss.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        action_norm_sq = zero.clone()
        seg_norm_sq = zero.clone()
        dot = zero.clone()
        for action_grad, seg_grad in zip(action_grads, seg_grads, strict=True):
            if action_grad is not None:
                action_norm_sq = action_norm_sq + action_grad.float().square().sum()
            if seg_grad is not None:
                seg_norm_sq = seg_norm_sq + seg_grad.float().square().sum()
            if action_grad is not None and seg_grad is not None:
                dot = dot + (action_grad.float() * seg_grad.float()).sum()

        eps = 1e-12
        action_norm = action_norm_sq.sqrt()
        seg_norm = seg_norm_sq.sqrt()
        denominator = (action_norm * seg_norm).clamp_min(eps)
        cosine = torch.where(denominator > eps, dot / denominator, zero)
        conflict = bool((dot < 0).item())

        projection_coefficient = zero
        apply_projection = (
            self.action_to_seg_conflict_projection and conflict and float(seg_norm_sq.item()) > eps
        )
        if apply_projection:
            projection_coefficient = dot / seg_norm_sq.clamp_min(eps)

        projected_action_grads: list[Tensor | None] = []
        projected_norm_sq = zero.clone()
        projected_dot = zero.clone()
        for action_grad, seg_grad in zip(action_grads, seg_grads, strict=True):
            if action_grad is None:
                projected_action_grads.append(None)
                continue
            projected = action_grad
            if seg_grad is not None and apply_projection:
                projected = action_grad - projection_coefficient.to(action_grad.dtype) * seg_grad
            projected_action_grads.append(projected)
            projected_norm_sq = projected_norm_sq + projected.float().square().sum()
            if seg_grad is not None:
                projected_dot = projected_dot + (projected.float() * seg_grad.float()).sum()

        projected_norm = projected_norm_sq.sqrt()
        projected_denominator = (projected_norm * seg_norm).clamp_min(eps)
        projected_cosine = torch.where(
            projected_denominator > eps,
            projected_dot / projected_denominator,
            zero,
        )
        applied_scale = zero
        if target_ratio > 0.0 and float(projected_norm.item()) > eps and float(seg_norm.item()) > eps:
            max_scale = target_ratio * seg_norm / projected_norm.clamp_min(eps)
            applied_scale = torch.minimum(torch.ones_like(max_scale), max_scale)
        applied_scale_value = float(applied_scale.item())

        for parameter, seg_grad, action_grad in zip(
            seg_parameters,
            seg_grads,
            projected_action_grads,
            strict=True,
        ):
            if seg_grad is None:
                parameter.grad = None
                continue
            parameter.grad = seg_grad
            if action_grad is not None and applied_scale_value > 0.0:
                parameter.grad.add_(action_grad, alpha=applied_scale_value)

        applied_action_norm = projected_norm * applied_scale
        logs.update(
            {
                "seg_supervised_grad_norm": float(seg_norm.cpu()),
                "action_to_seg_raw_grad_norm": float(action_norm.cpu()),
                "action_to_seg_raw_grad_ratio": float((action_norm / seg_norm.clamp_min(eps)).cpu()),
                "action_to_seg_grad_cosine": float(cosine.cpu()),
                "action_to_seg_projected_grad_cosine": float(projected_cosine.cpu()),
                "action_to_seg_grad_conflict": float(conflict),
                "action_to_seg_applied_scale": float(applied_scale.cpu()),
                "action_to_seg_applied_grad_norm": float(applied_action_norm.cpu()),
                "action_to_seg_applied_grad_ratio": float(
                    (applied_action_norm / seg_norm.clamp_min(eps)).cpu()
                ),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=str.upper,
        choices=[
            "1A",
            "1B",
            "2A",
            "2B",
            "2C",
            "3",
            "4A",
            "4B",
            "4C",
            "5",
            "SEM-1",
            "SEM-1-V2",
            "ASEM-1",
            "SSACT-1",
            "SSACT-3",
            "SEM-2",
            "VIEWFUS-V1",
            "UNET-SEM",
        ],
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
    parser.add_argument("--seed", type=int, default=1000)
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
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--semantic-temperature", type=float, default=1.0)
    parser.add_argument(
        "--semantic-adapter-base-channels",
        type=int,
        default=32,
        help="SEM-1-V2 base width for the lightweight five-channel semantic adapter.",
    )
    parser.add_argument(
        "--semantic-fusion-warmup-steps",
        type=int,
        default=20_000,
        help="SEM-1-V2 steps that train segmentation and RGB ACT before semantic residual fusion begins.",
    )
    parser.add_argument(
        "--semantic-fusion-ramp-steps",
        type=int,
        default=10_000,
        help="SEM-1-V2 steps used to linearly ramp semantic residual fusion from zero to full strength.",
    )
    parser.add_argument(
        "--viewfusion-homography-json",
        type=Path,
        default=Path("outputs/side_to_front_homography/homography_side_to_front.json"),
        help="Quality-filtered side-right-bottom to front-right-bottom homography for ViewFus-v1.",
    )
    parser.add_argument("--viewfusion-adapter-base-channels", type=int, default=16)
    parser.add_argument("--viewfusion-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--viewfusion-teacher-forcing-steps",
        type=int,
        default=20_000,
        help="Steps for which ACT and the fusion adapter use only dataset semantic labels.",
    )
    parser.add_argument(
        "--viewfusion-teacher-forcing-ramp-steps",
        type=int,
        default=20_000,
        help="Steps used to switch ACT/fusion inputs from labels to model predictions.",
    )
    parser.add_argument(
        "--action-to-seg-grad-ratio",
        type=float,
        default=0.1,
        help="Maximum ASEM-1 action-gradient norm as a fraction of the supervised segmentation-gradient norm.",
    )
    parser.add_argument(
        "--action-to-seg-warmup-steps",
        type=int,
        default=20_000,
        help="ASEM-1 steps trained like SEM-1 before action gradients reach the segmentation network.",
    )
    parser.add_argument(
        "--action-to-seg-ramp-steps",
        type=int,
        default=20_000,
        help="ASEM-1 linear ramp duration from zero to --action-to-seg-grad-ratio.",
    )
    parser.add_argument(
        "--no-action-to-seg-conflict-projection",
        action="store_false",
        dest="action_to_seg_conflict_projection",
        help="Disable removal of action-gradient components that oppose supervised segmentation.",
    )
    parser.set_defaults(action_to_seg_conflict_projection=True)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--metric-loss-weight", type=float, default=1.0)
    parser.add_argument("--metric-eps", type=float, default=1e-6)
    parser.add_argument(
        "--semantic-prediction-offsets",
        type=int,
        nargs="+",
        default=[1, 8, 24, 60],
        help="SSACT-1 future frame/action offsets used by the semantic dynamics loss.",
    )
    parser.add_argument("--semantic-dynamics-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--semantic-dynamics-loss-weight",
        type=float,
        default=0.1,
        help="SSACT-1 Gaussian semantic rollout NLL weight.",
    )
    parser.add_argument(
        "--semantic-states",
        type=Path,
        default=None,
        help="Offline SSACT-1 semantic targets generated by mycode/precompute_semantic_states.py.",
    )
    parser.add_argument(
        "--mask-quality-dir",
        type=Path,
        default=None,
        help="Offline SAM2 quality directory from sam2_mask_quality.py used to weight segmentation losses.",
    )
    parser.add_argument(
        "--mask-quality-min-score",
        type=float,
        default=0.60,
        help="Minimum offline label-quality score used by legacy hard weighting.",
    )
    parser.add_argument(
        "--mask-quality-weighting",
        choices=["hard", "soft"],
        default="soft",
        help="Use legacy binary quality gates or continuous score-based segmentation-loss weights.",
    )
    parser.add_argument(
        "--mask-quality-full-score",
        type=float,
        default=0.95,
        help="In soft mode, scores at or above this value receive full loss weight.",
    )
    parser.add_argument(
        "--mask-quality-weight-gamma",
        type=float,
        default=1.0,
        help="Exponent for soft quality weights: min(score/full_score, 1)^gamma.",
    )
    parser.add_argument(
        "--phase-labels",
        type=Path,
        default=None,
        help="Offline five-phase soft labels generated by mycode/semantic_phase_labels.py.",
    )
    parser.add_argument("--phase-history-length", type=int, default=16)
    parser.add_argument("--phase-history-stride", type=int, default=4)
    parser.add_argument("--phase-hidden-dim", type=int, default=128)
    parser.add_argument("--phase-loss-weight", type=float, default=0.2)
    parser.add_argument("--phase-min-confidence", type=float, default=0.10)
    parser.add_argument("--phase-teacher-forcing-steps", type=int, default=10_000)
    parser.add_argument("--phase-teacher-forcing-ramp-steps", type=int, default=20_000)
    parser.add_argument(
        "--stage-supervision",
        type=Path,
        default=None,
        help="Offline SSACT-3 event/stage targets generated by mycode/generate_stage_supervision.py.",
    )
    parser.add_argument(
        "--stage-conditioning-mode",
        choices=PhaseConditionedSemanticAdapter.VALID_MODES,
        default="attention-film",
        help="SSACT-3 phase-conditioned semantic fusion mode; modes are directly ablatable.",
    )
    parser.add_argument("--stage-predicted-input-warmup-steps", type=int, default=20_000)
    parser.add_argument("--stage-predicted-input-ramp-steps", type=int, default=10_000)
    parser.add_argument("--stage-phase-loss-weight", type=float, default=0.20)
    parser.add_argument("--stage-event-loss-weight", type=float, default=0.10)
    parser.add_argument("--stage-progress-loss-weight", type=float, default=0.10)
    parser.add_argument("--stage-transition-loss-weight", type=float, default=0.10)
    parser.add_argument("--stage-relation-loss-weight", type=float, default=0.10)
    parser.add_argument("--stage-attention-regularization-weight", type=float, default=0.01)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=100)
    parser.add_argument("--pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")
    parser.add_argument(
        "--pretrained-segmentation-checkpoint",
        type=Path,
        default=None,
        help="Frozen five-class TinyUNet checkpoint required by UNET-SEM.",
    )
    parser.add_argument(
        "--identity-embedding-gate-init",
        type=float,
        default=0.01,
        help="Initial camera and modality embedding gates for UNET-SEM.",
    )
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


def load_viewfusion_homography(args: argparse.Namespace) -> None:
    if args.experiment.upper() not in VIEW_FUSION_EXPERIMENTS:
        args.viewfusion_homography = None
        return
    path = args.viewfusion_homography_json.expanduser().resolve()
    payload = json.loads(path.read_text())
    anchor = payload.get("calibration", {}).get("anchor")
    if anchor != "side_right_bottom_to_front_right_bottom":
        raise ValueError(f"ViewFus-v1 requires the right-bottom homography, got {anchor!r} from {path}.")
    matrix = np.asarray(payload["homography_side_to_front"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Invalid homography in {path}: {matrix}.")
    args.viewfusion_homography_json = path
    args.viewfusion_homography = matrix.tolist()


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
        if not isinstance(image, Tensor) or image.ndim not in {4, 5}:
            raise ValueError(f"Expected batched BCHW or BTCHW tensor for '{key}', got {type(image).__name__}.")
        if tuple(image.shape[-2:]) == image_size:
            continue
        original_shape = image.shape
        if image.ndim == 5:
            image = image.flatten(0, 1)
        if key in rgb_keys:
            resized = F.interpolate(image, size=image_size, mode="bilinear", align_corners=False, antialias=True)
        else:
            resized = F.interpolate(image, size=image_size, mode="nearest")
        if len(original_shape) == 5:
            resized = resized.reshape(original_shape[0], original_shape[1], original_shape[2], *image_size)
        resized_batch[key] = resized
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

    if args.experiment.upper() in VIEW_FUSION_EXPERIMENTS and VIEWFUS_FUSED_FRONT_KEY not in available_keys:
        raise KeyError(
            f"{args.experiment} requires derived feature '{VIEWFUS_FUSED_FRONT_KEY}'. Run "
            "mycode/precompute_viewfus_front_semantic.py first."
        )

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
    semantic_keys = [semantic_image_key(rgb_key) for rgb_key in args.rgb_keys]
    if experiment in {"SEM-1", "ASEM-1", "SSACT-1", *FROZEN_SEMANTIC_EXPERIMENTS}:
        return [*semantic_keys, *args.rgb_keys]
    if experiment in {"SEM-1-V2", "SSACT-3"}:
        return list(args.rgb_keys)
    if experiment in VIEW_FUSION_EXPERIMENTS:
        if len(args.rgb_keys) != 2:
            raise ValueError(f"{experiment} requires exactly front and side RGB keys, got {args.rgb_keys}.")
        return [args.rgb_keys[0], args.rgb_keys[1], VIEWFUS_FUSED_FRONT_KEY]
    if experiment == "SEM-2":
        return semantic_keys
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


def act_identity_ids_for_experiment(
    args: argparse.Namespace,
) -> tuple[list[int] | None, list[int] | None]:
    experiment = args.experiment.upper()
    if experiment in FROZEN_SEMANTIC_EXPERIMENTS:
        view_ids = list(range(len(args.rgb_keys)))
        return [*view_ids, *view_ids], [1] * len(view_ids) + [0] * len(view_ids)
    if experiment in VIEW_FUSION_EXPERIMENTS:
        return [0, 1, 0], [0, 0, 1]
    return None, None


def act_uses_latent(experiment: str) -> bool:
    return experiment.upper() in {"2A", "2B", "3"}


def act_uses_semantic_env_state(experiment: str) -> bool:
    return experiment.upper() == "5"


def act_uses_phase_env_state(experiment: str) -> bool:
    return experiment.upper() in PHASE_CONDITIONED_EXPERIMENTS


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
    semantic_features = {
        semantic_image_key(rgb_key): PolicyFeature(type=FeatureType.VISUAL, shape=features[rgb_key].shape)
        for rgb_key in args.rgb_keys
    }
    missing = [
        key
        for key in [*input_keys, "action"]
        if key not in features and key not in semantic_features
    ]
    if missing:
        raise KeyError(f"Missing feature(s) for ACT policy: {missing}")

    input_features = {
        key: features[key] if key in features else semantic_features[key]
        for key in input_keys
    }
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
    if act_uses_phase_env_state(args.experiment):
        input_features[OBS_ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(5,))

    viewfusion_homography = getattr(args, "viewfusion_homography", None)
    if viewfusion_homography is not None and args.image_size is not None:
        original_shape = features[args.rgb_keys[0]].shape
        original_height, original_width = original_shape[-2:]
        resized_height, resized_width = args.image_size
        scale = np.diag(
            [resized_width / original_width, resized_height / original_height, 1.0]
        )
        viewfusion_homography = (
            scale @ np.asarray(viewfusion_homography, dtype=np.float64) @ np.linalg.inv(scale)
        ).tolist()

    image_camera_ids, image_modality_ids = act_identity_ids_for_experiment(args)
    identity_embedding_kwargs = {}
    if args.experiment.upper() in FROZEN_SEMANTIC_EXPERIMENTS:
        identity_embedding_kwargs = {
            "image_camera_embedding_mode": "gated",
            "image_camera_embedding_std": 0.02,
            "image_camera_embedding_gate_init": args.identity_embedding_gate_init,
            "image_modality_embedding_mode": "gated",
            "image_modality_embedding_std": 0.02,
            "image_modality_embedding_gate_init": args.identity_embedding_gate_init,
        }
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
        image_camera_ids=image_camera_ids,
        image_modality_ids=image_modality_ids,
        **identity_embedding_kwargs,
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
        dice_loss_weight=getattr(args, "dice_loss_weight", 1.0),
        semantic_loss_weight=args.semantic_loss_weight,
        metric_loss_weight=args.metric_loss_weight,
        metric_eps=args.metric_eps,
        semantic_temperature=getattr(args, "semantic_temperature", 1.0),
        semantic_adapter_base_channels=getattr(args, "semantic_adapter_base_channels", 32),
        semantic_fusion_warmup_steps=getattr(args, "semantic_fusion_warmup_steps", 20_000),
        semantic_fusion_ramp_steps=getattr(args, "semantic_fusion_ramp_steps", 10_000),
        viewfusion_homography=viewfusion_homography,
        viewfusion_adapter_base_channels=getattr(args, "viewfusion_adapter_base_channels", 16),
        viewfusion_loss_weight=getattr(args, "viewfusion_loss_weight", 1.0),
        viewfusion_teacher_forcing_steps=getattr(args, "viewfusion_teacher_forcing_steps", 20_000),
        viewfusion_teacher_forcing_ramp_steps=getattr(
            args, "viewfusion_teacher_forcing_ramp_steps", 20_000
        ),
        action_to_seg_grad_ratio=getattr(args, "action_to_seg_grad_ratio", 0.1),
        action_to_seg_warmup_steps=getattr(args, "action_to_seg_warmup_steps", 20_000),
        action_to_seg_ramp_steps=getattr(args, "action_to_seg_ramp_steps", 20_000),
        action_to_seg_conflict_projection=getattr(args, "action_to_seg_conflict_projection", True),
        semantic_prediction_offsets=list(getattr(args, "semantic_prediction_offsets", [1, 8, 24, 60])),
        semantic_dynamics_hidden_dim=getattr(args, "semantic_dynamics_hidden_dim", 256),
        semantic_dynamics_loss_weight=getattr(args, "semantic_dynamics_loss_weight", 0.1),
        phase_feature_reliability=getattr(args, "phase_feature_reliability", None),
        phase_history_length=getattr(args, "phase_history_length", 16),
        phase_hidden_dim=getattr(args, "phase_hidden_dim", 128),
        phase_loss_weight=getattr(args, "phase_loss_weight", 0.2),
        phase_teacher_forcing_steps=getattr(args, "phase_teacher_forcing_steps", 10_000),
        phase_teacher_forcing_ramp_steps=getattr(args, "phase_teacher_forcing_ramp_steps", 20_000),
        stage_feature_dim=getattr(args, "stage_feature_dim", None),
        stage_conditioning_mode=getattr(args, "stage_conditioning_mode", "attention-film"),
        stage_predicted_input_warmup_steps=getattr(
            args, "stage_predicted_input_warmup_steps", 20_000
        ),
        stage_predicted_input_ramp_steps=getattr(args, "stage_predicted_input_ramp_steps", 10_000),
        stage_phase_loss_weight=getattr(args, "stage_phase_loss_weight", 0.20),
        stage_event_loss_weight=getattr(args, "stage_event_loss_weight", 0.10),
        stage_progress_loss_weight=getattr(args, "stage_progress_loss_weight", 0.10),
        stage_transition_loss_weight=getattr(args, "stage_transition_loss_weight", 0.10),
        stage_relation_loss_weight=getattr(args, "stage_relation_loss_weight", 0.10),
        stage_attention_regularization_weight=getattr(
            args, "stage_attention_regularization_weight", 0.01
        ),
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        pretrained_segmentation_checkpoint=getattr(args, "pretrained_segmentation_checkpoint", None),
        mask_quality_weighting=getattr(args, "mask_quality_weighting", "hard"),
        mask_quality_full_score=getattr(args, "mask_quality_full_score", 0.95),
        mask_quality_weight_gamma=getattr(args, "mask_quality_weight_gamma", 1.0),
        phase_history_stride=getattr(args, "phase_history_stride", 1),
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
        dice_loss_weight=args.dice_loss_weight,
        semantic_loss_weight=args.semantic_loss_weight,
        metric_loss_weight=args.metric_loss_weight,
        metric_eps=args.metric_eps,
        semantic_temperature=args.semantic_temperature,
        semantic_adapter_base_channels=args.semantic_adapter_base_channels,
        semantic_fusion_warmup_steps=args.semantic_fusion_warmup_steps,
        semantic_fusion_ramp_steps=args.semantic_fusion_ramp_steps,
        viewfusion_homography_json=str(args.viewfusion_homography_json),
        viewfusion_homography=args.viewfusion_homography,
        viewfusion_adapter_base_channels=args.viewfusion_adapter_base_channels,
        viewfusion_loss_weight=args.viewfusion_loss_weight,
        viewfusion_teacher_forcing_steps=args.viewfusion_teacher_forcing_steps,
        viewfusion_teacher_forcing_ramp_steps=args.viewfusion_teacher_forcing_ramp_steps,
        action_to_seg_grad_ratio=args.action_to_seg_grad_ratio,
        action_to_seg_warmup_steps=args.action_to_seg_warmup_steps,
        action_to_seg_ramp_steps=args.action_to_seg_ramp_steps,
        action_to_seg_conflict_projection=args.action_to_seg_conflict_projection,
        semantic_prediction_offsets=list(args.semantic_prediction_offsets),
        semantic_dynamics_hidden_dim=args.semantic_dynamics_hidden_dim,
        semantic_dynamics_loss_weight=args.semantic_dynamics_loss_weight,
        semantic_states=str(args.semantic_states) if args.semantic_states is not None else None,
        mask_quality_dir=str(args.mask_quality_dir) if args.mask_quality_dir is not None else None,
        mask_quality_min_score=args.mask_quality_min_score,
        mask_quality_weighting=args.mask_quality_weighting,
        mask_quality_full_score=args.mask_quality_full_score,
        mask_quality_weight_gamma=args.mask_quality_weight_gamma,
        phase_labels=str(args.phase_labels) if args.phase_labels is not None else None,
        phase_history_length=args.phase_history_length,
        phase_history_stride=args.phase_history_stride,
        phase_hidden_dim=args.phase_hidden_dim,
        phase_loss_weight=args.phase_loss_weight,
        phase_min_confidence=args.phase_min_confidence,
        phase_teacher_forcing_steps=args.phase_teacher_forcing_steps,
        phase_teacher_forcing_ramp_steps=args.phase_teacher_forcing_ramp_steps,
        phase_feature_reliability=args.phase_feature_reliability,
        stage_supervision=(
            str(args.stage_supervision) if args.stage_supervision is not None else None
        ),
        stage_conditioning_mode=args.stage_conditioning_mode,
        stage_predicted_input_warmup_steps=args.stage_predicted_input_warmup_steps,
        stage_predicted_input_ramp_steps=args.stage_predicted_input_ramp_steps,
        stage_phase_loss_weight=args.stage_phase_loss_weight,
        stage_event_loss_weight=args.stage_event_loss_weight,
        stage_progress_loss_weight=args.stage_progress_loss_weight,
        stage_transition_loss_weight=args.stage_transition_loss_weight,
        stage_relation_loss_weight=args.stage_relation_loss_weight,
        stage_attention_regularization_weight=args.stage_attention_regularization_weight,
        stage_feature_dim=args.stage_feature_dim,
        seed=args.seed,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=args.pretrained_backbone_weights,
        pretrained_segmentation_checkpoint=(
            str(args.pretrained_segmentation_checkpoint)
            if args.pretrained_segmentation_checkpoint is not None
            else None
        ),
        identity_embedding_gate_init=args.identity_embedding_gate_init,
        canonical_mask_definition=list(CANONICAL_SEMANTIC_MASK_KEYS),
        no_gripper=args.no_gripper,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "mask_act_run_config.json", "w") as f:
        payload = asdict(run_cfg)
        payload["dataset_view_image_keys"] = image_keys_in_view
        payload["act_image_keys"] = act_image_keys_for_experiment(args)
        if args.experiment.upper() in VIEW_FUSION_EXPERIMENTS:
            payload["view_fusion"] = {
                "version": "ViewFus-v1",
                "fused_front_key": VIEWFUS_FUSED_FRONT_KEY,
                "semantic_source": "teacher_labels_then_model_predictions",
                "segmentation_module": "shared_two_view_unet",
                "fusion_module": "homography_aligned_fusion_adapter",
                "teacher_forcing_steps": args.viewfusion_teacher_forcing_steps,
                "teacher_forcing_ramp_steps": args.viewfusion_teacher_forcing_ramp_steps,
                "shared_visual_backbone": True,
                "image_camera_ids": [0, 1, 0],
                "image_modality_ids": [0, 0, 1],
                "camera_id_meanings": {"0": "front", "1": "side"},
                "modality_id_meanings": {"0": "rgb", "1": "semantic_rgb"},
            }
        if args.experiment.upper() in FROZEN_SEMANTIC_EXPERIMENTS:
            camera_ids, modality_ids = act_identity_ids_for_experiment(args)
            payload["frozen_semantic_input"] = {
                "segmentation_checkpoint": str(args.pretrained_segmentation_checkpoint),
                "segmentation_trainable": False,
                "representation": "five_class_softmax_to_semantic_rgb",
                "shared_visual_backbone": True,
                "image_camera_ids": camera_ids,
                "image_modality_ids": modality_ids,
                "camera_embedding": {
                    "mode": "gated",
                    "std": 0.02,
                    "gate_init": args.identity_embedding_gate_init,
                },
                "modality_embedding": {
                    "mode": "gated",
                    "std": 0.02,
                    "gate_init": args.identity_embedding_gate_init,
                },
                "camera_id_meanings": {
                    str(index): key.rsplit(".", 1)[-1]
                    for index, key in enumerate(args.rgb_keys)
                },
                "modality_id_meanings": {"0": "rgb", "1": "semantic_rgb"},
            }
        if args.mask_quality_dir is not None:
            quality_summary_path = args.mask_quality_dir.expanduser().resolve() / "summary.json"
            if not quality_summary_path.is_file():
                raise FileNotFoundError(f"Mask quality summary is missing: {quality_summary_path}")
            with open(quality_summary_path) as quality_summary_file:
                payload["mask_quality_summary"] = json.load(quality_summary_file)
        if args.experiment in SEMANTIC_EXPERIMENTS:
            mask_suffixes, _ = build_mask_layout(list(args.rgb_keys), list(args.mask_target_keys))
            payload["semantic_classes"] = ["background", *mask_suffixes]
            payload["semantic_palette_rgb"] = {
                name: [round(channel * 255) for channel in color]
                for name, color in SEMANTIC_PALETTE_BY_CLASS.items()
            }
            payload["semantic_map_action_gradient"] = args.experiment in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS
            if args.experiment in SEMANTIC_FEATURE_FUSION_EXPERIMENTS:
                payload["semantic_input_representation"] = "five_class_soft_probabilities"
                payload["semantic_fusion"] = {
                    "type": (
                        "phase_conditioned_relation_residual_at_resnet_layer4"
                        if args.experiment in STAGE_AWARE_EXPERIMENTS
                        else "per_view_residual_at_resnet_layer4"
                    ),
                    "shared_adapter_across_views": True,
                    "adds_transformer_tokens": False,
                    "probabilities_detached_from_action_loss": True,
                    "zero_initialized_output_projection": True,
                }
            if args.experiment in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS:
                payload["action_supervision_design"] = "mycode/ASEM_1_DESIGN.md"
            if args.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
                payload["semantic_state_definition"] = "mycode/SEMANTIC_SERVO_FEASIBILITY.md"
                payload["semantic_state_names"] = list(
                    SoftSemanticStateExtractor(include_confidence=False).feature_names(
                        [key.rsplit(".", 1)[-1] for key in args.rgb_keys]
                    )
                )
                semantic_state_summary_path = args.semantic_states.with_suffix(".json")
                if not semantic_state_summary_path.is_file():
                    raise FileNotFoundError(
                        f"Semantic-state summary is missing: {semantic_state_summary_path}"
                    )
                with open(semantic_state_summary_path) as semantic_state_summary_file:
                    payload["semantic_state_summary"] = json.load(semantic_state_summary_file)
            if args.experiment == "SSACT-1":
                payload["phase_design"] = "mycode/SSACT_1_DESIGN.md"
                payload["phase_names"] = ["uncover", "expose", "transport", "restore", "done"]
                payload["phase_feature_names"] = list(
                    PhaseSemanticFeatureExtractor(
                        torch.tensor(args.phase_feature_reliability, dtype=torch.float32)
                    ).feature_names([key.rsplit(".", 1)[-1] for key in args.rgb_keys])
                )
                phase_summary_path = args.phase_labels.with_suffix(".json")
                if phase_summary_path.is_file():
                    with open(phase_summary_path) as phase_summary_file:
                        payload["phase_label_summary"] = json.load(phase_summary_file)
            if args.experiment in STAGE_AWARE_EXPERIMENTS:
                payload["stage_design"] = "mycode/SSACT_3_STAGEFILM_DESIGN.md"
                payload["phase_names"] = list(STAGE_PHASE_NAMES)
                payload["event_names"] = list(STAGE_EVENT_NAMES)
                payload["transition_names"] = list(STAGE_TRANSITION_NAMES)
                payload["relation_names"] = list(STAGE_RELATION_NAMES)
                payload["semantic_state_names"] = list(
                    SoftSemanticStateExtractor(include_confidence=True).feature_names(
                        [key.rsplit(".", 1)[-1] for key in args.rgb_keys]
                    )
                )
                stage_summary_path = args.stage_supervision.with_suffix(".json")
                if not stage_summary_path.is_file():
                    raise FileNotFoundError(
                        f"Stage-supervision summary is missing: {stage_summary_path}"
                    )
                with open(stage_summary_path) as stage_summary_file:
                    payload["stage_supervision_summary"] = json.load(stage_summary_file)
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
    if model.uses_semantic_maps():
        rgbs = model._get_rgb_inputs(raw_batch, device=device)
        logits_by_view, _ = model.predict_view_logits_and_latent_from_rgbs(rgbs)
        probabilities_by_view = model.semantic_probabilities(logits_by_view)
        soft_maps = model.semantic_rgb_maps(probabilities_by_view)
        targets_by_view = None
        if not model.uses_frozen_semantic_segmenter():
            targets_by_view = model.build_semantic_targets(raw_batch, device=device)

        tiles = []
        for view_idx, rgb_key in enumerate(model.rgb_keys):
            view_name = rgb_key.rsplit(".", 1)[-1]
            hard_prediction = probabilities_by_view[view_idx].argmax(dim=1)
            hard_map = model.semantic_palette[hard_prediction].permute(0, 3, 1, 2)
            tiles.append(make_labeled_tile(raw_batch[rgb_key][0], f"rgb {view_name}"))
            if targets_by_view is not None:
                gt_map = model.semantic_palette[targets_by_view[view_idx]].permute(0, 3, 1, 2)
                tiles.append(make_labeled_tile(gt_map[0], f"gt semantic {view_name}"))
            tiles.extend(
                [
                    make_labeled_tile(soft_maps[view_idx][0], f"soft semantic {view_name}"),
                    make_labeled_tile(hard_map[0], f"hard preview {view_name}"),
                ]
            )

        columns = 4
        rows = (len(tiles) + columns - 1) // columns
        tile_w, tile_h = tiles[0].size
        grid = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
        for idx, tile in enumerate(tiles):
            grid.paste(tile, ((idx % columns) * tile_w, (idx // columns) * tile_h))
        grid.save(checkpoint_dir / f"semantic_preview_step_{step:06d}.png")
        if was_training:
            model.train()
        return

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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source_root = args.root.resolve()
    normalize_dataset_keys(args, source_root)
    if args.experiment.upper() in FROZEN_SEMANTIC_EXPERIMENTS:
        if args.pretrained_segmentation_checkpoint is None:
            raise ValueError(
                f"{args.experiment} requires --pretrained-segmentation-checkpoint."
            )
        args.pretrained_segmentation_checkpoint = (
            args.pretrained_segmentation_checkpoint.expanduser().resolve()
        )
        if not args.pretrained_segmentation_checkpoint.is_file():
            raise FileNotFoundError(
                f"Pretrained segmentation checkpoint not found: {args.pretrained_segmentation_checkpoint}"
            )
        if args.identity_embedding_gate_init < 0:
            raise ValueError("--identity-embedding-gate-init must be non-negative.")
    load_viewfusion_homography(args)
    mask_suffixes, _ = build_mask_layout(list(args.rgb_keys), list(args.mask_target_keys))
    if (
        args.experiment in (PREDICTIVE_SEMANTIC_EXPERIMENTS | STAGE_AWARE_EXPERIMENTS)
        and mask_suffixes != list(SEMANTIC_CLASSES)
    ):
        raise ValueError(
            f"{args.experiment} mask order must be {list(SEMANTIC_CLASSES)} so offline states align; "
            f"got {mask_suffixes}."
        )

    if args.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS and args.semantic_states is None:
        raise ValueError(
            f"{args.experiment} requires --semantic-states from "
            "mycode/precompute_semantic_states.py."
        )
    if args.semantic_states is not None and args.experiment not in PREDICTIVE_SEMANTIC_EXPERIMENTS:
        raise ValueError("--semantic-states is currently only used by SSACT-1.")
    if args.mask_quality_dir is not None and args.experiment not in SEMANTIC_EXPERIMENTS:
        raise ValueError(
            "--mask-quality-dir currently supports semantic experiments including SSACT-3. "
            "The legacy independent-BCE mask experiments require a separate classwise BCE masking path."
        )
    if args.mask_quality_dir is not None and args.experiment in FROZEN_SEMANTIC_EXPERIMENTS:
        raise ValueError(
            "UNET-SEM uses a frozen pretrained segmenter and has no segmentation loss to quality-weight."
        )
    if args.semantic_adapter_base_channels <= 0:
        raise ValueError(
            f"--semantic-adapter-base-channels must be positive, got {args.semantic_adapter_base_channels}."
        )
    if args.semantic_fusion_warmup_steps < 0 or args.semantic_fusion_ramp_steps < 0:
        raise ValueError(
            "--semantic-fusion-warmup-steps and --semantic-fusion-ramp-steps must be non-negative."
        )
    if args.viewfusion_adapter_base_channels <= 0:
        raise ValueError("--viewfusion-adapter-base-channels must be positive.")
    if args.viewfusion_loss_weight < 0:
        raise ValueError("--viewfusion-loss-weight must be non-negative.")
    if args.viewfusion_teacher_forcing_steps < 0 or args.viewfusion_teacher_forcing_ramp_steps < 0:
        raise ValueError("ViewFus-v1 teacher-forcing step counts must be non-negative.")
    if not 0.0 < args.mask_quality_full_score <= 1.0:
        raise ValueError(
            f"--mask-quality-full-score must be in (0, 1], got {args.mask_quality_full_score}."
        )
    if args.mask_quality_weight_gamma <= 0.0:
        raise ValueError(
            f"--mask-quality-weight-gamma must be positive, got {args.mask_quality_weight_gamma}."
        )
    if args.experiment == "SSACT-1" and args.phase_labels is None:
        raise ValueError(
            "SSACT-1 requires --phase-labels from mycode/semantic_phase_labels.py."
        )
    if args.phase_labels is not None and args.experiment != "SSACT-1":
        raise ValueError("--phase-labels is currently only used by SSACT-1.")
    if args.experiment in STAGE_AWARE_EXPERIMENTS and args.stage_supervision is None:
        raise ValueError(
            "SSACT-3 requires --stage-supervision from mycode/generate_stage_supervision.py."
        )
    if args.stage_supervision is not None and args.experiment not in STAGE_AWARE_EXPERIMENTS:
        raise ValueError("--stage-supervision is currently only used by SSACT-3.")

    if args.overwrite_output and args.resume_checkpoint is not None:
        raise ValueError("--overwrite-output cannot be used together with --resume-checkpoint.")

    if args.output_dir.exists() and args.overwrite_output:
        shutil.rmtree(args.output_dir)

    source_meta = LeRobotDatasetMetadata(args.repo_id, root=source_root)
    semantic_state_store = None
    if args.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
        semantic_state_store = SemanticStateStore(
            args.semantic_states.expanduser().resolve(),
            total_frames=source_meta.total_frames,
            rgb_keys=list(args.rgb_keys),
            minimum_quality_score=args.mask_quality_min_score,
        )
    phase_store = None
    if args.experiment == "SSACT-1":
        phase_store = SemanticPhaseStore(
            args.phase_labels.expanduser().resolve(),
            total_frames=source_meta.total_frames,
            num_views=len(args.rgb_keys),
            history_length=args.phase_history_length,
            history_stride=args.phase_history_stride,
            minimum_confidence=args.phase_min_confidence,
        )
        args.phase_feature_reliability = phase_store.feature_reliability.tolist()
    else:
        args.phase_feature_reliability = None
    stage_store = None
    if args.experiment in STAGE_AWARE_EXPERIMENTS:
        stage_store = StageSupervisionStore(
            args.stage_supervision.expanduser().resolve(),
            total_frames=source_meta.total_frames,
            history_length=args.phase_history_length,
            history_stride=args.phase_history_stride,
        )
        if stage_store.num_views != len(args.rgb_keys):
            raise ValueError(
                f"Stage supervision has {stage_store.num_views} views but training uses "
                f"{len(args.rgb_keys)}."
            )
        cached_rgb_keys = stage_store.rgb_keys.astype(str).tolist()
        if cached_rgb_keys != list(args.rgb_keys):
            raise ValueError(
                f"Stage supervision RGB keys {cached_rgb_keys} do not match training keys "
                f"{list(args.rgb_keys)}."
            )
        args.stage_feature_dim = stage_store.feature_dim
    else:
        args.stage_feature_dim = None

    image_keys_in_view = sorted(
        {
            *args.rgb_keys,
            *(
                []
                if args.experiment.upper() in FROZEN_SEMANTIC_EXPERIMENTS
                else args.mask_target_keys
            ),
            *(
                [VIEWFUS_FUSED_FRONT_KEY]
                if args.experiment.upper() in VIEW_FUSION_EXPERIMENTS
                else []
            ),
        }
    )
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
    if (phase_store is not None or stage_store is not None) and meta.total_frames != source_meta.total_frames:
        raise ValueError(
            f"Filtered dataset has {meta.total_frames} frames but offline labels cover "
            f"{source_meta.total_frames}."
        )
    delta_timestamps = {"action": [i / meta.fps for i in range(args.chunk_size)]}
    dataset = LeRobotDataset(
        args.repo_id,
        root=filtered_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-4,
        video_backend=args.video_backend,
    )

    mask_quality_store = None
    if args.mask_quality_dir is not None and semantic_state_store is None:
        quality_dir = args.mask_quality_dir.expanduser().resolve()
        mask_quality_store = MaskQualityStore(
            quality_dir,
            list(args.mask_target_keys),
            total_frames=meta.total_frames,
            minimum_score=args.mask_quality_min_score,
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

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
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
        generator=torch.Generator().manual_seed(args.seed),
    )
    dl_iter = cycle(dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Experiment: {args.experiment}")
    print(f"Dataset view: {filtered_root}")
    print(f"RGB keys: {args.rgb_keys}")
    print(f"ACT image keys: {act_image_keys_for_experiment(args)}")
    print(f"Mask supervision keys: {args.mask_target_keys}")
    print(f"Training image size: {args.image_size or 'dataset native resolution'}")
    print(f"Random seed: {args.seed}")
    if mask_quality_store is not None:
        if args.mask_quality_weighting == "soft":
            print(
                f"SAM2 label quality: {mask_quality_store.quality_dir}; continuous loss weights "
                f"min(score/{args.mask_quality_full_score:g}, 1)^{args.mask_quality_weight_gamma:g}."
            )
        else:
            print(
                f"SAM2 label quality: {mask_quality_store.quality_dir}; "
                f"minimum score={mask_quality_store.minimum_score:g}; low-quality labels masked from losses."
            )
    if phase_store is not None:
        print(
            f"Semantic phases: {phase_store.path}; history={phase_store.history_length}x"
            f"{phase_store.history_stride} frames; minimum confidence={phase_store.minimum_confidence:g}."
        )
    if stage_store is not None:
        print(
            f"Stage supervision: {stage_store.path}; history={stage_store.history_length}x"
            f"{stage_store.history_stride} frames; feature_dim={stage_store.feature_dim}."
        )
    if semantic_state_store is not None:
        print(
            f"Offline semantic states: {semantic_state_store.path}; future mask video decoding disabled; "
            f"minimum embedded quality score={semantic_state_store.minimum_quality_score:g}."
        )
    if args.experiment in SEMANTIC_EXPERIMENTS:
        print(f"Semantic classes: {['background', *build_mask_layout(args.rgb_keys, args.mask_target_keys)[0]]}")
        if args.experiment in FROZEN_SEMANTIC_EXPERIMENTS:
            camera_ids, modality_ids = act_identity_ids_for_experiment(args)
            print(
                f"Frozen segmentation: {args.pretrained_segmentation_checkpoint}; no segmentation "
                "parameters or losses are trained. Soft semantic RGB maps and RGB share ACT ResNet18."
            )
            print(
                f"Identity embeddings: camera IDs={camera_ids}, modality IDs={modality_ids}; "
                f"camera/modality mode=gated, std=0.02, gate_init={args.identity_embedding_gate_init:g}."
            )
        elif args.experiment in ACTION_SUPERVISED_SEMANTIC_EXPERIMENTS:
            semantic_description = (
                "Semantic loss: weighted multiclass CE + "
                f"{args.dice_loss_weight:g} * foreground Dice"
            )
            print(
                f"{semantic_description}; action gradient to semantic map: enabled with "
                f"{args.action_to_seg_warmup_steps} warmup steps, "
                f"{args.action_to_seg_ramp_steps} ramp steps, and at most "
                f"{args.action_to_seg_grad_ratio:g}x supervised segmentation gradient norm; "
                f"conflict projection: {args.action_to_seg_conflict_projection}."
            )
        else:
            semantic_description = (
                "Semantic loss: weighted multiclass CE + "
                f"{args.dice_loss_weight:g} * foreground Dice"
            )
            print(f"{semantic_description}; action gradient to semantic map: disabled.")
        if args.experiment in VIEW_FUSION_EXPERIMENTS:
            print(
                "View fusion: shared two-view U-Net -> fixed-homography fusion adapter -> fused front SEG; "
                f"fusion_loss_weight={args.viewfusion_loss_weight:g}; label teacher forcing="
                f"{args.viewfusion_teacher_forcing_steps} + "
                f"{args.viewfusion_teacher_forcing_ramp_steps} ramp steps; camera IDs=[0, 1, 0], "
                "modality IDs=[0, 0, 1]."
            )
        if args.experiment in SEMANTIC_FEATURE_FUSION_EXPERIMENTS:
            print(
                "Semantic fusion: five soft class probabilities -> shared lightweight adapter -> "
                "per-view RGB ResNet layer4 residual; no extra transformer tokens; segmentation detached "
                f"from action loss; warmup={args.semantic_fusion_warmup_steps}, "
                f"ramp={args.semantic_fusion_ramp_steps}."
            )
        if args.experiment in PREDICTIVE_SEMANTIC_EXPERIMENTS:
            print(
                "Semantic dynamics: expert-action conditioned Gaussian rollout at offsets "
                f"{args.semantic_prediction_offsets}; hidden_dim={args.semantic_dynamics_hidden_dim}; "
                f"loss_weight={args.semantic_dynamics_loss_weight:g}. This output is predictive, not "
                "CLF-certified until independent error-bound calibration is complete."
            )
        if args.experiment == "SSACT-1":
            print(
                "Phase model: ordered five-phase soft supervision; "
                f"history={args.phase_history_length} samples x {args.phase_history_stride} frames; "
                f"hidden_dim={args.phase_hidden_dim}; loss_weight={args.phase_loss_weight:g}; "
                f"teacher forcing={args.phase_teacher_forcing_steps} + "
                f"{args.phase_teacher_forcing_ramp_steps} ramp steps; "
                f"view/class reliability={args.phase_feature_reliability}."
            )
        if args.experiment in STAGE_AWARE_EXPERIMENTS:
            print(
                "Stage model: event-derived expose/separate/transport/restore/done supervision; "
                f"conditioning={args.stage_conditioning_mode}; history={args.phase_history_length}x"
                f"{args.phase_history_stride}; predicted-input warmup/ramp="
                f"{args.stage_predicted_input_warmup_steps}/{args.stage_predicted_input_ramp_steps}; "
                f"phase teacher forcing={args.phase_teacher_forcing_steps} + "
                f"{args.phase_teacher_forcing_ramp_steps} ramp."
            )
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
            if semantic_state_store is not None:
                raw_batch = semantic_state_store.add_batch_semantics(
                    raw_batch,
                    prediction_offsets=tuple(args.semantic_prediction_offsets),
                )
            if mask_quality_store is not None:
                raw_batch = mask_quality_store.add_batch_quality(
                    raw_batch,
                    mask_keys=list(args.mask_target_keys),
                    mask_key_map=model.mask_key_map,
                    num_views=len(args.rgb_keys),
                    num_classes=len(model.mask_suffixes),
                )
            if phase_store is not None:
                raw_batch = phase_store.add_batch_phase(raw_batch)
            if stage_store is not None:
                raw_batch = stage_store.add_batch(raw_batch)
            resize_mask_keys = [
                *(
                    []
                    if args.experiment.upper() in FROZEN_SEMANTIC_EXPERIMENTS
                    else args.mask_target_keys
                ),
                *(
                    [VIEWFUS_FUSED_FRONT_KEY]
                    if args.experiment.upper() in VIEW_FUSION_EXPERIMENTS
                    else []
                ),
            ]
            raw_batch = resize_training_visuals(
                raw_batch,
                rgb_keys=args.rgb_keys,
                mask_keys=resize_mask_keys,
                image_size=args.image_size,
            )
            batch = preprocessor(raw_batch)

            optimizer.zero_grad(set_to_none=True)
            model.set_training_step(step)
            loss, logs = model(batch, raw_batch)
            model.backward_training_loss(loss, logs)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip_norm)
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
