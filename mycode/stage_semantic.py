"""Modular stage-aware semantic components for SSACT-3-StageFiLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


PHASE_NAMES = ("expose", "separate", "transport", "restore", "done")
EVENT_NAMES = (
    "object_visible",
    "object_separated",
    "object_inside_region",
    "cover_restored",
    "object_lost",
    "separation_lost",
    "object_left_region",
    "semantic_valid",
)
TRANSITION_NAMES = ("stay", "advance", "rollback", "uncertain")
RELATION_NAMES = (
    "cover_area",
    "object_area",
    "object_cover_contact",
    "tool_object_distance",
    "object_region_distance",
    "object_region_contact",
    "region_area",
)
SEMANTIC_INPUT_NAMES = (
    "background",
    "occluder",
    "object",
    "region",
    "tool",
    "object_occluder_near",
    "object_region_near",
    "tool_object_near",
)

# Rows follow PHASE_NAMES and columns follow RELATION_NAMES. These masks make
# "what matters in this phase" an explicit, inspectable auxiliary objective.
PHASE_RELATION_MASK = torch.tensor(
    [
        [1, 1, 1, 0, 0, 0, 0],  # expose: cover and emerging object
        [0, 1, 1, 0, 0, 0, 0],  # separate: object-cover boundary/contact
        [0, 1, 0, 1, 1, 1, 1],  # transport: tool-object-region geometry
        [1, 0, 0, 0, 0, 1, 1],  # restore: cover while preserving placement
        [1, 0, 0, 0, 0, 1, 1],  # done: verify cover and placement
    ],
    dtype=torch.float32,
)

# The policy never hard-deletes a channel: the adapter uses 0.5 + attention.
PHASE_SEMANTIC_PRIOR = torch.tensor(
    [
        [0.05, 1.00, 1.00, 0.10, 0.20, 1.00, 0.10, 0.10],
        [0.05, 1.00, 1.00, 0.10, 0.40, 1.00, 0.10, 0.30],
        [0.05, 0.15, 1.00, 1.00, 1.00, 0.15, 1.00, 1.00],
        [0.05, 1.00, 0.60, 0.70, 0.30, 0.30, 0.80, 0.20],
        [0.05, 0.80, 0.50, 0.70, 0.10, 0.20, 0.80, 0.10],
    ],
    dtype=torch.float32,
)


def weighted_mean(loss: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    weight = weight.to(device=loss.device, dtype=loss.dtype)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / weight.expand_as(loss).sum().clamp_min(eps)


def weighted_soft_cross_entropy(logits: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    return weighted_mean(loss, weight)


class StageSupervisionStore:
    """Frame-level event-derived supervision and semantic histories."""

    def __init__(
        self,
        path: Path,
        *,
        total_frames: int,
        history_length: int,
        history_stride: int,
    ) -> None:
        if history_length <= 0 or history_stride <= 0:
            raise ValueError("Stage history length and stride must be positive.")
        if not path.is_file():
            raise FileNotFoundError(
                f"SSACT-3 stage supervision is missing: {path}. Run "
                "mycode/generate_stage_supervision.py first."
            )
        with np.load(path) as archive:
            required = {
                "semantic_features",
                "phase_probabilities",
                "phase_confidence",
                "event_targets",
                "event_weights",
                "progress_target",
                "progress_weight",
                "transition_target",
                "transition_weight",
                "relation_targets",
                "relation_weights",
                "episode_start_index",
                "rgb_keys",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise KeyError(f"Stage supervision '{path}' is missing {missing}.")
            for name in required:
                setattr(self, name, archive[name].copy())

        expected_rows = {
            "semantic_features": self.semantic_features,
            "phase_probabilities": self.phase_probabilities,
            "phase_confidence": self.phase_confidence,
            "event_targets": self.event_targets,
            "event_weights": self.event_weights,
            "progress_target": self.progress_target,
            "progress_weight": self.progress_weight,
            "transition_target": self.transition_target,
            "transition_weight": self.transition_weight,
            "relation_targets": self.relation_targets,
            "relation_weights": self.relation_weights,
            "episode_start_index": self.episode_start_index,
        }
        bad = {name: values.shape for name, values in expected_rows.items() if values.shape[0] != total_frames}
        if bad:
            raise ValueError(f"Stage supervision does not cover {total_frames} frames: {bad}")
        if self.phase_probabilities.shape[1] != len(PHASE_NAMES):
            raise ValueError("Stage phase target dimension does not match PHASE_NAMES.")
        if self.event_targets.shape != (total_frames, len(EVENT_NAMES)):
            raise ValueError("Stage event target shape does not match EVENT_NAMES.")
        if self.event_weights.shape != self.event_targets.shape:
            raise ValueError("Stage event weights must match event targets.")
        if self.relation_targets.shape != (total_frames, len(RELATION_NAMES)):
            raise ValueError("Stage relation target shape does not match RELATION_NAMES.")
        if self.relation_weights.shape != self.relation_targets.shape:
            raise ValueError("Stage relation weights must match relation targets.")
        if np.any((self.transition_target < 0) | (self.transition_target >= len(TRANSITION_NAMES))):
            raise ValueError("Stage transition targets contain invalid class indices.")
        floating = [
            self.semantic_features,
            self.phase_probabilities,
            self.phase_confidence,
            self.event_targets,
            self.event_weights,
            self.progress_target,
            self.progress_weight,
            self.transition_weight,
            self.relation_targets,
            self.relation_weights,
        ]
        if not all(np.isfinite(values).all() for values in floating):
            raise ValueError(f"Stage supervision '{path}' contains non-finite values.")
        self.path = path
        self.history_length = history_length
        self.history_stride = history_stride
        self.semantic_features = self.semantic_features.astype(np.float32, copy=False)

    @property
    def feature_dim(self) -> int:
        return int(self.semantic_features.shape[1])

    @property
    def num_views(self) -> int:
        return int(len(self.rgb_keys))

    def add_batch(self, raw_batch: dict[str, Any]) -> dict[str, Any]:
        indices = raw_batch["index"].detach().cpu().numpy().astype(np.int64)
        offsets = np.arange(self.history_length - 1, -1, -1, dtype=np.int64) * self.history_stride
        history = indices[:, None] - offsets[None, :]
        history = np.maximum(history, self.episode_start_index[indices, None])
        enriched = dict(raw_batch)
        enriched.update(
            {
                "stage_semantic_history": torch.from_numpy(self.semantic_features[history]),
                "stage_phase_target": torch.from_numpy(self.phase_probabilities[indices].astype(np.float32)),
                "stage_phase_weight": torch.from_numpy(self.phase_confidence[indices].astype(np.float32)),
                "stage_event_target": torch.from_numpy(self.event_targets[indices].astype(np.float32)),
                "stage_event_weight": torch.from_numpy(self.event_weights[indices].astype(np.float32)),
                "stage_progress_target": torch.from_numpy(self.progress_target[indices].astype(np.float32)),
                "stage_progress_weight": torch.from_numpy(self.progress_weight[indices].astype(np.float32)),
                "stage_transition_target": torch.from_numpy(
                    self.transition_target[indices].astype(np.int64)
                ),
                "stage_transition_weight": torch.from_numpy(
                    self.transition_weight[indices].astype(np.float32)
                ),
                "stage_relation_target": torch.from_numpy(self.relation_targets[indices].astype(np.float32)),
                "stage_relation_weight": torch.from_numpy(self.relation_weights[indices].astype(np.float32)),
            }
        )
        return enriched


class StageAwareTemporalModel(nn.Module):
    """Shared temporal belief with independently ablatable task heads."""

    def __init__(
        self,
        semantic_dim: int,
        robot_state_dim: int,
        hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.robot_state_dim = robot_state_dim
        self.hidden_dim = hidden_dim
        self.semantic_input = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.history = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.robot_input = (
            nn.Sequential(nn.Linear(robot_state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
            if robot_state_dim > 0
            else None
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * (2 if self.robot_input is not None else 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.phase_head = nn.Linear(hidden_dim, len(PHASE_NAMES))
        self.event_head = nn.Linear(hidden_dim, len(EVENT_NAMES))
        self.progress_head = nn.Linear(hidden_dim, 1)
        self.transition_head = nn.Linear(hidden_dim, len(TRANSITION_NAMES))
        self.relation_head = nn.Linear(hidden_dim, len(RELATION_NAMES))

    def forward(self, semantic_history: Tensor, robot_state: Tensor | None = None) -> dict[str, Tensor]:
        if semantic_history.ndim != 3 or semantic_history.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Expected semantic history (B, L, {self.semantic_dim}), got {tuple(semantic_history.shape)}."
            )
        encoded, _ = self.history(self.semantic_input(semantic_history))
        parts = [encoded[:, -1]]
        if self.robot_input is not None:
            if robot_state is None or robot_state.shape[-1] != self.robot_state_dim:
                raise ValueError(f"Expected current robot state with width {self.robot_state_dim}.")
            parts.append(self.robot_input(robot_state))
        context = self.fusion(torch.cat(parts, dim=-1))
        return {
            "context": context,
            "phase_logits": self.phase_head(context),
            "event_logits": self.event_head(context),
            "progress": torch.sigmoid(self.progress_head(context)).squeeze(-1),
            "transition_logits": self.transition_head(context),
            "relations": torch.sigmoid(self.relation_head(context)),
        }


class _AdapterDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class PhaseConditionedSemanticAdapter(nn.Module):
    """Relation-channel attention plus residual FiLM for an ACT image feature map."""

    VALID_MODES = ("none", "attention", "film", "attention-film")

    def __init__(
        self,
        output_channels: int,
        context_dim: int,
        *,
        base_channels: int = 32,
        mode: str = "attention-film",
        relation_radius_ratio: float = 0.03,
        attention_delta_scale: float = 0.25,
        film_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unknown stage conditioning mode '{mode}'.")
        if base_channels <= 0 or output_channels <= 0 or context_dim <= 0:
            raise ValueError("Adapter channel and context dimensions must be positive.")
        self.mode = mode
        self.relation_radius_ratio = relation_radius_ratio
        self.attention_delta_scale = attention_delta_scale
        self.film_scale = film_scale
        self.register_buffer("attention_prior", PHASE_SEMANTIC_PRIOR.clone())
        self.attention_delta = nn.Parameter(torch.zeros_like(PHASE_SEMANTIC_PRIOR))

        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.stem = nn.Sequential(
            nn.Conv2d(len(SEMANTIC_INPUT_NAMES), channels[0], 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
        )
        self.downsamples = nn.Sequential(
            _AdapterDownsample(channels[0], channels[1]),
            _AdapterDownsample(channels[1], channels[2]),
            _AdapterDownsample(channels[2], channels[3]),
            _AdapterDownsample(channels[3], channels[3]),
        )
        self.output_proj = nn.Conv2d(channels[3], output_channels, 1)
        self.film = nn.Linear(context_dim, 2 * output_channels)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def _relation_channels(self, probabilities: Tensor) -> Tensor:
        if probabilities.ndim != 4 or probabilities.shape[1] != 5:
            raise ValueError(f"Expected five semantic probabilities BCHW, got {tuple(probabilities.shape)}.")
        height, width = probabilities.shape[-2:]
        radius = max(1, round(min(height, width) * self.relation_radius_ratio))
        kernel = 2 * radius + 1

        def near(first: int, second: int) -> Tensor:
            first_map = probabilities[:, first : first + 1]
            second_map = probabilities[:, second : second + 1]
            first_near = F.max_pool2d(first_map, kernel, stride=1, padding=radius)
            second_near = F.max_pool2d(second_map, kernel, stride=1, padding=radius)
            return (first_map * second_near + second_map * first_near).clamp(0.0, 1.0)

        return torch.cat(
            [
                probabilities,
                near(2, 1),
                near(2, 3),
                near(4, 2),
            ],
            dim=1,
        )

    def phase_attention(self, phase_probabilities: Tensor) -> Tensor:
        if phase_probabilities.ndim != 2 or phase_probabilities.shape[1] != len(PHASE_NAMES):
            raise ValueError("Phase probabilities must have shape (B, 5).")
        learned = self.attention_prior + self.attention_delta_scale * torch.tanh(self.attention_delta)
        return phase_probabilities @ learned.clamp(0.0, 1.25)

    def attention_regularization(self) -> Tensor:
        return self.attention_delta.square().mean()

    def forward(
        self,
        probabilities: Tensor,
        phase_probabilities: Tensor,
        context: Tensor,
        *,
        confidence: Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> Tensor:
        channels = self._relation_channels(probabilities)
        if self.mode in {"attention", "attention-film"}:
            attention = self.phase_attention(phase_probabilities)
            channels = channels * (0.5 + attention.unsqueeze(-1).unsqueeze(-1))
        residual = self.output_proj(self.downsamples(self.stem(channels)))
        if self.mode in {"film", "attention-film"}:
            gamma, beta = self.film(context).chunk(2, dim=-1)
            gamma = self.film_scale * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
            beta = self.film_scale * beta.unsqueeze(-1).unsqueeze(-1)
            residual = residual * (1.0 + gamma) + beta
        if confidence is not None:
            residual = residual * confidence.clamp(0.0, 1.0).view(-1, 1, 1, 1)
        if output_size is not None and residual.shape[-2:] != output_size:
            residual = F.interpolate(residual, size=output_size, mode="bilinear", align_corners=False)
        return residual


def stage_losses(
    outputs: dict[str, Tensor],
    batch: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, float]]:
    device = outputs["phase_logits"].device
    phase_target = batch["stage_phase_target"].to(device=device, dtype=torch.float32)
    phase_weight = batch["stage_phase_weight"].to(device=device, dtype=torch.float32)
    event_target = batch["stage_event_target"].to(device=device, dtype=torch.float32)
    event_weight = batch["stage_event_weight"].to(device=device, dtype=torch.float32)
    progress_target = batch["stage_progress_target"].to(device=device, dtype=torch.float32)
    progress_weight = batch["stage_progress_weight"].to(device=device, dtype=torch.float32)
    transition_target = batch["stage_transition_target"].to(device=device, dtype=torch.long)
    transition_weight = batch["stage_transition_weight"].to(device=device, dtype=torch.float32)
    relation_target = batch["stage_relation_target"].to(device=device, dtype=torch.float32)
    relation_weight = batch["stage_relation_weight"].to(device=device, dtype=torch.float32)

    losses = {
        "phase": weighted_soft_cross_entropy(outputs["phase_logits"], phase_target, phase_weight),
        "event": weighted_mean(
            F.binary_cross_entropy_with_logits(outputs["event_logits"], event_target, reduction="none"),
            event_weight,
        ),
        "progress": weighted_mean(
            F.smooth_l1_loss(outputs["progress"], progress_target, reduction="none"),
            progress_weight,
        ),
        "transition": weighted_mean(
            F.cross_entropy(outputs["transition_logits"], transition_target, reduction="none"),
            transition_weight,
        ),
        "relation": weighted_mean(
            F.smooth_l1_loss(outputs["relations"], relation_target, reduction="none"),
            relation_weight,
        ),
    }
    with torch.no_grad():
        phase_accuracy = (
            outputs["phase_logits"].argmax(dim=-1) == phase_target.argmax(dim=-1)
        ).to(dtype=torch.float32)
        transition_accuracy = (
            outputs["transition_logits"].argmax(dim=-1) == transition_target
        ).to(dtype=torch.float32)
        logs = {
            "stage_phase_accuracy": float(weighted_mean(phase_accuracy, phase_weight).cpu()),
            "stage_transition_accuracy": float(weighted_mean(transition_accuracy, transition_weight).cpu()),
            "stage_phase_confidence": float(phase_weight.mean().cpu()),
            "stage_uncertain_target_ratio": float(
                (transition_target == TRANSITION_NAMES.index("uncertain")).float().mean().cpu()
            ),
        }
    return losses, logs
