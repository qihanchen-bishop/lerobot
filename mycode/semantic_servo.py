"""Structured semantic prediction and control utilities.

The learned components in this module are usable before CLF calibration.  The
CLF projector only becomes a certified filter after its dynamics, observation
error bounds, and operating domain have been calibrated independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SEMANTIC_CONTROL_CLASSES = ("cloth", "object", "goal", "actuator")


class PhaseSemanticFeatureExtractor(nn.Module):
    """Extract phase features with fixed per-view/class reliability gates.

    The output contains 12 values per view: four area ratios, normalized
    centroids for object/goal/actuator, object-goal distance, and
    actuator-object distance. Reliability gates are applied to features that
    depend on a class, allowing one unreliable view/class pair to be reduced
    without suppressing the complete view.
    """

    def __init__(
        self,
        reliability: Tensor,
        class_names: Sequence[str] = SEMANTIC_CONTROL_CLASSES,
        *,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        required = set(SEMANTIC_CONTROL_CLASSES)
        missing = required.difference(self.class_names)
        if missing:
            raise ValueError(f"Missing phase semantic classes: {sorted(missing)}")
        if reliability.ndim != 2 or reliability.shape[1] != len(self.class_names):
            raise ValueError(
                "reliability must have shape (views, classes), got "
                f"{tuple(reliability.shape)} for {len(self.class_names)} classes."
            )
        if torch.any((reliability < 0) | (reliability > 1)):
            raise ValueError("reliability values must be in [0, 1].")
        self.class_indices = {name: self.class_names.index(name) for name in required}
        self.register_buffer("reliability", reliability.to(dtype=torch.float32))
        self.eps = eps

    @property
    def features_per_view(self) -> int:
        return 12

    @property
    def output_dim(self) -> int:
        return self.reliability.shape[0] * self.features_per_view

    def feature_names(self, view_names: Sequence[str]) -> tuple[str, ...]:
        if len(view_names) != self.reliability.shape[0]:
            raise ValueError(
                f"Expected {self.reliability.shape[0]} view names, got {len(view_names)}."
            )
        names = []
        for view in view_names:
            prefix = f"{view}/"
            names.extend(prefix + f"area_{name}" for name in SEMANTIC_CONTROL_CLASSES)
            names.extend(
                prefix + f"{axis}_{name}"
                for name in ("object", "goal", "actuator")
                for axis in ("x", "y")
            )
            names.extend((prefix + "distance_object_goal", prefix + "distance_actuator_object"))
        return tuple(names)

    def forward(self, probabilities: Tensor) -> Tensor:
        if probabilities.ndim != 5:
            raise ValueError(
                "Expected phase probabilities (batch, views, classes, height, width), "
                f"got {tuple(probabilities.shape)}."
            )
        if probabilities.shape[1:3] != self.reliability.shape:
            raise ValueError(
                f"Expected {tuple(self.reliability.shape)} view/classes, got "
                f"{tuple(probabilities.shape[1:3])}."
            )

        probabilities = probabilities.clamp(0.0, 1.0)
        _, _, _, height, width = probabilities.shape
        x = torch.linspace(0.0, 1.0, width, device=probabilities.device, dtype=probabilities.dtype)
        y = torch.linspace(0.0, 1.0, height, device=probabilities.device, dtype=probabilities.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        gates = self.reliability.to(device=probabilities.device, dtype=probabilities.dtype)

        areas = probabilities.mean(dim=(-2, -1)) * gates.unsqueeze(0)
        centroids: dict[str, tuple[Tensor, Tensor]] = {}
        gated_centroid_features = []
        for name in ("object", "goal", "actuator"):
            class_idx = self.class_indices[name]
            semantic_map = probabilities[:, :, class_idx]
            mass = semantic_map.sum(dim=(-2, -1)).clamp_min(self.eps)
            centroid_x = (semantic_map * xx).sum(dim=(-2, -1)) / mass
            centroid_y = (semantic_map * yy).sum(dim=(-2, -1)) / mass
            centroids[name] = (centroid_x, centroid_y)
            gate = gates[:, class_idx].unsqueeze(0)
            gated_centroid_features.extend((centroid_x * gate, centroid_y * gate))

        def gated_distance(first: str, second: str) -> Tensor:
            first_x, first_y = centroids[first]
            second_x, second_y = centroids[second]
            distance = torch.sqrt((first_x - second_x).square() + (first_y - second_y).square())
            gate = torch.minimum(
                gates[:, self.class_indices[first]],
                gates[:, self.class_indices[second]],
            ).unsqueeze(0)
            return distance / math.sqrt(2) * gate

        centroid_features = torch.stack(gated_centroid_features, dim=-1)
        relation_features = torch.stack(
            [gated_distance("object", "goal"), gated_distance("actuator", "object")],
            dim=-1,
        )
        per_view = torch.cat([areas, centroid_features, relation_features], dim=-1)
        return per_view.flatten(start_dim=1)


class SoftSemanticStateExtractor(nn.Module):
    """Convert per-view soft class probabilities into normalized semantic metrics.

    Input probabilities have shape ``(batch, views, classes, height, width)``.
    Coordinates and distances are normalized to [0, 1], so the result does not
    depend on image resolution.  Metrics remain view-specific because combining
    camera coordinates requires a calibrated homography or camera extrinsics.
    """

    def __init__(
        self,
        class_names: Sequence[str] = SEMANTIC_CONTROL_CLASSES,
        *,
        eps: float = 1e-6,
        include_confidence: bool = True,
        contact_radius_ratio: float = 0.03,
    ) -> None:
        super().__init__()
        self.class_names = tuple(class_names)
        required = set(SEMANTIC_CONTROL_CLASSES)
        missing = required.difference(self.class_names)
        if missing:
            raise ValueError(f"Missing semantic control classes: {sorted(missing)}")
        self.class_indices = {name: self.class_names.index(name) for name in required}
        self.eps = eps
        self.include_confidence = include_confidence
        if contact_radius_ratio < 0:
            raise ValueError("contact_radius_ratio must be non-negative.")
        self.contact_radius_ratio = contact_radius_ratio

    @property
    def features_per_view(self) -> int:
        return len(self.feature_names_for_view("view"))

    def feature_names_for_view(self, view_name: str) -> tuple[str, ...]:
        prefix = f"{view_name}/"
        area_names = tuple(prefix + f"area_{name}" for name in SEMANTIC_CONTROL_CLASSES)
        centroid_names = tuple(
            prefix + f"{axis}_{name}"
            for name in ("object", "goal", "actuator")
            for axis in ("x", "y")
        )
        names = (
            *area_names,
            *centroid_names,
            prefix + "contact_object_goal",
            prefix + "contact_object_cloth",
            prefix + "distance_object_goal",
            prefix + "distance_actuator_object",
        )
        if self.include_confidence:
            names += (prefix + "segmentation_confidence",)
        return names

    def feature_names(self, view_names: Sequence[str]) -> tuple[str, ...]:
        return tuple(name for view in view_names for name in self.feature_names_for_view(view))

    def forward(self, probabilities: Tensor) -> Tensor:
        if probabilities.ndim != 5:
            raise ValueError(
                "Expected semantic probabilities with shape (batch, views, classes, height, width), "
                f"got {tuple(probabilities.shape)}."
            )
        if probabilities.shape[2] != len(self.class_names):
            raise ValueError(
                f"Expected {len(self.class_names)} classes {self.class_names}, "
                f"got {probabilities.shape[2]}."
            )

        probabilities = probabilities.clamp(0.0, 1.0)
        height, width = probabilities.shape[-2:]
        x = torch.linspace(0.0, 1.0, width, device=probabilities.device, dtype=probabilities.dtype)
        y = torch.linspace(0.0, 1.0, height, device=probabilities.device, dtype=probabilities.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        areas = probabilities.mean(dim=(-2, -1))
        selected = {
            name: probabilities[:, :, self.class_indices[name]]
            for name in SEMANTIC_CONTROL_CLASSES
        }

        centroids: dict[str, tuple[Tensor, Tensor]] = {}
        for name in ("object", "goal", "actuator"):
            semantic_map = selected[name]
            mass = semantic_map.sum(dim=(-2, -1)).clamp_min(self.eps)
            centroid_x = (semantic_map * xx).sum(dim=(-2, -1)) / mass
            centroid_y = (semantic_map * yy).sum(dim=(-2, -1)) / mass
            centroids[name] = (centroid_x, centroid_y)

        object_mass = selected["object"].sum(dim=(-2, -1)).clamp_min(self.eps)

        # Current labels are mutually exclusive. A direct product would therefore
        # be zero for hard targets, so use a local occupancy/contact proxy. Truly
        # amodal independent masks can set contact_radius_ratio=0.
        radius = round(min(height, width) * self.contact_radius_ratio)
        kernel_size = 2 * radius + 1

        def soft_contact(other: Tensor) -> Tensor:
            if radius > 0:
                batch, views = other.shape[:2]
                other = F.max_pool2d(
                    other.reshape(batch * views, 1, height, width),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                ).reshape(batch, views, height, width)
            return (selected["object"] * other).sum(dim=(-2, -1)) / object_mass

        contact_object_goal = soft_contact(selected["goal"])
        contact_object_cloth = soft_contact(selected["cloth"])

        def centroid_distance(first: str, second: str) -> Tensor:
            first_x, first_y = centroids[first]
            second_x, second_y = centroids[second]
            # Dividing by sqrt(2) maps the image-diagonal distance to [0, 1].
            return torch.sqrt((first_x - second_x).square() + (first_y - second_y).square()) / math.sqrt(2)

        background = (1.0 - probabilities.sum(dim=2, keepdim=True)).clamp(0.0, 1.0)
        distribution = torch.cat([background, probabilities], dim=2)
        entropy = -(distribution.clamp_min(self.eps) * distribution.clamp_min(self.eps).log()).sum(dim=2)
        confidence = 1.0 - entropy.mean(dim=(-2, -1)) / math.log(len(self.class_names) + 1)

        centroid_features = torch.stack(
            [coordinate for name in ("object", "goal", "actuator") for coordinate in centroids[name]],
            dim=-1,
        )
        relation_values = [
            contact_object_goal,
            contact_object_cloth,
            centroid_distance("object", "goal"),
            centroid_distance("actuator", "object"),
        ]
        if self.include_confidence:
            relation_values.append(confidence)
        relation_features = torch.stack(relation_values, dim=-1)
        per_view = torch.cat([areas, centroid_features, relation_features], dim=-1)
        return per_view.flatten(start_dim=1)


class ActionConditionedSemanticDynamics(nn.Module):
    """Predict future semantic states from current state, robot state, and actions."""

    def __init__(
        self,
        semantic_dim: int,
        robot_state_dim: int,
        action_dim: int,
        prediction_offsets: Sequence[int],
        *,
        hidden_dim: int = 256,
        min_log_std: float = -5.0,
        max_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        offsets = tuple(int(offset) for offset in prediction_offsets)
        if not offsets or any(offset <= 0 for offset in offsets) or tuple(sorted(set(offsets))) != offsets:
            raise ValueError("prediction_offsets must be unique, increasing positive action-step offsets.")
        self.semantic_dim = semantic_dim
        self.robot_state_dim = robot_state_dim
        self.action_dim = action_dim
        self.prediction_offsets = offsets
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.context = nn.Sequential(
            nn.Linear(semantic_dim + robot_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_input = nn.Linear(action_dim + semantic_dim, hidden_dim)
        self.rollout = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.delta_head = nn.Linear(hidden_dim, semantic_dim)
        self.log_std_head = nn.Linear(hidden_dim, semantic_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, math.log(0.1))

    def forward(self, semantic_state: Tensor, robot_state: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        if semantic_state.ndim != 2 or semantic_state.shape[-1] != self.semantic_dim:
            raise ValueError(
                f"Expected semantic_state (B, {self.semantic_dim}), got {tuple(semantic_state.shape)}."
            )
        if robot_state.ndim != 2 or robot_state.shape[-1] != self.robot_state_dim:
            raise ValueError(
                f"Expected robot_state (B, {self.robot_state_dim}), got {tuple(robot_state.shape)}."
            )
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim:
            raise ValueError(f"Expected actions (B, H, {self.action_dim}), got {tuple(actions.shape)}.")
        if actions.shape[1] < self.prediction_offsets[-1]:
            raise ValueError(
                f"Action horizon {actions.shape[1]} is shorter than prediction offset "
                f"{self.prediction_offsets[-1]}."
            )

        initial_hidden = self.context(torch.cat([semantic_state, robot_state], dim=-1)).unsqueeze(0)
        repeated_state = semantic_state.unsqueeze(1).expand(-1, actions.shape[1], -1)
        rollout_input = F.silu(self.action_input(torch.cat([actions, repeated_state], dim=-1)))
        hidden, _ = self.rollout(rollout_input, initial_hidden)
        indices = torch.as_tensor(
            [offset - 1 for offset in self.prediction_offsets],
            device=actions.device,
            dtype=torch.long,
        )
        selected_hidden = hidden.index_select(1, indices)
        mean = semantic_state.unsqueeze(1) + self.delta_head(selected_hidden)
        log_std = self.log_std_head(selected_hidden).clamp(self.min_log_std, self.max_log_std)
        return mean, log_std

    @staticmethod
    def gaussian_nll(
        mean: Tensor,
        log_std: Tensor,
        target: Tensor,
        valid_steps: Tensor | None = None,
    ) -> Tensor:
        if mean.shape != log_std.shape or mean.shape != target.shape:
            raise ValueError(
                f"mean, log_std, and target must have identical shapes; got "
                f"{tuple(mean.shape)}, {tuple(log_std.shape)}, and {tuple(target.shape)}."
            )
        loss = 0.5 * (
            (target - mean).square() * torch.exp(-2.0 * log_std)
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        if valid_steps is None:
            return loss.mean()
        if valid_steps.shape != mean.shape[:2]:
            raise ValueError(f"Expected valid_steps shape {mean.shape[:2]}, got {tuple(valid_steps.shape)}.")
        weights = valid_steps.to(dtype=loss.dtype).unsqueeze(-1)
        return (loss * weights).sum() / (weights.sum() * mean.shape[-1]).clamp_min(1.0)


class PhaseHistoryModel(nn.Module):
    """Five-phase classifier over semantic, robot-state, and previous-action history."""

    def __init__(
        self,
        semantic_dim: int,
        robot_state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(semantic_dim + robot_state_dim + action_dim, hidden_dim)
        self.history = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 5)

    def forward(self, semantic_history: Tensor, robot_history: Tensor, action_history: Tensor) -> Tensor:
        if (
            semantic_history.shape[:2] != robot_history.shape[:2]
            or semantic_history.shape[:2] != action_history.shape[:2]
        ):
            raise ValueError("Phase histories must have matching batch and time dimensions.")
        inputs = torch.cat([semantic_history, robot_history, action_history], dim=-1)
        hidden, _ = self.history(F.silu(self.input_proj(inputs)))
        return self.classifier(hidden[:, -1])


def hazard_nll(hazard_logits: Tensor, event_steps: Tensor, valid_steps: Tensor | None = None) -> Tensor:
    """Discrete survival loss for one-based event steps."""

    if hazard_logits.ndim != 2:
        raise ValueError(f"Expected hazard_logits (B, H), got {tuple(hazard_logits.shape)}.")
    batch_size, horizon = hazard_logits.shape
    if event_steps.shape != (batch_size,):
        raise ValueError(f"Expected event_steps ({batch_size},), got {tuple(event_steps.shape)}.")
    if torch.any((event_steps < 1) | (event_steps > horizon)):
        raise ValueError(f"event_steps must be between 1 and {horizon}.")
    positions = torch.arange(1, horizon + 1, device=hazard_logits.device).unsqueeze(0)
    at_risk = positions < event_steps.unsqueeze(1)
    event = positions == event_steps.unsqueeze(1)
    mask = at_risk | event
    if valid_steps is not None:
        if valid_steps.shape != hazard_logits.shape:
            raise ValueError("valid_steps must match hazard_logits.")
        mask &= valid_steps
    targets = event.to(dtype=hazard_logits.dtype)
    losses = F.binary_cross_entropy_with_logits(hazard_logits, targets, reduction="none")
    return (losses * mask).sum() / mask.sum().clamp_min(1)


@dataclass(frozen=True)
class CLFProjectionResult:
    action: Tensor
    slack: float
    constraint_before: float
    constraint_after: float
    active: bool


class BoxConstrainedCLFProjector:
    """Solve a diagonal-weight CLF-QP with box limits and one soft CLF constraint.

    This exact one-constraint solver covers joint increment/rate boxes.  General
    coupled collision or CBF constraints require a full QP solver.
    """

    def __init__(self, slack_penalty: float = 1_000.0, bisection_steps: int = 80) -> None:
        if slack_penalty <= 0:
            raise ValueError("slack_penalty must be positive.")
        self.slack_penalty = slack_penalty
        self.bisection_steps = bisection_steps

    @torch.no_grad()
    def project(
        self,
        proposed_action: Tensor,
        *,
        lie_f: Tensor | float,
        lie_g: Tensor,
        clf_value: Tensor | float,
        decay_rate: float,
        lower: Tensor,
        upper: Tensor,
        dynamics_margin: Tensor | float = 0.0,
        observation_margin: Tensor | float = 0.0,
        weights: Tensor | None = None,
    ) -> CLFProjectionResult:
        tensors = (proposed_action, lie_g, lower, upper)
        if any(tensor.ndim != 1 for tensor in tensors):
            raise ValueError("proposed_action, lie_g, lower, and upper must be one-dimensional.")
        if not all(tensor.shape == proposed_action.shape for tensor in tensors):
            raise ValueError("CLF-QP action tensors must have identical shapes.")
        if torch.any(lower > upper):
            raise ValueError("CLF-QP lower bounds must not exceed upper bounds.")
        if decay_rate < 0:
            raise ValueError("decay_rate must be non-negative.")

        device, dtype = proposed_action.device, proposed_action.dtype
        lie_f_t = torch.as_tensor(lie_f, device=device, dtype=dtype)
        value_t = torch.as_tensor(clf_value, device=device, dtype=dtype)
        margin = torch.as_tensor(dynamics_margin, device=device, dtype=dtype) + torch.as_tensor(
            observation_margin, device=device, dtype=dtype
        )
        rhs = -decay_rate * value_t - lie_f_t - margin
        weights = (
            torch.ones_like(proposed_action)
            if weights is None
            else weights.to(device=device, dtype=dtype)
        )
        if weights.shape != proposed_action.shape or torch.any(weights <= 0):
            raise ValueError("CLF-QP weights must be positive and match the action shape.")

        nominal = proposed_action.clamp(lower, upper)
        violation = torch.dot(lie_g, nominal) - rhs
        if float(violation) <= 0.0:
            return CLFProjectionResult(nominal, 0.0, float(violation), float(violation), False)

        def residual(multiplier: Tensor) -> Tensor:
            action = (proposed_action - multiplier * lie_g / weights).clamp(lower, upper)
            slack = multiplier / (2.0 * self.slack_penalty)
            return torch.dot(lie_g, action) - rhs - slack

        low = torch.zeros((), device=device, dtype=dtype)
        high = torch.ones((), device=device, dtype=dtype)
        for _ in range(80):
            if float(residual(high)) <= 0.0:
                break
            high *= 2.0
        else:
            raise RuntimeError("Could not bracket the CLF-QP dual solution.")

        for _ in range(self.bisection_steps):
            middle = (low + high) / 2.0
            if float(residual(middle)) > 0.0:
                low = middle
            else:
                high = middle
        multiplier = high
        action = (proposed_action - multiplier * lie_g / weights).clamp(lower, upper)
        slack = multiplier / (2.0 * self.slack_penalty)
        after = torch.dot(lie_g, action) - rhs - slack
        return CLFProjectionResult(action, float(slack), float(violation), float(after), True)


@dataclass(frozen=True)
class EventTriggerDecision:
    replan: bool
    reasons: tuple[str, ...]


class SemanticEventTrigger:
    """Deployment monitor for semantic confidence, innovation, and CLF progress."""

    def __init__(
        self,
        minimum_confidence: float = 0.5,
        maximum_innovation: float = 3.0,
        clf_tolerance: float = 0.0,
    ):
        self.minimum_confidence = minimum_confidence
        self.maximum_innovation = maximum_innovation
        self.clf_tolerance = clf_tolerance

    def evaluate(
        self,
        *,
        previous_phase: int,
        current_phase: int,
        segmentation_confidence: float,
        normalized_innovation: float,
        clf_value: float,
        certified_clf_upper_bound: float,
    ) -> EventTriggerDecision:
        reasons = []
        if current_phase != previous_phase:
            reasons.append("phase_changed")
        if segmentation_confidence < self.minimum_confidence:
            reasons.append("low_segmentation_confidence")
        if normalized_innovation > self.maximum_innovation:
            reasons.append("semantic_prediction_outlier")
        if clf_value > certified_clf_upper_bound + self.clf_tolerance:
            reasons.append("clf_progress_violation")
        return EventTriggerDecision(bool(reasons), tuple(reasons))


SSACT_PHASE_NAMES = ("uncover", "expose", "transport", "restore", "done")


def phase_semantic_clf(semantic_state: Tensor, phase_index: int) -> Tensor:
    """Return an interpretable phase objective from the 14 metrics per view.

    The thresholds are conservative runtime defaults expressed in normalized
    image coordinates. They are not a substitute for held-out calibration.
    """
    if semantic_state.ndim != 2 or semantic_state.shape[-1] % 14:
        raise ValueError(
            "semantic_state must have shape (batch, views * 14), got "
            f"{tuple(semantic_state.shape)}."
        )
    if not 0 <= phase_index < len(SSACT_PHASE_NAMES):
        raise ValueError(f"phase_index must be in [0, 4], got {phase_index}.")

    per_view = semantic_state.reshape(semantic_state.shape[0], -1, 14)
    cloth = per_view[..., 0]
    object_area = per_view[..., 1]
    object_goal_contact = per_view[..., 10]
    object_cloth_contact = per_view[..., 11]
    object_goal_distance = per_view[..., 12]
    actuator_object_distance = per_view[..., 13]

    # The front view is the primary task view. Side remains useful for object
    # and goal geometry but is downweighted because its tool mask is less stable.
    view_weights = semantic_state.new_ones(per_view.shape[1])
    if per_view.shape[1] > 1:
        view_weights[1:] = 0.65
    view_weights = view_weights / view_weights.sum()

    def weighted(values: Tensor) -> Tensor:
        return (values * view_weights.unsqueeze(0)).sum(dim=1)

    object_visible_error = torch.relu(0.002 - object_area) / 0.002
    if phase_index == 0:  # uncover: reduce cloth and make the object observable
        per_view_value = torch.relu(cloth - 0.48).square() + 0.25 * object_visible_error.square()
    elif phase_index == 1:  # expose: enlarge object visibility and separate it from cloth
        per_view_value = (
            object_visible_error.square()
            + 0.50 * object_cloth_contact.square()
            + 0.15 * actuator_object_distance.square()
        )
    elif phase_index == 2:  # transport: move object to the goal
        per_view_value = object_goal_distance.square() + 0.20 * (1.0 - object_goal_contact).square()
    elif phase_index == 3:  # restore: preserve placement while restoring cloth coverage
        per_view_value = torch.relu(0.55 - cloth).square() + 0.35 * object_goal_distance.square()
    else:  # done: absorbing target set monitor
        per_view_value = torch.relu(0.55 - cloth).square() + 0.50 * object_goal_distance.square()
    return weighted(per_view_value)


@dataclass(frozen=True)
class SSACTRuntimeConfig:
    mode: str = "active"
    adaptive_horizon: bool = True
    minimum_execution_steps: int = 1
    maximum_execution_steps: int = 4
    maximum_action_residual: float = 0.015
    clf_decay_rate: float = 0.03
    phase_confidence_threshold: float = 0.55
    uncertainty_threshold: float = 0.12
    innovation_threshold: float = 3.0
    slack_penalty: float = 10_000.0

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError("SSACT runtime mode must be off, shadow, or active.")
        if self.minimum_execution_steps < 1:
            raise ValueError("minimum_execution_steps must be positive.")
        if self.maximum_execution_steps < self.minimum_execution_steps:
            raise ValueError("maximum_execution_steps must not be smaller than the minimum.")
        if self.maximum_action_residual < 0:
            raise ValueError("maximum_action_residual must be non-negative.")
        if not 0 <= self.phase_confidence_threshold <= 1:
            raise ValueError("phase_confidence_threshold must be in [0, 1].")


class SSACTRuntimeController:
    """Runtime semantic servo and adaptive scheduler for an SSACT-1 checkpoint.

    The phase classifier and semantic dynamics are learned. The horizon rule and
    CLF targets are runtime control logic. Until independent residual bounds are
    supplied, the QP is an uncalibrated stability filter and must not be reported
    as a certified CLF controller.
    """

    def __init__(self, config: SSACTRuntimeConfig) -> None:
        self.config = config
        self.projector = BoxConstrainedCLFProjector(slack_penalty=config.slack_penalty)
        self.reset()

    def reset(self) -> None:
        self.previous_phase: int | None = None
        self.expected_state: Tensor | None = None
        self.expected_std: Tensor | None = None
        self.expected_clf: float | None = None
        self.latest_report: dict[str, Any] = {}

    @staticmethod
    def _phase_default_steps(phase_index: int) -> int:
        return (8, 4, 4, 8, 1)[phase_index]

    @staticmethod
    def _rollout_at_step(
        values: Tensor,
        prediction_offsets: Sequence[int],
        execution_steps: int,
    ) -> Tensor:
        """Interpolate a predicted semantic quantity at an execution boundary."""
        offsets = tuple(int(offset) for offset in prediction_offsets)
        if values.ndim != 3 or values.shape[1] != len(offsets):
            raise ValueError("Rollout values and prediction offsets do not match.")
        if execution_steps <= offsets[0]:
            return values[:, 0]
        if execution_steps >= offsets[-1]:
            return values[:, -1]
        for upper_index, upper_step in enumerate(offsets[1:], start=1):
            if execution_steps <= upper_step:
                lower_index = upper_index - 1
                lower_step = offsets[lower_index]
                weight = (execution_steps - lower_step) / (upper_step - lower_step)
                return torch.lerp(values[:, lower_index], values[:, upper_index], weight)
        raise RuntimeError("Unable to interpolate semantic rollout.")

    def _adaptive_steps(
        self,
        *,
        phase_index: int,
        phase_confidence: float,
        uncertainty: float,
        innovation: float,
        phase_changed: bool,
        nominal_progress: float,
    ) -> tuple[int, list[str]]:
        cfg = self.config
        steps = min(max(self._phase_default_steps(phase_index), cfg.minimum_execution_steps), cfg.maximum_execution_steps)
        reasons = [f"phase_default:{SSACT_PHASE_NAMES[phase_index]}"]
        if not cfg.adaptive_horizon:
            return cfg.maximum_execution_steps, ["adaptive_disabled"]
        if phase_changed:
            steps = cfg.minimum_execution_steps
            reasons.append("phase_changed")
        if phase_confidence < cfg.phase_confidence_threshold:
            steps = cfg.minimum_execution_steps
            reasons.append("low_phase_confidence")
        if innovation > cfg.innovation_threshold:
            steps = cfg.minimum_execution_steps
            reasons.append("semantic_prediction_outlier")
        if uncertainty > cfg.uncertainty_threshold:
            steps = min(steps, max(cfg.minimum_execution_steps, 2))
            reasons.append("high_dynamics_uncertainty")
        if nominal_progress >= 0:
            steps = min(steps, max(cfg.minimum_execution_steps, 2))
            reasons.append("nominal_clf_not_decreasing")
        return steps, reasons

    def apply(
        self,
        *,
        semantic_dynamics: nn.Module,
        semantic_state: Tensor,
        robot_state: Tensor,
        nominal_actions: Tensor,
        phase_probabilities: Tensor,
        prediction_offsets: Sequence[int],
    ) -> tuple[Tensor, dict[str, Any]]:
        if nominal_actions.ndim != 3 or nominal_actions.shape[0] != 1:
            raise ValueError("SSACT runtime currently requires one action chunk with batch size 1.")
        phase_probs = phase_probabilities.detach().float()
        phase_index = int(phase_probs[0].argmax())
        phase_confidence = float(phase_probs[0, phase_index])
        current_state = semantic_state.detach().float()
        current_clf_t = phase_semantic_clf(current_state, phase_index)
        current_clf = float(current_clf_t[0])

        innovation = 0.0
        if self.expected_state is not None and self.expected_std is not None:
            expected_state = self.expected_state.to(current_state)
            expected_std = self.expected_std.to(current_state).clamp_min(0.02)
            innovation = float(((current_state - expected_state).abs() / expected_std).mean())

        phase_changed = self.previous_phase is not None and phase_index != self.previous_phase
        device_type = nominal_actions.device.type
        with (
            torch.enable_grad(),
            torch.autocast(device_type=device_type, enabled=False),
            torch.backends.cudnn.flags(enabled=False),
        ):
            differentiable_actions = nominal_actions.detach().float().requires_grad_(True)
            predicted_means, predicted_log_stds = semantic_dynamics(
                current_state,
                robot_state.detach().float(),
                differentiable_actions,
            )
            predicted_stds = predicted_log_stds.exp()

            base_steps = min(
                max(self._phase_default_steps(phase_index), self.config.minimum_execution_steps),
                self.config.maximum_execution_steps,
            )
            base_mean = self._rollout_at_step(predicted_means, prediction_offsets, base_steps)
            base_std = self._rollout_at_step(predicted_stds, prediction_offsets, base_steps)
            predicted_clf_t = phase_semantic_clf(base_mean, phase_index)
            predicted_clf = float(predicted_clf_t.detach()[0])
            uncertainty = float(base_std.detach().mean())
            nominal_progress = predicted_clf - current_clf

            execution_steps, horizon_reasons = self._adaptive_steps(
                phase_index=phase_index,
                phase_confidence=phase_confidence,
                uncertainty=uncertainty,
                innovation=innovation,
                phase_changed=phase_changed,
                nominal_progress=nominal_progress,
            )
            execution_steps = min(execution_steps, nominal_actions.shape[1])
            selected_mean = self._rollout_at_step(predicted_means, prediction_offsets, execution_steps)
            selected_std = self._rollout_at_step(predicted_stds, prediction_offsets, execution_steps)
            predicted_clf_t = phase_semantic_clf(selected_mean, phase_index)
            predicted_clf = float(predicted_clf_t.detach()[0])
            uncertainty = float(selected_std.detach().mean())
            nominal_progress = predicted_clf - current_clf

            gradient = torch.autograd.grad(
                predicted_clf_t.sum(),
                differentiable_actions,
                retain_graph=False,
                create_graph=False,
            )[0][0, :execution_steps].reshape(-1)

        correction = torch.zeros_like(gradient)
        qp_active = False
        qp_slack = 0.0
        constraint_before = 0.0
        constraint_after = 0.0
        can_correct = (
            self.config.mode != "off"
            and phase_index != 4
            and phase_confidence >= self.config.phase_confidence_threshold
            and torch.isfinite(gradient).all()
            and float(gradient.norm()) > 1e-8
        )
        if can_correct:
            limit = self.config.maximum_action_residual
            result = self.projector.project(
                torch.zeros_like(gradient),
                lie_f=predicted_clf - current_clf,
                lie_g=gradient,
                clf_value=current_clf,
                decay_rate=self.config.clf_decay_rate,
                lower=torch.full_like(gradient, -limit),
                upper=torch.full_like(gradient, limit),
            )
            correction = result.action
            qp_active = result.active
            qp_slack = result.slack
            constraint_before = result.constraint_before
            constraint_after = result.constraint_after

        controlled = nominal_actions.clone()
        if self.config.mode == "active" and correction.numel():
            controlled[:, :execution_steps] += correction.reshape(1, execution_steps, -1).to(controlled)

        selected_mean = selected_mean.detach()
        selected_std = selected_std.detach()
        semantic_delta = selected_mean - current_state
        self.previous_phase = phase_index
        self.expected_state = selected_mean.cpu()
        self.expected_std = selected_std.cpu()
        self.expected_clf = predicted_clf
        report = {
            "phase_index": phase_index,
            "phase": SSACT_PHASE_NAMES[phase_index],
            "phase_confidence": phase_confidence,
            "phase_probabilities": [float(value) for value in phase_probs[0]],
            "phase_changed": phase_changed,
            "clf_value": current_clf,
            "predicted_clf": predicted_clf,
            "nominal_clf_progress": nominal_progress,
            "dynamics_uncertainty": uncertainty,
            "semantic_delta_l2": float(semantic_delta.norm()),
            "semantic_delta_max": float(semantic_delta.abs().max()),
            "normalized_innovation": innovation,
            "execution_steps": execution_steps,
            "horizon_reasons": horizon_reasons,
            "servo_mode": self.config.mode,
            "servo_applied": self.config.mode == "active" and bool(correction.abs().max() > 0),
            "qp_evaluated": can_correct,
            "correction_l2": float(correction.norm()),
            "correction_max": float(correction.abs().max()) if correction.numel() else 0.0,
            "qp_active": qp_active,
            "qp_slack": qp_slack,
            "qp_constraint_before": constraint_before,
            "qp_constraint_after": constraint_after,
            "clf_certified": False,
            "calibration_status": "uncalibrated",
            "learned_phase": True,
            "learned_semantic_dynamics": True,
            "learned_hazard": False,
            "adaptive_horizon_applied": self.config.adaptive_horizon,
            "adaptive_horizon_source": "phase+dynamics runtime rule",
        }
        self.latest_report = report
        return controlled, dict(report)
