"""Causal fixed-window residual control. No robot or transport dependencies.

Actions, state, residual bounds and command limits use dataset-native joint units,
not model-normalized coordinates. A plan's time zero is its observation time.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ResidualConfig:
    variant: str = "act"
    fps: int = 30
    fast_stride: int = 2
    slow_stride: int = 30
    history: int = 8
    hidden: int = 128
    correction_steps: int = 2
    plan_steps: int = 6
    slow_delay: int = 3
    fast_delay: int = 1
    image_height: int = 180
    image_width: int = 320
    max_observation_age: float = 0.15
    max_view_skew: float = 0.04
    history_gap: float | None = None
    residual_max_age: float | None = None
    unit: str = "dataset_native"
    seed: int = 1000

    def __post_init__(self):
        if self.variant not in {"act", "unet", "qtoken"}:
            raise ValueError("Unknown residual variant")
        for name in (
            "fps",
            "fast_stride",
            "slow_stride",
            "history",
            "hidden",
            "correction_steps",
            "plan_steps",
            "image_height",
            "image_width",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.slow_delay < 0 or self.fast_delay < 0:
            raise ValueError("Delays cannot be negative")
        if self.correction_steps < self.fast_stride or self.plan_steps < self.correction_steps:
            raise ValueError("Output horizon must cover the fast update interval")
        for value in (self.history_gap, self.residual_max_age):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Optional age limits must be finite and positive")
        if self.unit != "dataset_native":
            raise ValueError("v1 accepts explicitly dataset-native joint commands only")


def validate_joints(state_names: list[str], action_names: list[str], expected: list[str]) -> None:
    if state_names != expected or action_names != expected or len(set(expected)) != len(expected):
        raise ValueError("State/action joint order mismatch")


def observation_valid(times: list[float], now: float, cfg: ResidualConfig) -> bool:
    return (
        len(times) == 2
        and all(math.isfinite(t) for t in times)
        and all(-1e-6 <= now - t <= cfg.max_observation_age for t in times)
        and max(times) - min(times) <= cfg.max_view_skew
    )


@dataclass
class ActionPlan:
    plan_id: int
    observed_at: float
    ready_at: float
    actions: Tensor
    fps: int = 30

    def __post_init__(self):
        if (
            self.actions.ndim != 2
            or len(self.actions) == 0
            or not torch.isfinite(self.actions).all()
            or not math.isfinite(self.observed_at)
            or not math.isfinite(self.ready_at)
            or self.ready_at < self.observed_at
            or self.fps <= 0
        ):
            raise ValueError("Invalid plan")

    def at(self, target_time: float, count: int) -> tuple[Tensor, Tensor]:
        index = round((target_time - self.observed_at) * self.fps)
        indices = torch.arange(index, index + count, device=self.actions.device)
        valid = (indices >= 0) & (indices < len(self.actions))
        values = self.actions[indices.clamp(0, len(self.actions) - 1)]
        return values * valid[:, None], valid


def context_record(
    q: Tensor,
    previous_q: Tensor,
    previous_action: Tensor,
    plan: ActionPlan,
    observed_at: float,
    dt: float,
    scale: Tensor,
    cfg: ResidualConfig,
    switched: bool = False,
) -> Tensor:
    """Only measured/past state and the then-current plan enter a history record."""
    target = observed_at + cfg.fast_delay / cfg.fps
    suffix, valid = plan.at(target, cfg.plan_steps)
    velocity = (q - previous_q) / max(dt, 1 / cfg.fps)
    age = observed_at - plan.observed_at
    return torch.cat(
        (
            q / scale,
            velocity / scale,
            previous_action / scale,
            (suffix / scale).flatten(),
            ((suffix - q) / scale * valid[:, None]).flatten(),
            valid.float(),
            q.new_tensor([age, dt, float(switched), cfg.fast_delay / cfg.fps]),
        )
    )


class ResidualCorrector(nn.Module):
    """Fixed-window GRU; right padding is explicitly masked, never recurrent memory."""

    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        action_dim: int,
        bounds: Tensor,
        cfg: ResidualConfig,
        mode: str = "gru",
    ):
        super().__init__()
        if mode not in {"gru", "mlp"}:
            raise ValueError("v1 supports GRU and the no-history MLP control")
        if bounds.shape != (action_dim,) or not torch.isfinite(bounds).all() or (bounds <= 0).any():
            raise ValueError("Bounds must be positive per-joint native units")
        self.cfg, self.mode = cfg, mode
        self.spec = {
            "feature_dim": feature_dim,
            "context_dim": context_dim,
            "action_dim": action_dim,
            "cfg": asdict(cfg),
            "mode": mode,
        }
        self.visual_adapter = nn.Sequential(
            nn.Linear(feature_dim, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.SiLU()
        )
        self.input_adapter = nn.Sequential(nn.Linear(cfg.hidden + context_dim, cfg.hidden), nn.SiLU())
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, batch_first=True) if mode == "gru" else None
        self.head = nn.Linear(cfg.hidden, cfg.correction_steps * action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("bounds", bounds.clone())

    def forward(self, features: Tensor, context: Tensor, lengths: Tensor) -> Tensor:
        if features.ndim != 3 or context.shape[:2] != features.shape[:2]:
            raise ValueError("Expected aligned B,W,F feature and context windows")
        if (lengths < 1).any() or (lengths > features.shape[1]).any():
            raise ValueError("Invalid window length")
        x = self.input_adapter(torch.cat((self.visual_adapter(features), context), dim=-1))
        if self.gru is not None:
            x, _ = self.gru(x)
        h = x[torch.arange(len(x), device=x.device), lengths.to(x.device) - 1]
        return self.head(h).reshape(len(h), self.cfg.correction_steps, -1).tanh() * self.bounds


class OnlineWindow:
    def __init__(self, model: ResidualCorrector):
        self.model = model
        self.records: deque = deque(maxlen=model.cfg.history if model.mode == "gru" else 1)
        self.last_time = -math.inf

    def reset(self):
        self.records.clear()
        self.last_time = -math.inf

    @torch.no_grad()
    def update(self, feature: Tensor, context: Tensor, timestamp: float) -> Tensor | None:
        if timestamp <= self.last_time:
            return None  # A repeated image is not a new observation.
        if not torch.isfinite(feature).all() or not torch.isfinite(context).all():
            return None
        if timestamp - self.last_time > (self.model.cfg.history_gap or self.model.cfg.max_observation_age):
            self.records.clear()
        self.last_time = timestamp
        self.records.append((feature, context))
        f, c = zip(*self.records, strict=True)
        return self.model(
            torch.stack(f)[None], torch.stack(c)[None], torch.tensor([len(f)], device=feature.device)
        )[0]


class MultiRateExecutor:
    """Shadow-only scheduler with explicit plan ownership and stale-output rejection.

    It returns commands but has no robot I/O. Call accept_plan / accept_residual
    from completed workers and command at the fixed control clock.
    """

    def __init__(
        self,
        cfg: ResidualConfig,
        bounds: Tensor,
        lower: Tensor | None = None,
        upper: Tensor | None = None,
        max_velocity: Tensor | None = None,
        max_acceleration: Tensor | None = None,
    ):
        self.cfg, self.bounds = cfg, bounds
        self.lower, self.upper = lower, upper
        self.max_velocity, self.max_acceleration = max_velocity, max_acceleration
        self.reset()

    def reset(self):
        self.plan: ActionPlan | None = None
        self.pending: dict[int, Tensor] = {}
        self.pending_times: dict[int, float] = {}
        self.last_command: Tensor | None = None
        self.last_velocity: Tensor | None = None
        self.last_slot = -1
        self.expired_prefix_slots = 0

    def accept_plan(self, plan: ActionPlan, now: float) -> bool:
        if plan.fps != self.cfg.fps or plan.actions.shape[1] != len(self.bounds):
            raise ValueError("Plan clock/action dimension mismatch")
        if plan.ready_at > now or (self.plan is not None and plan.plan_id <= self.plan.plan_id):
            return False
        _, valid = plan.at(now, 1)
        if not bool(valid[0]):
            return False
        self.plan, self.pending = plan, {}
        self.pending_times = {}
        return True

    def accept_residual(
        self, plan_id: int, residual: Tensor, target_time: float, view_times: list[float], now: float
    ) -> bool:
        if (
            self.plan is None
            or plan_id != self.plan.plan_id
            or not observation_valid(view_times, now, self.cfg)
            or residual.shape != (self.cfg.correction_steps, len(self.bounds))
            or not torch.isfinite(residual).all()
        ):
            return False
        start = round(target_time * self.cfg.fps)
        first_available = max(math.ceil(now * self.cfg.fps - 1e-6), self.last_slot + 1)
        skipped = max(0, first_available - start)
        if skipped >= len(residual):
            return False
        self.expired_prefix_slots += skipped
        for j, r in enumerate(residual):
            if j >= skipped:
                self.pending[start + j] = r.clamp(-self.bounds, self.bounds)
                self.pending_times[start + j] = min(view_times)
        return True

    def command(self, now: float, enabled: bool = True) -> Tensor | None:
        slot = round(now * self.cfg.fps)
        if self.plan is None or slot <= self.last_slot:
            return None
        nominal, valid = self.plan.at(now, 1)
        if not bool(valid[0]):
            return None
        command = nominal[0].clone()
        correction = self.pending.pop(slot, None)
        observed_at = self.pending_times.pop(slot, -math.inf)
        for expired in [key for key in self.pending if key < slot]:
            self.pending.pop(expired)
            self.pending_times.pop(expired)
        if (
            enabled
            and correction is not None
            and now - observed_at <= (self.cfg.residual_max_age or self.cfg.max_observation_age)
        ):
            command += correction
        if self.lower is not None:
            command = torch.maximum(command, self.lower)
        if self.upper is not None:
            command = torch.minimum(command, self.upper)
        if self.last_command is not None and (
            self.max_velocity is not None or self.max_acceleration is not None
        ):
            dt = (slot - self.last_slot) / self.cfg.fps
            velocity = (command - self.last_command) / dt
            if self.max_velocity is not None:
                velocity = velocity.clamp(-self.max_velocity, self.max_velocity)
            if self.max_acceleration is not None and self.last_velocity is not None:
                velocity = self.last_velocity + (velocity - self.last_velocity).clamp(
                    -self.max_acceleration * dt, self.max_acceleration * dt
                )
            command = self.last_command + velocity * dt
            self.last_velocity = velocity
        self.last_command, self.last_slot = command.clone(), slot
        return command


class ResidualSequenceDataset(Dataset):
    """Episode-local windows over frozen cached features; no video decoding in training."""

    def __init__(self, episodes: list[dict[str, Any]], cfg: ResidualConfig, history: int | None = None):
        self.episodes, self.cfg = episodes, cfg
        self.history = history or cfg.history
        self.index = [(e, i) for e, ep in enumerate(episodes) for i in range(len(ep["features"]))]
        for ep in episodes:
            n = len(ep["features"])
            if not n or any(len(ep[k]) != n for k in ("context", "base", "target", "valid", "time")):
                raise ValueError("Misaligned cached episode")
            if not torch.all(torch.diff(ep["time"]) > 0):
                raise ValueError("Non-monotonic episode timestamps")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        e, i = self.index[item]
        ep = self.episodes[e]
        start = max(0, i - self.history + 1)
        # Reset history across dropped frames exactly like OnlineWindow.
        gap = self.cfg.history_gap or self.cfg.max_observation_age
        while start < i and ep["time"][start + 1] - ep["time"][start] > gap:
            start += 1
        gaps = torch.where(torch.diff(ep["time"][start : i + 1]) > gap)[0]
        if len(gaps):
            start += int(gaps[-1]) + 1
        length = i - start + 1
        result = {k: ep[k][i].float() for k in ("base", "target", "valid", "age")}
        for key in ("features", "context"):
            values = ep[key][start : i + 1].float()
            result[key] = F.pad(values, (0, 0, 0, self.history - length))
        result["lengths"] = torch.tensor(length)
        return result


def residual_loss(
    prediction: Tensor, batch: dict[str, Tensor], scale: Tensor, magnitude_weight: float = 1e-3
) -> Tensor:
    mask = batch["valid"].unsqueeze(-1)
    loss = F.smooth_l1_loss(
        (batch["base"] + prediction) / scale, batch["target"] / scale, beta=0.1, reduction="none"
    )
    return ((loss + magnitude_weight * (prediction / scale).square()) * mask).sum() / (
        mask.sum().clamp_min(1) * prediction.shape[-1]
    )


def diagnostic_metrics(
    base: Tensor, target: Tensor, residual: Tensor, valid: Tensor, scale: Tensor, bounds: Tensor, ages: Tensor
) -> dict[str, Any]:
    mask = valid.bool()
    b, t, r = base[mask], target[mask], residual[mask]
    if not len(b):
        raise ValueError("Empty evaluation")
    error, corrected = (b - t).abs(), (b + r - t).abs()
    result = {
        "base_mae": error.mean().item(),
        "corrected_mae": corrected.mean().item(),
        "base_normalized_mae": (error / scale).mean().item(),
        "corrected_normalized_mae": (corrected / scale).mean().item(),
        "base_per_joint_mae": error.mean(0).tolist(),
        "corrected_per_joint_mae": corrected.mean(0).tolist(),
        "residual_rms": r.square().mean().sqrt().item(),
        "saturation_fraction": (r.abs() >= bounds * 0.98).float().mean().item(),
        "target_reachable_fraction": ((t - b).abs() <= bounds).float().mean().item(),
        "count": len(b),
        "by_plan_age": {},
    }
    for lo, hi in ((0, 0.33), (0.33, 0.66), (0.66, 2.0)):
        selected = (ages >= lo) & (ages < hi)
        m = mask & selected[:, None]
        if m.any():
            result["by_plan_age"][f"{lo}-{hi}"] = {
                "base_mae": (base - target).abs()[m].mean().item(),
                "corrected_mae": (base + residual - target).abs()[m].mean().item(),
                "count": int(m.sum()),
            }
    return result
