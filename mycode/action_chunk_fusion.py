"""Action-chunk replanning and overlap fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActionMagnitudeReplanScheduler:
    """Choose a chunk prefix whose cumulative normalized joint motion reaches a budget."""

    min_steps: int
    max_steps: int
    movement_budget: float
    action_scales: tuple[float, ...]
    action_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.min_steps < 1 or self.max_steps < self.min_steps:
            raise ValueError("Auto-replan steps require 1 <= min_steps <= max_steps.")
        if self.movement_budget <= 0:
            raise ValueError("Auto-replan movement_budget must be greater than zero.")
        if not self.action_scales or any(scale <= 0 for scale in self.action_scales):
            raise ValueError("Auto-replan action scales must all be greater than zero.")
        if self.action_indices is not None:
            if len(self.action_indices) != len(self.action_scales):
                raise ValueError("Auto-replan action_indices and action_scales must have equal length.")
            if len(set(self.action_indices)) != len(self.action_indices) or min(self.action_indices) < 0:
                raise ValueError("Auto-replan action_indices must be unique and non-negative.")

    def select(self, action_chunk: Tensor) -> dict[str, Any]:
        """Return the selected execution length and inspectable motion statistics."""
        if action_chunk.ndim == 3:
            if action_chunk.shape[0] != 1:
                raise ValueError(
                    f"Only batch size 1 is supported; got shape {tuple(action_chunk.shape)}."
                )
            action_chunk = action_chunk.squeeze(0)
        if action_chunk.ndim != 2:
            raise ValueError(
                f"Expected action chunk shape (steps, action_dim); got {tuple(action_chunk.shape)}."
            )

        available_steps = int(action_chunk.shape[0])
        usable_max = min(self.max_steps, available_steps)
        if usable_max < self.min_steps:
            raise ValueError(
                f"Auto replan needs at least {self.min_steps} predicted actions; got {available_steps}."
            )

        if self.action_indices is None:
            if action_chunk.shape[1] != len(self.action_scales):
                raise ValueError(
                    "Action chunk dimension does not match auto-replan calibration: "
                    f"{action_chunk.shape[1]} != {len(self.action_scales)}."
                )
            selected_actions = action_chunk[:usable_max]
        else:
            if max(self.action_indices) >= action_chunk.shape[1]:
                raise ValueError(
                    "Auto-replan calibration references an unavailable action dimension: "
                    f"max index {max(self.action_indices)}, action dim {action_chunk.shape[1]}."
                )
            selected_actions = action_chunk[:usable_max, list(self.action_indices)]

        scales = selected_actions.new_tensor(self.action_scales).clamp_min(1e-6)
        normalized_delta = torch.diff(selected_actions, dim=0) / scales
        step_motion = torch.sqrt(torch.mean(normalized_delta.square(), dim=-1))
        cumulative_motion = torch.cat(
            (step_motion.new_zeros(1), torch.cumsum(step_motion, dim=0)), dim=0
        )
        eligible = cumulative_motion[self.min_steps - 1 : usable_max]
        crossings = torch.nonzero(eligible >= self.movement_budget, as_tuple=False).flatten()
        selected_steps = (
            self.min_steps + int(crossings[0].item()) if crossings.numel() else usable_max
        )

        cumulative_at_min = float(cumulative_motion[self.min_steps - 1].item())
        cumulative_at_selected = float(cumulative_motion[selected_steps - 1].item())
        cumulative_at_max = float(cumulative_motion[usable_max - 1].item())
        return {
            "selected_steps": selected_steps,
            "minimum_steps": self.min_steps,
            "maximum_steps": usable_max,
            "movement_budget": float(self.movement_budget),
            "cumulative_motion_at_min": cumulative_at_min,
            "cumulative_motion_at_selected": cumulative_at_selected,
            "cumulative_motion_at_max": cumulative_at_max,
            "mean_normalized_step_motion": float(step_motion.mean().item()),
            "clamped_to_minimum": selected_steps == self.min_steps
            and cumulative_at_min >= self.movement_budget,
            "clamped_to_maximum": selected_steps == usable_max
            and cumulative_at_max < self.movement_budget,
        }


def fuse_action_chunks(
    historical_actions: Tensor,
    new_actions: Tensor,
    fusion_steps: int,
    initial_history_weight: float,
) -> Tensor:
    """Blend aligned action chunks with a history weight that decays linearly to zero."""
    if historical_actions.ndim != 2 or new_actions.ndim != 2:
        raise ValueError("Action chunks must have shape (steps, action_dim).")
    if historical_actions.shape[1] != new_actions.shape[1]:
        raise ValueError(
            "Historical and new action chunks must have the same action dimension; "
            f"got {historical_actions.shape[1]} and {new_actions.shape[1]}."
        )
    if fusion_steps < 0:
        raise ValueError("fusion_steps must be non-negative.")
    if not 0.0 <= initial_history_weight <= 1.0:
        raise ValueError("initial_history_weight must be between 0 and 1.")

    fused = new_actions.clone()
    overlap = min(fusion_steps, historical_actions.shape[0], new_actions.shape[0])
    if overlap == 0:
        return fused

    if overlap == 1:
        history_weights = new_actions.new_tensor([initial_history_weight])
    else:
        history_weights = torch.linspace(
            initial_history_weight,
            0.0,
            overlap,
            device=new_actions.device,
            dtype=new_actions.dtype,
        )
    history_weights = history_weights.unsqueeze(-1)
    history = historical_actions[:overlap].to(device=new_actions.device, dtype=new_actions.dtype)
    fused[:overlap] = history * history_weights + fused[:overlap] * (1.0 - history_weights)
    return fused


@dataclass
class ActionChunkFusionPlanner:
    """Execute a fixed number of actions, then replan and blend with the old chunk remainder."""

    replan_steps: int
    fusion_steps: int = 0
    initial_history_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.replan_steps <= 0:
            raise ValueError("replan_steps must be greater than zero.")
        if self.fusion_steps < 0:
            raise ValueError("fusion_steps must be non-negative.")
        if not 0.0 <= self.initial_history_weight <= 1.0:
            raise ValueError("initial_history_weight must be between 0 and 1.")
        self._chunk: Tensor | None = None
        self._cursor = 0

    @property
    def needs_replan(self) -> bool:
        return self._chunk is None or self._cursor >= min(self.replan_steps, self._chunk.shape[0])

    @property
    def remaining_history_steps(self) -> int:
        if self._chunk is None:
            return 0
        return max(self._chunk.shape[0] - self._cursor, 0)

    def set_replan_steps(self, replan_steps: int) -> None:
        """Change the next chunk prefix length selected by an adaptive scheduler."""
        if replan_steps <= 0:
            raise ValueError("replan_steps must be greater than zero.")
        self.replan_steps = int(replan_steps)

    def update(self, new_chunk: Tensor) -> int:
        """Install a new chunk and return the number of aligned steps that were fused."""
        if new_chunk.ndim == 3:
            if new_chunk.shape[0] != 1:
                raise ValueError(f"Only batch size 1 is supported; got shape {tuple(new_chunk.shape)}.")
            new_chunk = new_chunk.squeeze(0)
        if new_chunk.ndim != 2 or new_chunk.shape[0] == 0:
            raise ValueError(f"Expected a non-empty (steps, action_dim) chunk; got {tuple(new_chunk.shape)}.")

        historical_actions = None if self._chunk is None else self._chunk[self._cursor :]
        overlap = 0
        if historical_actions is not None:
            overlap = min(self.fusion_steps, historical_actions.shape[0], new_chunk.shape[0])
            new_chunk = fuse_action_chunks(
                historical_actions,
                new_chunk,
                fusion_steps=self.fusion_steps,
                initial_history_weight=self.initial_history_weight,
            )
        else:
            new_chunk = new_chunk.clone()

        self._chunk = new_chunk
        self._cursor = 0
        return overlap

    def pop_action(self) -> Tensor:
        if self._chunk is None:
            raise RuntimeError("No action chunk is available; call update() first.")
        if self._cursor >= self._chunk.shape[0]:
            raise RuntimeError("The current action chunk is exhausted.")
        action = self._chunk[self._cursor]
        self._cursor += 1
        return action
