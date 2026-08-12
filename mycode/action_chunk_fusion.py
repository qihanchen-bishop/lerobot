"""Action-chunk replanning and overlap fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


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
