"""Frozen semantic and query-context descriptors for the shared residual core.

Missing/invalid auxiliary inputs produce zeros PLUS explicit validity flags.
Low confidence is not treated as a valid zero geometric distance.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def semantic_descriptor(
    probabilities: Tensor | None, classes: list[str], reference: Tensor, confidence_threshold: float = 0.6
) -> Tensor:
    """Spatial soft maps + per-class area/centroid/confidence/visibility.

    classes excludes background. Centers are image-normalized, not 3D positions.
    Pair geometry refers to centroids, deliberately separate from QToken's tool endpoint.
    """
    channels = len(classes) + 1
    dimension = channels * 6 + len(classes) * 5 + 8
    empty = reference.new_zeros((reference.shape[0], dimension))
    if probabilities is None:
        return empty
    if probabilities.ndim != 4 or probabilities.shape[:2] != (len(reference), channels):
        raise ValueError("Semantic class layout/batch mismatch")
    finite = torch.isfinite(probabilities).flatten(1).all(1)
    p = torch.nan_to_num(probabilities).clamp(0, 1)
    normalized = (p.sum(1) - 1).abs().flatten(1).amax(1) < 1e-3
    reliable = finite & normalized
    labels = p.argmax(1)
    b, _, height, width = p.shape
    y, x = torch.meshgrid(
        torch.linspace(0, 1, height, device=p.device),
        torch.linspace(0, 1, width, device=p.device),
        indexing="ij",
    )
    # Batch class reductions to avoid dozens of small per-class GPU launches.
    membership = labels[:, None] == torch.arange(1, channels, device=p.device)[None, :, None, None]
    area = membership.float().mean((-2, -1))
    weights = p[:, 1:] * membership
    total = weights.sum((-2, -1))
    confidence = total / membership.sum((-2, -1)).clamp_min(1)
    visible = (area >= 1e-4) & (confidence >= confidence_threshold) & reliable[:, None]
    mass = total.clamp_min(1e-8)
    center = torch.stack(((weights * x).sum((-2, -1)) / mass, (weights * y).sum((-2, -1)) / mass), -1)
    entries = torch.cat(
        (area[..., None], center * visible[..., None], confidence[..., None], visible[..., None]), -1
    ).flatten(1)
    centers = {name: center[:, i] for i, name in enumerate(classes)}
    visibility = {name: visible[:, i] for i, name in enumerate(classes)}
    pairs = []
    for first, second in (("object", "region"), ("object", "tool")):
        if first in centers and second in centers:
            valid = visibility[first] & visibility[second]
            delta = (centers[first] - centers[second]) * valid[:, None]
            pairs.append(torch.cat((delta, delta.norm(dim=-1, keepdim=True), valid[:, None]), -1))
        else:
            pairs.append(p.new_zeros((b, 4)))
    result = torch.cat((F.adaptive_avg_pool2d(p, (2, 3)).flatten(1), entries, *pairs), -1)
    return torch.where(reliable[:, None], result, empty)


def query_descriptor(
    tokens: Tensor | None,
    reference: Tensor,
    count: int = 3,
    dimension: int = 512,
    validity: Tensor | None = None,
) -> Tensor:
    """Fixed-order tokens padded to three slots, each with numeric validity."""
    out = reference.new_zeros((len(reference), count, dimension))
    flags = reference.new_zeros((len(reference), count))
    if tokens is not None:
        if tokens.ndim != 3 or tokens.shape[0] != len(reference) or tokens.shape[-1] != dimension:
            raise ValueError("Query token batch/channel mismatch")
        n = min(count, tokens.shape[1])
        valid = torch.isfinite(tokens[:, :n]).all(-1)
        if validity is not None:
            if validity.shape != tokens.shape[:2]:
                raise ValueError("Query validity shape mismatch")
            valid &= validity[:, :n].bool()
        out[:, :n] = torch.where(valid[..., None], tokens[:, :n], 0)
        flags[:, :n] = valid
    return torch.cat((out.flatten(1), flags), -1)


class QueryCapture:
    """Read original ACT query hidden states at metric-head inputs, without changing forward."""

    def __init__(self, act_model):
        if act_model.config.metric_mode != "encoder_tokens" or act_model.metric_heads is None:
            raise ValueError("Expected a QToken ACT checkpoint with separate metric heads")
        if len(act_model.metric_heads) != 3:
            raise ValueError("Expected three semantic queries")
        self.values: dict[int, Tensor] = {}
        self.handles = [
            head.register_forward_pre_hook(self._hook(index))
            for index, head in enumerate(act_model.metric_heads)
        ]

    def _hook(self, index):
        def capture(module, inputs):
            self.values[index] = inputs[0].detach().clone()

        return capture

    def clear(self):
        self.values.clear()

    def descriptor(self, reference):
        if len(self.values) != 3:
            return query_descriptor(None, reference)
        return query_descriptor(torch.stack([self.values[i] for i in range(3)], dim=1), reference)

    def close(self):
        for handle in self.handles:
            handle.remove()
