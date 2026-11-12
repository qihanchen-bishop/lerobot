"""UR-only semantic descriptors. No SO-ARM experiment registries are modified."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

VERSION = 1
LABELS = ("background", "occluder", "object", "region", "leftarm", "rightarm", "organ", "wire")
HEADS = ((0,), (1,), (2,), (3,), (4, 5))
HEAD_NAMES = ("object_area", "organ_area", "object_region_distance", "occluder_area", "object_position")
SEMANTIC_KEY = "observation.images.ur_semantic"
FRONT_KEY = "observation.images.front"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_signature(root):
    """Hash metadata/quality, and record video size/mtime without rereading terabytes."""
    root = Path(root)
    result = {}
    for folder in ("meta", "segmentation_quality"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file():
                result[str(path.relative_to(root))] = digest(path)
    for name in (
        "segmentation_manifest.json",
        "models/unet_front/model_config.json",
        "models/unet_front/filter_policy.json",
    ):
        result[name] = digest(root / name)
    for path in sorted((root / "videos").rglob("*.mp4")):
        stat = path.stat()
        result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return result


def reference_region(masks, quality, uncertain, *, min_frames=10, min_quality=0.8):
    """Only the episode's initial 20 frames are eligible; never search its future."""
    usable = (~uncertain) & (quality >= min_quality) & (masks.sum(axis=(1, 2)) >= 20)
    if usable.sum() < min_frames:
        return None, 0.0, int(usable.sum())
    reference = np.average(masks[usable], axis=0, weights=quality[usable]) >= 0.5
    if reference.sum() < 20:
        return None, 0.0, int(usable.sum())
    intersection = (masks[usable] & reference).sum(axis=(1, 2))
    union = (masks[usable] | reference).sum(axis=(1, 2))
    stability = float(np.median(intersection / np.maximum(union, 1)))
    if stability < 0.8:
        return None, 0.0, int(usable.sum())
    return reference, float(np.mean(quality[usable]) * stability), int(usable.sum())


def descriptors(ids, reference):
    h, w = ids.shape
    target = np.zeros(6, np.float32)
    valid = np.zeros(6, bool)
    for column, name in ((0, "object"), (1, "organ"), (3, "occluder")):
        target[column] = np.mean(ids == LABELS.index(name))
        valid[column] = True  # Reliable absence is a meaningful zero-area target.
    y, x = np.nonzero(ids == LABELS.index("object"))
    if x.size:
        target[4:] = (x.mean() / max(w - 1, 1), y.mean() / max(h - 1, 1))
        valid[4:] = True
        if reference is not None:
            ry, rx = np.nonzero(reference)
            if rx.size:
                target[2] = np.hypot(x.mean() - rx.mean(), y.mean() - ry.mean()) / np.hypot(w - 1, h - 1)
                valid[2] = True
    return target, valid


def descriptor_weights(valid, scores, uncertain, reference_quality):
    weights = np.zeros(valid.shape, np.float32)
    for columns, name in (((0, 4, 5), "object"), ((1,), "organ"), ((3,), "occluder")):
        weights[:, columns] = (scores[name] * ~uncertain[name])[:, None]
    weights[:, 2] = np.minimum(scores["object"], reference_quality) * ~uncertain["object"]
    return weights * valid


def query_loss(prediction, targets, valid, weights, scales):
    """Equal head weighting; do not normalize away uniformly low label quality."""
    error = F.smooth_l1_loss(prediction.float(), targets / scales, reduction="none", beta=0.1)
    losses, logs = [], {}
    for name, columns in zip(HEAD_NAMES, HEADS, strict=True):
        idx = list(columns)
        loss = (error[:, idx] * weights[:, idx]).sum() / valid[:, idx].sum().clamp_min(1)
        losses.append(loss)
        mae = ((prediction.detach()[:, idx] * scales[idx] - targets[:, idx]).abs() * weights[:, idx]).sum()
        logs[f"query_{name}_loss"] = loss.detach()
        logs[f"query_{name}_mae"] = mae / weights[:, idx].sum().clamp_min(1e-8)
        logs[f"query_{name}_supervised"] = (weights[:, idx] > 0).float().mean()
        logs[f"query_{name}_weight"] = weights[:, idx].mean()
    for name, present in (("visible", targets[:, 0] > 0), ("absent", targets[:, 0] == 0)):
        w = weights[:, 0] * present
        logs[f"query_object_area_{name}_mae"] = (
            (prediction.detach()[:, 0] * scales[0] - targets[:, 0]).abs() * w
        ).sum() / w.sum().clamp_min(1e-8)
    return torch.stack(losses).mean(), logs


class URSemanticCache:
    def __init__(self, directory, root, verify=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        m = self.manifest
        if m["version"] != VERSION or tuple(m["labels"]) != LABELS:
            raise ValueError("Not a compatible UR-SemQ cache.")
        if verify and m["source_signature"] != source_signature(root):
            raise ValueError("UR cache source changed; create a new versioned cache.")
        if digest(self.directory / "targets.npz") != m["targets_sha256"]:
            raise ValueError("Corrupt descriptor cache.")
        with np.load(self.directory / "targets.npz", allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key].copy() for key in arrays.files}
        self._labels = None
        shape = tuple(m["shape"])
        labels = np.load(self.directory / "semantic_labels.npy", mmap_mode="r")
        if labels.shape != shape or labels.dtype != np.uint8:
            raise ValueError("Invalid semantic cache shape/type.")
        for key in ("targets", "weights", "valid"):
            if self.arrays[key].shape != (shape[0], 6):
                raise ValueError(f"Invalid descriptor shape: {key}")
        if not np.isfinite(self.arrays["targets"]).all() or not np.isfinite(self.arrays["weights"]).all():
            raise ValueError("Non-finite descriptor cache.")
        if np.any((self.arrays["weights"] < 0) | (self.arrays["weights"] > 1)):
            raise ValueError("Descriptor weights must lie in [0, 1].")
        if self.arrays["valid"].dtype != bool or np.any(self.arrays["weights"][~self.arrays["valid"]] != 0):
            raise ValueError("Invalid geometry/weight combination.")

    def labels(self, index):
        if self._labels is None:
            self._labels = np.load(self.directory / "semantic_labels.npy", mmap_mode="r")
        return np.array(self._labels[index], copy=True)

    def __getstate__(self):
        return {**self.__dict__, "_labels": None}
