"""Shared task queries with view-specific readouts and filtered hard semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EXPERIMENT = "FINAL-VIEWQTOKEN"
SEM_ONLY_EXPERIMENT = "FINAL-SEM-FS-NOQ"
RGB_QUERY_EXPERIMENT = "FINAL-RGB-FS-QTOKEN"
FRONT_QUERY_EXPERIMENT = "FINAL-FRONTQTOKEN-FS"
FRONT_SYMMETRIC_EXPERIMENT = "FINAL-FRONTQTOKEN-FS-SEMMINUS1TO1"
FRONT_QUERY_EXPERIMENTS = {FRONT_QUERY_EXPERIMENT, FRONT_SYMMETRIC_EXPERIMENT}
QUERY_EXPERIMENTS = {EXPERIMENT, RGB_QUERY_EXPERIMENT, *FRONT_QUERY_EXPERIMENTS}
HARD_SEMANTIC_EXPERIMENTS = {EXPERIMENT, SEM_ONLY_EXPERIMENT, *FRONT_QUERY_EXPERIMENTS}
EXPERIMENTS = QUERY_EXPERIMENTS | HARD_SEMANTIC_EXPERIMENTS
VIEWS = ("front", "side")
CLASSES = ("occluder", "object", "region", "tool", "leftarm", "rightarm")
TASKS = ("visibility", "tool_object", "object_region")
HEADS = ((0,), (1,), (2, 3), (4, 5), (6,), (7,))
HEAD_NAMES = tuple(f"{task}_{view}" for task in TASKS for view in VIEWS)
OUTPUT_DIMS = (2, 4, 2)
CACHE_VERSION = 1
FRONT_HEADS = ((0,), (1,), (2, 3))
FRONT_HEAD_NAMES = ("visibility_front", "object_region_front", "tool_object_front")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ViewHeads(nn.Module):
    """Both independent heads read the same task token."""

    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.front = nn.Linear(hidden_dim, output_dim)
        self.side = nn.Linear(hidden_dim, output_dim)

    def forward(self, token):
        return torch.cat((self.front(token), self.side(token)), dim=-1)


def filter_class_ids(mask, labels, policy):
    # Keep the exporter's ordering (area descending, component ID breaks ties).
    if policy.get("distance_gt_area") or policy.get("contact_only_two") or policy.get("fill_holes"):
        raise ValueError("ViewQtoken supports the exported area/top-N rules only.")
    result = mask.copy()
    for class_id, name in enumerate(labels[1:], 1):
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (mask == class_id).astype(np.uint8), connectivity=8
        )
        ranked = sorted(range(1, count), key=lambda c: (-int(stats[c, cv2.CC_STAT_AREA]), c))
        eligible = [c for c in ranked if stats[c, cv2.CC_STAT_AREA] >= policy["min_area"]]
        limit = policy["component_limits"][name]
        keep = eligible[:limit] if limit else eligible
        result[(components > 0) & ~np.isin(components, keep)] = 0
    return result


class FrozenExportedUNet(nn.Module):
    """Portable TorchScript bundle; matches the exporter's RGB/OpenCV preprocessing."""

    def __init__(self, checkpoint_path, *, expected_labels):
        super().__init__()
        path = Path(checkpoint_path).resolve()
        root = path.parent
        self.bundle_config = json.loads((root / "model_config.json").read_text())
        self.labels = tuple(self.bundle_config["labels"])
        if self.labels != tuple(expected_labels):
            raise ValueError(f"Segmentation label mismatch: {self.labels} != {expected_labels}")
        self.filter_policy = json.loads((root / "filter_policy.json").read_text())["policy"]
        filter_class_ids(np.zeros((2, 2), np.uint8), self.labels, self.filter_policy)
        script = root / self.bundle_config["torchscript"]
        self.network = torch.jit.load(str(script), map_location="cpu").eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_path = str(path)
        self.model_sha256 = sha256(script)
        self.register_buffer("palette", torch.tensor(self.bundle_config["palette_rgb"], dtype=torch.float32) / 255)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, rgb):
        device = self.palette.device
        original_hw = tuple(rgb.shape[-2:])
        size = tuple(self.bundle_config["input_size_wh"])
        mean = np.asarray(self.bundle_config["normalization_mean"], np.float32)
        std = np.asarray(self.bundle_config["normalization_std"], np.float32)
        # Images enter this method in [0,1]; reproduce the uint8 RGB export path.
        images = (rgb.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
        prepared = np.stack([
            ((cv2.resize(im, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1)
            for im in images
        ])
        raw_ids = self.network(torch.from_numpy(prepared).to(device)).argmax(1).cpu().numpy().astype(np.uint8)
        masks = [filter_class_ids(
            cv2.resize(mask, original_hw[::-1], interpolation=cv2.INTER_NEAREST), self.labels, self.filter_policy
        ) for mask in raw_ids]
        ids = torch.from_numpy(np.stack(masks)).to(device=device, dtype=torch.long)
        return F.one_hot(ids, len(self.labels)).permute(0, 3, 1, 2).float()


def region_distance_map(shape, polygon):
    inside = np.zeros(shape, np.uint8)
    polygon = np.asarray(polygon, np.int32)
    if len(polygon) < 3 or cv2.contourArea(polygon) <= 0:
        raise ValueError("Region polygon is empty or degenerate.")
    cv2.fillPoly(inside, [polygon], 1)
    if not inside.any():
        raise ValueError("Region polygon falls outside the image.")
    return cv2.distanceTransform(1 - inside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def view_geometry(object_mask, tool_mask, region_mask, distance_map, view):
    """Return occupancy, vector x/y, region distance and geometry-only validity."""
    h, w = object_mask.shape
    target = np.zeros(4, np.float32)
    valid = np.zeros(4, bool)
    target[0] = object_mask.mean()
    valid[0] = True  # Reliable absence is a useful zero-visibility target.
    oy, ox = np.nonzero(object_mask)
    if not len(ox):
        return target, valid
    object_point = np.array([ox.mean() / max(w - 1, 1), oy.mean() / max(h - 1, 1)])
    if region_mask.any():
        target[3] = distance_map[object_mask].max() / max(np.hypot(w - 1, h - 1), 1)
        valid[3] = True
    ty, tx = np.nonzero(tool_mask)
    if not len(tx):
        return target, valid
    points = np.column_stack((tx / max(w - 1, 1), ty / max(h - 1, 1)))
    center = points.mean(0)
    reference = center
    if view == "front":
        if len(points) < 2:
            return target, valid
        centered = points - center
        values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
        if values[-1] <= 1e-12 or values[-1] - values[0] <= 1e-12:
            return target, valid
        axis = vectors[:, -1]
        projection = centered @ axis
        endpoints = center + np.array([projection.min(), projection.max()])[:, None] * axis
        reference = endpoints[np.lexsort((endpoints[:, 1], endpoints[:, 0]))[0]]
    elif view != "side":
        raise ValueError(f"Unknown view: {view}")
    target[1:3] = np.clip(object_point - reference, -1, 1)
    valid[1:3] = True
    return target, valid


def quality_weights(valid, scores, anomalies):
    """scores/anomalies have object, tool, region columns for one view."""
    q = np.stack((scores[:, 0], np.minimum(scores[:, 0], scores[:, 1]),
                  np.minimum(scores[:, 0], scores[:, 1]), np.minimum(scores[:, 0], scores[:, 2])), axis=-1)
    bad = np.stack((anomalies[:, 0], anomalies[:, 0] | anomalies[:, 1],
                    anomalies[:, 0] | anomalies[:, 1], anomalies[:, 0] | anomalies[:, 2]), axis=-1)
    return valid.astype(np.float32) * (~bad) * q


def query_loss(pred, target, valid, weights, *, experiment=EXPERIMENT):
    error = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
    losses, logs = [], {}
    names, heads = (FRONT_HEAD_NAMES, FRONT_HEADS) if experiment in FRONT_QUERY_EXPERIMENTS else (HEAD_NAMES, HEADS)
    for name, indices in zip(names, heads, strict=True):
        idx = list(indices)
        # Do not divide by sum(weights): uniformly low quality must reduce supervision.
        denominator = valid[:, idx].float().sum().clamp_min(1)
        loss = (error[:, idx] * weights[:, idx]).sum() / denominator
        losses.append(loss)
        logs[f"query_{name}_loss"] = float(loss.detach())
        logs[f"query_{name}_valid_ratio"] = float(valid[:, idx].float().mean())
        logs[f"query_{name}_weight"] = float(weights[:, idx].mean())
        logs[f"query_{name}_mae"] = float(
            ((pred.detach()[:, idx] - target[:, idx]).abs() * weights[:, idx]).sum()
            / weights[:, idx].sum().clamp_min(1e-8)
        )
    return torch.stack(losses).mean(), logs


def forward_training(model, batch, raw_batch):
    device = next(model.parameters()).device
    act_batch = dict(batch)
    if model.experiment in HARD_SEMANTIC_EXPERIMENTS:
        targets = model.build_semantic_targets(raw_batch, device=device)
        for view, rgb_key in enumerate(model.rgb_keys):
            palette = getattr(model, model._semantic_palette_buffer_names[view])
            hard_rgb = palette[targets[view]].permute(0, 3, 1, 2)
            act_batch[f"{rgb_key}_semantic"] = model.normalize_semantic_map(hard_rgb)
    action_loss, logs = model.act_policy(act_batch)
    if model.experiment == SEM_ONLY_EXPERIMENT:
        loss = model.action_loss_weight * action_loss
        logs.update(loss=float(loss.detach()), action_loss=float(action_loss.detach()), metric_loss=0.0, seg_loss=0.0)
        return loss, logs
    pred = logs.pop("metric_pred")
    metric_loss, metric_logs = query_loss(
        pred, raw_batch["view_query_targets"].to(device),
        raw_batch["view_query_valid"].to(device), raw_batch["view_query_weights"].to(device),
        experiment=model.experiment,
    )
    loss = model.action_loss_weight * action_loss + model.metric_loss_weight * metric_loss
    logs.update(metric_logs)
    logs.update(loss=float(loss.detach()), action_loss=float(action_loss.detach()),
                metric_loss=float(metric_loss.detach()), seg_loss=0.0)
    return loss, logs


def dataset_signature(root):
    """Bind supervision to labels, quality, episode layout and mask video revisions."""
    root = Path(root)
    files = [root / "meta/info.json", root / "segmentation_manifest.json"]
    files += sorted((root / "meta/episodes").glob("*/*.parquet"))
    files += sorted((root / "segmentation_quality").glob("*.npz"))
    hashes = {str(p.relative_to(root)): sha256(p) for p in files}
    videos = {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
              for p in sorted((root / "videos").rglob("*.mp4"))}
    return {"hashes": hashes, "videos": videos}


class ViewQueryStore:
    def __init__(self, path, root, *, experiment=EXPERIMENT):
        path = Path(path)
        self.metadata = json.loads((path.parent / "manifest.json").read_text())
        if self.metadata["version"] != CACHE_VERSION or self.metadata["source_signature"] != dataset_signature(root):
            raise ValueError("ViewQtoken cache is stale; prepare a new cache for this dataset.")
        if sha256(path) != self.metadata["targets_sha256"]:
            raise ValueError("ViewQtoken target cache checksum mismatch.")
        front = experiment in FRONT_QUERY_EXPERIMENTS
        if (self.metadata.get("experiment") == FRONT_QUERY_EXPERIMENT) != front:
            raise ValueError("Query cache uses a different target definition.")
        dimensions = 4 if front else 8
        total = json.loads((Path(root) / "meta/info.json").read_text())["total_frames"]
        with np.load(path, allow_pickle=False) as arrays:
            self.data = {key: arrays[key].copy() for key in ("targets", "valid", "weights")}
        for key, value in self.data.items():
            if value.shape != (total, dimensions) or not np.isfinite(value).all():
                raise ValueError(f"Invalid ViewQtoken {key} array.")
        if ((self.data["weights"] < 0) | (self.data["weights"] > 1)).any():
            raise ValueError("ViewQtoken weights must be in [0,1].")
        if self.data["valid"].dtype != np.bool_ or (self.data["weights"][~self.data["valid"]] != 0).any():
            raise ValueError("Invalid ViewQtoken validity/weight combination.")

    def add_batch(self, raw_batch):
        batch = dict(raw_batch)
        indices = batch["index"].cpu().numpy()
        if (indices < 0).any() or (indices >= len(self.data["targets"])).any():
            raise IndexError("ViewQtoken batch indices outside cache.")
        for key, values in self.data.items():
            batch[f"view_query_{key}"] = torch.from_numpy(values[indices])
        return batch


class SemanticLabelStore:
    """Lossless uint8 labels avoid decoding twelve binary mask videos per batch."""

    def __init__(self, path, root):
        path, root = Path(path), Path(root)
        metadata = json.loads((path / "manifest.json").read_text())
        if metadata["version"] != CACHE_VERSION or metadata["source_signature"] != dataset_signature(root):
            raise ValueError("Semantic cache is stale; prepare a new cache.")
        labels = metadata["semantic_labels"]
        if labels["views"] != list(VIEWS) or labels["classes"] != list(CLASSES):
            raise ValueError("Semantic cache view/class order mismatch.")
        info = json.loads((root / "meta/info.json").read_text())
        shapes = [info["features"][f"observation.images.{view}"]["shape"][:2] for view in VIEWS]
        if shapes[0] != shapes[1]:
            raise ValueError("Packed semantic cache requires matching view resolutions.")
        expected_shape = (info["total_frames"], len(VIEWS), *shapes[0])
        file = path / labels["file"]
        if sha256(file) != labels["sha256"]:
            raise ValueError("Semantic cache checksum mismatch.")
        self.data = np.load(file, mmap_mode="r", allow_pickle=False)
        if self.data.shape != expected_shape or self.data.dtype != np.uint8:
            raise ValueError("Semantic cache shape/dtype mismatch.")

    def add_batch(self, raw_batch):
        batch = dict(raw_batch)
        indices = batch["index"].cpu().numpy()
        if (indices < 0).any() or (indices >= len(self.data)).any():
            raise IndexError("Semantic batch indices outside cache.")
        batch["semantic_class_ids"] = torch.from_numpy(self.data[indices])
        return batch
