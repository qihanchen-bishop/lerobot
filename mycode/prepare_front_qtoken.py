#!/usr/bin/env python3
"""Cache legacy front query targets and lossless dual-view hard semantic inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
import torch

from prepare_view_qtoken import load_quality, mask_path, read_binary
from view_qtoken import (
    CACHE_VERSION, CLASSES, FRONT_QUERY_EXPERIMENT, VIEWS,
    SemanticLabelStore, ViewQueryStore, dataset_signature, quality_weights, sha256,
)


def legacy_targets(class_ids):
    # Reuse the original implementation, including PCA endpoint conventions.
    from train_mask_act_policy import front_query_metric_targets

    ids = torch.from_numpy(np.asarray(class_ids).copy()).long()
    probabilities = torch.nn.functional.one_hot(ids, len(CLASSES) + 1).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        targets, valid = front_query_metric_targets(probabilities, CLASSES)
    return targets.numpy(), valid.numpy()


def cache_episode(task):
    root, output, info, row = task
    root, output = Path(root), Path(output)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    start, length = int(row["dataset_from_index"]), int(row["length"])
    data = np.load(output / "semantic_labels.npy", mmap_mode="r+")
    targets = np.zeros((length, 4), np.float32)
    valid = np.zeros((length, 4), bool)
    for vi, view in enumerate(VIEWS):
        caps = []
        try:
            for name in CLASSES:
                path = str(mask_path(root, info, row, view, name))
                if hasattr(cv2, "CAP_PROP_N_THREADS"):
                    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
                else:
                    cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot decode {path}")
                caps.append(cap)
            for frame in range(length):
                ids = np.zeros(data.shape[-2:], np.uint8)
                for class_id, cap in enumerate(caps, 1):
                    mask = read_binary(cap)
                    if mask.shape != ids.shape or ((ids > 0) & mask).any():
                        raise ValueError(f"Mask size/overlap at episode {row['episode_index']}, {view}, {frame}")
                    ids[mask] = class_id
                data[start + frame, vi] = ids
            if any(cap.read()[0] for cap in caps):
                raise ValueError("Video longer than its episode.")
        finally:
            for cap in caps:
                cap.release()
    for offset in range(0, length, 8):
        end = min(offset + 8, length)
        targets[offset:end], valid[offset:end] = legacy_targets(data[start + offset:start + end, 0])
    # Workers write disjoint frame ranges; publish the manifest only after all finish.
    del data
    return start, targets, valid


def prepare(root, output, workers):
    root, output = root.resolve(), output.resolve()
    if (output / "manifest.json").is_file():
        ViewQueryStore(output / "targets.npz", root, experiment=FRONT_QUERY_EXPERIMENT)
        SemanticLabelStore(output, root)
        print(f"PASS: verified existing cache {output}", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"Incomplete/existing cache; choose a new output directory: {output}")
    info = json.loads((root / "meta/info.json").read_text())
    rows = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("*/*.parquet"))])
    rows = rows.sort_values("episode_index").to_dict("records")
    if [int(row["episode_index"]) for row in rows] != list(range(info["total_episodes"])):
        raise ValueError("Missing or duplicate episodes.")
    shapes = [info["features"][f"observation.images.{view}"]["shape"][:2] for view in VIEWS]
    if shapes[0] != shapes[1]:
        raise ValueError("Views must have equal resolution for this packed cache.")
    offset = 0
    for row in rows:
        if int(row["dataset_from_index"]) != offset:
            raise ValueError("Noncontiguous episode offsets.")
        offset += int(row["length"])
        if int(row["dataset_to_index"]) != offset:
            raise ValueError("Episode end offset mismatch.")
        for view in VIEWS:
            for name in CLASSES:
                if abs(float(row[f"videos/observation.images.{view}_{name}/from_timestamp"])) > 1e-6:
                    raise ValueError("Packed videos with nonzero offsets are not supported.")
    if offset != info["total_frames"]:
        raise ValueError("Frame total mismatch.")
    ep = np.concatenate([np.full(int(r["length"]), int(r["episode_index"]), np.int64) for r in rows])
    frames = np.concatenate([np.arange(int(r["length"])) for r in rows])
    quality = {name: load_quality(root, "front", name, ep, frames) for name in ("object", "tool", "region")}
    signature = dataset_signature(root)
    shape = (offset, len(VIEWS), *shapes[0])
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < int(np.prod(shape)) * 1.1:
        raise OSError("Insufficient free disk for lossless semantic cache.")
    output.mkdir(parents=True)
    labels_path = output / "semantic_labels.npy"
    data = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.uint8, shape=shape)
    del data
    targets, valid = np.zeros((offset, 4), np.float32), np.zeros((offset, 4), bool)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = ((str(root), str(output), info, row) for row in rows)
        for count, (start, values, validity) in enumerate(pool.map(cache_episode, tasks), 1):
            targets[start:start + len(values)] = values
            valid[start:start + len(values)] = validity
            print(f"cached {count}/{len(rows)} episodes", flush=True)
    scores = np.stack([quality[c][0] for c in ("object", "tool", "region")], -1)
    anomalies = np.stack([quality[c][1] for c in ("object", "tool", "region")], -1)
    # quality_weights uses occupancy, dx, dy, region; the legacy order differs.
    order = [0, 2, 3, 1]
    weights = quality_weights(valid[:, order], scores, anomalies)[:, [0, 3, 1, 2]]
    if signature != dataset_signature(root):
        raise RuntimeError("Source changed during preparation; cache not published.")
    np.savez_compressed(output / "targets.npz", targets=targets, valid=valid, weights=weights,
                        episode_index=ep, frame_index=frames)
    manifest = {
        "version": CACHE_VERSION, "experiment": FRONT_QUERY_EXPERIMENT,
        "source_root": str(root), "source_signature": signature,
        "targets_sha256": sha256(output / "targets.npz"),
        "output_order": ["visibility_front", "centroid_region_front", "tool_front_dx", "tool_front_dy"],
        "geometry": "original front_query_metric_targets, without changing PCA or centroid-distance definitions",
        "frames": offset, "valid_fraction": valid.mean(0).tolist(),
        "supervised_fraction": (weights > 0).mean(0).tolist(), "mean_weight": weights.mean(0).tolist(),
        "quality": "filtered class score times geometry validity times not-anomalous; pairwise minimum score",
        "semantic_labels": {"file": labels_path.name, "sha256": sha256(labels_path),
                            "views": list(VIEWS), "classes": list(CLASSES), "shape": list(shape),
                            "source": "binary masks, mean RGB >= 127.5; no resize or new filtering"},
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    ViewQueryStore(output / "targets.npz", root, experiment=FRONT_QUERY_EXPERIMENT)
    SemanticLabelStore(output, root)
    print(f"PASS: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    prepare(args.root, args.output_dir, args.workers)
