#!/usr/bin/env python3
"""Precompute final-ViewQtoken targets; never modify source masks or quality arrays."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from view_qtoken import (
    CACHE_VERSION, VIEWS, ViewQueryStore, dataset_signature, quality_weights,
    region_distance_map, sha256, view_geometry,
)


def mask_path(root, info, row, view, name):
    key = f"observation.images.{view}" + (f"_{name}" if name else "")
    return root / info["video_path"].format(
        video_key=key, chunk_index=int(row[f"videos/{key}/chunk_index"]),
        file_index=int(row[f"videos/{key}/file_index"]),
    )


def read_binary(cap):
    ok, image = cap.read()
    if not ok:
        raise ValueError("Mask video ended before episode metadata length.")
    return image.mean(axis=2) >= 127.5


def load_quality(root, view, name, episode_indices, frame_indices):
    path = root / f"segmentation_quality/observation_images_{view}_{name}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["episode_index"], episode_indices) or not np.array_equal(arrays["frame_index"], frame_indices):
            raise ValueError(f"Quality is not aligned to dataset indices: {path}")
        score, anomaly = arrays["quality_score"].copy(), arrays["uncertain"].copy()
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any() or anomaly.dtype != np.bool_:
        raise ValueError(f"Invalid quality values: {path}")
    return score, anomaly


def estimate_region(root, info, rows, quality, view, output):
    candidates = []
    for row in rows:
        start, length = int(row["dataset_from_index"]), int(row["length"])
        cap = cv2.VideoCapture(str(mask_path(root, info, row, view, "region")))
        try:
            for frame in np.unique(np.linspace(0, length - 1, 12, dtype=int)):
                if quality[0][start + frame] < .95 or quality[1][start + frame]:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                mask = read_binary(cap)
                if mask.sum() >= 20:
                    candidates.append((int(mask.sum()), mask, row, int(frame)))
        finally:
            cap.release()
    if len(candidates) < 3:
        raise ValueError(f"Insufficient reliable region observations for {view}.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:max(3, len(candidates) // 4)]
    consensus = np.mean(np.stack([item[1] for item in selected]), axis=0) >= .5
    points = cv2.findNonZero(consensus.astype(np.uint8))
    if points is None or len(points) < 3:
        raise ValueError(f"No stable region for {view}.")
    hull = cv2.convexHull(points).reshape(-1, 2)
    region_distance_map(consensus.shape, hull)
    # This diagnostic is an overlay of source data, not a synthetic scene.
    _, _, example, frame = selected[len(selected) // 2]
    cap = cv2.VideoCapture(str(mask_path(root, info, example, view, "")))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot read region diagnostic RGB frame.")
    cv2.polylines(image, [hull], True, (0, 0, 255), 2)
    if not cv2.imwrite(str(output / f"region_{view}.png"), image):
        raise OSError("Could not save region diagnostic.")
    return {"polygon": hull.tolist(), "candidate_frames": len(candidates),
            "selected_frames": len(selected), "example_episode": int(example["episode_index"]),
            "example_frame": frame,
            "method": "quality>=0.95 and not anomalous; top-area quartile; majority mask convex hull"}


def episode_targets(task):
    root, info, row, polygons = task
    root = Path(root)
    cv2.setNumThreads(1)
    n = int(row["length"])
    targets, validity = np.zeros((n, 8), np.float32), np.zeros((n, 8), bool)
    for vi, view in enumerate(VIEWS):
        shape = tuple(info["features"][f"observation.images.{view}"]["shape"][:2])
        distances = region_distance_map(shape, polygons[view]["polygon"])
        caps = [cv2.VideoCapture(str(mask_path(root, info, row, view, c))) for c in ("object", "tool", "region")]
        columns = [vi, 2 + 2 * vi, 3 + 2 * vi, 6 + vi]
        try:
            for frame in range(n):
                masks = [read_binary(cap) for cap in caps]
                if any(mask.shape != shape for mask in masks) or np.any(sum(mask.astype(np.uint8) for mask in masks) > 1):
                    raise ValueError("Mask size or exclusivity mismatch.")
                target, valid = view_geometry(*masks, distances, view)
                targets[frame, columns], validity[frame, columns] = target, valid
            if any(cap.read()[0] for cap in caps):
                raise ValueError("Mask video longer than episode metadata.")
        finally:
            for cap in caps:
                cap.release()
    return int(row["dataset_from_index"]), targets, validity


def prepare(root, output, workers):
    root, output = root.resolve(), output.resolve()
    if (output / "manifest.json").is_file():
        ViewQueryStore(output / "targets.npz", root)
        print(f"Verified existing cache: {output}", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"Incomplete/existing output, choose a new directory: {output}")
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("splits", {}).get("train") != f"0:{info['total_episodes']}":
        raise ValueError("This preparation command requires the all-training dataset; do not fit ROI on held-out episodes.")
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").glob("*/*.parquet"))]).sort_values("episode_index")
    if episodes.episode_index.tolist() != list(range(info["total_episodes"])):
        raise ValueError("Missing or duplicated episodes.")
    rows = episodes.to_dict("records")
    ep_indices = np.concatenate([np.full(int(row['length']), int(row['episode_index']), np.int64) for row in rows])
    frame_indices = np.concatenate([np.arange(int(row['length'])) for row in rows])
    if len(ep_indices) != info["total_frames"]:
        raise ValueError("Frame total mismatch.")
    offset = 0
    for row in rows:
        if int(row["dataset_from_index"]) != offset:
            raise ValueError("Non-contiguous episode offsets.")
        offset += int(row["length"])
        if int(row["dataset_to_index"]) != offset:
            raise ValueError("Episode end offset mismatch.")
        for view in VIEWS:
            for name in ("object", "tool", "region"):
                key = f"observation.images.{view}_{name}"
                if abs(float(row[f"videos/{key}/from_timestamp"])) > 1e-6:
                    raise ValueError("Packed videos require offset-aware preparation; not supported here.")
    signature = dataset_signature(root)
    quality = {view: {c: load_quality(root, view, c, ep_indices, frame_indices)
                      for c in ("object", "tool", "region")} for view in VIEWS}
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    polygons = {v: estimate_region(root, info, rows, quality[v]["region"], v, output) for v in VIEWS}
    total = info["total_frames"]
    targets, valid = np.zeros((total, 8), np.float32), np.zeros((total, 8), bool)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = ((str(root), info, row, polygons) for row in rows)
        for count, (start, values, validity) in enumerate(pool.map(episode_targets, tasks), 1):
            targets[start:start + len(values)], valid[start:start + len(values)] = values, validity
            print(f"targets {count}/{len(rows)} episodes", flush=True)
    weights = np.zeros_like(targets)
    for vi, view in enumerate(VIEWS):
        columns = [vi, 2 + 2 * vi, 3 + 2 * vi, 6 + vi]
        scores = np.stack([quality[view][c][0] for c in ("object", "tool", "region")], -1)
        anomalies = np.stack([quality[view][c][1] for c in ("object", "tool", "region")], -1)
        weights[:, columns] = quality_weights(valid[:, columns], scores, anomalies)
    if signature != dataset_signature(root):
        raise RuntimeError("Source data changed during preparation; cache not published.")
    np.savez_compressed(output / "targets.npz", targets=targets, valid=valid, weights=weights,
                        episode_index=ep_indices, frame_index=frame_indices)
    manifest = {"version": CACHE_VERSION, "source_root": str(root), "source_signature": signature,
                "targets_sha256": sha256(output / "targets.npz"), "regions": polygons,
                "output_order": ["visibility_front", "visibility_side", "tool_front_dx", "tool_front_dy",
                                 "tool_side_dx", "tool_side_dy", "region_front", "region_side"],
                "frames": total, "valid_fraction": valid.mean(0).tolist(),
                "supervised_fraction": (weights > 0).mean(0).tolist(), "mean_weight": weights.mean(0).tolist(),
                "quality": "filtered score times geometry validity times not-anomalous; no extra score threshold",
                "region_note": "Visible 2D projection containment, not a 3D task-success oracle. Inspect region_*.png."}
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output / "manifest.json")
    ViewQueryStore(output / "targets.npz", root)
    print(f"PASS: {output}; inspect region_front.png and region_side.png", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    prepare(args.root, args.output_dir, args.workers)
