"""Build a lossless, episode-local UR semantic/query cache without editing source data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from ur_semq import (
    LABELS,
    VERSION,
    URSemanticCache,
    descriptor_weights,
    descriptors,
    digest,
    reference_region,
    source_signature,
)


def cache_episode(task):
    root, output, info, row = task
    root, output = Path(root), Path(output)
    cv2.setNumThreads(1)
    start, length = int(row["dataset_from_index"]), int(row["length"])
    labels = np.load(output / "semantic_labels.npy", mmap_mode="r+")
    scores, uncertain = {}, {}
    for name in ("object", "organ", "occluder", "region"):
        with np.load(root / f"segmentation_quality/observation_images_front_{name}.npz") as data:
            scores[name] = data["quality_score"][start : start + length].copy()
            uncertain[name] = data["uncertain"][start : start + length].copy()
    caps = []
    try:
        for name in LABELS[1:]:
            key = f"observation.images.front_{name}"
            path = root / info["video_path"].format(
                video_key=key,
                chunk_index=row[f"videos/{key}/chunk_index"],
                file_index=row[f"videos/{key}/file_index"],
            )
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            if not cap.isOpened():
                raise ValueError(f"Cannot decode {path}")
            caps.append(cap)
        initial = []
        reference, reference_quality, usable = None, 0.0, 0
        targets, valid = np.zeros((length, 6), np.float32), np.zeros((length, 6), bool)
        for frame in range(length):
            ids = np.zeros(labels.shape[1:], np.uint8)
            for class_id, cap in enumerate(caps, 1):
                ok, image = cap.read()
                if not ok or image.shape[:2] != ids.shape:
                    raise ValueError(
                        f"Mask frame/shape mismatch: episode {row['episode_index']}, frame {frame}"
                    )
                # Exact integer equivalent of RGB mean >= 127.5, without float64 conversion.
                mask = image.sum(axis=2, dtype=np.uint16) >= 383
                if np.any(mask & (ids > 0)):
                    raise ValueError("Overlapping source class masks.")
                ids[mask] = class_id
            labels[start + frame] = ids
            if frame < 20:
                initial.append(ids == LABELS.index("region"))
                if frame == 19:
                    reference, reference_quality, usable = reference_region(
                        np.stack(initial), scores["region"][:20], uncertain["region"][:20]
                    )
            targets[frame], valid[frame] = descriptors(ids, reference if frame >= 20 else None)
        if any(cap.read()[0] for cap in caps):
            raise ValueError("Mask video longer than episode metadata.")
    finally:
        for cap in caps:
            cap.release()
    weights = descriptor_weights(valid, scores, uncertain, reference_quality)
    if reference is not None:
        np.save(output / "regions" / f"episode_{row['episode_index']:03d}.npy", reference)
    del labels
    return (
        start,
        targets,
        valid,
        weights,
        {
            "episode_index": int(row["episode_index"]),
            "usable_frames": usable,
            "reference_quality": reference_quality,
            "valid": reference is not None,
        },
    )


def prepare(root, output, workers):
    root, output = root.resolve(), output.resolve()
    if (output / "manifest.json").exists():
        URSemanticCache(output, root)
        print(f"Verified existing cache: {output}", flush=True)
        return
    if output.exists():
        raise FileExistsError(f"Incomplete/existing cache, choose a new directory: {output}")
    info = json.loads((root / "meta/info.json").read_text())
    model = json.loads((root / "models/unet_front/model_config.json").read_text())
    if tuple(model["labels"]) != LABELS:
        raise ValueError("Unexpected UR segmentation classes.")
    rows = sorted(
        [r for p in (root / "meta/episodes").glob("*/*.parquet") for r in pq.read_table(p).to_pylist()],
        key=lambda r: r["episode_index"],
    )
    if [r["episode_index"] for r in rows] != list(range(info["total_episodes"])):
        raise ValueError("Missing/duplicate episode metadata.")
    offset = 0
    for row in rows:
        if row["dataset_from_index"] != offset or row["length"] < 20:
            raise ValueError("Invalid episode offsets/length.")
        offset += row["length"]
        if row["dataset_to_index"] != offset:
            raise ValueError("Invalid episode end index.")
        for name in LABELS[1:]:
            if abs(row[f"videos/observation.images.front_{name}/from_timestamp"]) > 1e-6:
                raise ValueError("Packed videos with nonzero offsets are unsupported.")
    if offset != info["total_frames"]:
        raise ValueError("Frame total mismatch.")
    episode = np.concatenate([np.full(r["length"], r["episode_index"], np.int64) for r in rows])
    frame = np.concatenate([np.arange(r["length"]) for r in rows])
    for name in LABELS[1:]:
        with np.load(root / f"segmentation_quality/observation_images_front_{name}.npz") as q:
            if not np.array_equal(q["episode_index"], episode) or not np.array_equal(q["frame_index"], frame):
                raise ValueError("Quality arrays not frame-aligned.")
            if (
                not np.isfinite(q["quality_score"]).all()
                or np.any((q["quality_score"] < 0) | (q["quality_score"] > 1))
                or q["uncertain"].dtype != bool
            ):
                raise ValueError("Invalid quality values.")
    signature = source_signature(root)
    shape = (offset, *info["features"]["observation.images.front"]["shape"][:2])
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < np.prod(shape) * 1.1:
        raise OSError("Insufficient space for the lossless semantic cache.")
    output.mkdir()
    (output / "regions").mkdir()
    labels = np.lib.format.open_memmap(output / "semantic_labels.npy", mode="w+", dtype=np.uint8, shape=shape)
    del labels
    targets, valid, weights = (
        np.zeros((offset, 6), np.float32),
        np.zeros((offset, 6), bool),
        np.zeros((offset, 6), np.float32),
    )
    references = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = ((str(root), str(output), info, row) for row in rows)
        for i, (start, t, v, w, reference) in enumerate(pool.map(cache_episode, tasks), 1):
            (
                targets[start : start + len(t)],
                valid[start : start + len(t)],
                weights[start : start + len(t)],
            ) = t, v, w
            references.append(reference)
            print(f"cached {i}/{len(rows)} episodes", flush=True)
    scales = np.ones(6, np.float32)
    for column in (0, 1, 3):
        positive = targets[(weights[:, column] > 0) & (targets[:, column] > 0), column]
        scales[column] = max(float(np.quantile(positive, 0.95)) if len(positive) else 1.0, 0.001)
    np.savez_compressed(
        output / "targets.npz",
        targets=targets,
        valid=valid,
        weights=weights,
        episode_index=episode,
        frame_index=frame,
    )
    if signature != source_signature(root):
        raise RuntimeError("Source changed during cache generation.")
    manifest = {
        "version": VERSION,
        "labels": LABELS,
        "shape": shape,
        "source_root": str(root),
        "source_signature": signature,
        "palette_rgb": model["palette_rgb"],
        "targets_sha256": digest(output / "targets.npz"),
        "scales": scales.tolist(),
        "region_initial_frames": 20,
        "region_min_valid_frames": 10,
        "region_min_quality": 0.8,
        "region_min_consensus_iou": 0.8,
        "regions": references,
        "valid_fraction": valid.mean(0).tolist(),
        "supervised_fraction": (weights > 0).mean(0).tolist(),
        "quality": "class quality * not uncertain * geometry valid; distance uses frozen region quality",
    }
    (output / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(output / "manifest.json.tmp", output / "manifest.json")
    URSemanticCache(output, root)
    print(f"PASS {output}; scales={scales.tolist()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.root, args.output_dir, args.workers)
