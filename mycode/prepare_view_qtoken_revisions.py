#!/usr/bin/env python3
"""Derive v2/v3 targets without decoding or duplicating semantic videos."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from view_qtoken import (
    CLASSES, EXPERIMENT, VIEW_V2, VIEW_V3, SemanticLabelStore, ViewQueryStore,
    dataset_signature, front_tool_vectors, revision_layout, sha256,
)


def midpoint_chunk(task):
    path, start, stop = task
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    targets = np.zeros((stop - start, 4), np.float32)
    valid = np.zeros(stop - start, bool)
    for i, ids in enumerate(labels[start:stop, 0]):
        targets[i], valid[i] = front_tool_vectors(
            ids == CLASSES.index("object") + 1, ids == CLASSES.index("tool") + 1
        )
    return start, targets, valid


def derive_targets(original, experiment, vectors=None, vector_valid=None):
    if experiment == VIEW_V2:
        return {k: v[:, [0, 1, 2, 3, 6, 7]].copy() for k, v in original.items()}
    if experiment != VIEW_V3 or vectors is None or vector_valid is None:
        raise ValueError("v3 needs independently computed front PCA midpoint targets.")
    # Original front endpoint parity checks also verify cache/frame/class alignment.
    np.testing.assert_array_equal(original["valid"][:, 2], vector_valid)
    np.testing.assert_allclose(original["targets"][:, 2:4], vectors[:, :2], atol=2e-7, rtol=0)
    result = {k: v.copy() for k, v in original.items()}
    result["targets"][:, 4:6] = vectors[:, 2:4]
    result["valid"][:, 4:6] = vector_valid[:, None]
    result["weights"][:, 4:6] = original["weights"][:, 2:4]
    return result


def prepare(args):
    root = args.root.resolve()
    original = ViewQueryStore(args.base_cache / "targets.npz", root, experiment=EXPERIMENT)
    labels = SemanticLabelStore(args.semantic_label_cache, root)
    destinations = {VIEW_V2: args.output_parent / "final-ViewQtoken-v2",
                    VIEW_V3: args.output_parent / "final-ViewQtoken-v3"}
    needed = []
    for experiment, dest in destinations.items():
        if (dest / "manifest.json").is_file():
            ViewQueryStore(dest / "targets.npz", root, experiment=experiment)
            print(f"PASS existing {experiment}: {dest}", flush=True)
        elif dest.exists():
            raise FileExistsError(f"Incomplete cache; choose a new output parent: {dest}")
        else:
            needed.append(experiment)
    vectors = vector_valid = None
    if VIEW_V3 in needed:
        count = len(labels.data)
        vectors, vector_valid = np.zeros((count, 4), np.float32), np.zeros(count, bool)
        tasks = [(str(labels.data.filename), s, min(s + 512, count)) for s in range(0, count, 512)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, (start, values, valid) in enumerate(pool.map(midpoint_chunk, tasks), 1):
                end = start + len(values)
                vectors[start:end], vector_valid[start:end] = values, valid
                if n % 20 == 0 or n == len(tasks):
                    print(f"Front PCA geometry: {end}/{count}", flush=True)
    if original.metadata["source_signature"] != dataset_signature(root):
        raise ValueError("Source changed while preparing targets.")
    with np.load(args.base_cache / "targets.npz", allow_pickle=False) as source:
        alignment = {k: source[k].copy() for k in ("episode_index", "frame_index") if k in source}
    for experiment in needed:
        arrays = derive_targets(original.data, experiment, vectors, vector_valid)
        dest = destinations[experiment]
        dest.mkdir(parents=True)
        np.savez_compressed(dest / "targets.npz", **arrays, **alignment)
        metadata = dict(original.metadata)
        names, heads, groups = revision_layout(experiment)
        metadata.update(
            experiment=experiment, targets_sha256=sha256(dest / "targets.npz"),
            output_order=["visibility_front", "visibility_side", "tool_front_dx", "tool_front_dy"] +
                         (["tool_front_midpoint_dx", "tool_front_midpoint_dy"] if experiment == VIEW_V3 else []) +
                         ["region_front", "region_side"],
            supervised_fraction=(arrays["weights"] > 0).mean(0).tolist(),
            mean_weight=arrays["weights"].mean(0).tolist(),
            region_note="Visible 2D projection containment, not 3D success. Region previews are in the base_cache directory.",
            head_names=list(names), head_indices=[list(h) for h in heads],
            task_groups=[list(g) for g in groups],
            provenance={"base_cache": str(args.base_cache.resolve()),
                        "base_targets_sha256": original.metadata["targets_sha256"],
                        "semantic_label_cache": str(args.semantic_label_cache.resolve()),
                        "semantic_manifest_sha256": sha256(args.semantic_label_cache / "manifest.json"),
                        "front_endpoint_parity": "all frames, atol=2e-7" if experiment == VIEW_V3 else "exact subset"},
            geometry="original dual-view occupancy and region containment; front normalized-coordinate PCA left endpoint" +
                     (" and segment midpoint to object centroid" if experiment == VIEW_V3 else " to object centroid"),
            loss_reduction="mean of three task means; equal head weighting within each task",
        )
        (dest / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        ViewQueryStore(dest / "targets.npz", root, experiment=experiment)
        print(f"PASS {experiment}: {len(arrays['targets'])} frames; "
              f"valid={arrays['valid'].mean(0).tolist()}; weights={arrays['weights'].mean(0).tolist()}; {dest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--semantic-label-cache", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    prepare(parser.parse_args())
