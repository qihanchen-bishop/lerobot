#!/usr/bin/env python
"""Repair LeRobot v3 image statistics and optionally normalize vector parquet schemas."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import (
    aggregate_feature_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--image-key", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--normalize-vector-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_episode_tables(root: Path) -> list[tuple[Path, pa.Table, list[dict[str, Any]]]]:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet files under {root / 'meta' / 'episodes'}")
    tables = []
    for path in paths:
        table = pq.read_table(path)
        tables.append((path, table, table.to_pylist()))
    return tables


def video_path_for_episode(root: Path, image_key: str, row: dict[str, Any]) -> Path:
    chunk = row[f"videos/{image_key}/chunk_index"]
    file_index = row[f"videos/{image_key}/file_index"]
    return root / "videos" / image_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def compute_video_stats(
    episode_index: int,
    video_path: Path,
    expected_frames: int,
) -> tuple[int, dict[str, np.ndarray], int, tuple[int, int]]:
    selected_indices = set(sample_indices(expected_frames))
    sampled_images: list[np.ndarray] = []
    decoded_frames = 0
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index in selected_indices:
                image = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                sampled_images.append(auto_downsample_height_width(image))
            decoded_frames += 1

    if decoded_frames != expected_frames:
        raise ValueError(
            f"Episode {episode_index} video has {decoded_frames} frames; expected {expected_frames}: {video_path}"
        )
    if len(sampled_images) != len(selected_indices):
        raise ValueError(
            f"Episode {episode_index} sampled {len(sampled_images)} images; expected {len(selected_indices)}."
        )

    images = np.stack(sampled_images, axis=0)
    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    normalized = {
        key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
        for key, value in stats.items()
    }
    return episode_index, normalized, decoded_frames, tuple(images.shape[-2:])


def serialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    return value


def prepare_episode_metadata_updates(
    episode_tables: list[tuple[Path, pa.Table, list[dict[str, Any]]]],
    image_key: str,
    stats_by_episode: dict[int, dict[str, np.ndarray]],
) -> list[tuple[Path, pa.Table]]:
    prepared = []
    for path, table, rows in episode_tables:
        updated_rows = []
        for row in rows:
            episode_index = int(row["episode_index"])
            image_stats = stats_by_episode[episode_index]
            updated = dict(row)
            for stat_name, value in image_stats.items():
                field = f"stats/{image_key}/{stat_name}"
                if field not in table.schema.names:
                    raise KeyError(f"Episode metadata schema is missing {field}: {path}")
                updated[field] = serialize(value)
            updated_rows.append(updated)
        prepared.append((path, pa.Table.from_pylist(updated_rows, schema=table.schema)))
    return prepared


def vector_schema_is_fixed(table: pa.Table, vector_keys: list[str], vector_dims: dict[str, int]) -> bool:
    return all(
        pa.types.is_fixed_size_list(table.schema.field(key).type)
        and table.schema.field(key).type.list_size == vector_dims[key]
        for key in vector_keys
    )


def prepare_vector_schema_updates(
    root: Path,
    features: dict[str, dict[str, Any]],
) -> list[tuple[Path, pa.Table]]:
    vector_keys = [
        key
        for key, feature in features.items()
        if feature["dtype"] != "video" and len(feature["shape"]) == 1 and feature["shape"][0] > 1
    ]
    vector_dims = {key: int(features[key]["shape"][0]) for key in vector_keys}
    prepared = []
    for path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(path)
        if vector_schema_is_fixed(table, vector_keys, vector_dims):
            continue
        data = table.to_pydict()
        for key in vector_keys:
            lengths = {len(value) for value in data[key]}
            if lengths != {vector_dims[key]}:
                raise ValueError(f"{path} feature {key} has lengths {sorted(lengths)}; expected {vector_dims[key]}.")
            column_index = table.schema.get_field_index(key)
            source_field = table.schema.field(column_index)
            value_type = source_field.type.value_type
            flat_values = pa.array(np.asarray(data[key]).reshape(-1), type=value_type)
            fixed_values = pa.FixedSizeListArray.from_arrays(flat_values, vector_dims[key])
            fixed_field = source_field.with_type(pa.list_(value_type, vector_dims[key]))
            table = table.set_column(column_index, fixed_field, fixed_values)
        prepared.append((path, table))
    return prepared


def create_backup(
    root: Path,
    metadata_paths: list[Path],
    data_paths: list[Path],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = root / "meta" / "backups" / f"image_stats_fix_{timestamp}"
    backup_root.mkdir(parents=True, exist_ok=False)
    for path in metadata_paths:
        destination = backup_root / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for path in data_paths:
        destination = backup_root / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return backup_root


def write_parquet_updates(updates: list[tuple[Path, pa.Table]]) -> None:
    staged = []
    try:
        for path, table in updates:
            temporary = path.with_name(f".{path.name}.repair.tmp")
            pq.write_table(table, temporary)
            staged.append((path, temporary))
        for path, temporary in staged:
            os.replace(temporary, path)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}.")

    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "stats.json"
    info = json.loads(info_path.read_text())
    if args.image_key not in info["features"] or info["features"][args.image_key]["dtype"] != "video":
        raise KeyError(f"Not a declared video feature: {args.image_key}")

    episode_tables = load_episode_tables(root)
    episode_rows = [row for _, _, rows in episode_tables for row in rows]
    episode_ids = sorted(int(row["episode_index"]) for row in episode_rows)
    if episode_ids != list(range(info["total_episodes"])):
        raise ValueError(f"Episode metadata indices are not continuous: {episode_ids}")

    stats_by_episode: dict[int, dict[str, np.ndarray]] = {}
    downsampled_shapes = set()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for row in episode_rows:
            episode_index = int(row["episode_index"])
            video_path = video_path_for_episode(root, args.image_key, row)
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            futures.append(
                executor.submit(compute_video_stats, episode_index, video_path, int(row["length"]))
            )
        for completed, future in enumerate(as_completed(futures), start=1):
            episode_index, image_stats, _, downsampled_shape = future.result()
            stats_by_episode[episode_index] = image_stats
            downsampled_shapes.add(downsampled_shape)
            if completed % 10 == 0 or completed == len(futures):
                print(f"Computed image stats: {completed}/{len(futures)}", flush=True)

    global_image_stats = aggregate_feature_stats(
        [stats_by_episode[index] for index in range(info["total_episodes"])]
    )
    global_std = np.asarray(global_image_stats["std"]).reshape(-1)
    if not np.isfinite(global_std).all() or (global_std <= 0.02).any():
        raise ValueError(f"Recomputed image std is implausible: {global_std.tolist()}")

    episode_updates = prepare_episode_metadata_updates(
        episode_tables,
        args.image_key,
        stats_by_episode,
    )
    schema_updates = (
        prepare_vector_schema_updates(root, info["features"])
        if args.normalize_vector_schema
        else []
    )
    print(f"Downsampled image shapes used for stats: {sorted(downsampled_shapes)}")
    print(f"Global image mean: {np.asarray(global_image_stats['mean']).reshape(-1).tolist()}")
    print(f"Global image std: {global_std.tolist()}")
    print(f"Episode metadata files to update: {len(episode_updates)}")
    print(f"Data parquet schemas to normalize: {len(schema_updates)}")
    if args.dry_run:
        print("Dry run complete; no files were changed.")
        return

    metadata_paths = [stats_path, *(path for path, _ in episode_updates)]
    backup_root = create_backup(root, metadata_paths, [path for path, _ in schema_updates])
    print(f"Backup: {backup_root}")

    global_stats = json.loads(stats_path.read_text())
    global_stats[args.image_key] = serialize(global_image_stats)
    temporary_stats = stats_path.with_name(f".{stats_path.name}.repair.tmp")
    temporary_stats.write_text(json.dumps(global_stats, indent=4) + "\n")
    os.replace(temporary_stats, stats_path)
    write_parquet_updates([*episode_updates, *schema_updates])
    print("Repair complete.")


if __name__ == "__main__":
    main()
