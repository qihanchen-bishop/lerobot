#!/usr/bin/env python

"""Create a 5-DoF SO101 dataset by removing gripper state and action dims.

The source dataset is left untouched.  The output dataset keeps the same
episodes, task metadata and videos, but changes:

    action:             6 -> 5, drops "gripper.pos"
    observation.state:  6 -> 5, drops "gripper.pos"

Run from the repository root:

    conda run -n lerobot python mycode/make_so101_no_gripper_dataset.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DROP_NAME = "gripper.pos"
VECTOR_COLUMNS = ("action", "observation.state")


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def _drop_last_fixed_list(array: pa.ChunkedArray) -> pa.Array:
    values: list[list[float]] = []
    for chunk in array.chunks:
        py_values = chunk.to_pylist()
        values.extend([list(row[:-1]) for row in py_values])
    return pa.array(values, type=pa.list_(pa.float32(), list_size=5))


def convert_parquet(src_file: Path, dst_file: Path) -> None:
    table = pq.read_table(src_file)
    columns: list[pa.Array | pa.ChunkedArray] = []
    names: list[str] = []
    for name in table.column_names:
        if name in VECTOR_COLUMNS:
            columns.append(_drop_last_fixed_list(table[name]))
        else:
            columns.append(table[name])
        names.append(name)

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, names=names), dst_file)


def _drop_gripper_feature(feature: dict[str, Any]) -> None:
    names = feature.get("names")
    if not isinstance(names, list):
        raise ValueError(f"Expected feature names list, got {names!r}")
    if names[-1] != DROP_NAME:
        raise ValueError(f"Expected last feature name to be {DROP_NAME!r}, got {names[-1]!r}")
    feature["names"] = names[:-1]
    feature["shape"] = [5]


def convert_info(src: Path, dst: Path) -> None:
    info = json.loads(src.read_text())
    for key in VECTOR_COLUMNS:
        _drop_gripper_feature(info["features"][key])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(info, indent=4) + "\n")


def _truncate_stat_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 6 and all(isinstance(item, (int, float)) for item in value):
        return value[:-1]
    return value


def convert_stats(src: Path, dst: Path) -> None:
    stats = json.loads(src.read_text())
    for key in VECTOR_COLUMNS:
        for stat_name, stat_value in list(stats[key].items()):
            stats[key][stat_name] = _truncate_stat_value(stat_value)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(stats, indent=4) + "\n")


def convert_dataset(src_root: Path, dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(dst_root)

    for parquet in sorted((src_root / "data").glob("chunk-*/*.parquet")):
        convert_parquet(parquet, dst_root / parquet.relative_to(src_root))

    convert_info(src_root / "meta" / "info.json", dst_root / "meta" / "info.json")
    convert_stats(src_root / "meta" / "stats.json", dst_root / "meta" / "stats.json")

    for meta_file in (src_root / "meta").iterdir():
        if meta_file.name in {"info.json", "stats.json"}:
            continue
        target = dst_root / "meta" / meta_file.name
        if meta_file.is_dir():
            _copy_tree(meta_file, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(meta_file, target)

    _copy_tree(src_root / "videos", dst_root / "videos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("datanew/so101_single"),
        help="Source LeRobot dataset root.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("datanew/so101_single_no_gripper"),
        help="Output LeRobot dataset root.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output directory if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(args.src, args.dst, args.overwrite)
    print(f"Wrote no-gripper dataset to {args.dst}")


if __name__ == "__main__":
    main()
