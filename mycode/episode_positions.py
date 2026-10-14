"""Manual initial-object grid labels, independent of LeRobot model features."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

POSITIONS_PATH = Path("meta/episode_positions.jsonl")


def grid_position(rows: int, cols: int, cell: str) -> dict:
    if not 1 <= rows <= 20 or not 1 <= cols <= 20:
        raise ValueError("网格行数和列数必须在 1–20 之间。")
    match = re.fullmatch(r"\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*", cell)
    if match is None:
        raise ValueError("请先选择物体初始位置，再点击 Record。")
    x, y = map(int, match.groups())
    if x >= cols or y >= rows:
        raise ValueError("所选位置超出了当前网格范围。")
    return {
        "grid_rows": rows,
        "grid_cols": cols,
        "grid_x": x,
        "grid_y": y,
        "grid_cell": f"({x},{y})",
        "grid_origin": "top_left",
        "grid_x_direction": "right",
        "grid_y_direction": "down",
        "position_role": "object_initial_grid",
        "position_source": "manual_selection",
    }


def load_episode_positions(root: Path) -> dict[int, dict]:
    path = root / POSITIONS_PATH
    if not path.exists():
        return {}
    records = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            index = record["episode_index"]
            if type(index) is not int or index < 0 or index in records:
                raise ValueError("invalid or duplicate episode_index")
            position = grid_position(record["grid_rows"], record["grid_cols"], record["grid_cell"])
            if any(record.get(key) != position[key] for key in ("grid_x", "grid_y")):
                raise ValueError("grid coordinates disagree with grid_cell")
            records[index] = record
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid position metadata in {path}, line {line_number}: {exc}") from exc
    return records


def write_episode_positions(root: Path, records: dict[int, dict]) -> None:
    path = root / POSITIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for index in sorted(records):
                stream.write(json.dumps(records[index], ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_episode_position(root: Path, record: dict) -> None:
    records = load_episode_positions(root)
    records[record["episode_index"]] = record
    write_episode_positions(root, records)


def reindex_episode_positions(source: Path, destination: Path, old_to_new: dict[int, int]) -> None:
    if not (source / POSITIONS_PATH).exists():
        return
    records = load_episode_positions(source)
    remapped = {
        new: {**records[old], "episode_index": new} for old, new in old_to_new.items() if old in records
    }
    write_episode_positions(destination, remapped)


def position_counts(records: dict[int, dict]) -> list[dict]:
    counts = Counter(
        (record["task"], record["grid_rows"], record["grid_cols"], record["grid_y"], record["grid_x"])
        for record in records.values()
    )
    return [
        {"task": task, "grid_rows": rows, "grid_cols": cols, "grid_cell": f"({x},{y})", "episodes": count}
        for (task, rows, cols, y, x), count in sorted(counts.items())
    ]


def export_position_counts(path: Path, records: dict[int, dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("task", "grid_rows", "grid_cols", "grid_cell", "episodes")
        )
        writer.writeheader()
        writer.writerows(position_counts(records))
