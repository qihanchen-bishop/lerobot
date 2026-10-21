#!/usr/bin/env python3
"""Calibrate evaluation from episode metadata and decoded front occluder endpoints.

Run with the lerobot Python environment. Never modifies the source dataset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import av
import numpy as np
import pandas as pd


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"min": float(values.min()), "mean": float(values.mean()),
            "max": float(values.max()), "p05": float(np.quantile(values, .05)),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95))}


def analyze(root: Path, output: Path):
    info = json.loads((root / "meta/info.json").read_text())
    episodes = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))])
    assert episodes.episode_index.is_unique
    assert len(episodes) == info["total_episodes"]
    assert int(episodes.length.sum()) == info["total_frames"]
    fps = info["fps"]
    key = "observation.images.front_occluder"

    def endpoint(row):
        prefix = f"videos/{key}/"
        path = root / info["video_path"].format(video_key=key, chunk_index=int(row[prefix + "chunk_index"]),
                                               file_index=int(row[prefix + "file_index"]))
        target = float(row[prefix + "from_timestamp"]) + (int(row["length"]) - 1) / fps
        # Seek to the preceding keyframe, then decode through the exact episode endpoint.
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(max(0, int((target - .5) / stream.time_base)), stream=stream)
            selected = None
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                if abs(t - target) <= .5 / fps:
                    selected = frame.to_ndarray(format="gray")
                    break
                if t > target + .5 / fps:
                    break
            if selected is None:
                raise ValueError(f"No exact last frame: episode={row['episode_index']} target={target} {path}")
        return {"episode_index": int(row["episode_index"]), "frames": int(row["length"]),
                "duration_s": int(row["length"]) / fps,
                "front_occluder_last_ratio": float((selected >= 128).mean()),
                "last_timestamp_s": target, "decoded_timestamp_s": t,
                "mask_width": selected.shape[1], "mask_height": selected.shape[0],
                "mask_video": str(path.relative_to(root))}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(endpoint, episodes.to_dict("records")))
    table = pd.DataFrame(rows).sort_values("episode_index")
    duration = summarize(table.duration_s)
    coverage = summarize(table.front_occluder_last_ratio)
    # Include every demonstration endpoint, with one percentage point of tolerance.
    threshold = max(0., math.floor((coverage["min"] - .01) * 100) / 100)
    timeout = int(math.ceil(duration["max"] / 5) * 5)
    result = {"dataset": str(root.resolve()), "episodes": len(table), "frames": int(table.frames.sum()),
              "fps": fps, "duration_definition": "episode length / nominal fps (recorded timeline, not measured wall time)",
              "mask_definition": "decoded grayscale >= 128; front occluder pixels / full frame pixels; exact last episode frame",
              "duration_s": duration, "front_occluder_last_ratio": coverage,
              "recommended_episode_time_s": timeout,
              "recommended_occluder_complete_ratio": threshold,
              "selection_rule": "timeout: maximum demonstration duration rounded up to 5s; coverage: minimum endpoint minus 1 percentage point, rounded down to 1%",
              "limitation": "Successful demonstration endpoints alone cannot measure false-positive rate. Area coverage does not certify original cloth pose or geometry."}
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "episodes.csv", index=False)
    (output / "calibration.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    # Companion notebook records both the exact reproducible entry point and its result.
    source = "import sys\nfrom pathlib import Path\nsys.path.insert(0, " + repr(str(Path(__file__).resolve().parents[1])) + ")\nfrom mycode.analyze_finaldata_eval import analyze\nresult = analyze(Path(" + repr(str(root.resolve())) + "), Path(" + repr(str(output.resolve())) + "))\nresult"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                "cells": [{"cell_type": "markdown", "metadata": {}, "id": "scope", "source": ["# finaldata_seg evaluation calibration\n", "One row per episode. Exact endpoint timestamps, binary threshold 128, full-frame denominator.\n", result["limitation"]]},
                          {"cell_type": "code", "metadata": {}, "id": "analysis", "execution_count": None, "source": source.splitlines(True),
                           "outputs": []}]}
    (output / "analysis.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/finaldata_seg"))
    parser.add_argument("--output", type=Path, default=Path("result/finaldata_eval_calibration_20261020"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), indent=2, ensure_ascii=False))
