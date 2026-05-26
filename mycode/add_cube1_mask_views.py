#!/usr/bin/env python
"""Add SAM2 mask videos as LeRobot visual features for simdata/cube1."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


FEATURE_SPECS = {
    "mask_frames": {
        "key": "observation.images.mask_frames",
        "source_dir": "mask_frames",
        "pattern": "%05d.png",
        "lossless": True,
    },
    "mask_visual_frames": {
        "key": "observation.images.mask_visual_frames",
        "source_dir": "mask_visual_frames",
        "pattern": "%05d.png",
        "lossless": False,
    },
    "occluder": {
        "key": "observation.images.occluder",
        "source_dir": "object_mask_frames",
        "object_name": "occluder",
        "lossless": True,
    },
    "object": {
        "key": "observation.images.object",
        "source_dir": "object_mask_frames",
        "object_name": "object",
        "lossless": True,
    },
    "region": {
        "key": "observation.images.region",
        "source_dir": "object_mask_frames",
        "object_name": "region",
        "lossless": True,
    },
    "left_arm": {
        "key": "observation.images.left_arm",
        "source_dir": "object_mask_frames",
        "object_name": "leftarm",
        "lossless": True,
    },
    "right_arm": {
        "key": "observation.images.right_arm",
        "source_dir": "object_mask_frames",
        "object_name": "rightarm",
        "lossless": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("simdata/cube1"))
    parser.add_argument("--task-dir", default="task1")
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ffprobe_frame_count(path: Path) -> int:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return int(output)


def object_ids(mask_root: Path) -> dict[str, int]:
    with open(mask_root / "sam2_prompts.json") as f:
        prompts = json.load(f)
    return {obj["name"]: int(obj["id"]) for obj in prompts["objects"]}


def encode_png_sequence(
    input_pattern: Path,
    output_path: Path,
    fps: int,
    lossless: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-framerate",
        str(fps),
        "-i",
        str(input_pattern),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "0" if lossless else "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    if dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def feature_info(height: int, width: int, fps: int) -> dict:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def update_info_json(root: Path, feature_keys: list[str], dry_run: bool) -> None:
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    fps = int(info["fps"])
    camera_info = info["features"]["observation.images.camera"]
    height, width = camera_info["shape"][0], camera_info["shape"][1]
    for key in feature_keys:
        info["features"][key] = feature_info(height=height, width=width, fps=fps)

    if dry_run:
        print(f"Would update {info_path} with: {feature_keys}")
        return

    backup_path = info_path.with_suffix(".json.bak_before_masks")
    if not backup_path.exists():
        shutil.copy2(info_path, backup_path)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def neutral_image_stats(count: int) -> dict:
    zero = [[[0.0]], [[0.0]], [[0.0]]]
    half = [[[0.5]], [[0.5]], [[0.5]]]
    one = [[[1.0]], [[1.0]], [[1.0]]]
    return {
        "min": zero,
        "max": one,
        "mean": half,
        "std": half,
        "count": [count],
        "q01": zero,
        "q10": zero,
        "q50": half,
        "q90": one,
        "q99": one,
    }


def update_stats_json(root: Path, feature_keys: list[str], dry_run: bool) -> None:
    stats_path = root / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)

    count = stats.get("observation.images.camera", {}).get("count", [0])[0]
    for key in feature_keys:
        stats[key] = neutral_image_stats(count=count)

    if dry_run:
        print(f"Would update {stats_path} with: {feature_keys}")
        return

    backup_path = stats_path.with_suffix(".json.bak_before_masks")
    if not backup_path.exists():
        shutil.copy2(stats_path, backup_path)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")


def update_episodes_metadata(root: Path, feature_keys: list[str], dry_run: bool) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "Updating meta/episodes requires pandas. Run this script in the lerobot conda environment."
        ) from exc

    camera_key = "observation.images.camera"
    episodes_root = root / "meta" / "episodes"
    episode_files = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
    if dry_run:
        print(f"Would update episode metadata files: {[str(path) for path in episode_files]}")
        return

    for path in episode_files:
        backup_path = path.with_suffix(".parquet.bak_before_masks")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

        df = pd.read_parquet(path)
        for key in feature_keys:
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                src_col = f"videos/{camera_key}/{suffix}"
                dst_col = f"videos/{key}/{suffix}"
                if dst_col not in df.columns:
                    df[dst_col] = df[src_col]

            for src_col in [col for col in df.columns if col.startswith(f"stats/{camera_key}/")]:
                dst_col = src_col.replace(f"stats/{camera_key}/", f"stats/{key}/", 1)
                if dst_col not in df.columns:
                    df[dst_col] = df[src_col]

        df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    root = args.root
    task_root = root / args.task_dir
    chunk_dir_name = f"chunk-{args.chunk_index:03d}"
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    total_episodes = int(info["total_episodes"])

    for episode_index in range(total_episodes):
        episode_name = f"file-{episode_index:03d}"
        mask_root = task_root / f"{episode_name}_sam2_masks"
        ids_by_name = object_ids(mask_root)
        original_video = root / "videos" / "observation.images.camera" / chunk_dir_name / f"{episode_name}.mp4"
        frame_count = ffprobe_frame_count(original_video)

        for spec in FEATURE_SPECS.values():
            key = spec["key"]
            output_path = root / "videos" / key / chunk_dir_name / f"{episode_name}.mp4"
            if output_path.exists() and not args.overwrite:
                continue

            source_dir = mask_root / spec["source_dir"]
            if "object_name" in spec:
                object_id = ids_by_name[spec["object_name"]]
                pattern = source_dir / f"%05d_obj{object_id:03d}.png"
            else:
                pattern = source_dir / spec["pattern"]

            encode_png_sequence(
                input_pattern=pattern,
                output_path=output_path,
                fps=fps,
                lossless=spec["lossless"],
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

            if not args.dry_run:
                encoded_frames = ffprobe_frame_count(output_path)
                if encoded_frames != frame_count:
                    raise RuntimeError(
                        f"{output_path} has {encoded_frames} frames, expected {frame_count} from {original_video}"
                    )

        print(f"episode {episode_index:03d}: ok ({frame_count} frames)")

    feature_keys = [spec["key"] for spec in FEATURE_SPECS.values()]
    update_info_json(root, feature_keys, dry_run=args.dry_run)
    update_stats_json(root, feature_keys, dry_run=args.dry_run)
    update_episodes_metadata(root, feature_keys, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
