#!/usr/bin/env python
"""Create a resized copy of a LeRobot v3 dataset without modifying the source."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


DEFAULT_SOURCE = Path("data/soarmcube277_mask_task1")
DEFAULT_OUTPUT = Path("data/soarmcube277_mask_task1_480x640")
DEFAULT_RGB_KEYS = ["observation.images.left_front", "observation.images.camera"]


@dataclass(frozen=True)
class VideoJob:
    feature_key: str
    source: Path
    destination: Path
    is_mask: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--rgb-keys",
        nargs="+",
        default=DEFAULT_RGB_KEYS,
        help="Video feature keys resized with bilinear filtering. Other video features use nearest-neighbor.",
    )
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--rgb-crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument(
        "--gop-size",
        type=int,
        default=2,
        help="Keyframe interval. Small values make random frame access much faster at the cost of larger files.",
    )
    parser.add_argument("--ffmpeg", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_executable(explicit: Path | None, name: str) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"{name} executable does not exist: {candidate}")
        return candidate

    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)

    candidates = [
        Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / name,
        Path.home() / "anaconda3" / "envs" / "lerobot" / "bin" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {name}. Pass --ffmpeg /path/to/ffmpeg.")


def load_info(source: Path) -> dict[str, Any]:
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    with open(info_path) as f:
        return json.load(f)


def video_feature_keys(info: dict[str, Any]) -> list[str]:
    return sorted(key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video")


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if source == output:
        raise ValueError("--output must be different from --source.")
    if source in output.parents or output in source.parents:
        raise ValueError("Source and output directories must not contain one another.")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 yuv420p output requires even --height and --width values.")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive.")
    if args.gop_size <= 0:
        raise ValueError("--gop-size must be positive.")
    return source, output


def prepare_output(source: Path, output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)

    output.mkdir(parents=True)
    for item in source.iterdir():
        if item.name == "videos":
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    (output / "videos").mkdir()


def collect_jobs(source: Path, output: Path, feature_keys: list[str], rgb_keys: set[str]) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for feature_key in feature_keys:
        feature_dir = source / "videos" / feature_key
        if not feature_dir.is_dir():
            raise FileNotFoundError(f"Video feature directory not found: {feature_dir}")
        videos = sorted(feature_dir.rglob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No MP4 files found for video feature: {feature_key}")
        for video in videos:
            jobs.append(
                VideoJob(
                    feature_key=feature_key,
                    source=video,
                    destination=output / video.relative_to(source),
                    is_mask=feature_key not in rgb_keys,
                )
            )
    return jobs


def transcode_video(
    job: VideoJob,
    ffmpeg: Path,
    width: int,
    height: int,
    preset: str,
    rgb_crf: int,
    gop_size: int,
) -> None:
    job.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = job.destination.with_suffix(".tmp.mp4")
    temporary.unlink(missing_ok=True)
    scale_flags = "neighbor" if job.is_mask else "bilinear"
    crf = "0" if job.is_mask else str(rgb_crf)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(job.source),
        "-map",
        "0:v:0",
        "-vf",
        f"scale={width}:{height}:flags={scale_flags}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        crf,
        "-g",
        str(gop_size),
        "-keyint_min",
        str(gop_size),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-threads",
        "1",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced an empty video: {temporary}")
        temporary.replace(job.destination)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        detail = exc.stderr.strip() if exc.stderr else "unknown ffmpeg error"
        raise RuntimeError(f"Failed to resize {job.source}: {detail}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def probe_dimensions(ffprobe: Path, video: Path) -> tuple[int, int]:
    output = subprocess.check_output(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video),
        ],
        text=True,
    ).strip()
    width, height = output.split("x")
    return int(width), int(height)


def update_metadata(
    output: Path,
    info: dict[str, Any],
    feature_keys: list[str],
    width: int,
    height: int,
    source: Path,
    rgb_keys: set[str],
    gop_size: int,
) -> None:
    for key in feature_keys:
        feature = info["features"][key]
        feature["shape"] = [height, width, 3]
        feature_info = feature.setdefault("info", {})
        feature_info["video.height"] = height
        feature_info["video.width"] = width
        feature_info["video.codec"] = "h264"
        feature_info["video.pix_fmt"] = "yuv420p"

    video_bytes = sum(path.stat().st_size for path in (output / "videos").rglob("*.mp4"))
    info["video_files_size_in_mb"] = round(video_bytes / 1_000_000, 3)
    with open(output / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")

    conversion = {
        "source": str(source),
        "height": height,
        "width": width,
        "rgb_keys": sorted(rgb_keys.intersection(feature_keys)),
        "mask_keys": sorted(set(feature_keys).difference(rgb_keys)),
        "gop_size": gop_size,
        "note": "stats.json was copied from the source dataset; channel normalization statistics are unchanged.",
    }
    with open(output / "meta" / "resize_conversion.json", "w") as f:
        json.dump(conversion, f, indent=4)
        f.write("\n")


def main() -> None:
    args = parse_args()
    source, output = validate_args(args)
    ffmpeg = find_executable(args.ffmpeg, "ffmpeg")
    ffprobe = find_executable(ffmpeg.with_name("ffprobe"), "ffprobe")
    info = load_info(source)
    feature_keys = video_feature_keys(info)
    if not feature_keys:
        raise ValueError(f"Dataset contains no video features: {source}")

    rgb_keys = set(args.rgb_keys)
    unknown_rgb_keys = rgb_keys.difference(feature_keys)
    if unknown_rgb_keys:
        print(f"Ignoring RGB keys not present in this dataset: {sorted(unknown_rgb_keys)}")

    print(f"Source: {source}")
    print(f"Output: {output}")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Video features: {feature_keys}")
    print(f"ffmpeg: {ffmpeg}")
    prepare_output(source, output, args.overwrite)
    jobs = collect_jobs(source, output, feature_keys, rgb_keys)

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    transcode_video,
                    job,
                    ffmpeg,
                    args.width,
                    args.height,
                    args.preset,
                    args.rgb_crf,
                    args.gop_size,
                ): job
                for job in jobs
            }
            with tqdm(total=len(jobs), desc="resize videos", unit="video", dynamic_ncols=True) as progress:
                for future in as_completed(futures):
                    future.result()
                    progress.update()

        for job in tqdm(jobs, desc="verify videos", unit="video", dynamic_ncols=True):
            dimensions = probe_dimensions(ffprobe, job.destination)
            if dimensions != (args.width, args.height):
                raise RuntimeError(f"Unexpected dimensions for {job.destination}: {dimensions}")

        update_metadata(output, info, feature_keys, args.width, args.height, source, rgb_keys, args.gop_size)
    except Exception:
        print(f"Conversion failed. Partial output remains at: {output}")
        raise

    print(f"Converted {len(jobs)} videos successfully.")
    print(f"Resized dataset: {output}")


if __name__ == "__main__":
    main()
