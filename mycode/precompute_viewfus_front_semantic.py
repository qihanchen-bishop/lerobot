#!/usr/bin/env python3
"""Precompute the label-derived fused-front semantic stream for ViewFus-v1."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

try:
    from mycode.generate_modified_front_semantic_video import (
        AlphaBetaTracker,
        CLASSES,
        OnlineObjectTemplate,
        semantic_image,
    )
    from mycode.visualize_side_to_front_homography import (
        binary_mask,
        project_points,
        right_bottom_point,
        video_path,
    )
except ModuleNotFoundError:
    from generate_modified_front_semantic_video import (
        AlphaBetaTracker,
        CLASSES,
        OnlineObjectTemplate,
        semantic_image,
    )
    from visualize_side_to_front_homography import (
        binary_mask,
        project_points,
        right_bottom_point,
        video_path,
    )


FUSED_FRONT_KEY = "observation.images.front_fused_semantic"


@dataclass
class EpisodeSummary:
    episode: int
    frames: int
    exposure_frame: int | None
    latent_frames: int
    recovered_frames: int
    template_samples: int
    template_width_px: float
    template_height_px: float
    rgb_histogram: list[list[int]]
    output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bettersetup"))
    parser.add_argument(
        "--homography-json",
        type=Path,
        default=Path("outputs/side_to_front_homography/homography_side_to_front.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--minimum-object-area", type=int, default=200)
    parser.add_argument("--exposure-confirm-frames", type=int, default=5)
    parser.add_argument("--motion-only-frames", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return capture


def valid_existing_video(path: Path, expected_frames: int) -> bool:
    if not path.is_file():
        return False
    capture = cv2.VideoCapture(str(path))
    valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames
    capture.release()
    return valid


def video_rgb_histogram(path: Path) -> list[list[int]]:
    capture = open_video(path)
    histogram = np.zeros((3, 256), dtype=np.int64)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = frame[..., ::-1]
        for channel in range(3):
            histogram[channel] += np.bincount(rgb[..., channel].ravel(), minlength=256)
    capture.release()
    return histogram.tolist()


def generate_episode(
    dataset_root: str,
    episode: int,
    homography_values: list[list[float]],
    minimum_object_area: int,
    exposure_confirm_frames: int,
    motion_only_frames: int,
    overwrite: bool,
) -> EpisodeSummary:
    root = Path(dataset_root)
    homography = np.asarray(homography_values, dtype=np.float64)
    output_path = root / "videos" / FUSED_FRONT_KEY / "chunk-000" / f"file-{episode:03d}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    captures: dict[tuple[str, str], cv2.VideoCapture] = {}
    for class_name in CLASSES:
        captures[("front", class_name)] = open_video(video_path(root, "front", class_name, episode))
    captures[("side", "object")] = open_video(video_path(root, "side", "object", episode))
    frame_count = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures.values())
    fps = captures[("front", "object")].get(cv2.CAP_PROP_FPS) or 30.0
    width = int(captures[("front", "object")].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captures[("front", "object")].get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not overwrite and valid_existing_video(output_path, frame_count):
        for capture in captures.values():
            capture.release()
        return EpisodeSummary(
            episode, frame_count, None, 0, 0, 0, 0.0, 0.0, video_rgb_histogram(output_path), str(output_path)
        )

    template = OnlineObjectTemplate()
    tracker = AlphaBetaTracker()
    online_bias: np.ndarray | None = None
    paired_run = 0
    exposed = False
    exposure_frame: int | None = None
    last_side_update = -10_000
    latent_frames = 0
    recovered_frames = 0
    rgb_histogram = np.zeros((3, 256), dtype=np.int64)

    with tempfile.TemporaryDirectory(prefix=f"viewfus-{episode:03d}-") as temporary_dir:
        intermediate = Path(temporary_dir) / "semantic.mp4"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create temporary video: {intermediate}")

        for frame_index in range(frame_count):
            masks: dict[str, dict[str, np.ndarray]] = {"front": {}, "side": {}}
            for key, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to read {key} at frame {frame_index}.")
                masks[key[0]][key[1]] = binary_mask(frame)

            front_object = masks["front"]["object"]
            side_object = masks["side"]["object"]
            front_valid = int(front_object.sum()) >= minimum_object_area
            side_valid = int(side_object.sum()) >= minimum_object_area
            front_anchor = right_bottom_point(front_object) if front_valid else None
            side_anchor = right_bottom_point(side_object) if side_valid else None
            projected_side = (
                project_points(side_anchor[None], homography)[0] if side_anchor is not None else None
            )

            paired_run = paired_run + 1 if front_valid and side_valid else 0
            if not exposed and paired_run >= exposure_confirm_frames:
                exposed = True
                exposure_frame = frame_index - exposure_confirm_frames + 1
                assert front_anchor is not None and projected_side is not None
                online_bias = front_anchor - projected_side
                tracker.update(front_anchor, primary=True)

            tracker.predict()
            display_object: np.ndarray | None = None
            if not exposed:
                if projected_side is not None:
                    latent_frames += 1
            elif front_anchor is not None:
                tracker.update(front_anchor, primary=True)
                template.update(front_object)
                display_object = front_object
                if projected_side is not None:
                    residual = front_anchor - projected_side
                    if np.linalg.norm(residual) < 80:
                        online_bias = residual if online_bias is None else 0.95 * online_bias + 0.05 * residual
            elif projected_side is not None:
                corrected_side = projected_side + (0.0 if online_bias is None else online_bias)
                position = tracker.update(corrected_side, primary=False)
                display_object = template.render((height, width), position)
                last_side_update = frame_index
                recovered_frames += 1
            elif tracker.position is not None and frame_index - last_side_update <= motion_only_frames:
                display_object = template.render((height, width), tracker.position)

            fused = semantic_image(masks["front"], display_object)
            if not exposed and projected_side is not None:
                point = tuple(np.rint(projected_side).astype(int))
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    cv2.drawMarker(fused, point, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
            rgb = fused[..., ::-1]
            for channel in range(3):
                rgb_histogram[channel] += np.bincount(rgb[..., channel].ravel(), minlength=256)
            writer.write(fused)

        writer.release()
        for capture in captures.values():
            capture.release()
        temporary_output = output_path.with_suffix(".building.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(intermediate),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "10",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ],
            check=True,
        )
        os.replace(temporary_output, output_path)

    return EpisodeSummary(
        episode=episode,
        frames=frame_count,
        exposure_frame=exposure_frame,
        latent_frames=latent_frames,
        recovered_frames=recovered_frames,
        template_samples=template.count,
        template_width_px=template.mean_width,
        template_height_px=template.mean_height,
        rgb_histogram=rgb_histogram.tolist(),
        output=str(output_path),
    )


def histogram_stats(summaries: list[EpisodeSummary]) -> dict[str, list[float] | list[int]]:
    histogram = np.asarray([item.rgb_histogram for item in summaries], dtype=np.int64).sum(axis=0)
    values = np.arange(256, dtype=np.float64)
    counts = histogram.sum(axis=1)
    if np.any(counts == 0):
        raise RuntimeError("Cannot compute fused semantic statistics from an empty histogram.")
    mean = (histogram * values).sum(axis=1) / counts
    variance = (histogram * np.square(values)).sum(axis=1) / counts - np.square(mean)

    def quantile(probability: float) -> list[float]:
        targets = np.ceil(probability * counts).astype(np.int64).clip(min=1)
        return [
            float(np.searchsorted(np.cumsum(histogram[channel]), targets[channel]) / 255.0)
            for channel in range(3)
        ]

    return {
        "min": [float(np.flatnonzero(row)[0] / 255.0) for row in histogram],
        "max": [float(np.flatnonzero(row)[-1] / 255.0) for row in histogram],
        "mean": (mean / 255.0).tolist(),
        "std": (np.sqrt(np.maximum(variance, 0.0)) / 255.0).tolist(),
        "count": counts.tolist(),
        "q01": quantile(0.01),
        "q10": quantile(0.10),
        "q50": quantile(0.50),
        "q90": quantile(0.90),
        "q99": quantile(0.99),
    }


def update_dataset_metadata(dataset_root: Path, fps: int, summaries: list[EpisodeSummary]) -> None:
    info_path = dataset_root / "meta" / "info.json"
    stats_path = dataset_root / "meta" / "stats.json"
    info = json.loads(info_path.read_text())
    reference = info["features"]["observation.images.front"]
    feature = {
        "dtype": "video",
        "shape": list(reference["shape"]),
        "names": list(reference["names"]),
        "info": {
            "video.height": int(reference["shape"][0]),
            "video.width": int(reference["shape"][1]),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }
    info["features"][FUSED_FRONT_KEY] = feature
    temporary_info = info_path.with_suffix(".json.tmp")
    temporary_info.write_text(json.dumps(info, indent=4) + "\n")
    os.replace(temporary_info, info_path)

    stats = json.loads(stats_path.read_text())
    stats[FUSED_FRONT_KEY] = histogram_stats(summaries)
    temporary_stats = stats_path.with_suffix(".json.tmp")
    temporary_stats.write_text(json.dumps(stats, indent=4) + "\n")
    os.replace(temporary_stats, stats_path)

    episode_root = dataset_root / "meta" / "episodes"
    source_prefix = "videos/observation.images.front"
    target_prefix = f"videos/{FUSED_FRONT_KEY}"
    for parquet_path in sorted(episode_root.glob("chunk-*/*.parquet")):
        table = pq.read_table(parquet_path)
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            source_name = f"{source_prefix}/{suffix}"
            target_name = f"{target_prefix}/{suffix}"
            values = table[source_name]
            if target_name in table.column_names:
                table = table.set_column(table.column_names.index(target_name), target_name, values)
            else:
                table = table.append_column(target_name, values)
        temporary_parquet = parquet_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_parquet)
        os.replace(temporary_parquet, parquet_path)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    dataset_root = args.dataset_root.expanduser().resolve()
    homography_path = args.homography_json.expanduser().resolve()
    payload = json.loads(homography_path.read_text())
    anchor = payload.get("calibration", {}).get("anchor")
    if anchor != "side_right_bottom_to_front_right_bottom":
        raise ValueError(f"Expected right-bottom homography, got {anchor!r} from {homography_path}.")
    homography_values = payload["homography_side_to_front"]

    front_dir = dataset_root / "videos" / "observation.images.front" / "chunk-000"
    episodes = sorted(int(path.stem.split("-")[-1]) for path in front_dir.glob("file-*.mp4"))
    summaries: list[EpisodeSummary] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_episode,
                str(dataset_root),
                episode,
                homography_values,
                args.minimum_object_area,
                args.exposure_confirm_frames,
                args.motion_only_frames,
                args.overwrite,
            ): episode
            for episode in episodes
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="ViewFus-v1 labels"):
            summaries.append(future.result())

    summaries.sort(key=lambda item: item.episode)
    with open(dataset_root / "meta" / "info.json") as file:
        fps = int(json.load(file)["fps"])
    update_dataset_metadata(dataset_root, fps, summaries)
    summary = {
        "feature_key": FUSED_FRONT_KEY,
        "dataset_root": str(dataset_root),
        "homography": str(homography_path),
        "homography_method": payload.get("method"),
        "homography_anchor": anchor,
        "episodes": [asdict(item) for item in summaries],
    }
    summary_path = dataset_root / "viewfus_v1_precompute.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Generated {len(summaries)} episodes for {FUSED_FRONT_KEY}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
