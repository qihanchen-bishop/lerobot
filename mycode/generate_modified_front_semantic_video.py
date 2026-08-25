#!/usr/bin/env python3
"""Render a side-assisted canonical front semantic video.

The three panels are the front RGB observation, side RGB observation, and a
modified front semantic map. Before stable two-view exposure, a red cross marks
the latent object position projected from the side view. Once both views have
matched, the online front template is used to recover later front occlusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

try:
    from mycode.visualize_side_to_front_homography import (
        binary_mask,
        project_points,
        right_bottom_point,
        video_path,
    )
except ModuleNotFoundError:
    from visualize_side_to_front_homography import (
        binary_mask,
        project_points,
        right_bottom_point,
        video_path,
    )


CLASSES = ("occluder", "object", "region", "tool")
PALETTE_BGR = {
    "occluder": (255, 160, 50),
    "object": (80, 80, 255),
    "region": (120, 220, 60),
    "tool": (80, 210, 210),
}
HEADER_HEIGHT = 52


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bettersetup"))
    parser.add_argument(
        "--homography-json",
        type=Path,
        default=Path("outputs/side_to_front_homography/homography_side_to_front.json"),
    )
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/side_to_front_homography/episode_005_modified_front.mp4"),
    )
    parser.add_argument("--minimum-object-area", type=int, default=200)
    parser.add_argument("--exposure-confirm-frames", type=int, default=5)
    parser.add_argument("--motion-only-frames", type=int, default=30)
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return capture


def mask_from_frame(frame: np.ndarray) -> np.ndarray:
    return binary_mask(frame)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


class OnlineObjectTemplate:
    def __init__(self, canonical_size: int = 64) -> None:
        self.canonical_size = canonical_size
        self.mean_shape: np.ndarray | None = None
        self.mean_width = 35.0
        self.mean_height = 33.0
        self.count = 0

    def update(self, mask: np.ndarray) -> bool:
        component = largest_component(mask)
        ys, xs = np.nonzero(component)
        if len(xs) < 200:
            return False
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        width, height = x1 - x0, y1 - y0
        if not (15 <= width <= 60 and 12 <= height <= 55):
            return False
        normalized = cv2.resize(
            component[y0:y1, x0:x1].astype(np.float32),
            (self.canonical_size, self.canonical_size),
            interpolation=cv2.INTER_AREA,
        )
        self.count += 1
        if self.mean_shape is None:
            self.mean_shape = normalized
            self.mean_width = float(width)
            self.mean_height = float(height)
        else:
            rate = min(0.10, 1.0 / self.count)
            self.mean_shape = (1.0 - rate) * self.mean_shape + rate * normalized
            self.mean_width = (1.0 - rate) * self.mean_width + rate * width
            self.mean_height = (1.0 - rate) * self.mean_height + rate * height
        return True

    def render(self, shape: tuple[int, int], right_bottom: np.ndarray) -> np.ndarray:
        output = np.zeros(shape, dtype=bool)
        width = max(1, round(self.mean_width))
        height = max(1, round(self.mean_height))
        if self.mean_shape is None:
            normalized = np.ones((self.canonical_size, self.canonical_size), dtype=np.float32)
        else:
            normalized = self.mean_shape
        template = cv2.resize(normalized, (width, height), interpolation=cv2.INTER_LINEAR) >= 0.45
        anchor_x, anchor_y = np.rint(right_bottom).astype(int)
        x0 = anchor_x - width + 1
        y0 = anchor_y - height + 1
        x1, y1 = x0 + width, y0 + height
        dst_x0, dst_y0 = max(0, x0), max(0, y0)
        dst_x1, dst_y1 = min(shape[1], x1), min(shape[0], y1)
        if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
            return output
        src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
        src_x1, src_y1 = src_x0 + (dst_x1 - dst_x0), src_y0 + (dst_y1 - dst_y0)
        output[dst_y0:dst_y1, dst_x0:dst_x1] = template[src_y0:src_y1, src_x0:src_x1]
        return output


class AlphaBetaTracker:
    def __init__(self) -> None:
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float32)

    def predict(self) -> np.ndarray | None:
        if self.position is not None:
            self.position = self.position + self.velocity
        return None if self.position is None else self.position.copy()

    def update(self, measurement: np.ndarray, *, primary: bool) -> np.ndarray:
        measurement = measurement.astype(np.float32)
        if self.position is None:
            self.position = measurement.copy()
            self.velocity.fill(0.0)
            return self.position.copy()
        alpha, beta = (0.70, 0.16) if primary else (0.38, 0.06)
        innovation = measurement - self.position
        self.position = self.position + alpha * innovation
        self.velocity = 0.92 * self.velocity + beta * innovation
        return self.position.copy()


def semantic_image(masks: dict[str, np.ndarray], object_mask: np.ndarray | None) -> np.ndarray:
    height, width = next(iter(masks.values())).shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_name in ("region", "occluder", "tool"):
        image[masks[class_name]] = PALETTE_BGR[class_name]
    if object_mask is not None:
        image[object_mask] = PALETTE_BGR["object"]
    return image


def overlay_semantics(rgb: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    semantic = semantic_image(masks, masks["object"])
    selected = np.any(semantic != 0, axis=2)
    blended = cv2.addWeighted(rgb, 0.58, semantic, 0.42, 0.0)
    output = rgb.copy()
    output[selected] = blended[selected]
    return output


def title_panel(image: np.ndarray, title: str, detail: str) -> np.ndarray:
    output = np.zeros((image.shape[0] + HEADER_HEIGHT, image.shape[1], 3), dtype=np.uint8)
    output[HEADER_HEIGHT:] = image
    cv2.putText(output, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(output, detail, (10, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (210, 210, 210), 1)
    return output


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    homography_path = args.homography_json.expanduser().resolve()
    with homography_path.open() as file:
        homography_payload = json.load(file)
    calibration_anchor = homography_payload.get("calibration", {}).get("anchor")
    if calibration_anchor != "side_right_bottom_to_front_right_bottom":
        raise ValueError(
            f"Expected a right-bottom anchored homography, got {calibration_anchor!r} from "
            f"{homography_path}."
        )
    homography = np.asarray(homography_payload["homography_side_to_front"], dtype=np.float64)

    captures: dict[tuple[str, str | None], cv2.VideoCapture] = {}
    for camera in ("front", "side"):
        captures[(camera, None)] = open_video(video_path(dataset_root, camera, None, args.episode))
        for class_name in CLASSES:
            captures[(camera, class_name)] = open_video(
                video_path(dataset_root, camera, class_name, args.episode)
            )
    frame_count = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) for capture in captures.values())
    fps = captures[("front", None)].get(cv2.CAP_PROP_FPS) or 30.0
    width = int(captures[("front", None)].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captures[("front", None)].get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = OnlineObjectTemplate()
    tracker = AlphaBetaTracker()
    online_bias: np.ndarray | None = None
    paired_run = 0
    exposed = False
    exposure_frame: int | None = None
    last_side_update = -10_000
    recovered_frames = 0
    latent_frames = 0

    with tempfile.TemporaryDirectory(prefix="modified-front-") as temporary_dir:
        intermediate = Path(temporary_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (3 * width, height + HEADER_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create temporary video: {intermediate}")

        for frame_index in range(frame_count):
            frames: dict[tuple[str, str | None], np.ndarray] = {}
            for key, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to read {key} at frame {frame_index}.")
                frames[key] = frame
            masks = {
                camera: {
                    class_name: mask_from_frame(frames[(camera, class_name)])
                    for class_name in CLASSES
                }
                for camera in ("front", "side")
            }
            front_area = int(masks["front"]["object"].sum())
            side_area = int(masks["side"]["object"].sum())
            front_valid = front_area >= args.minimum_object_area
            side_valid = side_area >= args.minimum_object_area
            front_anchor = right_bottom_point(masks["front"]["object"]) if front_valid else None
            side_anchor = right_bottom_point(masks["side"]["object"]) if side_valid else None
            projected_side = (
                project_points(side_anchor[None], homography)[0] if side_anchor is not None else None
            )

            paired_run = paired_run + 1 if front_valid and side_valid else 0
            if not exposed and paired_run >= args.exposure_confirm_frames:
                exposed = True
                exposure_frame = frame_index - args.exposure_confirm_frames + 1
                assert front_anchor is not None and projected_side is not None
                online_bias = front_anchor - projected_side
                tracker.update(front_anchor, primary=True)

            tracker.predict()
            display_object: np.ndarray | None = None
            status = "NO SIDE OBJECT"
            if not exposed:
                if projected_side is not None:
                    status = "PRE-EXPOSE: SIDE-PROJECTED LATENT POSITION"
                    latent_frames += 1
                else:
                    status = "PRE-EXPOSE: OBJECT UNKNOWN"
            else:
                if front_anchor is not None:
                    tracker.update(front_anchor, primary=True)
                    template.update(masks["front"]["object"])
                    display_object = masks["front"]["object"]
                    status = "OBSERVED: FRONT OBJECT MASK"
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
                    status = "RECOVERED: SIDE + VELOCITY + FRONT TEMPLATE"
                elif (
                    tracker.position is not None
                    and frame_index - last_side_update <= args.motion_only_frames
                ):
                    display_object = template.render((height, width), tracker.position)
                    status = "PREDICTED: SHORT MOTION-ONLY HOLD"
                else:
                    status = "LOST: NO RELIABLE OBJECT OBSERVATION"

            modified = semantic_image(masks["front"], display_object)
            if not exposed and projected_side is not None:
                point = tuple(np.rint(projected_side).astype(int))
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    cv2.drawMarker(modified, point, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)

            front_panel = title_panel(
                overlay_semantics(frames[("front", None)], masks["front"]),
                "FRONT RGB + ORIGINAL MASKS",
                f"frame={frame_index} front_object_area={front_area}",
            )
            side_panel = title_panel(
                overlay_semantics(frames[("side", None)], masks["side"]),
                "SIDE RGB + ORIGINAL MASKS",
                f"frame={frame_index} side_object_area={side_area}",
            )
            modified_panel = title_panel(
                modified,
                "MODIFIED FRONT SEMANTIC",
                status,
            )
            writer.write(np.hstack([front_panel, side_panel, modified_panel]))

        writer.release()
        for capture in captures.values():
            capture.release()
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
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )

    summary = {
        "episode": args.episode,
        "frames": frame_count,
        "fps": fps,
        "exposure_frame": exposure_frame,
        "latent_position_frames": latent_frames,
        "recovered_template_frames": recovered_frames,
        "online_template_samples": template.count,
        "online_template_width_px": template.mean_width,
        "online_template_height_px": template.mean_height,
        "homography_method": homography_payload.get("method"),
        "homography_anchor": calibration_anchor,
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
