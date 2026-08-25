#!/usr/bin/env python3
"""Estimate and visualize a tabletop side-to-front homography.

The calibration uses quality-filtered right-bottom anchors from synchronized
object masks. The resulting homography is suitable for object/region position
transfer, not exact 3D silhouette reconstruction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CAMERA_KEYS = ("front", "side")
MASK_CLASSES = ("object", "region")
CALIBRATION_FRACTIONS = np.linspace(0.15, 0.85, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bettersetup"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/side_to_front_homography"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=167)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--quality-score-min", type=float, default=0.95)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def video_path(dataset_root: Path, camera: str, semantic_class: str | None, episode: int) -> Path:
    key = f"observation.images.{camera}"
    if semantic_class is not None:
        key += f"_{semantic_class}"
    return dataset_root / "videos" / key / "chunk-000" / f"file-{episode:03d}.mp4"


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    return frame


def frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def binary_mask(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) > 127


def largest_component_stats(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[component]
    return int(x), int(y), int(width), int(height), int(area)


def right_bottom_point(mask: np.ndarray, *, require_calibration_shape: bool = False) -> np.ndarray | None:
    """Return the right-bottom anchor of the largest object component."""
    stats = largest_component_stats(mask)
    if stats is None:
        return None
    x, y, width, height, area = stats
    if area < 25:
        return None
    if require_calibration_shape and not (
        200 <= area <= 2500
        and 18 <= width <= 60
        and 15 <= height <= 55
        and area / (width * height) >= 0.60
    ):
        return None
    return np.asarray([x + width - 1, y + height - 1], dtype=np.float32)


def quality_lookup(dataset_root: Path, camera: str) -> dict[tuple[int, int], tuple[float, bool]]:
    quality_path = (
        dataset_root
        / "segmentation_quality"
        / f"observation_images_{camera}_object.npz"
    )
    if not quality_path.is_file():
        raise FileNotFoundError(f"Object quality archive is missing: {quality_path}")
    with np.load(quality_path) as archive:
        return {
            (int(episode), int(frame)): (float(score), bool(uncertain))
            for episode, frame, score, uncertain in zip(
                archive["episode_index"],
                archive["frame_index"],
                archive["quality_score"],
                archive["uncertain"],
                strict=True,
            )
        }


def episode_indices(dataset_root: Path) -> list[int]:
    paths = sorted(
        (dataset_root / "videos" / "observation.images.front_object" / "chunk-000").glob("file-*.mp4")
    )
    return [int(path.stem.split("-")[-1]) for path in paths]


def calibration_correspondences(
    dataset_root: Path,
    quality_score_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side_points: list[np.ndarray] = []
    front_points: list[np.ndarray] = []
    point_episodes: list[int] = []
    quality = {camera: quality_lookup(dataset_root, camera) for camera in CAMERA_KEYS}
    for episode in episode_indices(dataset_root):
        paths = {
            camera: video_path(dataset_root, camera, "object", episode) for camera in CAMERA_KEYS
        }
        count = min(frame_count(path) for path in paths.values())
        for fraction in CALIBRATION_FRACTIONS:
            frame_index = int(count * fraction)
            frame_quality = [quality[camera].get((episode, frame_index)) for camera in CAMERA_KEYS]
            if any(
                item is None or item[0] < quality_score_min or item[1]
                for item in frame_quality
            ):
                continue
            points = {}
            for camera, path in paths.items():
                points[camera] = right_bottom_point(
                    binary_mask(read_frame(path, frame_index)),
                    require_calibration_shape=True,
                )
            if points["front"] is None or points["side"] is None:
                continue
            side_points.append(points["side"])
            front_points.append(points["front"])
            point_episodes.append(episode)
    if len(side_points) < 4:
        raise RuntimeError("At least four valid side/front object correspondences are required.")
    return np.stack(side_points), np.stack(front_points), np.asarray(point_episodes)


def fit_homography(
    side_points: np.ndarray,
    front_points: np.ndarray,
    ransac_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    homography, inliers = cv2.findHomography(
        side_points,
        front_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )
    if homography is None or inliers is None:
        raise RuntimeError("Homography estimation failed.")
    return homography, inliers.reshape(-1).astype(bool)


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def validation_metrics(
    side_points: np.ndarray,
    front_points: np.ndarray,
    episodes: np.ndarray,
    *,
    holdout_ratio: float,
    seed: int,
    ransac_threshold: float,
) -> dict[str, float | int]:
    unique_episodes = np.unique(episodes)
    rng = np.random.default_rng(seed)
    holdout_count = max(1, round(len(unique_episodes) * holdout_ratio))
    holdout_episodes = rng.choice(unique_episodes, holdout_count, replace=False)
    train = ~np.isin(episodes, holdout_episodes)
    homography, inliers = fit_homography(side_points[train], front_points[train], ransac_threshold)
    projected = project_points(side_points[~train], homography)
    errors = np.linalg.norm(projected - front_points[~train], axis=1)
    return {
        "num_correspondences": int(len(side_points)),
        "num_train_correspondences": int(train.sum()),
        "num_holdout_correspondences": int((~train).sum()),
        "train_ransac_inlier_ratio": float(inliers.mean()),
        "holdout_error_median_px": float(np.median(errors)),
        "holdout_error_p90_px": float(np.percentile(errors, 90)),
        "holdout_error_mean_px": float(errors.mean()),
        "holdout_error_max_px": float(errors.max()),
    }


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (0, 0, 0), thickness=-1)
    cv2.putText(output, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    return output


def overlay_masks(image: np.ndarray, masks: dict[str, np.ndarray], alpha: float = 0.45) -> np.ndarray:
    colors = {"object": (0, 0, 255), "region": (0, 255, 0)}
    output = image.copy()
    color_layer = np.zeros_like(image)
    selected = np.zeros(image.shape[:2], dtype=bool)
    for class_name, mask in masks.items():
        color_layer[mask] = colors[class_name]
        selected |= mask
    blended = cv2.addWeighted(image, 1.0 - alpha, color_layer, alpha, 0.0)
    output[selected] = blended[selected]
    for class_name, mask in masks.items():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, colors[class_name], 2)
    return output


def object_location_prior(
    side_mask: np.ndarray,
    homography: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Project the right-bottom anchor and render a compact front-view prior."""
    side_contact = right_bottom_point(side_mask)
    prior = np.zeros(shape, dtype=np.uint8)
    if side_contact is None:
        return prior.astype(bool), None
    front_contact = project_points(side_contact[None], homography)[0]
    x, y = np.rint(front_contact).astype(int)
    if 0 <= x < shape[1] and 0 <= y < shape[0]:
        # Shift left and upward from the right-bottom anchor instead of warping
        # the cube's non-planar side-view silhouette.
        cv2.ellipse(prior, (max(0, x - 17), max(0, y - 16)), (18, 14), 0, 0, 360, 1, thickness=-1)
    return prior.astype(bool), front_contact


def render_demo(
    dataset_root: Path,
    output_path: Path,
    homography: np.ndarray,
    episode: int,
    frame_index: int,
) -> dict[str, float | int | bool]:
    rgb = {
        camera: read_frame(video_path(dataset_root, camera, None, episode), frame_index)
        for camera in CAMERA_KEYS
    }
    masks = {
        camera: {
            class_name: binary_mask(
                read_frame(video_path(dataset_root, camera, class_name, episode), frame_index)
            )
            for class_name in MASK_CLASSES
        }
        for camera in CAMERA_KEYS
    }
    height, width = rgb["front"].shape[:2]
    projected_object, projected_contact = object_location_prior(
        masks["side"]["object"], homography, (height, width)
    )
    projected = {
        "object": projected_object,
        "region": cv2.warpPerspective(
            masks["side"]["region"].astype(np.uint8),
            homography,
            (width, height),
            flags=cv2.INTER_NEAREST,
        ).astype(bool),
    }

    front_panel = add_title(overlay_masks(rgb["front"], masks["front"]), "Front visible masks")
    side_panel = add_title(overlay_masks(rgb["side"], masks["side"]), "Side visible masks")
    projected_panel = add_title(
        overlay_masks(rgb["front"], projected),
        "Projected region + object location prior",
    )
    if projected_contact is not None:
        contact = tuple(np.rint(projected_contact).astype(int))
        cv2.drawMarker(projected_panel, contact, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    comparison = rgb["front"].copy()
    comparison = overlay_masks(comparison, projected, alpha=0.28)
    for class_name, mask in masks["front"].items():
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(comparison, contours, -1, (255, 255, 255), 2)
    comparison = add_title(comparison, "Projected fill; white = front observation")
    if projected_contact is not None:
        contact = tuple(np.rint(projected_contact).astype(int))
        cv2.drawMarker(comparison, contact, (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
    montage = np.vstack([np.hstack([front_panel, side_panel]), np.hstack([projected_panel, comparison])])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), montage):
        raise RuntimeError(f"Could not write visualization: {output_path}")

    metrics: dict[str, float | int | bool] = {
        "episode": episode,
        "frame_index": frame_index,
        "front_object_visible": bool(masks["front"]["object"].sum() >= 25),
        "side_object_visible": bool(masks["side"]["object"].sum() >= 25),
        "projected_object_prior_area_px": int(projected["object"].sum()),
    }
    if projected_contact is not None:
        metrics["projected_object_contact_x_px"] = float(projected_contact[0])
        metrics["projected_object_contact_y_px"] = float(projected_contact[1])
    intersection = np.logical_and(projected["region"], masks["front"]["region"]).sum()
    union = np.logical_or(projected["region"], masks["front"]["region"]).sum()
    metrics["region_projection_iou"] = float(intersection / union) if union else 0.0
    return metrics


def main() -> None:
    args = parse_args()
    if not 0.0 < args.holdout_ratio < 1.0:
        raise ValueError("--holdout-ratio must be between 0 and 1.")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.quality_score_min <= 1.0:
        raise ValueError("--quality-score-min must be in (0, 1].")
    side_points, front_points, episodes = calibration_correspondences(
        dataset_root,
        args.quality_score_min,
    )
    metrics = validation_metrics(
        side_points,
        front_points,
        episodes,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        ransac_threshold=args.ransac_threshold,
    )
    homography, inliers = fit_homography(
        side_points,
        front_points,
        args.ransac_threshold,
    )
    metrics["all_data_ransac_inlier_ratio"] = float(inliers.mean())
    metrics.update(
        render_demo(
            dataset_root,
            output_dir / "side_to_front_demo.png",
            homography,
            args.episode,
            args.frame_index,
        )
    )
    payload = {
        "method": "quality_filtered_object_right_bottom_ransac_homography",
        "calibration": {
            "anchor": "side_right_bottom_to_front_right_bottom",
            "quality_score_min": args.quality_score_min,
            "uncertain_allowed": False,
            "candidate_frames_per_episode": len(CALIBRATION_FRACTIONS),
            "shape_filter": {
                "area_px": [200, 2500],
                "width_px": [18, 60],
                "height_px": [15, 55],
                "minimum_fill_ratio": 0.60,
            },
        },
        "homography_side_to_front": homography.tolist(),
        "metrics": metrics,
        "limitations": [
            "Object and region transfer assumes tabletop-planar geometry.",
            "Warped object pixels are an approximate location prior, not an exact front silhouette.",
            "Tool and deformable occluder masks should not use this homography as dense-mask ground truth.",
        ],
    }
    result_path = output_dir / "homography_side_to_front.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Visualization: {output_dir / 'side_to_front_demo.png'}")


if __name__ == "__main__":
    main()
