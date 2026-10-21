"""Aligned recording of final Qtoken predictions and segmentation-derived geometry."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np

from view_qtoken import (
    CLASSES, FRONT_QUERY_EXPERIMENTS, QUERY_EXPERIMENTS, VIEWS,
    FrozenExportedUNet, region_distance_map, view_geometry,
)

METRIC_NAMES = (
    "visibility_front", "visibility_side", "tool_front_dx", "tool_front_dy",
    "tool_side_dx", "tool_side_dy", "region_front", "region_side",
)
FRONT_METRIC_NAMES = ("visibility_front", "centroid_region_front", "tool_front_dx", "tool_front_dy")
PREFIX = "observation.qtoken."


def recording_features(metric_names=METRIC_NAMES):
    features = {}
    for field in ("measured", "predicted", "error"):
        features[PREFIX + field] = {"dtype": "float32", "shape": (len(metric_names),), "names": list(metric_names)}
        features[PREFIX + field + "_valid"] = {"dtype": "bool", "shape": (len(metric_names),), "names": list(metric_names)}
    for field in ("inference_id", "source_control_step"):
        features[PREFIX + field] = {"dtype": "int64", "shape": (1,), "names": [field]}
    for field in ("source_timestamp_s", "ready_timestamp_s", "frame_timestamp_s", "age_s"):
        features[PREFIX + field] = {"dtype": "float32", "shape": (1,), "names": [field]}
    return features


class QTokenRecorder:
    """Capture immutable decisions in the worker; activate accepted decisions in the control loop."""

    def __init__(self, policy, run_config, model_paths, output_dir):
        if str(policy.experiment).upper() not in QUERY_EXPERIMENTS:
            raise ValueError("Qtoken recording requires a final supervised-query checkpoint.")
        self.policy = policy
        self.front_only = str(policy.experiment).upper() in FRONT_QUERY_EXPERIMENTS
        self.metric_names = FRONT_METRIC_NAMES if self.front_only else METRIC_NAMES
        self.views = ("front",) if self.front_only else VIEWS
        self.regions = {} if self.front_only else run_config["view_query"]["cache_metadata"]["regions"]
        order = run_config["view_query"]["cache_metadata"]["output_order"]
        if tuple(order) != self.metric_names:
            raise ValueError(f"Unexpected Qtoken output order: {order}")
        self.model_paths = list(model_paths)
        if len(self.model_paths) != 2:
            raise ValueError("Qtoken recording requires front and side segmenter bundles.")
        self.reference_hw = {
            view: tuple(policy.config.input_features[f"observation.images.{view}"].shape[-2:])
            for view in VIEWS
        }
        self.segmenters = {}
        self.distance_maps = {}
        self.output_dir = Path(output_dir)
        self.lock = threading.Lock()
        self.next_id = 0
        self.active = None
        self.metadata = {
            "schema_version": 1, "metric_names": list(self.metric_names),
            "supervised_views": list(self.views),
            "segmentation_model_paths": [str(path) for path in self.model_paths],
            "predictions_file": "qtoken_predictions.jsonl",
            "frame_feature_prefix": PREFIX,
            "measurement_source": "r3 filtered segmentation of the exact policy decision RGB frames; not ground truth",
            "prediction_source": "raw metric_pred from the same ACT forward pass as the action chunk; no clipping",
            "visibility": "object mask area / full image area",
            "tool_offset": "object centroid minus tool reference; x/(width-1), y/(height-1)",
            "tool_reference_front": "leftmost PCA endpoint (y breaks x ties)",
            "tool_reference_side": "tool mask centroid",
            "region_distance": "maximum object-pixel distance outside the fixed training ROI, divided by image diagonal",
            "regions": self.regions, "region_reference_hw": self.reference_hw,
            "validity": "same geometry visibility/degeneracy rules as training; no live annotation-quality weights",
            "error": "prediction minus measurement, valid only when both are valid",
            "time_alignment": "values held from the accepted inference_id; background predictions do not replace active values",
            "clock": "elapsed perf_counter wall time in seconds; frame_timestamp_s is independent of the nominal dataset timestamp",
            "source_control_step": "zero-based source RGB frame_index in this episode",
            "missing_values": "null in JSONL; zero with valid=false in frame arrays; inference_id=-1 before first decision",
        }
        if self.front_only:
            self.metadata.update(
                region_distance="object centroid to current segmented region centroid, divided by image diagonal",
                tool_reference_front="original front_query_metric_targets PCA endpoint with smaller x; endpoint_a on ties",
                validity="original front_query_metric_targets visibility rules; no PCA degeneracy rejection",
            )
            self.metadata.pop("tool_reference_side")

    def _class_ids(self, observation):
        import torch

        latest = self.policy.latest_inference_mask_preview()
        result = {}
        for index, view in enumerate(self.views):
            labels = ("background", "occluder", "object", "region", "tool", "leftarm", "rightarm")
            keys = [f"observation.images.{view}_{label}" for label in labels]
            if all(key in latest for key in keys):
                arrays = [np.asarray(latest[key].cpu() if hasattr(latest[key], "cpu") else latest[key]) for key in keys]
                result[view] = np.stack(arrays).argmax(0).astype(np.uint8)
            else:
                # RGBQtoken has no semantic branch. Measure its decision image with
                # the same r3 adapter used by semantic policies and live preview.
                if view not in self.segmenters:
                    self.segmenters[view] = FrozenExportedUNet(
                        self.model_paths[index], expected_labels=labels,
                    ).to(next(self.policy.parameters()).device).eval()
                rgb = np.asarray(observation[f"observation.images.{view}"], dtype=np.uint8)
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255
                result[view] = self.segmenters[view](tensor).argmax(1)[0].cpu().numpy().astype(np.uint8)
        return result

    def _measure(self, class_ids):
        if self.front_only:
            import torch
            from train_mask_act_policy import front_query_metric_targets

            # Call the training target function to preserve its centroid, PCA,
            # tie-breaking and visibility conventions exactly.
            ids = torch.from_numpy(np.asarray(class_ids["front"]).copy()).long()[None]
            probabilities = torch.nn.functional.one_hot(ids, len(CLASSES) + 1).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                values, valid = front_query_metric_targets(probabilities, CLASSES)
            return values[0].numpy(), valid[0].numpy()
        values, valid = np.zeros(8, np.float32), np.zeros(8, bool)
        for index, view in enumerate(VIEWS):
            ids = class_ids[view]
            shape = tuple(ids.shape)
            cache_key = (view, shape)
            if cache_key not in self.distance_maps:
                polygon = np.asarray(self.regions[view]["polygon"], dtype=np.float64)
                ref_h, ref_w = self.reference_hw[view]
                h, w = shape
                polygon = polygon * np.asarray([(w - 1) / max(ref_w - 1, 1), (h - 1) / max(ref_h - 1, 1)])
                self.distance_maps[cache_key] = region_distance_map(shape, np.rint(polygon).astype(np.int32))
            columns = [index, 2 + 2 * index, 3 + 2 * index, 6 + index]
            values[columns], valid[columns] = view_geometry(
                ids == 2, ids == 4, ids == 3, self.distance_maps[cache_key], view,
            )
        return values, valid

    def capture(self, observation, *, source_control_step, source_timestamp_s,
                inference_started_s, ready_timestamp_s):
        """Does not activate this result; async inference may finish before acceptance."""
        capture_started = time.perf_counter()
        predicted = self.policy.latest_inference_metrics()
        if predicted is None:
            raise RuntimeError("The Qtoken checkpoint did not expose its metric-head output.")
        predicted = np.asarray(predicted.cpu() if hasattr(predicted, "cpu") else predicted, dtype=np.float32)
        if predicted.shape != (1, len(self.metric_names)):
            raise ValueError(f"Expected a single decision with {len(self.metric_names)} Qtoken outputs, got {predicted.shape}")
        predicted = predicted[0].copy()
        ids = self._class_ids(observation)
        measured, measured_valid = self._measure(ids)
        predicted_valid = np.isfinite(predicted)
        error_valid = measured_valid & predicted_valid
        error = predicted - measured

        def named(values, validity):
            return {name: float(value) if available else None
                    for name, value, available in zip(self.metric_names, values, validity, strict=True)}

        with self.lock:
            inference_id = self.next_id
            self.next_id += 1
            report = {
                "schema_version": 1, "inference_id": inference_id,
                "source_control_step": int(source_control_step), "source_timestamp_s": float(source_timestamp_s),
                "inference_started_s": float(inference_started_s),
                "ready_timestamp_s": float(ready_timestamp_s + time.perf_counter() - capture_started),
                "metric_names": list(self.metric_names),
                "measured": named(measured, measured_valid), "predicted": named(predicted, predicted_valid),
                "error": named(error, error_valid), "measured_valid": measured_valid.tolist(),
                "predicted_valid": predicted_valid.tolist(), "error_valid": error_valid.tolist(),
                "segmentation_hw": {view: list(ids[view].shape) for view in self.views},
            }
            # The dataset directory is created by LeRobotDataset.create before inference.
            with (self.output_dir / "qtoken_predictions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
        return report

    def activate(self, report):
        self.active = report

    def frame(self, elapsed_s):
        report = self.active
        frame = {}
        for field in ("measured", "predicted", "error"):
            frame[PREFIX + field] = np.asarray([
                (report[field][name] if report[field][name] is not None else 0.) if report else 0.
                for name in self.metric_names
            ], dtype=np.float32)
            frame[PREFIX + field + "_valid"] = np.asarray(
                report[field + "_valid"] if report else [False] * len(self.metric_names), dtype=bool)
        for field in ("inference_id", "source_control_step"):
            frame[PREFIX + field] = np.asarray([report[field] if report else -1], dtype=np.int64)
        for field in ("source_timestamp_s", "ready_timestamp_s"):
            frame[PREFIX + field] = np.asarray([report[field] if report else 0.], dtype=np.float32)
        frame[PREFIX + "frame_timestamp_s"] = np.asarray([elapsed_s], dtype=np.float32)
        frame[PREFIX + "age_s"] = np.asarray([max(0., elapsed_s - report["source_timestamp_s"]) if report else 0.], dtype=np.float32)
        return frame
