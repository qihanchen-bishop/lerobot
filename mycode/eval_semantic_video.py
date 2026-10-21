"""Record existing live segmentation without scheduling additional inference."""
import threading


def full_semantic_features(sensor_features, rgb_keys):
    return {
        f"{key}_semantic": dict(sensor_features[key])
        for key in rgb_keys
        if key in sensor_features
        and key.rsplit(".", 1)[-1] in {"front", "side"}
        and sensor_features[key].get("dtype") in {"image", "video"}
    }


def semantic_source_features(semantic_features):
    features = {}
    for key in semantic_features:
        view = key.removesuffix("_semantic").rsplit(".", 1)[-1]
        for name, dtype in (("source_frame_index", "int64"), ("source_timestamp_s", "float32"),
                            ("valid", "bool")):
            features[f"observation.semantic.{view}.{name}"] = {
                "dtype": dtype, "shape": (1,), "names": [name],
            }
    return features


class LiveSemanticRecorder:
    def __init__(self, features, run_token):
        self.features = features
        self.run_token = run_token
        self.lock = threading.Lock()
        self.latest = {}
        self.samples = {}
        self.models = {}
        self.recorded_frames = 0
        self.closed = False

    def update(self, view, ids, palette, source, model):
        """Called by the existing segmentation worker using its filtered class IDs."""
        import numpy as np

        if source is None or source.get("run_token") != self.run_token:
            return
        key = f"observation.images.{view}_semantic"
        if key not in self.features:
            return
        colors = np.ascontiguousarray(np.asarray(palette, np.uint8)[ids])
        with self.lock:
            if self.closed:
                return
            self.latest[view] = (colors, int(source["frame_index"]), float(source["timestamp_s"]))
            self.samples[view] = self.samples.get(view, 0) + 1
            self.models[view] = {
                "model_path": str(model.checkpoint_path), "labels": list(model.labels),
                "model_sha256": getattr(model.model, "model_sha256", None),
                "filter_policy": getattr(model.model, "filter_policy", None),
            }

    def frame(self, observation):
        """Hold the last completed live result; source metadata exposes its age."""
        import numpy as np

        with self.lock:
            snapshot = dict(self.latest)
            self.recorded_frames += 1
        frame = {}
        for key in self.features:
            rgb_key = key.removesuffix("_semantic")
            view = rgb_key.rsplit(".", 1)[-1]
            sample = snapshot.get(view)
            valid = sample is not None
            image, index, timestamp = sample if valid else (np.zeros_like(observation[rgb_key]), -1, 0.)
            frame[key] = image
            prefix = f"observation.semantic.{view}."
            frame[prefix + "source_frame_index"] = np.asarray([index], np.int64)
            frame[prefix + "source_timestamp_s"] = np.asarray([timestamp], np.float32)
            frame[prefix + "valid"] = np.asarray([valid], np.bool_)
        return frame

    def finish(self):
        with self.lock:
            self.closed = True
            return {
                "status": "complete", "generation": "reuse_live_segmentation",
                "alignment": "latest_completed_live_result_held_between_updates",
                "representation": "filtered_hard_label_palette_rgb",
                "source_metadata_prefix": "observation.semantic",
                "before_first_result": "black_frame_with_valid_false",
                "samples_by_view": dict(self.samples), "models": dict(self.models),
                "recorded_frames": self.recorded_frames,
                "additional_segmentation_calls": 0,
            }
