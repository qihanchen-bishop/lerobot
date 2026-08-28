import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mycode"))

from gui_eval_lerobot_policy import (  # noqa: E402
    EvalPolicyApp,
    SegmentationPreviewModel,
    segmentation_model_paths_for_policy,
)


class StubVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class StubSemanticPolicy:
    def __init__(self, frames):
        self.frames = frames

    def latest_inference_semantic_input(self):
        return dict(self.frames)


class EvalSegmentationPreviewTest(unittest.TestCase):
    def test_newsetup_routes_front_to_v3_and_side_to_v2(self):
        checkpoint = (
            PROJECT_ROOT / "outputs/train/newsetup_SEM-1-N_newseg/checkpoint_step_100000"
        )

        front_path, side_path = segmentation_model_paths_for_policy(checkpoint)

        self.assertEqual(front_path.parent.name, "seg_v3")
        self.assertEqual(side_path.parent.name, "seg_v2")

    def test_view_specific_preview_models_preserve_checkpoint_classes(self):
        cases = (
            (
                "front",
                "seg_v3",
                [
                    "background",
                    "occluder",
                    "object",
                    "region",
                    "tool",
                    "leftarm",
                    "rightarm",
                ],
            ),
            ("side", "seg_v2", ["background", "occluder", "object", "region", "tool"]),
        )
        for view, version, expected_labels in cases:
            with self.subTest(view=view, version=version):
                model = SegmentationPreviewModel(
                    PROJECT_ROOT / f"mycode/tool/{version}/best.pt", "cpu"
                )

                overlays, summaries, measurements = model.predict(
                    {view: np.zeros((72, 128, 3), dtype=np.uint8)}
                )

                self.assertEqual(model.labels, expected_labels)
                self.assertEqual(overlays[view].shape, (72, 128, 3))
                self.assertEqual(
                    list(measurements[view]["class_ratios"]), expected_labels[1:]
                )
                self.assertTrue(all(label in summaries[view] for label in expected_labels[1:]))

    def test_custom_save_subfolder_overrides_storage_name_only(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {
            "dataset_root": StubVariable("/tmp/eval"),
            "save_subfolder": StubVariable("front-v3-ablation"),
            "policy_type": StubVariable("mask_act"),
        }

        path = app._policy_save_parent(
            "mask_act", policy_variant="SEM-1-N", create=False
        )

        self.assertEqual(path, Path("/tmp/eval/mask_act/front-v3-ablation"))

    def test_empty_save_subfolder_keeps_automatic_variant_name(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {
            "dataset_root": StubVariable("/tmp/eval"),
            "save_subfolder": StubVariable(""),
            "policy_type": StubVariable("mask_act"),
        }

        path = app._policy_save_parent(
            "mask_act", policy_variant="SEM-1-N", create=False
        )

        self.assertEqual(path, Path("/tmp/eval/mask_act/SEM-1-N"))

    def test_save_subfolder_rejects_nested_paths(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {"save_subfolder": StubVariable("nested/name")}

        with self.assertRaisesRegex(ValueError, "one directory name"):
            app._custom_save_subfolder()

    def test_policy_semantic_recording_uses_the_actual_act_input_key(self):
        dataset_features = {
            "observation.images.front": {
                "dtype": "video",
                "shape": (360, 640, 3),
                "names": ["height", "width", "channels"],
            }
        }
        policy_inputs = {
            "observation.images.front": object(),
            "observation.images.front_semantic": object(),
        }

        features = EvalPolicyApp._policy_semantic_recording_features(
            dataset_features,
            policy_inputs,
        )
        semantic = np.zeros((360, 640, 3), dtype=np.uint8)
        frame = EvalPolicyApp._latest_policy_semantic_frame(
            StubSemanticPolicy({"observation.images.front_semantic": semantic}),
            features,
        )

        self.assertEqual(list(features), ["observation.images.front_semantic"])
        self.assertEqual(features["observation.images.front_semantic"]["dtype"], "video")
        self.assertIs(frame["observation.images.front_semantic"], semantic)

    def test_custom_folder_result_lookup_keeps_policy_variant_filter(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {"save_subfolder": StubVariable("SEM-1-N-FRONT")}
        observed_paths = []

        def fake_read(path):
            observed_paths.append(path)
            return [
                {"policy_type": "mask_act", "policy_variant": "SEM-1-N"},
                {"policy_type": "mask_act", "policy_variant": "OTHER"},
            ]

        app._read_jsonl_records = fake_read
        records = app._result_records(Path("/tmp/eval"), "mask_act", "SEM-1-N")

        self.assertEqual(
            observed_paths,
            [Path("/tmp/eval/mask_act/SEM-1-N-FRONT/eval_results.jsonl")],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["policy_variant"], "SEM-1-N")

    def test_recent_save_subfolder_is_restored_for_the_same_checkpoint(self):
        checkpoint = PROJECT_ROOT / "outputs/train/newsetup_SEM-1-N_newseg/checkpoint_step_100000"
        with TemporaryDirectory(prefix="eval-subfolder-test-") as root:
            result_path = Path(root) / "mask_act/SEM-1-N-FRONT/eval_results.jsonl"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(
                    {
                        "policy_type": "mask_act",
                        "policy_path": str(checkpoint),
                        "saved_at": "2026-08-28T10:20:00",
                    }
                )
                + "\n"
            )
            app = object.__new__(EvalPolicyApp)
            app.vars = {
                "dataset_root": StubVariable(root),
                "policy_type": StubVariable("mask_act"),
                "checkpoint_path": StubVariable(str(checkpoint)),
            }

            restored = app._recent_save_subfolder_for_current_checkpoint()

        self.assertEqual(restored, "SEM-1-N-FRONT")


if __name__ == "__main__":
    unittest.main()
