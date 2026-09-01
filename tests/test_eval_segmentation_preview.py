import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mycode"))

from gui_eval_lerobot_policy import (  # noqa: E402
    DEFAULT_ACT_REPLAN_STEPS,
    DEFAULT_EPISODE_TIME_S,
    DEFAULT_TRIALS_PER_GRID,
    EvalPolicyApp,
    SegmentationPreviewModel,
    segmentation_model_paths_for_policy,
)


class StubVariable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class StubSemanticPolicy:
    def __init__(self, frames):
        self.frames = frames

    def latest_inference_semantic_input(self):
        return dict(self.frames)


class EvalSegmentationPreviewTest(unittest.TestCase):
    def test_evaluation_protocol_defaults(self):
        self.assertEqual(DEFAULT_EPISODE_TIME_S, 30)
        self.assertEqual(DEFAULT_ACT_REPLAN_STEPS, 30)
        self.assertEqual(DEFAULT_TRIALS_PER_GRID, 2)

    def test_standard_act_checkpoint_defaults_to_30_replan_steps(self):
        with TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory)
            (checkpoint / "config.json").write_text(
                json.dumps({"chunk_size": 60, "n_action_steps": 60}),
                encoding="utf-8",
            )
            app = object.__new__(EvalPolicyApp)
            app.vars = {
                key: StubVariable(value)
                for key, value in {
                    "policy_type": "act",
                    "model_chunk_size": "N/A",
                    "prediction_steps": "",
                    "n_action_steps": "",
                    "fusion_steps": "0",
                    "fusion_history_weight": "0",
                    "num_inference_steps": "",
                    "noise_scheduler_type": "checkpoint",
                    "execution_mode": "synchronous",
                    "camera_read_mode": "wait_new_frame",
                }.items()
            }
            app.use_amp = StubVariable(False)
            app._append_log = lambda _message: None

            app._refresh_action_chunk_settings(checkpoint)

            self.assertEqual(app.vars["prediction_steps"].get(), "60")
            self.assertEqual(app.vars["n_action_steps"].get(), "30")

    def test_model_parameter_statistics_reports_counts_and_memory(self):
        class StubTensor:
            def __init__(self, count, bytes_per_value, requires_grad=False):
                self._count = count
                self._bytes_per_value = bytes_per_value
                self.requires_grad = requires_grad

            def numel(self):
                return self._count

            def element_size(self):
                return self._bytes_per_value

        class StubModel:
            def parameters(self):
                return [StubTensor(100, 4, True), StubTensor(50, 2, False)]

            def buffers(self):
                return [StubTensor(20, 4)]

        statistics = EvalPolicyApp._model_parameter_statistics(StubModel())

        self.assertEqual(
            statistics,
            {
                "parameter_count": 150,
                "trainable_parameter_count": 100,
                "parameter_bytes": 500,
                "buffer_bytes": 80,
            },
        )

    def test_eval_observation_always_uses_robot_waiting_read(self):
        class StubRobot:
            def __init__(self):
                self.reads = 0

            def get_observation(self):
                self.reads += 1
                return {"observation.images.front": "new-frame"}

        app = object.__new__(EvalPolicyApp)
        robot = StubRobot()

        observation = app._get_eval_observation(robot)

        self.assertEqual(observation["observation.images.front"], "new-frame")
        self.assertEqual(robot.reads, 1)

    def test_camera_layout_is_required_before_connection(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {"checkpoint_path": StubVariable("/tmp/missing-checkpoint")}
        app.include_side_camera = StubVariable(False)

        with self.assertRaisesRegex(ValueError, "Could not determine"):
            app._sync_side_camera_for_checkpoint(required=True)

    def test_newsetup_routes_to_latest_view_specific_models(self):
        checkpoint = (
            PROJECT_ROOT / "outputs/train/newsetup_SEM-1-N_newseg/checkpoint_step_100000"
        )

        front_path, side_path = segmentation_model_paths_for_policy(checkpoint)

        self.assertEqual(front_path.parent.name, "unet_front_v4_r1")
        self.assertEqual(side_path.parent.name, "unet_side")

    def test_legacy_checkpoint_uses_same_system_segmenters(self):
        checkpoint = PROJECT_ROOT / "outputs/train/ACT_3object/checkpoint_step_100000"

        front_path, side_path = segmentation_model_paths_for_policy(checkpoint)

        self.assertEqual(front_path.parent.name, "unet_front_v4_r1")
        self.assertEqual(side_path.parent.name, "unet_side")

    def test_v5_strategies_route_to_latest_view_specific_models(self):
        for strategy in (
            "newsetup_UnetSem_FS",
            "newsetup_Stagev5_Rgb_F",
            "newsetup_AISatgev5_RGB_F",
        ):
            with self.subTest(strategy=strategy):
                checkpoint = PROJECT_ROOT / f"outputs/train/{strategy}/checkpoint_step_100000"

                front_path, side_path = segmentation_model_paths_for_policy(checkpoint)

                self.assertEqual(front_path.parent.name, "unet_front_v4_r1")
                self.assertEqual(side_path.parent.name, "unet_side")

    def test_neweval_layout_separates_view_strategy_and_runtime_configuration(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {
            "dataset_root": StubVariable(str(PROJECT_ROOT / "eval/neweval")),
            "save_subfolder": StubVariable(""),
            "policy_type": StubVariable("act"),
            "checkpoint_path": StubVariable("/tmp/checkpoint"),
            "prediction_steps": StubVariable("60"),
            "n_action_steps": StubVariable("20"),
            "num_inference_steps": StubVariable(""),
            "noise_scheduler_type": StubVariable("checkpoint"),
            "execution_mode": StubVariable("synchronous"),
            "camera_read_mode": StubVariable("wait_new_frame"),
            "fusion_steps": StubVariable("0"),
            "fusion_history_weight": StubVariable("0"),
            "ssact_servo_mode": StubVariable("off"),
            "ssact_min_execution_steps": StubVariable("1"),
            "ssact_max_execution_steps": StubVariable("4"),
            "ssact_max_action_residual": StubVariable("0.08"),
        }
        app.include_side_camera = StubVariable(False)
        app.use_amp = StubVariable(False)
        app.enable_ssact_adaptive_horizon = StubVariable(False)
        app.enable_auto_replan = StubVariable(False)
        app.lock_grippers = StubVariable(False)

        path = app._policy_save_parent("act", policy_variant="ACT-SINGLE-FRONT", create=False)

        self.assertEqual(
            path,
            PROJECT_ROOT
            / "eval/neweval/single_view/act/ACT-SINGLE-FRONT"
            / "replan-20__sync__pred-60__fusion-0-w0__amp-off",
        )

    def test_view_layout_is_derived_from_checkpoint_training_inputs(self):
        app = object.__new__(EvalPolicyApp)
        app.include_side_camera = StubVariable(False)
        app.vars = {
            "checkpoint_path": StubVariable(
                str(PROJECT_ROOT / "outputs/train/newsetup_act/100000/pretrained_model")
            )
        }
        self.assertEqual(app._view_layout_name(), "dual_view")

        app.vars["checkpoint_path"] = StubVariable(
            str(PROJECT_ROOT / "outputs/train/newsetup_act_single/100000/pretrained_model")
        )
        self.assertEqual(app._view_layout_name(), "single_view")

    def test_diffusion_runtime_folder_separates_scheduler_and_inference_steps(self):
        folder = EvalPolicyApp._runtime_configuration_dirname(
            {
                "policy_type": "diffusion",
                "replan_interval_steps": 24,
                "prediction_steps": 64,
                "num_inference_steps": 16,
                "noise_scheduler_type": "DDIM",
                "execution_mode": "asynchronous",
                "fusion_steps": 0,
                "fusion_history_weight": 0.0,
                "use_amp": True,
                "ssact_servo_mode": "off",
                "ssact_adaptive_horizon": False,
            }
        )

        self.assertEqual(
            folder,
            "replan-24__async__pred-64__ddim-16__fusion-0-w0__amp-on",
        )

    def test_auto_replan_runtime_folder_replaces_fixed_interval_name(self):
        folder = EvalPolicyApp._runtime_configuration_dirname(
            {
                "policy_type": "act",
                "replan_interval_steps": 30,
                "auto_replan": True,
                "prediction_steps": 60,
                "execution_mode": "synchronous",
                "fusion_steps": 0,
                "fusion_history_weight": 0.0,
                "use_amp": False,
                "ssact_servo_mode": "off",
                "ssact_adaptive_horizon": False,
            }
        )

        self.assertEqual(folder, "autoreplan__sync__pred-60__fusion-0-w0__amp-off")

    def test_config_presets_only_show_newsetup_non_embedding_runs(self):
        presets = EvalPolicyApp._newsetup_config_presets(PROJECT_ROOT / "mycode")

        for name in ("SS5_RGB_F", "SS5_U_F", "SS5_U_FS", "AI5_RGB_F"):
            self.assertIn(name, presets)
        self.assertIn("ACT_NEWSETUP", presets)
        self.assertNotIn("ACT", presets)
        self.assertFalse(any("CE" in name or "VIEWFUS" in name for name in presets))
        self.assertNotIn("auto_replan_bettersetup", presets)

    def test_checkpoint_visibility_uses_the_same_newsetup_filter(self):
        self.assertEqual(
            EvalPolicyApp._visible_newsetup_run_name(
                PROJECT_ROOT / "outputs/train/newsetup_Stagev5_U_FS/checkpoint_step_100000"
            ),
            "newsetup_Stagev5_U_FS",
        )
        self.assertIsNone(
            EvalPolicyApp._visible_newsetup_run_name(
                PROJECT_ROOT / "outputs/train/1A_3object/checkpoint_step_100000"
            )
        )
        self.assertIsNone(
            EvalPolicyApp._visible_newsetup_run_name(
                PROJECT_ROOT / "outputs/train/newsetup_CE_gated/100000/pretrained_model"
            )
        )

    def test_decision_and_segmentation_recording_features_follow_rgb_views(self):
        dataset_features = {
            "observation.images.front": {
                "dtype": "video",
                "shape": (360, 640, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.side": {
                "dtype": "video",
                "shape": (360, 640, 3),
                "names": ["height", "width", "channels"],
            },
        }
        rgb_keys = ["observation.images.front", "observation.images.side"]

        decision_features, decision_sources = EvalPolicyApp._policy_decision_recording_features(
            dataset_features, rgb_keys
        )
        segmentation_features, segmentation_views = (
            EvalPolicyApp._policy_segmentation_recording_features(dataset_features, rgb_keys)
        )

        self.assertEqual(
            decision_sources,
            {
                "observation.images.front_policy_input": "observation.images.front",
                "observation.images.side_policy_input": "observation.images.side",
            },
        )
        self.assertEqual(
            segmentation_views,
            {
                "observation.images.front_policy_segmentation": "front",
                "observation.images.side_policy_segmentation": "side",
            },
        )
        self.assertTrue(all(feature["dtype"] == "video" for feature in decision_features.values()))
        self.assertTrue(all(feature["dtype"] == "video" for feature in segmentation_features.values()))

    def test_view_specific_preview_models_preserve_checkpoint_classes(self):
        cases = (
            (
                "front",
                "unet_front_v4_r1",
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
            ("side", "unet_side", ["background", "occluder", "object", "region", "tool"]),
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
