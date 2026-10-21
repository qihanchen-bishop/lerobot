import json
from pathlib import Path
import queue
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mycode"))

from eval_protocol import CONFIG_DIR, deviation_records, load_calibration, stage_from_metrics
from gui_eval_lerobot_policy import EvalPolicyApp, SegmentationPreviewModel


class Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class FinaldataEvalProtocolTest(unittest.TestCase):
    def test_missing_local_calibration_uses_explicit_defaults(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            with patch("eval_protocol.CALIBRATION_PATH", path):
                with self.assertWarnsRegex(RuntimeWarning, "Using finaldata defaults"):
                    calibration = load_calibration()
                self.assertEqual(calibration["recommended_episode_time_s"], 40)
                self.assertEqual(calibration["recommended_occluder_complete_ratio"], .32)
                self.assertEqual(calibration["source"], "builtin_finaldata_defaults")
                self.assertFalse(path.exists())
                path.write_text(json.dumps({"recommended_episode_time_s": 45}))
                self.assertEqual(load_calibration(), {"recommended_episode_time_s": 45})

    def app(self):
        app = object.__new__(EvalPolicyApp)
        app.vars = {key: Variable(value) for key, value in {
            "task": "cube", "occluder_complete_ratio": ".32", "evaluation_stage": "未暴露",
        }.items()}
        app.enable_segmentation_preview = Variable(True)
        app.eval_running = True
        app.started_at = time.monotonic()
        app.stop_policy_requested = False
        app.log_queue = queue.Queue()
        app.segmentation_eval_metrics = app._new_segmentation_eval_metrics()
        return app

    @staticmethod
    def measurement(exposed=False, separated=False, in_region=False, coverage=.1):
        return {"class_ratios": {"object": .01 if exposed else 0, "occluder": coverage},
                "object_occluder_separated": separated, "object_in_region": in_region}

    def feed(self, app, front, side=None, count=7):
        measurements = {"front": front} if front is not None else {}
        if side is not None:
            measurements["side"] = side
        for _ in range(count):
            app._update_segmentation_eval_metrics(measurements)
        return app.segmentation_eval_metrics

    def test_five_front_only_stages_and_repeated_evidence(self):
        app = self.app()
        stages = [
            (self.measurement(), "未暴露"),
            (self.measurement(True), "暴露未分离"),
            (self.measurement(True, True), "分离未入区"),
            (self.measurement(True, True, True, .31), "入区未恢复"),
            (self.measurement(True, True, True, .32), "完成"),
        ]
        for frame, label in stages:
            metrics = self.feed(app, frame)
            self.assertEqual(metrics["evaluation_stage_label"], label)
        self.assertTrue(app.stop_policy_requested)
        self.assertEqual(metrics["segmentation_occluder_complete_threshold"], .32)
        # A temporary false frame must not erase voted completion; sustained loss must.
        self.feed(app, self.measurement(), count=1)
        self.assertTrue(metrics["final_task_success"])
        self.feed(app, self.measurement())
        self.assertFalse(metrics["final_task_success"])
        self.assertTrue(metrics["auto_task_complete"])
        self.assertEqual(metrics["evaluation_stage_label"], "未暴露")

    def test_side_success_and_missing_front_cannot_finish(self):
        app = self.app()
        success = self.measurement(True, True, True, .4)
        metrics = self.feed(app, self.measurement(), success)
        self.assertFalse(metrics["final_task_success"])
        self.assertEqual(metrics["evaluation_stage_label"], "未暴露")
        self.feed(app, self.measurement(True, True, True, .31))
        self.feed(app, None, success)
        self.assertFalse(metrics["final_object_exposed"])
        self.assertEqual(metrics["last_decision_source"], "front_missing")
        self.assertFalse(app.stop_policy_requested)

    def test_coverage_and_region_without_separation_do_not_complete(self):
        app = self.app()
        metrics = self.feed(app, self.measurement(True, False, True, .4))
        self.assertFalse(metrics["final_task_success"])
        self.assertEqual(stage_from_metrics(metrics)[1], "暴露未分离")

    def test_hiding_preview_does_not_disable_task_judgement(self):
        app = self.app()
        app.enable_segmentation_preview.set(False)
        metrics = self.feed(app, self.measurement(True, True, True, .4))
        self.assertTrue(metrics["segmentation_enabled"])
        self.assertFalse(metrics["segmentation_preview_enabled"])
        self.assertTrue(metrics["final_task_success"])

    def test_recovery_is_explicit_and_per_deviation(self):
        with self.assertRaisesRegex(ValueError, "是否已从该偏差恢复"):
            deviation_records({"air_sweep": "请选择"})
        records = deviation_records({"air_sweep": "已恢复", "tool_trapped": "未恢复"})
        self.assertEqual({r["code"]: r["recovered"] for r in records},
                         {"air_sweep": True, "tool_trapped": False})
        self.assertEqual(len({r["category"] for r in records}), 2)
        self.assertEqual(deviation_records({}), [])

    def test_config_folder_is_bounded_and_accepts_all_policy_families(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            config = {"policy_type": "act", "checkpoint_path": "outputs/train/finaldata/ACT"}
            (folder / "valid.json").write_text(json.dumps(config))
            (folder / "other.json").write_text('{"something": 1}')
            (folder / "broken.json").write_text('[')
            (folder / "nested").mkdir()
            (folder / "nested/hidden.json").write_text(json.dumps(config))
            self.assertEqual(list(EvalPolicyApp._config_presets_in_folder(folder)), ["valid"])
    @unittest.skipUnless(CONFIG_DIR.is_dir(), "Local evaluation presets are not synced")
    def test_local_finaldata_presets(self):
        presets = EvalPolicyApp._config_presets_in_folder(CONFIG_DIR)
        self.assertEqual(len(presets), 9)
        for path in presets.values():
            config = json.loads(path.read_text())
            self.assertTrue((ROOT / config["checkpoint_path"]).is_dir())
            self.assertEqual(config["episode_time_s"], 40)
            self.assertEqual(config["occluder_complete_ratio"], .32)
            self.assertEqual(config["n_action_steps"], 30)

    def test_exported_preview_matches_policy_filtered_hard_semantics(self):
        import torch
        from view_qtoken import FrozenExportedUNet

        # A tiny portable model isolates preprocessing, argmax and component filtering.
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            labels = ["background", "occluder", "object", "region", "tool", "leftarm", "rightarm"]
            network = torch.nn.Conv2d(3, len(labels), 1)
            with torch.no_grad():
                network.weight.zero_()
                network.bias.zero_()
                network.weight[1, 0, 0, 0] = 10
                network.bias[1] = -5
            torch.jit.trace(network, torch.zeros(1, 3, 8, 12)).save(str(folder / "model.ts"))
            config = {"labels": labels, "torchscript": "model.ts", "input_size_wh": [12, 8],
                      "normalization_mean": [0, 0, 0], "normalization_std": [1, 1, 1],
                      "palette_rgb": [[i * 30, 0, 0] for i in range(7)]}
            (folder / "model_config.json").write_text(json.dumps(config))
            (folder / "filter_policy.json").write_text(json.dumps({"policy": {
                "min_area": 3, "component_limits": {name: 1 for name in labels[1:]}}}))
            rgb = np.zeros((8, 12, 3), np.uint8)
            rgb[1:4, 1:4, 0] = 255
            rgb[6, 10, 0] = 255  # small component must be filtered from both paths
            policy_segmenter = FrozenExportedUNet(folder / "model.ts", expected_labels=labels)
            semantics = policy_segmenter(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255)
            self.assertTrue(torch.all((semantics == 0) | (semantics == 1)))
            self.assertTrue(torch.all(semantics.sum(dim=1) == 1))
            ids = semantics.argmax(1)[0].numpy()
            preview = SegmentationPreviewModel(folder, "cpu")
            _, _, measurements = preview.predict({"front": rgb})
            np.testing.assert_array_equal(preview.latest_class_ids["front"], ids)
            self.assertEqual(int((ids == 1).sum()), 9)
            self.assertEqual(measurements["front"]["class_ratios"]["occluder"], float((ids == 1).mean()))

    def test_policy_preview_preserves_hard_labels_and_soft_mixtures(self):
        from gui_eval_lerobot_policy import SEGMENTATION_PALETTE_V3

        ids = np.asarray([[0, 1, 2], [3, 4, 5], [6, 0, 1]])
        names = ("background", "occluder", "object", "region", "tool", "leftarm", "rightarm")
        masks = {f"observation.images.front_{name}": ((ids == idx) * 255).astype(np.uint8)
                 for idx, name in enumerate(names)}
        overlays, summaries = EvalPolicyApp._render_model_mask_previews(masks)
        np.testing.assert_array_equal(overlays["front"], np.asarray(SEGMENTATION_PALETTE_V3)[ids])
        self.assertTrue(summaries["front"].startswith("hard-7"))
        self.assertNotIn("confidence", summaries["front"])

        masks = {"front_background": np.asarray([[128]], np.uint8),
                 "front_object": np.asarray([[127]], np.uint8)}
        overlays, summaries = EvalPolicyApp._render_model_mask_previews(masks)
        self.assertTrue(summaries["front"].startswith("soft-2"))
        expected = (np.asarray([255, 80, 80]) * (127 / 255)).astype(np.uint8)
        np.testing.assert_array_equal(overlays["front"][0, 0], expected)

    def test_lfs_pointer_requires_exact_local_payload(self):
        from view_qtoken import resolve_torchscript_payload, sha256

        for view in ("front", "side"):
            script = ROOT / f"mycode/tool/finaldata_{view}_unet_r3/model.ts"
            payload = resolve_torchscript_payload(script)
            manifest = json.loads((script.parent / "bundle_manifest.json").read_text())
            self.assertEqual(sha256(payload), manifest["model_sha256"])
        with TemporaryDirectory() as directory:
            pointer = Path(directory) / "model.ts"
            pointer.write_text("version https://git-lfs.github.com/spec/v1\n"
                               + "oid sha256:" + "0" * 64 + "\nsize 1\n")
            with self.assertRaisesRegex(FileNotFoundError, "no matching local payload"):
                resolve_torchscript_payload(pointer)


if __name__ == "__main__":
    unittest.main()
