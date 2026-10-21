import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mycode"))

from qtoken_eval import METRIC_NAMES, PREFIX, QTokenRecorder, recording_features
from lerobot.policies.act.modeling_act import ACTPolicy


class StubQueryPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.experiment = "FINAL-VIEWQTOKEN"
        self.config = SimpleNamespace(input_features={
            f"observation.images.{view}": SimpleNamespace(shape=(3, 10, 10))
            for view in ("front", "side")
        })
        self.predicted = torch.tensor([[.02, .03, .5, -.3, .4, -.2, .01, .02]])
        self.ids = np.zeros((10, 10), np.uint8)
        self.ids[4:7, 6:10] = 3
        self.ids[5, 7] = 2
        self.ids[7, 1:6] = 4

    def latest_inference_metrics(self):
        return self.predicted.clone()

    def latest_inference_mask_preview(self):
        labels = ("background", "occluder", "object", "region", "tool", "leftarm", "rightarm")
        return {f"observation.images.{view}_{name}": (self.ids == index).astype(np.uint8) * 255
                for view in ("front", "side") for index, name in enumerate(labels)}


def run_config():
    return {"view_query": {"cache_metadata": {
        "output_order": list(METRIC_NAMES),
        "regions": {view: {"polygon": [[6, 4], [9, 4], [9, 6], [6, 6]]}
                    for view in ("front", "side")},
    }}}


class QTokenEvaluationTest(unittest.TestCase):
    def capture(self, recorder, step=0):
        return recorder.capture({}, source_control_step=step, source_timestamp_s=step / 30,
                                inference_started_s=step / 30, ready_timestamp_s=step / 30 + .01)

    def recorder(self, policy, folder):
        return QTokenRecorder(policy, run_config(), ["front/model.ts", "side/model.ts"], folder)

    def test_exact_training_geometry_and_raw_unclipped_predictions(self):
        with TemporaryDirectory() as folder:
            policy = StubQueryPolicy()
            policy.predicted[0, 0] = -2.  # raw head values must not be clipped to [0, 1]
            recorder = self.recorder(policy, folder)
            report = self.capture(recorder)
            actual = [report["measured"][name] for name in METRIC_NAMES]
            np.testing.assert_allclose(actual, [.01, .01, 6 / 9, -2 / 9, 4 / 9, -2 / 9, 0, 0], atol=1e-7)
            self.assertEqual(report["predicted"]["visibility_front"], -2.)
            self.assertAlmostEqual(report["error"]["visibility_front"], -2.01, places=6)
            self.assertTrue(all(report["measured_valid"]))
            saved = json.loads((Path(folder) / "qtoken_predictions.jsonl").read_text())
            self.assertEqual(saved, report)

    def test_missing_geometry_and_nonfinite_outputs_are_explicitly_invalid(self):
        with TemporaryDirectory() as folder:
            policy = StubQueryPolicy()
            policy.ids[policy.ids == 2] = 0
            policy.predicted[0, 0] = float("nan")
            recorder = self.recorder(policy, folder)
            report = self.capture(recorder)
            self.assertEqual(report["measured"]["visibility_front"], 0.)
            self.assertTrue(report["measured_valid"][0])
            self.assertIsNone(report["measured"]["tool_front_dx"])
            self.assertIsNone(report["measured"]["region_side"])
            self.assertIsNone(report["predicted"]["visibility_front"])
            self.assertIsNone(report["error"]["visibility_front"])
            recorder.activate(report)
            frame = recorder.frame(.2)
            self.assertFalse(frame[PREFIX + "predicted_valid"][0])
            self.assertEqual(frame[PREFIX + "predicted"][0], 0)

    def test_async_completion_keeps_active_decision_until_acceptance(self):
        with TemporaryDirectory() as folder:
            policy = StubQueryPolicy()
            recorder = self.recorder(policy, folder)
            initial = recorder.frame(0)
            self.assertEqual(initial[PREFIX + "inference_id"].item(), -1)
            self.assertFalse(initial[PREFIX + "measured_valid"].any())
            first = self.capture(recorder)
            recorder.activate(first)
            policy.predicted[0, 0] = .7
            pending = self.capture(recorder, step=30)
            held = recorder.frame(1.2)
            self.assertEqual(held[PREFIX + "inference_id"].item(), 0)
            self.assertAlmostEqual(float(held[PREFIX + "predicted"][0]), .02)
            recorder.activate(pending)
            current = recorder.frame(1.2)
            self.assertEqual(current[PREFIX + "inference_id"].item(), 1)
            self.assertEqual(current[PREFIX + "source_control_step"].item(), 30)
            self.assertAlmostEqual(current[PREFIX + "age_s"].item(), .2)
            self.assertAlmostEqual(current[PREFIX + "predicted"][0].item(), .7)
            self.assertEqual(first["predicted"]["visibility_front"], float(np.float32(.02)))

    def test_qtoken_numeric_columns_are_saved_in_lerobot_parquet(self):
        import pandas as pd
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with TemporaryDirectory() as directory:
            recorder = self.recorder(StubQueryPolicy(), directory)
            recorder.activate(self.capture(recorder))
            features = {
                "action": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
                "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
                **recording_features(),
            }
            root = Path(directory) / "dataset"
            dataset = LeRobotDataset.create("local/qtoken-test", 30, root=root,
                                           features=features, robot_type="test", use_videos=False)
            try:
                for step in range(2):
                    dataset.add_frame({"action": np.zeros(2, np.float32),
                                       "observation.state": np.zeros(2, np.float32),
                                       **recorder.frame(step / 30), "task": "test"})
                dataset.save_episode()
            finally:
                dataset.finalize()
            table = pd.read_parquet(next((root / "data").rglob("*.parquet")))
            self.assertEqual(len(table), 2)
            self.assertEqual(table.iloc[0][PREFIX + "inference_id"], 0)
            np.testing.assert_array_equal(table.iloc[1][PREFIX + "measured_valid"], np.ones(8, bool))
            np.testing.assert_allclose(table.iloc[1][PREFIX + "measured"][:2], [.01, .01])

    def test_act_captures_metrics_from_the_same_single_forward_and_clears_on_reset(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, batch):
                self.calls += 1
                return torch.ones(1, 2, 2), (None, None), {"metric_pred": torch.arange(8)[None].float()}

        policy = object.__new__(ACTPolicy)
        torch.nn.Module.__init__(policy)
        policy.config = SimpleNamespace(image_features={}, action_target="dataset_action",
                                        temporal_ensemble_coeff=None, n_action_steps=2)
        policy.model = Network()
        policy.reset()
        self.assertIsNone(policy.latest_inference_metrics())
        actions = policy.predict_action_chunk({})
        self.assertEqual(policy.model.calls, 1)
        self.assertEqual(tuple(actions.shape), (1, 2, 2))
        torch.testing.assert_close(policy.latest_inference_metrics(), torch.arange(8)[None].float())
        copy = policy.latest_inference_metrics()
        copy.zero_()
        self.assertEqual(policy.latest_inference_metrics()[0, -1].item(), 7)
        policy.reset()
        self.assertIsNone(policy.latest_inference_metrics())


if __name__ == "__main__":
    unittest.main()
