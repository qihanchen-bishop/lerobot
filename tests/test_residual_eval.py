import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from residual_eval import ACTResidualRuntime
from mycode.slow_fast_residual import ResidualConfig, ResidualCorrector
from gui_eval_lerobot_policy import EvalPolicyApp


class ResidualEvalTest(unittest.TestCase):
    def test_direct_script_import_without_project_on_python_path(self):
        script_dir = Path(__file__).resolve().parents[1] / "mycode"
        code = (
            f"import sys; sys.path.insert(0, {str(script_dir)!r}); "
            "from residual_eval import checkpoint_identity, ACTResidualRuntime; "
            "from mycode.train_slow_fast_residual import BasePolicyAdapter"
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-c", code], cwd=directory,
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "best.pt"
        self.cfg = ResidualConfig(fast_stride=6, fast_delay=3, correction_steps=9, plan_steps=9)
        m = ResidualCorrector(4, 223, 10, torch.ones(10), self.cfg, "mlp")
        self.joints = [str(i) for i in range(10)]
        torch.save({"spec": m.spec, "model": m.state_dict(), "cache_id": "test",
                    "contract": {"joints": self.joints, "unit": "dataset_native", "scale": [1]*10}}, self.path)
        self.policy = torch.nn.Linear(1, 1)
        self.policy.config = SimpleNamespace(action_target="dataset_action", chunk_size=60,
            image_features={"observation.images.front": None, "observation.images.side": None})
        self.settings = dict(fps=30, replan_interval_steps=30, prediction_steps=60,
                             fusion_steps=0, auto_replan=False, use_amp=False, execution_mode="asynchronous")

    def tearDown(self):
        self.directory.cleanup()

    def load(self, mode="mlp", joints=None):
        return ACTResidualRuntime(self.path, mode, self.policy, None, None,
                                  self.joints, joints or self.joints, self.settings)

    def test_checkpoint_mode_and_joint_order_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "mode/variant"):
            self.load("gru")
        with self.assertRaisesRegex(ValueError, "joint order"):
            self.load(joints=list(reversed(self.joints)))

    def test_timing_mismatch_is_rejected(self):
        self.settings["replan_interval_steps"] = 60
        with self.assertRaisesRegex(ValueError, "replan=30"):
            self.load()

    def test_real_loader_preserves_bounds_and_empty_history(self):
        runtime = self.load()
        try:
            torch.testing.assert_close(runtime.model.bounds, torch.ones(10))
            self.assertEqual(len(runtime.window.records), 0)
        finally:
            runtime.close()

    def test_result_folders_separate_disabled_mlp_and_gru(self):
        config = {**self.settings, "policy_type": "act", "residual_path": str(self.path),
                  "ssact_servo_mode": "off"}
        paths = [EvalPolicyApp._runtime_configuration_dirname({**config, "residual_mode": mode})
                 for mode in ("off", "mlp", "gru")]
        self.assertEqual(len(set(paths)), 3)
        self.assertNotIn("residual", paths[0])


if __name__ == "__main__":
    unittest.main()
