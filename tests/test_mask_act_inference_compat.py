from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mycode"))

from mask_act_inference import _load_mask_act_state_dict_compatibly  # noqa: E402


class StubMaskACTPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.register_buffer("semantic_palette_view_0", torch.arange(21).reshape(7, 3))
        self._semantic_palette_buffer_names = ["semantic_palette_view_0"]


class MaskACTInferenceCompatibilityTest(unittest.TestCase):
    def test_legacy_checkpoint_may_omit_derived_view_palette(self):
        model = StubMaskACTPolicy()

        initialized = _load_mask_act_state_dict_compatibly(
            model,
            {"weight": torch.tensor([3.0, 4.0])},
        )

        self.assertEqual(initialized, ["semantic_palette_view_0"])
        torch.testing.assert_close(model.weight, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(
            model.semantic_palette_view_0,
            torch.arange(21).reshape(7, 3),
        )

    def test_missing_learned_weight_is_still_rejected(self):
        model = StubMaskACTPolicy()

        with self.assertRaisesRegex(RuntimeError, "missing learned/state keys"):
            _load_mask_act_state_dict_compatibly(model, {})


if __name__ == "__main__":
    unittest.main()
