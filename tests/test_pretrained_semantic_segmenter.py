from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mycode"))

from pretrained_semantic_segmenter import EXPECTED_LABELS, FrozenTinyUNetSegmenter  # noqa: E402
from train_mask_act_policy import act_identity_ids_for_experiment  # noqa: E402


def test_seg_v2_checkpoint_is_frozen_and_outputs_resized_probabilities():
    segmenter = FrozenTinyUNetSegmenter(PROJECT_ROOT / "mycode/tool/seg_v2/best.pt")
    segmenter.train()

    probabilities = segmenter(torch.rand(2, 3, 96, 128))

    assert segmenter.labels == EXPECTED_LABELS
    assert probabilities.shape == (2, 5, 96, 128)
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert not segmenter.training
    assert not segmenter.network.training
    assert all(not parameter.requires_grad for parameter in segmenter.parameters())
    assert probabilities.grad_fn is None


def test_unet_sem_identity_ids_cover_single_and_dual_view_modalities():
    class Args:
        experiment = "UNET-SEM"

    args = Args()
    args.rgb_keys = ["observation.images.front"]
    assert act_identity_ids_for_experiment(args) == ([0, 0], [1, 0])

    args.rgb_keys = ["observation.images.front", "observation.images.side"]
    assert act_identity_ids_for_experiment(args) == ([0, 1, 0, 1], [1, 1, 0, 0])
