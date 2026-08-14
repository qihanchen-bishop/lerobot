from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from train_mask_act_policy import mask_quality_scores_to_weights  # noqa: E402
from train_mask_act_policy import MaskACTPolicy  # noqa: E402


def test_soft_quality_weights_are_continuous_and_saturate_at_full_score():
    scores = torch.tensor([0.0, 0.30, 0.60, 0.95, 1.0])

    weights = mask_quality_scores_to_weights(scores, full_score=0.95, gamma=1.0)

    torch.testing.assert_close(
        weights,
        torch.tensor([0.0, 0.30 / 0.95, 0.60 / 0.95, 1.0, 1.0]),
    )


def test_soft_quality_weight_gamma_controls_attenuation():
    scores = torch.tensor([0.475, 0.95])

    weights = mask_quality_scores_to_weights(scores, full_score=0.95, gamma=2.0)

    torch.testing.assert_close(weights, torch.tensor([0.25, 1.0]))


def test_segmentation_ce_scales_each_sample_by_its_quality():
    policy = MaskACTPolicy.__new__(MaskACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.semantic_class_weights = torch.tensor([0.5, 1.0])
    policy.dice_loss_weight = 0.0
    policy.mask_suffixes = ["object"]
    logits = torch.tensor(
        [
            [[[5.0]], [[-5.0]]],
            [[[-5.0]], [[5.0]]],
        ],
        requires_grad=True,
    )
    targets = torch.ones((2, 1, 1), dtype=torch.long)

    _, weighted_ce, _, _ = policy.semantic_segmentation_loss(
        [logits],
        [targets],
        [torch.tensor([[0.25], [1.0]])],
    )
    expected = (
        torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        * torch.tensor([[[0.25]], [[1.0]]])
    ).sum() / 1.25

    torch.testing.assert_close(weighted_ce, expected)


@pytest.mark.parametrize(
    ("full_score", "gamma", "message"),
    [(0.0, 1.0, "full_score"), (1.1, 1.0, "full_score"), (0.95, 0.0, "gamma")],
)
def test_soft_quality_weights_reject_invalid_parameters(full_score, gamma, message):
    with pytest.raises(ValueError, match=message):
        mask_quality_scores_to_weights(
            torch.ones(1),
            full_score=full_score,
            gamma=gamma,
        )
