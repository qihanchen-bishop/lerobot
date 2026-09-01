from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "mycode"))

from pretrained_semantic_segmenter import (  # noqa: E402
    EXPECTED_LABELS,
    ActionSupervisedTinyUNetSegmenter,
    FrozenTinyUNetSegmenter,
)
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


def test_seg_v3_checkpoint_supports_front_only_seven_class_probabilities():
    expected_labels = (
        "background",
        "occluder",
        "object",
        "region",
        "tool",
        "leftarm",
        "rightarm",
    )
    segmenter = FrozenTinyUNetSegmenter(
        PROJECT_ROOT / "mycode/tool/seg_v3/best.pt",
        expected_labels=expected_labels,
    )

    probabilities = segmenter(torch.rand(1, 3, 72, 128))

    assert segmenter.labels == expected_labels
    assert segmenter.input_size == (270, 480)
    assert probabilities.shape == (1, 7, 72, 128)
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert all(not parameter.requires_grad for parameter in segmenter.parameters())


def test_unet_sem_identity_ids_cover_single_and_dual_view_modalities():
    class Args:
        experiment = "UNET-SEM"

    args = Args()
    args.rgb_keys = ["observation.images.front"]
    assert act_identity_ids_for_experiment(args) == ([0, 0], [1, 0])

    args.rgb_keys = ["observation.images.front", "observation.images.side"]
    assert act_identity_ids_for_experiment(args) == ([0, 1, 0, 1], [1, 1, 0, 0])

    args.experiment = "UNET-SEM-V3-F-NOEMB"
    args.rgb_keys = ["observation.images.front"]
    assert act_identity_ids_for_experiment(args) == (None, None)


def test_action_supervised_segmenter_only_trains_fuse0_convolutions_and_head():
    expected_labels = (
        "background",
        "occluder",
        "object",
        "region",
        "tool",
        "leftarm",
        "rightarm",
    )
    segmenter = ActionSupervisedTinyUNetSegmenter(
        PROJECT_ROOT / "data/bettersetup_v5/models/unet_front_v4_r1/best.pt",
        expected_labels=expected_labels,
    )
    segmenter.train()

    trainable_names = {name for name, _ in segmenter.trainable_named_parameters()}
    assert trainable_names == {
        "fuse0.net.0.weight",
        "fuse0.net.3.weight",
        "head.weight",
        "head.bias",
    }
    assert not segmenter.network.training
    assert all(
        not module.training
        for module in segmenter.network.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )

    logits = segmenter(torch.rand(1, 3, 36, 64))
    assert logits.shape == (1, 7, 36, 64)
    assert logits.requires_grad
    logits.square().mean().backward()
    assert all(parameter.grad is not None for _, parameter in segmenter.trainable_named_parameters())
    assert all(
        parameter.grad is None
        for parameter in segmenter.parameters()
        if not parameter.requires_grad
    )
