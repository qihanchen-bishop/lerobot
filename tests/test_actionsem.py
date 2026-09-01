from argparse import Namespace
from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from train_mask_act_policy import MaskACTPolicy, act_image_keys_for_experiment  # noqa: E402


FRONT = "observation.images.front"
SIDE = "observation.images.side"
FRONT_CLASSES = ["occluder", "object", "region", "tool", "leftarm", "rightarm"]
SIDE_CLASSES = ["occluder", "object", "region", "tool"]


def test_actionsem_visual_inputs_keep_semantics_before_rgb_without_embeddings():
    front_args = Namespace(experiment="ACTIONSEM-F", rgb_keys=[FRONT])
    dual_args = Namespace(experiment="ACTIONSEM-FS", rgb_keys=[FRONT, SIDE])

    assert act_image_keys_for_experiment(front_args) == [f"{FRONT}_semantic", FRONT]
    assert act_image_keys_for_experiment(dual_args) == [
        f"{FRONT}_semantic",
        f"{SIDE}_semantic",
        FRONT,
        SIDE,
    ]


def test_actionsem_asymmetric_losses_use_view_specific_classes_and_quality():
    policy = MaskACTPolicy.__new__(MaskACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.dice_loss_weight = 1.0
    policy.mask_suffixes = FRONT_CLASSES
    policy.view_mask_suffixes = [FRONT_CLASSES, SIDE_CLASSES]
    policy.rgb_keys = [FRONT, SIDE]
    policy._semantic_class_weight_buffer_names = ["weights_front", "weights_side"]
    policy.register_buffer("weights_front", torch.tensor([0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0]))
    policy.register_buffer("weights_side", torch.tensor([0.5, 1.0, 2.0, 1.0, 1.0]))

    front_logits = torch.randn(2, 7, 8, 8, requires_grad=True)
    side_logits = torch.randn(2, 5, 8, 8, requires_grad=True)
    front_target = torch.randint(0, 7, (2, 8, 8))
    side_target = torch.randint(0, 5, (2, 8, 8))
    front_quality = torch.rand(2, 6)
    side_quality = torch.rand(2, 4)

    seg_losses, _, _, logs = policy.semantic_segmentation_losses_by_view(
        [front_logits, side_logits],
        [front_target, side_target],
        [front_quality, side_quality],
    )
    torch.stack(seg_losses).mean().backward()

    assert len(seg_losses) == 2
    assert front_logits.grad is not None
    assert side_logits.grad is not None
    assert "dice_front_leftarm" in logs
    assert "dice_side_tool" in logs
    assert "dice_side_leftarm" not in logs
    assert "dice_object" in logs


def test_actionsem_conflict_projection_caps_action_gradient_per_view():
    policy = MaskACTPolicy.__new__(MaskACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.action_to_seg_conflict_projection = True
    parameter = torch.nn.Parameter(torch.zeros(2))

    metrics = policy.apply_guarded_action_gradients(
        [parameter],
        [torch.tensor([-1.0, 1.0])],
        [torch.tensor([1.0, 0.0])],
        target_ratio=0.05,
    )

    torch.testing.assert_close(parameter.grad, torch.tensor([1.0, 0.05]))
    assert metrics["action_to_seg_grad_conflict"] == 1.0
    assert abs(metrics["action_to_seg_projected_grad_cosine"]) < 1e-6
    assert abs(metrics["action_to_seg_applied_grad_ratio"] - 0.05) < 1e-6


def test_legacy_asem_backward_still_uses_one_shared_segmentation_group():
    policy = MaskACTPolicy.__new__(MaskACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.experiment = "ASEM-1"
    policy.seg_net = torch.nn.Linear(1, 1, bias=False)
    policy.seg_net.weight.data.fill_(1.0)
    policy.action_loss_weight = 1.0
    policy.seg_loss_weight = 1.0
    policy.action_to_seg_conflict_projection = True
    action_loss = -policy.seg_net.weight.sum()
    seg_loss = 0.5 * policy.seg_net.weight.square().sum()
    policy._asem_backward_losses = (action_loss, [seg_loss], 0.05)
    logs = {}

    policy.backward_training_loss(action_loss + seg_loss, logs)

    torch.testing.assert_close(policy.seg_net.weight.grad, torch.ones_like(policy.seg_net.weight))
    assert logs["action_to_seg_grad_conflict"] == 1.0
