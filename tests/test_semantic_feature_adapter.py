from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from train_mask_act_policy import MaskACTPolicy, SemanticFeatureAdapter  # noqa: E402


def test_semantic_feature_adapter_matches_resnet18_layer4_shape_and_starts_at_zero():
    adapter = SemanticFeatureAdapter(num_classes=5, output_channels=512, base_channels=8)

    residual = adapter(torch.rand(2, 5, 64, 96))

    assert residual.shape == (2, 512, 2, 3)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


def test_semantic_feature_residual_detaches_segmentation_probabilities():
    policy = MaskACTPolicy.__new__(MaskACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.semantic_adapter = SemanticFeatureAdapter(num_classes=5, output_channels=16, base_channels=8)
    torch.nn.init.normal_(policy.semantic_adapter.output_proj.weight)
    probabilities = torch.rand(2, 5, 32, 32, requires_grad=True)

    residual = policy.semantic_feature_residuals([probabilities], scale=0.5)[0]
    residual.square().mean().backward()

    assert probabilities.grad is None
    assert policy.semantic_adapter.output_proj.weight.grad is not None


def test_semantic_fusion_schedule_warms_up_then_ramps():
    policy = MaskACTPolicy.__new__(MaskACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.experiment = "SEM-1-V2"
    policy.semantic_fusion_warmup_steps = 20
    policy.semantic_fusion_ramp_steps = 10
    policy.train()

    expected = {0: 0.0, 20: 0.0, 25: 0.5, 30: 1.0, 40: 1.0}
    for step, scale in expected.items():
        policy._training_step = step
        assert policy.scheduled_semantic_fusion_scale() == scale

    policy.eval()
    assert policy.scheduled_semantic_fusion_scale() == 1.0
