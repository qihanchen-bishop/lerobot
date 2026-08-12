import math

import torch
from torch import nn

from mycode.semantic_servo import (
    ActionConditionedSemanticDynamics,
    BoxConstrainedCLFProjector,
    PhaseSemanticFeatureExtractor,
    SemanticEventTrigger,
    SSACTRuntimeConfig,
    SSACTRuntimeController,
    SoftSemanticStateExtractor,
    hazard_nll,
)


def test_soft_semantic_state_extracts_normalized_geometry():
    probabilities = torch.zeros(1, 2, 4, 10, 20)
    probabilities[:, :, 0, :5] = 1.0  # cloth
    probabilities[:, :, 1, 4:6, 8:12] = 1.0  # object
    probabilities[:, :, 2, 4:6, 10:14] = 1.0  # goal
    probabilities[:, :, 3, 4:6, 2:4] = 1.0  # actuator
    # Resolve deliberately overlapping hard test shapes into soft exclusive labels.
    probabilities[:, :, 0] *= 1.0 - probabilities[:, :, 1:].amax(dim=2)
    probabilities[:, :, 2] *= 1.0 - probabilities[:, :, 1]

    extractor = SoftSemanticStateExtractor()
    state = extractor(probabilities)

    assert state.shape == (1, 30)
    per_view = state.reshape(1, 2, 15)
    assert torch.all((per_view >= 0.0) & (per_view <= 1.0))
    assert torch.all(per_view[:, :, 12] < per_view[:, :, 13])  # object-goal is closer than actuator-object
    assert torch.allclose(per_view[:, :, 14], torch.ones_like(per_view[:, :, 14]), atol=1e-4)


def test_phase_features_gate_only_the_unreliable_view_class():
    probabilities = torch.zeros(1, 2, 4, 8, 8)
    probabilities[:, :, 0, :4] = 1.0
    probabilities[:, :, 1, 4:6, 2:4] = 1.0
    probabilities[:, :, 2, 6:, 5:7] = 1.0
    probabilities[:, :, 3, 2:4, 1:3] = 1.0
    reliability = torch.ones(2, 4)
    reliability[1, 3] = 0.25
    extractor = PhaseSemanticFeatureExtractor(reliability)

    features = extractor(probabilities).reshape(1, 2, 12)

    assert features.shape == (1, 2, 12)
    unchanged = torch.tensor([0, 1, 2, 4, 5, 6, 7])
    assert torch.allclose(features[:, 0].index_select(1, unchanged), features[:, 1].index_select(1, unchanged))
    assert torch.allclose(features[:, 1, 3], features[:, 0, 3] * 0.25)
    assert torch.allclose(features[:, 1, 8:10], features[:, 0, 8:10] * 0.25)
    assert torch.allclose(features[:, 1, 10], features[:, 0, 10])
    assert torch.allclose(features[:, 1, 11], features[:, 0, 11] * 0.25)


def test_action_conditioned_dynamics_shapes_and_padding_loss():
    model = ActionConditionedSemanticDynamics(
        semantic_dim=6,
        robot_state_dim=3,
        action_dim=2,
        prediction_offsets=[1, 3],
    )
    current = torch.rand(4, 6)
    robot = torch.rand(4, 3)
    actions = torch.rand(4, 3, 2)
    mean, log_std = model(current, robot, actions)

    assert mean.shape == (4, 2, 6)
    assert log_std.shape == mean.shape
    targets = mean.detach().clone()
    valid = torch.tensor([[True, True], [True, False], [False, False], [True, True]])
    loss = model.gaussian_nll(mean, log_std, targets, valid)
    assert torch.isfinite(loss)


def test_action_conditioned_dynamics_delta_head_is_a_semantic_state_delta():
    model = ActionConditionedSemanticDynamics(
        semantic_dim=3,
        robot_state_dim=2,
        action_dim=1,
        prediction_offsets=[1],
    )
    with torch.no_grad():
        model.delta_head.weight.zero_()
        model.delta_head.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    current = torch.tensor([[0.4, 0.5, 0.6]])
    mean, _ = model(current, torch.zeros(1, 2), torch.zeros(1, 1, 1))

    assert torch.allclose(mean[:, 0], current + torch.tensor([[0.1, -0.2, 0.3]]))


def test_ssact_runtime_separates_predicted_semantic_delta_from_qp_action_correction():
    class DeltaSemanticDynamics(nn.Module):
        def forward(self, semantic_state, robot_state, actions):
            predictions = []
            for offset in (1, 8):
                semantic_delta = torch.zeros_like(semantic_state)
                semantic_delta[:, 0] = 0.4 * actions[:, :offset, 0].mean(dim=1)
                predictions.append(semantic_state + semantic_delta)
            means = torch.stack(predictions, dim=1)
            return means, torch.full_like(means, -3.0)

    semantic_state = torch.zeros(1, 28)
    semantic_state[:, [0, 14]] = 0.8
    nominal_actions = torch.zeros(1, 8, 10)
    controller = SSACTRuntimeController(
        SSACTRuntimeConfig(
            mode="active",
            adaptive_horizon=True,
            minimum_execution_steps=1,
            maximum_execution_steps=4,
            maximum_action_residual=0.015,
        )
    )

    controlled, report = controller.apply(
        semantic_dynamics=DeltaSemanticDynamics(),
        semantic_state=semantic_state,
        robot_state=torch.zeros(1, 10),
        nominal_actions=nominal_actions,
        phase_probabilities=torch.tensor([[0.9, 0.025, 0.025, 0.025, 0.025]]),
        prediction_offsets=(1, 8),
    )

    assert report["semantic_delta_l2"] == 0.0
    assert report["qp_active"]
    assert report["correction_max"] > 0.0
    assert not torch.equal(controlled[:, : report["execution_steps"]], nominal_actions[:, : report["execution_steps"]])
    assert torch.equal(controlled[:, report["execution_steps"] :], nominal_actions[:, report["execution_steps"] :])


def test_hazard_nll_prefers_correct_event_logits():
    events = torch.tensor([2, 3])
    good = torch.tensor([[-8.0, 8.0, 0.0], [-8.0, -8.0, 8.0]])
    bad = -good
    assert hazard_nll(good, events) < hazard_nll(bad, events)


def test_box_clf_projector_satisfies_soft_constraint_and_bounds():
    projector = BoxConstrainedCLFProjector(slack_penalty=10_000.0)
    result = projector.project(
        torch.tensor([1.0, 0.5]),
        lie_f=0.0,
        lie_g=torch.tensor([1.0, 0.0]),
        clf_value=1.0,
        decay_rate=0.5,
        lower=torch.tensor([-1.0, -1.0]),
        upper=torch.tensor([1.0, 1.0]),
    )

    assert result.active
    assert result.constraint_before > 0.0
    assert result.constraint_after <= 1e-5
    assert -1.0 <= result.action[0] <= 1.0
    assert math.isclose(float(result.action[1]), 0.5, abs_tol=1e-5)


def test_event_trigger_reports_all_replan_reasons():
    decision = SemanticEventTrigger().evaluate(
        previous_phase=1,
        current_phase=2,
        segmentation_confidence=0.2,
        normalized_innovation=4.0,
        clf_value=1.0,
        certified_clf_upper_bound=0.8,
    )
    assert decision.replan
    assert set(decision.reasons) == {
        "phase_changed",
        "low_segmentation_confidence",
        "semantic_prediction_outlier",
        "clf_progress_violation",
    }
