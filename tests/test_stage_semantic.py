from pathlib import Path
import sys

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mycode"))
from generate_stage_supervision import _stable_state_machine  # noqa: E402
from stage_semantic import (  # noqa: E402
    EVENT_NAMES,
    PHASE_NAMES,
    RELATION_NAMES,
    TRANSITION_NAMES,
    PhaseConditionedSemanticAdapter,
    StageAwareTemporalModel,
    StageSupervisionStore,
    stage_losses,
)
from train_mask_act_policy import MaskACTPolicy  # noqa: E402


def test_stable_state_machine_uses_persistence_and_can_rollback():
    scores = np.zeros((30, 4), dtype=np.float32)
    scores[2:, 0] = 1.0
    scores[7:, 1] = 1.0
    scores[12:, 2] = 1.0
    scores[17:, 3] = 1.0
    scores[10:13, 1] = 0.0
    confidence = np.ones_like(scores)

    phases = _stable_state_machine(
        scores,
        confidence,
        on_threshold=0.8,
        off_threshold=0.35,
        advance_frames=2,
        rollback_frames=2,
        minimum_transition_quality=0.5,
    )

    assert phases[0] == PHASE_NAMES.index("expose")
    assert PHASE_NAMES.index("transport") in phases
    assert np.any(np.diff(phases) < 0)
    assert phases[-1] == PHASE_NAMES.index("done")


def test_stage_model_and_weighted_losses_have_expected_outputs():
    batch_size = 4
    model = StageAwareTemporalModel(semantic_dim=30, robot_state_dim=10, hidden_dim=32)
    outputs = model(torch.rand(batch_size, 8, 30), torch.rand(batch_size, 10))
    target_batch = {
        "stage_phase_target": torch.nn.functional.one_hot(
            torch.arange(batch_size) % len(PHASE_NAMES), len(PHASE_NAMES)
        ).float(),
        "stage_phase_weight": torch.ones(batch_size),
        "stage_event_target": torch.rand(batch_size, len(EVENT_NAMES)),
        "stage_event_weight": torch.ones(batch_size, len(EVENT_NAMES)),
        "stage_progress_target": torch.rand(batch_size),
        "stage_progress_weight": torch.ones(batch_size),
        "stage_transition_target": torch.arange(batch_size) % len(TRANSITION_NAMES),
        "stage_transition_weight": torch.ones(batch_size),
        "stage_relation_target": torch.rand(batch_size, len(RELATION_NAMES)),
        "stage_relation_weight": torch.ones(batch_size, len(RELATION_NAMES)),
    }

    losses, logs = stage_losses(outputs, target_batch)
    total = sum(losses.values())
    total.backward()

    assert set(losses) == {"phase", "event", "progress", "transition", "relation"}
    assert all(torch.isfinite(loss) for loss in losses.values())
    assert "stage_phase_accuracy" in logs
    assert model.phase_head.weight.grad is not None


def test_stage_adapter_starts_at_zero_and_action_path_is_detached():
    policy = MaskACTPolicy.__new__(MaskACTPolicy)
    torch.nn.Module.__init__(policy)
    policy.experiment = "SSACT-3"
    policy.semantic_adapter = PhaseConditionedSemanticAdapter(
        output_channels=16,
        context_dim=12,
        base_channels=8,
    )
    probabilities = torch.rand(2, 5, 64, 96, requires_grad=True)
    phase_logits = torch.rand(2, len(PHASE_NAMES), requires_grad=True)
    phase = torch.softmax(phase_logits, dim=-1)
    context = torch.rand(2, 12, requires_grad=True)

    initial = policy.semantic_feature_residuals(
        [probabilities],
        scale=1.0,
        phase_probabilities=phase,
        stage_context=context,
        stage_confidence=torch.ones(2),
    )[0]
    torch.testing.assert_close(initial, torch.zeros_like(initial))

    torch.nn.init.normal_(policy.semantic_adapter.output_proj.weight)
    residual = policy.semantic_feature_residuals(
        [probabilities],
        scale=1.0,
        phase_probabilities=phase,
        stage_context=context,
        stage_confidence=torch.ones(2),
    )[0]
    residual.square().mean().backward()

    assert probabilities.grad is None
    assert phase_logits.grad is None
    assert context.grad is None
    assert policy.semantic_adapter.output_proj.weight.grad is not None


def test_stage_store_history_never_crosses_episode_boundary(tmp_path: Path):
    total_frames = 6
    feature_dim = 4
    path = tmp_path / "stage.npz"
    np.savez_compressed(
        path,
        semantic_features=np.arange(total_frames * feature_dim, dtype=np.float32).reshape(
            total_frames, feature_dim
        ),
        phase_probabilities=np.eye(len(PHASE_NAMES), dtype=np.float32)[[0, 0, 0, 1, 1, 1]],
        phase_confidence=np.ones(total_frames, dtype=np.float32),
        event_targets=np.zeros((total_frames, len(EVENT_NAMES)), dtype=np.float32),
        event_weights=np.ones((total_frames, len(EVENT_NAMES)), dtype=np.float32),
        progress_target=np.zeros(total_frames, dtype=np.float32),
        progress_weight=np.ones(total_frames, dtype=np.float32),
        transition_target=np.zeros(total_frames, dtype=np.int64),
        transition_weight=np.ones(total_frames, dtype=np.float32),
        relation_targets=np.zeros((total_frames, len(RELATION_NAMES)), dtype=np.float32),
        relation_weights=np.ones((total_frames, len(RELATION_NAMES)), dtype=np.float32),
        episode_start_index=np.asarray([0, 0, 0, 3, 3, 3]),
        rgb_keys=np.asarray(["observation.images.front"]),
    )
    store = StageSupervisionStore(path, total_frames=total_frames, history_length=3, history_stride=1)

    enriched = store.add_batch({"index": torch.tensor([1, 3, 5])})
    history = enriched["stage_semantic_history"].numpy()

    np.testing.assert_array_equal(history[0, :, 0], [0, 0, 4])
    np.testing.assert_array_equal(history[1, :, 0], [12, 12, 12])
    np.testing.assert_array_equal(history[2, :, 0], [12, 16, 20])
