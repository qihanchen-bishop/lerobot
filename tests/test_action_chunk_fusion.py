import pytest
import torch

from mycode.action_chunk_fusion import ActionChunkFusionPlanner, fuse_action_chunks


def test_fuse_action_chunks_aligns_history_and_decays_weight_to_zero():
    historical = torch.arange(50, 100, dtype=torch.float32).unsqueeze(-1)
    new = torch.full((100, 1), 100.0)

    fused = fuse_action_chunks(
        historical,
        new,
        fusion_steps=25,
        initial_history_weight=0.5,
    )

    assert fused[0].item() == pytest.approx(75.0)
    assert fused[24].item() == pytest.approx(100.0)
    assert fused[25].item() == pytest.approx(100.0)
    assert torch.all(fused[:25] <= 100.0)


def test_planner_replans_after_requested_steps_and_uses_unexecuted_history():
    planner = ActionChunkFusionPlanner(
        replan_steps=2,
        fusion_steps=2,
        initial_history_weight=0.5,
    )
    planner.update(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))

    assert planner.pop_action().item() == 0.0
    assert planner.pop_action().item() == 1.0
    assert planner.needs_replan

    overlap = planner.update(torch.tensor([[10.0], [20.0], [30.0], [40.0]]))

    assert overlap == 2
    assert planner.pop_action().item() == pytest.approx(6.0)
    assert planner.pop_action().item() == pytest.approx(20.0)


def test_planner_preserves_new_chunk_when_fusion_is_disabled():
    planner = ActionChunkFusionPlanner(replan_steps=1, fusion_steps=0, initial_history_weight=0.5)
    planner.update(torch.tensor([[[1.0], [2.0]]]))
    assert planner.pop_action().item() == 1.0
    planner.update(torch.tensor([[[3.0], [4.0]]]))
    assert planner.pop_action().item() == 3.0


def test_planner_accepts_a_runtime_execution_length_update():
    planner = ActionChunkFusionPlanner(replan_steps=4)
    planner.update(torch.arange(6, dtype=torch.float32).unsqueeze(-1))
    planner.set_replan_steps(1)

    assert planner.pop_action().item() == 0.0
    assert planner.needs_replan
