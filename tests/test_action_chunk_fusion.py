import pytest
import torch

from mycode.action_chunk_fusion import (
    ActionChunkFusionPlanner,
    ActionMagnitudeReplanScheduler,
    fuse_action_chunks,
)


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


def test_action_magnitude_scheduler_replans_earlier_for_larger_motion():
    scheduler = ActionMagnitudeReplanScheduler(
        min_steps=30,
        max_steps=60,
        movement_budget=0.5,
        action_scales=(1.0,),
    )
    large_motion = torch.arange(60, dtype=torch.float32).unsqueeze(-1) * 0.02
    small_motion = torch.arange(60, dtype=torch.float32).unsqueeze(-1) * 0.005

    large_report = scheduler.select(large_motion)
    small_report = scheduler.select(small_motion)

    assert large_report["selected_steps"] == 30
    assert small_report["selected_steps"] == 60
    assert large_report["cumulative_motion_at_max"] > small_report["cumulative_motion_at_max"]


def test_action_magnitude_scheduler_uses_calibrated_action_indices():
    scheduler = ActionMagnitudeReplanScheduler(
        min_steps=2,
        max_steps=4,
        movement_budget=0.5,
        action_scales=(1.0,),
        action_indices=(1,),
    )
    chunk = torch.tensor([[100.0, 0.0], [200.0, 0.1], [300.0, 0.6], [400.0, 0.7]])

    report = scheduler.select(chunk)

    assert report["selected_steps"] == 3
