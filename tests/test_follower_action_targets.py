import pandas as pd
import torch

from lerobot.policies.act.modeling_act import (
    NORMALIZATION_EPS,
    decode_follower_anchor_delta_actions,
    decode_follower_delta_actions,
    decode_follower_joint_anchor_delta_gripper_actions,
    decode_follower_joint_delta_gripper_actions,
    follower_anchor_delta_targets,
    follower_joint_anchor_delta_gripper_targets,
)
from mycode.train_lerobot_policy import _replace_action_with_next_follower_state


def test_follower_next_state_targets_shift_within_each_episode():
    dataframe = pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1, 1],
            "frame_index": [0, 1, 0, 1, 2],
            "observation.state": [[0.0], [1.0], [10.0], [11.0], [12.0]],
            "action": [[100.0], [101.0], [110.0], [111.0], [112.0]],
        }
    )

    result = _replace_action_with_next_follower_state(
        dataframe,
        action_key="action",
        follower_state_key="observation.state",
    )

    assert result["action"].tolist() == [[1.0], [1.0], [11.0], [12.0], [12.0]]
    assert dataframe["action"].tolist() == [[100.0], [101.0], [110.0], [111.0], [112.0]]


def test_follower_next_state_targets_follow_frame_index_order():
    dataframe = pd.DataFrame(
        {
            "episode_index": [4, 4, 4],
            "frame_index": [2, 0, 1],
            "observation.state": [[2.0], [0.0], [1.0]],
            "action": [[20.0], [0.0], [10.0]],
        }
    )

    result = _replace_action_with_next_follower_state(
        dataframe,
        action_key="action",
        follower_state_key="observation.state",
    )

    assert result["action"].tolist() == [[2.0], [1.0], [2.0]]


def test_follower_delta_targets_are_one_step_differences():
    dataframe = pd.DataFrame(
        {
            "episode_index": [0, 0, 1, 1, 1],
            "frame_index": [0, 1, 0, 1, 2],
            "observation.state": [[0.0], [1.0], [10.0], [11.0], [13.0]],
            "action": [[100.0], [101.0], [110.0], [111.0], [112.0]],
        }
    )

    result = _replace_action_with_next_follower_state(
        dataframe,
        action_key="action",
        follower_state_key="observation.state",
        target_mode="follower_delta",
    )

    assert result["action"].tolist() == [[1.0], [0.0], [1.0], [2.0], [0.0]]


def test_follower_joint_delta_gripper_targets_mix_delta_and_absolute_values():
    dataframe = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "observation.state": [[1.0, 4.0, 0.0], [2.0, 3.0, 0.0], [4.0, 6.0, 1.0]],
            "action": [[0.0, 0.0, 0.0]] * 3,
        }
    )

    result = _replace_action_with_next_follower_state(
        dataframe,
        action_key="action",
        follower_state_key="observation.state",
        target_mode="follower_joint_delta_gripper_absolute",
    )

    assert result["action"].tolist() == [
        [1.0, -1.0, 0.0],
        [2.0, 3.0, 1.0],
        [0.0, 0.0, 1.0],
    ]


def test_follower_delta_decoder_returns_absolute_targets_after_postprocessing():
    raw_deltas = torch.tensor([[[1.0], [-2.0], [3.0]]])
    delta_mean = torch.tensor([0.25])
    delta_std = torch.tensor([2.0])
    normalized_deltas = (raw_deltas - delta_mean) / delta_std

    encoded_actions = decode_follower_delta_actions(
        normalized_deltas,
        torch.tensor([[0.0]]),
        delta_mean=delta_mean,
        delta_std=delta_std,
        state_mean=torch.tensor([10.0]),
        state_std=torch.tensor([4.0]),
    )
    postprocessed_actions = encoded_actions * delta_std + delta_mean

    torch.testing.assert_close(postprocessed_actions, torch.tensor([[[11.0], [9.0], [12.0]]]))


def test_follower_joint_delta_gripper_decoder_integrates_only_joints():
    action_mean = torch.tensor([0.5, -0.5, 0.25])
    action_std = torch.tensor([2.0, 4.0, 0.5])
    raw_joint_deltas = torch.tensor([[[1.0, -2.0], [-0.5, 3.0], [2.0, -1.0]]])
    normalized_predictions = torch.empty(1, 3, 3)
    normalized_predictions[..., :-1] = (
        raw_joint_deltas - action_mean[:-1]
    ) / action_std[:-1]
    normalized_predictions[..., -1] = torch.tensor([[-2.0, 3.0, 0.0]])

    encoded_actions = decode_follower_joint_delta_gripper_actions(
        normalized_predictions,
        torch.tensor([[0.0, 0.0, 0.0]]),
        action_mean=action_mean,
        action_std=action_std,
        state_mean=torch.tensor([10.0, 20.0, 0.0]),
        state_std=torch.tensor([1.0, 1.0, 1.0]),
    )
    postprocessed_actions = encoded_actions * action_std + action_mean

    torch.testing.assert_close(
        postprocessed_actions,
        torch.tensor([[[11.0, 18.0, 0.0], [10.5, 21.0, 1.0], [12.5, 20.0, 1.0]]]),
    )


def test_follower_joint_anchor_delta_gripper_targets_keep_absolute_gripper():
    action_mean = torch.tensor([8.0, 0.25])
    action_std = torch.tensor([2.0, 0.5])
    state_mean = torch.tensor([10.0, 0.25])
    state_std = torch.tensor([4.0, 0.5])
    raw_future = torch.tensor([[[11.0, 0.0], [13.0, 1.0], [9.0, 1.0]]])
    normalized_future = (raw_future - action_mean) / action_std

    targets = follower_joint_anchor_delta_gripper_targets(
        normalized_future,
        torch.tensor([[0.0, -0.5]]),
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
    )

    expected = normalized_future.clone()
    expected[..., 0] = torch.tensor([[0.5, 1.5, -0.5]])
    torch.testing.assert_close(targets, expected)


def test_follower_joint_anchor_delta_gripper_decoder_does_not_accumulate_joints():
    action_mean = torch.tensor([8.0, 0.25])
    action_std = torch.tensor([2.0, 0.5])
    predictions = torch.tensor([[[0.5, -2.0], [1.5, 3.0], [-0.5, 0.0]]])

    encoded_actions = decode_follower_joint_anchor_delta_gripper_actions(
        predictions,
        torch.tensor([[0.0, -0.5]]),
        action_mean=action_mean,
        action_std=action_std,
        state_mean=torch.tensor([10.0, 0.25]),
        state_std=torch.tensor([4.0, 0.5]),
    )
    postprocessed_actions = encoded_actions * action_std + action_mean

    torch.testing.assert_close(
        postprocessed_actions,
        torch.tensor([[[11.0, 0.0], [13.0, 1.0], [9.0, 1.0]]]),
    )


def test_follower_anchor_delta_targets_use_one_planning_frame_anchor():
    action_mean = torch.tensor([8.0])
    action_std = torch.tensor([2.0])
    state_mean = torch.tensor([10.0])
    state_std = torch.tensor([4.0])
    raw_future = torch.tensor([[[11.0], [13.0], [9.0]]])
    normalized_future = (raw_future - action_mean) / (action_std + NORMALIZATION_EPS)

    targets = follower_anchor_delta_targets(
        normalized_future,
        torch.tensor([[0.0]]),
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
    )

    expected_offsets = (raw_future - 10.0) / (action_std + NORMALIZATION_EPS)
    torch.testing.assert_close(targets, expected_offsets)


def test_follower_anchor_delta_decoder_does_not_accumulate_offsets():
    action_mean = torch.tensor([8.0])
    action_std = torch.tensor([2.0])
    offsets = torch.tensor([[[0.5], [1.5], [-0.5]]])

    encoded_actions = decode_follower_anchor_delta_actions(
        offsets,
        torch.tensor([[0.0]]),
        action_mean=action_mean,
        action_std=action_std,
        state_mean=torch.tensor([10.0]),
        state_std=torch.tensor([4.0]),
    )
    postprocessed_actions = encoded_actions * action_std + action_mean

    torch.testing.assert_close(postprocessed_actions, torch.tensor([[[11.0], [13.0], [9.0]]]))
