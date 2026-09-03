import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import (
    ANCHOR_PRED,
    MOTION_OFFSET_PRED,
    ACTPolicy,
    decode_anchor_motion_actions,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def make_anchor_motion_config(**overrides) -> ACTConfig:
    kwargs = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
        "chunk_size": 4,
        "n_action_steps": 4,
        "dim_model": 32,
        "n_heads": 4,
        "dim_feedforward": 64,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "use_vae": False,
        "latent_dim": 4,
        "action_representation": "anchor_offset",
    }
    kwargs.update(overrides)
    return ACTConfig(**kwargs)


def test_decode_anchor_motion_actions_uses_one_shared_anchor():
    anchor = torch.tensor([[[1.0, -2.0]]])
    offsets = torch.tensor([[[0.5, 1.0], [2.0, -3.0]]])

    decoded = decode_anchor_motion_actions(anchor, offsets)

    torch.testing.assert_close(
        decoded,
        torch.tensor([[[1.0, -2.0], [1.5, -1.0], [3.0, -5.0]]]),
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"action_representation": "unknown"}, "action_representation"),
        ({"chunk_size": 1, "n_action_steps": 1}, "chunk_size >= 2"),
        (
            {
                "anchor_loss_weight": 0.0,
                "motion_loss_weight": 0.0,
                "reconstruction_loss_weight": 0.0,
            },
            "At least one",
        ),
    ],
)
def test_anchor_motion_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_anchor_motion_config(**overrides)


def test_anchor_motion_model_outputs_absolute_chunk_and_backpropagates():
    torch.manual_seed(7)
    policy = ACTPolicy(make_anchor_motion_config())
    batch = {
        OBS_STATE: torch.randn(2, 4),
        OBS_ENV_STATE: torch.randn(2, 3),
        ACTION: torch.randn(2, 4, 4),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }

    loss, logs = policy(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert policy.model.anchor_head is not policy.model.action_head
    assert policy.model.anchor_head.weight.grad is not None
    assert policy.model.action_head.weight.grad is not None
    assert policy.model.anchor_motion_gate.grad is not None
    assert set(
        [
            "anchor_l1",
            "motion_offset_l1",
            "reconstructed_action_l1",
            "endpoint_l1",
            "velocity_l1",
            "anchor_motion_gate",
        ]
    ).issubset(logs)

    policy.eval()
    actions, _, aux = policy.model({OBS_STATE: batch[OBS_STATE], OBS_ENV_STATE: batch[OBS_ENV_STATE]})
    expected = decode_anchor_motion_actions(aux[ANCHOR_PRED], aux[MOTION_OFFSET_PRED])
    torch.testing.assert_close(actions, expected)


class _FixedAnchorMotionModel(nn.Module):
    def __init__(self, actions, anchor, offsets):
        super().__init__()
        self.actions = actions
        self.anchor = anchor
        self.offsets = offsets

    def forward(self, batch):
        return self.actions, (None, None), {
            ANCHOR_PRED: self.anchor,
            MOTION_OFFSET_PRED: self.offsets,
            "anchor_motion_gate": torch.tensor(0.1),
        }


def test_anchor_motion_loss_supervises_anchor_motion_and_reconstruction_separately():
    policy = ACTPolicy(make_anchor_motion_config())
    target = torch.tensor(
        [[[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]]
    )
    anchor_target = target[:, :1]
    offset_target = target[:, 1:] - anchor_target
    anchor_pred = anchor_target + 1.0
    offset_pred = offset_target + 2.0
    actions_pred = decode_anchor_motion_actions(anchor_pred, offset_pred)
    policy.model = _FixedAnchorMotionModel(actions_pred, anchor_pred, offset_pred)

    loss, logs = policy(
        {
            OBS_STATE: torch.zeros(1, 4),
            OBS_ENV_STATE: torch.zeros(1, 3),
            ACTION: target,
            "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
        }
    )

    assert logs["anchor_l1"] == pytest.approx(1.0)
    assert logs["motion_offset_l1"] == pytest.approx(2.0)
    assert logs["reconstructed_action_l1"] == pytest.approx(2.5)
    assert loss.item() == pytest.approx(2.0)
