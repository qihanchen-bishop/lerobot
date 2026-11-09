import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.diffusion import modeling_diffusion as module
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors


def config(target="follower_joint_delta_gripper_absolute", n_obs=1):
    return DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(1,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        action_target=target,
        n_obs_steps=n_obs,
        horizon=8,
        n_action_steps=3,
        device="cpu",
        pretrained_backbone_weights=None,
    )


def stats():
    return {
        "observation.state": {
            "min": torch.tensor([-100.0, -100.0, 0.0]),
            "max": torch.tensor([100.0, 100.0, 1.0]),
        },
        "action": {"min": torch.tensor([-2.0, -4.0, 0.0]), "max": torch.tensor([2.0, 4.0, 1.0])},
    }


class FixedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def generate_actions(self, batch, noise=None):
        raw = torch.tensor([[[1.0, -2.0, 1.0], [0.5, 1.0, 0.0], [-1.0, 0.5, 1.0]]])
        s = stats()["action"]
        return (2 * (raw - s["min"]) / (s["max"] - s["min"]) - 1).expand(
            batch["observation.state"].shape[0], -1, -1
        )


@pytest.mark.parametrize("n_obs", [1, 2])
def test_queue_integrates_once_at_current_state(monkeypatch, n_obs):
    monkeypatch.setattr(module, "DiffusionModel", FixedModel)
    cfg = config(n_obs=n_obs)
    policy = module.DiffusionPolicy(cfg, dataset_stats=stats())
    _, post = make_diffusion_pre_post_processors(cfg, stats())

    def batch(q):
        return {
            "observation.state": torch.tensor([[q / 100.0, 0.2, 0.0]]),
            "observation.environment_state": torch.zeros(1, 1),
        }

    actions = [post(policy.select_action(batch(q))).clone() for q in [10.0, 50.0, 70.0, 30.0]]
    expected = torch.tensor([[11.0, 18.0, 1.0], [11.5, 19.0, 0.0], [10.5, 19.5, 1.0], [31.0, 18.0, 1.0]])
    torch.testing.assert_close(torch.cat(actions), expected, atol=2e-5, rtol=0)
    policy.reset()
    torch.testing.assert_close(
        post(policy.select_action(batch(0.0))), torch.tensor([[1.0, 18.0, 1.0]]), atol=2e-5, rtol=0
    )


def test_checkpoint_restores_delta_statistics(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "DiffusionModel", FixedModel)
    policy = module.DiffusionPolicy(config(), dataset_stats=stats())
    policy.save_pretrained(tmp_path)
    loaded = module.DiffusionPolicy.from_pretrained(tmp_path, local_files_only=True)
    assert loaded.config.action_target == policy.config.action_target
    for k, value in policy.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], value)
    batch = {"observation.state": torch.zeros(1, 3), "observation.environment_state": torch.zeros(1, 1)}
    torch.testing.assert_close(policy.select_action(dict(batch)), loaded.select_action(dict(batch)))


def test_absolute_dp_unchanged(monkeypatch):
    monkeypatch.setattr(module, "DiffusionModel", FixedModel)
    policy = module.DiffusionPolicy(config("dataset_action"))
    assert not any(k.startswith("follower_") for k in policy.state_dict())
    batch = {"observation.state": torch.zeros(1, 3), "observation.environment_state": torch.zeros(1, 1)}
    expected = FixedModel(config()).generate_actions(batch)[:, 0]
    torch.testing.assert_close(policy.select_action(batch), expected)


def test_uninitialized_decoder_fails_closed(monkeypatch):
    monkeypatch.setattr(module, "DiffusionModel", FixedModel)
    policy = module.DiffusionPolicy(config())
    with pytest.raises(ValueError, match="statistics"):
        policy.select_action(
            {"observation.state": torch.zeros(1, 3), "observation.environment_state": torch.zeros(1, 1)}
        )


def test_wrong_normalization_rejected():
    with pytest.raises(ValueError, match="MIN_MAX"):
        DiffusionConfig(
            action_target="follower_joint_delta_gripper_absolute",
            normalization_mapping={"STATE": NormalizationMode.MEAN_STD, "ACTION": NormalizationMode.MIN_MAX},
        )


def test_constant_ranges_and_binary_threshold():
    decoded = module.decode_follower_joint_delta_minmax_actions(
        torch.tensor([[[0.0, 0.0], [0.0, -0.01]]]),
        torch.tensor([[-1.0, -1.0]]),
        action_min=torch.tensor([0.0, 0.0]),
        action_max=torch.tensor([0.0, 1.0]),
        state_min=torch.tensor([2.0, 0.0]),
        state_max=torch.tensor([2.0, 1.0]),
    )
    raw = (decoded + 1) / 2 * torch.tensor([1e-8, 1.0])
    torch.testing.assert_close(raw[0, :, 0], torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(raw[0, :, 1], torch.tensor([1.0, 0.0]))
