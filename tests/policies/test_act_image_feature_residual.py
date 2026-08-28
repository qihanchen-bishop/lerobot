import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, ACTPolicy, IMAGE_FEATURE_RESIDUALS
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def test_act_accepts_one_feature_residual_per_image_without_adding_image_tokens():
    config = ACTConfig(
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    model = ACT(config).eval()
    image = torch.rand(1, 3, 64, 64)
    base_batch = {OBS_IMAGES: [image], OBS_STATE: torch.rand(1, 2)}

    with torch.no_grad():
        actions_without_residual = model(base_batch)[0]
        actions_with_zero_residual = model(
            {**base_batch, IMAGE_FEATURE_RESIDUALS: [torch.zeros(1, 512, 2, 2)]}
        )[0]

    torch.testing.assert_close(actions_without_residual, actions_with_zero_residual)


def test_act_adds_camera_and_modality_embeddings_to_shared_backbone_features():
    image_features = {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        "observation.images.front_seg": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    config = ACTConfig(
        input_features={**image_features, OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_camera_ids=[0, 1, 0],
        image_modality_ids=[0, 0, 1],
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    model = ACT(config)
    batch = {
        OBS_IMAGES: [torch.rand(1, 3, 64, 64) for _ in range(3)],
        OBS_STATE: torch.rand(1, 2),
    }

    model(batch)[0].square().mean().backward()

    assert model.image_camera_embed.num_embeddings == 2
    assert model.image_modality_embed.num_embeddings == 2
    assert model.image_camera_embed.weight.grad is not None
    assert model.image_modality_embed.weight.grad is not None


def test_act_rejects_image_identity_ids_with_wrong_length():
    with pytest.raises(ValueError, match="one id per image feature"):
        ACTConfig(
            input_features={
                "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            image_camera_ids=[0],
        )


@pytest.mark.parametrize("mode", ["default", "zero", "gated"])
def test_camera_embedding_ablation_preserves_all_shared_parameter_initialization(mode):
    def make_config(camera_ids, camera_mode="default"):
        return ACTConfig(
            input_features={
                "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            image_camera_ids=camera_ids,
            image_camera_embedding_mode=camera_mode,
            pretrained_backbone_weights=None,
            chunk_size=2,
            n_action_steps=2,
            use_vae=False,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )

    torch.manual_seed(1234)
    baseline = ACT(make_config(None))
    torch.manual_seed(1234)
    camera_model = ACT(make_config([0, 1], mode))

    baseline_parameters = dict(baseline.named_parameters())
    camera_parameters = dict(camera_model.named_parameters())
    expected_extra = {"image_camera_embed.weight"}
    if mode == "gated":
        expected_extra.add("image_camera_embed_gate")
    assert set(camera_parameters) - set(baseline_parameters) == expected_extra
    for name, parameter in baseline_parameters.items():
        torch.testing.assert_close(parameter, camera_parameters[name])


def test_zero_camera_embedding_starts_at_baseline_and_receives_gradient():
    config = ACTConfig(
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_camera_ids=[0, 1],
        image_camera_embedding_mode="zero",
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    model = ACT(config)
    torch.testing.assert_close(
        model.image_camera_embed.weight,
        torch.zeros_like(model.image_camera_embed.weight),
    )

    model(
        {
            OBS_IMAGES: [torch.rand(1, 3, 64, 64) for _ in range(2)],
            OBS_STATE: torch.rand(1, 2),
        }
    )[0].square().mean().backward()

    assert model.image_camera_embed.weight.grad is not None
    assert model.image_camera_embed.weight.grad.abs().sum() > 0


def test_gated_camera_embedding_starts_with_baseline_output_and_trains_gate():
    def make_config(camera_ids=None, mode="default"):
        return ACTConfig(
            input_features={
                "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            },
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            image_camera_ids=camera_ids,
            image_camera_embedding_mode=mode,
            pretrained_backbone_weights=None,
            chunk_size=2,
            n_action_steps=2,
            use_vae=False,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
        )

    torch.manual_seed(1234)
    baseline = ACT(make_config()).eval()
    torch.manual_seed(1234)
    gated = ACT(make_config([0, 1], "gated")).eval()
    batch = {
        OBS_IMAGES: [torch.rand(1, 3, 64, 64) for _ in range(2)],
        OBS_STATE: torch.rand(1, 2),
    }

    baseline_actions = baseline(batch)[0]
    gated_actions = gated(batch)[0]

    torch.testing.assert_close(gated_actions, baseline_actions)
    assert gated.image_camera_embed.weight.std().item() == pytest.approx(0.02, abs=0.005)
    assert gated.image_camera_embed_gate.item() == 0.0
    gated_actions.square().mean().backward()
    assert gated.image_camera_embed_gate.grad is not None
    assert gated.image_camera_embed_gate.grad.abs() > 0


def test_nonzero_gated_camera_and_modality_embeddings_train_from_first_step():
    config = ACTConfig(
        input_features={
            "observation.images.front_seg": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.side_seg": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_camera_ids=[0, 1, 0, 1],
        image_camera_embedding_mode="gated",
        image_camera_embedding_gate_init=0.01,
        image_modality_ids=[1, 1, 0, 0],
        image_modality_embedding_mode="gated",
        image_modality_embedding_gate_init=0.01,
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    model = ACT(config)
    actions = model(
        {
            OBS_IMAGES: [torch.rand(1, 3, 64, 64) for _ in range(4)],
            OBS_STATE: torch.rand(1, 2),
        }
    )[0]
    actions.square().mean().backward()

    assert model.image_camera_embed_gate.item() == pytest.approx(0.01)
    assert model.image_modality_embed_gate.item() == pytest.approx(0.01)
    assert model.image_camera_embed.weight.grad is not None
    assert model.image_camera_embed.weight.grad.abs().sum() > 0
    assert model.image_modality_embed.weight.grad is not None
    assert model.image_modality_embed.weight.grad.abs().sum() > 0
    assert model.image_camera_embed_gate.grad is not None
    assert model.image_modality_embed_gate.grad is not None


def test_gated_camera_diagnostics_measure_scale_and_restore_action_ablation_state():
    config = ACTConfig(
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_camera_ids=[0, 1],
        image_camera_embedding_mode="gated",
        image_camera_embedding_gate_init=0.01,
        pretrained_backbone_weights=None,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    policy = ACTPolicy(config)
    batch = {
        "observation.images.front": torch.rand(1, 3, 64, 64),
        "observation.images.side": torch.rand(1, 3, 64, 64),
        OBS_STATE: torch.rand(1, 2),
        ACTION: torch.rand(1, 2, 2),
        "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
    }

    loss, logs = policy(batch)
    assert logs["camera_embedding_gate"] == pytest.approx(0.01)
    assert logs["camera_embedding_effective_rms"] > 0
    assert logs["camera_image_0_feature_rms"] > 0
    assert logs["camera_image_0_embedding_to_feature_ratio"] > 0
    loss.backward()
    gradient_diagnostics = policy.camera_embedding_gradient_diagnostics()
    assert gradient_diagnostics["camera_embedding_gate_grad_abs"] > 0
    assert gradient_diagnostics["camera_embedding_weight_grad_rms"] > 0

    saved_gate = policy.model.image_camera_embed_gate.detach().clone()
    saved_ids = list(policy.config.image_camera_ids)
    diagnostics = policy.camera_embedding_action_ablation(batch)

    assert diagnostics["camera_embedding_action_disable_delta_rms"] > 0
    assert diagnostics["camera_embedding_action_swap_delta_rms"] > 0
    torch.testing.assert_close(policy.model.image_camera_embed_gate, saved_gate)
    assert policy.config.image_camera_ids == saved_ids
    assert policy.training
