import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT, IMAGE_FEATURE_RESIDUALS
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
