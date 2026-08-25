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
