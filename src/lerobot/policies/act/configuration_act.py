#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Configuration class for the Action Chunking Transformers policy.

    Defaults are configured for training on bimanual Aloha tasks like "insertion" or "transfer".

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_features` and `output_features`.

    Notes on the inputs and outputs:
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.images." they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - May optionally work without an "observation.state" key for the proprioceptive robot state.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        chunk_size: The size of the action prediction "chunks" in units of environment steps.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            This should be no greater than the chunk size. For example, if the chunk size size 100, you may
            set this to 50. This would mean that the model predicts 100 steps worth of actions, runs 50 in the
            environment, and throws the other 50 out.
        input_features: A dictionary defining the PolicyFeature of the input data for the policy. The key represents
            the input data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        output_features: A dictionary defining the PolicyFeature of the output data for the policy. The key represents
            the output data name, and the value is PolicyFeature, which consists of FeatureType and shape attributes.
        normalization_mapping: A dictionary that maps from a str value of FeatureType (e.g., "STATE", "VISUAL") to
            a corresponding NormalizationMode (e.g., NormalizationMode.MIN_MAX)
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        pretrained_backbone_weights: Pretrained weights from torchvision to initialize the backbone.
            `None` means no pretrained weights.
        replace_final_stride_with_dilation: Whether to replace the ResNet's final 2x2 stride with a dilated
            convolution.
        pre_norm: Whether to use "pre-norm" in the transformer blocks.
        dim_model: The transformer blocks' main hidden dimension.
        n_heads: The number of heads to use in the transformer blocks' multi-head attention.
        dim_feedforward: The dimension to expand the transformer's hidden dimension to in the feed-forward
            layers.
        feedforward_activation: The activation to use in the transformer block's feed-forward layers.
        n_encoder_layers: The number of transformer layers to use for the transformer encoder.
        n_decoder_layers: The number of transformer layers to use for the transformer decoder.
        use_vae: Whether to use a variational objective during training. This introduces another transformer
            which is used as the VAE's encoder (not to be confused with the transformer encoder - see
            documentation in the policy class).
        latent_dim: The VAE's latent dimension.
        n_vae_encoder_layers: The number of transformer layers to use for the VAE's encoder.
        temporal_ensemble_coeff: Coefficient for the exponential weighting scheme to apply for temporal
            ensembling. Defaults to None which means temporal ensembling is not used. `n_action_steps` must be
            1 when using this feature, as inference needs to happen at every step to form an ensemble. For
            more information on how ensembling works, please see `ACTTemporalEnsembler`.
        dropout: Dropout to use in the transformer layers (see code for details).
        kl_weight: The weight to use for the KL-divergence component of the loss if the variational objective
            is enabled. Loss is then calculated as: `reconstruction_loss + kl_weight * kld_loss`.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Note: Although the original ACT implementation has 7 for `n_decoder_layers`, there is a bug in the code
    # that means only the first layer is used. Here we match the original implementation by setting this to 1.
    # See this issue https://github.com/tonyzhaozh/act/issues/25#issue-2258740521.
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    # Note: the value used in ACT when temporal ensembling is enabled is 0.01.
    temporal_ensemble_coeff: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0
    action_representation: str = "absolute"
    anchor_loss_weight: float = 0.25
    motion_loss_weight: float = 0.25
    reconstruction_loss_weight: float = 0.5
    anchor_motion_gate_init: float = 0.1
    action_target: str = "dataset_action"
    follower_state_key: str = "observation.state"
    gripper_loss_weight: float = 0.2
    gripper_positive_weight: float = 1.0
    metric_mode: str | None = None
    metric_dim: int = 2
    image_camera_ids: list[int] | None = None
    image_camera_embedding_mode: str = "default"
    image_camera_embedding_std: float = 0.02
    image_camera_embedding_gate_init: float = 0.0
    image_modality_ids: list[int] | None = None
    image_modality_embedding_mode: str = "default"
    image_modality_embedding_std: float = 0.02
    image_modality_embedding_gate_init: float = 0.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )
        if self.metric_mode not in {None, "encoder_tokens", "decoder_autoregressive"}:
            raise ValueError(
                "`metric_mode` must be None, 'encoder_tokens', or 'decoder_autoregressive'. "
                f"Got {self.metric_mode}."
            )
        if self.metric_dim <= 0:
            raise ValueError(f"`metric_dim` must be positive. Got {self.metric_dim}.")
        if self.action_representation not in {"absolute", "anchor_offset"}:
            raise ValueError(
                "`action_representation` must be 'absolute' or 'anchor_offset'. "
                f"Got {self.action_representation!r}."
            )
        if self.action_representation == "anchor_offset" and self.chunk_size < 2:
            raise ValueError("`anchor_offset` action representation requires `chunk_size >= 2`.")
        if self.action_target not in {
            "dataset_action",
            "follower_next_state",
            "follower_delta",
            "follower_anchor_delta",
            "follower_joint_delta_gripper_absolute",
            "follower_joint_anchor_delta_gripper_absolute",
        }:
            raise ValueError(
                "`action_target` must be 'dataset_action', 'follower_next_state', "
                "'follower_delta', 'follower_anchor_delta', or "
                "'follower_joint_delta_gripper_absolute', or "
                "'follower_joint_anchor_delta_gripper_absolute'. "
                f"Got {self.action_target!r}."
            )
        if self.action_target != "dataset_action" and self.action_representation != "absolute":
            raise ValueError(
                f"{self.action_target} requires the standard absolute ACT decoder representation."
            )
        action_loss_weights = {
            "anchor_loss_weight": self.anchor_loss_weight,
            "motion_loss_weight": self.motion_loss_weight,
            "reconstruction_loss_weight": self.reconstruction_loss_weight,
        }
        if any(weight < 0 for weight in action_loss_weights.values()):
            raise ValueError(f"Anchor-and-motion loss weights must be non-negative: {action_loss_weights}.")
        if self.action_representation == "anchor_offset" and sum(action_loss_weights.values()) <= 0:
            raise ValueError("At least one anchor-and-motion action loss weight must be positive.")
        if self.gripper_loss_weight < 0:
            raise ValueError(f"`gripper_loss_weight` must be non-negative, got {self.gripper_loss_weight}.")
        if self.gripper_positive_weight <= 0:
            raise ValueError(
                f"`gripper_positive_weight` must be positive, got {self.gripper_positive_weight}."
            )
        image_count = len(self.image_features)
        if self.image_camera_embedding_mode not in {"default", "zero", "gated"}:
            raise ValueError(
                "`image_camera_embedding_mode` must be 'default', 'zero', or 'gated'. "
                f"Got {self.image_camera_embedding_mode!r}."
            )
        if self.image_camera_embedding_std <= 0:
            raise ValueError(
                f"`image_camera_embedding_std` must be positive. Got {self.image_camera_embedding_std}."
            )
        if self.image_camera_ids is None and self.image_camera_embedding_mode != "default":
            raise ValueError(
                "A non-default `image_camera_embedding_mode` requires `image_camera_ids`."
            )
        if self.image_modality_embedding_mode not in {"default", "zero", "gated"}:
            raise ValueError(
                "`image_modality_embedding_mode` must be 'default', 'zero', or 'gated'. "
                f"Got {self.image_modality_embedding_mode!r}."
            )
        if self.image_modality_embedding_std <= 0:
            raise ValueError(
                "`image_modality_embedding_std` must be positive. "
                f"Got {self.image_modality_embedding_std}."
            )
        if self.image_modality_ids is None and self.image_modality_embedding_mode != "default":
            raise ValueError(
                "A non-default `image_modality_embedding_mode` requires `image_modality_ids`."
            )
        for name, ids in (
            ("image_camera_ids", self.image_camera_ids),
            ("image_modality_ids", self.image_modality_ids),
        ):
            if ids is None:
                continue
            if len(ids) != image_count:
                raise ValueError(
                    f"`{name}` must contain one id per image feature ({image_count}), got {ids}."
                )
            if any(identifier < 0 for identifier in ids):
                raise ValueError(f"`{name}` values must be non-negative, got {ids}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.action_target in {
            "follower_delta",
            "follower_anchor_delta",
            "follower_joint_delta_gripper_absolute",
            "follower_joint_anchor_delta_gripper_absolute",
        }:
            state_feature = self.input_features.get(self.follower_state_key)
            action_feature = self.output_features.get("action")
            if state_feature is None:
                raise ValueError(
                    f"{self.action_target} requires input feature {self.follower_state_key!r}."
                )
            if action_feature is None or state_feature.shape != action_feature.shape:
                raise ValueError(
                    f"{self.action_target} requires matching follower state and action dimensions, got "
                    f"state={getattr(state_feature, 'shape', None)} and "
                    f"action={getattr(action_feature, 'shape', None)}."
                )
            if (
                self.action_target
                in {
                    "follower_joint_delta_gripper_absolute",
                    "follower_joint_anchor_delta_gripper_absolute",
                }
                and action_feature.shape[0] < 2
            ):
                raise ValueError(
                    "follower_joint_delta_gripper_absolute requires at least one joint and one gripper."
                )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
