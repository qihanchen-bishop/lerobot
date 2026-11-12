"""Offline inference adapter for UR-SemQ checkpoints; contains no robot interfaces."""

import json
from pathlib import Path

import torch
from ur_semq import FRONT_KEY, LABELS, SEMANTIC_KEY, digest
from view_qtoken import FrozenExportedUNet

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


class URSemQInference:
    """Raw RGB/depth-preview [0,1] BCHW + follower radians -> absolute commands."""

    def __init__(self, checkpoint, device="cuda"):
        checkpoint = Path(checkpoint)
        self.metadata = json.loads((checkpoint / "ur_semq.json").read_text())
        if self.metadata["experiment"] != "UR-SemQ-v1" or tuple(self.metadata["labels"]) != LABELS:
            raise ValueError("Not a UR-SemQ-v1 checkpoint.")
        if digest(checkpoint / "semantic_model/model.ts") != self.metadata["semantic_model_sha256"]:
            raise ValueError("Segmentation bundle hash mismatch.")
        cfg = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        if not isinstance(cfg, ACTConfig):
            raise ValueError("UR-SemQ requires an ACT checkpoint.")
        cfg.device = device
        self.policy = (
            ACTPolicy.from_pretrained(checkpoint, config=cfg, local_files_only=True, strict=True)
            .to(device)
            .eval()
        )
        self.pre, self.post = make_pre_post_processors(
            cfg,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "normalizer_processor": {"device": device},
            },
        )
        self.segmenter = FrozenExportedUNet(
            checkpoint / "semantic_model/model.ts", expected_labels=LABELS
        ).to(device)
        self.scales = torch.tensor(self.metadata["scales"])

    def reset(self):
        self.policy.reset()

    @torch.inference_mode()
    def predict_action_chunk(self, observations):
        batch = dict(observations)
        probabilities = self.segmenter(batch[FRONT_KEY])
        ids = probabilities.argmax(1)
        batch[SEMANTIC_KEY] = self.segmenter.palette[ids].permute(0, 3, 1, 2).contiguous()
        actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        descriptors = self.policy.latest_inference_metrics() * self.scales
        return actions, descriptors
