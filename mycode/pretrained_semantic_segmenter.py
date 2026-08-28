"""Frozen TinyUNet semantic segmentation for policy-side semantic inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


EXPECTED_LABELS = ("background", "occluder", "object", "region", "tool")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """Architecture used by ``tool/seg_v2/best.pt``."""

    def __init__(self, num_classes: int, width: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 6
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            SeparableConv(c1, c1),
        )
        self.down1 = SeparableConv(c1, c2, stride=2)
        self.down2 = SeparableConv(c2, c3, stride=2)
        self.down3 = SeparableConv(c3, c4, stride=2)
        self.fuse2 = SeparableConv(c4 + c3, c3)
        self.fuse1 = SeparableConv(c3 + c2, c2)
        self.fuse0 = SeparableConv(c2 + c1, c1)
        self.head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skip0 = self.stem(x)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        x = self.down3(skip2)
        x = F.interpolate(x, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse2(torch.cat([x, skip2], dim=1))
        x = F.interpolate(x, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse1(torch.cat([x, skip1], dim=1))
        x = F.interpolate(x, size=skip0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse0(torch.cat([x, skip0], dim=1))
        return self.head(x)


class FrozenTinyUNetSegmenter(nn.Module):
    """Load a TinyUNet checkpoint and always run it without parameter updates."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        expected_labels: tuple[str, ...] | None = EXPECTED_LABELS,
    ) -> None:
        super().__init__()
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained segmentation checkpoint not found: {path}")

        checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
        labels = tuple(checkpoint.get("labels", ()))
        if expected_labels is not None and labels != expected_labels:
            raise ValueError(
                f"Segmentation checkpoint labels must be {expected_labels}, got {labels} from {path}."
            )
        if not labels or labels[0] != "background" or len(set(labels)) != len(labels):
            raise ValueError(
                "Segmentation checkpoint labels must be unique, non-empty, and start with "
                f"'background'; got {labels} from {path}."
            )
        image_size = checkpoint.get("image_size")
        if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
            raise ValueError(f"Segmentation checkpoint has invalid image_size: {image_size!r}")
        width = int(checkpoint.get("args", {}).get("width", 0))
        if width <= 0:
            width = int(checkpoint["model"]["stem.0.weight"].shape[0])

        self.network = TinyUNet(num_classes=len(labels), width=width)
        self.network.load_state_dict(checkpoint["model"], strict=True)
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

        input_width, input_height = (int(value) for value in image_size)
        self.input_size = (input_height, input_width)
        self.labels = labels
        self.checkpoint_path = str(path)
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.train(False)

    def train(self, mode: bool = True) -> FrozenTinyUNetSegmenter:
        # The policy itself trains, but this pretrained network and its BatchNorm
        # statistics must remain frozen.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB input, got {tuple(rgb.shape)}.")
        output_size = tuple(rgb.shape[-2:])
        resized = rgb.to(dtype=torch.float32).clamp(0.0, 1.0)
        if output_size != self.input_size:
            resized = F.interpolate(
                resized,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        normalized = (resized - self.mean) / self.std
        probabilities = torch.softmax(self.network(normalized), dim=1)
        if output_size != self.input_size:
            probabilities = F.interpolate(
                probabilities,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return probabilities
