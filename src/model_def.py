"""ResNet-34 vision model adapted for 5-channel geospatial imagery."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DataCenterVisionNet(nn.Module):
    """ResNet-34 backbone with 5-channel input and 3-class impact head."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=5,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = rgb_mean
            if old_conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.backbone.conv1 = new_conv

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
