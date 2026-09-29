"""Transfer-learning backbones for static sign-language digit images."""

from __future__ import annotations

from torch import nn
from torchvision import models

SUPPORTED_IMAGE_MODELS = ("mobilenet_v3_small", "efficientnet_b0")


def build_digit_model(name: str, num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED_IMAGE_MODELS:
        raise ValueError(
            f"Unsupported image model {name!r}; choose one of {SUPPORTED_IMAGE_MODELS}"
        )
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
