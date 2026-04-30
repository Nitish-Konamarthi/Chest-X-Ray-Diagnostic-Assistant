"""
DenseNet model definitions for CheXNet.

Provides DenseNet-121 (primary), DenseNet-169, and DenseNet-201 architectures
with pre-trained ImageNet weights and a sigmoid classification head for
multi-label chest pathology detection.
"""

import torch
import torch.nn as nn
import torchvision


class DenseNet121(nn.Module):
    """CheXNet — DenseNet-121 with sigmoid output for 14-class pathology detection."""

    def __init__(self, class_count: int = 14, pretrained: bool = True):
        super().__init__()

        if pretrained:
            try:
                from torchvision.models import DenseNet121_Weights
                self.densenet121 = torchvision.models.densenet121(
                    weights=DenseNet121_Weights.IMAGENET1K_V1,
                )
            except (ImportError, TypeError):
                self.densenet121 = torchvision.models.densenet121(pretrained=True)
        else:
            self.densenet121 = torchvision.models.densenet121(weights=None)

        in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features, class_count),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet121(x)


class DenseNet169(nn.Module):
    """DenseNet-169 variant (alternative architecture)."""

    def __init__(self, class_count: int = 14, pretrained: bool = True):
        super().__init__()

        if pretrained:
            try:
                from torchvision.models import DenseNet169_Weights
                self.densenet169 = torchvision.models.densenet169(
                    weights=DenseNet169_Weights.IMAGENET1K_V1,
                )
            except (ImportError, TypeError):
                self.densenet169 = torchvision.models.densenet169(pretrained=True)
        else:
            self.densenet169 = torchvision.models.densenet169(weights=None)

        in_features = self.densenet169.classifier.in_features
        self.densenet169.classifier = nn.Sequential(
            nn.Linear(in_features, class_count),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet169(x)


class DenseNet201(nn.Module):
    """DenseNet-201 variant (alternative architecture)."""

    def __init__(self, class_count: int = 14, pretrained: bool = True):
        super().__init__()

        if pretrained:
            try:
                from torchvision.models import DenseNet201_Weights
                self.densenet201 = torchvision.models.densenet201(
                    weights=DenseNet201_Weights.IMAGENET1K_V1,
                )
            except (ImportError, TypeError):
                self.densenet201 = torchvision.models.densenet201(pretrained=True)
        else:
            self.densenet201 = torchvision.models.densenet201(weights=None)

        in_features = self.densenet201.classifier.in_features
        self.densenet201.classifier = nn.Sequential(
            nn.Linear(in_features, class_count),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet201(x)
