"""
AlexNet feature encoder with optional FC layers for dendritic modeling experiments.

Copied and extended from:
    dendritic_modeling/networks/architectures/classical/cnn/alexnet.py

Provides AlexNetFCEncoder: a frozen pretrained AlexNet backbone that optionally
includes 0, 1, or 2 of AlexNet's fully-connected layers (fc6, fc7). Used as the
encoder in Bernardo's FC-replacement experiments — the backbone is always frozen;
only the downstream EI layer and output head are trained.

AlexNet classifier structure (torchvision):
    classifier[0]: Dropout(0.5)
    classifier[1]: Linear(9216 -> 4096)  = fc6
    classifier[2]: ReLU
    classifier[3]: Dropout(0.5)
    classifier[4]: Linear(4096 -> 4096)  = fc7
    classifier[5]: ReLU
    classifier[6]: Linear(4096 -> 1000)  = fc8  (in AlexNetFC8Decoder, not here)

num_fc_layers controls how many FC layers are included in the encoder:
    0: features only          -> output_dim = 9216  (K=2 condition)
    1: features + fc6         -> output_dim = 4096  (K=1 condition)
    2: features + fc6 + fc7   -> output_dim = 4096  (baseline condition)
"""

import torch
from torchvision import models
from torchvision.models import AlexNet_Weights

from dendritic_modeling.networks.base import BaseNetwork


class AlexNetFCEncoder(BaseNetwork):
    """
    Frozen pretrained AlexNet encoder with configurable FC depth.

    The entire encoder is frozen (requires_grad=False). Dropout layers from the
    original classifier are excluded so that the encoder always behaves
    deterministically regardless of train/eval mode.

    Args:
        num_fc_layers: Number of FC layers (excluding fc8) to include.
            0 -> conv features only, output_dim=9216 (K=2)
            1 -> features + fc6,     output_dim=4096 (K=1)
            2 -> features + fc6+fc7, output_dim=4096 (baseline)
    """

    def __init__(self, num_fc_layers: int = 0, **kwargs):
        super().__init__()

        if num_fc_layers not in (0, 1, 2):
            raise ValueError(f"num_fc_layers must be 0, 1, or 2, got {num_fc_layers}")

        alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        layers = list(alexnet.features)
        layers.append(torch.nn.AdaptiveAvgPool2d((6, 6)))
        layers.append(torch.nn.Flatten(start_dim=-3))

        if num_fc_layers >= 1:
            # Add fc6 (skip dropout at classifier[0])
            layers.append(alexnet.classifier[1])  # Linear(9216 -> 4096)
            layers.append(alexnet.classifier[2])  # ReLU

        if num_fc_layers >= 2:
            # Add fc7 (skip dropout at classifier[3])
            layers.append(alexnet.classifier[4])  # Linear(4096 -> 4096)
            layers.append(alexnet.classifier[5])  # ReLU

        self.layers = torch.nn.Sequential(*layers)

        self.output_dim = 4096 if num_fc_layers >= 1 else 9216

        # Freeze entire encoder — only the EI layer and output head are trained
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (C, H, W)
            x = x[None, ...]  # Add batch dimension: (1, C, H, W)

        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        output: torch.Tensor = self.layers(x)
        return output.reshape(*batch_shape, -1)

    def get_output_dim(self):
        return self.output_dim


__all__ = ["AlexNetFCEncoder"]
