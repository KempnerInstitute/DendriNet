"""
AlexNet-based Input Transform for neural networks.

This module implements AlexNetInputTransform which uses a pre-trained AlexNet
to transform inputs into excitatory and inhibitory pathways.
"""

import torch
from torch import nn
from torchvision import models

from dendritic_modeling.networks.base import BaseNetwork

class AlexNet(BaseNetwork):

    def __init__(self):
        super().__init__()

        # Load AlexNet
        alexnet = models.alexnet(pretrained=True)
        self.hidden = nn.Sequential(
            *list(alexnet.features), 
            nn.AdaptiveAvgPool2d((6, 6)), 
            nn.Flatten(start_dim=-3)
        )
        self.output_dim = 9216

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)
        


__all__ = ["AlexNet"] 