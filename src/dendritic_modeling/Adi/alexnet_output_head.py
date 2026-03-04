"""
AlexNet fc8 output head for dendritic modeling experiments.

Copied and extended from:
    dendritic_modeling/networks/architectures/classical/cnn/alexnet.py

Provides AlexNetFC8Decoder: pretrained AlexNet fc8 weights (Linear(4096 -> 1000))
loaded as the starting point for the output head, but kept fully trainable.

Bernardo's plan: the weights between the excitatory layer and fc8 are free to
update during training (no constraints, no freezing). Pretrained init gives a
better starting point than random init for a 1000-class head.

The Classifier model applies LogSoftmax after this decoder, consistent with
the cat_nll loss function used in training.
"""

import torch
from torchvision import models
from torchvision.models import AlexNet_Weights

from dendritic_modeling.networks.base import BaseNetwork


class AlexNetFC8Decoder(BaseNetwork):
    """
    Pretrained AlexNet fc8 layer: Linear(4096 -> 1000), fully trainable.

    Weights are initialized from AlexNet_Weights.IMAGENET1K_V1 but are NOT
    frozen — they are updated during training alongside the EI layer.
    The EI layer's n_excitatory_cells must be set to 4096 to match this
    layer's input dimension.
    """

    def __init__(self, **kwargs):
        super().__init__()

        alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # classifier[6]: Linear(4096 -> 1000) with pretrained weights
        self.fc8 = alexnet.classifier[6]

        self.output_dim = 1000

        # Weights remain trainable — requires_grad=True by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc8(x)

    def get_output_dim(self):
        return self.output_dim


__all__ = ["AlexNetFC8Decoder"]
