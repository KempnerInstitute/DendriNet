"""
AlexNet-based Input Transform for neural networks.

This module implements AlexNetInputTransform which uses a pre-trained AlexNet
to transform inputs into excitatory and inhibitory pathways.
"""

import torch.nn as nn


class AlexNetInputTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input)
    using convolutional layers.
    """

    def __init__(self, excitatory_dim: int, inhibitory_dim: int):
        super().__init__()
        import torchvision.models as models

        # Load AlexNet without classifier
        alexnet = models.alexnet(pretrained=True)
        self.shared = nn.Sequential(
            *list(alexnet.features), nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten()
        )

        # Create new heads for excitatory and inhibitory outputs
        self.head_exc = nn.Sequential(nn.Linear(9216, excitatory_dim), nn.Sigmoid())
        self.head_inh = nn.Sequential(nn.Linear(9216, inhibitory_dim), nn.Sigmoid())

        self.excitatory_dim = excitatory_dim
        self.inhibitory_dim = inhibitory_dim

    def forward(self, x):
        x = self.shared(x)
        excitatory_input = self.head_exc(x)
        inhibitory_input = self.head_inh(x)
        return excitatory_input, inhibitory_input


__all__ = ["AlexNetInputTransform"] 