"""
CNN-based Input Transform for neural networks.

This module implements CNNInputTransform which uses convolutional neural networks
to transform inputs into excitatory and inhibitory pathways.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.activations.factory import ActivationFactory


class CNN(BaseNetwork):

    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (channels, height, width)
        hidden_dims: List[int],  # List of channel sizes for each conv layer
        kernel_sizes: List[int],  # Kernel size for all conv layers
        strides: List[int],  # Stride for all conv layers
        activation: str = "relu",
        output_dim: Optional[int] = None
    ):
        super().__init__()

        # Input validation
        input_dim, height, width = input_shape

        # Build CNN backbone for excitatory pathway
        layers = []
        in_dim = input_dim
        current_h, current_w = height, width

        activation_factory = ActivationFactory()

        for hidden_dim, kernel_size, stride in zip(
            hidden_dims, kernel_sizes, strides
        ):
            # Add convolutional layer
            conv = nn.Conv2d(in_dim, hidden_dim, kernel_size, stride)
            if activation == "relu":
                nn.init.kaiming_normal_(conv.weight)
                nn.init.zeros_(conv.bias)
            else:
                nn.init.xavier_normal_(conv.weight)
                nn.init.zeros_(conv.bias)
            layers.append(conv)

            layers.append(
                activation_factory.create(
                    act_type=activation,
                    output_dim= hidden_dim
                )
            )

            in_dim = hidden_dim
            current_h = ((current_h - kernel_size) // stride) + 1
            current_w = ((current_w - kernel_size) // stride) + 1

        # Flatten layer
        layers.append(nn.Flatten(start_dim=-3))
        self.output_dim = hidden_dims[-1] * current_h * current_w

        if output_dim is not None:
            layers.append(nn.Linear(self.output_dim, output_dim))
            self.output_dim = output_dim
       
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct shape (add batch dimension if needed)
        if x.dim() == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension: (1, C, H, W)

        return self.layers(x)


__all__ = ["CNN"] 