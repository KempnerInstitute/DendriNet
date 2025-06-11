"""
CNN-based Input Transform for neural networks.

This module implements CNNInputTransform which uses convolutional neural networks
to transform inputs into excitatory and inhibitory pathways.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class CNNInputTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input)
    using convolutional layers.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (channels, height, width)
        excitatory_hidden_dims: List[int],  # List of channel sizes for each conv layer
        inhibitory_hidden_dims: List[int],  # List of channel sizes for each conv layer
        excitatory_kernel_sizes: List[int],  # Kernel size for all conv layers
        inhibitory_kernel_sizes: List[int],  # Kernel size for all conv layers
        excitatory_strides: List[int],  # Stride for all conv layers
        inhibitory_strides: List[int],  # Stride for all conv layers
    ):
        super().__init__()

        # Input validation
        input_dim, height, width = input_shape

        # Build CNN backbone for excitatory pathway
        exc_layers = []
        in_dim = input_dim
        current_h, current_w = height, width

        for hidden_dim, kernel_size, stride in zip(
            excitatory_hidden_dims[:-1],
            excitatory_kernel_sizes[:-1],
            excitatory_strides[:-1],
        ):
            # Add convolutional layer
            exc_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size, stride)
            nn.init.kaiming_normal_(exc_conv.weight)
            nn.init.zeros_(exc_conv.bias)
            exc_layers.append(exc_conv)
            exc_layers.append(nn.ReLU())

            in_dim = hidden_dim
            current_h = ((current_h - kernel_size) // stride) + 1
            current_w = ((current_w - kernel_size) // stride) + 1

        # Add final convolutional layer
        exc_conv = nn.Conv2d(
            in_dim,
            excitatory_hidden_dims[-1],
            excitatory_kernel_sizes[-1],
            excitatory_strides[-1],
        )
        nn.init.xavier_normal_(exc_conv.weight)
        nn.init.zeros_(exc_conv.bias)
        exc_layers.append(exc_conv)
        exc_layers.append(nn.Sigmoid())

        current_h = (
            (current_h - excitatory_kernel_sizes[-1]) // excitatory_strides[-1]
        ) + 1
        current_w = (
            (current_w - excitatory_kernel_sizes[-1]) // excitatory_strides[-1]
        ) + 1

        # Flatten layer
        exc_layers.append(nn.Flatten(start_dim=-3))

        # Calculate flattened dimension
        self.excitatory_dim = excitatory_hidden_dims[-1] * current_h * current_w

        # Build CNN backbone for inhibitory pathway
        inh_layers = []
        in_dim = input_dim
        current_h, current_w = height, width

        for hidden_dim, kernel_size, stride in zip(
            inhibitory_hidden_dims[:-1],
            inhibitory_kernel_sizes[:-1],
            inhibitory_strides[:-1],
        ):
            # Add convolutional layer
            inh_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size, stride)
            nn.init.kaiming_normal_(inh_conv.weight)
            nn.init.zeros_(inh_conv.bias)
            inh_layers.append(inh_conv)
            inh_layers.append(nn.ReLU())

            in_dim = hidden_dim
            current_h = ((current_h - kernel_size) // stride) + 1
            current_w = ((current_w - kernel_size) // stride) + 1

        # Add final convolutional layer
        inh_conv = nn.Conv2d(
            in_dim,
            inhibitory_hidden_dims[-1],
            inhibitory_kernel_sizes[-1],
            inhibitory_strides[-1],
        )
        nn.init.xavier_normal_(inh_conv.weight)
        nn.init.zeros_(inh_conv.bias)
        inh_layers.append(inh_conv)
        inh_layers.append(nn.Sigmoid())

        current_h = (
            (current_h - inhibitory_kernel_sizes[-1]) // inhibitory_strides[-1]
        ) + 1
        current_w = (
            (current_w - inhibitory_kernel_sizes[-1]) // inhibitory_strides[-1]
        ) + 1

        # Flatten layer
        inh_layers.append(nn.Flatten(start_dim=-3))

        # Calculate flattened dimension
        self.inhibitory_dim = inhibitory_hidden_dims[-1] * current_h * current_w

        # Create sequential models
        self.exc_encoding = nn.Sequential(*exc_layers)
        self.inh_encoding = nn.Sequential(*inh_layers)

    def forward(self, x):
        # Ensure input has correct shape (add batch dimension if needed)
        if x.dim() == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension: (1, C, H, W)

        excitatory_input = self.exc_encoding(x)
        inhibitory_input = self.inh_encoding(x)
        return excitatory_input, inhibitory_input


__all__ = ["CNNInputTransform"] 