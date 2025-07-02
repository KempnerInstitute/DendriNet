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


# change nomenclature to encoder and decoder

class CNNDownsample(BaseNetwork):

    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (channels, height, width)
        hidden_dims: List[int],  # List of channel sizes for each conv layer
        kernel_sizes: List[int],  # Kernel size for all conv layers
        strides: List[int],  # Stride for all conv layers
        activation: str = "relu",
        output_dim: Optional[int] = None,
        **kwargs
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

        self.output_dim = (hidden_dims[-1], current_h, current_w)

        if output_dim is not None:
            layers.append(nn.Flatten(start_dim=-3))
            final_hidden_flat = 1
            for dim in self.output_dim:
                final_hidden_flat *= dim
            layers.append(nn.Linear(final_hidden_flat, output_dim))
            self.output_dim = (output_dim,)
       
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct shape (add batch dimension if needed)
        if x.dim() == 3:  # (C, H, W)
            x = x[None, ...]  # Add batch dimension: (1, C, H, W)
        
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        
        output: torch.Tensor = self.layers(x)
        return output.reshape(*batch_shape, *self.output_dim)



class CNNUpsample(BaseNetwork):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W) - starting shape
        target_shape: Tuple[int, int, int],  # (C, H, W) - target output shape
        hidden_dims: List[int],
        activation: str = "relu"
    ):
        super().__init__()
        
        input_c, input_h, input_w = input_shape
        target_c, target_h, target_w = target_shape
        
        if input_h != input_w or target_h != target_w:
            raise ValueError("Currently only supports square spatial dimensions")
        
        if input_h > target_h:
            raise ValueError("Input spatial size must be <= target spatial size")
        
        # Auto-calculate transpose conv parameters from input_shape to target_shape
        kernels, strides, paddings = self._calculate_transpose_params(
            input_h, target_h, len(hidden_dims) + 1
        )
        
        # Build transpose convolution layers
        layers = []
        in_channels = input_c  # Start with input channels
        activation_factory = ActivationFactory()
        
        for i, out_channels in enumerate(hidden_dims + [target_c]):
            conv_transpose = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernels[i],
                stride=strides[i],
                padding=paddings[i]
            )
            
            # Initialize weights
            if activation == "relu":
                nn.init.kaiming_normal_(conv_transpose.weight)
                nn.init.zeros_(conv_transpose.bias)
            else:
                nn.init.xavier_normal_(conv_transpose.weight)
                nn.init.zeros_(conv_transpose.bias)
                
            layers.append(conv_transpose)
            
            # Add activation (except for final layer)
            if i < len(hidden_dims):
                layers.append(activation_factory.create(activation, out_channels))
            
            in_channels = out_channels
            
        self.layers = nn.Sequential(*layers)
        self.output_dim = (target_c, target_h, target_w)
        
    def _calculate_transpose_params(
        self, 
        input_size: int, 
        target_size: int, 
        num_layers: int
    ) -> Tuple[List[int], List[int], List[int]]:
        """Calculate transpose convolution parameters from input_size to target_size."""
        
        if input_size == target_size:
            # No upsampling needed, just use 1x1 convs
            kernels = [1] * num_layers
            strides = [1] * num_layers
            paddings = [0] * num_layers
            return kernels, strides, paddings
        
        # Work backwards from target to input to ensure exact sizing
        kernels = []
        strides = []
        paddings = []
        
        # Calculate intermediate sizes working backwards
        sizes = [target_size]
        current_size = target_size
        
        # Work backwards to find intermediate sizes
        for i in range(num_layers):
            if i == num_layers - 1:
                current_size = input_size
            else:
                # Try to divide by 2 (most common upsampling factor)
                if current_size % 2 == 0 and current_size // 2 >= input_size:
                    current_size = current_size // 2
                else:
                    # If can't divide by 2, find the best divisor
                    best_divisor = 2
                    for divisor in [3, 5, 7, 11, 13, 17, 19]:
                        if current_size >= divisor and current_size // divisor >= input_size:
                            best_divisor = divisor
                            break
                    current_size = max(input_size, current_size // best_divisor)
            sizes.append(current_size)
        
        # Ensure we end at input_size
        sizes.reverse()  # Now goes from input_size to target_size
        
        # Calculate parameters for each transition
        for i in range(num_layers):
            input_s = sizes[i]
            output_s = sizes[i + 1]
            
            stride = output_s // input_s
            
            # Choose kernel based on stride
            if stride == 1:
                kernel = 1
                padding = 0
            elif stride == 2:
                kernel = 4
                padding = 1
            elif stride <= 4:
                kernel = stride * 2
                padding = kernel // 2 - 1
            else:
                kernel = stride + 2
                padding = 1
            
            # Verify the math works out
            # output_size = (input_size - 1) * stride - 2 * padding + kernel
            expected_output = (input_s - 1) * stride - 2 * padding + kernel
            
            # Adjust padding if needed to hit exact target
            if expected_output != output_s:
                padding = max(0, ((input_s - 1) * stride + kernel - output_s) // 2)
            
            kernels.append(kernel)
            strides.append(stride)
            paddings.append(padding)
        
        return kernels, strides, paddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[None, ...]
        
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        
        output: torch.Tensor = self.layers(x)
        return output.reshape(*batch_shape, *self.output_dim)


__all__ = ["CNNDownsample", "CNNUpsample"] 