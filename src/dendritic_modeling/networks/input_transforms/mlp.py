"""
MLP-based Input Transform for neural networks.

This module implements MLPInputTransform which uses multilayer perceptrons
to transform inputs into excitatory and inhibitory pathways.
"""

import torch
import torch.nn as nn
from typing import List


class MLPInputTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input).
    This is a generalized version that can support different network architectures.
    By default, it uses an MLP with configurable hidden dimensions.
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        independent_heads: bool = False,
        excitatory_dim: int = None, 
        inhibitory_dim: int = None,
        output_dim: int = None
    ):
        super().__init__()

        if independent_heads:
            # Build MLP backbone
            exc_layers = []
            inh_layers = []
            in_features = input_dim
            for hdim in hidden_dims:
                exc_layer = nn.Linear(in_features, hdim)
                nn.init.kaiming_normal_(exc_layer.weight)
                nn.init.zeros_(exc_layer.bias)
                
                exc_layers.append(exc_layer)
                exc_layers.append(nn.ReLU())

                inh_layer = nn.Linear(in_features, hdim)
                nn.init.kaiming_normal_(inh_layer.weight)
                nn.init.zeros_(inh_layer.bias)
                inh_layers.append(inh_layer)
                inh_layers.append(nn.ReLU())

                in_features = hdim
            
            exc_head = nn.Linear(in_features, excitatory_dim)
            nn.init.xavier_normal_(exc_head.weight)
            nn.init.zeros_(exc_head.bias)
            exc_layers.append(exc_head)
            exc_layers.append(nn.Sigmoid())
            
            inh_head = nn.Linear(in_features, inhibitory_dim)
            nn.init.xavier_normal_(inh_head.weight)
            nn.init.zeros_(inh_head.bias)
            inh_layers.append(inh_head)
            inh_layers.append(nn.Sigmoid())

            self.exc_encoding = nn.Sequential(*exc_layers)
            self.inh_encoding = nn.Sequential(*inh_layers)

            self.excitatory_dim = excitatory_dim
            self.inhibitory_dim = inhibitory_dim
        
        else:
            encoding_layers = []
            in_features = input_dim
            for hdim in hidden_dims:
                layer = nn.Linear(in_features, hdim)
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                encoding_layers.append(layer)
                encoding_layers.append(nn.ReLU())
                in_features = hdim
            
            head = nn.Linear(in_features, output_dim)
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)
            encoding_layers.append(head)
            encoding_layers.append(nn.Sigmoid())

            self.encoding = nn.Sequential(*encoding_layers)

            self.excitatory_dim = output_dim
            self.inhibitory_dim = output_dim
        
        self.independent_heads = independent_heads

    def forward(self, x):
        if self.independent_heads:
            excitatory_input = self.exc_encoding(x)
            inhibitory_input = self.inh_encoding(x)
        else:
            x = self.encoding(x)
            excitatory_input = x
            inhibitory_input = x
        return excitatory_input, inhibitory_input


__all__ = ["MLPInputTransform"] 