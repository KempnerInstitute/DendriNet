"""
Baseline MLP models.

This module contains simple MLP implementations for baseline comparisons.
"""
from typing import List, Optional
from copy import deepcopy

import torch
from torch import nn

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.activations.factory import ActivationFactory


class MLP(BaseNetwork):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int],
        activation: str = "relu",
        output_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        activation_factory = ActivationFactory()

        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)

        layers = []

        # Create hidden layers
        for i in range(1, len(hidden_dims)):
            fc_linear = torch.nn.Linear(hidden_dims[i-1], hidden_dims[i])
            if activation == "relu":
                nn.init.kaiming_normal_(fc_linear.weight)
                nn.init.zeros_(fc_linear.bias)
            else:
                nn.init.xavier_normal_(fc_linear.weight)
                nn.init.zeros_(fc_linear.bias)
            layers.append(fc_linear)
            
            layers.append(
                activation_factory.create(
                    act_type=activation,
                    output_dim= hidden_dims[i]
                )
            )

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            layers.append(nn.Linear(self.output_dim, output_dim))
            self.output_dim = output_dim

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.layers(x)


__all__ = ["MLP"] 