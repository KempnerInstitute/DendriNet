"""
Baseline MLP models.

This module contains simple MLP implementations for baseline comparisons.
"""
from typing import List, Optional

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
        output_dim: Optional[int] = None
    ):
        super().__init__()

        activation_factory = ActivationFactory()

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for h_dim in hidden_dims:
            fc_linear = torch.nn.Linear(prev_dim, h_dim)
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
                    output_dim= h_dim
                )
            )
            prev_dim = h_dim

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            layers.append(nn.Linear(self.output_dim, output_dim))
            self.output_dim = output_dim

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        return self.layers(x)


__all__ = ["MLP"] 