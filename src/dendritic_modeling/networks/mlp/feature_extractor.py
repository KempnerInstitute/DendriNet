"""
Feature Extractor MLP.

This module implements a feature extractor based on Multi-Layer Perceptron.
"""

import torch.nn as nn
from typing import List


class FeatureExtractorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        layers = []
        in_features = input_dim
        for hdim in hidden_dims[:-1]:
            layer = nn.Linear(in_features, hdim)
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU())
            in_features = hdim
        
        encoding_head = nn.Linear(in_features, hidden_dims[-1])
        nn.init.xavier_normal_(encoding_head.weight)
        nn.init.zeros_(encoding_head.bias)
        layers.append(encoding_head)
        layers.append(nn.Sigmoid())

        self.encoding = nn.Sequential(*layers)

        self.excitatory_dim = hidden_dims[-1]
        self.inhibitory_dim = hidden_dims[-1]

        self.readout = True
        self.final = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.encoding(x)
        if self.readout:
            return self.final(x)
        else:
            excitatory_input = x
            inhibitory_input = x
            return excitatory_input, inhibitory_input
    
    def set_readout(self, readout: bool):
        self.readout = readout

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False


__all__ = ["FeatureExtractorMLP"] 