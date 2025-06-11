"""
MLP Excitation-Inhibition Network.

This module implements MLPExcInhNetwork and MLPExcInhLayer for simpler 
excitation-inhibition architectures using standard MLP layers.
"""

import torch
import torch.nn as nn
from typing import List


class MLPExcInhLayer(nn.Module):
    def __init__(self, in_excit, in_inhib, out_features, activation=None):
        super().__init__()
        self.in_excit = int(in_excit)
        self.in_inhib = int(in_inhib)
        self.out_features = int(out_features)
        if self.in_excit > 0:
            self.excit_pre_w = nn.Parameter(
                torch.zeros(self.out_features, self.in_excit)
            )
            nn.init.xavier_normal_(self.excit_pre_w)
        else:
            self.excit_pre_w = None
        if self.in_inhib > 0:
            self.inhib_pre_w = nn.Parameter(
                torch.zeros(self.out_features, self.in_inhib)
            )
            nn.init.xavier_normal_(self.inhib_pre_w)
        else:
            self.inhib_pre_w = None
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.activation = activation

    def decay_weights(self, weight_decay):
        with torch.no_grad():
            if self.excit_pre_w is not None:
                self.excit_pre_w.sub_(weight_decay * self.excit_pre_w)
            if self.inhib_pre_w is not None:
                self.inhib_pre_w.sub_(weight_decay * self.inhib_pre_w)
            self.bias.sub_(weight_decay * self.bias)

    def forward(self, x):
        batch_size = x.size(0)
        if self.in_excit > 0 and self.in_inhib > 0:
            x_excit = x[:, : self.in_excit]
            x_inhib = x[:, self.in_excit : self.in_excit + self.in_inhib]
        elif self.in_excit > 0:
            x_excit = x
            x_inhib = None
        else:
            x_excit = None
            x_inhib = x
        out = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        if self.excit_pre_w is not None and x_excit is not None:
            w_excit = self.excit_pre_w.exp()
            out += x_excit.matmul(w_excit.t())
        if self.inhib_pre_w is not None and x_inhib is not None:
            w_inhib = -self.inhib_pre_w.exp()
            out += x_inhib.matmul(w_inhib.t())
        out += self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


class MLPExcInhNetwork(nn.Module):
    """
    A simple MLP network with separate excitatory and inhibitory pathways.
    This is a placeholder implementation.
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int], output_dim: int = 10, **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Create the network
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            in_excit = layer.in_excit
            in_inhib = layer.in_inhib
            if x.size(1) < in_excit + in_inhib:
                pad_size = (in_excit + in_inhib) - x.size(1)
                x = torch.cat(
                    [x, torch.zeros(x.size(0), pad_size, device=x.device)], dim=1
                )
            x = layer(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x


__all__ = ["MLPExcInhLayer", "MLPExcInhNetwork"] 