"""
Simple MLP Layer.

This module implements a simple MLP layer with configurable activation functions.
"""

import torch
import torch.nn as nn
from typing import Optional


class SimpleMLPLayer(nn.Module):
    """
    A simple MLP layer with configurable activation and optional batch normalization.
    """

    def __init__(
        self,
        in_features,
        out_features,
        activation_type=None,
        trainable_activation=True,
        activation_wrapper=None,
        positive_weights=False,
        init_m=4.0,
        use_batch_norm=False,
        weight_init="kaiming_uniform",
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.positive_weights = positive_weights
        self.use_batch_norm = use_batch_norm
        
        # Linear layer
        self.linear = nn.Linear(in_features, out_features)
        
        # Weight initialization
        if weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.linear.weight)
        elif weight_init == "kaiming_normal":
            nn.init.kaiming_normal_(self.linear.weight)
        elif weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.linear.weight)
        elif weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.linear.weight)
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")
            
        nn.init.zeros_(self.linear.bias)
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_features)
        
        # Activation
        self.activation = None
        if activation_type:
            if activation_type == "relu":
                self.activation = nn.ReLU()
            elif activation_type == "sigmoid":
                self.activation = nn.Sigmoid()
            elif activation_type == "tanh":
                self.activation = nn.Tanh()
            elif activation_type == "param_linear_sigmoid":
                # Custom parameterized activation
                self.activation = nn.Sequential(
                    nn.Linear(out_features, out_features),
                    nn.Sigmoid()
                )
                if trainable_activation:
                    nn.init.constant_(self.activation[0].weight, init_m)
                    nn.init.zeros_(self.activation[0].bias)
            else:
                raise ValueError(f"Unknown activation type: {activation_type}")
                
        # Apply positive weight constraint if needed
        if positive_weights:
            with torch.no_grad():
                self.linear.weight.abs_()

    def decay_weights(self, weight_decay):
        """Apply weight decay to the linear layer weights."""
        if weight_decay > 0:
            with torch.no_grad():
                self.linear.weight.data *= (1 - weight_decay)

    def forward(self, x):
        """Forward pass through the layer."""
        x = self.linear(x)
        
        if self.use_batch_norm:
            x = self.batch_norm(x)
            
        if self.activation is not None:
            x = self.activation(x)
            
        # Apply positive weight constraint if needed
        if self.positive_weights:
            with torch.no_grad():
                self.linear.weight.abs_()
                
        return x


__all__ = ["SimpleMLPLayer"] 