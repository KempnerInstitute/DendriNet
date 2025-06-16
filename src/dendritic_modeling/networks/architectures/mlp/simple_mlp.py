"""
Simple MLP Network.

This module implements SimpleMLPNet, a modular MLP implementation that
uses SimpleMLPLayer and supports all activation functions from ActivationFactory.
"""

import torch
import torch.nn as nn

from dendritic_modeling.networks.mlp.simple_layer import SimpleMLPLayer


class SimpleMLPNet(nn.Module):
    """
    A modular MLP implementation that uses SimpleMLPLayer.
    Supports all activation functions from ActivationFactory.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nb_layers=2,  # default to 2-layer network
        activation_type="param_linear_sigmoid",
        weight_decay_rate=None,
        learning_strategy="regression",
        trainable_activation=True,
        grad_clip_value=1.0,
        activation_wrapper=None,
        init_m=4.0,
        seed=None,
        use_batch_norm=False,  # Batch normalization parameter
        weight_init="kaiming_uniform",  # Updated default value with more options
        **kwargs,
    ):
        super(SimpleMLPNet, self).__init__()
        self.weight_decay_rate = weight_decay_rate
        self.learning_strategy = learning_strategy
        self.input_dim = int(input_dim)
        self.grad_clip_value = grad_clip_value
        self.init_m = init_m
        self.use_batch_norm = use_batch_norm
        self.weight_init = weight_init

        if seed is not None:
            self.set_seed(seed)

        # Build multi-layer network
        layers = []

        # First layer: input_dim -> hidden_dim
        layers.append(
            SimpleMLPLayer(
                in_features=input_dim,
                out_features=hidden_dim,
                activation_type=activation_type,
                trainable_activation=trainable_activation,
                activation_wrapper=activation_wrapper,
                init_m=self.init_m,
                use_batch_norm=use_batch_norm,
                weight_init=weight_init,
            )
        )

        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(nb_layers - 2):
            layers.append(
                SimpleMLPLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    activation_type=activation_type,
                    trainable_activation=trainable_activation,
                    activation_wrapper=activation_wrapper,
                    init_m=self.init_m,
                    use_batch_norm=use_batch_norm,
                    weight_init=weight_init,
                )
            )

        # Output layer: hidden_dim -> output_dim (no activation)
        layers.append(
            SimpleMLPLayer(
                in_features=hidden_dim,
                out_features=output_dim,
                activation_type=None,
                trainable_activation=trainable_activation,
                activation_wrapper=activation_wrapper,
                init_m=self.init_m,
                use_batch_norm=False,  # No batch norm on output layer
                weight_init=weight_init,
            )
        )

        self.layers = nn.ModuleList(layers)

    def decay_weights(self, weight_decay=None):
        """Apply weight decay to all layers."""
        if weight_decay is not None:
            weight_decay = self.weight_decay_rate
            for layer in self.layers:
                layer.decay_weights(weight_decay)

    def clip_gradients(self):
        """Apply gradient clipping to all parameters."""
        if self.grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.grad_clip_value
            )

    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x

    def set_seed(self, seed):
        """Set the seed for the network."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def branch_layers(self):
        """Compatibility with existing codebase expectations."""
        return []

    @property
    def n_branch_layers(self):
        """Compatibility with existing codebase expectations."""
        return 0


__all__ = ["SimpleMLPNet"] 