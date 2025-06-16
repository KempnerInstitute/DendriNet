"""
Sigmoid-based Input Transform for neural networks.

This module implements SigmoidInputNetTransform which applies
a sigmoid activation to ensure values are in [0, 1] range.
"""

import torch

from dendritic_modeling.networks.input_transforms.base import InputNetTransform


class SigmoidInputNetTransform(InputNetTransform):
    """
    Extension of InputNetTransform that applies a sigmoid activation to the output,
    ensuring values are in the range [0, 1] before entering the dendrinet.
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        excitatory_dim,
        inhibitory_dim,
        network_type="MLPInputTransform",
    ):
        super().__init__(
            input_dim, hidden_dims, excitatory_dim, inhibitory_dim, network_type
        )

    def forward(self, x):
        # Call the parent's forward method to get outputs
        excit_out, inhib_out = InputNetTransform.forward(self, x)

        # Apply sigmoid activation to ensure outputs are in [0, 1]
        excit_out = torch.sigmoid(excit_out)
        inhib_out = torch.sigmoid(inhib_out)

        return excit_out, inhib_out


__all__ = ["SigmoidInputNetTransform"] 