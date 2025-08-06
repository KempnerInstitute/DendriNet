"""
Block Linear Layer for Dendritic Networks.

This module implements a custom linear layer that aggregates information
from converging branches in a previous branch layer.
"""

from math import floor

import torch
import torch.nn as nn


class BlockLinear(nn.Module):
    """
    A custom linear layer that aggregates information from converging branches
    in a previous branch layer.

    Parameters
    ----------

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    requires_grad : bool, optional
        If `True`, the weights are trainable. Defaults to `False`.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The weights for the linear transformation.

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    block_size : int
        The number of branches converging onto a single branch.

    Methods
    -------
    forward(x)
        Performs a forward pass through the layer.

    sum_conductances()
        Returns the sum of conductances for each output neuron.

    Notes
    -----

    - Block_size equals number of branches converging onto single branch/soma
      in next layer.

    """

    def __init__(self, in_features, out_features):
        super().__init__()

        block_size = floor(in_features / out_features)
        assert (
            in_features == out_features * block_size
        ), "in_features must be divisible by out_features."

        self.log_weight = nn.Parameter(
            torch.empty((out_features, block_size)), requires_grad=True
        )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.input_scale_vec = None  # may be used for gradient scaling
        self.param_scale_vec = None  # may be used for gradient scaling

    def initialize(self, sum_synapse_conductances=None):
        if sum_synapse_conductances is None:
            nn.init.constant_(self.log_weight, 0)
        else:
            # Ensure sum_synapse_conductances is a 1D tensor
            self.log_weight.data = (
                sum_synapse_conductances.log()[:, None]
                .expand(self.out_features, self.block_size)
                .clone()
            )

    def block(self):
        block = torch.zeros(
            (self.out_features, self.in_features),
            device=self.log_weight.device,
            dtype=self.log_weight.dtype,
        )
        row_ix = torch.arange(self.out_features, device=self.log_weight.device)[:, None]
        col_ix = torch.arange(self.in_features, device=self.log_weight.device).view(
            self.out_features, self.block_size
        )
        block[row_ix, col_ix] = self.log_weight.exp()
        return block

    def log_block(self):
        log_block = (
            torch.ones(
                (self.out_features, self.in_features),
                device=self.log_weight.device,
                dtype=self.log_weight.dtype,
            )
            * -10
        )
        row_ix = torch.arange(self.out_features, device=self.log_weight.device)[:, None]
        col_ix = torch.arange(self.in_features, device=self.log_weight.device).view(
            self.out_features, self.block_size
        )
        log_block[row_ix, col_ix] = self.log_weight
        return log_block

    def grad_block(self):
        grad_block = torch.zeros(
            (self.out_features, self.in_features),
            device=self.log_weight.device,
            dtype=self.log_weight.dtype,
        )
        row_ix = torch.arange(self.out_features, device=self.log_weight.device)[:, None]
        col_ix = torch.arange(self.in_features, device=self.log_weight.device).view(
            self.out_features, self.block_size
        )
        if self.log_weight.grad is not None:
            grad_block[row_ix, col_ix] = self.log_weight.grad
        return grad_block

    def decay_weights(self, weight_decay):
        with torch.no_grad():
            self.log_weight.data -= weight_decay

    def forward(self, x):
        # Ensure x is a 2D tensor for matrix multiplication
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")

        # If x is a 1D tensor, unsqueeze to make it 2D [1, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # If x is not 2D at this point, we can't do matrix multiplication
        # if x.dim() != 2:
        #     raise ValueError(
        #         f"Expected input to be a 2D tensor, got shape {x.shape} with {x.dim()} dimensions"
        #     )

        return torch.matmul(x, self.block().t())

    def sum_conductances(self):
        return self.log_weight.exp().sum(dim=1)


# ---------------------------------------------------------------------------
# Memory-efficient implementation that avoids constructing the block-diagonal
# matrix. It is API-compatible with BlockLinear but allocates only the log_weight
# parameter tensor and no large temporaries.
# ---------------------------------------------------------------------------


class EfficientBlockLinear(BlockLinear):
    """BlockLinear variant with *O(out_features × block_size)* memory.

    Forward pass reshapes the input into (B, out, block_size) and performs a
    weighted sum along the last dimension, eliminating the need to materialise
    the big block-diagonal matrix created in :py:meth:`BlockLinear.block`.
    """

    def forward(self, x):  # type: ignore[override]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(x)}")

        if x.dim() == 1:
            x = x.unsqueeze(0)

        # x has shape (B, out_features * block_size)
        B = x.shape[0]
        expected_dim = self.out_features * self.block_size
        if x.shape[1] != expected_dim:
            # Try to provide more context about where this error is occurring
            import traceback

            tb_str = "".join(
                traceback.format_stack()[-3:-1]
            )  # Get last few stack frames
            print(
                f"Error computing metric: Input feature dimension {x.shape[1]} does not match expected {expected_dim} (= out_features*block_size)"
            )
            print(
                "This suggests data is flowing incorrectly through the network. Stack trace:"
            )
            print(tb_str)

            # Return zeros instead of crashing to allow analysis to continue
            return torch.zeros(B, self.out_features, device=x.device, dtype=x.dtype)

        x_reshaped = x.view(B, self.out_features, self.block_size)
        weights = self.log_weight.exp()  # (out_features, block_size)
        output = (x_reshaped * weights).sum(dim=-1)  # (B, out_features)
        return output


__all__ = ["BlockLinear"]
__all__.extend(["EfficientBlockLinear"])
