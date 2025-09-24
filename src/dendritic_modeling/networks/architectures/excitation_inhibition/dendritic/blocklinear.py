"""
Block Linear Layer for Dendritic Networks.

This module implements a custom linear layer that aggregates information
from converging branches in a previous branch layer.
"""

from math import floor

import torch
import torch.nn as nn

from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    apply_weight_transform,
    inverse_weight_transform,
)


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

    def __init__(
        self, in_features, out_features, weight_transform: WeightTransformType = "exp"
    ):
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
        self.weight_transform = weight_transform
        self.input_scale_vec = None  # may be used for gradient scaling
        self.param_scale_vec = None  # may be used for gradient scaling

    def initialize(self, g_branch: float = 1.0):
        """
        Initialize the block weights to a specific conductance value.

        Args:
            g_branch: The desired conductance value (positive) after weight transformation.
                     The method handles conversion to the appropriate raw parameter space
                     based on self.weight_transform.
        """
        # Convert the desired conductance to the appropriate raw parameter value
        raw_value = inverse_weight_transform(
            torch.tensor(
                g_branch, dtype=self.log_weight.dtype, device=self.log_weight.device
            ),
            self.weight_transform,
        ).item()

        # Initialize the raw parameters
        nn.init.constant_(self.log_weight, raw_value)

    def weight(self):
        """Return transformed per-block weights with shape (out_features, block_size)."""
        return apply_weight_transform(self.log_weight, self.weight_transform)

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
        block[row_ix, col_ix] = self.weight()
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
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")

        original_shape = x.shape

        # Handle different input dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            # Handle higher dimensional inputs by flattening batch dimensions
            # This is common during analysis when data has extra batch dimensions
            x = x.reshape(-1, x.shape[-1])

        # x should now be 2D: [batch_size, features]
        if x.dim() != 2:
            raise ValueError(
                f"BlockLinear expects 2D input tensor after reshaping, got shape {x.shape} with {x.dim()} dimensions"
            )

        # Validate input dimension
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: got {x.shape[1]}, expected {self.in_features}"
            )

        # Perform the forward computation
        output = torch.matmul(x, self.block().t())

        # Reshape output to match input shape structure if needed
        if len(original_shape) > 2:
            # Restore the original batch dimensions
            output_shape = [*original_shape[:-1], self.out_features]
            output = output.reshape(output_shape)

        return output

    def sum_conductances(self):
        return self.weight().sum(dim=1)


# ---------------------------------------------------------------------------
# Memory-efficient implementation that avoids constructing the block-diagonal
# matrix. It is API-compatible with BlockLinear but allocates only the log_weight
# parameter tensor and no large temporaries.
# ---------------------------------------------------------------------------


class EfficientBlockLinear(BlockLinear):
    """BlockLinear variant with *O(out_features x block_size)* memory.

    Forward pass reshapes the input into (B, out, block_size) and performs a
    weighted sum along the last dimension, eliminating the need to materialise
    the big block-diagonal matrix created in :py:meth:`BlockLinear.block`.
    """

    def forward(self, x):  # type: ignore[override]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(x)}")

        original_shape = x.shape

        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            # Handle higher dimensional inputs by flattening batch dimensions
            # This is common during analysis when data has extra batch dimensions
            feature_dim = x.shape[-1]
            x = x.reshape(-1, feature_dim)

        # x should now be 2D: [B, features]
        B = x.shape[0]
        expected_dim = self.out_features * self.block_size

        # Validate input dimension
        if x.shape[1] != expected_dim:
            # Check if this might be a transposed or mishaped tensor
            if x.shape[1] % self.block_size == 0:
                # Could be wrong number of output features
                implied_out = x.shape[1] // self.block_size
                print(
                    f"Warning: EfficientBlockLinear expecting {expected_dim} features (out={self.out_features}, block={self.block_size}), "
                    f"but got {x.shape[1]} features. This implies out_features should be {implied_out}."
                )

            # Return zeros to allow analysis to continue
            print(
                f"Warning: EfficientBlockLinear input shape mismatch: original shape {original_shape}, "
                f"reshaped to {x.shape}, expected [B, {expected_dim}]. Returning zeros."
            )

            # If original was higher-dim, return zeros with appropriate shape
            if len(original_shape) > 2:
                out_shape = [*original_shape[:-1], self.out_features]
                return torch.zeros(out_shape, device=x.device, dtype=x.dtype)
            else:
                return torch.zeros(B, self.out_features, device=x.device, dtype=x.dtype)

        x_reshaped = x.view(B, self.out_features, self.block_size)
        weights = apply_weight_transform(
            self.log_weight, self.weight_transform
        )  # (out_features, block_size)
        output = (x_reshaped * weights).sum(dim=-1)  # (B, out_features)

        # Reshape output to match input shape structure if needed
        if len(original_shape) > 2:
            # Restore the original batch dimensions
            output_shape = [*original_shape[:-1], self.out_features]
            output = output.reshape(output_shape)

        return output


__all__ = ["BlockLinear", "EfficientBlockLinear"]
