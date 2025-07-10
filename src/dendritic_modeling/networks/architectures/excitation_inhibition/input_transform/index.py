"""
Index-based Input Transform for neural networks.

This module implements IndexInputTransform which randomly splits
input dimensions into excitatory and inhibitory channels.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class IndexInputTransform(nn.Module):
    def __init__(
        self,
        input_dim: int,
        excitatory_dim: Optional[int] = None,
        inhibitory_dim: Optional[int] = None
    ):
        super().__init__()

        try:
            assert excitatory_dim is not None and inhibitory_dim is not None, (
                "error in IndexInputTransform: "
                "either excitatory_dim or inhibitory_dim must be provided"
            )
        except AssertionError as e:
            logger.error(e)
            excitatory_dim = int(input_dim / 2)
            inhibitory_dim = int(input_dim - excitatory_dim)

        if excitatory_dim is None:
            excitatory_dim = int(input_dim - inhibitory_dim)
        if inhibitory_dim is None:
            inhibitory_dim = int(input_dim - excitatory_dim)

        try:
            assert excitatory_dim + inhibitory_dim == input_dim, (
                "error in IndexInputTransform: "
                "excitatory_dim + inhibitory_dim must equal input_dim, "
                f"got {excitatory_dim} + {inhibitory_dim} != {input_dim}"
            )
        except AssertionError as e:
            logger.error(e)
            inhibitory_dim = int(input_dim - excitatory_dim)

        dims = torch.randperm(input_dim)
        exc_dims = dims[:excitatory_dim]
        inh_dims = dims[excitatory_dim:]

        self.exc_dims: torch.Tensor
        self.register_buffer("exc_dims", exc_dims)
        self.inh_dims: torch.Tensor
        self.register_buffer("inh_dims", inh_dims)

        self.excitatory_dim = excitatory_dim
        self.inhibitory_dim = inhibitory_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x[None, :]

        excitatory_input = x[..., self.exc_dims]
        inhibitory_input = x[..., self.inh_dims]
        return excitatory_input, inhibitory_input

__all__ = ["IndexInputTransform"]
