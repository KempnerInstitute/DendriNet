"""Explicit input-domain contracts without changing values or derivatives."""

import torch
from torch import nn


def require_nonnegative_input(x: torch.Tensor) -> torch.Tensor:
    """Check each input batch and return it unchanged, including at zero.

    This is for measured or generated nonnegative rates. Unlike rectification,
    it rejects invalid data and preserves the identity derivative at zero.
    The explicit check synchronizes a CUDA input; use a rectifying adapter
    when the input domain is not guaranteed by the data source.
    """
    if not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
        raise ValueError("nonnegative input requires finite values >= 0")
    return x


class NonnegativeInputCheck(nn.Module):
    """An identity map with an enforced finite, nonnegative input domain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return require_nonnegative_input(x)
