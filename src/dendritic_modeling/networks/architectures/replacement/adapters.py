"""Input adapters shared by replacement modules."""

from __future__ import annotations

import torch
import torch.nn as nn

_VALID_INPUT_TRANSFORMS = frozenset({"identity", "relu", "signed_split"})


def validate_input_transform(transform: str) -> str:
    """Return a normalized replacement input transform name."""
    normalized = str(transform)
    if normalized not in _VALID_INPUT_TRANSFORMS:
        raise ValueError("input_transform must be identity, relu, or signed_split")
    return normalized


def transformed_feature_dim(input_dim: int, transform: str) -> int:
    """Feature dimension after applying a replacement input adapter."""
    normalized = validate_input_transform(transform)
    multiplier = 2 if normalized == "signed_split" else 1
    return int(input_dim) * multiplier


class NonNegativeInputAdapter(nn.Module):
    """Map signed features into the non-negative domain expected by dendrites.

    ``identity`` is available for callers that already guarantee non-negative
    inputs.  ``relu`` keeps only positive activity.  ``signed_split`` preserves
    both polarities as ``[ReLU(x), ReLU(-x)]`` using non-negative channels.
    """

    def __init__(
        self,
        input_dim: int,
        transform: str = "signed_split",
        *,
        validate_shape: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.transform = validate_input_transform(transform)
        self.output_dim = transformed_feature_dim(self.input_dim, self.transform)
        self.validate_shape = bool(validate_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.validate_shape and x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, got "
                f"{x.shape[-1]}"
            )
        if self.transform == "identity":
            return x
        if self.transform == "relu":
            return torch.relu(x)
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"transform={self.transform!r}"
        )
