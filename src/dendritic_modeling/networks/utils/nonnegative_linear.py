"""Positive afferent projection with an unconstrained intrinsic offset."""

import torch
from torch import nn
from torch.nn import functional as F

from .input_domain import require_nonnegative_input
from .weight_transforms import inverse_softplus


class NonnegativeLinear(nn.Module):
    """``y = softplus(raw_weight) @ x + bias`` for nonnegative input rates.

    Effective weights start uniformly in ``(0, 1 / in_features)``; the intrinsic
    offset starts at zero. A downstream rectifier produces output rates.
    This is a new model option, not a conversion of signed trained weights.
    """

    weight_transform = "softplus"

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("projection dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        initial = torch.empty(out_features, in_features).uniform_(0, 1 / in_features)
        self.raw_weight = nn.Parameter(inverse_softplus(initial))
        self.bias = nn.Parameter(torch.zeros(out_features))

    @property
    def weight(self):
        return F.softplus(self.raw_weight)

    def forward(self, x):
        return F.linear(require_nonnegative_input(x), self.weight, self.bias)
