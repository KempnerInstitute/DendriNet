"""PointNeuronNetwork — minimal fc7 replacement for point-neuron baseline.

Single Linear layer followed by ReLU. Directly comparable to the noproj
EINet sweep: same input/output dims, same zero-padding via _ZeroPadCore,
same training protocol. The only difference from EINet is the absence of
dendritic branching, the E/I circuit, and sparse connectivity.

Any accuracy gap between this and EINet at the same N_E isolates the
contribution of dendritic morphology and the EI circuit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointNeuronNetwork(nn.Module):
    """Linear(input_dim → n_e) → ReLU point-neuron core.

    Args:
        input_dim: Dimensionality of the encoder output (4096 for AlexNet fc6).
        n_e: Number of point neurons (matches N_E in the EINet noproj sweep).
    """

    def __init__(self, input_dim: int, n_e: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_e)
        self.relu   = nn.ReLU()
        self.output_dim = n_e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))

    def get_output_dim(self) -> int:
        return self.output_dim

    # ------------------------------------------------------------------
    # Stubs so the training framework doesn't error when it calls these
    # on the core network (EINet exposes them; we provide no-ops).
    # ------------------------------------------------------------------
    def decay_weights(self, weight_decay: float, weight_boosting: bool = False) -> None:
        pass

    def apply_rewiring(self) -> None:
        pass
