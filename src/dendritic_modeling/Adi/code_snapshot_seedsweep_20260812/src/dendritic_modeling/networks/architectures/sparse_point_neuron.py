"""Sparse point-neuron baseline: TopK-sparse Linear(input_dim → n_e) → ReLU.

Directly comparable to EINet noproj experiments. Same input/output dims,
same fc8 decoder, same training protocol. The only difference from EINet is
the absence of dendritic branching and the E/I circuit — the feedforward
excitatory drive per output unit is matched via K.

For 2x2 EINet (branch_factors=[2,2], ee=40): each output neuron has 6 branch
stages (4 tips + 2 intermediate), each receiving 40 ee synapses, so K=240
matches the EE feedforward connections per output unit exactly.

    EINet 2x2 EE per output: 6 branches × 40 ee = 240
    SparsePointNeuron K=240: exactly 240 active inputs per output
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    TopKLinear,
)


class SparsePointNeuronNetwork(nn.Module):
    """TopKLinear(input_dim → n_e, K) → ReLU sparse point-neuron core.

    Args:
        input_dim: Dimensionality of the encoder output (4096 for AlexNet fc6).
        n_e: Number of output neurons (matches N_E in the EINet sweep).
        k: Active connections per output neuron. Set to 240 to match 2x2 EINet
           (6 branch stages × 40 ee synapses each).
    """

    def __init__(self, input_dim: int, n_e: int, k: int):
        super().__init__()
        self.sparse_linear = TopKLinear(
            in_features=input_dim,
            out_features=n_e,
            K=k,
            weight_transform="softplus",
            init_method="xavier_normal",
        )
        # TopKLinear.__init__ leaves pre_w as torch.empty(); initialize() is what
        # applies init_method. The EINet path does this in DendriticBranchLayer
        # (branch_layer.py) — without it, pre_w is all zeros at production size and
        # every output neuron selects the same K inputs.
        self.sparse_linear.initialize()
        self.relu = nn.ReLU()
        self.output_dim = n_e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.sparse_linear(x))

    def get_output_dim(self) -> int:
        return self.output_dim

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False) -> None:
        self.sparse_linear.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self) -> None:
        pass
