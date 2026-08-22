"""SignedSparsePointNeuronNetwork — parameter-matched sparse point-neuron baseline.

This is PointNeuronNetwork with only the K largest-magnitude weights per output
neuron active. Same nn.Linear, same initialization, same SIGNED weights, same
bias, same ReLU, same param group. The number of active weights per neuron is the
only difference from the dense point-neuron baseline, so the comparison between
them isolates sparsity and nothing else.

Deliberately a standalone copy of PointNeuronNetwork rather than a flag on it:
point_neuron.py is what every completed dense baseline runs on, and it must not
be modified.

At K=315 this matches the 2x2 EINet's full synaptic budget (EE + EI + IE),
confirmed exact against ExcitationInhibitionNetwork.get_effective_params():

    EE:  16384 tips x 40 + 8192 layer1 x 40       =   983,040
    EI:  16384 tips x 10 + 8192 layer1 x 10       =   245,760
    IE:  (1024 inh: 4096 tips + 2048 layer1) x 10 =    61,440
                                            total = 1,290,240
    1,290,240 / 4096 excitatory neurons           =       315

Unlike SparsePointNeuronNetwork (TopKLinear + softplus), weights here are signed
and there is a bias — matching the dense point neuron. A positive-only single
pathway cannot express inhibition at all, whereas the EINet can via
`voltage = +branch_excitation - branch_inhibition` (branch_layer.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignedSparsePointNeuronNetwork(nn.Module):
    """Linear(input_dim → n_e) → ReLU with only K active weights per neuron.

    Args:
        input_dim: Dimensionality of the encoder output (4096 for AlexNet fc6).
        n_e: Number of point neurons (matches N_E in the EINet noproj sweep).
        k: Active weights per output neuron. 315 matches the 2x2 EINet at N_E=4096.
    """

    def __init__(self, input_dim: int, n_e: int, k: int):
        super().__init__()
        if not 1 <= k <= input_dim:
            raise ValueError(f"k must be in [1, {input_dim}], got {k}")
        self.linear = nn.Linear(input_dim, n_e)
        self.relu   = nn.ReLU()
        self.output_dim = n_e
        self.k = k

    def weight_mask(self) -> torch.Tensor:
        """0/1 mask keeping the k largest-|w| weights per output neuron."""
        w = self.linear.weight
        idx = torch.topk(w.abs(), self.k, dim=-1, largest=True, sorted=False)[1]
        return torch.zeros_like(w).scatter_(1, idx, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mask() * self.linear.weight
        return self.relu(F.linear(x, weight, self.linear.bias))

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
