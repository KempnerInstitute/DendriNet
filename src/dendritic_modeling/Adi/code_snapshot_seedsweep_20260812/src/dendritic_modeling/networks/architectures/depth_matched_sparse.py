"""DepthMatchedSparseNetwork — depth control for the sparse point-neuron baseline.

SparsePointNeuronNetwork is a single TopKLinear + activation, while the EINet is a
*sequence* of nonlinear branch stages. Comparing them therefore conflates two
things: sparse structured connectivity, and depth. This core isolates the first
by giving the point-neuron baseline the same number of stages and the same stage
widths as the EINet tree, with the same total active weight budget.

Stage widths are derived from the morphology exactly as DendriNet does, so a
2x2 morphology at N_E=4096 gives:

    4096 (input) -> 16384 -> 8192 -> 4096      3 TopKLinear + reactivation stages

with K active inputs per unit at every stage. Matching the 2x2 EINet's full
synaptic budget (EE + EI + IE = 1,290,240) gives K exactly:

    1,290,240 / (16384 + 8192 + 4096) = 45

Why param_tanh and not ReLU
---------------------------
`weight_transform="softplus"` makes every effective weight strictly positive, and
the encoder output is non-negative, so a TopKLinear stage followed by ReLU can only
amplify — there is no possible cancellation. Measured at init on real fc6 input, one
stage takes the mean from 0.61 to 19.3 (~32x); stacking three compounds it and
training goes non-finite in the first epoch.

The EINet is also three stages deep and does not do this because every branch layer
ends in a *bounded* reactivation. We use the identical mechanism, built through the
same ActivationFactory with the same parameters, so each stage output lies in (0, 1)
exactly as the EINet's branch and soma outputs do. Its learnable parameters are
ParametricActivation instances, so they land in the `reactivation_lr` param group
alongside the EINet's (models/base.py).

Note this is a depth-and-width control, NOT a reimplementation of the tree. The
EINet is not a plain MLP: DendriNet.forward feeds the same input x to *every*
branch layer, with BlockLinear carrying child->parent aggregation, plus inhibition.
Here each stage consumes only the previous stage's output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    TopKLinear,
)


class DepthMatchedSparseNetwork(nn.Module):
    """Stacked TopKLinear + reactivation stages matching an EINet morphology's depth.

    Args:
        input_dim: Dimensionality of the encoder output (4096 for AlexNet fc6).
        n_e: Number of output neurons (matches N_E in the EINet sweep).
        branch_factors: Morphology branch factors, e.g. [2, 2] for 2x2. Stage
            widths are derived from these the same way DendriNet derives its
            compartment counts.
        k: Active connections per unit, applied at every stage.
        reactivation_type: Activation name passed to ActivationFactory. Matches
            the EINet's `core.reactivation.type`.
        reactivation_init_m: Initial slope, matching `core.reactivation.init_m`.
        reactivation_init_b: Initial midpoint, matching `core.reactivation.init_b`.
    """

    def __init__(
        self,
        input_dim: int,
        n_e: int,
        branch_factors: list[int],
        k: int,
        reactivation_type: str = "param_tanh",
        reactivation_init_m: float = 1.5,
        reactivation_init_b: float = 0.5,
    ):
        super().__init__()

        # Same derivation as DendriNet: widen by each branch factor from the
        # soma outward, then reverse so the widest (tip) stage comes first.
        layer_sizes = [n_e]
        for factor in branch_factors:
            layer_sizes.append(layer_sizes[-1] * factor)
        layer_sizes.reverse()  # e.g. [16384, 8192, 4096] for 2x2 at N_E=4096

        dims = [input_dim] + layer_sizes
        stages = []
        reactivations = []
        for i in range(len(layer_sizes)):
            stage = TopKLinear(
                in_features=dims[i],
                out_features=dims[i + 1],
                K=k,
                weight_transform="softplus",
                init_method="xavier_normal",
            )
            # TopKLinear.__init__ leaves pre_w as torch.empty(); initialize() is
            # what applies init_method. See sparse_point_neuron.py.
            stage.initialize()
            stages.append(stage)

            # Same construction as DendriticBranchLayer (branch_layer.py).
            reactivations.append(
                ActivationFactory.create(
                    act_type=reactivation_type,
                    output_dim=dims[i + 1],
                    init_m=reactivation_init_m,
                    init_b=reactivation_init_b,
                )
            )

        self.stages = nn.ModuleList(stages)
        self.reactivations = nn.ModuleList(reactivations)
        self.stage_dims = dims
        self.output_dim = n_e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage, reactivation in zip(self.stages, self.reactivations):
            x = reactivation(stage(x))
        return x

    def get_output_dim(self) -> int:
        return self.output_dim

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False) -> None:
        for stage in self.stages:
            stage.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self) -> None:
        pass
