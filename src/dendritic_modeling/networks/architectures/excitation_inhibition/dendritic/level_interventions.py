"""Labelled interventions on single ``DendriNet`` levels.

Interventions are passed to :meth:`DendriNet.forward_levels`; ``forward``
never applies them. They replace a level's output in the pass where they are
given and change no parameter or buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, eq=False)
class CurrentInjection:
    """Replace a shunting level's inhibitory conductance by injected current.

    The inhibition beyond a reference input acts as a current at a fixed
    voltage ``anchor`` instead of as a conductance:
    ``V = (E + C - (I - I_ref) * anchor) / (leak + E + G + I_ref)``, followed
    by the level's activation, where ``I`` and ``I_ref`` are the level's
    inhibitory conductances for the actual and the reference inhibitory input,
    ``C`` and ``G`` the child current and coupling conductance and ``E`` the
    excitatory conductance (absent at levels without that pathway). The
    level's ordinary forward still runs first, so its diagnostics describe the
    unperturbed conductance state. ``anchor`` has one value per compartment
    (for example a training-set mean voltage); it is the caller's to
    calibrate.
    """

    reference_inhibitory_input: torch.Tensor
    anchor: torch.Tensor

    def output(self, layer: Any, x, inhibitory_input, branch_input):
        """The level's output under the intervention."""
        if not layer.use_shunting:
            raise ValueError("CurrentInjection needs a shunting level")
        if layer.branch_inhibition is None or inhibitory_input is None:
            raise ValueError("CurrentInjection needs a level with inhibitory input")
        # Terms are summed in the order of the level's own forward pass.
        numerator = 0
        base = layer.effective_leak
        if layer.input_excitatory and layer.branch_excitation is not None:
            excitation = layer.branch_excitation(x)
            numerator = numerator + excitation
            base = base + excitation
        if layer.input_branches and branch_input is not None:
            numerator = numerator + layer.branches_to_output(branch_input)
            base = base + layer.branches_to_output.sum_conductances()
        inhibition = layer.branch_inhibition(inhibitory_input)
        reference = layer.branch_inhibition(self.reference_inhibitory_input)
        numerator = numerator - (inhibition - reference) * self.anchor
        denominator = base + reference
        if layer.epsilon:
            denominator = denominator + layer.epsilon
        return layer.reactivation(numerator / denominator)


__all__ = ["CurrentInjection"]
