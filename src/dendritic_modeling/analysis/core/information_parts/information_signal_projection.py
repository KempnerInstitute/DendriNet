"""Shared projection helpers for information-analysis signal variants."""

from __future__ import annotations

import numpy as np
import torch


def compute_projected_branch_output(
    module,
    excitation: np.ndarray,
    inhibition: np.ndarray,
    branch_input: np.ndarray,
) -> np.ndarray:
    """Apply the layer's voltage rule and reactivation to projected signals.

    The projected excitatory, inhibitory and upstream currents replace the
    recorded ones in the layer's own rule (``output_from_currents``), so its
    leak, integration mode, the coupling conductance of its child branches,
    its signal rounding and its gate are those of the forward pass. The rule
    is evaluated in float64.
    """
    parameter = next(module.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")

    def current(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float64, device=device)

    child_conductance = None
    if getattr(module, "input_branches", False):
        child_conductance = module.branches_to_output.sum_conductances()
    with torch.no_grad():
        output = module.output_from_currents(
            current(excitation),
            current(inhibition),
            branch_current=current(branch_input),
            branch_conductance=child_conductance,
        )
    return output.double().cpu().numpy()


__all__ = ["compute_projected_branch_output"]
