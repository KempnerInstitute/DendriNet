"""Closed-form passive branch/soma equilibria; no recurrent state update."""

import torch


def passive_star_soma_voltage(
    excitation: torch.Tensor,
    membrane_conductance: torch.Tensor,
    axial_conductance: torch.Tensor,
    soma_leak: float = 1.0,
) -> torch.Tensor:
    """Solve independent passive stars, reducing the final branch dimension.

    For branch current E, membrane load h and axial coupling g, the branch
    equilibrium is (E + g*s)/(h + g). The soma has leak to reversal zero
    and no direct input. Inputs broadcast; h and soma_leak must be positive
    and g nonnegative. Excitation is a current, not an extra conductance;
    callers construct h according to their model's reversal potentials.

    This positive-denominator expression avoids cancellation at large g. Its
    operation order is fixed, so results are reproducible bit for bit. It adds
    no model state, changes no layer defaults and performs no casts.
    """
    a = membrane_conductance + axial_conductance
    return (axial_conductance * excitation / a).sum(-1) / (
        soma_leak + (axial_conductance * membrane_conductance / a).sum(-1)
    )
