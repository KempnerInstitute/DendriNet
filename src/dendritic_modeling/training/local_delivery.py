"""Signal delivery for local rules with autograd eligibilities.

A local rule of this kind trains every level of a ``DendriNet`` by the
gradient of a surrogate objective: the sum over levels of the level's output
times a delivered error, with each level's child input detached
(``DendriNet.forward_levels(..., detach_child_inputs=True)``). Autograd then
gives each level's parameters their exact local eligibility times the
delivered error. The delivered error is the soma error repeated over each
neuron's compartments, times a gain built from the quantities below. The
delivered error and its gains are supplied signals, not differentiated:
``delivered_error_objective`` rejects graph-connected ones, and the gain
quantities are computed without gradients.

The functions here compute those quantities for one level; which products
form a rule, and any permutation or noise applied to them, belongs to the
protocol that uses them. The delivered error equals the backpropagated one
when a level's gain is the product, over every ancestor on its path to the
soma (the soma included), of the ancestor's activation slope, coupling ratio
and relative resistance (1 at an ancestor without inhibition).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def soma_error(loss: torch.Tensor, soma_output: torch.Tensor) -> torch.Tensor:
    """Detached gradient of ``loss`` with respect to the soma outputs."""
    return torch.autograd.grad(loss, soma_output, retain_graph=True)[0].detach()


def _require_shunting(layer: Any, name: str) -> None:
    if not getattr(layer, "use_shunting", False):
        raise ValueError(f"{name} is defined for shunting levels")


def _uninhibited_denominator(layer: Any, excitatory_input) -> torch.Tensor:
    """D0 + epsilon: the level's forward denominator without inhibition.

    Leak, the excitatory conductance where the level has that pathway, the
    coupling conductance, and the level's ``epsilon``, which the forward pass
    adds to its denominator (``V = N / (D + epsilon)``).
    """
    base = layer.effective_leak
    if layer.branch_excitation is not None:
        if excitatory_input is None:
            raise ValueError(
                f"level {layer.layer_idx} has an excitatory pathway; pass "
                "excitatory_input"
            )
        base = base + layer.branch_excitation(excitatory_input)
    if layer.input_branches:
        base = base + layer.branches_to_output.sum_conductances()
    if layer.epsilon:
        base = base + layer.epsilon
    return base


@torch.no_grad()
def relative_resistance(
    layer: Any, inhibitory_input: torch.Tensor, excitatory_input=None
) -> torch.Tensor:
    """(D0 + eps) / (D0 + I + eps) for every compartment of a shunting level.

    ``I`` is the level's inhibitory conductance for ``inhibitory_input`` and
    ``D0`` its total conductance without inhibition: leak plus coupling
    conductance, plus the excitatory conductance at levels with an excitatory
    pathway (pass ``excitatory_input`` there). ``eps`` is the level's
    ``epsilon``, which the forward pass adds to the denominator. This is the
    input resistance left by inhibition relative to the uninhibited compartment.
    """
    _require_shunting(layer, "relative_resistance")
    if layer.branch_inhibition is None:
        raise ValueError(f"level {layer.layer_idx} has no inhibitory pathway")
    base = _uninhibited_denominator(layer, excitatory_input)
    inhibition = layer.branch_inhibition(inhibitory_input)
    return base / (base + inhibition)


@torch.no_grad()
def activation_slope(layer: Any) -> torch.Tensor:
    """The level's activation derivative at its last forward voltage.

    Needs branch diagnostics on the level (``set_branch_diagnostics(True)``
    before the forward pass). Supported activations: identity and tanh.
    """
    voltage = layer.get_branch_diagnostics().get("V")
    if voltage is None:
        raise ValueError(
            f"level {layer.layer_idx} has no stored voltage; enable branch "
            "diagnostics before the forward pass"
        )
    if isinstance(layer.reactivation, nn.Identity):
        return torch.ones_like(voltage)
    if isinstance(layer.reactivation, nn.Tanh):
        return 1 - voltage.tanh().square()
    raise NotImplementedError(
        f"activation_slope supports identity and tanh activations, not "
        f"{type(layer.reactivation).__name__}"
    )


@torch.no_grad()
def coupling_ratios(layer: Any, excitatory_input=None) -> torch.Tensor:
    """g_child / (D0 + eps) for every child of every parent compartment.

    ``D0`` and ``eps`` are as in ``relative_resistance``: the parent's
    conductance without inhibition (leak plus coupling conductance, plus the
    excitatory conductance at levels with an excitatory pathway; pass
    ``excitatory_input`` there) and its ``epsilon``. Shape
    ``[parents, children per parent]``, or ``[batch, parents, children per
    parent]`` when the excitatory conductance enters. Times
    ``relative_resistance`` it is the exact ratio g_child / (D + eps) by which a
    child's output moves its parent's voltage.
    """
    _require_shunting(layer, "coupling_ratios")
    if not layer.input_branches:
        raise ValueError(f"level {layer.layer_idx} has no child branches")
    base = _uninhibited_denominator(layer, excitatory_input)
    return layer.branches_to_output.weight() / base[..., None]


def expand_to_children(values: torch.Tensor, branch_factor: int) -> torch.Tensor:
    """Repeat per-parent values over each parent's children (last dimension)."""
    return values.repeat_interleave(branch_factor, dim=-1)


def delivered_error_objective(
    level_outputs: Sequence[torch.Tensor],
    error: torch.Tensor,
    gains: Sequence[torch.Tensor | None],
) -> torch.Tensor:
    """Sum over levels of output x (error repeated over compartments x gain).

    ``error`` has one column per neuron; each level's outputs are ordered
    neuron by neuron, so the error is repeated over each neuron's
    compartments. A gain of ``None`` delivers the repeated error unchanged.
    The error and the gains must be detached: a graph-connected one would add
    its own derivative to the update, so it raises. Add the result to the
    loss and backpropagate.
    """
    if len(level_outputs) != len(gains):
        raise ValueError("give one gain (or None) per level output")
    if error.requires_grad:
        raise ValueError(
            "the delivered error must be detached; soma_error returns it detached"
        )
    if any(gain is not None and gain.requires_grad for gain in gains):
        raise ValueError(
            "gains must be detached (compute them under torch.no_grad() or detach "
            "them); a graph-connected gain would add its own derivative"
        )
    n_soma = error.shape[-1]
    total = 0
    for output, gain in zip(level_outputs, gains):
        if output.shape[-1] % n_soma:
            raise ValueError(
                f"a level with {output.shape[-1]} compartments does not divide "
                f"into {n_soma} neurons"
            )
        delivered = error.repeat_interleave(output.shape[-1] // n_soma, dim=-1)
        if gain is not None:
            delivered = delivered * gain
        total = total + (output * delivered).sum()
    return total


__all__ = [
    "activation_slope",
    "coupling_ratios",
    "delivered_error_objective",
    "expand_to_children",
    "relative_resistance",
    "soma_error",
]
