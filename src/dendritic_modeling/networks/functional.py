"""Stateless current integration, without constructing synapses or layer owners.

These functions compute pre-reactivation voltages. They are not full networks:
they do not route inputs, generate noise, update weights, or apply output gates.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


def _shunting_current_components(
    excitation,
    recurrent_excitation,
    inhibition,
    recurrent_inhibition,
    child_current,
    child_conductance,
    leak,
):
    """Unchecked arithmetic shared by the checked API and the training paths.

    The recurrent/current-substitution path and the routed E/I control call it
    directly. Keep the order and the scalar-zero shortcuts: regrouping sums can
    change last bits in existing models and checkpoint analyses.
    """
    numerator = excitation + recurrent_excitation
    if torch.is_tensor(child_current) or child_current:
        numerator = numerator + child_current
    denominator = leak + excitation
    for term in (recurrent_excitation, inhibition, recurrent_inhibition):
        if torch.is_tensor(term) or term:
            denominator = denominator + term
    if child_conductance is not None:
        denominator = denominator + child_conductance
    return numerator, denominator


def _raw_current_sum(
    excitation, recurrent_excitation, inhibition, recurrent_inhibition, child_current
):
    voltage = excitation + recurrent_excitation - inhibition - recurrent_inhibition
    if torch.is_tensor(child_current) or child_current:
        voltage = voltage + child_current
    return voltage


def _reference(excitation):
    if not isinstance(excitation, torch.Tensor) or not excitation.is_floating_point():
        raise TypeError("excitation must be a floating-point torch.Tensor")
    if not excitation.numel():
        raise ValueError("excitation must be nonempty")
    return excitation


def _check_operand(value, reference, name, *, nonnegative=True, positive=False):
    if isinstance(value, torch.Tensor):
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"{name} must share excitation's dtype and device")
        try:
            shape = torch.broadcast_shapes(reference.shape, value.shape)
        except RuntimeError as error:
            raise ValueError(f"{name} must broadcast to excitation's shape") from error
        if shape != reference.shape:
            raise ValueError(f"{name} must not expand excitation's shape")
        # Each test synchronizes with the host; run only those that apply.
        finite = bool(torch.isfinite(value).all())
        below_zero = nonnegative and not positive and bool((value < 0).any())
        at_or_below_zero = positive and bool((value <= 0).any())
    elif isinstance(value, Real) and not isinstance(value, bool):
        finite = math.isfinite(value)
        below_zero = value < 0
        at_or_below_zero = value <= 0
    else:
        raise TypeError(f"{name} must be a real scalar or floating-point tensor")
    if not finite:
        raise ValueError(f"{name} must be finite")
    if positive and at_or_below_zero:
        raise ValueError(f"{name} must be positive")
    if nonnegative and below_zero:
        raise ValueError(f"{name} must be nonnegative")
    # A positive scalar can still round to zero in a low-precision dtype.
    if positive and not isinstance(value, torch.Tensor):
        if not torch.tensor(value, dtype=reference.dtype).item() > 0:
            raise ValueError(f"{name} rounds to zero in {reference.dtype}")


def shunting_voltage(
    excitation: torch.Tensor,
    inhibition: float | torch.Tensor = 0.0,
    *,
    recurrent_excitation: float | torch.Tensor = 0.0,
    recurrent_inhibition: float | torch.Tensor = 0.0,
    child_current: float | torch.Tensor = 0.0,
    child_conductance: float | torch.Tensor | None = None,
    leak: float | torch.Tensor = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Return ``(E + E_rec + C) / (L + E + E_rec + I + I_rec + G + epsilon)``.

    ``C`` is the summed child current and ``G`` its coupling conductance, not
    another excitatory input. A nonzero child current requires an explicit
    coupling conductance. E/I currents and coupling must be nonnegative; child
    current can be signed. Leak is strictly positive, including tensor-valued
    learnable leaks. This API does not implement a zero-leak mathematical
    quotient, a partial-division exponent, or a post-voltage reactivation.

    Tensor operands must share excitation's dtype/device and broadcast to its
    shape without expanding it. Autograd is preserved for every tensor input.
    The returned shape, dtype and device match excitation. Validation reads
    tensor values, so this checked API is intended for explicit-current models
    and analyses, not as a replacement for the fused layer training path; for
    the same reason it does not run under ``torch.func`` transforms such as
    ``vmap``.
    """
    reference = _reference(excitation)
    for name, value in (
        ("excitation", excitation),
        ("inhibition", inhibition),
        ("recurrent_excitation", recurrent_excitation),
        ("recurrent_inhibition", recurrent_inhibition),
    ):
        _check_operand(value, reference, name)
    _check_operand(child_current, reference, "child_current", nonnegative=False)
    _check_operand(leak, reference, "leak", positive=True)
    if not isinstance(epsilon, Real) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a real scalar")
    _check_operand(epsilon, reference, "epsilon")
    has_child_current = (
        bool((child_current != 0).any())
        if isinstance(child_current, torch.Tensor)
        else bool(child_current)
    )
    if child_conductance is None:
        if has_child_current:
            raise ValueError("nonzero child_current requires child_conductance")
    else:
        _check_operand(child_conductance, reference, "child_conductance")
    numerator, denominator = _shunting_current_components(
        excitation,
        recurrent_excitation,
        inhibition,
        recurrent_inhibition,
        child_current,
        child_conductance,
        leak,
    )
    return numerator / (denominator + epsilon)


def additive_voltage(
    excitation: torch.Tensor,
    inhibition: float | torch.Tensor = 0.0,
    *,
    recurrent_excitation: float | torch.Tensor = 0.0,
    recurrent_inhibition: float | torch.Tensor = 0.0,
    child_current: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Return the raw additive control ``E + E_rec - I - I_rec + C``.

    Currents obey the same shape, dtype/device and sign contract as
    :func:`shunting_voltage`. This is the unnormalized arithmetic comparator:
    it ignores coupling conductance, does not divide by leak, may be negative,
    and applies neither an output gate nor across-feature standardization.
    Any normalization of currents by a reference leak is the caller's explicit
    unit convention, not an implicit change to the native additive rule.
    """
    reference = _reference(excitation)
    for name, value in (
        ("excitation", excitation),
        ("inhibition", inhibition),
        ("recurrent_excitation", recurrent_excitation),
        ("recurrent_inhibition", recurrent_inhibition),
    ):
        _check_operand(value, reference, name)
    _check_operand(child_current, reference, "child_current", nonnegative=False)
    return _raw_current_sum(
        excitation,
        recurrent_excitation,
        inhibition,
        recurrent_inhibition,
        child_current,
    )


__all__ = ["additive_voltage", "shunting_voltage"]
