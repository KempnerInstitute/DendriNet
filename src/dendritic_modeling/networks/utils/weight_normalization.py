"""Row normalization of synaptic weights to a fixed p-norm.

``normalize_weight_rows`` scales every row of synaptic weights to p-norm
``gamma``. Rows whose norm is at least machine epsilon use the historical
expression, so their values and gradients (reverse and forward mode) are
bit-identical to earlier releases. That expression floors the norm at machine
epsilon, which shrank smaller rows, and the L2 norm of very small weights
underflows to zero.

Smaller rows are normalized after division by their largest weight. Rows of
``exp`` and ``softplus`` weights do this in log space, on their pre-transform
parameters, so they keep finite gradients when the weights are subnormal or
underflow. For ``exp`` the result depends exactly on differences of those
parameters. For ``softplus`` the parameter stands in for the log weight: every
weight of a small row lies below machine epsilon, where
``softplus(x) = exp(x) (1 - exp(x) / 2 + ...)`` differs from ``exp(x)`` by a
relative error below half of machine epsilon. Both transforms are then subject
to the usual rounding of the normalization arithmetic, which grows with the
number of contacts as for rows above machine epsilon. The exact gradients of
other small rows grow as ``gamma`` over the row norm and overflow to infinity
only when that norm is subnormal. Rows without contacts (``exp`` and
``softplus``) or without nonzero weights (other transforms) are returned
unchanged, and orders ``p <= 0``, which are not norms, keep the historical
expression for every row.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from dendritic_modeling.networks.utils.weight_transforms import (
    STRICTLY_POSITIVE_WEIGHT_TRANSFORMS,
)


class _KeepEntries(torch.autograd.Function):
    """Return ``values`` where ``keep`` holds and ``fill`` elsewhere.

    Outside ``keep`` the gradient is negative zero, which leaves any gradient
    it is added to unchanged bit for bit (positive zero would turn a negative
    zero positive). The rows the small-row branch does not handle therefore
    receive exactly the historical gradient.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(values, keep, fill):
        return torch.where(keep, values, fill)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1])
        ctx.save_for_forward(inputs[1])

    @staticmethod
    def backward(ctx, grad):
        (keep,) = ctx.saved_tensors
        return torch.where(keep, grad, -0.0), None, None

    @staticmethod
    def jvp(ctx, values_tangent, keep_tangent, fill_tangent):
        if values_tangent is None:
            return None
        (keep,) = ctx.saved_tensors
        return torch.where(keep, values_tangent, 0.0)


def _keep_entries(
    values: torch.Tensor, keep: torch.Tensor, fill: float | torch.Tensor
) -> torch.Tensor:
    # torch.compile cannot trace a function with a custom jvp; compiled code
    # does not reproduce eager rounding, so it needs no negative-zero gradient.
    if torch.compiler.is_compiling():
        return torch.where(keep, values, fill)
    return _KeepEntries.apply(values, keep, fill)


def normalize_weight_rows(
    weight: torch.Tensor,
    order: float,
    gamma: float | torch.Tensor,
    *,
    pre_weight: torch.Tensor | None = None,
    transform: str | None = None,
    contacts: torch.Tensor | None = None,
    sum_floor: float | None = None,
) -> torch.Tensor:
    """Scale each row of ``weight`` (its last dimension) to p-norm ``gamma``.

    Args:
        weight: Synaptic weights; entries that are not contacts are zero.
        order: The p of the norm, as for ``torch.norm``. Orders ``p <= 0``
            keep the historical expression for every row.
        gamma: Target norm of every row.
        pre_weight: Pre-transform parameters in the layout of ``weight``, with
            ``weight == transform(pre_weight)`` on the contacts.
        transform: The weight transform. With ``pre_weight`` and ``exp`` or
            ``softplus``, rows below machine epsilon normalize in log space.
        contacts: Boolean mask of the contacts in ``weight`` for the log-space
            rows; ``None`` means every entry is a contact.
        sum_floor: Explicit clamped-sum convention for nonnegative, order-1
            weights: ``weight / clamp(sum(weight), min=sum_floor) * gamma``.
            Must remain finite and strictly positive in ``weight.dtype``.
            Small rows then shrink rather than have unit norm. This opt-in
            convention preserves reduced-model replays; ``None`` retains the
            existing robust p-norm algorithm, including every default call.

    Returns:
        The normalized weights.
    """
    if sum_floor is not None:
        if (
            order != 1
            or isinstance(sum_floor, bool)
            or not isinstance(sum_floor, Real)
            or not math.isfinite(sum_floor)
            or sum_floor <= 0
        ):
            raise ValueError("sum_floor requires order=1 and a finite positive floor")
        if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError("clamped-sum weights must be finite and nonnegative")
        return divide_rows_by_sum(weight, sum_floor) * gamma
    norm = torch.norm(weight, p=order, dim=-1, keepdim=True)
    eps = torch.finfo(weight.dtype).eps
    # The historical expression, kept for every row whose norm is >= eps.
    safe_norm = norm.clamp_min(eps)
    historical = torch.where(norm > 0, (weight / safe_norm) * gamma, weight)
    if not order > 0:
        # With a zero entry such an order's "norm" is zero; nothing to rescale.
        return historical

    below_eps = norm.detach() < eps
    if pre_weight is not None and transform in STRICTLY_POSITIVE_WEIGHT_TRANSFORMS:
        if contacts is None:
            small = active = below_eps
        else:
            active = contacts & below_eps
            small = active.any(dim=-1, keepdim=True)
        unit = _log_space_unit_rows(pre_weight, small, active)
    else:
        peak = weight.detach().abs().amax(dim=-1, keepdim=True)
        small = below_eps & (peak > 0)
        unit = _keep_entries(weight, small, 0.0)
        unit = unit / torch.where(small, peak, torch.ones_like(peak))
    unit_norm = torch.linalg.vector_norm(unit, ord=order, dim=-1, keepdim=True)
    scale = gamma / torch.where(small, unit_norm, torch.ones_like(unit_norm))
    return torch.where(small, unit * scale, historical)


def divide_rows_by_sum(
    weight: torch.Tensor, floor: float | None = None
) -> torch.Tensor:
    """Divide each row of nonnegative ``weight`` by its sum.

    With ``floor=None`` every row with a positive sum sums to one and all-zero
    rows stay zero. Unlike the p-norm path of :func:`normalize_weight_rows`,
    the sum is differentiated at zero entries too, as straight-through routing
    needs. A positive ``floor`` divides by ``max(sum, floor)``, the clamped-sum
    convention, so rows below the floor shrink. The scalar floor must remain
    finite and strictly positive in ``weight.dtype``. Callers guarantee finite,
    nonnegative weights; ``normalize_weight_rows(..., sum_floor=...)`` checks
    them first.
    """
    if floor is not None:
        if (
            isinstance(floor, bool)
            or not isinstance(floor, Real)
            or not math.isfinite(floor)
            or floor <= 0
        ):
            raise ValueError("floor must be finite and positive")
        converted = torch.tensor(floor, dtype=weight.dtype, device="cpu").item()
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError(f"floor must remain finite and positive in {weight.dtype}")
    total = weight.sum(dim=-1, keepdim=True)
    if floor is None:
        return weight / torch.where(total > 0, total, torch.ones_like(total))
    return weight / total.clamp_min(floor)


def _log_space_unit_rows(
    pre_weight: torch.Tensor,
    small: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """``exp(x - max x)`` on the active entries of small rows, else 0.

    Inactive entries are filled with the shift, which keeps the exponential
    away from underflow (much slower on CPUs), and zeroed afterwards.
    """
    peak = torch.where(active, pre_weight.detach(), float("-inf"))
    peak = peak.amax(dim=-1, keepdim=True)
    shift = torch.where(small, peak, torch.zeros_like(peak))
    log_weight = _keep_entries(pre_weight, active, shift)
    return torch.where(active, (log_weight - shift).exp_(), 0.0)


__all__ = ["divide_rows_by_sum", "normalize_weight_rows"]
