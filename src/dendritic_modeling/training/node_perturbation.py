"""Node-perturbation estimates of errors from loss changes.

A node-perturbation estimate replaces the derivative of a loss with respect to
perturbable quantities (activations, or the compartment coordinates of
``perturb_compartments``) by the correlation of loss changes with random
perturbations. For per-example losses ``l_b(h_b)`` and antithetic draws
``n ~ N(0, I)`` of scale ``s``::

    g_hat_b = mean over draws of [l_b(h_b + s n) - l_b(h_b - s n)] / (2 s) * n

``g_hat`` is unbiased for the gradient when ``l_b`` is quadratic in ``h_b``;
otherwise it estimates the gradient of the loss smoothed by the Gaussian
perturbation, with a bias of order ``s**2``. With orthonormal basis columns
``Q`` (perturbations ``Q z``, ``z ~ N(0, I_m)``) it estimates ``Q Q^T`` times
that smoothed gradient, which approaches the projected gradient as ``s``
vanishes. Each draw costs two loss evaluations per example; the examples of a
batch are independent members of one evaluation.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


@dataclass(frozen=True)
class NodePerturbationEstimate:
    """Running estimates after each requested number of draws.

    ``estimates`` maps a draw count to the estimate of shape ``[B, D]``;
    ``rank`` is the number of perturbed directions per example (``D``, or the
    basis width). One draw costs two loss evaluations per example.
    """

    estimates: dict[int, torch.Tensor]
    scale: float
    rank: int

    @staticmethod
    def evaluations_per_example(draws: int) -> int:
        """Loss evaluations per example behind an estimate from ``draws`` draws."""
        return 2 * int(draws)


def group_basis(
    groups: Sequence[Sequence[int]],
    size: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Orthonormal columns that each average one group of coordinates.

    Column ``k`` is the normalized indicator of ``groups[k]`` among ``size``
    coordinates. Groups must be non-empty, disjoint and within range;
    coordinates in no group are not perturbed.
    """
    size = _positive_int(size, "size")
    if not groups:
        raise ValueError("give at least one group")
    seen: set[int] = set()
    basis = torch.zeros(size, len(groups), dtype=dtype, device=device)
    for column, group in enumerate(groups):
        members = list(group)
        if any(
            isinstance(index, bool) or not isinstance(index, numbers.Integral)
            for index in members
        ):
            raise TypeError(f"group {column} must hold integer indices, got {members}")
        members = [int(index) for index in members]
        if not members:
            raise ValueError(f"group {column} is empty")
        outside = [index for index in members if not 0 <= index < size]
        if outside:
            raise ValueError(
                f"group {column} has indices outside [0, {size}): {outside}"
            )
        repeated = seen.intersection(members) or (len(set(members)) != len(members))
        if repeated:
            raise ValueError(
                f"groups must be disjoint; group {column} repeats an index"
            )
        seen.update(members)
        basis[members, column] = 1.0 / math.sqrt(len(members))
    return basis


def check_orthonormal_basis(basis: torch.Tensor, size: int) -> None:
    """Raise unless ``basis`` is ``[size, m]`` with orthonormal columns."""
    if not isinstance(basis, torch.Tensor) or basis.ndim != 2:
        raise ValueError("a basis is a two-dimensional tensor [size, columns]")
    if basis.shape[0] != size or basis.shape[1] < 1:
        raise ValueError(
            f"a basis for {size} coordinates needs shape [{size}, m] with m >= 1, "
            f"got {tuple(basis.shape)}"
        )
    tolerance = 1e-10 if basis.dtype == torch.float64 else 1e-5
    gram = basis.T @ basis
    identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    if not torch.allclose(gram, identity, atol=tolerance, rtol=0):
        raise ValueError("basis columns must be orthonormal")


def node_perturbation_estimates(
    loss: Callable[[torch.Tensor], torch.Tensor],
    shape: Sequence[int],
    *,
    scale: float,
    draws: Iterable[int],
    generator: torch.Generator,
    basis: torch.Tensor | None = None,
    chunk: int = 16,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> NodePerturbationEstimate:
    """Antithetic node-perturbation estimates of per-example errors.

    ``loss(offset)`` maps perturbations of shape ``[c, B, D]`` (``shape`` is
    ``(B, D)``) to per-example losses of shape ``[c, B]``, without reducing
    over examples, and is called with ``+scale * noise`` and ``-scale *
    noise`` for each chunk of at most ``chunk`` draws. ``basis`` restricts the
    noise to orthonormal columns (``[D, m]``; see :func:`group_basis`).
    ``generator`` makes the draws reproducible, independently of ``chunk``;
    the draws are made on ``device``, by default the generator's, which must
    be the generator's device. The estimates for all counts in ``draws`` share
    the same draws in order, so larger counts extend smaller ones. Returns a
    :class:`NodePerturbationEstimate`.
    """
    shape = tuple(shape)
    if len(shape) != 2:
        raise ValueError(f"shape must be two positive sizes (B, D), got {shape}")
    batch, size = (_positive_int(value, "each size in shape") for value in shape)
    if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
        raise TypeError(f"scale must be a number, got {scale!r}")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    if isinstance(draws, (str, bytes)) or not isinstance(draws, Iterable):
        raise TypeError(f"draws must be a collection of counts, got {draws!r}")
    counts = sorted({_positive_int(value, "each draw count") for value in draws})
    if not counts:
        raise ValueError("give at least one draw count")
    chunk = _positive_int(chunk, "chunk")
    if not isinstance(generator, torch.Generator):
        raise TypeError("give a torch.Generator for reproducible draws")
    # Tensors carry a concrete device (cuda:0, not cuda); compare those.
    source = torch.empty(0, device=generator.device).device
    device = source if device is None else torch.empty(0, device=device).device
    if source != device:
        raise ValueError(f"the generator is on {source}, the draws are on {device}")
    if basis is not None:
        if basis.device != device or basis.dtype != dtype:
            raise ValueError("the basis must have the draws' device and dtype")
        check_orthonormal_basis(basis, size)
    width = size if basis is None else basis.shape[1]

    def evaluate(offset: torch.Tensor) -> torch.Tensor:
        value = loss(offset)
        if not isinstance(value, torch.Tensor) or value.shape != offset.shape[:2]:
            raise ValueError(
                "loss must return per-example losses of shape [draws, batch], "
                f"{tuple(offset.shape[:2])}; got "
                f"{tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}"
            )
        return value

    total = torch.zeros(batch, size, dtype=dtype, device=device)
    estimates: dict[int, torch.Tensor] = {}
    done = 0
    for target in counts:
        while done < target:
            c = min(chunk, target - done)
            # One draw at a time: the sampler's values depend on the tensor
            # size, so drawing per draw keeps estimates independent of chunk.
            z = torch.stack(
                [
                    torch.randn(
                        (batch, width), generator=generator, dtype=dtype, device=device
                    )
                    for _ in range(c)
                ]
            )
            noise = z if basis is None else z @ basis.T
            with torch.no_grad():
                change = (evaluate(scale * noise) - evaluate(-scale * noise)) / 2.0
            total += (change[..., None] * noise).sum(0) / scale
            done += c
        estimates[target] = total / target
    return NodePerturbationEstimate(estimates=estimates, scale=scale, rank=width)


__all__ = [
    "NodePerturbationEstimate",
    "check_orthonormal_basis",
    "group_basis",
    "node_perturbation_estimates",
]
