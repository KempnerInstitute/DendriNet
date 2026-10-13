"""Optimizer-step maintenance plumbing for sparse synapse leaves.

The trainer issues two maintenance calls per optimizer update:
``decay_weights(rate, weight_boosting)`` before the update and
``apply_rewiring()`` after it. Sparse synapse leaves (modules owning a
``pre_w`` parameter) implement them; containers forward them to the leaves
they own. This module provides both halves of that contract:

* :func:`maintenance_applies` is the leaf-side guard. A leaf whose weights are
  frozen (``requires_grad=False``, e.g. layers frozen by the layer-wise
  trainer) is not maintained, and inside :func:`restrict_maintenance_to` a leaf
  whose weights the stepping optimizer does not own is not maintained either.
* :func:`iter_maintenance_targets` and :func:`forward_maintenance` reach every
  sparse leaf of a module tree exactly once: a module that implements the
  method is called and is responsible for its own descendants; any other
  module is searched recursively.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn

MAINTENANCE_METHODS = ("decay_weights", "apply_rewiring")

_OWNED_PARAMETER_IDS: ContextVar[frozenset[int] | None] = ContextVar(
    "dendritic_sparse_maintenance_owned_parameter_ids", default=None
)
_VISITED_MODULES: ContextVar[dict[str, set[int]] | None] = ContextVar(
    "dendritic_sparse_maintenance_visited_modules", default=None
)


def maintenance_applies(parameter: torch.Tensor | None) -> bool:
    """Return whether optimizer-step maintenance may change ``parameter``."""
    if parameter is None or not bool(getattr(parameter, "requires_grad", False)):
        return False
    owned = _OWNED_PARAMETER_IDS.get()
    return owned is None or id(parameter) in owned


def optimizer_parameters(optimizer) -> Iterator[torch.Tensor]:
    """Yield every parameter held by ``optimizer``'s parameter groups."""
    for group in optimizer.param_groups:
        yield from group["params"]


@contextmanager
def restrict_maintenance_to(parameters: Iterable[torch.Tensor]) -> Iterator[None]:
    """Maintain only leaves whose weights are among ``parameters``."""
    parameters = tuple(parameters)
    if any(getattr(p, "_is_flat_param", False) for p in parameters):
        raise ValueError(
            "Sparse maintenance cannot resolve flattened optimizer ownership; "
            "use original parameters or disable maintenance for this wrapper"
        )
    token = _OWNED_PARAMETER_IDS.set(frozenset(id(p) for p in parameters))
    try:
        yield
    finally:
        _OWNED_PARAMETER_IDS.reset(token)


def _is_maintenance_target(module: nn.Module, method_name: str) -> bool:
    if not callable(getattr(module, method_name, None)):
        return False
    # Sparse synapse leaves own ``pre_w``; containers forward to their
    # children. A childless module without ``pre_w`` that happens to define the
    # method (``BlockLinear``) is not a sparse synapse and was never part of
    # optimizer-step maintenance.
    return hasattr(module, "pre_w") or next(module.children(), None) is not None


def iter_maintenance_targets(
    modules: Iterable[nn.Module | None],
    method_name: str,
) -> Iterator[nn.Module]:
    """Yield the outermost modules implementing ``method_name``, each once.

    ``modules`` are visited in order, depth first in registration order. A
    module that implements the method is yielded and not searched further
    (it forwards to its own leaves); any other module is searched.
    """
    if method_name not in MAINTENANCE_METHODS:
        raise ValueError(
            f"method_name must be one of {MAINTENANCE_METHODS}, got {method_name!r}"
        )
    traversal = _VISITED_MODULES.get()
    seen: set[int] = (
        set() if traversal is None else traversal.setdefault(method_name, set())
    )

    def visit(module: nn.Module) -> Iterator[nn.Module]:
        if id(module) in seen:
            return
        seen.add(id(module))
        if _is_maintenance_target(module, method_name):
            yield module
            return
        for child in module.children():
            yield from visit(child)

    for module in modules:
        if isinstance(module, nn.Module):
            yield from visit(module)


def forward_maintenance(owner: nn.Module, method_name: str, *args) -> None:
    """Forward a maintenance call to every sparse leaf below ``owner`` once.

    ``owner`` itself is not called; use this inside a container's own
    ``decay_weights``/``apply_rewiring``.
    """
    # Nested forwarding containers share a traversal, including aliased leaves.
    # A fresh outer call must start a new traversal for the next optimizer step.
    token = None
    if _VISITED_MODULES.get() is None:
        token = _VISITED_MODULES.set({})
    try:
        for target in iter_maintenance_targets(owner.children(), method_name):
            getattr(target, method_name)(*args)
    finally:
        if token is not None:
            _VISITED_MODULES.reset(token)


def apply_maintenance(model: nn.Module, method_name: str, *args) -> None:
    """Issue a maintenance call on ``model``, or on its leaves if it has none."""
    method = getattr(model, method_name, None)
    if callable(method):
        method(*args)
    else:
        forward_maintenance(model, method_name, *args)


__all__ = [
    "MAINTENANCE_METHODS",
    "apply_maintenance",
    "forward_maintenance",
    "iter_maintenance_targets",
    "maintenance_applies",
    "optimizer_parameters",
    "restrict_maintenance_to",
]


_SPARSE_STATE_FROZEN: ContextVar[bool] = ContextVar(
    "dendritic_sparse_state_frozen", default=False
)


@contextmanager
def frozen_sparse_state():
    """Keep forward passes from changing sparse-synapse training state.

    Inside the context, dense-to-sparse schedules do not advance, variance
    Top-K statistics are not updated and release synapses do not track source
    activity or impact. A forward uses the current state: variance Top-K then
    selects from the statistics without this batch. Auxiliary training-mode
    passes (for example a no-grad target forward) use it so that only
    training forwards advance this state.
    """
    token = _SPARSE_STATE_FROZEN.set(True)
    try:
        yield
    finally:
        _SPARSE_STATE_FROZEN.reset(token)


def sparse_state_frozen() -> bool:
    """Whether :func:`frozen_sparse_state` is active."""
    return _SPARSE_STATE_FROZEN.get()


def forward_stateful_modules(module: nn.Module) -> list[str]:
    """Names of submodules whose training forward changes their state."""
    return [
        name
        for name, child in module.named_modules()
        if callable(getattr(child, "forward_mutates_state", None))
        and child.forward_mutates_state()
    ]


# Data-parallel training gives every rank its own batch. Synapses whose
# training forward keeps statistics of that batch (variance Top-K input
# moments, release impact and source activity) declare the class attribute
# ``reduce_statistics_across_ranks = False``; the data-parallel wrappers set it
# on each instance, and those forwards then reduce their batch sums over all
# ranks, so every rank keeps the statistics of the global batch and makes the
# same selections and rewiring decisions.


def mark_rank_reduced_statistics(model: nn.Module) -> list[str]:
    """Make the model's batch-statistics synapses reduce over the ranks.

    Called by the data-parallel wrappers when more than one rank trains. Every
    rank must then run the same training forwards (as data-parallel training
    does), since each such forward is a collective. Returns the marked names.
    """
    names = []
    for name, module in model.named_modules():
        if hasattr(type(module), "reduce_statistics_across_ranks"):
            module.reduce_statistics_across_ranks = True
            names.append(name)
    return names


def reduces_statistics_across_ranks(module: nn.Module) -> bool:
    """Whether a training forward of ``module`` reduces its batch statistics."""
    return (
        bool(getattr(module, "reduce_statistics_across_ranks", False))
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def all_reduce_sums(*values: torch.Tensor) -> list[torch.Tensor]:
    """Sum each tensor over all ranks in one float64 collective."""
    flat = torch.cat([value.detach().reshape(-1).to(torch.float64) for value in values])
    torch.distributed.all_reduce(flat)
    reduced, offset = [], 0
    for value in values:
        reduced.append(flat[offset : offset + value.numel()].reshape(value.shape))
        offset += value.numel()
    return reduced


__all__ += [
    "all_reduce_sums",
    "forward_stateful_modules",
    "frozen_sparse_state",
    "mark_rank_reduced_statistics",
    "reduces_statistics_across_ranks",
    "sparse_state_frozen",
]
