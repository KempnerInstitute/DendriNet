"""Copying and pickling modules that keep tensors of their last forward pass.

Several modules keep tensors of their most recent forward pass for analyses:
population outputs, routing information, branch currents. After a training
forward these tensors belong to an autograd graph, and ``copy.deepcopy``
cannot copy such tensors. A module lists these attributes and returns
:func:`detach_forward_caches` of its state from ``__getstate__``: copies and
pickles then hold the cached values without their graph, while the module
itself, its forward pass and what analyses read from it are unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import torch


def detach_graph_tensors(value: Any) -> Any:
    """Return ``value`` with every tensor that has an autograd graph detached.

    Dicts, lists and tuples are rebuilt around their detached entries, so the
    containers of the module that owns ``value`` are not modified. Leaf
    tensors (parameters, buffers) and other values are returned as they are.
    """
    if torch.is_tensor(value):
        return value.detach() if value.grad_fn is not None else value
    if isinstance(value, dict):
        detached = copy.copy(value)
        for key, item in value.items():
            detached[key] = detach_graph_tensors(item)
        return detached
    if isinstance(value, list):
        return [detach_graph_tensors(item) for item in value]
    if isinstance(value, tuple):
        items = [detach_graph_tensors(item) for item in value]
        if hasattr(value, "_fields"):  # named tuple
            return type(value)(*items)
        return type(value)(items)
    return value


def detach_forward_caches(
    state: dict[str, Any], names: Iterable[str]
) -> dict[str, Any]:
    """Detach the named forward caches in a ``__getstate__`` state dictionary.

    ``state`` is the shallow copy of the module's ``__dict__`` that
    ``nn.Module.__getstate__`` returns; only its entries are replaced.
    """
    for name in names:
        if name in state:
            state[name] = detach_graph_tensors(state[name])
    return state


__all__ = ["detach_forward_caches", "detach_graph_tensors"]
