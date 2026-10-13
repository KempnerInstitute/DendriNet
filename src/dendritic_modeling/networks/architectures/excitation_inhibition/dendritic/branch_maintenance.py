"""Maintenance and inspection helpers for dendritic branch layers."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks.utils.sparse_maintenance import forward_maintenance


def decay_branch_synapse_weights(
    owner: Any,
    weight_decay,
    weight_boosting: bool = False,
) -> None:
    """Apply custom decay to all active sparse synapse sublayers."""
    forward_maintenance(owner, "decay_weights", weight_decay, weight_boosting)


def apply_branch_rewiring(owner: Any) -> None:
    """Apply one optimizer step of rewiring to all sparse synapse sublayers.

    Each sublayer owns its schedule: DeepST and indexed-rewire layers keep a
    checkpointed step counter and rewire every ``rewire_frequency`` calls, so
    this container path and the direct leaf helpers follow the same schedule.
    """
    forward_maintenance(owner, "apply_rewiring")


def branch_rewiring_step_count(owner: Any) -> int:
    """Return the optimizer steps seen by a branch layer's rewiring synapses.

    The count is read from the sublayers' checkpointed ``rewire_step``
    buffers, so it survives ``state_dict`` round trips.
    """
    count = 0
    for layer in (
        owner.branch_excitation,
        owner.branch_inhibition,
        owner.branch_recurrent,
        owner.branch_rec_inhibition,
    ):
        counter = getattr(layer, "rewire_step", None)
        if isinstance(counter, torch.Tensor):
            count = max(count, int(counter))
    return count


def collect_from_branch_synapse_layers(owner: Any, method_name: str):
    """Collect a named method result from all active sparse synapse sublayers."""
    result = {}
    if owner.branch_excitation is not None:
        result["exc"] = getattr(owner.branch_excitation, method_name)()
    if owner.branch_recurrent is not None:
        result["rec_exc"] = getattr(owner.branch_recurrent, method_name)()
    if owner.branch_inhibition is not None:
        result["inh"] = getattr(owner.branch_inhibition, method_name)()
    if owner.branch_rec_inhibition is not None:
        result["rec_inh"] = getattr(owner.branch_rec_inhibition, method_name)()
    return result if result else None


__all__ = [
    "apply_branch_rewiring",
    "branch_rewiring_step_count",
    "collect_from_branch_synapse_layers",
    "decay_branch_synapse_weights",
]
