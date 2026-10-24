"""Explicit re-initialization recipes for ``DendriNet``.

A recipe redraws a built network's log-conductances from the global random
stream at the moment it is applied, so a protocol controls exactly where in
its random stream the draws fall (for example after building a readout).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)

OFFSET_TARGETS = ("excitatory", "inhibitory", "coupling")


@dataclass(frozen=True)
class LogOffset:
    """A constant added to one level's log-weights after the draws."""

    depth: int
    target: str
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError("log offset depth must be an integer")
        if self.target not in OFFSET_TARGETS:
            raise ValueError(
                f"log offset target must be one of {OFFSET_TARGETS}, got {self.target!r}"
            )
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("log offset value must be a number")

    @classmethod
    def from_config(cls, value: LogOffset | Mapping[str, Any]) -> LogOffset:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            known = {item.name for item in fields(cls)}
            unknown = sorted(set(value) - known)
            if unknown:
                raise ValueError(f"Unknown log offset keys: {unknown}")
            return cls(**dict(value))
        raise TypeError("a log offset must be a LogOffset or mapping")


@dataclass(frozen=True)
class LogNormalReinitialization:
    """Redraw synaptic and coupling log-weights from normal distributions.

    Levels are visited in construction order (terminal level first, soma
    last). At each level the excitatory, then the inhibitory synaptic
    log-weights are drawn from N(``synapse_log_mean``, ``synapse_log_sd``)
    over the full dense tensors (entries outside the connection mask
    included), then the level's coupling log-weights from
    N(``coupling_log_mean``, ``coupling_log_sd``). The ``log_offsets`` are
    added after all draws, in the order given. Requires exponential weight
    transforms (the parameters are log-conductances) and dense synapse layers.
    """

    synapse_log_sd: float
    coupling_log_sd: float
    synapse_log_mean: float = 0.0
    coupling_log_mean: float = 0.0
    log_offsets: tuple[LogOffset, ...] = ()

    def __post_init__(self) -> None:
        for name in ("synapse_log_sd", "coupling_log_sd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        offsets = self.log_offsets
        if isinstance(offsets, (str, bytes, Mapping)) or not isinstance(
            offsets, Sequence
        ):
            raise TypeError("log_offsets must be a list of offsets")
        object.__setattr__(
            self, "log_offsets", tuple(LogOffset.from_config(item) for item in offsets)
        )


def _synapse_targets(layer: Any) -> list[tuple[str, Any]]:
    targets = []
    for target, attribute in (
        ("excitatory", "branch_excitation"),
        ("inhibitory", "branch_inhibition"),
    ):
        synapse = getattr(layer, attribute, None)
        if synapse is None:
            continue
        if not isinstance(synapse, TopKLinear):
            raise ValueError(
                "LogNormalReinitialization needs dense synapse layers (the TopK "
                f"family); level {layer.layer_idx} uses {type(synapse).__name__}"
            )
        targets.append((target, synapse.pre_w))
    return targets


def apply_lognormal_reinitialization(
    branch_layers: Sequence[Any], recipe: LogNormalReinitialization
) -> None:
    """Apply ``recipe`` to the branch layers of one DendriNet."""
    for layer in branch_layers:
        if str(getattr(layer, "weight_transform", "")).lower() != "exp":
            raise ValueError(
                "LogNormalReinitialization needs weight_transform: exp (log-"
                f"conductance parameters); level {layer.layer_idx} uses "
                f"{layer.weight_transform!r}"
            )
        for attribute in ("branch_recurrent", "branch_rec_inhibition"):
            if getattr(layer, attribute, None) is not None:
                raise ValueError(
                    "LogNormalReinitialization does not cover recurrent synapses"
                )
    by_depth = {layer.layer_idx: layer for layer in branch_layers}
    for offset in recipe.log_offsets:
        layer = by_depth.get(offset.depth)
        if layer is None:
            raise ValueError(f"log offset depth {offset.depth} is not a level")
        if offset.target == "coupling":
            if not getattr(layer, "input_branches", False):
                raise ValueError(f"level {offset.depth} has no couplings to offset")
        elif offset.target not in dict(_synapse_targets(layer)):
            raise ValueError(f"level {offset.depth} has no {offset.target} pathway")
    with torch.no_grad():
        for layer in branch_layers:
            for _target, weight in _synapse_targets(layer):
                weight.normal_(recipe.synapse_log_mean, recipe.synapse_log_sd)
            if getattr(layer, "input_branches", False):
                layer.branches_to_output.log_weight.normal_(
                    recipe.coupling_log_mean, recipe.coupling_log_sd
                )
        for offset in recipe.log_offsets:
            layer = by_depth[offset.depth]
            if offset.target == "coupling":
                layer.branches_to_output.log_weight.add_(offset.value)
            else:
                dict(_synapse_targets(layer))[offset.target].add_(offset.value)


__all__ = [
    "LogNormalReinitialization",
    "LogOffset",
    "apply_lognormal_reinitialization",
]
