"""Per-level structure overrides for ``DendriNet``.

A DendriNet builds every level from the same synapse counts, connectivity and
activation. A level override changes one level: it removes a synaptic
pathway, declares the inputs each compartment contacts, or gives the level its
own activation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch

from dendritic_modeling.networks.activations.factory import (
    get_registered_activation_types,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)

PATHWAYS = ("excitatory", "inhibitory")


def _contact_lists(value: Any, name: str) -> tuple[tuple[int, ...], ...] | None:
    """Normalize per-compartment contact lists to nested tuples of ints."""
    if value is None:
        return None
    message = f"{name} must be a list of input-index lists, one per compartment"
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise TypeError(message)
    rows = []
    for row in value:
        if isinstance(row, (str, bytes, Mapping)) or not isinstance(row, Sequence):
            raise TypeError(message)
        indices = []
        for index in row:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"{name} entries must be integer input indices")
            indices.append(index)
        if not indices:
            raise ValueError(f"every compartment in {name} needs at least one contact")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} repeats an input within one compartment")
        rows.append(tuple(indices))
    if not rows:
        raise ValueError(f"{name} must list at least one compartment")
    return tuple(rows)


@dataclass(frozen=True)
class DendriticLevelOverride:
    """Structure of one DendriNet level that differs from the shared settings.

    ``depth`` counts levels from the soma: 0 is the soma, 1 the level whose
    compartments couple into the soma, and the terminal level has depth
    ``len(branch_factors)``.

    ``excitatory: false`` or ``inhibitory: false`` remove that feedforward
    pathway from the level. The pathway is built and initialized with the
    others and removed afterwards, so every remaining parameter receives the
    same random draws and initial values as in the model with the pathway
    (matched initialization across pathway ablations).

    ``excitatory_contacts`` and ``inhibitory_contacts`` declare the inputs
    each compartment of the level contacts: a list of input indices per
    compartment, either for every compartment of the level or once for the
    compartments of one neuron, repeated for every neuron (compartments are
    ordered neuron by neuron). They become the pathway's connection mask,
    which is derived from configuration and not saved in checkpoints.
    Selection by the pathway's synapse count still applies among the declared
    contacts; declare at most that many per compartment for all of them to be
    active.

    ``activation`` names the level's activation (any registered activation
    type; ``identity`` or ``none`` for none). ``None`` keeps the network-wide
    setting.
    """

    depth: int
    excitatory: bool = True
    inhibitory: bool = True
    activation: str | None = None
    excitatory_contacts: tuple[tuple[int, ...], ...] | None = None
    inhibitory_contacts: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.depth, bool) or not isinstance(self.depth, int):
            raise TypeError("level override depth must be an integer")
        if self.depth < 0:
            raise ValueError("level override depth must be >= 0 (0 is the soma)")
        for pathway in PATHWAYS:
            if not isinstance(getattr(self, pathway), bool):
                raise TypeError(f"level override {pathway} must be true or false")
        if self.activation is not None:
            if not isinstance(self.activation, str):
                raise TypeError("level override activation must be a string")
            name = self.activation.lower()
            if name not in {"none", "identity"} and (
                name not in get_registered_activation_types()
            ):
                raise ValueError(f"Unknown level activation: {self.activation}")
        for pathway in PATHWAYS:
            key = f"{pathway}_contacts"
            contacts = _contact_lists(getattr(self, key), key)
            object.__setattr__(self, key, contacts)
            if contacts is not None and not getattr(self, pathway):
                raise ValueError(
                    f"level {self.depth} removes its {pathway} pathway and also "
                    f"declares {key}"
                )

    @classmethod
    def from_config(
        cls, value: DendriticLevelOverride | Mapping[str, Any]
    ) -> DendriticLevelOverride:
        """Normalize an override object or mapping."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            known = {item.name for item in fields(cls)}
            unknown = sorted(set(value) - known)
            if unknown:
                raise ValueError(
                    f"Unknown level override keys: {unknown}; expected {sorted(known)}"
                )
            if "depth" not in value:
                raise ValueError("a level override needs a depth")
            return cls(**dict(value))
        raise TypeError("a level override must be a DendriticLevelOverride or mapping")

    def to_config(self) -> dict[str, Any]:
        """Return the override as a plain mapping (lists for contacts)."""
        result: dict[str, Any] = {"depth": self.depth}
        defaults = {item.name: item.default for item in fields(self)}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "depth" or value == defaults[item.name]:
                continue
            if item.name.endswith("_contacts"):
                value = [list(row) for row in value]
            result[item.name] = value
        return result

    @property
    def is_identity(self) -> bool:
        """Whether the override leaves its level unchanged."""
        return (
            self.excitatory
            and self.inhibitory
            and self.activation is None
            and self.excitatory_contacts is None
            and self.inhibitory_contacts is None
        )


def normalize_level_overrides(values: Any) -> tuple[DendriticLevelOverride, ...]:
    """Return overrides sorted by depth, rejecting repeated depths."""
    if values is None:
        return ()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise TypeError("level_overrides must be a list of level overrides")
    overrides = tuple(DendriticLevelOverride.from_config(value) for value in values)
    depths = [override.depth for override in overrides]
    if len(set(depths)) != len(depths):
        raise ValueError(f"level_overrides repeats a depth: {sorted(depths)}")
    return tuple(sorted(overrides, key=lambda override: override.depth))


def overrides_by_depth(
    overrides: Sequence[DendriticLevelOverride], n_branch_layers: int
) -> dict[int, DendriticLevelOverride]:
    """Index overrides by depth after checking every depth exists."""
    result = {}
    for override in overrides:
        if override.depth > n_branch_layers:
            raise ValueError(
                f"level override depth {override.depth} exceeds the terminal "
                f"level's depth ({n_branch_layers})"
            )
        result[override.depth] = override
    return result


def contacts_mask(
    contacts: Sequence[Sequence[int]],
    *,
    rows: int,
    in_features: int,
    n_soma: int,
    name: str,
) -> torch.Tensor:
    """Boolean connection mask ``[rows, in_features]`` from contact lists."""
    per_neuron = rows // n_soma if rows % n_soma == 0 else None
    if len(contacts) not in {rows, per_neuron}:
        raise ValueError(
            f"{name} lists {len(contacts)} compartments; the level has {rows} "
            f"({per_neuron} per neuron)"
        )
    for row in contacts:
        for index in row:
            if not 0 <= index < in_features:
                raise ValueError(
                    f"{name} contains input {index}, outside [0, {in_features - 1}]"
                )
    mask = torch.zeros(rows, in_features, dtype=torch.bool)
    for row in range(rows):
        mask[row, list(contacts[row % len(contacts)])] = True
    return mask


def level_synapse_config(synapse_config: Any, activation: str | None) -> Any:
    """The synapse config of a level with its own activation."""
    if activation is None:
        return synapse_config
    return replace(
        synapse_config,
        reactivation=replace(
            synapse_config.reactivation,
            dendritic_activation=activation.lower(),
            soma_type=None,
        ),
    )


def remove_synaptic_pathway(layer: Any, pathway: str) -> None:
    """Remove a feedforward pathway from a built branch layer.

    The layer is left as if it had been constructed without that pathway:
    no synapse module, no input flag and no pathway entries in its
    ``branch_config``.
    """
    if pathway == "excitatory":
        if getattr(layer, "branch_excitation", None) is None:
            raise ValueError(
                f"level {layer.layer_idx} has no excitatory pathway to remove"
            )
        layer.branch_excitation = None
        layer.input_excitatory = False
        layer.__dict__.pop("excitatory_input_dim", None)
        layer.branch_config = replace(
            layer.branch_config,
            excitatory_input_dim=None,
            excitatory_synapses_per_branch=None,
            excitatory_connection_indices=None,
            excitatory_connection_mask=None,
        )
    elif pathway == "inhibitory":
        if getattr(layer, "branch_inhibition", None) is None:
            raise ValueError(
                f"level {layer.layer_idx} has no inhibitory pathway to remove"
            )
        layer.branch_inhibition = None
        layer.input_inhibitory = False
        layer.__dict__.pop("inhibitory_input_dim", None)
        layer.branch_config = replace(
            layer.branch_config,
            inhibitory_input_dim=None,
            inhibitory_synapses_per_branch=None,
            inhibitory_connection_indices=None,
            inhibitory_connection_mask=None,
        )
    else:
        raise ValueError(f"Unknown pathway {pathway!r}; expected one of {PATHWAYS}")


def declare_contacts(synapse: Any, mask: torch.Tensor) -> None:
    """Use configuration-declared contacts as a synapse's connection mask.

    The mask is kept as a non-persistent buffer: configuration rebuilds it,
    so checkpoints keep the layout of the same model without declared
    contacts.
    """
    if not isinstance(synapse, TopKLinear):
        raise ValueError(
            "declared contacts need a dense synapse layer with a connection mask "
            f"(the TopK family); {type(synapse).__name__} stores its contacts "
            "differently"
        )
    if tuple(mask.shape) != (synapse.out_features, synapse.in_features):
        raise ValueError(
            f"declared contacts have shape {tuple(mask.shape)}, the synapse layer "
            f"{(synapse.out_features, synapse.in_features)}"
        )
    synapse.register_buffer(
        "connection_mask",
        mask.to(device=synapse.pre_w.device, dtype=torch.bool),
        persistent=False,
    )


__all__ = [
    "DendriticLevelOverride",
    "contacts_mask",
    "declare_contacts",
    "level_synapse_config",
    "normalize_level_overrides",
    "overrides_by_depth",
    "remove_synaptic_pathway",
]
