"""Shared named-population configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import (
    deep_merge_dicts,
    has_enabled_synapse_types,
)
from dendritic_modeling.config.keys import apply_renamed_fields, config_from_payload
from dendritic_modeling.config.reactivation import (
    DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY,
    warn_unused_gate_init,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)

# Morphology fields owned by each population definition. Copies placed in
# ``population_defaults`` or a nested ``population`` mapping would be overwritten.
_DEFINITION_ONLY_KEYS = ("n_neurons", "branch_factors")


@dataclass
class PopulationDefinitionConfig(BaseConfig):
    """Configuration for one named population in a population-network layer."""

    name: str = "pyr"
    polarity: str = "excitatory"  # excitatory | inhibitory
    n_neurons: int = 64
    branch_factors: list[int] = field(default_factory=lambda: [1])
    population: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("population name must be non-empty")
        self.polarity = str(self.polarity).lower()
        if self.polarity not in {"excitatory", "inhibitory"}:
            raise ValueError(
                f"polarity must be 'excitatory' or 'inhibitory', got {self.polarity!r}"
            )
        if self.n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0, got {self.n_neurons}")
        for branch_factor in self.branch_factors:
            if int(branch_factor) <= 0:
                raise ValueError(
                    f"branch_factors entries must be positive, got {branch_factor}"
                )


def as_population_definition_config(
    population: PopulationDefinitionConfig | dict[str, Any],
    *,
    field_name: str = "populations",
) -> PopulationDefinitionConfig:
    """Coerce a population payload into ``PopulationDefinitionConfig``."""

    if isinstance(population, PopulationDefinitionConfig):
        return population
    if isinstance(population, dict):
        return config_from_payload(
            PopulationDefinitionConfig, population, section=field_name
        )
    raise TypeError(
        f"{field_name} entries must be PopulationDefinitionConfig or dict, "
        f"got {type(population).__name__}"
    )


def population_level_dims(n_soma: int, branch_factors: list[int]) -> list[int]:
    """Return leaf-to-soma level dimensions for a population morphology."""

    sizes = [int(n_soma)]
    for branch_factor in branch_factors:
        sizes.append(sizes[-1] * int(branch_factor))
    sizes.reverse()
    return sizes


def population_config_from_definition(
    population: PopulationDefinitionConfig,
    defaults: dict[str, Any],
    *,
    layer_idx: int,
    network_layer_idx: int = 0,
) -> PopulationConfig:
    """Merge layer defaults with a population override and build config.

    ``layer_idx`` is the population's position in its layer (its
    ``structured_layer_idx``); ``network_layer_idx`` is the layer's index in
    the network.
    """

    for source, values in (
        ("population_defaults", defaults),
        (f"populations[{population.name!r}].population", population.population),
    ):
        misplaced = [key for key in _DEFINITION_ONLY_KEYS if key in values]
        if misplaced:
            raise ValueError(
                f"{source} sets {misplaced}, but each population definition "
                "always supplies these values; set n_neurons and branch_factors "
                f"on the definition of population {population.name!r} instead"
            )
    payload = deep_merge_dicts(defaults, dict(population.population))
    if (
        "reactivation_init_policy" not in payload
        and payload.get("use_shunting") is False
    ):
        payload["reactivation_init_policy"] = DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY
    payload["n_neurons"] = int(population.n_neurons)
    payload["branch_factors"] = list(population.branch_factors)
    payload.setdefault("structured_layer_idx", layer_idx)
    payload.setdefault("structured_network_layer_idx", network_layer_idx)
    section = (
        f"model.core.population_network.layers[{network_layer_idx}]"
        f".populations[{population.name}]"
    )
    payload = apply_renamed_fields(PopulationConfig, payload, section=section)
    warn_unused_gate_init(
        payload,
        where=(
            "the population settings of "
            f"model.core.population_network.layers[{network_layer_idx}]"
        ),
        policy_key="reactivation_init_policy",
        value_keys=("reactivation_init_m", "reactivation_init_b"),
    )
    return config_from_payload(PopulationConfig, payload, section=section)


def population_requires_temporal_state(pop_config: PopulationConfig) -> bool:
    """Return whether population dynamics need state across sequence timesteps."""

    return (
        str(pop_config.dynamics_mode).lower() == "spike"
        or has_enabled_synapse_types(pop_config.synapse_types)
        or bool(pop_config.dendritic_spikes_enabled)
        or bool(pop_config.soma_feedback_enabled)
    )


def validate_stepped_credit_modes(
    name: str, populations: dict[str, PopulationConfig]
) -> None:
    """All populations in a layer that steps through time need exact gradients."""
    broadcast = sorted(
        key
        for key, pop in populations.items()
        if pop.autograd_credit_mode.lower() != "exact"
    )
    if broadcast:
        raise ValueError(
            f"population_network layer {name!r} steps its populations through time, "
            f"but {broadcast} set autograd_credit_mode='soma_broadcast', which "
            "applies only to feedforward layers. Set autograd_credit_mode to 'exact' "
            "for stepped populations."
        )


__all__ = [
    "PopulationDefinitionConfig",
    "as_population_definition_config",
    "population_config_from_definition",
    "population_level_dims",
    "population_requires_temporal_state",
    "validate_stepped_credit_modes",
]
