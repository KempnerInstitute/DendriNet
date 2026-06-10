"""Shared named-population configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import (
    deep_merge_dicts,
    has_enabled_synapse_types,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)


@dataclass
class CellTypeConfig(BaseConfig):
    """Configuration for one named cell type/population."""

    name: str = "pyr"
    polarity: str = "excitatory"  # excitatory | inhibitory
    n_neurons: int = 64
    branch_factors: list[int] = field(default_factory=lambda: [1])
    population: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("cell type name must be non-empty")
        self.polarity = str(self.polarity).lower()
        if self.polarity not in {"excitatory", "inhibitory"}:
            raise ValueError(
                f"polarity must be 'excitatory' or 'inhibitory', got {self.polarity!r}"
            )
        if self.n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0, got {self.n_neurons}")
        if not self.branch_factors:
            raise ValueError("branch_factors must be non-empty")


def as_cell_type_config(
    cell_type: CellTypeConfig | dict[str, Any],
    *,
    field_name: str = "populations",
) -> CellTypeConfig:
    """Coerce a population/cell-type payload into ``CellTypeConfig``."""

    if isinstance(cell_type, CellTypeConfig):
        return cell_type
    if isinstance(cell_type, dict):
        return CellTypeConfig(**cell_type)
    raise TypeError(
        f"{field_name} entries must be CellTypeConfig or dict, "
        f"got {type(cell_type).__name__}"
    )


def population_level_dims(n_soma: int, branch_factors: list[int]) -> list[int]:
    """Return leaf-to-soma level dimensions for a population morphology."""

    sizes = [int(n_soma)]
    for branch_factor in branch_factors:
        sizes.append(sizes[-1] * int(branch_factor))
    sizes.reverse()
    return sizes


def population_config_from_cell(
    cell: CellTypeConfig,
    defaults: dict[str, Any],
    *,
    layer_idx: int,
) -> PopulationConfig:
    """Merge layer defaults with a cell override and build ``PopulationConfig``."""

    payload = deep_merge_dicts(defaults, dict(cell.population))
    payload["n_neurons"] = int(cell.n_neurons)
    payload["branch_factors"] = list(cell.branch_factors)
    payload.setdefault("structured_layer_idx", layer_idx)
    return PopulationConfig(**payload)


def population_requires_temporal_state(pop_config: PopulationConfig) -> bool:
    """Return whether population dynamics need state across sequence timesteps."""

    return (
        str(pop_config.dynamics_mode).lower() == "spike"
        or has_enabled_synapse_types(pop_config.synapse_types)
        or bool(pop_config.dendritic_spikes_enabled)
        or bool(pop_config.soma_feedback_enabled)
    )


__all__ = [
    "CellTypeConfig",
    "as_cell_type_config",
    "population_config_from_cell",
    "population_level_dims",
    "population_requires_temporal_state",
]
