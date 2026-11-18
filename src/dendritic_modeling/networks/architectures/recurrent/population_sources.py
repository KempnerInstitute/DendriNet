"""Population-network source normalization and stream helpers."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.config.keys import config_from_payload
from dendritic_modeling.config.population_routing import (
    canonical_input_source as _canonical_external_source,
)
from dendritic_modeling.networks.architectures.recurrent.population_configs import (
    PopulationProjectionConfig,
)


def as_population_projection_config(
    connection: PopulationProjectionConfig | dict[str, Any],
) -> PopulationProjectionConfig:
    if isinstance(connection, PopulationProjectionConfig):
        return connection
    if isinstance(connection, dict):
        payload = dict(connection)
        if "probability" not in payload and "p_connect" in payload:
            payload["probability"] = payload.pop("p_connect")
        return config_from_payload(
            PopulationProjectionConfig,
            payload,
            section="model.core.population_network connection",
        )
    raise TypeError(
        "connections entries must be PopulationProjectionConfig or dict, "
        f"got {type(connection).__name__}"
    )


def _concat_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return torch.cat(values, dim=-1)


_as_projection_config = as_population_projection_config


def _split_qualified_source(source: str) -> tuple[str, str] | None:
    if "." not in source:
        return None
    layer_name, population_name = source.split(".", 1)
    if not layer_name or not population_name:
        raise ValueError(
            "qualified population sources must use 'layer.population', "
            f"got {source!r}"
        )
    return layer_name, population_name


__all__ = [
    "_as_projection_config",
    "_canonical_external_source",
    "_concat_or_none",
    "_split_qualified_source",
    "as_population_projection_config",
]
