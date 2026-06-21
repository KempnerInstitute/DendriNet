"""Unified recurrent E/I architecture factory helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.model_aliases import warn_alias_conflicts


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of config-like objects into a plain mapping."""

    return to_plain_dict(obj)


def _apply_population_alias_overrides(
    population: dict[str, Any],
    alias_overrides: dict[str, object],
    *,
    core_type: str,
    context: str,
) -> dict[str, Any]:
    """Apply recurrent type-alias semantics to one PopulationConfig payload."""

    if not alias_overrides:
        return population
    updated = dict(population)
    warn_alias_conflicts(core_type, updated, alias_overrides, context=context)
    for key, value in alias_overrides.items():
        updated[key] = value
    return updated


def _coerce_population_config(
    population: Any,
    *,
    population_config_cls: type[Any],
    alias_overrides: dict[str, object],
    core_type: str,
    context: str,
) -> Any:
    """Normalize one optional E/I population config payload."""

    if population is not None and not isinstance(population, dict):
        plain_population = _to_plain_mapping(population)
        if plain_population:
            population = plain_population
    if isinstance(population, dict):
        population = _apply_population_alias_overrides(
            population,
            alias_overrides,
            core_type=core_type,
            context=context,
        )
        return population_config_cls(**population)
    return population


def build_unified_ei_layer_configs(
    *,
    layers: Sequence[Any],
    ei_layer_config_cls: type[Any],
    population_config_cls: type[Any],
    alias_overrides: dict[str, object],
    core_type: str,
) -> list[Any]:
    """Build typed unified E/I layer configs from mappings or config objects."""

    layers_cfg = []
    for layer_idx, layer in enumerate(layers):
        if isinstance(layer, ei_layer_config_cls):
            if not alias_overrides:
                layers_cfg.append(layer)
                continue
            layer = asdict(layer)
        if not isinstance(layer, dict):
            raise ValueError(
                "Each unified E-I layer config must be a dict or EILayerConfig"
            )
        layer = dict(layer)
        exc = _coerce_population_config(
            layer.get("excitatory"),
            population_config_cls=population_config_cls,
            alias_overrides=alias_overrides,
            core_type=core_type,
            context=f"model.core.unified_ei.layers[{layer_idx}].excitatory",
        )
        inh = _coerce_population_config(
            layer.get("inhibitory"),
            population_config_cls=population_config_cls,
            alias_overrides=alias_overrides,
            core_type=core_type,
            context=f"model.core.unified_ei.layers[{layer_idx}].inhibitory",
        )
        if isinstance(exc, population_config_cls):
            layer["excitatory"] = exc
        if isinstance(inh, population_config_cls):
            layer["inhibitory"] = inh
        layers_cfg.append(ei_layer_config_cls(**layer))
    return layers_cfg


__all__ = [
    "_apply_population_alias_overrides",
    "build_unified_ei_layer_configs",
]
