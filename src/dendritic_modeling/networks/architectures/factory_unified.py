"""Unified recurrent E/I architecture factory helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, fields, is_dataclass
from typing import Any

from dendritic_modeling.config.conversion import (
    set_derived_input_dim,
    to_plain_dict as _to_plain_mapping,
)
from dendritic_modeling.config.keys import apply_renamed_fields, config_from_payload
from dendritic_modeling.config.legacy import canonicalize_unified_ei_config
from dendritic_modeling.config.model_aliases import (
    get_unified_ei_alias_overrides,
    warn_alias_conflicts,
)


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
    initialization_seed: int | None = None,
    initialization_namespace: str = "",
) -> Any:
    """Normalize one optional E/I population config payload."""

    if population is not None and not isinstance(population, dict):
        plain_population = _to_plain_mapping(population)
        if plain_population:
            population = plain_population
    if isinstance(population, dict):
        population = dict(population)
        if initialization_seed is not None:
            population.setdefault("initialization_seed", initialization_seed)
        if not population.get("initialization_namespace"):
            population["initialization_namespace"] = initialization_namespace
        population = _apply_population_alias_overrides(
            population,
            alias_overrides,
            core_type=core_type,
            context=context,
        )
        population = apply_renamed_fields(
            population_config_cls, population, section=context
        )
        return config_from_payload(population_config_cls, population, section=context)
    return population


def build_unified_ei_layer_configs(
    *,
    layers: Sequence[Any],
    ei_layer_config_cls: type[Any],
    population_config_cls: type[Any],
    alias_overrides: dict[str, object],
    core_type: str,
    initialization_seed: int | None = None,
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
        if "inhibitory" not in layer:
            # EILayerConfig's default would add a 16-cell inhibitory population.
            raise ValueError(
                f"model.core.unified_ei.layers[{layer_idx}] does not say whether "
                "the layer has an inhibitory population. Write 'inhibitory:' with "
                "a population mapping, or 'inhibitory: null' for none (omitting "
                "it built a default 16-cell population before 2026-10-08)."
            )
        layer = dict(layer)
        exc = _coerce_population_config(
            layer.get("excitatory"),
            population_config_cls=population_config_cls,
            alias_overrides=alias_overrides,
            core_type=core_type,
            context=f"model.core.unified_ei.layers[{layer_idx}].excitatory",
            initialization_seed=initialization_seed,
            initialization_namespace=f"unified.layer.{layer_idx}.excitatory",
        )
        inh = _coerce_population_config(
            layer.get("inhibitory"),
            population_config_cls=population_config_cls,
            alias_overrides=alias_overrides,
            core_type=core_type,
            context=f"model.core.unified_ei.layers[{layer_idx}].inhibitory",
            initialization_seed=initialization_seed,
            initialization_namespace=f"unified.layer.{layer_idx}.inhibitory",
        )
        if isinstance(exc, population_config_cls):
            layer["excitatory"] = exc
        if isinstance(inh, population_config_cls):
            layer["inhibitory"] = inh
        layers_cfg.append(
            config_from_payload(
                ei_layer_config_cls,
                layer,
                section=f"model.core.unified_ei.layers[{layer_idx}]",
            )
        )
    return layers_cfg


# Synapse counts of a population whose source may be missing, and the
# per-level list that replaces each when it is written.
_SOURCED_COUNTS = {
    "ff_inhibitory_synapses": "ff_inhibitory_synapses_by_level",
    "rec_inhibitory_synapses": "rec_inhibitory_synapses_by_level",
}


def _written_counts(population: Any) -> dict[str, Any]:
    """The source-dependent synapse counts a population payload writes.

    For a mapping these are its keys; for a typed population, the fields
    whose value differs from the default (an unset count cannot be told from
    the default there).
    """
    names = set(_SOURCED_COUNTS) | set(_SOURCED_COUNTS.values())
    if isinstance(population, dict):
        return {key: population[key] for key in names if key in population}
    if is_dataclass(population):
        defaults = type(population)()
        return {
            item.name: getattr(population, item.name)
            for item in fields(population)
            if item.name in names
            and getattr(population, item.name) != getattr(defaults, item.name)
        }
    return {}


def _positive_counts(written: dict[str, Any]) -> list[str]:
    """Written count keys that build synapses (a per-level list replaces its count)."""
    positive = []
    for count, by_level in _SOURCED_COUNTS.items():
        if written.get(by_level):
            if any(int(value) > 0 for value in written[by_level]):
                positive.append(by_level)
        elif int(written.get(count) or 0) > 0:
            positive.append(count)
    return positive


def _layer_fact(layer: Any, name: str, default: Any) -> Any:
    if isinstance(layer, dict):
        return layer.get(name, default)
    return getattr(layer, name, default)


def unsourced_unified_counts(
    layers: Sequence[Any], *, use_transfer: bool
) -> list[tuple[int, str, str, str]]:
    """Written inhibitory synapse counts of unified E/I layers without an input.

    ``layers`` are the layer payloads as written: mappings, whose keys show
    what was written, or typed layer configs, whose non-default counts count
    as written (an omitted count cannot be told from the default there). The
    network wires each layer's feedforward inhibitory input from the
    transferred stream (first layer, or any layer with
    ``direct_ff_inhibitory_to_excitatory``) or from the previous layer's
    inhibitory population; excitatory cells read their own layer's inhibitory
    population, or that input directly when the layer has none and sets
    ``direct_ff_inhibitory_to_excitatory``. Recurrent counts of non-recurrent
    layers are rejected by the layer itself.

    Returns ``(layer index, population, key, reason)`` for every count key
    above 0 that would build no synapses.
    """
    found = []
    previous_inhibitory = False
    for index, layer in enumerate(layers):
        direct = bool(_layer_fact(layer, "direct_ff_inhibitory_to_excitatory", False))
        recurrent = bool(_layer_fact(layer, "recurrent", False))
        has_inhibitory = _layer_fact(layer, "inhibitory", None) is not None
        layer_input = (use_transfer and (direct or index == 0)) or (
            index > 0 and previous_inhibitory
        )
        no_input = (
            "use_transfer is false"
            if index == 0
            else f"layer {index - 1} has no inhibitory population"
        )
        missing = {
            "inhibitory": {
                "ff_inhibitory_synapses": None if layer_input else no_input,
            },
            "excitatory": {
                "ff_inhibitory_synapses": (
                    None
                    if has_inhibitory or (direct and layer_input)
                    else "the layer has no inhibitory population"
                    + (
                        ""
                        if direct
                        else " and direct_ff_inhibitory_to_excitatory is false"
                    )
                ),
                "rec_inhibitory_synapses": (
                    None
                    if has_inhibitory or not recurrent
                    else "the layer has no inhibitory population"
                ),
            },
        }
        for role, counts in missing.items():
            population = _layer_fact(layer, role, None)
            if population is None:
                continue
            written = _written_counts(population)
            for count, reason in counts.items():
                if reason is None:
                    continue
                by_level = _SOURCED_COUNTS[count]
                for key in _positive_counts(
                    {k: written[k] for k in (count, by_level) if k in written}
                ):
                    found.append((index, role, key, reason))
        previous_inhibitory = has_inhibitory
    return found


def _reject_unsourced_unified_counts(
    raw_layers: Sequence[Any], *, use_transfer: bool
) -> None:
    """Fail on written inhibitory synapse counts whose input does not exist."""
    problems = []
    for index, role, key, reason in unsourced_unified_counts(
        raw_layers, use_transfer=use_transfer
    ):
        value = _layer_fact(_layer_fact(raw_layers[index], role, None), key, None)
        problems.append(
            f"layers[{index}].{role}.{key}={value!r}: {reason}, so these synapses "
            "would have no input"
        )
    if problems:
        raise ValueError(
            "model.core.unified_ei sets synapse counts without a source; set them "
            "to 0 or add the source:\n- " + "\n- ".join(problems)
        )


def _build_unified_ei_architecture(
    type: str,
    parameters: dict[str, Any],
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> Any:
    """Build the unified recurrent E/I architecture family."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        EILayerConfig,
        EINetwork,
        EINetworkConfig,
        PopulationConfig,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating unified E-I architecture (type='{type}')"
        )

    raw_params = _to_plain_mapping(parameters)
    unified_cfg = raw_params.get("unified_ei", raw_params.get("ei_unified", {}))
    if not unified_cfg and "layers" in raw_params:
        # Allow passing EINetworkConfig-shaped dict directly. Core-level keys
        # are consumed here, not by the unified_ei section validator.
        unified_cfg = {
            key: value
            for key, value in raw_params.items()
            if key not in {"type", "initialization_seed", "biological_neuron"}
        }

    if not isinstance(unified_cfg, dict):
        raise ValueError(
            "Unified E-I config must be a mapping, got "
            f"{unified_cfg.__class__.__name__}"
        )

    unified_cfg = canonicalize_unified_ei_config(unified_cfg)
    set_derived_input_dim(unified_cfg, input_dim, section="model.core.unified_ei")

    alias_overrides = get_unified_ei_alias_overrides(type)
    layers_cfg = build_unified_ei_layer_configs(
        layers=unified_cfg.get("layers", []),
        ei_layer_config_cls=EILayerConfig,
        population_config_cls=PopulationConfig,
        alias_overrides=alias_overrides,
        core_type=type,
        initialization_seed=raw_params.get("initialization_seed"),
    )

    if not layers_cfg:
        raise ValueError(
            "Unified E-I architecture requires at least one layer in unified_ei.layers"
        )
    _reject_unsourced_unified_counts(
        unified_cfg.get("layers", []),
        use_transfer=bool(unified_cfg.get("use_transfer", False)),
    )

    unified_cfg["layers"] = layers_cfg
    return EINetwork(
        config_from_payload(
            EINetworkConfig, unified_cfg, section="model.core.unified_ei"
        )
    )


__all__ = [
    "_apply_population_alias_overrides",
    "_build_unified_ei_architecture",
    "build_unified_ei_layer_configs",
    "unsourced_unified_counts",
]
