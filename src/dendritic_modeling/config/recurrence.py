"""Shared recurrent-core detection for model setup and validation."""

from __future__ import annotations

import warnings
from typing import Any

from dendritic_modeling.config.conversion import (
    deep_merge_dicts,
    has_enabled_synapse_types,
    to_plain_dict,
)

_BASELINE_RNN_TYPES = {"gru", "lstm", "vanilla_rnn", "rnn"}
_HETEROGENEOUS_LEAK_CTRNN_TYPES = {
    "heterogeneous_leak_ctrnn",
    "heterogeneous_ctrnn",
}
_LEGENDRE_MEMORY_TYPES = {"legendre_memory"}
_POPULATION_NETWORK_TYPES = {
    "population_network",
}
_UNIFIED_EI_TYPES = {
    "ei_unified",
    "unified_ei",
    "ei_net",
    "unified_einet",
    "rnn_dendritic_shunting",
    "rnn_dendritic_additive",
    "rnn_dendritic_normalized_additive",
    "rnn_flat_shunting",
    "rnn_flat_additive",
    "rnn_flat_normalized_additive",
}
# model.core.recurrent selected the legacy DendriticRNNCell stack, removed on
# 2026-02-27. load_config retires the key; plain-dict cores fail here.
LEGACY_RECURRENT_ENABLED_MESSAGE = (
    "model.core.recurrent.enabled=true selects the removed legacy recurrent "
    "cell stack; no builder consumes it. Use connectivity.rec_* synapse counts "
    "(EINet types), model.core.unified_ei, or model.core.population_network "
    "with recurrent layers instead."
)

# Core types built from the structured ``architecture``/``connectivity``
# schema, feedforward or recurrent. This is the registry the factory uses.
_STRUCTURED_EI_TYPES = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_normalized_additive",
    "dendritic_mlp",
    "dendritic_signed",
    "flat_shunting",
    "flat_additive",
    "flat_normalized_additive",
    "flat_mlp",
    "flat_signed",
}


def _get_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# Older placements of the structured-EINet feature blocks. ``load_config``
# rejects them; programmatic configs may still use them (with a warning).
_NESTED_FEATURE_BLOCKS = {
    "dendritic_spikes": ("dynamics",),
    "soma_feedback": ("dynamics", "recurrent_ei"),
}


def resolve_feature_block(core_cfg: Any, name: str) -> dict[str, Any]:
    """Return ``model.core.<name>`` for a structured EINet core.

    ``name`` is ``"dendritic_spikes"`` or ``"soma_feedback"``. Programmatic
    configs may still nest the block under ``dynamics`` (and ``soma_feedback``
    under ``recurrent_ei``); a nested block is used, with a deprecation
    warning, when the top-level block is empty. Different non-empty blocks in
    two places raise, because one of them would be ignored.
    """
    top = to_plain_dict(_get_value(core_cfg, name, {}) or {})
    nested: list[tuple[str, dict[str, Any]]] = []
    for parent in _NESTED_FEATURE_BLOCKS[name]:
        container = _get_value(core_cfg, parent, None)
        if parent == "dynamics" and not container:
            container = _get_value(core_cfg, "spiking", None)
        block = to_plain_dict(to_plain_dict(container or {}).get(name, {}) or {})
        if block:
            nested.append((f"model.core.{parent}.{name}", block))
    if not nested:
        return top
    reference_path, reference = (f"model.core.{name}", top) if top else nested[0]
    for path, block in nested:
        if block != reference:
            raise ValueError(
                f"{path} conflicts with {reference_path}; set the block once, "
                f"as model.core.{name}"
            )
    if not top:
        warnings.warn(
            f"{reference_path} is deprecated; set model.core.{name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return reference


def _core_type(core_cfg: Any) -> str:
    return str(_get_value(core_cfg, "type", "") or "").lower()


def _is_positive(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_is_positive(item) for item in value)
    if value is None:
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _any_positive_in(mapping: Any, keys: tuple[str, ...]) -> bool:
    for key in keys:
        if _is_positive(_get_value(mapping, key, 0)):
            return True
    return False


def _legacy_recurrent_enabled(core_cfg: Any) -> bool:
    rec = _get_value(core_cfg, "recurrent", {})
    if isinstance(rec, dict):
        enabled = bool(rec.get("enabled", False))
    else:
        enabled = bool(getattr(rec, "enabled", False))
    if enabled:
        raise ValueError(LEGACY_RECURRENT_ENABLED_MESSAGE)
    return False


def _has_unified_recurrent_layers(core_cfg: Any) -> bool:
    unified = _get_value(core_cfg, "unified_ei", None)
    if unified is None:
        unified = _get_value(core_cfg, "ei_unified", {})
    unified = to_plain_dict(unified)
    if not unified:
        return False
    if bool(unified.get("recurrent", False)):
        return True
    for layer in unified.get("layers", []) or []:
        if isinstance(layer, dict):
            if bool(layer.get("recurrent", False)):
                return True
            if _is_positive(layer.get("rec_synapses", 0)):
                return True
            if _is_positive(layer.get("rec_excitatory_synapses", 0)):
                return True
            if _is_positive(layer.get("rec_inhibitory_synapses", 0)):
                return True
        elif bool(getattr(layer, "recurrent", False)):
            return True
    return False


def _has_structured_einet_recurrence(core_cfg: Any) -> bool:
    conn = _get_value(core_cfg, "connectivity", {})
    if _any_positive_in(
        conn,
        (
            "rec_ee_synapses_per_branch",
            "rec_ie_synapses_per_branch",
            "rec_ei_synapses_per_branch",
            "rec_ii_synapses_per_branch",
        ),
    ):
        return True

    dynamics = to_plain_dict(_get_value(core_cfg, "dynamics", {}))
    soma_feedback = resolve_feature_block(core_cfg, "soma_feedback")
    if bool(
        soma_feedback.get(
            "enabled",
            dynamics.get("soma_feedback_enabled", False),
        )
    ):
        return True
    if (
        str(dynamics.get("mode", dynamics.get("dynamics_mode", "rate"))).lower()
        == "spike"
    ):
        return True

    dendritic_spikes = resolve_feature_block(core_cfg, "dendritic_spikes")
    if bool(
        dendritic_spikes.get(
            "enabled",
            dynamics.get("dendritic_spikes_enabled", False),
        )
    ):
        return True
    if has_enabled_synapse_types(_get_value(core_cfg, "synapse_types", {})):
        return True

    recurrent_ei = _get_value(core_cfg, "recurrent_ei", {})
    if not recurrent_ei:
        return False

    enabled = bool(_get_value(recurrent_ei, "enabled", False))
    if enabled:
        per_layer = _get_value(recurrent_ei, "recurrent_layers", None)
        if per_layer is None:
            return True
        if isinstance(per_layer, list):
            return any(bool(v) for v in per_layer)
        return bool(per_layer)

    return _any_positive_in(
        recurrent_ei,
        (
            "rec_excitatory_synapses",
            "rec_inhibitory_synapses",
            "rec_excitatory_synapses_per_layer",
            "rec_inhibitory_synapses_per_layer",
        ),
    )


def _population_has_temporal_state(population: dict[str, Any]) -> bool:
    """Return whether a population needs state across sequence timesteps."""
    if str(population.get("dynamics_mode", "rate")).lower() == "spike":
        return True
    if has_enabled_synapse_types(population.get("synapse_types", {})):
        return True
    if bool(population.get("dendritic_spikes_enabled", False)):
        return True
    if bool(population.get("soma_feedback_enabled", False)):
        return True
    return False


def _population_network_requires_recurrent_wrapper(core_cfg: Any) -> bool:
    mp = to_plain_dict(_get_value(core_cfg, "population_network", {}))
    if not mp:
        mp = to_plain_dict(core_cfg)
    output_mode = str(mp.get("output_mode", "last")).lower()
    if output_mode in {"all", "mean"}:
        return True
    for layer in mp.get("layers", []) or []:
        layer_payload = to_plain_dict(layer)
        if bool(layer_payload.get("recurrent", False)):
            return True
        defaults = to_plain_dict(layer_payload.get("population_defaults", {}))
        if _population_has_temporal_state(defaults):
            return True
        for cell in layer_payload.get("populations", []) or []:
            cell_payload = to_plain_dict(cell)
            population = deep_merge_dicts(
                defaults,
                to_plain_dict(cell_payload.get("population", {})),
            )
            if _population_has_temporal_state(population):
                return True
        for conn in layer_payload.get("connections", []) or []:
            conn_payload = to_plain_dict(conn)
            if str(conn_payload.get("timing", "same_step")).lower() == "delayed":
                return True
    return False


def is_recurrent_core_config(core_cfg: Any) -> bool:
    """Detect whether a model core config should use recurrent wrappers."""
    core_type = _core_type(core_cfg)
    if core_type in _BASELINE_RNN_TYPES:
        return True
    if core_type in _HETEROGENEOUS_LEAK_CTRNN_TYPES:
        return True
    if core_type in _LEGENDRE_MEMORY_TYPES:
        return True
    if core_type in _POPULATION_NETWORK_TYPES:
        return _population_network_requires_recurrent_wrapper(core_cfg)
    if core_type in _UNIFIED_EI_TYPES:
        return _has_unified_recurrent_layers(core_cfg)
    if core_type in _STRUCTURED_EI_TYPES:
        return _has_structured_einet_recurrence(core_cfg) or _legacy_recurrent_enabled(
            core_cfg
        )
    return _legacy_recurrent_enabled(core_cfg)


__all__ = ["is_recurrent_core_config", "resolve_feature_block"]
