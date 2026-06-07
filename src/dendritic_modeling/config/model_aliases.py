"""Shared semantics for public model type aliases."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from dataclasses import asdict as dataclass_asdict, is_dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

RECURRENT_ADDITIVE_REACTIVATION_INIT_M = 1.0

_CORE_MORPHOLOGY_ALIAS_OVERRIDES: dict[str, dict[str, object]] = {
    "dendritic_shunting": {"use_shunting": True},
    "flat_shunting": {"use_shunting": True},
    "dendritic_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
    },
    "flat_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
    },
    "dendritic_mlp": {"use_shunting": False},
    "flat_mlp": {"use_shunting": False},
    "mlp": {"use_shunting": False},
}

_UNIFIED_EI_ALIAS_OVERRIDES: dict[str, dict[str, object]] = {
    "rnn_dendritic_shunting": {
        "use_shunting": True,
    },
    "rnn_dendritic_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
        "reactivation_init_m": RECURRENT_ADDITIVE_REACTIVATION_INIT_M,
        "reactivation_init_b": 0.0,
        "reactivation_init_policy": "fixed",
    },
    "rnn_flat_shunting": {
        "use_shunting": True,
        "branch_factors": [1],
    },
    "rnn_flat_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
        "reactivation_init_m": RECURRENT_ADDITIVE_REACTIVATION_INIT_M,
        "reactivation_init_b": 0.0,
        "reactivation_init_policy": "fixed",
        "branch_factors": [1],
    },
}

_STRUCTURED_RECURRENT_ALIAS_MAP: dict[str, str] = {
    "dendritic_additive": "rnn_dendritic_additive",
}


def get_core_morphology_alias_overrides(core_type: str) -> dict[str, object]:
    """Return feedforward/core morphology flags implied by ``core_type``."""
    return copy.deepcopy(
        _CORE_MORPHOLOGY_ALIAS_OVERRIDES.get(str(core_type).lower(), {})
    )


def get_unified_ei_alias_overrides(core_type: str) -> dict[str, object]:
    """Return recurrent unified-EI type-alias overrides for ``core_type``."""
    return copy.deepcopy(_UNIFIED_EI_ALIAS_OVERRIDES.get(str(core_type).lower(), {}))


def get_structured_recurrent_alias_overrides(core_type: str) -> dict[str, object]:
    """Return recurrent structured-EINet overrides for legacy aliases."""
    alias = _STRUCTURED_RECURRENT_ALIAS_MAP.get(str(core_type).lower())
    if alias is None:
        return {}
    return get_unified_ei_alias_overrides(alias)


def warn_alias_conflicts(
    core_type: str,
    explicit_values: Mapping[str, object],
    alias_overrides: Mapping[str, object],
    *,
    context: str,
    skip_keys: set[str] | frozenset[str] | None = None,
) -> None:
    """Warn when an explicit config field conflicts with a type-alias contract."""
    if not alias_overrides:
        return
    skip = set(skip_keys or ())
    for key, alias_value in alias_overrides.items():
        if key in skip or key not in explicit_values:
            continue
        explicit_value = explicit_values[key]
        if explicit_value is None or explicit_value == alias_value:
            continue
        warnings.warn(
            f"{context}: core type {str(core_type)!r} overrides {key}="
            f"{explicit_value!r} with alias value {alias_value!r}. Remove the "
            "conflicting field or use a generic explicit type such as 'einet' "
            "or 'ei_unified'.",
            UserWarning,
            stacklevel=2,
        )


def _copy_config_payload(config: Any) -> dict[str, Any]:
    """Copy a config-like payload into a mutable plain dict."""
    if isinstance(config, DictConfig):
        result = OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        result = dataclass_asdict(config)
    else:
        result = copy.deepcopy(config)
    return result if isinstance(result, dict) else {}


def _has_recurrent_connectivity(core: dict[str, Any]) -> bool:
    connectivity = core.get("connectivity", {})
    if not isinstance(connectivity, dict):
        return False
    for key in (
        "rec_ee_synapses_per_branch",
        "rec_ie_synapses_per_branch",
        "rec_ei_synapses_per_branch",
        "rec_ii_synapses_per_branch",
    ):
        value = connectivity.get(key, 0)
        if isinstance(value, (list, tuple)):
            if any(float(v or 0) > 0 for v in value):
                return True
        elif float(value or 0) > 0:
            return True
    recurrent_ei = core.get("recurrent_ei", {})
    return isinstance(recurrent_ei, dict) and bool(recurrent_ei.get("enabled", False))


def canonicalize_model_core_flags(config: Any) -> dict[str, Any]:
    """Return a config dict with feedforward/core type aliases made explicit.

    This is the config-layer source of truth for public aliases such as
    ``dendritic_shunting`` and ``dendritic_additive``.  Script utilities keep
    compatibility wrappers that delegate here.
    """
    result = _copy_config_payload(config)
    if not result:
        return {}

    model_cfg = result.get("model")
    if isinstance(model_cfg, dict) and isinstance(model_cfg.get("core"), dict):
        core_cfg = model_cfg["core"]
    elif {"architecture", "connectivity"} <= set(result.keys()):
        # Also support direct core-config dicts passed straight to the factory.
        core_cfg = result
    else:
        return result

    core_type = str(core_cfg.get("type", "")).lower()
    morphology_cfg = core_cfg.setdefault("morphology", {})
    if not isinstance(morphology_cfg, dict):
        morphology_cfg = {}
        core_cfg["morphology"] = morphology_cfg

    overrides = get_core_morphology_alias_overrides(core_type)
    if overrides:
        warn_alias_conflicts(
            core_type,
            morphology_cfg,
            overrides,
            context="model.core.morphology",
        )
        morphology_cfg.update(overrides)

    # Feedforward additive aliases historically needed an additive-safe fixed
    # gate bootstrap only when no explicit initialization policy was present.
    if core_type == "dendritic_additive" and _has_recurrent_connectivity(core_cfg):
        reactivation_cfg = core_cfg.setdefault("reactivation", {})
        if not isinstance(reactivation_cfg, dict):
            reactivation_cfg = {}
            core_cfg["reactivation"] = reactivation_cfg
        recurrent_additive_reactivation = {
            "init_m": RECURRENT_ADDITIVE_REACTIVATION_INIT_M,
            "init_b": 0.0,
            "init_policy": "fixed",
        }
        warn_alias_conflicts(
            core_type,
            reactivation_cfg,
            recurrent_additive_reactivation,
            context="model.core.reactivation",
        )
        morphology_cfg["use_additive_normalization"] = True
        reactivation_cfg.update(recurrent_additive_reactivation)
    elif core_type in {"dendritic_additive", "flat_additive"}:
        reactivation_cfg = core_cfg.setdefault("reactivation", {})
        if not isinstance(reactivation_cfg, dict):
            reactivation_cfg = {}
            core_cfg["reactivation"] = reactivation_cfg
        if not reactivation_cfg.get("init_policy"):
            reactivation_cfg["init_m"] = 0.1
            reactivation_cfg["init_b"] = 0.0
            reactivation_cfg["init_policy"] = "fixed"

    return result


def canonicalize_rnn_core_flags(config: Any) -> dict[str, Any]:
    """Return a config dict with recurrent unified-EI aliases made explicit."""
    result = _copy_config_payload(config)
    if not result:
        return {}

    model_cfg = result.get("model")
    if not isinstance(model_cfg, dict):
        return result

    core_cfg = model_cfg.get("core")
    if not isinstance(core_cfg, dict):
        return result

    core_type = str(core_cfg.get("type", "")).lower()
    overrides = get_unified_ei_alias_overrides(core_type)
    if not overrides:
        return result

    unified_cfg = core_cfg.get("unified_ei") or core_cfg.get("ei_unified")
    if not isinstance(unified_cfg, dict):
        return result
    layers = unified_cfg.get("layers")
    if not isinstance(layers, list):
        return result

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for population_key in ("excitatory", "inhibitory"):
            population = layer.get(population_key)
            if not isinstance(population, dict):
                continue
            warn_alias_conflicts(
                core_type,
                population,
                overrides,
                context=f"model.core.unified_ei.layers[].{population_key}",
            )
            population.update(overrides)

    return result


__all__ = [
    "RECURRENT_ADDITIVE_REACTIVATION_INIT_M",
    "canonicalize_model_core_flags",
    "canonicalize_rnn_core_flags",
    "get_core_morphology_alias_overrides",
    "get_structured_recurrent_alias_overrides",
    "get_unified_ei_alias_overrides",
    "warn_alias_conflicts",
]
