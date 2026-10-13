"""Compatibility adapters for deprecated config spellings.

Active builders should consume canonical config keys.  Deprecated public YAML
spellings are normalized here so compatibility stays explicit and contained.
"""

from __future__ import annotations

import copy
import difflib
import logging
from collections.abc import Mapping
from typing import Any

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.inhibitory_wiring import strip_retired_routing_keys
from dendritic_modeling.config.keys import field_default

logger = logging.getLogger(__name__)


def normalize_transfer_config(transfer: Any) -> dict[str, Any]:
    """Return transfer params without the retired inhibitory-routing keys.

    Structured EINet builders check those keys against the rest of the core
    first (``inhibitory_wiring.check_retired_inhibitory_routing``).
    """
    return strip_retired_routing_keys(to_plain_dict(transfer))


def _is_legacy_unified_ei_layer(layer: dict[str, Any]) -> bool:
    """Return True for the pre-population unified-EI layer shorthand."""
    return any(
        key in layer
        for key in (
            "tau",
            "ff_synapses",
            "rec_synapses",
            "ie_synapses",
            "compartment",
        )
    )


def _legacy_branch_factors(layer: dict[str, Any], *, inhibitory: bool) -> list[int]:
    """Translate legacy recurrent compartment names to a population morphology."""
    key = "inhibitory_branch_factors" if inhibitory else "branch_factors"
    factors = layer.get(key)
    if factors is not None:
        return list(factors)
    return [1]


def _canonicalize_legacy_unified_ei_layer(
    layer: dict[str, Any],
    unified_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Convert old recurrent layer shorthand to explicit population configs."""
    ff_synapses = int(layer.get("ff_synapses", layer.get("ff_excitatory_synapses", 0)))
    rec_synapses = int(
        layer.get("rec_synapses", layer.get("rec_excitatory_synapses", 0))
    )
    ie_synapses = int(layer.get("ie_synapses", layer.get("ff_inhibitory_synapses", 0)))

    if (
        ff_synapses == 0
        and rec_synapses > 0
        and not bool(unified_cfg.get("allow_recurrent_only_layers", False))
    ):
        # A zero-initialized recurrent-only layer has no upstream drive and can
        # make lower layers/input projections inert.  Keep the migration adapter
        # behavior aligned with the canonical recurrent examples.
        ff_synapses = int(unified_cfg.get("legacy_recurrent_only_ff_synapses", 2))

    tau = float(layer.get("tau", unified_cfg.get("tau_base", 50.0)))
    i_tau = float(unified_cfg.get("i_tau", tau))
    learnable_tau = bool(
        unified_cfg.get("learnable_tau", unified_cfg.get("learn_taus", False))
    )
    soma_mode = str(unified_cfg.get("soma_mode", layer.get("soma_mode", ""))).lower()
    use_shunting = bool(layer.get("use_shunting", soma_mode not in {"additive", "sum"}))
    recurrent = bool(
        layer.get(
            "recurrent",
            rec_synapses > 0 or bool(unified_cfg.get("recurrent", False)),
        )
    )

    base_population = {
        "branch_factors": _legacy_branch_factors(layer, inhibitory=False),
        "ff_excitatory_synapses": ff_synapses,
        "ff_inhibitory_synapses": ie_synapses,
        "rec_excitatory_synapses": rec_synapses,
        "rec_inhibitory_synapses": ie_synapses,
        "use_shunting": use_shunting,
        "use_additive_normalization": soma_mode in {"additive", "sum"},
        "reactivate": bool(layer.get("reactivate", True)),
        "reactivation_type": layer.get("reactivation_type", "param_tanh"),
        "weight_transform": str(layer.get("weight_transform", "softplus")),
        "tau_base": tau,
        "tau_ratio": float(layer.get("tau_ratio", 1.0)),
        "learnable_tau": learnable_tau,
        "somatic_synapses": bool(layer.get("somatic_synapses", True)),
    }

    n_inhibitory = int(unified_cfg.get("n_inhibitory", 16) or 0)
    inhibitory = None
    if n_inhibitory > 0 and (ff_synapses > 0 or rec_synapses > 0):
        inhibitory = {
            **base_population,
            "n_neurons": n_inhibitory,
            "branch_factors": _legacy_branch_factors(layer, inhibitory=True),
            "ff_excitatory_synapses": ff_synapses,
            "ff_inhibitory_synapses": 0,
            "rec_excitatory_synapses": rec_synapses,
            "rec_inhibitory_synapses": 0,
            "tau_base": i_tau,
        }

    canonical = {
        key: value
        for key, value in layer.items()
        if key
        not in {
            "tau",
            "tau_ratio",
            "ff_synapses",
            "rec_synapses",
            "ie_synapses",
            "compartment",
            "use_shunting",
            "reactivate",
            "reactivation_type",
            "weight_transform",
            "somatic_synapses",
        }
    }
    canonical.update(
        {
            "recurrent": recurrent,
            "dt": float(layer.get("dt", unified_cfg.get("dt", 1.0))),
            "excitatory": {
                **base_population,
                "n_neurons": int(unified_cfg.get("n_excitatory", 64)),
            },
            "inhibitory": inhibitory,
        }
    )
    return canonical


def canonicalize_unified_ei_config(unified_cfg: Any) -> dict[str, Any]:
    """Normalize old/new unified-EI config spellings before dataclass build."""
    canonical = to_plain_dict(unified_cfg)
    raw_keys = set(canonical)
    if (
        "input_projection_dims" not in canonical
        and "input_projection_layers" in canonical
    ):
        logger.warning(
            "unified_ei.input_projection_layers is deprecated; use "
            "unified_ei.input_projection_dims instead."
        )
        canonical["input_projection_dims"] = list(canonical["input_projection_layers"])

    if "transfer_params" in canonical:
        canonical["transfer_params"] = normalize_transfer_config(
            canonical["transfer_params"]
        )

    canonical_layers = []
    uses_legacy_layers = False
    for layer in canonical.get("layers", []) or []:
        layer_payload = to_plain_dict(layer)
        if layer_payload and _is_legacy_unified_ei_layer(layer_payload):
            uses_legacy_layers = True
            logger.warning(
                "Legacy unified_ei.layers[] shorthand is deprecated; use explicit "
                "unified_ei.layers[].excitatory / inhibitory population blocks."
            )
            canonical_layers.append(
                _canonicalize_legacy_unified_ei_layer(layer_payload, canonical)
            )
        else:
            canonical_layers.append(layer)
    canonical["layers"] = canonical_layers

    _reject_unused_unified_ei_keys(
        canonical, raw_keys=raw_keys, uses_legacy_layers=uses_legacy_layers
    )
    return {
        key: value
        for key, value in canonical.items()
        if key in _UNIFIED_EI_CANONICAL_KEYS
    }


_UNIFIED_EI_CANONICAL_KEYS = frozenset(
    {
        "layers",
        "input_dim",
        "use_transfer",
        "transfer_params",
        "input_projection_dims",
        "output_mode",
        "store_routing",
    }
)
# Top-level keys read only while expanding the legacy layers[] shorthand.
_UNIFIED_EI_LEGACY_LAYER_KEYS = frozenset(
    {
        "n_excitatory",
        "n_inhibitory",
        "tau_base",
        "i_tau",
        "learnable_tau",
        "learn_taus",
        "soma_mode",
        "recurrent",
        "dt",
        "allow_recurrent_only_layers",
        "legacy_recurrent_only_ff_synapses",
    }
)


def _reject_unused_unified_ei_keys(
    canonical: dict[str, Any],
    *,
    raw_keys: set[str],
    uses_legacy_layers: bool,
) -> None:
    """Fail on top-level unified_ei keys that the builder would drop."""
    unknown = sorted(
        key
        for key in raw_keys
        if key not in _UNIFIED_EI_CANONICAL_KEYS
        and key not in _UNIFIED_EI_LEGACY_LAYER_KEYS
        and key != "input_projection_layers"
    )
    if unknown:
        known = sorted(
            _UNIFIED_EI_CANONICAL_KEYS
            | _UNIFIED_EI_LEGACY_LAYER_KEYS
            | {"input_projection_layers"}
        )
        hints = {key: difflib.get_close_matches(key, known, n=1) for key in unknown}
        raise ValueError(
            "Unknown model.core.unified_ei key(s): "
            + ", ".join(
                f"{key}" + (f" (did you mean {hints[key][0]!r}?)" if hints[key] else "")
                for key in unknown
            )
        )
    if not uses_legacy_layers:
        unused = sorted(raw_keys & _UNIFIED_EI_LEGACY_LAYER_KEYS)
        if unused:
            raise ValueError(
                f"model.core.unified_ei keys {unused} are read only by the legacy "
                "layers[] shorthand; with explicit excitatory/inhibitory population "
                "blocks set them per layer or per population instead"
            )
    if "input_projection_layers" in raw_keys and "input_projection_dims" in raw_keys:
        raise ValueError(
            "Set only model.core.unified_ei.input_projection_dims; the deprecated "
            "input_projection_layers alias is ignored when both are present"
        )


# Deprecated flat analysis parameters and the grouped fields they name. The
# analyzers read the grouped fields, whose non-empty defaults used to shadow the
# flat spellings; the loader and the params classes now move flat values onto
# their grouped twins.
INFORMATION_FLAT_PARAMS: dict[str, tuple[str, ...]] = {
    "method": ("estimator", "method"),
    "n_neighbors": ("estimator", "kraskov", "n_neighbors"),
    "n_bins": ("estimator", "binned", "n_bins"),
    "copula_type": ("estimator", "copula", "type"),
    "decoder_cv_folds": ("estimator", "decoder", "cv_folds"),
    "decoder_C": ("estimator", "decoder", "C"),
    "decoder_standardize": ("estimator", "decoder", "standardize"),
    "decoder_seed": ("estimator", "decoder", "seed"),
    "decoder_continuous_strategy": ("estimator", "decoder", "continuous", "strategy"),
    "decoder_continuous_max_total_dim": (
        "estimator",
        "decoder",
        "continuous",
        "max_total_dim",
    ),
    "decoder_gaussian_ridge": ("estimator", "decoder", "continuous", "gaussian_ridge"),
    "binning_strategy": ("estimator", "binned", "binning_strategy"),
    "binning_bias_correction": ("estimator", "binned", "bias_correction"),
    "max_samples": ("compute", "max_samples"),
    "include_vinf": ("compute", "include_vinf"),
    "normalize_mi": ("compute", "normalize_mi"),
    "verbose": ("compute", "verbose"),
    "computation_level": ("compute", "computation_level"),
    "branch_aggregation": ("compute", "branch_aggregation"),
    "per_neuron_analysis": ("compute", "per_neuron_analysis"),
    "per_einet_analysis": ("compute", "per_einet_analysis"),
    "per_layer_analysis": ("compute", "per_layer_analysis"),
    "compute_lda_weights": ("compute", "compute_lda_weights"),
    "layer_total_topk": ("compute", "layer_total_topk"),
    "components": ("selection", "components"),
    "scopes": ("selection", "scopes"),
    "views": ("selection", "views"),
    "signal_variants": ("selection", "signal_variants"),
    "pid_binarize": ("pid", "binarize"),
    "pid_binarize_method": ("pid", "binarize_method"),
    "pid_binarize_threshold": ("pid", "binarize_threshold"),
    "mi_null_shuffles": ("baselines", "label_shuffle_null", "n_shuffles"),
    "mi_null_seed": ("baselines", "label_shuffle_null", "seed"),
    "lda_n_shuffles": ("baselines", "random_weights", "n_shuffles"),
}

CORRELATION_FLAT_PARAMS: dict[str, tuple[str, ...]] = {
    "components": ("selection", "components"),
    "pairs": ("selection", "pairs"),
    "scopes": ("selection", "scopes"),
    "computation_level": ("compute", "computation_level"),
    "max_samples": ("compute", "max_samples"),
    "per_layer_analysis": ("compute", "per_layer_analysis"),
    "per_einet_analysis": ("compute", "per_einet_analysis"),
    "per_neuron_analysis": ("compute", "per_neuron_analysis"),
    "n_synapse_pairs": ("sampling", "n_synapse_pairs"),
    "sampling_strategy": ("sampling", "sampling_strategy"),
    "stratified_categories": ("sampling", "stratified_categories"),
    "max_pairs_per_category": ("sampling", "max_pairs_per_category"),
}


def _warn_moved(section: str, moved: list[str], table: dict[str, tuple[str, ...]]):
    targets = ", ".join(f"{name} -> {'.'.join(table[name])}" for name in moved)
    logger.warning(
        "Flat %s keys are deprecated and were moved to their grouped fields: %s",
        section,
        targets,
    )


def migrate_flat_params(
    params: dict[str, Any],
    table: dict[str, tuple[str, ...]],
    *,
    params_cls: type,
    section: str,
) -> None:
    """Move deprecated flat keys of a loaded params mapping onto grouped keys.

    Operates in place on the mapping read from a config file. A flat key equal
    to its former default is dropped. A flat key whose grouped twin is also
    written raises unless both agree.
    """
    moved = []
    for name, path in table.items():
        if name not in params:
            continue
        value = params.pop(name)
        node = params
        for key in path[:-1]:
            child = node.get(key)
            if child is None:
                child = {}
            elif isinstance(child, Mapping):
                child = dict(child)
            else:
                raise ValueError(f"{section}.{key} must be a mapping")
            node[key] = child
            node = child
        leaf = path[-1]
        grouped = f"{section}.{'.'.join(path)}"
        default = field_default(params_cls, name)
        if leaf in node:
            if node[leaf] == value or value == default:
                continue
            raise ValueError(
                f"{section}.{name}={value!r} conflicts with {grouped}="
                f"{node[leaf]!r}; keep only {grouped}"
            )
        if value == default:
            continue
        node[leaf] = value
        moved.append(name)
    if moved:
        _warn_moved(section, moved, table)


def migrate_flat_fields(
    params: Any, table: dict[str, tuple[str, ...]], *, section: str
) -> None:
    """Copy deprecated flat fields that differ from their defaults to grouped ones.

    Used by ``__post_init__`` of params built in code. A flat field that
    contradicts a grouped field already changed from its default raises.
    """
    moved = []
    for name, path in table.items():
        value = getattr(params, name)
        if value == field_default(type(params), name):
            continue
        parent = params
        for key in path[:-1]:
            parent = getattr(parent, key)
        leaf = path[-1]
        current = getattr(parent, leaf)
        if current == value:
            continue
        grouped = f"{section}.{'.'.join(path)}"
        if current != field_default(type(parent), leaf):
            raise ValueError(
                f"{section}.{name}={value!r} conflicts with {grouped}="
                f"{current!r}; keep only {grouped}"
            )
        setattr(parent, leaf, copy.deepcopy(value))
        moved.append(name)
    if moved:
        _warn_moved(section, moved, table)


__all__ = [
    "CORRELATION_FLAT_PARAMS",
    "INFORMATION_FLAT_PARAMS",
    "canonicalize_unified_ei_config",
    "migrate_flat_fields",
    "migrate_flat_params",
    "normalize_transfer_config",
]
