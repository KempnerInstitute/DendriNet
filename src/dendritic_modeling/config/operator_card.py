"""Operator card: the dendritic-operator axes a run actually used.

The card records, for one model, its input domain, weight domain, integration
rule, gate nonlinearity, morphology, synapse mechanism and output contract.
When the constructed model is available the axes are read from its dendritic
branch layers, so the card describes what was built for every core family
(including population networks, whose options live in per-population blocks).
Without a model, only structured EINet configs can be summarized; other core
types are marked ``unresolved``.

Reading is best-effort and side-effect free, so writing the card can never
break training.
"""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.biological import (
    BIOLOGICAL_CORE_TYPES,
    resolve_biological_neuron,
)
from dendritic_modeling.config.conversion import to_plain_dict

OPERATOR_CARD_SCHEMA_VERSION = "dendritic_operator_card_v3"


def _get(section: Any, key: str, default: Any = None) -> Any:
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _integration_rule(use_shunting: bool, additive_mode: str, zscore: bool) -> str:
    if use_shunting:
        return "shunting"
    if zscore:
        return "additive_raw_feature_zscore"
    return f"additive_{additive_mode}"


def _summary(values: list[Any]) -> Any:
    """One value when all layers agree, otherwise the sorted distinct values."""
    distinct = []
    for value in values:
        if value not in distinct:
            distinct.append(value)
    if len(distinct) == 1:
        return distinct[0]
    return sorted(distinct, key=repr)


def _layer_facts(layer: Any) -> dict[str, Any]:
    weight_transform = str(_get(layer, "weight_transform", "exp")).lower()
    reactivate = bool(_get(layer, "reactivate", False))
    gate = type(layer.reactivation).__name__ if reactivate else "none"
    return {
        "integration_rule": _integration_rule(
            bool(_get(layer, "use_shunting", True)),
            str(_get(layer, "additive_mode", "raw")),
            bool(_get(layer, "use_additive_normalization", False)),
        ),
        "weight_transform": weight_transform,
        "signed": weight_transform == "identity",
        "gate": gate,
        "gate_init_policy": (
            _get(layer, "reactivation_init_policy", None) if reactivate else None
        ),
        "synapse_mechanism": str(_get(layer, "topk_type", "standard")),
        "branch_factors": list(_get(layer, "branch_factors", []) or []),
        "leak_conductance": float(_get(layer, "leak_conductance", 1.0)),
        "signal_rounding": str(_get(layer, "signal_rounding", "none")),
    }


def _card_from_model(model: Any) -> dict[str, Any] | None:
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
        DendriticBranchLayer,
    )

    layers = [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]
    if not layers:
        return None
    facts = [_layer_facts(layer) for layer in layers]

    def axis(name: str) -> Any:
        return _summary([fact[name] for fact in facts])

    return {
        "source": "constructed_model",
        "dendritic_layers": len(layers),
        "weight_domain": {
            "weight_transform": axis("weight_transform"),
            "signed": axis("signed"),
        },
        "integration_rule": axis("integration_rule"),
        "gates": {"type": axis("gate"), "init_policy": axis("gate_init_policy")},
        "morphology": {
            "branch_factors": axis("branch_factors"),
            "leak_conductance": axis("leak_conductance"),
            "signal_rounding": axis("signal_rounding"),
        },
        "topology": {"synapse_mechanism": axis("synapse_mechanism")},
    }


def _card_from_einet_config(core_config: Any) -> dict[str, Any]:
    from dendritic_modeling.config.model_aliases import (
        get_core_morphology_alias_overrides,
    )

    architecture = _get(core_config, "architecture", {})
    connectivity = _get(core_config, "connectivity", {})
    reactivation = _get(core_config, "reactivation", {})
    sparsity = _get(core_config, "sparsity", {})
    morphology = to_plain_dict(_get(core_config, "morphology", {}))
    morphology.update(
        get_core_morphology_alias_overrides(
            str(_get(core_config, "type", "einet")).lower()
        )
    )
    weight_transform = str(morphology.get("weight_transform") or "softplus").lower()
    enabled = bool(_get(reactivation, "enabled", True))
    return {
        "source": "config",
        "weight_domain": {
            "weight_transform": weight_transform,
            "signed": weight_transform == "identity",
        },
        "integration_rule": _integration_rule(
            bool(morphology.get("use_shunting", True)),
            str(morphology.get("additive_mode") or "raw"),
            bool(morphology.get("use_additive_normalization", False)),
        ),
        "gates": {
            "type": _get(reactivation, "type", "param_tanh") if enabled else "none",
            "init_policy": (
                _get(reactivation, "init_policy", "analytical") if enabled else None
            ),
        },
        "morphology": {
            "excitatory_layer_sizes": _get(
                architecture, "excitatory_layer_sizes", None
            ),
            "inhibitory_layer_sizes": _get(
                architecture, "inhibitory_layer_sizes", None
            ),
            "excitatory_branch_factors": _get(
                architecture, "excitatory_branch_factors", None
            ),
            "somatic_synapses": morphology.get("somatic_synapses"),
        },
        "topology": {
            "synapse_mechanism": _get(sparsity, "type", "standard"),
            "ee_synapses_per_branch": _get(
                connectivity, "ee_synapses_per_branch_per_layer", None
            ),
            "ie_synapses_per_branch": _get(
                connectivity, "ie_synapses_per_branch_per_layer", None
            ),
        },
    }


def build_operator_card(core_config: Any, model: Any = None) -> dict[str, Any]:
    """Summarize the operator a core configuration (and its built model) used."""
    from dendritic_modeling.config.recurrence import (
        _STRUCTURED_EI_TYPES,
        _UNIFIED_EI_TYPES,
    )

    core_type = str(_get(core_config, "type", "einet") or "einet")
    structured = core_type.lower() in _STRUCTURED_EI_TYPES
    card: dict[str, Any] = {
        "schema_version": OPERATOR_CARD_SCHEMA_VERSION,
        "core_type": core_type,
    }

    if core_type.lower() in BIOLOGICAL_CORE_TYPES:
        biological = {
            "declared": resolve_biological_neuron(
                core_type, _get(core_config, "biological_neuron", None)
            ),
            "scope": "neuronal_core",
        }
        if model is not None:
            from dendritic_modeling.networks.architectures.biological_constraints import (
                biological_violations,
            )

            biological["violations"] = biological_violations(model)
        card["biological_neuron"] = biological
        transfer = _get(core_config, "transfer", {})
        if (
            core_type.lower() == "population_network"
            or core_type.lower() in _UNIFIED_EI_TYPES
        ):
            section = (
                "population_network"
                if core_type.lower() == "population_network"
                else "unified_ei"
            )
            population = _get(core_config, section, {})
            transfer = (
                _get(population, "transfer_params", {})
                if _get(population, "use_transfer", False)
                else {}
            )
        built_core = getattr(model, "core_network", model)
        if built_core is not None:
            transfer = getattr(built_core, "transfer_fn", None)
        card["external_afferents"] = {
            "input_source_identity": _get(transfer, "input_source_identity", "shared"),
            "independent_pathways": _get(transfer, "independent_pathways", False),
        }

    spatial_types = {"hierarchical_dendritic_conv", "spatial_dendritic_conv"}
    if core_type.lower() in spatial_types:
        spatial = _get(core_config, "spatial", {})
        card["input_domain"] = {
            "input_transform": _get(spatial, "input_transform", "identity"),
            "input_scale": _get(spatial, "input_scale", 1.0),
        }
        card["output_contract"] = {
            "adapter_mode": _get(spatial, "output_adapter_mode", "identity"),
            "adapter_initial_scale": _get(spatial, "output_initial_scale", 1.0),
            "adapter_initial_threshold": _get(
                spatial, "output_initial_threshold", 0.25
            ),
        }

    resolved = _card_from_model(model) if model is not None else None
    if resolved is None and model is not None:
        card["source"] = "not_applicable"
        return card
    if resolved is None and structured:
        resolved = _card_from_einet_config(core_config)
    if resolved is None:
        card["source"] = "unresolved"
        return card
    card.update(resolved)
    return card


__all__ = ["OPERATOR_CARD_SCHEMA_VERSION", "build_operator_card"]
