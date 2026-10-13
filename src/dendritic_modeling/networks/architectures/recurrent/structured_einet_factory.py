"""Build structured recurrent E/I networks from structured architecture configs."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.inhibitory_wiring import check_retired_inhibitory_routing
from dendritic_modeling.networks.architectures.recurrent.structured_population_kwargs import (
    _structured_population_adaptive_init_kwargs,
    _structured_population_blocklinear_kwargs,
    _structured_population_deepst_kwargs,
    _structured_population_dendritic_spike_kwargs,
    _structured_population_dynamics_kwargs,
    _structured_population_morphology_kwargs,
    _structured_population_reactivation_kwargs,
    _structured_population_soma_feedback_kwargs,
    _structured_population_sparsity_kwargs,
    _structured_population_timing_kwargs,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_config_builders import (
    _build_structured_recurrent_layer_config,
    _build_structured_recurrent_layers,
    _build_structured_recurrent_network_config,
    _layer_value,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_features import (
    _detect_recurrent_feature_flags,
    _has_positive_layer_value,
    _resolve_recurrent_synapse_counts,
    _should_build_structured_recurrent_einet,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_layer_policy import (
    _effective_layer_recurrence_enabled,
    _layer_recurrence_enabled,
    _resolve_layer_excitatory_synapse_counts,
    _resolve_layer_inhibitory_synapse_counts,
    _resolve_layer_recurrent_timing,
    _resolve_recurrent_input_projection_dims,
    _resolve_recurrent_store_routing,
    _should_build_inhibitory_population,
    _stateful_layer_features_enabled,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_options import (
    _require_recurrent_excitatory_sizes,
    _resolve_recurrent_morphology_options,
    _resolve_recurrent_reactivation_options,
    _resolve_recurrent_use_transfer,
    _resolve_structured_recurrent_build_options,
    _validate_structured_recurrent_type,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_population_builders import (
    _build_structured_excitatory_population_config,
    _build_structured_inhibitory_population_config,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_sections import (
    _normalize_sparsity_type,
    _parse_structured_recurrent_sections,
    _to_plain_mapping,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _LayerPopulationSynapseCounts,
    _RecurrentFeatureFlags,
    _RecurrentReactivationOptions,
    _RecurrentSynapseCounts,
    _StructuredRecurrentBuildOptions,
    _StructuredRecurrentSections,
)


def _reject_inert_recurrent_settings(
    core_type: str, recurrent_cfg: dict[str, Any], *, hint: str
) -> None:
    """Fail when ``recurrent_ei`` settings reach a feedforward build.

    ``recurrent_ei`` holds the temporal settings of the recurrent EINet. A
    feedforward build reads none of them; ``enabled: false`` states that choice.
    """
    settings = sorted(
        key
        for key, value in recurrent_cfg.items()
        if value not in (None, [], {}) and not (key == "enabled" and not value)
    )
    if settings:
        raise ValueError(
            f"core type {core_type!r} builds a feedforward network here, which does "
            "not apply "
            + ", ".join(f"model.core.recurrent_ei.{key}" for key in settings)
            + f". {hint}"
        )


def reject_recurrent_request_for_feedforward_core(
    core_type: str,
    raw_params: dict[str, Any],
) -> None:
    """Fail when a feedforward-only core is configured with recurrent features.

    Spatial cores, ``ss_mlp`` and ``dendritic_alexnet`` always build the
    feedforward EINet. Recurrent synapses, stateful dynamics and
    ``recurrent_ei`` settings are therefore ignored by these cores.
    """
    sections = _parse_structured_recurrent_sections(raw_params)
    counts = _resolve_recurrent_synapse_counts(
        connectivity=sections.connectivity,
        recurrent_cfg=sections.recurrent_cfg,
    )
    flags = _detect_recurrent_feature_flags(sections, counts)
    if not _should_build_structured_recurrent_einet(sections, flags):
        _reject_inert_recurrent_settings(
            core_type,
            sections.recurrent_cfg,
            hint=(
                "Remove these settings or use an EINet, unified_ei or "
                "population_network core for recurrent dynamics."
            ),
        )
        return
    requested = [
        name
        for name, active in (
            ("connectivity.rec_* synapses", flags.has_rec_synapses),
            ("recurrent_ei.enabled", bool(sections.recurrent_cfg.get("enabled"))),
            ("soma_feedback.enabled", flags.has_soma_feedback),
            ("dynamics.mode=spike", flags.has_spiking_dynamics),
            ("dendritic_spikes.enabled", flags.has_dendritic_spikes),
            ("synapse_types.enabled", flags.has_synapse_types),
        )
        if active
    ]
    raise ValueError(
        f"core type {core_type!r} builds a feedforward dendritic network and "
        f"does not apply {', '.join(requested)}. Remove these options or use an "
        "EINet, unified_ei or population_network core for recurrent dynamics."
    )


def _build_recurrent_einet_from_structured_config(
    *,
    type: str,
    raw_params: dict[str, Any],
    input_dim: int,
):
    """Translate structured EINet config + recurrent synapse counts to recurrent EINetwork.

    Recurrence is triggered when any ``connectivity.rec_*`` synapse count is > 0,
    or (for backward compatibility) when ``recurrent_ei.enabled`` is True.
    """
    from dendritic_modeling.networks.architectures.recurrent import (
        EILayerConfig,
        EINetwork,
        EINetworkConfig,
        PopulationConfig,
    )

    sections = _parse_structured_recurrent_sections(raw_params)
    counts = _resolve_recurrent_synapse_counts(
        connectivity=sections.connectivity,
        recurrent_cfg=sections.recurrent_cfg,
    )
    flags = _detect_recurrent_feature_flags(sections, counts)

    # Backward compat: old configs use recurrent_ei.enabled: true to activate
    # recurrence.  New configs rely solely on connectivity.rec_* > 0.
    if not _should_build_structured_recurrent_einet(sections, flags):
        _reject_inert_recurrent_settings(
            type,
            sections.recurrent_cfg,
            hint=(
                "A structured EINet is recurrent when connectivity.rec_* synapses, "
                "stateful dynamics or recurrent_ei.enabled: true are set; add one "
                "or remove these settings."
            ),
        )
        return None

    check_retired_inhibitory_routing(raw_params, recurrent=True)
    options = _resolve_structured_recurrent_build_options(
        core_type=type,
        sections=sections,
        counts=counts,
    )

    layers = _build_structured_recurrent_layers(
        ei_layer_config_cls=EILayerConfig,
        population_config_cls=PopulationConfig,
        sections=sections,
        counts=counts,
        flags=flags,
        excitatory_sizes=options.excitatory_sizes,
        inhibitory_sizes=options.inhibitory_sizes,
        explicit_recurrent_layers=options.explicit_recurrent_layers,
        use_shunting=options.use_shunting,
        use_additive_normalization=options.use_additive_normalization,
        reactivation_options=options.reactivation_options,
    )
    if not any(layer.recurrent for layer in layers):
        raise ValueError(
            "model.core.recurrent_ei.enabled=true requests a recurrent network, but "
            "no layer has recurrent synapses, stateful dynamics or a true "
            "recurrent_ei.recurrent_layers entry, so the network would run "
            "feedforward. Add connectivity.rec_* synapses or remove "
            "recurrent_ei.enabled."
        )

    cfg = _build_structured_recurrent_network_config(
        network_config_cls=EINetworkConfig,
        sections=sections,
        layers=layers,
        input_dim=input_dim,
        use_transfer=options.use_transfer,
    )
    return EINetwork(cfg)


__all__ = [
    "_LayerPopulationSynapseCounts",
    "_RecurrentFeatureFlags",
    "_RecurrentReactivationOptions",
    "_RecurrentSynapseCounts",
    "_StructuredRecurrentBuildOptions",
    "_StructuredRecurrentSections",
    "_build_recurrent_einet_from_structured_config",
    "_build_structured_excitatory_population_config",
    "_build_structured_inhibitory_population_config",
    "_build_structured_recurrent_layer_config",
    "_build_structured_recurrent_layers",
    "_build_structured_recurrent_network_config",
    "_detect_recurrent_feature_flags",
    "_effective_layer_recurrence_enabled",
    "_has_positive_layer_value",
    "_layer_recurrence_enabled",
    "_layer_value",
    "_normalize_sparsity_type",
    "_parse_structured_recurrent_sections",
    "_require_recurrent_excitatory_sizes",
    "_resolve_layer_excitatory_synapse_counts",
    "_resolve_layer_inhibitory_synapse_counts",
    "_resolve_layer_recurrent_timing",
    "_resolve_recurrent_input_projection_dims",
    "_resolve_recurrent_morphology_options",
    "_resolve_recurrent_reactivation_options",
    "_resolve_recurrent_store_routing",
    "_resolve_recurrent_synapse_counts",
    "_resolve_recurrent_use_transfer",
    "_resolve_structured_recurrent_build_options",
    "_should_build_inhibitory_population",
    "_should_build_structured_recurrent_einet",
    "_stateful_layer_features_enabled",
    "_structured_population_adaptive_init_kwargs",
    "_structured_population_blocklinear_kwargs",
    "_structured_population_deepst_kwargs",
    "_structured_population_dendritic_spike_kwargs",
    "_structured_population_dynamics_kwargs",
    "_structured_population_morphology_kwargs",
    "_structured_population_reactivation_kwargs",
    "_structured_population_soma_feedback_kwargs",
    "_structured_population_sparsity_kwargs",
    "_structured_population_timing_kwargs",
    "_to_plain_mapping",
    "_validate_structured_recurrent_type",
]
