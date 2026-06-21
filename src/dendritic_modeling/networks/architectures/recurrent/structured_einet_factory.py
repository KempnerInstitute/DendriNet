"""Build structured recurrent E/I networks from structured architecture configs."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from dendritic_modeling.config.conversion import (
    layer_value,
    normalize_sparsity_type,
    to_plain_dict,
)
from dendritic_modeling.config.legacy import normalize_transfer_config
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
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_features import (
    _detect_recurrent_feature_flags,
    _has_positive_layer_value,
    _resolve_recurrent_synapse_counts,
    _should_build_structured_recurrent_einet,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_layer_policy import (
    _direct_ff_inhibitory_enabled,
    _effective_layer_recurrence_enabled,
    _layer_recurrence_enabled,
    _resolve_layer_excitatory_synapse_counts,
    _resolve_layer_inhibitory_synapse_counts,
    _resolve_layer_recurrent_timing,
    _resolve_recurrent_input_projection_dims,
    _resolve_recurrent_store_routing,
    _resolve_recurrent_transfer_params,
    _should_build_inhibitory_population,
    _stateful_layer_features_enabled,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_options import (
    _require_recurrent_excitatory_sizes,
    _resolve_recurrent_morphology_options,
    _resolve_recurrent_reactivation_options,
    _resolve_recurrent_use_transfer,
    _resolve_structured_recurrent_build_options,
    _resolve_transfer_inhibitory_mode,
    _validate_direct_inhibitory_stream,
    _validate_structured_recurrent_type,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _LayerPopulationSynapseCounts,
    _RecurrentFeatureFlags,
    _RecurrentReactivationOptions,
    _RecurrentSynapseCounts,
    _StructuredRecurrentBuildOptions,
    _StructuredRecurrentSections,
)

logger = logging.getLogger(__name__)


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of config-like objects into a plain mapping."""
    return to_plain_dict(obj)


def _layer_value(
    value: Any, layer_idx: int, default: Any = 0, *, allow_empty_list: bool = False
) -> Any:
    """Read scalar/list per-layer values with repeat-last semantics."""
    return layer_value(
        value,
        layer_idx,
        default,
        allow_empty_list=allow_empty_list,
    )


def _normalize_sparsity_type(sparsity_type: Any) -> str:
    """Normalize public sparsity aliases to the internal TopK strategy names."""
    return normalize_sparsity_type(sparsity_type)


def _parse_structured_recurrent_sections(
    raw_params: dict[str, Any],
) -> _StructuredRecurrentSections:
    recurrent_cfg = _to_plain_mapping(raw_params.get("recurrent_ei", {}))
    architecture = _to_plain_mapping(raw_params.get("architecture", {}))
    connectivity = _to_plain_mapping(raw_params.get("connectivity", {}))
    structured_alias_present = "structured_connectivity" in connectivity
    structured_alias = connectivity.pop("structured_connectivity", {})
    if structured_alias_present:
        logger.warning(
            "model.core.connectivity.structured_connectivity is deprecated; "
            "use model.core.connectivity.structured instead."
        )
    structured_connectivity = connectivity.pop("structured", structured_alias)
    transfer = normalize_transfer_config(raw_params.get("transfer", {}))
    morphology = _to_plain_mapping(raw_params.get("morphology", {}))
    sparsity = _to_plain_mapping(raw_params.get("sparsity", {}))
    reactivation = _to_plain_mapping(raw_params.get("reactivation", {}))
    blocklinear = _to_plain_mapping(raw_params.get("blocklinear", {}))
    implementation = _to_plain_mapping(raw_params.get("implementation", {}))
    synapse_types = _to_plain_mapping(raw_params.get("synapse_types", {}))
    dynamics = _to_plain_mapping(
        raw_params.get("dynamics", raw_params.get("spiking", {}))
    )
    dendritic_spikes = _to_plain_mapping(
        raw_params.get("dendritic_spikes", dynamics.get("dendritic_spikes", {}))
    )
    soma_feedback = _to_plain_mapping(
        raw_params.get(
            "soma_feedback",
            recurrent_cfg.get("soma_feedback", dynamics.get("soma_feedback", {})),
        )
    )
    deepst = _to_plain_mapping(sparsity.get("deepst", {}))
    if "annealed_topk" in sparsity and "dense_to_sparse" not in sparsity:
        logger.warning(
            "model.core.sparsity.annealed_topk is deprecated; use "
            "model.core.sparsity.dense_to_sparse instead."
        )
    dense_to_sparse = _to_plain_mapping(
        sparsity.get("dense_to_sparse", sparsity.get("annealed_topk", {}))
    )
    indexed = _to_plain_mapping(sparsity.get("indexed", {}))
    return _StructuredRecurrentSections(
        recurrent_cfg=recurrent_cfg,
        architecture=architecture,
        connectivity=connectivity,
        structured_connectivity=structured_connectivity,
        transfer=transfer,
        morphology=morphology,
        sparsity=sparsity,
        reactivation=reactivation,
        blocklinear=blocklinear,
        implementation=implementation,
        synapse_types=synapse_types,
        dynamics=dynamics,
        dendritic_spikes=dendritic_spikes,
        soma_feedback=soma_feedback,
        deepst=deepst,
        dense_to_sparse=dense_to_sparse,
        indexed=indexed,
    )


def _build_structured_inhibitory_population_config(
    *,
    excitatory_config: Any,
    architecture: dict[str, Any],
    layer_idx: int,
    n_inh: int,
    synapses: _LayerPopulationSynapseCounts,
) -> Any:
    """Create inhibitory PopulationConfig by replacing pathway-specific fields."""
    return replace(
        excitatory_config,
        n_neurons=n_inh,
        branch_factors=list(architecture.get("inhibitory_branch_factors", [1]) or [1]),
        ff_excitatory_synapses=synapses.ff_excitatory,
        ff_inhibitory_synapses=synapses.ff_inhibitory,
        rec_excitatory_synapses=synapses.rec_excitatory,
        rec_inhibitory_synapses=synapses.rec_inhibitory,
        structured_layer_idx=layer_idx,
        ff_excitatory_pathway="ei",
        ff_inhibitory_pathway="ii",
        rec_excitatory_pathway="rec_ei",
        rec_inhibitory_pathway="rec_ii",
    )


def _build_structured_excitatory_population_config(
    *,
    population_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
    layer_idx: int,
    n_exc: Any,
    use_shunting: bool,
    use_additive_normalization: bool,
    reactivation_options: _RecurrentReactivationOptions,
    tau_base: float,
    tau_ratio: float,
) -> Any:
    architecture = sections.architecture
    connectivity = sections.connectivity
    sparsity = sections.sparsity
    morphology = sections.morphology
    reactivation = sections.reactivation
    blocklinear = sections.blocklinear
    implementation = sections.implementation
    recurrent_cfg = sections.recurrent_cfg
    deepst = sections.deepst
    dense_to_sparse = sections.dense_to_sparse
    indexed = sections.indexed
    dynamics = sections.dynamics
    dendritic_spikes = sections.dendritic_spikes
    soma_feedback = sections.soma_feedback
    synapses = _resolve_layer_excitatory_synapse_counts(
        connectivity=connectivity,
        counts=counts,
        layer_idx=layer_idx,
    )

    return population_config_cls(
        n_neurons=int(n_exc),
        branch_factors=list(architecture.get("excitatory_branch_factors", [1]) or [1]),
        ff_excitatory_synapses=synapses.ff_excitatory,
        ff_inhibitory_synapses=synapses.ff_inhibitory,
        rec_excitatory_synapses=synapses.rec_excitatory,
        rec_inhibitory_synapses=synapses.rec_inhibitory,
        **_structured_population_sparsity_kwargs(
            sparsity=sparsity,
            dense_to_sparse=dense_to_sparse,
            indexed=indexed,
        ),
        **_structured_population_morphology_kwargs(
            morphology=morphology,
            use_shunting=use_shunting,
            use_additive_normalization=use_additive_normalization,
        ),
        **_structured_population_reactivation_kwargs(
            reactivation=reactivation,
            reactivation_options=reactivation_options,
        ),
        **_structured_population_blocklinear_kwargs(
            blocklinear=blocklinear,
            implementation=implementation,
        ),
        **_structured_population_timing_kwargs(
            recurrent_cfg=recurrent_cfg,
            tau_base=tau_base,
            tau_ratio=tau_ratio,
        ),
        **_structured_population_deepst_kwargs(deepst),
        **_structured_population_adaptive_init_kwargs(implementation),
        structured_connectivity=_to_plain_mapping(sections.structured_connectivity),
        structured_layer_idx=layer_idx,
        ff_excitatory_pathway="ee",
        ff_inhibitory_pathway="ie",
        rec_excitatory_pathway="rec_ee",
        rec_inhibitory_pathway="rec_ie",
        synapse_types=sections.synapse_types,
        **_structured_population_dynamics_kwargs(dynamics),
        **_structured_population_dendritic_spike_kwargs(
            dendritic_spikes=dendritic_spikes,
            dynamics=dynamics,
        ),
        **_structured_population_soma_feedback_kwargs(
            soma_feedback=soma_feedback,
            dynamics=dynamics,
        ),
    )


def _build_structured_recurrent_layer_config(
    *,
    ei_layer_config_cls: type[Any],
    population_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
    flags: _RecurrentFeatureFlags,
    layer_idx: int,
    n_exc: Any,
    n_inh: int,
    input_mode: int,
    transfer_inhibitory_mode: str,
    explicit_recurrent_layers: Any,
    use_shunting: bool,
    use_additive_normalization: bool,
    reactivation_options: _RecurrentReactivationOptions,
) -> Any:
    """Build one recurrent E/I layer config from structured sections."""
    architecture = sections.architecture
    connectivity = sections.connectivity
    recurrent_cfg = sections.recurrent_cfg

    # Per-layer recurrence
    rec_enabled = _layer_recurrence_enabled(
        layer_idx=layer_idx,
        explicit_recurrent_layers=explicit_recurrent_layers,
        counts=counts,
    )

    timing = _resolve_layer_recurrent_timing(
        recurrent_cfg=recurrent_cfg,
        layer_idx=layer_idx,
    )

    exc_cfg = _build_structured_excitatory_population_config(
        population_config_cls=population_config_cls,
        sections=sections,
        counts=counts,
        layer_idx=layer_idx,
        n_exc=n_exc,
        use_shunting=use_shunting,
        use_additive_normalization=use_additive_normalization,
        reactivation_options=reactivation_options,
        tau_base=timing.tau_base,
        tau_ratio=timing.tau_ratio,
    )

    inh_synapses = _resolve_layer_inhibitory_synapse_counts(
        connectivity=connectivity,
        counts=counts,
        layer_idx=layer_idx,
    )
    build_inhibitory_population = _should_build_inhibitory_population(
        n_inh=n_inh,
        synapses=inh_synapses,
    )

    inh_cfg = None
    if build_inhibitory_population:
        inh_cfg = _build_structured_inhibitory_population_config(
            excitatory_config=exc_cfg,
            architecture=architecture,
            layer_idx=layer_idx,
            n_inh=n_inh,
            synapses=inh_synapses,
        )

    rec_enabled = _effective_layer_recurrence_enabled(
        base_recurrent=rec_enabled,
        flags=flags,
        exc_cfg=exc_cfg,
        inh_cfg=inh_cfg,
    )

    direct_ff_inh = _direct_ff_inhibitory_enabled(
        input_mode=input_mode,
        transfer_inhibitory_mode=transfer_inhibitory_mode,
        layer_idx=layer_idx,
    )

    return ei_layer_config_cls(
        excitatory=exc_cfg,
        inhibitory=inh_cfg,
        recurrent=rec_enabled,
        dt=timing.dt,
        direct_ff_inhibitory_to_excitatory=direct_ff_inh,
    )


def _build_structured_recurrent_layers(
    *,
    ei_layer_config_cls: type[Any],
    population_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
    flags: _RecurrentFeatureFlags,
    excitatory_sizes: list[Any],
    inhibitory_sizes: list[Any],
    input_mode: int,
    transfer_inhibitory_mode: str,
    explicit_recurrent_layers: Any,
    use_shunting: bool,
    use_additive_normalization: bool,
    reactivation_options: _RecurrentReactivationOptions,
) -> list[Any]:
    """Build the per-layer recurrent E/I config sequence."""
    layers: list[Any] = []
    for layer_idx, n_exc in enumerate(excitatory_sizes):
        n_inh = int(_layer_value(inhibitory_sizes, layer_idx, default=0))
        layers.append(
            _build_structured_recurrent_layer_config(
                ei_layer_config_cls=ei_layer_config_cls,
                population_config_cls=population_config_cls,
                sections=sections,
                counts=counts,
                flags=flags,
                layer_idx=layer_idx,
                n_exc=n_exc,
                n_inh=n_inh,
                input_mode=input_mode,
                transfer_inhibitory_mode=transfer_inhibitory_mode,
                explicit_recurrent_layers=explicit_recurrent_layers,
                use_shunting=use_shunting,
                use_additive_normalization=use_additive_normalization,
                reactivation_options=reactivation_options,
            )
        )
    return layers


def _build_structured_recurrent_network_config(
    *,
    network_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    layers: list[Any],
    input_dim: int,
    use_transfer: bool,
) -> Any:
    """Assemble the recurrent network config from resolved sections."""
    transfer_params = _resolve_recurrent_transfer_params(
        transfer=sections.transfer,
        recurrent_cfg=sections.recurrent_cfg,
    )
    input_projection_dims = _resolve_recurrent_input_projection_dims(
        architecture=sections.architecture,
        recurrent_cfg=sections.recurrent_cfg,
    )
    store_routing = _resolve_recurrent_store_routing(
        implementation=sections.implementation,
        recurrent_cfg=sections.recurrent_cfg,
    )

    return network_config_cls(
        layers=layers,
        input_dim=input_dim,
        use_transfer=use_transfer,
        transfer_params=transfer_params,
        input_projection_dims=input_projection_dims,
        output_mode=str(sections.recurrent_cfg.get("output_mode", "last")),
        store_routing=store_routing,
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
        return None

    options = _resolve_structured_recurrent_build_options(
        core_type=type,
        sections=sections,
    )

    layers = _build_structured_recurrent_layers(
        ei_layer_config_cls=EILayerConfig,
        population_config_cls=PopulationConfig,
        sections=sections,
        counts=counts,
        flags=flags,
        excitatory_sizes=options.excitatory_sizes,
        inhibitory_sizes=options.inhibitory_sizes,
        input_mode=options.input_mode,
        transfer_inhibitory_mode=options.transfer_inhibitory_mode,
        explicit_recurrent_layers=options.explicit_recurrent_layers,
        use_shunting=options.use_shunting,
        use_additive_normalization=options.use_additive_normalization,
        reactivation_options=options.reactivation_options,
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
    "_StructuredRecurrentBuildOptions",
    "_build_recurrent_einet_from_structured_config",
    "_detect_recurrent_feature_flags",
    "_direct_ff_inhibitory_enabled",
    "_effective_layer_recurrence_enabled",
    "_has_positive_layer_value",
    "_layer_recurrence_enabled",
    "_require_recurrent_excitatory_sizes",
    "_resolve_layer_excitatory_synapse_counts",
    "_resolve_layer_inhibitory_synapse_counts",
    "_resolve_layer_recurrent_timing",
    "_resolve_recurrent_input_projection_dims",
    "_resolve_recurrent_morphology_options",
    "_resolve_recurrent_reactivation_options",
    "_resolve_recurrent_store_routing",
    "_resolve_recurrent_synapse_counts",
    "_resolve_recurrent_transfer_params",
    "_resolve_recurrent_use_transfer",
    "_resolve_structured_recurrent_build_options",
    "_resolve_transfer_inhibitory_mode",
    "_should_build_inhibitory_population",
    "_should_build_structured_recurrent_einet",
    "_stateful_layer_features_enabled",
    "_validate_direct_inhibitory_stream",
    "_validate_structured_recurrent_type",
]
