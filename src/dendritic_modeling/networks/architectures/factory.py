from dataclasses import asdict, replace
from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.config.conversion import (
    has_enabled_synapse_types,
    layer_value,
    normalize_sparsity_type,
    to_plain_dict,
)
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    get_structured_recurrent_alias_overrides,
    get_unified_ei_alias_overrides,
    warn_alias_conflicts,
)
from dendritic_modeling.networks.activations import resolve_dendritic_activation
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    CNNAutoencoder,
    CNNVariationalAutoencoder,
    MLPAutoencoder,
    MLPVariationalAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    EffectiveMatchedParamMLP,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    PointMatchedParamMLP,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.registry import (
    build_registered_architecture,
    get_registered_architecture_names,
    reserve_architecture_names,
)

# All EINet type strings handled via ConfigurableEINetwork (+ dendritic_mlp special-cased)
_EINET_TYPES: set[str] = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_mlp",
    "dendritic_signed",
    "flat_shunting",
    "flat_additive",
    "flat_mlp",
    "flat_signed",
}

_UNIFIED_EI_TYPES: set[str] = {
    "ei_unified",
    "unified_ei",
    "ei_net",
    "unified_einet",
    # Parallel aliases that mirror the BP / feedforward vocabulary
    # (`dendritic_shunting`, `dendritic_additive`, `flat_shunting`,
    # `flat_additive`). An `rnn_` prefix disambiguates them from the
    # non-recurrent BP types. The factory below canonicalizes each alias
    # to the appropriate `use_shunting` / `use_additive_normalization`
    # pair on every configured population, leaving the `ei_unified` path
    # unchanged for backward compatibility.
    "rnn_dendritic_shunting",
    "rnn_dendritic_additive",
    "rnn_flat_shunting",
    "rnn_flat_additive",
}

# Spatial dendritic cores: flatten encoder spatial output → EINet → project.
# Tensor-map baseline: flatten → DendriNet → project (no spatial locality).
_SPATIAL_DENDRITIC_TYPES: set[str] = {
    "hierarchical_dendritic_tensor_map",
    "dendritic_tensor_map",
    "tensor_map",
}

# Spatial dendritic conv: F.unfold → shared DendriNet on local patches → fold.
# Preserves spatial structure (B, C_out, H_out, W_out).
# Canonical name: hierarchical_dendritic_conv
_SPATIAL_DENDRITIC_CONV_TYPES: set[str] = {
    "hierarchical_dendritic_conv",
    "spatial_dendritic_conv",
}

_BASELINE_RNN_TYPES: dict[str, str] = {
    "gru": "gru",
    "lstm": "lstm",
    "vanilla_rnn": "vanilla",
    "rnn": "vanilla",
}

_POPULATION_GRAPH_TYPES: set[str] = {
    "population_graph",
    "cell_type_graph",
    "cell_types",
}

_MULTI_POPULATION_TYPES: set[str] = {
    "multi_population_ei",
    "multi_pop_ei",
    "multi_population",
    "population_layers",
}

_BUILTIN_ARCHITECTURE_TYPES: frozenset[str] = frozenset(
    set(_EINET_TYPES)
    | set(_UNIFIED_EI_TYPES)
    | set(_BASELINE_RNN_TYPES)
    | set(_POPULATION_GRAPH_TYPES)
    | set(_MULTI_POPULATION_TYPES)
    | set(_SPATIAL_DENDRITIC_TYPES)
    | set(_SPATIAL_DENDRITIC_CONV_TYPES)
    | {
        "active_param_mlp",
        "alexnet",
        "cnn",
        "cnn_down",
        "cnn_downsample",
        "cnn_up",
        "cnn_upsample",
        "cnnautoencoder",
        "cnnvariationalautoencoder",
        "cnndownsample",
        "cnnupsample",
        "dendritic_rnn",
        "dendrnn",
        "effectivematchedparammlp",
        "identity",
        "mlp",
        "mlpautoencoder",
        "mlpvariationalautoencoder",
        "none",
        "pathway_router",
        "point_mlp",
        "router",
        "ss_mlp",
        "ss_mlp_flat",
        "total_param_mlp",
    }
)

reserve_architecture_names(_BUILTIN_ARCHITECTURE_TYPES)


def get_available_architectures() -> list[str]:
    """Return built-in and extension architecture names."""
    return sorted(
        set(_BUILTIN_ARCHITECTURE_TYPES) | set(get_registered_architecture_names())
    )


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

    recurrent_cfg = _to_plain_mapping(raw_params.get("recurrent_ei", {}))
    architecture = _to_plain_mapping(raw_params.get("architecture", {}))
    connectivity = _to_plain_mapping(raw_params.get("connectivity", {}))
    structured_connectivity = connectivity.pop(
        "structured",
        connectivity.pop("structured_connectivity", {}),
    )
    transfer = _to_plain_mapping(raw_params.get("transfer", {}))
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
    dense_to_sparse = _to_plain_mapping(
        sparsity.get("dense_to_sparse", sparsity.get("annealed_topk", {}))
    )

    # --- Resolve recurrent synapse counts (new location first, old fallback) ---
    def _rec_syn(new_key, *old_keys, fallback=0):
        if new_key in connectivity:
            return connectivity[new_key]
        for k in old_keys:
            if k in recurrent_cfg:
                return recurrent_cfg[k]
        return fallback

    rec_ee = _rec_syn(
        "rec_ee_synapses_per_branch",
        "rec_excitatory_synapses_per_layer",
        "rec_excitatory_synapses",
    )
    rec_ie = _rec_syn(
        "rec_ie_synapses_per_branch",
        "rec_inhibitory_synapses_per_layer",
        "rec_inhibitory_synapses",
    )
    rec_ei = _rec_syn(
        "rec_ei_synapses_per_branch",
        "inh_rec_excitatory_synapses_per_layer",
        "inh_rec_excitatory_synapses",
        fallback=rec_ee,
    )
    rec_ii = _rec_syn(
        "rec_ii_synapses_per_branch",
        "inh_rec_inhibitory_synapses_per_layer",
        "inh_rec_inhibitory_synapses",
        fallback=rec_ie,
    )

    # --- Determine whether recurrence is active ---
    def _any_positive(val):
        if isinstance(val, list):
            return any(v > 0 for v in val)
        return val is not None and val > 0

    has_rec_synapses = any(_any_positive(v) for v in [rec_ee, rec_ie, rec_ei, rec_ii])
    has_soma_feedback = bool(
        soma_feedback.get("enabled", dynamics.get("soma_feedback_enabled", False))
    )
    has_spiking_dynamics = (
        str(dynamics.get("mode", dynamics.get("dynamics_mode", "rate"))).lower()
        == "spike"
    )
    has_dendritic_spikes = bool(
        dendritic_spikes.get(
            "enabled",
            dynamics.get("dendritic_spikes_enabled", False),
        )
    )
    has_synapse_types = has_enabled_synapse_types(synapse_types)
    has_stateful_features = (
        has_soma_feedback
        or has_spiking_dynamics
        or has_dendritic_spikes
        or has_synapse_types
    )
    # Backward compat: old configs use recurrent_ei.enabled: true to activate
    # recurrence.  New configs rely solely on connectivity.rec_* > 0.
    explicit_enabled = recurrent_cfg.get("enabled", None)
    if not has_rec_synapses and not explicit_enabled and not has_stateful_features:
        return None

    if type in {"dendritic_mlp", "flat_shunting", "flat_additive"}:
        raise ValueError(
            f"Recurrent mode is not supported for architecture type '{type}'. "
            "Use 'EINet', 'dendritic_shunting', or 'dendritic_additive'."
        )

    excitatory_sizes = list(architecture.get("excitatory_layer_sizes", []))
    if not excitatory_sizes:
        raise ValueError(
            "Recurrent mode requires non-empty architecture.excitatory_layer_sizes"
        )

    inhibitory_sizes = list(architecture.get("inhibitory_layer_sizes", []))
    input_mode = int(transfer.get("input_mode", 0))
    has_inhibitory_population = any(int(size or 0) > 0 for size in inhibitory_sizes)
    has_direct_ie_synapses = any(
        int(
            _layer_value(
                connectivity.get("ie_synapses_per_branch_per_layer", []), i, default=0
            )
        )
        > 0
        for i in range(len(excitatory_sizes))
    )
    allow_direct_inhibitory_stream = bool(
        transfer.get("allow_direct_inhibitory_stream", False)
        or transfer.get("allow_direct_inhibitory_input", False)
    )
    if (
        input_mode == 1
        and not has_inhibitory_population
        and has_direct_ie_synapses
        and not allow_direct_inhibitory_stream
    ):
        raise ValueError(
            "input_mode=1 with empty inhibitory_layer_sizes and nonzero "
            "ie_synapses_per_branch_per_layer would use the transferred "
            "inhibitory stream directly instead of building inhibitory "
            "neurons. Set architecture.inhibitory_layer_sizes to positive "
            "values for an inhibitory population, or set "
            "transfer.allow_direct_inhibitory_stream=true to opt in to the "
            "direct stream explicitly."
        )

    # use_transfer: keep legacy EINet behavior by default (TransferLayer always on),
    # with backward-compatible explicit override via recurrent_ei.use_transfer.
    if "use_transfer" in recurrent_cfg:
        use_transfer = bool(recurrent_cfg["use_transfer"])
    else:
        use_transfer = True

    # Preserve legacy EINet type aliases for shunting/additive behavior.
    core_morphology_overrides = get_core_morphology_alias_overrides(type)
    warn_alias_conflicts(
        type,
        morphology,
        core_morphology_overrides,
        context="model.core.morphology",
    )
    structured_alias_overrides = get_structured_recurrent_alias_overrides(type)

    use_shunting = bool(morphology.get("use_shunting", True))
    if "use_shunting" in core_morphology_overrides:
        use_shunting = bool(core_morphology_overrides["use_shunting"])

    use_additive_normalization = bool(
        morphology.get("use_additive_normalization", False)
    )
    if "use_additive_normalization" in core_morphology_overrides:
        use_additive_normalization = bool(
            core_morphology_overrides["use_additive_normalization"]
        )
    if "use_additive_normalization" in structured_alias_overrides:
        use_additive_normalization = bool(
            structured_alias_overrides["use_additive_normalization"]
        )

    reactivation_init_m = float(reactivation.get("init_m", 1.5))
    reactivation_init_b = float(reactivation.get("init_b", 0.5))
    reactivation_init_policy = str(reactivation.get("init_policy", "analytical"))
    reactivate, reactivation_type = resolve_dendritic_activation(
        reactivation.get("dendritic_activation"),
        bool(reactivation.get("enabled", True)),
        reactivation.get("type", "param_tanh"),
    )
    if structured_alias_overrides:
        warn_alias_conflicts(
            type,
            reactivation,
            {
                "init_m": structured_alias_overrides.get("reactivation_init_m"),
                "init_b": structured_alias_overrides.get("reactivation_init_b"),
                "init_policy": structured_alias_overrides.get(
                    "reactivation_init_policy"
                ),
            },
            context="model.core.reactivation",
        )
        reactivation_init_m = float(
            structured_alias_overrides.get("reactivation_init_m", reactivation_init_m)
        )
        reactivation_init_b = float(
            structured_alias_overrides.get("reactivation_init_b", reactivation_init_b)
        )
        reactivation_init_policy = str(
            structured_alias_overrides.get(
                "reactivation_init_policy", reactivation_init_policy
            )
        )

    # Per-layer recurrence: infer from synapse counts, backward compat with explicit flag
    explicit_recurrent_layers = recurrent_cfg.get(
        "recurrent_layers", recurrent_cfg.get("recurrent_per_layer", None)
    )

    dt_value = recurrent_cfg.get("dt", 1.0)

    # transfer_inhibitory_mode: new location first, old fallback
    transfer_inhibitory_mode = str(
        transfer.get(
            "inhibitory_mode",
            recurrent_cfg.get("transfer_inhibitory_mode", "first"),
        )
    ).lower()
    if transfer_inhibitory_mode not in {"none", "first", "all"}:
        raise ValueError(
            "transfer.inhibitory_mode must be one of ('none', 'first', 'all')"
        )

    layers: list[EILayerConfig] = []
    for layer_idx, n_exc in enumerate(excitatory_sizes):
        n_inh = int(_layer_value(inhibitory_sizes, layer_idx, default=0))
        # A positive inhibitory layer size requests an explicit inhibitory
        # population. input_mode=1 only controls whether the transfer layer
        # supplies an external inhibitory stream.
        build_inhibitory_population = n_inh > 0

        # Per-layer recurrence
        if explicit_recurrent_layers is not None:
            rec_enabled = bool(
                _layer_value(explicit_recurrent_layers, layer_idx, default=True)
            )
        else:
            layer_rec_total = sum(
                int(_layer_value(v, layer_idx, default=0))
                for v in [rec_ee, rec_ie, rec_ei, rec_ii]
            )
            rec_enabled = layer_rec_total > 0

        dt = float(_layer_value(dt_value, layer_idx, default=1.0))

        # Per-layer tau_base / tau_ratio
        tau_base = float(
            _layer_value(recurrent_cfg.get("tau_base", 50.0), layer_idx, default=50.0)
        )
        tau_ratio = float(
            _layer_value(recurrent_cfg.get("tau_ratio", 3.0), layer_idx, default=3.0)
        )

        exc_cfg = PopulationConfig(
            n_neurons=int(n_exc),
            branch_factors=list(
                architecture.get("excitatory_branch_factors", [1]) or [1]
            ),
            ff_excitatory_synapses=int(
                _layer_value(
                    connectivity.get("ee_synapses_per_branch_per_layer", []),
                    layer_idx,
                    default=0,
                )
            ),
            ff_inhibitory_synapses=int(
                _layer_value(
                    connectivity.get("ie_synapses_per_branch_per_layer", []),
                    layer_idx,
                    default=0,
                )
            ),
            rec_excitatory_synapses=int(_layer_value(rec_ee, layer_idx, default=0)),
            rec_inhibitory_synapses=int(_layer_value(rec_ie, layer_idx, default=0)),
            topk_init_method=str(sparsity.get("init_method", "xavier_normal")),
            topk_noise_level=float(sparsity.get("noise_level", 0.0)),
            topk_type=_normalize_sparsity_type(sparsity.get("type", "standard")),
            topk_weight_norm_order=sparsity.get("weight_norm_order"),
            topk_gamma=float(sparsity.get("gamma", 1.0)),
            topk_temperature=float(sparsity.get("temperature", 0.5)),
            topk_ultrafast=bool(sparsity.get("ultrafast", False)),
            topk_strategy=str(sparsity.get("gradient_scaling", "none")),
            dense_to_sparse_initial_density=float(
                dense_to_sparse.get("initial_density", 1.0)
            ),
            dense_to_sparse_initial_k=dense_to_sparse.get("initial_k"),
            dense_to_sparse_start_step=int(dense_to_sparse.get("start_step", 0)),
            dense_to_sparse_end_step=int(dense_to_sparse.get("end_step", 1000)),
            dense_to_sparse_update_interval=int(
                dense_to_sparse.get("update_interval", 1)
            ),
            dense_to_sparse_schedule=str(dense_to_sparse.get("schedule", "cubic")),
            dense_to_sparse_freeze_on_end=bool(
                dense_to_sparse.get("freeze_on_end", False)
            ),
            dense_to_sparse_advance_on_forward=bool(
                dense_to_sparse.get("advance_on_forward", True)
            ),
            dense_to_sparse_prune_metric=str(
                dense_to_sparse.get("prune_metric", "weight")
            ),
            weight_transform=str(morphology.get("weight_transform", "softplus")),
            use_shunting=use_shunting,
            use_additive_normalization=use_additive_normalization,
            allow_self_recurrence=bool(morphology.get("allow_self_recurrence", True)),
            reactivate=reactivate,
            reactivation_type=reactivation_type,
            reactivation_init_m=reactivation_init_m,
            reactivation_init_b=reactivation_init_b,
            reactivation_init_policy=reactivation_init_policy,
            reactivation_occupancy_quantile_low=reactivation.get(
                "occupancy_quantile_low"
            ),
            reactivation_occupancy_quantile_high=reactivation.get(
                "occupancy_quantile_high"
            ),
            reactivation_occupancy_target_low=reactivation.get("occupancy_target_low"),
            reactivation_occupancy_target_high=reactivation.get(
                "occupancy_target_high"
            ),
            reactivation_calibration_min_quantile_width=float(
                reactivation.get("calibration_min_quantile_width", 1e-3)
            ),
            reactivation_calibration_max_m=float(
                reactivation.get("calibration_max_m", 50.0)
            ),
            reactivation_calibration_revert_on_invalid=bool(
                reactivation.get("calibration_revert_on_invalid", True)
            ),
            reactivation_sigma_aware_k=float(reactivation.get("sigma_aware_k", 0.25)),
            reactivation_strategy=str(reactivation.get("gradient_scaling", "none")),
            blocklinear_strategy=str(blocklinear.get("gradient_scaling", "none")),
            level_taus=list(recurrent_cfg.get("level_taus", [])),
            tau_base=tau_base,
            tau_ratio=tau_ratio,
            learnable_tau=bool(recurrent_cfg.get("learnable_tau", False)),
            epsilon=float(recurrent_cfg.get("epsilon", 1e-8)),
            somatic_synapses=bool(morphology.get("somatic_synapses", False)),
            dbl_init_method=str(
                morphology.get("dbl_init_method", "analytical_expectation")
            ),
            efficient_blocklinear=bool(blocklinear.get("efficient", False)),
            print_hooks=bool(implementation.get("print_hooks", False)),
            excitatory_target_density=float(
                deepst.get("excitatory_target_density", 0.1)
            ),
            inhibitory_target_density=float(
                deepst.get("inhibitory_target_density", 0.1)
            ),
            use_noise=bool(deepst.get("use_noise", True)),
            sigma=float(deepst.get("sigma", 0.05)),
            rewiring_mode=str(deepst.get("rewiring_mode", "global")),
            rewire_frequency=int(deepst.get("rewire_frequency", 1)),
            synapses_per_branch=deepst.get("synapses_per_branch"),
            freeze_excitatory_connectivity=bool(
                deepst.get("freeze_excitatory_connectivity", False)
            ),
            freeze_inhibitory_connectivity=bool(
                deepst.get("freeze_inhibitory_connectivity", False)
            ),
            init_method=str(deepst.get("init_method", "xavier_normal")),
            weight_threshold=float(deepst.get("weight_threshold", 1e-6)),
            adaptive_initialization=bool(
                implementation.get("adaptive_initialization", True)
            ),
            structured_connectivity=_to_plain_mapping(structured_connectivity),
            structured_layer_idx=layer_idx,
            ff_excitatory_pathway="ee",
            ff_inhibitory_pathway="ie",
            rec_excitatory_pathway="rec_ee",
            rec_inhibitory_pathway="rec_ie",
            synapse_types=synapse_types,
            dynamics_mode=str(
                dynamics.get("mode", dynamics.get("dynamics_mode", "rate"))
            ),
            spike_threshold=float(
                dynamics.get("threshold", dynamics.get("spike_threshold", 1.0))
            ),
            spike_reset=float(dynamics.get("reset", dynamics.get("spike_reset", 0.0))),
            spike_tau=float(dynamics.get("tau", dynamics.get("spike_tau", 20.0))),
            spike_refractory_steps=int(
                dynamics.get(
                    "refractory_steps", dynamics.get("spike_refractory_steps", 0)
                )
            ),
            spike_surrogate_beta=float(
                dynamics.get(
                    "surrogate_beta", dynamics.get("spike_surrogate_beta", 10.0)
                )
            ),
            spike_readout=str(
                dynamics.get("readout", dynamics.get("spike_readout", "spikes"))
            ),
            spike_readout_tau=float(
                dynamics.get("readout_tau", dynamics.get("spike_readout_tau", 20.0))
            ),
            dendritic_spikes_enabled=bool(
                dendritic_spikes.get(
                    "enabled",
                    dynamics.get("dendritic_spikes_enabled", False),
                )
            ),
            dendritic_spike_mode=str(
                dendritic_spikes.get(
                    "mode",
                    dynamics.get("dendritic_spike_mode", "plateau"),
                )
            ),
            dendritic_spike_levels=dendritic_spikes.get(
                "levels",
                dynamics.get("dendritic_spike_levels", "non_soma"),
            ),
            dendritic_spike_threshold=float(
                dendritic_spikes.get(
                    "threshold",
                    dynamics.get("dendritic_spike_threshold", 0.7),
                )
            ),
            dendritic_spike_plateau_amplitude=float(
                dendritic_spikes.get(
                    "plateau_amplitude",
                    dynamics.get("dendritic_spike_plateau_amplitude", 1.0),
                )
            ),
            dendritic_spike_plateau_tau=float(
                dendritic_spikes.get(
                    "plateau_tau",
                    dynamics.get("dendritic_spike_plateau_tau", 20.0),
                )
            ),
            dendritic_spike_refractory_steps=int(
                dendritic_spikes.get(
                    "refractory_steps",
                    dynamics.get("dendritic_spike_refractory_steps", 0),
                )
            ),
            dendritic_spike_surrogate_beta=float(
                dendritic_spikes.get(
                    "surrogate_beta",
                    dynamics.get("dendritic_spike_surrogate_beta", 10.0),
                )
            ),
            dendritic_spike_propagation=str(
                dendritic_spikes.get(
                    "propagation",
                    dynamics.get("dendritic_spike_propagation", "additive"),
                )
            ),
            soma_feedback_enabled=bool(
                soma_feedback.get(
                    "enabled",
                    dynamics.get("soma_feedback_enabled", False),
                )
            ),
            soma_feedback_mode=str(
                soma_feedback.get(
                    "mode",
                    dynamics.get("soma_feedback_mode", "additive"),
                )
            ),
            soma_feedback_source=str(
                soma_feedback.get(
                    "source",
                    dynamics.get("soma_feedback_source", "output"),
                )
            ),
            soma_feedback_levels=soma_feedback.get(
                "levels",
                dynamics.get("soma_feedback_levels", "non_soma"),
            ),
            soma_feedback_strength=float(
                soma_feedback.get(
                    "strength",
                    dynamics.get("soma_feedback_strength", 0.1),
                )
            ),
            soma_feedback_learnable_strength=bool(
                soma_feedback.get(
                    "learnable_strength",
                    dynamics.get("soma_feedback_learnable_strength", False),
                )
            ),
            soma_feedback_per_level=bool(
                soma_feedback.get(
                    "per_level",
                    dynamics.get("soma_feedback_per_level", True),
                )
            ),
            soma_feedback_init_std=float(
                soma_feedback.get(
                    "init_std",
                    dynamics.get("soma_feedback_init_std", 0.0),
                )
            ),
            soma_feedback_reversal=float(
                soma_feedback.get(
                    "reversal",
                    dynamics.get("soma_feedback_reversal", 1.0),
                )
            ),
        )

        inh_cfg = None
        if build_inhibitory_population:
            inh_cfg = replace(
                exc_cfg,
                n_neurons=n_inh,
                branch_factors=list(
                    architecture.get("inhibitory_branch_factors", [1]) or [1]
                ),
                ff_excitatory_synapses=int(
                    _layer_value(
                        connectivity.get("ei_synapses_per_branch_per_layer", []),
                        layer_idx,
                        default=0,
                    )
                ),
                ff_inhibitory_synapses=int(
                    _layer_value(
                        connectivity.get("ii_synapses_per_branch_per_layer", []),
                        layer_idx,
                        default=0,
                    )
                ),
                rec_excitatory_synapses=int(_layer_value(rec_ei, layer_idx, default=0)),
                rec_inhibitory_synapses=int(_layer_value(rec_ii, layer_idx, default=0)),
                structured_layer_idx=layer_idx,
                ff_excitatory_pathway="ei",
                ff_inhibitory_pathway="ii",
                rec_excitatory_pathway="rec_ei",
                rec_inhibitory_pathway="rec_ii",
            )

        rec_enabled = (
            rec_enabled
            or has_spiking_dynamics
            or exc_cfg.dendritic_spikes_enabled
            or has_synapse_types
            or exc_cfg.soma_feedback_enabled
            or (
                inh_cfg is not None
                and (
                    inh_cfg.dendritic_spikes_enabled
                    or has_synapse_types
                    or inh_cfg.soma_feedback_enabled
                )
            )
        )

        direct_ff_inh = False
        if input_mode == 1:
            if transfer_inhibitory_mode == "all":
                direct_ff_inh = True
            elif transfer_inhibitory_mode == "first":
                direct_ff_inh = layer_idx == 0

        layers.append(
            EILayerConfig(
                excitatory=exc_cfg,
                inhibitory=inh_cfg,
                recurrent=rec_enabled,
                dt=dt,
                direct_ff_inhibitory_to_excitatory=direct_ff_inh,
            )
        )

    transfer_params = dict(transfer)
    transfer_params.update(_to_plain_mapping(recurrent_cfg.get("transfer", {})))
    transfer_params.update(_to_plain_mapping(recurrent_cfg.get("transfer_params", {})))

    # input_projection_dims: new location first, old fallback
    raw_ipd = architecture.get(
        "input_projection_dims",
        recurrent_cfg.get("input_projection_dims", []),
    )
    ipd = list(raw_ipd) if isinstance(raw_ipd, list) else []

    # store_routing: new location first, old fallback
    store_routing = bool(
        implementation.get("store_routing", recurrent_cfg.get("store_routing", False))
    )

    cfg = EINetworkConfig(
        layers=layers,
        input_dim=input_dim,
        use_transfer=use_transfer,
        transfer_params=transfer_params,
        input_projection_dims=ipd,
        output_mode=str(recurrent_cfg.get("output_mode", "last")),
        store_routing=store_routing,
    )
    return EINetwork(cfg)


def get_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int] = None,
    suffix_input_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory method for creating network architectures.

    Args:
        type: Architecture type name
        parameters: Either flat parameters dict or structured config
        input_dim: Optional input dimension (needed for EINet from config)
        suffix_input_dim: Optional suffix input dimension. Used by spatial
            dendritic cores (e.g., ``hierarchical_dendritic_tensor_map``) to
            add an output projection when the EINet output dim doesn't match
            the downstream decoder's expected input.

    Returns:
        Initialized network module
    """
    type = (type or "").lower()
    if type not in _BUILTIN_ARCHITECTURE_TYPES:
        registered = build_registered_architecture(
            type,
            parameters,
            input_dim=input_dim,
            suffix_input_dim=suffix_input_dim,
        )
        if registered is not None:
            return registered

    # ------------------------------------------------------------------
    # Unified E-I architecture (opt-in, isolated from legacy code paths)
    # ------------------------------------------------------------------
    if type in _UNIFIED_EI_TYPES:
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
            # Allow passing EINetworkConfig-shaped dict directly.
            unified_cfg = raw_params

        if not isinstance(unified_cfg, dict):
            raise ValueError(
                "Unified E-I config must be a mapping, got "
                f"{unified_cfg.__class__.__name__}"
            )

        unified_cfg = dict(unified_cfg)
        unified_cfg["input_dim"] = input_dim

        # Resolve recurrent-type aliases to explicit population flags. When
        # `type` is `ei_unified` / `unified_ei` / `ei_net`, this is a no-op;
        # otherwise it sets use_shunting / use_additive_normalization / flat
        # branch_factors to match the alias on E and I populations.
        alias_overrides = get_unified_ei_alias_overrides(type)

        layers_cfg = []
        for layer_idx, layer in enumerate(unified_cfg.get("layers", [])):
            if isinstance(layer, EILayerConfig):
                if not alias_overrides:
                    layers_cfg.append(layer)
                    continue
                layer = asdict(layer)
            if not isinstance(layer, dict):
                raise ValueError(
                    "Each unified E-I layer config must be a dict or EILayerConfig"
                )
            layer = dict(layer)
            exc = layer.get("excitatory")
            inh = layer.get("inhibitory")
            if exc is not None and not isinstance(exc, dict):
                exc_plain = _to_plain_mapping(exc)
                if exc_plain:
                    exc = exc_plain
            if inh is not None and not isinstance(inh, dict):
                inh_plain = _to_plain_mapping(inh)
                if inh_plain:
                    inh = inh_plain
            if isinstance(exc, dict):
                exc = _apply_population_alias_overrides(
                    exc,
                    alias_overrides,
                    core_type=type,
                    context=(f"model.core.unified_ei.layers[{layer_idx}].excitatory"),
                )
                layer["excitatory"] = PopulationConfig(**exc)
            if isinstance(inh, dict):
                inh = _apply_population_alias_overrides(
                    inh,
                    alias_overrides,
                    core_type=type,
                    context=(f"model.core.unified_ei.layers[{layer_idx}].inhibitory"),
                )
                layer["inhibitory"] = PopulationConfig(**inh)
            layers_cfg.append(EILayerConfig(**layer))

        if not layers_cfg:
            raise ValueError(
                "Unified E-I architecture requires at least one layer in unified_ei.layers"
            )

        unified_cfg["layers"] = layers_cfg
        return EINetwork(EINetworkConfig(**unified_cfg))

    # ------------------------------------------------------------------
    # Baseline RNN models (GRU, LSTM, Vanilla RNN)
    # ------------------------------------------------------------------
    if type in _BASELINE_RNN_TYPES:
        from dendritic_modeling.networks.architectures.recurrent import (
            BaselineRNN,
            BaselineRNNConfig,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating baseline RNN (type='{type}')"
            )

        raw_params = _to_plain_mapping(parameters)
        baseline_cfg = raw_params.get("baseline_rnn", raw_params)
        if not isinstance(baseline_cfg, dict):
            baseline_cfg = {}
        baseline_cfg = dict(baseline_cfg)
        baseline_cfg["cell_type"] = _BASELINE_RNN_TYPES[type]
        baseline_cfg["input_dim"] = input_dim
        return BaselineRNN(BaselineRNNConfig(**baseline_cfg))

    # ------------------------------------------------------------------
    # Named cell-type / population-graph recurrent core
    # ------------------------------------------------------------------
    if type in _POPULATION_GRAPH_TYPES:
        from dendritic_modeling.networks.architectures.recurrent import (
            CellTypeConfig,
            PopulationGraphConfig,
            PopulationGraphNetwork,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating population graph (type='{type}')"
            )

        raw_params = _to_plain_mapping(parameters)
        graph_cfg = raw_params.get("population_graph", raw_params)
        if not isinstance(graph_cfg, dict):
            graph_cfg = {}
        graph_cfg = dict(graph_cfg)
        graph_cfg["input_dim"] = input_dim
        graph_cfg["cell_types"] = [
            cell if isinstance(cell, CellTypeConfig) else CellTypeConfig(**cell)
            for cell in graph_cfg.get("cell_types", [])
        ]
        return PopulationGraphNetwork(PopulationGraphConfig(**graph_cfg))

    # ------------------------------------------------------------------
    # Multi-population E-I layers with directional same-step edges
    # ------------------------------------------------------------------
    if type in _MULTI_POPULATION_TYPES:
        from dendritic_modeling.networks.architectures.recurrent import (
            MultiPopulationLayerConfig,
            MultiPopulationNetwork,
            MultiPopulationNetworkConfig,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating multi-population E-I architecture (type='{type}')"
            )

        raw_params = _to_plain_mapping(parameters)
        mp_cfg = raw_params.get(
            "multi_population",
            raw_params.get("multi_population_ei", raw_params),
        )
        if not isinstance(mp_cfg, dict):
            mp_cfg = {}
        mp_cfg = dict(mp_cfg)
        mp_cfg["input_dim"] = input_dim
        mp_cfg["layers"] = [
            (
                layer
                if isinstance(layer, MultiPopulationLayerConfig)
                else MultiPopulationLayerConfig(**layer)
            )
            for layer in mp_cfg.get("layers", [])
        ]
        return MultiPopulationNetwork(MultiPopulationNetworkConfig(**mp_cfg))

    # ------------------------------------------------------------------
    # Legacy recurrent type aliases (removed -- use unified E-I instead)
    # ------------------------------------------------------------------
    if type in ("dendritic_rnn", "dendrnn"):
        raise ValueError(
            f"Architecture type '{type}' has been removed. "
            f"Use 'ei_unified' (or 'unified_ei', 'ei_net') with the new "
            f"EINetworkConfig format instead."
        )

    # ------------------------------------------------------------------
    # Spatial dendritic cores (flatten → EINet → project)
    # ------------------------------------------------------------------
    if type in _SPATIAL_DENDRITIC_TYPES:
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
            HierarchicalDendriticTensorMap,
        )

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating spatial dendritic core (type='{type}')"
            )

        return HierarchicalDendriticTensorMap(
            config=parameters,
            input_dim=input_dim,
            suffix_input_dim=suffix_input_dim,
        )

    # ------------------------------------------------------------------
    # Spatial dendritic conv (unfold → shared DendriNet → spatial output)
    # ------------------------------------------------------------------
    if type in _SPATIAL_DENDRITIC_CONV_TYPES:
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
            HierarchicalDendriticConv,
        )

        raw_params = _to_plain_mapping(parameters)
        spatial_cfg = _to_plain_mapping(raw_params.get("spatial", {}))
        kernel_size = spatial_cfg.get("kernel_size", 3)
        stride = spatial_cfg.get("stride", 1)
        padding = spatial_cfg.get("padding", 0)

        if input_dim is None:
            raise ValueError(
                f"input_dim required when creating spatial dendritic conv (type='{type}')"
            )

        # input_dim here is the number of input channels (C_in).
        # For spatial cores, input_dim = prefix_output_spec.channels.
        return HierarchicalDendriticConv(
            config=parameters,
            in_channels=input_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    # ------------------------------------------------------------------
    # EINet variants (require structured config)
    # ------------------------------------------------------------------
    if type in _EINET_TYPES:
        raw_params = _to_plain_mapping(parameters)
        if not ("architecture" in raw_params and "connectivity" in raw_params):
            raise ValueError(
                "EINet now requires structured config with 'architecture', 'connectivity', etc. "
                "sections. Flat parameters are no longer supported."
            )
        if input_dim is None:
            raise ValueError(
                "input_dim required when passing structured config to EINet"
            )

        recurrent_einet = _build_recurrent_einet_from_structured_config(
            type=type, raw_params=raw_params, input_dim=input_dim
        )
        if recurrent_einet is not None:
            return recurrent_einet

        # Derive shunting/additive settings from the shared type-alias table.
        synapse_mode: Optional[str] = None
        use_shunting: Optional[bool] = None
        weight_transform: Optional[str] = None
        flatten_dendrites: bool = False
        core_morphology_overrides = get_core_morphology_alias_overrides(type)

        if "use_shunting" in core_morphology_overrides:
            use_shunting = bool(core_morphology_overrides["use_shunting"])

        if "flat" in type:
            flatten_dendrites = True

        if "signed" in type:
            weight_transform = "identity"
            use_shunting = False

        if "mlp" in type:
            synapse_mode = "mlp"
            weight_transform = "identity"
            use_shunting = False

        params_for_build = _to_plain_mapping(parameters)
        params_for_build.setdefault("type", type)

        return ConfigurableEINetwork(
            config=params_for_build,
            input_dim=input_dim,
            synapse_mode=synapse_mode,
            use_shunting=use_shunting,
            weight_transform=weight_transform,
            flatten_dendrites=flatten_dendrites,
        )

    # ------------------------------------------------------------------
    # Point-neuron MLP baseline (parameter-matched, structured config)
    # ------------------------------------------------------------------
    raw_params = _to_plain_mapping(parameters)
    if type in ["point_mlp", "mlp"] and (
        "architecture" in raw_params and "connectivity" in raw_params
    ):
        if input_dim is None:
            raise ValueError("input_dim required when passing structured config to mlp")

        from dendritic_modeling.scripts.script_utils.config_utils import (
            prepare_ei_network_params,
        )

        params = prepare_ei_network_params(raw_params, input_dim)
        return PointMatchedParamMLP(**params)

    # ------------------------------------------------------------------
    # MR-style structured-config MLP baselines
    # ------------------------------------------------------------------
    if type == "total_param_mlp":
        return MatchedTotalParamMLP(parameters, input_dim)

    if type == "active_param_mlp":
        return MatchedActiveParamMLP(parameters, input_dim)

    if type == "ss_mlp":
        return SparseStructuredMLP(parameters, input_dim)

    if type == "ss_mlp_flat":
        return SparseStructuredMLP(parameters, input_dim, flatten_dendrites=True)

    # ------------------------------------------------------------------
    # All other architectures expect flat parameters
    # ------------------------------------------------------------------
    if isinstance(parameters, DictConfig):
        raise ValueError(
            f"Architecture {type} expects flat parameters, not structured config"
        )

    if type == "alexnet":
        return AlexNet(**parameters)
    elif type in ["cnn", "cnn_down", "cnndownsample", "cnn_downsample"]:
        return CNNDownsample(**parameters)
    elif type in ["cnn_up", "cnnupsample", "cnn_upsample"]:
        return CNNUpsample(**parameters)
    elif type == "mlp":
        return MLP(**parameters)
    elif type == "effectivematchedparammlp":
        return EffectiveMatchedParamMLP(**parameters)
    elif type == "mlpautoencoder":
        return MLPAutoencoder(**parameters)
    elif type == "cnnautoencoder":
        return CNNAutoencoder(**parameters)
    elif type == "mlpvariationalautoencoder":
        return MLPVariationalAutoencoder(**parameters)
    elif type == "cnnvariationalautoencoder":
        return CNNVariationalAutoencoder(**parameters)
    elif type in ["pathway_router", "router"]:
        return PathwayRouter(**parameters)
    elif type in ["identity", "none", None]:
        return Identity(**parameters)
    else:
        raise ValueError(f"Invalid architecture type: {type}")
