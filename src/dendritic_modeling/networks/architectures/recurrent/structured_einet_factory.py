"""Build structured recurrent E/I networks from structured architecture configs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from dendritic_modeling.config.conversion import (
    has_enabled_synapse_types,
    layer_value,
    normalize_sparsity_type,
    to_plain_dict,
)
from dendritic_modeling.config.legacy import normalize_transfer_config
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    get_structured_recurrent_alias_overrides,
    warn_alias_conflicts,
)
from dendritic_modeling.networks.activations import resolve_dendritic_activation

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


@dataclass(frozen=True)
class _StructuredRecurrentSections:
    recurrent_cfg: dict[str, Any]
    architecture: dict[str, Any]
    connectivity: dict[str, Any]
    structured_connectivity: Any
    transfer: dict[str, Any]
    morphology: dict[str, Any]
    sparsity: dict[str, Any]
    reactivation: dict[str, Any]
    blocklinear: dict[str, Any]
    implementation: dict[str, Any]
    synapse_types: dict[str, Any]
    dynamics: dict[str, Any]
    dendritic_spikes: dict[str, Any]
    soma_feedback: dict[str, Any]
    deepst: dict[str, Any]
    dense_to_sparse: dict[str, Any]
    indexed: dict[str, Any]


@dataclass(frozen=True)
class _RecurrentSynapseCounts:
    rec_ee: Any
    rec_ie: Any
    rec_ei: Any
    rec_ii: Any


@dataclass(frozen=True)
class _LayerPopulationSynapseCounts:
    ff_excitatory: int
    ff_inhibitory: int
    rec_excitatory: int
    rec_inhibitory: int

    @property
    def has_incoming_synapses(self) -> bool:
        return any(
            value > 0
            for value in (
                self.ff_excitatory,
                self.ff_inhibitory,
                self.rec_excitatory,
                self.rec_inhibitory,
            )
        )


@dataclass(frozen=True)
class _RecurrentFeatureFlags:
    has_rec_synapses: bool
    has_soma_feedback: bool
    has_spiking_dynamics: bool
    has_dendritic_spikes: bool
    has_synapse_types: bool

    @property
    def has_stateful_features(self) -> bool:
        return (
            self.has_soma_feedback
            or self.has_spiking_dynamics
            or self.has_dendritic_spikes
            or self.has_synapse_types
        )


@dataclass(frozen=True)
class _RecurrentReactivationOptions:
    reactivate: bool
    reactivation_type: str
    init_m: float
    init_b: float
    init_policy: str


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


def _resolve_recurrent_synapse_counts(
    *,
    connectivity: dict[str, Any],
    recurrent_cfg: dict[str, Any],
) -> _RecurrentSynapseCounts:
    def _rec_syn(new_key: str, *old_keys: str, fallback: Any = 0) -> Any:
        if new_key in connectivity:
            return connectivity[new_key]
        for key in old_keys:
            if key in recurrent_cfg:
                return recurrent_cfg[key]
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
    return _RecurrentSynapseCounts(
        rec_ee=rec_ee,
        rec_ie=rec_ie,
        rec_ei=rec_ei,
        rec_ii=rec_ii,
    )


def _has_positive_layer_value(value: Any) -> bool:
    if isinstance(value, list):
        return any(v > 0 for v in value)
    return value is not None and value > 0


def _detect_recurrent_feature_flags(
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
) -> _RecurrentFeatureFlags:
    has_rec_synapses = any(
        _has_positive_layer_value(value)
        for value in (counts.rec_ee, counts.rec_ie, counts.rec_ei, counts.rec_ii)
    )
    has_soma_feedback = bool(
        sections.soma_feedback.get(
            "enabled", sections.dynamics.get("soma_feedback_enabled", False)
        )
    )
    has_spiking_dynamics = (
        str(
            sections.dynamics.get(
                "mode", sections.dynamics.get("dynamics_mode", "rate")
            )
        ).lower()
        == "spike"
    )
    has_dendritic_spikes = bool(
        sections.dendritic_spikes.get(
            "enabled",
            sections.dynamics.get("dendritic_spikes_enabled", False),
        )
    )
    has_synapse_types = has_enabled_synapse_types(sections.synapse_types)
    return _RecurrentFeatureFlags(
        has_rec_synapses=has_rec_synapses,
        has_soma_feedback=has_soma_feedback,
        has_spiking_dynamics=has_spiking_dynamics,
        has_dendritic_spikes=has_dendritic_spikes,
        has_synapse_types=has_synapse_types,
    )


def _should_build_structured_recurrent_einet(
    sections: _StructuredRecurrentSections,
    flags: _RecurrentFeatureFlags,
) -> bool:
    explicit_enabled = sections.recurrent_cfg.get("enabled", None)
    return (
        flags.has_rec_synapses or bool(explicit_enabled) or flags.has_stateful_features
    )


def _validate_structured_recurrent_type(core_type: str) -> None:
    if core_type in {
        "dendritic_mlp",
        "flat_shunting",
        "flat_additive",
        "flat_normalized_additive",
    }:
        raise ValueError(
            f"Recurrent mode is not supported for architecture type '{core_type}'. "
            "Use 'EINet', 'dendritic_shunting', 'dendritic_additive', or "
            "'dendritic_normalized_additive'."
        )


def _require_recurrent_excitatory_sizes(
    architecture: dict[str, Any],
) -> list[Any]:
    excitatory_sizes = list(architecture.get("excitatory_layer_sizes", []))
    if not excitatory_sizes:
        raise ValueError(
            "Recurrent mode requires non-empty architecture.excitatory_layer_sizes"
        )
    return excitatory_sizes


def _validate_direct_inhibitory_stream(
    *,
    transfer: dict[str, Any],
    architecture: dict[str, Any],
    connectivity: dict[str, Any],
    excitatory_sizes: list[Any],
) -> tuple[list[Any], int]:
    inhibitory_sizes = list(architecture.get("inhibitory_layer_sizes", []))
    input_mode = int(transfer.get("input_mode", 0))
    has_inhibitory_population = any(int(size or 0) > 0 for size in inhibitory_sizes)
    has_direct_ie_synapses = any(
        int(
            _layer_value(
                connectivity.get("ie_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        )
        > 0
        for layer_idx in range(len(excitatory_sizes))
    )
    allow_direct_inhibitory_stream = bool(
        transfer.get("allow_direct_inhibitory_stream", False)
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
    return inhibitory_sizes, input_mode


def _resolve_recurrent_use_transfer(recurrent_cfg: dict[str, Any]) -> bool:
    # use_transfer: keep legacy EINet behavior by default (TransferLayer always on),
    # with backward-compatible explicit override via recurrent_ei.use_transfer.
    if "use_transfer" in recurrent_cfg:
        return bool(recurrent_cfg["use_transfer"])
    return True


def _resolve_recurrent_morphology_options(
    *,
    core_type: str,
    morphology: dict[str, Any],
) -> tuple[dict[str, object], bool, bool]:
    # Preserve legacy EINet type aliases for shunting/additive behavior.
    core_morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        morphology,
        core_morphology_overrides,
        context="model.core.morphology",
    )
    structured_alias_overrides = get_structured_recurrent_alias_overrides(core_type)

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
    return structured_alias_overrides, use_shunting, use_additive_normalization


def _resolve_recurrent_reactivation_options(
    *,
    core_type: str,
    reactivation: dict[str, Any],
    structured_alias_overrides: dict[str, object],
) -> _RecurrentReactivationOptions:
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
            core_type,
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
    return _RecurrentReactivationOptions(
        reactivate=reactivate,
        reactivation_type=reactivation_type,
        init_m=reactivation_init_m,
        init_b=reactivation_init_b,
        init_policy=reactivation_init_policy,
    )


def _resolve_transfer_inhibitory_mode(
    *,
    transfer: dict[str, Any],
    recurrent_cfg: dict[str, Any],
) -> str:
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
    return transfer_inhibitory_mode


def _layer_recurrence_enabled(
    *,
    layer_idx: int,
    explicit_recurrent_layers: Any,
    counts: _RecurrentSynapseCounts,
) -> bool:
    if explicit_recurrent_layers is not None:
        return bool(_layer_value(explicit_recurrent_layers, layer_idx, default=True))
    layer_rec_total = sum(
        int(_layer_value(value, layer_idx, default=0))
        for value in (counts.rec_ee, counts.rec_ie, counts.rec_ei, counts.rec_ii)
    )
    return layer_rec_total > 0


def _direct_ff_inhibitory_enabled(
    *,
    input_mode: int,
    transfer_inhibitory_mode: str,
    layer_idx: int,
) -> bool:
    if input_mode != 1:
        return False
    if transfer_inhibitory_mode == "all":
        return True
    if transfer_inhibitory_mode == "first":
        return layer_idx == 0
    return False


def _resolve_layer_inhibitory_synapse_counts(
    *,
    connectivity: dict[str, Any],
    counts: _RecurrentSynapseCounts,
    layer_idx: int,
) -> _LayerPopulationSynapseCounts:
    """Resolve feedforward/recurrent synapse counts for one inhibitory population."""
    return _LayerPopulationSynapseCounts(
        ff_excitatory=int(
            _layer_value(
                connectivity.get("ei_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        ff_inhibitory=int(
            _layer_value(
                connectivity.get("ii_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        rec_excitatory=int(_layer_value(counts.rec_ei, layer_idx, default=0)),
        rec_inhibitory=int(_layer_value(counts.rec_ii, layer_idx, default=0)),
    )


def _resolve_layer_excitatory_synapse_counts(
    *,
    connectivity: dict[str, Any],
    counts: _RecurrentSynapseCounts,
    layer_idx: int,
) -> _LayerPopulationSynapseCounts:
    """Resolve feedforward/recurrent synapse counts for one excitatory population."""
    return _LayerPopulationSynapseCounts(
        ff_excitatory=int(
            _layer_value(
                connectivity.get("ee_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        ff_inhibitory=int(
            _layer_value(
                connectivity.get("ie_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        rec_excitatory=int(_layer_value(counts.rec_ee, layer_idx, default=0)),
        rec_inhibitory=int(_layer_value(counts.rec_ie, layer_idx, default=0)),
    )


def _should_build_inhibitory_population(
    *,
    n_inh: int,
    synapses: _LayerPopulationSynapseCounts,
) -> bool:
    # Positive inhibitory sizes without incoming synapses preserve legacy direct-stream
    # behavior: the size can be present in the config without constructing a population.
    return n_inh > 0 and synapses.has_incoming_synapses


def _structured_population_sparsity_kwargs(
    *,
    sparsity: dict[str, Any],
    dense_to_sparse: dict[str, Any],
    indexed: dict[str, Any],
) -> dict[str, Any]:
    """Translate structured sparsity config into PopulationConfig kwargs."""
    return {
        "topk_init_method": str(sparsity.get("init_method", "xavier_normal")),
        "topk_noise_level": float(sparsity.get("noise_level", 0.0)),
        "topk_type": _normalize_sparsity_type(sparsity.get("type", "standard")),
        "topk_weight_norm_order": sparsity.get("weight_norm_order"),
        "topk_gamma": float(sparsity.get("gamma", 1.0)),
        "topk_temperature": float(sparsity.get("temperature", 0.0)),
        "topk_ultrafast": bool(sparsity.get("ultrafast", False)),
        "topk_strategy": str(sparsity.get("gradient_scaling", "none")),
        "dense_to_sparse_initial_density": float(
            dense_to_sparse.get("initial_density", 1.0)
        ),
        "dense_to_sparse_initial_k": dense_to_sparse.get("initial_k"),
        "dense_to_sparse_start_step": int(dense_to_sparse.get("start_step", 0)),
        "dense_to_sparse_end_step": int(dense_to_sparse.get("end_step", 1000)),
        "dense_to_sparse_update_interval": int(
            dense_to_sparse.get("update_interval", 1)
        ),
        "dense_to_sparse_schedule": str(dense_to_sparse.get("schedule", "cubic")),
        "dense_to_sparse_freeze_on_end": bool(
            dense_to_sparse.get("freeze_on_end", False)
        ),
        "dense_to_sparse_advance_on_forward": bool(
            dense_to_sparse.get("advance_on_forward", True)
        ),
        "dense_to_sparse_prune_metric": str(
            dense_to_sparse.get("prune_metric", "weight")
        ),
        "indexed_seed": indexed.get("seed", sparsity.get("seed")),
        "indexed_candidate_size": indexed.get("candidate_size"),
        "indexed_selection": str(indexed.get("selection", "standard")),
        "indexed_output_chunk_size": int(indexed.get("output_chunk_size", 2048)),
        "indexed_rewire_frequency": int(indexed.get("rewire_frequency", 100)),
        "indexed_rewire_quantile": float(indexed.get("rewire_quantile", 0.05)),
        "indexed_rewire_until_step": indexed.get("rewire_until_step"),
    }


def _structured_population_deepst_kwargs(deepst: dict[str, Any]) -> dict[str, Any]:
    """Translate structured DeepST config into PopulationConfig kwargs."""
    return {
        "excitatory_target_density": float(
            deepst.get("excitatory_target_density", 0.1)
        ),
        "inhibitory_target_density": float(
            deepst.get("inhibitory_target_density", 0.1)
        ),
        "use_noise": bool(deepst.get("use_noise", True)),
        "sigma": float(deepst.get("sigma", 0.05)),
        "rewiring_mode": str(deepst.get("rewiring_mode", "global")),
        "rewire_frequency": int(deepst.get("rewire_frequency", 1)),
        "synapses_per_branch": deepst.get("synapses_per_branch"),
        "freeze_excitatory_connectivity": bool(
            deepst.get("freeze_excitatory_connectivity", False)
        ),
        "freeze_inhibitory_connectivity": bool(
            deepst.get("freeze_inhibitory_connectivity", False)
        ),
        "init_method": str(deepst.get("init_method", "xavier_normal")),
        "weight_threshold": float(deepst.get("weight_threshold", 1e-6)),
    }


def _structured_population_adaptive_init_kwargs(
    implementation: dict[str, Any],
) -> dict[str, Any]:
    """Translate adaptive-initialization implementation settings."""
    return {
        "adaptive_initialization": bool(
            implementation.get("adaptive_initialization", False)
        ),
        "adaptive_initialization_policy": str(
            implementation.get(
                "adaptive_initialization_policy",
                "preserve_shunting_center",
            )
        ),
        "adaptive_target_conductance": float(
            implementation.get("adaptive_target_conductance", 5.0)
        ),
    }


def _structured_population_dynamics_kwargs(
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    """Translate somatic spiking dynamics config into PopulationConfig kwargs."""
    return {
        "dynamics_mode": str(
            dynamics.get("mode", dynamics.get("dynamics_mode", "rate"))
        ),
        "spike_threshold": float(
            dynamics.get("threshold", dynamics.get("spike_threshold", 1.0))
        ),
        "spike_reset": float(dynamics.get("reset", dynamics.get("spike_reset", 0.0))),
        "spike_tau": float(dynamics.get("tau", dynamics.get("spike_tau", 20.0))),
        "spike_refractory_steps": int(
            dynamics.get("refractory_steps", dynamics.get("spike_refractory_steps", 0))
        ),
        "spike_surrogate_beta": float(
            dynamics.get("surrogate_beta", dynamics.get("spike_surrogate_beta", 10.0))
        ),
        "spike_readout": str(
            dynamics.get("readout", dynamics.get("spike_readout", "spikes"))
        ),
        "spike_readout_tau": float(
            dynamics.get("readout_tau", dynamics.get("spike_readout_tau", 20.0))
        ),
    }


def _structured_population_dendritic_spike_kwargs(
    *,
    dendritic_spikes: dict[str, Any],
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    """Translate dendritic-spike config into PopulationConfig kwargs."""
    return {
        "dendritic_spikes_enabled": bool(
            dendritic_spikes.get(
                "enabled",
                dynamics.get("dendritic_spikes_enabled", False),
            )
        ),
        "dendritic_spike_mode": str(
            dendritic_spikes.get(
                "mode",
                dynamics.get("dendritic_spike_mode", "plateau"),
            )
        ),
        "dendritic_spike_levels": dendritic_spikes.get(
            "levels",
            dynamics.get("dendritic_spike_levels", "non_soma"),
        ),
        "dendritic_spike_threshold": float(
            dendritic_spikes.get(
                "threshold",
                dynamics.get("dendritic_spike_threshold", 0.7),
            )
        ),
        "dendritic_spike_plateau_amplitude": float(
            dendritic_spikes.get(
                "plateau_amplitude",
                dynamics.get("dendritic_spike_plateau_amplitude", 1.0),
            )
        ),
        "dendritic_spike_plateau_tau": float(
            dendritic_spikes.get(
                "plateau_tau",
                dynamics.get("dendritic_spike_plateau_tau", 20.0),
            )
        ),
        "dendritic_spike_refractory_steps": int(
            dendritic_spikes.get(
                "refractory_steps",
                dynamics.get("dendritic_spike_refractory_steps", 0),
            )
        ),
        "dendritic_spike_surrogate_beta": float(
            dendritic_spikes.get(
                "surrogate_beta",
                dynamics.get("dendritic_spike_surrogate_beta", 10.0),
            )
        ),
        "dendritic_spike_propagation": str(
            dendritic_spikes.get(
                "propagation",
                dynamics.get("dendritic_spike_propagation", "additive"),
            )
        ),
    }


def _structured_population_soma_feedback_kwargs(
    *,
    soma_feedback: dict[str, Any],
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    """Translate soma-feedback config into PopulationConfig kwargs."""
    return {
        "soma_feedback_enabled": bool(
            soma_feedback.get(
                "enabled",
                dynamics.get("soma_feedback_enabled", False),
            )
        ),
        "soma_feedback_mode": str(
            soma_feedback.get(
                "mode",
                dynamics.get("soma_feedback_mode", "additive"),
            )
        ),
        "soma_feedback_source": str(
            soma_feedback.get(
                "source",
                dynamics.get("soma_feedback_source", "output"),
            )
        ),
        "soma_feedback_levels": soma_feedback.get(
            "levels",
            dynamics.get("soma_feedback_levels", "non_soma"),
        ),
        "soma_feedback_strength": float(
            soma_feedback.get(
                "strength",
                dynamics.get("soma_feedback_strength", 0.1),
            )
        ),
        "soma_feedback_learnable_strength": bool(
            soma_feedback.get(
                "learnable_strength",
                dynamics.get("soma_feedback_learnable_strength", False),
            )
        ),
        "soma_feedback_per_level": bool(
            soma_feedback.get(
                "per_level",
                dynamics.get("soma_feedback_per_level", True),
            )
        ),
        "soma_feedback_init_std": float(
            soma_feedback.get(
                "init_std",
                dynamics.get("soma_feedback_init_std", 0.0),
            )
        ),
        "soma_feedback_reversal": float(
            soma_feedback.get(
                "reversal",
                dynamics.get("soma_feedback_reversal", 1.0),
            )
        ),
    }


def _structured_population_morphology_kwargs(
    *,
    morphology: dict[str, Any],
    use_shunting: bool,
    use_additive_normalization: bool,
) -> dict[str, Any]:
    """Translate morphology config into PopulationConfig kwargs."""
    return {
        "weight_transform": str(morphology.get("weight_transform", "softplus")),
        "use_shunting": use_shunting,
        "use_additive_normalization": use_additive_normalization,
        "allow_self_recurrence": bool(morphology.get("allow_self_recurrence", True)),
        "somatic_synapses": bool(morphology.get("somatic_synapses", False)),
        "dbl_init_method": str(
            morphology.get("dbl_init_method", "analytical_expectation")
        ),
    }


def _structured_population_reactivation_kwargs(
    *,
    reactivation: dict[str, Any],
    reactivation_options: _RecurrentReactivationOptions,
) -> dict[str, Any]:
    """Translate reactivation config into PopulationConfig kwargs."""
    return {
        "reactivate": reactivation_options.reactivate,
        "reactivation_type": reactivation_options.reactivation_type,
        "reactivation_init_m": reactivation_options.init_m,
        "reactivation_init_b": reactivation_options.init_b,
        "reactivation_init_policy": reactivation_options.init_policy,
        "reactivation_occupancy_quantile_low": reactivation.get(
            "occupancy_quantile_low"
        ),
        "reactivation_occupancy_quantile_high": reactivation.get(
            "occupancy_quantile_high"
        ),
        "reactivation_occupancy_target_low": reactivation.get("occupancy_target_low"),
        "reactivation_occupancy_target_high": reactivation.get("occupancy_target_high"),
        "reactivation_calibration_min_quantile_width": float(
            reactivation.get("calibration_min_quantile_width", 1e-3)
        ),
        "reactivation_calibration_max_m": float(
            reactivation.get("calibration_max_m", 50.0)
        ),
        "reactivation_calibration_revert_on_invalid": bool(
            reactivation.get("calibration_revert_on_invalid", True)
        ),
        "reactivation_sigma_aware_k": float(reactivation.get("sigma_aware_k", 0.25)),
        "reactivation_strategy": str(reactivation.get("gradient_scaling", "none")),
    }


def _structured_population_blocklinear_kwargs(
    *,
    blocklinear: dict[str, Any],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    """Translate BlockLinear implementation settings into PopulationConfig kwargs."""
    return {
        "blocklinear_strategy": str(blocklinear.get("gradient_scaling", "none")),
        "efficient_blocklinear": bool(blocklinear.get("efficient", False)),
        "print_hooks": bool(implementation.get("print_hooks", False)),
    }


def _structured_population_timing_kwargs(
    *,
    recurrent_cfg: dict[str, Any],
    tau_base: float,
    tau_ratio: float,
) -> dict[str, Any]:
    """Translate recurrent timing config into PopulationConfig kwargs."""
    return {
        "level_taus": list(recurrent_cfg.get("level_taus", [])),
        "tau_base": tau_base,
        "tau_ratio": tau_ratio,
        "learnable_tau": bool(recurrent_cfg.get("learnable_tau", False)),
        "epsilon": float(recurrent_cfg.get("epsilon", 1e-8)),
    }


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

    recurrent_cfg = sections.recurrent_cfg
    architecture = sections.architecture
    connectivity = sections.connectivity
    transfer = sections.transfer
    morphology = sections.morphology
    reactivation = sections.reactivation
    implementation = sections.implementation
    has_spiking_dynamics = flags.has_spiking_dynamics
    has_synapse_types = flags.has_synapse_types

    _validate_structured_recurrent_type(type)
    excitatory_sizes = _require_recurrent_excitatory_sizes(architecture)
    inhibitory_sizes, input_mode = _validate_direct_inhibitory_stream(
        transfer=transfer,
        architecture=architecture,
        connectivity=connectivity,
        excitatory_sizes=excitatory_sizes,
    )
    use_transfer = _resolve_recurrent_use_transfer(recurrent_cfg)
    (
        structured_alias_overrides,
        use_shunting,
        use_additive_normalization,
    ) = _resolve_recurrent_morphology_options(
        core_type=type,
        morphology=morphology,
    )
    reactivation_options = _resolve_recurrent_reactivation_options(
        core_type=type,
        reactivation=reactivation,
        structured_alias_overrides=structured_alias_overrides,
    )

    # Per-layer recurrence: infer from synapse counts, backward compat with explicit flag
    explicit_recurrent_layers = recurrent_cfg.get(
        "recurrent_layers", recurrent_cfg.get("recurrent_per_layer", None)
    )

    dt_value = recurrent_cfg.get("dt", 1.0)

    # transfer_inhibitory_mode: new location first, old fallback
    transfer_inhibitory_mode = _resolve_transfer_inhibitory_mode(
        transfer=transfer,
        recurrent_cfg=recurrent_cfg,
    )

    layers: list[EILayerConfig] = []
    for layer_idx, n_exc in enumerate(excitatory_sizes):
        n_inh = int(_layer_value(inhibitory_sizes, layer_idx, default=0))

        # Per-layer recurrence
        rec_enabled = _layer_recurrence_enabled(
            layer_idx=layer_idx,
            explicit_recurrent_layers=explicit_recurrent_layers,
            counts=counts,
        )

        dt = float(_layer_value(dt_value, layer_idx, default=1.0))

        # Per-layer tau_base / tau_ratio
        tau_base = float(
            _layer_value(recurrent_cfg.get("tau_base", 50.0), layer_idx, default=50.0)
        )
        tau_ratio = float(
            _layer_value(recurrent_cfg.get("tau_ratio", 3.0), layer_idx, default=3.0)
        )

        exc_cfg = _build_structured_excitatory_population_config(
            population_config_cls=PopulationConfig,
            sections=sections,
            counts=counts,
            layer_idx=layer_idx,
            n_exc=n_exc,
            use_shunting=use_shunting,
            use_additive_normalization=use_additive_normalization,
            reactivation_options=reactivation_options,
            tau_base=tau_base,
            tau_ratio=tau_ratio,
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
            inh_cfg = replace(
                exc_cfg,
                n_neurons=n_inh,
                branch_factors=list(
                    architecture.get("inhibitory_branch_factors", [1]) or [1]
                ),
                ff_excitatory_synapses=inh_synapses.ff_excitatory,
                ff_inhibitory_synapses=inh_synapses.ff_inhibitory,
                rec_excitatory_synapses=inh_synapses.rec_excitatory,
                rec_inhibitory_synapses=inh_synapses.rec_inhibitory,
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

        direct_ff_inh = _direct_ff_inhibitory_enabled(
            input_mode=input_mode,
            transfer_inhibitory_mode=transfer_inhibitory_mode,
            layer_idx=layer_idx,
        )

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
    transfer_params = normalize_transfer_config(transfer_params)

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


__all__ = ["_build_recurrent_einet_from_structured_config"]
