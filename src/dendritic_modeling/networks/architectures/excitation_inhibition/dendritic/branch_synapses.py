"""Sparse synapse layer construction for dendritic branch layers."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
    BUILTIN_SPARSE_LAYER_TYPES,
    get_sparse_layer,
    get_sparse_layer_class,
    normalize_sparse_layer_type,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset


def _indexed_synapse_seed(
    owner: Any,
    *,
    pathway: str,
) -> int | None:
    """Derive a stable, independent indexed seed for one synaptic pathway.

    ``synapse_type`` alone is not a sufficient namespace: feedforward and
    recurrent excitation previously received the same seed (as did the two
    inhibitory pathways).  The population-network builder namespaces the base
    seed by layer and target population; this final offset separates pathway
    and dendritic depth without depending on Python's randomized ``hash``.
    """
    if owner.indexed_seed is None:
        return None
    return (
        int(owner.indexed_seed)
        + stable_seed_offset("branch_synapse", pathway, int(owner.layer_idx))
    ) % ((1 << 63) - 1)


def _pathway_gamma(owner: Any, synapse_type: str) -> float:
    """Use inhibitory gamma only for inhibitory sparse pathways."""
    return owner.topk_gamma if synapse_type == "inh" else 1.0


def _indexed_runtime_kwargs(owner: Any) -> dict[str, Any]:
    """Return config-controlled indexed storage and execution options."""
    return {
        "index_dtype": owner.indexed_index_dtype,
        "workspace_mb": owner.indexed_workspace_mb,
        "cache_transformed_weights": owner.indexed_cache_transformed_weights,
        "recompute_backward": owner.indexed_recompute_backward,
        "projection_backend": owner.indexed_projection_backend,
        "persistent_indices": owner.indexed_persistent_indices,
        "init_mode": owner.indexed_init_mode,
        "support_group_rows": getattr(owner, "indexed_support_group_rows", 1),
        "support_col_block": getattr(owner, "indexed_support_col_block", 1),
    }


# Sparse layer types whose constructors receive no TopK-score noise.
_NOISE_FREE_LAYER_TYPES = frozenset(
    {"indexed", "indexed_rewire", "indexed_release", "deepst", "credit_deepst"}
)
# Sparse layer types that never normalize their weights.
_UNNORMALIZED_LAYER_TYPES = frozenset(
    {"stochastic", "variance", "deepst", "credit_deepst"}
)
_DEFAULT_DEEPST_INIT_METHOD = "xavier_normal"

logger = logging.getLogger(__name__)

_INDEXED_FAMILY = frozenset(
    {"indexed", "indexed_rewire", "indexed_release", "indexed_dynamic"}
)
# IndexedSynapseConfig fields and the layer types that read them. The seed is
# not listed: builders namespace it for every population, and DeepST reads it.
_INDEXED_FIELD_TYPES: dict[str, frozenset[str]] = {
    "rewire_frequency": frozenset({"indexed_rewire"}),
    "rewire_quantile": frozenset({"indexed_rewire"}),
    "rewire_until_step": frozenset({"indexed_rewire"}),
    "candidate_size": frozenset({"indexed_dynamic"}),
    "selection": frozenset({"indexed_dynamic"}),
    "support_group_rows": _INDEXED_FAMILY,
    "support_col_block": _INDEXED_FAMILY,
    "output_chunk_size": _INDEXED_FAMILY,
    "index_dtype": _INDEXED_FAMILY,
    "workspace_mb": _INDEXED_FAMILY,
    "cache_transformed_weights": _INDEXED_FAMILY,
    "recompute_backward": _INDEXED_FAMILY,
    "projection_backend": _INDEXED_FAMILY,
    "persistent_indices": _INDEXED_FAMILY,
    "init_mode": _INDEXED_FAMILY,
    "release_p_min": frozenset({"indexed_release"}),
    "release_p_max": frozenset({"indexed_release"}),
    "release_weight_scale": frozenset({"indexed_release"}),
    "release_impact_decay": frozenset({"indexed_release"}),
    "release_impact_metric": frozenset({"indexed_release"}),
    "release_max_conductance": frozenset({"indexed_release"}),
    "release_track_source_activity": frozenset({"indexed_release"}),
}
# Fields that only set how an indexed layer executes or stores its indices.
_INDEXED_EXECUTION_FIELDS = frozenset(
    {
        "output_chunk_size",
        "index_dtype",
        "workspace_mb",
        "cache_transformed_weights",
        "recompute_backward",
        "projection_backend",
        "persistent_indices",
        "init_mode",
    }
)
_DENSE_TO_SPARSE_TYPES = frozenset({"dense_to_sparse", "annealed_topk"})
_REPORTED_EXECUTION_OPTIONS: set[tuple[str, str]] = set()


def _is_plugin_indexed_type(layer_type: str) -> bool:
    """Whether a plugin registered ``layer_type`` as a fixed-index layer subclass.

    Such a layer is built like ``indexed`` (seed, chunk size, storage and
    execution options) and reads the indexed options that ``indexed`` reads.
    """
    return layer_type not in BUILTIN_SPARSE_LAYER_TYPES and issubclass(
        get_sparse_layer_class(layer_type), IndexedSparseLinear
    )


def _option_reader_type(layer_type: str) -> str:
    """The layer type whose option set ``layer_type`` reads."""
    return "indexed" if _is_plugin_indexed_type(layer_type) else layer_type


def _unread_family_options(owner: Any, layer_type: str) -> tuple[list[str], list[str]]:
    """Non-default indexed and dense-to-sparse options ``layer_type`` does not read.

    Returns the options that would change the model and the execution-only ones.
    """
    config = getattr(owner, "synapse_config", None)
    behavior, execution = [], []
    indexed = getattr(config, "indexed", None)
    if indexed is not None:
        for item in dataclasses.fields(indexed):
            readers = _INDEXED_FIELD_TYPES.get(item.name)
            value = getattr(indexed, item.name)
            if readers is None or layer_type in readers or value == item.default:
                continue
            name = (
                item.name
                if item.name.startswith("release_")
                else f"indexed_{item.name}"
            )
            target = execution if item.name in _INDEXED_EXECUTION_FIELDS else behavior
            target.append(f"{name}={value!r}")
    dense = getattr(config, "dense_to_sparse", None)
    if dense is not None and layer_type not in _DENSE_TO_SPARSE_TYPES:
        for item in dataclasses.fields(dense):
            value = getattr(dense, item.name)
            if value != item.default:
                behavior.append(f"dense_to_sparse_{item.name}={value!r}")
    return behavior, execution


def _unread_synapses_per_branch(owner: Any, *, layer_type: str, K) -> str | None:
    """Describe a DeepST ``synapses_per_branch`` that the layer does not use.

    Only ``constant_branch`` DeepST reads it, and only for a bank built without
    a synapse count. Dendritic branch layers always pass the pathway's count
    ``K``, which is then the per-branch budget, so a written value other than
    that count is never used.
    """
    value = getattr(owner, "synapses_per_branch", None)
    if value is None:
        return None
    deepst = layer_type in {"deepst", "credit_deepst"}
    if deepst and owner.rewiring_mode == "constant_branch":
        if K is None or value == K:
            return None
        reason = (
            "constant_branch keeps each pathway's own synapse count active per "
            f"branch, {K} for this bank, so remove it"
        )
    elif deepst:
        reason = (
            f"rewiring_mode {owner.rewiring_mode!r} keeps the target density of "
            "the allowed contacts, so remove it"
        )
    else:
        reason = "only DeepST synapses read it, so remove it"
    return (
        f"sparsity.deepst.synapses_per_branch={value!r} (population "
        f"synapses_per_branch: {reason})"
    )


def _reject_unapplied_sparse_options(
    owner: Any,
    *,
    layer_type: str,
    noise_level: float,
    K=None,
) -> None:
    """Fail on sparse-synapse options that the selected layer type ignores.

    ``sparsity.init_method`` (``topk_init_method``) is deliberately not checked:
    every ``dbl_init_method`` overwrites the constructed weights, so it only
    sets the discarded constructor draw, whose consumption of the global
    random stream depends on the method (``indexed_rewire`` regrows contacts
    with the pruned weight). Branch-layer DeepST weights live in log space,
    where every registered ``deepst.init_method`` regrows contacts at
    ``weight_threshold``; only ``null`` (uniform regrowth) differs.
    """
    unapplied = []
    options = getattr(owner, "topk_options", None) or {}
    if options and layer_type in BUILTIN_SPARSE_LAYER_TYPES:
        unapplied.append(
            f"sparsity.options={dict(options)!r} (only plugin sparse layer types "
            "take constructor options)"
        )
    requested = layer_type
    layer_type = _option_reader_type(layer_type)
    unread_budget = _unread_synapses_per_branch(owner, layer_type=layer_type, K=K)
    if unread_budget is not None:
        unapplied.append(unread_budget)
    if float(noise_level or 0.0) != 0.0 and layer_type in _NOISE_FREE_LAYER_TYPES:
        unapplied.append(f"sparsity.noise_level={noise_level!r}")
    if layer_type in _UNNORMALIZED_LAYER_TYPES:
        if owner.topk_weight_norm_order is not None:
            unapplied.append(
                f"sparsity.weight_norm_order={owner.topk_weight_norm_order!r}"
            )
        if owner.topk_gamma is not None and float(owner.topk_gamma) != 1.0:
            unapplied.append(f"sparsity.gamma={owner.topk_gamma!r}")
    variance_score = getattr(owner, "topk_variance_score", "absolute")
    if layer_type != "variance" and variance_score != "absolute":
        unapplied.append(f"sparsity.variance_score={variance_score!r}")
    if layer_type in {"deepst", "credit_deepst"} and owner.init_method not in (
        None,
        "None",
        _DEFAULT_DEEPST_INIT_METHOD,
    ):
        unapplied.append(
            f"sparsity.deepst.init_method={owner.init_method!r} (log-space DeepST "
            "regrowth is identical for every registered method; use null for "
            "uniform regrowth)"
        )
    behavior, execution = _unread_family_options(owner, layer_type)
    unapplied.extend(behavior)
    if unapplied:
        raise ValueError(
            f"sparsity type {requested!r} does not apply " + "; ".join(unapplied)
        )
    for option in execution:
        if (requested, option) not in _REPORTED_EXECUTION_OPTIONS:
            _REPORTED_EXECUTION_OPTIONS.add((requested, option))
            logger.info(
                "sparsity type %r does not read the indexed execution option %s",
                requested,
                option,
            )


def create_branch_sparse_layer(
    owner: Any,
    *,
    in_features: int,
    out_features: int,
    K: int | None,
    init_method: str,
    noise_level: float,
    synapse_type: str = "exc",
    pathway: str | None = None,
    forbidden_input_index_per_output: Any = None,
    connection_indices: Any = None,
    connection_mask: Any = None,
):
    """Create a sparse synapse layer using a branch layer's configuration."""
    pathway = pathway or synapse_type
    requested_K = K
    # Keep invalid sweep configs runnable by capping requested synapse count to
    # available presynaptic features for this layer.
    if K is not None and K > in_features:
        K = in_features

    layer_type = normalize_sparse_layer_type(owner.topk_type)
    _reject_unapplied_sparse_options(
        owner,
        layer_type=layer_type,
        noise_level=noise_level,
        K=requested_K,
    )
    if connection_indices is not None and layer_type != "indexed":
        raise ValueError(
            "Direct structured connection indices currently require "
            "sparsity.type='indexed'. Adaptive rewiring would leave the "
            "configured spatial regions."
        )

    if layer_type == "standard":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

    # Alternative strategies only incur overhead if explicitly requested.
    if layer_type == "stochastic":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            temperature=getattr(owner, "topk_temperature", 0.5),
            ultrafast=getattr(owner, "topk_ultrafast", True),
            weight_transform=owner.weight_transform,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

    if layer_type == "variance":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            weight_transform=owner.weight_transform,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
            variance_score=getattr(owner, "topk_variance_score", "absolute"),
        )

    if layer_type in {"dense_to_sparse", "annealed_topk"}:
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
            initial_density=owner.dense_to_sparse_initial_density,
            initial_k=owner.dense_to_sparse_initial_k,
            start_step=owner.dense_to_sparse_start_step,
            end_step=owner.dense_to_sparse_end_step,
            update_interval=owner.dense_to_sparse_update_interval,
            schedule=owner.dense_to_sparse_schedule,
            freeze_on_end=owner.dense_to_sparse_freeze_on_end,
            advance_on_forward=owner.dense_to_sparse_advance_on_forward,
            prune_metric=owner.dense_to_sparse_prune_metric,
        )

    if layer_type in {"deepst", "credit_deepst"}:
        constant_branch = owner.rewiring_mode == "constant_branch"
        resolved_synapses_per_branch = (
            K if constant_branch and K is not None else owner.synapses_per_branch
        )
        resolved_target_density = (
            float(K) / int(in_features)
            if constant_branch and K is not None
            else (
                owner.excitatory_target_density
                if synapse_type == "exc"
                else owner.inhibitory_target_density
            )
        )
        credit_kwargs = {}
        if layer_type == "credit_deepst":
            credit_kwargs = {
                "credit_trace_decay": owner.credit_trace_decay,
                "credit_candidate_pool_size": owner.credit_candidate_pool_size,
                "credit_turnover_fraction": owner.credit_turnover_fraction,
                "credit_weak_active_pool_fraction": (
                    owner.credit_weak_active_pool_fraction
                ),
                "credit_min_observations": owner.credit_min_observations,
                "credit_swap_margin": owner.credit_swap_margin,
                "credit_force_turnover": owner.credit_force_turnover,
                "credit_selection": owner.credit_selection,
                "credit_warmup_steps": owner.credit_warmup_steps,
            }
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            target_density=resolved_target_density,
            sigma=owner.sigma,
            use_noise=owner.use_noise,
            rewiring_mode=owner.rewiring_mode,
            synapses_per_branch=resolved_synapses_per_branch,
            param_space="log",
            init_method=owner.init_method,
            freeze_connectivity=(
                owner.freeze_excitatory_connectivity
                if synapse_type == "exc"
                else owner.freeze_inhibitory_connectivity
            ),
            weight_threshold=owner.weight_threshold,
            weight_transform=owner.weight_transform,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
            rewire_frequency=owner.rewire_frequency,
            seed=_indexed_synapse_seed(owner, pathway=pathway),
            **credit_kwargs,
        )

    if layer_type in {"indexed", "indexed_rewire", "indexed_release"}:
        extra_kwargs = {}
        if layer_type == "indexed_rewire":
            extra_kwargs = {
                "rewire_frequency": owner.indexed_rewire_frequency,
                "rewire_quantile": owner.indexed_rewire_quantile,
                "rewire_until_step": owner.indexed_rewire_until_step,
            }
        if layer_type == "indexed_release":
            cfg = owner.synapse_config.indexed
            extra_kwargs = {
                "release_p_min": cfg.release_p_min,
                "release_p_max": cfg.release_p_max,
                "release_weight_scale": cfg.release_weight_scale,
                "release_impact_decay": cfg.release_impact_decay,
                "release_impact_metric": cfg.release_impact_metric,
                "release_max_conductance": cfg.release_max_conductance,
                "release_track_source_activity": cfg.release_track_source_activity,
                "signal_rounding": owner.signal_rounding,
            }
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            connection_indices=connection_indices,
            connection_mask=connection_mask,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            seed=_indexed_synapse_seed(owner, pathway=pathway),
            output_chunk_size=owner.indexed_output_chunk_size,
            **_indexed_runtime_kwargs(owner),
            **extra_kwargs,
        )

    if layer_type == "indexed_dynamic":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            candidate_size=owner.indexed_candidate_size,
            selection=owner.indexed_selection,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            temperature=owner.topk_temperature,
            ultrafast=owner.topk_ultrafast,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            connection_mask=connection_mask,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            seed=_indexed_synapse_seed(owner, pathway=pathway),
            output_chunk_size=owner.indexed_output_chunk_size,
            **_indexed_runtime_kwargs(owner),
        )

    # Plugin types: a fixed-index subclass is built like ``indexed``; any other
    # type receives the generic sparse options. Both receive their own options.
    options = dict(getattr(owner, "topk_options", None) or {})
    if _is_plugin_indexed_type(layer_type):
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            connection_mask=connection_mask,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            seed=_indexed_synapse_seed(owner, pathway=pathway),
            output_chunk_size=owner.indexed_output_chunk_size,
            **_indexed_runtime_kwargs(owner),
            **options,
        )
    return get_sparse_layer(
        layer_type,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method=init_method,
        noise_level=noise_level,
        weight_transform=owner.weight_transform,
        weight_norm_order=owner.topk_weight_norm_order,
        gamma=_pathway_gamma(owner, synapse_type),
        forbidden_input_index_per_output=forbidden_input_index_per_output,
        connection_mask=connection_mask,
        **options,
    )


def create_registered_branch_sparse_layer(
    owner: Any,
    gradient_scaler: Any,
    *,
    in_features: int,
    out_features: int,
    K: int | None,
    init_method: str,
    noise_level: float,
    synapse_type: str = "exc",
    pathway: str | None = None,
    forbidden_input_index_per_output: Any = None,
    connection_indices: Any = None,
    connection_mask: Any = None,
):
    """Create a sparse branch pathway and register its dynamic gradient scaler."""
    layer = create_branch_sparse_layer(
        owner,
        in_features=in_features,
        out_features=out_features,
        K=K,
        init_method=init_method,
        noise_level=noise_level,
        synapse_type=synapse_type,
        pathway=pathway,
        forbidden_input_index_per_output=forbidden_input_index_per_output,
        connection_indices=connection_indices,
        connection_mask=connection_mask,
    )
    gradient_scaler.register_topk_dynamic(layer)
    return layer


__all__ = [
    "create_branch_sparse_layer",
    "create_registered_branch_sparse_layer",
]
