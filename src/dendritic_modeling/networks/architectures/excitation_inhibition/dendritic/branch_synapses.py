"""Sparse synapse layer construction for dendritic branch layers."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
    get_sparse_layer,
    normalize_sparse_layer_type,
)


def _indexed_synapse_seed(owner: Any, synapse_type: str) -> int | None:
    """Derive the deterministic per-layer/per-pathway indexed sparse seed."""
    if owner.indexed_seed is None:
        return None
    synapse_offset = {"exc": 0, "inh": 1}.get(synapse_type, 2)
    return int(owner.indexed_seed) + (1009 * int(owner.layer_idx)) + synapse_offset


def _pathway_gamma(owner: Any, synapse_type: str) -> float:
    """Use inhibitory gamma only for inhibitory sparse pathways."""
    return owner.topk_gamma if synapse_type == "inh" else 1.0


def create_branch_sparse_layer(
    owner: Any,
    *,
    in_features: int,
    out_features: int,
    K: int | None,
    init_method: str,
    noise_level: float,
    synapse_type: str = "exc",
    forbidden_input_index_per_output: Any = None,
    connection_mask: Any = None,
):
    """Create a sparse synapse layer using a branch layer's configuration."""
    # Keep invalid sweep configs runnable by capping requested synapse count to
    # available presynaptic features for this layer.
    if K is not None and K > in_features:
        K = in_features

    layer_type = normalize_sparse_layer_type(owner.topk_type)

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

    if layer_type == "deepst":
        owner.step_counter = 0
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            target_density=(
                owner.excitatory_target_density
                if synapse_type == "exc"
                else owner.inhibitory_target_density
            ),
            sigma=owner.sigma,
            use_noise=owner.use_noise,
            rewiring_mode=owner.rewiring_mode,
            synapses_per_branch=owner.synapses_per_branch,
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
        )

    if layer_type in {"indexed", "indexed_rewire"}:
        extra_kwargs = {}
        if layer_type == "indexed_rewire":
            extra_kwargs = {
                "rewire_frequency": owner.indexed_rewire_frequency,
                "rewire_quantile": owner.indexed_rewire_quantile,
                "rewire_until_step": owner.indexed_rewire_until_step,
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
            connection_mask=connection_mask,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            seed=_indexed_synapse_seed(owner, synapse_type),
            output_chunk_size=owner.indexed_output_chunk_size,
            use_fused_gather=owner.indexed_use_fused_gather,
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
            seed=_indexed_synapse_seed(owner, synapse_type),
            output_chunk_size=owner.indexed_output_chunk_size,
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
    forbidden_input_index_per_output: Any = None,
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
        forbidden_input_index_per_output=forbidden_input_index_per_output,
        connection_mask=connection_mask,
    )
    gradient_scaler.register_topk_dynamic(layer)
    return layer


__all__ = [
    "create_branch_sparse_layer",
    "create_registered_branch_sparse_layer",
]
