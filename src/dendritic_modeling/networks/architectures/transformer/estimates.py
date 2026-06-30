"""Parameter estimates for transformer feed-forward replacements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dendritic_modeling.config.conversion import normalize_sparsity_type


def _normalize_topk_type(value: Any) -> str:
    return normalize_sparsity_type(value)


def estimate_deepseek_dense_ffn_params(
    hidden_size: int = 7168,
    intermediate_size: int = 18432,
    *,
    gated: bool = True,
    bias: bool = False,
) -> int:
    """Estimate parameters in a DeepSeek/Llama-style dense FFN block."""
    projections = 3 if gated else 2
    params = projections * int(hidden_size) * int(intermediate_size)
    if bias:
        params += (2 if gated else 1) * int(intermediate_size) + int(hidden_size)
    return params


def estimate_dendritic_ffn_params(
    *,
    hidden_size: int,
    dendritic_units: int,
    branch_factors: Sequence[int],
    synapses_per_branch: int,
    candidate_size: int | None = None,
    input_transform: str = "signed_split",
    topk_type: str = "indexed",
    output_bias: bool = False,
    somatic_synapses: bool = True,
) -> dict[str, int]:
    """Estimate stored and active parameters for a dendritic FFN replacement."""
    topk_type = _normalize_topk_type(topk_type)
    input_dim = int(hidden_size) * (2 if input_transform == "signed_split" else 1)
    layer_sizes = [int(dendritic_units)]
    for branch_factor in branch_factors:
        layer_sizes.append(layer_sizes[-1] * int(branch_factor))
    dendritic_layer_sizes = list(reversed(layer_sizes))
    if not somatic_synapses:
        dendritic_layer_sizes = dendritic_layer_sizes[:-1]
    branch_units = sum(dendritic_layer_sizes)

    active_synapses = branch_units * int(synapses_per_branch)
    if topk_type in {"indexed", "indexed_rewire"}:
        stored_synapses = active_synapses
    elif topk_type == "indexed_dynamic":
        if candidate_size is None:
            candidate_size = min(
                input_dim,
                max(int(synapses_per_branch), 4 * int(synapses_per_branch)),
            )
        stored_synapses = branch_units * int(candidate_size)
    else:
        stored_synapses = branch_units * input_dim

    blocklinear = sum(
        int(parent) * int(branch_factor)
        for parent, branch_factor in zip(layer_sizes, branch_factors)
    )
    projection = int(dendritic_units) * int(hidden_size)
    if output_bias:
        projection += int(hidden_size)
    return {
        "input_dim": input_dim,
        "branch_units": branch_units,
        "active_synapses": active_synapses,
        "stored_synapses": stored_synapses,
        "blocklinear": blocklinear,
        "output_projection": projection,
        "stored_total": stored_synapses + blocklinear + projection,
        "active_total": active_synapses + blocklinear + projection,
    }
