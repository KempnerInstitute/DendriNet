"""Native signed dense, factorized and fixed-index readouts.

Construction order preserves the historical parameter-budget experiment RNG
sequence, including the initial draws made by ``nn.Linear`` constructors.
"""

from __future__ import annotations

import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)


def make_signed_readout(
    input_dim: int,
    output_dim: int,
    *,
    rank: int | None = None,
    topk: int | None = None,
    bias: bool = False,
    init_std: float = 0.02,
    seed: int | None = None,
    projection_backend: str = "eager",
) -> nn.Module:
    """Construct a signed readout with one explicit parameterization.

    Omitting both budgets produces a dense map. ``rank`` chooses two dense
    factors, and ``topk`` chooses fixed indexed support. Indexed initialization
    follows the synapse initializer; its weights are not rescaled by init_std.
    An affine bias for indexed readouts must be applied by the caller.
    """
    input_dim, output_dim = int(input_dim), int(output_dim)
    if input_dim < 1 or output_dim < 1:
        raise ValueError("input_dim and output_dim must be positive")
    if rank is not None and topk is not None:
        raise ValueError("rank and topk are mutually exclusive")
    if topk is None:
        if rank is None:
            projection = nn.Linear(input_dim, output_dim, bias=bool(bias))
            nn.init.normal_(projection.weight, mean=0.0, std=float(init_std))
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)
            return projection
        rank = int(rank)
        if rank < 1 or rank > min(input_dim, output_dim):
            raise ValueError("rank must be in [1, min(input_dim, output_dim)]")
        first = nn.Linear(input_dim, rank, bias=False)
        second = nn.Linear(rank, output_dim, bias=bool(bias))
        nn.init.normal_(first.weight, mean=0.0, std=float(init_std))
        nn.init.normal_(second.weight, mean=0.0, std=float(init_std))
        if second.bias is not None:
            nn.init.zeros_(second.bias)
        return nn.Sequential(first, second)
    topk = int(topk)
    if topk < 1 or topk > input_dim:
        raise ValueError("topk must be in [1, input_dim]")
    if bias:
        raise ValueError("sparse output projections currently require bias=false")
    return IndexedSparseLinear(
        in_features=input_dim,
        out_features=output_dim,
        K=topk,
        init_method="xavier_normal",
        weight_transform="identity",
        seed=seed,
        output_chunk_size=2048,
        index_dtype="int64",
        workspace_mb=None,
        cache_transformed_weights=False,
        recompute_backward=False,
        projection_backend=str(projection_backend),
        persistent_indices=True,
        init_mode="per_rank",
        support_group_rows=1,
        support_col_block=1,
    )
