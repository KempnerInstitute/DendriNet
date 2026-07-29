"""Shared utilities for indexed sparse synapse layers."""

from __future__ import annotations

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    validate_connection_mask,
)


def _sample_indices_from_mask(
    *,
    out_features: int,
    in_features: int,
    K: int,
    connection_mask: torch.Tensor | None,
    generator: torch.Generator | None,
    row_chunk_size: int = 2048,
) -> torch.Tensor:
    """Sample exactly ``K`` unique allowed input indices for each output."""
    allowed = validate_connection_mask(connection_mask, out_features, in_features)
    indices = torch.empty(out_features, K, dtype=torch.long)

    # The common unmasked sparse case must not construct one full randperm of
    # all inputs for every output row. At large neuronal scales that makes
    # initialization proportional to out_features * in_features even though
    # only out_features * K indices are retained. Rejection sampling is exact
    # and efficient while K is sparse relative to in_features.
    if allowed is None and K * 4 <= in_features:
        max_chunk_entries = 8_000_000
        rows_per_chunk = max(
            1,
            min(
                int(row_chunk_size),
                max_chunk_entries // max(K, 1),
            ),
        )
        for start in range(0, out_features, rows_per_chunk):
            end = min(start + rows_per_chunk, out_features)
            values = torch.randint(
                in_features,
                (end - start, K),
                dtype=torch.long,
                generator=generator,
            )
            for _ in range(128):
                sorted_values, order = values.sort(dim=1)
                duplicate_sorted = torch.zeros_like(values, dtype=torch.bool)
                duplicate_sorted[:, 1:] = sorted_values[:, 1:] == sorted_values[:, :-1]
                if not bool(duplicate_sorted.any()):
                    break
                duplicate = torch.zeros_like(duplicate_sorted)
                duplicate.scatter_(1, order, duplicate_sorted)
                duplicate_count = int(duplicate.sum().item())
                values[duplicate] = torch.randint(
                    in_features,
                    (duplicate_count,),
                    dtype=torch.long,
                    generator=generator,
                )
            else:
                raise RuntimeError(
                    "Unable to sample unique indexed connections after 128 "
                    "rejection rounds."
                )
            indices[start:end] = values
        return indices

    for out_idx in range(out_features):
        if allowed is None:
            row_allowed = torch.arange(in_features, dtype=torch.long)
        else:
            row_allowed = torch.nonzero(allowed[out_idx], as_tuple=False).flatten()

        if row_allowed.numel() < K:
            raise ValueError(
                "IndexedSparseLinear requires at least K allowed inputs per "
                f"output row; row {out_idx} has {row_allowed.numel()} allowed "
                f"inputs but K={K}."
            )

        perm = torch.randperm(row_allowed.numel(), generator=generator)[:K]
        indices[out_idx] = row_allowed[perm]

    return indices


def _connection_generator(seed: int | None) -> torch.Generator:
    """Create a CPU generator; seed=None follows the process/global RNG state."""
    if seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def _scale_for_sparse_target(
    param_scale: torch.Tensor,
    *,
    target: torch.Tensor,
    connection_indices: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Broadcast or gather a gradient scale onto a sparse [out, K] tensor."""
    scale = param_scale.detach().to(device=target.device, dtype=target.dtype)
    if scale.ndim == 1:
        scale = scale[:, None]
    if scale.shape == target.shape:
        return scale
    if scale.ndim == 2 and scale.shape[1] == in_features:
        row = torch.arange(target.shape[0], device=target.device)[:, None]
        return scale[row, connection_indices.to(target.device)]
    return scale


def _allowed_mask_with_forbidden(
    *,
    connection_mask: torch.Tensor | None,
    forbidden_input_index_per_output: torch.Tensor | None,
    out_features: int,
    in_features: int,
) -> torch.Tensor | None:
    """Merge hard connectivity masks with per-row forbidden input indices."""
    allowed = validate_connection_mask(connection_mask, out_features, in_features)
    if forbidden_input_index_per_output is None:
        return allowed

    forbidden = torch.as_tensor(
        forbidden_input_index_per_output, dtype=torch.long
    ).view(-1)
    if forbidden.numel() != out_features:
        raise ValueError(
            "forbidden_input_index_per_output must have one index per "
            f"output feature ({out_features}), got {forbidden.numel()}"
        )
    valid = (forbidden >= -1) & (forbidden < in_features)
    if not bool(valid.all()):
        raise ValueError(
            "forbidden_input_index_per_output entries must be -1 or in "
            f"[0, {in_features - 1}]"
        )

    if allowed is None:
        allowed = torch.ones(out_features, in_features, dtype=torch.bool)
    else:
        allowed = allowed.clone()
    rows = torch.arange(out_features, dtype=torch.long)
    active = forbidden >= 0
    if bool(active.any()):
        allowed[rows[active], forbidden[active]] = False
    if allowed.numel() > 0 and bool((allowed.sum(dim=1) == 0).any()):
        raise ValueError(
            "connection_mask and forbidden_input_index_per_output leave at "
            "least one output row with no allowed inputs"
        )
    return allowed


__all__ = [
    "_allowed_mask_with_forbidden",
    "_connection_generator",
    "_sample_indices_from_mask",
    "_scale_for_sparse_target",
]
