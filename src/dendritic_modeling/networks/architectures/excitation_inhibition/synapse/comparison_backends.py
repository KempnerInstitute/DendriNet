"""Explicit A/B backends for fixed indexed sparse projections.

This module is intentionally benchmark-facing.  ``iss373_hs`` kernels and the
Ryanconfig kernels remain in separate modules, so selecting one can never
replace the other or share a lazy-compiled CUDA extension.
"""

from __future__ import annotations

from typing import Final

import torch


_ALIASES: Final[dict[str, str]] = {
    # Keep the spelling used by the imported iss373 example configs usable.
    "eager": "control_eager",
    "recompute": "iss_recompute",
    "triton": "iss_triton",
    "cuda": "iss_cuda",
    "auto": "iss_auto",
}
INDEXED_COMPARISON_BACKENDS: Final[frozenset[str]] = frozenset(
    {
        "control_eager",
        "iss_recompute",
        "iss_triton",
        "iss_cuda",
        "iss_auto",
        "ryan_fused",
        "ryan_triton",
        "ryan_cuda",
        "ryan_hybrid",
        "ryan_auto",
    }
)


def normalize_indexed_comparison_backend(value: object | None) -> str | None:
    """Validate an opt-in comparison backend, retaining legacy aliases."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in INDEXED_COMPARISON_BACKENDS:
        choices = ", ".join(sorted(INDEXED_COMPARISON_BACKENDS | _ALIASES.keys()))
        raise ValueError(
            "indexed_comparison_backend must be one of "
            f"{{{choices}}}, got {value!r}"
        )
    return normalized


class _Iss373RecomputeProjection(torch.autograd.Function):
    """The upstream recompute control, preserved separately for comparison."""

    @staticmethod
    def forward(ctx, flat_x, weight, connection_indices, output_chunk_size):
        ctx.output_chunk_size = int(output_chunk_size)
        ctx.save_for_backward(flat_x, weight, connection_indices)
        out_features, synapses_per_output = weight.shape
        out = flat_x.new_empty(flat_x.shape[0], out_features)
        for start in range(0, out_features, ctx.output_chunk_size):
            end = min(start + ctx.output_chunk_size, out_features)
            idx_chunk = connection_indices[start:end]
            selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                flat_x.shape[0], end - start, synapses_per_output
            )
            out[:, start:end] = (
                selected * weight[start:end].unsqueeze(0)
            ).sum(dim=-1)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        flat_x, weight, connection_indices = ctx.saved_tensors
        out_features, synapses_per_output = weight.shape
        accumulation_dtype = (
            torch.float32
            if flat_x.dtype in {torch.bfloat16, torch.float16}
            else flat_x.dtype
        )
        grad_x = (
            torch.zeros_like(flat_x, dtype=accumulation_dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight = (
            torch.empty_like(weight) if ctx.needs_input_grad[1] else None
        )
        for start in range(0, out_features, ctx.output_chunk_size):
            end = min(start + ctx.output_chunk_size, out_features)
            idx_chunk = connection_indices[start:end]
            grad_chunk = grad_output[:, start:end]
            if grad_weight is not None:
                selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                    flat_x.shape[0], end - start, synapses_per_output
                )
                grad_weight[start:end] = (
                    selected * grad_chunk.unsqueeze(-1)
                ).sum(dim=0)
            if grad_x is not None:
                contributions = grad_chunk.to(accumulation_dtype).unsqueeze(
                    -1
                ) * weight[start:end].to(accumulation_dtype).unsqueeze(0)
                grad_x.index_add_(
                    1,
                    idx_chunk.reshape(-1),
                    contributions.reshape(flat_x.shape[0], -1),
                )
        if grad_x is not None and grad_x.dtype != flat_x.dtype:
            grad_x = grad_x.to(flat_x.dtype)
        return grad_x, grad_weight, None, None


def iss373_recompute_sparse_gather(
    flat_x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    *,
    output_chunk_size: int,
) -> torch.Tensor:
    """Run the upstream custom-autograd recompute control."""
    return _Iss373RecomputeProjection.apply(
        flat_x, weight, connection_indices, output_chunk_size
    )


def _require_cuda_backend(backend: str, available: bool, reason: str) -> None:
    if not available:
        raise RuntimeError(
            f"indexed_comparison_backend={backend!r} is unavailable: {reason}. "
            "Use an explicit portable control or auto backend instead."
        )


def dispatch_iss373_backend(
    backend: str,
    flat_x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    *,
    output_chunk_size: int,
) -> tuple[torch.Tensor, str]:
    """Execute an upstream comparison backend and return its resolved label."""
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.iss373_kernels.cuda_indexed_gather import (
        cuda_gather_available,
        cuda_gather_unavailability_reason,
        cuda_sparse_gather,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.iss373_kernels.triton_indexed_gather import (
        triton_gather_available,
        triton_sparse_gather,
    )

    if backend == "iss_auto":
        if flat_x.is_cuda and cuda_gather_available():
            backend = "iss_cuda"
        elif flat_x.is_cuda and triton_gather_available():
            backend = "iss_triton"
        else:
            backend = "iss_recompute"
    if backend == "iss_recompute":
        return (
            iss373_recompute_sparse_gather(
                flat_x,
                connection_indices,
                weight,
                output_chunk_size=output_chunk_size,
            ),
            backend,
        )
    if backend == "iss_cuda":
        _require_cuda_backend(
            backend,
            flat_x.is_cuda and cuda_gather_available(),
            cuda_gather_unavailability_reason(),
        )
        return (
            cuda_sparse_gather(
                flat_x, connection_indices, weight, chunk_size=output_chunk_size
            ),
            backend,
        )
    if backend == "iss_triton":
        _require_cuda_backend(
            backend,
            flat_x.is_cuda and triton_gather_available(),
            "Triton and a CUDA device are required",
        )
        return (
            triton_sparse_gather(
                flat_x, connection_indices, weight, chunk_size=output_chunk_size
            ),
            backend,
        )
    raise ValueError(f"not an iss373 backend: {backend!r}")


__all__ = [
    "INDEXED_COMPARISON_BACKENDS",
    "dispatch_iss373_backend",
    "normalize_indexed_comparison_backend",
]
