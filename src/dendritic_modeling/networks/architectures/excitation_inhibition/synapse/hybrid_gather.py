"""Hybrid sparse gather: Triton v3 FORWARD married to the fused aten BACKWARD.

Motivation (bench_highk.py, 2026-08-11, H200, the 1M model's hot shapes):
at K=300-3968 the v3 K-streaming forward wins up to 1.56x over the fused
aten forward, but the v3 atomic backward loses 4-6.5x. The fused backward
(grad_w recompute + grouped index_add_) is the best backward in the stack at
high K. This Function takes the best column of each: forward via the v3
kernel (bf16-native, int32 indices, transposed-coalesced), backward exactly
the fused path's (identical math and dtypes -- its equivalence receipts
carry over unchanged).

Requires Triton (query :func:`hybrid_gather_available`); indices int32/int64.
"""
from __future__ import annotations

import torch

from .fused_gather import _ChunkedSparseGatherMulSum
from . import triton_gather as tg


def hybrid_gather_available() -> bool:
    """Whether the hybrid path (Triton forward) can be used."""
    return tg.HAS_TRITON


class _HybridSparseGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx32, idx_flat, weight, out_features, K, chunk_size):
        # v3 forward: bf16-native, fp32 accumulation in-kernel
        compute_dtype = (
            torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
        )
        xf = x.contiguous().to(compute_dtype)
        wf = weight.contiguous().to(compute_dtype)
        M, IN = xf.shape
        xT = xf.t().contiguous()
        out = torch.empty(M, out_features, device=xf.device, dtype=torch.float32)
        B_M, B_O, B_K = tg._FWD_TILES
        grid = (tg.triton.cdiv(out_features, B_O), tg.triton.cdiv(M, B_M))
        tg._sparse_gather_fwd_t[grid](
            xT, idx32, wf, out, M, out_features, IN, K, B_M, B_O, B_K
        )
        ctx.save_for_backward(x, idx_flat, weight)
        ctx.meta = (out_features, K, chunk_size)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # Exactly the fused path's backward (aten recompute + index_add_) --
        # but through a shim: the borrowed code reads needs_input_grad with
        # ITS argument order (x, idx, weight), while this Function's order is
        # (x, idx32, idx_flat, weight, ...). Passing ctx raw makes it read
        # idx_flat's False as "weight needs no grad" and silently skip grad_w.
        class _Shim:
            saved_tensors = ctx.saved_tensors
            meta = ctx.meta
            needs_input_grad = (
                ctx.needs_input_grad[0],
                False,
                ctx.needs_input_grad[3],
            )

        g = _ChunkedSparseGatherMulSum.backward(_Shim, grad_out)
        grad_x, _, grad_w = g[0], g[1], g[2]
        return grad_x, None, None, grad_w, None, None, None


def hybrid_sparse_gather(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """v3-forward / fused-backward gather; same contract as the siblings."""
    out_features, K = connection_indices.shape
    idx32 = connection_indices.contiguous().to(torch.int32)
    # aten index_select/index_add_ accept IntTensor: keep indices int32
    idx_flat = idx32.reshape(-1)
    return _HybridSparseGather.apply(
        x, idx32, idx_flat, weight, out_features, K, chunk_size
    )
