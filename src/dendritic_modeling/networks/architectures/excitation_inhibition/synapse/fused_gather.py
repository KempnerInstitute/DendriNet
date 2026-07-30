"""Fused sparse gather-multiply-sum with a custom, memory-bounded backward.

Drop-in replacement for the chunked gather+mul+sum in ``indexed_sparse.py`` that
computes, for a fixed sparse fan-in::

    out[m, o] = sum_k  x[m, connection_indices[o, k]] * weight[o, k]

Motivation (measured on A100 MIG, out=90000, K=40, N=128, fp32):
  * autograd's generic advanced-index backward (``indexing_backward_kernel``) runs
    at ~3% of peak HBM bandwidth -- it was ~84% of full-model GPU time.
  * this backward uses an explicit, grouped ``index_add_`` scatter -> ~7.6x faster
    backward and ~28x lower peak memory (``selected`` is recomputed, not saved).

Numerically identical forward + grad_w; grad_x matches the baseline exactly in
fp64. In fp32 grad_x differs only by accumulation-order rounding (~1e-3 relative on
large fan-in), which is far below bf16/AMP training precision. See the isolated
benchmark and correctness harness in
``profilings/weight_transform_opt/bench_kernel-rzw-optimized.py``.
"""
from __future__ import annotations

import torch


class _ChunkedSparseGatherMulSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx_flat, weight, out_features, K, chunk_size):
        # x: [M, IN]   idx_flat: [out_features*K] long   weight: [out_features, K]
        M = x.shape[0]
        out = x.new_empty(M, out_features)
        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            idx_chunk = idx_flat[start * K:end * K]
            selected = x.index_select(1, idx_chunk).reshape(M, end - start, K)
            out[:, start:end] = (selected * weight[start:end].unsqueeze(0)).sum(dim=-1)
        # save only small tensors; `selected` is recomputed in backward to bound memory
        ctx.save_for_backward(x, idx_flat, weight)
        ctx.meta = (out_features, K, chunk_size)
        return out

    @staticmethod
    def backward(ctx, grad_out):                        # grad_out: [M, out_features]
        x, idx_flat, weight = ctx.saved_tensors
        out_features, K, chunk_size = ctx.meta
        M = x.shape[0]
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        # grad_x is a scatter: every one of the out_features*K synapses adds into
        # one of in_features slots, so each slot takes out_features*K/in_features
        # atomic adds (~1600 for the recurrent configs). Accumulate in fp32 when
        # the input is reduced-precision, for two reasons:
        #   * precision -- summing that many bf16 terms into a bf16 slot loses
        #     bits the eager baseline keeps (aten's indexing_backward accumulates
        #     in opmath_t = float), so fp32 here matches the baseline rather than
        #     diverging from it;
        #   * speed -- bf16 atomicAdd is emulated on sm_80 and collapses under
        #     this contention. Measured on the 5k/5k recurrent layer shapes:
        #     bf16 index_add_ is ~105x slower than fp32 on A100 (170 ms vs 1.6 ms)
        #     and ~1.9x slower on Blackwell. This one cast is the whole reason the
        #     fused path was 8.8x slower than the loop baseline on A100.
        acc_dtype = (
            torch.float32
            if x.dtype in (torch.bfloat16, torch.float16)
            else x.dtype
        )
        grad_x = (
            torch.zeros(x.shape, dtype=acc_dtype, device=x.device)
            if needs_x
            else None
        )
        grad_w = torch.empty_like(weight) if needs_w else None
        for start in range(0, out_features, chunk_size):
            end = min(start + chunk_size, out_features)
            idx_chunk = idx_flat[start * K:end * K]
            g = grad_out[:, start:end].unsqueeze(-1)                       # [M, chunk, 1]
            if needs_w:
                selected = x.index_select(1, idx_chunk).reshape(M, end - start, K)
                grad_w[start:end] = (g * selected).sum(dim=0)              # [chunk, K]
            if needs_x:
                grad_sel = (
                    g.to(acc_dtype) * weight[start:end].to(acc_dtype).unsqueeze(0)
                ).reshape(M, (end - start) * K)
                grad_x.index_add_(1, idx_chunk, grad_sel)                  # grouped scatter
        if grad_x is not None and grad_x.dtype != x.dtype:
            grad_x = grad_x.to(x.dtype)
        return grad_x, None, grad_w, None, None, None


def fused_sparse_gather(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """Return ``out[m, o] = sum_k x[m, connection_indices[o, k]] * weight[o, k]``.

    ``x`` is 2-D ``[M, in_features]``; ``connection_indices`` and ``weight`` are
    both ``[out_features, K]`` (indices ``long``). Differentiable w.r.t. ``x`` and
    ``weight``; ``connection_indices`` is treated as fixed.
    """
    out_features, K = connection_indices.shape
    idx_flat = connection_indices.reshape(-1)
    return _ChunkedSparseGatherMulSum.apply(x, idx_flat, weight, out_features, K, chunk_size)
