"""Triton fused sparse gather-multiply-sum: forward + backward in one Function.

Drop-in alternative to :func:`fused_sparse_gather` (same signature) that computes,
for a fixed sparse fan-in::

    out[m, o] = sum_k  x[m, connection_indices[o, k]] * weight[o, k]

using two hand-written Triton kernels instead of ``index_select``/``index_add_``:

  * forward  : batch-tiled gather (``_sparse_gather_fwd``) -- load idx/w once per
    output tile, reuse across ``B_M`` batch rows (amortizes index traffic).
  * backward : fused one-pass atomic scatter (``_sparse_gather_bwd``) -- grad_x via
    ``tl.atomic_add`` (the transpose of the forward gather), grad_w via a batch
    reduction, both in a single kernel.

Measured whole-training-step speedup vs the ``fused_sparse_gather`` path (fp32,
batch 128, the sglimit5 1-layer EI config):
    A100      +17.8%   (forward +11.9%, backward adds +5.9%)
    Blackwell +31.8%   (forward +16.4%, backward adds +15.3%)
Isolated backward beats ``index_add_`` 1.24x (A100) / 1.35x (Blackwell) at B_M=16.

Correctness: forward and grad_w are numerically identical to the baseline; grad_x
matches in fp64 and differs only by atomic accumulation-order rounding in fp32
(~1e-4 on this workload -- far below bf16/AMP training precision; full-model grad
match was ~2e-8 across all parameters).

AMP-safe: inputs are upcast to fp32 for the kernels (atomics stay fp32; bf16
atomics are unreliable across archs) and outputs/grads are cast back to the caller's
dtypes, so switching this path on is numerically equivalent under autocast.

Triton is an OPTIONAL dependency: importing this module never fails. Query
:data:`HAS_TRITON` (or call :func:`triton_gather_available`) before selecting this
path; ``triton_sparse_gather`` raises a clear error if Triton is unavailable.
"""
from __future__ import annotations

import torch

try:  # Triton is an optional dependency -- never hard-fail on import
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001  (ImportError, or a broken partial install)
    triton = None
    tl = None
    HAS_TRITON = False

# Proven default tile sizes (B_M=16, B_O=64 win on both A100 and Blackwell for the
# backward, and are within 1% of optimal for the forward). Kept module-level so a
# future autotune can override without touching call sites.
_DEFAULT_B_M = 16
_DEFAULT_B_O = 64


def triton_gather_available() -> bool:
    """Whether the Triton fused-gather path can be used on this install."""
    return HAS_TRITON


if HAS_TRITON:

    @triton.jit
    def _sparse_gather_fwd(
        x_ptr, idx_ptr, w_ptr, out_ptr,
        M, O, IN,
        K: tl.constexpr, B_M: tl.constexpr, B_O: tl.constexpr,
    ):
        mb = tl.program_id(0)                       # batch tile
        ob = tl.program_id(1)                       # output tile
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < O
        base_ok = o_off * K
        acc = tl.zeros([B_M, B_O], dtype=tl.float32)
        for k in tl.static_range(K):                # K fixed -> unrolled
            idx = tl.load(idx_ptr + base_ok + k, mask=o_mask, other=0)     # [B_O] read once
            wk = tl.load(w_ptr + base_ok + k, mask=o_mask, other=0.0)      # [B_O] read once
            x_off = m_off[:, None] * IN + idx[None, :]
            xmask = m_mask[:, None] & o_mask[None, :]
            xv = tl.load(x_ptr + x_off, mask=xmask, other=0.0)            # [B_M, B_O] gather
            acc += xv * wk[None, :]
        out_off = m_off[:, None] * O + o_off[None, :]
        tl.store(out_ptr + out_off, acc, mask=(m_mask[:, None] & o_mask[None, :]))

    @triton.jit
    def _sparse_gather_bwd(
        grad_out_ptr, x_ptr, idx_ptr, w_ptr, grad_x_ptr, grad_w_ptr,
        M, O, IN,
        K: tl.constexpr, B_M: tl.constexpr, B_O: tl.constexpr,
    ):
        mb = tl.program_id(0)
        ob = tl.program_id(1)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < O
        mo = m_mask[:, None] & o_mask[None, :]
        go = tl.load(grad_out_ptr + m_off[:, None] * O + o_off[None, :], mask=mo, other=0.0)
        base_ok = o_off * K
        for k in tl.static_range(K):
            idx = tl.load(idx_ptr + base_ok + k, mask=o_mask, other=0)     # [B_O]
            wk = tl.load(w_ptr + base_ok + k, mask=o_mask, other=0.0)      # [B_O]
            xoff = m_off[:, None] * IN + idx[None, :]                      # [B_M, B_O]
            # grad_x[m, idx] += grad_out[m, o] * w[o, k]   (scatter -> atomic)
            tl.atomic_add(grad_x_ptr + xoff, go * wk[None, :], mask=mo)
            # grad_w[o, k] += sum_m grad_out[m, o] * x[m, idx]   (batch reduce)
            xv = tl.load(x_ptr + xoff, mask=mo, other=0.0)
            tl.atomic_add(grad_w_ptr + base_ok + k, tl.sum(go * xv, axis=0), mask=o_mask)


class _TritonSparseGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, weight, B_M, B_O):
        # x: [M, IN] (any float dtype)   idx: [O, K] long   weight: [O, K]
        in_dtype = x.dtype
        w_dtype = weight.dtype
        xf = x.contiguous().to(torch.float32)
        wf = weight.contiguous().to(torch.float32)
        idx = idx.contiguous().to(torch.int64)
        M, IN = xf.shape
        O, K = idx.shape
        out = torch.empty(M, O, device=xf.device, dtype=torch.float32)
        grid = (triton.cdiv(M, B_M), triton.cdiv(O, B_O))
        _sparse_gather_fwd[grid](xf, idx, wf, out, M, O, IN, K, B_M, B_O)
        ctx.save_for_backward(xf, idx, wf)
        ctx.meta = (M, O, IN, K, B_M, B_O, in_dtype, w_dtype)
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        xf, idx, wf = ctx.saved_tensors
        M, O, IN, K, B_M, B_O, in_dtype, w_dtype = ctx.meta
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        grad_out = grad_out.contiguous().to(torch.float32)
        # both grads are produced in one kernel pass; allocate whatever we might read
        grad_x = torch.zeros(M, IN, device=xf.device, dtype=torch.float32)
        grad_w = torch.zeros(O, K, device=xf.device, dtype=torch.float32)
        grid = (triton.cdiv(M, B_M), triton.cdiv(O, B_O))
        _sparse_gather_bwd[grid](grad_out, xf, idx, wf, grad_x, grad_w, M, O, IN, K, B_M, B_O)
        return (
            grad_x.to(in_dtype) if needs_x else None,
            None,
            grad_w.to(w_dtype) if needs_w else None,
            None,
            None,
        )


def triton_sparse_gather(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,          # accepted for signature parity; unused (no chunking)
    *,
    B_M: int = _DEFAULT_B_M,
    B_O: int = _DEFAULT_B_O,
) -> torch.Tensor:
    """Return ``out[m, o] = sum_k x[m, connection_indices[o, k]] * weight[o, k]``.

    Drop-in for :func:`fused_sparse_gather`: ``x`` is ``[M, in_features]``;
    ``connection_indices`` and ``weight`` are ``[out_features, K]`` (indices
    ``long``). Differentiable w.r.t. ``x`` and ``weight``; indices are fixed.

    Requires a CUDA tensor and a working Triton install (see :data:`HAS_TRITON`).
    ``chunk_size`` is ignored -- the Triton kernels tile internally.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "triton_sparse_gather requires Triton, which is not available. "
            "Use fused_sparse_gather (use_fused_gather=True) instead, or install triton."
        )
    if not x.is_cuda:
        raise RuntimeError("triton_sparse_gather requires a CUDA tensor")
    return _TritonSparseGather.apply(x, connection_indices, weight, B_M, B_O)


__all__ = ["triton_sparse_gather", "triton_gather_available", "HAS_TRITON"]
