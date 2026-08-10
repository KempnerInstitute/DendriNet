"""Triton fused sparse gather-multiply-sum: forward + backward in one Function.

Drop-in alternative to :func:`fused_sparse_gather` (same signature) that computes,
for a fixed sparse fan-in::

    out[m, o] = sum_k  x[m, connection_indices[o, k]] * weight[o, k]

v3 TRANSPOSED-COALESCED design (2026-08-07), replacing the original batch-tiled
kernels after a measured 4-6x per-op win (bench_triton_v2.py, Blackwell b512 K40):

  * forward  : gathers read from ``xT [IN, M]`` so the address ``idx*M + m`` is
    CONTIGUOUS over the batch -- every gather is a coalesced row segment and the
    kernel runs at the DRAM roofline for its traffic (1.55 ms at 16k->144k vs
    9.38 ms for the CUDA picker's kernel; the one-off transpose is ~0.05 ms).
  * backward : SPLIT. grad_w via a transposed reduction over m (``goT``/``xT``
    rows, coalesced, NO atomics); grad_x via coalesced ``tl.atomic_add`` into
    ``gxT [IN, M]`` then one transpose back. fp32 atomics only.
  * indices  : int32 end-to-end (the model's buffer dtype) -- the old kernels
    upcast to int64 EVERY call (an [O,K] alloc + copy per layer per step).

Measured per-op vs the old Triton kernels (Blackwell, batch 512, K=40):
    forward 3.7-14.6x faster, backward 2.8-5.8x faster; also beats the CUDA
    extension's picker at every swept FF shape (its staged design is occupancy-
    capped on sm120; brute coalesced traffic wins at this bandwidth). See
    profilings/dendrinet_harness/rnn_prof/bench_triton_v2.py and the ledger.

Correctness (fp64-reference, batch 512, all swept shapes): forward <= 3.1e-7,
grad_w <= 2.2e-5, grad_x <= 4.6e-6 -- fp32 reassociation only. grad_x remains
atomic and therefore run-to-run nondeterministic at the last-ulp level, exactly
like the previous kernels and the CUDA path.

AMP-safe: inputs are upcast to fp32 for the kernels (atomics stay fp32; bf16
atomics are unreliable across archs) and outputs/grads are cast back to the
caller's dtypes, so switching this path on is numerically equivalent under
autocast.

Triton is an OPTIONAL dependency: importing this module never fails. Query
:data:`HAS_TRITON` (or call :func:`triton_gather_available`) before selecting
this path; ``triton_sparse_gather`` raises a clear error if Triton is missing.
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

# Swept on Blackwell (bench_triton_v2.py): uniform winners across the FF grid.
# (B_M, B_O, B_K) for the forward and grad_x scatter; (B_O, B_M) for grad_w.
_FWD_TILES = (128, 16, 8)
_GX_TILES = (32, 32, 8)
_GW_TILES = (64, 64)


def triton_gather_available() -> bool:
    """Whether the Triton fused-gather path can be used on this install."""
    return HAS_TRITON


if HAS_TRITON:

    @triton.jit
    def _sparse_gather_fwd_t(
        xT_ptr, idx_ptr, w_ptr, out_ptr, M, O, IN, K,
        B_M: tl.constexpr, B_O: tl.constexpr, B_K: tl.constexpr,
    ):
        """out[m, o] = sum_k xT[idx[o, k], m] * w[o, k] -- coalesced over m.
        Output tiles ride grid axis 0 (2^31 limit): 20M-output RNN modules
        overflow the 65,535 cap of axis 1."""
        ob = tl.program_id(0)
        mb = tl.program_id(1)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < O
        acc = tl.zeros([B_O, B_M], dtype=tl.float32)
        for kb in range(0, tl.cdiv(K, B_K)):
            k_off = kb * B_K + tl.arange(0, B_K)
            k_mask = k_off < K
            ok = o_off[:, None] * K + k_off[None, :]
            om = o_mask[:, None] & k_mask[None, :]
            i2 = tl.load(idx_ptr + ok, mask=om, other=0)
            w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
            xo = i2[:, :, None] * M + m_off[None, None, :]        # [B_O, B_K, B_M]
            xm = om[:, :, None] & m_mask[None, None, :]
            xv = tl.load(xT_ptr + xo, mask=xm, other=0.0).to(tl.float32)
            acc += tl.sum(xv * w2[:, :, None].to(tl.float32), axis=1)
        oo = m_off[None, :] * O + o_off[:, None]
        tl.store(out_ptr + oo, acc, mask=(m_mask[None, :] & o_mask[:, None]))

    @triton.jit
    def _sparse_gather_gw_t(
        goT_ptr, xT_ptr, idx_ptr, gw_ptr, M, O, IN, K,
        B_O: tl.constexpr, B_M: tl.constexpr,
    ):
        """grad_w[o, k] = sum_m go[m, o] * x[m, idx[o, k]] -- transposed,
        coalesced over m, no atomics. Grid: (O tiles, K)."""
        ob = tl.program_id(0)
        k = tl.program_id(1)
        o_off = ob * B_O + tl.arange(0, B_O)
        o_mask = o_off < O
        idx = tl.load(idx_ptr + o_off * K + k, mask=o_mask, other=0)
        acc = tl.zeros([B_O], dtype=tl.float32)
        for mb in range(0, tl.cdiv(M, B_M)):
            m_off = mb * B_M + tl.arange(0, B_M)
            m_mask = m_off < M
            mm = o_mask[:, None] & m_mask[None, :]
            go = tl.load(goT_ptr + o_off[:, None] * M + m_off[None, :],
                         mask=mm, other=0.0).to(tl.float32)
            xv = tl.load(xT_ptr + idx[:, None] * M + m_off[None, :],
                         mask=mm, other=0.0).to(tl.float32)
            acc += tl.sum(go * xv, axis=1)
        tl.store(gw_ptr + o_off * K + k, acc, mask=o_mask)

    @triton.jit
    def _sparse_gather_gx_t(
        goT_ptr, idx_ptr, w_ptr, gxT_ptr, M, O, IN, K,
        B_M: tl.constexpr, B_O: tl.constexpr, B_K: tl.constexpr,
    ):
        """gxT[idx[o, k], m] += go[m, o] * w[o, k] -- coalesced fp32 atomics.
        Output tiles on grid axis 0 (2^31 limit), as in the forward."""
        ob = tl.program_id(0)
        mb = tl.program_id(1)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < O
        om_go = o_mask[:, None] & m_mask[None, :]
        go = tl.load(goT_ptr + o_off[:, None] * M + m_off[None, :],
                     mask=om_go, other=0.0)
        for kb in range(0, tl.cdiv(K, B_K)):
            k_off = kb * B_K + tl.arange(0, B_K)
            k_mask = k_off < K
            ok = o_off[:, None] * K + k_off[None, :]
            om = o_mask[:, None] & k_mask[None, :]
            i2 = tl.load(idx_ptr + ok, mask=om, other=0)
            w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
            xo = i2[:, :, None] * M + m_off[None, None, :]        # [B_O, B_K, B_M]
            xm = om[:, :, None] & m_mask[None, None, :]
            tl.atomic_add(gxT_ptr + xo,
                          w2[:, :, None].to(tl.float32) * go[:, None, :].to(tl.float32),
                          mask=xm)


class _TritonSparseGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, weight):
        # x: [M, IN] (any float dtype)   idx: [O, K] int32/int64   weight: [O, K]
        in_dtype = x.dtype
        w_dtype = weight.dtype
        # bf16 inputs pass through at half the bytes -- every kernel casts its
        # loads to fp32 in registers, so ALL accumulation stays fp32 (the sm_80
        # atomics + precision lesson). fp32 inputs are unchanged.
        compute_dtype = (
            torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
        )
        xf = x.contiguous().to(compute_dtype)
        wf = weight.contiguous().to(compute_dtype)
        idx = idx.contiguous().to(torch.int32)      # no-op for the int32 buffer
        M, IN = xf.shape
        O, K = idx.shape
        xT = xf.t().contiguous()
        out = torch.empty(M, O, device=xf.device, dtype=torch.float32)
        B_M, B_O, B_K = _FWD_TILES
        grid = (triton.cdiv(O, B_O), triton.cdiv(M, B_M))
        _sparse_gather_fwd_t[grid](xT, idx, wf, out, M, O, IN, K, B_M, B_O, B_K)
        # save the untransposed xf: backward re-transposes (cheap) rather than
        # doubling the saved-activation footprint.
        ctx.save_for_backward(xf, idx, wf)
        ctx.meta = (M, O, IN, K, in_dtype, w_dtype)
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        xf, idx, wf = ctx.saved_tensors
        M, O, IN, K, in_dtype, w_dtype = ctx.meta
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        goT = grad_out.contiguous().to(xf.dtype).t().contiguous()
        grad_x = grad_w = None
        if needs_w:
            xT = xf.t().contiguous()
            grad_w = torch.empty(O, K, device=xf.device, dtype=torch.float32)
            B_O, B_M = _GW_TILES
            _sparse_gather_gw_t[(triton.cdiv(O, B_O), K)](
                goT, xT, idx, grad_w, M, O, IN, K, B_O, B_M)
        if needs_x:
            gxT = torch.zeros(IN, M, device=xf.device, dtype=torch.float32)
            B_M2, B_O2, B_K2 = _GX_TILES
            grid = (triton.cdiv(O, B_O2), triton.cdiv(M, B_M2))
            _sparse_gather_gx_t[grid](goT, idx, wf, gxT, M, O, IN, K,
                                      B_M2, B_O2, B_K2)
            grad_x = gxT.t().contiguous()
        return (
            grad_x.to(in_dtype) if grad_x is not None else None,
            None,
            grad_w.to(w_dtype) if grad_w is not None else None,
        )


def triton_sparse_gather(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,          # accepted for signature parity; unused
    *,
    B_M: int | None = None,          # accepted for back-compat; tiles are fixed
    B_O: int | None = None,
) -> torch.Tensor:
    """Return ``out[m, o] = sum_k x[m, connection_indices[o, k]] * weight[o, k]``.

    Drop-in for :func:`fused_sparse_gather`: ``x`` is ``[M, in_features]``;
    ``connection_indices`` and ``weight`` are ``[out_features, K]``.
    Differentiable w.r.t. ``x`` and ``weight``; indices are fixed.

    Requires a CUDA tensor and a working Triton install (see :data:`HAS_TRITON`).
    ``chunk_size``/``B_M``/``B_O`` are ignored (v3 tiles are fixed, swept-in).
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "triton_sparse_gather requires Triton, which is not available. "
            "Use fused_sparse_gather (use_fused_gather=True) instead, or install triton."
        )
    if not x.is_cuda:
        raise RuntimeError("triton_sparse_gather requires a CUDA tensor")
    return _TritonSparseGather.apply(x, connection_indices, weight)


__all__ = ["triton_sparse_gather", "triton_gather_available", "HAS_TRITON"]
