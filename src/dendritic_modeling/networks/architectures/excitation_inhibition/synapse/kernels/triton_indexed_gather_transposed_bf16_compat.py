"""BF16-product-compatible transposed Triton indexed projection.

This is an explicit numerical-compatibility backend for the fixed-index
projection

.. math::

   y_{m,o} = \\sum_k x_{m,i[o,k]} w_{o,k}.

PyTorch eager materializes the elementwise product in BF16 when both operands
are BF16, so every product is rounded to BF16 before the reduction.  The
ordinary :mod:`triton_indexed_gather_transposed` backend intentionally converts
both operands to FP32 before multiplying.  That is normally the more accurate
operation, but its different rounding points can accumulate through a deep
composition and fail a comparison against the eager execution of the same BF16
model.

The kernels here preserve the transposed, coalesced layout while explicitly
rounding each forward, ``grad_weight``, and ``grad_x`` product to BF16 before
FP32 reduction.  The final tensors remain BF16, matching the eager dtype
contract.  This mode accepts BF16 activations and BF16 weights only and fails
closed for mixed or other floating-point dtypes.  It is never selected by
``auto`` and does not change the ordinary transposed backend.

Triton remains optional: importing this module is safe without a CUDA device or
a working Triton installation.  Call :func:`triton_bf16_compat_available`
before selecting the backend.
"""

from __future__ import annotations

import warnings

import torch

try:  # Triton is optional; package import must remain portable.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False


_FWD_TILES = (128, 16, 8)  # B_M, B_O, B_K; preserve transposed backend tiling.
_GW_TILES = (64, 64)  # B_O, B_M.
_GX_TILES = (32, 32, 8)  # atomic fallback B_M, B_O, B_K.
_GX_DET_TILES = (16, 128)  # B_P, B_M.
_GX_DET_NSPLIT = 4
_GX_DET_REDUCE_BLOCK = 1024
_DET_GX_WORKSPACE_CAP_BYTES = 2 * 1024**3
_warned_gradx_fallback = False


def triton_bf16_compat_available() -> bool:
    """Return whether CUDA and the optional Triton dependency are available."""

    return HAS_TRITON and torch.cuda.is_available()


if HAS_TRITON:

    @triton.jit
    def _bf16_product_as_fp32(lhs, rhs):
        """Multiply in FP32, round the product to BF16, then widen for sum."""

        return (
            (lhs.to(tl.float32) * rhs.to(tl.float32))
            .to(
                tl.bfloat16,
                fp_downcast_rounding="rtne",
            )
            .to(tl.float32)
        )

    @triton.jit
    def _sparse_gather_fwd_bf16_compat(
        xT_ptr,
        idx_ptr,
        w_ptr,
        out_ptr,
        M,
        NUM_OUTPUTS,
        IN,
        K,
        B_M: tl.constexpr,
        B_O: tl.constexpr,
        B_K: tl.constexpr,
    ):
        """Transposed forward with eager-compatible BF16 product rounding."""

        ob = tl.program_id(0)
        mb = tl.program_id(1)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < NUM_OUTPUTS
        acc = tl.zeros([B_O, B_M], dtype=tl.float32)
        for kb in range(tl.cdiv(K, B_K)):
            k_off = kb * B_K + tl.arange(0, B_K)
            k_mask = k_off < K
            ok = o_off[:, None] * K + k_off[None, :]
            om = o_mask[:, None] & k_mask[None, :]
            indices = tl.load(idx_ptr + ok, mask=om, other=0)
            weights = tl.load(w_ptr + ok, mask=om, other=0.0)
            x_offsets = indices[:, :, None] * M + m_off[None, None, :]
            x_mask = om[:, :, None] & m_mask[None, None, :]
            values = tl.load(xT_ptr + x_offsets, mask=x_mask, other=0.0)
            products = _bf16_product_as_fp32(
                values,
                weights[:, :, None],
            )
            acc += tl.sum(products, axis=1)
        out_offsets = m_off[None, :] * NUM_OUTPUTS + o_off[:, None]
        tl.store(
            out_ptr + out_offsets,
            acc.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=m_mask[None, :] & o_mask[:, None],
        )

    @triton.jit
    def _sparse_gather_gw_bf16_compat(
        goT_ptr,
        xT_ptr,
        idx_ptr,
        grad_weight_ptr,
        M,
        NUM_OUTPUTS,
        IN,
        K,
        B_O: tl.constexpr,
        B_M: tl.constexpr,
    ):
        """Weight gradient with BF16-rounded products and FP32 reduction."""

        ob = tl.program_id(0)
        k = tl.program_id(1)
        o_off = ob * B_O + tl.arange(0, B_O)
        o_mask = o_off < NUM_OUTPUTS
        indices = tl.load(idx_ptr + o_off * K + k, mask=o_mask, other=0)
        acc = tl.zeros([B_O], dtype=tl.float32)
        for mb in range(tl.cdiv(M, B_M)):
            m_off = mb * B_M + tl.arange(0, B_M)
            m_mask = m_off < M
            mask = o_mask[:, None] & m_mask[None, :]
            grad_output = tl.load(
                goT_ptr + o_off[:, None] * M + m_off[None, :],
                mask=mask,
                other=0.0,
            )
            values = tl.load(
                xT_ptr + indices[:, None] * M + m_off[None, :],
                mask=mask,
                other=0.0,
            )
            acc += tl.sum(
                _bf16_product_as_fp32(grad_output, values),
                axis=1,
            )
        tl.store(
            grad_weight_ptr + o_off * K + k,
            acc.to(tl.bfloat16, fp_downcast_rounding="rtne"),
            mask=o_mask,
        )

    @triton.jit
    def _sparse_gather_gx_bf16_compat_atomic(
        goT_ptr,
        idx_ptr,
        w_ptr,
        grad_xT_ptr,
        M,
        NUM_OUTPUTS,
        IN,
        K,
        B_M: tl.constexpr,
        B_O: tl.constexpr,
        B_K: tl.constexpr,
    ):
        """Atomic fallback for grad_x with products rounded before scatter."""

        ob = tl.program_id(0)
        mb = tl.program_id(1)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < NUM_OUTPUTS
        grad_mask = o_mask[:, None] & m_mask[None, :]
        grad_output = tl.load(
            goT_ptr + o_off[:, None] * M + m_off[None, :],
            mask=grad_mask,
            other=0.0,
        )
        for kb in range(tl.cdiv(K, B_K)):
            k_off = kb * B_K + tl.arange(0, B_K)
            k_mask = k_off < K
            ok = o_off[:, None] * K + k_off[None, :]
            support_mask = o_mask[:, None] & k_mask[None, :]
            indices = tl.load(idx_ptr + ok, mask=support_mask, other=0)
            weights = tl.load(w_ptr + ok, mask=support_mask, other=0.0)
            x_offsets = indices[:, :, None] * M + m_off[None, None, :]
            contribution_mask = support_mask[:, :, None] & m_mask[None, None, :]
            contributions = _bf16_product_as_fp32(
                weights[:, :, None],
                grad_output[:, None, :],
            )
            tl.atomic_add(
                grad_xT_ptr + x_offsets,
                contributions,
                mask=contribution_mask,
            )

    @triton.jit
    def _sparse_gather_gx_bf16_compat_stage1(
        output_sorted_ptr,
        weight_sorted_ptr,
        row_ptr,
        goT_ptr,
        workspace_ptr,
        M,
        IN,
        NSPLIT,
        B_P: tl.constexpr,
        B_M: tl.constexpr,
    ):
        """Deterministic CSR grad_x stage with BF16-rounded products."""

        pid0 = tl.program_id(0)
        mb = tl.program_id(1)
        input_index = pid0 // NSPLIT
        split = pid0 % NSPLIT
        m_off = mb * B_M + tl.arange(0, B_M)
        m_mask = m_off < M
        start = tl.load(row_ptr + input_index)
        end = tl.load(row_ptr + input_index + 1)
        segment_size = end - start
        chunk_size = (segment_size + NSPLIT - 1) // NSPLIT
        lo = start + split * chunk_size
        hi = tl.minimum(lo + chunk_size, end)
        count = hi - lo
        acc = tl.zeros([B_M], dtype=tl.float32)
        for pb in range(tl.cdiv(count, B_P)):
            p_off = lo + pb * B_P + tl.arange(0, B_P)
            p_mask = p_off < hi
            outputs = tl.load(output_sorted_ptr + p_off, mask=p_mask, other=0)
            weights = tl.load(weight_sorted_ptr + p_off, mask=p_mask, other=0.0)
            grad_output = tl.load(
                goT_ptr + outputs.to(tl.int64)[:, None] * M + m_off[None, :],
                mask=p_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(
                _bf16_product_as_fp32(grad_output, weights[:, None]),
                axis=0,
            )
        base = (split.to(tl.int64) * IN + input_index) * M
        tl.store(workspace_ptr + base + m_off, acc, mask=m_mask)

    @triton.jit
    def _reduce_gradx_workspace(
        workspace_ptr,
        output_ptr,
        NUMEL,
        N_BLOCKS,
        B: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * B + tl.arange(0, B)
        mask = offsets < NUMEL
        acc = tl.zeros([B], dtype=tl.float32)
        pointer = workspace_ptr + offsets
        for _ in range(N_BLOCKS):
            acc += tl.load(pointer, mask=mask, other=0.0)
            pointer += NUMEL
        tl.store(output_ptr + offsets, acc, mask=mask)


def _warn_gradx_fallback(reason: str) -> None:
    from dendritic_modeling.utils.reproducibility import alert_not_deterministic

    alert_not_deterministic(
        "The atomic grad_x fallback of triton_transposed_bf16_compat " f"({reason})",
        "Use projection_backend 'eager' for a BF16-product-compatible "
        "deterministic projection of this layer.",
    )
    global _warned_gradx_fallback
    if _warned_gradx_fallback:
        return
    _warned_gradx_fallback = True
    warnings.warn(
        "triton_transposed_bf16_compat deterministic grad_x "
        f"{reason}; using the BF16-product-compatible atomic fallback. "
        "The gradient remains numerically compatible but is not bitwise "
        "deterministic across runs.",
        RuntimeWarning,
        stacklevel=3,
    )


def _deterministic_gradx(
    grad_output_t: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
    *,
    batch_size: int,
    num_outputs: int,
    in_features: int,
    fan_in: int,
    source_indices: torch.Tensor,
) -> torch.Tensor | None:
    """Return deterministic FP32 grad_xT, or ``None`` past safe limits."""

    # Share the shape limits and the maintained, version-aware CSR setup cache
    # with the ordinary transposed backend.  The setup depends only on the
    # immutable support and does not encode any numerical semantics.
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed import (
        _gradx_setup,
        deterministic_gradx_limit,
    )

    limit = deterministic_gradx_limit(
        batch_size=batch_size,
        in_features=in_features,
        out_features=num_outputs,
        synapses_per_output=fan_in,
    )
    if limit is not None:
        _warn_gradx_fallback(limit)
        return None
    slice_bytes = 4 * int(in_features) * int(batch_size)

    permutation, output_sorted, row_ptr = _gradx_setup(
        source_indices,
        indices,
        in_features,
    )
    weight_sorted = weight.reshape(-1).index_select(0, permutation)
    nsplit = max(
        1,
        min(_GX_DET_NSPLIT, _DET_GX_WORKSPACE_CAP_BYTES // max(1, slice_bytes)),
    )
    workspace = torch.empty(
        nsplit,
        in_features,
        batch_size,
        device=grad_output_t.device,
        dtype=torch.float32,
    )
    block_positions, block_batch = _GX_DET_TILES
    _sparse_gather_gx_bf16_compat_stage1[
        (in_features * nsplit, triton.cdiv(batch_size, block_batch))
    ](
        output_sorted,
        weight_sorted,
        row_ptr,
        grad_output_t,
        workspace,
        batch_size,
        in_features,
        nsplit,
        B_P=block_positions,
        B_M=block_batch,
    )
    if nsplit == 1:
        return workspace[0]
    grad_x_t = torch.empty(
        in_features,
        batch_size,
        device=grad_output_t.device,
        dtype=torch.float32,
    )
    numel = int(in_features) * int(batch_size)
    _reduce_gradx_workspace[(triton.cdiv(numel, _GX_DET_REDUCE_BLOCK),)](
        workspace,
        grad_x_t,
        numel,
        nsplit,
        _GX_DET_REDUCE_BLOCK,
    )
    return grad_x_t


class _TritonSparseGatherTransposedBF16Compat(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        connection_indices: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        x_contiguous = x.contiguous()
        weight_contiguous = weight.contiguous()
        source_indices = connection_indices
        indices = connection_indices.contiguous().to(torch.int32)
        batch_size, in_features = x_contiguous.shape
        num_outputs, fan_in = indices.shape
        x_transposed = x_contiguous.t().contiguous()
        output = torch.empty(
            batch_size,
            num_outputs,
            device=x.device,
            dtype=torch.bfloat16,
        )
        block_batch, block_output, block_fan_in = _FWD_TILES
        grid = (
            triton.cdiv(num_outputs, block_output),
            triton.cdiv(batch_size, block_batch),
        )
        with torch.cuda.device(x.device):
            _sparse_gather_fwd_bf16_compat[grid](
                x_transposed,
                indices,
                weight_contiguous,
                output,
                batch_size,
                num_outputs,
                in_features,
                fan_in,
                block_batch,
                block_output,
                block_fan_in,
            )
        ctx.save_for_backward(
            x_contiguous,
            indices,
            weight_contiguous,
            source_indices,
        )
        ctx.shape = (batch_size, num_outputs, in_features, fan_in)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, indices, weight, source_indices = ctx.saved_tensors
        batch_size, num_outputs, in_features, fan_in = ctx.shape
        needs_x, _, needs_weight = ctx.needs_input_grad[:3]
        if grad_output.dtype != torch.bfloat16:
            raise RuntimeError(
                "triton_transposed_bf16_compat backward requires a BF16 "
                f"output gradient, got {grad_output.dtype}"
            )
        grad_output_t = grad_output.contiguous().t().contiguous()
        grad_x = None
        grad_weight = None
        with torch.cuda.device(x.device):
            if needs_weight:
                x_transposed = x.t().contiguous()
                grad_weight = torch.empty(
                    num_outputs,
                    fan_in,
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                block_output, block_batch = _GW_TILES
                _sparse_gather_gw_bf16_compat[
                    (triton.cdiv(num_outputs, block_output), fan_in)
                ](
                    grad_output_t,
                    x_transposed,
                    indices,
                    grad_weight,
                    batch_size,
                    num_outputs,
                    in_features,
                    fan_in,
                    block_output,
                    block_batch,
                )
            if needs_x:
                grad_x_t = _deterministic_gradx(
                    grad_output_t,
                    indices,
                    weight,
                    batch_size=batch_size,
                    num_outputs=num_outputs,
                    in_features=in_features,
                    fan_in=fan_in,
                    source_indices=source_indices,
                )
                if grad_x_t is None:
                    grad_x_t = torch.zeros(
                        in_features,
                        batch_size,
                        device=x.device,
                        dtype=torch.float32,
                    )
                    block_batch, block_output, block_fan_in = _GX_TILES
                    grid = (
                        triton.cdiv(num_outputs, block_output),
                        triton.cdiv(batch_size, block_batch),
                    )
                    _sparse_gather_gx_bf16_compat_atomic[grid](
                        grad_output_t,
                        indices,
                        weight,
                        grad_x_t,
                        batch_size,
                        num_outputs,
                        in_features,
                        fan_in,
                        block_batch,
                        block_output,
                        block_fan_in,
                    )
                grad_x = grad_x_t.t().contiguous().to(torch.bfloat16)
        return grad_x, None, grad_weight


def _validate_inputs(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must be two-dimensional, got shape {tuple(x.shape)}")
    if connection_indices.ndim != 2 or weight.ndim != 2:
        raise ValueError("connection_indices and weight must be two-dimensional")
    if connection_indices.shape != weight.shape:
        raise ValueError(
            "connection_indices and weight must have identical shapes, got "
            f"{tuple(connection_indices.shape)} and {tuple(weight.shape)}"
        )
    if x.shape[1] < 1:
        raise ValueError("x must have a nonempty input-feature dimension")
    if x.shape[0] < 1:
        raise ValueError("x must have a nonempty flattened batch dimension")
    if connection_indices.shape[0] < 1 or connection_indices.shape[1] < 1:
        raise ValueError("connection_indices must have nonempty output and K axes")
    if connection_indices.dtype not in {torch.int32, torch.int64}:
        raise TypeError("connection_indices must use int32 or int64")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(
            "triton_transposed_bf16_compat requires BF16 x and BF16 weight; "
            f"got {x.dtype} and {weight.dtype}"
        )
    if x.device != connection_indices.device or x.device != weight.device:
        raise ValueError("x, connection_indices, and weight must share one device")
    if not x.is_cuda:
        raise RuntimeError(
            "triton_transposed_bf16_compat requires tensors on a CUDA device"
        )


def triton_sparse_gather_bf16_compat(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,
    *,
    B_M: int | None = None,
    B_O: int | None = None,
) -> torch.Tensor:
    """Apply the explicit BF16-product-compatible transposed projection.

    ``chunk_size``, ``B_M``, and ``B_O`` are accepted for API parity with the
    ordinary transposed backend; the prospectively fixed kernel tiles are used.
    The operation is differentiable with respect to ``x`` and ``weight``.
    """

    del chunk_size, B_M, B_O
    _validate_inputs(x, connection_indices, weight)
    if not triton_bf16_compat_available():
        raise RuntimeError(
            "triton_transposed_bf16_compat requires a working Triton install"
        )
    return _TritonSparseGatherTransposedBF16Compat.apply(
        x,
        connection_indices,
        weight,
    )


__all__ = [
    "HAS_TRITON",
    "triton_bf16_compat_available",
    "triton_sparse_gather_bf16_compat",
]
