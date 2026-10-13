"""Shared utilities for indexed sparse synapse layers."""

from __future__ import annotations

import hashlib
import warnings
from typing import NamedTuple

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    validate_connection_mask,
)


def _resolve_index_dtype(
    index_dtype: str | torch.dtype,
    *,
    in_features: int,
) -> torch.dtype:
    """Resolve a compact, valid dtype for stored presynaptic indices."""
    if isinstance(index_dtype, torch.dtype):
        resolved = index_dtype
    else:
        normalized = str(index_dtype).strip().lower()
        aliases = {
            "auto": "auto",
            "int32": torch.int32,
            "torch.int32": torch.int32,
            "int64": torch.int64,
            "long": torch.int64,
            "torch.int64": torch.int64,
        }
        if normalized not in aliases:
            raise ValueError("index_dtype must be one of: auto, int32, int64")
        value = aliases[normalized]
        if value == "auto":
            resolved = (
                torch.int32
                if in_features <= torch.iinfo(torch.int32).max
                else torch.int64
            )
        else:
            resolved = value

    if resolved not in {torch.int32, torch.int64}:
        raise ValueError("index_dtype must resolve to torch.int32 or torch.int64")
    if resolved == torch.int32 and in_features > torch.iinfo(torch.int32).max:
        raise ValueError(
            "int32 connection indices require in_features <= "
            f"{torch.iinfo(torch.int32).max}, got {in_features}"
        )
    return resolved


def _validate_forbidden_indices(
    forbidden_input_index_per_output: torch.Tensor | None,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor | None:
    """Validate and normalize one optional forbidden input index per row."""
    if forbidden_input_index_per_output is None:
        return None

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
    return forbidden


# Output rows drawn per rejection-sampling chunk. The sampled topology depends
# on this value, so it is a fixed part of the sampler rather than the forward
# workspace setting ``output_chunk_size``. It equals that setting's default, so
# topologies sampled with the default are unchanged.
_TOPOLOGY_SAMPLING_ROWS = 2048


def _sample_indices_from_mask(
    *,
    out_features: int,
    in_features: int,
    K: int,
    connection_mask: torch.Tensor | None,
    generator: torch.Generator | None,
    forbidden_input_index_per_output: torch.Tensor | None = None,
    index_dtype: str | torch.dtype = torch.int64,
) -> torch.Tensor:
    """Sample exactly ``K`` unique allowed input indices for each output.

    The result depends only on the arguments and the generator state, not on
    any execution setting.
    """
    allowed = validate_connection_mask(connection_mask, out_features, in_features)
    forbidden = _validate_forbidden_indices(
        forbidden_input_index_per_output,
        out_features=out_features,
        in_features=in_features,
    )
    resolved_dtype = _resolve_index_dtype(index_dtype, in_features=in_features)
    indices = torch.empty(out_features, K, dtype=resolved_dtype)

    # The common unmasked sparse case must not construct one full randperm of
    # all inputs for every output row. At large neuronal scales that makes
    # initialization proportional to out_features * in_features even though
    # only out_features * K indices are retained. Rejection sampling is exact
    # and efficient while K is sparse relative to in_features.
    if allowed is None and K * 4 <= in_features:
        if forbidden is not None and bool(
            (in_features - (forbidden >= 0).to(torch.long) < K).any()
        ):
            raise ValueError(
                "Indexed sparse sampling requires at least K allowed inputs "
                "per output row after applying forbidden indices."
            )
        max_chunk_entries = 8_000_000
        rows_per_chunk = max(
            1,
            min(_TOPOLOGY_SAMPLING_ROWS, max_chunk_entries // max(K, 1)),
        )
        for start in range(0, out_features, rows_per_chunk):
            end = min(start + rows_per_chunk, out_features)
            values = torch.randint(
                in_features,
                (end - start, K),
                dtype=resolved_dtype,
                generator=generator,
            )
            for _ in range(128):
                sorted_values, order = values.sort(dim=1)
                duplicate_sorted = torch.zeros_like(values, dtype=torch.bool)
                duplicate_sorted[:, 1:] = sorted_values[:, 1:] == sorted_values[:, :-1]
                invalid = torch.zeros_like(duplicate_sorted)
                invalid.scatter_(1, order, duplicate_sorted)
                if forbidden is not None:
                    chunk_forbidden = forbidden[start:end].to(dtype=resolved_dtype)
                    invalid |= (chunk_forbidden[:, None] >= 0) & (
                        values == chunk_forbidden[:, None]
                    )
                if not bool(invalid.any()):
                    break
                invalid_count = int(invalid.sum().item())
                values[invalid] = torch.randint(
                    in_features,
                    (invalid_count,),
                    dtype=resolved_dtype,
                    generator=generator,
                )
            else:
                raise RuntimeError(
                    "Unable to sample unique indexed connections after 128 "
                    "rejection rounds."
                )
            indices[start:end] = values
        return indices

    full_arange = (
        torch.arange(in_features, dtype=torch.long) if allowed is None else None
    )

    # Masked sampling intentionally keeps this seeded per-row randperm contract.
    # A rejection sampler is also mathematically valid, but consumes the RNG
    # stream differently and therefore changes the graph generated by an
    # existing seed. That would break topology-regenerated checkpoints whose
    # indices are non-persistent. A future sampling algorithm must use an
    # explicit topology version rather than silently changing this path.
    for out_idx in range(out_features):
        if allowed is None:
            row_allowed = full_arange
        else:
            row_allowed = torch.nonzero(allowed[out_idx], as_tuple=False).flatten()
        if forbidden is not None and forbidden[out_idx] >= 0:
            row_allowed = row_allowed[row_allowed != forbidden[out_idx]]

        if row_allowed.numel() < K:
            raise ValueError(
                "IndexedSparseLinear requires at least K allowed inputs per "
                f"output row; row {out_idx} has {row_allowed.numel()} allowed "
                f"inputs but K={K}."
            )

        perm = torch.randperm(row_allowed.numel(), generator=generator)[:K]
        indices[out_idx] = row_allowed[perm].to(dtype=resolved_dtype)

    return indices


def topology_fingerprint(indices: torch.Tensor) -> int:
    """Return a 62-bit hash of a connection-index tensor.

    The hash covers the shape and the index values (as int64), so it does not
    depend on the index storage dtype or device.
    """
    data = indices.detach().to(device="cpu", dtype=torch.int64).contiguous()
    digest = hashlib.sha256(str(tuple(data.shape)).encode())
    digest.update(data.numpy().tobytes())
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 62) - 1)


def register_topology_fingerprint(module: torch.nn.Module) -> None:
    """Record the fingerprint of a module's regenerated ``connection_indices``.

    Modules that do not save their indices rebuild them from the seed when a
    checkpoint is loaded. The fingerprint is saved instead, so
    :func:`check_topology_fingerprint` can refuse a checkpoint whose topology
    differs from the regenerated one.
    """
    module.register_buffer(
        "topology_fingerprint",
        torch.tensor(topology_fingerprint(module.connection_indices)),
        persistent=True,
    )


def check_topology_fingerprint(
    module: torch.nn.Module,
    state_dict,
    prefix: str,
    missing_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Compare a loaded fingerprint with the module's regenerated topology."""
    key = prefix + "topology_fingerprint"
    if key in missing_keys:
        missing_keys.remove(key)
        warnings.warn(
            f"{prefix or type(module).__name__}: this checkpoint predates topology "
            "fingerprints, so the connection indices regenerated from the seed "
            "cannot be checked against the trained ones.",
            stacklevel=3,
        )
        module.topology_fingerprint.fill_(
            topology_fingerprint(module.connection_indices)
        )
        return
    if key not in state_dict:
        return
    expected = topology_fingerprint(module.connection_indices)
    if int(state_dict[key]) != expected:
        error_msgs.append(
            f"{key}: the checkpoint was trained on a different connection topology "
            "than the one regenerated from the seed (a different seed, sampler or "
            "library version). Load it with the library version that wrote it, or "
            "save checkpoints with persistent_indices=True."
        )


def sample_structured_indices(
    *,
    out_features: int,
    in_features: int,
    K: int,
    generator: torch.Generator,
    index_dtype: str | torch.dtype = "int64",
    support_group_rows: int = 1,
    support_col_block: int = 1,
) -> torch.Tensor:
    """Sample hardware-structured fixed support (row groups x column blocks).

    Two orthogonal constraints make the resulting sparse topology tile-shaped
    for block-sparse execution:

    - ``support_group_rows``: consecutive output rows in groups of this size
      share one sampled column support (their weights stay independent). Each
      (group_rows x K) tile is then dense, and index storage shrinks by the
      same factor.
    - ``support_col_block``: sampled columns arrive as contiguous runs of this
      length, so gathers are coalesced and tiles align to fixed column blocks.

    ``support_group_rows=1, support_col_block=1`` reproduces unstructured
    sampling semantics but through this dedicated path; callers keep the
    historical sampler when no structure is requested. Fails closed when K or
    ``in_features`` is not divisible by ``support_col_block``.
    """

    group_rows = int(support_group_rows)
    col_block = int(support_col_block)
    if group_rows < 1 or col_block < 1:
        raise ValueError(
            "support_group_rows and support_col_block must be >= 1, got "
            f"{support_group_rows} and {support_col_block}"
        )
    if K % col_block:
        raise ValueError(f"K={K} must be divisible by support_col_block={col_block}")
    if in_features % col_block:
        raise ValueError(
            f"in_features={in_features} must be divisible by "
            f"support_col_block={col_block}"
        )
    blocks_total = in_features // col_block
    blocks_per_row = K // col_block
    if blocks_per_row > blocks_total:
        raise ValueError(
            f"K={K} with support_col_block={col_block} needs "
            f"{blocks_per_row} column blocks but only {blocks_total} exist"
        )

    resolved_dtype = _resolve_index_dtype(index_dtype, in_features=in_features)
    n_groups = (out_features + group_rows - 1) // group_rows
    offsets = torch.arange(col_block, dtype=torch.long)
    indices = torch.empty(out_features, K, dtype=resolved_dtype)
    for group in range(n_groups):
        chosen_blocks = torch.randperm(blocks_total, generator=generator)[
            :blocks_per_row
        ]
        columns = (chosen_blocks[:, None] * col_block + offsets[None, :]).reshape(-1)
        row_start = group * group_rows
        row_end = min(row_start + group_rows, out_features)
        indices[row_start:row_end] = columns.to(resolved_dtype)
    return indices


def _workspace_limited_chunk_size(
    *,
    output_chunk_size: int,
    workspace_mb: float | None,
    out_features: int,
    flattened_batch_size: int,
    synapses_per_output: int,
    element_size: int,
) -> int:
    """Cap output rows so the gathered-input workspace fits a byte budget."""
    chunk_size = min(int(output_chunk_size), int(out_features))
    if workspace_mb is None or flattened_batch_size == 0:
        return chunk_size
    bytes_per_output = (
        int(flattened_batch_size) * int(synapses_per_output) * int(element_size)
    )
    budget_bytes = int(float(workspace_mb) * 1024 * 1024)
    return min(chunk_size, max(1, budget_bytes // max(1, bytes_per_output)))


class _IndexedSparseProjectionFunction(torch.autograd.Function):
    """Indexed projection whose backward re-gathers instead of saving gathers."""

    @staticmethod
    def forward(
        ctx,
        flat_x: torch.Tensor,
        weight: torch.Tensor,
        connection_indices: torch.Tensor,
        output_chunk_size: int,
    ) -> torch.Tensor:
        ctx.output_chunk_size = int(output_chunk_size)
        ctx.save_for_backward(flat_x, weight, connection_indices)
        out_features, synapses_per_output = weight.shape
        out = flat_x.new_empty(flat_x.shape[0], out_features)

        for start in range(0, out_features, ctx.output_chunk_size):
            end = min(start + ctx.output_chunk_size, out_features)
            idx_chunk = connection_indices[start:end]
            selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                flat_x.shape[0],
                end - start,
                synapses_per_output,
            )
            out[:, start:end] = (selected * weight[start:end].unsqueeze(0)).sum(dim=-1)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
        grad_weight = torch.empty_like(weight) if ctx.needs_input_grad[1] else None

        for start in range(0, out_features, ctx.output_chunk_size):
            end = min(start + ctx.output_chunk_size, out_features)
            idx_chunk = connection_indices[start:end]
            grad_chunk = grad_output[:, start:end]

            if grad_weight is not None:
                selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                    flat_x.shape[0],
                    end - start,
                    synapses_per_output,
                )
                grad_weight[start:end] = (selected * grad_chunk.unsqueeze(-1)).sum(
                    dim=0
                )

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


def _indexed_sparse_projection(
    *,
    flat_x: torch.Tensor,
    weight: torch.Tensor,
    connection_indices: torch.Tensor,
    output_chunk_size: int,
    recompute_backward: bool,
) -> torch.Tensor:
    """Apply an indexed projection with optional gather recomputation."""
    if recompute_backward and torch.is_grad_enabled():
        return _IndexedSparseProjectionFunction.apply(
            flat_x,
            weight,
            connection_indices,
            output_chunk_size,
        )

    out_features, synapses_per_output = weight.shape
    out = flat_x.new_empty(flat_x.shape[0], out_features)
    for start in range(0, out_features, output_chunk_size):
        end = min(start + output_chunk_size, out_features)
        idx_chunk = connection_indices[start:end]
        selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
            flat_x.shape[0],
            end - start,
            synapses_per_output,
        )
        out[:, start:end] = (selected * weight[start:end].unsqueeze(0)).sum(dim=-1)
    return out


class ContiguousBlockLayout(NamedTuple):
    """Index-free connectivity: output row ``r`` reads inputs ``[b_r K, (b_r + 1) K)``.

    The whole topology is one block id per row; ``order``/``inverse`` group the
    rows by block (stable) so the projection runs as one GEMM per block.
    """

    block: int
    order: torch.Tensor
    inverse: torch.Tensor
    counts: tuple[int, ...]
    identity: bool


def contiguous_block_layout(
    connection_indices: torch.Tensor, in_features: int
) -> ContiguousBlockLayout:
    """Validate the ``contiguous_blocks`` layout and return its row grouping.

    Every row must read the ``K`` consecutive inputs of one aligned block, in
    order, and ``in_features`` must be a multiple of ``K``.
    """

    rows, block = connection_indices.shape
    if rows < 1 or block < 1 or in_features % block:
        raise ValueError(
            f"contiguous_blocks needs in_features ({in_features}) to be a multiple "
            f"of the contacts per row ({block})"
        )
    indices = connection_indices.to(torch.long)
    start = indices[:, 0]
    if bool((start % block).ne(0).any()):
        raise ValueError("contiguous_blocks rows must start on a block boundary")
    offsets = torch.arange(block, device=indices.device)
    chunk = max(1, (1 << 24) // block)
    for first in range(0, rows, chunk):
        part = indices[first : first + chunk]
        if not torch.equal(part, start[first : first + chunk, None] + offsets):
            raise ValueError(
                "contiguous_blocks rows must read consecutive inputs of one block, "
                "in order"
            )
    block_ids = start // block
    blocks = in_features // block
    order = torch.argsort(block_ids, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(rows, device=order.device)
    counts = tuple(int(c) for c in torch.bincount(block_ids, minlength=blocks).tolist())
    identity = bool(torch.equal(order, torch.arange(rows, device=order.device)))
    return ContiguousBlockLayout(block, order, inverse, counts, identity)


def _contiguous_block_projection(
    flat_x: torch.Tensor, weight: torch.Tensor, layout: ContiguousBlockLayout
) -> torch.Tensor:
    """``out[:, r] = x[:, block(r)] @ w[r]`` as one batched GEMM (balanced blocks)
    or one GEMM per block."""

    rows, block = weight.shape
    if block != layout.block or len(layout.inverse) != rows:
        raise ValueError("contiguous block layout does not match the weight shape")
    dtype = torch.promote_types(flat_x.dtype, weight.dtype)
    batch = flat_x.shape[0]
    blocks = flat_x.shape[1] // block
    x_blocks = flat_x.to(dtype).reshape(batch, blocks, block).transpose(0, 1)
    ordered = weight.to(dtype)
    if not layout.identity:
        ordered = ordered.index_select(0, layout.order)
    per_block = layout.counts[0]
    if all(count == per_block for count in layout.counts):
        out = torch.bmm(
            x_blocks, ordered.reshape(blocks, per_block, block).transpose(1, 2)
        )
        out = out.transpose(0, 1).reshape(batch, rows)
    else:
        pieces, first = [], 0
        for index, count in enumerate(layout.counts):
            if count:
                pieces.append(x_blocks[index] @ ordered[first : first + count].T)
                first += count
        out = torch.cat(pieces, dim=1)
    if not layout.identity:
        out = out.index_select(1, layout.inverse)
    return out


def _scatter_dense_projection(
    flat_x: torch.Tensor, weight: torch.Tensor, connection_indices: torch.Tensor
) -> torch.Tensor:
    """``out[n, r] = sum_k weight[r, k] * x[n, idx[r, k]]`` through one dense GEMM.

    Each row's contacts are scatter-added into a dense ``[out, in]`` matrix, so a
    repeated index accumulates exactly as in the gather-sum; autograd returns the
    weight gradient by gathering the dense gradient back at the indices.
    """

    dense = weight.new_zeros(weight.shape[0], flat_x.shape[1]).scatter_add(
        1, connection_indices.long(), weight
    )
    return torch.nn.functional.linear(flat_x, dense.to(flat_x.dtype))


def _dispatch_indexed_sparse_projection(
    *,
    flat_x: torch.Tensor,
    weight: torch.Tensor,
    connection_indices: torch.Tensor,
    output_chunk_size: int,
    projection_backend: str,
    backend_is_resolved: bool = False,
    block_layout: ContiguousBlockLayout | None = None,
    auto_requested: bool = False,
) -> tuple[torch.Tensor, str]:
    """Apply an indexed projection through one validated execution backend.

    ``block_layout`` is an optional cached :func:`contiguous_block_layout` for
    the ``contiguous_blocks`` backend (validated here when omitted).

    Returns the output and the resolution of the request, which callers may
    cache. Triton kernels with int32 element offsets never run on a shape that
    overflows them: for an ``auto`` request (``auto_requested`` marks a cached
    resolution of one) this call alone runs a kernel that fits, leaving the
    returned resolution unchanged for later calls; an explicit request raises
    ``ValueError``. Likewise, under ``torch.use_deterministic_algorithms(True)``
    a projection whose gradients a backend would compute with atomic adds runs
    on a deterministic backend for an ``auto`` request, and an explicit request
    raises ``RuntimeError`` (warns in ``warn_only`` mode).
    """
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
        indexed_kernel_shape_violations,
        nondeterministic_gradient_reason,
        normalize_indexed_projection_backend,
        resolve_indexed_projection_backend,
    )

    if backend_is_resolved:
        resolved = normalize_indexed_projection_backend(projection_backend)
        if resolved == "auto":
            raise ValueError("a resolved indexed projection backend cannot be 'auto'")
    else:
        auto_requested = auto_requested or (
            normalize_indexed_projection_backend(projection_backend) == "auto"
        )
        resolved = resolve_indexed_projection_backend(
            projection_backend,
            device=flat_x.device,
            in_features=flat_x.shape[1],
        )
    shape = {
        "batch_size": int(flat_x.shape[0]),
        "in_features": int(flat_x.shape[1]),
        "out_features": int(connection_indices.shape[0]),
        "synapses_per_output": int(connection_indices.shape[1]),
    }
    backend = resolved
    overflow = indexed_kernel_shape_violations(resolved, **shape)
    if overflow:
        if not auto_requested:
            raise ValueError(
                f"indexed_projection_backend={resolved!r} computes int32 element "
                f"offsets and cannot run this projection ({', '.join(overflow)} "
                ">= 2^31). Use 'auto', 'recompute' or 'eager' for this shape."
            )
        backend = resolve_indexed_projection_backend(
            "auto", device=flat_x.device, **shape
        )
    if torch.are_deterministic_algorithms_enabled() and torch.is_grad_enabled():
        reason = nondeterministic_gradient_reason(
            backend,
            **shape,
            input_gradient=flat_x.requires_grad,
            weight_gradient=weight.requires_grad,
        )
        if reason is not None:
            if auto_requested:
                backend = resolve_indexed_projection_backend(
                    "auto", device=flat_x.device, **shape
                )
            else:
                from dendritic_modeling.utils.reproducibility import (
                    alert_not_deterministic,
                )

                alert_not_deterministic(
                    f"indexed_projection_backend={backend!r} ({reason})",
                    "Use 'auto', which selects a backend with deterministic "
                    "gradients, or 'recompute'.",
                )
    if backend == "contiguous_blocks":
        layout = block_layout
        if layout is None:
            layout = contiguous_block_layout(connection_indices, flat_x.shape[1])
        output = _contiguous_block_projection(flat_x, weight, layout)
    elif backend == "scatter_dense":
        output = _scatter_dense_projection(flat_x, weight, connection_indices)
    elif backend in {"triton_transposed", "triton_transposed_atomic"}:
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed import (
            triton_sparse_gather,
        )

        output = triton_sparse_gather(
            flat_x,
            connection_indices,
            weight,
            chunk_size=output_chunk_size,
            deterministic_input_gradient=backend == "triton_transposed",
        )
    elif backend == "triton_transposed_bf16_compat":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed_bf16_compat import (
            triton_sparse_gather_bf16_compat,
        )

        output = triton_sparse_gather_bf16_compat(
            flat_x,
            connection_indices,
            weight,
            chunk_size=output_chunk_size,
        )
    elif backend == "triton_fused":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_fused_chunked import (
            triton_fused_sparse_gather,
        )

        # Fused chunked kernel: one launch per pass sharing the idx/w reads
        # and the x gather across forward / grad_w / grad_x. auto resolves
        # here for giant input widths, where it is 3.4x (H100) / 8.5x
        # (Blackwell) faster than chunked recompute.
        output = triton_fused_sparse_gather(
            flat_x,
            connection_indices,
            weight,
            chunk_size=output_chunk_size,
        )
    elif backend == "triton_ell":
        from dendritic_modeling.kernels.triton_ell import ell_gather_gemm_train

        # Fused ELL gather-GEMM with the feature-major activation layout
        # (coalesced gathers; measured 7-9x over eager at bench shapes).
        # ``weight`` is already transformed by the caller, so the kernel
        # runs with the identity transform; gradients flow through the
        # transformed tensor into pre_w as with every other backend. The
        # per-call transpose is one small copy of [M, D] and is amortized
        # against the eliminated [M, O, K] traffic; sharing the transposed
        # activations across E/I banks is a follow-up optimization.
        flat_x_fm = flat_x.t().contiguous().t()
        output = ell_gather_gemm_train(
            flat_x_fm,
            connection_indices,
            weight,
            transform="identity",
        ).to(flat_x.dtype)
    else:
        output = _indexed_sparse_projection(
            flat_x=flat_x,
            weight=weight,
            connection_indices=connection_indices,
            output_chunk_size=output_chunk_size,
            recompute_backward=backend == "recompute",
        )
    return output, resolved


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
    forbidden = _validate_forbidden_indices(
        forbidden_input_index_per_output,
        out_features=out_features,
        in_features=in_features,
    )
    if forbidden is None:
        return allowed

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
    "ContiguousBlockLayout",
    "_allowed_mask_with_forbidden",
    "_connection_generator",
    "_dispatch_indexed_sparse_projection",
    "_indexed_sparse_projection",
    "_resolve_index_dtype",
    "_sample_indices_from_mask",
    "_scale_for_sparse_target",
    "_validate_forbidden_indices",
    "_workspace_limited_chunk_size",
    "check_topology_fingerprint",
    "contiguous_block_layout",
    "register_topology_fingerprint",
    "topology_fingerprint",
]
