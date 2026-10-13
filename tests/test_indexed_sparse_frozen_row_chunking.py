"""Row-chunked frozen transform: bitwise parity and workspace bounds.

``_forward_frozen_chunked_transform`` used to chunk only the output axis
while gathering the FULL flattened batch per chunk, so ``workspace_mb``
could not actually bound the gather once a single output column over the
full batch exceeded the budget (the 2026-09-01 battery OOM ledger).  The
fixed implementation also chunks the row axis.  These tests pin:

* bitwise equality against the pre-fix implementation (copied verbatim
  below as the reference) across odd shapes, chunk boundaries, and K,
* the workspace cap on the tile sizes the code actually chooses,
  by tensor-size arithmetic (no CUDA required),
* exact degeneration to a single row chunk whenever the whole flattened
  batch fits the budget (or no budget is configured),
* CPU determinism of the row-chunked path.
"""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    indexed_sparse,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _workspace_limited_chunk_size,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
    _workspace_limited_row_chunk_size,
)
from dendritic_modeling.networks.utils.weight_transforms import apply_weight_transform


def _reference_frozen_chunked_transform(
    layer: IndexedSparseLinear,
    flat_x: torch.Tensor,
    connection_indices: torch.Tensor,
    output_chunk_size: int,
) -> torch.Tensor:
    """Pre-fix implementation, copied verbatim (output chunking only)."""
    out = flat_x.new_empty(flat_x.shape[0], layer.out_features)
    for start in range(0, layer.out_features, output_chunk_size):
        end = min(start + output_chunk_size, layer.out_features)
        idx_chunk = connection_indices[start:end].to(torch.long)
        selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
            flat_x.shape[0],
            end - start,
            layer.K,
        )
        weight_chunk = apply_weight_transform(
            layer.pre_w[start:end],
            layer.weight_transform,
        )
        out[:, start:end] = (selected * weight_chunk.unsqueeze(0)).sum(dim=-1)
    return out


def _make_layer(
    *,
    in_features: int,
    out_features: int,
    K: int,
    output_chunk_size: int,
    workspace_mb: float | None,
    weight_transform: str = "exp",
    seed: int = 7,
) -> IndexedSparseLinear:
    layer = IndexedSparseLinear(
        in_features,
        out_features,
        K,
        weight_transform=weight_transform,
        output_chunk_size=output_chunk_size,
        workspace_mb=workspace_mb,
        seed=seed,
    )
    layer.pre_w.requires_grad_(False)
    layer.eval()
    return layer


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))


@pytest.mark.parametrize("activation_dtype", [torch.float32, torch.bfloat16])
def test_unconfigured_frozen_workspace_is_bounded(monkeypatch, activation_dtype):
    """An unset budget must not recreate the wide-cell calibration OOM."""
    torch.manual_seed(111)
    layer = _make_layer(
        in_features=11,
        out_features=8,
        K=7,
        output_chunk_size=2048,
        workspace_mb=None,
    )
    x = torch.randn(41, 11).to(activation_dtype)
    reference = _reference_frozen_chunked_transform(
        layer, x, layer.connection_indices, layer.output_chunk_size
    )
    budget_mb = 0.0005
    monkeypatch.setattr(indexed_sparse, "_DEFAULT_FROZEN_WORKSPACE_MB", budget_mb)
    observed = []
    original = indexed_sparse._workspace_limited_row_chunk_size

    def record_rows(**kwargs):
        rows = original(**kwargs)
        observed.append(
            rows * kwargs["output_chunk_size"] * layer.K * kwargs["element_size"]
        )
        return rows

    monkeypatch.setattr(
        indexed_sparse, "_workspace_limited_row_chunk_size", record_rows
    )
    assert torch.equal(layer(x), reference)
    assert observed and max(observed) <= int(budget_mb * 1024 * 1024)
    assert layer.workspace_mb is None


def test_default_frozen_workspace_leaves_trainable_dispatch_unchanged(monkeypatch):
    layer = _make_layer(
        in_features=11,
        out_features=8,
        K=7,
        output_chunk_size=2048,
        workspace_mb=None,
    )
    layer.pre_w.requires_grad_(True)
    monkeypatch.setattr(indexed_sparse, "_DEFAULT_FROZEN_WORKSPACE_MB", 0.0001)
    x = torch.randn(41, 11, requires_grad=True)
    assert not layer._can_use_frozen_chunked_transform(x)
    layer(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.pre_w.grad is not None and torch.isfinite(layer.pre_w.grad).all()


@pytest.mark.parametrize("rows", [1, 3, 1025])
@pytest.mark.parametrize(
    ("out_features", "output_chunk_size"),
    [
        (7, 7),  # exact single chunk
        (8, 3),  # ragged final chunk (3, 3, 2)
        (5, 2048),  # configured chunk larger than out_features
    ],
)
@pytest.mark.parametrize("K", [1, 7])
@pytest.mark.parametrize(
    "workspace_mb",
    [None, 0.001, 0.01, 1024.0],
)
def test_row_chunked_transform_is_bitwise_equal_to_reference(
    rows: int,
    out_features: int,
    output_chunk_size: int,
    K: int,
    workspace_mb: float | None,
):
    torch.manual_seed(1234)
    in_features = 11
    layer = _make_layer(
        in_features=in_features,
        out_features=out_features,
        K=K,
        output_chunk_size=output_chunk_size,
        workspace_mb=workspace_mb,
    )
    flat_x = torch.randn(rows, in_features)
    effective_chunk = layer._forward_output_chunk_size(flat_x)
    reference = _reference_frozen_chunked_transform(
        layer, flat_x, layer.connection_indices, effective_chunk
    )
    result = layer._forward_frozen_chunked_transform(
        flat_x, layer.connection_indices, effective_chunk
    )
    assert _bitwise_equal(result, reference)


@pytest.mark.parametrize("weight_transform", ["identity", "softplus"])
def test_row_chunked_transform_bitwise_across_weight_transforms(
    weight_transform: str,
):
    torch.manual_seed(99)
    layer = _make_layer(
        in_features=13,
        out_features=9,
        K=4,
        output_chunk_size=4,
        workspace_mb=0.001,
        weight_transform=weight_transform,
    )
    flat_x = torch.randn(257, 13)
    effective_chunk = layer._forward_output_chunk_size(flat_x)
    reference = _reference_frozen_chunked_transform(
        layer, flat_x, layer.connection_indices, effective_chunk
    )
    result = layer._forward_frozen_chunked_transform(
        flat_x, layer.connection_indices, effective_chunk
    )
    assert _bitwise_equal(result, reference)


def test_forward_frozen_path_matches_reference_under_tiny_budget():
    """End-to-end forward through the frozen path stays bitwise identical."""
    torch.manual_seed(5)
    layer = _make_layer(
        in_features=11,
        out_features=8,
        K=7,
        output_chunk_size=2048,
        workspace_mb=0.01,
    )
    x = torch.randn(41, 25, 11)  # 1025 flattened rows, long-sequence shape
    flat_x = x.reshape(-1, 11)
    assert layer._can_use_frozen_chunked_transform(flat_x)
    effective_chunk = layer._forward_output_chunk_size(flat_x)
    # The tiny budget must actually engage row chunking for this shape.
    row_chunk = _workspace_limited_row_chunk_size(
        flattened_batch_size=flat_x.shape[0],
        output_chunk_size=min(effective_chunk, layer.out_features),
        workspace_mb=layer.workspace_mb,
        synapses_per_output=layer.K,
        element_size=flat_x.element_size(),
    )
    assert row_chunk < flat_x.shape[0]
    reference = _reference_frozen_chunked_transform(
        layer, flat_x, layer.connection_indices, effective_chunk
    )
    out = layer(x)
    assert _bitwise_equal(out, reference.reshape(41, 25, layer.out_features))


@pytest.mark.parametrize("rows", [1, 3, 1025, 100_000])
@pytest.mark.parametrize("K", [1, 7])
@pytest.mark.parametrize("workspace_mb", [0.001, 0.01, 4.0])
def test_chosen_tiles_respect_workspace_budget(rows: int, K: int, workspace_mb: float):
    """The tile the code chooses is bounded by tensor-size arithmetic alone.

    Bound: row_chunk * output_chunk * K * element_size <= budget, except
    that the irreducible minimum tile (one row, one output column, K
    synapses) is allowed when even that exceeds the budget.
    """
    out_features = 33
    element_size = 4  # float32
    configured_chunk = 2048
    budget_bytes = int(workspace_mb * 1024 * 1024)
    output_chunk = _workspace_limited_chunk_size(
        output_chunk_size=configured_chunk,
        workspace_mb=workspace_mb,
        out_features=out_features,
        flattened_batch_size=rows,
        synapses_per_output=K,
        element_size=element_size,
    )
    row_chunk = _workspace_limited_row_chunk_size(
        flattened_batch_size=rows,
        output_chunk_size=min(output_chunk, out_features),
        workspace_mb=workspace_mb,
        synapses_per_output=K,
        element_size=element_size,
    )
    tile_bytes = row_chunk * min(output_chunk, out_features) * K * element_size
    minimum_tile_bytes = 1 * 1 * K * element_size
    assert tile_bytes <= max(budget_bytes, minimum_tile_bytes)
    # Regression: the pre-fix workspace gathered ALL rows per output chunk.
    prefix_workspace_bytes = rows * min(output_chunk, out_features) * K * element_size
    if prefix_workspace_bytes > budget_bytes:
        assert row_chunk < rows  # row chunking engages exactly where it OOMed
    else:
        assert row_chunk == rows  # otherwise: single row chunk, old behavior


def test_row_chunk_degenerates_without_budget_or_under_ample_budget():
    for workspace_mb in (None, 1024.0):
        assert (
            _workspace_limited_row_chunk_size(
                flattened_batch_size=1025,
                output_chunk_size=8,
                workspace_mb=workspace_mb,
                synapses_per_output=7,
                element_size=4,
            )
            == 1025
        )


def test_row_chunk_handles_degenerate_sizes():
    # Empty flattened batch: helper returns a positive step for range().
    assert (
        _workspace_limited_row_chunk_size(
            flattened_batch_size=0,
            output_chunk_size=8,
            workspace_mb=0.001,
            synapses_per_output=7,
            element_size=4,
        )
        == 1
    )
    # Budget below even one (row, output, K) tile still yields a valid step.
    assert (
        _workspace_limited_row_chunk_size(
            flattened_batch_size=10,
            output_chunk_size=1,
            workspace_mb=1e-6,
            synapses_per_output=64,
            element_size=4,
        )
        == 1
    )


def test_empty_batch_shape_is_preserved():
    layer = _make_layer(
        in_features=11,
        out_features=5,
        K=3,
        output_chunk_size=2,
        workspace_mb=0.001,
    )
    flat_x = torch.empty(0, 11)
    out = layer._forward_frozen_chunked_transform(
        flat_x, layer.connection_indices, layer._forward_output_chunk_size(flat_x)
    )
    assert out.shape == (0, 5)


def test_row_chunked_transform_is_deterministic_on_cpu():
    torch.manual_seed(11)
    layer = _make_layer(
        in_features=11,
        out_features=8,
        K=7,
        output_chunk_size=3,
        workspace_mb=0.001,
    )
    flat_x = torch.randn(1025, 11)
    effective_chunk = layer._forward_output_chunk_size(flat_x)
    first = layer._forward_frozen_chunked_transform(
        flat_x, layer.connection_indices, effective_chunk
    )
    for _ in range(3):
        again = layer._forward_frozen_chunked_transform(
            flat_x, layer.connection_indices, effective_chunk
        )
        assert _bitwise_equal(first, again)


@pytest.mark.parametrize(
    "backend, chunked",
    [
        ("eager", True),
        ("recompute", True),
        ("auto", True),
        ("contiguous_blocks", False),
        ("scatter_dense", False),
        ("triton_transposed", False),
        ("triton_transposed_atomic", False),
        ("triton_transposed_bf16_compat", False),
        ("triton_fused", False),
        ("triton_ell", False),
    ],
)
def test_a_frozen_layer_tiles_only_backends_that_gather(backend, chunked):
    # Fused kernels never materialize the [batch, out, K] gather, so a frozen
    # layer configured with one runs that kernel instead of the eager tiles.
    layer = _make_layer(
        in_features=11, out_features=8, K=3, output_chunk_size=2048, workspace_mb=None
    )
    layer.projection_backend = backend
    assert layer._can_use_frozen_chunked_transform(torch.randn(4, 11)) is chunked
