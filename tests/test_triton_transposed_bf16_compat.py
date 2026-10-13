"""Prospective tests for the explicit BF16-product compatibility backend."""

from __future__ import annotations

from math import sqrt

import pytest
import torch
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _indexed_sparse_projection,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    indexed_projection_backend_available,
    normalize_indexed_projection_backend,
    resolve_indexed_projection_backend,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed_bf16_compat import (
    triton_bf16_compat_available,
    triton_sparse_gather_bf16_compat,
)

BACKEND = "triton_transposed_bf16_compat"
BF16_RTOL = 3e-2
BF16_ATOL = 3e-2


def _eager_projection(
    x: torch.Tensor,
    indices: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return _indexed_sparse_projection(
        flat_x=x,
        weight=weight,
        connection_indices=indices,
        output_chunk_size=3,
        recompute_backward=False,
    )


def test_bf16_compat_backend_is_explicit_and_unavailable_on_cpu():
    assert normalize_indexed_projection_backend(BACKEND) == BACKEND
    assert not indexed_projection_backend_available(BACKEND, device="cpu")
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_indexed_projection_backend(BACKEND, device="cpu")


def test_bf16_compat_backend_does_not_change_auto_cpu_policy():
    assert resolve_indexed_projection_backend("auto", device="cpu") == "recompute"


@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float16),
    ],
)
def test_bf16_compat_function_fails_closed_on_unsupported_dtypes(
    input_dtype,
    weight_dtype,
):
    x = torch.randn(2, 8, dtype=input_dtype)
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    weight = torch.randn(2, 2, dtype=weight_dtype)
    with pytest.raises(TypeError, match="requires BF16 x and BF16 weight"):
        triton_sparse_gather_bf16_compat(x, indices, weight)


def test_bf16_compat_function_does_not_fallback_on_cpu():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    weight = torch.randn(2, 2, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires tensors on a CUDA device"):
        triton_sparse_gather_bf16_compat(x, indices, weight)


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("fan_in", [99, 1188, 1551, 1724])
def test_eager_bf16_contract_is_rounded_products_then_fp32_sum(
    batch_size,
    fan_in,
):
    generator = torch.Generator().manual_seed(20260831 + batch_size + fan_in)
    in_features = 2048
    out_features = 3
    x = torch.randn(
        batch_size,
        in_features,
        generator=generator,
        dtype=torch.bfloat16,
    )
    indices = torch.stack(
        [
            torch.randperm(in_features, generator=generator)[:fan_in]
            for _ in range(out_features)
        ]
    ).to(torch.int32)
    weight = torch.randn(
        out_features,
        fan_in,
        generator=generator,
        dtype=torch.bfloat16,
    )
    selected = x[:, indices.reshape(-1)].reshape(
        batch_size,
        out_features,
        fan_in,
    )
    eager = _eager_projection(x, indices, weight)
    explicit_contract = (selected * weight.unsqueeze(0)).float().sum(-1).bfloat16()
    assert torch.equal(eager, explicit_contract)


@pytest.mark.skipif(
    not triton_bf16_compat_available(),
    reason="requires CUDA + Triton",
)
def test_indexed_sparse_linear_dispatches_bf16_compat_for_training():
    indices = torch.tensor(
        [[0, 2, 4], [1, 3, 5], [0, 3, 6], [1, 4, 7]],
        dtype=torch.int32,
    )
    eager = (
        IndexedSparseLinear(
            8,
            4,
            3,
            connection_indices=indices,
            weight_transform="identity",
            projection_backend="eager",
        )
        .cuda()
        .bfloat16()
    )
    compat = (
        IndexedSparseLinear(
            8,
            4,
            3,
            connection_indices=indices,
            weight_transform="identity",
            projection_backend=BACKEND,
        )
        .cuda()
        .bfloat16()
    )
    compat.load_state_dict(eager.state_dict())
    eager_input = torch.randn(
        5,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    compat_input = eager_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn(5, 4, device="cuda", dtype=torch.bfloat16)

    eager_output = eager(eager_input)
    compat_output = compat(compat_input)
    eager_output.backward(grad_output)
    compat_output.backward(grad_output)

    assert compat._last_resolved_projection_backend == BACKEND
    torch.testing.assert_close(
        compat_output,
        eager_output,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        compat_input.grad,
        eager_input.grad,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        compat.pre_w.grad,
        eager.pre_w.grad,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )


@pytest.mark.skipif(
    not triton_bf16_compat_available(),
    reason="requires CUDA + Triton",
)
@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("fan_in", [99, 1188, 1551, 1724])
@pytest.mark.parametrize("projection_family", ["gated", "ei_like"])
def test_bf16_compat_matches_eager_at_production_fan_in(
    batch_size,
    fan_in,
    projection_family,
):
    """Cover deployed K values, decode/prefill M, gating, and E/I arithmetic."""

    generator = torch.Generator(device="cuda").manual_seed(
        20260831 + batch_size + fan_in
    )
    in_features = 2048
    out_features = 7
    indices_a = torch.stack(
        [
            torch.randperm(
                in_features,
                generator=generator,
                device="cuda",
                dtype=torch.int32,
            )[:fan_in]
            for _ in range(out_features)
        ]
    )
    indices_b = torch.stack(
        [
            torch.randperm(
                in_features,
                generator=generator,
                device="cuda",
                dtype=torch.int32,
            )[:fan_in]
            for _ in range(out_features)
        ]
    )
    eager_input = torch.randn(
        batch_size,
        in_features,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    compat_input = eager_input.detach().clone().requires_grad_(True)
    scale = 1.0 / sqrt(fan_in)

    if projection_family == "gated":
        eager_a = (
            torch.randn(
                out_features,
                fan_in,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * scale
        ).requires_grad_(True)
        eager_b = (
            torch.randn(
                out_features,
                fan_in,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * scale
        ).requires_grad_(True)
        compat_a = eager_a.detach().clone().requires_grad_(True)
        compat_b = eager_b.detach().clone().requires_grad_(True)
        eager_output = F.silu(
            _eager_projection(eager_input, indices_a, eager_a)
        ) * _eager_projection(eager_input, indices_b, eager_b)
        compat_output = F.silu(
            triton_sparse_gather_bf16_compat(compat_input, indices_a, compat_a)
        ) * triton_sparse_gather_bf16_compat(
            compat_input,
            indices_b,
            compat_b,
        )
    else:
        # Positive E and I conductance-like banks followed by signed integration.
        eager_a = (
            torch.rand(
                out_features,
                fan_in,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * scale
        ).requires_grad_(True)
        eager_b = (
            torch.rand(
                out_features,
                fan_in,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * scale
        ).requires_grad_(True)
        compat_a = eager_a.detach().clone().requires_grad_(True)
        compat_b = eager_b.detach().clone().requires_grad_(True)
        eager_output = _eager_projection(
            eager_input,
            indices_a,
            eager_a,
        ) - _eager_projection(eager_input, indices_b, eager_b)
        compat_output = triton_sparse_gather_bf16_compat(
            compat_input,
            indices_a,
            compat_a,
        ) - triton_sparse_gather_bf16_compat(
            compat_input,
            indices_b,
            compat_b,
        )

    grad_output = torch.randn(
        eager_output.shape,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    eager_output.backward(grad_output)
    compat_output.backward(grad_output)

    assert eager_output.dtype == compat_output.dtype == torch.bfloat16
    assert torch.isfinite(compat_output).all()
    torch.testing.assert_close(
        compat_output,
        eager_output,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        compat_input.grad,
        eager_input.grad,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        compat_a.grad,
        eager_a.grad,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
    torch.testing.assert_close(
        compat_b.grad,
        eager_b.grad,
        rtol=BF16_RTOL,
        atol=BF16_ATOL,
    )
