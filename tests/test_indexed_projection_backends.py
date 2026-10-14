"""Correctness and dispatch tests for indexed projection backends."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.config.model import SparsityConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    kernels as indexed_kernels,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.degree_grouped_indexed import (
    DegreeGroupedIndexedLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _indexed_sparse_projection,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_dynamic import (
    IndexedDynamicTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    backend as indexed_backend_module,
    indexed_projection_backend_available,
    normalize_indexed_projection_backend,
    normalize_indexed_projection_options,
    resolve_indexed_projection_backend,
    triton_fused_chunked,
    triton_indexed_gather_transposed,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("AUTO", "auto"),
        (" eager ", "eager"),
        ("recompute", "recompute"),
        ("TRITON", "triton_transposed"),  # retired alias resolves to v3
        ("triton_transposed", "triton_transposed"),
        (" Triton_Fused ", "triton_fused"),
        ("triton_ell", "triton_ell"),
    ],
)
def test_indexed_projection_backend_names_are_normalized(value, expected):
    assert normalize_indexed_projection_backend(value) == expected


def test_indexed_projection_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="indexed_projection_backend"):
        normalize_indexed_projection_backend("loopish")


def test_structured_config_normalizes_backend_and_rejects_ignored_alias():
    config = SparsityConfig(indexed={"projection_backend": " AUTO "})
    assert config.indexed["projection_backend"] == "auto"

    with pytest.raises(ValueError, match=r"comparison_backend.*previously ignored"):
        SparsityConfig(
            indexed={
                "projection_backend": "recompute",
                "comparison_backend": "alternate",
            }
        )


def test_structured_config_rejects_unknown_indexed_runtime_option():
    with pytest.raises(ValueError, match=r"Unknown.*projecton_backend"):
        SparsityConfig(indexed={"projecton_backend": "auto"})


def test_legacy_recompute_flag_maps_to_backend_and_rejects_conflicts():
    assert (
        normalize_indexed_projection_options("eager", recompute_backward=True)
        == "recompute"
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_indexed_projection_options(
            "triton_transposed", recompute_backward=True
        )


def test_auto_backend_is_portable_on_cpu():
    assert indexed_projection_backend_available("auto", device="cpu")
    assert resolve_indexed_projection_backend("auto", device="cpu") == "recompute"


@pytest.mark.parametrize("backend", ["triton_transposed", "triton_fused", "triton_ell"])
def test_explicit_accelerator_backend_does_not_silently_fallback_on_cpu(backend):
    assert not indexed_projection_backend_available(backend, device="cpu")
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_indexed_projection_backend(backend, device="cpu")


def test_auto_backend_ignores_in_features_on_cpu():
    assert (
        resolve_indexed_projection_backend("auto", device="cpu", in_features=1_000_000)
        == "recompute"
    )


def test_auto_backend_prefers_fused_kernel_for_giant_inputs(monkeypatch):
    """auto policy: triton_fused at in_features >= 100k, else transposed."""
    available = {"triton_fused", "triton_transposed"}

    def fake_available(backend, *, device=None):
        return (
            indexed_backend_module.normalize_indexed_projection_backend(backend)
            in available
        )

    monkeypatch.setattr(
        indexed_backend_module,
        "indexed_projection_backend_available",
        fake_available,
    )
    resolve = indexed_backend_module.resolve_indexed_projection_backend
    assert resolve("auto", device="cuda", in_features=100_000) == "triton_fused"
    assert resolve("auto", device="cuda", in_features=1_000_000) == "triton_fused"
    # Below the threshold, and with the width unknown, existing behavior holds.
    assert resolve("auto", device="cuda", in_features=99_999) == "triton_transposed"
    assert resolve("auto", device="cuda", in_features=8) == "triton_transposed"
    assert resolve("auto", device="cuda") == "triton_transposed"

    available.discard("triton_fused")
    assert resolve("auto", device="cuda", in_features=1_000_000) == "triton_transposed"


def test_auto_layer_matches_eager_forward_and_gradients_on_cpu():
    indices = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32)
    eager = IndexedSparseLinear(
        6,
        3,
        2,
        connection_indices=indices,
        weight_transform="softplus",
        projection_backend="eager",
    )
    automatic = IndexedSparseLinear(
        6,
        3,
        2,
        connection_indices=indices,
        weight_transform="softplus",
        projection_backend="auto",
    )
    automatic.load_state_dict(eager.state_dict())
    eager_input = torch.randn(4, 6, requires_grad=True)
    automatic_input = eager_input.detach().clone().requires_grad_(True)

    eager_output = eager(eager_input)
    automatic_output = automatic(automatic_input)
    eager_output.square().sum().backward()
    automatic_output.square().sum().backward()

    torch.testing.assert_close(automatic_output, eager_output)
    torch.testing.assert_close(automatic_input.grad, eager_input.grad)
    torch.testing.assert_close(automatic.pre_w.grad, eager.pre_w.grad)
    assert automatic._last_resolved_projection_backend == "recompute"


def test_layer_reuses_resolved_backend_on_same_device(monkeypatch):
    layer = IndexedSparseLinear(
        5,
        2,
        2,
        seed=9,
        projection_backend="auto",
    )
    value = torch.randn(3, 5)
    first = layer(value)

    def unexpected_resolution(*_args, **_kwargs):
        raise AssertionError("backend resolution should be cached")

    monkeypatch.setattr(
        indexed_kernels,
        "resolve_indexed_projection_backend",
        unexpected_resolution,
    )
    torch.testing.assert_close(layer(value), first)


def test_dynamic_and_degree_grouped_layers_share_backend_contract():
    dynamic = IndexedDynamicTopKLinear(
        8,
        4,
        2,
        candidate_size=4,
        seed=3,
        projection_backend="auto",
    )
    grouped = DegreeGroupedIndexedLinear(
        8,
        4,
        [1, 2, 3, 2],
        seed=5,
        projection_backend="auto",
    )
    value = torch.randn(3, 8, requires_grad=True)

    (dynamic(value).sum() + grouped(value).sum()).backward()

    assert dynamic._last_resolved_projection_backend == "recompute"
    assert all(
        group.projection._last_resolved_projection_backend == "recompute"
        for group in grouped.groups
    )
    assert value.grad is not None


def test_degree_grouped_validates_backend_even_without_active_groups():
    with pytest.raises(ValueError, match="indexed_projection_backend"):
        DegreeGroupedIndexedLinear(
            4,
            3,
            [0, 0, 0],
            seed=5,
            projection_backend="unknown",
        )


def test_recompute_backend_accumulates_bfloat16_input_gradient_in_float32():
    batch_size, input_width, output_width, fan_in = 3, 11, 96, 4
    generator = torch.Generator().manual_seed(31)
    indices = torch.stack(
        [
            torch.randperm(input_width, generator=generator)[:fan_in]
            for _ in range(output_width)
        ]
    ).to(torch.int32)
    value = torch.randn(batch_size, input_width, generator=generator).to(torch.bfloat16)
    value.requires_grad_(True)
    weight = torch.randn(output_width, fan_in, generator=generator).to(torch.bfloat16)
    weight.requires_grad_(True)
    grad_output = torch.randn(batch_size, output_width, generator=generator).to(
        torch.bfloat16
    )

    output = _indexed_sparse_projection(
        flat_x=value,
        weight=weight,
        connection_indices=indices,
        output_chunk_size=13,
        recompute_backward=True,
    )
    output.backward(grad_output)

    expected = torch.zeros(batch_size, input_width, dtype=torch.float32)
    for output_index in range(output_width):
        contribution = grad_output[:, output_index].float().unsqueeze(1) * weight[
            output_index
        ].float().unsqueeze(0)
        expected.index_add_(1, indices[output_index], contribution)
    torch.testing.assert_close(value.grad, expected.to(torch.bfloat16), rtol=0, atol=0)


@pytest.mark.requires_gpu
@pytest.mark.parametrize("backend", ["triton_transposed", "triton_fused"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 9])
def test_native_backend_matches_recompute_on_cuda(backend, dtype, batch_size):
    if not indexed_projection_backend_available(backend, device="cuda"):
        pytest.skip(f"{backend} backend is unavailable")
    indices = torch.tensor(
        [[0, 2, 4], [1, 3, 5], [0, 3, 6], [1, 4, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    reference = (
        IndexedSparseLinear(
            8,
            4,
            3,
            connection_indices=indices.cpu(),
            projection_backend="recompute",
        )
        .cuda()
        .to(dtype)
    )
    accelerated = (
        IndexedSparseLinear(
            8,
            4,
            3,
            connection_indices=indices.cpu(),
            projection_backend=backend,
        )
        .cuda()
        .to(dtype)
    )
    accelerated.load_state_dict(reference.state_dict())
    reference_input = torch.randn(
        batch_size,
        8,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    accelerated_input = reference_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn(batch_size, 4, device="cuda", dtype=dtype)

    reference_output = reference(reference_input)
    accelerated_output = accelerated(accelerated_input)
    reference_output.backward(grad_output)
    accelerated_output.backward(grad_output)

    tolerance = (
        2e-5 if dtype == torch.float32 else (2e-3 if dtype == torch.float16 else 3e-2)
    )
    torch.testing.assert_close(
        accelerated_output,
        reference_output,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        accelerated_input.grad,
        reference_input.grad,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        accelerated.pre_w.grad,
        reference.pre_w.grad,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.requires_gpu
@pytest.mark.parametrize("backend", ["triton_transposed", "triton_fused"])
def test_native_backend_matches_mixed_dtype_eager_contract(backend):
    if not indexed_projection_backend_available(backend, device="cuda"):
        pytest.skip(f"{backend} backend is unavailable")
    native = {
        "triton_transposed": triton_indexed_gather_transposed.triton_sparse_gather,
        "triton_fused": triton_fused_chunked.triton_fused_sparse_gather,
    }[backend]
    indices = torch.tensor(
        [[0, 2, 4], [1, 3, 5], [0, 3, 6], [1, 4, 7]],
        device="cuda",
        dtype=torch.int32,
    )
    reference_input = torch.randn(
        3,
        8,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    native_input = reference_input.detach().clone().requires_grad_(True)
    reference_weight = torch.randn(
        4,
        3,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    native_weight = reference_weight.detach().clone().requires_grad_(True)
    grad_output = torch.randn(3, 4, device="cuda", dtype=torch.float32)

    reference_output = _indexed_sparse_projection(
        flat_x=reference_input,
        weight=reference_weight,
        connection_indices=indices,
        output_chunk_size=2,
        recompute_backward=True,
    )
    native_output = native(native_input, indices, native_weight)
    reference_output.backward(grad_output)
    native_output.backward(grad_output)

    assert reference_output.dtype == native_output.dtype == torch.bfloat16
    torch.testing.assert_close(native_output, reference_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(
        native_input.grad,
        reference_input.grad,
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        native_weight.grad,
        reference_weight.grad,
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ("batch_size", "in_features", "out_features", "fan_in"),
    [
        (4, 32, 16, 8),  # small, M=4 single-m-block regime of record
        (5, 33, 17, 7),  # non-divisible vs every tile (B_O=2, B_K=128)
        (37, 45, 23, 5),  # M > 16: multi-m-block atomic grad_w fallback path
    ],
)
def test_triton_fused_matches_eager_forward_and_gradients(
    batch_size, in_features, out_features, fan_in
):
    if not indexed_projection_backend_available("triton_fused", device="cuda"):
        pytest.skip("triton_fused backend is unavailable")
    torch.manual_seed(7)
    indices = torch.randint(
        in_features, (out_features, fan_in), device="cuda", dtype=torch.int32
    )
    indices[0] = indices[-1]  # duplicate rows: shared inputs stress grad_x atomics
    eager_input = torch.randn(
        batch_size, in_features, device="cuda", requires_grad=True
    )
    fused_input = eager_input.detach().clone().requires_grad_(True)
    eager_weight = 0.1 * torch.randn(out_features, fan_in, device="cuda")
    eager_weight.requires_grad_(True)
    fused_weight = eager_weight.detach().clone().requires_grad_(True)
    grad_output = torch.randn(batch_size, out_features, device="cuda")

    eager_output = _indexed_sparse_projection(
        flat_x=eager_input,
        weight=eager_weight,
        connection_indices=indices,
        output_chunk_size=8,
        recompute_backward=False,
    )
    fused_output = triton_fused_chunked.triton_fused_sparse_gather(
        fused_input, indices, fused_weight
    )
    eager_output.backward(grad_output)
    fused_output.backward(grad_output)

    assert fused_output.dtype == eager_output.dtype
    torch.testing.assert_close(fused_output, eager_output, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(fused_input.grad, eager_input.grad, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        fused_weight.grad, eager_weight.grad, rtol=2e-5, atol=2e-5
    )
