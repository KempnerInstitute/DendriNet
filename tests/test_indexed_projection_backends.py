"""Correctness and dispatch tests for indexed projection backends."""

from __future__ import annotations

import pytest
import torch

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
    cuda_indexed_gather,
    indexed_projection_backend_available,
    normalize_indexed_projection_backend,
    normalize_indexed_projection_options,
    resolve_indexed_projection_backend,
    triton_indexed_gather,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("AUTO", "auto"),
        ("cuda", "cuda"),
        (" eager ", "eager"),
        ("recompute", "recompute"),
        ("TRITON", "triton"),
    ],
)
def test_indexed_projection_backend_names_are_normalized(value, expected):
    assert normalize_indexed_projection_backend(value) == expected


def test_indexed_projection_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="indexed_projection_backend"):
        normalize_indexed_projection_backend("loopish")


def test_legacy_recompute_flag_maps_to_backend_and_rejects_conflicts():
    assert (
        normalize_indexed_projection_options("eager", recompute_backward=True)
        == "recompute"
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        normalize_indexed_projection_options("cuda", recompute_backward=True)


def test_auto_backend_is_portable_on_cpu():
    assert indexed_projection_backend_available("auto", device="cpu")
    assert resolve_indexed_projection_backend("auto", device="cpu") == "recompute"


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_explicit_accelerator_backend_does_not_silently_fallback_on_cpu(backend):
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_indexed_projection_backend(backend, device="cpu")


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


@pytest.mark.parametrize(
    ("capability", "shared_bytes", "expected"),
    [
        ((8, 0), 167_936, ("nostage", 512, 512)),
        ((9, 0), 232_448, ("nostage", 512, 512)),
        ((12, 0), 101_376, ("hybrid", 2048, 512)),
        ((10, 0), 232_448, ("staged", 2048, 512)),
        ((7, 5), 98_304, ("staged", 2048, 512)),
    ],
)
def test_cuda_forward_picker_preserves_measured_architecture_bands(
    monkeypatch,
    capability,
    shared_bytes,
    expected,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(cuda_indexed_gather, "_optin_shared", lambda: shared_bytes)

    # The selected shape is a measured winner on every explicitly optimized
    # architecture. Unknown architectures retain the conservative staged path.
    assert cuda_indexed_gather._pick_fwd_variant(24_000, 48_000, 10) == expected


def _measured_picker_winners(capability):
    inputs = (3_072, 8_000, 16_000, 24_000)
    outputs = (2_000, 48_000, 144_000)
    fan_ins = (10, 15, 20, 25, 30, 40, 50, 60, 70, 80)
    winners = {}
    if capability == (9, 0):
        for fan_in in (10, 15):
            for output in outputs:
                winners[(24_000, output, fan_in)] = "nostage"
        for fan_in in (20, 25, 30, 40, 50, 60):
            winners[(24_000, 144_000, fan_in)] = "nostage"
    elif capability == (12, 0):
        for input_width in (16_000, 24_000):
            for fan_in in (10, 15, 20):
                for output in outputs:
                    winners[(input_width, output, fan_in)] = "hybrid"
        for fan_in in (25, 30, 40, 50, 60):
            winners[(16_000, 2_000, fan_in)] = "hybrid"
            if fan_in <= 50:
                winners[(16_000, 48_000, fan_in)] = "hybrid"
        for fan_in in (25, 30, 40, 50, 60, 70, 80):
            winners[(16_000, 144_000, fan_in)] = "nostage"
        for fan_in in (25, 30, 40):
            winners[(24_000, 144_000, fan_in)] = "nostage"
    else:
        for fan_in in (10, 15, 20, 25, 30, 40):
            for output in outputs:
                winners[(24_000, output, fan_in)] = "nostage"
        for fan_in in (50, 60, 70, 80):
            for output in (48_000, 144_000):
                winners[(24_000, output, fan_in)] = "nostage"
        for fan_in in (10, 15):
            for output in outputs:
                winners[(16_000, output, fan_in)] = "nostage"
        winners[(16_000, 48_000, 20)] = "nostage"
        for fan_in in (20, 25, 30, 40):
            winners[(16_000, 144_000, fan_in)] = "nostage"
    return inputs, outputs, fan_ins, winners


@pytest.mark.parametrize(
    ("capability", "shared_bytes"),
    [((8, 0), 167_936), ((9, 0), 232_448), ((12, 0), 101_376)],
)
def test_cuda_forward_picker_matches_complete_measured_winner_grid(
    monkeypatch,
    capability,
    shared_bytes,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(cuda_indexed_gather, "_optin_shared", lambda: shared_bytes)
    inputs, outputs, fan_ins, expected = _measured_picker_winners(capability)
    actual = {}
    for input_width in inputs:
        for output in outputs:
            for fan_in in fan_ins:
                variant, _, _ = cuda_indexed_gather._pick_fwd_variant(
                    input_width,
                    output,
                    fan_in,
                )
                if variant != "staged":
                    actual[(input_width, output, fan_in)] = variant

    assert actual == expected


@pytest.mark.parametrize(
    ("out_features", "expected_p"),
    [(2_000, 96), (20_000, 256), (48_000, 384), (144_000, 1024), (300_000, 2048)],
)
def test_cuda_backward_picker_preserves_validated_tiles(out_features, expected_p):
    assert cuda_indexed_gather._pick_bwd_wpo_params(out_features) == (
        expected_p,
        512,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["triton", "cuda"])
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend", ["triton", "cuda"])
def test_native_backend_matches_mixed_dtype_eager_contract(backend):
    if not indexed_projection_backend_available(backend, device="cuda"):
        pytest.skip(f"{backend} backend is unavailable")
    native = (
        cuda_indexed_gather.cuda_sparse_gather
        if backend == "cuda"
        else triton_indexed_gather.triton_sparse_gather
    )
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
