"""Indexed projections under ``torch.use_deterministic_algorithms(True)``.

Backends whose gradients accumulate with atomic adds vary in the last bits
from run to run. Under deterministic algorithms ``auto`` avoids them and an
explicit request raises (warns in ``warn_only`` mode) once it would compute
such a gradient.
"""

from __future__ import annotations

import hashlib
import warnings

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _dispatch_indexed_sparse_projection,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    NONDETERMINISTIC_INDEXED_BACKENDS,
    backend as backend_module,
    nondeterministic_gradient_reason,
    triton_indexed_gather_transposed,
)
from dendritic_modeling.utils.reproducibility import alert_not_deterministic


@pytest.fixture
def deterministic_algorithms():
    """Set PyTorch's deterministic mode for one test and restore it after."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()

    def set_mode(mode: str) -> None:
        torch.use_deterministic_algorithms(mode != "off", warn_only=mode == "warn")

    yield set_mode
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _shape(batch_size=4, in_features=16, out_features=8, synapses_per_output=3):
    return {
        "batch_size": batch_size,
        "in_features": in_features,
        "out_features": out_features,
        "synapses_per_output": synapses_per_output,
    }


def test_alert_follows_the_deterministic_mode(deterministic_algorithms):
    deterministic_algorithms("off")
    alert_not_deterministic("op", "remedy")

    deterministic_algorithms("warn")
    with pytest.warns(UserWarning, match=r"op is not deterministic.*remedy"):
        alert_not_deterministic("op", "remedy")

    deterministic_algorithms("strict")
    with pytest.raises(RuntimeError, match=r"op is not deterministic.*remedy"):
        alert_not_deterministic("op", "remedy")


@pytest.mark.parametrize("backend", sorted(NONDETERMINISTIC_INDEXED_BACKENDS))
def test_atomic_backends_are_nondeterministic_only_for_input_gradients(backend):
    def reason(**gradients):
        return nondeterministic_gradient_reason(backend, **_shape(), **gradients)

    assert "input gradient" in reason(input_gradient=True, weight_gradient=False)
    assert reason(input_gradient=False, weight_gradient=False) is None
    assert reason(input_gradient=False, weight_gradient=True) is None


def test_fused_weight_gradient_is_atomic_past_one_batch_block():
    def reason(batch_size):
        return nondeterministic_gradient_reason(
            "triton_fused",
            **_shape(batch_size=batch_size),
            input_gradient=False,
            weight_gradient=True,
        )

    assert reason(16) is None
    assert "weight gradient" in reason(17)


@pytest.mark.parametrize(
    "backend",
    ["auto", "eager", "recompute", "scatter_dense", "contiguous_blocks"],
)
def test_portable_backends_are_deterministic(backend):
    assert (
        nondeterministic_gradient_reason(
            backend, **_shape(), input_gradient=True, weight_gradient=True
        )
        is None
    )


@pytest.mark.parametrize(
    "backend", ["triton_transposed", "triton_transposed_bf16_compat"]
)
def test_transposed_kernels_report_their_atomic_fallback(backend, monkeypatch):
    def reason():
        return nondeterministic_gradient_reason(
            backend, **_shape(), input_gradient=True, weight_gradient=True
        )

    assert reason() is None
    monkeypatch.setattr(
        triton_indexed_gather_transposed, "_DET_GX_WORKSPACE_CAP_BYTES", 64
    )
    assert "falls back to atomic adds" in reason()


def _available(names):
    def available(backend, *, device=None):
        return backend_module.normalize_indexed_projection_backend(backend) in names

    return available


def test_auto_selects_deterministic_kernels_under_deterministic_algorithms(
    monkeypatch, deterministic_algorithms
):
    monkeypatch.setattr(
        backend_module,
        "indexed_projection_backend_available",
        _available({"triton_fused", "triton_transposed"}),
    )
    resolve = backend_module.resolve_indexed_projection_backend
    deterministic_algorithms("off")
    assert resolve("auto", device="cuda", in_features=200_000) == "triton_fused"

    deterministic_algorithms("strict")
    assert resolve("auto", device="cuda", in_features=200_000) == "triton_transposed"
    assert resolve("auto", device="cuda", in_features=8) == "triton_transposed"
    # Past the deterministic input-gradient workspace, recompute.
    monkeypatch.setattr(
        triton_indexed_gather_transposed, "_DET_GX_WORKSPACE_CAP_BYTES", 64
    )
    assert resolve("auto", device="cuda", **_shape()) == "recompute"
    # Explicit requests are checked when the projection runs.
    assert resolve("triton_fused", device="cuda") == "triton_fused"


def _project(backend, *, auto_requested=False, input_gradient=True):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(4, 16, generator=generator, requires_grad=input_gradient)
    weight = torch.randn(8, 3, generator=generator, requires_grad=True)
    indices = torch.randint(0, 16, (8, 3), generator=generator)
    return _dispatch_indexed_sparse_projection(
        flat_x=x,
        weight=weight,
        connection_indices=indices,
        output_chunk_size=8,
        projection_backend=backend,
        backend_is_resolved=True,
        auto_requested=auto_requested,
    )


@pytest.mark.parametrize("backend", sorted(NONDETERMINISTIC_INDEXED_BACKENDS))
def test_explicit_atomic_backend_raises_under_deterministic_algorithms(
    backend, deterministic_algorithms
):
    deterministic_algorithms("strict")
    with pytest.raises(RuntimeError, match=f"indexed_projection_backend='{backend}'"):
        _project(backend)


def test_cached_auto_resolution_moves_to_a_deterministic_backend(
    deterministic_algorithms,
):
    # A layer resolved auto to triton_fused before deterministic algorithms
    # were enabled; on CPU the deterministic choice is recompute.
    deterministic_algorithms("strict")
    output, resolved = _project("triton_fused", auto_requested=True)
    reference, _ = _project("recompute")

    assert resolved == "triton_fused"
    assert torch.equal(output, reference)


def test_inference_does_not_check_gradients(deterministic_algorithms):
    deterministic_algorithms("strict")
    with torch.no_grad():
        output, _ = _project("recompute")
    assert output.shape == (4, 8)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def _gradient_digests(backend: str, in_features: int, repeats: int = 4) -> set[str]:
    torch.manual_seed(0)
    layer = IndexedSparseLinear(
        in_features, 256, 32, seed=1, projection_backend=backend
    ).cuda()
    x = torch.randn(4, in_features, device="cuda")
    weights = torch.linspace(-1, 1, 256, device="cuda")
    digests = set()
    for _ in range(repeats):
        layer.zero_grad(set_to_none=True)
        inputs = x.clone().requires_grad_(True)
        (layer(inputs) * weights).sum().backward()
        digests.add(_digest(inputs.grad) + _digest(layer.pre_w.grad))
    return digests


@pytest.mark.requires_gpu
def test_auto_gradients_are_bitwise_reproducible_under_strict_mode(
    deterministic_algorithms,
):
    if not backend_module.indexed_projection_backend_available(
        "triton_transposed", device="cuda"
    ):
        pytest.skip("Triton kernels are unavailable")
    deterministic_algorithms("strict")
    # At this width auto would otherwise select the atomic fused kernel.
    assert len(_gradient_digests("auto", 120_000)) == 1
    with pytest.raises(RuntimeError, match="triton_transposed_atomic"):
        _gradient_digests("triton_transposed_atomic", 4096)

    deterministic_algorithms("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _gradient_digests("triton_transposed_atomic", 4096, repeats=1)
    assert any("not deterministic" in str(warning.message) for warning in caught)
