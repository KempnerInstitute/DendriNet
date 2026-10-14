"""Correctness tests for the fused ELL gather-GEMM kernels.

CUDA-gated: on CPU-only nodes the wrappers fall back to the eager
reference, so only the fallback contract is checked there.
"""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.kernels import (
    ell_gather_gemm,
    ell_shunting_forward,
    reference_gather_gemm,
    reference_shunting_forward,
    triton_available,
)

CUDA = pytest.mark.skipif(not triton_available(), reason="requires CUDA + triton")


def _case(m=48, d=512, o=96, k=72, device="cpu", index_dtype=torch.int32):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(m, d, generator=generator).to(device)
    indices = torch.stack(
        [torch.randperm(d, generator=generator)[:k] for _ in range(o)]
    ).to(device=device, dtype=index_dtype)
    pre_w = torch.randn(o, k, generator=generator).to(device)
    return x, indices, pre_w


def test_cpu_fallback_matches_reference() -> None:
    x, indices, pre_w = _case()
    torch.testing.assert_close(
        ell_gather_gemm(x, indices, pre_w, "softplus"),
        reference_gather_gemm(x, indices, pre_w, "softplus"),
    )


@CUDA
@pytest.mark.parametrize("transform", ["identity", "softplus", "exp"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gather_gemm_matches_reference_on_cuda(transform, index_dtype) -> None:
    x, indices, pre_w = _case(device="cuda", index_dtype=index_dtype)
    fused = ell_gather_gemm(x, indices, pre_w, transform)
    expected = reference_gather_gemm(x, indices, pre_w, transform)
    torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_gather_gemm_supports_transposed_activation_layout() -> None:
    x, indices, pre_w = _case(device="cuda")
    x_dm = x.t().contiguous().t()  # [M, D] view over [D, M]-contiguous storage
    fused = ell_gather_gemm(x_dm, indices, pre_w, "softplus")
    expected = reference_gather_gemm(x, indices, pre_w, "softplus")
    torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_gather_gemm_ragged_block_edges() -> None:
    # M and K deliberately not multiples of the block sizes.
    x, indices, pre_w = _case(m=37, d=300, o=13, k=41, device="cuda")
    fused = ell_gather_gemm(x, indices, pre_w, "softplus", block_m=32, block_k=32)
    expected = reference_gather_gemm(x, indices, pre_w, "softplus")
    torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_shunting_forward_matches_reference_on_cuda() -> None:
    x, idx_exc, w_exc = _case(device="cuda")
    _, idx_inh, w_inh = _case(o=96, k=40, device="cuda")
    fused = ell_shunting_forward(
        x, idx_exc, w_exc, idx_inh, w_inh, somatic=0.3, conductance=0.1
    )
    expected = reference_shunting_forward(
        x, idx_exc, w_exc, idx_inh, w_inh, somatic=0.3, conductance=0.1
    )
    torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4)
    assert bool((fused >= 0).all()) and bool((fused <= 1.3).all())


@CUDA
@pytest.mark.parametrize("transform", ["identity", "softplus", "exp"])
def test_train_gradients_match_reference(transform) -> None:
    from dendritic_modeling.kernels.triton_ell import ell_gather_gemm_train

    x, indices, pre_w = _case(m=33, d=200, o=17, k=29, device="cuda")
    x_ref = x.clone().requires_grad_(True)
    w_ref = pre_w.clone().requires_grad_(True)
    x_fused = x.clone().requires_grad_(True)
    w_fused = pre_w.clone().requires_grad_(True)

    reference = reference_gather_gemm(x_ref, indices, w_ref, transform)
    fused = ell_gather_gemm_train(x_fused, indices, w_fused, transform)
    torch.testing.assert_close(fused, reference, rtol=1e-4, atol=1e-4)

    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    fused.backward(upstream)
    torch.testing.assert_close(w_fused.grad, w_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=1e-3, atol=1e-3)


def test_train_cpu_fallback_is_differentiable() -> None:
    from dendritic_modeling.kernels.triton_ell import ell_gather_gemm_train

    x, indices, pre_w = _case(m=8, d=32, o=5, k=6)
    x.requires_grad_(True)
    pre_w.requires_grad_(True)
    ell_gather_gemm_train(x, indices, pre_w).sum().backward()
    assert x.grad is not None and pre_w.grad is not None


def test_layer_backend_rejects_cpu_loudly() -> None:
    """Explicit accelerator backends fail loudly off-GPU (no silent demotion,
    matching the triton/cuda backend policy)."""
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
        IndexedSparseLinear,
    )

    layer = IndexedSparseLinear(
        in_features=64,
        out_features=12,
        K=16,
        seed=3,
        weight_transform="softplus",
        projection_backend="triton_ell",
    )
    if triton_available():
        pytest.skip("CUDA present; CPU rejection not applicable")
    with pytest.raises(RuntimeError, match="triton_ell"):
        layer(torch.rand(5, 64))


@CUDA
def test_layer_backend_matches_eager_on_cuda_forward_backward() -> None:
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
        IndexedSparseLinear,
    )

    common = {
        "in_features": 256,
        "out_features": 32,
        "K": 48,
        "seed": 3,
        "weight_transform": "softplus",
    }
    eager = IndexedSparseLinear(projection_backend="eager", **common).cuda()
    fused = IndexedSparseLinear(projection_backend="triton_ell", **common).cuda()
    with torch.no_grad():
        fused.pre_w.copy_(eager.pre_w)
        fused.connection_indices.copy_(eager.connection_indices)

    x = torch.rand(17, 256, device="cuda")
    out_eager = eager(x)
    out_fused = fused(x)
    torch.testing.assert_close(out_fused, out_eager, rtol=1e-4, atol=1e-4)

    out_eager.sum().backward()
    out_fused.sum().backward()
    torch.testing.assert_close(fused.pre_w.grad, eager.pre_w.grad, rtol=1e-3, atol=1e-3)


def test_glu_cpu_fallback_matches_reference() -> None:
    from dendritic_modeling.kernels import ell_glu_forward, reference_glu_forward

    x, idx_g, w_g = _case()
    _, idx_u, w_u = _case(k=48)
    torch.testing.assert_close(
        ell_glu_forward(x, idx_g, w_g, idx_u, w_u, "identity"),
        reference_glu_forward(x, idx_g, w_g, idx_u, w_u, "identity"),
    )


@CUDA
@pytest.mark.parametrize("transform", ["identity", "softplus"])
def test_glu_matches_reference_on_cuda(transform) -> None:
    from dendritic_modeling.kernels import ell_glu_forward, reference_glu_forward

    x, idx_g, w_g = _case(device="cuda")
    _, idx_u, w_u = _case(k=40, device="cuda")
    fused = ell_glu_forward(x, idx_g, w_g, idx_u, w_u, transform)
    expected = reference_glu_forward(x, idx_g, w_g, idx_u, w_u, transform)
    torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_glu_ragged_and_feature_major() -> None:
    from dendritic_modeling.kernels import ell_glu_forward, reference_glu_forward

    x, idx_g, w_g = _case(m=37, d=300, o=13, k=41, device="cuda")
    _, idx_u, w_u = _case(m=37, d=300, o=13, k=29, device="cuda")
    x_dm = x.t().contiguous().t()
    fused = ell_glu_forward(
        x_dm, idx_g, w_g, idx_u, w_u, "identity", block_m=32, block_k=32
    )
    expected = reference_glu_forward(x, idx_g, w_g, idx_u, w_u, "identity")
    torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-4)


@CUDA
def test_glu_train_gradients_match_reference() -> None:
    from dendritic_modeling.kernels import ell_glu_train, reference_glu_forward

    x, idx_g, w_g = _case(m=33, d=200, o=17, k=29, device="cuda")
    _, idx_u, w_u = _case(m=33, d=200, o=17, k=23, device="cuda")
    x_ref = x.clone().requires_grad_(True)
    wg_ref = w_g.clone().requires_grad_(True)
    wu_ref = w_u.clone().requires_grad_(True)
    x_fused = x.clone().requires_grad_(True)
    wg_fused = w_g.clone().requires_grad_(True)
    wu_fused = w_u.clone().requires_grad_(True)

    reference = reference_glu_forward(x_ref, idx_g, wg_ref, idx_u, wu_ref, "identity")
    fused = ell_glu_train(x_fused, idx_g, wg_fused, idx_u, wu_fused, "identity")
    torch.testing.assert_close(fused, reference, rtol=1e-4, atol=1e-4)

    upstream = torch.randn_like(reference)
    reference.backward(upstream)
    fused.backward(upstream)
    torch.testing.assert_close(wg_fused.grad, wg_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(wu_fused.grad, wu_ref.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=1e-3, atol=1e-3)


def test_glu_train_cpu_fallback_is_differentiable() -> None:
    from dendritic_modeling.kernels import ell_glu_train

    x, idx_g, w_g = _case(m=8, d=32, o=5, k=6)
    _, idx_u, w_u = _case(m=8, d=32, o=5, k=6)
    x.requires_grad_(True)
    w_g.requires_grad_(True)
    w_u.requires_grad_(True)
    ell_glu_train(x, idx_g, w_g, idx_u, w_u).sum().backward()
    assert x.grad is not None and w_g.grad is not None and w_u.grad is not None
