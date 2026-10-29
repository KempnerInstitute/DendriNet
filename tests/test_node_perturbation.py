"""Antithetic node-perturbation estimates (``training.node_perturbation``)."""

from __future__ import annotations

import math

import pytest
import torch

from dendritic_modeling.training.node_perturbation import (
    NodePerturbationEstimate,
    check_orthonormal_basis,
    group_basis,
    node_perturbation_estimates,
)

BATCH, SIZE = 6, 5


def _quadratic(seed=0):
    """Per-example losses 0.5 h'A h + b'h with known gradients at h = point."""
    generator = torch.Generator().manual_seed(seed)
    root = torch.randn(BATCH, SIZE, SIZE, generator=generator, dtype=torch.float64)
    a = root @ root.transpose(1, 2)
    b = torch.randn(BATCH, SIZE, generator=generator, dtype=torch.float64)
    point = torch.randn(BATCH, SIZE, generator=generator, dtype=torch.float64)
    calls = []

    def loss(offset):
        calls.append(offset.shape[0])
        h = point + offset
        return 0.5 * torch.einsum("cbi,bij,cbj->cb", h, a, h) + (h * b).sum(-1)

    gradient = torch.einsum("bij,bj->bi", a, point) + b
    return loss, gradient, calls


def _estimate(loss, **kwargs):
    settings = {
        "scale": 0.1,
        "draws": [16],
        "generator": torch.Generator().manual_seed(7),
        "dtype": torch.float64,
    }
    settings.update(kwargs)
    return node_perturbation_estimates(loss, (BATCH, SIZE), **settings)


def test_quadratic_estimates_do_not_depend_on_the_scale():
    # Each antithetic draw gives exactly (g . n) n for a quadratic loss.
    loss, _, _ = _quadratic()
    small = _estimate(loss, scale=1e-3).estimates[16]
    large = _estimate(loss, scale=10.0).estimates[16]
    assert torch.allclose(small, large, rtol=1e-9, atol=1e-9)


def test_estimates_converge_to_the_gradient():
    loss, gradient, _ = _quadratic()
    estimate = _estimate(loss, draws=[20000], chunk=500).estimates[20000]
    cosine = (estimate * gradient).sum(-1) / (
        estimate.norm(dim=-1) * gradient.norm(dim=-1)
    )
    assert cosine.min() > 0.98


def test_a_basis_keeps_estimates_in_its_span_and_converges_to_the_projection():
    loss, gradient, _ = _quadratic()
    basis = group_basis([[0, 1], [2, 3]], SIZE, dtype=torch.float64)
    projector = basis @ basis.T
    found = _estimate(loss, basis=basis, draws=[4, 20000], chunk=500)
    assert found.rank == 2
    small = found.estimates[4]
    assert torch.allclose(
        small - small @ projector, torch.zeros_like(small), atol=1e-12
    )
    projected = gradient @ projector
    assert torch.allclose(
        found.estimates[20000], projected, rtol=0, atol=0.06 * projected.abs().max()
    )


def test_larger_counts_extend_the_draws_of_smaller_ones():
    loss, _, _ = _quadratic()
    nested = _estimate(loss, draws=[4, 16], chunk=3).estimates
    alone = _estimate(loss, draws=[16], chunk=5).estimates[16]
    assert torch.allclose(nested[16], alone, rtol=1e-12, atol=1e-12)
    assert not torch.allclose(nested[4], nested[16])


def test_the_same_generator_seed_reproduces_the_estimate_and_counts_evaluations():
    loss, _, calls = _quadratic()
    first = _estimate(loss, draws=[10], chunk=4)
    assert calls == [4, 4, 4, 4, 2, 2]  # plus and minus per chunk
    second = _estimate(loss, draws=[10], chunk=4)
    assert torch.equal(first.estimates[10], second.estimates[10])
    assert NodePerturbationEstimate.evaluations_per_example(10) == 20


def test_group_basis_columns_are_orthonormal_group_averages():
    basis = group_basis([[0, 2], [4]], SIZE, dtype=torch.float64)
    check_orthonormal_basis(basis, SIZE)
    assert torch.equal(
        basis[:, 0], torch.tensor([1, 0, 1, 0, 0], dtype=torch.float64) / math.sqrt(2)
    )
    assert torch.equal(basis[:, 1], torch.tensor([0, 0, 0, 0, 1], dtype=torch.float64))


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("width", [3, 7, 31, 129])
def test_group_basis_accepts_roundoff_in_its_dtype(dtype, width):
    basis = group_basis([list(range(width)), [width]], width + 1, dtype=dtype)
    check_orthonormal_basis(basis, width + 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_bases_accept_roundoff_but_reject_wrong_norms_and_correlations(dtype):
    generator = torch.Generator().manual_seed(23)
    exact, _ = torch.linalg.qr(
        torch.randn(17, 3, generator=generator, dtype=torch.float64)
    )
    basis = exact.to(dtype)
    check_orthonormal_basis(basis, 17)
    scaled = basis.clone()
    scaled[:, 0] *= 1.1
    with pytest.raises(ValueError, match="orthonormal"):
        check_orthonormal_basis(scaled, 17)
    correlated = basis.clone()
    correlated[:, 1] += 0.1 * correlated[:, 0]
    correlated[:, 1] /= correlated[:, 1].norm()
    with pytest.raises(ValueError, match="orthonormal"):
        check_orthonormal_basis(correlated, 17)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_group_basis_can_estimate_linear_credit(dtype):
    basis = group_basis([[0, 1, 2], [3, 4]], SIZE, dtype=dtype)
    gradient = torch.ones(BATCH, SIZE, dtype=dtype)

    def loss(offset):
        return (offset * gradient).sum(-1)

    found = _estimate(loss, basis=basis, dtype=dtype, draws=[256], chunk=16)
    estimate = found.estimates[256].float()
    exact = gradient.float()
    cosine = (estimate * exact).sum() / (estimate.norm() * exact.norm())
    assert torch.isfinite(estimate).all()
    assert cosine > 0.95


@pytest.mark.parametrize(
    "groups, message",
    [
        ([[0, 1], [1, 2]], "disjoint"),
        ([[0, 0]], "disjoint"),
        ([[0], []], "empty"),
        ([[0, 5]], "outside"),
        ([], "at least one"),
    ],
)
def test_group_basis_rejects_invalid_groups(groups, message):
    with pytest.raises(ValueError, match=message):
        group_basis(groups, SIZE)


def test_group_basis_needs_integer_indices_and_size():
    with pytest.raises(TypeError, match="integer indices"):
        group_basis([[0.0, 1]], SIZE)
    with pytest.raises(TypeError, match="size must be an integer"):
        group_basis([[0, 1]], float(SIZE))


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"scale": 0.0}, "scale must be positive"),
        ({"scale": math.nan}, "scale must be positive"),
        ({"scale": True}, "scale must be a number"),
        ({"draws": []}, "at least one draw count"),
        ({"draws": [0, 4]}, "draw count must be positive"),
        ({"draws": [1.5]}, "draw count must be an integer"),
        ({"draws": [True]}, "draw count must be an integer"),
        ({"draws": 4}, "collection of counts"),
        ({"chunk": 0}, "chunk must be positive"),
        ({"chunk": 2.0}, "chunk must be an integer"),
        ({"basis": torch.ones(SIZE, 2, dtype=torch.float64)}, "orthonormal"),
        ({"basis": torch.eye(SIZE + 1, dtype=torch.float64)}, "shape"),
        ({"basis": torch.eye(SIZE)}, "device and dtype"),
        ({"generator": None}, "torch.Generator"),
    ],
)
def test_invalid_settings_are_refused(settings, message):
    loss, _, _ = _quadratic()
    with pytest.raises((ValueError, TypeError), match=message):
        _estimate(loss, **settings)


def test_a_loss_must_return_per_example_values():
    def summed(offset):
        return offset.sum()

    with pytest.raises(ValueError, match="per-example losses"):
        _estimate(summed)


@pytest.mark.requires_gpu
def test_draws_on_the_gpu_need_a_gpu_generator():
    loss_cpu, _, _ = _quadratic()

    def loss(offset):
        return loss_cpu(offset.cpu()).to(offset.device)

    with pytest.raises(ValueError, match="generator is on"):
        _estimate(loss, device="cuda")
    found = _estimate(
        loss, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)
    )
    assert found.estimates[16].device.type == "cuda"


@pytest.mark.requires_gpu
def test_an_unindexed_cuda_device_accepts_a_basis_on_the_current_device():
    loss_cpu, _, _ = _quadratic()

    def loss(offset):
        return loss_cpu(offset.cpu()).to(offset.device)

    # Tensors report cuda:0; the requested device "cuda" must still match them.
    basis = group_basis([[0, 1], [2, 3, 4]], SIZE, dtype=torch.float64, device="cuda")
    found = _estimate(
        loss,
        device="cuda",
        basis=basis,
        generator=torch.Generator(device="cuda").manual_seed(7),
    )
    assert found.estimates[16].device == basis.device
    # Without a device the draws are made on the generator's.
    default = _estimate(
        loss, basis=basis, generator=torch.Generator(device="cuda").manual_seed(7)
    )
    assert torch.equal(default.estimates[16], found.estimates[16])
