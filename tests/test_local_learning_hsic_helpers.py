"""Unit tests for the extracted pure HSIC / additive-gain helpers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_hsic import (
    clamp_phi,
    compute_additive_pseudo_signals,
    compute_dendritic_normalization,
    compute_hsic_gradient,
    compute_kernel_matrix,
)


def _hsic_cfg(kernel="linear"):
    return SimpleNamespace(
        hsic=SimpleNamespace(kernel=kernel, sigma=1.0, coef0=1.0, degree=2, weight=1.0)
    )


def test_kernel_matrix_linear_rbf_poly():
    X = torch.randn(4, 3)
    Y = torch.randn(4, 3)
    torch.testing.assert_close(
        compute_kernel_matrix(_hsic_cfg("linear"), X, Y), X @ Y.t()
    )
    # rbf is symmetric PSD-ish with unit diagonal for X==X
    K = compute_kernel_matrix(_hsic_cfg("rbf"), X, X)
    torch.testing.assert_close(torch.diagonal(K), torch.ones(4))
    torch.testing.assert_close(K, K.t())
    # poly
    P = compute_kernel_matrix(_hsic_cfg("poly"), X, Y)
    torch.testing.assert_close(P, (1.0 + X @ Y.t()) ** 2)
    # unknown -> linear fallback
    torch.testing.assert_close(
        compute_kernel_matrix(_hsic_cfg("bogus"), X, Y), X @ Y.t()
    )


def test_hsic_gradient_linear_shapes_and_self_term():
    z = torch.randn(5, 4)
    y = torch.randn(5, 2)
    g_self = compute_hsic_gradient(
        _hsic_cfg("linear"), z, y, weight=1.0, grad_type="self"
    )
    torch.testing.assert_close(g_self, (2.0 / 5.0) * z)
    g_tgt = compute_hsic_gradient(
        _hsic_cfg("linear"), z, y, weight=1.0, grad_type="target"
    )
    assert g_tgt.shape == z.shape
    # rbf path runs and returns correct shape
    assert compute_hsic_gradient(_hsic_cfg("rbf"), z, y, 1.0, "self").shape == z.shape
    assert (
        compute_hsic_gradient(_hsic_cfg("poly"), z, y, 1.0, "target").shape == z.shape
    )


def test_additive_pseudo_signals():
    v_n = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    rec = {"exc_out": torch.tensor([[1.0, 0.0], [0.0, 1.0]]), "inh_out": None}
    pseudo_R, pseudo_drive = compute_additive_pseudo_signals(rec, v_n)
    torch.testing.assert_close(
        pseudo_R, 1.0 / (1.0 + torch.tensor([[1.0, 0.0], [0.0, 1.0]]) + 1e-8)
    )
    torch.testing.assert_close(pseudo_drive, v_n.mean(dim=0, keepdim=True) - v_n + 1.0)


def test_clamp_phi_bounds_and_swap():
    cfg = SimpleNamespace(
        five_factor=SimpleNamespace(phi_clamp_min=0.25, phi_clamp_max=4.0)
    )
    assert clamp_phi(cfg, 10.0) == 4.0
    assert clamp_phi(cfg, 0.01) == 0.25
    assert clamp_phi(cfg, 1.0) == 1.0
    # inverted bounds are swapped
    swapped = SimpleNamespace(
        five_factor=SimpleNamespace(phi_clamp_min=4.0, phi_clamp_max=0.25)
    )
    assert clamp_phi(swapped, 1.0) == 1.0


def test_dendritic_normalization_passthrough_without_blk():
    grad = torch.randn(3, 4)
    torch.testing.assert_close(compute_dendritic_normalization({}, grad), grad)

    class _Blk:
        def sum_conductances(self):
            return torch.tensor([1.0, 3.0, 1.0])

    out = compute_dendritic_normalization({"blk_module": _Blk()}, torch.ones(3, 2))
    torch.testing.assert_close(
        out, torch.ones(3, 2) / (torch.tensor([[1.0], [3.0], [1.0]]) + 1e-8)
    )
