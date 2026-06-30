"""Tests for reactivation local-gradient helper formulas."""

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_reactivation_mixin import (
    _reactivation_bias_local_grad,
    _reactivation_log_m_local_grad,
)


def test_reactivation_log_m_local_grad_preserves_formula_with_bias():
    error_signal = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    v_n = torch.tensor([[0.3, 0.6], [0.8, 0.1]])
    v_out = torch.tensor([[0.2, 0.7], [0.9, 0.4]])
    log_m = torch.log(torch.tensor([1.5, 0.75]))
    b = torch.tensor([0.1, 0.25])
    rho = torch.tensor([2.0, -0.5])
    branch_scale = torch.tensor([0.25, 3.0])

    actual = _reactivation_log_m_local_grad(
        error_signal=error_signal,
        v_n=v_n,
        v_out=v_out,
        log_m=log_m,
        b=b,
        rho=rho,
        phi=1.25,
        branch_scale=branch_scale,
    )

    slope = 1 - (2 * v_out - 1) ** 2
    scaled = error_signal * rho.unsqueeze(0) * 1.25 * branch_scale.unsqueeze(0)
    expected = (scaled * slope * (v_n - b)).mean(dim=0) * log_m.exp()
    torch.testing.assert_close(actual, expected)


def test_reactivation_log_m_local_grad_preserves_formula_without_bias():
    error_signal = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    v_n = torch.tensor([[0.3, 0.6], [0.8, 0.1]])
    v_out = torch.tensor([[0.2, 0.7], [0.9, 0.4]])
    log_m = torch.log(torch.tensor([1.5, 0.75]))

    actual = _reactivation_log_m_local_grad(
        error_signal=error_signal,
        v_n=v_n,
        v_out=v_out,
        log_m=log_m,
        b=None,
        rho=2.0,
        phi=1.25,
        branch_scale=0.5,
    )

    slope = 1 - (2 * v_out - 1) ** 2
    expected = (error_signal * 2.0 * 1.25 * 0.5 * slope * v_n).mean(dim=0)
    expected = expected * log_m.exp()
    torch.testing.assert_close(actual, expected)


def test_reactivation_bias_local_grad_preserves_formula():
    error_signal = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    v_out = torch.tensor([[0.2, 0.7], [0.9, 0.4]])
    log_m = torch.log(torch.tensor([1.5, 0.75]))
    rho = torch.tensor([2.0, -0.5])
    branch_scale = torch.tensor([0.25, 3.0])

    actual = _reactivation_bias_local_grad(
        error_signal=error_signal,
        v_out=v_out,
        log_m=log_m,
        rho=rho,
        phi=1.25,
        branch_scale=branch_scale,
    )

    slope = 1 - (2 * v_out - 1) ** 2
    scaled = error_signal * rho.unsqueeze(0) * 1.25 * branch_scale.unsqueeze(0)
    expected = (scaled * (-slope * log_m.exp())).mean(dim=0)
    torch.testing.assert_close(actual, expected)
