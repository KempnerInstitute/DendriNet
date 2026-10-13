"""Tests for exact log-time-constant sensitivity formulas."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.utils.tau_sensitivity import (
    decay_log_tau_sensitivity,
    exponential_decay,
    trace_update_log_tau_sensitivity,
)


def test_decay_sensitivity_matches_autograd() -> None:
    log_tau = torch.log(
        torch.tensor([5.0, 25.0, 125.0], dtype=torch.float64)
    ).requires_grad_()
    jacobian = torch.autograd.functional.jacobian(
        lambda value: torch.exp(-1.0 / value.exp()), log_tau
    )
    expected = decay_log_tau_sensitivity(log_tau.exp(), dt=1.0)
    torch.testing.assert_close(torch.diagonal(jacobian), expected)
    torch.testing.assert_close(jacobian, torch.diag(expected))


def test_trace_update_sensitivity_matches_autograd() -> None:
    previous = torch.tensor([0.2, 0.7], dtype=torch.float64)
    drive = torch.tensor([0.8, 0.1], dtype=torch.float64)
    log_tau = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    def update(value: torch.Tensor) -> torch.Tensor:
        rho = torch.exp(-0.5 / value.exp())
        return rho * previous + (1.0 - rho) * drive

    expected = torch.autograd.functional.jacobian(update, log_tau)
    observed = trace_update_log_tau_sensitivity(previous, drive, log_tau.exp(), dt=0.5)
    torch.testing.assert_close(observed, expected)


def test_reference_schedule_has_unequal_local_parameter_sensitivity() -> None:
    tau = torch.tensor([125.0, 25.0, 5.0], dtype=torch.float64)
    sensitivity = decay_log_tau_sensitivity(tau)
    ratio = float(sensitivity[-1] / sensitivity[0])

    assert ratio == pytest.approx(20.62, rel=2e-3)
    assert bool(torch.all(sensitivity[1:] > sensitivity[:-1]))


@pytest.mark.parametrize(
    "tau,dt",
    (
        (torch.tensor([0.0]), 1.0),
        (torch.tensor([1.0]), 0.0),
        (torch.tensor([float("nan")]), 1.0),
    ),
)
def test_invalid_tau_or_dt_fails_closed(tau: torch.Tensor, dt: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        exponential_decay(tau, dt=dt)
