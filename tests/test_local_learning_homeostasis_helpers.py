"""Unit tests for the extracted pure homeostasis helpers (local credit assignment)."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_homeostasis import (
    compute_gate_homeostasis_aux_grads,
    compute_inhibitory_homeostasis_factor,
    compute_voltage_homeostasis_error,
)


def test_inhibitory_homeostasis_disabled_returns_none():
    cfg = SimpleNamespace(inhibitory_homeostasis=None)
    assert (
        compute_inhibitory_homeostasis_factor(
            cfg, {}, torch.tensor(0.3), torch.zeros(2)
        )
        is None
    )
    cfg2 = SimpleNamespace(
        inhibitory_homeostasis=SimpleNamespace(enabled=False, weight=1.0)
    )
    assert (
        compute_inhibitory_homeostasis_factor(
            cfg2, {}, torch.tensor(0.3), torch.zeros(2)
        )
        is None
    )


def test_inhibitory_homeostasis_r_tot_mode():
    cfg = SimpleNamespace(
        inhibitory_homeostasis=SimpleNamespace(
            enabled=True, weight=1.0, mode="r_tot", target_r_tot=0.3
        )
    )
    R_tot = torch.tensor([0.5, 0.2])
    out = compute_inhibitory_homeostasis_factor(cfg, {}, R_tot, torch.zeros(2))
    torch.testing.assert_close(out, -1.0 * (R_tot - 0.3) * (R_tot**2))


def test_inhibitory_homeostasis_voltage_mode():
    cfg = SimpleNamespace(
        inhibitory_homeostasis=SimpleNamespace(
            enabled=True, weight=1.0, mode="voltage", target_voltage=0.1
        ),
        three_factor=SimpleNamespace(e_rev_inh=0.0),
    )
    v_n = torch.tensor([0.4, 0.2])
    R_tot = torch.tensor([1.0, 1.0])
    out = compute_inhibitory_homeostasis_factor(cfg, {}, R_tot, v_n)
    torch.testing.assert_close(out, 1.0 * (v_n - 0.1) * R_tot * (0.0 - v_n))


def test_inhibitory_homeostasis_unknown_mode_returns_none():
    cfg = SimpleNamespace(
        inhibitory_homeostasis=SimpleNamespace(enabled=True, weight=1.0, mode="bogus")
    )
    assert (
        compute_inhibitory_homeostasis_factor(
            cfg, {}, torch.tensor(0.3), torch.zeros(2)
        )
        is None
    )


def test_voltage_homeostasis_error():
    cfg = SimpleNamespace(
        voltage_homeostasis=SimpleNamespace(
            enabled=True, weight=2.0, target_voltage=0.5
        )
    )
    v_n = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(
        compute_voltage_homeostasis_error(cfg, v_n), 2.0 * (0.5 - v_n)
    )
    assert (
        compute_voltage_homeostasis_error(
            SimpleNamespace(voltage_homeostasis=None), v_n
        )
        is None
    )


def test_gate_homeostasis_guards_return_none():
    cfg = SimpleNamespace(gate_homeostasis=None)
    assert compute_gate_homeostasis_aux_grads(cfg, None, None, None) == (None, None)
    # non-ParametricActivation react module is rejected
    cfg2 = SimpleNamespace(gate_homeostasis=SimpleNamespace(enabled=True))
    assert compute_gate_homeostasis_aux_grads(
        cfg2, torch.nn.ReLU(), torch.zeros(2, 3), torch.zeros(2, 3)
    ) == (None, None)
