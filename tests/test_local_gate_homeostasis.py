import types

import torch

from dendritic_modeling.networks.activations.parametric import (
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _make_local_ca(**gate_kwargs) -> LocalCreditAssignment:
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer.local_cfg = types.SimpleNamespace(
        gate_homeostasis=types.SimpleNamespace(
            enabled=True,
            center_weight=float(gate_kwargs.get("center_weight", 1.0)),
            saturation_weight=float(gate_kwargs.get("saturation_weight", 1.0)),
            target_mean=float(gate_kwargs.get("target_mean", 0.5)),
            target_saturation=float(gate_kwargs.get("target_saturation", 0.2)),
            low_threshold=float(gate_kwargs.get("low_threshold", 0.1)),
            high_threshold=float(gate_kwargs.get("high_threshold", 0.9)),
            saturation_slope=float(gate_kwargs.get("saturation_slope", 20.0)),
        )
    )
    return trainer


def test_gate_homeostasis_aux_grads_match_local_objective_for_parametric_tanh() -> None:
    torch.manual_seed(0)
    trainer = _make_local_ca(center_weight=0.7, saturation_weight=0.4)
    react = ParametricTanh(output_dim=3, init_m=1.3, init_b=0.45)
    v_n = torch.randn(12, 3) * 0.4 + 0.5
    v_out = react(v_n)

    grad_log_m, grad_b = trainer._compute_gate_homeostasis_aux_grads(
        react_module=react,
        v_n=v_n,
        v_out=v_out,
    )

    y = react(v_n)
    mean_r = y.mean(dim=0)
    soft_lo = torch.sigmoid(20.0 * (0.1 - y))
    soft_hi = torch.sigmoid(20.0 * (y - 0.9))
    total_sat = soft_lo.mean(dim=0) + soft_hi.mean(dim=0)
    loss = 0.7 * ((mean_r - 0.5) ** 2).sum() + 0.4 * ((total_sat - 0.2) ** 2).sum()
    expected_log_m, expected_b = torch.autograd.grad(loss, [react.log_m, react.b])

    torch.testing.assert_close(grad_log_m, expected_log_m, atol=1e-6, rtol=1e-6)
    assert grad_b is not None
    torch.testing.assert_close(grad_b, expected_b, atol=1e-6, rtol=1e-6)


def test_gate_homeostasis_aux_grads_support_only_m_variant() -> None:
    torch.manual_seed(1)
    trainer = _make_local_ca(center_weight=0.5, saturation_weight=0.3)
    react = ParametricTanhOnlyM(output_dim=2, init_m=1.1, fixed_b=0.5)
    v_n = torch.randn(10, 2) * 0.2 + 0.5
    v_out = react(v_n)

    grad_log_m, grad_b = trainer._compute_gate_homeostasis_aux_grads(
        react_module=react,
        v_n=v_n,
        v_out=v_out,
    )

    assert grad_log_m is not None
    assert grad_b is None
    assert torch.isfinite(grad_log_m).all()
