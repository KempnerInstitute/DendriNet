import types

import torch

from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _make_local_ca(*, hsic_weight: float) -> LocalCreditAssignment:
    # Avoid constructing the full trainer; we only need `_compute_hsic_gradient`.
    local_ca = LocalCreditAssignment.__new__(LocalCreditAssignment)
    local_ca.local_cfg = types.SimpleNamespace(
        hsic=types.SimpleNamespace(
            kernel="linear",
            sigma=1.0,
            degree=2,
            coef0=1.0,
            weight=float(hsic_weight),
        )
    )
    return local_ca


def test_hsic_weight_gates_self_and_target_gradients_linear_kernel() -> None:
    torch.manual_seed(0)
    z = torch.randn(8, 5)
    y = torch.randn(8, 3)

    # If hsic.weight == 0, both self and target auxiliary gradients must be zero.
    local_ca0 = _make_local_ca(hsic_weight=0.0)
    grad_self0 = local_ca0._compute_hsic_gradient(z, z, weight=1.0, grad_type="self")
    grad_tgt0 = local_ca0._compute_hsic_gradient(z, y, weight=1.0, grad_type="target")
    assert torch.allclose(grad_self0, torch.zeros_like(grad_self0))
    assert torch.allclose(grad_tgt0, torch.zeros_like(grad_tgt0))

    # hsic.weight should scale both self and target gradients linearly.
    local_ca1 = _make_local_ca(hsic_weight=1.0)
    grad_self1 = local_ca1._compute_hsic_gradient(z, z, weight=1.0, grad_type="self")
    grad_tgt1 = local_ca1._compute_hsic_gradient(z, y, weight=1.0, grad_type="target")

    local_ca2 = _make_local_ca(hsic_weight=2.0)
    grad_self2 = local_ca2._compute_hsic_gradient(z, z, weight=1.0, grad_type="self")
    grad_tgt2 = local_ca2._compute_hsic_gradient(z, y, weight=1.0, grad_type="target")

    assert torch.allclose(grad_self2, 2.0 * grad_self1, atol=1e-6, rtol=1e-6)
    assert torch.allclose(grad_tgt2, 2.0 * grad_tgt1, atol=1e-6, rtol=1e-6)
