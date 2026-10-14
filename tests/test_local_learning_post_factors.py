"""Tests for local-learning post-factor assembly helpers."""

from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_post_factors_mixin import (
    LocalLearningPostFactorsMixin,
    _activation_derivative_tensor,
    _apply_output_modulator,
    _apply_post_factor_modulators,
    _conductance_post_factor,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerModulators,
)


def _modulators(
    *,
    rho=1.0,
    phi=1.0,
    branch_scale=1.0,
) -> _LayerModulators:
    return _LayerModulators(
        rho=rho,
        phi=phi,
        branch_scale=branch_scale,
        dendritic_branch_scale=1.0,
        role_synaptic_alignment=None,
        role_block_alignment=None,
    )


class _PostFactorProbe(LocalLearningPostFactorsMixin):
    """Minimal object exposing the post-factor mixin contract."""

    def __init__(self, *, additive_mode: str = "none", use_driving_force: bool = True):
        self.local_cfg = SimpleNamespace(
            three_factor=SimpleNamespace(
                additive_gain_mode=additive_mode,
                use_driving_force=use_driving_force,
                e_rev_exc=1.5,
                theta=0.25,
            )
        )
        self.calls: list[str] = []

    def _get_layer_activation_derivative(self, *, rec, v_n, v_out):
        self.calls.append("activation")
        return rec.get("activation_derivative")

    def _compute_additive_pseudo_signals(self, rec, v_n):
        self.calls.append("pseudo")
        return rec["pseudo_R"], rec["pseudo_drive"]

    def _get_additive_running_inv_std(self, layer_idx, v_n):
        self.calls.append(f"running:{layer_idx}")
        return torch.full_like(v_n, 0.5)


def test_activation_derivative_tensor_preserves_tensor_and_fallback():
    v_n = torch.ones(2, 3, dtype=torch.float64)
    derivative = torch.arange(3, dtype=torch.float32)

    actual = _activation_derivative_tensor(derivative, v_n)
    fallback = _activation_derivative_tensor(None, v_n)

    assert actual.dtype == v_n.dtype
    assert actual.device == v_n.device
    torch.testing.assert_close(actual, derivative.to(dtype=v_n.dtype))
    torch.testing.assert_close(fallback, torch.ones_like(v_n))


def test_conductance_post_factor_preserves_driving_force_modes():
    e_v = torch.tensor([[0.2, 0.4]])
    v_n = torch.tensor([[0.5, 0.75]])
    r_tot = torch.tensor([[2.0, 3.0]])

    driving = _conductance_post_factor(
        e_v=e_v,
        v_n=v_n,
        r_tot=r_tot,
        use_driving_force=True,
        e_rev_exc=1.5,
        theta=0.25,
    )
    non_driving = _conductance_post_factor(
        e_v=e_v,
        v_n=v_n,
        r_tot=r_tot,
        use_driving_force=False,
        e_rev_exc=1.5,
        theta=0.25,
    )

    torch.testing.assert_close(driving, e_v * r_tot * (1.5 - v_n))
    torch.testing.assert_close(non_driving, e_v * r_tot * (v_n - 0.25))


def test_apply_output_modulator_preserves_scalar_and_neuronwise_modes():
    post_factor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    scalar = _apply_output_modulator(post_factor, 0.5)
    neuronwise = _apply_output_modulator(post_factor, torch.tensor([0.5, 2.0]))

    torch.testing.assert_close(scalar, post_factor * 0.5)
    torch.testing.assert_close(
        neuronwise,
        post_factor * torch.tensor([[0.5, 2.0]]),
    )


def test_apply_post_factor_modulators_preserves_order_and_tensor_modes():
    post_factor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    modulators = _modulators(
        rho=torch.tensor([0.5, 2.0]),
        phi=3.0,
        branch_scale=torch.tensor([4.0, 0.25]),
    )

    actual = _apply_post_factor_modulators(post_factor, modulators)

    expected = post_factor
    expected = expected * torch.tensor([[0.5, 2.0]])
    expected = expected * 3.0
    expected = expected * torch.tensor([[4.0, 0.25]])
    torch.testing.assert_close(actual, expected)


def test_compute_local_voltage_error_preserves_derivative_and_fallback():
    v_n = torch.ones(2, 3, dtype=torch.float64)
    e_n = torch.full_like(v_n, 2.0)
    derivative = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    probe = _PostFactorProbe()

    actual = probe._compute_local_voltage_error(
        rec={"activation_derivative": derivative},
        e_n=e_n,
        v_n=v_n,
    )
    fallback = probe._compute_local_voltage_error(rec={}, e_n=e_n, v_n=v_n)

    torch.testing.assert_close(actual, e_n * derivative.to(dtype=v_n.dtype))
    torch.testing.assert_close(fallback, e_n)
    assert probe.calls == ["activation", "activation"]


def test_compute_local_post_factors_preserves_raw_and_modulated_conductance():
    probe = _PostFactorProbe(use_driving_force=True)
    v_n = torch.tensor([[0.5, 0.75]])
    e_n = torch.tensor([[2.0, 3.0]])
    derivative = torch.tensor([[0.25, 0.5]])
    r_tot = torch.tensor([[2.0, 4.0]])
    modulators = _modulators(
        rho=torch.tensor([0.5, 2.0]),
        phi=3.0,
        branch_scale=torch.tensor([4.0, 0.25]),
    )

    result = probe._compute_local_post_factors(
        rec={"activation_derivative": derivative},
        e_n=e_n,
        v_n=v_n,
        r_tot=r_tot,
        layer_dynamics_mode="conductance",
        layer_idx=2,
        modulators=modulators,
    )

    e_v = e_n * derivative
    raw = e_v * r_tot * (1.5 - v_n)
    expected = _apply_post_factor_modulators(raw, modulators)
    torch.testing.assert_close(result.e_v, e_v)
    torch.testing.assert_close(result.post_factor_raw, raw)
    torch.testing.assert_close(result.post_factor, expected)
    assert probe.calls == ["activation"]


def test_compute_additive_post_factor_preserves_modes():
    v_n = torch.tensor([[0.5, 0.75]])
    e_v = torch.tensor([[2.0, 3.0]])

    pseudo = _PostFactorProbe(additive_mode="input_dependent")
    pseudo_R = torch.tensor([[0.25, 0.5]])
    pseudo_drive = torch.tensor([[4.0, 2.0]])
    pseudo_factor = pseudo._compute_additive_post_factor(
        rec={"pseudo_R": pseudo_R, "pseudo_drive": pseudo_drive},
        e_v=e_v,
        v_n=v_n,
        layer_idx=1,
    )
    torch.testing.assert_close(pseudo_factor, e_v * pseudo_R * pseudo_drive)
    assert pseudo.calls == ["pseudo"]

    running = _PostFactorProbe(additive_mode="running_stats")
    running_factor = running._compute_additive_post_factor(
        rec={},
        e_v=e_v,
        v_n=v_n,
        layer_idx=7,
    )
    torch.testing.assert_close(running_factor, e_v * 0.5)
    assert running.calls == ["running:7"]

    neutral = _PostFactorProbe(additive_mode="none")
    neutral_factor = neutral._compute_additive_post_factor(
        rec={},
        e_v=e_v,
        v_n=v_n,
        layer_idx=1,
    )
    assert neutral_factor is e_v
