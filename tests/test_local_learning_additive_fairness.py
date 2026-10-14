"""Tests for additive-aware LocalCA dynamics modes."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


class _SingleBranchModel(nn.Module):
    """Small model exposing a single dendritic branch layer."""

    def __init__(self, *, use_shunting: bool, topk_type: str = "standard"):
        super().__init__()
        indexed_options = {}
        if topk_type == "indexed":
            indices = torch.tensor([[0, 1], [2, 3]])
            indexed_options = {
                "excitatory_connection_indices": indices,
                "inhibitory_connection_indices": indices,
            }
        self.branch = DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=4,
            excitatory_synapses_per_branch=2,
            inhibitory_input_dim=4,
            inhibitory_synapses_per_branch=2,
            input_branch_factor=2,
            use_shunting=use_shunting,
            reactivate=False,
            topk_noise_level=0.0,
            topk_type=topk_type,
            weight_transform="identity",
            **indexed_options,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x, inhibitory_input=x, branch_input=x)


class _DummyBlock:
    """Minimal block module for path-factor unit tests."""

    def __init__(self, weight: torch.Tensor, block_size: int):
        self._weight = weight
        self.block_size = block_size

    def weight(self) -> torch.Tensor:
        return self._weight


def _make_trainer(
    *,
    dynamics_mode: str,
    error_broadcast_mode: str = "per_soma",
    broadcast_rank: int = 4,
    use_conductance_scaling: bool = True,
    use_driving_force: bool = True,
    use_path_propagation: bool = False,
    path_factor_mode: str = "per_branch",
    inhibitory_homeostasis: dict | None = None,
) -> LocalCreditAssignment:
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer.local_cfg = trainer._build_local_rule_config(
        {
            "rule_variant": "3f",
            "error_broadcast_mode": error_broadcast_mode,
            "broadcast_rank": broadcast_rank,
            "normalize_by_batch": True,
            "update_inactive_weights": True,
            "three_factor": {
                "dynamics_mode": dynamics_mode,
                "use_conductance_scaling": use_conductance_scaling,
                "use_driving_force": use_driving_force,
                "theta": 0.0,
                "e_rev_exc": 1.0,
                "e_rev_inh": 0.0,
            },
            "morphology_aware": {
                "use_path_propagation": use_path_propagation,
                "path_factor_mode": path_factor_mode,
            },
            "inhibitory_homeostasis": inhibitory_homeostasis
            or {"enabled": False, "weight": 0.0},
        }
    )
    trainer._layer_stats = {}
    trainer._decoder_cache = {}
    trainer._broadcast_cache = {}
    return trainer


def _capture_record(
    trainer: LocalCreditAssignment, model: nn.Module, x_batch: torch.Tensor
) -> dict:
    records, handles = trainer._attach_local_recorders(model)
    try:
        _ = model(x_batch)
    finally:
        for handle in handles:
            handle.remove()
    assert len(records) == 1
    return records[0]


def _run_local_update(
    trainer: LocalCreditAssignment,
    model: _SingleBranchModel,
    rec: dict,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    trainer._apply_local_rule_gradients(
        model=model,
        layer_records=[rec],
        v0=torch.zeros_like(delta),
        delta=delta,
        y_target=None,
        update_reactivation=False,
    )
    exc_grad = model.branch.branch_excitation.pre_w.grad.detach().clone()
    inh_grad = model.branch.branch_inhibition.pre_w.grad.detach().clone()
    den_grad = model.branch.branches_to_output.log_weight.grad.detach().clone()
    return exc_grad, inh_grad, den_grad


def test_topk_voltage_homeostasis_gradient_moves_voltages_to_the_target() -> None:
    """The homeostasis error is dL/dV, applied through each path's sensitivity."""
    trainer = _make_trainer(dynamics_mode="conductance")
    trainer.local_cfg.voltage_homeostasis.enabled = True
    trainer.local_cfg.voltage_homeostasis.weight = 2.0
    trainer.local_cfg.voltage_homeostasis.target_voltage = 0.25

    v_n = torch.tensor([[0.1, 0.4], [0.3, -0.2]], dtype=torch.float32)
    x_pre = torch.tensor([[0.5, -0.1, 0.2], [0.7, 0.3, -0.4]], dtype=torch.float32)
    r_tot = torch.tensor([[0.8, 1.2], [0.5, 1.5]], dtype=torch.float32)

    conductance_grad = trainer._compute_topk_voltage_homeostasis_grad(
        v_n=v_n,
        x_pre=x_pre,
        r_tot=r_tot,
        layer_dynamics_mode="conductance",
        batch_size=v_n.size(0),
        conductance_reversal=1.1,
        additive_sign=1.0,
    )
    voltage_error = 2.0 * (v_n - 0.25)
    expected_conductance = (voltage_error * r_tot * (1.1 - v_n)).t() @ x_pre
    expected_conductance = expected_conductance / float(v_n.size(0))
    torch.testing.assert_close(conductance_grad, expected_conductance)

    additive_grad = trainer._compute_topk_voltage_homeostasis_grad(
        v_n=v_n,
        x_pre=x_pre,
        r_tot=r_tot,
        layer_dynamics_mode="additive",
        batch_size=v_n.size(0),
        conductance_reversal=0.0,
        additive_sign=-1.0,
    )
    expected_additive = (-voltage_error).t() @ x_pre / float(v_n.size(0))
    torch.testing.assert_close(additive_grad, expected_additive)

    trainer.local_cfg.voltage_homeostasis.enabled = False
    assert (
        trainer._compute_topk_voltage_homeostasis_grad(
            v_n=v_n,
            x_pre=x_pre,
            r_tot=r_tot,
            layer_dynamics_mode="conductance",
            batch_size=v_n.size(0),
            conductance_reversal=1.1,
            additive_sign=1.0,
        )
        is None
    )


@pytest.mark.parametrize("below_target", [True, False])
@pytest.mark.parametrize("reversal", [1.0, 0.0])  # excitatory, inhibitory
def test_topk_voltage_homeostasis_descent_moves_voltages_to_the_target(
    below_target, reversal
) -> None:
    trainer = _make_trainer(dynamics_mode="conductance")
    trainer.local_cfg.voltage_homeostasis.enabled = True
    trainer.local_cfg.voltage_homeostasis.weight = 1.0
    trainer.local_cfg.voltage_homeostasis.target_voltage = 0.5
    v_n = torch.full((4, 3), 0.2 if below_target else 0.8)
    grad = trainer._compute_topk_voltage_homeostasis_grad(
        v_n=v_n,
        x_pre=torch.rand(4, 5) + 0.1,
        r_tot=torch.rand(4, 3) + 0.5,
        layer_dynamics_mode="conductance",
        batch_size=4,
        conductance_reversal=reversal,
        additive_sign=1.0,
    )
    # A descent step changes the conductance by -grad; excitation raises V and
    # inhibition lowers it, so the step must move V toward the target.
    voltage_change_sign = -torch.sign(grad) * (1.0 if reversal > 0.5 else -1.0)
    expected = 1.0 if below_target else -1.0
    assert torch.all(voltage_change_sign == expected)


def test_local_rule_updates_compact_indexed_synapses_without_dense_gradients() -> None:
    trainer = _make_trainer(dynamics_mode="conductance")
    model = _SingleBranchModel(use_shunting=True, topk_type="indexed")
    rec = _capture_record(trainer, model, torch.rand(3, 4))

    exc_grad, inh_grad, dendritic_grad = _run_local_update(
        trainer,
        model,
        rec,
        delta=torch.tensor([[0.2, -0.1], [-0.3, 0.4], [0.1, 0.2]]),
    )

    assert exc_grad.shape == (2, 2)
    assert inh_grad.shape == (2, 2)
    assert dendritic_grad.shape == model.branch.branches_to_output.log_weight.shape
    assert torch.isfinite(exc_grad).all()
    assert torch.isfinite(inh_grad).all()
    assert torch.isfinite(dendritic_grad).all()


def test_local_rule_gradient_dispatch_passes_records_in_order(monkeypatch) -> None:
    """Batch-level local-rule dispatch should preserve per-record call context."""
    trainer = _make_trainer(dynamics_mode="additive")
    records = [
        {"v_n": torch.zeros(2, 2)},
        {"v_n": torch.zeros(2, 3)},
    ]
    v0 = torch.zeros(2, 2)
    delta = torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float32)
    y_target = torch.tensor([0, 1])
    model = object()
    broadcast_state = object()
    captured = []

    def fake_prepare(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["layer_records"] is records
        assert kwargs["delta"] is delta
        torch.testing.assert_close(
            kwargs["delta_scalar"],
            trainer._reduce_error_to_scalar(delta),
        )
        return broadcast_state

    def fake_apply(**kwargs):
        captured.append(kwargs)

    monkeypatch.setattr(trainer, "_prepare_local_rule_broadcast_state", fake_prepare)
    monkeypatch.setattr(trainer, "_apply_local_rule_record_gradients", fake_apply)

    trainer._apply_local_rule_gradients(
        model=model,
        layer_records=records,
        v0=v0,
        delta=delta,
        y_target=y_target,
        update_reactivation=False,
    )

    assert [entry["rec"] for entry in captured] == records
    assert [entry["layer_idx"] for entry in captured] == [0, 1]
    assert [entry["num_layers"] for entry in captured] == [2, 2]
    assert [entry["batch_size"] for entry in captured] == [2, 2]
    assert all(entry["v0"] is v0 for entry in captured)
    assert all(entry["delta"] is delta for entry in captured)
    assert all(entry["broadcast_state"] is broadcast_state for entry in captured)
    assert all(entry["y_target"] is y_target for entry in captured)
    assert all(entry["update_reactivation"] is False for entry in captured)


def test_additive_auto_uses_additive_derivatives() -> None:
    """Auto mode should use additive-consistent derivatives for additive layers."""
    torch.manual_seed(0)
    model = _SingleBranchModel(use_shunting=False)
    trainer_auto = _make_trainer(
        dynamics_mode="auto",
        use_conductance_scaling=True,
        use_driving_force=True,
    )
    trainer_auto_no_shunt_terms = _make_trainer(
        dynamics_mode="auto",
        use_conductance_scaling=False,
        use_driving_force=False,
    )

    x_batch = torch.tensor(
        [
            [0.2, 0.5, 0.9, 0.1],
            [0.7, 0.1, 0.3, 0.6],
            [0.4, 0.8, 0.6, 0.2],
            [0.9, 0.2, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    rec = _capture_record(trainer_auto, model, x_batch)
    delta = torch.tensor(
        [
            [0.1, -0.2],
            [-0.4, 0.3],
            [0.2, 0.1],
            [-0.1, 0.4],
        ],
        dtype=torch.float32,
    )

    exc_grad, inh_grad, den_grad = _run_local_update(trainer_auto, model, rec, delta)
    exc_grad_ref, inh_grad_ref, den_grad_ref = _run_local_update(
        trainer_auto_no_shunt_terms, model, rec, delta
    )

    # Additive auto mode should ignore conductance/driving-force toggles.
    torch.testing.assert_close(exc_grad, exc_grad_ref)
    torch.testing.assert_close(inh_grad, inh_grad_ref)
    torch.testing.assert_close(den_grad, den_grad_ref)

    # Additive formulas:
    # dV/dg_exc = x, dV/dg_inh = -x, dV/dg_den = V_child.
    x_exc = rec["x_exc"]
    x_inh = rec["x_inh"]
    blk = model.branch.branches_to_output
    x_blk = rec["x_blk_raw"].view(
        rec["x_blk_raw"].size(0), blk.out_features, blk.block_size
    )
    expected_exc = delta.t() @ x_exc / float(delta.size(0))
    expected_inh = (-delta).t() @ x_inh / float(delta.size(0))
    expected_den = (delta.unsqueeze(-1) * x_blk).mean(dim=0)

    torch.testing.assert_close(exc_grad, expected_exc)
    torch.testing.assert_close(inh_grad, expected_inh)
    torch.testing.assert_close(den_grad, expected_den)


def test_shunting_auto_matches_conductance_mode() -> None:
    """Auto mode must preserve existing behavior for shunting layers."""
    torch.manual_seed(1)
    model = _SingleBranchModel(use_shunting=True)
    trainer_auto = _make_trainer(dynamics_mode="auto")
    trainer_conductance = _make_trainer(dynamics_mode="conductance")

    x_batch = torch.tensor(
        [
            [0.3, 0.4, 0.7, 0.2],
            [0.6, 0.2, 0.5, 0.1],
            [0.1, 0.8, 0.9, 0.4],
        ],
        dtype=torch.float32,
    )
    rec = _capture_record(trainer_auto, model, x_batch)
    delta = torch.tensor(
        [
            [0.2, -0.1],
            [-0.3, 0.4],
            [0.1, 0.2],
        ],
        dtype=torch.float32,
    )

    grads_auto = _run_local_update(trainer_auto, model, rec, delta)
    grads_conductance = _run_local_update(trainer_conductance, model, rec, delta)

    for grad_auto, grad_conductance in zip(grads_auto, grads_conductance):
        torch.testing.assert_close(grad_auto, grad_conductance)


def test_low_rank_broadcast_keeps_vector_structure() -> None:
    """Low-rank broadcast should produce neuron-specific signals."""
    trainer = _make_trainer(
        dynamics_mode="conductance",
        error_broadcast_mode="low_rank",
        broadcast_rank=2,
    )
    delta = torch.tensor(
        [[0.4, -0.2], [0.1, 0.3], [-0.5, 0.7]],
        dtype=torch.float32,
    )

    e_n = trainer._compute_low_rank_broadcast(
        delta=delta,
        layer_idx=1,
        out_features=6,
        device=delta.device,
        dtype=delta.dtype,
    )

    assert e_n.shape == (3, 6)
    assert torch.std(e_n, dim=1).mean().item() > 1e-4


def test_path_propagation_uses_proximal_conductances() -> None:
    """Per-branch path factors should propagate through proximal layers."""
    trainer = _make_trainer(
        dynamics_mode="conductance",
        use_path_propagation=True,
        path_factor_mode="per_branch",
    )

    distal_v = torch.zeros(2, 4)
    proximal_v = torch.zeros(2, 2)
    soma_v = torch.zeros(2, 1)

    proximal_blk = BlockLinear(4, 2, weight_transform="identity")
    proximal_blk.log_weight.data = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    soma_blk = BlockLinear(2, 1, weight_transform="identity")
    soma_blk.log_weight.data = torch.tensor([[0.3, 0.7]])

    layer_records = [
        {"v_n": distal_v},
        {"v_n": proximal_v, "blk_module": proximal_blk},
        {"v_n": soma_v, "blk_module": soma_blk},
    ]

    path_factors = trainer._precompute_path_propagation_factors(layer_records)

    expected_soma = torch.ones_like(soma_v)
    expected_proximal = torch.tensor([[0.15, 0.35], [0.15, 0.35]])
    expected_distal = torch.tensor(
        [[0.01875, 0.0375, 0.0875, 0.11666667]] * 2,
        dtype=torch.float32,
    )

    torch.testing.assert_close(path_factors[2], expected_soma)
    torch.testing.assert_close(path_factors[1], expected_proximal, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(path_factors[0], expected_distal, atol=1e-6, rtol=1e-6)


def test_path_transport_preserves_soma_vector_structure() -> None:
    """Path transport should recursively expand vector soma errors through the tree."""
    trainer = _make_trainer(
        dynamics_mode="conductance",
        error_broadcast_mode="path_transport",
    )

    distal_v = torch.zeros(2, 8)
    proximal_v = torch.zeros(2, 4)
    soma_v = torch.zeros(2, 2)

    proximal_blk = BlockLinear(8, 4, weight_transform="identity")
    proximal_blk.log_weight.data = torch.tensor(
        [[0.2, 0.4], [0.6, 0.8], [1.0, 1.2], [1.4, 1.6]]
    )
    soma_blk = BlockLinear(4, 2, weight_transform="identity")
    soma_blk.log_weight.data = torch.tensor([[0.3, 0.7], [0.5, 0.9]])

    layer_records = [
        {"v_n": distal_v},
        {"v_n": proximal_v, "blk_module": proximal_blk},
        {"v_n": soma_v, "blk_module": soma_blk},
    ]
    delta = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    delta_scalar = trainer._reduce_error_to_scalar(delta)

    transported = trainer._precompute_path_transport_errors(
        layer_records=layer_records,
        delta=delta,
        delta_scalar=delta_scalar,
    )

    expected_soma = delta
    expected_proximal = torch.tensor(
        [[0.15, 0.35, 0.41666666, 0.75], [0.45, 1.05, 0.8333333, 1.5]],
        dtype=torch.float32,
    )
    expected_distal = torch.tensor(
        [
            [0.01875, 0.0375, 0.0875, 0.11666667, 0.13020833, 0.15625, 0.2625, 0.3],
            [0.05625, 0.1125, 0.2625, 0.35, 0.26041666, 0.3125, 0.525, 0.6],
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(transported[2], expected_soma)
    torch.testing.assert_close(transported[1], expected_proximal, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(transported[0], expected_distal, atol=1e-6, rtol=1e-6)


def test_inhibitory_homeostasis_pushes_toward_target_rtot() -> None:
    """Homeostatic inhibitory updates should increase inhibition when R_tot is high."""
    torch.manual_seed(7)
    model = _SingleBranchModel(use_shunting=True)
    trainer = _make_trainer(
        dynamics_mode="conductance",
        inhibitory_homeostasis={
            "enabled": True,
            "mode": "r_tot",
            "weight": 0.5,
            "target_r_tot": 0.0,
        },
    )

    x_batch = torch.tensor(
        [
            [0.2, 0.5, 0.9, 0.1],
            [0.7, 0.1, 0.3, 0.6],
            [0.4, 0.8, 0.6, 0.2],
            [0.9, 0.2, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    rec = _capture_record(trainer, model, x_batch)
    R_tot = 1.0 / (trainer._compute_layer_total_conductance(rec, rec["v_n"]) + 1e-8)
    trainer.local_cfg.inhibitory_homeostasis.target_r_tot = (
        float(R_tot.mean().item()) * 0.5
    )

    delta = torch.zeros((x_batch.size(0), 2), dtype=torch.float32)
    exc_grad, inh_grad, den_grad = _run_local_update(trainer, model, rec, delta)

    torch.testing.assert_close(exc_grad, torch.zeros_like(exc_grad))
    torch.testing.assert_close(den_grad, torch.zeros_like(den_grad))
    assert inh_grad.mean().item() < -1e-6


def test_additive_local_rule_uses_activation_derivative() -> None:
    """Voltage-space local updates should include the branch activation slope."""
    torch.manual_seed(11)
    model = _SingleBranchModel(use_shunting=False)
    model.branch.reactivation = nn.Tanh()
    trainer = _make_trainer(
        dynamics_mode="auto",
        use_conductance_scaling=False,
        use_driving_force=False,
    )

    x_batch = torch.tensor(
        [
            [0.2, 0.5, 0.9, 0.1],
            [0.7, 0.1, 0.3, 0.6],
            [0.4, 0.8, 0.6, 0.2],
            [0.9, 0.2, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    rec = _capture_record(trainer, model, x_batch)
    delta = torch.tensor(
        [
            [0.1, -0.2],
            [-0.4, 0.3],
            [0.2, 0.1],
            [-0.1, 0.4],
        ],
        dtype=torch.float32,
    )

    exc_grad, inh_grad, den_grad = _run_local_update(trainer, model, rec, delta)

    v_n = rec["v_n"]
    act_deriv = 1.0 - torch.tanh(v_n).square()
    effective_error = delta * act_deriv
    x_exc = rec["x_exc"]
    x_inh = rec["x_inh"]
    blk = model.branch.branches_to_output
    x_blk = rec["x_blk_raw"].view(
        rec["x_blk_raw"].size(0), blk.out_features, blk.block_size
    )

    expected_exc = effective_error.t() @ x_exc / float(delta.size(0))
    expected_inh = (-effective_error).t() @ x_inh / float(delta.size(0))
    expected_den = (effective_error.unsqueeze(-1) * x_blk).mean(dim=0)

    torch.testing.assert_close(exc_grad, expected_exc, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(inh_grad, expected_inh, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(den_grad, expected_den, atol=1e-6, rtol=1e-6)


def test_path_transport_includes_parent_activation_derivative() -> None:
    """Recursive transport should include the proximal activation slope."""
    trainer = _make_trainer(
        dynamics_mode="conductance",
        error_broadcast_mode="path_transport",
    )

    child_v = torch.zeros(1, 4)
    parent_v = torch.tensor([[0.5, -0.4]], dtype=torch.float32)
    parent_layer = type(
        "Layer", (), {"reactivation": nn.Tanh(), "effective_leak": 1.0}
    )()
    blk = _DummyBlock(
        weight=torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32),
        block_size=2,
    )

    layer_records = [
        {"v_n": child_v},
        {
            "v_n": parent_v,
            "v_out": torch.tanh(parent_v),
            "blk_module": blk,
            "layer": parent_layer,
        },
    ]
    delta = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    delta_scalar = trainer._reduce_error_to_scalar(delta)

    transported = trainer._precompute_path_transport_errors(
        layer_records=layer_records,
        delta=delta,
        delta_scalar=delta_scalar,
    )

    parent_act = 1.0 - torch.tanh(parent_v).square()
    expected_child = torch.tensor(
        [
            [
                1.0 * parent_act[0, 0] * 0.2,
                1.0 * parent_act[0, 0] * 0.4,
                2.0 * parent_act[0, 1] * 0.6,
                2.0 * parent_act[0, 1] * 0.8,
            ]
        ],
        dtype=torch.float32,
    )

    torch.testing.assert_close(transported[1], delta, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(transported[0], expected_child, atol=1e-6, rtol=1e-6)
