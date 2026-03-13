"""Tests for additive-aware LocalCA dynamics modes."""

from __future__ import annotations

import torch
import torch.nn as nn

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


class _SingleBranchModel(nn.Module):
    """Small model exposing a single dendritic branch layer."""

    def __init__(self, *, use_shunting: bool):
        super().__init__()
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
            weight_transform="identity",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x, inhibitory_input=x, branch_input=x)


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
