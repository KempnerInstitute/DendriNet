"""Tests for additive-aware LocalCA dynamics modes."""

from __future__ import annotations

import torch
import torch.nn as nn

from dendritic_modeling.networks import DendriticBranchLayer
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
    use_conductance_scaling: bool = True,
    use_driving_force: bool = True,
) -> LocalCreditAssignment:
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer.local_cfg = trainer._build_local_rule_config(
        {
            "rule_variant": "3f",
            "error_broadcast_mode": "per_soma",
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
        }
    )
    trainer._layer_stats = {}
    trainer._decoder_cache = {}
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
