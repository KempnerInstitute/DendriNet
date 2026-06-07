"""Tests for trace-based STDP local learning."""

import torch
import torch.nn as nn

from dendritic_modeling.config.training import LocalRuleConfig, STDPConfig
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


class _RecurrentBranchModel(nn.Module):
    """Minimal branch model exposing feedforward and recurrent TopK pathways."""

    def __init__(self) -> None:
        super().__init__()
        self.branch = DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=3,
            excitatory_synapses_per_branch=2,
            recurrent_input_dim=2,
            recurrent_synapses_per_branch=1,
            rec_inhibitory_input_dim=2,
            rec_inhibitory_synapses_per_branch=1,
            use_shunting=False,
            reactivate=False,
            topk_noise_level=0.0,
            weight_transform="identity",
        )

    def forward(
        self,
        x: torch.Tensor,
        rec_exc: torch.Tensor,
        rec_inh: torch.Tensor,
    ) -> torch.Tensor:
        return self.branch(
            x,
            recurrent_input=rec_exc,
            rec_inhibitory_input=rec_inh,
        )


def _make_trainer(local_rule_config: dict) -> LocalCreditAssignment:
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer.local_cfg = trainer._build_local_rule_config(local_rule_config)
    trainer._layer_stats = {}
    trainer._decoder_cache = {}
    trainer._broadcast_cache = {}
    trainer._stdp_traces = {}
    return trainer


def test_stdp_rule_variant_enables_trace_rule() -> None:
    cfg = LocalRuleConfig(rule_variant="stdp")

    assert cfg.rule_variant == "stdp"
    assert cfg.stdp.enabled


def test_stdp_config_loads_from_nested_local_rule_dict() -> None:
    trainer = _make_trainer(
        {
            "rule_variant": "stdp",
            "stdp": {
                "apply_to": ["exc", "rec_inh"],
                "activity_mode": "binary",
                "tau_pre": 5.0,
            },
        }
    )

    assert trainer.local_cfg.stdp.enabled
    assert trainer.local_cfg.stdp.apply_to == ["exc", "rec_inh"]
    assert trainer.local_cfg.stdp.activity_mode == "binary"
    assert trainer.local_cfg.stdp.tau_pre == 5.0


def test_stdp_config_accepts_single_apply_to_string() -> None:
    cfg = STDPConfig(apply_to="all")

    assert cfg.apply_to == ["all"]


def test_stdp_trace_update_potentiates_prior_pre_activity() -> None:
    layer = TopKLinear(
        in_features=3,
        out_features=2,
        K=3,
        weight_transform="identity",
    )
    layer.pre_w.data.fill_(1.0)
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer.local_cfg = LocalRuleConfig(
        rule_variant="stdp",
        stdp=STDPConfig(
            enabled=True,
            apply_to=["exc"],
            activity_mode="identity",
            tau_pre=1.0,
            tau_post=1.0,
            a_plus=1.0,
            a_minus=0.0,
            learning_rate_scale=0.25,
        ),
        update_inactive_weights=False,
    )
    trainer._stdp_traces = {}

    first_pre = torch.tensor([[2.0, 0.0, 0.0]])
    first_post = torch.zeros(1, 2)
    trainer._apply_stdp_gradient(
        layer=layer,
        pre=first_pre,
        post=first_post,
        mask=torch.ones_like(layer.pre_w),
        pathway="exc",
    )

    layer.pre_w.grad = None
    second_pre = torch.zeros(1, 3)
    second_post = torch.tensor([[1.0, 0.0]])
    trainer._apply_stdp_gradient(
        layer=layer,
        pre=second_pre,
        post=second_post,
        mask=torch.ones_like(layer.pre_w),
        pathway="exc",
    )

    assert layer.pre_w.grad is not None
    assert layer.pre_w.grad[0, 0] < 0.0
    assert torch.count_nonzero(layer.pre_w.grad[1]).item() == 0


def test_local_recorders_and_3f_rule_update_recurrent_topk_pathways() -> None:
    torch.manual_seed(12)
    model = _RecurrentBranchModel()
    for module in model.modules():
        if isinstance(module, TopKLinear):
            module.pre_w.data.fill_(0.5)

    trainer = _make_trainer(
        {
            "rule_variant": "3f",
            "error_broadcast_mode": "per_soma",
            "normalize_by_batch": True,
            "update_inactive_weights": True,
            "three_factor": {
                "dynamics_mode": "additive",
                "use_conductance_scaling": False,
                "use_driving_force": False,
            },
        }
    )
    x = torch.tensor(
        [[0.2, 0.6, 0.1], [0.4, 0.3, 0.7], [0.8, 0.5, 0.2]],
        dtype=torch.float32,
    )
    rec_exc = torch.tensor([[0.5, 0.1], [0.2, 0.9], [0.7, 0.4]])
    rec_inh = torch.tensor([[0.3, 0.6], [0.8, 0.1], [0.2, 0.5]])

    records, handles = trainer._attach_local_recorders(model)
    try:
        _ = model(x, rec_exc, rec_inh)
    finally:
        for handle in handles:
            handle.remove()

    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec.get("rec_exc_module"), TopKLinear)
    assert isinstance(rec.get("rec_inh_module"), TopKLinear)
    assert torch.equal(rec["x_rec_exc"], rec_exc)
    assert torch.equal(rec["x_rec_inh"], rec_inh)

    trainer._apply_local_rule_gradients(
        model=model,
        layer_records=records,
        v0=torch.zeros(3, 2),
        delta=torch.tensor([[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5]]),
        y_target=None,
        update_reactivation=False,
    )

    rec_exc_grad = model.branch.branch_recurrent.pre_w.grad
    rec_inh_grad = model.branch.branch_rec_inhibition.pre_w.grad
    assert rec_exc_grad is not None
    assert rec_inh_grad is not None
    assert torch.count_nonzero(rec_exc_grad).item() > 0
    assert torch.count_nonzero(rec_inh_grad).item() > 0
