"""Staged trainers keep fixed parameters fixed and restore trainability exactly.

Options that change which parameters train after DistributedDataParallel has
wrapped the model raise before the first step instead of failing in its
reducer.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from test_operator_config_contract import _config
from test_trainer_configuration import (
    IN_DIM,
    _data,
    _einet_model,
    _make_trainer,
    _optimizer,
    _trainer_config,
    _wrap,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from dendritic_modeling.networks import ParametricActivation
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.training.strategies import TrainOnlyMReactivation
from dendritic_modeling.training.strategies.trainability import (
    restore_requires_grad,
    snapshot_requires_grad,
    train_only,
)


def _model_with_fixed_parameters(seed=13):
    """EINet whose linear_sigmoid gate slopes are fixed, with a frozen readout bias."""
    torch.manual_seed(seed)
    config = deepcopy(_config("einet", "shunting", "linear_sigmoid"))
    model = _wrap(get_architecture("einet", config, input_dim=IN_DIM))
    model.decoder_network.bias.requires_grad_(False)
    return model


def _flags(model):
    return {name: param.requires_grad for name, param in model.named_parameters()}


def _fixed_values(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not param.requires_grad
    }


def _assert_unchanged(model, values):
    parameters = dict(model.named_parameters())
    for name, value in values.items():
        assert torch.equal(parameters[name], value), name


def _gate_parameters(model, name):
    return [
        getattr(module, name)
        for module in model.modules()
        if isinstance(module, ParametricActivation)
    ]


def test_train_only_keeps_fixed_parameters_fixed_and_restores_exactly():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model[1].bias.requires_grad_(False)
    head = nn.Linear(1, 1)
    snapshot = snapshot_requires_grad(model)

    train_only(model, [model[1], head], snapshot)

    assert [p.requires_grad for p in model.parameters()] == [False, False, True, False]
    assert all(p.requires_grad for p in head.parameters())
    restore_requires_grad(snapshot)
    assert [p.requires_grad for p in model.parameters()] == [True, True, True, False]


def test_the_test_model_has_fixed_gate_slopes():
    fixed = _fixed_values(_model_with_fixed_parameters())
    assert sum(name.endswith("log_m") for name in fixed) >= 2
    assert "decoder_network.bias" in fixed


def test_freeze_layers_keeps_fixed_parameters_and_restores_flags():
    model = _model_with_fixed_parameters()
    flags, fixed = _flags(model), _fixed_values(model)
    trainer, _ = _make_trainer(
        "freeze_layers",
        model,
        pretrain_epochs=1,
        epochs_per_layer=1,
        final_tune_epochs=1,
    )

    trainer.train(model, _data(), _data(seed=1))

    assert _flags(model) == flags
    _assert_unchanged(model, fixed)


def test_freeze_layers_restores_flags_when_a_layer_phase_fails():
    model = _model_with_fixed_parameters()
    flags = _flags(model)
    trainer, _ = _make_trainer(
        "freeze_layers", model, pretrain_epochs=0, epochs_per_layer=1
    )
    during = []

    def failing_loop(**_kwargs):
        during.append(_flags(model))
        raise RuntimeError("layer phase failed")

    trainer._run_training_loop = failing_loop
    with pytest.raises(RuntimeError, match="layer phase failed"):
        trainer.train(model, _data(), _data(seed=1))

    assert during and during[0] != flags  # the failure came inside the phase
    assert _flags(model) == flags


def test_freeze_layers_keeps_frozen_reactivation_mode_through_layer_phases():
    model = _einet_model()
    gates = [
        (param, param.detach().clone())
        for name in ("log_m", "b")
        for param in _gate_parameters(model, name)
    ]
    trainer, _ = _make_trainer(
        "freeze_layers",
        model,
        pretrain_epochs=1,
        epochs_per_layer=1,
        final_tune_epochs=1,
        reactivation_update_mode="frozen",
    )

    trainer.train(model, _data(), _data(seed=1))

    assert all(torch.equal(param, before) for param, before in gates)


@pytest.mark.parametrize("strategy", ["freeze_branches", "freeze_branch_kl"])
def test_two_step_keeps_fixed_parameters_and_restores_flags(strategy):
    model = _model_with_fixed_parameters()
    flags, fixed = _flags(model), _fixed_values(model)
    trainer, _ = _make_trainer(strategy, model, epochs=1, pretrain_epochs=1)

    trainer.train(model, _data(), _data(seed=1))

    assert _flags(model) == flags
    _assert_unchanged(model, fixed)
    fixed_ids = {id(p) for p in model.parameters() if not p.requires_grad}
    local_ids = {
        id(param)
        for local in trainer.local_optimizers
        for group in getattr(local, "optimizer", local).param_groups
        for param in group["params"]
    }
    assert not fixed_ids & local_ids


def test_two_step_restores_flags_when_stage_one_fails(monkeypatch):
    from dendritic_modeling.training.strategies import two_step

    model = _model_with_fixed_parameters()
    flags = _flags(model)
    trainer, _ = _make_trainer("freeze_branches", model, epochs=1, pretrain_epochs=1)
    during = []

    def failing_features(*_args, **_kwargs):
        during.append(_flags(model))
        raise RuntimeError("stage 1 failed")

    trainer.local_heads = [torch.nn.Linear(3, 3)]  # sized without a forward
    monkeypatch.setattr(two_step, "_local_features", failing_features)
    with pytest.raises(RuntimeError, match="stage 1 failed"):
        trainer.train(model, _data(), _data(seed=1))

    assert during and during[0] != flags
    assert _flags(model) == flags


def _train_only_m_trainer(model):
    trainer = TrainOnlyMReactivation(_optimizer(model), **_trainer_config(epochs=1))
    trainer.device = "cpu"
    return trainer


def test_train_only_m_reactivation_holds_gate_midpoints_fixed():
    model = _einet_model()
    flags = _flags(model)
    midpoints = [(p, p.detach().clone()) for p in _gate_parameters(model, "b")]
    slopes = [(p, p.detach().clone()) for p in _gate_parameters(model, "log_m")]

    _train_only_m_trainer(model).train(model, _data(), _data(seed=1))

    assert all(torch.equal(param, before) for param, before in midpoints)
    assert any(not torch.equal(param, before) for param, before in slopes)
    assert _flags(model) == flags


def test_train_only_m_reactivation_rejects_models_without_trainable_midpoints():
    model = _model_with_fixed_parameters()  # linear_sigmoid gates: fixed b
    with pytest.raises(ValueError, match="no reactivation gate with a trainable"):
        _train_only_m_trainer(model).train(model, _data(), _data(seed=1))


# ----------------------------------------------------------------------------
# DistributedDataParallel: a one-process gloo group is enough for the checks
# ----------------------------------------------------------------------------


@pytest.fixture
def process_group(tmp_path):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.usefixtures("process_group")
@pytest.mark.parametrize("mode", ["quantile", "frozen"])
def test_ddp_rejects_reactivation_modes_that_freeze_registered_parameters(mode):
    model = _einet_model()
    trainer, _ = _make_trainer(
        "standard",
        model,
        reactivation_update_mode=mode,
        recalibrate_reactivation_every=int(mode == "quantile"),
    )
    with pytest.raises(NotImplementedError, match="freezes the reactivation"):
        trainer.train(DistributedDataParallel(model), _data(), _data(seed=1))


@pytest.mark.usefixtures("process_group")
@pytest.mark.parametrize(
    "strategy, mode", [("standard", "backprop"), ("fa", "quantile")]
)
def test_ddp_accepts_trainers_whose_trainable_set_stays_registered(strategy, mode):
    model = _einet_model()
    trainer, _ = _make_trainer(
        strategy,
        model,
        reactivation_update_mode=mode,
        recalibrate_reactivation_every=int(mode == "quantile"),
    )
    trainer._initialize_attributes(DistributedDataParallel(model), 1)


@pytest.mark.usefixtures("process_group")
@pytest.mark.parametrize(
    "strategy, options",
    [
        ("freeze_layers", {}),
        (
            "multi_stage",
            {"multi_stage_config": {"stages": [{"learning_strategy": "standard"}]}},
        ),
    ],
)
def test_staged_trainers_reject_distributed_training(strategy, options):
    model = _einet_model()
    trainer, _ = _make_trainer(strategy, model, **options)
    with pytest.raises(NotImplementedError, match="does not support distributed"):
        trainer.train(DistributedDataParallel(model), _data(), _data(seed=1))


@pytest.mark.usefixtures("process_group")
@pytest.mark.parametrize(
    "strategy", ["voltage_stabilization", "homeostatic_control", "ei_equilibrium"]
)
def test_ddp_rejects_warmups_whose_loss_skips_parameters(strategy):
    model = _einet_model()
    trainer, _ = _make_trainer(strategy, model, pretrain_epochs=1)
    with pytest.raises(NotImplementedError, match="reaches only some parameters"):
        trainer.train(DistributedDataParallel(model), _data(), _data(seed=1))


@pytest.mark.usefixtures("process_group")
def test_ddp_rejects_train_only_m_reactivation():
    model = _einet_model()
    with pytest.raises(NotImplementedError, match="freezes the reactivation"):
        _train_only_m_trainer(model).train(
            DistributedDataParallel(model), _data(), _data(seed=1)
        )
