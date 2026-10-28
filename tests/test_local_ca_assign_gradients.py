"""``LocalCreditAssignment.assign_gradients``: a training step's update, no step."""

from __future__ import annotations

import pytest
import torch
from test_local_learning_credit_controls import (
    _core_state,
    _same,
    _trainability,
    _tree_model,
    _variance_tree_model,
)
from test_trainer_configuration import _data, _einet_model, _make_trainer, _mlp_model


def _gradients(model):
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


@pytest.mark.parametrize(
    "local_rule_config",
    [
        {},
        {"error_broadcast_mode": "path_transport"},
        {
            "hsic": {
                "enabled": True,
                "weight": 0.2,
                "kernel": "linear",
                "warmup_epochs": 4,
            }
        },
    ],
)
def test_the_gradients_are_those_of_the_first_training_step(local_rule_config):
    batch = _data(n=16)
    stepped = _einet_model()
    trainer, optimizer = _make_trainer(
        "local_ca",
        stepped,
        epochs=1,
        batch_size=16,
        local_rule_config=local_rule_config,
    )
    applied = []
    optimizer.step = lambda: applied.append(_gradients(stepped))
    trainer.train(stepped, batch, _data(n=8, seed=1))

    model = _einet_model()
    fresh, _ = _make_trainer(
        "local_ca",
        model,
        epochs=1,
        batch_size=16,
        local_rule_config=local_rule_config,
    )
    model.train()
    loss = fresh.assign_gradients(model, *batch.tensors)
    assigned = _gradients(model)

    assert len(applied) == 1 and assigned
    assert applied[0].keys() == assigned.keys()
    for name, gradient in applied[0].items():
        assert torch.equal(assigned[name], gradient), name
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_no_parameter_or_optimizer_state_changes():
    model = _einet_model()
    trainer, optimizer = _make_trainer("local_ca", model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    trainer.assign_gradients(model, *_data(n=16).tensors)
    assert _gradients(model)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name
    assert len(optimizer.optimizer.state) == 0


def test_a_trainer_is_prepared_on_first_use_without_a_best_state():
    model = _einet_model()
    trainer, _ = _make_trainer("local_ca", model)
    assert not hasattr(trainer, "epoch_counter")
    trainer.assign_gradients(model, *_data(n=16).tensors)
    assert trainer.epoch_counter == 0
    assert not hasattr(trainer, "best_state_dict")


def test_a_model_without_branch_layers_is_rejected():
    model = _mlp_model()
    trainer, _ = _make_trainer("local_ca", model)
    with pytest.raises(ValueError, match="this model has none"):
        trainer.assign_gradients(model, *_data(n=16).tensors)


def test_epoch_selects_that_epochs_update_modes():
    model = _einet_model()
    trainer, _ = _make_trainer(
        "local_ca", model, local_rule_config={"freeze_decoder_epochs": 1}
    )
    x, y = _data(n=16).tensors
    trainer.assign_gradients(model, x, y)  # the first epoch: decoder frozen
    assert not any(name.startswith("decoder_network.") for name in _gradients(model))
    trainer.assign_gradients(model, x, y, epoch=2)
    assert any(name.startswith("decoder_network.") for name in _gradients(model))
    with pytest.raises(ValueError, match="counts from 1"):
        trainer.assign_gradients(model, x, y, epoch=0)


@pytest.mark.parametrize("build", [_tree_model, _variance_tree_model])
def test_core_update_mode_none_holds_the_core_as_training_does(build):
    # A training forward would update the variance Top-K statistics; training
    # with the core held fixed does not, and neither may assign_gradients.
    model = build()
    before = _core_state(model)
    trainability = _trainability(model)
    trainer, _ = _make_trainer(
        "local_ca", model, local_rule_config={"core_update_mode": "none"}
    )
    model.train()
    trainer.assign_gradients(model, *_data(n=16).tensors)
    assert _same(before, _core_state(model))
    assert _trainability(model) == trainability
    assigned = _gradients(model)
    assert assigned and not any(name.startswith("core_network.") for name in assigned)


def test_core_update_mode_none_gradients_are_those_of_the_first_training_step():
    config = {"core_update_mode": "none"}
    batch = _data(n=16)
    stepped = _variance_tree_model()
    trainer, optimizer = _make_trainer(
        "local_ca", stepped, epochs=1, batch_size=16, local_rule_config=config
    )
    applied = []
    optimizer.step = lambda: applied.append(_gradients(stepped))
    trainer.train(stepped, batch, _data(n=8, seed=1))

    model = _variance_tree_model()
    fresh, _ = _make_trainer(
        "local_ca", model, epochs=1, batch_size=16, local_rule_config=config
    )
    model.train()
    fresh.assign_gradients(model, *batch.tensors)
    assigned = _gradients(model)

    assert len(applied) == 1 and assigned
    assert applied[0].keys() == assigned.keys()
    for name, gradient in applied[0].items():
        assert torch.equal(assigned[name], gradient), name


@pytest.mark.parametrize("failure", [None, "initialization", "batch"])
def test_frozen_core_restores_pre_initialization_gate_flags(monkeypatch, failure):
    model = _einet_model()
    before = _trainability(model)
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        local_rule_config={"core_update_mode": "none"},
        reactivation_update_mode="frozen",
        recalibrate_reactivation_every=0,
    )
    initialize = trainer._initialize_attributes

    def initialize_then_fail(*args, **kwargs):
        initialize(*args, **kwargs)
        assert _trainability(model) != before
        raise RuntimeError("initialization failed")

    def fail_batch(*args, **kwargs):
        assert trainer.epoch_counter == 2
        raise RuntimeError("batch failed")

    if failure == "initialization":
        monkeypatch.setattr(trainer, "_initialize_attributes", initialize_then_fail)
    elif failure == "batch":
        monkeypatch.setattr(trainer, "_assign_batch_gradients", fail_batch)
    if failure is None:
        trainer.assign_gradients(model, *_data(n=16).tensors, epoch=2)
        assert trainer.epoch_counter == 0
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            trainer.assign_gradients(model, *_data(n=16).tensors, epoch=2)
    assert _trainability(model) == before
    # The trainer must also retain the original flags, rather than cache the
    # temporary frozen core when initialization prepares the gate policy.
    assert all(
        flag == before[name]
        for name, param in model.named_parameters()
        for gate, flag in trainer._reactivation_param_states
        if gate is param
    )


def test_incompatible_quantile_policy_restores_pre_initialization_gate_flags():
    model = _einet_model()
    before = _trainability(model)
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        local_rule_config={"core_update_mode": "none"},
        reactivation_update_mode="quantile",
        recalibrate_reactivation_every=1,
    )
    with pytest.raises(ValueError, match="recalibrate_reactivation_every=1"):
        trainer.assign_gradients(model, *_data(n=16).tensors)
    assert _trainability(model) == before


def test_explicit_epoch_scales_hsic_and_restores_the_counter():
    gradients = {}
    for label, enabled, epoch in (
        ("off", False, 1),
        ("half", True, 2),
        ("full", True, 4),
    ):
        model = _einet_model()
        trainer, _ = _make_trainer(
            "local_ca",
            model,
            local_rule_config={
                "hsic": {
                    "enabled": enabled,
                    "weight": 0.2,
                    "kernel": "linear",
                    "warmup_epochs": 4,
                }
            },
        )
        trainer._initialize_attributes(model, trainer.epochs, snapshot_best=False)
        trainer.epoch_counter = 9
        trainer.assign_gradients(model, *_data(n=16).tensors, epoch=epoch)
        assert trainer.epoch_counter == 9
        gradients[label] = _gradients(model)
    assert (
        gradients["off"].keys() == gradients["half"].keys() == gradients["full"].keys()
    )
    assert any(
        not torch.equal(gradients["off"][k], gradients["full"][k])
        for k in gradients["off"]
    )
    for name, base in gradients["off"].items():
        torch.testing.assert_close(
            gradients["half"][name], base + 0.5 * (gradients["full"][name] - base)
        )


def test_failed_batch_restores_an_existing_epoch_counter(monkeypatch):
    model = _einet_model()
    trainer, _ = _make_trainer("local_ca", model)
    trainer.assign_gradients(model, *_data(n=16).tensors)
    trainer.epoch_counter = 7

    def fail(*args, **kwargs):
        assert trainer.epoch_counter == 3
        raise RuntimeError("batch failed")

    monkeypatch.setattr(trainer, "_assign_batch_gradients", fail)
    with pytest.raises(RuntimeError, match="batch failed"):
        trainer.assign_gradients(model, *_data(n=16).tensors, epoch=3)
    assert trainer.epoch_counter == 7
