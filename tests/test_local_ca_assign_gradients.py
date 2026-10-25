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
    "local_rule_config", [{}, {"error_broadcast_mode": "path_transport"}]
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
