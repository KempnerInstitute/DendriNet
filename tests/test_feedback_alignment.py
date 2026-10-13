"""FA and DFA follow their published rules with exact local derivatives."""

from __future__ import annotations

import logging
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from test_operator_config_contract import _config
from test_trainer_configuration import (
    IN_DIM,
    N_CLASSES,
    _einet_model,
    _make_trainer,
    _mlp_model,
    _recurrent_model,
)

from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.training import factory
from dendritic_modeling.training.strategies.feedback_alignment import (
    FeedbackAlignmentTrainer,
)


def _batch(n=16, seed=0, *, loss="ce"):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n, IN_DIM, generator=generator)
    labels = torch.randint(0, N_CLASSES, (n,), generator=generator)
    if loss == "mse":
        return x, nn.functional.one_hot(labels, N_CLASSES).float()
    return x, labels


def _gradients(model):
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _rule_gradients(trainer, model, x, y):
    model.zero_grad(set_to_none=True)
    trainer.compute_gradients(model, x, y)
    return _gradients(model)


def _backprop_gradients(trainer, model, x, y):
    model.zero_grad(set_to_none=True)
    trainer.loss_function.from_predictions(model(x), y).backward()
    return _gradients(model)


def _identity_decoder_einet(seed=13):
    """An EINet whose somas are the class scores."""
    config = deepcopy(_config("einet", "shunting", "param_tanh"))
    config["architecture"]["excitatory_layer_sizes"] = [N_CLASSES]
    torch.manual_seed(seed)
    core = get_architecture("einet", config, input_dim=IN_DIM)
    return Classifier(
        Identity(IN_DIM), core, Identity(N_CLASSES), output_scale_mode="fixed"
    )


@pytest.mark.parametrize("mode", ["fa", "dfa"])
@pytest.mark.parametrize("loss", ["ce", "mse"])
def test_output_layer_receives_the_exact_loss_gradient(mode, loss):
    model = _mlp_model()
    trainer, _ = _make_trainer(mode, model, loss_function=loss)
    x, y = _batch(loss=loss)

    rule = _rule_gradients(trainer, model, x, y)
    backprop = _backprop_gradients(trainer, model, x, y)

    for name in ("decoder_network.weight", "decoder_network.bias"):
        torch.testing.assert_close(rule[name], backprop[name])


def test_fa_with_transposed_weights_is_backpropagation():
    model = _mlp_model()
    trainer, _ = _make_trainer("fa", model)
    x, y = _batch()
    _rule_gradients(trainer, model, x, y)  # builds the plan and the matrices
    modules = dict(model.named_modules())
    assert set(trainer.feedback) == {"decoder_network", "core_network.layers.2"}
    for name in trainer.feedback:
        trainer.feedback[name] = modules[name].weight.detach().clone()

    rule = _rule_gradients(trainer, model, x, y)
    backprop = _backprop_gradients(trainer, model, x, y)

    assert rule.keys() == backprop.keys()
    for name in backprop:
        torch.testing.assert_close(rule[name], backprop[name])


def test_dfa_hidden_layer_uses_its_projection_and_the_unit_derivative():
    model = _mlp_model()
    trainer, _ = _make_trainer("dfa", model)
    x, y = _batch()
    rule = _rule_gradients(trainer, model, x, y)

    first = model.core_network.layers[0]
    logits = model(x)
    error = torch.autograd.grad(
        trainer.loss_function.from_predictions(logits, y), logits
    )[0]
    feedback = trainer.feedback["core_network.layers.2:input"]
    pre_activation = first(x).detach()
    delta = (error @ feedback) * (pre_activation > 0)
    torch.testing.assert_close(rule["core_network.layers.0.weight"], delta.t() @ x)
    torch.testing.assert_close(rule["core_network.layers.0.bias"], delta.sum(0))
    assert (delta == 0).any()  # inactive units received no update


def test_dfa_gives_inactive_top_k_candidates_no_gradient():
    model = _einet_model()
    trainer, _ = _make_trainer("dfa", model)
    x, y = _batch()

    rule = _rule_gradients(trainer, model, x, y)
    backprop = _backprop_gradients(trainer, model, x, y)

    synapse_banks = [name for name in rule if name.endswith("pre_w")]
    assert synapse_banks
    for name in synapse_banks:
        inactive = backprop[name] == 0
        assert inactive.any(), name
        assert torch.all(rule[name][inactive] == 0), name


def test_dfa_soma_of_an_identity_decoder_network_gets_the_exact_error():
    model = _identity_decoder_einet()
    trainer, _ = _make_trainer("dfa", model)
    x, y = _batch()

    rule = _rule_gradients(trainer, model, x, y)
    backprop = _backprop_gradients(trainer, model, x, y)

    soma = [name for name in backprop if ".branch_layers.2." in name]
    assert soma
    for name in soma:
        torch.testing.assert_close(rule[name], backprop[name])
    distal = [name for name in backprop if ".branch_layers.0." in name]
    assert any(not torch.allclose(rule[name], backprop[name]) for name in distal)


def test_feedback_matrices_follow_the_trainer_seed_not_the_global_rng():
    def feedback(seed, global_seed):
        model = _mlp_model()
        trainer, _ = _make_trainer("dfa", model, seed=seed)
        torch.manual_seed(global_seed)
        _rule_gradients(trainer, model, *_batch())
        return trainer.feedback

    first, again, other = feedback(3, 0), feedback(3, 1), feedback(4, 0)
    assert first.keys() == again.keys() == other.keys()
    assert all(torch.equal(first[key], again[key]) for key in first)
    assert not any(torch.equal(first[key], other[key]) for key in first)


def test_shunting_strategy_names_run_fa_and_dfa(caplog, monkeypatch):
    monkeypatch.setattr(factory, "_WARNED_STRATEGY_ALIASES", set())
    model = _einet_model()
    with caplog.at_level(logging.WARNING, logger="dendritic_modeling.training"):
        trainer, _ = _make_trainer("shunting_dfa", model)
    assert type(trainer) is FeedbackAlignmentTrainer
    assert trainer.mode == "dfa"
    assert "Strategy 'shunting_dfa' now runs as 'dfa'" in caplog.text
    with pytest.raises(ValueError, match="e_rev_exc"):
        _make_trainer("shunting_dfa", model, e_rev_exc=1.0)


def test_recurrent_cells_are_rejected():
    model = _recurrent_model()
    trainer, _ = _make_trainer("dfa", model)
    x = torch.rand(4, 3, IN_DIM)
    with pytest.raises(ValueError, match="use the recurrent or standard strategy"):
        trainer.compute_gradients(model, x, torch.zeros(4, dtype=torch.long))


def test_parameters_several_blocks_read_are_reported(caplog):
    model = _einet_model()
    model.encoder_network = nn.Sequential(nn.Linear(IN_DIM, IN_DIM), nn.Sigmoid())
    trainer, _ = _make_trainer("dfa", model)
    with caplog.at_level(logging.WARNING):
        _rule_gradients(trainer, model, *_batch())
    assert trainer.uncovered_parameters == [
        "encoder_network.0.weight",
        "encoder_network.0.bias",
    ]
    assert "several blocks read them" in caplog.text


@pytest.mark.parametrize("mode", ["fa", "dfa"])
def test_epoch_losses_are_reduced_across_ranks(monkeypatch, tmp_path, mode):
    """FA/DFA training losses are rank-invariant, as in the standard trainer."""
    import torch.distributed as dist
    from torch.utils.data import DataLoader, TensorDataset

    model = _mlp_model()
    loader = DataLoader(TensorDataset(*_batch(20)), batch_size=8)
    trainer, _ = _make_trainer(mode, deepcopy(model), epochs=1)
    single = trainer._epoch_train(trainer.optimizer.model, loader)

    payloads = []

    def all_reduce(tensor, op=None):
        assert op == dist.ReduceOp.SUM
        payloads.append(tensor.detach().clone())
        tensor.mul_(2)  # a second rank with the same values

    trainer, _ = _make_trainer(mode, model, epochs=1)
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        monkeypatch.setattr(dist, "get_world_size", lambda *_args: 2)
        monkeypatch.setattr(dist, "all_reduce", all_reduce)
        reduced = trainer._epoch_train(model, loader)
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()

    assert reduced == single
    epoch = payloads[-1]  # (total, base, regularization, batches)
    assert epoch.dtype == torch.float64 and epoch.tolist()[3] == 3.0
    assert epoch[0].item() / 3 == single[0]
