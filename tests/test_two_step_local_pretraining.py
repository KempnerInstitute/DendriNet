"""Two-step local pretraining trains each layer on its input in the model's forward.

Stage 1 validates like a standard epoch, its KL teacher runs in eval mode, and
the result is selected among stage-2 epochs.
"""

from __future__ import annotations

import math
from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config
from test_trainer_configuration import (
    IN_DIM,
    N_CLASSES,
    _data,
    _einet_model,
    _make_trainer,
    _population_model,
    _wrap,
)
from torch import nn

from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.training.loss.functions import LossFunction


def _two_layer_einet(seed=13):
    """EINet of two layers whose branches receive inhibitory synapses."""
    config = deepcopy(_config("einet", "shunting", "param_tanh"))
    config["architecture"].update(
        excitatory_layer_sizes=[4, 3],
        inhibitory_layer_sizes=[2, 2],
        inhibitory_branch_factors=[2],
    )
    config["connectivity"].update(
        ee_synapses_per_branch_per_layer=[2, 2],
        ie_synapses_per_branch_per_layer=[2, 2],
        ei_synapses_per_branch_per_layer=[2, 2],
        ii_synapses_per_branch_per_layer=[0, 0],
    )
    torch.manual_seed(seed)
    return _wrap(get_architecture("einet", config, input_dim=IN_DIM))


@pytest.mark.parametrize("builder", [_two_layer_einet, _population_model])
def test_forward_until_matches_every_layer_of_the_forward(builder):
    model = builder().eval()
    core = model.core_network
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    handles = [layer.register_forward_hook(capture) for layer in core.layers]
    x = model.encoder_network(_data(8)[:][0])
    with torch.no_grad():
        full = core(x)
        for handle in handles:
            handle.remove()
        for index in range(len(core.layers)):
            assert torch.equal(core.forward_until(x, index), captured[index])
        assert torch.equal(core.forward_until(x, len(core.layers) - 1), full)
    with pytest.raises(IndexError):
        core.forward_until(x, len(core.layers))


def _model_with_dropout_readout():
    model = _einet_model()
    width = model.decoder_network.in_features
    model.decoder_network = nn.Sequential(nn.Dropout(0.5), nn.Linear(width, N_CLASSES))
    model.output_dim = N_CLASSES
    return model


def test_stage_one_validation_is_a_standard_eval_mode_epoch():
    model = _model_with_dropout_readout()
    trainer, _ = _make_trainer(
        "freeze_branches", model, epochs=0, pretrain_epochs=1, batch_size=16
    )
    valid = _data(20, seed=1)  # batches of 16 and 4: sample weighting matters

    results = trainer.train(model, _data(), valid)

    _, valid_loader = trainer._dataloaders(_data(), valid)
    assert results["valid_losses"] == [
        trainer._non_distributed_validation_loss(model, valid_loader)
    ]


def test_the_kl_teacher_and_validation_run_in_eval_mode():
    model = _model_with_dropout_readout()
    trainer, _ = _make_trainer(
        "freeze_branch_kl", model, epochs=0, pretrain_epochs=1, batch_size=16
    )
    modes = []
    model.register_forward_pre_hook(
        lambda module, _inputs: modes.append(module.training)
    )

    trainer.train(model, _data(), _data(16, seed=1))

    assert modes and not any(modes)


def test_two_step_selects_its_result_among_stage_two_epochs():
    model = _einet_model()
    trainer, _ = _make_trainer(
        "freeze_branches",
        model,
        epochs=2,
        pretrain_epochs=1,
        load_best_state_dict=True,
    )
    values = iter([0.1, 0.5, 0.7])  # stage 1, then two stage-2 epochs
    trainer._non_distributed_validation_loss = lambda *_args: next(values)

    results = trainer.train(model, _data(), _data(seed=1))

    assert results["best_loss"] == 0.5
    assert results["best_epoch"] == 2  # counted across both stages
    assert results["phases"] == [
        {
            "name": "local_pretraining",
            "epochs": 1,
            "loss": f"local head {trainer.loss_function._loss_name}",
        },
        {"name": "end_to_end", "epochs": 2},
    ]
    assert {len(results[key]) for key in ("train_losses", "train_losses_reg")} == {3}


def test_stage_two_never_falls_back_to_a_stage_one_state():
    model = _einet_model()
    trainer, _ = _make_trainer(
        "freeze_branches", model, epochs=1, pretrain_epochs=1, load_best_state_dict=True
    )
    values = iter([0.1, math.nan])
    trainer._non_distributed_validation_loss = lambda *_args: next(values)
    with pytest.raises(RuntimeError, match="no epoch had a finite validation loss"):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize("strategy", ["freeze_branches", "freeze_branch_kl"])
def test_a_loss_that_needs_the_whole_model_is_rejected(strategy):
    class WholeModelLoss(LossFunction):
        def forward(self, model, x, y):
            return model(x).sum()

    model = _einet_model()
    trainer, _ = _make_trainer(strategy, model, pretrain_epochs=1)
    trainer.loss_function = WholeModelLoss()
    with pytest.raises(ValueError, match="needs the whole model"):
        trainer.train(model, _data(), _data(seed=1))


def test_cores_without_forward_until_are_rejected():
    class LayeredCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(IN_DIM, 4)])
            self.output_dim = 4

        def forward(self, x):
            return self.layers[0](x)

    model = _wrap(LayeredCore())
    trainer, _ = _make_trainer("freeze_branches", model, pretrain_epochs=1)
    with pytest.raises(ValueError, match="forward_until"):
        trainer.train(model, _data(), _data(seed=1))
