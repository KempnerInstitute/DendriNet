import torch
from torch import nn

from dendritic_modeling.training.strategies.layerwise import (
    LayerWiseTrainer,
    _empty_loss_histories,
    _layer_logger_info,
    _layerwise_total_epochs,
    _loss_histories_from_results,
    _ordered_layer_training_plan,
    _prepend_loss_histories,
    _prepend_loss_histories_to_results,
    _restore_best_state_and_unfreeze,
    _set_all_parameters_trainable,
    _set_only_module_trainable,
    _strategy_filename_prefix,
    _strategy_logger_prefix,
)
from dendritic_modeling.training.strategies.standard import Trainer


def test_ordered_layer_training_plan_preserves_forward_order():
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 1)])

    strategy, plan = _ordered_layer_training_plan(layers, reverse_training=False)

    assert strategy == "Freeze-Layers"
    assert [idx for idx, _layer in plan] == [0, 1, 2]
    assert [layer for _idx, layer in plan] == list(layers)


def test_ordered_layer_training_plan_preserves_reverse_indices():
    layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 1)])

    strategy, plan = _ordered_layer_training_plan(layers, reverse_training=True)

    assert strategy == "Freeze-Layers-Reversed"
    assert [idx for idx, _layer in plan] == [2, 1, 0]
    assert [layer for _idx, layer in plan] == list(reversed(layers))


def test_layerwise_total_epochs_matches_legacy_formula():
    assert (
        _layerwise_total_epochs(
            pretrain_epochs=3,
            n_layers=4,
            epochs_per_layer=5,
            final_tune_epochs=6,
        )
        == 29
    )


def test_strategy_prefix_helpers_preserve_legacy_strings():
    assert _strategy_filename_prefix("Freeze-Layers-Reversed") == (
        "freeze_layers_reversed_"
    )
    assert _strategy_logger_prefix("Freeze-Layers-Reversed") == (
        "[Freeze-Layers-Reversed] "
    )
    assert _layer_logger_info("Freeze-Layers-Reversed", 3) == (
        "[Freeze-Layers-Reversed] layer=3, "
    )


def test_set_only_module_trainable_freezes_model_except_target_module():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

    _set_only_module_trainable(model, model[1])

    assert all(not param.requires_grad for param in model[0].parameters())
    assert all(param.requires_grad for param in model[1].parameters())


def test_set_all_parameters_trainable_unfreezes_every_parameter():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    for param in model.parameters():
        param.requires_grad = False

    _set_all_parameters_trainable(model)

    assert all(param.requires_grad for param in model.parameters())


def test_restore_best_state_and_unfreeze_loads_state_and_unfreezes_parameters():
    source_model = nn.Linear(2, 1)
    target_model = nn.Linear(2, 1)
    with torch.no_grad():
        source_model.weight.fill_(0.75)
        source_model.bias.fill_(-0.5)
    for param in target_model.parameters():
        param.requires_grad = False

    _restore_best_state_and_unfreeze(target_model, source_model.state_dict())

    assert torch.equal(target_model.weight, source_model.weight)
    assert torch.equal(target_model.bias, source_model.bias)
    assert all(param.requires_grad for param in target_model.parameters())


def test_loss_histories_from_results_uses_legacy_keys_and_empty_defaults():
    assert _loss_histories_from_results({}) == ([], [], [], [])

    histories = _loss_histories_from_results(
        {
            "train_losses": [1.0],
            "valid_losses": [2.0],
            "train_losses_base": [3.0],
            "train_losses_reg": [4.0],
        }
    )

    assert histories == ([1.0], [2.0], [3.0], [4.0])


def test_empty_loss_histories_returns_four_independent_lists():
    train, valid, base, reg = _empty_loss_histories()

    train.append(1.0)

    assert train == [1.0]
    assert valid == []
    assert base == []
    assert reg == []


def test_prepend_loss_histories_preserves_metric_order():
    combined = _prepend_loss_histories(
        ([1.0], [2.0], [3.0], [4.0]),
        ([10.0], [20.0], [30.0], [40.0]),
    )

    assert combined == ([1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0])


def test_prepend_loss_histories_to_results_mutates_final_results_in_place():
    results = {"train_losses": [10.0], "valid_losses": [20.0], "other": "kept"}

    returned = _prepend_loss_histories_to_results(
        results,
        ([1.0], [2.0], [3.0], [4.0]),
    )

    assert returned is results
    assert results == {
        "train_losses": [1.0, 10.0],
        "valid_losses": [2.0, 20.0],
        "train_losses_base": [3.0],
        "train_losses_reg": [4.0],
        "other": "kept",
    }


def test_set_only_module_trainable_does_not_change_parameter_values():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    before = [param.detach().clone() for param in model.parameters()]

    _set_only_module_trainable(model, model[0])

    after = [param.detach() for param in model.parameters()]
    for before_param, after_param in zip(before, after):
        assert torch.equal(before_param, after_param)


def test_train_layer_plan_sets_prefixes_and_trains_each_target_layer():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    trainer = LayerWiseTrainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=7,
    )
    trainer.epochs_per_layer = 3
    calls = []
    plot_calls = []

    def _fake_run_training_loop(**kwargs):
        trainable_by_layer = [
            [param.requires_grad for param in module.parameters()] for module in model
        ]
        calls.append(
            {
                "n_epochs": kwargs["n_epochs"],
                "logger_info": kwargs["logger_info"],
                "wandb_info": kwargs["wandb_info"],
                "trainable_by_layer": trainable_by_layer,
            }
        )
        layer_idx = kwargs["wandb_info"]["layer"]
        kwargs["train_losses"].append(float(layer_idx))
        kwargs["valid_losses"].append(float(layer_idx + 10))
        kwargs["train_losses_base"].append(float(layer_idx + 20))
        kwargs["train_losses_reg"].append(float(layer_idx + 30))
        return (
            kwargs["train_losses"],
            kwargs["valid_losses"],
            kwargs["train_losses_base"],
            kwargs["train_losses_reg"],
        )

    trainer._run_training_loop = _fake_run_training_loop
    trainer._loss_plotting = lambda *histories: plot_calls.append(histories)

    histories = trainer._train_layer_plan(
        model,
        train_loader=object(),
        valid_loader=object(),
        layer_plan=[(1, model[1]), (0, model[0])],
        learning_strategy="Freeze-Layers-Reversed",
    )

    assert trainer.filename_prefix == "freeze_layers_reversed_"
    assert trainer.logger_info_prefix == "[Freeze-Layers-Reversed] "
    assert histories == ([1.0, 0.0], [11.0, 10.0], [21.0, 20.0], [31.0, 30.0])
    assert plot_calls == [histories]
    assert calls == [
        {
            "n_epochs": 3,
            "logger_info": "[Freeze-Layers-Reversed] layer=1, ",
            "wandb_info": {"layer": 1},
            "trainable_by_layer": [[False, False], [True, True]],
        },
        {
            "n_epochs": 3,
            "logger_info": "[Freeze-Layers-Reversed] layer=0, ",
            "wandb_info": {"layer": 0},
            "trainable_by_layer": [[True, True], [False, False]],
        },
    ]


def test_run_final_tuning_preserves_legacy_prefix_reset(monkeypatch):
    model = nn.Linear(2, 1)
    trainer = LayerWiseTrainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=9,
    )
    trainer.final_tune_epochs = 4
    trainer.filename_prefix = "previous_"
    trainer.logger_info_prefix = "[Previous] "
    train_data = object()
    valid_data = object()
    calls = []

    def _fake_train(self, model_arg, train_data_arg, valid_data_arg):
        calls.append(
            {
                "epochs": self.epochs,
                "filename_prefix": self.filename_prefix,
                "logger_info_prefix": self.logger_info_prefix,
                "model": model_arg,
                "train_data": train_data_arg,
                "valid_data": valid_data_arg,
            }
        )
        return {"train_losses": [4.0]}

    monkeypatch.setattr(Trainer, "train", _fake_train)

    results = trainer._run_final_tuning(model, train_data, valid_data)

    assert results == {"train_losses": [4.0]}
    assert calls == [
        {
            "epochs": 4,
            "filename_prefix": "final_tune_",
            "logger_info_prefix": "[Final Tune] ",
            "model": model,
            "train_data": train_data,
            "valid_data": valid_data,
        }
    ]
    assert trainer.epochs == 9
    assert trainer.filename_prefix == ""
    assert trainer.logger_info_prefix == "[Final Tune] "


def test_pretrain_restores_epoch_and_filename_after_failure(monkeypatch):
    model = nn.Linear(2, 1)
    trainer = LayerWiseTrainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=9,
    )
    trainer.pretrain_epochs = 4
    trainer.filename_prefix = "previous_"
    trainer.logger_info_prefix = "[Previous] "

    def _failing_train(self, model_arg, train_data_arg, valid_data_arg):
        assert model_arg is model
        assert train_data_arg is train_data
        assert valid_data_arg is valid_data
        assert self.epochs == 4
        assert self.filename_prefix == "pretrain_"
        assert self.logger_info_prefix == "[Pretrain] "
        raise RuntimeError("pretrain failed")

    train_data = object()
    valid_data = object()
    monkeypatch.setattr(Trainer, "train", _failing_train)

    try:
        trainer._pretrain(model, train_data=train_data, valid_data=valid_data)
    except RuntimeError as exc:
        assert str(exc) == "pretrain failed"
    else:
        raise AssertionError("Expected pretrain failure")

    assert trainer.epochs == 9
    assert trainer.filename_prefix == ""
    assert trainer.logger_info_prefix == "[Pretrain] "


def test_run_final_tuning_restores_epoch_and_filename_after_failure(monkeypatch):
    model = nn.Linear(2, 1)
    trainer = LayerWiseTrainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=9,
    )
    trainer.final_tune_epochs = 4
    trainer.filename_prefix = "previous_"
    trainer.logger_info_prefix = "[Previous] "

    def _failing_train(self, model_arg, train_data_arg, valid_data_arg):
        assert model_arg is model
        assert train_data_arg is train_data
        assert valid_data_arg is valid_data
        assert self.epochs == 4
        assert self.filename_prefix == "final_tune_"
        assert self.logger_info_prefix == "[Final Tune] "
        raise RuntimeError("final tune failed")

    train_data = object()
    valid_data = object()
    monkeypatch.setattr(Trainer, "train", _failing_train)

    try:
        trainer._run_final_tuning(model, train_data=train_data, valid_data=valid_data)
    except RuntimeError as exc:
        assert str(exc) == "final tune failed"
    else:
        raise AssertionError("Expected final tuning failure")

    assert trainer.epochs == 9
    assert trainer.filename_prefix == ""
    assert trainer.logger_info_prefix == "[Final Tune] "
