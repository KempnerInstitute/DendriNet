"""Tests for train_experiments analysis orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.scripts.training import train_experiments


class _FakeAnalysisManager:
    def __init__(self, *, track_training_performance: bool = True):
        self.config = SimpleNamespace(
            performance_analysis=SimpleNamespace(training=track_training_performance)
        )
        self.run_calls = []
        self.pruning_results = None
        self.set_pruning_calls = []

    def set_pruning_results(self, results):
        self.pruning_results = results
        self.set_pruning_calls.append(results)

    def run_analysis(self, *, filename: str, training: bool):
        self.run_calls.append({"filename": filename, "training": training})


class _ParamModel:
    def __init__(self, *params):
        self._params = params

    def parameters(self):
        return iter(self._params)


class _StateDictModel:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


@dataclass
class _RunConfig:
    value: int


def test_resolve_final_epoch_for_performance_plot_handles_missing_sections():
    """Final-epoch metadata should remain optional."""
    assert train_experiments._resolve_final_epoch_for_performance_plot(None) is None
    assert (
        train_experiments._resolve_final_epoch_for_performance_plot(SimpleNamespace())
        is None
    )

    training_config = SimpleNamespace(
        main=SimpleNamespace(common=SimpleNamespace(epochs=17))
    )
    assert (
        train_experiments._resolve_final_epoch_for_performance_plot(training_config)
        == 17
    )


def test_apply_cli_overrides_updates_config_fields():
    config = SimpleNamespace(
        training=SimpleNamespace(main=SimpleNamespace(strategy="standard")),
        outputs=SimpleNamespace(results_dir="old_dir", run_name="old_run"),
        experiment=SimpleNamespace(seed=1),
    )
    args = SimpleNamespace(
        learning_strategy="local_ca",
        output_dir="new_dir",
        run_name="new_run",
        seed=99,
    )

    train_experiments._apply_cli_overrides(config, args)

    assert config.training.main.strategy == "local_ca"
    assert config.outputs.results_dir == "new_dir"
    assert config.outputs.run_name == "new_run"
    assert config.experiment.seed == 99


def test_apply_cli_overrides_ignores_missing_args_object():
    config = SimpleNamespace(
        training=SimpleNamespace(main=SimpleNamespace(strategy="standard")),
        outputs=SimpleNamespace(results_dir="old_dir", run_name="old_run"),
        experiment=SimpleNamespace(seed=1),
    )

    train_experiments._apply_cli_overrides(config, None)

    assert config.training.main.strategy == "standard"
    assert config.outputs.results_dir == "old_dir"
    assert config.outputs.run_name == "old_run"
    assert config.experiment.seed == 1


def test_resolve_main_training_settings_from_dict_config():
    main_train_config = {
        "strategy": "recurrent",
        "optimizer": {"name": "sgd", "lr": 0.02},
        "common": {
            "param_groups": {"lr": 0.03, "split_params": False},
            "weight_decay_rate": 0.4,
            "weight_boosting": True,
            "epochs": 12,
        },
    }

    settings = train_experiments._resolve_main_training_settings(main_train_config)

    assert settings.common_config is main_train_config["common"]
    assert settings.strategy == "recurrent"
    assert settings.optimizer_config.name == "sgd"
    assert settings.optimizer_config.lr == 0.02
    assert settings.param_groups == {"lr": 0.03, "split_params": False}
    assert settings.weight_decay_rate == 0.4
    assert settings.weight_boosting is True
    assert settings.common_epochs == 12


def test_resolve_main_training_settings_from_object_config():
    common_config = SimpleNamespace(
        param_groups={"lr": 0.05},
        weight_decay_rate=0.2,
        epochs=9,
    )
    main_train_config = SimpleNamespace(
        strategy="standard",
        optimizer=SimpleNamespace(name="adamw", lr=0.01),
        common=common_config,
    )

    settings = train_experiments._resolve_main_training_settings(main_train_config)

    assert settings.common_config is common_config
    assert settings.strategy == "standard"
    assert settings.optimizer_config.name == "adamw"
    assert settings.optimizer_config.lr == 0.01
    assert settings.param_groups == {"lr": 0.05}
    assert settings.weight_decay_rate == 0.2
    assert settings.weight_boosting is False
    assert settings.common_epochs == 9


def test_install_encoder_network_preserves_identity_instance():
    model = SimpleNamespace()
    encoder = train_experiments.Identity(input_dim=5)

    train_experiments._install_encoder_network(model, encoder)

    assert model.encoder_network is encoder


def test_install_encoder_network_uses_identity_for_encoder_wrappers():
    model = SimpleNamespace()
    encoder = SimpleNamespace(output_dim=7)
    encoder_network = SimpleNamespace(encoder=encoder)

    train_experiments._install_encoder_network(model, encoder_network)

    assert isinstance(model.encoder_network, train_experiments.Identity)
    assert model.encoder_network.output_dim == 7


def test_install_encoder_network_uses_direct_encoder_without_encoder_attr():
    model = SimpleNamespace()
    encoder_network = SimpleNamespace(output_dim=3)

    train_experiments._install_encoder_network(model, encoder_network)

    assert model.encoder_network is encoder_network


def test_record_model_size_summary_logs_and_saves_count(monkeypatch, tmp_path):
    model = _ParamModel(torch.zeros(2, 3), torch.zeros(4))
    config = SimpleNamespace(training=SimpleNamespace())
    wandb_logs = []
    saved_payloads = []

    monkeypatch.setattr(train_experiments.wandb, "run", object())
    monkeypatch.setattr(
        train_experiments.wandb,
        "log",
        lambda payload: wandb_logs.append(payload),
    )
    monkeypatch.setattr(
        train_experiments,
        "save_dict",
        lambda payload, save_path, name: saved_payloads.append(
            (payload, save_path, name)
        ),
    )
    monkeypatch.setattr(
        train_experiments,
        "suggest_optimal_config_for_model_size",
        lambda nparams: {},
    )

    nparams = train_experiments._record_model_size_summary(model, config, str(tmp_path))

    assert nparams == 10
    assert wandb_logs == [{"nparams": 10}]
    assert saved_payloads == [({"nparams": 10}, str(tmp_path), "nparams")]


def test_record_model_size_summary_logs_only_changed_suggestions(
    monkeypatch,
    tmp_path,
    caplog,
):
    model = _ParamModel(torch.zeros(5))
    config = SimpleNamespace(training=SimpleNamespace(lr=0.01))
    saved_payloads = []

    monkeypatch.setattr(train_experiments.wandb, "run", None)
    monkeypatch.setattr(
        train_experiments,
        "save_dict",
        lambda payload, save_path, name: saved_payloads.append(
            (payload, save_path, name)
        ),
    )
    monkeypatch.setattr(
        train_experiments,
        "suggest_optimal_config_for_model_size",
        lambda nparams: {"lr": 0.01, "batch_size": 128},
    )

    with caplog.at_level("INFO", logger=train_experiments.__name__):
        train_experiments._record_model_size_summary(model, config, str(tmp_path))

    messages = [record.getMessage() for record in caplog.records]
    assert saved_payloads == [({"nparams": 5}, str(tmp_path), "nparams")]
    assert any("MODEL SIZE RECOMMENDATIONS" in message for message in messages)
    assert any("batch_size = 128" in message for message in messages)
    assert all("lr = 0.01" not in message for message in messages)


def test_save_run_config_applies_canonicalizers_in_order(monkeypatch):
    calls = []
    saved_payloads = []

    def fake_model_canonicalizer(payload):
        calls.append(("model", dict(payload)))
        payload = dict(payload)
        payload["model"] = True
        return payload

    def fake_rnn_canonicalizer(payload):
        calls.append(("rnn", dict(payload)))
        payload = dict(payload)
        payload["rnn"] = True
        return payload

    monkeypatch.setattr(
        train_experiments,
        "canonicalize_model_core_flags",
        fake_model_canonicalizer,
    )
    monkeypatch.setattr(
        train_experiments,
        "canonicalize_rnn_core_flags",
        fake_rnn_canonicalizer,
    )
    monkeypatch.setattr(
        train_experiments,
        "save_dict",
        lambda payload, save_path, name: saved_payloads.append(
            (payload, save_path, name)
        ),
    )

    train_experiments._save_run_config(_RunConfig(value=3), "/tmp/run")

    assert calls == [
        ("model", {"value": 3}),
        ("rnn", {"value": 3, "model": True}),
    ]
    assert saved_payloads == [
        ({"value": 3, "model": True, "rnn": True}, "/tmp/run", "config.json")
    ]


def test_load_or_train_encoder_datasets_non_ddp_calls_loader_once(monkeypatch):
    encoder_network = object()
    training_config = SimpleNamespace(encoder=object())
    model_config = SimpleNamespace(encoder=object())
    data_config = object()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    result = (object(), object(), object(), object())
    loader_calls = []

    def fake_loader(**kwargs):
        loader_calls.append(kwargs)
        return result

    def fail_barrier():
        raise AssertionError("barrier should not run outside DDP autoencoder path")

    monkeypatch.setattr(train_experiments, "load_train_encoder_network", fake_loader)
    monkeypatch.setattr(train_experiments.dist, "barrier", fail_barrier)

    encoded = train_experiments._load_or_train_encoder_datasets(
        encoder_network=encoder_network,
        training_config=training_config,
        model_config=model_config,
        data_config=data_config,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        world_size=1,
        is_main=True,
    )

    assert encoded is result
    assert len(loader_calls) == 1
    assert loader_calls[0]["encoder_network"] is encoder_network
    assert loader_calls[0]["encoder_train_config"] is training_config.encoder
    assert loader_calls[0]["train_ds"] is train_ds
    assert loader_calls[0]["valid_ds"] is valid_ds
    assert loader_calls[0]["test_ds"] is test_ds
    assert loader_calls[0]["encoder_network_config"] is model_config.encoder
    assert loader_calls[0]["task_config"] is data_config


def test_load_or_train_encoder_datasets_ddp_main_loads_before_barrier(monkeypatch):
    result = (object(), object(), object(), object())
    events = []

    def fake_loader(**kwargs):
        events.append("load")
        return result

    def fake_barrier():
        events.append("barrier")

    monkeypatch.setattr(train_experiments, "load_train_encoder_network", fake_loader)
    monkeypatch.setattr(train_experiments.dist, "barrier", fake_barrier)

    encoded = train_experiments._load_or_train_encoder_datasets(
        encoder_network=train_experiments.BaseAutoencoder(),
        training_config=SimpleNamespace(encoder=object()),
        model_config=SimpleNamespace(encoder=object()),
        data_config=object(),
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        world_size=2,
        is_main=True,
    )

    assert encoded is result
    assert events == ["load", "barrier"]


def test_load_or_train_encoder_datasets_ddp_worker_loads_after_barrier(monkeypatch):
    result = (object(), object(), object(), object())
    events = []

    def fake_loader(**kwargs):
        events.append("load")
        return result

    def fake_barrier():
        events.append("barrier")

    monkeypatch.setattr(train_experiments, "load_train_encoder_network", fake_loader)
    monkeypatch.setattr(train_experiments.dist, "barrier", fake_barrier)

    encoded = train_experiments._load_or_train_encoder_datasets(
        encoder_network=train_experiments.BaseAutoencoder(),
        training_config=SimpleNamespace(encoder=object()),
        model_config=SimpleNamespace(encoder=object()),
        data_config=object(),
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        world_size=2,
        is_main=False,
    )

    assert encoded is result
    assert events == ["barrier", "load"]


def test_maybe_plot_multistage_training_performance_plots_when_available(
    monkeypatch,
    tmp_path,
):
    performance_epochs = tmp_path / "performance" / "epochs"
    performance_epochs.mkdir(parents=True)
    analysis_config = SimpleNamespace(
        performance_analysis=SimpleNamespace(training=True)
    )
    main_train_config = SimpleNamespace(
        strategies=SimpleNamespace(
            multi_stage=SimpleNamespace(
                stages=[SimpleNamespace(epochs=2), SimpleNamespace()]
            )
        )
    )
    analysis_manager = SimpleNamespace(pruning_results={"accuracy": 0.7})
    plot_calls = []

    class FakePerformancePlotter:
        def plot_performance_evolution(self, **kwargs):
            plot_calls.append(kwargs)

    monkeypatch.setattr(
        train_experiments,
        "PerformancePlotter",
        FakePerformancePlotter,
    )

    train_experiments._maybe_plot_multistage_training_performance(
        strategy="multi_stage",
        analysis_config=analysis_config,
        run_save_path=str(tmp_path),
        main_train_config=main_train_config,
        common_epochs=3,
        analysis_manager=analysis_manager,
    )

    assert plot_calls == [
        {
            "performance_dir": str(performance_epochs),
            "save_path": str(tmp_path / "performance"),
            "pruning_performance": {"accuracy": 0.7},
            "final_epoch": 5,
        }
    ]


def test_maybe_plot_multistage_training_performance_skips_disabled_paths(
    monkeypatch,
    tmp_path,
):
    analysis_config = SimpleNamespace(
        performance_analysis=SimpleNamespace(training=False)
    )

    monkeypatch.setattr(
        train_experiments,
        "PerformancePlotter",
        lambda: pytest.fail("plotter should not be constructed"),
    )

    train_experiments._maybe_plot_multistage_training_performance(
        strategy="standard",
        analysis_config=analysis_config,
        run_save_path=str(tmp_path),
        main_train_config=SimpleNamespace(),
        common_epochs=3,
        analysis_manager=SimpleNamespace(),
    )
    train_experiments._maybe_plot_multistage_training_performance(
        strategy="multi_stage",
        analysis_config=analysis_config,
        run_save_path=str(tmp_path),
        main_train_config=SimpleNamespace(),
        common_epochs=3,
        analysis_manager=SimpleNamespace(),
    )


def test_maybe_apply_pruning_step_skips_disabled_config(monkeypatch):
    training_results = {"loss": 1.0}
    trainer_config = {"pruning": {}}

    monkeypatch.setattr(
        train_experiments,
        "get_trainer",
        lambda **kwargs: pytest.fail("pruning trainer should not be constructed"),
    )

    result = train_experiments._maybe_apply_pruning_step(
        original_pruning_config={"enabled": False},
        is_main=True,
        main_trainer_config_dict=trainer_config,
        strategy="standard",
        optimizer=object(),
        analysis_manager=object(),
        model=object(),
        common_epochs=3,
        encoded_train_ds=object(),
        encoded_valid_ds=object(),
        training_results=training_results,
    )

    assert result is training_results
    assert training_results == {"loss": 1.0}
    assert trainer_config == {"pruning": {}}


def test_maybe_apply_pruning_step_runs_and_merges_results(monkeypatch):
    original_pruning_config = {"enabled": True, "amount": 0.2}
    trainer_config = {"pruning": {}, "reporting": {"existing": True}}
    training_results = {"loss": 1.0}
    optimizer = object()
    analysis_manager = object()
    model = object()
    train_ds = object()
    valid_ds = object()
    trainer_calls = []
    initialize_calls = []
    prune_calls = []

    class FakePruningTrainer:
        def _initialize_attributes(self, received_model, epochs):
            initialize_calls.append((received_model, epochs))

        def _apply_pruning_and_evaluate(self, **kwargs):
            prune_calls.append(kwargs)
            return {"post_pruning_loss": 0.8}

    def fake_get_trainer(**kwargs):
        trainer_calls.append(kwargs)
        return FakePruningTrainer()

    monkeypatch.setattr(train_experiments, "get_trainer", fake_get_trainer)

    result = train_experiments._maybe_apply_pruning_step(
        original_pruning_config=original_pruning_config,
        is_main=True,
        main_trainer_config_dict=trainer_config,
        strategy="standard",
        optimizer=optimizer,
        analysis_manager=analysis_manager,
        model=model,
        common_epochs=7,
        encoded_train_ds=train_ds,
        encoded_valid_ds=valid_ds,
        training_results=training_results,
    )

    assert result is training_results
    assert training_results == {"loss": 1.0, "post_pruning_loss": 0.8}
    assert trainer_config["pruning"] is original_pruning_config
    assert trainer_config["reporting"] == {"existing": True}
    assert trainer_calls == [
        {
            "strategy": "standard",
            "optimizer": optimizer,
            "trainer_config_dict": trainer_config,
            "analysis_manager": analysis_manager,
        }
    ]
    assert initialize_calls == [(model, 7)]
    assert prune_calls == [
        {"model": model, "train_data": train_ds, "valid_data": valid_ds}
    ]


def test_save_final_model_and_analysis_runs_final_analysis(monkeypatch, tmp_path):
    model_state = {"weight": torch.ones(1)}
    model = _StateDictModel(model_state)
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    analysis_manager = object()
    training_config = object()
    training_results = {"loss": 1.0}
    save_calls = []
    analysis_calls = []

    monkeypatch.setattr(
        train_experiments.torch,
        "save",
        lambda state, path: save_calls.append((state, path)),
    )
    monkeypatch.setattr(
        train_experiments,
        "run_analysis",
        lambda **kwargs: analysis_calls.append(kwargs),
    )

    train_experiments._save_final_model_and_analysis(
        unwrapped_model=model,
        run_save_path=str(tmp_path),
        any_analysis_enabled=True,
        strategy="standard",
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        analysis_manager=analysis_manager,
        training_config=training_config,
        training_results=training_results,
    )

    assert save_calls == [(model_state, str(tmp_path / "final_model.pt"))]
    assert analysis_calls == [
        {
            "train_ds": train_ds,
            "valid_ds": valid_ds,
            "test_ds": test_ds,
            "run_save_path": str(tmp_path),
            "analysis_manager": analysis_manager,
            "training_config": training_config,
            "training": False,
            "training_results": training_results,
        }
    ]


def test_save_final_model_and_analysis_skips_multistage_analysis(
    monkeypatch,
    tmp_path,
):
    model_state = {"weight": torch.ones(1)}
    save_calls = []

    monkeypatch.setattr(
        train_experiments.torch,
        "save",
        lambda state, path: save_calls.append((state, path)),
    )
    monkeypatch.setattr(
        train_experiments,
        "run_analysis",
        lambda **kwargs: pytest.fail("multi-stage final analysis should be skipped"),
    )

    train_experiments._save_final_model_and_analysis(
        unwrapped_model=_StateDictModel(model_state),
        run_save_path=str(tmp_path),
        any_analysis_enabled=True,
        strategy="multi_stage",
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        analysis_manager=object(),
        training_config=object(),
        training_results={},
    )

    assert save_calls == [(model_state, str(tmp_path / "final_model.pt"))]


def test_broadcast_run_save_path_skips_single_process(monkeypatch):
    monkeypatch.setattr(
        train_experiments.dist,
        "broadcast",
        lambda *args, **kwargs: pytest.fail("single-process path should not broadcast"),
    )

    result = train_experiments._broadcast_run_save_path(
        "/tmp/local",
        world_size=1,
        is_main=True,
        device="cpu",
    )

    assert result == "/tmp/local"


def test_broadcast_run_save_path_rank0_broadcasts_original_path(monkeypatch):
    calls = []

    def fake_broadcast(tensor, src):
        calls.append((tensor.clone(), src))

    monkeypatch.setattr(train_experiments.dist, "broadcast", fake_broadcast)

    result = train_experiments._broadcast_run_save_path(
        "/tmp/rank0",
        world_size=2,
        is_main=True,
        device="cpu",
    )

    assert result == "/tmp/rank0"
    assert [src for _, src in calls] == [0, 0]
    assert calls[0][0].tolist() == [len(b"/tmp/rank0")]
    assert bytes(calls[1][0].tolist()).decode("utf-8") == "/tmp/rank0"


def test_broadcast_run_save_path_worker_adopts_rank0_path(monkeypatch):
    shared_path = b"/tmp/rank0"
    calls = []

    def fake_broadcast(tensor, src):
        calls.append((tensor.dtype, src))
        if tensor.dtype == torch.long:
            tensor.fill_(len(shared_path))
        else:
            tensor.copy_(torch.tensor(list(shared_path), dtype=torch.uint8))

    monkeypatch.setattr(train_experiments.dist, "broadcast", fake_broadcast)

    result = train_experiments._broadcast_run_save_path(
        "/tmp/local-worker",
        world_size=2,
        is_main=False,
        device="cpu",
    )

    assert result == shared_path.decode("utf-8")
    assert calls == [(torch.long, 0), (torch.uint8, 0)]


def test_run_analysis_attaches_pruning_results_and_plots_evolution(
    monkeypatch,
    tmp_path,
):
    """Final analysis should preserve pruning metadata and plot epoch curves."""
    performance_epochs = tmp_path / "performance" / "epochs"
    performance_epochs.mkdir(parents=True)
    manager = _FakeAnalysisManager(track_training_performance=True)
    training_results = {"pre_pruning_performance": {"accuracy": 0.8}}
    training_config = SimpleNamespace(
        main=SimpleNamespace(common=SimpleNamespace(epochs=5))
    )
    plot_calls = []

    class FakePerformancePlotter:
        def plot_performance_evolution(self, **kwargs):
            plot_calls.append(kwargs)

    monkeypatch.setattr(
        train_experiments,
        "PerformancePlotter",
        FakePerformancePlotter,
    )

    train_experiments.run_analysis(
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        run_save_path=str(tmp_path),
        analysis_manager=manager,
        training=False,
        training_results=training_results,
        training_config=training_config,
    )

    assert manager.run_calls == [{"filename": "final", "training": False}]
    assert manager.set_pruning_calls == [training_results]
    assert plot_calls == [
        {
            "performance_dir": str(performance_epochs),
            "save_path": str(tmp_path / "performance"),
            "pruning_performance": training_results,
            "final_epoch": 5,
        }
    ]


def test_run_analysis_skips_performance_plot_during_training(monkeypatch, tmp_path):
    """Training-phase analysis should not generate final performance plots."""
    (tmp_path / "performance" / "epochs").mkdir(parents=True)
    manager = _FakeAnalysisManager(track_training_performance=True)
    plot_calls = []

    class FakePerformancePlotter:
        def plot_performance_evolution(self, **kwargs):
            plot_calls.append(kwargs)

    monkeypatch.setattr(
        train_experiments,
        "PerformancePlotter",
        FakePerformancePlotter,
    )

    train_experiments.run_analysis(
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        run_save_path=str(tmp_path),
        analysis_manager=manager,
        training=True,
        training_results={"post_pruning_performance": {"accuracy": 0.9}},
        training_config=None,
    )

    assert manager.run_calls == [{"filename": "final", "training": True}]
    assert plot_calls == []


def test_run_analysis_does_not_attach_non_pruning_results(tmp_path):
    """Only pruning result keys should be attached to the analysis manager."""
    manager = _FakeAnalysisManager(track_training_performance=False)

    train_experiments.run_analysis(
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        run_save_path=str(tmp_path),
        analysis_manager=manager,
        training=False,
        training_results={"train_losses": [1.0]},
        training_config=None,
    )

    assert manager.run_calls == [{"filename": "final", "training": False}]
    assert manager.set_pruning_calls == []
