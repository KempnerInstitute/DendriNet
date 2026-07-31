"""Tests for train_experiments analysis orchestration helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.scripts.training import train_experiments


def _enable_package_caplog(monkeypatch) -> None:
    """Let caplog see package logs while preserving normal logger configuration."""
    monkeypatch.setattr(logging.getLogger("dendritic_modeling"), "propagate", True)


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
        self.core_network = SimpleNamespace(is_recurrent=False)

    def parameters(self):
        return iter(self._params)


class _SummaryModel:
    def __init__(self, *, is_recurrent):
        self.core_network = SimpleNamespace(is_recurrent=is_recurrent)
        self._params = [
            torch.nn.Parameter(torch.zeros(2, 3)),
            torch.nn.Parameter(torch.zeros(4), requires_grad=False),
        ]

    def parameters(self):
        return iter(self._params)


class _ParamGroupModel:
    def __init__(self):
        self.param_group_calls = []

    def get_param_groups(self, param_groups):
        self.param_group_calls.append(param_groups)
        return ("model-param-groups",)


class _StateDictModel:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


class _MovableModel:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


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
        training=SimpleNamespace(
            main=SimpleNamespace(
                strategy="standard",
                common=SimpleNamespace(epochs=10),
            )
        ),
        outputs=SimpleNamespace(results_dir="old_dir", run_name="old_run"),
        experiment=SimpleNamespace(seed=1),
    )
    args = SimpleNamespace(
        learning_strategy="local_ca",
        output_dir="new_dir",
        run_name="new_run",
        seed=99,
        epochs=3,
    )

    train_experiments._apply_cli_overrides(config, args)

    assert config.training.main.strategy == "local_ca"
    assert config.outputs.results_dir == "new_dir"
    assert config.outputs.run_name == "new_run"
    assert config.experiment.seed == 99
    assert config.training.main.common.epochs == 3


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


def test_main_validate_only_stops_before_training(monkeypatch):
    config = SimpleNamespace(
        experiment=object(),
        data=object(),
        model=object(),
        training=object(),
        analysis=object(),
        wandb=object(),
        outputs=object(),
    )
    prepared = train_experiments._PreparedTrainingConfig(
        config=config,
        experiment_config=config.experiment,
        data_config=config.data,
        model_config=config.model,
        training_config=config.training,
        analysis_config=config.analysis,
        wandb_config=config.wandb,
        outputs_config=config.outputs,
    )
    model = object()
    environment = train_experiments._TrainingEnvironment(
        run_save_path="/tmp/preflight",
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        model=model,
        unwrapped_model=model,
        encoder_network=object(),
    )

    monkeypatch.setattr(train_experiments, "setup_distributed", lambda: (0, 0, 1))
    monkeypatch.setattr(
        train_experiments,
        "_load_and_prepare_training_config",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        train_experiments, "_validate_sealed_test_analysis", lambda *args: None
    )
    monkeypatch.setattr(
        train_experiments,
        "_setup_training_environment",
        lambda **kwargs: environment,
    )
    recorded = []
    cleaned = []
    monkeypatch.setattr(
        train_experiments,
        "_record_model_size_summary",
        lambda *args: recorded.append(args),
    )
    monkeypatch.setattr(
        train_experiments,
        "_cleanup_training_runtime",
        lambda *args, **kwargs: cleaned.append((args, kwargs)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_prepare_encoded_training_state",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("training started")),
    )

    train_experiments.main("config.yaml", args=SimpleNamespace(validate_only=True))

    assert len(recorded) == 1
    assert len(cleaned) == 1


def test_load_and_prepare_training_config_returns_sections_and_disables_worker_wandb(
    monkeypatch,
):
    experiment_config = object()
    data_config = object()
    model_config = object()
    training_config = object()
    analysis_config = object()
    wandb_config = SimpleNamespace(use_wandb=True)
    outputs_config = object()
    config = SimpleNamespace(
        experiment=experiment_config,
        data=data_config,
        model=model_config,
        training=training_config,
        analysis=analysis_config,
        wandb=wandb_config,
        outputs=outputs_config,
    )
    args = object()
    events = []

    monkeypatch.setattr(
        train_experiments,
        "load_config",
        lambda path: events.append(("load", path)) or config,
    )
    monkeypatch.setattr(
        train_experiments,
        "validate_training_stability",
        lambda cfg: events.append(("validate", cfg)) or True,
    )
    monkeypatch.setattr(
        train_experiments,
        "apply_automatic_safeguards",
        lambda cfg: events.append(("safeguards", cfg)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_apply_cli_overrides",
        lambda cfg, received_args: events.append(("overrides", cfg, received_args)),
    )

    result = train_experiments._load_and_prepare_training_config(
        "/tmp/config.yaml",
        args,
        is_main=False,
        world_size=2,
    )

    assert result == train_experiments._PreparedTrainingConfig(
        config=config,
        experiment_config=experiment_config,
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        analysis_config=analysis_config,
        wandb_config=wandb_config,
        outputs_config=outputs_config,
    )
    assert wandb_config.use_wandb is False
    assert events == [
        ("load", "/tmp/config.yaml"),
        ("validate", config),
        ("safeguards", config),
        ("overrides", config, args),
    ]


def test_load_and_prepare_training_config_raises_on_failed_validation(monkeypatch):
    config = SimpleNamespace(wandb=SimpleNamespace(use_wandb=True))

    monkeypatch.setattr(train_experiments, "load_config", lambda path: config)
    monkeypatch.setattr(
        train_experiments,
        "validate_training_stability",
        lambda cfg: False,
    )
    monkeypatch.setattr(
        train_experiments,
        "apply_automatic_safeguards",
        lambda cfg: pytest.fail("safeguards should not run after failed validation"),
    )
    monkeypatch.setattr(
        train_experiments,
        "_apply_cli_overrides",
        lambda cfg, args: pytest.fail(
            "overrides should not run after failed validation"
        ),
    )

    with pytest.raises(ValueError, match="Invalid configuration"):
        train_experiments._load_and_prepare_training_config(
            "/tmp/config.yaml",
            object(),
            is_main=True,
            world_size=1,
        )


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
    resources = {"trainable_parameters": 6, "active_synapses": 4}
    monkeypatch.setattr(
        train_experiments,
        "model_resource_summary",
        lambda received_model: resources,
    )

    nparams = train_experiments._record_model_size_summary(model, config, str(tmp_path))

    assert nparams == 10
    assert wandb_logs == [{"nparams": 10}]
    assert saved_payloads == [
        ({"nparams": 10}, str(tmp_path), "nparams"),
        (resources, str(tmp_path), "model_resources.json"),
    ]


def test_record_model_size_summary_logs_only_changed_suggestions(
    monkeypatch,
    tmp_path,
    caplog,
):
    _enable_package_caplog(monkeypatch)
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
    resources = {"trainable_parameters": 5, "active_synapses": 0}
    monkeypatch.setattr(
        train_experiments,
        "model_resource_summary",
        lambda received_model: resources,
    )

    with caplog.at_level("INFO", logger=train_experiments.__name__):
        train_experiments._record_model_size_summary(model, config, str(tmp_path))

    messages = [record.getMessage() for record in caplog.records]
    assert saved_payloads == [
        ({"nparams": 5}, str(tmp_path), "nparams"),
        (resources, str(tmp_path), "model_resources.json"),
    ]
    assert any("MODEL SIZE RECOMMENDATIONS" in message for message in messages)
    assert any("batch_size = 128" in message for message in messages)
    assert all("lr = 0.01" not in message for message in messages)


def test_save_training_summary_recursively_removes_state_dicts(monkeypatch):
    saved_payloads = []
    monkeypatch.setattr(
        train_experiments,
        "save_dict",
        lambda payload, save_path, name: saved_payloads.append(
            (payload, save_path, name)
        ),
    )
    results = {
        "best_loss": 0.25,
        "best_state_dict": {"weight": torch.ones(100)},
        "stages": [
            {
                "name": "fit",
                "results": {
                    "valid_losses": (0.5, 0.25),
                    "final_state_dict": {"weight": torch.zeros(100)},
                },
            }
        ],
    }

    train_experiments._save_training_summary(results, "/tmp/run")

    assert saved_payloads == [
        (
            {
                "best_loss": 0.25,
                "stages": [
                    {
                        "name": "fit",
                        "results": {"valid_losses": [0.5, 0.25]},
                    }
                ],
            },
            "/tmp/run",
            "training_summary.json",
        )
    ]


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


def test_create_analysis_manager_uses_encoded_data_and_cuda(monkeypatch):
    model = object()
    analysis_config = object()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    manager_calls = []

    class FakeAnalysisManager:
        def __init__(self, **kwargs):
            manager_calls.append(kwargs)

    monkeypatch.setattr(train_experiments, "AnalysisManager", FakeAnalysisManager)
    monkeypatch.setattr(train_experiments.torch.cuda, "is_available", lambda: True)

    manager = train_experiments._create_analysis_manager(
        unwrapped_model=model,
        encoded_train_ds=train_ds,
        encoded_valid_ds=valid_ds,
        encoded_test_ds=test_ds,
        analysis_config=analysis_config,
        run_save_path="/tmp/run",
    )

    assert isinstance(manager, FakeAnalysisManager)
    assert manager_calls == [
        {
            "model": model,
            "data": {"train": train_ds, "valid": valid_ds, "test": test_ds},
            "analysis_config": analysis_config,
            "save_root": "/tmp/run",
            "device": "cuda",
        }
    ]


def test_create_analysis_manager_uses_cpu_when_cuda_unavailable(monkeypatch):
    manager_calls = []

    class FakeAnalysisManager:
        def __init__(self, **kwargs):
            manager_calls.append(kwargs)

    monkeypatch.setattr(train_experiments, "AnalysisManager", FakeAnalysisManager)
    monkeypatch.setattr(train_experiments.torch.cuda, "is_available", lambda: False)

    train_experiments._create_analysis_manager(
        unwrapped_model=object(),
        encoded_train_ds=object(),
        encoded_valid_ds=object(),
        encoded_test_ds=object(),
        analysis_config=object(),
        run_save_path="/tmp/run",
    )

    assert manager_calls[0]["device"] == "cpu"


def test_prepare_encoded_training_state_runs_steps_in_order(monkeypatch):
    config = object()
    unwrapped_model = object()
    encoder_network = object()
    encoded_encoder = object()
    training_config = object()
    model_config = object()
    data_config = object()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    encoded_train_ds = object()
    encoded_valid_ds = object()
    encoded_test_ds = object()
    analysis_config = object()
    analysis_manager = object()
    events = []

    monkeypatch.setattr(
        train_experiments,
        "_load_or_train_encoder_datasets",
        lambda **kwargs: (
            events.append(("load", kwargs))
            or (encoded_encoder, encoded_train_ds, encoded_valid_ds, encoded_test_ds)
        ),
    )
    monkeypatch.setattr(
        train_experiments,
        "_create_analysis_manager",
        lambda **kwargs: events.append(("analysis", kwargs)) or analysis_manager,
    )
    monkeypatch.setattr(
        train_experiments,
        "_install_encoder_network",
        lambda model, encoder: events.append(("install", model, encoder)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_maybe_calibrate_data_driven_reactivation",
        lambda **kwargs: events.append(("calibrate", kwargs)),
    )

    result = train_experiments._prepare_encoded_training_state(
        config=config,
        unwrapped_model=unwrapped_model,
        encoder_network=encoder_network,
        training_config=training_config,
        model_config=model_config,
        data_config=data_config,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        analysis_config=analysis_config,
        run_save_path="/tmp/run",
        world_size=2,
        is_main=True,
    )

    assert result == train_experiments._EncodedTrainingState(
        encoder_network=encoded_encoder,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        encoded_test_ds=encoded_test_ds,
        analysis_manager=analysis_manager,
    )
    assert events == [
        (
            "load",
            {
                "encoder_network": encoder_network,
                "training_config": training_config,
                "model_config": model_config,
                "data_config": data_config,
                "train_ds": train_ds,
                "valid_ds": valid_ds,
                "test_ds": test_ds,
                "world_size": 2,
                "is_main": True,
            },
        ),
        (
            "analysis",
            {
                "unwrapped_model": unwrapped_model,
                "encoded_train_ds": encoded_train_ds,
                "encoded_valid_ds": encoded_valid_ds,
                "encoded_test_ds": encoded_test_ds,
                "analysis_config": analysis_config,
                "run_save_path": "/tmp/run",
            },
        ),
        ("install", unwrapped_model, encoded_encoder),
        (
            "calibrate",
            {
                "config": config,
                "model": unwrapped_model,
                "train_dataset": encoded_train_ds,
                "is_main": True,
            },
        ),
    ]


def test_log_model_summary_before_training_handles_recurrent_model(
    monkeypatch,
    caplog,
):
    _enable_package_caplog(monkeypatch)
    model = _SummaryModel(is_recurrent=True)
    config = SimpleNamespace(
        training=SimpleNamespace(main=SimpleNamespace(strategy="standard"))
    )
    wandb_logs = []

    monkeypatch.setattr(train_experiments.wandb, "run", object())
    monkeypatch.setattr(
        train_experiments.wandb,
        "log",
        lambda payload: wandb_logs.append(payload),
    )
    monkeypatch.setattr(
        train_experiments.torchinfo,
        "summary",
        lambda *args, **kwargs: pytest.fail("recurrent model should skip torchinfo"),
    )

    with caplog.at_level("INFO", logger=train_experiments.__name__):
        train_experiments._log_model_summary_before_training(model, object(), config)

    assert wandb_logs == [{"total_params": 10, "trainable_params": 6}]
    assert "Recurrent model detected" in caplog.text
    assert "Skipping torchinfo summary for recurrent model" in caplog.text


def test_log_model_summary_before_training_uses_torchinfo(monkeypatch):
    model = _SummaryModel(is_recurrent=False)
    sample = torch.zeros(3, 4, 5)
    config = SimpleNamespace(
        training=SimpleNamespace(main=SimpleNamespace(strategy="standard"))
    )
    summary_calls = []
    wandb_logs = []

    monkeypatch.setattr(train_experiments.wandb, "run", object())
    monkeypatch.setattr(
        train_experiments.wandb,
        "log",
        lambda payload: wandb_logs.append(payload),
    )
    monkeypatch.setattr(
        train_experiments.torchinfo,
        "summary",
        lambda *args, **kwargs: (
            summary_calls.append((args, kwargs))
            or SimpleNamespace(total_params=11, trainable_params=7)
        ),
    )

    train_experiments._log_model_summary_before_training(
        model,
        [(sample, torch.tensor(1))],
        config,
    )

    assert len(summary_calls) == 1
    args, kwargs = summary_calls[0]
    assert args == (model,)
    assert kwargs == {
        "input_size": (100, 3, 4, 5),
        "col_names": ["input_size", "output_size", "num_params"],
        "col_width": 25,
        "row_settings": ["depth"],
        "depth": 7,
    }
    assert wandb_logs == [{"total_params": 11, "trainable_params": 7}]


def test_setup_training_environment_runs_rank0_setup_outputs(monkeypatch):
    config = object()
    model_config = object()
    training_config = object()
    data_config = object()
    wandb_config = object()
    outputs_config = object()
    experiment_config = object()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    built_model = object()
    unwrapped_model = object()
    wrapped_model = SimpleNamespace(module=unwrapped_model)
    encoder_network = object()
    events = []

    def fake_setup_environment(**kwargs):
        events.append(("setup", kwargs))
        return (
            "/tmp/local-run",
            train_ds,
            valid_ds,
            test_ds,
            built_model,
            encoder_network,
        )

    def fake_broadcast(path, *, world_size, is_main):
        events.append(("broadcast", path, world_size, is_main))
        return "/tmp/shared-run"

    def fake_wrap(model, *, world_size, local_rank, is_main):
        events.append(("wrap", model, world_size, local_rank, is_main))
        return wrapped_model

    monkeypatch.setattr(train_experiments, "setup_environment", fake_setup_environment)
    monkeypatch.setattr(train_experiments, "_broadcast_run_save_path", fake_broadcast)
    monkeypatch.setattr(train_experiments, "_wrap_ddp_model", fake_wrap)
    monkeypatch.setattr(
        train_experiments,
        "_dump_init_gate_stats_diagnostic",
        lambda model, path: events.append(("dump", model, path)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_log_model_summary_before_training",
        lambda model, dataset, cfg: events.append(("summary", model, dataset, cfg)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_save_run_config",
        lambda cfg, path: events.append(("save_config", cfg, path)),
    )

    result = train_experiments._setup_training_environment(
        config=config,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        wandb_config=wandb_config,
        outputs_config=outputs_config,
        experiment_config=experiment_config,
        world_size=2,
        local_rank=1,
        is_main=True,
    )

    assert result == train_experiments._TrainingEnvironment(
        run_save_path="/tmp/shared-run",
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        model=wrapped_model,
        unwrapped_model=unwrapped_model,
        encoder_network=encoder_network,
    )
    assert events == [
        (
            "setup",
            {
                "model_config": model_config,
                "training_config": training_config,
                "data_config": data_config,
                "wandb_config": wandb_config,
                "outputs_config": outputs_config,
                "experiment_config": experiment_config,
                "is_main": True,
            },
        ),
        ("broadcast", "/tmp/local-run", 2, True),
        ("wrap", built_model, 2, 1, True),
        ("dump", unwrapped_model, "/tmp/shared-run"),
        ("summary", unwrapped_model, train_ds, config),
        ("save_config", config, "/tmp/shared-run"),
    ]


def test_setup_training_environment_worker_skips_rank0_outputs(monkeypatch):
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    built_model = object()
    encoder_network = object()
    events = []

    monkeypatch.setattr(
        train_experiments,
        "setup_environment",
        lambda **kwargs: (
            events.append(("setup", kwargs))
            or (
                "/tmp/worker-run",
                train_ds,
                valid_ds,
                test_ds,
                built_model,
                encoder_network,
            )
        ),
    )
    monkeypatch.setattr(
        train_experiments,
        "_broadcast_run_save_path",
        lambda path, *, world_size, is_main: (
            events.append(("broadcast", path, world_size, is_main)) or "/tmp/shared-run"
        ),
    )
    monkeypatch.setattr(
        train_experiments,
        "_wrap_ddp_model",
        lambda model, *, world_size, local_rank, is_main: (
            events.append(("wrap", model, world_size, local_rank, is_main)) or model
        ),
    )
    monkeypatch.setattr(
        train_experiments,
        "_dump_init_gate_stats_diagnostic",
        lambda *args: pytest.fail("worker rank should not write gate diagnostics"),
    )
    monkeypatch.setattr(
        train_experiments,
        "_log_model_summary_before_training",
        lambda *args: pytest.fail("worker rank should not log model summary"),
    )
    monkeypatch.setattr(
        train_experiments,
        "_save_run_config",
        lambda *args: pytest.fail("worker rank should not save config"),
    )

    result = train_experiments._setup_training_environment(
        config=object(),
        model_config=object(),
        training_config=object(),
        data_config=object(),
        wandb_config=object(),
        outputs_config=object(),
        experiment_config=object(),
        world_size=2,
        local_rank=1,
        is_main=False,
    )

    assert result.run_save_path == "/tmp/shared-run"
    assert result.train_ds is train_ds
    assert result.valid_ds is valid_ds
    assert result.test_ds is test_ds
    assert result.model is built_model
    assert result.unwrapped_model is built_model
    assert result.encoder_network is encoder_network
    assert [event[0] for event in events] == ["setup", "broadcast", "wrap"]


def test_run_main_training_phase_builds_trainer_and_runs_training(
    monkeypatch,
    tmp_path,
):
    unwrapped_model = _ParamGroupModel()
    wrapped_model = object()
    train_ds = object()
    valid_ds = object()
    analysis_manager = object()
    base_optimizer = object()
    training_results = {"loss": 0.1}
    prepared_config = {"pruning": {"enabled": True}, "existing": "value"}
    calls = []

    main_train_config = SimpleNamespace(
        strategy="standard",
        optimizer=SimpleNamespace(name="adam", lr=0.01),
        common=SimpleNamespace(
            param_groups={"split_params": True},
            weight_decay_rate=0.2,
            weight_boosting=True,
            epochs=9,
        ),
    )
    training_config = SimpleNamespace(main=main_train_config)
    model_config = SimpleNamespace(task="sequence-classification")
    experiment_config = SimpleNamespace(seed=123)

    class FakeWeightDecayOptimizer:
        def __init__(self, **kwargs):
            calls.append(("weight_decay_optimizer", kwargs))

    class FakeTrainer:
        def train(self, **kwargs):
            calls.append(("train", kwargs))
            return training_results

    def fake_prepare_trainer_config(**kwargs):
        calls.append(("prepare", kwargs))
        return dict(prepared_config)

    def fake_normalize_param_groups(param_groups):
        calls.append(("normalize_param_groups", param_groups))
        return ("normalized-param-groups",)

    def fake_create_optimizer(param_groups, optimizer_config):
        calls.append(("create_optimizer", param_groups, optimizer_config))
        return base_optimizer

    def fake_get_trainer(**kwargs):
        calls.append(("get_trainer", kwargs))
        return FakeTrainer()

    monkeypatch.setattr(
        train_experiments,
        "prepare_trainer_config",
        fake_prepare_trainer_config,
    )
    monkeypatch.setattr(
        train_experiments,
        "normalize_param_groups_config",
        fake_normalize_param_groups,
    )
    monkeypatch.setattr(
        train_experiments,
        "create_optimizer",
        fake_create_optimizer,
    )
    monkeypatch.setattr(
        train_experiments,
        "CustomWeightDecayOptimizer",
        FakeWeightDecayOptimizer,
    )
    monkeypatch.setattr(train_experiments, "get_trainer", fake_get_trainer)

    result = train_experiments._run_main_training_phase(
        unwrapped_model=unwrapped_model,
        model=wrapped_model,
        training_config=training_config,
        model_config=model_config,
        experiment_config=experiment_config,
        run_save_path=str(tmp_path),
        encoded_train_ds=train_ds,
        encoded_valid_ds=valid_ds,
        analysis_manager=analysis_manager,
    )

    assert (tmp_path / "main_network").is_dir()
    assert unwrapped_model.param_group_calls == [("normalized-param-groups",)]
    assert result.main_train_config is main_train_config
    assert result.trainer_config_dict == {
        "pruning": {},
        "existing": "value",
        "seed": 123,
        "loader_seed": 123,
        "task": "sequence-classification",
    }
    assert result.optimizer is not None
    assert result.strategy == "standard"
    assert result.common_epochs == 9
    assert result.original_pruning_config == {"enabled": True}
    assert result.training_results is training_results

    assert calls[0] == (
        "prepare",
        {
            "main_train_config": main_train_config,
            "experiment_config": experiment_config,
            "save_path": str(tmp_path / "main_network"),
        },
    )
    assert calls[1] == (
        "normalize_param_groups",
        {"split_params": True},
    )
    assert calls[2][0] == "create_optimizer"
    assert calls[2][1] == ("model-param-groups",)
    assert calls[2][2].name == "adam"
    assert calls[3] == (
        "weight_decay_optimizer",
        {
            "model": unwrapped_model,
            "optimizer": base_optimizer,
            "weight_decay": 0.2,
            "weight_boosting": True,
        },
    )
    assert calls[4] == (
        "get_trainer",
        {
            "strategy": "standard",
            "optimizer": result.optimizer,
            "trainer_config_dict": result.trainer_config_dict,
            "analysis_manager": analysis_manager,
        },
    )
    assert calls[5] == (
        "train",
        {"model": wrapped_model, "train_data": train_ds, "valid_data": valid_ds},
    )


def test_run_main_training_phase_requires_multistage_config(monkeypatch, tmp_path):
    main_train_config = SimpleNamespace(
        strategy="multi_stage",
        optimizer=SimpleNamespace(name="adam", lr=0.01),
        common=SimpleNamespace(
            param_groups={},
            weight_decay_rate=0.0,
            epochs=3,
        ),
    )

    monkeypatch.setattr(
        train_experiments,
        "prepare_trainer_config",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        train_experiments,
        "normalize_param_groups_config",
        lambda param_groups: param_groups,
    )
    monkeypatch.setattr(
        train_experiments,
        "create_optimizer",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        train_experiments,
        "CustomWeightDecayOptimizer",
        lambda **kwargs: object(),
    )

    with pytest.raises(ValueError, match="multi_stage_config"):
        train_experiments._run_main_training_phase(
            unwrapped_model=_ParamGroupModel(),
            model=object(),
            training_config=SimpleNamespace(main=main_train_config),
            model_config=SimpleNamespace(task="task"),
            experiment_config=SimpleNamespace(seed=0),
            run_save_path=str(tmp_path),
            encoded_train_ds=object(),
            encoded_valid_ds=object(),
            analysis_manager=object(),
        )


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


def test_finalize_main_training_phase_runs_rank0_outputs_and_pruning(monkeypatch):
    unwrapped_model = object()
    config = object()
    analysis_config = SimpleNamespace(any_enabled=lambda: True)
    main_train_config = object()
    trainer_config = {"pruning": {}}
    optimizer = object()
    analysis_manager = object()
    model = object()
    encoded_train_ds = object()
    encoded_valid_ds = object()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    training_config = object()
    training_results = {"loss": 1.0}
    pruned_results = {"loss": 1.0, "post_pruning_loss": 0.7}
    events = []

    main_training = train_experiments._MainTrainingRun(
        main_train_config=main_train_config,
        trainer_config_dict=trainer_config,
        optimizer=optimizer,
        strategy="standard",
        common_epochs=5,
        original_pruning_config={"enabled": True},
        training_results=training_results,
    )

    monkeypatch.setattr(
        train_experiments,
        "_record_model_size_summary",
        lambda *args: events.append(("size", args)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_maybe_plot_multistage_training_performance",
        lambda **kwargs: events.append(("plot", kwargs)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_maybe_apply_pruning_step",
        lambda **kwargs: events.append(("pruning", kwargs)) or pruned_results,
    )
    monkeypatch.setattr(
        train_experiments,
        "_save_final_model_and_analysis",
        lambda **kwargs: events.append(("save", kwargs)),
    )
    monkeypatch.setattr(
        train_experiments,
        "_save_training_summary",
        lambda *args: events.append(("summary", args)),
    )

    result = train_experiments._finalize_main_training_phase(
        is_main=True,
        unwrapped_model=unwrapped_model,
        config=config,
        run_save_path="/tmp/run",
        analysis_config=analysis_config,
        main_training=main_training,
        analysis_manager=analysis_manager,
        model=model,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        training_config=training_config,
        training_results=training_results,
    )

    assert result is pruned_results
    assert [event[0] for event in events] == [
        "size",
        "plot",
        "pruning",
        "summary",
        "save",
    ]
    assert events[0][1] == (unwrapped_model, config, "/tmp/run")
    assert events[1][1] == {
        "strategy": "standard",
        "analysis_config": analysis_config,
        "run_save_path": "/tmp/run",
        "main_train_config": main_train_config,
        "common_epochs": 5,
        "analysis_manager": analysis_manager,
    }
    assert events[2][1] == {
        "original_pruning_config": {"enabled": True},
        "is_main": True,
        "main_trainer_config_dict": trainer_config,
        "strategy": "standard",
        "optimizer": optimizer,
        "analysis_manager": analysis_manager,
        "model": model,
        "common_epochs": 5,
        "encoded_train_ds": encoded_train_ds,
        "encoded_valid_ds": encoded_valid_ds,
        "training_results": training_results,
    }
    assert events[3][1] == (pruned_results, "/tmp/run")
    assert events[4][1] == {
        "unwrapped_model": unwrapped_model,
        "run_save_path": "/tmp/run",
        "any_analysis_enabled": True,
        "strategy": "standard",
        "train_ds": train_ds,
        "valid_ds": valid_ds,
        "test_ds": test_ds,
        "analysis_manager": analysis_manager,
        "training_config": training_config,
        "training_results": pruned_results,
    }


def test_finalize_main_training_phase_workers_only_run_pruning(monkeypatch):
    training_results = {"loss": 1.0}
    pruned_results = {"loss": 0.8}
    events = []

    main_training = train_experiments._MainTrainingRun(
        main_train_config=object(),
        trainer_config_dict={"pruning": {}},
        optimizer=object(),
        strategy="standard",
        common_epochs=3,
        original_pruning_config={},
        training_results=training_results,
    )

    monkeypatch.setattr(
        train_experiments,
        "_record_model_size_summary",
        lambda *args: pytest.fail("worker rank should not record size summary"),
    )
    monkeypatch.setattr(
        train_experiments,
        "_maybe_plot_multistage_training_performance",
        lambda **kwargs: pytest.fail("worker rank should not plot performance"),
    )
    monkeypatch.setattr(
        train_experiments,
        "_save_final_model_and_analysis",
        lambda **kwargs: pytest.fail("worker rank should not save final model"),
    )
    monkeypatch.setattr(
        train_experiments,
        "_save_training_summary",
        lambda *args: pytest.fail("worker rank should not save training summary"),
    )
    monkeypatch.setattr(
        train_experiments,
        "_maybe_apply_pruning_step",
        lambda **kwargs: events.append(kwargs) or pruned_results,
    )

    result = train_experiments._finalize_main_training_phase(
        is_main=False,
        unwrapped_model=object(),
        config=object(),
        run_save_path="/tmp/run",
        analysis_config=SimpleNamespace(
            any_enabled=lambda: pytest.fail("worker rank should not inspect analysis")
        ),
        main_training=main_training,
        analysis_manager=object(),
        model=object(),
        encoded_train_ds=object(),
        encoded_valid_ds=object(),
        train_ds=object(),
        valid_ds=object(),
        test_ds=object(),
        training_config=object(),
        training_results=training_results,
    )

    assert result is pruned_results
    assert len(events) == 1
    assert events[0]["is_main"] is False
    assert events[0]["training_results"] is training_results


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


def test_wrap_ddp_model_skips_single_process(monkeypatch):
    model = _MovableModel()

    monkeypatch.setattr(
        train_experiments.torch.nn.parallel,
        "DistributedDataParallel",
        lambda *args, **kwargs: pytest.fail("single-process path should not wrap DDP"),
    )

    wrapped = train_experiments._wrap_ddp_model(
        model,
        world_size=1,
        local_rank=0,
        is_main=True,
    )

    assert wrapped is model
    assert model.to_calls == []


def test_wrap_ddp_model_moves_and_wraps_ddp(monkeypatch):
    model = _MovableModel()
    ddp_calls = []

    class FakeDDP:
        def __init__(self, received_model, device_ids):
            self.received_model = received_model
            self.device_ids = device_ids
            ddp_calls.append((received_model, device_ids))

    monkeypatch.setattr(
        train_experiments.torch.nn.parallel,
        "DistributedDataParallel",
        FakeDDP,
    )

    wrapped = train_experiments._wrap_ddp_model(
        model,
        world_size=2,
        local_rank=3,
        is_main=True,
    )

    assert isinstance(wrapped, FakeDDP)
    assert wrapped.received_model is model
    assert wrapped.device_ids == [3]
    assert ddp_calls == [(model, [3])]
    assert [str(device) for device in model.to_calls] == ["cuda:3"]


def test_dump_init_gate_stats_diagnostic_calls_writer(monkeypatch):
    calls = []
    model = object()

    monkeypatch.setattr(
        train_experiments,
        "dump_init_gate_stats",
        lambda received_model, path: calls.append((received_model, path)),
    )

    train_experiments._dump_init_gate_stats_diagnostic(model, "/tmp/run")

    assert calls == [(model, "/tmp/run")]


def test_dump_init_gate_stats_diagnostic_logs_failure(monkeypatch, caplog):
    _enable_package_caplog(monkeypatch)

    def failing_dump(*args):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(train_experiments, "dump_init_gate_stats", failing_dump)

    with caplog.at_level("WARNING", logger=train_experiments.__name__):
        train_experiments._dump_init_gate_stats_diagnostic(object(), "/tmp/run")

    assert "init_gate_stats diagnostic failed: diagnostic failed" in caplog.text


def test_cleanup_training_runtime_finishes_wandb_and_distributed(monkeypatch):
    finish_calls = []
    destroy_calls = []

    monkeypatch.setattr(train_experiments.wandb, "run", object())
    monkeypatch.setattr(
        train_experiments.wandb,
        "finish",
        lambda: finish_calls.append("finish"),
    )
    monkeypatch.setattr(
        train_experiments.dist,
        "is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        train_experiments.dist,
        "destroy_process_group",
        lambda: destroy_calls.append("destroy"),
    )

    train_experiments._cleanup_training_runtime(
        SimpleNamespace(use_wandb=True),
        world_size=2,
    )

    assert finish_calls == ["finish"]
    assert destroy_calls == ["destroy"]


def test_cleanup_training_runtime_skips_disabled_paths(monkeypatch):
    monkeypatch.setattr(train_experiments.wandb, "run", None)
    monkeypatch.setattr(
        train_experiments.wandb,
        "finish",
        lambda: pytest.fail("wandb.finish should not run"),
    )
    monkeypatch.setattr(
        train_experiments.dist,
        "is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        train_experiments.dist,
        "destroy_process_group",
        lambda: pytest.fail("single-process run should not destroy process group"),
    )

    train_experiments._cleanup_training_runtime(
        SimpleNamespace(use_wandb=False),
        world_size=1,
    )


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
