"""Tests for train_experiments analysis orchestration helpers."""

from __future__ import annotations

from types import SimpleNamespace

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
