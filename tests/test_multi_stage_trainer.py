import json
from types import SimpleNamespace

from dendritic_modeling.config.multi_stage_training import (
    MultiStageTrainingConfig,
    TrainingStageConfig,
)
from dendritic_modeling.training.strategies.multi_stage import (
    _build_stage_trainer_config,
    _concatenate_stage_performance_epochs,
    _epoch_files_by_local_epoch,
    _prepare_stage_analysis_root,
    _run_between_stage_analysis,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_epoch_files_by_local_epoch_sorts_numeric_epoch_files(tmp_path):
    epochs_dir = tmp_path / "epochs"
    epochs_dir.mkdir()
    _write_json(epochs_dir / "epoch10.json", {"epoch": 10})
    _write_json(epochs_dir / "epoch2.json", {"epoch": 2})
    _write_json(epochs_dir / "epoch1", {"epoch": 1})
    _write_json(epochs_dir / "epoch_bad.json", {"epoch": "bad"})
    _write_json(epochs_dir / "metrics.json", {"ignored": True})

    assert _epoch_files_by_local_epoch(str(epochs_dir)) == [
        (1, "epoch1"),
        (2, "epoch2.json"),
        (10, "epoch10.json"),
    ]


def test_concatenate_stage_performance_epochs_preserves_stage_and_epoch_order(
    tmp_path,
):
    main_epochs_dir = tmp_path / "performance" / "epochs"
    _write_json(main_epochs_dir / "epoch99.json", {"old": True})
    (main_epochs_dir / "notes.json").write_text("{}")

    stage1_epochs = tmp_path / "stage_1" / "performance" / "epochs"
    _write_json(stage1_epochs / "epoch10.json", {"stage": 1, "epoch": 10})
    _write_json(stage1_epochs / "epoch2.json", {"stage": 1, "epoch": 2})
    _write_json(stage1_epochs / "epoch3.json", {})
    (stage1_epochs / "epoch4.json").write_text("{not-json")
    _write_json(stage1_epochs / "epoch_bad.json", {"ignored": True})

    stage2_epochs = tmp_path / "stage_2" / "performance" / "epochs"
    _write_json(stage2_epochs / "epoch1.json", {"stage": 2, "epoch": 1})

    _concatenate_stage_performance_epochs(
        [TrainingStageConfig(), TrainingStageConfig(), TrainingStageConfig()],
        str(tmp_path),
    )

    assert not (main_epochs_dir / "epoch99.json").exists()
    assert (main_epochs_dir / "notes.json").exists()
    assert json.loads((main_epochs_dir / "epoch1.json").read_text()) == {
        "stage": 1,
        "epoch": 2,
    }
    assert json.loads((main_epochs_dir / "epoch2.json").read_text()) == {
        "stage": 1,
        "epoch": 10,
    }
    assert json.loads((main_epochs_dir / "epoch3.json").read_text()) == {
        "stage": 2,
        "epoch": 1,
    }
    assert not (main_epochs_dir / "epoch4.json").exists()


def test_prepare_stage_analysis_root_clears_epoch_files_only(tmp_path):
    stage_epochs = tmp_path / "stage_1" / "performance" / "epochs"
    _write_json(stage_epochs / "epoch1.json", {"old": True})
    _write_json(stage_epochs / "metrics.json", {"keep": True})
    analysis_manager = SimpleNamespace(save_root=str(tmp_path))

    _prepare_stage_analysis_root(analysis_manager, str(tmp_path), "stage_1")

    assert analysis_manager.save_root == str(tmp_path / "stage_1")
    assert not (stage_epochs / "epoch1.json").exists()
    assert (stage_epochs / "metrics.json").exists()


def test_run_between_stage_analysis_restores_save_root_after_success(tmp_path):
    calls = []

    class _AnalysisManager:
        def __init__(self):
            self.save_root = str(tmp_path)

        def run_analysis(self, *, filename, training):
            calls.append((self.save_root, filename, training))

    analysis_manager = _AnalysisManager()

    _run_between_stage_analysis(analysis_manager, "stage_1", str(tmp_path))

    assert calls == [(str(tmp_path / "stage_1"), "stage_1", False)]
    assert analysis_manager.save_root == str(tmp_path)


def test_build_stage_trainer_config_applies_stage_overrides_and_global_defaults():
    base_config = {
        "epochs": 5,
        "batch_size": 16,
        "loss_function": "cross_entropy",
    }
    multi_stage_config = MultiStageTrainingConfig(
        stages=[],
        global_batch_size=32,
        global_shuffle=False,
        global_grad_clip_value=0.25,
    )
    stage_config = TrainingStageConfig(
        epochs=2,
        batch_size=None,
        trainer_config={"loss_function": "mse", "extra": "value"},
        param_groups={"encoder": {"lr": 1e-3}},
        local_rule_config={"rule": "stdp"},
    )

    stage_trainer_config = _build_stage_trainer_config(
        base_config,
        multi_stage_config,
        stage_config,
    )

    assert stage_trainer_config == {
        "epochs": 2,
        "batch_size": 32,
        "loss_function": "mse",
        "extra": "value",
        "param_groups": {"encoder": {"lr": 1e-3}},
        "shuffle": False,
        "grad_clip_value": 0.25,
        "local_rule_config": {"rule": "stdp"},
    }
    assert base_config == {
        "epochs": 5,
        "batch_size": 16,
        "loss_function": "cross_entropy",
    }


def test_build_stage_trainer_config_keeps_existing_shuffle_and_grad_clip():
    base_config = {"shuffle": True, "grad_clip_value": 3.0}
    multi_stage_config = MultiStageTrainingConfig(
        stages=[],
        global_shuffle=False,
        global_grad_clip_value=0.25,
    )
    stage_config = TrainingStageConfig(batch_size=64)

    stage_trainer_config = _build_stage_trainer_config(
        base_config,
        multi_stage_config,
        stage_config,
    )

    assert stage_trainer_config["batch_size"] == 64
    assert stage_trainer_config["shuffle"] is True
    assert stage_trainer_config["grad_clip_value"] == 3.0
