import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dendritic_modeling.config.multi_stage_training import (
    MultiStageTrainingConfig,
    TrainingStageConfig,
)
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies.multi_stage import (
    STAGE_CHECKPOINT_NAME,
    MultiStageTrainer,
    _append_stage_result,
    _build_stage_optimizer,
    _build_stage_trainer,
    _build_stage_trainer_config,
    _concatenate_stage_performance_epochs,
    _epoch_files_by_local_epoch,
    _finalize_multi_stage_results,
    _initial_multi_stage_results,
    _iter_resettable_modules,
    _load_stage_checkpoint,
    _prepare_stage_analysis_root,
    _reset_module_weights,
    _run_between_stage_analysis,
    _stage_checkpoint_optimizer_state,
    _stage_directory,
    _update_best_stage,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class _ParamGroupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(2, 2)
        self.decoder = nn.Linear(2, 1)
        self.seen_param_groups_config = None

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_param_groups(self, param_groups_config):
        self.seen_param_groups_config = param_groups_config
        return [
            {"params": list(self.encoder.parameters()), "lr": param_groups_config.lr},
            {
                "params": list(self.decoder.parameters()),
                "lr": param_groups_config.decoder_lr,
            },
        ]

    def decay_weights(self, weight_decay, weight_boosting=False):
        self.last_decay = (weight_decay, weight_boosting)

    def apply_rewiring(self):
        self.rewired = True


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
        ["stage_1", "stage_2", "stage_3"],
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


def test_run_between_stage_analysis_restores_save_root_after_failure(tmp_path):
    calls = []

    class _AnalysisManager:
        def __init__(self):
            self.save_root = str(tmp_path)

        def run_analysis(self, *, filename, training):
            calls.append((self.save_root, filename, training))
            raise RuntimeError("analysis failed")

    analysis_manager = _AnalysisManager()

    try:
        _run_between_stage_analysis(analysis_manager, "stage_1", str(tmp_path))
    except RuntimeError as exc:
        assert str(exc) == "analysis failed"
    else:
        raise AssertionError("Expected analysis failure")

    assert calls == [(str(tmp_path / "stage_1"), "stage_1", False)]
    assert analysis_manager.save_root == str(tmp_path)


def test_build_stage_trainer_config_applies_stage_overrides():
    base_config = {
        "epochs": 5,
        "batch_size": 16,
        "shuffle": True,
        "grad_clip_value": 3.0,
        "loss_function": "cross_entropy",
    }
    multi_stage_config = MultiStageTrainingConfig(stages=[])
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
        "batch_size": 16,
        "shuffle": True,
        "grad_clip_value": 3.0,
        "loss_function": "mse",
        "extra": "value",
        "param_groups": {"encoder": {"lr": 1e-3}},
        "local_rule_config": {"rule": "stdp"},
    }
    assert base_config["loss_function"] == "cross_entropy"


@pytest.mark.parametrize("key", ["epochs", "batch_size", "param_groups"])
def test_stage_level_keys_inside_trainer_config_raise(key):
    with pytest.raises(ValueError, match="write them at stage level"):
        TrainingStageConfig(trainer_config={key: 1})


def test_stage_names_must_be_unique_directory_names():
    with pytest.raises(ValueError, match="more than once"):
        MultiStageTrainingConfig(
            stages=[TrainingStageConfig(stage_name="a"), {"stage_name": "a"}]
        )
    with pytest.raises(ValueError, match="plain file name"):
        TrainingStageConfig(stage_name="../escape")


def test_the_former_best_stage_flag_loads_as_final_state():
    from dendritic_modeling.config import config_from_mapping

    def final_state(**block):
        stages = [{"learning_strategy": "standard", "epochs": 1}]
        config = config_from_mapping(
            {
                "training": {
                    "main": {"strategies": {"multi_stage": {**block, "stages": stages}}}
                }
            }
        )
        return config.training.main.strategies.multi_stage.final_state

    assert final_state() == "last_stage"
    assert final_state(use_best_from_each_stage=True) == "best_stage"
    assert final_state(use_best_from_each_stage=False) == "last_stage"
    with pytest.raises(ValueError, match="final_state='best' is not supported"):
        final_state(final_state="best")


def test_build_stage_optimizer_continues_base_optimizer_without_reset():
    model = nn.Linear(2, 1)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    stage_optimizer = _build_stage_optimizer(
        model,
        base_optimizer,
        TrainingStageConfig(reset_optimizer=False),
        stage_idx=2,
    )

    # The optimizer state continues; the wrapper runs sparse maintenance.
    assert isinstance(stage_optimizer, CustomWeightDecayOptimizer)
    assert stage_optimizer.optimizer is base_optimizer
    assert stage_optimizer.weight_decay == 0.0


def test_build_stage_optimizer_reuses_matching_wrapper_without_reset():
    model = nn.Linear(2, 1)
    base_optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.02),
        weight_decay=0.2,
        weight_boosting=True,
    )

    stage_optimizer = _build_stage_optimizer(
        model,
        base_optimizer,
        TrainingStageConfig(reset_optimizer=False),
        stage_idx=0,
        stage_trainer_config={"synapse_decay_rate": 0.2, "weight_boosting": True},
    )

    assert stage_optimizer is base_optimizer


def test_build_stage_optimizer_reset_rebuilds_from_param_groups():
    model = nn.Linear(2, 1)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.02, eps=1e-6)

    stage_optimizer = _build_stage_optimizer(
        model,
        base_optimizer,
        TrainingStageConfig(reset_optimizer=True),
        stage_idx=2,
        stage_trainer_config={"param_groups": {"lr": 0.05}},
    )

    assert isinstance(stage_optimizer, CustomWeightDecayOptimizer)
    assert isinstance(stage_optimizer.optimizer, torch.optim.Adam)
    assert stage_optimizer.optimizer is not base_optimizer
    assert stage_optimizer.optimizer.param_groups[0]["lr"] == 0.05
    assert stage_optimizer.optimizer.defaults["eps"] == 1e-6


def test_build_stage_optimizer_wraps_reset_optimizer_when_base_uses_custom_decay():
    model = _ParamGroupModel()
    base_optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.5),
        weight_decay=0.2,
    )

    stage_optimizer = _build_stage_optimizer(
        model,
        base_optimizer,
        TrainingStageConfig(reset_optimizer=True),
        stage_idx=2,
    )

    assert isinstance(stage_optimizer, CustomWeightDecayOptimizer)
    assert isinstance(stage_optimizer.optimizer, torch.optim.SGD)
    assert stage_optimizer.weight_decay == 0.2
    assert stage_optimizer.optimizer.defaults["momentum"] == 0.5
    # Rebuilt from model.get_param_groups(ParamGroupsConfig()) defaults.
    assert [group["lr"] for group in stage_optimizer.optimizer.param_groups] == [
        0.01,
        0.001,
    ]


def test_build_stage_optimizer_uses_model_param_groups_when_configured():
    model = _ParamGroupModel()
    base_optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    stage_optimizer = _build_stage_optimizer(
        model,
        base_optimizer,
        TrainingStageConfig(param_groups={"lr": 0.03, "decoder_lr": 0.004}),
        stage_idx=2,
    )

    assert isinstance(stage_optimizer, CustomWeightDecayOptimizer)
    assert isinstance(stage_optimizer.optimizer, torch.optim.Adam)
    assert model.seen_param_groups_config.lr == 0.03
    assert model.seen_param_groups_config.decoder_lr == 0.004
    assert [group["lr"] for group in stage_optimizer.param_groups] == [0.03, 0.004]


def test_load_stage_checkpoint_accepts_wrapped_model_state_dict(tmp_path):
    source_model = nn.Linear(2, 1)
    target_model = nn.Linear(2, 1)
    with torch.no_grad():
        source_model.weight.fill_(1.5)
        source_model.bias.fill_(-0.25)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": source_model.state_dict()}, checkpoint_path)

    _load_stage_checkpoint(target_model, str(checkpoint_path))

    for source_param, target_param in zip(
        source_model.parameters(),
        target_model.parameters(),
    ):
        assert torch.equal(source_param, target_param)


def test_load_stage_checkpoint_accepts_plain_state_dict(tmp_path):
    source_model = nn.Linear(2, 1)
    target_model = nn.Linear(2, 1)
    with torch.no_grad():
        source_model.weight.fill_(2.5)
        source_model.bias.fill_(0.75)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(source_model.state_dict(), checkpoint_path)

    _load_stage_checkpoint(target_model, str(checkpoint_path))

    for source_param, target_param in zip(
        source_model.parameters(),
        target_model.parameters(),
    ):
        assert torch.equal(source_param, target_param)


def test_build_stage_trainer_delegates_to_factory(monkeypatch):
    calls = []
    stage_optimizer = object()
    stage_trainer_config = {"epochs": 3}
    analysis_manager = object()
    expected_trainer = object()

    def _fake_get_trainer(**kwargs):
        calls.append(kwargs)
        return expected_trainer

    monkeypatch.setattr(
        "dendritic_modeling.training.factory.get_trainer",
        _fake_get_trainer,
    )

    stage_trainer = _build_stage_trainer(
        TrainingStageConfig(learning_strategy="local_ca"),
        stage_optimizer,
        stage_trainer_config,
        analysis_manager,
    )

    assert stage_trainer is expected_trainer
    assert calls == [
        {
            "strategy": "local_ca",
            "optimizer": stage_optimizer,
            "trainer_config_dict": stage_trainer_config,
            "analysis_manager": analysis_manager,
        }
    ]


def test_execute_stage_restores_analysis_root_after_failure(tmp_path, monkeypatch):
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    analysis_manager = SimpleNamespace(save_root=str(tmp_path))
    trainer = MultiStageTrainer(
        multi_stage_config=MultiStageTrainingConfig(stages=[TrainingStageConfig()]),
        base_optimizer=optimizer,
        base_trainer_config={"epochs": 1},
        analysis_manager=analysis_manager,
    )

    class _FailingTrainer:
        def train(self, model, train_data, valid_data):
            raise RuntimeError("stage failed")

    monkeypatch.setattr(
        "dendritic_modeling.training.factory.get_trainer",
        lambda **kwargs: _FailingTrainer(),
    )

    try:
        trainer._execute_stage(
            model,
            train_data=[],
            valid_data=[],
            stage_config=TrainingStageConfig(),
            stage_idx=0,
            stage_name="stage_1",
            original_save_root=str(tmp_path),
        )
    except RuntimeError as exc:
        assert str(exc) == "stage failed"
    else:
        raise AssertionError("Expected stage failure")

    assert analysis_manager.save_root == str(tmp_path)


def test_each_stage_gets_its_own_created_directory(tmp_path):
    directory = _stage_directory(str(tmp_path), "pretrain")

    assert directory == str(tmp_path / "pretrain")
    assert (tmp_path / "pretrain").is_dir()
    assert STAGE_CHECKPOINT_NAME == "stage_checkpoint.pt"


def test_initial_multi_stage_results_shape():
    results = _initial_multi_stage_results("last_stage")

    assert results == {
        "stages": [],
        "final_state": "last_stage",
        "final_stage": None,
        "best_loss": float("inf"),
        "best_epoch": None,
        "final_state_dict": None,
    }


def test_append_stage_result_preserves_legacy_stage_record_shape():
    all_results = _initial_multi_stage_results("last_stage")
    stage_results = {"best_loss": 0.25}

    _append_stage_result(all_results, "stage_2", 1, stage_results)

    assert all_results["stages"] == [
        {
            "stage_name": "stage_2",
            "stage_idx": 1,
            "results": stage_results,
        }
    ]


def test_update_best_stage_uses_strictly_lower_best_loss():
    previous_state = {"weight": torch.tensor([1.0])}
    candidate_state = {"weight": torch.tensor([2.0])}

    best_loss, best_state, best_stage = _update_best_stage(
        {"best_loss": 0.3, "best_state_dict": candidate_state},
        best_overall_loss=0.5,
        best_overall_state_dict=previous_state,
        best_stage_name="stage_1",
        stage_name="stage_2",
    )

    assert best_loss == 0.3
    assert best_state is candidate_state
    assert best_stage == "stage_2"

    best_loss, best_state, best_stage = _update_best_stage(
        {"best_loss": 0.3, "best_state_dict": {"weight": torch.tensor([3.0])}},
        best_overall_loss=0.3,
        best_overall_state_dict=candidate_state,
        best_stage_name="stage_2",
        stage_name="stage_3",
    )

    assert best_loss == 0.3
    assert best_state is candidate_state
    assert best_stage == "stage_2"


def test_update_best_stage_ignores_results_without_best_loss():
    previous_state = {"weight": torch.tensor([1.0])}

    best_loss, best_state, best_stage = _update_best_stage(
        {"loss": 0.1},
        best_overall_loss=0.5,
        best_overall_state_dict=previous_state,
        best_stage_name="stage_1",
        stage_name="stage_2",
    )

    assert best_loss == 0.5
    assert best_state is previous_state
    assert best_stage == "stage_1"


def test_finalize_records_the_final_stage_and_a_copy_of_its_weights():
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(-0.1)
    all_results = _initial_multi_stage_results("last_stage")

    _finalize_multi_stage_results(
        all_results, model, "stage_2", {"best_loss": 0.2, "best_epoch": 3}
    )
    with torch.no_grad():
        model.weight.fill_(9.0)  # later changes do not reach the recorded copy

    assert all_results["final_stage"] == "stage_2"
    assert all_results["best_loss"] == 0.2
    assert all_results["best_epoch"] == 3
    assert torch.equal(
        all_results["final_state_dict"]["weight"], torch.full((1, 2), 0.5)
    )
    assert all_results["final_state_dict"]["bias"].device.type == "cpu"


def test_resettable_modules_preserve_order_and_reset_rule():
    calls = []

    class ResetAware(nn.Module):
        def __init__(self, label):
            super().__init__()
            self.label = label
            self.weight = nn.Parameter(torch.ones(1))

        def reset_parameters(self):
            calls.append(self.label)
            with torch.no_grad():
                self.weight.zero_()

    first = ResetAware("first")
    second = ResetAware("second")
    model = nn.Sequential(first, nn.Identity(), nn.Sequential(second))

    resettable = list(_iter_resettable_modules(model))
    for module in resettable:
        _reset_module_weights(module)

    assert resettable == [first, second]
    assert calls == ["first", "second"]
    assert first.weight.item() == 0.0
    assert second.weight.item() == 0.0


def test_stage_checkpoint_optimizer_state_prefers_trainer_optimizer():
    model = nn.Linear(2, 1)
    stage_optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    trainer_optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    stage_trainer = SimpleNamespace(optimizer=trainer_optimizer)

    state = _stage_checkpoint_optimizer_state(stage_trainer, stage_optimizer)

    assert state["param_groups"][0]["lr"] == 0.03


def test_stage_checkpoint_optimizer_state_unwraps_custom_optimizer():
    model = _ParamGroupModel()
    wrapped_optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.04),
        weight_decay=0.1,
    )
    stage_trainer = SimpleNamespace()

    state = _stage_checkpoint_optimizer_state(stage_trainer, wrapped_optimizer)

    assert state["param_groups"][0]["lr"] == 0.04
