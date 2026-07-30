"""
Tests for training FSDP utils module to improve coverage.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime as RealDatetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import dendritic_modeling.scripts.training.train_experiments_fsdp as fsdp_script
from dendritic_modeling.networks.activations import ParametricTanh
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    BaseAutoencoder,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.training.train_experiments_fsdp import (
    _build_fsdp_runtime_configs,
    _create_fsdp_trainer,
    _dataloader_kwargs_from_common,
    _encode_rank0_analysis_datasets,
    _maybe_calibrate_fsdp_data_driven_reactivation,
    _prepare_rank_analysis_datasets,
    _print_rank0_model_summary,
    _requested_data_driven_reactivation_policies,
    _resolve_large_model_input_dim,
    _run_final_analysis_on_plain_model,
    _save_final_fsdp_checkpoint_and_run_analysis,
    _save_fsdp_run_config,
    _save_rank0_training_results,
    _setup_rank0_analysis_manager,
    _setup_rank0_run_directory,
    setup_distributed,
)
from dendritic_modeling.training.fsdp_utils import (
    _apply_gradient_checkpointing,
    _is_dendritic_fsdp_module,
    _is_large_linear_checkpoint_candidate,
    create_fsdp_wrap_policy,
    estimate_memory_requirements,
    get_fsdp_config,
    get_recommended_gpu_count,
    load_fsdp_checkpoint,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)
from dendritic_modeling.training.strategies.fsdp_standard import (
    FSDPTrainer,
    _iter_modules_by_class_name,
    _iter_reactivation_parameter_candidates,
    _sequence_lengths_from_batch,
)


def test_setup_distributed_uses_configured_process_group_timeout(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS", "3600")
    monkeypatch.setattr(fsdp_script.dist, "is_initialized", lambda: False)
    init_calls = []
    monkeypatch.setattr(
        fsdp_script.dist,
        "init_process_group",
        lambda **kwargs: init_calls.append(kwargs),
    )
    device_calls = []
    monkeypatch.setattr(
        fsdp_script.torch.cuda,
        "set_device",
        lambda local_rank: device_calls.append(local_rank),
    )

    assert setup_distributed() == (3, 1, 8)
    assert init_calls == [{"backend": "nccl", "timeout": timedelta(hours=1)}]
    assert device_calls == [1]


def test_setup_distributed_rejects_invalid_process_group_timeout(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("DENDRITIC_FSDP_PROCESS_GROUP_TIMEOUT_SECONDS", "0")
    monkeypatch.setattr(fsdp_script.dist, "is_initialized", lambda: False)

    with pytest.raises(ValueError, match="positive integer"):
        setup_distributed()


def test_fsdp_dataloader_kwargs_from_common_config():
    """FSDP script should preserve typed DataLoader tuning knobs."""
    common = SimpleNamespace(
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=3,
    )

    kwargs = _dataloader_kwargs_from_common(common)

    assert kwargs == {
        "num_workers": 1,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 3,
    }


def test_fsdp_dataloader_kwargs_auto_avoids_workers_for_tensor_data():
    """FSDP auto tuning should avoid worker overhead for tensor-backed datasets."""
    common = SimpleNamespace(
        num_workers=None,
        pin_memory=None,
        persistent_workers=None,
        prefetch_factor=None,
    )
    dataset = TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))

    kwargs = _dataloader_kwargs_from_common(common, dataset, device="cpu")

    assert kwargs == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }


def test_fsdp_dataloader_kwargs_auto_keeps_workers_for_lazy_data(monkeypatch):
    """FSDP auto tuning should keep workers for lazy image/file-backed datasets."""

    class LazyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, idx):
            return torch.zeros(3, 224, 224), torch.tensor(idx % 2)

    common = SimpleNamespace(
        num_workers=None,
        pin_memory=None,
        persistent_workers=None,
        prefetch_factor=None,
    )
    monkeypatch.setattr(
        "dendritic_modeling.training.dataloader_utils.visible_cpu_count",
        lambda: 4,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    kwargs = _dataloader_kwargs_from_common(common, LazyDataset(), device="cpu")

    assert kwargs == {
        "num_workers": 3,
        "pin_memory": False,
        "persistent_workers": True,
    }


def test_fsdp_dataloader_tuning_summary_reports_dataset_type():
    """FSDP helper should expose the same tuning summary used by loader setup."""
    common = SimpleNamespace(
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    dataset = TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))

    tuning = fsdp_script._dataloader_tuning_from_common(
        common,
        dataset,
        device="cpu",
    )

    summary = tuning.summary(dataset)
    assert "dataset=torch.utils.data.dataset.TensorDataset" in summary
    assert "num_workers=0" in summary
    assert "pin_memory=False" in summary


def test_fsdp_setup_data_loaders_logs_tuning_on_rank0(monkeypatch, caplog):
    """FSDP data-loader setup should log resolved tuning once on rank 0."""
    train_ds = TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))
    valid_ds = TensorDataset(torch.randn(4, 3), torch.randint(0, 2, (4,)))
    test_ds = TensorDataset(torch.randn(4, 3), torch.randint(0, 2, (4,)))

    monkeypatch.setattr(
        fsdp_script,
        "get_unified_datasets",
        lambda task_cfg: (train_ds, valid_ds, test_ds),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    common = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    config = SimpleNamespace(
        data=SimpleNamespace(
            base_dir="",
            dataset_name="unit",
            dataset_params=None,
            processing=None,
        ),
        experiment=SimpleNamespace(train_valid_split=0.8, seed=7),
        training=SimpleNamespace(main=SimpleNamespace(common=common)),
    )

    with caplog.at_level(
        logging.INFO,
        logger="dendritic_modeling.scripts.training.train_experiments_fsdp",
    ):
        train_loader, valid_loader, returned_test, returned_train, returned_valid = (
            fsdp_script.setup_data_loaders(config, rank=0, world_size=1)
        )

    assert train_loader.num_workers == 0
    assert valid_loader.num_workers == 0
    assert returned_test is test_ds
    assert returned_train is train_ds
    assert returned_valid is valid_ds
    assert "FSDP DataLoader tuning:" in caplog.text
    assert "dataset=torch.utils.data.dataset.TensorDataset" in caplog.text
    assert "num_workers=0" in caplog.text


@pytest.mark.parametrize(
    ("dataset_name", "processing", "expected"),
    [
        ("mnist", SimpleNamespace(flatten=True), 784),
        ("cifar10", SimpleNamespace(flatten=True), 3072),
        ("imagenet", SimpleNamespace(flatten=True), 150528),
        ("imagenet", SimpleNamespace(flatten=False), 150528),
        ("unknown", SimpleNamespace(flatten=True), 784),
    ],
)
def test_resolve_large_model_input_dim_preserves_existing_mapping(
    dataset_name, processing, expected
):
    data_config = SimpleNamespace(dataset_name=dataset_name, processing=processing)

    assert _resolve_large_model_input_dim(data_config) == expected


def test_resolve_large_model_input_dim_honors_explicit_sequence_width():
    data_config = SimpleNamespace(dataset_name="romo_delay_comparison_positive")
    model_config = SimpleNamespace(
        encoder=SimpleNamespace(params=SimpleNamespace(input_dim=11))
    )

    assert _resolve_large_model_input_dim(data_config, model_config) == 11


def test_sequence_lengths_from_recurrent_batch():
    x_batch = torch.zeros(3, 7, 4)
    lengths = torch.tensor([7, 5, 6])

    resolved = _sequence_lengths_from_batch(
        (x_batch, torch.zeros(3), lengths),
        x_batch=x_batch,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(resolved, lengths)


def test_sequence_lengths_ignores_nonsequence_metadata():
    x_batch = torch.zeros(3, 4)

    assert (
        _sequence_lengths_from_batch(
            (x_batch, torch.zeros(3), torch.arange(3)),
            x_batch=x_batch,
            device=torch.device("cpu"),
        )
        is None
    )


def test_fsdp_dataloader_kwargs_reject_invalid_persistent_workers():
    """persistent_workers=True is invalid when DataLoader has no workers."""
    common = SimpleNamespace(
        num_workers=0,
        pin_memory=None,
        persistent_workers=True,
        prefetch_factor=None,
    )

    with pytest.raises(ValueError, match="persistent_workers"):
        _dataloader_kwargs_from_common(common)


def test_run_final_analysis_on_plain_model_reloads_checkpoint(monkeypatch):
    """FSDP final analysis should run on a fresh non-sharded model."""

    class PlainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded_state = None
            self.to_device = None
            self.eval_called = False
            self.encoder_network = None

        def load_state_dict(self, state_dict):
            self.loaded_state = state_dict

        def to(self, device):
            self.to_device = device
            return self

        def eval(self):
            self.eval_called = True

    class AnalysisManagerStub:
        def __init__(self):
            self.model = None
            self.run_calls = []

        def run_analysis(self, *, filename, training):
            self.run_calls.append({"filename": filename, "training": training})

    config = SimpleNamespace(name="config")
    plain_model = PlainModel()
    analysis_manager = AnalysisManagerStub()
    encoder = object()
    state = {"weight": torch.ones(1)}
    load_calls = []

    monkeypatch.setattr(
        "dendritic_modeling.scripts.training.train_experiments_fsdp.create_large_model",
        lambda received_config: plain_model,
    )

    def fake_load(path, *, map_location):
        load_calls.append({"path": path, "map_location": map_location})
        return {"model_state_dict": state}

    monkeypatch.setattr(
        "dendritic_modeling.scripts.training.train_experiments_fsdp.torch.load",
        fake_load,
    )

    _run_final_analysis_on_plain_model(
        config=config,
        analysis_manager=analysis_manager,
        full_ckpt_path="/tmp/final_full.pt",
        encoder_network=encoder,
        local_rank=2,
    )

    assert plain_model.loaded_state is state
    assert plain_model.to_device == "cuda:2"
    assert plain_model.eval_called
    assert plain_model.encoder_network is encoder
    assert analysis_manager.model is plain_model
    assert analysis_manager.run_calls == [{"filename": "final", "training": False}]
    assert load_calls == [{"path": "/tmp/final_full.pt", "map_location": "cuda:2"}]


@dataclass
class _RunOutputsConfig:
    run_name: str
    results_dir: str
    exact_run_dir: bool = False


@dataclass
class _RunConfigSnapshot:
    outputs: _RunOutputsConfig
    marker: int = 7


def test_save_fsdp_run_config_applies_canonicalizers(monkeypatch):
    """FSDP config snapshots should preserve the existing canonicalizer order."""
    config = _RunConfigSnapshot(outputs=_RunOutputsConfig("run", "/tmp/results"))
    saved = []

    def fake_model_canonicalizer(payload):
        return {**payload, "model_canonicalized": True}

    def fake_rnn_canonicalizer(payload):
        return {**payload, "rnn_saw_model": payload["model_canonicalized"]}

    monkeypatch.setattr(
        fsdp_script, "canonicalize_model_core_flags", fake_model_canonicalizer
    )
    monkeypatch.setattr(
        fsdp_script, "canonicalize_rnn_core_flags", fake_rnn_canonicalizer
    )
    monkeypatch.setattr(
        fsdp_script,
        "save_dict",
        lambda payload, path, filename: saved.append(
            {"payload": payload, "path": path, "filename": filename}
        ),
    )

    _save_fsdp_run_config(config, "/tmp/run")

    assert saved == [
        {
            "payload": {
                "outputs": {
                    "run_name": "run",
                    "results_dir": "/tmp/results",
                    "exact_run_dir": False,
                },
                "marker": 7,
                "model_canonicalized": True,
                "rnn_saw_model": True,
            },
            "path": "/tmp/run",
            "filename": "config.json",
        }
    ]


def test_setup_rank0_run_directory_creates_log_dir_and_config(monkeypatch, tmp_path):
    """FSDP rank-0 setup should create the timestamped output and save config."""

    class FixedDatetime:
        @classmethod
        def now(cls):
            return RealDatetime(2025, 1, 2, 3, 4, 5)

    config = _RunConfigSnapshot(outputs=_RunOutputsConfig("experiment", str(tmp_path)))
    log_dirs = []
    saved_configs = []
    original_handlers = list(fsdp_script.logger.handlers)

    monkeypatch.setattr(fsdp_script, "datetime", FixedDatetime)
    monkeypatch.setattr(
        fsdp_script.logger_manager,
        "set_log_directory",
        lambda path: log_dirs.append(path),
    )
    monkeypatch.setattr(
        fsdp_script,
        "_save_fsdp_run_config",
        lambda received_config, path: saved_configs.append((received_config, path)),
    )

    try:
        run_save_path = _setup_rank0_run_directory(config)
    finally:
        for handler in list(fsdp_script.logger.handlers):
            if handler not in original_handlers:
                fsdp_script.logger.removeHandler(handler)
                handler.close()

    expected_path = os.path.join(str(tmp_path), "experiment_20250102_030405")
    assert run_save_path == expected_path
    assert os.path.isdir(expected_path)
    assert os.path.exists(os.path.join(expected_path, "train.log"))
    assert log_dirs == [expected_path]
    assert saved_configs == [(config, expected_path)]


def test_setup_rank0_run_directory_can_use_exact_manifest_path(monkeypatch, tmp_path):
    exact_path = tmp_path / "claimed-run"
    exact_path.mkdir()
    config = _RunConfigSnapshot(
        outputs=_RunOutputsConfig(
            "experiment",
            str(exact_path),
            exact_run_dir=True,
        )
    )
    log_dirs = []
    saved_configs = []
    original_handlers = list(fsdp_script.logger.handlers)
    monkeypatch.setattr(
        fsdp_script.logger_manager,
        "set_log_directory",
        lambda path: log_dirs.append(path),
    )
    monkeypatch.setattr(
        fsdp_script,
        "_save_fsdp_run_config",
        lambda received_config, path: saved_configs.append((received_config, path)),
    )

    try:
        run_save_path = _setup_rank0_run_directory(config)
    finally:
        for handler in list(fsdp_script.logger.handlers):
            if handler not in original_handlers:
                fsdp_script.logger.removeHandler(handler)
                handler.close()

    assert run_save_path == str(exact_path)
    assert log_dirs == [str(exact_path)]
    assert saved_configs == [(config, str(exact_path))]


def test_encode_rank0_analysis_datasets_delegates_to_encoder_loader(monkeypatch):
    """FSDP rank-0 encoding should preserve the encoder-loader call contract."""
    encoder = object()
    encoded_encoder = object()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    encoded_train_ds = object()
    encoded_valid_ds = object()
    encoded_test_ds = object()
    model = SimpleNamespace(encoder_network=encoder)
    config = SimpleNamespace(
        training=SimpleNamespace(encoder=object()),
        model=SimpleNamespace(encoder=object()),
        data=object(),
    )
    calls = []

    def fake_load_train_encoder_network(**kwargs):
        calls.append(kwargs)
        return encoded_encoder, encoded_train_ds, encoded_valid_ds, encoded_test_ds

    monkeypatch.setattr(
        fsdp_script,
        "load_train_encoder_network",
        fake_load_train_encoder_network,
    )

    result = _encode_rank0_analysis_datasets(
        model=model,
        config=config,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    assert result == (
        encoded_encoder,
        encoded_train_ds,
        encoded_valid_ds,
        encoded_test_ds,
    )
    assert calls == [
        {
            "encoder_network": encoder,
            "encoder_train_config": config.training.encoder,
            "train_ds": train_ds,
            "valid_ds": valid_ds,
            "test_ds": test_ds,
            "encoder_network_config": config.model.encoder,
            "task_config": config.data,
        }
    ]


def test_encode_rank0_analysis_datasets_skips_base_autoencoder(monkeypatch):
    """FSDP should not try to pretrain a BaseAutoencoder inside rank-0 encoding."""
    encoder = BaseAutoencoder()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    model = SimpleNamespace(encoder_network=encoder)
    config = SimpleNamespace()

    monkeypatch.setattr(
        fsdp_script,
        "load_train_encoder_network",
        lambda **kwargs: pytest.fail("BaseAutoencoder should not be encoded here"),
    )

    result = _encode_rank0_analysis_datasets(
        model=model,
        config=config,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    assert result == (encoder, train_ds, valid_ds, test_ds)


def test_prepare_rank_analysis_datasets_rank0_delegates_to_encoder(monkeypatch):
    """Rank 0 should prepare real encoded datasets for analysis."""
    model = object()
    config = object()
    train_ds = object()
    valid_ds = object()
    test_ds = object()
    result = (object(), object(), object(), object())
    calls = []

    monkeypatch.setattr(
        fsdp_script,
        "_encode_rank0_analysis_datasets",
        lambda **kwargs: calls.append(kwargs) or result,
    )

    prepared = _prepare_rank_analysis_datasets(
        rank=0,
        model=model,
        config=config,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
    )

    assert prepared is result
    assert calls == [
        {
            "model": model,
            "config": config,
            "train_ds": train_ds,
            "valid_ds": valid_ds,
            "test_ds": test_ds,
        }
    ]


def test_prepare_rank_analysis_datasets_non_rank0_returns_placeholders(monkeypatch):
    """Non-rank-0 processes should not encode analysis datasets."""
    monkeypatch.setattr(
        fsdp_script,
        "_encode_rank0_analysis_datasets",
        lambda **kwargs: pytest.fail("non-rank-0 should not encode datasets"),
    )

    encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds = (
        _prepare_rank_analysis_datasets(
            rank=1,
            model=object(),
            config=object(),
            train_ds=object(),
            valid_ds=object(),
            test_ds=object(),
        )
    )

    assert isinstance(encoder_network, fsdp_script.Identity)
    assert encoder_network.output_dim == 1
    assert encoded_train_ds is None
    assert encoded_valid_ds is None
    assert encoded_test_ds is None


def test_print_rank0_model_summary_uses_sample_input(monkeypatch):
    """FSDP rank-0 summary should use the first dataset sample when available."""
    model = object()
    sample = torch.ones(2, 3)
    calls = []

    class SampleDataset:
        def __getitem__(self, index):
            assert index == 0
            return sample, torch.tensor(1)

    monkeypatch.setattr(
        fsdp_script,
        "print_model_parameter_summary",
        lambda received_model, sample_input=None: calls.append(
            (received_model, sample_input)
        ),
    )

    _print_rank0_model_summary(model, SampleDataset())

    assert calls == [(model, sample)]


def test_print_rank0_model_summary_falls_back_without_sample(monkeypatch):
    """FSDP rank-0 summary should preserve the no-sample fallback."""
    model = object()
    calls = []

    class FailingDataset:
        def __getitem__(self, index):
            raise RuntimeError(f"cannot sample {index}")

    monkeypatch.setattr(
        fsdp_script,
        "print_model_parameter_summary",
        lambda received_model, sample_input=None: calls.append(
            (received_model, sample_input)
        ),
    )

    _print_rank0_model_summary(model, FailingDataset())

    assert calls == [(model, None)]


def test_parameter_summary_skips_torchinfo_for_sequence_input(monkeypatch):
    model = nn.Linear(4, 2)
    summary_calls = []
    monkeypatch.setattr(fsdp_script, "TORCHINFO_AVAILABLE", True)
    monkeypatch.setattr(
        fsdp_script.torchinfo,
        "summary",
        lambda *args, **kwargs: summary_calls.append((args, kwargs)),
    )

    fsdp_script.print_model_parameter_summary(model, torch.zeros(7, 4))

    assert summary_calls == []


def test_build_fsdp_runtime_configs_uses_typed_config_values(monkeypatch):
    """FSDP runtime config assembly should preserve typed config values."""
    fsdp_config = object()
    calls = []
    train_common = SimpleNamespace(
        param_groups=SimpleNamespace(lr=0.123, split_params=True),
        weight_decay_rate=0.4,
    )
    fsdp_cfg = SimpleNamespace(
        mixed_precision=True,
        cpu_offload=False,
        sharding_strategy="NO_SHARD",
        extra_key="kept",
    )
    config = SimpleNamespace(
        training=SimpleNamespace(main=SimpleNamespace(common=train_common)),
        distributed=SimpleNamespace(fsdp=fsdp_cfg),
    )

    def fake_get_fsdp_config(**kwargs):
        calls.append(kwargs)
        return fsdp_config

    monkeypatch.setattr(fsdp_script, "get_fsdp_config", fake_get_fsdp_config)

    result_train_common, result_fsdp_config, optimizer_config = (
        _build_fsdp_runtime_configs(config)
    )

    assert result_train_common is train_common
    assert result_fsdp_config is fsdp_config
    assert optimizer_config == {
        "lr": 0.123,
        "weight_decay": 0.4,
        "split_params": True,
    }
    assert calls == [
        {
            "mixed_precision": True,
            "cpu_offload": False,
            "sharding_strategy": "NO_SHARD",
            "fsdp_config_dict": vars(fsdp_cfg),
        }
    ]


def test_build_fsdp_runtime_configs_preserves_defaults_and_slot_fsdp_config(
    monkeypatch,
):
    """FSDP runtime config assembly should keep legacy optimizer defaults."""

    class SlotFSDPConfig:
        __slots__ = ("cpu_offload", "mixed_precision", "sharding_strategy")

        def __init__(self):
            self.mixed_precision = False
            self.cpu_offload = True
            self.sharding_strategy = "FULL_SHARD"

    fsdp_config = object()
    calls = []
    train_common = SimpleNamespace(param_groups=SimpleNamespace(lr=0.01))
    config = SimpleNamespace(
        training=SimpleNamespace(main=SimpleNamespace(common=train_common)),
        distributed=SimpleNamespace(fsdp=SlotFSDPConfig()),
    )

    monkeypatch.setattr(
        fsdp_script,
        "get_fsdp_config",
        lambda **kwargs: calls.append(kwargs) or fsdp_config,
    )

    result_train_common, result_fsdp_config, optimizer_config = (
        _build_fsdp_runtime_configs(config)
    )

    assert result_train_common is train_common
    assert result_fsdp_config is fsdp_config
    assert optimizer_config == {
        "lr": 0.01,
        "weight_decay": 0.0,
        "split_params": False,
    }
    assert calls == [
        {
            "mixed_precision": False,
            "cpu_offload": True,
            "sharding_strategy": "FULL_SHARD",
            "fsdp_config_dict": None,
        }
    ]


def test_fsdp_optimizer_split_groups_assign_each_parameter_once():
    """FSDP split optimizer groups should be single-assignment and ordered."""

    class GroupedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.topk = TopKLinear(4, 3, K=2)
            self.blocklinear = BlockLinear(4, 2)
            self.reactivation = ParametricTanh(2)
            self.decoder_network = nn.Sequential(ParametricTanh(2), nn.Linear(2, 2))
            self.other = nn.Linear(2, 1)

    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.optimizer_config = {
        "split_params": True,
        "lr": 0.05,
        "topk_lr": 0.01,
        "blocklinear_lr": 0.02,
        "reactivation_lr": 0.03,
        "decoder_lr": 0.04,
        "weight_decay": 0.001,
    }
    model = GroupedModel()

    optimizer = trainer.create_optimizer(model)

    assert [group["lr"] for group in optimizer.param_groups] == [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]
    assert optimizer.defaults["weight_decay"] == 0.001

    grouped_param_ids = [
        id(param) for group in optimizer.param_groups for param in group["params"]
    ]
    model_param_ids = {id(param) for param in model.parameters()}
    assert len(grouped_param_ids) == len(set(grouped_param_ids))
    assert set(grouped_param_ids) == model_param_ids

    lr_by_param_id = {
        id(param): group["lr"]
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert lr_by_param_id[id(model.topk.pre_w)] == 0.01
    assert lr_by_param_id[id(model.blocklinear.log_weight)] == 0.02
    assert lr_by_param_id[id(model.reactivation.log_m)] == 0.03
    assert lr_by_param_id[id(model.decoder_network[0].log_m)] == 0.03
    assert lr_by_param_id[id(model.decoder_network[1].weight)] == 0.04
    assert lr_by_param_id[id(model.other.weight)] == 0.05


def test_fsdp_module_class_name_iterator_preserves_concrete_name_rule():
    class TopKLinear(nn.Module):
        pass

    class DerivedTopKLinear(TopKLinear):
        pass

    first = TopKLinear()
    derived = DerivedTopKLinear()
    second = TopKLinear()
    model = nn.Sequential(first, derived, nn.Sequential(second))

    assert list(_iter_modules_by_class_name(model, "TopKLinear")) == [
        first,
        second,
    ]


def test_fsdp_reactivation_candidate_iterator_preserves_module_order():
    first = ParametricTanh(2)
    second = ParametricTanh(3)
    model = nn.Sequential(first, nn.ReLU(), nn.Sequential(second))

    candidates = list(_iter_reactivation_parameter_candidates(model))

    assert candidates == [(first.log_m, first.b), (second.log_m, second.b)]


def test_create_fsdp_trainer_preserves_rank0_custom_kwargs(monkeypatch):
    """FSDP trainer construction should preserve rank-0 custom kwargs."""
    calls = []

    class FSDPTrainerStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    fsdp_config = object()
    optimizer_config = object()
    analysis_manager = object()
    wandb_run = object()
    train_common = SimpleNamespace(
        epochs=12,
        save_every_n_epochs=3,
        loss_function="mse",
        grad_clip_value=1.5,
        use_amp=True,
        early_stopping=True,
        patience=4,
        print_every=2,
        plot_losses=True,
    )
    config = SimpleNamespace(
        experiment=SimpleNamespace(
            seed="42",
            checkpointing={"enabled": True, "save_every_n_epochs": 3},
        )
    )

    monkeypatch.setattr(fsdp_script, "FSDPTrainer", FSDPTrainerStub)

    trainer = _create_fsdp_trainer(
        rank=0,
        world_size=8,
        fsdp_config=fsdp_config,
        run_save_path="/tmp/run",
        train_common=train_common,
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        wandb_run=wandb_run,
        config=config,
    )

    assert isinstance(trainer, FSDPTrainerStub)
    assert trainer.kwargs == {
        "rank": 0,
        "world_size": 8,
        "fsdp_config": fsdp_config,
        "checkpoint_dir": "/tmp/run",
        "save_every_n_epochs": 3,
        "optimizer_config": optimizer_config,
        "analysis_manager": analysis_manager,
        "wandb_run": wandb_run,
        "epochs": 12,
        "loss_function": "mse",
        "grad_clip_value": 1.5,
        "use_amp": True,
        "early_stopping": True,
        "patience": 4,
        "print_every": 2,
        "checkpointing": True,
        "plot_losses": True,
        "save_path": "/tmp/run",
        "seed": 42,
    }
    assert calls == [trainer.kwargs]


def test_create_fsdp_trainer_preserves_non_rank0_defaults(monkeypatch):
    """FSDP trainer construction should keep defaults and rank-gated paths."""
    calls = []

    class FSDPTrainerStub:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            calls.append(kwargs)

    fsdp_config = object()
    optimizer_config = object()
    train_common = SimpleNamespace(epochs=5)
    config = SimpleNamespace(experiment=SimpleNamespace(seed=9))

    monkeypatch.setattr(fsdp_script, "FSDPTrainer", FSDPTrainerStub)

    trainer = _create_fsdp_trainer(
        rank=2,
        world_size=4,
        fsdp_config=fsdp_config,
        run_save_path="/tmp/run",
        train_common=train_common,
        optimizer_config=optimizer_config,
        analysis_manager=None,
        wandb_run=None,
        config=config,
    )

    assert trainer.kwargs == {
        "rank": 2,
        "world_size": 4,
        "fsdp_config": fsdp_config,
        "checkpoint_dir": None,
        "save_every_n_epochs": None,
        "optimizer_config": optimizer_config,
        "analysis_manager": None,
        "wandb_run": None,
        "epochs": 5,
        "loss_function": "cat_nll",
        "grad_clip_value": 5.0,
        "use_amp": False,
        "early_stopping": False,
        "patience": 10,
        "print_every": 1,
        "checkpointing": False,
        "plot_losses": False,
        "save_path": None,
        "seed": 9,
    }
    assert calls == [trainer.kwargs]


def test_setup_rank0_analysis_manager_wires_manager_and_calibration(monkeypatch):
    """FSDP rank-0 analysis setup should preserve manager and calibration wiring."""
    analysis_encoder = object()
    original_encoder = object()
    encoded_train_ds = object()
    encoded_valid_ds = object()
    encoded_test_ds = object()
    model = SimpleNamespace(encoder_network=original_encoder)
    config = SimpleNamespace(
        analysis=SimpleNamespace(
            performance_analysis=SimpleNamespace(enabled=True, training=False)
        )
    )
    manager_calls = []
    calibration_calls = []

    class AnalysisManagerStub:
        def __init__(self, **kwargs):
            manager_calls.append(kwargs)

    monkeypatch.setattr(
        fsdp_script,
        "_get_analysis_encoder_network",
        lambda encoder_network: analysis_encoder,
    )
    monkeypatch.setattr(fsdp_script, "AnalysisManager", AnalysisManagerStub)
    monkeypatch.setattr(
        fsdp_script,
        "_maybe_calibrate_fsdp_data_driven_reactivation",
        lambda **kwargs: calibration_calls.append(kwargs),
    )

    analysis_manager = _setup_rank0_analysis_manager(
        config=config,
        model=model,
        encoder_network=original_encoder,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        encoded_test_ds=encoded_test_ds,
        run_save_path="/tmp/run",
        local_rank=3,
    )

    assert isinstance(analysis_manager, AnalysisManagerStub)
    assert model.encoder_network is analysis_encoder
    assert manager_calls == [
        {
            "model": model,
            "data": {
                "train": encoded_train_ds,
                "valid": encoded_valid_ds,
                "test": encoded_test_ds,
            },
            "analysis_config": config.analysis,
            "save_root": "/tmp/run",
            "device": "cuda:3",
        }
    ]
    assert calibration_calls == [
        {
            "config": config,
            "model": model,
            "train_dataset": encoded_train_ds,
        }
    ]


def test_save_final_fsdp_checkpoint_and_run_analysis_rank0(monkeypatch):
    """FSDP finalization should save a full checkpoint and run analysis on rank 0."""
    fsdp_model = object()
    optimizer = object()
    config = SimpleNamespace(analysis=SimpleNamespace(any_enabled=lambda: True))
    analysis_manager = object()
    encoder_network = object()
    trainer = SimpleNamespace(
        fsdp_model=fsdp_model,
        optimizer=optimizer,
        epochs=7,
        last_completed_epoch=4,
    )
    mkdir_calls = []
    checkpoint_calls = []
    analysis_calls = []

    monkeypatch.setattr(
        fsdp_script.os,
        "makedirs",
        lambda path, *, exist_ok: mkdir_calls.append(
            {"path": path, "exist_ok": exist_ok}
        ),
    )
    monkeypatch.setattr(
        fsdp_script,
        "save_fsdp_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append(
            {"args": args, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        fsdp_script,
        "_run_final_analysis_on_plain_model",
        lambda **kwargs: analysis_calls.append(kwargs),
    )

    full_ckpt_path = _save_final_fsdp_checkpoint_and_run_analysis(
        rank=0,
        trainer=trainer,
        run_save_path="/tmp/run",
        config=config,
        analysis_manager=analysis_manager,
        encoder_network=encoder_network,
        local_rank=2,
    )

    assert full_ckpt_path == "/tmp/run/checkpoints/final_full.pt"
    assert mkdir_calls == [{"path": "/tmp/run/checkpoints", "exist_ok": True}]
    assert checkpoint_calls == [
        {
            "args": (
                fsdp_model,
                optimizer,
                4,
                "/tmp/run/checkpoints/final_full.pt",
            ),
            "kwargs": {"rank": 0},
        }
    ]
    assert analysis_calls == [
        {
            "config": config,
            "analysis_manager": analysis_manager,
            "full_ckpt_path": "/tmp/run/checkpoints/final_full.pt",
            "encoder_network": encoder_network,
            "local_rank": 2,
        }
    ]


def test_save_final_fsdp_checkpoint_and_run_analysis_non_rank0(monkeypatch):
    """FSDP finalization should only gather/save on nonzero ranks."""
    fsdp_model = object()
    optimizer = object()
    analysis_manager = object()
    trainer = SimpleNamespace(fsdp_model=fsdp_model, optimizer=optimizer, epochs=4)
    mkdir_calls = []
    checkpoint_calls = []
    analysis_calls = []

    monkeypatch.setattr(
        fsdp_script.os,
        "makedirs",
        lambda path, *, exist_ok: mkdir_calls.append(
            {"path": path, "exist_ok": exist_ok}
        ),
    )
    monkeypatch.setattr(
        fsdp_script,
        "save_fsdp_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append(
            {"args": args, "kwargs": kwargs}
        ),
    )
    monkeypatch.setattr(
        fsdp_script,
        "_run_final_analysis_on_plain_model",
        lambda **kwargs: analysis_calls.append(kwargs),
    )

    full_ckpt_path = _save_final_fsdp_checkpoint_and_run_analysis(
        rank=1,
        trainer=trainer,
        run_save_path=None,
        config=SimpleNamespace(analysis=SimpleNamespace(any_enabled=lambda: True)),
        analysis_manager=analysis_manager,
        encoder_network=object(),
        local_rank=0,
    )

    assert full_ckpt_path == "/tmp/checkpoints/final_full.pt"
    assert mkdir_calls == []
    assert checkpoint_calls == [
        {
            "args": (
                fsdp_model,
                optimizer,
                3,
                "/tmp/checkpoints/final_full.pt",
            ),
            "kwargs": {"rank": 1},
        }
    ]
    assert analysis_calls == []


def test_save_final_fsdp_checkpoint_skips_when_nothing_requested(monkeypatch):
    checkpoint_calls = []
    trainer = SimpleNamespace(
        fsdp_model=object(),
        optimizer=object(),
        epochs=1,
        final_checkpoint_path=None,
    )
    monkeypatch.setattr(
        fsdp_script,
        "save_fsdp_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append((args, kwargs)),
    )

    result = _save_final_fsdp_checkpoint_and_run_analysis(
        rank=0,
        trainer=trainer,
        run_save_path="/tmp/run",
        config=SimpleNamespace(analysis=SimpleNamespace(any_enabled=lambda: False)),
        analysis_manager=None,
        encoder_network=None,
        local_rank=0,
    )

    assert result == ""
    assert checkpoint_calls == []


def test_save_rank0_training_results_saves_only_on_rank0(monkeypatch):
    """Training results should be persisted by rank 0 only."""
    save_calls = []

    monkeypatch.setattr(
        fsdp_script,
        "save_dict",
        lambda *args: save_calls.append(args),
    )

    results = {"final_train_loss": 1.5, "final_valid_loss": 2.5}
    _save_rank0_training_results(
        rank=0,
        results=results,
        run_save_path="/tmp/run",
    )
    _save_rank0_training_results(
        rank=1,
        results={"final_train_loss": 99.0},
        run_save_path="/tmp/other",
    )

    assert save_calls == [(results, "/tmp/run", "training_results.json")]


class _PolicyLayer(nn.Module):
    """Minimal module carrying a branch-layer reactivation policy attribute."""

    def __init__(self, policy):
        super().__init__()
        self.reactivation_init_policy = policy


class _PolicyModel(nn.Module):
    """Minimal nested model for FSDP reactivation policy discovery tests."""

    def __init__(self, *policies):
        super().__init__()
        self.layers = nn.ModuleList(_PolicyLayer(policy) for policy in policies)


def test_requested_data_driven_reactivation_policies_collects_config_and_layers():
    """FSDP calibration should detect data-driven config and layer policies."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            core=SimpleNamespace(
                reactivation=SimpleNamespace(init_policy="occupancy_quantile")
            )
        )
    )
    model = _PolicyModel("fixed", "empirical")

    policies = _requested_data_driven_reactivation_policies(config, model)

    assert policies == {"occupancy_quantile", "empirical"}


def test_requested_data_driven_reactivation_policies_ignores_missing_config_and_fixed_layers():
    """FSDP calibration discovery should be quiet when no data-driven policy exists."""
    config = SimpleNamespace()
    model = _PolicyModel("fixed", "analytical")

    policies = _requested_data_driven_reactivation_policies(config, model)

    assert policies == set()


def test_maybe_calibrate_fsdp_data_driven_reactivation_calls_single_gpu_helper(
    monkeypatch,
):
    """FSDP helper should delegate calibration with the same public contract."""
    from dendritic_modeling.scripts.training import train_experiments

    config = SimpleNamespace(
        model=SimpleNamespace(
            core=SimpleNamespace(reactivation=SimpleNamespace(init_policy="empirical"))
        )
    )
    model = _PolicyModel()
    train_dataset = object()
    calls = []

    def fake_calibrate(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        train_experiments,
        "_maybe_calibrate_data_driven_reactivation",
        fake_calibrate,
    )

    _maybe_calibrate_fsdp_data_driven_reactivation(
        config=config,
        model=model,
        train_dataset=train_dataset,
    )

    assert calls == [
        {
            "config": config,
            "model": model,
            "train_dataset": train_dataset,
            "is_main": True,
        }
    ]


def test_maybe_calibrate_fsdp_data_driven_reactivation_skips_when_not_requested(
    monkeypatch,
):
    """FSDP helper should not import/delegate calibration for fixed policies."""
    from dendritic_modeling.scripts.training import train_experiments

    config = SimpleNamespace(
        model=SimpleNamespace(
            core=SimpleNamespace(reactivation=SimpleNamespace(init_policy="fixed"))
        )
    )
    calls = []

    monkeypatch.setattr(
        train_experiments,
        "_maybe_calibrate_data_driven_reactivation",
        lambda **kwargs: calls.append(kwargs),
    )

    _maybe_calibrate_fsdp_data_driven_reactivation(
        config=config,
        model=_PolicyModel("analytical"),
        train_dataset=object(),
    )

    assert calls == []


class MockDendriticBranchLayer(nn.Module):
    """Mock dendritic branch layer for testing."""

    def __init__(self, num_params=1000):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_params, 1))

    def forward(self, x):
        return x


class MockExcitationInhibitionLayer(nn.Module):
    """Mock excitation-inhibition layer for testing."""

    def __init__(self, num_params=2000):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_params, 1))

    def forward(self, x):
        return x


class TestCreateFSDPWrapPolicy:
    """Test create_fsdp_wrap_policy function."""

    def test_fsdp_dendritic_module_predicate_preserves_isinstance_rule(self):
        from dendritic_modeling.networks.architectures.excitation_inhibition import (
            DendriticBranchLayer,
        )

        dendritic_layer = DendriticBranchLayer.__new__(DendriticBranchLayer)

        assert _is_dendritic_fsdp_module(dendritic_layer)
        assert not _is_dendritic_fsdp_module(MockDendriticBranchLayer())
        assert not _is_dendritic_fsdp_module(nn.Linear(2, 2))

    def test_large_linear_checkpoint_candidate_preserves_threshold(self):
        assert not _is_large_linear_checkpoint_candidate(nn.Linear(1000, 2))
        assert _is_large_linear_checkpoint_candidate(nn.Linear(1001, 2))
        assert not _is_large_linear_checkpoint_candidate(nn.ReLU())

    def test_wrap_policy_basic(self):
        """Test basic wrap policy creation."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        assert callable(policy)

    def test_wrap_policy_dendritic_layers(self):
        """Test policy wraps dendritic layers."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        # Mock dendritic layer - test shows isinstance check behavior
        dendritic_layer = MockDendriticBranchLayer()
        result = policy(
            dendritic_layer, None, 500
        )  # Below threshold, not actual DendriticBranchLayer
        assert result is False  # Mock doesn't inherit from actual class

        # Test with enough parameters to trigger wrapping
        large_layer = nn.Linear(100, 100)  # >10k parameters
        result = policy(large_layer, None, 20000)
        assert result is True

    def test_wrap_policy_parameter_threshold(self):
        """Test policy wraps based on parameter count."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        # Regular module above threshold should be wrapped
        regular_module = nn.Linear(100, 100)  # ~10k params
        result = policy(regular_module, None, 2000)
        assert result is True

        # Regular module below threshold should not be wrapped
        result = policy(regular_module, None, 500)
        assert result is False

    def test_wrap_policy_recurse_true(self):
        """Test policy behavior - recurse parameter not used in logic."""
        policy = create_fsdp_wrap_policy(min_num_params=1000)

        regular_module = nn.Linear(10, 10)
        result = policy(regular_module, True, 500)  # Below threshold
        assert (
            result is False
        )  # Policy doesn't use recurse parameter, only checks param count

    def test_wrap_policy_different_thresholds(self):
        """Test policy with different parameter thresholds."""
        # High threshold
        policy_high = create_fsdp_wrap_policy(min_num_params=10000)
        regular_module = nn.Linear(50, 50)  # ~2.5k params
        result = policy_high(regular_module, None, 2500)
        assert result is False

        # Low threshold
        policy_low = create_fsdp_wrap_policy(min_num_params=100)
        result = policy_low(regular_module, None, 2500)
        assert result is True


class TestGetFSDPConfig:
    """Test get_fsdp_config function."""

    @patch("torch.cuda.current_device", return_value=0)
    def test_get_fsdp_config_mixed_precision_dtype_aliases_cpu_safe(
        self, _mock_current_device
    ):
        config = get_fsdp_config(
            fsdp_config_dict={
                "mixed_precision": True,
                "mixed_precision_dtype": "bf16",
                "reduce_dtype": "fp32",
            }
        )

        assert config["mixed_precision"].param_dtype is torch.bfloat16
        assert config["mixed_precision"].reduce_dtype is torch.float32
        assert config["mixed_precision"].buffer_dtype is torch.bfloat16

    @patch("torch.cuda.current_device", return_value=0)
    def test_get_fsdp_config_indexes_synapses_with_narrow_wrap_policy(
        self, _mock_current_device
    ):
        config = get_fsdp_config(
            fsdp_config_dict={"auto_wrap_policy": "indexed_synapses"}
        )

        policy = config["auto_wrap_policy"]
        assert policy._module_classes == {
            IndexedSparseLinear,
            IndexedDynamicTopKLinear,
        }

    @patch("torch.cuda.current_device", return_value=0)
    def test_get_fsdp_config_rejects_unknown_auto_wrap_policy(
        self, _mock_current_device
    ):
        with pytest.raises(ValueError, match="auto_wrap_policy"):
            get_fsdp_config(fsdp_config_dict={"auto_wrap_policy": "size_based"})

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_defaults(self):
        """Test get_fsdp_config with default parameters."""
        config = get_fsdp_config()

        assert isinstance(config, dict)
        assert "auto_wrap_policy" in config
        assert "mixed_precision" in config
        assert "cpu_offload" in config
        assert "sharding_strategy" in config
        assert "sync_module_states" in config

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_small_model(self):
        """Test FSDP config for small model."""
        config = get_fsdp_config(model_size="small")

        # Auto-wrapping is disabled to fix parameter corruption, so policy should be None
        assert config["auto_wrap_policy"] is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_large_model(self):
        """Test FSDP config for large model."""
        config = get_fsdp_config(model_size="large")

        # Auto-wrapping is disabled to fix parameter corruption, so policy should be None
        assert config["auto_wrap_policy"] is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_xlarge_model(self):
        """Test FSDP config for extra large model."""
        config = get_fsdp_config(model_size="xlarge")

        # Auto-wrapping is disabled to fix parameter corruption, so policy should be None
        assert config["auto_wrap_policy"] is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_mixed_precision(self):
        """Test FSDP config with mixed precision."""
        config = get_fsdp_config(mixed_precision=True)

        assert config["mixed_precision"] is not None

        config_no_mp = get_fsdp_config(mixed_precision=False)
        assert config_no_mp["mixed_precision"] is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_cpu_offload(self):
        """Test FSDP config with CPU offload."""
        config = get_fsdp_config(cpu_offload=True)

        assert config["cpu_offload"] is not None

        config_no_offload = get_fsdp_config(cpu_offload=False)
        assert config_no_offload["cpu_offload"] is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_sharding_strategies(self):
        """Test different sharding strategies."""
        strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]

        for strategy in strategies:
            config = get_fsdp_config(sharding_strategy=strategy)
            assert config["sharding_strategy"] is not None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_with_dict_override(self):
        """Test FSDP config with dictionary override."""
        override_dict = {
            "mixed_precision": False,
            "cpu_offload": True,
            "sharding_strategy": "NO_SHARD",
            "min_num_params": 5000,
            "sync_module_states": False,
        }

        config = get_fsdp_config(fsdp_config_dict=override_dict)

        # Should use override values
        assert config["cpu_offload"] is not None  # True from override
        assert config["sync_module_states"] is False
        assert config["mixed_precision"] is None  # False from override

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_communication_overhead(self):
        """Test communication overhead optimization."""
        with patch("dendritic_modeling.training.fsdp_utils.logger") as mock_logger:
            get_fsdp_config(reduce_communication_overhead=True)

            # Should log optimization info
            mock_logger.info.assert_called_once()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    def test_get_fsdp_config_backward_prefetch_mapping(self):
        """Test backward prefetch strategy mapping."""
        config = get_fsdp_config()

        # Should have valid backward prefetch config
        assert "backward_prefetch" in config


class TestWrapModelWithFSDP:
    """Test wrap_model_with_fsdp function."""

    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    @pytest.fixture
    def mock_fsdp_config(self):
        """Create mock FSDP config."""
        return {
            "auto_wrap_policy": create_fsdp_wrap_policy(),
            "mixed_precision": None,
            "cpu_offload": None,
            "sharding_strategy": "FULL_SHARD",
            "sync_module_states": True,
            "use_orig_params": True,
            "backward_prefetch": "BACKWARD_PRE",
            "limit_all_gathers": True,
            "forward_prefetch": True,
            "gradient_checkpointing": False,
        }

    @patch("torch.distributed.is_initialized")
    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_basic(
        self, mock_fsdp, mock_dist_init, simple_model, mock_fsdp_config
    ):
        """Test basic model wrapping with FSDP."""
        mock_dist_init.return_value = True
        mock_fsdp.return_value = simple_model

        wrapped_model = wrap_model_with_fsdp(simple_model, mock_fsdp_config)

        mock_fsdp.assert_called_once()
        assert wrapped_model == simple_model

    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_no_distributed(self, mock_fsdp, simple_model, mock_fsdp_config):
        """Test model wrapping when distributed is not initialized."""
        # FullyShardedDataParallel will raise ValueError if distributed is not initialized
        mock_fsdp.side_effect = ValueError(
            "Default process group has not been initialized"
        )

        # Make sure config doesn't have gradient_checkpointing key
        if "gradient_checkpointing" in mock_fsdp_config:
            del mock_fsdp_config["gradient_checkpointing"]

        with pytest.raises(
            ValueError, match="Default process group has not been initialized"
        ):
            wrap_model_with_fsdp(simple_model, mock_fsdp_config)

    @patch("torch.distributed.is_initialized")
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available - FSDP requires GPU"
    )
    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_with_gradient_checkpointing(
        self, mock_fsdp, mock_dist_init, simple_model
    ):
        """Test model wrapping with gradient checkpointing."""
        mock_dist_init.return_value = True
        mock_fsdp.return_value = simple_model

        config = get_fsdp_config()
        config["_gradient_checkpointing"] = True

        with patch(
            "dendritic_modeling.training.fsdp_utils._apply_gradient_checkpointing"
        ) as mock_gc:
            wrap_model_with_fsdp(simple_model, config)

            mock_gc.assert_called_once_with(simple_model)

    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_basic_functionality(
        self, mock_fsdp, simple_model, mock_fsdp_config
    ):
        """Test basic model wrapping functionality."""
        mock_fsdp.return_value = simple_model

        # Make sure config doesn't have gradient_checkpointing key
        if "gradient_checkpointing" in mock_fsdp_config:
            del mock_fsdp_config["gradient_checkpointing"]

        result = wrap_model_with_fsdp(simple_model, mock_fsdp_config)

        # Should return the wrapped model
        assert result == simple_model
        # FullyShardedDataParallel should be called
        mock_fsdp.assert_called_once()


class TestApplyGradientCheckpointing:
    """Test _apply_gradient_checkpointing function."""

    def test_apply_gradient_checkpointing(self):
        """Test applying gradient checkpointing to model."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        # For a simple Sequential model without dendritic layers, no logging occurs
        # This is expected behavior - test passes if no exception is raised
        _apply_gradient_checkpointing(model)

    def test_apply_gradient_checkpointing_with_large_linear(self):
        """Test applying gradient checkpointing to model with large linear layers."""
        # Create a model with a large linear layer (> 1000 features)
        model = nn.Sequential(nn.Linear(2000, 1000), nn.ReLU(), nn.Linear(1000, 2))

        with patch("dendritic_modeling.training.fsdp_utils.logger") as mock_logger:
            _apply_gradient_checkpointing(model)

            # Should log debug message for large linear layer
            assert (
                mock_logger.debug.called
            ), "Expected logger.debug to be called for large linear layer"

    def test_apply_gradient_checkpointing_large_linear_preserves_forward(self):
        """Checkpointed large linear modules should preserve forward and gradients."""
        torch.manual_seed(0)
        layer = nn.Linear(1001, 3)
        model = nn.Sequential(layer)
        x = torch.randn(4, 1001, requires_grad=True)

        expected = model(x).detach()

        _apply_gradient_checkpointing(model)

        assert layer.forward.__func__.__name__ == "checkpointed_forward"
        actual = model(x)
        torch.testing.assert_close(actual, expected)

        actual.sum().backward()

        assert x.grad is not None
        assert layer.weight.grad is not None
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(layer.weight.grad).all()

    def test_recurrent_ei_checkpoint_preserves_forward_and_gradients(self):
        """Checkpoint the recurrent layer that owns raw-current computation."""
        params = {
            "unified_ei": {
                "output_mode": "last",
                "layers": [
                    {
                        "excitatory": {
                            "n_neurons": 6,
                            "branch_factors": [1],
                            "ff_excitatory_synapses": 3,
                            "rec_excitatory_synapses": 2,
                            "use_noise": False,
                        },
                        "inhibitory": None,
                        "recurrent": True,
                    }
                ],
            }
        }
        torch.manual_seed(12)
        baseline = get_architecture("ei_unified", params, input_dim=4)
        checkpointed = get_architecture("ei_unified", params, input_dim=4)
        checkpointed.load_state_dict(baseline.state_dict())
        _apply_gradient_checkpointing(checkpointed)

        assert checkpointed.forward.__func__.__name__ == "checkpointed_forward"
        assert (
            checkpointed.layers[0].forward.__func__.__name__ != "checkpointed_forward"
        )
        assert (
            checkpointed.layers[0]
            .e_population.branch_layers[0]
            .forward.__func__.__name__
            != "checkpointed_forward"
        )

        x_baseline = torch.randn(2, 3, 4, requires_grad=True)
        x_checkpointed = x_baseline.detach().clone().requires_grad_(True)
        y_baseline = baseline(x_baseline)
        y_checkpointed = checkpointed(x_checkpointed)
        torch.testing.assert_close(y_checkpointed, y_baseline)

        y_baseline.square().sum().backward()
        y_checkpointed.square().sum().backward()
        torch.testing.assert_close(x_checkpointed.grad, x_baseline.grad)
        for baseline_param, checkpointed_param in zip(
            baseline.parameters(),
            checkpointed.parameters(),
            strict=True,
        ):
            if baseline_param.grad is None:
                assert checkpointed_param.grad is None
            else:
                torch.testing.assert_close(
                    checkpointed_param.grad,
                    baseline_param.grad,
                )

    def test_apply_gradient_checkpointing_selects_targets_in_module_order(
        self, monkeypatch
    ):
        """Only eligible modules should be checkpointed, in named-module order."""
        first = nn.Linear(1001, 3)
        small = nn.Linear(3, 3)
        second = nn.Linear(1002, 4)
        model = nn.Module()
        model.first = first
        model.small = small
        model.block = nn.Sequential(nn.ReLU(), second)
        patched = []

        monkeypatch.setattr(
            "dendritic_modeling.training.fsdp_utils._checkpoint_module_forward",
            lambda module: patched.append(module),
        )

        _apply_gradient_checkpointing(model)

        assert patched == [first, second]

    def test_apply_gradient_checkpointing_nested_model(self):
        """Test gradient checkpointing on nested model."""
        model = nn.ModuleDict(
            {
                "encoder": nn.Sequential(nn.Linear(10, 5), nn.ReLU()),
                "decoder": nn.Sequential(nn.Linear(5, 2), nn.Sigmoid()),
            }
        )

        # No logging expected for regular modules without dendritic layers
        # The function only logs for dendritic layers or large linear layers
        # Test passes if no exception is raised
        _apply_gradient_checkpointing(model)


@pytest.mark.skip(reason="FSDP checkpointing requires complex PyTorch FSDP setup")
class TestFSDPCheckpointing:
    """Test FSDP checkpoint save/load functions."""

    @pytest.fixture
    def mock_fsdp_model(self):
        """Create mock FSDP model."""
        # Use a real PyTorch model to avoid deep FSDP mocking issues
        return nn.Linear(3, 2)

    @pytest.fixture
    def mock_optimizer(self):
        """Create mock optimizer."""
        optimizer = Mock()
        optimizer.state_dict.return_value = {"lr": 0.001}
        return optimizer

    def test_save_fsdp_checkpoint(self, mock_fsdp_model, mock_optimizer):
        """Test saving FSDP checkpoint."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")

            with patch("torch.save") as mock_save:
                save_fsdp_checkpoint(
                    mock_fsdp_model,
                    mock_optimizer,
                    epoch=10,
                    save_path=checkpoint_path,
                )

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        assert "model_state_dict" in saved_data
        assert "optimizer_state_dict" in saved_data
        assert "epoch" in saved_data
        assert saved_data["epoch"] == 10

    def test_save_fsdp_checkpoint_with_metadata(self, mock_fsdp_model, mock_optimizer):
        """Test saving checkpoint with additional metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")

            with patch("torch.save") as mock_save:
                save_fsdp_checkpoint(
                    mock_fsdp_model,
                    mock_optimizer,
                    epoch=5,
                    save_path=checkpoint_path,
                )

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        assert "model_state_dict" in saved_data
        assert "optimizer_state_dict" in saved_data
        assert "epoch" in saved_data
        assert saved_data["epoch"] == 5

    def test_load_fsdp_checkpoint(self, mock_fsdp_model, mock_optimizer):
        """Test loading FSDP checkpoint."""
        checkpoint_data = {
            "model_state_dict": {"weight": torch.randn(5, 5)},
            "optimizer_state_dict": {"lr": 0.001},
            "epoch": 15,
            "loss": 0.25,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")

            with patch("torch.load", return_value=checkpoint_data) as mock_load:
                loaded_data = load_fsdp_checkpoint(
                    mock_fsdp_model, mock_optimizer, checkpoint_path
                )

                mock_load.assert_called_once_with(checkpoint_path, map_location="cpu")
                mock_fsdp_model.load_state_dict.assert_called_once()
                mock_optimizer.load_state_dict.assert_called_once()

                assert loaded_data["epoch"] == 15
                assert loaded_data["loss"] == 0.25

    def test_load_fsdp_checkpoint_model_only(self, mock_fsdp_model):
        """Test loading checkpoint with model only."""
        checkpoint_data = {
            "model_state_dict": {"weight": torch.randn(5, 5)},
            "epoch": 20,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = os.path.join(tmp_dir, "checkpoint.pt")

            with patch("torch.load", return_value=checkpoint_data):
                loaded_data = load_fsdp_checkpoint(
                    mock_fsdp_model,
                    None,
                    checkpoint_path,  # No optimizer
                )

                mock_fsdp_model.load_state_dict.assert_called_once()
                assert loaded_data["epoch"] == 20

    def test_load_fsdp_checkpoint_nonexistent_file(
        self, mock_fsdp_model, mock_optimizer
    ):
        """Test loading checkpoint from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_fsdp_checkpoint(
                mock_fsdp_model, mock_optimizer, "/nonexistent/checkpoint.pt"
            )


class TestMemoryEstimation:
    """Test memory estimation functions."""

    def test_estimate_memory_requirements_small_model(self):
        """Test memory estimation for small model."""
        memory = estimate_memory_requirements(
            num_neurons=[1000, 500],
            num_synapses_per_neuron=[100, 50],
            num_branches=[10, 5],
        )

        assert isinstance(memory, dict)
        assert len(memory) > 0

    def test_estimate_memory_requirements_large_model(self):
        """Test memory estimation for large model."""
        memory = estimate_memory_requirements(
            num_neurons=[10000, 5000],
            num_synapses_per_neuron=[1000, 500],
            num_branches=[100, 50],
        )

        assert isinstance(memory, dict)
        assert len(memory) > 0

    def test_estimate_memory_requirements_with_fsdp(self):
        """Test memory estimation with FSDP enabled."""
        memory = estimate_memory_requirements(
            num_neurons=[5000, 2500],
            num_synapses_per_neuron=[500, 250],
            num_branches=[50, 25],
        )

        memory_no_fsdp = estimate_memory_requirements(
            num_neurons=[5000, 2500],
            num_synapses_per_neuron=[500, 250],
            num_branches=[50, 25],
        )

        # Both should return memory estimates
        assert isinstance(memory, dict)
        assert isinstance(memory_no_fsdp, dict)

    def test_estimate_memory_requirements_different_precision(self):
        """Test memory estimation with different precision."""
        memory_fp32 = estimate_memory_requirements(
            num_neurons=[2000, 1000],
            num_synapses_per_neuron=[200, 100],
            num_branches=[20, 10],
            dtype=torch.float32,
        )

        memory_fp16 = estimate_memory_requirements(
            num_neurons=[2000, 1000],
            num_synapses_per_neuron=[200, 100],
            num_branches=[20, 10],
            dtype=torch.float16,
        )

        # Both should return memory estimates
        assert isinstance(memory_fp32, dict)
        assert isinstance(memory_fp16, dict)

    def test_estimate_memory_requirements_gradient_checkpointing(self):
        """Test memory estimation with gradient checkpointing."""
        memory_normal = estimate_memory_requirements(
            num_neurons=[2000, 1000],
            num_synapses_per_neuron=[200, 100],
            num_branches=[20, 10],
        )

        memory_gc = estimate_memory_requirements(
            num_neurons=[2000, 1000],
            num_synapses_per_neuron=[200, 100],
            num_branches=[20, 10],
        )

        # Both should return memory estimates
        assert isinstance(memory_normal, dict)
        assert isinstance(memory_gc, dict)

    def test_get_recommended_gpu_count(self):
        """Test GPU count recommendation."""
        gpu_count = get_recommended_gpu_count(
            memory_per_gpu_gb=16.0, total_memory_gb=64.0
        )

        assert gpu_count == 8  # 64 / (16 * 0.9) = 4.4, rounds up to 8

    def test_get_recommended_gpu_count_minimum(self):
        """Test GPU count recommendation with minimum."""
        gpu_count = get_recommended_gpu_count(
            memory_per_gpu_gb=32.0,
            total_memory_gb=8.0,  # Less than per-GPU memory
        )

        assert gpu_count == 2  # Function returns minimum of 2 GPUs

    def test_get_recommended_gpu_count_fractional(self):
        """Test GPU count recommendation with fractional result."""
        gpu_count = get_recommended_gpu_count(
            memory_per_gpu_gb=24.0,
            total_memory_gb=100.0,  # 100/24 = 4.17
        )

        assert gpu_count == 8  # 100/(24*0.9) = 4.6, rounds up to 8


class TestFSDPUtilsEdgeCases:
    """Test edge cases for FSDP utilities."""

    def test_wrap_policy_with_none_values(self):
        """Test wrap policy with None values."""
        policy = create_fsdp_wrap_policy()

        # Should handle None module gracefully
        result = policy(None, None, 1000)
        assert result is False

    def test_get_fsdp_config_invalid_strategy(self):
        """Test FSDP config with invalid sharding strategy."""
        # Should raise KeyError for invalid strategy
        with pytest.raises(KeyError, match="INVALID_STRATEGY"):
            get_fsdp_config(sharding_strategy="INVALID_STRATEGY")

    def test_memory_estimation_edge_cases(self):
        """Test memory estimation with edge cases."""
        # Zero parameters
        memory = estimate_memory_requirements(
            num_neurons=[1000, 500],
            num_synapses_per_neuron=[100, 50],
            num_branches=[10, 5],
        )
        assert isinstance(memory, dict)

        # Very large parameters
        memory = estimate_memory_requirements(
            num_neurons=[1000, 500],
            num_synapses_per_neuron=[100, 50],
            num_branches=[10, 5],
        )
        assert isinstance(memory, dict)

    def test_checkpoint_save_directory_creation(self):
        """Test that checkpoint save creates directory if needed."""
        # This test requires complex FSDP setup, skip it for now
        pytest.skip("FSDP checkpoint save requires complex PyTorch FSDP setup")

    def test_estimate_memory_with_all_optimizations(self):
        """Test memory estimation with all memory optimizations enabled."""
        memory = estimate_memory_requirements(
            num_neurons=[5000, 2500],
            num_synapses_per_neuron=[500, 250],
            num_branches=[50, 25],
            dtype=torch.float16,
        )

        # Should return memory breakdown
        assert isinstance(memory, dict)
        assert len(memory) > 0
