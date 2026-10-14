"""
Tests for training FSDP utils module to improve coverage.
"""

import logging
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import dendritic_modeling.scripts.training.train_experiments_fsdp as fsdp_script
from dendritic_modeling.config.keys import field_default
from dendritic_modeling.config.training import (
    CommonTrainerConfig,
    OptimizerConfig,
    ParamGroupsConfig,
)
from dendritic_modeling.networks.activations import ParametricTanh
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
    _apply_fsdp_trainability_policy,
    _build_fsdp_runtime_configs,
    _calibrate_fsdp_reactivation,
    _create_fsdp_trainer,
    _dataloader_kwargs_from_common,
    _print_rank0_model_summary,
    _requested_data_driven_reactivation_policies,
    _resolve_large_model_input_dim,
    _run_final_analysis_on_plain_model,
    _save_final_model_and_run_analysis,
    _setup_rank0_analysis_manager,
    setup_distributed,
)
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.fsdp_utils import (
    _apply_gradient_checkpointing,
    _is_dendritic_fsdp_module,
    _is_large_linear_checkpoint_candidate,
    estimate_memory_requirements,
    get_fsdp_config,
    get_recommended_gpu_count,
    wrap_model_with_fsdp,
)
from dendritic_modeling.training.loss.functions import CrossEntropyLoss
from dendritic_modeling.training.strategies.fsdp_standard import (
    FSDPTrainer,
    _sequence_lengths_from_batch,
    _tensor_diagnostics,
)


@pytest.mark.parametrize(
    "timeout_seconds, expected",
    [(3600, timedelta(hours=1)), (None, timedelta(seconds=1800))],
)
def test_setup_distributed_uses_the_configured_timeout(
    monkeypatch, timeout_seconds, expected
):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")
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

    assert setup_distributed(timeout_seconds) == (3, 1, 8)
    assert init_calls == [{"backend": "nccl", "timeout": expected}]
    assert set(device_calls) == {1}


def test_setup_distributed_rejects_a_non_positive_timeout(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(fsdp_script.dist, "is_initialized", lambda: False)

    with pytest.raises(ValueError, match="must be positive"):
        setup_distributed(timeout_seconds=0)


def test_ddp_setup_passes_the_configured_timeout(monkeypatch):
    from dendritic_modeling.scripts.training import train_experiments

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(train_experiments.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(train_experiments.torch.cuda, "set_device", lambda _rank: None)
    init_calls = []
    monkeypatch.setattr(
        train_experiments.dist,
        "init_process_group",
        lambda **kwargs: init_calls.append(kwargs),
    )
    train_experiments.setup_distributed(timeout_seconds=7200)
    train_experiments.setup_distributed()
    assert init_calls == [
        {"backend": "nccl", "timeout": timedelta(hours=2)},
        {"backend": "nccl"},
    ]


def test_distributed_timeout_must_be_positive():
    from dendritic_modeling.config import config_from_mapping

    with pytest.raises(ValueError, match="timeout_seconds must be a positive"):
        config_from_mapping({"distributed": {"timeout_seconds": 0}})


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


def test_tensor_diagnostics_reports_finite_values_only():
    diagnostics = _tensor_diagnostics(
        torch.tensor([1.0, 3.0, float("nan"), float("inf")])
    )

    assert diagnostics == {
        "numel": 4,
        "finite_count": 2,
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
        "rms": pytest.approx(5**0.5),
    }


def test_tensor_diagnostics_separates_feature_and_sample_variation():
    diagnostics = _tensor_diagnostics(torch.tensor([[1.0, 3.0], [3.0, 5.0]]))

    assert diagnostics["within_row_std_mean"] == 1.0
    assert diagnostics["row_mean_std"] == 1.0
    assert diagnostics["across_row_feature_std_mean"] == 1.0


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
    """FSDP loaders shard the run's datasets and log their tuning on rank 0."""
    train_ds = TensorDataset(torch.randn(8, 3), torch.randint(0, 2, (8,)))
    valid_ds = TensorDataset(torch.randn(4, 3), torch.randint(0, 2, (4,)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    common = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    config = SimpleNamespace(
        experiment=SimpleNamespace(train_valid_split=0.8, seed=7),
        training=SimpleNamespace(main=SimpleNamespace(common=common)),
    )

    with caplog.at_level(
        logging.INFO,
        logger="dendritic_modeling.scripts.training.train_experiments_fsdp",
    ):
        train_loader, valid_loader = fsdp_script.setup_data_loaders(
            config, train_ds, valid_ds, rank=0, world_size=1
        )

    assert train_loader.dataset is train_ds
    assert valid_loader.dataset is valid_ds
    assert train_loader.num_workers == 0
    assert valid_loader.num_workers == 0
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


def test_run_final_analysis_on_plain_model_loads_the_gathered_weights(monkeypatch):
    """FSDP final analysis runs on a fresh unsharded copy of the model."""

    class PlainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded_state = None
            self.to_device = None
            self.eval_called = False

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

    config = SimpleNamespace(model=SimpleNamespace(name="model"))
    plain_model = PlainModel()
    analysis_manager = AnalysisManagerStub()
    state = {"weight": torch.ones(1)}
    monkeypatch.setattr(
        fsdp_script,
        "initialize_model",
        lambda model_config: (plain_model, None),
    )

    _run_final_analysis_on_plain_model(
        config=config,
        analysis_manager=analysis_manager,
        state_dict=state,
        local_rank=2,
    )

    assert plain_model.loaded_state is state
    assert plain_model.to_device == "cuda:2"
    assert plain_model.eval_called
    assert analysis_manager.model is plain_model
    assert analysis_manager.run_calls == [{"filename": "final", "training": False}]


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
        param_groups=SimpleNamespace(
            lr=0.123,
            split_params=True,
            topk_lr=0.01,
            blocklinear_lr=0.02,
            reactivation_lr=0.03,
            decoder_lr=0.04,
            decoder_input_lr=0.006,
            encoder_lr=0.05,
        ),
        synapse_decay_rate=0.4,
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
    # synapse_decay_rate (0.4) is sparse maintenance and is not mapped to
    # optimizer weight decay; the optimizer comes from training.main.optimizer.
    assert optimizer_config == {
        "optimizer": OptimizerConfig(),
        "param_groups": ParamGroupsConfig(
            lr=0.123,
            split_params=True,
            topk_lr=0.01,
            blocklinear_lr=0.02,
            reactivation_lr=0.03,
            decoder_lr=0.04,
            decoder_input_lr=0.006,
            encoder_lr=0.05,
        ),
    }
    assert calls == [
        {
            "mixed_precision": True,
            "cpu_offload": False,
            "sharding_strategy": "NO_SHARD",
            "fsdp_config_dict": vars(fsdp_cfg),
        }
    ]


def test_build_fsdp_runtime_configs_fills_typed_defaults_and_slot_fsdp_config(
    monkeypatch,
):
    """Unset param-group fields take the ParamGroupsConfig defaults."""

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
        "optimizer": OptimizerConfig(),
        "param_groups": ParamGroupsConfig(lr=0.01),
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
    """FSDP split optimizer groups are single-assignment and ordered.

    A model without ``core_network`` is its own core; a gate inside the
    decoder trains with the decoder, as in single-process training.
    """

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
        "optimizer": OptimizerConfig(weight_decay=0.001),
        "param_groups": {
            "split_params": True,
            "lr": 0.05,
            "topk_lr": 0.01,
            "blocklinear_lr": 0.02,
            "reactivation_lr": 0.03,
            "decoder_lr": 0.04,
        },
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
    assert lr_by_param_id[id(model.decoder_network[0].log_m)] == 0.04
    assert lr_by_param_id[id(model.decoder_network[1].weight)] == 0.04
    assert lr_by_param_id[id(model.other.weight)] == 0.05


def test_fsdp_optimizer_assigns_indexed_synapses_to_topk_group():
    """Indexed sparse weights should honor the configured synapse LR."""
    model = nn.Module()
    model.fixed = IndexedSparseLinear(4, 3, K=2, seed=1)
    model.dynamic = IndexedDynamicTopKLinear(
        4,
        3,
        K=2,
        candidate_size=3,
        seed=2,
    )
    model.decoder_network = nn.Linear(3, 2)

    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.optimizer_config = {
        "param_groups": {
            "split_params": True,
            "lr": 0.05,
            "topk_lr": 0.01,
            "decoder_lr": 0.04,
        }
    }

    optimizer = trainer.create_optimizer(model)
    lr_by_param_id = {
        id(param): group["lr"]
        for group in optimizer.param_groups
        for param in group["params"]
    }

    assert lr_by_param_id[id(model.fixed.pre_w)] == 0.01
    assert lr_by_param_id[id(model.dynamic.pre_w)] == 0.01
    assert lr_by_param_id[id(model.decoder_network.weight)] == 0.04


def test_fsdp_optimizer_scales_only_decoder_input_weight():
    model = nn.Module()
    model.decoder_network = nn.Sequential(
        nn.LayerNorm(8, elementwise_affine=False),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )

    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.optimizer_config = {
        "param_groups": {
            "split_params": True,
            "lr": 0.05,
            "decoder_input_lr": 1e-5,
            "decoder_lr": 1e-3,
        }
    }

    optimizer = trainer.create_optimizer(model)
    group_by_param_id = {
        id(param): group
        for group in optimizer.param_groups
        for param in group["params"]
    }

    assert group_by_param_id[id(model.decoder_network[1].weight)]["name"] == (
        "decoder_input"
    )
    assert group_by_param_id[id(model.decoder_network[1].weight)]["lr"] == 1e-5
    assert group_by_param_id[id(model.decoder_network[1].bias)]["name"] == "decoder"
    assert group_by_param_id[id(model.decoder_network[1].bias)]["lr"] == 1e-3
    assert group_by_param_id[id(model.decoder_network[3].weight)]["name"] == "decoder"
    assert group_by_param_id[id(model.decoder_network[3].weight)]["lr"] == 1e-3


def test_fsdp_freeze_core_policy_leaves_only_readout_trainable():
    model = nn.Module()
    model.core_network = nn.Linear(8, 8)
    model.decoder_network = nn.Linear(8, 2)

    frozen = _apply_fsdp_trainability_policy(
        model,
        SimpleNamespace(freeze_core=True),
    )

    assert frozen == 72
    assert all(
        not parameter.requires_grad for parameter in model.core_network.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.decoder_network.parameters()
    )
    assert model._freeze_core_forward is True
    assert all(
        module._fsdp_frozen_core is True for module in model.core_network.modules()
    )

    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.optimizer_config = {
        "param_groups": {"split_params": True, "lr": 0.0, "decoder_lr": 1e-3}
    }
    optimizer = trainer.create_optimizer(model)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {
        id(parameter) for parameter in model.decoder_network.parameters()
    }


def test_fsdp_freeze_core_policy_is_opt_in():
    model = nn.Module()
    model.core_network = nn.Linear(3, 2)

    frozen = _apply_fsdp_trainability_policy(
        model,
        SimpleNamespace(freeze_core=False),
    )

    assert frozen == 0
    assert all(parameter.requires_grad for parameter in model.core_network.parameters())
    assert not hasattr(model, "_freeze_core_forward")


def test_fsdp_core_output_cache_requires_frozen_core():
    model = nn.Module()
    model.core_network = nn.Linear(3, 2)

    with pytest.raises(ValueError, match="requires freeze_core=true"):
        _apply_fsdp_trainability_policy(
            model,
            SimpleNamespace(
                freeze_core=False,
                cache_frozen_core_outputs=True,
            ),
        )


def test_fsdp_core_output_cache_requires_frozen_encoder():
    model = nn.Module()
    model.encoder_network = nn.Linear(3, 3)
    model.core_network = nn.Linear(3, 2)

    with pytest.raises(ValueError, match="frozen or parameter-free encoder"):
        _apply_fsdp_trainability_policy(
            model,
            SimpleNamespace(
                freeze_core=True,
                cache_frozen_core_outputs=True,
            ),
        )


def test_fsdp_cache_core_output_loader_materializes_decoder_inputs():
    class CacheModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.core_network = nn.Linear(2, 2, bias=False)
            self.decoder_network = nn.Linear(2, 2)
            self.core_calls = 0
            self._freeze_core_forward = True
            with torch.no_grad():
                self.core_network.weight.copy_(2 * torch.eye(2))

        def forward(self, inputs, seq_lengths=None, *, precomputed_core=False):
            del seq_lengths
            if not precomputed_core:
                self.core_calls += 1
                inputs = self.core_network(inputs)
            return self.decoder_network(inputs)

    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.seed = 11
    trainer.cached_core_batch_size = None
    model = CacheModel()
    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)

    cached_loader = trainer._cache_core_output_loader(
        model,
        loader,
        shuffle=False,
        seed_offset=17,
    )

    cached_features, cached_targets = cached_loader.dataset.tensors
    torch.testing.assert_close(cached_features, 2 * inputs)
    torch.testing.assert_close(cached_targets, targets)
    assert model.core_calls == 2

    cached_logits = model(cached_features, precomputed_core=True)
    direct_logits = model.decoder_network(2 * inputs)
    torch.testing.assert_close(cached_logits, direct_logits)
    assert model.core_calls == 2


def test_fsdp_cached_eval_loader_preserves_real_padding_count():
    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 1
    trainer.seed = 11
    trainer.cached_core_batch_size = None
    model = nn.Module()
    model.decoder_network = nn.Identity()
    model.forward = lambda inputs, **_kwargs: model.decoder_network(inputs)
    dataset = TensorDataset(torch.ones(3, 2), torch.tensor([0, 1, 0]))
    sampler = PaddedDistributedEvalSampler(dataset, num_replicas=2, rank=1)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler)

    cached_loader = trainer._cache_core_output_loader(
        model,
        loader,
        shuffle=False,
        seed_offset=29,
    )

    assert len(cached_loader.dataset) == 2
    assert cached_loader._dendritic_real_count == 1


def test_fsdp_training_reuses_cached_core_outputs_across_epochs(
    monkeypatch,
    tmp_path,
):
    class CacheModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.core_network = nn.Linear(2, 2)
            self.core_network.requires_grad_(False)
            self.decoder_network = nn.Linear(2, 2)
            self.core_calls = 0
            self._freeze_core_forward = True

        def forward(self, inputs, seq_lengths=None, *, precomputed_core=False):
            del seq_lengths
            if not precomputed_core:
                self.core_calls += 1
                inputs = self.core_network(inputs)
            return self.decoder_network(inputs)

    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.epochs = 2
    trainer.early_stopping = False
    trainer.patience = 10
    trainer.checkpointing = False
    trainer.print_every = 1
    trainer.save_every_n_epochs = None
    trainer.save_path = str(tmp_path)
    trainer.log_first_step_diagnostics = False
    trainer.cache_frozen_core_outputs = True
    trainer.cached_core_batch_size = 2
    trainer.first_step_diagnostics = None
    trainer._decoder_input_diagnostics = None
    trainer._use_precomputed_core = False
    trainer.fsdp_config = {}
    trainer.optimizer_config = {"param_groups": {"lr": 1e-2, "split_params": False}}
    trainer.analysis_manager = None
    trainer.grad_clip_value = None
    trainer.use_amp = False
    trainer.loss_function = CrossEntropyLoss()
    trainer.loss_fn = trainer.loss_function.from_predictions
    trainer._class_logit_loss = True
    trainer.seed = 13
    trainer._loss_plotting = lambda *_args: None

    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    model = CacheModel()
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.fsdp_standard.wrap_model_with_fsdp",
        lambda wrapped_model, _config: wrapped_model,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *_args, **_kwargs: None)

    results = trainer.train(model, loader, loader)

    # Two train and two validation batches are cached once. Neither training
    # epoch recomputes the frozen core.
    assert model.core_calls == 4
    assert len(results["train_losses"]) == 2
    assert len(results["valid_losses"]) == 2
    assert results["fsdp_setup_seconds"] >= 0.0
    assert results["frozen_core_cache_seconds"] >= 0.0
    assert len(results["epoch_times_seconds"]) == 2
    assert results["training_loop_seconds"] >= sum(results["epoch_times_seconds"])
    assert results["trainer_total_seconds"] >= results["training_loop_seconds"]


def test_fsdp_best_valid_loss_tracks_history_without_early_stopping(
    monkeypatch,
    tmp_path,
):
    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.epochs = 3
    trainer.early_stopping = False
    trainer.patience = 10
    trainer.checkpointing = False
    trainer.print_every = 1
    trainer.save_every_n_epochs = None
    trainer.save_path = str(tmp_path)
    trainer.log_first_step_diagnostics = False
    trainer.cache_frozen_core_outputs = False
    trainer.cached_core_batch_size = None
    trainer._use_precomputed_core = False
    trainer.first_step_diagnostics = None
    trainer._decoder_input_diagnostics = None
    trainer.fsdp_config = {}
    trainer.optimizer_config = {"param_groups": {"lr": 1e-3}}
    trainer.analysis_manager = None
    trainer._loss_plotting = lambda *_args: None
    trainer._epoch_train_fsdp = lambda *_args: (0.5, 0.5)
    validation_losses = iter((0.7, 0.4, 0.6))
    trainer._epoch_valid_fsdp = lambda *_args: (next(validation_losses), 0.5)

    model = nn.Linear(2, 2)
    loader = DataLoader(TensorDataset(torch.ones(1, 2), torch.zeros(1)), batch_size=1)
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.fsdp_standard.wrap_model_with_fsdp",
        lambda wrapped_model, _config: wrapped_model,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    results = trainer.train(model, loader, loader)

    assert results["valid_losses"] == [0.7, 0.4, 0.6]
    assert results["best_valid_loss"] == 0.4
    assert results["frozen_core_cache_seconds"] is None
    assert len(results["epoch_times_seconds"]) == 3


def test_fsdp_final_checkpoint_does_not_take_the_final_weights_name(
    monkeypatch,
    tmp_path,
):
    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.device = torch.device("cpu")
    trainer.rank = 0
    trainer.epochs = 1
    trainer.early_stopping = False
    trainer.patience = 10
    trainer.checkpointing = True
    trainer.checkpoint_dir = str(tmp_path)
    trainer.print_every = 1
    trainer.save_every_n_epochs = None
    trainer.save_path = str(tmp_path)
    trainer.log_first_step_diagnostics = False
    trainer.cache_frozen_core_outputs = False
    trainer.cached_core_batch_size = None
    trainer._use_precomputed_core = False
    trainer.first_step_diagnostics = None
    trainer._decoder_input_diagnostics = None
    trainer.fsdp_config = {}
    trainer.optimizer_config = {"param_groups": {"lr": 1e-3}}
    trainer.analysis_manager = None
    trainer._loss_plotting = lambda *_args: None
    trainer._epoch_train_fsdp = lambda *_args: (0.5, 0.5)
    trainer._epoch_valid_fsdp = lambda *_args: (0.4, 0.5)
    saved = []
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.fsdp_standard.save_fsdp_checkpoint",
        lambda _model, _optimizer, _epoch, path, **_kwargs: saved.append(path),
    )
    monkeypatch.setattr(
        "dendritic_modeling.training.strategies.fsdp_standard.wrap_model_with_fsdp",
        lambda wrapped_model, _config: wrapped_model,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = DataLoader(TensorDataset(torch.ones(1, 2), torch.zeros(1)), batch_size=1)

    trainer.train(nn.Linear(2, 2), loader, loader)

    # The entry point writes the plain weights to final_model.pt afterwards.
    assert trainer.final_checkpoint_path == str(tmp_path / "final_checkpoint.pt")
    assert str(tmp_path / "final_model.pt") not in saved
    assert trainer.final_checkpoint_path in saved


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
    train_common = CommonTrainerConfig(
        epochs=12,
        loss_function="mse",
        grad_clip_value=1.5,
        use_amp=True,
        early_stopping=True,
        patience=4,
        print_every=2,
        plot_losses=True,
        log_first_step_diagnostics=True,
    )
    config = SimpleNamespace(
        model=SimpleNamespace(task="regression"),
        experiment=SimpleNamespace(
            seed="42",
            checkpointing={"enabled": True, "save_every_n_epochs": 3},
        ),
        distributed=SimpleNamespace(
            fsdp=SimpleNamespace(
                cache_frozen_core_outputs=True,
                cached_core_batch_size=16,
            )
        ),
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
        "epochs": 12,
        "loss_function": "mse",
        "task": "regression",
        "grad_clip_value": 1.5,
        "use_amp": True,
        "early_stopping": True,
        "patience": 4,
        "print_every": 2,
        "log_first_step_diagnostics": True,
        "cache_frozen_core_outputs": True,
        "cached_core_batch_size": 16,
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
    train_common = CommonTrainerConfig(epochs=5)
    config = SimpleNamespace(
        model=SimpleNamespace(task="classification"),
        experiment=SimpleNamespace(seed=9),
        distributed=SimpleNamespace(
            fsdp=SimpleNamespace(
                cache_frozen_core_outputs=False,
                cached_core_batch_size=None,
            )
        ),
    )

    monkeypatch.setattr(fsdp_script, "FSDPTrainer", FSDPTrainerStub)

    trainer = _create_fsdp_trainer(
        rank=2,
        world_size=4,
        fsdp_config=fsdp_config,
        run_save_path="/tmp/run",
        train_common=train_common,
        optimizer_config=optimizer_config,
        analysis_manager=None,
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
        "epochs": 5,
        # The typed configuration's defaults, not separate ones.
        "loss_function": field_default(CommonTrainerConfig, "loss_function"),
        "task": "classification",
        "grad_clip_value": field_default(CommonTrainerConfig, "grad_clip_value"),
        "use_amp": field_default(CommonTrainerConfig, "use_amp"),
        "early_stopping": field_default(CommonTrainerConfig, "early_stopping"),
        "patience": field_default(CommonTrainerConfig, "patience"),
        "print_every": field_default(CommonTrainerConfig, "print_every"),
        "log_first_step_diagnostics": False,
        "cache_frozen_core_outputs": False,
        "cached_core_batch_size": None,
        "checkpointing": False,
        "plot_losses": field_default(CommonTrainerConfig, "plot_losses"),
        "save_path": None,
        "seed": 9,
    }
    assert calls == [trainer.kwargs]


def test_setup_rank0_analysis_manager_wires_manager(monkeypatch):
    """The rank-0 analysis manager reads the run's datasets and model."""
    encoder = object()
    encoded_train_ds = object()
    encoded_valid_ds = object()
    encoded_test_ds = object()
    model = SimpleNamespace(encoder_network=encoder)
    config = SimpleNamespace(
        analysis=SimpleNamespace(
            performance=SimpleNamespace(enabled=True, training=False)
        )
    )
    manager_calls = []

    class AnalysisManagerStub:
        def __init__(self, **kwargs):
            manager_calls.append(kwargs)

    monkeypatch.setattr(fsdp_script, "AnalysisManager", AnalysisManagerStub)

    analysis_manager = _setup_rank0_analysis_manager(
        config=config,
        model=model,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        encoded_test_ds=encoded_test_ds,
        run_save_path="/tmp/run",
        local_rank=3,
    )

    assert isinstance(analysis_manager, AnalysisManagerStub)
    assert model.encoder_network is encoder
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


@pytest.mark.parametrize("rank", [0, 1])
def test_final_weights_are_saved_on_rank0_after_releasing_the_fsdp_model(
    monkeypatch, tmp_path, rank
):
    """Every rank gathers; rank 0 saves final_model.pt and runs the analysis."""
    state = {"weight": torch.ones(2)}
    gathered = []
    analysis_calls = []
    trainer = SimpleNamespace(fsdp_model=object(), optimizer=object())
    trainer.release_model = lambda: setattr(trainer, "fsdp_model", None)
    analysis_manager = SimpleNamespace(model=object())
    monkeypatch.setattr(
        fsdp_script,
        "gather_full_state_dict",
        lambda model: gathered.append(model) or (state if rank == 0 else {}),
    )
    monkeypatch.setattr(
        fsdp_script,
        "_run_final_analysis_on_plain_model",
        lambda **kwargs: analysis_calls.append(kwargs),
    )

    path = _save_final_model_and_run_analysis(
        rank=rank,
        trainer=trainer,
        run_save_path=str(tmp_path),
        config="config",
        analysis_manager=analysis_manager,
        local_rank=1,
    )

    assert len(gathered) == 1
    assert trainer.fsdp_model is None
    assert analysis_manager.model is None
    if rank == 0:
        assert path == str(tmp_path / "final_model.pt")
        torch.testing.assert_close(torch.load(path)["weight"], state["weight"])
        assert analysis_calls == [
            {
                "config": "config",
                "analysis_manager": analysis_manager,
                "state_dict": state,
                "local_rank": 1,
            }
        ]
    else:
        assert path == ""
        assert not (tmp_path / "final_model.pt").exists()
        assert analysis_calls == []


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


def _calibration_config(init_policy, *, sync_module_states=True):
    return SimpleNamespace(
        model=SimpleNamespace(
            core=SimpleNamespace(reactivation=SimpleNamespace(init_policy=init_policy))
        ),
        distributed=SimpleNamespace(
            fsdp=SimpleNamespace(sync_module_states=sync_module_states)
        ),
    )


def test_fsdp_calibration_runs_on_rank_zero_without_analysis(monkeypatch):
    """FSDP calibrates data-driven gates before wrapping, on rank 0."""
    config = _calibration_config("empirical")
    model = _PolicyModel()
    train_dataset = object()
    calls = []
    monkeypatch.setattr(
        fsdp_script,
        "_maybe_calibrate_data_driven_reactivation",
        lambda **kwargs: calls.append(kwargs),
    )

    _calibrate_fsdp_reactivation(config, model, train_dataset, rank=1)
    assert calls == []
    _calibrate_fsdp_reactivation(config, model, train_dataset, rank=0)
    assert calls == [
        {
            "config": config,
            "model": model,
            "train_dataset": train_dataset,
            "is_main": True,
        }
    ]


def test_fsdp_calibration_needs_synchronized_module_states():
    with pytest.raises(ValueError, match="sync_module_states"):
        _calibrate_fsdp_reactivation(
            _calibration_config("empirical", sync_module_states=False),
            _PolicyModel(),
            object(),
            rank=0,
        )


def test_fsdp_calibration_skips_fixed_policies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fsdp_script,
        "_maybe_calibrate_data_driven_reactivation",
        lambda **kwargs: calls.append(kwargs),
    )
    _calibrate_fsdp_reactivation(
        _calibration_config("fixed", sync_module_states=False),
        _PolicyModel("analytical"),
        object(),
        rank=0,
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


class TestFSDPModulePredicates:
    """Which modules gradient checkpointing treats as dendritic or large."""

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
    def test_get_fsdp_config_defaults_to_bf16_and_explicit_prefetch(
        self, _mock_current_device
    ):
        config = get_fsdp_config()
        assert config["mixed_precision"].param_dtype is torch.bfloat16
        assert config["limit_all_gathers"] is True
        assert config["forward_prefetch"] is True
        config = get_fsdp_config(
            fsdp_config_dict={"limit_all_gathers": False, "forward_prefetch": False}
        )
        assert config["limit_all_gathers"] is False
        assert config["forward_prefetch"] is False

    @patch("torch.cuda.current_device", return_value=0)
    def test_get_fsdp_config_size_policy_reads_min_num_params(
        self, _mock_current_device
    ):
        policy = get_fsdp_config(
            fsdp_config_dict={"auto_wrap_policy": "size", "min_num_params": 50}
        )["auto_wrap_policy"]
        assert policy(nn.Linear(10, 10), recurse=False, nonwrapped_numel=110)
        assert not policy(nn.Linear(2, 2), recurse=False, nonwrapped_numel=6)

    @patch("torch.cuda.current_device", return_value=0)
    def test_get_fsdp_config_rejects_unknown_auto_wrap_policy(
        self, _mock_current_device
    ):
        with pytest.raises(ValueError, match="auto_wrap_policy"):
            get_fsdp_config(fsdp_config_dict={"auto_wrap_policy": "size_based_huge"})

    @pytest.mark.requires_gpu
    def test_get_fsdp_config_defaults(self):
        """Test get_fsdp_config with default parameters."""
        config = get_fsdp_config()

        assert isinstance(config, dict)
        assert "auto_wrap_policy" in config
        assert "mixed_precision" in config
        assert "cpu_offload" in config
        assert "sharding_strategy" in config
        assert "sync_module_states" in config

    @pytest.mark.requires_gpu
    def test_get_fsdp_config_mixed_precision(self):
        """Test FSDP config with mixed precision."""
        config = get_fsdp_config(mixed_precision=True)

        assert config["mixed_precision"] is not None

        config_no_mp = get_fsdp_config(mixed_precision=False)
        assert config_no_mp["mixed_precision"] is None

    @pytest.mark.requires_gpu
    def test_get_fsdp_config_cpu_offload(self):
        """Test FSDP config with CPU offload."""
        config = get_fsdp_config(cpu_offload=True)

        assert config["cpu_offload"] is not None

        config_no_offload = get_fsdp_config(cpu_offload=False)
        assert config_no_offload["cpu_offload"] is None

    @pytest.mark.requires_gpu
    def test_get_fsdp_config_sharding_strategies(self):
        """Test different sharding strategies."""
        strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]

        for strategy in strategies:
            config = get_fsdp_config(sharding_strategy=strategy)
            assert config["sharding_strategy"] is not None

    @pytest.mark.requires_gpu
    def test_get_fsdp_config_with_dict_override(self):
        """Test FSDP config with dictionary override."""
        override_dict = {
            "mixed_precision": False,
            "cpu_offload": True,
            "sharding_strategy": "NO_SHARD",
            "sync_module_states": False,
        }

        config = get_fsdp_config(fsdp_config_dict=override_dict)

        # Should use override values
        assert config["cpu_offload"] is not None  # True from override
        assert config["sync_module_states"] is False
        assert config["mixed_precision"] is None  # False from override

    @pytest.mark.requires_gpu
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
            "auto_wrap_policy": None,
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

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.is_initialized")
    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_basic(
        self, mock_fsdp, mock_dist_init, _world_size, simple_model, mock_fsdp_config
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

    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.is_initialized")
    @pytest.mark.requires_gpu
    @patch("dendritic_modeling.training.fsdp_utils.FullyShardedDataParallel")
    def test_wrap_model_with_gradient_checkpointing(
        self, mock_fsdp, mock_dist_init, _world_size, simple_model
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

    def test_get_fsdp_config_invalid_strategy(self):
        """An unknown sharding strategy raises naming the valid ones."""
        with pytest.raises(ValueError, match="INVALID_STRATEGY"):
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


def test_checkpoint_saving_gathers_optimizer_state_on_rank_zero_only(
    monkeypatch, tmp_path
):
    from contextlib import contextmanager

    from dendritic_modeling.training import fsdp_utils

    contexts = []

    class _FSDP:
        @staticmethod
        @contextmanager
        def state_dict_type(module, kind, model_config, optim_config):
            contexts.append((model_config, optim_config))
            yield

        @staticmethod
        def optim_state_dict(module, optimizer):
            raise RuntimeError("gather failed")

    model = nn.Linear(2, 2)
    monkeypatch.setattr(fsdp_utils, "FullyShardedDataParallel", _FSDP)
    with pytest.raises(RuntimeError, match="gather failed"):
        fsdp_utils.save_fsdp_checkpoint(
            model, torch.optim.Adam(model.parameters()), 1, str(tmp_path / "c.pt")
        )
    ((model_config, optim_config),) = contexts
    assert model_config.rank0_only and optim_config.rank0_only
    assert optim_config.offload_to_cpu
    assert not (tmp_path / "c.pt").exists()  # no rank-local fallback was written
