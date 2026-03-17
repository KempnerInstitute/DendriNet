import warnings

import torch
import yaml
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from dendritic_modeling.analysis.metrics.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
    materialize_dataset,
)
from dendritic_modeling.config import load_config
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models.classifier import Classifier
from dendritic_modeling.networks.architectures.classical.identity import Identity


class NoSliceDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            raise TypeError("slice access disabled")
        return self.x[idx], self.y[idx]


def test_legacy_performance_runtime_maps_to_shared_runtime(tmp_path):
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
        "wandb": {"use_wandb": False},
        "outputs": {"results_dir": str(tmp_path / "outputs")},
        "analysis": {
            "performance": {
                "enabled": True,
                "training_runtime": {
                    "mode": "stream",
                    "batch_size": 32,
                    "num_workers": 0,
                    "splits": ["valid"],
                },
                "final_runtime": {
                    "mode": "materialize",
                    "batch_size": 64,
                    "num_workers": 0,
                    "splits": ["test"],
                },
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))

    cfg = load_config(str(config_path))

    assert cfg.analysis.runtime.training.mode == "stream"
    assert cfg.analysis.runtime.training.batch_size == 32
    assert cfg.analysis.runtime.final.mode == "materialize"
    assert cfg.analysis.performance_analysis.training_splits == ["valid"]
    assert cfg.analysis.performance_analysis.final_splits == ["test"]


def test_single_gpu_without_runtime_defaults_to_materialize(tmp_path):
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
        "wandb": {"use_wandb": False},
        "outputs": {"results_dir": str(tmp_path / "outputs")},
        "distributed": {"mode": "none"},
        "analysis": {"performance": {"enabled": True}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))

    cfg = load_config(str(config_path))

    assert cfg.analysis.runtime.training.mode == "materialize"
    assert cfg.analysis.runtime.final.mode == "materialize"


def test_distributed_without_runtime_keeps_auto_default(tmp_path):
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
        "wandb": {"use_wandb": False},
        "outputs": {"results_dir": str(tmp_path / "outputs")},
        "distributed": {"mode": "fsdp"},
        "analysis": {"performance": {"enabled": True}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))

    cfg = load_config(str(config_path))

    assert cfg.analysis.runtime.training.mode == "auto"
    assert cfg.analysis.runtime.final.mode == "auto"


def test_explicit_runtime_overrides_single_gpu_default(tmp_path):
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
        "wandb": {"use_wandb": False},
        "outputs": {"results_dir": str(tmp_path / "outputs")},
        "distributed": {"mode": "none"},
        "analysis": {
            "runtime": {
                "final": {
                    "mode": "stream",
                    "batch_size": 32,
                    "num_workers": 0,
                }
            },
            "performance": {"enabled": True},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))

    cfg = load_config(str(config_path))

    assert cfg.analysis.runtime.training.mode == "materialize"
    assert cfg.analysis.runtime.final.mode == "stream"
    assert cfg.analysis.runtime.final.batch_size == 32


def test_materialize_dataset_streams_without_slice_support():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    y = torch.arange(10, dtype=torch.long)
    dataset = NoSliceDataset(x, y)
    runtime = EvaluationRuntimeConfig(mode="stream", batch_size=3, num_workers=0)

    materialized_x, materialized_y = materialize_dataset(
        dataset, runtime, explicit_max_samples=5
    )

    torch.testing.assert_close(materialized_x, x[:5])
    torch.testing.assert_close(materialized_y, y[:5])


def test_performance_analyzer_stream_matches_materialize():
    x = torch.randn(64, 3)
    y = x.argmax(dim=-1)
    dataset = TensorDataset(x, y)
    model = Classifier(
        encoder_network=Identity(3),
        core_network=Identity(3),
        decoder_network=Identity(3),
    )

    analyzer = PerformanceAnalyzer(
        type(
            "Params",
            (),
            {
                "accuracy": True,
                "auc": True,
                "categorical_loglikelihood": True,
                "mse": False,
                "cosine_similarity": False,
            },
        )()
    )

    materialize = EvaluationRuntimeConfig(mode="materialize", batch_size=16)
    stream = EvaluationRuntimeConfig(mode="stream", batch_size=16, num_workers=0)

    res_materialize = analyzer.analyze(
        model=model,
        test_ds=dataset,
        device="cpu",
        runtime=materialize,
        splits=["test"],
    )
    res_stream = analyzer.analyze(
        model=model,
        test_ds=dataset,
        device="cpu",
        runtime=stream,
        splits=["test"],
    )

    for metric_name in res_materialize:
        lhs = res_materialize[metric_name]["test"]
        rhs = res_stream[metric_name]["test"]
        assert abs(float(lhs) - float(rhs)) < 1e-6


def test_analysis_device_context_supports_parameterless_models():
    class EmptyModule(nn.Module):
        def forward(self, x):
            return x

    model = EmptyModule()
    model.train()

    with analysis_device_context(model, "cpu") as analysis_device:
        assert str(analysis_device) == "cpu"
        assert not model.training

    assert model.training


def test_performance_stream_on_cpu_emits_no_pin_memory_warning():
    x = torch.randn(32, 3)
    y = x.argmax(dim=-1)
    dataset = TensorDataset(x, y)
    model = Classifier(
        encoder_network=Identity(3),
        core_network=Identity(3),
        decoder_network=Identity(3),
    )

    analyzer = PerformanceAnalyzer(
        type(
            "Params",
            (),
            {
                "accuracy": True,
                "auc": False,
                "categorical_loglikelihood": False,
                "mse": False,
                "cosine_similarity": False,
            },
        )()
    )

    runtime = EvaluationRuntimeConfig(mode="stream", batch_size=8, num_workers=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyzer.analyze(
            model=model,
            test_ds=dataset,
            device="cpu",
            runtime=runtime,
            splits=["test"],
        )

    pin_memory_warnings = [
        str(w.message) for w in caught if "pin_memory" in str(w.message)
    ]
    assert not pin_memory_warnings


def test_iter_analysis_batches_materialize_yields_single_full_batch():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    y = torch.arange(10, dtype=torch.long)
    dataset = TensorDataset(x, y)
    runtime = EvaluationRuntimeConfig(mode="materialize", batch_size=3, num_workers=0)

    batches = list(iter_analysis_batches(dataset, runtime, explicit_max_samples=5))

    assert len(batches) == 1
    batch_x, batch_y = batches[0]
    torch.testing.assert_close(batch_x, x[:5])
    torch.testing.assert_close(batch_y, y[:5])


def test_iter_analysis_batches_materialize_falls_back_without_slice_support():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    y = torch.arange(10, dtype=torch.long)
    dataset = NoSliceDataset(x, y)
    runtime = EvaluationRuntimeConfig(mode="materialize", batch_size=3, num_workers=0)

    batches = list(iter_analysis_batches(dataset, runtime))

    assert len(batches) == 4
    batch_x = torch.cat([batch[0] for batch in batches], dim=0)
    batch_y = torch.cat([batch[1] for batch in batches], dim=0)
    torch.testing.assert_close(batch_x, x)
    torch.testing.assert_close(batch_y, y)
