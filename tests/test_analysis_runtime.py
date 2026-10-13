import json
import logging
import warnings

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dendritic_modeling.analysis.metrics.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.utils import runtime as runtime_utils
from dendritic_modeling.analysis.utils.runtime import (
    _resolve_analysis_num_workers,
    _resolve_analysis_persistent_workers,
    _resolve_analysis_prefetch_factor,
    analysis_device_context,
    analysis_loader_kwargs_from_runtime,
    evaluation_kwargs_from_runtime,
    iter_analysis_batches,
    materialize_dataset,
    run_model_over_analysis_batches,
    run_no_grad_analysis_batches,
)
from dendritic_modeling.config import load_config
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models.classifier import Classifier
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.training.utils import evaluation as evaluation_utils


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


def test_runtime_public_exports_include_shared_batch_helpers():
    assert "run_model_over_analysis_batches" in runtime_utils.__all__
    assert "run_no_grad_analysis_batches" in runtime_utils.__all__


def test_legacy_performance_runtime_maps_to_shared_runtime(tmp_path):
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
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
    assert cfg.analysis.performance.training_splits == ["validation"]
    assert cfg.analysis.performance.final_splits == ["test"]


def test_evaluation_kwargs_preserve_worker_runtime_options():
    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=16,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=5,
        in_order=False,
        max_batches=3,
        materialize_threshold_mb=128,
    )

    kwargs = evaluation_kwargs_from_runtime(runtime)

    assert kwargs == {
        "eval_mode": "stream",
        "eval_batch_size": 16,
        "eval_num_workers": 2,
        "eval_pin_memory": False,
        "eval_persistent_workers": True,
        "eval_prefetch_factor": 5,
        "eval_in_order": False,
        "eval_max_batches": 3,
        "eval_materialize_threshold_mb": 128,
        "eval_seed": 0,
    }


def test_analysis_loader_kwargs_preserve_worker_runtime_options():
    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=16,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=5,
        in_order=False,
    )

    kwargs = analysis_loader_kwargs_from_runtime(runtime, device="cpu")

    assert kwargs == {
        "num_workers": 2,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 5,
        "in_order": False,
    }


def test_analysis_loader_kwargs_omit_prefetch_without_workers():
    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=5,
    )

    kwargs = analysis_loader_kwargs_from_runtime(runtime, device="cpu")

    assert kwargs == {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
    }


def test_analysis_loader_kwargs_omit_default_in_order():
    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=16,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=5,
        in_order=None,
    )

    kwargs = analysis_loader_kwargs_from_runtime(runtime, device="cpu")

    assert "in_order" not in kwargs


def test_analysis_loader_kwargs_omit_unsupported_in_order(monkeypatch):
    monkeypatch.setattr(
        "dendritic_modeling.analysis.utils.runtime.dataloader_supports_kwarg",
        lambda name: name != "in_order",
    )
    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=16,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=5,
        in_order=False,
    )

    kwargs = analysis_loader_kwargs_from_runtime(runtime, device="cpu")

    assert "in_order" not in kwargs
    assert kwargs["prefetch_factor"] == 5


def test_evaluation_kwargs_omit_default_in_order():
    kwargs = evaluation_kwargs_from_runtime(EvaluationRuntimeConfig())

    assert "eval_in_order" not in kwargs


def test_analysis_prefetch_resolver_preserves_worker_gating_and_clamp():
    assert _resolve_analysis_prefetch_factor(None, num_workers=2) is None
    assert _resolve_analysis_prefetch_factor(5, num_workers=0) is None
    assert _resolve_analysis_prefetch_factor(0, num_workers=2) == 1
    assert _resolve_analysis_prefetch_factor(5, num_workers=2) == 5


def test_analysis_worker_resolvers_preserve_clamp_and_persistent_guard():
    assert _resolve_analysis_num_workers(EvaluationRuntimeConfig(num_workers=-3)) == 0
    assert _resolve_analysis_num_workers(EvaluationRuntimeConfig(num_workers=2)) == 2
    assert _resolve_analysis_persistent_workers(False, num_workers=0) is False
    assert _resolve_analysis_persistent_workers(True, num_workers=2) is True
    with pytest.raises(ValueError, match="persistent_workers=True"):
        _resolve_analysis_persistent_workers(True, num_workers=0)


def test_single_gpu_without_runtime_defaults_to_materialize(tmp_path):
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
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


def test_explicit_runtime_ignores_legacy_split_fields(tmp_path):
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
        "outputs": {"results_dir": str(tmp_path / "outputs")},
        "analysis": {
            "runtime": {
                "training": {
                    "mode": "stream",
                    "batch_size": 17,
                    "num_workers": 0,
                    "splits": ["train"],
                },
                "final": {
                    "mode": "materialize",
                    "batch_size": 23,
                    "num_workers": 0,
                    "splits": ["test"],
                },
            },
            "performance": {"enabled": True},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))

    cfg = load_config(str(config_path))

    assert cfg.analysis.runtime.training.mode == "stream"
    assert cfg.analysis.runtime.training.batch_size == 17
    assert cfg.analysis.runtime.final.mode == "materialize"
    assert cfg.analysis.runtime.final.batch_size == 23
    assert not hasattr(cfg.analysis.runtime.training, "splits")
    assert not hasattr(cfg.analysis.runtime.final, "splits")


def test_materialize_dataset_streams_without_slice_support():
    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    y = torch.arange(10, dtype=torch.long)
    dataset = NoSliceDataset(x, y)
    runtime = EvaluationRuntimeConfig(mode="stream", batch_size=3, num_workers=0)

    result = materialize_dataset(dataset, runtime, explicit_max_samples=5)
    assert len(result) == 2
    materialized_x, materialized_y = result

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

    # Oracle: the identity classifier's logits are x, so its one-vs-rest AUC is
    # sklearn's on softmax(x), and its accuracy is 1 (labels are argmax x).
    from sklearn.metrics import roc_auc_score

    expected_auc = roc_auc_score(
        y.numpy(), torch.softmax(x, dim=-1).numpy(), multi_class="ovr"
    )
    assert float(res_materialize["auc"]["test"]) == pytest.approx(expected_auc)
    assert float(res_materialize["accuracy"]["test"]) == pytest.approx(1.0)


def test_evaluate_metric_generic_forwards_seq_lengths_across_runtime_modes():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y = torch.arange(4, dtype=torch.long)
    seq_lengths = torch.tensor([1, 2, 3, 1])
    dataset = TensorDataset(x, y, seq_lengths)
    model = nn.Identity()
    seen_seq_lengths = []

    def metric_with_lengths(
        model,
        inputs,
        labels,
        reduce_dim=0,
        move_device=True,
        device="cpu",
        seq_lengths=None,
    ):
        assert seq_lengths is not None
        seen_seq_lengths.append(seq_lengths.clone())
        return seq_lengths.float().sum()

    def metric_without_lengths(
        model,
        inputs,
        labels,
        reduce_dim=0,
        move_device=True,
        device="cpu",
    ):
        return labels.float().sum()

    for mode in ("materialize", "stream"):
        with_lengths = evaluation_utils._evaluate_metric_generic(
            metric_func=metric_with_lengths,
            metric_name="seq_length_sum",
            model=model,
            test_ds=dataset,
            move_device=False,
            device="cpu",
            eval_mode=mode,
            eval_batch_size=2,
            eval_num_workers=0,
            eval_pin_memory=False,
        )
        without_lengths = evaluation_utils._evaluate_metric_generic(
            metric_func=metric_without_lengths,
            metric_name="label_sum",
            model=model,
            test_ds=dataset,
            move_device=False,
            device="cpu",
            eval_mode=mode,
            eval_batch_size=2,
            eval_num_workers=0,
            eval_pin_memory=False,
        )

        assert with_lengths == (None, None, float(seq_lengths.sum()))
        assert without_lengths == (None, None, float(y.sum()))

    assert len(seen_seq_lengths) == 2
    for observed in seen_seq_lengths:
        torch.testing.assert_close(observed, seq_lengths)


def test_evaluate_metric_generic_saves_tensor_metrics_via_shared_serialization(
    tmp_path,
):
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y = torch.arange(4, dtype=torch.long)
    dataset = TensorDataset(x, y)
    model = nn.Identity()

    def vector_metric(
        model,
        inputs,
        labels,
        reduce_dim=0,
        move_device=True,
        device="cpu",
    ):
        return torch.tensor([1.25, 2.5])

    result = evaluation_utils._evaluate_metric_generic(
        metric_func=vector_metric,
        metric_name="vector_metric",
        model=model,
        test_ds=dataset,
        move_device=False,
        device="cpu",
        save_path=str(tmp_path),
        filename="vector_metric.json",
    )

    assert result[0] is None
    assert result[1] is None
    torch.testing.assert_close(result[2], torch.tensor([1.25, 2.5]))
    assert json.loads((tmp_path / "vector_metric.json").read_text()) == {
        "train vector_metric": None,
        "valid vector_metric": None,
        "test vector_metric": [1.25, 2.5],
    }


def test_evaluate_metric_generic_materialize_falls_back_to_stream_without_slice():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y = torch.arange(4, dtype=torch.long)
    dataset = NoSliceDataset(x, y)
    model = nn.Identity()
    calls = []

    def label_sum_metric(
        model,
        inputs,
        labels,
        reduce_dim=0,
        move_device=True,
        device="cpu",
    ):
        calls.append((model, inputs.clone(), labels.clone(), move_device, device))
        return labels.float().sum()

    result = evaluation_utils._evaluate_metric_generic(
        metric_func=label_sum_metric,
        metric_name="label_sum",
        model=model,
        test_ds=dataset,
        move_device=False,
        device="cpu",
        eval_mode="materialize",
        eval_batch_size=2,
        eval_num_workers=0,
        eval_pin_memory=False,
    )

    assert result == (None, None, float(y.sum()))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][1], x)
    torch.testing.assert_close(calls[0][2], y)
    assert calls[0][3] is False
    assert calls[0][4] == "cpu"


def test_evaluate_metric_generic_materialized_nan_retries_stream(caplog):
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y = torch.arange(4, dtype=torch.long)
    dataset = TensorDataset(x, y)
    model = nn.Identity()
    calls = []

    def flaky_metric(
        model,
        inputs,
        labels,
        reduce_dim=0,
        move_device=True,
        device="cpu",
    ):
        calls.append(type(model).__name__)
        if len(calls) == 1:
            return torch.tensor(float("nan"))
        return labels.float().sum()

    with caplog.at_level(
        logging.WARNING,
        logger="dendritic_modeling.training.utils.evaluation",
    ):
        result = evaluation_utils._evaluate_metric_generic(
            metric_func=flaky_metric,
            metric_name="flaky",
            model=model,
            test_ds=dataset,
            move_device=False,
            device="cpu",
            eval_mode="materialize",
            eval_batch_size=2,
            eval_num_workers=0,
            eval_pin_memory=False,
        )

    assert result == (None, None, float(y.sum()))
    assert calls == ["Identity", "_CachedOutputModel"]
    assert "Materialized evaluation for flaky returned NaN" in caplog.text


def test_stream_model_outputs_forwards_seq_lengths_to_compatible_model():
    class SeqLengthAwareIdentity(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_seq_lengths = []

        def forward(self, x, seq_lengths=None):
            assert seq_lengths is not None
            self.seen_seq_lengths.append(seq_lengths.detach().cpu().clone())
            return x

    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    y = torch.arange(4, dtype=torch.long)
    seq_lengths = torch.tensor([1, 2, 3, 1])
    loader = DataLoader(TensorDataset(x, y, seq_lengths), batch_size=2)
    model = SeqLengthAwareIdentity()

    outputs, labels, observed_seq_lengths = evaluation_utils._stream_model_outputs(
        model=model,
        loader=loader,
        device="cpu",
    )

    torch.testing.assert_close(outputs, x)
    torch.testing.assert_close(labels, y)
    torch.testing.assert_close(observed_seq_lengths, seq_lengths)
    assert len(model.seen_seq_lengths) == 2
    torch.testing.assert_close(model.seen_seq_lengths[0], seq_lengths[:2])
    torch.testing.assert_close(model.seen_seq_lengths[1], seq_lengths[2:])


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


def test_run_model_over_analysis_batches_manages_hooks_and_model_state():
    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)
            self.seen_batches = []
            with torch.no_grad():
                self.linear.weight.copy_(torch.eye(2))

        def forward(self, x):
            self.seen_batches.append((self.training, x.detach().cpu().clone()))
            return self.linear(x)

    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    y = torch.arange(6)
    dataset = TensorDataset(x, y)
    runtime = EvaluationRuntimeConfig(mode="stream", batch_size=2, num_workers=0)
    model = RecordingModel()
    model.train()

    hook_outputs = []
    removed_counts = []

    def _hook(_module, _inputs, output):
        hook_outputs.append(output.detach().cpu().clone())

    def _attach():
        return [model.linear.register_forward_hook(_hook)]

    def _remove(handles):
        handle_list = list(handles)
        removed_counts.append(len(handle_list))
        for handle in handle_list:
            handle.remove()

    run_model_over_analysis_batches(
        model=model,
        dataset=dataset,
        device="cpu",
        runtime=runtime,
        explicit_max_samples=None,
        attach_hooks=_attach,
        remove_hooks=_remove,
    )

    assert model.training
    assert len(model.seen_batches) == 3
    assert [training for training, _batch in model.seen_batches] == [False] * 3
    torch.testing.assert_close(
        torch.cat([batch for _training, batch in model.seen_batches], dim=0),
        x,
    )
    torch.testing.assert_close(torch.cat(hook_outputs, dim=0), x)
    assert removed_counts == [1]
    assert len(model.linear._forward_hooks) == 0


def test_run_no_grad_analysis_batches_restores_state_and_batches_runtime():
    class RecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.records = []

        def forward(self, x):
            self.records.append((self.training, torch.is_grad_enabled(), x.cpu()))
            return self.linear(x)

    x = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    y = torch.arange(8)
    dataset = TensorDataset(x, y)
    runtime = EvaluationRuntimeConfig(mode="stream", batch_size=3, num_workers=0)
    model = RecordingModel()
    model.train()

    def _process(batch, device):
        _ = model(batch[0].to(device))

    run_no_grad_analysis_batches(
        model=model,
        dataset=dataset,
        device="cpu",
        runtime=runtime,
        explicit_max_samples=7,
        process_batch=_process,
    )

    assert model.training
    assert [record[0] for record in model.records] == [False, False, False]
    assert [record[1] for record in model.records] == [False, False, False]
    torch.testing.assert_close(
        torch.cat([record[2] for record in model.records], dim=0),
        x[:7],
    )


def test_run_model_over_analysis_batches_removes_hooks_after_forward_error():
    class FailingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            output = self.linear(x)
            if self.calls == 2:
                raise RuntimeError("batch failed")
            return output

    x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    y = torch.arange(6)
    dataset = TensorDataset(x, y)
    runtime = EvaluationRuntimeConfig(mode="stream", batch_size=2, num_workers=0)
    model = FailingModel()
    model.train()

    def _attach():
        return [model.linear.register_forward_hook(lambda *_args: None)]

    def _remove(handles):
        for handle in handles:
            handle.remove()

    with pytest.raises(RuntimeError, match="batch failed"):
        run_model_over_analysis_batches(
            model=model,
            dataset=dataset,
            device="cpu",
            runtime=runtime,
            explicit_max_samples=None,
            attach_hooks=_attach,
            remove_hooks=_remove,
        )

    assert model.training
    assert len(model.linear._forward_hooks) == 0


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


def test_performance_stream_forwards_worker_runtime_options(monkeypatch):
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
    recorded_kwargs = {}

    def recording_loader(dataset, **kwargs):
        recorded_kwargs.update(kwargs)
        return DataLoader(dataset, batch_size=kwargs["batch_size"], shuffle=False)

    monkeypatch.setattr(evaluation_utils, "_make_eval_loader", recording_loader)

    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=8,
        num_workers=2,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=5,
        in_order=False,
    )
    analyzer.analyze(
        model=model,
        test_ds=dataset,
        device="cpu",
        runtime=runtime,
        splits=["test"],
    )

    assert recorded_kwargs["batch_size"] == 8
    assert recorded_kwargs["num_workers"] == 2
    assert recorded_kwargs["pin_memory"] is False
    assert recorded_kwargs["persistent_workers"] is True
    assert recorded_kwargs["prefetch_factor"] == 5
    assert recorded_kwargs["in_order"] is False


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


def test_unknown_runtime_mode_raises_at_load(tmp_path):
    """A misspelled mode used to behave as 'auto' without notice."""
    config_data = {
        "experiment": {"seed": 0},
        "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
        "model": {"task": "classification"},
        "training": {},
        "outputs": {"results_dir": str(tmp_path / "outputs")},
        "analysis": {"runtime": {"final": {"mode": "streaming"}}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data))

    with pytest.raises(ValueError, match=r"mode='streaming'.*did you mean 'stream'"):
        load_config(str(config_path))


def test_runtime_mode_spellings_are_canonical():
    assert EvaluationRuntimeConfig(mode="Stream").mode == "stream"
    assert EvaluationRuntimeConfig(mode="MATERIALIZE").mode == "materialize"
    assert EvaluationRuntimeConfig().mode == "auto"
    with pytest.raises(ValueError, match="choose one of auto, materialize, stream"):
        EvaluationRuntimeConfig(mode="lazy")


def test_materialize_decision_rejects_unknown_modes_built_in_code():
    from types import SimpleNamespace

    from dendritic_modeling.analysis.utils.dataset_runtime import (
        should_materialize_dataset,
    )
    from dendritic_modeling.training.utils._evaluation.runtime import (
        _should_materialize,
    )

    dataset = TensorDataset(torch.zeros(4, 2), torch.zeros(4))
    assert should_materialize_dataset(dataset, SimpleNamespace(mode="stream")) is False
    with pytest.raises(ValueError, match="mode='bogus'"):
        should_materialize_dataset(dataset, SimpleNamespace(mode="bogus"))
    with pytest.raises(ValueError, match="mode='bogus'"):
        _should_materialize(dataset, mode="bogus")
