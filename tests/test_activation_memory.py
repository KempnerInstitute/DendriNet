"""A materialized split whose one forward pass cannot fit is streamed at once."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.utils import (
    activation_memory as memory,
    runtime as analysis_runtime,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.training.utils._evaluation import runtime as evaluation
from dendritic_modeling.training.utils.evaluation import (
    accuracy_score,
    categorical_loglikelihood_score,
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
)


class _Core(torch.nn.Module):
    """A small core whose activations grow with the batch; counts its calls."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.input_dim, self.output_dim = 12, 3
        self.hidden = torch.nn.Linear(12, width)
        self.out = torch.nn.Linear(width, 3)
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))
        self.batch_sizes: list[int] = []

    def forward(self, x):
        self.calls += 1
        self.batch_sizes.append(int(x.shape[0]))
        return self.out(torch.relu(self.hidden(x)))


def _classifier(width: int = 64) -> Classifier:
    torch.manual_seed(0)
    return Classifier(
        encoder_network=Identity(input_dim=12),
        core_network=_Core(width),
        decoder_network=Identity(input_dim=3),
    )


def _dataset(n: int = 300) -> TensorDataset:
    generator = torch.Generator().manual_seed(4)
    return TensorDataset(
        torch.rand(n, 12, generator=generator),
        torch.randint(0, 3, (n,), generator=generator),
    )


def test_no_bound_is_used_off_cuda():
    model, dataset = _classifier(), _dataset()
    items = dataset[:]
    assert memory.available_device_bytes("cpu") is None
    assert memory.forward_peak_lower_bound(model, items[0], device="cpu") is None
    cannot_fit, _ = memory.full_pass_cannot_fit(model, items, device="cpu")
    assert cannot_fit is False
    assert model.core_network.batch_sizes == []  # the model never ran
    # The CPU evaluation path is the one it was: one full batch per metric.
    _evaluate(model, dataset, eval_mode="materialize")
    assert model.core_network.batch_sizes == [300, 300]


def test_held_bytes_extrapolate_only_when_affine_in_the_batch():
    def trace(held):
        return SimpleNamespace(operations=["a", "b"], held=held)

    sizes = memory.PROBE_BATCH_SIZES  # (2, 4, 8)
    growth = memory._affine_growth(
        [trace([20, 10]), trace([40, 10]), trace([80, 10])], sizes
    )
    assert growth == [(80, 40), (10, 0)]
    # 10 bytes per example, constant 10 bytes: not linear in the batch.
    assert (
        memory._affine_growth(
            [trace([20, 10]), trace([40, 12]), trace([80, 10])], sizes
        )
        is None
    )
    mismatched = SimpleNamespace(operations=["a"], held=[80])
    assert (
        memory._affine_growth([trace([20, 10]), trace([40, 10]), mismatched], sizes)
        is None
    )


def test_probes_leave_the_model_and_random_state_unchanged():
    model = _classifier()
    inputs = _dataset()[:][0]
    before = torch.random.get_rng_state()
    growth = memory._probe_growth(model, inputs, None, torch.device("cpu"))
    assert torch.equal(before, torch.random.get_rng_state())
    assert int(model.core_network.calls) == 0  # the probes ran on a copy
    assert growth is not None
    # The activations of a linear layer grow by width * 4 bytes per example.
    assert max(per_step for _held, per_step in growth) >= 64 * 4 * 4


def test_hooks_and_batch_adaptive_modules_disable_the_bound():
    model = _classifier()
    assert not memory.has_forward_hooks(model)
    handle = model.core_network.hidden.register_forward_hook(lambda *args: None)
    assert memory.has_forward_hooks(model)
    handle.remove()
    model.core_network.out.workspace_mb = 64.0
    assert memory.batch_adaptive_modules(model) == ["core_network.out"]
    model.core_network.hidden.batch_adaptive_memory = True
    assert memory.batch_adaptive_modules(model) == [
        "core_network.hidden",
        "core_network.out",
    ]


def _evaluate(model, dataset, **kwargs):
    common = {"device": "cpu", "eval_batch_size": 64, "eval_num_workers": 0}
    common.update(kwargs)
    return (
        evaluate_accuracy(model, train_ds=dataset, **common)[0],
        evaluate_categorical_loglikelihood(model, train_ds=dataset, **common)[0],
    )


def test_a_pass_that_fits_keeps_its_single_batch(monkeypatch):
    """Runs that fit are evaluated exactly as before: one full batch."""
    calls = []

    def fits(model, items, *, device, n_examples=None):
        calls.append(len(items[0]))
        return False, "fits"

    monkeypatch.setattr(evaluation, "available_device_bytes", lambda device: 2**30)
    monkeypatch.setattr(evaluation, "full_pass_cannot_fit", fits)
    model, dataset = _classifier(), _dataset()
    accuracy, loglik = _evaluate(model, dataset, eval_mode="materialize")
    inputs, labels = dataset[:]
    assert accuracy == accuracy_score(model, inputs, labels, device="cpu").item()
    assert (
        loglik
        == categorical_loglikelihood_score(model, inputs, labels, device="cpu").item()
    )
    assert calls == [300, 300]
    assert model.core_network.batch_sizes[:2] == [300, 300]


def test_a_pass_that_cannot_fit_is_streamed_without_a_full_batch(monkeypatch, caplog):
    def cannot_fit(model, items, *, device, n_examples=None):
        return True, "the pass needs at least 9.00 GiB"

    monkeypatch.setattr(evaluation, "available_device_bytes", lambda device: 2**30)
    monkeypatch.setattr(evaluation, "full_pass_cannot_fit", cannot_fit)
    model, dataset = _classifier(), _dataset()
    with caplog.at_level(logging.INFO):
        streamed_by_bound = _evaluate(model, dataset, eval_mode="materialize")
    assert max(model.core_network.batch_sizes) == 64
    assert "Streaming evaluation for accuracy: the pass needs" in caplog.text
    assert "returned NaN" not in caplog.text

    reference = _classifier()
    assert streamed_by_bound == _evaluate(reference, dataset, eval_mode="stream")


class _CountingDataset(TensorDataset):
    """Counts reads, so a test can show the split is read as the pass reads it."""

    def __init__(self, *tensors):
        super().__init__(*tensors)
        self.reads = 0

    def __getitem__(self, index):
        self.reads += 1
        return super().__getitem__(index)


def test_analysis_batches_stream_when_their_one_batch_cannot_fit(monkeypatch):
    model = _classifier()
    dataset = _CountingDataset(*_dataset()[:])
    runtime = EvaluationRuntimeConfig(mode="materialize", batch_size=50, num_workers=0)
    helper = analysis_runtime.materialized_batch_or_streaming_runtime
    # Off CUDA no bound is used and the split is not read here.
    assert helper(model=model, dataset=dataset, device="cpu", runtime=runtime) == (
        None,
        runtime,
    )
    assert dataset.reads == 0

    monkeypatch.setattr(analysis_runtime, "available_device_bytes", lambda d: 2**30)
    monkeypatch.setattr(
        analysis_runtime,
        "full_pass_cannot_fit",
        lambda *args, **kwargs: (True, "too large"),
    )
    batch, streamed = helper(
        model=model, dataset=dataset, device="cpu", runtime=runtime
    )
    assert batch is None and streamed.mode == "stream" and runtime.mode == "materialize"

    def run():
        analysis_runtime.run_model_over_analysis_batches(
            model=model,
            dataset=dataset,
            device="cpu",
            runtime=runtime,
            explicit_max_samples=None,
            attach_hooks=list,
            remove_hooks=lambda handles: None,
        )

    run()
    assert max(model.core_network.batch_sizes) == 50

    # A pass that fits processes the split read once, as one batch.
    monkeypatch.setattr(
        analysis_runtime,
        "full_pass_cannot_fit",
        lambda *args, **kwargs: (False, "fits"),
    )
    model.core_network.batch_sizes.clear()
    dataset.reads = 0
    run()
    assert model.core_network.batch_sizes == [300]
    assert dataset.reads == 1


@pytest.mark.requires_gpu
def test_the_bound_never_exceeds_the_measured_cuda_peak():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = _classifier(width=512).cuda().eval()
    for n in (64, 1024, 4096):
        inputs = torch.rand(n, 12)
        bound = memory.forward_peak_lower_bound(model, inputs, device="cuda")
        torch.cuda.synchronize()
        batch = inputs.cuda()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            output = model(batch)
        torch.cuda.synchronize()
        measured = torch.cuda.max_memory_allocated() - base
        del output, batch
        assert bound is not None and 0 < bound <= measured


@pytest.mark.requires_gpu
def test_a_capped_cuda_device_streams_instead_of_running_out_of_memory(caplog):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = _classifier(width=4096).cuda().eval()
    dataset = _dataset(n=40000)
    total = torch.cuda.mem_get_info(0)[1]
    # A cap below the full pass (40000 x 4096 floats is 0.6 GiB) but above
    # one streamed batch.
    cap = (torch.cuda.memory_allocated(0) + 300 * 2**20) / total
    torch.cuda.set_per_process_memory_fraction(cap, 0)
    try:
        with caplog.at_level(logging.INFO):
            capped = _evaluate(model, dataset, device="cuda", eval_mode="materialize")
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
    assert "needs at least" in caplog.text
    assert "returned NaN" not in caplog.text
    assert max(model.core_network.batch_sizes) == 64
    assert capped == _evaluate(model, dataset, device="cuda", eval_mode="stream")
