"""Tests for reproducible posthoc Poisson exposure transfer."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.poisson_exposure_transfer import (
    POISSON_EXPOSURE_DRAW_SEED_RULE,
    POSTHOC_INTEGRATION_NOTE,
    PoissonExposureTransferAnalyzer,
    poisson_exposure_draw_seed,
    poisson_exposure_duration_key,
    validate_poisson_exposure_durations,
)
from dendritic_modeling.config import PoissonExposureTransferParams


class _ToyPoissonDataset(Dataset):
    def __init__(self, duration: float) -> None:
        self.duration = float(duration)
        self.rates = torch.tensor(
            [
                [0.2, 1.0],
                [1.2, 0.1],
                [0.4, 0.8],
                [1.4, 0.2],
                [0.3, 1.1],
                [1.1, 0.3],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1])

    def __len__(self) -> int:
        return len(self.rates)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = torch.poisson(self.duration * self.rates[index]) / self.duration
        return value, self.labels[index]


class _DifferenceClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs[:, 0] - inputs[:, 1] + self.anchor
        return torch.stack((-score, score), dim=-1)


def _legacy_accuracy(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    seed: int,
    batch_size: int,
) -> tuple[int, int]:
    """Reproduce the reference manual-seed/DataLoader loop."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        correct = 0
        total = 0
        model.eval()
        for inputs, labels in loader:
            predictions = model(inputs).argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return correct, total


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_duration_keys_and_draw_seeds_are_order_independent() -> None:
    forward = [0.05, 0.1, 0.3162, 1.0, 3.1623]
    reverse = list(reversed(forward))

    assert validate_poisson_exposure_durations(forward) == tuple(forward)
    assert {
        duration: poisson_exposure_draw_seed(73000, duration, 3) for duration in forward
    } == {
        duration: poisson_exposure_draw_seed(73000, duration, 3) for duration in reverse
    }
    assert len({poisson_exposure_duration_key(value) for value in forward}) == 5
    assert POISSON_EXPOSURE_DRAW_SEED_RULE == (
        "base + 100 * round(test_duration * 1e6) + draw"
    )


@pytest.mark.parametrize(
    "durations,match",
    [
        ([], "At least one"),
        ([0.0], "finite and positive"),
        ([float("nan")], "finite and positive"),
        ([1e-8], "micro-duration resolution"),
        ([0.1, 0.1000004], "collide"),
    ],
)
def test_duration_validation_fails_closed(
    durations: list[float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_poisson_exposure_durations(durations)
    with pytest.raises(ValueError, match=match):
        PoissonExposureTransferParams(test_durations=durations)


def test_transfer_params_round_trip_through_yaml(tmp_path: Path) -> None:
    path = tmp_path / "poisson_exposure_transfer.yaml"
    expected = PoissonExposureTransferParams(
        test_durations=[0.1, 1.0],
        draws_per_duration=3,
        base_draw_seed=91,
        batch_size=4,
        max_samples=5,
        fingerprint_realized_datasets=True,
    )

    expected.save(str(path))
    observed = PoissonExposureTransferParams.load(str(path))

    assert observed.asdict() == expected.asdict()


def test_transfer_matches_frozen_runner_numerics_and_restores_rng_state() -> None:
    params = PoissonExposureTransferParams(
        test_durations=[0.2, 1.0],
        draws_per_duration=2,
        base_draw_seed=73000,
        batch_size=2,
        max_samples=5,
    )
    analyzer = PoissonExposureTransferAnalyzer(params)
    model = _DifferenceClassifier()
    model.train()
    factory_calls: list[float] = []

    def _factory(duration: float) -> Dataset:
        factory_calls.append(duration)
        # Even a custom factory that touches ambient streams may not leak state.
        random.random()
        np.random.random()
        torch.rand(1)
        return _ToyPoissonDataset(duration)

    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    result = analyzer.analyze(
        model,
        dataset_factory=_factory,
        train_duration=0.2,
        device="cpu",
    )

    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert model.training is True
    assert factory_calls == [0.2, 1.0]
    assert set(result) == {
        "test_durations",
        "draws_per_duration",
        "base_draw_seed",
        "draw_seed_rule",
        "max_samples",
        "common_random_numbers",
        "records",
    }
    assert len(result["records"]) == 4

    for record in result["records"]:
        expected_seed = poisson_exposure_draw_seed(
            params.base_draw_seed,
            record["test_duration"],
            record["draw"],
        )
        expected_correct, expected_total = _legacy_accuracy(
            model,
            Subset(
                _ToyPoissonDataset(record["test_duration"]),
                range(params.max_samples or 0),
            ),
            seed=expected_seed,
            batch_size=params.batch_size,
        )
        assert set(record) == {
            "train_duration",
            "test_duration",
            "test_duration_key",
            "draw",
            "draw_seed",
            "n_correct",
            "n_samples",
            "accuracy",
        }
        assert record["draw_seed"] == expected_seed
        assert record["n_correct"] == expected_correct
        assert record["n_samples"] == expected_total == 5
        assert record["accuracy"] == expected_correct / expected_total


def test_optional_fingerprints_describe_exact_common_draws() -> None:
    params = PoissonExposureTransferParams(
        test_durations=[0.5],
        draws_per_duration=2,
        base_draw_seed=20,
        batch_size=3,
        fingerprint_realized_datasets=True,
    )
    analyzer = PoissonExposureTransferAnalyzer(params)
    model = _DifferenceClassifier()

    first = analyzer.analyze(
        model,
        dataset_factory=_ToyPoissonDataset,
        train_duration=0.5,
    )
    second = analyzer.analyze(
        model,
        dataset_factory=_ToyPoissonDataset,
        train_duration=1.0,
    )

    assert first["fingerprint_realized_datasets"] is True
    first_by_draw = {
        record["draw"]: record["realized_dataset_fingerprint"]
        for record in first["records"]
    }
    second_by_draw = {
        record["draw"]: record["realized_dataset_fingerprint"]
        for record in second["records"]
    }
    assert first_by_draw == second_by_draw
    assert all(
        record["definition"] == "dataset-content-fingerprint-v1"
        and len(record["content_sha256"]) == 64
        for record in first_by_draw.values()
    )
    assert len({record["content_sha256"] for record in first_by_draw.values()}) == 2


def test_posthoc_tool_is_exported_but_not_manager_registered() -> None:
    assert get_analyzer_spec("poisson_exposure_transfer") is None
    assert "posthoc-only" in POSTHOC_INTEGRATION_NOTE
    assert "fixed test dataset" in POSTHOC_INTEGRATION_NOTE


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"draws_per_duration": 0}, r"\[1, 100\]"),
        ({"draws_per_duration": 101}, r"\[1, 100\]"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"max_samples": 0}, "max_samples must be positive"),
    ],
)
def test_transfer_param_counts_are_validated(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        PoissonExposureTransferParams(**kwargs)
