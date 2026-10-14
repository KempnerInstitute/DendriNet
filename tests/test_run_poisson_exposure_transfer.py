"""Tests for the standard frozen-checkpoint Poisson transfer runner."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.datasets.poisson import (
    POISSON_GENERATOR_DEFAULTS,
    PoissonGeneratorDataset,
)
from dendritic_modeling.datasets.standard_datasets import (
    load_poisson_generator_datasets,
)
from dendritic_modeling.datasets.unified_options import _resolve_unified_dataset_options
from dendritic_modeling.scripts.analysis import run_poisson_exposure_transfer as runner
from dendritic_modeling.scripts.analysis.run_poisson_exposure_transfer import (
    _validate_training_contract,
)


def _config(**poisson_overrides):
    params = {
        "base_dataset": "cifar10",
        "multiplicative_gain": False,
        "poisson_sampling": True,
        "stimulus_duration": 0.1,
        **poisson_overrides,
    }
    return _config_with_block(params)


def _config_with_block(params):
    dataset_params = {} if params is None else {"poisson_generator": params}
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_name="poisson_generator",
            dataset_params=dataset_params,
        )
    )


def test_poisson_transfer_training_contract_returns_declared_state() -> None:
    params, duration, dataset = _validate_training_contract(_config())
    assert params["poisson_sampling"] is True
    assert duration == pytest.approx(0.1)
    assert dataset == "cifar10"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"multiplicative_gain": True, "max_gain_factor": 4.0}, "exactly 1"),
        ({"poisson_sampling": False}, "poisson_sampling=true"),
        ({"stimulus_duration": 0.0}, "finite and positive"),
    ],
)
def test_poisson_transfer_training_contract_rejects_invalid_modes(
    overrides: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_training_contract(_config(**overrides))


def test_poisson_transfer_requires_poisson_dataset_name() -> None:
    config = _config()
    config.data.dataset_name = "cifar10"
    with pytest.raises(ValueError, match="dataset_name=poisson_generator"):
        _validate_training_contract(config)


def test_omitted_gain_switch_with_a_bound_is_rejected() -> None:
    """multiplicative_gain defaults to true: a bound of 4 trained on gains 1-4."""
    config = _config_with_block(
        {
            "base_dataset": "mnist",
            "poisson_sampling": True,
            "stimulus_duration": 1,
            "max_gain_factor": 4,
        }
    )
    with pytest.raises(ValueError, match=r"gains drawn from \[1, 4\]"):
        _validate_training_contract(config)


def test_omitted_poisson_sampling_is_the_default_true() -> None:
    config = _config_with_block({"base_dataset": "mnist", "stimulus_duration": 0.5})
    params, duration, dataset = _validate_training_contract(config)
    assert params["poisson_sampling"] is True
    assert params["multiplicative_gain"] is True
    assert duration == 0.5
    assert dataset == "mnist"


def test_missing_block_takes_every_loader_default() -> None:
    params, duration, dataset = _validate_training_contract(_config_with_block(None))
    assert params == dict(POISSON_GENERATOR_DEFAULTS)
    assert duration == 1.0
    assert dataset == "mnist"


@pytest.mark.parametrize(
    "gain",
    [
        {"multiplicative_gain": False, "max_gain_factor": 4.0},
        {"multiplicative_gain": True},
        {"multiplicative_gain": True, "fixed_gain_factor": 1.0},
        {"multiplicative_gain": True, "max_gain_factor": 1.0},
        {"multiplicative_gain": True, "max_gain_tau_ratio": 2.0},  # 2 * 0.5 = 1
        {"multiplicative_gain": True, "max_gain_factor": 1.0, "uniform_gain": False},
        {"multiplicative_gain": True, "gain_sampling": "linear_uniform"},
    ],
)
def test_unit_gain_trainings_are_accepted(gain: dict) -> None:
    config = _config_with_block({"stimulus_duration": 0.5, **gain})
    _validate_training_contract(config)


@pytest.mark.parametrize(
    "gain,match",
    [
        ({"fixed_gain_factor": 2.0}, "a fixed gain of 2"),
        ({"fixed_gain_factor": 2.0, "max_gain_factor": 1.0}, "a fixed gain of 2"),
        ({"max_gain_tau_ratio": 4.0}, r"gains drawn from \[1, 2\]"),
        ({"max_gain_factor": 1.5, "uniform_gain": False}, r"\[1, 1.5\]"),
    ],
)
def test_gain_trainings_are_rejected(gain: dict, match: str) -> None:
    config = _config_with_block({"stimulus_duration": 0.5, **gain})
    with pytest.raises(ValueError, match=match):
        _validate_training_contract(config)


def test_contract_raises_the_loader_error_for_unbounded_per_element_gain() -> None:
    config = _config_with_block({"uniform_gain": False})
    with pytest.raises(ValueError, match="uniform_gain=False requires"):
        _validate_training_contract(config)


def test_poisson_defaults_have_one_source() -> None:
    """The loader, the dataset and the transfer check read the same defaults."""
    options = _resolve_unified_dataset_options(
        {"dataset": "poisson_generator", "parameters": {}}
    )
    for key, default in POISSON_GENERATOR_DEFAULTS.items():
        assert getattr(options, key) == default, key
    for function in (PoissonGeneratorDataset.__init__, load_poisson_generator_datasets):
        signature = inspect.signature(function)
        for key, default in POISSON_GENERATOR_DEFAULTS.items():
            if key in signature.parameters and key != "base_dataset":
                assert signature.parameters[key].default == default, (function, key)


def test_gain_is_unity_matches_the_items_drawn() -> None:
    inputs = torch.rand(64, 5) + 0.5
    labels = torch.zeros(64, dtype=torch.long)
    for kwargs, unity in (
        ({}, True),
        ({"max_gain_factor": 1.0, "gain_sampling": "linear_uniform"}, True),
        ({"fixed_gain_factor": 1.0}, True),
        ({"multiplicative_gain": False, "fixed_gain_factor": 3.0}, True),
        ({"max_gain_factor": 4.0}, False),
        ({"fixed_gain_factor": 3.0}, False),
    ):
        dataset = PoissonGeneratorDataset(
            inputs, labels, poisson_sampling=False, **kwargs
        )
        assert dataset.gain_is_unity is unity, kwargs
        drawn, _ = dataset[:]
        assert torch.equal(drawn, inputs) is unity, kwargs
    base = PoissonGeneratorDataset(inputs, labels, max_gain_factor=4.0)
    assert base.with_gain_override(fixed=1.0).gain_is_unity
    assert base.with_gain_override(max_gain=1.0).gain_is_unity
    assert not base.with_gain_override(fixed=2.0).gain_is_unity


def test_rebuilt_test_split_flattens_like_the_training_loader(monkeypatch) -> None:
    """A flatten written in the dataset block wins over data.processing."""
    captured = {}

    def fake_loader(**kwargs):
        captured.update(kwargs)
        return None, None, "test"

    monkeypatch.setattr(runner, "load_poisson_generator_datasets", fake_loader)
    config = SimpleNamespace(
        data=SimpleNamespace(
            dataset_name="poisson_generator",
            dataset_params={"poisson_generator": {"flatten": False}},
            processing=SimpleNamespace(flatten=True, normalize=None),
            base_dir=None,
        ),
        experiment=SimpleNamespace(train_valid_split=0.9, seed=3),
    )
    params, duration, _ = _validate_training_contract(config)
    assert runner._test_dataset(config, params, duration) == "test"
    assert captured["flatten"] is False
    assert captured["base_dataset"] == "mnist"
    assert captured["multiplicative_gain"] is False
