"""Tests for the standard frozen-checkpoint Poisson transfer runner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    return SimpleNamespace(
        data=SimpleNamespace(
            dataset_name="poisson_generator",
            dataset_params={"poisson_generator": params},
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
        ({"multiplicative_gain": True}, "multiplicative_gain=false"),
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
