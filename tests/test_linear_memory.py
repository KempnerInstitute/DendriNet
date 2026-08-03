"""Tests for held-out linear memory-curve estimation."""

from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.analysis.utils import estimate_linear_memory_curve


def _lag_state(drive: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """Return state coordinates that exactly carry selected scalar-drive lags."""

    states = np.zeros((*drive.shape, len(lags)), dtype=np.float64)
    for coordinate, lag in enumerate(lags):
        states[:, lag:, coordinate] = drive[:, :-lag] if lag else drive
    return states


def test_recovers_held_out_lags_and_clips_integrated_capacity() -> None:
    generator = np.random.default_rng(91)
    train_drive = generator.normal(size=(4, 500))
    test_drive = generator.normal(size=(3, 400))
    train_states = _lag_state(train_drive, (1, 3))
    test_states = _lag_state(test_drive, (1, 3))

    result = estimate_linear_memory_curve(
        train_states=train_states,
        train_drive=train_drive,
        test_states=test_states,
        test_drive=test_drive,
        lags=(1, 2, 3, 4),
        ridge_alpha=1e-8,
        horizon_threshold=0.95,
    )

    assert result.r2_by_lag[[0, 2]] == pytest.approx([1.0, 1.0], abs=1e-10)
    assert np.all(result.r2_by_lag[[1, 3]] < 0.05)
    assert result.integrated_memory_capacity == pytest.approx(2.0, abs=0.05)
    assert result.threshold_horizon == 3
    assert result.metadata.train_rows == 4 * (500 - 4)
    assert result.metadata.test_rows == 3 * (400 - 4)
    assert result.metadata.target_layout == "lag_major_then_drive_coordinate"
    assert result.coefficients.shape == (2, 4, 1)
    assert result.r2_by_lag_and_drive.shape == (4, 1)


def test_vector_drive_uses_one_max_lag_window_and_train_only_means() -> None:
    generator = np.random.default_rng(17)
    train_drive = generator.normal(size=(2, 80, 2)) + np.array([5.0, -3.0])
    test_drive = generator.normal(size=(3, 65, 2)) + np.array([-4.0, 7.0])
    train_states = np.zeros((2, 80, 3), dtype=np.float64)
    test_states = np.zeros((3, 65, 3), dtype=np.float64)
    train_states[:, 2:, :2] = train_drive[:, :-2, :]
    test_states[:, 2:, :2] = test_drive[:, :-2, :]
    train_states[:, :, 2] = generator.normal(size=(2, 80))
    test_states[:, :, 2] = generator.normal(size=(3, 65))

    result = estimate_linear_memory_curve(
        train_states=train_states,
        train_drive=train_drive,
        test_states=test_states,
        test_drive=test_drive,
        lags=(0, 2),
        ridge_alpha=1e-7,
        horizon_threshold=0.9,
    )

    expected_state_mean = train_states[:, 2:, :].reshape(-1, 3).mean(axis=0)
    expected_target_means = (
        np.stack([train_drive[:, 2:, :], train_drive[:, :-2, :]], axis=2)
        .reshape(-1, 4)
        .mean(axis=0)
        .reshape(2, 2)
    )
    assert result.state_train_mean == pytest.approx(expected_state_mean)
    assert result.target_train_mean == pytest.approx(expected_target_means)
    assert result.r2_by_lag_and_drive[1] == pytest.approx([1.0, 1.0], abs=1e-9)
    assert result.metadata.drive_dimension == 2
    assert result.metadata.centering.endswith("training_means_only")
    assert result.threshold_horizon == 2


def test_estimate_is_deterministic_and_returns_read_only_arrays() -> None:
    generator = np.random.default_rng(2)
    train_drive = generator.normal(size=100)
    test_drive = generator.normal(size=90)
    train_states = _lag_state(train_drive[np.newaxis, :], (1,))[0]
    test_states = _lag_state(test_drive[np.newaxis, :], (1,))[0]
    kwargs = {
        "train_states": train_states,
        "train_drive": train_drive,
        "test_states": test_states,
        "test_drive": test_drive,
        "lags": (1, 2),
        "ridge_alpha": 1e-5,
        "horizon_threshold": 0.5,
    }

    first = estimate_linear_memory_curve(**kwargs)
    second = estimate_linear_memory_curve(**kwargs)

    assert np.array_equal(first.coefficients, second.coefficients)
    assert np.array_equal(first.r2_by_lag, second.r2_by_lag)
    assert first.to_dict() == second.to_dict()
    assert not first.coefficients.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.r2_by_lag[0] = 0.0


def test_ridge_predictions_are_invariant_to_uniform_state_units() -> None:
    generator = np.random.default_rng(712)
    train_drive = generator.normal(size=(3, 180))
    test_drive = generator.normal(size=(2, 150))
    train_states = _lag_state(train_drive, (1, 4))
    test_states = _lag_state(test_drive, (1, 4))
    common = {
        "train_drive": train_drive,
        "test_drive": test_drive,
        "lags": (1, 2, 4),
        "ridge_alpha": 1e-3,
        "horizon_threshold": 0.5,
    }

    reference = estimate_linear_memory_curve(
        train_states=train_states,
        test_states=test_states,
        **common,
    )
    scaled = estimate_linear_memory_curve(
        train_states=train_states * 1e-6,
        test_states=test_states * 1e-6,
        **common,
    )

    assert scaled.r2_by_lag == pytest.approx(reference.r2_by_lag, abs=1e-12)
    assert scaled.correlation_by_lag == pytest.approx(
        reference.correlation_by_lag,
        abs=1e-12,
    )
    assert scaled.integrated_memory_capacity == pytest.approx(
        reference.integrated_memory_capacity,
        abs=1e-12,
    )
    assert scaled.metadata.ridge_penalty_scale == pytest.approx(
        reference.metadata.ridge_penalty_scale * 1e-12,
        rel=1e-12,
    )


def test_shared_train_test_storage_is_rejected() -> None:
    drive = np.arange(20, dtype=np.float64)
    states = np.column_stack((drive, drive**2))

    with pytest.raises(ValueError, match="separate, non-overlapping"):
        estimate_linear_memory_curve(
            train_states=states,
            train_drive=drive,
            test_states=states,
            test_drive=drive,
            lags=(1,),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda arrays: arrays.__setitem__("lags", (2, 1)), "strictly increasing"),
        (
            lambda arrays: arrays["test_drive"].__setitem__(0, np.nan),
            "finite",
        ),
        (
            lambda arrays: arrays.__setitem__(
                "test_drive", np.ones_like(arrays["test_drive"])
            ),
            "targets must vary",
        ),
        (
            lambda arrays: arrays.__setitem__("horizon_threshold", 1.1),
            "at most 1",
        ),
    ],
)
def test_invalid_memory_curve_contract_fails_closed(mutation, message) -> None:
    generator = np.random.default_rng(8)
    train_drive = generator.normal(size=30)
    test_drive = generator.normal(size=25)
    arrays = {
        "train_states": _lag_state(train_drive[np.newaxis, :], (1,))[0],
        "train_drive": train_drive,
        "test_states": _lag_state(test_drive[np.newaxis, :], (1,))[0],
        "test_drive": test_drive,
        "lags": (1, 2),
        "ridge_alpha": 1e-6,
        "horizon_threshold": 0.1,
    }
    mutation(arrays)

    with pytest.raises((TypeError, ValueError), match=message):
        estimate_linear_memory_curve(**arrays)
