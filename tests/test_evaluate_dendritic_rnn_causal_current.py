from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_causal_current import (
    CausalCurrentEvaluationError,
    build_phase_masks,
    load_study_config,
    target_margin,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_causal_current_config_loads() -> None:
    config, path = load_study_config(
        REPO_ROOT / "configs/dendritic_rnn/causal_current_pilot_v1.yaml"
    )
    assert path.is_file()
    assert config["interventions"]["soma_relative_depths"] == [0, 1, 2]
    assert set(config["interventions"]["endpoints"]) == {
        "ff_write",
        "rec_late_delay",
    }


def test_phase_masks_follow_each_variable_delay() -> None:
    lengths = np.array([108, 148], dtype=np.int64)
    masks = build_phase_masks(
        lengths,
        timesteps=148,
        stimulus_duration=20,
        response_duration=8,
        late_delay_steps=20,
    )
    assert masks["first_stimulus"].shape == (2, 148)
    assert np.array_equal(masks["first_stimulus"].sum(axis=1), [20, 20])
    assert np.array_equal(masks["late_delay"].sum(axis=1), [20, 20])
    assert np.flatnonzero(masks["late_delay"][0])[[0, -1]].tolist() == [60, 79]
    assert np.flatnonzero(masks["late_delay"][1])[[0, -1]].tolist() == [100, 119]
    assert not np.any(masks["first_stimulus"] & masks["late_delay"])


def test_phase_masks_reject_delays_shorter_than_window() -> None:
    with pytest.raises(CausalCurrentEvaluationError, match="cannot support"):
        build_phase_masks(
            np.array([60], dtype=np.int64),
            timesteps=60,
            stimulus_duration=20,
            response_duration=8,
            late_delay_steps=20,
        )


def test_binary_target_margin() -> None:
    logits = np.array([[1.0, 3.0], [4.0, -1.0]], dtype=np.float32)
    labels = np.array([1, 0], dtype=np.int64)
    assert np.array_equal(target_margin(logits, labels), np.array([2.0, 5.0]))
