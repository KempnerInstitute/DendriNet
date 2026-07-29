from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.scripts.analysis import (
    evaluate_dendritic_rnn_stimulus_blanks as blanks,
)


def test_blank_signal_changes_only_declared_first_stimulus_channels() -> None:
    inputs = np.ones((8192, 148, 11), dtype=np.float32)
    delays = np.full(8192, 60, dtype=np.int64)

    corrupted, certificate = blanks.blank_stimulus_signal(
        inputs, delays, "first_stimulus_signal_zero"
    )

    assert np.all(corrupted[:, :20, :8] == 0.0)
    assert np.array_equal(corrupted[:, 20:, :8], inputs[:, 20:, :8])
    assert np.array_equal(corrupted[:, :, 8:], inputs[:, :, 8:])
    assert certificate["outside_mask_bitwise_unchanged"] is True


def test_blank_signal_uses_trial_specific_second_stimulus_onset() -> None:
    inputs = np.ones((8192, 148, 11), dtype=np.float32)
    delays = np.full(8192, 60, dtype=np.int64)
    delays[1] = 100

    corrupted, _ = blanks.blank_stimulus_signal(
        inputs, delays, "second_stimulus_signal_zero"
    )

    assert np.all(corrupted[0, 80:100, :8] == 0.0)
    assert np.all(corrupted[1, 120:140, :8] == 0.0)
    assert np.all(corrupted[1, 80:100, :8] == 1.0)


def test_blank_signal_rejects_unknown_policy() -> None:
    inputs = np.ones((8192, 148, 11), dtype=np.float32)
    delays = np.full(8192, 60, dtype=np.int64)

    with pytest.raises(blanks.StimulusBlankEvaluationError):
        blanks.blank_stimulus_signal(inputs, delays, "unknown")
