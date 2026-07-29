"""Tests for matched-support class mutual information."""

from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.analysis.tools.class_information import (
    MatchedClassInformationAnalyzer,
    matched_support_class_information,
)


def test_matched_class_information_detects_signal_and_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 80)
    centers = np.eye(3, dtype=np.float32) * 5.0
    values = centers[labels] + rng.normal(0, 0.2, (len(labels), 3))
    analyzer = MatchedClassInformationAnalyzer(
        n_neighbors=5,
        n_label_shuffles=3,
        seed=19,
    )

    first = analyzer.estimate(values, labels)
    second = analyzer.estimate(values, labels)

    assert first == second
    assert first["information_units"] == "bits"
    assert first["feature_count"] == 3
    assert first["standardized_per_coordinate"] is True
    assert first["preprocessing_uses_labels"] is False
    assert first["observed_mi_bits"] > 1.0
    assert first["excess_mi_over_label_shuffle_bits"] > 0.8
    assert len(first["label_shuffle_mi_bits"]) == 3


def test_matched_class_information_is_invariant_to_coordinate_units() -> None:
    rng = np.random.default_rng(31)
    labels = np.repeat(np.arange(3), 80)
    values = rng.normal(size=(labels.size, 4))
    values[:, 0] += labels * 0.8
    transformed = values * np.array([1e-3, 7.0, 120.0, 0.04]) + np.array(
        [4.0, -11.0, 2.5, 900.0]
    )

    original = matched_support_class_information(
        values,
        labels,
        n_neighbors=5,
        n_label_shuffles=5,
        seed=9,
    )
    rescaled = matched_support_class_information(
        transformed,
        labels,
        n_neighbors=5,
        n_label_shuffles=5,
        seed=9,
    )

    assert rescaled["observed_mi_bits"] == pytest.approx(
        original["observed_mi_bits"], abs=1e-10
    )
    assert rescaled["label_shuffle_mi_bits"] == pytest.approx(
        original["label_shuffle_mi_bits"], abs=1e-10
    )


def test_matched_class_information_rejects_misaligned_inputs() -> None:
    analyzer = MatchedClassInformationAnalyzer(n_label_shuffles=1)
    with pytest.raises(ValueError, match="feature/label counts"):
        analyzer.estimate(np.ones((5, 2)), np.arange(4))
