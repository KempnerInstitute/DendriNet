"""Tests for held-out class-accessibility analysis."""

from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.analysis.tools.class_accessibility import (
    ClassAccessibilityProbeAnalyzer,
    LinearClassAccessibilityAnalyzer,
    cross_validated_linear_decode,
)


def _separable(seed: int, samples_per_class: int = 20):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), samples_per_class)
    centers = np.eye(3, dtype=np.float32) * 5.0
    values = centers[labels] + rng.normal(0, 0.15, size=(len(labels), 3))
    return values.astype(np.float32), labels


def test_validation_to_test_accessibility_is_held_out_and_reproducible() -> None:
    valid_x, valid_y = _separable(1)
    test_x, test_y = _separable(2)
    analyzer = LinearClassAccessibilityAnalyzer(
        ridge_alpha=1.0,
        n_label_shuffles=4,
        seed=17,
    )
    first = analyzer.validation_to_test(valid_x, valid_y, test_x, test_y)
    second = analyzer.validation_to_test(valid_x, valid_y, test_x, test_y)
    assert first == second
    assert first["primary_fit_split"] == "valid"
    assert first["primary_evaluation_split"] == "test"
    assert first["primary_test_used_for_selection"] is False
    assert first["heldout_test_balanced_accuracy"] == pytest.approx(1.0)
    assert "not mutual information" in first["primary_interpretation"]


def test_cross_validated_accessibility_validates_feature_contract() -> None:
    values, labels = _separable(3)
    result = cross_validated_linear_decode(
        values,
        labels,
        folds=5,
        seed=11,
        ridge_alpha=1.0,
        n_label_shuffles=2,
    )
    assert result["observed_balanced_accuracy"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="feature/label counts"):
        cross_validated_linear_decode(
            values[:-1],
            labels,
            folds=5,
            seed=11,
            ridge_alpha=1.0,
            n_label_shuffles=0,
        )


def test_nonlinear_probe_recovers_xor_without_touching_test_for_selection() -> None:
    def _xor(seed: int, repeats: int = 100):
        rng = np.random.default_rng(seed)
        corners = np.asarray(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        values = np.repeat(corners, repeats, axis=0)
        values += rng.normal(0.0, 0.08, size=values.shape)
        labels = np.repeat(np.asarray([0, 1, 1, 0]), repeats)
        order = rng.permutation(labels.size)
        return values[order], labels[order]

    valid_x, valid_y = _xor(4)
    test_x, test_y = _xor(5)
    nonlinear = ClassAccessibilityProbeAnalyzer(
        probe_type="mlp",
        n_label_shuffles=1,
        seed=19,
        mlp_hidden_dims=(32, 16),
        mlp_learning_rate_init=1e-2,
        mlp_max_iter=400,
        mlp_early_stopping=False,
    ).validation_to_test(valid_x, valid_y, test_x, test_y)
    linear = ClassAccessibilityProbeAnalyzer(
        probe_type="ridge",
        n_label_shuffles=1,
        seed=19,
    ).validation_to_test(valid_x, valid_y, test_x, test_y)

    assert nonlinear["probe_family"] == "nonlinear"
    assert nonlinear["primary_test_used_for_selection"] is False
    assert nonlinear["heldout_test_balanced_accuracy"] > 0.95
    assert linear["heldout_test_balanced_accuracy"] < 0.7
