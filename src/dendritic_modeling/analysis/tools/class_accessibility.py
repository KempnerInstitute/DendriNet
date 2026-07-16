"""Held-out linear class-accessibility analysis for internal representations.

This module deliberately calls the quantity *class accessibility*, not mutual
information.  A fixed linear decoder can establish that class structure is
available to a downstream readout under a declared feature budget; it does not
decompose information or prove that the trained network uses that structure.

The validation-to-test estimator is the primary publication-safe path.  Its
feature support must be selected without test labels before this module is
called.  Label-shuffle controls permute only decoder-fitting labels and are
then evaluated against the untouched true test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _aligned_feature_matrix(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    targets = np.asarray(labels).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} features must be two-dimensional")
    if matrix.shape[0] != targets.shape[0]:
        raise ValueError(f"{name} feature/label counts do not match")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} features cannot be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} features must all be finite")
    return matrix, targets


def _ridge_decoder(ridge_alpha: float):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=float(ridge_alpha), class_weight="balanced"),
    )


def cross_validated_linear_decode(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
    ridge_alpha: float,
    n_label_shuffles: int,
) -> dict[str, Any]:
    """Return stratified fold-held-out ridge decoding and a shuffle floor."""

    values, labels = _aligned_feature_matrix(values, labels, name="Decoder")
    if int(folds) < 2:
        raise ValueError("At least two cross-validation folds are required")
    _, class_counts = np.unique(labels, return_counts=True)
    if int(class_counts.min()) < int(folds):
        raise ValueError(
            f"Every class needs at least {int(folds)} samples for stratified CV"
        )
    if int(n_label_shuffles) < 0:
        raise ValueError("n_label_shuffles cannot be negative")

    splitter = StratifiedKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(seed),
    )
    splits = list(splitter.split(values, labels))

    def _scores(targets: np.ndarray) -> tuple[list[float], list[float]]:
        balanced_scores: list[float] = []
        accuracy_scores: list[float] = []
        for train_indices, heldout_indices in splits:
            decoder = _ridge_decoder(ridge_alpha)
            decoder.fit(values[train_indices], targets[train_indices])
            predictions = decoder.predict(values[heldout_indices])
            balanced_scores.append(
                float(
                    balanced_accuracy_score(
                        targets[heldout_indices], predictions, adjusted=False
                    )
                )
            )
            accuracy_scores.append(
                float(accuracy_score(targets[heldout_indices], predictions))
            )
        return balanced_scores, accuracy_scores

    observed_balanced, observed_accuracy = _scores(labels)
    rng = np.random.default_rng(int(seed) + 104729)
    null_balanced: list[list[float]] = []
    null_accuracy: list[list[float]] = []
    for _ in range(int(n_label_shuffles)):
        balanced, accuracy = _scores(rng.permutation(labels))
        null_balanced.append(balanced)
        null_accuracy.append(accuracy)

    flat_null_balanced = [score for scores in null_balanced for score in scores]
    flat_null_accuracy = [score for scores in null_accuracy for score in scores]
    return {
        "estimator": "StandardScaler + RidgeClassifier",
        "interpretation": "cross-validated linear decoding; not mutual information",
        "folds": int(folds),
        "splitter": "StratifiedKFold(shuffle=True)",
        "cv_seed": int(seed),
        "ridge_alpha": float(ridge_alpha),
        "n_label_shuffles": int(n_label_shuffles),
        "observed_balanced_accuracy": float(np.mean(observed_balanced)),
        "observed_accuracy": float(np.mean(observed_accuracy)),
        "observed_balanced_accuracy_by_fold": observed_balanced,
        "observed_accuracy_by_fold": observed_accuracy,
        "label_shuffle_balanced_accuracy": (
            float(np.mean(flat_null_balanced)) if flat_null_balanced else None
        ),
        "label_shuffle_accuracy": (
            float(np.mean(flat_null_accuracy)) if flat_null_accuracy else None
        ),
        "label_shuffle_balanced_accuracy_by_shuffle_and_fold": null_balanced,
        "label_shuffle_accuracy_by_shuffle_and_fold": null_accuracy,
    }


def validation_to_test_linear_decode(
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
    ridge_alpha: float,
    n_label_shuffles: int,
) -> dict[str, Any]:
    """Fit a fixed ridge probe on validation and evaluate on untouched test."""

    validation_values, validation_labels = _aligned_feature_matrix(
        validation_values,
        validation_labels,
        name="Validation",
    )
    test_values, test_labels = _aligned_feature_matrix(
        test_values,
        test_labels,
        name="Test",
    )
    if validation_values.shape[1] != test_values.shape[1]:
        raise ValueError("Validation and test feature counts must match")
    if int(n_label_shuffles) < 0:
        raise ValueError("n_label_shuffles cannot be negative")

    def _fit_score(targets: np.ndarray) -> tuple[float, float]:
        decoder = _ridge_decoder(ridge_alpha)
        decoder.fit(validation_values, targets)
        predictions = decoder.predict(test_values)
        return (
            float(balanced_accuracy_score(test_labels, predictions, adjusted=False)),
            float(accuracy_score(test_labels, predictions)),
        )

    observed_balanced, observed_accuracy = _fit_score(validation_labels)
    rng = np.random.default_rng(int(seed) + 130363)
    null_scores = [
        _fit_score(rng.permutation(validation_labels))
        for _ in range(int(n_label_shuffles))
    ]
    return {
        "primary_estimator": "validation-fit StandardScaler + RidgeClassifier",
        "primary_interpretation": (
            "fixed validation-to-test linear decoding; not mutual information"
        ),
        "primary_fit_split": "valid",
        "primary_evaluation_split": "test",
        "primary_test_used_for_selection": False,
        "heldout_test_balanced_accuracy": observed_balanced,
        "heldout_test_accuracy": observed_accuracy,
        "heldout_test_label_shuffle_balanced_accuracy": (
            float(np.mean([score[0] for score in null_scores])) if null_scores else None
        ),
        "heldout_test_label_shuffle_accuracy": (
            float(np.mean([score[1] for score in null_scores])) if null_scores else None
        ),
        "heldout_test_label_shuffle_balanced_accuracy_by_shuffle": [
            score[0] for score in null_scores
        ],
        "heldout_test_label_shuffle_accuracy_by_shuffle": [
            score[1] for score in null_scores
        ],
        "label_shuffle_semantics": (
            "permute validation training labels; score against true test labels"
        ),
    }


@dataclass(frozen=True)
class LinearClassAccessibilityAnalyzer:
    """Configured facade for the reusable array-level accessibility estimators."""

    ridge_alpha: float = 1.0
    n_label_shuffles: int = 20
    seed: int = 0

    def validation_to_test(
        self,
        validation_values: np.ndarray,
        validation_labels: np.ndarray,
        test_values: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, Any]:
        return validation_to_test_linear_decode(
            validation_values,
            validation_labels,
            test_values,
            test_labels,
            seed=self.seed,
            ridge_alpha=self.ridge_alpha,
            n_label_shuffles=self.n_label_shuffles,
        )

    def cross_validated(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        *,
        folds: int,
    ) -> dict[str, Any]:
        return cross_validated_linear_decode(
            values,
            labels,
            folds=folds,
            seed=self.seed,
            ridge_alpha=self.ridge_alpha,
            n_label_shuffles=self.n_label_shuffles,
        )


__all__ = [
    "LinearClassAccessibilityAnalyzer",
    "cross_validated_linear_decode",
    "validation_to_test_linear_decode",
]
