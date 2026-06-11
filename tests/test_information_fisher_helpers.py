"""Equivalence + unit tests for the extracted Fisher/gaussian estimator helpers.

The InformationAnalyzer methods now delegate to pure functions in
information_helpers; these tests assert the delegation is exact (the wrapper
passes the analyzer's config through unchanged) and that the pure functions
behave sensibly on degenerate and well-separated inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.analysis.core.information_helpers import (
    fisher_dprime2_binary,
    fisher_dprime2_pairwise_avg,
    fisher_mi_from_f,
    gaussian_fisher_proxy,
)
from dendritic_modeling.config import InformationAnalysisParams


@pytest.fixture
def analyzer():
    return InformationAnalyzer(InformationAnalysisParams())


def _fixture_data(seed=0, n=60, d=4, n_classes=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)
    x = rng.normal(size=(n, d)) + labels[:, None]  # class-separated signal
    return x, labels


def test_wrappers_delegate_exactly(analyzer):
    """Analyzer methods must equal the free functions given the analyzer's config."""
    x, labels = _fixture_data()
    eps = float(getattr(analyzer, "gaussian_fisher_eps", 1e-8))
    agg = str(getattr(analyzer, "gaussian_fisher_aggregation", "sum"))
    mit = str(getattr(analyzer, "gaussian_fisher_mi_transform", "half_log1p"))

    assert analyzer._gaussian_fisher_proxy(x, labels) == gaussian_fisher_proxy(
        x, labels, eps=eps, aggregation=agg, mi_transform=mit
    )
    assert analyzer._fisher_dprime2_binary(
        x[labels < 2], labels[labels < 2]
    ) == fisher_dprime2_binary(x[labels < 2], labels[labels < 2], eps=eps)
    assert analyzer._fisher_dprime2_pairwise_avg(
        x, labels
    ) == fisher_dprime2_pairwise_avg(x, labels, eps=eps)
    assert analyzer._fisher_mi_from_f(2.5) == fisher_mi_from_f(2.5, mi_transform=mit)


def test_degenerate_inputs_return_zero():
    x = np.zeros((1, 3))
    labels = np.array([0])
    assert gaussian_fisher_proxy(x, labels) == (0.0, 0.0)
    assert fisher_dprime2_binary(x, labels) == 0.0
    assert fisher_dprime2_pairwise_avg(x, labels) == 0.0
    assert fisher_mi_from_f(0.0) == 0.0
    assert fisher_mi_from_f(float("nan")) == 0.0


def test_separation_increases_fisher_signal():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 30 + [1] * 30)
    near = rng.normal(size=(60, 2)) + labels[:, None] * 0.1
    far = rng.normal(size=(60, 2)) + labels[:, None] * 5.0
    f_near, mi_near = gaussian_fisher_proxy(near, labels)
    f_far, mi_far = gaussian_fisher_proxy(far, labels)
    assert f_far > f_near >= 0.0
    assert mi_far >= mi_near >= 0.0
    assert fisher_dprime2_binary(far, labels) > fisher_dprime2_binary(near, labels)


def test_mi_transforms():
    assert fisher_mi_from_f(3.0, mi_transform="none") == 3.0
    assert fisher_mi_from_f(3.0, mi_transform="log1p") == pytest.approx(np.log1p(3.0))
    assert fisher_mi_from_f(3.0, mi_transform="half_log1p") == pytest.approx(
        0.5 * np.log1p(3.0)
    )
