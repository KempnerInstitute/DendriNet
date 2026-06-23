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
from dendritic_modeling.analysis.core.information_parts.information_fisher_mixin import (
    InformationFisherMetricsMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_helpers import (
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


class _RecordingLogger:
    def __init__(self):
        self.warning_messages = []

    def warning(self, message):
        self.warning_messages.append(message)


class _FisherHarness(InformationFisherMetricsMixin):
    include_vinf = True

    def __init__(self):
        self.logger = _RecordingLogger()
        self.proxy_calls = []
        self.dprime_shapes = []

    def _gaussian_fisher_proxy(self, x, labels):
        self.proxy_calls.append((x.copy(), labels.copy()))
        value = float(len(self.proxy_calls))
        return value, value + 0.5

    def _fisher_dprime2_pairwise_avg(self, x, labels):
        self.dprime_shapes.append((x.shape, labels.copy()))
        return 8.0 if x.shape[1] == 3 else 3.0

    def _fisher_mi_from_f(self, f):
        return f + 10.0


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


def test_compute_gaussian_fisher_preserves_variable_order_and_flow_increment():
    harness = _FisherHarness()
    E = np.array([[1.0], [2.0], [3.0]])
    I_var = np.array([[4.0], [5.0], [6.0]])
    Vout = np.array([[7.0], [8.0], [9.0]])
    C = np.array([0, 1, 0])
    Vinf = np.array([[10.0], [11.0], [12.0]])
    Vb = np.array([[13.0], [14.0], [15.0]])

    metrics = harness._compute_gaussian_fisher(E, I_var, Vout, C, Vinf=Vinf, Vb=Vb)

    assert list(metrics) == [
        "I(E;C)_fisher_F",
        "I(E;C)_fisher",
        "I(I;C)_fisher_F",
        "I(I;C)_fisher",
        "I(Vb;C)_fisher_F",
        "I(Vb;C)_fisher",
        "I(Vout;C)_fisher_F",
        "I(Vout;C)_fisher",
        "I(Vinf;C)_fisher_F",
        "I(Vinf;C)_fisher",
        "I(Vb;C|E,I)_fisher_F",
        "I(Vb;C|E,I)_fisher",
    ]
    assert metrics["I(E;C)_fisher_F"] == 1.0
    assert metrics["I(Vinf;C)_fisher"] == 5.5
    assert metrics["I(Vb;C|E,I)_fisher_F"] == 5.0
    assert metrics["I(Vb;C|E,I)_fisher"] == 15.0
    assert [call[0].shape for call in harness.proxy_calls] == [
        E.shape,
        I_var.shape,
        Vb.shape,
        Vout.shape,
        Vinf.shape,
    ]
    assert [shape for shape, _labels in harness.dprime_shapes] == [(3, 3), (3, 2)]
    assert harness.logger.warning_messages == []


def test_compute_gaussian_fisher_respects_include_vinf_flag():
    harness = _FisherHarness()
    harness.include_vinf = False
    E = np.array([[1.0], [2.0], [3.0]])
    I_var = np.array([[4.0], [5.0], [6.0]])
    Vout = np.array([[7.0], [8.0], [9.0]])
    C = np.array([0, 1, 0])
    Vinf = np.array([[10.0], [11.0], [12.0]])

    metrics = harness._compute_gaussian_fisher(E, I_var, Vout, C, Vinf=Vinf)

    assert "I(Vinf;C)_fisher_F" not in metrics
    assert [call[0].shape for call in harness.proxy_calls] == [
        E.shape,
        I_var.shape,
        Vout.shape,
    ]
