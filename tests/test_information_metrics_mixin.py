"""Focused tests for information metric helper dispatch."""

import numpy as np

from dendritic_modeling.analysis.core.information_metrics_mixin import (
    InformationMetricsMixin,
)


class _RecordingLogger:
    def __init__(self):
        self.debug_messages = []
        self.warning_messages = []

    def debug(self, message):
        self.debug_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


class _RecordingEstimator:
    def __init__(self):
        self.calls = []

    def mutual_information(self, x, y):
        self.calls.append(("mi", x.copy(), y.copy()))
        return float(len(self.calls))

    def conditional_mutual_information(self, x, y, condition):
        self.calls.append(("cmi", x.copy(), y.copy(), condition.copy()))
        return float(len(self.calls))

    def entropy(self, x):
        self.calls.append(("entropy", x.copy()))
        return 2.0


class _FailingConditionalEstimator(_RecordingEstimator):
    def conditional_mutual_information(self, x, y, condition):
        self.calls.append(("cmi", x.copy(), y.copy(), condition.copy()))
        raise RuntimeError("synthetic estimator failure")


class _MetricHarness(InformationMetricsMixin):
    def __init__(self):
        self.logger = _RecordingLogger()
        self.estimator = _RecordingEstimator()
        self.estimators = {"recording": self.estimator}
        self.method = "recording"
        self.include_vinf = True
        self.normalize_mi = False
        self.compute_ablation_aligned_mi = False
        self.compute_upstream_unique_cmi = False


def _arrays():
    return {
        "E": np.array([[1.0], [2.0], [3.0]]),
        "I": np.array([[4.0], [5.0], [6.0]]),
        "Vout": np.array([[7.0], [8.0], [9.0]]),
        "C": np.array([0, 1, 0]),
        "Vinf": np.array([[10.0], [11.0], [12.0]]),
        "Vb": np.array([[13.0], [14.0], [15.0]]),
    }


def test_estimate_mi_and_cmi_dispatch_to_active_estimator():
    harness = _MetricHarness()
    arrays = _arrays()

    mi_value = harness._estimate_mi("I(E;C)", arrays["E"], arrays["C"])
    cmi_value = harness._estimate_cmi(
        "I(E;I|C)",
        arrays["E"],
        arrays["I"],
        arrays["C"].reshape(-1, 1),
    )

    assert mi_value == 1.0
    assert cmi_value == 2.0
    assert [call[0] for call in harness.estimator.calls] == ["mi", "cmi"]
    assert "Computing I(E;C)" in harness.logger.debug_messages[0]
    assert "I(E;I|C) = 2.0000" in harness.logger.debug_messages[-1]


def test_estimate_cmi_or_zero_preserves_failure_fallback():
    harness = _MetricHarness()
    harness.estimator = _FailingConditionalEstimator()
    harness.estimators = {"recording": harness.estimator}
    arrays = _arrays()

    value = harness._estimate_cmi_or_zero(
        "I(Vb;C|E,I)",
        arrays["Vb"],
        arrays["C"].reshape(-1, 1),
        np.hstack([arrays["E"], arrays["I"]]),
    )

    assert value == 0.0
    assert [call[0] for call in harness.estimator.calls] == ["cmi"]
    assert harness.logger.warning_messages == [
        "  Failed to compute I(Vb;C|E,I): synthetic estimator failure"
    ]
    assert "I(Vb;C|E,I) = 0.0000" in harness.logger.debug_messages[-1]


def test_compute_basic_mi_preserves_keys_order_and_joint_input():
    harness = _MetricHarness()
    arrays = _arrays()

    metrics = harness._compute_basic_mi(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        arrays["C"],
        Vinf=arrays["Vinf"],
        Vb=arrays["Vb"],
    )

    assert metrics == {
        "I(E;C)": 1.0,
        "I(I;C)": 2.0,
        "I(Vb;C)": 3.0,
        "I(Vout;C)": 4.0,
        "I(Vinf;C)": 5.0,
        "I(E,I;C)": 6.0,
    }
    assert [call[0] for call in harness.estimator.calls] == ["mi"] * 6
    np.testing.assert_array_equal(
        harness.estimator.calls[-1][1],
        np.hstack([arrays["E"], arrays["I"]]),
    )
    np.testing.assert_array_equal(harness.estimator.calls[-1][2], arrays["C"])


def test_compute_pairwise_mi_preserves_keys_order_and_optional_inputs():
    harness = _MetricHarness()
    arrays = _arrays()

    metrics = harness._compute_pairwise_mi(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        Vinf=arrays["Vinf"],
        Vb=arrays["Vb"],
    )

    assert metrics == {
        "I(E;I)": 1.0,
        "I(E;Vout)": 2.0,
        "I(I;Vout)": 3.0,
        "I(E;Vb)": 4.0,
        "I(I;Vb)": 5.0,
        "I(Vb;Vout)": 6.0,
        "I(E;Vinf)": 7.0,
        "I(I;Vinf)": 8.0,
        "I(Vout;Vinf)": 9.0,
    }
    assert [call[0] for call in harness.estimator.calls] == ["mi"] * 9


def test_compute_conditional_mi_preserves_base_keys_order_and_condition_shape():
    harness = _MetricHarness()
    arrays = _arrays()

    metrics = harness._compute_conditional_mi(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        arrays["C"],
        Vinf=arrays["Vinf"],
        Vb=arrays["Vb"],
    )

    assert metrics == {
        "I(E;I|C)": 1.0,
        "I(E;Vout|C)": 2.0,
        "I(I;Vout|C)": 3.0,
        "I(E;Vb|C)": 4.0,
        "I(I;Vb|C)": 5.0,
        "I(Vb;Vout|C)": 6.0,
        "I(E;Vinf|C)": 7.0,
        "I(I;Vinf|C)": 8.0,
    }
    assert [call[0] for call in harness.estimator.calls] == ["cmi"] * 8
    for call in harness.estimator.calls:
        np.testing.assert_array_equal(call[3], arrays["C"].reshape(-1, 1))


def test_compute_conditional_mi_preserves_ablation_keys_with_upstream_inputs():
    harness = _MetricHarness()
    harness.compute_ablation_aligned_mi = True
    arrays = _arrays()

    metrics = harness._compute_conditional_mi(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        arrays["C"],
        Vinf=arrays["Vinf"],
        Vb=arrays["Vb"],
    )

    expected_keys = [
        "I(E;I|C)",
        "I(E;Vout|C)",
        "I(I;Vout|C)",
        "I(E;Vb|C)",
        "I(I;Vb|C)",
        "I(Vb;Vout|C)",
        "I(Vout;C|E)",
        "I(Vout;C|I)",
        "I(Vout;C|Vb)",
        "I(Vb;C|Vout)",
        "I(Vout;C|E,I)",
        "I(E;Vout|I,Vb)",
        "I(I;Vout|E,Vb)",
        "I(Vb;Vout|E,I)",
        "I(E;C|Vb)",
        "I(I;C|Vb)",
        "I(E,I;C|Vb)",
        "I(E;C|I,Vb)",
        "I(I;C|E,Vb)",
        "I(E;Vinf|C)",
        "I(I;Vinf|C)",
        "I(Vinf;C|Vout)",
        "I(Vout;C|Vinf)",
    ]
    assert metrics == {
        metric_key: float(index)
        for index, metric_key in enumerate(expected_keys, start=1)
    }
    assert [call[0] for call in harness.estimator.calls] == ["cmi"] * len(expected_keys)

    EI = np.hstack([arrays["E"], arrays["I"]])
    IVb = np.hstack([arrays["I"], arrays["Vb"]])
    EVb = np.hstack([arrays["E"], arrays["Vb"]])
    expected_conditions = {
        "I(Vout;C|E)": arrays["E"],
        "I(Vout;C|I)": arrays["I"],
        "I(Vout;C|Vb)": arrays["Vb"],
        "I(Vb;C|Vout)": arrays["Vout"],
        "I(Vout;C|E,I)": EI,
        "I(E;Vout|I,Vb)": IVb,
        "I(I;Vout|E,Vb)": EVb,
        "I(Vb;Vout|E,I)": EI,
        "I(E;C|Vb)": arrays["Vb"],
        "I(I;C|Vb)": arrays["Vb"],
        "I(E,I;C|Vb)": arrays["Vb"],
        "I(E;C|I,Vb)": IVb,
        "I(I;C|E,Vb)": EVb,
        "I(Vinf;C|Vout)": arrays["Vout"],
        "I(Vout;C|Vinf)": arrays["Vinf"],
    }
    for metric_key, expected_condition in expected_conditions.items():
        call = harness.estimator.calls[int(metrics[metric_key]) - 1]
        np.testing.assert_array_equal(call[3], expected_condition)


def test_compute_conditional_mi_preserves_ablation_keys_without_upstream_inputs():
    harness = _MetricHarness()
    harness.compute_ablation_aligned_mi = True
    arrays = _arrays()

    metrics = harness._compute_conditional_mi(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        arrays["C"],
    )

    expected_keys = [
        "I(E;I|C)",
        "I(E;Vout|C)",
        "I(I;Vout|C)",
        "I(Vout;C|E)",
        "I(Vout;C|I)",
        "I(E;C|I)",
        "I(I;C|E)",
        "I(E;Vout|I)",
        "I(I;Vout|E)",
        "I(Vout;C|E,I)",
    ]
    assert metrics == {
        metric_key: float(index)
        for index, metric_key in enumerate(expected_keys, start=1)
    }

    EI = np.hstack([arrays["E"], arrays["I"]])
    expected_conditions = {
        "I(Vout;C|E)": arrays["E"],
        "I(Vout;C|I)": arrays["I"],
        "I(E;C|I)": arrays["I"],
        "I(I;C|E)": arrays["E"],
        "I(E;Vout|I)": arrays["I"],
        "I(I;Vout|E)": arrays["E"],
        "I(Vout;C|E,I)": EI,
    }
    for metric_key, expected_condition in expected_conditions.items():
        call = harness.estimator.calls[int(metrics[metric_key]) - 1]
        np.testing.assert_array_equal(call[3], expected_condition)
