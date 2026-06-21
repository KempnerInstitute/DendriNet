"""Focused tests for information metric helper dispatch."""

import numpy as np

from dendritic_modeling.analysis.core import information_pid_mixin
from dendritic_modeling.analysis.core.information_metrics_mixin import (
    InformationMetricsMixin,
    _metric_variable_for_null_baseline,
    _parse_mi_key_for_null_baseline,
)
from dendritic_modeling.analysis.core.information_pid_mixin import (
    InformationPidMetricsMixin,
)


class _RecordingLogger:
    def __init__(self):
        self.debug_messages = []
        self.info_messages = []
        self.warning_messages = []

    def debug(self, message):
        self.debug_messages.append(message)

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


class _RecordingEstimator:
    def __init__(self):
        self.calls = []

    def mutual_information(self, x, y):
        self.calls.append(("mi", x.copy(), y.copy()))
        return float(len(self.calls))

    def mutual_information_null(self, x, y, *, n_shuffles, seed):
        self.calls.append(("mi_null", x.copy(), y.copy(), n_shuffles, seed))
        return float(seed), float(n_shuffles)

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
        self.mi_null_shuffles = 0
        self.mi_null_seed = 0
        self.mi_null_components = {"basic"}


class _MetricOrchestrationHarness(InformationMetricsMixin):
    def __init__(self):
        self.logger = _RecordingLogger()
        self.method = "recording"
        self.compute_basic_mi = True
        self.compute_pairwise_mi = True
        self.compute_conditional_mi = True
        self.compute_gaussian_fisher = True
        self.compute_pid = True
        self.pid_available = True
        self.calls = []

    def _record_metric_call(
        self,
        name,
        E,
        I_var,
        Vout,
        C,
        Vinf=None,
        Vb=None,
    ):
        self.calls.append(
            (
                name,
                E.shape,
                I_var.shape,
                Vout.shape,
                C.shape,
                None if Vinf is None else Vinf.shape,
                None if Vb is None else Vb.shape,
            )
        )
        return {"name": name}

    def _compute_basic_mi(self, E, I_var, Vout, C, Vinf=None, Vb=None):
        return self._record_metric_call("basic", E, I_var, Vout, C, Vinf, Vb)

    def _compute_pairwise_mi(self, E, I_var, Vout, Vinf=None, Vb=None):
        return self._record_metric_call(
            "pairwise", E, I_var, Vout, np.array([]), Vinf, Vb
        )

    def _compute_conditional_mi(self, E, I_var, Vout, C, Vinf=None, Vb=None):
        return self._record_metric_call("conditional", E, I_var, Vout, C, Vinf, Vb)

    def _compute_gaussian_fisher(self, E, I_var, Vout, C, Vinf=None, Vb=None):
        return self._record_metric_call("gaussian", E, I_var, Vout, C, Vinf, Vb)

    def _compute_pid(self, E, I_var, Vout, C):
        return self._record_metric_call("pid", E, I_var, Vout, C)


class _PidHarness(InformationPidMetricsMixin):
    def __init__(self):
        self.logger = _RecordingLogger()
        self.estimator = _RecordingEstimator()
        self.estimators = {"recording": self.estimator}
        self.method = "recording"
        self.pid_binarize = False
        self.pid_binarize_method = "median"
        self.pid_binarize_threshold = 0.5


def _arrays():
    return {
        "E": np.array([[1.0], [2.0], [3.0]]),
        "I": np.array([[4.0], [5.0], [6.0]]),
        "Vout": np.array([[7.0], [8.0], [9.0]]),
        "C": np.array([0, 1, 0]),
        "Vinf": np.array([[10.0], [11.0], [12.0]]),
        "Vb": np.array([[13.0], [14.0], [15.0]]),
    }


def test_parse_mi_key_for_null_baseline_preserves_legacy_key_rules():
    assert _parse_mi_key_for_null_baseline(None) is None
    assert _parse_mi_key_for_null_baseline("not_a_metric") is None
    assert _parse_mi_key_for_null_baseline("I(E,C)") is None
    assert _parse_mi_key_for_null_baseline(" I(E, I; C | Vb, Vout) ") == (
        ["E", "I"],
        "C",
        ["Vb", "Vout"],
    )
    assert _parse_mi_key_for_null_baseline("I(E;C)") == (["E"], "C", [])
    assert _parse_mi_key_for_null_baseline("I(E,I;Vout|C)") == (
        ["E", "I"],
        "Vout",
        ["C"],
    )


def test_metric_variable_for_null_baseline_preserves_token_availability_rules():
    arrays = _arrays()
    C_col = arrays["C"].reshape(-1, 1)
    common_kwargs = {
        "E": arrays["E"],
        "I_var": arrays["I"],
        "Vout": arrays["Vout"],
        "C_col": C_col,
        "Vinf": arrays["Vinf"],
        "Vb": arrays["Vb"],
        "has_exc_synapses": True,
        "has_inh_synapses": True,
        "has_branch_input": True,
    }

    assert _metric_variable_for_null_baseline("C", **common_kwargs) is C_col
    assert _metric_variable_for_null_baseline("E_lin", **common_kwargs) is arrays["E"]
    assert _metric_variable_for_null_baseline("I_null", **common_kwargs) is arrays["I"]
    assert (
        _metric_variable_for_null_baseline("Vout_shuffled", **common_kwargs)
        is arrays["Vout"]
    )
    assert _metric_variable_for_null_baseline("Vinf", **common_kwargs) is arrays["Vinf"]
    assert _metric_variable_for_null_baseline("Vb_lda", **common_kwargs) is arrays["Vb"]
    assert _metric_variable_for_null_baseline("unknown", **common_kwargs) is None

    assert (
        _metric_variable_for_null_baseline(
            "E",
            **{**common_kwargs, "has_exc_synapses": False},
        )
        is None
    )
    assert (
        _metric_variable_for_null_baseline(
            "I",
            **{**common_kwargs, "has_inh_synapses": False},
        )
        is None
    )
    assert (
        _metric_variable_for_null_baseline(
            "Vb",
            **{**common_kwargs, "has_branch_input": False},
        )
        is None
    )


def test_compute_metrics_preserves_category_order_logs_and_shape_normalization():
    harness = _MetricOrchestrationHarness()
    arrays = _arrays()

    results = harness.compute_metrics(
        arrays["E"].ravel(),
        arrays["I"].ravel(),
        arrays["Vout"].ravel(),
        arrays["C"].reshape(-1, 1),
        Vinf=arrays["Vinf"].ravel(),
        Vb=arrays["Vb"].ravel(),
    )

    assert list(results) == [
        "method",
        "n_samples",
        "basic_mi",
        "pairwise_mi",
        "conditional_mi",
        "gaussian_fisher",
        "pid",
    ]
    assert results["method"] == "recording"
    assert results["n_samples"] == 3
    assert harness.logger.info_messages == [
        "Computing basic MI metrics...",
        "Computing pairwise MI metrics...",
        "Computing conditional MI metrics...",
        "Computing Gaussian/Fisher proxy metrics...",
        "Computing PID metrics...",
    ]
    assert [call[0] for call in harness.calls] == [
        "basic",
        "pairwise",
        "conditional",
        "gaussian",
        "pid",
    ]
    for _, e_shape, i_shape, vout_shape, *_rest in harness.calls:
        assert e_shape == (3, 1)
        assert i_shape == (3, 1)
        assert vout_shape == (3, 1)
    assert harness.calls[0][4:] == ((3,), (3, 1), (3, 1))
    assert harness.calls[-1][4:] == ((3,), None, None)


def test_add_label_shuffle_null_baselines_uses_extracted_parsing_and_lookup():
    harness = _MetricHarness()
    harness.mi_null_shuffles = 2
    harness.mi_null_seed = 10
    harness.mi_null_components = {"basic"}
    arrays = _arrays()
    result = {
        "basic_mi": {
            "I(E;C)": 0.1,
            "I(I;C)": 0.2,
            "I(Vb;C)": 0.3,
            "not_a_metric": 0.4,
        }
    }

    harness._add_label_shuffle_null_baselines_in_place(
        result,
        E=arrays["E"],
        I_var=arrays["I"],
        Vout=arrays["Vout"],
        C=arrays["C"],
        Vinf=arrays["Vinf"],
        Vb=arrays["Vb"],
        has_exc_synapses=True,
        has_inh_synapses=True,
        has_branch_input=False,
        seed_offset=5,
    )

    assert result["basic_mi"]["I(E;C)_null"] == 15.0
    assert result["basic_mi"]["I(E;C)_null_std"] == 2.0
    assert result["basic_mi"]["I(I;C)_null"] == 16.0
    assert result["basic_mi"]["I(I;C)_null_std"] == 2.0
    assert "I(Vb;C)_null" not in result["basic_mi"]
    assert "not_a_metric_null" not in result["basic_mi"]
    assert [call[0] for call in harness.estimator.calls] == [
        "mi_null",
        "mi_null",
    ]
    assert [call[4] for call in harness.estimator.calls] == [15, 16]


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


def test_estimate_timed_metric_preserves_debug_message_shape():
    harness = _MetricHarness()

    value = harness._estimate_timed_metric("I(custom;C)", lambda: 3.25)

    assert value == 3.25
    assert harness.logger.debug_messages[0] == "  Computing I(custom;C)..."
    assert harness.logger.debug_messages[1].startswith("  I(custom;C) = 3.2500 (took ")
    assert harness.logger.debug_messages[1].endswith("s)")


def test_add_mi_metrics_preserves_spec_order_and_existing_entries():
    harness = _MetricHarness()
    arrays = _arrays()
    metrics = {"existing": -1.0}

    harness._add_mi_metrics(
        metrics,
        [
            ("I(E;C)", arrays["E"], arrays["C"]),
            ("I(I;C)", arrays["I"], arrays["C"]),
        ],
    )

    assert list(metrics) == ["existing", "I(E;C)", "I(I;C)"]
    assert metrics == {"existing": -1.0, "I(E;C)": 1.0, "I(I;C)": 2.0}
    assert [call[0] for call in harness.estimator.calls] == ["mi", "mi"]


def test_add_cmi_metrics_preserves_spec_order_and_existing_entries():
    harness = _MetricHarness()
    arrays = _arrays()
    C_col = arrays["C"].reshape(-1, 1)
    metrics = {"existing": -1.0}

    harness._add_cmi_metrics(
        metrics,
        [
            ("I(E;I|C)", arrays["E"], arrays["I"], C_col),
            ("I(E;Vout|C)", arrays["E"], arrays["Vout"], C_col),
        ],
    )

    assert list(metrics) == ["existing", "I(E;I|C)", "I(E;Vout|C)"]
    assert metrics == {"existing": -1.0, "I(E;I|C)": 1.0, "I(E;Vout|C)": 2.0}
    assert [call[0] for call in harness.estimator.calls] == ["cmi", "cmi"]
    for call in harness.estimator.calls:
        np.testing.assert_array_equal(call[3], C_col)


def test_estimate_class_entropy_for_normalization_preserves_debug_shape():
    harness = _MetricHarness()
    arrays = _arrays()

    entropy = harness._estimate_class_entropy_for_normalization(arrays["C"])

    assert entropy == 2.0
    assert [call[0] for call in harness.estimator.calls] == ["entropy"]
    np.testing.assert_array_equal(
        harness.estimator.calls[0][1],
        arrays["C"].reshape(-1, 1),
    )
    assert harness.logger.debug_messages[0] == (
        "  Computing entropy H(C) for normalization..."
    )
    assert harness.logger.debug_messages[1].startswith("  H(C) = 2.0000 (took ")
    assert harness.logger.debug_messages[1].endswith("s)")


def test_add_entropy_normalized_metrics_preserves_order_and_values():
    harness = _MetricHarness()
    arrays = _arrays()
    metrics = {"I(E;C)": 1.0, "I(I;C)": 3.0}

    harness._add_entropy_normalized_metrics(metrics, arrays["C"])

    assert list(metrics) == [
        "I(E;C)",
        "I(I;C)",
        "I(E;C)_normalized",
        "I(I;C)_normalized",
    ]
    assert metrics["I(E;C)_normalized"] == 0.5
    assert metrics["I(I;C)_normalized"] == 1.5


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


def test_add_pid_metric_preserves_key_schema_unique_terms_and_estimator_order(
    monkeypatch,
):
    harness = _PidHarness()
    arrays = _arrays()
    pid_results = {}

    def fake_pid(sources, target):
        assert sources[0] is arrays["E"]
        assert sources[1] is arrays["I"]
        assert target is arrays["Vout"]
        return 0.5, 0.25

    monkeypatch.setattr(information_pid_mixin, "pid_synergy_redundancy", fake_pid)

    harness._add_pid_metric(
        pid_results,
        source1_array=arrays["E"],
        source1_name="E",
        source2_array=arrays["I"],
        source2_name="I",
        target_array=arrays["Vout"],
        target_name="Vout",
    )

    assert pid_results == {
        "PID(E,I->Vout)": {
            "synergy": 0.5,
            "redundancy": 0.25,
            "unique_E": 0.75,
            "unique_I": 1.75,
        }
    }
    assert [call[0] for call in harness.estimator.calls] == ["mi", "mi"]
    np.testing.assert_array_equal(harness.estimator.calls[0][1], arrays["E"])
    np.testing.assert_array_equal(harness.estimator.calls[0][2], arrays["Vout"])
    np.testing.assert_array_equal(harness.estimator.calls[1][1], arrays["I"])
    np.testing.assert_array_equal(harness.estimator.calls[1][2], arrays["Vout"])
    assert harness.logger.info_messages[0] == "  Computing PID(E,I->Vout)..."
    assert harness.logger.warning_messages == []


def test_compute_pid_preserves_metric_order_and_reshaped_class_target(monkeypatch):
    harness = _PidHarness()
    arrays = _arrays()
    pid_calls = []

    def fake_pid(sources, target):
        pid_calls.append((sources[0].copy(), sources[1].copy(), target.copy()))
        return float(len(pid_calls)), 0.0

    monkeypatch.setattr(information_pid_mixin, "pid_synergy_redundancy", fake_pid)

    results = harness._compute_pid(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        arrays["C"],
    )

    assert list(results) == [
        "PID(E,I->C)",
        "PID(E,Vout->C)",
        "PID(I,Vout->C)",
        "PID(E,I->Vout)",
    ]
    np.testing.assert_array_equal(pid_calls[0][2], arrays["C"].reshape(-1, 1))
    np.testing.assert_array_equal(pid_calls[1][2], arrays["C"].reshape(-1, 1))
    np.testing.assert_array_equal(pid_calls[2][2], arrays["C"].reshape(-1, 1))
    np.testing.assert_array_equal(pid_calls[3][2], arrays["Vout"])
    assert [call[0] for call in harness.estimator.calls] == ["mi"] * 8


def test_compute_pid_preserves_binarized_preprocessing_order(monkeypatch):
    harness = _PidHarness()
    harness.pid_binarize = True
    harness.pid_binarize_method = "mean"
    arrays = {
        "E": np.array([[0.0], [2.0], [4.0]]),
        "I": np.array([[1.0], [3.0], [5.0]]),
        "Vout": np.array([[2.0], [4.0], [6.0]]),
        "C": np.array([0.0, 1.0, 3.0]),
    }
    pid_calls = []

    def fake_pid(sources, target):
        pid_calls.append((sources[0].copy(), sources[1].copy(), target.copy()))
        return float(len(pid_calls)), 0.0

    monkeypatch.setattr(information_pid_mixin, "pid_synergy_redundancy", fake_pid)

    results = harness._compute_pid(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        arrays["C"],
    )

    E_pid = np.array([[0], [0], [1]])
    I_pid = np.array([[0], [0], [1]])
    Vout_pid = np.array([[0], [0], [1]])
    C_pid = np.array([[0], [0], [1]])

    assert list(results) == [
        "PID(E,I->C)",
        "PID(E,Vout->C)",
        "PID(I,Vout->C)",
        "PID(E,I->Vout)",
    ]
    np.testing.assert_array_equal(pid_calls[0][0], E_pid)
    np.testing.assert_array_equal(pid_calls[0][1], I_pid)
    np.testing.assert_array_equal(pid_calls[0][2], C_pid)
    np.testing.assert_array_equal(pid_calls[1][0], E_pid)
    np.testing.assert_array_equal(pid_calls[1][1], Vout_pid)
    np.testing.assert_array_equal(pid_calls[1][2], C_pid)
    np.testing.assert_array_equal(pid_calls[2][0], I_pid)
    np.testing.assert_array_equal(pid_calls[2][1], Vout_pid)
    np.testing.assert_array_equal(pid_calls[2][2], C_pid)
    np.testing.assert_array_equal(pid_calls[3][0], E_pid)
    np.testing.assert_array_equal(pid_calls[3][1], I_pid)
    np.testing.assert_array_equal(pid_calls[3][2], Vout_pid)
    assert any(
        message.startswith("  Data binarization completed")
        for message in harness.logger.info_messages
    )


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


def test_compute_conditional_mi_preserves_upstream_unique_cmi_insertion_order():
    harness = _MetricHarness()
    harness.compute_ablation_aligned_mi = True
    harness.compute_upstream_unique_cmi = True
    arrays = _arrays()

    metrics = harness._compute_conditional_mi(
        arrays["E"],
        arrays["I"],
        arrays["Vout"],
        arrays["C"],
        Vinf=arrays["Vinf"],
        Vb=arrays["Vb"],
    )

    metric_keys = list(metrics)
    unique_idx = metric_keys.index("I(Vb;C|E,I)")
    assert metric_keys[unique_idx - 1] == "I(Vout;C|E,I)"
    assert metric_keys[unique_idx + 1] == "I(E;Vout|I,Vb)"
    assert metrics["I(Vb;C|E,I)"] == metrics["I(Vout;C|E,I)"] + 1.0

    EI = np.hstack([arrays["E"], arrays["I"]])
    unique_call = harness.estimator.calls[int(metrics["I(Vb;C|E,I)"]) - 1]
    assert unique_call[0] == "cmi"
    np.testing.assert_array_equal(unique_call[1], arrays["Vb"])
    np.testing.assert_array_equal(unique_call[2], arrays["C"].reshape(-1, 1))
    np.testing.assert_array_equal(unique_call[3], EI)


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
