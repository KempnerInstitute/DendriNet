"""Focused tests for information estimator helper semantics."""

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from dendritic_modeling.utils.information.estimators import (
    BaseInformationEstimator,
    BinnedEstimator,
    EstimationMethod,
    EstimatorConfig,
)
from dendritic_modeling.utils.information.nearest_neighbors import (
    compute_mi_cc,
    compute_mi_cd,
)


class _ChainRuleEstimator(BaseInformationEstimator):
    def __init__(self, mi_values):
        super().__init__(EstimatorConfig(method=EstimationMethod.KRASKOV))
        self._mi_values = iter(mi_values)
        self.calls = []

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        self.calls.append((x.copy(), y.copy()))
        return float(next(self._mi_values))

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        return self._chain_rule_cmi(x, y, z)


def test_chain_rule_cmi_preserves_backend_call_order_and_concatenation():
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[4.0], [5.0], [6.0]])
    z = np.array([[7.0], [8.0], [9.0]])
    estimator = _ChainRuleEstimator([1.25, 0.5])

    cmi = estimator.conditional_mutual_information(x, y, z)

    assert cmi == 0.75
    assert len(estimator.calls) == 2
    np.testing.assert_array_equal(estimator.calls[0][0], x)
    np.testing.assert_array_equal(estimator.calls[0][1], np.hstack([y, z]))
    np.testing.assert_array_equal(estimator.calls[1][0], x)
    np.testing.assert_array_equal(estimator.calls[1][1], z)


def test_chain_rule_cmi_preserves_nonnegative_clamp():
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[4.0], [5.0], [6.0]])
    z = np.array([[7.0], [8.0], [9.0]])
    estimator = _ChainRuleEstimator([0.25, 0.75])

    assert estimator.conditional_mutual_information(x, y, z) == 0.0


def test_binned_cmi_applies_miller_madow_sampling_bias_correction():
    x = np.array([0, 0, 1, 1, 0, 0, 1, 1]).reshape(-1, 1)
    y = x.copy()
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1]).reshape(-1, 1)
    uncorrected = BinnedEstimator(
        EstimatorConfig(
            method=EstimationMethod.BINNED,
            n_bins=10,
            method_params={"bias_correction": "none"},
        )
    )
    corrected = BinnedEstimator(
        EstimatorConfig(
            method=EstimationMethod.BINNED,
            n_bins=10,
            method_params={"bias_correction": "miller_madow"},
        )
    )

    uncorrected_cmi = uncorrected.conditional_mutual_information(x, y, z)
    corrected_cmi = corrected.conditional_mutual_information(x, y, z)

    # k_xz=4, k_yz=4, k_z=2, k_xyz=4, N=8.
    expected_correction = (4 + 4 - 2 - 4) / (2 * 8)
    assert np.isclose(corrected_cmi - uncorrected_cmi, expected_correction)


def test_mixed_knn_mi_matches_sklearn_neighbor_rank_semantics():
    rng = np.random.default_rng(20260729)
    labels = np.repeat(np.arange(4), 500)
    signal = labels.astype(float)[:, None] * 0.2
    values = signal + rng.normal(size=(labels.size, 1))

    observed = compute_mi_cd(values, labels, n_neighbors=10)
    expected = mutual_info_classif(
        values,
        labels,
        discrete_features=False,
        n_neighbors=10,
        random_state=0,
    )[0]

    assert np.isclose(observed, expected, atol=2e-4)


def test_continuous_knn_mi_recovers_correlated_gaussian_information():
    rng = np.random.default_rng(314159)
    correlation = 0.7
    x = rng.normal(size=(6000, 1))
    y = correlation * x + np.sqrt(1.0 - correlation**2) * rng.normal(size=(6000, 1))

    observed = compute_mi_cc(x, y, n_neighbors=10)
    expected = -0.5 * np.log(1.0 - correlation**2)

    assert np.isclose(observed, expected, atol=0.04)
