"""Focused tests for information estimator helper semantics."""

import numpy as np

from dendritic_modeling.utils.information.estimators import (
    BaseInformationEstimator,
    EstimationMethod,
    EstimatorConfig,
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
