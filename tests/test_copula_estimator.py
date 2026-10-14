"""Gaussian-copula mutual information against closed forms."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from dendritic_modeling.config.analysis import InformationEstimatorCopulaParams
from dendritic_modeling.utils.information.estimators import (
    CopulaEstimator,
    EstimatorConfig,
)


def _estimator(copula_type: str = "gaussian") -> CopulaEstimator:
    return CopulaEstimator(EstimatorConfig(method="copula", copula_type=copula_type))


def _bivariate(r: float, n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = r * x + math.sqrt(1 - r**2) * rng.normal(size=n)
    return x, y


@pytest.mark.parametrize("r", [0.0, 0.5, 0.9, 0.999])
def test_bivariate_gaussian_information(r):
    x, y = _bivariate(r)
    expected = -0.5 * math.log(1 - r**2)
    assert _estimator().mutual_information(x, y) == pytest.approx(expected, abs=0.02)


def test_information_ignores_monotone_transforms():
    x, y = _bivariate(0.7)
    estimator = _estimator()
    assert estimator.mutual_information(np.exp(x), y**3) == pytest.approx(
        estimator.mutual_information(x, y), abs=1e-12
    )


def test_vector_information_matches_the_gaussian_closed_form():
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.normal(size=(n, 2))
    y = x @ np.array([[0.6], [0.3]]) + 0.5 * rng.normal(size=(n, 1))
    covariance = np.array([[1.0, 0.0, 0.6], [0.0, 1.0, 0.3], [0.6, 0.3, 0.7]])
    expected = 0.5 * math.log(np.linalg.det(covariance[:2, :2]) * 0.7) - 0.5 * math.log(
        np.linalg.det(covariance)
    )
    assert _estimator().mutual_information(x, y) == pytest.approx(expected, abs=0.02)


def test_class_information_does_not_depend_on_label_order():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 4, size=8_000)
    x = np.array([0.0, 2.0, -1.0, 0.5])[labels] + rng.normal(size=labels.size)
    relabelled = np.array([3, 0, 2, 1])[labels]
    estimator = _estimator()
    information = estimator.mutual_information(x, labels)
    assert information > 0.3
    assert estimator.mutual_information(x, relabelled) == pytest.approx(information)
    assert estimator.mutual_information(labels, x) == pytest.approx(information)


def test_independent_labels_carry_no_information():
    rng = np.random.default_rng(3)
    information = _estimator().mutual_information(
        rng.normal(size=20_000), rng.integers(0, 10, size=20_000)
    )
    assert abs(information) < 0.01


@pytest.mark.parametrize("classes", [2, 10, 100])
def test_label_entropy_is_the_plug_in_entropy(classes):
    labels = np.repeat(np.arange(classes), 50)
    assert _estimator().entropy(labels) == pytest.approx(math.log(classes))


def test_continuous_entropy_is_undefined():
    with pytest.raises(ValueError, match="no entropy for continuous"):
        _estimator().entropy(np.linspace(0.0, 1.0, 100))


def test_conditional_information_of_a_markov_chain_vanishes():
    rng = np.random.default_rng(4)
    n = 20_000
    x = rng.normal(size=n)
    z = x + rng.normal(size=n)
    y = z + rng.normal(size=n)
    estimator = _estimator()
    assert estimator.mutual_information(x, y) > 0.1
    assert abs(estimator.conditional_mutual_information(x, y, z)) < 0.01


def test_conditioning_on_labels_averages_within_class_information():
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 2, size=20_000)
    r = np.where(labels == 0, 0.0, 0.8)
    x = rng.normal(size=labels.size)
    y = r * x + np.sqrt(1 - r**2) * rng.normal(size=labels.size)
    expected = 0.5 * (-0.5 * math.log(1 - 0.8**2))
    assert _estimator().conditional_mutual_information(x, y, labels) == pytest.approx(
        expected, abs=0.02
    )


def test_constant_variables_carry_no_information():
    x, _ = _bivariate(0.5)
    assert _estimator().mutual_information(x, np.ones_like(x)) == 0.0


@pytest.mark.parametrize("copula_type", ["dvc", "student"])
def test_unsupported_copula_types_raise(copula_type):
    with pytest.raises(ValueError, match="copula_type"):
        EstimatorConfig(method="copula", copula_type=copula_type)
    with pytest.raises(ValueError, match=r"estimator\.copula\.type"):
        InformationEstimatorCopulaParams(type=copula_type)


def test_the_kernel_copula_takes_two_continuous_scalars():
    x, y = _bivariate(0.8, n=2_000)
    estimator = _estimator("kernel")
    assert estimator.mutual_information(x, y) > 0.2
    with pytest.raises(ValueError, match="two continuous"):
        estimator.mutual_information(x, (y > 0).astype(int))
    with pytest.raises(ValueError, match="two continuous"):
        estimator.mutual_information(np.column_stack([x, y]), y)


def test_constructing_the_estimator_leaves_sys_path_alone():
    before = list(sys.path)
    _estimator()
    assert sys.path == before
