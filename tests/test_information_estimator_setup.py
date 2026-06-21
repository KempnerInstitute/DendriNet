"""Tests for information estimator construction helpers."""

from types import SimpleNamespace

from dendritic_modeling.analysis.core.information_parts.information_estimator_setup import (
    _build_information_estimators,
)
from dendritic_modeling.utils.information.estimators import (
    BinnedEstimator,
    DecoderEstimator,
    EstimationMethod,
)


def test_build_information_estimators_constructs_decoder_with_nested_params():
    estimators = _build_information_estimators(
        params=SimpleNamespace(
            decoder_cv_folds=3,
            decoder_C=0.5,
            decoder_standardize=False,
            decoder_seed=10,
            decoder_continuous_strategy="none",
            decoder_continuous_max_total_dim=2,
            decoder_gaussian_ridge=1e-6,
            binning_strategy="uniform",
            binning_bias_correction="none",
        ),
        estimator_cfg={
            "decoder": {
                "cv_folds": 7,
                "C": 1.25,
                "standardize": True,
                "seed": 42,
                "continuous": {
                    "strategy": "gaussian",
                    "max_total_dim": 9,
                    "gaussian_ridge": 2e-5,
                },
            },
            "binned": {
                "binning_strategy": "quantile",
                "bias_correction": "miller_madow",
            },
        },
        method="decoder",
        n_neighbors=5,
        n_bins=13,
        copula_type="student",
        verbose=True,
    )

    assert set(estimators) == {"decoder"}
    estimator = estimators["decoder"]
    assert isinstance(estimator, DecoderEstimator)
    assert estimator.config.method == EstimationMethod.DECODER
    assert estimator.config.n_neighbors == 5
    assert estimator.config.n_bins == 13
    assert estimator.config.copula_type == "student"
    assert estimator.config.verbose is True
    assert estimator.config.method_params == {
        "cv_folds": 7,
        "C": 1.25,
        "standardize": True,
        "seed": 42,
        "continuous_strategy": "gaussian",
        "continuous_max_dim": 9,
        "gaussian_ridge": 2e-5,
        "binning_strategy": "quantile",
        "bias_correction": "miller_madow",
    }
    assert estimator.cv_folds == 7
    assert estimator.C == 1.25
    assert estimator.standardize is True
    assert estimator.seed == 42
    assert estimator.continuous_strategy == "gaussian"
    assert estimator.continuous_max_dim == 9
    assert estimator.gaussian_ridge == 2e-5
    assert estimator.binning_strategy == "quantile"
    assert estimator.bias_correction == "miller_madow"


def test_build_information_estimators_preserves_binned_key_and_legacy_params():
    estimators = _build_information_estimators(
        params=SimpleNamespace(
            binning_strategy="uniform",
            binning_bias_correction="pt",
        ),
        estimator_cfg=None,
        method="binned",
        n_neighbors=3,
        n_bins=8,
        copula_type="gaussian",
        verbose=False,
    )

    assert set(estimators) == {"binned"}
    estimator = estimators["binned"]
    assert isinstance(estimator, BinnedEstimator)
    assert estimator.config.method == EstimationMethod.BINNED
    assert estimator.config.method_params == {
        "binning_strategy": "uniform",
        "bias_correction": "pt",
    }
    assert estimator.n_bins == 8
    assert estimator.binning_strategy == "uniform"
    assert estimator.bias_correction == "pt"
