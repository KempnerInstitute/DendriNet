"""Focused tests for information-analysis config helper semantics."""

from types import SimpleNamespace

from dendritic_modeling.analysis.core.information import (
    _build_information_estimator_method_params,
    _get_config_path,
    _get_config_value,
    _normalize_config_token,
    _parse_information_component_selection,
)


class _RaisingAttribute:
    @property
    def value(self):
        raise RuntimeError("attribute access failed")


class _RaisingMapping:
    def get(self, key, default=None):
        raise RuntimeError("mapping access failed")


def test_get_config_value_preserves_attribute_mapping_and_none_fallbacks():
    assert _get_config_value(None, "value", "fallback") == "fallback"
    assert _get_config_value(SimpleNamespace(value=3), "value", "fallback") == 3
    assert (
        _get_config_value(SimpleNamespace(value=None), "value", "fallback")
        == "fallback"
    )
    assert _get_config_value({"value": 4}, "value", "fallback") == 4
    assert _get_config_value({"value": None}, "value", "fallback") == "fallback"
    assert _get_config_value(_RaisingAttribute(), "value", "fallback") == "fallback"
    assert _get_config_value(_RaisingMapping(), "value", "fallback") == "fallback"


def test_get_config_path_preserves_nested_fallback_behavior():
    attr_config = SimpleNamespace(
        estimator=SimpleNamespace(kraskov=SimpleNamespace(n_neighbors=7))
    )
    dict_config = {"estimator": {"kraskov": {"n_neighbors": 11}}}
    missing_config = {"estimator": {"kraskov": {"n_neighbors": None}}}

    assert (
        _get_config_path(
            attr_config,
            ("estimator", "kraskov", "n_neighbors"),
            10,
        )
        == 7
    )
    assert (
        _get_config_path(
            dict_config,
            ("estimator", "kraskov", "n_neighbors"),
            10,
        )
        == 11
    )
    assert (
        _get_config_path(
            missing_config,
            ("estimator", "kraskov", "n_neighbors"),
            10,
        )
        == 10
    )
    assert _get_config_path(None, ("estimator",), "fallback") == "fallback"


def test_normalize_config_token_preserves_legacy_selector_normalization():
    assert _normalize_config_token(" Upstream-Unique CMI ") == "upstream_unique_cmi"
    assert _normalize_config_token("Gaussian Fisher") == "gaussian_fisher"
    assert _normalize_config_token("basic_mi") == "basic_mi"
    assert _normalize_config_token(123) == "123"


def test_parse_information_component_selection_preserves_aliases():
    selection = _parse_information_component_selection(
        [
            "basic",
            "pairwise_mi",
            "conditional",
            "pid",
            "Gaussian Fisher",
            "ablation",
            "layer-total",
            "upstream unique",
        ]
    )

    assert selection is not None
    assert selection.compute_basic_mi is True
    assert selection.compute_pairwise_mi is True
    assert selection.compute_conditional_mi is True
    assert selection.compute_pid is True
    assert selection.compute_gaussian_fisher is True
    assert selection.compute_ablation_aligned_mi is True
    assert selection.compute_layer_total_proxies is True
    assert selection.compute_upstream_unique_cmi is True


def test_parse_information_component_selection_preserves_empty_list_semantics():
    assert _parse_information_component_selection([]) is None
    assert _parse_information_component_selection(()) is None

    blank_selection = _parse_information_component_selection([" ", ""])
    assert blank_selection is not None
    assert blank_selection.compute_basic_mi is False
    assert blank_selection.compute_pairwise_mi is False
    assert blank_selection.compute_conditional_mi is False
    assert blank_selection.compute_pid is False
    assert blank_selection.compute_gaussian_fisher is False
    assert blank_selection.compute_ablation_aligned_mi is False
    assert blank_selection.compute_layer_total_proxies is False
    assert blank_selection.compute_upstream_unique_cmi is False


def test_build_information_estimator_method_params_decoder_prefers_nested_config():
    params = SimpleNamespace(
        decoder_cv_folds=3,
        decoder_C=0.5,
        decoder_standardize=False,
        decoder_seed=10,
        decoder_continuous_strategy="none",
        decoder_continuous_max_total_dim=2,
        decoder_gaussian_ridge=1e-6,
        binning_strategy="uniform",
        binning_bias_correction="none",
    )
    estimator_cfg = {
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
    }

    method_params = _build_information_estimator_method_params(
        params=params,
        estimator_cfg=estimator_cfg,
        method="decoder",
    )

    assert method_params == {
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


def test_build_information_estimator_method_params_decoder_uses_legacy_fallbacks():
    params = SimpleNamespace(
        decoder_cv_folds=4,
        decoder_C=0.75,
        decoder_standardize=False,
        decoder_seed=11,
        decoder_continuous_strategy="binned",
        decoder_continuous_max_total_dim=5,
        decoder_gaussian_ridge=3e-6,
        binning_strategy="uniform",
        binning_bias_correction="pt",
    )

    method_params = _build_information_estimator_method_params(
        params=params,
        estimator_cfg=None,
        method="decoder",
    )

    assert method_params == {
        "cv_folds": 4,
        "C": 0.75,
        "standardize": False,
        "seed": 11,
        "continuous_strategy": "binned",
        "continuous_max_dim": 5,
        "gaussian_ridge": 3e-6,
        "binning_strategy": "uniform",
        "bias_correction": "pt",
    }


def test_build_information_estimator_method_params_binned_and_other_methods():
    params = SimpleNamespace(
        binning_strategy="uniform",
        binning_bias_correction="pt",
    )
    estimator_cfg = {
        "binned": {
            "binning_strategy": "quantile",
            "bias_correction": "miller_madow",
        }
    }

    assert _build_information_estimator_method_params(
        params=params,
        estimator_cfg=estimator_cfg,
        method="Binned",
    ) == {
        "binning_strategy": "quantile",
        "bias_correction": "miller_madow",
    }
    assert _build_information_estimator_method_params(
        params=params,
        estimator_cfg=None,
        method="binned",
    ) == {
        "binning_strategy": "uniform",
        "bias_correction": "pt",
    }
    assert (
        _build_information_estimator_method_params(
            params=params,
            estimator_cfg=estimator_cfg,
            method="kraskov",
        )
        == {}
    )
