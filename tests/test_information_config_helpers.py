"""Focused tests for information-analysis config helper semantics."""

from types import SimpleNamespace

from dendritic_modeling.analysis.core.information import (
    _build_information_estimator_method_params,
    _build_information_gaussian_fisher_config,
    _build_information_label_shuffle_null_config,
    _build_information_pid_config,
    _build_information_signal_variant_selection,
    _get_config_path,
    _get_config_value,
    _normalize_config_token,
    _parse_information_component_selection,
    _parse_information_scope_selection,
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


def test_parse_information_scope_selection_preserves_aliases():
    selection = _parse_information_scope_selection(["neuron", "per_einet", "Per Layer"])

    assert selection is not None
    assert selection.per_neuron_analysis is True
    assert selection.per_einet_analysis is True
    assert selection.per_layer_analysis is True


def test_parse_information_scope_selection_preserves_empty_list_semantics():
    assert _parse_information_scope_selection([]) is None
    assert _parse_information_scope_selection(()) is None

    blank_selection = _parse_information_scope_selection([" ", ""])
    assert blank_selection is not None
    assert blank_selection.per_neuron_analysis is False
    assert blank_selection.per_einet_analysis is False
    assert blank_selection.per_layer_analysis is False


def test_build_information_signal_variant_selection_prefers_grouped_config():
    params = SimpleNamespace(
        compute_lda_weights=False,
        lda_n_shuffles=2,
        random_weights_seed=3,
    )

    selection = _build_information_signal_variant_selection(
        params=params,
        selection_cfg=None,
        compute_cfg={"compute_lda_weights": True},
        baselines_cfg={"random_weights": {"n_shuffles": 7, "seed": 11}},
    )

    assert selection.compute_lda_weights is True
    assert selection.lda_n_shuffles == 7
    assert selection.random_weights_seed == 11


def test_build_information_signal_variant_selection_uses_legacy_fallbacks():
    params = SimpleNamespace(
        compute_lda_weights=True,
        lda_n_shuffles=2,
        random_weights_seed=3,
    )

    legacy_selection = _build_information_signal_variant_selection(
        params=params,
        selection_cfg=None,
        compute_cfg=None,
        baselines_cfg=None,
    )
    signal_shuffle_selection = _build_information_signal_variant_selection(
        params=params,
        selection_cfg=None,
        compute_cfg=None,
        baselines_cfg={"signal_shuffle": {"n_shuffles": 5, "seed": 13}},
    )

    assert legacy_selection.compute_lda_weights is True
    assert legacy_selection.lda_n_shuffles == 2
    assert legacy_selection.random_weights_seed == 3
    assert signal_shuffle_selection.compute_lda_weights is True
    assert signal_shuffle_selection.lda_n_shuffles == 5
    assert signal_shuffle_selection.random_weights_seed == 13


def test_build_information_signal_variant_selection_applies_variant_overrides():
    params = SimpleNamespace(
        compute_lda_weights=True,
        lda_n_shuffles=2,
        random_weights_seed=3,
    )
    baselines_cfg = {"random_weights": {"n_shuffles": 7, "seed": 11}}

    network_only = _build_information_signal_variant_selection(
        params=params,
        selection_cfg={"signal_variants": ["network"]},
        compute_cfg=None,
        baselines_cfg=baselines_cfg,
    )
    lda_only = _build_information_signal_variant_selection(
        params=params,
        selection_cfg={"signal_variants": ["lda"]},
        compute_cfg=None,
        baselines_cfg=baselines_cfg,
    )
    random_only = _build_information_signal_variant_selection(
        params=params,
        selection_cfg={"signal_variants": ["shuffle"]},
        compute_cfg=None,
        baselines_cfg=baselines_cfg,
    )
    blank_only = _build_information_signal_variant_selection(
        params=params,
        selection_cfg={"signal_variants": [" ", ""]},
        compute_cfg=None,
        baselines_cfg=baselines_cfg,
    )

    assert network_only.compute_lda_weights is False
    assert network_only.lda_n_shuffles == 0
    assert network_only.random_weights_seed == 11
    assert lda_only.compute_lda_weights is True
    assert lda_only.lda_n_shuffles == 0
    assert lda_only.random_weights_seed == 11
    assert random_only.compute_lda_weights is True
    assert random_only.lda_n_shuffles == 7
    assert random_only.random_weights_seed == 11
    assert blank_only.compute_lda_weights is False
    assert blank_only.lda_n_shuffles == 0
    assert blank_only.random_weights_seed == 11


def test_build_information_signal_variant_selection_preserves_empty_list_semantics():
    params = SimpleNamespace(
        compute_lda_weights=True,
        signal_variants=[],
        lda_n_shuffles=2,
        random_weights_seed=3,
    )

    selection = _build_information_signal_variant_selection(
        params=params,
        selection_cfg={"signal_variants": []},
        compute_cfg=None,
        baselines_cfg=None,
    )

    assert selection.compute_lda_weights is True
    assert selection.lda_n_shuffles == 2
    assert selection.random_weights_seed == 3


def test_build_information_label_shuffle_null_config_prefers_grouped_config():
    params = SimpleNamespace(
        mi_null_shuffles=2,
        mi_null_seed=3,
        mi_null_components=["basic"],
    )

    null_config = _build_information_label_shuffle_null_config(
        params=params,
        baselines_cfg={
            "label_shuffle_null": {
                "n_shuffles": 7,
                "seed": 11,
                "components": ["Pairwise MI", "conditional-mi"],
            }
        },
    )

    assert null_config.mi_null_shuffles == 7
    assert null_config.mi_null_seed == 11
    assert null_config.mi_null_components == {"pairwise_mi", "conditional_mi"}


def test_build_information_label_shuffle_null_config_uses_legacy_fallbacks():
    params = SimpleNamespace(
        mi_null_shuffles=2,
        mi_null_seed=3,
        mi_null_components=["basic", "Layer Total"],
    )

    null_config = _build_information_label_shuffle_null_config(
        params=params,
        baselines_cfg=None,
    )

    assert null_config.mi_null_shuffles == 2
    assert null_config.mi_null_seed == 3
    assert null_config.mi_null_components == {"basic", "layer_total"}


def test_build_information_label_shuffle_null_config_preserves_default_components():
    missing_components = _build_information_label_shuffle_null_config(
        params=SimpleNamespace(mi_null_shuffles=None, mi_null_seed=None),
        baselines_cfg=None,
    )
    empty_components = _build_information_label_shuffle_null_config(
        params=SimpleNamespace(mi_null_components=[]),
        baselines_cfg=None,
    )
    blank_components = _build_information_label_shuffle_null_config(
        params=SimpleNamespace(mi_null_components=[" ", ""]),
        baselines_cfg=None,
    )

    assert missing_components.mi_null_shuffles == 0
    assert missing_components.mi_null_seed == 0
    assert missing_components.mi_null_components == {"basic"}
    assert empty_components.mi_null_components == {"basic"}
    assert blank_components.mi_null_components == {"basic"}


def test_build_information_pid_config_prefers_grouped_config():
    params = SimpleNamespace(
        pid_binarize=False,
        pid_binarize_method="legacy",
        pid_binarize_threshold=0.25,
    )

    pid_config = _build_information_pid_config(
        params=params,
        pid_cfg={
            "binarize": True,
            "binarize_method": "quantile",
            "binarize_threshold": 0.75,
        },
    )

    assert pid_config.pid_binarize is True
    assert pid_config.pid_binarize_method == "quantile"
    assert pid_config.pid_binarize_threshold == 0.75


def test_build_information_pid_config_uses_legacy_fallbacks():
    params = SimpleNamespace(
        pid_binarize=True,
        pid_binarize_method="mean",
        pid_binarize_threshold=0.6,
    )

    pid_config = _build_information_pid_config(params=params, pid_cfg=None)

    assert pid_config.pid_binarize is True
    assert pid_config.pid_binarize_method == "mean"
    assert pid_config.pid_binarize_threshold == 0.6


def test_build_information_pid_config_preserves_defaults():
    pid_config = _build_information_pid_config(params=SimpleNamespace(), pid_cfg=None)

    assert pid_config.pid_binarize is False
    assert pid_config.pid_binarize_method == "median"
    assert pid_config.pid_binarize_threshold == 0.5


def test_build_information_gaussian_fisher_config_normalizes_grouped_config():
    gaussian_fisher_config = _build_information_gaussian_fisher_config(
        gaussian_fisher_cfg={
            "aggregation": " Mean ",
            "mi_transform": " Log1P ",
            "ridge": 2e-6,
            "eps": 3e-6,
        }
    )

    assert gaussian_fisher_config.gaussian_fisher_aggregation == "mean"
    assert gaussian_fisher_config.gaussian_fisher_mi_transform == "log1p"
    assert gaussian_fisher_config.gaussian_fisher_eps == 2e-6


def test_build_information_gaussian_fisher_config_uses_eps_fallback():
    gaussian_fisher_config = _build_information_gaussian_fisher_config(
        gaussian_fisher_cfg={"eps": 3e-6}
    )

    assert gaussian_fisher_config.gaussian_fisher_aggregation == "sum"
    assert gaussian_fisher_config.gaussian_fisher_mi_transform == "half_log1p"
    assert gaussian_fisher_config.gaussian_fisher_eps == 3e-6


def test_build_information_gaussian_fisher_config_preserves_defaults():
    gaussian_fisher_config = _build_information_gaussian_fisher_config(
        gaussian_fisher_cfg=None
    )

    assert gaussian_fisher_config.gaussian_fisher_aggregation == "sum"
    assert gaussian_fisher_config.gaussian_fisher_mi_transform == "half_log1p"
    assert gaussian_fisher_config.gaussian_fisher_eps == 1e-8


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
