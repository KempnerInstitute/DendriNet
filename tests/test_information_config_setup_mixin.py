"""Focused tests for information analyzer config setup assignments."""

from types import SimpleNamespace

from dendritic_modeling.analysis.core.information_parts.information_config_setup_mixin import (
    InformationConfigSetupMixin,
)


class _WarningLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args):
        self.warnings.append((message, args))


class _ConfigHarness(InformationConfigSetupMixin):
    def __init__(self):
        self.logger = _WarningLogger()


def test_config_setup_mixin_assigns_core_null_pid_and_metric_options():
    harness = _ConfigHarness()
    params = SimpleNamespace(
        method="binned",
        max_samples=100,
        n_neighbors=3,
        n_bins=12,
        copula_type="legacy",
        mi_null_shuffles=2,
        mi_null_seed=3,
        mi_null_components=["basic"],
        compute_basic_mi=True,
        compute_pairwise_mi=False,
        compute_conditional_mi=True,
        compute_pid=True,
        pid_binarize=False,
        pid_binarize_method="legacy",
        pid_binarize_threshold=0.25,
        compute_ablation_aligned_mi=False,
        compute_upstream_unique_cmi=True,
        compute_layer_total_proxies=False,
        layer_total_topk=7,
    )

    harness._configure_estimator_core(
        params=params,
        estimator_cfg={
            "method": "decoder",
            "kraskov": {"n_neighbors": 9},
            "binned": {"n_bins": 21},
            "copula": {"type": "student"},
        },
        compute_cfg={"max_samples": 250},
    )
    harness._configure_label_shuffle_null(
        params=params,
        baselines_cfg={
            "label_shuffle_null": {
                "n_shuffles": 7,
                "seed": 11,
                "components": ["Pairwise MI", "conditional-mi"],
            }
        },
    )
    harness._configure_legacy_component_selection(params=params)
    harness._configure_pid_preprocessing(
        params=params,
        pid_cfg={
            "binarize": True,
            "binarize_method": "quantile",
            "binarize_threshold": 0.75,
        },
    )
    harness._configure_ablation_aligned_metrics(
        params=params,
        compute_cfg={"layer_total_topk": 13},
    )
    harness._configure_gaussian_fisher_metrics(
        gaussian_fisher_cfg={
            "aggregation": " Mean ",
            "mi_transform": " Log1P ",
            "ridge": 2e-6,
        }
    )

    assert harness.method == "decoder"
    assert harness.max_samples == 250
    assert harness.n_neighbors == 9
    assert harness.n_bins == 21
    assert harness.copula_type == "student"
    assert harness.mi_null_shuffles == 7
    assert harness.mi_null_seed == 11
    assert harness.mi_null_components == {"pairwise_mi", "conditional_mi"}
    assert harness.compute_basic_mi is True
    assert harness.compute_pairwise_mi is False
    assert harness.compute_conditional_mi is True
    assert harness.compute_pid is True
    assert harness.compute_gaussian_fisher is False
    assert harness.compute_soma_coupling_mi is False
    assert harness.pid_binarize is True
    assert harness.pid_binarize_method == "quantile"
    assert harness.pid_binarize_threshold == 0.75
    assert harness.compute_ablation_aligned_mi is False
    assert harness.compute_upstream_unique_cmi is True
    assert harness.compute_layer_total_proxies is False
    assert harness.layer_total_topk == 13
    assert harness.gaussian_fisher_aggregation == "mean"
    assert harness.gaussian_fisher_mi_transform == "log1p"
    assert harness.gaussian_fisher_eps == 2e-6


def test_config_setup_mixin_component_selection_preserves_empty_override_semantics():
    harness = _ConfigHarness()
    params = SimpleNamespace(
        compute_basic_mi=True,
        compute_pairwise_mi=True,
        compute_conditional_mi=False,
        compute_pid=False,
    )

    harness._configure_legacy_component_selection(params=params)
    harness._configure_component_selection_override(
        params=params,
        selection_cfg={"components": []},
    )

    assert harness.compute_basic_mi is True
    assert harness.compute_pairwise_mi is True
    assert harness.compute_conditional_mi is False
    assert harness.compute_pid is False
    assert harness.compute_gaussian_fisher is False

    harness._configure_component_selection_override(
        params=params,
        selection_cfg={
            "components": [
                "conditional",
                "Gaussian Fisher",
                "ablation",
                "layer-total",
                "upstream unique",
                "soma coupling",
            ]
        },
    )

    assert harness.compute_basic_mi is False
    assert harness.compute_pairwise_mi is False
    assert harness.compute_conditional_mi is True
    assert harness.compute_pid is False
    assert harness.compute_gaussian_fisher is True
    assert harness.compute_ablation_aligned_mi is True
    assert harness.compute_layer_total_proxies is True
    assert harness.compute_upstream_unique_cmi is True
    assert harness.compute_soma_coupling_mi is True


def test_config_setup_mixin_disables_pid_request_with_warning():
    harness = _ConfigHarness()
    harness.compute_pid = True

    harness._disable_pid_if_requested()

    assert harness.compute_pid is False
    assert harness.logger.warnings == [
        (
            "PID requested via selection.components, but PID is currently disabled "
            "and will be skipped.",
            (),
        )
    ]


def test_config_setup_mixin_runtime_options_warn_on_invalid_output_units():
    harness = _ConfigHarness()

    harness._configure_runtime_options(
        params=SimpleNamespace(output_units="bits"),
        compute_cfg={
            "include_vinf": True,
            "normalize_mi": True,
            "output_units": " Bad Units ",
            "analysis_split": "valid",
            "verbose": True,
        },
    )

    assert harness.include_vinf is True
    assert harness.normalize_mi is True
    assert harness.output_units == "bits"
    assert harness.analysis_split == "validation"
    assert harness.verbose is True
    assert harness.logger.warnings == [
        ("Unknown output_units=%r; falling back to 'bits'.", ("bad_units",))
    ]


def test_config_setup_mixin_granularity_and_scope_assignments():
    harness = _ConfigHarness()
    params = SimpleNamespace(
        computation_level="single_branch",
        branch_aggregation="mean",
        branch_sample_count_per_layer=None,
        branch_sample_seed=0,
        branch_sample_strategy="uniform",
        per_neuron_analysis=False,
        per_einet_analysis=False,
        per_layer_analysis=False,
    )

    harness._configure_granularity_defaults(
        params=params,
        compute_cfg={
            "computation_level": "all_branch",
            "branch_aggregation": "sample",
            "branch_sample_count_per_layer": 16,
            "branch_sample_seed": 23,
            "branch_sample_strategy": "parent-soma-balanced",
            "per_neuron_analysis": True,
            "per_einet_analysis": False,
            "per_layer_analysis": False,
        },
    )
    harness._configure_scope_selection_override(
        params=params,
        selection_cfg={"scopes": [], "views": ["per_einet", "Per Layer"]},
    )

    assert harness.computation_level == "all_branch"
    assert harness.branch_aggregation == "sample"
    assert harness.branch_sample_count_per_layer == 16
    assert harness.branch_sample_seed == 23
    assert harness.branch_sample_strategy == "parent_soma_balanced"
    assert harness.per_neuron_analysis is False
    assert harness.per_einet_analysis is True
    assert harness.per_layer_analysis is True


def test_config_setup_mixin_signal_variants_assign_grouped_overrides():
    harness = _ConfigHarness()
    params = SimpleNamespace(
        compute_lda_weights=True,
        lda_n_shuffles=2,
        random_weights_seed=3,
    )

    harness._configure_signal_variants(
        params=params,
        selection_cfg={"signal_variants": ["shuffle"]},
        compute_cfg=None,
        baselines_cfg={"random_weights": {"n_shuffles": 7, "seed": 11}},
    )

    assert harness.compute_lda_weights is True
    assert harness.lda_n_shuffles == 7
    assert harness.random_weights_seed == 11
