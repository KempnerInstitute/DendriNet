from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.analysis.core.information_parts.information_results import (
    average_mi_results,
    compute_layer_statistics,
    compute_variance_analysis,
    filter_result_metrics_in_place,
)
from dendritic_modeling.analysis.core.information_parts.information_summary import (
    format_information_summary,
)


def test_format_information_summary_preserves_legacy_section_output():
    summary = format_information_summary(
        {
            "method": "kraskov",
            "n_samples": 8,
            "basic_mi": {"I(E;C)": 0.25, "I(E;C)_normalized": 0.5},
            "pairwise_mi": {"I(E;I)": 0.125},
            "conditional_mi": {"I(E;I|C)": 0.375},
            "gaussian_fisher": {"F(E;C)": 1.25},
            "pid": {"PID(E,I->C)": {"synergy": 0.75, "redundancy": 0.125}},
        }
    )

    assert summary == (
        "Information Analysis Summary\n"
        "Method: kraskov\n"
        "Samples: 8\n"
        "\n"
        "Basic MI with C:\n"
        "  I(E;C): 0.2500\n"
        "\n"
        "Pairwise MI:\n"
        "  I(E;I): 0.1250\n"
        "\n"
        "Conditional MI given C:\n"
        "  I(E;I|C): 0.3750\n"
        "\n"
        "Gaussian/Fisher proxy:\n"
        "  F(E;C): 1.2500\n"
        "\n"
        "PID Results:\n"
        "  PID(E,I->C):\n"
        "    synergy: 0.7500\n"
        "    redundancy: 0.1250"
    )


def test_average_mi_results_handles_missing_and_nested_pid_metrics():
    results = [
        {
            "method": "kraskov",
            "n_samples": 8,
            "basic_mi": {"I(E;C)": 0.2, "I(I;C)": np.nan},
            "pid": {"PID(E,I->C)": {"synergy": 0.1}},
        },
        {
            "method": "kraskov",
            "n_samples": 8,
            "basic_mi": {"I(E;C)": 0.4, "I(Vout;C)": 0.5},
            "pid": {"PID(E,I->C)": {"synergy": 0.3, "redundancy": 0.05}},
        },
    ]

    averaged = average_mi_results(results, method="kraskov")

    assert averaged["basic_mi"]["I(E;C)"] == pytest.approx(0.3)
    assert averaged["basic_mi"]["I(E;C)_std"] == pytest.approx(0.1)
    assert averaged["basic_mi"]["I(Vout;C)"] == pytest.approx(0.5)
    assert "I(I;C)" not in averaged["basic_mi"]
    assert averaged["pid"]["PID(E,I->C)"]["synergy"] == pytest.approx(0.2)
    assert averaged["pid"]["PID(E,I->C)"]["redundancy"] == pytest.approx(0.05)


def test_average_mi_results_covers_metric_categories_and_skips_nonfinite_values():
    results = [
        {
            "method": "kraskov",
            "n_samples": 8,
            "pairwise_mi": {"I(E;I)": 0.2, "I(E;bad)": "skip"},
            "conditional_mi": {"I(E;I|C)": np.inf},
            "soma_coupling_mi": {"I(I;S)": 0.5},
            "gaussian_fisher": {"F(E;C)": 0.7},
        },
        {
            "method": "kraskov",
            "n_samples": 8,
            "pairwise_mi": {"I(E;I)": 0.4},
            "conditional_mi": {"I(E;I|C)": 1.2},
            "soma_coupling_mi": {"I(I;S)": 0.9},
            "gaussian_fisher": {"F(E;C)": 0.9},
        },
    ]

    averaged = average_mi_results(results, method="kraskov")

    assert averaged["pairwise_mi"]["I(E;I)"] == pytest.approx(0.3)
    assert averaged["pairwise_mi"]["I(E;I)_std"] == pytest.approx(0.1)
    assert "I(E;bad)" not in averaged["pairwise_mi"]
    assert averaged["conditional_mi"]["I(E;I|C)"] == pytest.approx(1.2)
    assert averaged["soma_coupling_mi"]["I(I;S)"] == pytest.approx(0.7)
    assert averaged["gaussian_fisher"]["F(E;C)"] == pytest.approx(0.8)


def test_compute_layer_statistics_adds_class_targeted_bounded_proxies():
    branch_results = [
        {"basic_mi": {"I(E;C)": 0.2, "I(E;I)": 0.7}},
        {"basic_mi": {"I(E;C)": 0.3, "I(E;I)": 0.9}},
    ]

    stats = compute_layer_statistics(branch_results, entropy_C=1.0, topk=1)

    assert stats["I(E;C)_mean"] == pytest.approx(0.25)
    assert stats["I(E;C)_sum_mean"] == pytest.approx(0.5)
    assert stats["I(E;C)_sum_clipped_mean"] == pytest.approx(0.5)
    assert stats["I(E;C)_union_mean"] == pytest.approx(1.0 * (1.0 - 0.8 * 0.7))
    assert stats["I(E;C)_top1_sum_mean"] == pytest.approx(0.3)
    assert stats["I(E;C)_scalar_mi_sum_proxy_mean"] == pytest.approx(0.5)
    assert stats["I(E;C)_top1_scalar_mi_sum_proxy_mean"] == pytest.approx(0.3)
    assert stats["I(E;C)_independent_evidence_union_proxy_mean"] == pytest.approx(
        1.0 * (1.0 - 0.8 * 0.7)
    )
    assert "I(E;I)_sum_mean" not in stats


def test_compute_layer_statistics_uses_reported_null_standard_deviations():
    branch_results = [
        {"basic_mi": {"I(E;C)_null": 0.2, "I(E;C)_null_std": 0.3}},
        {"basic_mi": {"I(E;C)_null": 0.4, "I(E;C)_null_std": 0.5}},
    ]

    stats = compute_layer_statistics(branch_results)

    assert stats["I(E;C)_null_mean"] == pytest.approx(0.3)
    assert stats["I(E;C)_null_std"] == pytest.approx(np.sqrt((0.3**2 + 0.5**2) / 2))


def test_filter_result_metrics_in_place_uses_variable_availability():
    result = {
        "basic_mi": {
            "I(E;C)": 0.1,
            "I(I;C)": 0.2,
            "I(Vb;C)": 0.3,
            "I(Vout;C)": 0.4,
        }
    }

    filter_result_metrics_in_place(
        result,
        has_exc_synapses=True,
        has_inh_synapses=False,
        has_branch_input=False,
    )

    assert result["basic_mi"] == {"I(E;C)": 0.1, "I(Vout;C)": 0.4}


def test_compute_variance_analysis_squares_layer_stds():
    variance = compute_variance_analysis(
        {
            "computation_level": "single_branch",
            "layer_statistics": {"layer0": {"I(E;C)_std": 0.25, "I(E;C)_mean": 0.5}},
        }
    )

    assert variance["layer_variances"]["layer0"]["I(E;C)_variance"] == pytest.approx(
        0.0625
    )
    assert variance["variance_interpretation"]["computation_level"] == "single_branch"
