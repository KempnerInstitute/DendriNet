import torch

from dendritic_modeling.analysis.tools.compartment_snr import (
    CompartmentSNRAnalyzer,
    _aggregate_non_depth_values,
    _summarize_population_values,
)
from dendritic_modeling.config.analysis import CompartmentSNRAnalysisParams


def _expected_population_stats(prefix, values):
    tensor = torch.tensor(values)
    return {
        f"{prefix}_mean": tensor.mean().item(),
        f"{prefix}_std": tensor.std(correction=0).item(),
    }


def test_compartment_snr_aggregate_helper_excludes_depth():
    records = [
        {"depth": 0, "exc_d'^2": 1.0, "inh_d'^2": 2.0},
        {"depth": 0, "exc_d'^2": 3.0, "inh_d'^2": 4.0},
    ]

    assert _aggregate_non_depth_values(records) == {
        "exc_d'^2": [1.0, 3.0],
        "inh_d'^2": [2.0, 4.0],
    }


def test_compartment_snr_population_summary_helper_matches_current_reductions():
    actual = _summarize_population_values(
        {
            "vout_d_total": [1.0, 3.0],
            "vout_d_max": [2.0, 4.0],
        },
        {"depth": 1},
    )

    assert actual == {
        "depth": 1,
        **_expected_population_stats("vout_d_total", [1.0, 3.0]),
        **_expected_population_stats("vout_d_max", [2.0, 4.0]),
    }


def test_compartment_snr_branch_aggregation_matches_population_reductions():
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())
    branch_statistics = {
        "layer0": {
            0: {"depth": 0, "exc_d'^2": 1.0, "inh_d'^2": 2.0},
            1: {"depth": 0, "exc_d'^2": 3.0, "inh_d'^2": 4.0},
        },
        "layer1": {
            0: {"depth": 1, "exc_d'^2": 5.0, "inh_d'^2": 6.0},
            1: {"depth": 1, "exc_d'^2": 7.0, "inh_d'^2": 8.0},
        },
    }

    layer_stats = analyzer._layer_branch_aggregation(branch_statistics)
    global_stats = analyzer._global_branch_aggregation(branch_statistics)

    assert layer_stats == {
        "layer0": {
            "depth": 0,
            **_expected_population_stats("exc_d'^2", [1.0, 3.0]),
            **_expected_population_stats("inh_d'^2", [2.0, 4.0]),
        },
        "layer1": {
            "depth": 1,
            **_expected_population_stats("exc_d'^2", [5.0, 7.0]),
            **_expected_population_stats("inh_d'^2", [6.0, 8.0]),
        },
    }
    assert global_stats == {
        **_expected_population_stats("exc_d'^2", [1.0, 3.0, 5.0, 7.0]),
        **_expected_population_stats("inh_d'^2", [2.0, 4.0, 6.0, 8.0]),
    }


def test_compartment_snr_global_layer_aggregation_matches_population_reductions():
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())
    layer_statistics = {
        "layer0": {"depth": 0, "vout_d_total": 1.0, "vout_d_max": 2.0},
        "layer1": {"depth": 1, "vout_d_total": 3.0, "vout_d_max": 4.0},
    }

    global_stats = analyzer._global_layer_aggregation(layer_statistics)

    assert global_stats == {
        **_expected_population_stats("vout_d_total", [1.0, 3.0]),
        **_expected_population_stats("vout_d_max", [2.0, 4.0]),
    }


def test_compartment_snr_univariate_discriminability_alias_is_numerically_identical():
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams())
    data = torch.tensor([0.0, 0.2, 1.0, 1.2])
    labels = torch.tensor([0, 0, 1, 1])

    corrected = analyzer._compute_discriminability_univariate(
        data,
        labels,
        "exc",
        {},
    )
    legacy = analyzer._compute_discriminality_univariate(
        data,
        labels,
        "exc",
        {},
    )

    assert corrected == legacy
    assert corrected["exc_d'^2"] > 0.0
