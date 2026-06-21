import torch

from dendritic_modeling.analysis.tools.compartment_statistics import (
    CompartmentStatisticsAnalyzer,
)
from dendritic_modeling.config.analysis import CompartmentStatisticsAnalysisParams


def _expected_tensor_stats(prefix, values):
    tensor = torch.tensor(values)
    return {
        f"{prefix}_mean": tensor.mean().item(),
        f"{prefix}_var": tensor.var().item(),
        f"{prefix}_std": tensor.std().item(),
        f"{prefix}_min": tensor.min().item(),
        f"{prefix}_max": tensor.max().item(),
        f"{prefix}_q1": tensor.quantile(0.25).item(),
        f"{prefix}_q2": tensor.median().item(),
        f"{prefix}_q3": tensor.quantile(0.75).item(),
    }


def test_compartment_statistics_layer_and_global_aggregation_match_tensor_reductions():
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(compute_entropy=False)
    )
    branch_statistics = {
        "layer0": {
            0: {"depth": 0, "exc_weight_mean": 1.0, "inh_weight_mean": 2.0},
            1: {"depth": 0, "exc_weight_mean": 3.0, "inh_weight_mean": 4.0},
        },
        "layer1": {
            0: {"depth": 1, "exc_weight_mean": 5.0, "inh_weight_mean": 6.0},
            1: {"depth": 1, "exc_weight_mean": 7.0, "inh_weight_mean": 8.0},
        },
    }

    layer_stats = analyzer._compute_layer_statistics(branch_statistics)
    global_stats = analyzer._compute_global_statistics(branch_statistics)

    assert layer_stats == {
        "layer0": {
            "depth": 0,
            **_expected_tensor_stats("exc_weight_mean", [1.0, 3.0]),
            **_expected_tensor_stats("inh_weight_mean", [2.0, 4.0]),
        },
        "layer1": {
            "depth": 1,
            **_expected_tensor_stats("exc_weight_mean", [5.0, 7.0]),
            **_expected_tensor_stats("inh_weight_mean", [6.0, 8.0]),
        },
    }
    assert global_stats == {
        **_expected_tensor_stats("exc_weight_mean", [1.0, 3.0, 5.0, 7.0]),
        **_expected_tensor_stats("inh_weight_mean", [2.0, 4.0, 6.0, 8.0]),
    }
