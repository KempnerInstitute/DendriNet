import math

import pytest
import torch

from dendritic_modeling.analysis.tools.compartment_snr import (
    CompartmentSNRAnalyzer,
    _aggregate_non_depth_values,
    _summarize_population_values,
)
from dendritic_modeling.config.analysis import CompartmentSNRAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer


def _count_forward_and_pre_hooks(module: torch.nn.Module) -> int:
    return sum(
        len(child._forward_hooks) + len(child._forward_pre_hooks)
        for child in module.modules()
    )


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


def test_compartment_snr_layer_capture_hooks_remove_partial_hooks(
    monkeypatch,
) -> None:
    analyzer = CompartmentSNRAnalyzer(CompartmentSNRAnalysisParams(input_analysis=True))
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=3,
        inhibitory_synapses_per_branch=2,
        reactivate=False,
        topk_noise_level=0.0,
    )

    def _failing_register(*args, **kwargs):
        raise RuntimeError("compartment snr hook registration failed")

    monkeypatch.setattr(
        layer.branch_inhibition,
        "register_forward_hook",
        _failing_register,
    )

    with pytest.raises(RuntimeError, match="compartment snr hook registration failed"):
        analyzer._attach_layer_capture_hooks("layer", layer)

    assert _count_forward_and_pre_hooks(layer) == 0


def _discriminability(x, y, permutations=50):
    from dendritic_modeling.analysis.tools.compartment_snr import class_discriminability

    return class_discriminability(
        x,
        y,
        null_permutations=permutations,
        null_quantile=0.95,
        generator=torch.Generator().manual_seed(0),
    )


def test_label_free_activity_has_no_significant_direction():
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(1000, 64, generator=generator)
    y = torch.randint(0, 5, (1000,), generator=generator)
    result = _discriminability(x, y)
    assert result["n_sig_dirs"] == 0.0
    # In-sample eigenvalues of noise: about (C - 1) D / (n - 1) in total.
    assert abs(result["d_total"] - 4 * 64 / 999) < 0.1
    assert abs(result["d_total"] - result["d_total_null_mean"]) < 0.1
    assert result["dim_over_n"] == 64 / 1000


def test_a_one_dimensional_class_signal_gives_one_direction():
    generator = torch.Generator().manual_seed(2)
    y = torch.randint(0, 4, (1000,), generator=generator)
    x = torch.randn(1000, 16, generator=generator)
    x[:, 0] += y.float()
    result = _discriminability(x, y)
    assert result["n_sig_dirs"] == 1.0
    assert 0.0 < result["d_max"] <= 1.0
    assert result["d_total"] <= 3.0


def test_discriminability_without_null_or_signal():
    generator = torch.Generator().manual_seed(3)
    y = torch.randint(0, 3, (200,), generator=generator)
    x = torch.randn(200, 4, generator=generator)
    without_null = _discriminability(x, y, permutations=0)
    assert math.isnan(without_null["n_sig_dirs"])
    constant = _discriminability(torch.ones(200, 4), y)
    assert constant["d_total"] == 0.0 and constant["n_sig_dirs"] == 0.0
