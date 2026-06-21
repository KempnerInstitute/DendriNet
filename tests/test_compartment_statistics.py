import numpy as np
import pytest
import torch
from scipy.stats import differential_entropy

from dendritic_modeling.analysis.tools.compartment_statistics import (
    CompartmentStatisticsAnalyzer,
    _empty_tensor_statistics,
    _tensor_statistics,
)
from dendritic_modeling.config.analysis import CompartmentStatisticsAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer


def _count_forward_hooks(module: torch.nn.Module) -> int:
    return sum(len(child._forward_hooks) for child in module.modules())


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


def test_compartment_statistics_tensor_helper_matches_current_reductions():
    actual = _tensor_statistics(
        torch.tensor([1.0, 3.0, 5.0]),
        "exc_weights",
        compute_mean=True,
        compute_variance=True,
        compute_min_max=True,
        compute_percentiles=True,
    )

    assert actual == _expected_tensor_stats("exc_weights", [1.0, 3.0, 5.0])


def test_compartment_statistics_empty_helper_respects_enabled_fields():
    assert _empty_tensor_statistics(
        "empty",
        compute_mean=True,
        compute_variance=True,
        compute_min_max=True,
        compute_percentiles=True,
    ) == {
        "empty_mean": 0.0,
        "empty_var": 0.0,
        "empty_std": 0.0,
        "empty_min": 0.0,
        "empty_max": 0.0,
        "empty_q1": 0.0,
        "empty_q2": 0.0,
        "empty_q3": 0.0,
    }
    assert _empty_tensor_statistics(
        "empty",
        compute_mean=False,
        compute_variance=False,
        compute_min_max=True,
        compute_percentiles=False,
    ) == {
        "empty_min": 0.0,
        "empty_max": 0.0,
    }


def test_compartment_statistics_tensor_helper_handles_empty_inputs():
    actual = _tensor_statistics(
        torch.tensor([]),
        "empty",
        compute_mean=True,
        compute_variance=False,
        compute_min_max=False,
        compute_percentiles=True,
    )

    assert actual == {
        "empty_mean": 0.0,
        "empty_q1": 0.0,
        "empty_q2": 0.0,
        "empty_q3": 0.0,
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


def test_compartment_statistics_entropy_sample_shaping_matches_current_rules():
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(compute_entropy=True)
    )
    input_tensor = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    activation_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    torch.testing.assert_close(
        torch.as_tensor(
            analyzer._entropy_samples_for_prefix(input_tensor, "exc_inputs")
        ),
        torch.tensor([2.0, 6.0, 3.0, 7.0]),
    )
    torch.testing.assert_close(
        torch.as_tensor(
            analyzer._entropy_samples_for_prefix(
                activation_tensor,
                "exc_activations",
            )
        ),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert analyzer._entropy_samples_for_prefix(input_tensor, "exc_weights") is None


def test_compartment_statistics_generic_entropy_matches_current_estimator():
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(compute_entropy=True)
    )
    analyzer.dequantize = False
    samples = torch.linspace(0.0, 1.0, steps=20)
    analyzer.raw_dict = {
        "layer0": {
            0: {
                "depth": 0,
                "exc_activations": samples,
            }
        }
    }

    branch_stats = analyzer._compute_branch_statistics()["layer0"][0]

    expected = float(differential_entropy(samples.numpy(), method="vasicek"))
    assert np.isclose(branch_stats["exc_activations_entropy"], expected)


def test_compartment_statistics_generic_entropy_zeroes_degenerate_samples():
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(compute_entropy=True)
    )
    analyzer.raw_dict = {
        "layer0": {
            0: {
                "depth": 0,
                "exc_activations": torch.ones(4),
            }
        }
    }

    branch_stats = analyzer._compute_branch_statistics()["layer0"][0]

    assert branch_stats["exc_activations_entropy"] == 0.0


def test_compartment_statistics_per_class_entropy_matches_current_estimator():
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(
            compute_entropy=True,
            per_class_analysis=True,
        )
    )
    class_zero = torch.linspace(0.0, 1.0, steps=20).reshape(10, 2)
    class_one = torch.linspace(2.0, 3.0, steps=20).reshape(10, 2)
    analyzer.raw_dict = {
        "layer0": {
            0: {
                "depth": 0,
                "exc_activations_per_class": {
                    0: [class_zero],
                    1: [class_one],
                },
            }
        }
    }

    branch_stats = analyzer._compute_branch_statistics()["layer0"][0]

    expected_zero = differential_entropy(class_zero.flatten().numpy(), method="vasicek")
    expected_one = differential_entropy(class_one.flatten().numpy(), method="vasicek")
    assert np.isclose(
        branch_stats["exc_activations_per_class_class_0_entropy"],
        expected_zero,
    )
    assert np.isclose(
        branch_stats["exc_activations_per_class_class_1_entropy"],
        expected_one,
    )


def test_compartment_statistics_synapse_capture_hooks_remove_partial_hooks(
    monkeypatch,
) -> None:
    analyzer = CompartmentStatisticsAnalyzer(CompartmentStatisticsAnalysisParams())
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=3,
        inhibitory_synapses_per_branch=2,
        reactivate=False,
        topk_noise_level=0.0,
    )
    synapse_modules = analyzer._iter_synapse_modules(layer)

    def _failing_register(*args, **kwargs):
        raise RuntimeError("compartment statistics hook registration failed")

    monkeypatch.setattr(
        layer.branch_inhibition,
        "register_forward_hook",
        _failing_register,
    )

    with pytest.raises(
        RuntimeError,
        match="compartment statistics hook registration failed",
    ):
        analyzer._attach_synapse_capture_hooks("layer", synapse_modules)

    assert _count_forward_hooks(layer) == 0
