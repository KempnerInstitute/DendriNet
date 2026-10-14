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

    branch_statistics = analyzer._compute_branch_statistics()
    branch_stats = branch_statistics["layer0"][0]

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

    branch_statistics = analyzer._compute_branch_statistics()
    branch_stats = branch_statistics["layer0"][0]

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

    branch_statistics = analyzer._compute_branch_statistics()
    branch_stats = branch_statistics["layer0"][0]

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
    assert "exc_activations_per_class" not in branch_stats
    layer_stats = analyzer._compute_layer_statistics(branch_statistics)
    global_stats = analyzer._compute_global_statistics(branch_statistics)
    assert layer_stats["layer0"]["depth"] == 0
    assert global_stats["exc_activations_per_class_class_0_mean_mean"] == pytest.approx(
        class_zero.mean().item()
    )


def test_compartment_statistics_per_class_capture_uses_cpu_labels():
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(
            compute_entropy=False,
            per_class_analysis=True,
        )
    )
    layer = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=3,
        inhibitory_synapses_per_branch=2,
        reactivate=False,
        topk_noise_level=0.0,
    )
    inputs = torch.randn(4, 3)
    labels = torch.tensor([0, 0, 1, 1])

    analyzer.collect_raw_data(layer, inputs, labels)

    assert analyzer.class_labels is not None
    assert analyzer.class_labels.device.type == "cpu"
    branch = analyzer.raw_dict[""][0]
    assert set(branch["exc_inputs_per_class"]) == {0, 1}
    assert set(branch["exc_activations_per_class"]) == {0, 1}


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


def _tree_model(ie_synapses: int):
    from dendritic_modeling.config import TransferFunctionConfig
    from dendritic_modeling.models import Classifier
    from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity

    torch.manual_seed(0)
    return Classifier(
        encoder_network=Identity(input_dim=8),
        core_network=ExcitationInhibitionNetwork(
            input_dim=8,
            transfer_params=TransferFunctionConfig().asdict(),
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[0],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            ee_synapses_per_branch_per_layer=[2],
            ei_synapses_per_branch_per_layer=[0],
            ie_synapses_per_branch_per_layer=[ie_synapses],
            ii_synapses_per_branch_per_layer=[0],
        ),
        decoder_network=Identity(input_dim=4),
    )


def _dataset():
    generator = torch.Generator().manual_seed(1)
    return torch.utils.data.TensorDataset(
        torch.rand(40, 8, generator=generator),
        torch.randint(0, 4, (40,), generator=generator),
    )


def test_excitatory_only_tree_skips_the_ratio_plot_and_records_why(tmp_path, caplog):
    """E/I ratios need inhibitory statistics, which excitatory-only trees lack."""
    import logging

    analyzer = CompartmentStatisticsAnalyzer(CompartmentStatisticsAnalysisParams())
    with caplog.at_level(logging.INFO):
        results = analyzer.analyze(
            _tree_model(ie_synapses=0), _dataset(), save_path=str(tmp_path)
        )

    assert results["branch_statistics"]
    assert (tmp_path / "final").is_file()
    assert not (tmp_path / "ratio_pairwise_comparison.png").exists()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    (missing,) = analyzer.analysis_missing_outputs
    assert missing["output"] == "ratio_pairwise_comparison.png"
    assert "no branch has inh_" in missing["reason"]
    assert "no inhibitory statistics" in missing["reason"]


def test_ratio_plot_draws_the_branches_that_have_inhibitory_statistics(tmp_path):
    from dendritic_modeling.analysis.tools.compartment_statistics import (
        plot_ratio_comparison,
        plot_ratio_pairwise_comparison,
    )

    def branch(depth, inhibition):
        stats = {"depth": depth}
        for side in ("exc", "inh") if inhibition else ("exc",):
            for field in ("weights", "inputs", "activations"):
                for statistic in ("mean", "std", "entropy"):
                    stats[f"{side}_{field}_{statistic}"] = 1.0 + depth
        return stats

    mixed = {"layer0": {0: branch(0, True), 1: branch(1, True), 2: branch(1, False)}}
    notes = plot_ratio_pairwise_comparison(
        mixed, weights_ratio=True, inputs_ratio=True, save_path=str(tmp_path)
    )
    assert notes == [
        "1 of 3 branches lack inh_inputs_mean, inh_inputs_std, inh_weights_mean, "
        "inh_weights_std (branches without inhibitory synapses have no inhibitory "
        "statistics) and are left out"
    ]
    assert (tmp_path / "ratio_pairwise_comparison.png").is_file()

    other = tmp_path / "unselected"
    assert plot_ratio_pairwise_comparison(mixed, save_path=str(other)) == []
    assert not other.exists()

    excitatory_only = {"layer0": {0: branch(0, False)}}
    (note,) = plot_ratio_comparison(excitatory_only, str(tmp_path))
    assert note.startswith("not drawn: no branch has inh_")
    assert not (tmp_path / "another_comparison.png").exists()


def test_compartment_statistics_status_lists_the_skipped_plot(tmp_path):
    from dendritic_modeling.analysis import AnalysisManager
    from dendritic_modeling.config import AnalysisConfig

    config = AnalysisConfig()
    config.compartment_statistics.enabled = True
    data = {"train": _dataset(), "valid": _dataset(), "test": _dataset()}
    manager = AnalysisManager(
        model=_tree_model(ie_synapses=0),
        data=data,
        analysis_config=config,
        save_root=str(tmp_path),
    )
    record = manager.run_analysis()
    status = record["analyses"]["compartment_statistics"]
    assert status["status"] == "completed"
    assert status["missing_outputs"][0]["output"] == "ratio_pairwise_comparison.png"


def test_compartment_statistics_reports_a_model_without_branch_layers():
    analyzer = CompartmentStatisticsAnalyzer(CompartmentStatisticsAnalysisParams())
    assert analyzer.analyze(torch.nn.Linear(2, 2), _dataset()) is None
    assert analyzer.analysis_skip_reason == "the model has no dendritic branch layers"


def test_per_class_plot_draws_each_depth_from_the_per_class_means(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from dendritic_modeling.analysis.tools.compartment_statistics import (
        plot_per_class_analysis,
    )

    def branch(depth, classes):
        stats = {"depth": depth}
        for metric in ("exc_inputs", "inh_inputs", "exc_activations"):
            for k in classes:
                stats[f"{metric}_per_class_class_{k}_mean"] = depth + k
                stats[f"{metric}_per_class_class_{k}_std"] = 99.0
        return stats

    statistics = {"layer0": {0: branch(0, (0, 1, 2)), 1: branch(2, (0, 2))}}
    drawn = []
    original = plt.Axes.errorbar

    def record(ax, x, y, *args, **kwargs):
        drawn.append((ax.get_title() or "", kwargs.get("label"), list(x), list(y)))
        return original(ax, x, y, *args, **kwargs)

    plt.Axes.errorbar = record
    try:
        assert plot_per_class_analysis(statistics, str(tmp_path)) == []
    finally:
        plt.Axes.errorbar = original
    assert (tmp_path / "per_class_analysis.png").is_file()
    # Three metrics have values, each drawn for both depths; standard
    # deviations of the branches are not mixed into the class means.
    assert len(drawn) == 6
    labels = {label for _title, label, _x, _y in drawn}
    assert labels == {"Depth 0", "Depth 2"}
    depth2 = next(y for _t, label, _x, y in drawn if label == "Depth 2")
    assert depth2[0] == 2.0 and np.isnan(depth2[1]) and depth2[2] == 4.0

    (note,) = plot_per_class_analysis(
        {"layer0": {0: {"depth": 0}}}, str(tmp_path / "x")
    )
    assert note.startswith("not drawn: no branch has per-class statistics")
    assert not (tmp_path / "x").exists()


def test_per_class_analysis_writes_its_figure(tmp_path):
    analyzer = CompartmentStatisticsAnalyzer(
        CompartmentStatisticsAnalysisParams(per_class_analysis=True)
    )
    analyzer.analyze(_tree_model(ie_synapses=2), _dataset(), save_path=str(tmp_path))
    assert (tmp_path / "per_class_analysis.png").is_file()
    assert not [
        item
        for item in analyzer.analysis_missing_outputs
        if item["output"] == "per_class_analysis.png"
    ]
