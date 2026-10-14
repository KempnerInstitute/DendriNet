"""Analyzer figures tell their series apart without colour."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from test_plot_styles import _assert_legends_read_without_colour


@pytest.fixture
def kept_figures(monkeypatch):
    """Keep every figure an analyzer draws instead of saving or closing it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    close = plt.close
    close("all")
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        matplotlib.figure.Figure, "savefig", lambda self, *args, **kwargs: None
    )
    yield lambda: [plt.figure(number) for number in plt.get_fignums()]
    close("all")


@pytest.fixture
def first_frames(monkeypatch):
    """Draw the first frame of every animation instead of encoding a video."""
    import matplotlib.animation as animation

    class FirstFrame:
        def __init__(self, fig, func, *args, init_func=None, **kwargs):
            if init_func is not None:
                init_func()
            func(0)

        def save(self, *args, **kwargs):
            return None

    monkeypatch.setattr(animation, "FuncAnimation", FirstFrame)


def _scatter_marker_counts(figures):
    """Number of distinct marker shapes in every scatter of ``figures``."""
    from matplotlib.collections import PathCollection

    return [
        len({np.round(path.vertices, 6).tobytes() for path in collection.get_paths()})
        for fig in figures
        for ax in fig.axes
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    ]


def _ablation_layer(scale=1.0):
    return {
        target: {
            f"module_{depth}": {"depth": depth, "accuracy_drop": scale * (depth + 1)}
            for depth in range(3)
        }
        for target in ("excitation", "inhibition", "all_synapses")
    }


def test_ablation_and_layer_contribution_bars_have_their_own_hatches(
    kept_figures, first_frames, tmp_path
):
    from dendritic_modeling.analysis.core.ablation import (
        create_video_layer_ablation_vs_training,
    )
    from dendritic_modeling.analysis.metrics.layer_contribution import (
        plot_layer_contribution,
    )

    epochs = tmp_path / "epochs"
    epochs.mkdir()
    for epoch in (1, 2):
        (epochs / f"epoch{epoch}.json").write_text(
            json.dumps(
                {
                    "layer": {"lesion": _ablation_layer(0.1 * epoch)},
                    "baseline": {"accuracy": 0.9},
                }
            )
        )
    create_video_layer_ablation_vs_training(str(tmp_path), "accuracy", "gif")
    plot_layer_contribution(
        {
            target: {
                f"module_{depth}": {"depth": depth, "accuracy": 0.5 + 0.1 * depth}
                for depth in range(3)
            }
            for target in ("excitation", "inhibition")
        }
    )
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=2)


def _layer_statistics(metric_keys, value=0.5):
    return {
        f"layer_{depth}": {
            "depth": depth,
            **dict.fromkeys(metric_keys, value + 0.1 * depth),
        }
        for depth in range(3)
    }


def test_compartment_snr_series_read_without_colour(
    kept_figures, first_frames, tmp_path
):
    from dendritic_modeling.analysis.tools import compartment_snr

    types = ("vout", "exc", "inh", "upstream", "vinf")
    epochs = [1, 2, 3]
    compartment_snr.plot_branch_analysis_global_agg_vs_training(
        {
            f"{act}_d'^2_{stat}": [0.1, 0.2, 0.3]
            for act in types
            for stat in ("mean", "std")
        },
        epochs,
        str(tmp_path),
    )
    compartment_snr.plot_branch_analysis_layer_agg_vs_training(
        {
            f"layer_{depth}": {
                "depth": depth,
                **{
                    f"{act}_d'^2_{stat}": [0.1, 0.2, 0.3]
                    for act in types
                    for stat in ("mean", "std")
                },
            }
            for depth in range(3)
        },
        epochs,
        str(tmp_path),
    )
    keys = [f"{act}_d'^2_{stat}" for act in types for stat in ("mean", "std")]
    compartment_snr.plot_branch_analysis_layer_agg_final(_layer_statistics(keys))
    compartment_snr.plot_layer_analysis_layer_agg_final(
        _layer_statistics([f"{act}_d_max" for act in types])
    )
    both = [f"{act}_d'^2/D" for act in ("vout", "upstream")] + [
        f"{act}_d'^2" for act in ("exc", "inh")
    ]
    compartment_snr.plot_layer_analysis_layer_agg_final_both_metrics(
        _layer_statistics(both)
    )
    compartment_snr.plot_layer_analysis_layer_agg_vs_training(
        {
            f"layer_{depth}": {
                "depth": depth,
                **{f"{act}_d_max": [0.2, 0.3, 0.4] for act in types},
            }
            for depth in range(3)
        },
        epochs,
        str(tmp_path / "video"),
    )
    compartment_snr.plot_layer_analysis_layer_agg_vs_training_both_metrics(
        {
            f"layer_{depth}": {"depth": depth, **{key: [0.2, 0.3, 0.4] for key in both}}
            for depth in range(3)
        },
        epochs,
        str(tmp_path / "video_both"),
    )
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=11)


def _branch_statistics():
    generator = np.random.default_rng(0)
    keys = [
        f"{side}_{kind}_{stat}"
        for side in ("exc", "inh")
        for kind in ("inputs", "weights", "activations")
        for stat in ("mean", "std", "entropy")
    ]
    return {
        "core_network.layers.0.excitatory_cells.branch_layers.0": {
            branch: {
                "depth": branch % 2,
                **{key: float(generator.uniform(0.5, 1.5)) for key in keys},
                **{
                    f"{act}_d'^2": float(generator.uniform(0.1, 1.0))
                    for act in ("vout", "exc", "inh")
                },
            }
            for branch in range(8)
        }
    }


def test_depth_scatters_mark_each_depth_with_its_own_marker(kept_figures, tmp_path):
    from dendritic_modeling.analysis.tools import (
        compartment_snr,
        compartment_statistics,
    )

    branches = _branch_statistics()
    compartment_snr.plot_branch_analysis_branch_agg_final(branches, str(tmp_path))
    compartment_statistics.plot_ratio_comparison(branches, str(tmp_path))
    compartment_statistics.plot_ratio_pairwise_comparison(
        branches, weights_ratio=True, activations_ratio=True, save_path=str(tmp_path)
    )
    figures = kept_figures()
    assert len(figures) == 3
    for fig in figures:
        # Two depths: every facet draws both markers.
        assert set(_scatter_marker_counts([fig])) == {2}


def test_compartment_statistics_series_read_without_colour(kept_figures, tmp_path):
    from dendritic_modeling.analysis.tools import compartment_statistics
    from dendritic_modeling.analysis.tools.compartment_statistics import (
        CompartmentStatisticsAnalyzer,
    )

    layers = {
        name: {
            f"{side}_{kind}_mean_{stat}": 0.3 + 0.1 * index
            for side in ("exc", "inh")
            for kind in ("weights", "inputs", "activations")
            for stat in ("mean", "std")
        }
        for index, name in enumerate(("Soma", "Distal Layer 1", "Distal Layer 2"))
    }
    names = list(layers)
    for plot in (
        compartment_statistics._plot_compartment_weight_strength,
        compartment_statistics._plot_compartment_weight_comparisons,
        compartment_statistics._plot_compartment_input_strength,
        compartment_statistics._plot_compartment_input_comparisons,
        compartment_statistics._plot_compartment_activation_strength,
        compartment_statistics._plot_compartment_activation_comparisons,
    ):
        plot(layers, names, str(tmp_path))
    analyzer = CompartmentStatisticsAnalyzer.__new__(CompartmentStatisticsAnalyzer)
    analyzer.logger = logging.getLogger(__name__)
    analyzer._plot_enhanced_statistics(
        {
            f"core_network.layers.0.excitatory_cells.branch_layers.{branch}": {
                "layer_weight_min": 0.1,
                "layer_weight_max": 0.9,
                "dendritic_strength": 1.0,
            }
            for branch in range(2)
        },
        str(tmp_path),
        somatic_synapses=True,
    )
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=7)


def test_weight_analyzer_min_max_bars_have_their_own_hatches(kept_figures, tmp_path):
    from dendritic_modeling.analysis.tools.weight import WeightAnalyzer

    analyzer = WeightAnalyzer.__new__(WeightAnalyzer)
    analyzer.logger = logging.getLogger(__name__)
    analyzer._plot_enhanced_statistics(
        {
            f"core_network.layers.0.excitatory_cells.branch_layers.{branch}": {
                "layer_weight_min": 0.1,
                "layer_weight_max": 0.9,
                "dendritic_strength": 1.0,
            }
            for branch in range(2)
        },
        str(tmp_path),
        somatic_synapses=True,
    )
    _assert_legends_read_without_colour(kept_figures())


def test_trace_and_spectrum_figures_read_without_colour(kept_figures, tmp_path):
    from dendritic_modeling.analysis.tools.dendritic_timetraces import (
        DendriticTimetracesAnalyzer,
    )
    from dendritic_modeling.analysis.tools.jacobian_spectrum import (
        JacobianSpectrumAnalyzer,
    )

    traces = DendriticTimetracesAnalyzer.__new__(DendriticTimetracesAnalyzer)
    traces._save_figures(
        {
            "layers": {
                "layer_0": {
                    "n_levels": 2,
                    "u": [0.0, 1.0, 0.0, 0.0],
                    "tau": [5.0, 25.0],
                    "V": [[0.1, 0.2, 0.3, 0.2], [0.0, 0.1, 0.2, 0.3]],
                    "V_hat": [[0.1, 0.2, 0.2, 0.2], [0.0, 0.1, 0.1, 0.2]],
                }
            }
        },
        str(tmp_path),
        "traces",
    )
    spectrum = JacobianSpectrumAnalyzer.__new__(JacobianSpectrumAnalyzer)
    spectrum._save_figures(
        {
            "layers": {
                "layer_0": {
                    "n_levels": 3,
                    "levels": [
                        {
                            "level": level,
                            "tau": 5.0 * (level + 1),
                            "sigma_rec": [3.0, 2.0, 1.0],
                            "top_singular_value_rec": 3.0,
                            "top_singular_value_inh": 1.0,
                        }
                        for level in range(3)
                    ],
                }
            }
        },
        str(tmp_path),
        "spectrum",
    )
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=4)


def test_receptive_field_tuning_series_read_without_colour(kept_figures, tmp_path):
    from dendritic_modeling.analysis.tools import receptive_fields

    tuning = {
        side: {"soma 0": {"pos": 0.6, "neg": 0.4}, "soma 1": {"pos": 0.5, "neg": 0.3}}
        for side in ("exc", "inh")
    }
    receptive_fields._plot_agg_tuning_final(str(tmp_path), "png", tuning)
    receptive_fields._plot_agg_tuning_vs_training(
        {
            side: {
                "soma 0": {"pos": [0.5, 0.6], "neg": [0.3, 0.4]},
            }
            for side in ("exc", "inh")
        },
        [1, 2],
        str(tmp_path),
        "png",
    )
    generator = np.random.default_rng(0)
    receptive_fields._plot_pref_unpref_scatter_by_compartment(
        {
            "soma 0": {
                group: {
                    "x": list(generator.uniform(0.1, 1.0, 20)),
                    "y": list(generator.uniform(0.1, 1.0, 20)),
                }
                for group in ("preferred", "unpreferred")
            }
        },
        "x",
        "y",
        False,
        str(tmp_path),
        "png",
        "pref_unpref",
    )
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=3)
