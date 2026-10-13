"""Plot series stay distinguishable without colour."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from dendritic_modeling.plotting.styles import (
    CATEGORY_STYLES,
    PALETTE,
    category_style,
    depth_style,
    indexed_style,
    split_style,
)
from dendritic_modeling.scripts.sweeps.plotters.information_plotter import (
    InformationPlotter,
)
from dendritic_modeling.scripts.sweeps.utils.metrics import (
    NETWORK_CATEGORY_LINESTYLES,
    NETWORK_CATEGORY_MARKERS,
)


def _shape(style):
    return (style.marker, style.linestyle)


def test_every_network_category_has_its_own_marker_and_line_style():
    shapes = [_shape(style) for style in CATEGORY_STYLES.values()]
    assert len(shapes) == len(set(shapes))
    assert category_style("shunting") == category_style("dendritic_shunting")
    for name in CATEGORY_STYLES:
        assert NETWORK_CATEGORY_MARKERS[name] == CATEGORY_STYLES[name].marker
        assert NETWORK_CATEGORY_LINESTYLES[name] == CATEGORY_STYLES[name].linestyle


def test_depths_and_indexed_series_have_distinct_shapes():
    assert len({_shape(depth_style(d)) for d in range(12)}) == 12
    assert len({_shape(indexed_style(i)) for i in range(48)}) == 48
    with pytest.raises(ValueError):
        indexed_style(-1)


def test_splits_and_metric_groups_have_distinct_shapes():
    splits = {_shape(split_style(name)) for name in ("train", "valid", "test")}
    assert len(splits) == 3
    assert split_style("validation") == split_style("valid")
    plotter = InformationPlotter.__new__(InformationPlotter)
    for group in InformationPlotter.METRIC_GROUPS.values():
        styles = [plotter._metric_style(metric) for metric in group["metrics"]]
        assert len({(s["marker"], s["linestyle"]) for s in styles}) == len(styles)
        assert len({s["color"] for s in styles}) == len(styles)


# Machado et al. (2009) simulation matrices for full dichromacy, in linear RGB.
_DICHROMACY = {
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
}


def _linear_rgb(hex_color):
    srgb = np.array([int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


@pytest.mark.parametrize("deficiency", sorted(_DICHROMACY))
def test_palette_colours_stay_apart_under_dichromacy(deficiency):
    simulated = {
        color: np.clip(_DICHROMACY[deficiency] @ _linear_rgb(color), 0, 1)
        for color in PALETTE
    }
    distances = [
        float(np.linalg.norm(simulated[a] - simulated[b]))
        for a, b in itertools.combinations(PALETTE, 2)
    ]
    assert min(distances) > 0.05


# ---------------------------------------------------------------------------
# Sweep plots: every series of an axes differs in marker or line style.
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_figures(monkeypatch):
    """Keep the figures sweep plotters save instead of writing them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dendritic_modeling.scripts.sweeps.plotters.base_plotter import BasePlotter

    figures = []

    def keep(self, fig, output_path, dpi=None, close=True):
        figures.append(fig)
        return output_path

    monkeypatch.setattr(BasePlotter, "save_figure", keep)
    yield figures
    plt.close("all")


def _series_shapes(ax):
    """(marker, line style) of the data series of ``ax`` (reference lines have no marker)."""
    return [
        (line.get_marker(), line.get_linestyle())
        for line in ax.get_lines()
        if line.get_marker() not in (None, "None", "", " ")
    ]


def _assert_distinct_shapes(figures, *, minimum_series=2):
    checked = 0
    for fig in figures:
        for ax in fig.axes:
            shapes = _series_shapes(ax)
            if len(shapes) >= minimum_series:
                assert len(set(shapes)) == len(shapes), shapes
                checked += 1
    assert checked, "no plot with several series was drawn"


def _noise_frame():
    import pandas as pd

    row = {}
    for noise in ("uniform", "gaussian"):
        for metric in ("accuracy", "auc"):
            for level in ("0.1", "0.2", "0.4"):
                row[f"noise_{noise}_{metric}_{level}_mean"] = 0.9 - float(level)
    return pd.DataFrame([row, row])


def test_noise_plotter_series_differ_without_colour(captured_figures, tmp_path):
    from dendritic_modeling.scripts.sweeps.plotters.noise_plotter import NoisePlotter

    NoisePlotter().generate_all(_noise_frame(), tmp_path)
    _assert_distinct_shapes(captured_figures)


def test_noise_analyzer_series_differ_without_colour(captured_figures, tmp_path):
    from dendritic_modeling.scripts.sweeps.analyzers.noise_analyzer import (
        NoiseSweepAnalyzer,
    )

    analyzer = NoiseSweepAnalyzer()
    analyzer._plot_noise_robustness(_noise_frame(), tmp_path)
    analyzer._plot_noise_type_comparison(_noise_frame(), tmp_path)
    assert len(captured_figures) == 3
    _assert_distinct_shapes(captured_figures)
    for fig in captured_figures:
        for ax in fig.axes:
            assert len(_series_shapes(ax)) == 2


def test_weight_plotter_layer_and_category_series_differ(captured_figures, tmp_path):
    import pandas as pd

    rows = []
    for category in ("dendritic_shunting", "dendritic_additive"):
        for ratio in (0.5, 1.0, 2.0):
            row = {"network_category": category, "ei_ratio": ratio}
            for layer in ("_L0", "_L1"):
                row[f"weight_exc_weight_mean_mean{layer}_mean"] = 1.0 + ratio
                row[f"weight_inh_weight_mean_mean{layer}_mean"] = 0.5 + ratio
            rows.append(row)
    from dendritic_modeling.scripts.sweeps.plotters.weight_plotter import WeightPlotter

    WeightPlotter().generate_all(pd.DataFrame(rows), tmp_path)
    _assert_distinct_shapes(captured_figures)
    # Two layers of two categories: four series in each plot.
    assert {len(_series_shapes(fig.axes[0])) for fig in captured_figures} == {4}


def test_layerwise_information_panel_series_differ(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.DataFrame(
        {
            "ei_ratio": [0.5, 1.0, 2.0],
            **{f"layer_mi_V_C_depth{d}_mean": [0.1, 0.2, 0.3] for d in range(4)},
        }
    )
    plotter = InformationPlotter()
    fig, ax = plt.subplots()
    try:
        plotter._plot_layerwise_ivc_panel(data, "ei_ratio", "E/I", True, ax)
        shapes = _series_shapes(ax)
        assert len(shapes) == 4 and len(set(shapes)) == 4
    finally:
        plt.close(fig)


def test_ablation_and_rule_variant_series_differ(captured_figures, tmp_path):
    import pandas as pd

    from dendritic_modeling.scripts.sweeps.analyzers.ablation_analyzer import (
        AblationSweepAnalyzer,
    )
    from dendritic_modeling.scripts.sweeps.analyzers.local_learning_analyzer import (
        LocalLearningSweepAnalyzer,
    )

    ablation = pd.DataFrame(
        [
            {
                f"ablation_{method}_{target}_accuracy_drop_depth{depth}_mean": 0.1
                * (depth + 1)
                for method in ("lesion", "shuffle")
                for target in ("excitation", "inhibition")
                for depth in range(3)
            }
        ]
    )
    AblationSweepAnalyzer()._plot_layer_contributions(ablation, tmp_path)
    rules = pd.DataFrame(
        {
            "rule_variant": ["3f", "3f", "4f", "4f", "5f", "5f"],
            "ei_ratio": [0.5, 2.0] * 3,
            "test_accuracy_mean": [0.5, 0.6, 0.55, 0.65, 0.6, 0.7],
        }
    )
    LocalLearningSweepAnalyzer()._generate_rule_variant_comparison(rules, tmp_path)
    counts = sorted(len(_series_shapes(fig.axes[0])) for fig in captured_figures)
    assert counts == [3, 4]
    _assert_distinct_shapes(captured_figures)


# ---------------------------------------------------------------------------
# Library plots: every legend entry of an axes differs from the others in
# marker, line style or hatch, not only in colour.
# ---------------------------------------------------------------------------


def test_bar_and_unmarked_series_helpers_never_repeat_early():
    from dendritic_modeling.plotting.styles import (
        HATCHES,
        bar_kwargs,
        hatch_color,
        indexed_hatch,
        indexed_linestyle,
    )

    assert len({indexed_hatch(i) for i in range(len(HATCHES))}) == len(HATCHES)
    assert indexed_hatch(0) == ""
    assert len({indexed_linestyle(i) for i in range(4)}) == 4
    assert hatch_color("#000000") == "white" and hatch_color("#F0E442") == "black"
    assert bar_kwargs("#0072B2", "") == {"color": "#0072B2"}
    assert bar_kwargs("#0072B2", "//", alpha=0.8) == {
        "color": "#0072B2",
        "alpha": 0.8,
        "hatch": "//",
        "edgecolor": "white",
        "linewidth": 0,
    }


def _legend_styles(ax):
    """Labels and non-colour styles of the legend entries of ``ax``."""
    from matplotlib.collections import PathCollection
    from matplotlib.container import BarContainer, ErrorbarContainer
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles, labels = ax.get_legend_handles_labels()
    styles = []
    for handle in handles:
        if isinstance(handle, ErrorbarContainer):
            line = handle.lines[0]
            styles.append(("line", line.get_marker(), line.get_linestyle()))
        elif isinstance(handle, BarContainer):
            styles.append(("bar", handle.patches[0].get_hatch() or ""))
        elif isinstance(handle, Line2D):
            styles.append(("line", handle.get_marker(), handle.get_linestyle()))
        elif isinstance(handle, PathCollection):
            vertices = handle.get_paths()[0].vertices
            styles.append(("scatter", tuple(np.round(vertices, 6).ravel())))
        elif isinstance(handle, Patch):
            styles.append(("patch", handle.get_linestyle(), handle.get_hatch() or ""))
        else:
            styles.append((type(handle).__name__,))
    return labels, styles


def _assert_legends_read_without_colour(figures, *, minimum_axes=1):
    """Every axes with several legend entries tells them apart without colour."""
    checked = 0
    for fig in figures:
        for ax in fig.axes:
            labels, styles = _legend_styles(ax)
            if len(styles) < 2:
                continue
            assert len(set(styles)) == len(styles), list(zip(labels, styles))
            checked += 1
    assert checked >= minimum_axes, checked


@pytest.fixture
def kept_figures(monkeypatch):
    """Keep every figure a plotting function draws instead of saving or closing it."""
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


def _ablation_results():
    return {
        target: {
            f"layer_{depth}": {"depth": depth, "accuracy_drop": 0.1 * (depth + 1)}
            for depth in range(3)
        }
        for target in ("excitation", "inhibition", "all_synapses")
    }


def test_ablation_bar_series_have_their_own_hatches(kept_figures):
    from dendritic_modeling.plotting.visualizations import general_analysis_plots

    general_analysis_plots.plot_layer_ablation(
        {"lesion": _ablation_results()}, {"accuracy": 0.9}, save_path=None
    )
    general_analysis_plots.plot_per_layer_ablation(_ablation_results())
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=2)


def test_overlaid_weight_histograms_have_their_own_line_styles(kept_figures):
    from dendritic_modeling.plotting.visualizations import general_analysis_plots

    generator = np.random.default_rng(0)
    general_analysis_plots.plot_weight_distributions(
        {name: [generator.random(40)] for name in ("exc", "inh", "branch")}
    )
    _assert_legends_read_without_colour(kept_figures())


def test_performance_evolution_and_noise_sweep_curves_read_without_colour(
    kept_figures, tmp_path
):
    import json

    import pandas as pd

    from dendritic_modeling.plotting.visualizations import general_analysis_plots

    for epoch in (1, 2, 3):
        (tmp_path / f"epoch_{epoch}_performance.json").write_text(
            json.dumps({"epoch": epoch, "accuracy": 0.5 + 0.1 * epoch, "loss": 1.0})
        )
    general_analysis_plots.plot_performance_evolution(str(tmp_path))
    rows = [
        {
            "noise_type": "gaussian",
            "n_branches": branches,
            "ei_ratio": branches / 2,
            "noise_level": level,
            "accuracy": 0.9 - level,
            "loss": level,
            "precision": 0.8,
            "recall": 0.7,
        }
        for branches in (2, 4, 8)
        for level in (0.1, 0.2)
    ]
    csv_path = tmp_path / "noise.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    general_analysis_plots.plot_noise_sweep_branch_results(str(csv_path))
    general_analysis_plots.plot_noise_sweep_ei_results(str(csv_path))
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=9)


def test_loss_curves_tell_training_from_validation_without_colour(kept_figures):
    from dendritic_modeling.plotting.visualizations import general_analysis_plots
    from dendritic_modeling.plotting.visualizations.performance_plots import (
        PerformancePlotter,
    )

    losses = [1.0 / (epoch + 1) for epoch in range(12)]
    general_analysis_plots.plot_loss_curves(
        losses,
        [loss + 0.1 for loss in losses],
        "CE",
        train_losses_base=[loss * 0.9 for loss in losses],
        train_losses_reg=[loss * 0.1 for loss in losses],
    )
    PerformancePlotter().plot_loss_curves(losses, [loss + 0.1 for loss in losses], "CE")
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=2)


def test_information_plot_series_read_without_colour(kept_figures, tmp_path):
    import logging

    from dendritic_modeling.plotting.visualizations import information_plots

    _, ax = __import__("matplotlib.pyplot").pyplot.subplots()
    layer_results = {
        name: {
            "I(E;C)_mean": 0.3,
            "I(I;C)_mean": 0.2,
            "I(Vout;C)_mean": 0.4,
        }
        for name in ("Soma", "Distal Layer 1")
    }
    information_plots.PerLayerPlotter().plot_basic_mi_layers(
        layer_results, list(layer_results), ax
    )
    depth_stats = {
        depth: {"I(E;C)_mean": 0.2 + 0.1 * i, "I(E;C)_std": 0.05}
        for i, depth in enumerate(("Soma", "Distal Layer 1"))
    }
    information_plots._generate_layer_comparison_plots(
        {"excitatory": {"Layer 0": depth_stats, "Layer 1": depth_stats}},
        str(tmp_path),
        logging.getLogger(__name__),
    )
    statistics = {
        f"core_network.layers.{layer}.excitatory_cells.branch_layers.{branch}": {
            "I(E;C)_mean": 0.2 + 0.1 * layer + 0.05 * branch,
            "I(E;C)_std": 0.05,
            "I(E_lin;C)_mean": 0.1 + 0.1 * layer,
            "I(E_lin;C)_std": 0.05,
        }
        for layer in (0, 1)
        for branch in (0, 1)
    }
    information_plots.plot_info_vs_ei_depth_per_dendritic_layer(
        statistics, str(tmp_path)
    )
    information_plots.plot_fully_aggregated_analysis(statistics, str(tmp_path))
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=5)


def test_synaptic_weight_and_correlation_series_read_without_colour(
    kept_figures, tmp_path
):
    import logging

    import matplotlib.pyplot as plt

    from dendritic_modeling.plotting.visualizations import (
        correlation_plots,
        synaptic_weight_plots,
    )

    layers = {
        name: {
            "exc_n_active_synapses_mean": 4.0,
            "inh_n_active_synapses_mean": 2.0,
            "exc_weight_mean_mean": 0.5,
            "inh_weight_mean_mean": 0.3,
            "exc_weight_var_mean": 0.1,
            "inh_weight_var_mean": 0.05,
        }
        for name in ("Soma", "Distal Layer 1")
    }
    logger = logging.getLogger(__name__)
    for plot in (
        synaptic_weight_plots._plot_total_synaptic_strength,
        synaptic_weight_plots._plot_weight_comparisons,
        synaptic_weight_plots._plot_active_synapse_counts,
    ):
        plot(layers, list(layers), str(tmp_path), logger)

    _, axes = plt.subplots(1, 3)
    correlation_plots.plot_total_vs_noise_correlations(
        {
            f"{kind}_{part}_mean": 0.2
            for kind in ("EE", "II", "EI")
            for part in ("total", "noise")
        },
        axes[0],
    )
    generator = np.random.default_rng(0)
    tuning = {
        f"{var}_selectivity": {
            "preferred_class": generator.integers(0, 3, size=20),
            "selectivity_index": generator.random(20),
        }
        for var in ("E", "I", "Vout")
    }
    correlation_plots._plot_class_preference_distribution(tuning, axes[1], [0, 1, 2])
    correlation_plots._plot_selectivity_distributions(
        tuning, axes[2], "selectivity_index"
    )
    _assert_legends_read_without_colour(kept_figures(), minimum_axes=7)


def test_the_error_bar_helper_takes_a_marker_and_line_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dendritic_modeling.plotting.visualizations.plotting_utils import add_error_bars

    fig, ax = plt.subplots()
    try:
        default = add_error_bars(ax, [0, 1], [1, 2], [0.1, 0.1], "#0072B2", "a")
        styled = add_error_bars(
            ax,
            [0, 1],
            [1, 2],
            [0.1, 0.1],
            "#D55E00",
            "b",
            marker="s",
            linestyle="--",
        )
        assert (default.lines[0].get_marker(), default.lines[0].get_linestyle()) == (
            "o",
            "-",
        )
        assert (styled.lines[0].get_marker(), styled.lines[0].get_linestyle()) == (
            "s",
            "--",
        )
    finally:
        plt.close(fig)
