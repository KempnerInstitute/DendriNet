"""Smoke tests for ablation sweep plotting/analyzer paths."""

from pathlib import Path

import pandas as pd

from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_layer_ablation,
)
from dendritic_modeling.scripts.sweeps.analyzers.ablation_analyzer import (
    AblationSweepAnalyzer,
)
from dendritic_modeling.scripts.sweeps.plotters.ablation_plotter import AblationPlotter


def test_layer_ablation_plot_does_not_relabel_depth_one_as_soma() -> None:
    figures = plot_layer_ablation(
        layer_results={
            "lesion": {
                "excitation": {
                    "branch_layers.0": {"depth": 2, "accuracy_drop": 0.2},
                    "branch_layers.1": {"depth": 1, "accuracy_drop": 0.1},
                }
            }
        },
        baselines={"accuracy": 0.5},
        somatic_synapses=False,
    )

    assert figures is not None
    labels = [tick.get_text() for tick in figures[0].axes[0].get_xticklabels()]
    assert labels == ["Distal 1", "Distal 2"]


def test_ablation_plotter_generates_canonical_multiword_plot(tmp_path: Path):
    data = pd.DataFrame(
        {
            "network_category": [
                "dendritic_shunting",
                "dendritic_shunting",
                "dendritic_additive",
                "dendritic_additive",
            ],
            "ei_ratio": [0.5, 1.0, 0.5, 1.0],
            "ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth0_mean": [
                0.10,
                0.20,
                0.15,
                0.25,
            ],
            "ablation_mean_clamp_upstream_pred_label_mi_bits_drop_depth1_mean": [
                0.12,
                0.22,
                0.18,
                0.28,
            ],
        }
    )

    paths = AblationPlotter().generate_all(data, tmp_path)
    names = {path.name for path in paths}
    assert "mean_clamp_upstream_pred_label_mi_bits_drop_ablation.png" in names


def test_ablation_plotter_generates_flat_mlp_plot(tmp_path: Path):
    data = pd.DataFrame(
        {
            "network_category": ["flat_mlp", "flat_mlp"],
            "ei_ratio": [0.5, 1.0],
            "ablation_lesion_excitation_accuracy_drop_depth0_mean": [0.10, 0.20],
        }
    )

    paths = AblationPlotter().generate_all(data, tmp_path)
    names = {path.name for path in paths}
    assert "lesion_excitation_accuracy_drop_ablation.png" in names


def test_ablation_plotter_generates_module_plot(tmp_path: Path):
    data = pd.DataFrame(
        {
            "network_category": [
                "dendritic_shunting",
                "dendritic_shunting",
                "dendritic_additive",
                "dendritic_additive",
            ],
            "ei_ratio": [0.5, 1.0, 0.5, 1.0],
            "ablation_module__lesion__excitation__accuracy_drop__core_layers_0__depth0_mean": [
                0.10,
                0.20,
                0.15,
                0.25,
            ],
            "ablation_module__lesion__excitation__accuracy_drop__core_layers_1__depth1_mean": [
                0.12,
                0.22,
                0.18,
                0.28,
            ],
        }
    )

    paths = AblationPlotter().generate_all(data, tmp_path)
    names = {path.name for path in paths}
    assert "lesion_excitation_accuracy_drop_module_ablation.png" in names


def test_ablation_sweep_analyzer_generates_exc_inh_plot_on_canonical_columns(
    tmp_path: Path,
):
    data = pd.DataFrame(
        {
            "network_category": [
                "dendritic_shunting",
                "dendritic_shunting",
                "dendritic_additive",
                "dendritic_additive",
            ],
            "ablation_lesion_excitation_accuracy_drop_depth0_mean": [
                0.10,
                0.20,
                0.15,
                0.25,
            ],
            "ablation_lesion_inhibition_accuracy_drop_depth0_mean": [
                0.05,
                0.10,
                0.07,
                0.11,
            ],
        }
    )

    paths = AblationSweepAnalyzer()._plot_exc_inh_contributions(data, tmp_path)
    names = {path.name for path in paths}
    assert "excitation_vs_inhibition_lesion_accuracy_drop.png" in names
