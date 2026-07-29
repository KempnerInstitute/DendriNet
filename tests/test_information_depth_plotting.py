from __future__ import annotations

from dendritic_modeling.plotting.visualizations.information_plots import (
    plot_info_vs_ei_depth_per_dendritic_layer,
)


def test_ei_depth_plot_accepts_population_module_paths(tmp_path) -> None:
    layer_statistics = {
        "core_network.layers.0.populations.e0.branch_layers.0": {
            "I(E;C)_mean": 0.1,
            "I(E;C)_std": 0.01,
            "has_synapses": True,
        },
        "core_network.layers.1.populations.e1.branch_layers.0": {
            "I(E;C)_mean": 0.2,
            "I(E;C)_std": 0.02,
            "has_synapses": True,
        },
    }
    soma_relative_depths = dict.fromkeys(layer_statistics, 2)

    plot_info_vs_ei_depth_per_dendritic_layer(
        layer_statistics,
        save_path=str(tmp_path),
        layer_soma_relative_depths=soma_relative_depths,
    )

    assert (tmp_path / "ei_depth_comparison_excitatory_cells_basic_mi.png").is_file()
