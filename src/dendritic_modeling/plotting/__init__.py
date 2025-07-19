"""
Plotting package for dendritic modeling.

This package contains visualization and plotting utilities organized into logical modules:
- analysis: Unified analysis plotting with object-oriented design for all analysis types
- training: Training-related plotting functions (loss curves, sweeps, metrics)
- network: Network visualization functions (parameters, weights, architecture)
- basic: Basic plotting utilities
- utils: Utility functions (CSV export, tree visualization, helper functions)
- base: Base classes and configuration for the plotting system

Each module contains focused plotting functions for specific aspects of dendritic modeling.
"""

# Import base classes and configuration
# Import analysis plotting classes and functions
from dendritic_modeling.plotting.analysis import (
    AblationPlotter,
    ActivationPlotter,
    InformationAnalysisPlotter,
    LayerAnalysisPlotter,
    NoiseAnalysisPlotter,
    WeightDistributionPlotter,
    dendrinet_activations_todict,
    einet_activations_todict,
    plot_ablation_results,
    plot_dendrinet_activations,
    plot_einet_activations,
    plot_information_analysis,
    plot_layer_contributions,
    plot_noise_robustness,
    plot_weight_distributions,
    plot_layerwise_ablation,
    plot_layerwise_contributions,
    plot_ablation_comparison,
)
from dendritic_modeling.plotting.base import (
    BarPlotter,
    BasePlotter,
    BoxPlotter,
    ComparisonPlotter,
    LinePlotter,
    PlotConfig,
)

# Import network plotting functions
from dendritic_modeling.plotting.network import (
    dendrinet_gradients_todict,
    dendrinet_params_todict,
    einet_gradients_todict,
    einet_params_todict,
    plot_branch_gradients,
    plot_branch_weights,
    plot_dendrinet_gradients,
    plot_dendrinet_params,
    plot_einet_params,
    plot_excitation_inhibition_gradients,
    plot_excitation_inhibition_weights,
)

# Import training plotting functions
from dendritic_modeling.plotting.training import (
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_loss_curves,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
)

# Import utility functions
from dendritic_modeling.plotting.utils import (
    build_dendritic_tree_graph,
    dendrinet_activations_to_csv,
    einet_activations_to_csv,
    format_layer_labels,
    plot_activation_tuning,
    plot_depth_info,
    plot_voltage_before_activation,
    reverse_layer_indices,
    split_mnist_inputs,
    visualize_dendritic_tree,
)

__all__ = [
    "AblationPlotter",
    "ActivationPlotter",
    "BarPlotter",
    "BasePlotter",
    "BoxPlotter",
    "ComparisonPlotter",
    "InformationAnalysisPlotter",
    "LayerAnalysisPlotter",
    "LinePlotter",
    "NoiseAnalysisPlotter",
    "PlotConfig",
    "WeightDistributionPlotter",
    "add_colorbars",
    "build_dendritic_tree_graph",
    "dendrinet_activations_to_csv",
    "dendrinet_activations_todict",
    "dendrinet_gradients_todict",
    "dendrinet_params_todict",
    "einet_activations_to_csv",
    "einet_activations_todict",
    "einet_gradients_todict",
    "einet_params_todict",
    "format_layer_labels",
    "plot_ablation_results",
    "plot_activation_tuning",
    "plot_branch_gradients",
    "plot_branch_sweep_results",
    "plot_branch_weights",
    "plot_dendrinet_activations",
    "plot_dendrinet_gradients",
    "plot_dendrinet_params",
    "plot_depth_info",
    "plot_ei_sweep_results",
    "plot_einet_activations",
    "plot_einet_params",
    "plot_excitation_inhibition_gradients",
    "plot_excitation_inhibition_weights",
    "plot_information_analysis",
    "plot_layer_contributions",
    "plot_loss_curves",
    "plot_nll_loss_curves",
    "plot_noise_robustness",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    "plot_voltage_before_activation",
    "plot_weight_distributions",
    "plot_with_color",
    "reverse_layer_indices",
    "shape_helper",
    "split_mnist_inputs",
    "visualize_dendritic_tree",
    "plot_layerwise_ablation",
    "plot_layerwise_contributions",
    "plot_ablation_comparison",
]
