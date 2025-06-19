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
from dendritic_modeling.plotting.base import PlotConfig, BasePlotter, LinePlotter, BarPlotter, BoxPlotter, ComparisonPlotter

# Import analysis plotting classes and functions
from dendritic_modeling.plotting.analysis import (
    # Plotter classes
    LayerAnalysisPlotter,
    AblationPlotter,
    NoiseAnalysisPlotter,
    WeightDistributionPlotter,
    InformationAnalysisPlotter,
    ActivationPlotter,
    # Convenience functions
    plot_layer_contributions,
    plot_ablation_results,
    plot_noise_robustness,
    plot_weight_distributions,
    plot_information_analysis,
    # Helper functions
    dendrinet_activations_todict,
    einet_activations_todict,
)

# Import training plotting functions
from dendritic_modeling.plotting.training import (
    plot_loss_curves,
    plot_ei_sweep_results,
    plot_branch_sweep_results,
    plot_noise_sweep_ei_results,
    plot_noise_sweep_branch_results,
)

# Import network plotting functions
from dendritic_modeling.plotting.network import (
    plot_dendrinet_params,
    plot_einet_params,
    plot_branch_weights,
    plot_excitation_inhibition_weights,
    visualize_dendritic_tree,
    plot_dendrinet_gradients,
    plot_branch_gradients,
    plot_excitation_inhibition_gradients,
    plotWithColor,
    add_colorbars,
    dendrinet_params_todict,
    einet_params_todict,
    dendrinet_gradients_todict,
    einet_gradients_todict,
)

# Import basic plotting functions
from dendritic_modeling.plotting.basic import (
    plotWithColor,
    add_colorbars,
    shape_helper,
    plot_NLL_loss_curves as plot_NLL_loss_curves_basic,
)

# Import utility functions
from dendritic_modeling.plotting.utils import (
    reverse_layer_indices,
    format_layer_labels,
    dendrinet_activations_to_csv,
    einet_activations_to_csv,
    build_dendritic_tree_graph,
    visualize_dendritic_tree,
    plot_activation_tuning,
    plot_voltage_before_activation,
    split_MNIST_inputs,
    plot_depth_info,
)

__all__ = [
    # Base classes and configuration
    "PlotConfig",
    "BasePlotter", 
    "LinePlotter",
    "BarPlotter",
    "BoxPlotter",
    "ComparisonPlotter",
    
    # Analysis plotting classes
    "LayerAnalysisPlotter",
    "AblationPlotter",
    "NoiseAnalysisPlotter",
    "WeightDistributionPlotter",
    "InformationAnalysisPlotter",
    "ActivationPlotter",
    
    # Analysis convenience functions
    "plot_layer_contributions",
    "plot_ablation_results",
    "plot_noise_robustness",
    "plot_weight_distributions",
    "plot_information_analysis",
    "dendrinet_activations_todict",
    "einet_activations_todict",
    
    # Training plotting functions
    "plot_loss_curves",
    "plot_ei_sweep_results",
    "plot_branch_sweep_results",
    "plot_noise_sweep_ei_results",
    "plot_noise_sweep_branch_results",
    
    # Network plotting functions
    "plot_dendrinet_params",
    "plot_einet_params",
    "plot_branch_weights",
    "plot_excitation_inhibition_weights",
    "visualize_dendritic_tree",
    "plot_dendrinet_gradients",
    "plot_branch_gradients",
    "plot_excitation_inhibition_gradients",
    "plotWithColor",
    "add_colorbars",
    "dendrinet_params_todict",
    "einet_params_todict",
    "dendrinet_gradients_todict",
    "einet_gradients_todict",
    
    # Basic plotting functions (excluding plotWithColor and add_colorbars as they're in network)
    "shape_helper",
    "plot_NLL_loss_curves_basic",
    
    # Utility functions
    "reverse_layer_indices",
    "format_layer_labels",
    "dendrinet_activations_to_csv",
    "einet_activations_to_csv",
    "build_dendritic_tree_graph",
    "visualize_dendritic_tree",
    "plot_activation_tuning",
    "plot_voltage_before_activation",
    "split_MNIST_inputs",
    "plot_depth_info",
] 