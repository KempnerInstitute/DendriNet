"""
Plotting package for dendritic modeling.

This package contains visualization and plotting utilities organized into logical modules:
- analysis: Analysis-related plotting functions (activations, tuning curves, distributions)
- training: Training-related plotting functions (loss curves, sweeps, metrics)  
- network: Network visualization functions (parameters, weights, architecture)
- utils: Utility functions (CSV export, tree visualization, helper functions)

Each module contains focused plotting functions for specific aspects of dendritic modeling.
"""

# Import all plotting functions from organized modules
from dendritic_modeling.plotting.training import (
    plot_loss_curves,
    plot_ei_sweep_results,
    plot_branch_sweep_results, 
    plot_noise_sweep_ei_results,
    plot_noise_sweep_branch_results,
)

from dendritic_modeling.plotting.network import (
    plotWithColor,
    add_colorbars,
    plot_dendrinet_params,
    plot_einet_params,
    plot_branch_weights,
    plot_excitation_inhibition_weights,
    build_dendritic_tree_graph,
    visualize_dendritic_tree,
    plot_dendrinet_gradients,
    plot_branch_gradients,
    plot_excitation_inhibition_gradients,
    shape_helper,
    dendrinet_params_todict,
    einet_params_todict,
    dendrinet_gradients_todict,
    einet_gradients_todict,
)

from dendritic_modeling.plotting.analysis import (
    plot_dendrinet_activations,
    plot_einet_activations,
    activation_boxplots,
    activation_means,
    plot_activation_tuning,
    plot_voltage_before_activation,
    plot_einet_profiles,
    plot_depth_info,
    dendrinet_activations_todict,
    einet_activations_todict,
)

from dendritic_modeling.plotting.utils import (
    dendrinet_activations_to_csv,
    einet_activations_to_csv,
    build_dendritic_tree_graph,
    visualize_dendritic_tree,
    split_MNIST_inputs,
)

# Export all functions for easy import
__all__ = [
    # Training functions
    "plot_loss_curves",
    "plot_ei_sweep_results",
    "plot_branch_sweep_results",
    "plot_noise_sweep_ei_results", 
    "plot_noise_sweep_branch_results",
    
    # Network functions
    "plotWithColor",
    "add_colorbars",
    "plot_dendrinet_params",
    "plot_einet_params", 
    "plot_branch_weights",
    "plot_excitation_inhibition_weights",
    "build_dendritic_tree_graph",
    "visualize_dendritic_tree",
    "plot_dendrinet_gradients",
    "plot_branch_gradients",
    "plot_excitation_inhibition_gradients",
    "shape_helper",
    "dendrinet_params_todict",
    "einet_params_todict",
    "dendrinet_gradients_todict",
    "einet_gradients_todict",
    
    # Analysis functions
    "plot_dendrinet_activations",
    "plot_einet_activations",
    "activation_boxplots",
    "activation_means",
    "plot_activation_tuning",
    "plot_voltage_before_activation",
    "plot_einet_profiles",
    "plot_depth_info",
    "dendrinet_activations_todict",
    "einet_activations_todict",
    
    # Utility functions
    "dendrinet_activations_to_csv",
    "einet_activations_to_csv",
    "split_MNIST_inputs",
] 