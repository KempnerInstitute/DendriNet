"""Visualization plotting functions for analysis, networks, and correlations."""

# Analysis visualizations (updated for modular structure)
from .information_analysis import (
    plot_information_analysis,
    plot_per_layer_information,
)

from .general_analysis_plots import (
    plot_ablation_results,
    plot_layer_contributions,
    plot_noise_robustness,
    plot_weight_distributions,
    plot_dendrinet_activations,
    plot_einet_activations,
    plot_performance_metrics,
    plot_performance_evolution,
    plot_synapse_turnover,
    # Training and sweep functions
    plot_loss_curves,
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
)

# Data extraction utilities
from ..core.utils import (
    dendritic_activations_todict,
    einet_activations_todict,
)

# Network visualizations  
from .network import (
    plot_branch_gradients,
    plot_branch_weights,
    plot_dendrinet_gradients,
    plot_dendrinet_params,
    plot_einet_params,
    plot_excitation_inhibition_gradients,
    plot_excitation_inhibition_weights,
    visualize_dendritic_tree,
)

__all__ = [
    # Information analysis (now modular)
    "plot_information_analysis",
    "plot_per_layer_information",
    
    # General analysis
    "plot_ablation_results",
    "plot_layer_contributions", 
    "plot_noise_robustness",
    "plot_weight_distributions",
    "plot_dendrinet_activations",
    "plot_einet_activations",
    "plot_performance_metrics",
    "plot_performance_evolution",
    "plot_synapse_turnover",
    
    # Training/sweep functions
    "plot_loss_curves",
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    
    # Legacy data utilities
    "dendritic_activations_todict",
    "einet_activations_todict",
    
    # Network
    "plot_branch_gradients",
    "plot_branch_weights",
    "plot_dendrinet_gradients", 
    "plot_dendrinet_params",
    "plot_einet_params",
    "plot_excitation_inhibition_gradients",
    "plot_excitation_inhibition_weights",
    "visualize_dendritic_tree",
] 