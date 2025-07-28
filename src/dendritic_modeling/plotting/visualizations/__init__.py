"""Visualization plotting functions for analysis, networks, and correlations."""

# Analysis visualizations (now includes training functions)
from .analysis import (
    dendritic_activations_todict,
    einet_activations_todict,
    plot_ablation_results,
    plot_information_analysis,
    plot_layer_contributions,
    plot_noise_robustness,
    plot_weight_distributions,
    plot_performance_metrics,
    plot_performance_evolution,
    plot_per_layer_information,
    # Training and sweep functions (merged from training.py)
    plot_loss_curves,
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
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
    # Analysis (includes former training functions)
    "dendritic_activations_todict",
    "einet_activations_todict",
    "plot_ablation_results",
    "plot_information_analysis",
    "plot_layer_contributions", 
    "plot_noise_robustness",
    "plot_weight_distributions",
    "plot_performance_metrics",
    "plot_performance_evolution",
    "plot_per_layer_information",
    
    # Training/sweep functions (now in analysis.py)
    "plot_loss_curves",
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    
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