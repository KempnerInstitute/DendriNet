"""Visualization plotting functions for analysis, networks, training, and correlations."""

# Analysis visualizations
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
)

# Training visualizations
from .training import (
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_loss_curves,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
)

# Correlation visualizations
from .correlation_plots import (
    plot_correlation_matrices,
    plot_tuning_curves,
    plot_correlation_vs_information,
)

__all__ = [
    # Analysis
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
    
    # Network
    "plot_branch_gradients",
    "plot_branch_weights",
    "plot_dendrinet_gradients", 
    "plot_dendrinet_params",
    "plot_einet_params",
    "plot_excitation_inhibition_gradients",
    "plot_excitation_inhibition_weights",
    
    # Training
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    "plot_loss_curves",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    
    # Correlation
    "plot_correlation_matrices",
    "plot_tuning_curves",
    "plot_correlation_vs_information",
] 