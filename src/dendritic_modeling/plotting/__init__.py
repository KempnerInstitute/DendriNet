"""
Plotting package for dendritic modeling.

This package contains visualization and plotting utilities organized into logical modules:
- core: Core utilities and helper functions
- visualizations: Specialized plotting functions for analysis, networks, training, and correlations  
- managers: Centralized plotting orchestration and management

The reorganized structure provides better organization and maintainability.
"""

# Core utilities - removed unused plotting functions

# Import from visualizations
from .visualizations import (
    # Analysis functions
    dendritic_activations_todict,
    einet_activations_todict,
    plot_ablation_results,
    plot_information_analysis,
    plot_layer_contributions,
    plot_weight_distributions,
    plot_performance_metrics,
    plot_performance_evolution,
    plot_per_layer_information,
    plot_synapse_turnover,
    plot_active_synapses_per_branch,
    
    
    # Training functions
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_loss_curves,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
)

# Note: PlotManager was removed as it was unused

__all__ = [
    # Analysis functions
    "dendritic_activations_todict",
    "einet_activations_todict", 
    "plot_ablation_results",
    "plot_information_analysis",
    "plot_layer_contributions",

    "plot_weight_distributions",
    "plot_performance_metrics",
    "plot_performance_evolution",
    "plot_per_layer_information",
    "plot_synapse_turnover",
    "plot_active_synapses_per_branch",
    
    
    # Training functions
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    "plot_loss_curves",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",

]
