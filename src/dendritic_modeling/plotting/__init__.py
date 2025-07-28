"""
Plotting package for dendritic modeling.

This package contains visualization and plotting utilities organized into logical modules:
- core: Core utilities and helper functions
- visualizations: Specialized plotting functions for analysis, networks, training, and correlations  
- managers: Centralized plotting orchestration and management

The reorganized structure provides better organization and maintainability.
"""

# Import from core utilities
from .core import (
    plot_activation_tuning,
    plot_depth_info,
    plot_voltage_before_activation,
)

# Import from visualizations
from .visualizations import (
    # Analysis functions
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
    
    # Network functions
    plot_branch_gradients,
    plot_branch_weights,
    plot_dendrinet_gradients,
    plot_dendrinet_params,
    plot_einet_params,
    plot_excitation_inhibition_gradients,
    plot_excitation_inhibition_weights,
    visualize_dendritic_tree,
    
    # Training functions
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_loss_curves,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
)

# Import managers
from .managers import PlotManager

__all__ = [
    # Core utilities
    "plot_activation_tuning",
    "plot_depth_info",
    "plot_voltage_before_activation",
    
    # Analysis functions
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
    
    # Network functions
    "plot_branch_gradients",
    "plot_branch_weights", 
    "plot_dendrinet_gradients",
    "plot_dendrinet_params",
    "plot_einet_params",
    "plot_excitation_inhibition_gradients",
    "plot_excitation_inhibition_weights",
    "visualize_dendritic_tree",
    
    # Training functions
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    "plot_loss_curves",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    
    # Managers
    "PlotManager",
]
