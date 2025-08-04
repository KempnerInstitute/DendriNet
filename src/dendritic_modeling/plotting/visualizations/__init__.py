"""Visualization plotting functions for analysis, networks, and correlations."""

# Analysis visualizations (updated for modular structure)
from .information_analysis import (
    plot_information_analysis,
    plot_per_layer_information,
)

from .general_analysis_plots import (
    plot_ablation_results,
    plot_per_layer_ablation,
    plot_layer_contributions,
    plot_weight_distributions,
    plot_performance_metrics,
    plot_performance_evolution,
    plot_synapse_turnover,
    plot_active_synapses_per_branch,
    plot_active_synapses_summary_by_layer,
    plot_pruning_performance_comparison,
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

# Network visualizations module exists but all functions have been removed as unused

__all__ = [
    # Information analysis (now modular)
    "plot_information_analysis",
    "plot_per_layer_information",
    
    # General analysis
    "plot_ablation_results",
    "plot_per_layer_ablation",
    "plot_layer_contributions", 
    "plot_weight_distributions",
    "plot_performance_metrics",
    "plot_performance_evolution",
    "plot_synapse_turnover",
    "plot_active_synapses_per_branch",
    "plot_active_synapses_summary_by_layer",
    "plot_pruning_performance_comparison",
    
    # Training/sweep functions
    "plot_loss_curves",
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    
    # Legacy data utilities
    "dendritic_activations_todict",
    "einet_activations_todict",
] 