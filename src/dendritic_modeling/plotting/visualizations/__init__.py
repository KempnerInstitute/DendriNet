"""Visualization plotting functions for analysis, networks, and correlations."""

# Analysis visualizations (updated for modular structure)
# Data extraction utilities
from ..core.utils import dendritic_activations_todict, einet_activations_todict
from .general_analysis_plots import (  # Training and sweep functions
    plot_ablation_results,
    plot_active_synapses_per_branch,
    plot_active_synapses_summary_by_layer,
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_layer_contributions,
    plot_loss_curves,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
    plot_per_layer_ablation,
    plot_performance_evolution,
    plot_performance_metrics,
    plot_pruning_performance_comparison,
    plot_synapse_turnover,
    plot_weight_distributions,
)
from .information_analysis import (
    plot_information_analysis, 
    plot_per_layer_information,
    plot_structured_ei_network_analysis,
    plot_layer_wise_aggregated_analysis,
    plot_network_wise_aggregated_analysis,
    plot_fully_aggregated_analysis,
)

# Network visualizations module exists but all functions have been removed as unused

__all__ = [
    # Legacy data utilities
    "dendritic_activations_todict",
    "einet_activations_todict",
    # General analysis
    "plot_ablation_results",
    "plot_active_synapses_per_branch",
    "plot_active_synapses_summary_by_layer",
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    # Information analysis (now modular)
    "plot_information_analysis",
    "plot_layer_contributions",
    "plot_structured_ei_network_analysis",
    "plot_layer_wise_aggregated_analysis",
    "plot_network_wise_aggregated_analysis",
    "plot_fully_aggregated_analysis",
    # Training/sweep functions
    "plot_loss_curves",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    "plot_per_layer_ablation",
    "plot_per_layer_information",
    "plot_performance_evolution",
    "plot_performance_metrics",
    "plot_pruning_performance_comparison",
    "plot_synapse_turnover",
    "plot_weight_distributions",
]
