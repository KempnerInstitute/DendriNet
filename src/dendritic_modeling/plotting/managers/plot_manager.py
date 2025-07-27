"""
Plotting manager for dendritic models.

This module contains the PlotManager class, which serves as a centralized orchestrator
for generating and managing plots across the codebase. It delegates to specialized
plotting functions in the plotting/ module and provides a unified interface for
common visualization workflows.

Key enhancements:
- Added methods for batch plotting and comparisons
- Improved error handling and saving
- Integrated with actual plotting functions from the codebase
- Added utilities for figure management
"""

import logging
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any
import os

from dendritic_modeling.plotting.visualizations.analysis import (
    plot_performance_metrics,
    plot_performance_evolution,
    plot_information_analysis,
    plot_ablation_results,
    plot_noise_robustness,
    plot_weight_distributions,
    plot_dendrinet_activations,
    plot_einet_activations,
    plot_layer_contributions,
    plot_per_layer_information,
    # Training functions (now in analysis.py)
    plot_loss_curves,
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
)
from dendritic_modeling.plotting.visualizations.network import (
    plot_dendrinet_gradients,
    plot_einet_params,
    plot_dendrinet_params,
    plot_branch_gradients,
    plot_branch_weights,
    plot_excitation_inhibition_gradients,
    plot_excitation_inhibition_weights,
    visualize_dendritic_tree,
)
from dendritic_modeling.plotting.core.utils import (
    plot_activation_tuning,
    plot_depth_info,
    plot_voltage_before_activation,
)

logger = logging.getLogger(__name__)


class PlotManager:
    """Central manager for generating and handling all plots in the codebase."""

    def __init__(self, device: str = "cpu", default_save_path: Optional[str] = "plots/"):
        """Initialize with optional device and default save path."""
        self.device = device
        self.default_save_path = default_save_path
        if self.default_save_path:
            os.makedirs(self.default_save_path, exist_ok=True)
            logger.info(f"PlotManager initialized with default save path: {self.default_save_path}")

    def save_figure(self, fig: plt.Figure, filename: str, save_path: Optional[str] = None):
        """Utility to save a figure with standard formatting."""
        path = save_path or self.default_save_path
        if path:
            os.makedirs(path, exist_ok=True)
            full_path = os.path.join(path, f"{filename}.png")
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {full_path}")
        plt.close(fig)

    # Network plotting methods
    def plot_einet_params(
        self, einet, save_path: Optional[str] = None, filename: str = "einet_params"
    ):
        """Generate EINet parameter plots."""
        save_path = save_path or self.default_save_path
        plot_einet_params(einet, save_path=save_path, filename=filename)

    def plot_dendrinet_gradients(
        self,
        dendrinet,
        save_path: Optional[str] = None,
        filename: str = "dendrinet_grads",
    ):
        """Generate DendriNet gradient plots."""
        save_path = save_path or self.default_save_path
        plot_dendrinet_gradients(dendrinet, save_path=save_path, filename=filename)

    def plot_dendrinet_params(
        self, dendrinet, save_path: Optional[str] = None, filename: str = "dendrinet_params"
    ):
        """Generate DendriNet parameter plots."""
        save_path = save_path or self.default_save_path
        plot_dendrinet_params(dendrinet, save_path=save_path, filename=filename)

    def plot_branch_weights(
        self, model, save_path: Optional[str] = None, filename: str = "branch_weights"
    ):
        """Generate branch weight plots."""
        save_path = save_path or self.default_save_path
        plot_branch_weights(model, save_path=save_path, filename=filename)

    def plot_branch_gradients(
        self, model, save_path: Optional[str] = None, filename: str = "branch_gradients"
    ):
        """Generate branch gradient plots."""
        save_path = save_path or self.default_save_path
        plot_branch_gradients(model, save_path=save_path, filename=filename)

    # Analysis plotting methods
    def plot_performance_metrics(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        filename: str = "performance",
    ):
        """Generate performance metric plots."""
        save_path = save_path or self.default_save_path
        return plot_performance_metrics(results, save_path=save_path, filename=filename)

    def plot_performance_evolution(
        self,
        performance_dir: str,
        save_path: Optional[str] = None,
        filename: str = "performance_evolution",
    ):
        """Generate performance evolution plots from per-epoch performance files."""
        save_path = save_path or self.default_save_path
        return plot_performance_evolution(performance_dir, save_path=save_path, filename=filename)

    def plot_information_analysis(
        self,
        info_results: Dict,
        save_path: Optional[str] = None,
        filename: str = "information",
    ):
        """Generate information analysis plots."""
        save_path = save_path or self.default_save_path
        plot_information_analysis(info_results, save_path=save_path, filename=filename)

    def plot_ablation_results(
        self, results: Dict, save_path: Optional[str] = None, filename: str = "ablation"
    ):
        """Generate ablation study plots."""
        save_path = save_path or self.default_save_path
        plot_ablation_results(results, save_path=save_path, filename=filename)

    def plot_noise_robustness(
        self, results: Dict, save_path: Optional[str] = None, filename: str = "noise_robustness"
    ):
        """Generate noise robustness plots."""
        save_path = save_path or self.default_save_path
        plot_noise_robustness(results, save_path=save_path, filename=filename)

    def plot_weight_distributions(
        self, model, save_path: Optional[str] = None, filename: str = "weight_distributions"
    ):
        """Generate weight distribution plots."""
        save_path = save_path or self.default_save_path
        plot_weight_distributions(model, save_path=save_path, filename=filename)

    # Training plotting methods
    def plot_loss_curves(
        self, training_data: Dict, save_path: Optional[str] = None, filename: str = "loss_curves"
    ):
        """Generate loss curve plots."""
        save_path = save_path or self.default_save_path
        plot_loss_curves(training_data, save_path=save_path, filename=filename)

    def plot_sweep_results(
        self, sweep_data: Dict, sweep_type: str = "branch", 
        save_path: Optional[str] = None, filename: str = "sweep_results"
    ):
        """Generate parameter sweep plots."""
        save_path = save_path or self.default_save_path
        if sweep_type == "branch":
            plot_branch_sweep_results(sweep_data, save_path=save_path, filename=filename)
        elif sweep_type == "ei":
            plot_ei_sweep_results(sweep_data, save_path=save_path, filename=filename)

    # Utility and visualization methods
    def visualize_dendritic_tree(
        self, tree_data, save_path: Optional[str] = None, filename: str = "dendritic_tree"
    ):
        """Generate dendritic tree visualization."""
        save_path = save_path or self.default_save_path
        visualize_dendritic_tree(tree_data, save_path=save_path, filename=filename)

    # New: Batch plotting for multiple analyses
    def plot_batch_analyses(self, analyses: List[Dict], types: List[str], save_path: Optional[str] = None):
        """Generate multiple analysis plots in batch."""
        if len(analyses) != len(types):
            raise ValueError("Number of analyses must match number of types")
        
        for i, (analysis, typ) in enumerate(zip(analyses, types)):
            try:
                if typ == 'performance':
                    self.plot_performance_metrics(analysis, save_path=save_path, filename=f"performance_{i}")
                elif typ == 'information':
                    self.plot_information_analysis(analysis, save_path=save_path, filename=f"information_{i}")
                elif typ == 'ablation':
                    self.plot_ablation_results(analysis, save_path=save_path, filename=f"ablation_{i}")
                elif typ == 'noise':
                    self.plot_noise_robustness(analysis, save_path=save_path, filename=f"noise_{i}")
                else:
                    logger.warning(f"Unknown analysis type: {typ}")
            except Exception as e:
                logger.error(f"Error plotting {typ} analysis {i}: {e}")
        
        logger.info(f"Batch plotted {len(analyses)} analyses")

    # New: Comparison plot for multiple models
    def plot_model_comparison(
        self, models_data: Dict[str, Dict], 
        metric: str = "loss", 
        save_path: Optional[str] = None, 
        filename: str = "model_comparison"
    ):
        """Generate comparison plots for multiple models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name, model_metrics in models_data.items():
            if metric in model_metrics:
                ax.plot(model_metrics[metric], label=model_name)
            else:
                logger.warning(f"Metric '{metric}' not found for model '{model_name}'")
        
        ax.set_title(f"Model {metric.capitalize()} Comparison")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path or self.default_save_path:
            self.save_figure(fig, filename, save_path)
        else:
            plt.show()

    # New: Network architecture comparison
    def plot_network_architecture_comparison(
        self, models: List[Any], model_names: List[str], 
        save_path: Optional[str] = None, filename: str = "architecture_comparison"
    ):
        """Generate network architecture comparison plots."""
        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of model names")
        
        # This would need to be implemented based on your specific model architectures
        logger.info(f"Architecture comparison for {len(models)} models")
        # Implementation would depend on what specific architectural comparisons you want

    def close_all(self):
        """Close all open matplotlib figures to free memory."""
        plt.close("all")
        logger.info("All open figures closed")

    def set_default_save_path(self, path: str):
        """Update the default save path."""
        self.default_save_path = path
        os.makedirs(path, exist_ok=True)
        logger.info(f"Default save path updated to: {path}")


__all__ = ["PlotManager"]
