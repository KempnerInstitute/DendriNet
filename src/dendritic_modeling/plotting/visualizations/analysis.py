"""
Unified analysis plotting module for dendritic modeling.

This module provides a comprehensive set of plotting utilities for analyzing
neural network models, including:
- Layer-wise analysis plots
- Ablation study results
- Noise robustness analysis
- Weight distribution plots
- Information-theoretic analysis
- Activation pattern visualization
"""

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

# Removed base.py imports as they were unused

# ============================================================================
# Helper Functions for Activation Extraction
# ============================================================================

def dendritic_activations_todict(
    dendrinet,
    exc_input_list,
    inh_input_list,
):
    """
    Extract DendriNet activations to dictionary format.

    Args:
        dendrinet: DendriNet model
        exc_input_list: list of excitatory inputs
        inh_input_list: list of inhibitory inputs

    Returns:
        Dictionary containing activations organized by soma and branch
    """
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    n_branch_layers = dendrinet.n_branch_layers

    plot_dict = {}
    for n in range(n_soma):
        plot_dict[f"soma{n}"] = {}

    output_list = [None] * len(exc_input_list)
    for i in range(n_branch_layers + 1):
        branches_per_soma = int(layer_sizes[i] / n_soma)

        chunk_list = []

        for j in range(len(exc_input_list)):
            output_list[j] = dendrinet.branch_layers[i](
                exc_input_list[j], inh_input_list[j], output_list[j]
            )
            chunk_list.append(output_list[j].chunk(n_soma, dim=-1))

        level = (
            "soma layer"
            if (i == n_branch_layers)
            else f"branch layer {n_branch_layers - i}"
        )

        for n in range(n_soma):
            plot_dict[f"soma{n}"][level] = {}
            for b in range(branches_per_soma):
                activation_list = []
                for k in range(len(exc_input_list)):
                    chunk = chunk_list[k][n][:, b]
                    activation_list.append(chunk)

                plot_dict[f"soma{n}"][level][f"branch{b}"] = {}
                plot_dict[f"soma{n}"][level][f"branch{b}"][
                    "activation list"
                ] = activation_list

    return plot_dict


def einet_activations_todict(einet, input_list):
    """
    Extract EINet activations to dictionary format.

    Args:
        einet: EINet model
        input_list: list of input tensors

    Returns:
        Dictionary containing activations organized by layer and cell type
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)

    plot_dict = {}

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dict[f"eilayer{i+1}"] = {}

        plot_dict[f"eilayer{i+1}"]["inh cells"] = dendritic_activations_todict(
            dendrinet=inh_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], inh_input_list[j]))
        inh_input_list = inh_temp

        plot_dict[f"eilayer{i+1}"]["exc cells"] = dendritic_activations_todict(
            dendrinet=exc_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
        )

        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], inh_input_list[j]))
        exc_input_list = exc_temp

    return plot_dict


# ============================================================================
# Analysis Plotting Functions  
# ============================================================================

# ============================================================================
# Analysis Plotters
# ============================================================================

# ============================================================================
# Specialized Activation Plotters
# ============================================================================

# ============================================================================
# Convenience Functions (for backward compatibility)
# ============================================================================

def plot_layer_contributions(results: dict[str, Any], **kwargs):
    """Convenience function for layer contribution plotting."""
    plotter = LayerAnalysisPlotter()
    return plotter.plot_layer_contributions(results, **kwargs)

def plot_ablation_results(results: dict[str, dict[str, float]], **kwargs):
    """Convenience function for ablation results plotting."""
    plotter = AblationPlotter()
    return plotter.plot_ablation_results(results, **kwargs)

def plot_noise_robustness(
    results: dict[str, list[float]], noise_levels: list[float], **kwargs
):
    """Convenience function for noise robustness plotting."""
    plotter = NoiseAnalysisPlotter()
    return plotter.plot_noise_robustness(results, noise_levels, **kwargs)

def plot_weight_distributions(weight_data: dict[str, list[np.ndarray]], **kwargs):
    """Convenience function for weight distribution plotting."""
    plotter = WeightDistributionPlotter()
    return plotter.plot_weight_distributions(weight_data, **kwargs)

def plot_information_analysis(results: dict[str, Any], **kwargs):
    """Convenience function for information analysis plotting."""
    plotter = InformationAnalysisPlotter()
    # Handle the filename/filename_prefix parameter mapping
    if "filename" in kwargs and "filename_prefix" not in kwargs:
        kwargs["filename_prefix"] = kwargs.pop("filename")
    return plotter.plot_information_metrics(results, **kwargs)

def plot_synapse_turnover(turnover_results: dict, save_path: str = None, filename: str = "synapse_turnover"):
    """
    Generate plots for synaptic turnover analysis results.
    
    Args:
        turnover_results: Dictionary containing turnover analysis results with structure:
                         - For old format: {"exc": {...}, "inh": {...}, "both": {...}}
                         - For new format: {"incremental": {...}, "cumulative": {...}, "n_snapshots": N}
        save_path: Directory to save the plot
        filename: Filename for the saved plot (without extension)
    
    Returns:
        List of matplotlib Figure objects
    """
    import matplotlib.pyplot as plt
    import os
    from dendritic_modeling import logger
    
    # Check if we have the new format with incremental and cumulative data
    if "incremental" in turnover_results and "cumulative" in turnover_results:
        # New format with both incremental and cumulative data
        figures = []
        
        # Create incremental turnover plot (t vs t-1)
        fig1 = _plot_turnover_data(
            turnover_results["incremental"],
            "Incremental Synaptic Turnover (t vs t-1)",
            "incremental"
        )
        figures.append(fig1)
        
        # Create cumulative turnover plot (t vs t=0)
        fig2 = _plot_turnover_data(
            turnover_results["cumulative"],
            "Cumulative Synaptic Turnover (t vs t=0)",
            "cumulative"
        )
        figures.append(fig2)
        
        # Save plots
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            # Save incremental plot
            incremental_path = os.path.join(save_path, f"{filename}_incremental.png")
            fig1.savefig(incremental_path, dpi=300, bbox_inches='tight')
            logger.info(f"Incremental turnover plot saved to {incremental_path}")
            
            # Save cumulative plot
            cumulative_path = os.path.join(save_path, f"{filename}_cumulative.png")
            fig2.savefig(cumulative_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cumulative turnover plot saved to {cumulative_path}")
            
            plt.close(fig1)
            plt.close(fig2)
        
        return figures
    else:
        # Old format - single plot
        fig = _plot_turnover_data(
            turnover_results,
            "Synaptic Turnover Analysis",
            "standard"
        )
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            full_path = os.path.join(save_path, f"{filename}.png")
            fig.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Synapse turnover plot saved to {full_path}")
            plt.close(fig)
        
        return [fig]

def _plot_turnover_data(turnover_data: dict, title: str, plot_type: str) -> plt.Figure:
    """
    Helper function to create a turnover plot.
    
    Args:
        turnover_data: Dictionary with "exc", "inh", "both" keys
        title: Main title for the plot
        plot_type: One of "incremental", "cumulative", or "standard"
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    from dendritic_modeling import logger
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Expected data structure
    expected_keys = ["exc", "inh", "both"]
    metric_keys = ["active", "stable", "transient"]
    
    if all(key in turnover_data for key in expected_keys):
        # Check if we have time series data
        exc_data = turnover_data["exc"]
        n_timepoints = len(exc_data["active"])
        
        # Create epochs array based on plot type
        if n_timepoints > 1:
            # Multiple snapshots during training
            if plot_type == "incremental":
                # For incremental, epochs start from 1 (first comparison is epoch 1 vs 0)
                epochs = list(range(1, n_timepoints + 1))
                epoch_label = "Epoch"
            elif plot_type == "cumulative":
                # For cumulative, epochs also start from 1 (comparing with epoch 0)
                epochs = list(range(1, n_timepoints + 1))
                epoch_label = "Epoch"
            else:
                # Standard format
                epochs = list(range(1, n_timepoints + 1))
                epoch_label = "Epoch"
        else:
            # Only one comparison point
            epochs = [0, 1]
            epoch_label = "Time"
            # Duplicate the single data point for visualization
            for synapse_type in ["exc", "inh", "both"]:
                for metric in ["active", "stable", "transient"]:
                    if len(turnover_data[synapse_type][metric]) == 1:
                        turnover_data[synapse_type][metric] = [
                            turnover_data[synapse_type][metric][0],
                            turnover_data[synapse_type][metric][0]
                        ]
        
        # Plot excitatory turnover
        ax = axes[0, 0]
        exc_data = turnover_data["exc"]
        
        ax.plot(epochs, exc_data["active"], label="Active", marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.plot(epochs, exc_data["stable"], label="Stable", marker='s', linestyle='--', linewidth=2, markersize=8)
        ax.plot(epochs, exc_data["transient"], label="Transient", marker='^', linestyle=':', linewidth=2, markersize=8)
        ax.set_title("Excitatory Synapses", fontsize=14)
        ax.set_xlabel(epoch_label, fontsize=12)
        ax.set_ylabel("Number of Synapses", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if n_timepoints == 1:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Initial", "Final"])
        
        # Plot inhibitory turnover
        ax = axes[0, 1]
        inh_data = turnover_data["inh"]
        
        ax.plot(epochs, inh_data["active"], label="Active", marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.plot(epochs, inh_data["stable"], label="Stable", marker='s', linestyle='--', linewidth=2, markersize=8)
        ax.plot(epochs, inh_data["transient"], label="Transient", marker='^', linestyle=':', linewidth=2, markersize=8)
        ax.set_title("Inhibitory Synapses", fontsize=14)
        ax.set_xlabel(epoch_label, fontsize=12)
        ax.set_ylabel("Number of Synapses", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if n_timepoints == 1:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Initial", "Final"])
        
        # Plot combined turnover
        ax = axes[1, 0]
        both_data = turnover_data["both"]
        
        ax.plot(epochs, both_data["active"], label="Active", marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.plot(epochs, both_data["stable"], label="Stable", marker='s', linestyle='--', linewidth=2, markersize=8)
        ax.plot(epochs, both_data["transient"], label="Transient", marker='^', linestyle=':', linewidth=2, markersize=8)
        ax.set_title("Combined (Excitatory + Inhibitory)", fontsize=14)
        ax.set_xlabel(epoch_label, fontsize=12)
        ax.set_ylabel("Number of Synapses", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if n_timepoints == 1:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Initial", "Final"])
        
        # Plot turnover rates (transient/active ratio)
        ax = axes[1, 1]
        exc_rate = [t/a if a > 0 else 0 for t, a in zip(exc_data["transient"], exc_data["active"])]
        inh_rate = [t/a if a > 0 else 0 for t, a in zip(inh_data["transient"], inh_data["active"])]
        both_rate = [t/a if a > 0 else 0 for t, a in zip(both_data["transient"], both_data["active"])]
        
        ax.plot(epochs, exc_rate, label="Excitatory Rate", marker='o', linestyle='-', linewidth=2, markersize=8)
        ax.plot(epochs, inh_rate, label="Inhibitory Rate", marker='s', linestyle='--', linewidth=2, markersize=8)
        ax.plot(epochs, both_rate, label="Combined Rate", marker='^', linestyle=':', linewidth=2, markersize=8)
        ax.set_title("Turnover Rates (Transient/Active)", fontsize=14)
        ax.set_xlabel(epoch_label, fontsize=12)
        ax.set_ylabel("Turnover Rate", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)  # Turnover rate is between 0 and 1
        if n_timepoints == 1:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Initial", "Final"])
        
    else:
        # Fallback for unknown data structure
        ax = axes[0, 0]
        ax.text(0.5, 0.5, f"Unexpected data structure\nAvailable keys: {list(turnover_data.keys())}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        
        # Hide other subplots
        for i in range(2):
            for j in range(2):
                if i != 0 or j != 0:
                    axes[i, j].set_visible(False)
        
        logger.warning(f"Unknown data structure for synapse turnover. Available keys: {list(turnover_data.keys())}")
    
    plt.tight_layout()
    return fig

def plot_dendrinet_activations(dendrinet, exc_input_list, inh_input_list, **kwargs):
    """Convenience function for DendriNet activation plotting."""
    plotter = ActivationPlotter()
    return plotter.plot_dendrinet_activations(
        dendrinet, exc_input_list, inh_input_list, **kwargs
    )

def plot_einet_activations(einet, input_list, **kwargs):
    """Convenience function for EINet activation plotting."""
    plotter = ActivationPlotter()
    return plotter.plot_einet_activations(einet, input_list, **kwargs)

def plot_performance_metrics(
    results: dict,
    save_path: Optional[str] = None,
    filename: str = "performance",
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """Generate bar plots for performance metrics across datasets."""
    plots = {}
    for metric_name, metric_values in results.items():
        if isinstance(metric_values, dict) and len(metric_values) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            datasets = list(metric_values.keys())
            values = list(metric_values.values())
            bars = ax.bar(datasets, values)
            ax.set_xlabel("Dataset")
            ax.set_ylabel(metric_name.replace("_", " ").title())
            ax.set_title(f"Performance: {metric_name.replace('_', ' ').title()}")
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                )
            plt.tight_layout()
            if save_path:
                plot_path = os.path.join(save_path, f"{filename}_{metric_name}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
            plots[metric_name] = (fig, ax)
    return plots

def plot_performance_evolution(
    performance_dir: str,
    save_path: Optional[str] = None,
    filename: str = "performance_evolution",
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """
    Plot performance metrics evolution over epochs from per-epoch performance files.
    
    Args:
        performance_dir: Directory containing per-epoch performance files (epoch1, epoch2, etc.)
                        Can be either the main performance dir or the epochs subdirectory
        save_path: Directory to save plots (optional)
        filename: Base filename for saved plots
        
    Returns:
        Dictionary mapping metric names to (figure, axes) tuples
    """
    import glob
    import json
    import re
    
    # Check if this is the epochs subdirectory or main directory
    epochs_subdir = os.path.join(performance_dir, "epochs")
    if os.path.exists(epochs_subdir) and performance_dir.endswith("performance"):
        # If we're in the main performance directory and epochs subfolder exists, use that
        search_dir = epochs_subdir
        print(f"Found epochs subdirectory, using: {search_dir}")
    else:
        # Use the provided directory (could be epochs subdir or main dir for backward compatibility)
        search_dir = performance_dir
    
    # Find all epoch files
    epoch_files = glob.glob(os.path.join(search_dir, "epoch*"))
    if not epoch_files:
        print(f"No epoch files found in {search_dir}")
        return {}
    
    # Extract epoch numbers and sort
    epoch_data = []
    for file_path in epoch_files:
        match = re.search(r'epoch(\d+)$', file_path)
        if match:
            epoch_num = int(match.group(1))
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                epoch_data.append((epoch_num, data))
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not load {file_path}: {e}")
                continue
    
    if not epoch_data:
        print("No valid epoch data found")
        return {}
    
    # Sort by epoch number
    epoch_data.sort(key=lambda x: x[0])
    epochs = [x[0] for x in epoch_data]
    
    print(f"Found {len(epoch_data)} epoch files spanning epochs {min(epochs)} to {max(epochs)}")
    
    # Collect metrics data
    metrics_over_time = {}
    for epoch_num, data in epoch_data:
        for metric_name, metric_values in data.items():
            if isinstance(metric_values, dict):
                if metric_name not in metrics_over_time:
                    metrics_over_time[metric_name] = {
                        'train': [], 'valid': [], 'test': []
                    }
                
                # Extract values for each dataset
                for dataset in ['train', 'valid', 'test']:
                    value = metric_values.get(dataset, None)
                    metrics_over_time[metric_name][dataset].append(value)
    
    if not metrics_over_time:
        print("No metrics data found in epoch files")
        return {}
    
    # Generate plots
    plots = {}
    for metric_name, dataset_data in metrics_over_time.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each dataset
        colors = {'train': '#1f77b4', 'valid': '#ff7f0e', 'test': '#2ca02c'}
        lines_plotted = 0
        
        for dataset, values in dataset_data.items():
            # Filter out None values
            valid_epochs = []
            valid_values = []
            for i, value in enumerate(values):
                if value is not None:
                    valid_epochs.append(epochs[i])
                    valid_values.append(value)
            
            if valid_values:
                ax.plot(valid_epochs, valid_values, 'o-', 
                       color=colors.get(dataset, '#d62728'), 
                       label=dataset.capitalize(),
                       linewidth=2, markersize=6)
                lines_plotted += 1
        
        if lines_plotted == 0:
            print(f"Warning: No valid data found for metric {metric_name}")
            plt.close(fig)
            continue
        
        ax.set_xlabel("Training Epoch", fontsize=12)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"{metric_name.replace('_', ' ').title()} Evolution During Training", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set integer ticks for epochs if reasonable number
        if len(epochs) <= 20:
            ax.set_xticks(epochs)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plot_path = os.path.join(save_path, f"{filename}_{metric_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Saved epoch evolution plot: {plot_path}")
            plt.close()
        
        plots[metric_name] = (fig, ax)
        
    return plots

def plot_per_layer_information(layer_results: dict[str, dict[str, Any]], **kwargs):
    """Convenience function for per-layer information plotting with mean and variance."""
    plotter = InformationAnalysisPlotter()
    return plotter.plot_per_layer_information(layer_results, **kwargs)
