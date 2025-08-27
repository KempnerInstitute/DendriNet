"""
Correlation analysis plotting functions.

This module provides visualization functions for hierarchical correlation analysis results,
similar to the information analysis plots.
"""

import os
from typing import Any, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from .plotting_utils import (
    create_figure,
    save_plot,
    handle_plot_error,
    get_color_scheme,
)


def plot_hierarchical_correlations(
    results: dict[str, Any],
    save_path: str,
    filename_prefix: str = "correlation"
) -> None:
    """Plot hierarchical correlation analysis results.
    
    Args:
        results: Correlation analysis results with hierarchical structure
        save_path: Directory to save plots
        filename_prefix: Prefix for saved files
    """
    try:
        # Create output directory
        os.makedirs(save_path, exist_ok=True)
        
        # Plot network-level summary
        if "network_level" in results:
            plot_network_summary(
                results["network_level"], 
                save_path=os.path.join(save_path, f"{filename_prefix}_network_summary.png")
            )
            
        # Plot per-layer results if available
        if "per_layer_analysis" in results and results.get("per_layer_analysis_enabled", False):
            plot_per_layer_correlations(
                results["per_layer_analysis"],
                save_path=os.path.join(save_path, f"{filename_prefix}_per_layer.png")
            )
            
        # Plot per-EI network results if available
        if "per_einet_analysis" in results and results.get("per_einet_analysis_enabled", False):
            plot_per_ei_correlations(
                results["per_einet_analysis"],
                save_path=os.path.join(save_path, f"{filename_prefix}_per_ei_network.png")
            )
            
    except Exception as e:
        handle_plot_error("hierarchical correlations", e)


def plot_network_summary(network_results: dict[str, Any], save_path: str) -> None:
    """Plot network-level correlation summary.
    
    Args:
        network_results: Network-level correlation results
        save_path: Path to save the plot
    """
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Get summary statistics
        summary_stats = network_results.get("summary_stats", {})
        
        # Plot 1: Mean correlation strengths
        ax1 = fig.add_subplot(gs[0, :])
        plot_correlation_strengths(summary_stats, ax1)
        ax1.set_title("Mean Correlation Strengths (Network-wide)", fontsize=14, fontweight='bold')
        
        # Plot 2: Total vs Noise correlations comparison
        ax2 = fig.add_subplot(gs[1, 0])
        plot_total_vs_noise_correlations(summary_stats, ax2)
        ax2.set_title("Total vs Noise Correlations", fontsize=12)
        
        # Plot 3: Signal correlations (if available)
        ax3 = fig.add_subplot(gs[1, 1])
        plot_signal_correlations(summary_stats, ax3)
        ax3.set_title("Signal Correlations", fontsize=12)
        
        # Plot 4: E-I balance
        ax4 = fig.add_subplot(gs[1, 2])
        plot_ei_balance(summary_stats, ax4)
        ax4.set_title("E-I Correlation Balance", fontsize=12)
        
        # Plot 5: Tuning summary (if available)
        if "tuning" in network_results:
            ax5 = fig.add_subplot(gs[2, :])
            plot_tuning_summary(network_results["tuning"], ax5)
            ax5.set_title("Tuning Curves Summary", fontsize=12)
        
        plt.suptitle("Network-wide Correlation Analysis", fontsize=16, fontweight='bold')
        save_plot(fig, save_path)
        
    except Exception as e:
        handle_plot_error("network summary", e)


def plot_correlation_strengths(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot mean correlation strengths for different correlation types."""
    correlation_types = ['EE', 'II', 'EI', 'E_Vout', 'I_Vout']
    conditions = ['total', 'noise']
    
    data = []
    labels = []
    colors = []
    color_scheme = get_color_scheme()
    
    for corr_type in correlation_types:
        for cond in conditions:
            key = f"{corr_type}_{cond}_mean"
            if key in stats:
                data.append(stats[key])
                labels.append(f"{corr_type}\n{cond}")
                colors.append(color_scheme['excitatory'] if 'E' in corr_type else color_scheme['inhibitory'])
    
    if data:
        bars = ax.bar(range(len(data)), data, color=colors, alpha=0.7)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Mean |Correlation|')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)


def plot_total_vs_noise_correlations(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot comparison of total vs noise correlations."""
    correlation_types = ['EE', 'II', 'EI']
    
    total_vals = []
    noise_vals = []
    
    for corr_type in correlation_types:
        total_key = f"{corr_type}_total_mean"
        noise_key = f"{corr_type}_noise_mean"
        
        if total_key in stats and noise_key in stats:
            total_vals.append(stats[total_key])
            noise_vals.append(stats[noise_key])
    
    if total_vals and noise_vals:
        x = np.arange(len(correlation_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, total_vals, width, label='Total', alpha=0.7)
        bars2 = ax.bar(x + width/2, noise_vals, width, label='Noise', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(correlation_types)
        ax.set_ylabel('Mean |Correlation|')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')


def plot_signal_correlations(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot signal correlations (between-class differences)."""
    correlation_types = ['EE', 'II', 'EI', 'E_Vout', 'I_Vout']
    
    signal_vals = []
    labels = []
    
    for corr_type in correlation_types:
        signal_key = f"{corr_type}_signal_mean"
        if signal_key in stats:
            signal_vals.append(stats[signal_key])
            labels.append(corr_type)
    
    if signal_vals:
        bars = ax.bar(range(len(signal_vals)), signal_vals, alpha=0.7)
        ax.set_xticks(range(len(signal_vals)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Mean Signal Correlation')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')


def plot_ei_balance(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot E-I correlation balance metrics."""
    metrics = {}
    
    # E/I ratio for different conditions
    if 'EE_total_mean' in stats and 'II_total_mean' in stats:
        metrics['E/I Total'] = stats['EE_total_mean'] / max(stats['II_total_mean'], 1e-6)
    
    if 'EE_noise_mean' in stats and 'II_noise_mean' in stats:
        metrics['E/I Noise'] = stats['EE_noise_mean'] / max(stats['II_noise_mean'], 1e-6)
    
    # E-I cross-correlation strength
    if 'EI_total_mean' in stats:
        metrics['E-I Cross'] = stats['EI_total_mean']
    
    if metrics:
        bars = ax.bar(range(len(metrics)), list(metrics.values()), alpha=0.7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Balance')
        ax.grid(True, alpha=0.3, axis='y')


def plot_tuning_summary(tuning: dict[str, np.ndarray], ax: plt.Axes) -> None:
    """Plot summary of tuning curves."""
    if 'E_tuning' in tuning and 'I_tuning' in tuning:
        # Create violin plots for tuning distributions
        data = [tuning['E_tuning'].flatten(), tuning['I_tuning'].flatten()]
        
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)
        
        # Customize colors
        colors = get_color_scheme()
        parts['bodies'][0].set_facecolor(colors['excitatory'])
        parts['bodies'][1].set_facecolor(colors['inhibitory'])
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Excitatory', 'Inhibitory'])
        ax.set_ylabel('Tuning Strength')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')


def plot_per_layer_correlations(
    per_layer_results: dict[str, dict[str, Any]],
    save_path: str
) -> None:
    """Plot per-layer correlation results.
    
    Args:
        per_layer_results: Dictionary with layer names as keys
        save_path: Path to save the plot
    """
    try:
        n_layers = len(per_layer_results)
        fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5 * n_layers), squeeze=False)
        
        for idx, (layer_name, layer_data) in enumerate(per_layer_results.items()):
            # Plot correlation matrix heatmap
            if "correlations" in layer_data:
                plot_layer_correlation_matrix(
                    layer_data["correlations"], 
                    axes[idx, 0], 
                    f"Layer: {layer_name}"
                )
            
            # Plot summary statistics
            if "summary_stats" in layer_data:
                plot_layer_summary_stats(
                    layer_data["summary_stats"], 
                    axes[idx, 1]
                )
            
            # Plot tuning if available
            if "tuning" in layer_data:
                plot_layer_tuning(
                    layer_data["tuning"], 
                    axes[idx, 2]
                )
        
        plt.suptitle("Per-Layer Correlation Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_plot(fig, save_path)
        
    except Exception as e:
        handle_plot_error("per-layer correlations", e)


def plot_correlation_matrices(
    correlations: dict[str, dict[str, np.ndarray]],
    save_path: str,
    max_size: int = 50
) -> None:
    """Plot correlation matrices for different types and conditions.
    
    Args:
        correlations: Dictionary with correlation matrices
        save_path: Path to save the plot
        max_size: Maximum size to display (for large matrices)
    """
    try:
        # Count available correlation types
        corr_types = ['EE', 'II', 'EI']
        conditions = ['total', 'noise']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for row, condition in enumerate(conditions):
            for col, corr_type in enumerate(corr_types):
                ax = axes[row, col]
                
                if corr_type in correlations and condition in correlations[corr_type]:
                    matrix = correlations[corr_type][condition]
                    
                    # Limit size for visualization
                    if matrix.shape[0] > max_size:
                        matrix = matrix[:max_size, :max_size]
                    
                    # Plot heatmap
                    im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', 
                                  vmin=-1, vmax=1, interpolation='nearest')
                    ax.set_title(f'{corr_type} - {condition}')
                    ax.set_xlabel('Neuron Index')
                    ax.set_ylabel('Neuron Index')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'Not Available', ha='center', va='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])
        
        plt.suptitle("Correlation Matrices", fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_plot(fig, save_path)
        
    except Exception as e:
        handle_plot_error("correlation matrices", e)


def plot_tuning_curves(
    tuning: dict[str, np.ndarray],
    save_path: str
) -> None:
    """Plot detailed tuning curves for E, I, and Vout.
    
    Args:
        tuning: Dictionary with tuning arrays
        save_path: Path to save the plot
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot histograms
        if 'E_tuning' in tuning:
            axes[0, 0].hist(tuning['E_tuning'], bins=50, alpha=0.7, color='red')
            axes[0, 0].set_xlabel('Tuning Strength')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Excitatory Tuning Distribution')
            axes[0, 0].axvline(0, color='k', linestyle='--', alpha=0.5)
        
        if 'I_tuning' in tuning:
            axes[0, 1].hist(tuning['I_tuning'], bins=50, alpha=0.7, color='blue')
            axes[0, 1].set_xlabel('Tuning Strength')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Inhibitory Tuning Distribution')
            axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
        
        # Plot E vs I tuning scatter
        if 'E_tuning' in tuning and 'I_tuning' in tuning:
            # Limit points for visualization
            n_points = min(1000, len(tuning['E_tuning']))
            indices = np.random.choice(len(tuning['E_tuning']), n_points, replace=False)
            
            axes[1, 0].scatter(tuning['E_tuning'][indices], 
                             tuning['I_tuning'][indices], 
                             alpha=0.5, s=20)
            axes[1, 0].set_xlabel('Excitatory Tuning')
            axes[1, 0].set_ylabel('Inhibitory Tuning')
            axes[1, 0].set_title('E vs I Tuning Correlation')
            axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
            
            # Add correlation coefficient
            corr = np.corrcoef(tuning['E_tuning'], tuning['I_tuning'])[0, 1]
            axes[1, 0].text(0.05, 0.95, f'r = {corr:.3f}', 
                          transform=axes[1, 0].transAxes, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Plot Vout tuning if available
        if 'Vout_tuning' in tuning:
            axes[1, 1].hist(tuning['Vout_tuning'], bins=50, alpha=0.7, color='green')
            axes[1, 1].set_xlabel('Tuning Strength')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Output (Vout) Tuning Distribution')
            axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.5)
        
        plt.suptitle("Tuning Curves Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_plot(fig, save_path)
        
    except Exception as e:
        handle_plot_error("tuning curves", e)


def plot_layer_correlation_matrix(
    correlations: dict[str, dict[str, np.ndarray]],
    ax: plt.Axes,
    title: str
) -> None:
    """Plot a single layer's correlation matrix."""
    # For visualization, we'll show the E-I correlation matrix
    if 'EI' in correlations and 'total' in correlations['EI']:
        matrix = correlations['EI']['total']
        
        # Limit size for visualization
        max_size = 30
        if matrix.shape[0] > max_size or matrix.shape[1] > max_size:
            matrix = matrix[:max_size, :max_size]
        
        im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', 
                      vmin=-0.5, vmax=0.5, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Inhibitory')
        ax.set_ylabel('Excitatory')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, 'No E-I correlations', ha='center', va='center',
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_layer_summary_stats(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot summary statistics for a single layer."""
    # Select key statistics to display
    key_stats = ['EE_total_mean', 'II_total_mean', 'EI_total_mean', 
                'E_Vout_total_mean', 'I_Vout_total_mean']
    
    values = []
    labels = []
    
    for stat in key_stats:
        if stat in stats:
            values.append(stats[stat])
            labels.append(stat.replace('_total_mean', '').replace('_', '-'))
    
    if values:
        bars = ax.bar(range(len(values)), values, alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel('Mean |Correlation|')
        ax.set_title('Correlation Summary')
        ax.grid(True, alpha=0.3, axis='y')


def plot_layer_tuning(tuning: dict[str, np.ndarray], ax: plt.Axes) -> None:
    """Plot tuning summary for a single layer."""
    if 'E_tuning' in tuning and 'I_tuning' in tuning:
        mean_e = np.mean(np.abs(tuning['E_tuning']))
        mean_i = np.mean(np.abs(tuning['I_tuning']))
        
        bars = ax.bar([0, 1], [mean_e, mean_i], color=['red', 'blue'], alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['E', 'I'])
        ax.set_ylabel('Mean |Tuning|')
        ax.set_title('Tuning Strength')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, val in zip(bars, [mean_e, mean_i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')


def plot_per_ei_correlations(
    ei_results: dict[str, dict[str, Any]],
    save_path: str
) -> None:
    """Plot per-EI network correlation results.
    
    Args:
        ei_results: Dictionary with E/I specific results
        save_path: Path to save the plot
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot excitatory network results
        if "excitatory_results" in ei_results:
            exc_data = ei_results["excitatory_results"]
            if "summary_stats" in exc_data:
                plot_ei_network_stats(exc_data["summary_stats"], axes[0, 0], "Excitatory Network")
            
        # Plot inhibitory network results
        if "inhibitory_results" in ei_results:
            inh_data = ei_results["inhibitory_results"]
            if "summary_stats" in inh_data:
                plot_ei_network_stats(inh_data["summary_stats"], axes[0, 1], "Inhibitory Network")
            
        # Plot combined results
        if "combined_results" in ei_results:
            comb_data = ei_results["combined_results"]
            if "correlations" in comb_data and "EI" in comb_data["correlations"]:
                plot_ei_cross_correlation(comb_data["correlations"]["EI"], axes[0, 2])
        
        # Plot comparison across networks
        plot_ei_comparison(ei_results, axes[1, :])
        
        plt.suptitle("Per-EI Network Correlation Analysis", fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_plot(fig, save_path)
        
    except Exception as e:
        handle_plot_error("per-EI correlations", e)


def plot_ei_network_stats(stats: dict[str, float], ax: plt.Axes, title: str) -> None:
    """Plot statistics for a single E or I network."""
    # Extract relevant stats
    if title == "Excitatory Network":
        stat_keys = ['EE_total_mean', 'EE_noise_mean', 'E_Vout_total_mean']
        labels = ['E-E Total', 'E-E Noise', 'E-Vout']
        color = 'red'
    else:
        stat_keys = ['II_total_mean', 'II_noise_mean', 'I_Vout_total_mean']
        labels = ['I-I Total', 'I-I Noise', 'I-Vout']
        color = 'blue'
    
    values = [stats.get(key, 0) for key in stat_keys]
    
    bars = ax.bar(range(len(values)), values, color=color, alpha=0.7)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Mean |Correlation|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')


def plot_ei_cross_correlation(ei_corr: dict[str, np.ndarray], ax: plt.Axes) -> None:
    """Plot E-I cross-correlation summary."""
    conditions = ['total', 'noise', 'signal']
    values = []
    
    for cond in conditions:
        if cond in ei_corr:
            values.append(np.mean(np.abs(ei_corr[cond])))
        else:
            values.append(0)
    
    bars = ax.bar(range(len(conditions)), values, color='purple', alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Mean |E-I Correlation|')
    ax.set_title('E-I Cross-Correlation')
    ax.grid(True, alpha=0.3, axis='y')


def plot_ei_comparison(ei_results: dict[str, Any], axes: np.ndarray) -> None:
    """Plot comparison across E and I networks."""
    # Aggregate comparison plot
    ax = axes[0]
    
    exc_stats = ei_results.get("excitatory_results", {}).get("summary_stats", {})
    inh_stats = ei_results.get("inhibitory_results", {}).get("summary_stats", {})
    
    # Compare self-correlations
    exc_self = exc_stats.get('EE_total_mean', 0)
    inh_self = inh_stats.get('II_total_mean', 0)
    
    bars = ax.bar([0, 1], [exc_self, inh_self], color=['red', 'blue'], alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['E-E', 'I-I'])
    ax.set_ylabel('Mean Self-Correlation')
    ax.set_title('E vs I Self-Correlation Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for ax in axes[1:]:
        ax.axis('off')