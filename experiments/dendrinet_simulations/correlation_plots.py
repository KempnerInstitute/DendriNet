"""
Correlation analysis plotting utilities for dendrinet simulations.

This module contains plotting functions for visualizing correlation analysis results
in dendritic network simulation experiments. Moved from the main plotting package
to be colocated with dendrinet simulation experiments.
"""

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

def plot_correlation_matrices(
    correlation_results: dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (15, 10),
    cmap: str = "coolwarm",
) -> plt.Figure:
    """
    Plot correlation matrices for different layer types.

    Args:
        correlation_results: Dictionary with layer names as keys and correlation matrices as values
        save_path: Path to save the figure
        figsize: Figure size
        cmap: Colormap for correlation matrices

    Returns:
        Figure object
    """
    n_layers = len(correlation_results)
    fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=figsize)
    axes = axes.flatten() if n_layers > 1 else [axes]

    for idx, (layer_name, corr_matrix) in enumerate(correlation_results.items()):
        ax = axes[idx]

        # Plot correlation matrix
        im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{layer_name} Correlations")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Neuron Index")

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(len(correlation_results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig

def plot_tuning_curves(
    tuning_results: dict[str, dict[str, np.ndarray]],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot tuning curves for different properties.

    Args:
        tuning_results: Dictionary with property names as keys and tuning data as values
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Figure object
    """
    n_properties = len(tuning_results)
    fig, axes = plt.subplots(2, (n_properties + 1) // 2, figsize=figsize)
    axes = axes.flatten() if n_properties > 1 else [axes]

    for idx, (property_name, tuning_data) in enumerate(tuning_results.items()):
        ax = axes[idx]

        # Extract stimulus values and responses
        if "stimulus_values" in tuning_data and "responses" in tuning_data:
            stimulus_values = tuning_data["stimulus_values"]
            responses = tuning_data["responses"]

            # Plot mean response with error bars if available
            if responses.ndim > 1:
                mean_response = np.mean(responses, axis=0)
                std_response = np.std(responses, axis=0)
                ax.errorbar(
                    stimulus_values,
                    mean_response,
                    yerr=std_response,
                    marker="o",
                    capsize=5,
                    label="Mean ± STD",
                )
            else:
                ax.plot(stimulus_values, responses, "o-", label="Response")

            ax.set_xlabel(f"{property_name} Value")
            ax.set_ylabel("Response")
            ax.set_title(f"{property_name} Tuning Curve")
            ax.grid(True, alpha=0.3)
            ax.legend()

    # Hide unused subplots
    for idx in range(len(tuning_results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig

def plot_information_analysis(
    info_results: dict[str, Any],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot comprehensive information analysis results.

    Args:
        info_results: Dictionary containing information analysis results
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Basic MI plot
    if "basic_mi" in info_results:
        ax1 = fig.add_subplot(gs[0, :2])
        basic_mi = info_results["basic_mi"]

        keys = [k for k in basic_mi.keys() if not k.endswith("_normalized")]
        values = [basic_mi[k] for k in keys]

        bars = ax1.bar(keys, values)
        ax1.set_ylabel("Mutual Information (bits)")
        ax1.set_title("Basic Mutual Information with Class Labels")
        ax1.tick_params(axis="x", rotation=45)

        # Color code bars
        colors = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"]
        for bar, color in zip(bars, colors[: len(bars)]):
            bar.set_color(color)

    # 2. Pairwise MI heatmap
    if "pairwise_mi" in info_results:
        ax2 = fig.add_subplot(gs[0, 2])
        pairwise_mi = info_results["pairwise_mi"]

        # Create matrix from pairwise values
        variables = ["E", "I", "Vout"]
        if "I(E;Vinf)" in pairwise_mi:
            variables.append("Vinf")

        n_vars = len(variables)
        mi_matrix = np.zeros((n_vars, n_vars))

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    key = f"I({var1};{var2})"
                    if key in pairwise_mi:
                        mi_matrix[i, j] = pairwise_mi[key]

        sns.heatmap(
            mi_matrix,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=variables,
            yticklabels=variables,
            ax=ax2,
            cbar_kws={"label": "MI (bits)"},
        )
        ax2.set_title("Pairwise Mutual Information")

    # 3. Conditional MI plot
    if "conditional_mi" in info_results:
        ax3 = fig.add_subplot(gs[1, :2])
        conditional_mi = info_results["conditional_mi"]

        keys = list(conditional_mi.keys())
        values = list(conditional_mi.values())

        bars = ax3.bar(keys, values)
        ax3.set_ylabel("Conditional MI (bits)")
        ax3.set_title("Conditional Mutual Information given Class")
        ax3.tick_params(axis="x", rotation=45)

        # Color code bars
        for bar, color in zip(bars, colors[: len(bars)]):
            bar.set_color(color)

    # 4. PID results
    if "pid" in info_results:
        ax4 = fig.add_subplot(gs[1, 2])

        # Extract PID components for (E,I)->C
        pid_ei_c = info_results["pid"].get("PID(E,I->C)", {})
        if pid_ei_c:
            components = ["Synergy", "Redundancy", "Unique E", "Unique I"]
            values = [
                pid_ei_c.get("synergy", 0),
                pid_ei_c.get("redundancy", 0),
                pid_ei_c.get("unique_E", 0),
                pid_ei_c.get("unique_I", 0),
            ]

            bars = ax4.bar(components, values)
            ax4.set_ylabel("Information (bits)")
            ax4.set_title("PID: (E, I_var) → C")
            ax4.tick_params(axis="x", rotation=45)

            # Color code bars
            pid_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
            for bar, color in zip(bars, pid_colors):
                bar.set_color(color)

    # 5. Summary statistics
    if "summary_stats" in info_results:
        ax5 = fig.add_subplot(gs[2, :])
        stats_text = info_results.get("summary_text", "")
        ax5.text(
            0.05,
            0.95,
            stats_text,
            transform=ax5.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis("off")
        ax5.set_title("Summary Statistics")

    plt.suptitle("Information Analysis Results", fontsize=16)

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig

def plot_correlation_vs_information(
    correlation_data: dict[str, np.ndarray],
    information_data: dict[str, float],
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot relationship between correlation structure and information content.

    Args:
        correlation_data: Dictionary with correlation matrices by layer
        information_data: Dictionary with information metrics by layer
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Extract correlation statistics
    layers = []
    mean_corrs = []
    info_values = []

    for layer_name in correlation_data.keys():
        if layer_name in information_data:
            corr_matrix = correlation_data[layer_name]
            # Get upper triangle (excluding diagonal)
            upper_tri = np.triu(corr_matrix, k=1)
            mean_corr = np.mean(np.abs(upper_tri[upper_tri != 0]))

            layers.append(layer_name)
            mean_corrs.append(mean_corr)
            info_values.append(information_data[layer_name])

    # Plot 1: Mean correlation by layer
    ax1.bar(layers, mean_corrs)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Absolute Correlation")
    ax1.set_title("Average Correlation Strength by Layer")
    ax1.tick_params(axis="x", rotation=45)

    # Plot 2: Correlation vs Information scatter
    ax2.scatter(mean_corrs, info_values, s=100)
    for i, layer in enumerate(layers):
        ax2.annotate(
            layer,
            (mean_corrs[i], info_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax2.set_xlabel("Mean Absolute Correlation")
    ax2.set_ylabel("Mutual Information (bits)")
    ax2.set_title("Correlation vs Information Content")
    ax2.grid(True, alpha=0.3)

    # Add trend line if enough points
    if len(mean_corrs) > 2:
        z = np.polyfit(mean_corrs, info_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(mean_corrs), max(mean_corrs), 100)
        ax2.plot(
            x_trend,
            p(x_trend),
            "r--",
            alpha=0.8,
            label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
        )
        ax2.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
            exist_ok=True,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig

