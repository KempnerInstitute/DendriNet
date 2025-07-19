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

from dendritic_modeling.plotting.base import (
    BarPlotter,
    BoxPlotter,
    ComparisonPlotter,
    LinePlotter,
    PlotConfig,
)

# ============================================================================
# Helper Functions for Activation Extraction
# ============================================================================


def dendrinet_activations_todict(
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

        plot_dict[f"eilayer{i+1}"]["inh cells"] = dendrinet_activations_todict(
            dendrinet=inh_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], inh_input_list[j]))
        inh_input_list = inh_temp

        plot_dict[f"eilayer{i+1}"]["exc cells"] = dendrinet_activations_todict(
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
# Analysis Plotters
# ============================================================================


class LayerAnalysisPlotter:
    """Plotter for layer-wise contribution analysis."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.line_plotter = LinePlotter(self.config)
        self.bar_plotter = BarPlotter(self.config)

    def plot_layer_contributions(
        self,
        results: dict[str, Any],
        metrics: Optional[list[str]] = None,
        save_path: Optional[str] = None,
        filename: str = "layer_contributions",
        reverse_layers: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot layer contribution analysis with dual metrics.

        Args:
            results: Dictionary containing 'accuracies' and 'losses' by layer
            metrics: list of metrics to plot (default: ['accuracy', 'loss'])
            save_path: Directory to save plots
            filename: Base filename
            reverse_layers: If True, reverse layer order (soma to distal)

        Returns:
            Tuple of (figure, axes)
        """
        if metrics is None:
            metrics = ["accuracy", "loss"]

        # Extract data
        layers = list(results.get("accuracies", results.get("accuracy", {})).keys())
        layer_indices = list(range(len(layers)))

        if reverse_layers:
            layer_indices = layer_indices[::-1]

        # Prepare data for plotting
        plot_data = {}

        if "accuracy" in metrics:
            accuracies = list(
                results.get("accuracies", results.get("accuracy", {})).values()
            )
            if reverse_layers:
                accuracies = accuracies[::-1]
            plot_data["Accuracy"] = (layer_indices, accuracies)

        if "loss" in metrics:
            losses = list(results.get("losses", results.get("loss", {})).values())
            if reverse_layers:
                losses = losses[::-1]
            plot_data["Loss"] = (layer_indices, losses)

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=self.config.figsize)

        # Plot accuracy on primary axis
        if "Accuracy" in plot_data:
            x, y = plot_data["Accuracy"]
            color = self.config.colors.get("primary", "blue")
            ax1.plot(
                x, y, "o-", color=color, linewidth=2, markersize=8, label="Accuracy"
            )
            ax1.set_xlabel(
                "Layer (Soma → Distal)" if reverse_layers else "Layer Index",
                fontsize=self.config.fontsize["label"],
            )
            ax1.set_ylabel(
                "Accuracy", color=color, fontsize=self.config.fontsize["label"]
            )
            ax1.tick_params(axis="y", labelcolor=color)
            ax1.grid(True, alpha=0.3)

        # Plot loss on secondary axis
        if "Loss" in plot_data and len(plot_data) > 1:
            ax2 = ax1.twinx()
            x, y = plot_data["Loss"]
            color = self.config.colors.get("secondary", "red")
            ax2.plot(x, y, "s-", color=color, linewidth=2, markersize=8, label="Loss")
            ax2.set_ylabel("Loss", color=color, fontsize=self.config.fontsize["label"])
            ax2.tick_params(axis="y", labelcolor=color)

        # Customize plot
        ax1.set_title(
            "Layer-wise Contribution Analysis", fontsize=self.config.fontsize["title"]
        )

        # Set x-axis labels
        if reverse_layers:
            ax1.set_xticks(layer_indices)
            ax1.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])

        # Add legend
        if len(plot_data) > 1:
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        else:
            ax1.legend()

        plt.tight_layout()

        # Save if requested
        if save_path:
            self._save_plot(fig, save_path, filename)

        return fig, (ax1, ax2) if "ax2" in locals() else ax1

    def _save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """Save plot with metadata."""
        os.makedirs(save_path, exist_ok=True)
        for fmt in self.config.save_formats:
            fig.savefig(
                os.path.join(save_path, f"{filename}.{fmt}"),
                dpi=self.config.dpi,
                bbox_inches="tight",
            )


class AblationPlotter:
    """Plotter for ablation study results."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.bar_plotter = BarPlotter(self.config)
        self.comparison_plotter = ComparisonPlotter(self.config)

    def plot_ablation_results(
        self,
        results: dict[str, dict[str, float]],
        baseline_key: str = "baseline",
        save_path: Optional[str] = None,
        filename: str = "ablation_results",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot ablation study results with baseline comparison.

        Args:
            results: Dictionary of condition -> metrics
            baseline_key: Key for baseline condition
            save_path: Directory to save plots
            filename: Base filename

        Returns:
            Tuple of (figure, axes)
        """
        # Extract baseline
        baseline = results.get(baseline_key, {})
        conditions = [k for k in results.keys() if k != baseline_key]

        # Prepare data for comparison
        comparison_data = {}
        for condition in conditions:
            comparison_data[condition] = results[condition]

        # Create comparison plot
        fig, axes = self.comparison_plotter.plot(
            data=comparison_data,
            baseline=baseline,
            title="Ablation Study Results",
            metric_names=list(baseline.keys()) if baseline else None,
        )

        # Save if requested
        if save_path:
            self._save_plot(fig, save_path, filename)

        return fig, axes

    def plot_relative_changes(
        self,
        results: dict[str, dict[str, float]],
        baseline_key: str = "baseline",
        metric: str = "accuracy",
        save_path: Optional[str] = None,
        filename: str = "relative_changes",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot relative changes from baseline."""
        baseline_value = results.get(baseline_key, {}).get(metric, 1.0)

        conditions = []
        relative_changes = []

        for condition, metrics in results.items():
            if condition != baseline_key:
                conditions.append(condition)
                value = metrics.get(metric, 0)
                relative_change = ((value - baseline_value) / baseline_value) * 100
                relative_changes.append(relative_change)

        # Create bar plot
        result = self.bar_plotter.plot(
            data=relative_changes,
            labels=conditions,
            title=f"Relative Change in {metric.capitalize()} from Baseline",
            xlabel="Ablation Condition",
            ylabel="Relative Change (%)",
            colors=["red" if x < 0 else "green" for x in relative_changes],
        )

        # Handle return value
        if isinstance(result, plt.Figure):
            fig, ax = result, result.axes[0] if result.axes else None
        else:
            fig, ax = result.figure, result

        # Add zero line
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Save if requested
        if save_path:
            self._save_plot(fig, save_path, filename)

        return fig, ax

    def _save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """Save plot with metadata."""
        os.makedirs(save_path, exist_ok=True)
        for fmt in self.config.save_formats:
            fig.savefig(
                os.path.join(save_path, f"{filename}.{fmt}"),
                dpi=self.config.dpi,
                bbox_inches="tight",
            )


class NoiseAnalysisPlotter:
    """Plotter for noise robustness analysis."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.line_plotter = LinePlotter(self.config)

    def plot_noise_robustness(
        self,
        results: dict[str, list[float]],
        noise_levels: list[float],
        save_path: Optional[str] = None,
        filename: str = "noise_robustness",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot noise robustness curves for different models/conditions."""
        # Prepare data for line plotter
        plot_data = {}
        for label, values in results.items():
            plot_data[label] = (noise_levels, values)

        # Create plot
        result = self.line_plotter.plot(
            data=plot_data,
            title="Noise Robustness Analysis",
            xlabel="Noise Level",
            ylabel="Accuracy",
        )

        # Handle return value
        if isinstance(result, plt.Figure):
            fig, ax = result, result.axes[0] if result.axes else None
        else:
            fig, ax = result.figure, result

        # Customize
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Save if requested
        if save_path:
            self._save_plot(fig, save_path, filename)

        return fig, ax

    def _save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """Save plot with metadata."""
        os.makedirs(save_path, exist_ok=True)
        for fmt in self.config.save_formats:
            fig.savefig(
                os.path.join(save_path, f"{filename}.{fmt}"),
                dpi=self.config.dpi,
                bbox_inches="tight",
            )


class WeightDistributionPlotter:
    """Plotter for weight distribution analysis."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.box_plotter = BoxPlotter(self.config)

    def plot_weight_distributions(
        self,
        weight_data: dict[str, list[np.ndarray]],
        title: str = "Weight Distributions by Layer",
        save_path: Optional[str] = None,
        filename: str = "weight_distributions",
    ):
        """Plot weight distributions across layers."""
        # Flatten weight arrays for each layer
        data_arrays = []
        labels = []
        colors = []

        color_map = {
            "excitatory": self.config.colors.get("excitatory", "#e74c3c"),
            "inhibitory": self.config.colors.get("inhibitory", "#3498db"),
            "combined": self.config.colors.get("combined", "#9b59b6"),
        }

        for layer_name, weights in weight_data.items():
            if isinstance(weights, list):
                flattened = np.concatenate([w.flatten() for w in weights])
            else:
                flattened = weights.flatten()

            data_arrays.append(flattened)
            labels.append(layer_name)

            # Determine color based on layer type
            if "exc" in layer_name.lower():
                colors.append(color_map["excitatory"])
            elif "inh" in layer_name.lower():
                colors.append(color_map["inhibitory"])
            else:
                colors.append(color_map["combined"])

        # Create plot
        result = self.box_plotter.plot(
            data=data_arrays,
            labels=labels,
            title=title,
            ylabel="Weight Value",
            colors=colors,
        )

        # Handle the return value - could be fig or ax
        if isinstance(result, plt.Figure):
            fig, ax = result, result.axes[0] if result.axes else None
        else:
            fig, ax = result.figure, result

        # Save if requested
        if save_path:
            self._save_plot(fig, save_path, filename)

        return fig, ax

    def _save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """Save plot with metadata."""
        os.makedirs(save_path, exist_ok=True)
        for fmt in self.config.save_formats:
            fig.savefig(
                os.path.join(save_path, f"{filename}.{fmt}"),
                dpi=self.config.dpi,
                bbox_inches="tight",
            )


class InformationAnalysisPlotter:
    """Plotter for information-theoretic analysis results."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
        self.line_plotter = LinePlotter(self.config)
        self.bar_plotter = BarPlotter(self.config)

    def _get_fig_ax(self, result):
        """Helper to get (fig, ax) from plotter result."""
        if isinstance(result, plt.Figure):
            return result, result.axes[0] if result.axes else None
        else:
            # It's an axes
            return result.figure, result

    def plot_information_metrics(
        self,
        results: dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """
        Create multiple plots for information analysis results.

        Args:
            results: Dictionary containing information analysis results
            save_path: Directory to save plots
            filename_prefix: Prefix for saved files

        Returns:
            Dictionary mapping plot names to (figure, axes) tuples
        """
        plots = {}

        # Extract number of layers
        num_layers = results.get("num_layers", 5)
        layer_indices = list(range(num_layers))

        # 1. Main information metrics plot
        if any(key in results for key in ["I_E_C", "I_I_C", "I_V_C"]):
            fig, ax = self._plot_main_metrics(results, layer_indices)
            plots["main_metrics"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_main_metrics")

        # 2. Normalized information plot
        if "H_C" in results:
            fig, ax = self._plot_normalized_metrics(results, layer_indices)
            plots["normalized_metrics"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_normalized")

        # 3. Information decomposition plot
        if any(key in results for key in ["I_E_C_given_I", "I_I_C_given_E"]):
            fig, ax = self._plot_decomposition(results, layer_indices)
            plots["decomposition"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_decomposition")

        # 4. Synergy/Redundancy plot (if PID analysis available)
        if "synergy" in results or "redundancy" in results:
            fig, ax = self._plot_synergy_redundancy(results, layer_indices)
            plots["synergy_redundancy"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_synergy_redundancy")

        # 5. Layer comparison bar plot
        if any(key in results for key in ["I_E_C", "I_I_C", "I_V_C"]):
            fig, ax = self._plot_layer_comparison(results)
            plots["layer_comparison"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_layer_comparison")

        return plots

    def _plot_main_metrics(
        self, results: dict, layer_indices: list[int]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot main information metrics (I(E;C), I(I;C), I(V;C))."""
        plot_data = {}

        if "I_E_C" in results:
            plot_data["I(E;C)"] = (layer_indices, results["I_E_C"])
        if "I_I_C" in results:
            plot_data["I(I;C)"] = (layer_indices, results["I_I_C"])
        if "I_V_C" in results:
            plot_data["I(V;C)"] = (layer_indices, results["I_V_C"])

        fig = self.line_plotter.plot(
            data=plot_data,
            title="Information Content Across Layers",
            xlabel="Layer (Soma → Distal)",
            ylabel="Mutual Information (bits)",
        )

        # Get the axes from the figure
        ax = fig.axes[0] if fig.axes else None

        # Customize x-axis
        if ax:
            ax.set_xticks(layer_indices)
            ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
            ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_normalized_metrics(
        self, results: dict, layer_indices: list[int]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot normalized information metrics."""
        plot_data = {}
        h_c = results.get("H_C", 1.0)  # Avoid division by zero

        if "I_E_C" in results and h_c > 0:
            plot_data["I(E;C)/H(C)"] = (
                layer_indices,
                [x / h_c for x in results["I_E_C"]],
            )
        if "I_I_C" in results and h_c > 0:
            plot_data["I(I;C)/H(C)"] = (
                layer_indices,
                [x / h_c for x in results["I_I_C"]],
            )
        if "I_V_C" in results and h_c > 0:
            plot_data["I(V;C)/H(C)"] = (
                layer_indices,
                [x / h_c for x in results["I_V_C"]],
            )

        result = self.line_plotter.plot(
            data=plot_data,
            title="Normalized Information Content",
            xlabel="Layer (Soma → Distal)",
            ylabel="Normalized Mutual Information",
        )

        fig, ax = self._get_fig_ax(result)

        # Customize x-axis
        if ax:
            ax.set_xticks(layer_indices)
            ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
            ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_decomposition(
        self, results: dict, layer_indices: list[int]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot information decomposition (conditional MI and interactions)."""
        plot_data = {}

        if "I_E_C_given_I" in results:
            plot_data["I(E;C|I)"] = (layer_indices, results["I_E_C_given_I"])
        if "I_I_C_given_E" in results:
            plot_data["I(I;C|E)"] = (layer_indices, results["I_I_C_given_E"])
        if "I_E_I" in results:
            plot_data["I(E;I)"] = (layer_indices, results["I_E_I"])

        result = self.line_plotter.plot(
            data=plot_data,
            title="Information Decomposition",
            xlabel="Layer (Soma → Distal)",
            ylabel="Information (bits)",
        )

        fig, ax = self._get_fig_ax(result)

        if ax:
            ax.set_xticks(layer_indices)
            ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
            ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_synergy_redundancy(
        self, results: dict, layer_indices: list[int]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot synergy and redundancy from PID analysis."""
        plot_data = {}

        if "synergy" in results:
            plot_data["Synergy"] = (layer_indices, results["synergy"])
        if "redundancy" in results:
            plot_data["Redundancy"] = (layer_indices, results["redundancy"])
        if "unique_E" in results:
            plot_data["Unique(E)"] = (layer_indices, results["unique_E"])
        if "unique_I" in results:
            plot_data["Unique(I_var)"] = (layer_indices, results["unique_I"])

        result = self.line_plotter.plot(
            data=plot_data,
            title="Partial Information Decomposition",
            xlabel="Layer (Soma → Distal)",
            ylabel="Information (bits)",
        )

        fig, ax = self._get_fig_ax(result)

        if ax:
            ax.set_xticks(layer_indices)
            ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
            ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_layer_comparison(self, results: dict) -> tuple[plt.Figure, plt.Axes]:
        """Create bar plot comparing information metrics across layers."""
        num_layers = results.get("num_layers", 5)

        # Prepare data for grouped bar plot
        metrics = []
        if "I_E_C" in results:
            metrics.append(
                (
                    "I(E;C)",
                    results["I_E_C"],
                    self.config.colors.get("excitatory", "#e74c3c"),
                )
            )
        if "I_I_C" in results:
            metrics.append(
                (
                    "I(I;C)",
                    results["I_I_C"],
                    self.config.colors.get("inhibitory", "#3498db"),
                )
            )
        if "I_V_C" in results:
            metrics.append(
                (
                    "I(V;C)",
                    results["I_V_C"],
                    self.config.colors.get("combined", "#9b59b6"),
                )
            )

        if not metrics:
            return None, None

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        x = np.arange(num_layers)
        width = 0.8 / len(metrics)

        for i, (label, values, color) in enumerate(metrics):
            offset = (i - len(metrics) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)

        ax.set_xlabel("Layer (Soma → Distal)", fontsize=self.config.fontsize["label"])
        ax.set_ylabel(
            "Mutual Information (bits)", fontsize=self.config.fontsize["label"]
        )
        ax.set_title(
            "Information Comparison Across Layers",
            fontsize=self.config.fontsize["title"],
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in range(num_layers)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        return fig, ax

    def _save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """Save plot with metadata."""
        os.makedirs(save_path, exist_ok=True)
        for fmt in self.config.save_formats:
            fig.savefig(
                os.path.join(save_path, f"{filename}.{fmt}"),
                dpi=self.config.dpi,
                bbox_inches="tight",
            )


# ============================================================================
# Specialized Activation Plotters
# ============================================================================


class ActivationPlotter:
    """Specialized plotter for neural activation patterns."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def plot_dendrinet_activations(
        self,
        dendrinet,
        exc_input_list,
        inh_input_list,
        save_path: Optional[str] = None,
        filename: str = "dendrinet_activations",
    ):
        """Plot DendriNet activations with hierarchical organization."""
        plot_dict = dendrinet_activations_todict(
            dendrinet, exc_input_list, inh_input_list
        )

        # Create visualizations for each soma
        for soma_key, soma_data in plot_dict.items():
            fig, axes = plt.subplots(
                len(soma_data), 1, figsize=(12, 4 * len(soma_data))
            )
            if len(soma_data) == 1:
                axes = [axes]

            fig.suptitle(
                f"DendriNet Activations - {soma_key}",
                fontsize=self.config.fontsize["title"],
            )

            for i, (level_key, level_data) in enumerate(soma_data.items()):
                # Collect all branch activations for this level
                all_activations = []
                for _branch_key, branch_data in level_data.items():
                    if "activation list" in branch_data:
                        for act in branch_data["activation list"]:
                            all_activations.append(act.detach().cpu().numpy())

                if all_activations:
                    # Create heatmap
                    act_array = np.array(all_activations).T
                    im = axes[i].imshow(act_array, aspect="auto", cmap="viridis")
                    axes[i].set_title(
                        f"{level_key}", fontsize=self.config.fontsize["label"]
                    )
                    axes[i].set_xlabel("Input Sample")
                    axes[i].set_ylabel("Branch Activation")
                    plt.colorbar(im, ax=axes[i])

            plt.tight_layout()

            if save_path:
                self._save_plot(fig, save_path, f"{filename}_{soma_key}")

            plt.close()

    def plot_einet_activations(
        self,
        einet,
        input_list,
        save_path: Optional[str] = None,
        filename: str = "einet_activations",
    ):
        """Plot EINet activations showing excitatory and inhibitory components."""
        plot_dict = einet_activations_todict(einet, input_list)

        # Create visualizations for each layer
        for layer_key, layer_data in plot_dict.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(
                f"EINet Activations - {layer_key}",
                fontsize=self.config.fontsize["title"],
            )

            # Plot inhibitory cells
            if "inh cells" in layer_data:
                inh_acts = []
                for soma_data in layer_data["inh cells"].values():
                    for level_data in soma_data.values():
                        for branch_data in level_data.values():
                            if "activation list" in branch_data:
                                for act in branch_data["activation list"]:
                                    inh_acts.append(act.detach().cpu().numpy())

                if inh_acts:
                    inh_array = np.array(inh_acts).T
                    im1 = ax1.imshow(
                        inh_array,
                        aspect="auto",
                        cmap="Blues" if "Blues" in plt.colormaps() else "viridis",
                    )
                    ax1.set_title(
                        "Inhibitory Cells", fontsize=self.config.fontsize["label"]
                    )
                    ax1.set_xlabel("Activation Sample")
                    ax1.set_ylabel("Neuron Index")
                    plt.colorbar(im1, ax=ax1)

            # Plot excitatory cells
            if "exc cells" in layer_data:
                exc_acts = []
                for soma_data in layer_data["exc cells"].values():
                    for level_data in soma_data.values():
                        for branch_data in level_data.values():
                            if "activation list" in branch_data:
                                for act in branch_data["activation list"]:
                                    exc_acts.append(act.detach().cpu().numpy())

                if exc_acts:
                    exc_array = np.array(exc_acts).T
                    im2 = ax2.imshow(
                        exc_array,
                        aspect="auto",
                        cmap="Reds" if "Reds" in plt.colormaps() else "viridis",
                    )
                    ax2.set_title(
                        "Excitatory Cells", fontsize=self.config.fontsize["label"]
                    )
                    ax2.set_xlabel("Activation Sample")
                    ax2.set_ylabel("Neuron Index")
                    plt.colorbar(im2, ax=ax2)

            plt.tight_layout()

            if save_path:
                self._save_plot(fig, save_path, f"{filename}_{layer_key}")

            plt.close()

    def _save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """Save plot with metadata."""
        os.makedirs(save_path, exist_ok=True)
        for fmt in self.config.save_formats:
            fig.savefig(
                os.path.join(save_path, f"{filename}.{fmt}"),
                dpi=self.config.dpi,
                bbox_inches="tight",
            )


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
