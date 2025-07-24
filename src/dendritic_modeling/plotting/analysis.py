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
            plot_data["Accuracy"] = accuracies

        if "loss" in metrics:
            losses = list(results.get("losses", results.get("loss", {})).values())
            if reverse_layers:
                losses = losses[::-1]
            plot_data["Loss"] = losses

        # Create figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=self.config.figsize)

        # Plot accuracy on primary axis
        if "Accuracy" in plot_data:
            y = plot_data["Accuracy"]
            color = self.config.colors.get("primary", "blue")
            ax1.plot(
                layer_indices, y, "o-", color=color, linewidth=2, markersize=8, label="Accuracy"
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
            y = plot_data["Loss"]
            color = self.config.colors.get("secondary", "red")
            ax2.plot(layer_indices, y, "s-", color=color, linewidth=2, markersize=8, label="Loss")
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

    def plot_layerwise_ablation(
        self,
        results: dict[str, list[float]],
        save_path: Optional[str] = None,
        filename: str = "layerwise_ablation",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot layer-wise ablation analysis results showing performance drops.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)

        layer_idx = results.get("layer_idx", list(range(len(results["baseline_acc"]))))
        ax.plot(layer_idx, results["baseline_acc"], "o-", label="Baseline")
        if "inh_ablated_acc" in results:
            ax.plot(layer_idx, results["inh_ablated_acc"], "s-", label="Inhibition Ablated")
        if "exc_ablated_acc" in results:
            ax.plot(layer_idx, results["exc_ablated_acc"], "^-", label="Excitation Ablated")

        ax.set_xlabel("Layer Index (higher = closer to input)", fontsize=self.config.fontsize["label"])
        ax.set_ylabel("Performance Metric", fontsize=self.config.fontsize["label"])
        ax.set_title("Layer-wise Ablation Analysis", fontsize=self.config.fontsize["title"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path, filename)

        return fig, ax

    def plot_layerwise_contributions(
        self,
        results: dict[str, list[float]],
        save_path: Optional[str] = None,
        filename: str = "layerwise_contributions",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot layer-wise contribution analysis showing performance drops when ablated.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)

        layer_idx = np.array(results.get("layer_idx", list(range(len(results["baseline_acc"])))))
        width = 0.35

        if "inh_ablated_acc" in results:
            inh_diff = [b - i for b, i in zip(results["baseline_acc"], results["inh_ablated_acc"])]
            ax.bar(layer_idx - width/2, inh_diff, width, label="Inhibition Contribution", alpha=0.7)

        if "exc_ablated_acc" in results:
            exc_diff = [b - e for b, e in zip(results["baseline_acc"], results["exc_ablated_acc"])]
            ax.bar(layer_idx + width/2, exc_diff, width, label="Excitation Contribution", alpha=0.7)

        ax.set_xlabel("Layer Index (higher = closer to input)", fontsize=self.config.fontsize["label"])
        ax.set_ylabel("Performance Drop when Ablated", fontsize=self.config.fontsize["label"])
        ax.set_title("Contribution per Layer", fontsize=self.config.fontsize["title"])
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_plot(fig, save_path, filename)

        return fig, ax

    def plot_ablation_comparison(
        self,
        normal_results: dict[str, Any],
        ablated_results: dict[str, dict[str, Any]],
        metric: str = "accuracy",
        save_path: Optional[str] = None,
        filename: str = "ablation_comparison",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Compare performance under different ablation conditions.
        """
        fig, ax = plt.subplots(figsize=self.config.figsize)

        layers = sorted(normal_results.keys())
        normal_values = [normal_results[layer][metric] for layer in layers]
        ax.plot(layers, normal_values, "o-", linewidth=2, markersize=8, label="Normal")

        colors = list(self.config.colors.values())
        for idx, (ablation_type, layer_data) in enumerate(ablated_results.items()):
            values = [layer_data.get(layer, {}).get(metric, 0) for layer in layers]
            ax.plot(layers, values, "s--", linewidth=2, markersize=6, label=f"{ablation_type} Ablated", color=colors[idx % len(colors)])

        ax.set_xlabel("Layer Index", fontsize=self.config.fontsize["label"])
        ax.set_ylabel(f"{metric.capitalize()}", fontsize=self.config.fontsize["label"])
        ax.set_title("Performance Under Ablation Conditions", fontsize=self.config.fontsize["title"])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

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
            plot_data[label] = values

        # Create plot
        result = self.line_plotter.plot(
            data=plot_data,
            x_values=noise_levels,
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

    def plot_information_metrics(
        self,
        results: dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """
        Create multiple plots for information analysis results.
        
        Handles both new format (nested dictionaries) and old format (flat dictionary).

        Args:
            results: Dictionary containing information analysis results
            save_path: Directory to save plots
            filename_prefix: Prefix for saved files

        Returns:
            Dictionary mapping plot names to (figure, axes) tuples
        """
        plots = {}

        # Check if this is the new format (nested) or old format (flat)
        is_new_format = "basic_mi" in results or "pairwise_mi" in results
        
        if is_new_format:
            # Handle new format with nested dictionaries
            return self._plot_new_format(results, save_path, filename_prefix)
        else:
            # Handle old format (July 1st version) with flat structure
            return self._plot_old_format(results, save_path, filename_prefix)

    def _plot_new_format(
        self,
        results: dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """Plot information metrics in the new nested format."""
        plots = {}

        # 1. Basic MI plot
        if "basic_mi" in results:
            fig, ax = self._plot_basic_mi_new(results["basic_mi"])
            plots["basic_mi"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_basic_mi")

        # 2. Pairwise MI plot
        if "pairwise_mi" in results:
            fig, ax = self._plot_pairwise_mi_new(results["pairwise_mi"])
            plots["pairwise_mi"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_pairwise_mi")

        # 3. Conditional MI plot
        if "conditional_mi" in results:
            fig, ax = self._plot_conditional_mi_new(results["conditional_mi"])
            plots["conditional_mi"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_conditional_mi")

        # 4. PID plot
        if "pid" in results:
            fig, ax = self._plot_pid_new(results["pid"])
            plots["pid"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_pid")

        # 5. Summary comparison plot
        if "basic_mi" in results:
            fig, ax = self._plot_summary_comparison_new(results)
            plots["summary"] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_summary")

        return plots

    def _plot_old_format(
        self,
        results: dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """Plot information metrics in the old flat format (July 1st version)."""
        plots = {}

        # Extract number of layers
        num_layers = results.get('num_layers', 5)
        layer_indices = list(range(num_layers))

        # 1. Main information metrics plot
        if any(key in results for key in ['I_E_C', 'I_I_C', 'I_V_C']):
            fig, ax = self._plot_main_metrics(results, layer_indices)
            plots['main_metrics'] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_main_metrics")

        # 2. Normalized information plot
        if 'H_C' in results:
            fig, ax = self._plot_normalized_metrics(results, layer_indices)
            plots['normalized_metrics'] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_normalized")

        # 3. Information decomposition plot
        if any(key in results for key in ['I_E_C_given_I', 'I_I_C_given_E']):
            fig, ax = self._plot_decomposition(results, layer_indices)
            plots['decomposition'] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_decomposition")

        # 4. Synergy/Redundancy plot (if PID analysis available)
        if 'synergy' in results or 'redundancy' in results:
            fig, ax = self._plot_synergy_redundancy(results, layer_indices)
            plots['synergy_redundancy'] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_synergy_redundancy")

        # 5. Layer comparison bar plot
        if any(key in results for key in ['I_E_C', 'I_I_C', 'I_V_C']):
            fig, ax = self._plot_layer_comparison(results)
            plots['layer_comparison'] = (fig, ax)
            if save_path:
                self._save_plot(fig, save_path, f"{filename_prefix}_layer_comparison")

        return plots

    def _plot_basic_mi_new(self, basic_mi: dict) -> tuple[plt.Figure, plt.Axes]:
        """Plot basic MI metrics from new format."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Extract metrics
        metrics = {k: v for k, v in basic_mi.items() if not k.endswith('_normalized') and not k.endswith('_std')}
        
        # Create bar plot
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(labels, values, alpha=0.8)
        
        # Color code by type
        colors = []
        for label in labels:
            if 'E' in label and 'I' not in label:
                colors.append(self.config.colors.get('excitatory', '#e74c3c'))
            elif 'I' in label and 'E' not in label:
                colors.append(self.config.colors.get('inhibitory', '#3498db'))
            else:
                colors.append(self.config.colors.get('combined', '#9b59b6'))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel('Mutual Information (bits)', fontsize=self.config.fontsize["label"])
        ax.set_title('Basic Mutual Information with Class Labels', fontsize=self.config.fontsize["title"])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig, ax

    def _plot_pairwise_mi_new(self, pairwise_mi: dict) -> tuple[plt.Figure, plt.Axes]:
        """Plot pairwise MI metrics from new format."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Extract metrics
        metrics = {k: v for k, v in pairwise_mi.items() if not k.endswith('_std')}
        
        # Create bar plot
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        ax.bar(labels, values, color=self.config.colors.get('primary', '#2c3e50'), alpha=0.8)

        ax.set_ylabel('Mutual Information (bits)', fontsize=self.config.fontsize["label"])
        ax.set_title('Pairwise Mutual Information', fontsize=self.config.fontsize["title"])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig, ax

    def _plot_conditional_mi_new(self, conditional_mi: dict) -> tuple[plt.Figure, plt.Axes]:
        """Plot conditional MI metrics from new format."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Extract metrics
        metrics = {k: v for k, v in conditional_mi.items() if not k.endswith('_std')}
        
        # Create bar plot
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        ax.bar(labels, values, color=self.config.colors.get('secondary', '#27ae60'), alpha=0.8)

        ax.set_ylabel('Conditional MI (bits)', fontsize=self.config.fontsize["label"])
        ax.set_title('Conditional Mutual Information given Class', fontsize=self.config.fontsize["title"])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig, ax

    def _plot_pid_new(self, pid_results: dict) -> tuple[plt.Figure, plt.Axes]:
        """Plot PID results from new format."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Process PID results
        pid_names = []
        synergy_values = []
        redundancy_values = []
        unique1_values = []
        unique2_values = []

        for pid_key, pid_data in pid_results.items():
            pid_names.append(pid_key)
            synergy_values.append(pid_data.get('synergy', 0))
            redundancy_values.append(pid_data.get('redundancy', 0))
            
            # Find unique values
            unique_keys = [k for k in pid_data.keys() if k.startswith('unique_')]
            if len(unique_keys) >= 2:
                unique1_values.append(pid_data[unique_keys[0]])
                unique2_values.append(pid_data[unique_keys[1]])
            else:
                unique1_values.append(0)
                unique2_values.append(0)

        # Create grouped bar plot
        x = np.arange(len(pid_names))
        width = 0.2

        ax.bar(x - 1.5*width, synergy_values, width, label='Synergy', color='#e74c3c')
        ax.bar(x - 0.5*width, redundancy_values, width, label='Redundancy', color='#3498db')
        ax.bar(x + 0.5*width, unique1_values, width, label='Unique 1', color='#2ecc71')
        ax.bar(x + 1.5*width, unique2_values, width, label='Unique 2', color='#f39c12')

        ax.set_xlabel('PID Decomposition', fontsize=self.config.fontsize["label"])
        ax.set_ylabel('Information (bits)', fontsize=self.config.fontsize["label"])
        ax.set_title('Partial Information Decomposition', fontsize=self.config.fontsize["title"])
        ax.set_xticks(x)
        ax.set_xticklabels(pid_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig, ax

    def _plot_summary_comparison_new(self, results: dict) -> tuple[plt.Figure, plt.Axes]:
        """Create summary comparison plot for new format."""
        fig, ax = plt.subplots(figsize=self.config.figsize)

        if "basic_mi" not in results:
            return fig, ax

        # Extract key metrics for comparison
        basic_mi = results["basic_mi"]
        
        # Get normalized values if available
        if any(k.endswith('_normalized') for k in basic_mi.keys()):
            metrics = {
                'I(E;C)/H(C)': basic_mi.get('I(E;C)_normalized', 0),
                'I(I;C)/H(C)': basic_mi.get('I(I;C)_normalized', 0),
                'I(Vout;C)/H(C)': basic_mi.get('I(Vout;C)_normalized', 0),
            }
            ylabel = 'Normalized Mutual Information'
        else:
            metrics = {
                'I(E;C)': basic_mi.get('I(E;C)', 0),
                'I(I;C)': basic_mi.get('I(I;C)', 0),
                'I(Vout;C)': basic_mi.get('I(Vout;C)', 0),
            }
            ylabel = 'Mutual Information (bits)'

        # Add joint MI if available
        if 'I(E,I;C)' in basic_mi:
            if any(k.endswith('_normalized') for k in basic_mi.keys()):
                metrics['I(E,I;C)/H(C)'] = basic_mi.get('I(E,I;C)_normalized', 0)
            else:
                metrics['I(E,I;C)'] = basic_mi.get('I(E,I;C)', 0)

        # Create bar plot
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(labels, values, alpha=0.8)
        
        # Color code
        colors = []
        for label in labels:
            if 'E' in label and 'I' not in label:
                colors.append(self.config.colors.get('excitatory', '#e74c3c'))
            elif 'I' in label and 'E' not in label:
                colors.append(self.config.colors.get('inhibitory', '#3498db'))
            elif 'Vout' in label:
                colors.append(self.config.colors.get('output', '#27ae60'))
            else:
                colors.append(self.config.colors.get('combined', '#9b59b6'))
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel(ylabel, fontsize=self.config.fontsize["label"])
        ax.set_title('Information Content Summary', fontsize=self.config.fontsize["title"])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return fig, ax

    # Keep the old format plotting methods for compatibility
    def _plot_main_metrics(self, results: dict, layer_indices: list[int]) -> tuple[plt.Figure, plt.Axes]:
        """Plot main information metrics (I(E;C), I(I;C), I(V;C))."""
        plot_data = {}

        if 'I_E_C' in results:
            plot_data['I(E;C)'] = (layer_indices, results['I_E_C'])
        if 'I_I_C' in results:
            plot_data['I(I;C)'] = (layer_indices, results['I_I_C'])
        if 'I_V_C' in results:
            plot_data['I(V;C)'] = (layer_indices, results['I_V_C'])

        fig, ax = self.line_plotter.plot(
            data=plot_data,
            title="Information Content Across Layers",
            xlabel="Layer (Soma → Distal)",
            ylabel="Mutual Information (bits)",
            legend_title="Information Metric",
            colors={
                'I(E;C)': self.config.colors.get('excitatory', '#e74c3c'),
                'I(I;C)': self.config.colors.get('inhibitory', '#3498db'),
                'I(V;C)': self.config.colors.get('combined', '#9b59b6')
            }
        )

        # Customize x-axis
        ax.set_xticks(layer_indices)
        ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
        ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_normalized_metrics(self, results: dict, layer_indices: list[int]) -> tuple[plt.Figure, plt.Axes]:
        """Plot normalized information metrics."""
        H_C = results.get('H_C', 1.0)
        plot_data = {}

        if 'I_E_C' in results:
            plot_data['I(E;C)/H(C)'] = (layer_indices, [i/H_C for i in results['I_E_C']])
        if 'I_I_C' in results:
            plot_data['I(I;C)/H(C)'] = (layer_indices, [i/H_C for i in results['I_I_C']])
        if 'I_V_C' in results:
            plot_data['I(V;C)/H(C)'] = (layer_indices, [i/H_C for i in results['I_V_C']])

        fig, ax = self.line_plotter.plot(
            data=plot_data,
            title="Normalized Information Content",
            xlabel="Layer (Soma → Distal)",
            ylabel="Normalized Mutual Information",
            legend_title="Normalized Metric",
            colors={
                'I(E;C)/H(C)': self.config.colors.get('excitatory', '#e74c3c'),
                'I(I;C)/H(C)': self.config.colors.get('inhibitory', '#3498db'),
                'I(V;C)/H(C)': self.config.colors.get('combined', '#9b59b6')
            }
        )

        ax.set_ylim(0, 1.05)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
        ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_decomposition(self, results: dict, layer_indices: list[int]) -> tuple[plt.Figure, plt.Axes]:
        """Plot information decomposition (conditional MI and interactions)."""
        plot_data = {}

        if 'I_E_C_given_I' in results:
            plot_data['I(E;C|I)'] = (layer_indices, results['I_E_C_given_I'])
        if 'I_I_C_given_E' in results:
            plot_data['I(I;C|E)'] = (layer_indices, results['I_I_C_given_E'])
        if 'I_E_I' in results:
            plot_data['I(E;I)'] = (layer_indices, results['I_E_I'])

        fig, ax = self.line_plotter.plot(
            data=plot_data,
            title="Information Decomposition",
            xlabel="Layer (Soma → Distal)",
            ylabel="Information (bits)",
            legend_title="Information Component"
        )

        ax.set_xticks(layer_indices)
        ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
        ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_synergy_redundancy(self, results: dict, layer_indices: list[int]) -> tuple[plt.Figure, plt.Axes]:
        """Plot synergy and redundancy from PID analysis."""
        plot_data = {}

        if 'synergy' in results:
            plot_data['Synergy'] = (layer_indices, results['synergy'])
        if 'redundancy' in results:
            plot_data['Redundancy'] = (layer_indices, results['redundancy'])
        if 'unique_E' in results:
            plot_data['Unique(E)'] = (layer_indices, results['unique_E'])
        if 'unique_I' in results:
            plot_data['Unique(I)'] = (layer_indices, results['unique_I'])

        fig, ax = self.line_plotter.plot(
            data=plot_data,
            title="Partial Information Decomposition",
            xlabel="Layer (Soma → Distal)",
            ylabel="Information (bits)",
            legend_title="PID Component"
        )

        ax.set_xticks(layer_indices)
        ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in layer_indices])
        ax.grid(True, alpha=0.3)

        return fig, ax

    def _plot_layer_comparison(self, results: dict) -> tuple[plt.Figure, plt.Axes]:
        """Create bar plot comparing information metrics across layers."""
        num_layers = results.get('num_layers', 5)

        # Prepare data for grouped bar plot
        metrics = []
        if 'I_E_C' in results:
            metrics.append(('I(E;C)', results['I_E_C'], self.config.colors.get('excitatory', '#e74c3c')))
        if 'I_I_C' in results:
            metrics.append(('I(I;C)', results['I_I_C'], self.config.colors.get('inhibitory', '#3498db')))
        if 'I_V_C' in results:
            metrics.append(('I(V;C)', results['I_V_C'], self.config.colors.get('combined', '#9b59b6')))

        if not metrics:
            return None, None

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        x = np.arange(num_layers)
        width = 0.8 / len(metrics)

        for i, (label, values, color) in enumerate(metrics):
            offset = (i - len(metrics)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)

        ax.set_xlabel('Layer (Soma → Distal)', fontsize=self.config.fontsize["label"])
        ax.set_ylabel('Mutual Information (bits)', fontsize=self.config.fontsize["label"])
        ax.set_title('Information Comparison Across Layers', fontsize=self.config.fontsize["title"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" if i > 0 else "Soma" for i in range(num_layers)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        return fig, ax

    def _save_plot(self, fig: plt.Figure, save_path: str, filename: str):
        """Save plot with metadata."""
        os.makedirs(save_path, exist_ok=True)
        for fmt in self.config.save_formats:
            fig.savefig(
                os.path.join(save_path, f"{filename}.{fmt}"),
                dpi=self.config.dpi,
                bbox_inches='tight'
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


def plot_synapse_turnover(turnover_results: dict, save_path: str = None, filename: str = "synapse_turnover"):
    """
    Generate plots for synaptic turnover analysis results.
    
    Args:
        turnover_results: Dictionary containing turnover analysis results with structure:
                         {"exc": {"active": [], "stable": [], "transient": []},
                          "inh": {"active": [], "stable": [], "transient": []},
                          "both": {"active": [], "stable": [], "transient": []}}
        save_path: Directory to save the plot
        filename: Filename for the saved plot (without extension)
    
    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import os
    from dendritic_modeling import logger
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Synaptic Turnover Analysis", fontsize=16)
    
    # Expected data structure from SynapticTurnoverAnalyzer
    expected_keys = ["exc", "inh", "both"]
    metric_keys = ["active", "stable", "transient"]
    
    if all(key in turnover_results for key in expected_keys):
        # Plot excitatory turnover
        ax = axes[0, 0]
        exc_data = turnover_results["exc"]
        epochs = range(len(exc_data["active"]))
        
        ax.plot(epochs, exc_data["active"], label="Active", marker='o', linestyle='-')
        ax.plot(epochs, exc_data["stable"], label="Stable", marker='s', linestyle='--')
        ax.plot(epochs, exc_data["transient"], label="Transient", marker='^', linestyle=':')
        ax.set_title("Excitatory Synapses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of Synapses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot inhibitory turnover
        ax = axes[0, 1]
        inh_data = turnover_results["inh"]
        
        ax.plot(epochs, inh_data["active"], label="Active", marker='o', linestyle='-')
        ax.plot(epochs, inh_data["stable"], label="Stable", marker='s', linestyle='--')
        ax.plot(epochs, inh_data["transient"], label="Transient", marker='^', linestyle=':')
        ax.set_title("Inhibitory Synapses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of Synapses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot combined turnover
        ax = axes[1, 0]
        both_data = turnover_results["both"]
        
        ax.plot(epochs, both_data["active"], label="Active", marker='o', linestyle='-')
        ax.plot(epochs, both_data["stable"], label="Stable", marker='s', linestyle='--')
        ax.plot(epochs, both_data["transient"], label="Transient", marker='^', linestyle=':')
        ax.set_title("Combined (Excitatory + Inhibitory)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of Synapses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot turnover rates (transient/active ratio)
        ax = axes[1, 1]
        exc_rate = [t/a if a > 0 else 0 for t, a in zip(exc_data["transient"], exc_data["active"])]
        inh_rate = [t/a if a > 0 else 0 for t, a in zip(inh_data["transient"], inh_data["active"])]
        both_rate = [t/a if a > 0 else 0 for t, a in zip(both_data["transient"], both_data["active"])]
        
        ax.plot(epochs, exc_rate, label="Excitatory Rate", marker='o', linestyle='-')
        ax.plot(epochs, inh_rate, label="Inhibitory Rate", marker='s', linestyle='--')
        ax.plot(epochs, both_rate, label="Combined Rate", marker='^', linestyle=':')
        ax.set_title("Turnover Rates (Transient/Active)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Turnover Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        # Fallback for unknown data structure
        ax = axes[0, 0]
        ax.text(0.5, 0.5, f"Unexpected data structure\nAvailable keys: {list(turnover_results.keys())}", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Synapse Turnover Analysis")
        
        # Hide other subplots
        for i in range(2):
            for j in range(2):
                if i != 0 or j != 0:
                    axes[i, j].set_visible(False)
        
        logger.warning(f"Unknown data structure for synapse turnover. Available keys: {list(turnover_results.keys())}")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{filename}.png")
        fig.savefig(full_path, dpi=300, bbox_inches='tight')
        logger.info(f"Synapse turnover plot saved to {full_path}")
    
    plt.close(fig)
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
