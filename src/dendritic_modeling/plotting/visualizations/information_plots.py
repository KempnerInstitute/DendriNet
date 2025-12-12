"""
Clean modular information analysis plotting for dendritic modeling.

This module provides specialized plotting classes for information-theoretic analysis.
"""

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# Import utilities
from .plotting_utils import (
    add_error_bars,
    annotate_bars,
    convert_layer_names,
    create_figure,
    create_heatmap,
    create_subplots,
    get_color_scheme,
    handle_plot_error,
    save_plot,
    setup_basic_plot,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


class PlotConfig:
    """Configuration for plot appearance."""

    def __init__(self):
        self.figsize = (10, 6)
        self.dpi = 300
        self.fontsize = {"title": 16, "label": 14, "tick": 12, "legend": 12}
        self.colors = get_color_scheme()


# ============================================================================
# Base Plotter Class
# ============================================================================


class BasePlotter:
    """Base class for all plotters with shared functionality."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def create_subplot(
        self, figsize: Optional[tuple[int, int]] = None
    ) -> tuple[plt.Figure, plt.Axes]:
        return create_figure(figsize or self.config.figsize)

    def create_subplots(
        self, rows: int, cols: int, figsize: Optional[tuple[int, int]] = None
    ) -> tuple[plt.Figure, np.ndarray]:
        return create_subplots(rows, cols, figsize or (15, 10))

    def save(self, fig: plt.Figure, save_path: str, filename: str):
        save_plot(fig, save_path, filename, dpi=self.config.dpi)

    def get_color(self, color_type: str) -> str:
        return self.config.colors.get(color_type, self.config.colors["primary"])


# ============================================================================
# Specialized Plotter Classes
# ============================================================================


class BasicMIPlotter(BasePlotter):
    """Handles basic MI plotting."""

    def plot(self, basic_mi: dict, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            _, ax = self.create_subplot()

        metrics = {
            k: v
            for k, v in basic_mi.items()
            if not k.endswith("_normalized") and not k.endswith("_std")
        }

        if metrics:
            labels = list(metrics.keys())
            values = list(metrics.values())
            colors = [
                (
                    self.get_color("excitatory")
                    if "E" in label and "I" not in label
                    else (
                        self.get_color("inhibitory")
                        if "I" in label and "E" not in label
                        else self.get_color("combined")
                    )
                )
                for label in labels
            ]

            bars = ax.bar(labels, values, color=colors, alpha=0.8)
            setup_basic_plot(
                ax,
                "Basic Mutual Information with Class Labels",
                "MI Type",
                "MI (bits)",
                self.config.fontsize,
            )
            ax.tick_params(axis="x", rotation=45)
            annotate_bars(ax, bars, values)
        else:
            ax.text(
                0.5,
                0.5,
                "No basic MI metrics found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Basic Mutual Information")

        return ax


class PairwiseMIPlotter(BasePlotter):
    """Handles pairwise MI plotting."""

    def plot_heatmap(self, pairwise_mi: dict, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            _, ax = self.create_subplot()

        # Start with base variables and add optional ones if present
        variables = ["E", "I"]
        if any("Vb" in key for key in pairwise_mi.keys()):
            variables.append("Vb")
        variables.append("Vout")
        if any("Vinf" in key for key in pairwise_mi.keys()):
            variables.append("Vinf")

        # Add LDA versions if present
        if any("E_lin" in key for key in pairwise_mi.keys()):
            variables.append("E_lin")
        if any("I_lin" in key for key in pairwise_mi.keys()):
            variables.append("I_lin")
        if any("Vb_lin" in key for key in pairwise_mi.keys()):
            variables.append("Vb_lin")

        n_vars = len(variables)
        mi_matrix = np.zeros((n_vars, n_vars))

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    for key in [f"I({var1};{var2})", f"I({var2};{var1})"]:
                        if key in pairwise_mi:
                            mi_matrix[i, j] = pairwise_mi[key]
                            break

        create_heatmap(
            ax,
            mi_matrix,
            variables,
            variables,
            "Pairwise Mutual Information Matrix",
            cmap="YlOrRd",
        )
        return ax


class PerLayerPlotter(BasePlotter):
    """Handles per-layer information plotting."""

    def plot_basic_mi_layers(
        self, layer_results: dict, layer_names: list, ax: plt.Axes
    ):
        # Base metrics - check what's available in the data
        possible_metrics = [
            "I(E;C)_mean",
            "I(I;C)_mean",
            "I(Vb;C)_mean",
            "I(Vout;C)_mean",
            "I(E_lin;C)_mean",
            "I(I_lin;C)_mean",
            "I(Vb_lin;C)_mean",
        ]
        possible_labels = [
            "I(E;C)",
            "I(I;C)",
            "I(Vb;C)",
            "I(Vout;C)",
            "I(E_lin;C)",
            "I(I_lin;C)",
            "I(Vb_lin;C)",
        ]

        # Filter to only metrics present in data
        metrics_to_plot = []
        metric_labels = []
        for metric, label in zip(possible_metrics, possible_labels):
            # Check if any layer has this metric
            if any(metric in layer_results.get(ln, {}) for ln in layer_names):
                metrics_to_plot.append(metric)
                metric_labels.append(label)

        # Fallback to original if none found
        if not metrics_to_plot:
            metrics_to_plot = ["I(E;C)_mean", "I(I;C)_mean", "I(Vout;C)_mean"]
            metric_labels = ["I(E;C)", "I(I;C)", "I(Vout;C)"]

        # Define colors matching the number of metrics
        color_map = {
            "I(E;C)": self.get_color("excitatory"),
            "I(I;C)": self.get_color("inhibitory"),
            "I(Vb;C)": "#FFB347",  # Orange for branch input
            "I(Vout;C)": self.get_color("combined"),
            "I(E_lin;C)": "#90EE90",  # Light green for E LDA
            "I(I_lin;C)": "#FFB6C1",  # Light pink for I LDA
            "I(Vb_lin;C)": "#FFD700",  # Gold for Vb LDA
        }
        colors = [color_map.get(label, "#808080") for label in metric_labels]

        x = np.arange(len(layer_names))
        width = 0.8 / len(metrics_to_plot)  # Dynamic width based on number of metrics

        for i, (metric, label, color) in enumerate(
            zip(metrics_to_plot, metric_labels, colors)
        ):
            values = []
            errors = []
            for layer_name in layer_names:
                values.append(layer_results[layer_name].get(metric, 0))
                std_key = metric.replace("_mean", "_std")
                errors.append(layer_results[layer_name].get(std_key, 0))

            bars = ax.bar(
                x + i * width,
                values,
                width,
                yerr=errors,
                label=label,
                color=color,
                alpha=0.8,
                capsize=4,
                error_kw={"linewidth": 1.5, "capthick": 1.5},
            )

            for bar, val, err in zip(bars, values, errors):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + err + 0.01,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        setup_basic_plot(
            ax,
            "Basic MI Across Layers (Mean ± Std)",
            "Layer (Soma → Distal)",
            "MI (bits)",
            self.config.fontsize,
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(layer_names, rotation=45)
        ax.legend(loc="upper right")

    def plot_info_flow_layers(
        self, layer_results: dict, layer_names: list, ax: plt.Axes
    ):
        layer_indices = np.arange(len(layer_names))
        vout_values = np.array(
            [layer_results[name].get("I(Vout;C)_mean", 0) for name in layer_names]
        )
        vout_errors = np.array(
            [layer_results[name].get("I(Vout;C)_std", 0) for name in layer_names]
        )

        add_error_bars(
            ax,
            layer_indices,
            vout_values,
            vout_errors,
            self.get_color("combined"),
            "I(Vout;C)",
            capsize=6,
        )

        ax.fill_between(
            layer_indices,
            vout_values - vout_errors,
            vout_values + vout_errors,
            alpha=0.2,
            color=self.get_color("combined"),
        )

        for i, (val, err) in enumerate(zip(vout_values, vout_errors)):
            if val > 0:
                ax.annotate(
                    f"{val:.3f}±{err:.3f}",
                    (layer_indices[i], val + err + 0.02),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                    },
                )

        setup_basic_plot(
            ax,
            "Information Flow Across Layers",
            "Layer Position (Soma → Distal)",
            "I(Vout;C) (bits)",
            self.config.fontsize,
        )
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45)
        ax.legend()


# ============================================================================
# Main Information Analysis Plotter
# ============================================================================


class InformationAnalysisPlotter(BasePlotter):
    """Main plotter orchestrating information analysis plots."""

    def __init__(self, config: PlotConfig = None):
        super().__init__(config)
        self.basic_plotter = BasicMIPlotter(self.config)
        self.pairwise_plotter = PairwiseMIPlotter(self.config)
        self.layer_plotter = PerLayerPlotter(self.config)

    def plot_information_metrics(
        self,
        results: dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """Create comprehensive plots using specialized plotters."""
        plots = {}

        try:
            # Basic MI plot
            if "basic_mi" in results:
                fig, ax = self.create_subplot()
                self.basic_plotter.plot(results["basic_mi"], ax)
                plots["basic_mi"] = (fig, ax)

            # Pairwise MI heatmap
            if "pairwise_mi" in results:
                fig, ax = self.create_subplot()
                self.pairwise_plotter.plot_heatmap(results["pairwise_mi"], ax)
                plots["pairwise_mi"] = (fig, ax)

            # Comprehensive summary
            if any(key in results for key in ["basic_mi", "pairwise_mi"]):
                fig, axes = self._plot_comprehensive_summary(results)
                plots["summary"] = (fig, axes)

            return plots

        except Exception as e:
            return handle_plot_error("information analysis plotting", e)

    def plot_per_layer_information(
        self, layer_results: dict[str, dict[str, Any]], **kwargs
    ):
        """Plot detailed per-layer information with comprehensive analysis."""
        try:
            save_path = kwargs.get("save_path")
            filename_prefix = kwargs.get("filename_prefix", "per_layer_info")

            if not layer_results:
                logger.warning("No layer results provided")
                return None

            # Try to get somatic_synapses from kwargs
            somatic_synapses = kwargs.get("somatic_synapses", True)
            converted_results = convert_layer_names(
                layer_results, somatic_synapses=somatic_synapses
            )
            if not converted_results:
                converted_results = layer_results

            layer_names = list(converted_results.keys())
            logger.info(f"Plotting per-layer information for layers: {layer_names}")

            # Create comprehensive plots for all information types
            plots = []

            # 1. Basic MI vs Layer Depth (Mean + Variance)
            fig1 = self._plot_layer_depth_analysis(
                converted_results,
                layer_names,
                "basic_mi",
                "Basic Mutual Information vs Layer Depth",
            )
            plots.append(fig1)
            if save_path and fig1:
                self.save(fig1, save_path, f"{filename_prefix}_basic_mi_depth")

            # 2. Pairwise MI vs Layer Depth
            fig2 = self._plot_layer_depth_analysis(
                converted_results,
                layer_names,
                "pairwise_mi",
                "Pairwise Mutual Information vs Layer Depth",
            )
            plots.append(fig2)
            if save_path and fig2:
                self.save(fig2, save_path, f"{filename_prefix}_pairwise_mi_depth")

            # 3. Conditional MI vs Layer Depth
            fig3 = self._plot_layer_depth_analysis(
                converted_results,
                layer_names,
                "conditional_mi",
                "Conditional Mutual Information vs Layer Depth",
            )
            plots.append(fig3)
            if save_path and fig3:
                self.save(fig3, save_path, f"{filename_prefix}_conditional_mi_depth")

            # 4. Branch Variance Analysis
            fig4 = self._plot_branch_variance_analysis(converted_results, layer_names)
            plots.append(fig4)

            # 5. Original detailed plot for compatibility
            fig5, axes = self.create_subplots(2, 2, figsize=(16, 12))
            self.layer_plotter.plot_basic_mi_layers(
                converted_results, layer_names, axes[0, 0]
            )
            self.layer_plotter.plot_info_flow_layers(
                converted_results, layer_names, axes[1, 0]
            )
            self._plot_layer_heatmap(axes[1, 1], converted_results, layer_names)
            plt.suptitle("Per-Layer Information Analysis (Soma → Distal)", fontsize=16)
            plt.tight_layout()
            plots.append(fig5)

            return plots

        except Exception as e:
            logger.error(f"Error in per-layer information plotting: {e}")
            return handle_plot_error("per-layer information plotting", e)

    def _plot_comprehensive_summary(
        self, results: dict[str, Any]
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create comprehensive summary plot."""
        fig, axes = self.create_subplots(2, 2, figsize=(15, 10))

        # Basic MI
        if "basic_mi" in results:
            self.basic_plotter.plot(results["basic_mi"], axes[0, 0])

        # Pairwise MI heatmap
        if "pairwise_mi" in results:
            self.pairwise_plotter.plot_heatmap(results["pairwise_mi"], axes[0, 1])

        # Summary text
        self._add_summary_text(axes[1, 0], results)

        plt.suptitle("Information Analysis Summary", fontsize=16)
        return fig, axes

    def _add_summary_text(self, ax: plt.Axes, results: dict[str, Any]):
        """Add summary statistics text."""
        summary_lines = []

        if "basic_mi" in results:
            basic_mi = results["basic_mi"]
            # Include all available basic MI metrics (E, I, Vb, Vout, and combinations)
            for key in ["I(E;C)", "I(I;C)", "I(Vb;C)", "I(Vout;C)", "I(E,I;C)"]:
                if key in basic_mi:
                    summary_lines.append(f"{key}: {basic_mi[key]:.4f} bits")

        if summary_lines:
            summary_text = "\n".join(summary_lines)
            ax.text(
                0.05,
                0.95,
                summary_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontfamily="monospace",
                fontsize=10,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Summary Statistics")

    def _plot_layer_heatmap(self, ax: plt.Axes, layer_results: dict, layer_names: list):
        """Plot layer information as heatmap."""
        # Check which metrics are available
        possible_metrics = [
            "I(E;C)_mean",
            "I(I;C)_mean",
            "I(Vb;C)_mean",
            "I(Vout;C)_mean",
            "I(E;I)_mean",
            "I(E_lin;C)_mean",
            "I(I_lin;C)_mean",
            "I(Vb_lin;C)_mean",
        ]
        possible_labels = [
            "I(E;C)",
            "I(I;C)",
            "I(Vb;C)",
            "I(Vout;C)",
            "I(E;I)",
            "I(E_lin;C)",
            "I(I_lin;C)",
            "I(Vb_lin;C)",
        ]

        metrics = []
        metric_labels = []
        for metric, label in zip(possible_metrics, possible_labels):
            if any(metric in layer_results.get(ln, {}) for ln in layer_names):
                metrics.append(metric)
                metric_labels.append(label)

        # Fallback if none found
        if not metrics:
            metrics = ["I(E;C)_mean", "I(I;C)_mean", "I(Vout;C)_mean", "I(E;I)_mean"]
            metric_labels = ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E;I)"]

        data_matrix = []
        for metric in metrics:
            row = [layer_results[name].get(metric, 0) for name in layer_names]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)
        create_heatmap(
            ax,
            data_matrix,
            layer_names,
            metric_labels,
            "Information Metrics Heatmap",
            annotate=True,
        )

    def _plot_layer_depth_analysis(
        self, layer_results: dict, layer_names: list, metric_type: str, title: str
    ) -> plt.Figure:
        """Plot information metrics vs layer depth with mean and variance."""
        try:
            # Define metric patterns for each type
            if metric_type == "basic_mi":
                metric_patterns = [
                    "I(E;C)",
                    "I(I;C)",
                    "I(Vb;C)",
                    "I(Vout;C)",
                    "I(E,I;C)",
                    "I(E_lin;C)",
                    "I(I_lin;C)",
                    "I(Vb_lin;C)",
                ]
            elif metric_type == "pairwise_mi":
                metric_patterns = [
                    "I(E;I)",
                    "I(E;Vout)",
                    "I(I;Vout)",
                    "I(E;Vb)",
                    "I(I;Vb)",
                    "I(Vb;Vout)",
                    "I(E_lin;I_lin)",
                    "I(E_lin;Vb_lin)",
                    "I(I_lin;Vb_lin)",
                ]
            elif metric_type == "conditional_mi":
                metric_patterns = [
                    "I(E;I|C)",
                    "I(E;Vout|C)",
                    "I(I;Vout|C)",
                    "I(E;Vb|C)",
                    "I(I;Vb|C)",
                    "I(Vb;Vout|C)",
                    "I(E_lin;I_lin|C)",
                    "I(E_lin;Vb_lin|C)",
                    "I(I_lin;Vb_lin|C)",
                ]
            else:
                metric_patterns = []

            # Get all metrics of this type by checking for pattern matches
            metrics = set()
            for layer_data in layer_results.values():
                for key in layer_data.keys():
                    if "_mean" in key:
                        # Extract base metric name by removing "_mean"
                        base_metric = key.replace("_mean", "")
                        # Check if this base metric matches any of our patterns
                        if any(pattern == base_metric for pattern in metric_patterns):
                            metrics.add(base_metric)

            metrics = sorted(metrics)
            logger.info(f"Found {metric_type} metrics: {metrics}")

            # Determine grid size based on number of metrics
            n_metrics = len(metrics)
            if n_metrics == 0:
                logger.warning(f"No {metric_type} metrics found")
                return None

            # Create appropriate grid: 2x2 for 3-4 metrics, 2x3 for 5-6, 3x3 for 7-9, etc.
            ncols = min(3, (n_metrics + 1) // 2 + (n_metrics % 2))
            nrows = (n_metrics + ncols - 1) // ncols

            fig, axes = self.create_subplots(
                nrows, ncols, figsize=(7 * ncols, 5 * nrows)
            )
            if nrows == 1 and ncols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # Plot each metric
            for idx, metric in enumerate(metrics):  # Plot all metrics, not just first 4
                ax = axes.flat[idx]

                means = []
                stds = []
                valid_layers = []

                for layer_name in layer_names:
                    if layer_name in layer_results:
                        mean_key = f"{metric}_mean"
                        std_key = f"{metric}_std"

                        if mean_key in layer_results[layer_name]:
                            means.append(layer_results[layer_name][mean_key])
                            # Reduce error bar size by dividing by 5
                            stds.append(layer_results[layer_name].get(std_key, 0) / 5.0)
                            valid_layers.append(layer_name)

                if means:
                    x_pos = range(len(valid_layers))

                    # Plot means with error bars
                    ax.errorbar(
                        x_pos,
                        means,
                        yerr=stds,
                        marker="o",
                        capsize=5,
                        linewidth=2,
                        markersize=6,
                        label=f"{metric} (mean ± std/5)",
                    )

                    # Plot variance as filled area
                    ax.fill_between(
                        x_pos,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3,
                    )

                    ax.set_xlabel("Layer (Soma → Distal)")
                    ax.set_ylabel(f"{metric} (bits)")
                    ax.set_title(f"{metric}")
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(valid_layers, rotation=45)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {metric}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

            # Hide unused subplots
            for idx in range(len(metrics), len(axes)):
                axes[idx].axis("off")

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting {metric_type} vs depth: {e}")
            return None

    def _plot_branch_variance_analysis(
        self, layer_results: dict, layer_names: list
    ) -> plt.Figure:
        """Plot analysis of variance between branches in each layer."""
        try:
            fig, axes = self.create_subplots(2, 2, figsize=(15, 10))

            # Get all std metrics (variance indicators)
            std_metrics = []
            for layer_data in layer_results.values():
                for key in layer_data.keys():
                    if "_std" in key and key not in std_metrics:
                        std_metrics.append(key)

            logger.info(f"Found variance metrics: {std_metrics}")

            # Plot variance for different metric types
            for idx, std_metric in enumerate(std_metrics[:4]):  # Limit to 4 subplots
                ax = axes.flat[idx]

                variances = []
                valid_layers = []

                for layer_name in layer_names:
                    if (
                        layer_name in layer_results
                        and std_metric in layer_results[layer_name]
                    ):
                        variances.append(layer_results[layer_name][std_metric])
                        valid_layers.append(layer_name)

                if variances:
                    x_pos = range(len(valid_layers))
                    bars = ax.bar(
                        x_pos,
                        variances,
                        alpha=0.7,
                        color=plt.cm.viridis(idx / max(1, len(std_metrics) - 1)),
                    )

                    # Add value labels on bars
                    for bar, val in zip(bars, variances):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + height * 0.01,
                            f"{val:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                    ax.set_xlabel("Layer (Soma → Distal)")
                    ax.set_ylabel("Standard Deviation (bits)")
                    ax.set_title(f'Branch Variance: {std_metric.replace("_std", "")}')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(valid_layers, rotation=45)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No variance data for {std_metric}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

            plt.suptitle(
                "Branch Variance Analysis (Variance Between Branches per Layer)",
                fontsize=16,
            )
            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting branch variance: {e}")
            return None


# ============================================================================
# Convenience Functions
# ============================================================================


def plot_per_layer_information(layer_results: dict[str, dict[str, Any]], **kwargs):
    """Convenience function for per-layer information plotting."""
    plotter = InformationAnalysisPlotter()
    return plotter.plot_per_layer_information(layer_results, **kwargs)


# ============================================================================
# Dendritic Depth Analysis Functions
# ============================================================================


def plot_structured_ei_network_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
):
    """Generate structured plots comparing different layers with color-coded lines.

    Creates 2 plots:
    - Excitatory Networks: Layer 0 vs Layer 1 as different colored lines
    - Inhibitory Networks: Layer 0 vs Layer 1 as different colored lines
    X-axis: [Soma, Layer 1, Layer 2] (dendritic depth)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating structured EI network plots with layer comparison...")

    # Group layers by major layer and network type
    layer_groups = {}
    for layer_name in layer_statistics.keys():
        parts = layer_name.split(".")
        if len(parts) >= 6:
            major_layer = parts[2]  # layers.0 or layers.1
            network_type = parts[3]  # inhibitory_cells or excitatory_cells
            branch_layer = parts[5]  # branch_layers.0, branch_layers.1, branch_layers.2

            group_key = f"{major_layer}_{network_type}"
            if group_key not in layer_groups:
                layer_groups[group_key] = {}

            layer_groups[group_key][branch_layer] = layer_statistics[layer_name]

    logger.info(
        f"Found {len(layer_groups)} major layer groups: {list(layer_groups.keys())}"
    )

    # Reorganize by network type instead of individual layers
    network_comparison_data = {
        "excitatory": {},  # Will contain Layer 0 and Layer 1 data
        "inhibitory": {},  # Will contain Layer 0 and Layer 1 data
    }

    # Process each group and organize by network type
    for group_key, group_data in layer_groups.items():
        major_layer, network_type = group_key.split("_", 1)

        # Create ordered layer data for this group (only include existing branches)
        ordered_data = {}
        branch_to_label = {"0": "Soma", "1": "Layer 1", "2": "Layer 2"}

        # Only process branches that actually exist in the data
        for branch_idx in sorted(group_data.keys()):  # Use available branches only
            if branch_idx in branch_to_label:
                label = branch_to_label[branch_idx]
                ordered_data[label] = group_data[branch_idx]

        if ordered_data:
            # Get clean network type name
            net_type_clean = network_type.split("_")[0]  # "excitatory" or "inhibitory"
            layer_label = f"Layer {major_layer[-1]}"  # "Layer 0" or "Layer 1"

            # Store data organized for comparison plotting
            if net_type_clean not in network_comparison_data:
                network_comparison_data[net_type_clean] = {}

            network_comparison_data[net_type_clean][layer_label] = ordered_data

    # Generate comparison plots
    _generate_layer_comparison_plots(network_comparison_data, save_path, logger)


def _generate_layer_comparison_plots(
    network_comparison_data: dict, save_path: str, logger: logging.Logger
):
    """Generate plots comparing Layer 0 vs Layer 1 for each network type."""
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # Define colors for different layers
    layer_colors = {
        "Layer 0": "#1f77b4",  # Blue
        "Layer 1": "#ff7f0e",  # Orange
    }

    # Information metrics to plot
    metric_types = [
        (
            "basic_mi",
            [
                "I(E;C)",
                "I(I;C)",
                "I(Vb;C)",
                "I(Vout;C)",
                "I(E,I;C)",
                "I(E_lin;C)",
                "I(I_lin;C)",
                "I(Vb_lin;C)",
            ],
        ),
        (
            "pairwise_mi",
            [
                "I(E;I)",
                "I(E;Vout)",
                "I(I;Vout)",
                "I(E;Vb)",
                "I(I;Vb)",
                "I(Vb;Vout)",
                "I(E_lin;I_lin)",
                "I(E_lin;Vb_lin)",
                "I(I_lin;Vb_lin)",
            ],
        ),
        (
            "conditional_mi",
            [
                "I(E;I|C)",
                "I(E;Vout|C)",
                "I(I;Vout|C)",
                "I(E;Vb|C)",
                "I(I;Vb|C)",
                "I(Vb;Vout|C)",
                "I(E_lin;I_lin|C)",
                "I(E_lin;Vb_lin|C)",
                "I(I_lin;Vb_lin|C)",
            ],
        ),
    ]

    for network_type, layer_data in network_comparison_data.items():
        logger.info(f"Generating comparison plots for {network_type} networks")

        for metric_category, metrics in metric_types:
            # Create figure with subplots for each metric - dynamic sizing
            n_metrics = len(metrics)
            n_cols = 3  # Use 3 columns
            n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for idx, metric in enumerate(metrics):
                if idx >= len(axes):
                    break
                ax = axes[idx]

                # First, determine all available depths from all layers
                all_available_depths = set()
                for _layer_name, depth_data in layer_data.items():
                    all_available_depths.update(depth_data.keys())

                # Sort depths in logical order (soma to distal)
                # Create dynamic depth ordering for any number of distal layers
                def depth_sort_key(depth_name):
                    if depth_name == "Soma":
                        return 0
                    elif depth_name.startswith("Distal Layer"):
                        try:
                            return int(depth_name.split()[-1])
                        except ValueError:
                            return 99
                    else:
                        return 99

                sorted_depths = sorted(all_available_depths, key=depth_sort_key)

                # Plot each layer as a different colored line
                for layer_name, depth_data in layer_data.items():
                    if not depth_data:
                        continue

                    # Extract dendritic depths and metric values
                    depths = []
                    values = []
                    errors = []

                    # Use the sorted depths that were determined for all layers
                    for depth_label in sorted_depths:
                        if depth_label in depth_data:
                            depth_stats = depth_data[depth_label]
                            if f"{metric}_mean" in depth_stats:
                                depths.append(depth_label)
                                values.append(depth_stats[f"{metric}_mean"])
                                # Use std/5 for smaller error bars
                                errors.append(depth_stats.get(f"{metric}_std", 0) / 5.0)

                    if depths and values:
                        x_pos = np.arange(len(depths))
                        color = layer_colors.get(layer_name, "#808080")

                        # Plot line with error bars
                        ax.errorbar(
                            x_pos,
                            values,
                            yerr=errors,
                            marker="o",
                            linewidth=2,
                            markersize=6,
                            label=layer_name,
                            color=color,
                            capsize=3,
                        )

                        # Fill between for variance
                        ax.fill_between(
                            x_pos,
                            [v - e for v, e in zip(values, errors)],
                            [v + e for v, e in zip(values, errors)],
                            alpha=0.2,
                            color=color,
                        )

                # Customize subplot using the actual depths found in the data
                if sorted_depths:  # Only customize if we have data
                    ax.set_xlabel("Dendritic Depth")
                    ax.set_ylabel(f"{metric} (bits)")
                    ax.set_title(f"{metric}")
                    ax.set_xticks(range(len(sorted_depths)))
                    ax.set_xticklabels(sorted_depths, rotation=0)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {metric}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

            # Remove empty subplots
            for idx in range(len(metrics), len(axes)):
                fig.delaxes(axes[idx])

            # Set overall title and save
            fig.suptitle(
                f"{network_type.title()} Networks: {metric_category.replace('_', ' ').title()} vs Dendritic Depth",
                fontsize=16,
            )
            plt.tight_layout()

            filename = f"layer_comparison_{network_type}_{metric_category}_depth.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved comparison plot: {filename}")


def plot_layer_wise_aggregated_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
):
    """Generate layer-wise aggregated plots when per_layer is enabled but per_einet is disabled.

    This aggregates E/I networks within each layer but keeps layers separate.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating layer-wise aggregated plots...")
    # Implementation similar to _plot_structured_ei_network_analysis but aggregating E/I
    # This would be called when per_layer=True, per_einet=False
    # Placeholder for now


def plot_network_wise_aggregated_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
):
    """Generate network-wise aggregated plots when per_einet is enabled but per_layer is disabled.

    This aggregates layers but keeps E/I networks separate.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating network-wise aggregated plots...")
    # Implementation would aggregate across layers but keep E/I separate
    # This would be called when per_layer=False, per_einet=True
    # Placeholder for now


def plot_info_vs_ei_depth_per_dendritic_layer(
    layer_statistics: dict[str, Any],
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
) -> None:
    """Plot information vs EI layer depth for each dendritic branch layer.

    This creates the complement to the existing plots - instead of showing
    info vs dendritic depth for each EI layer, this shows info vs EI depth
    for each dendritic branch layer.

    Args:
        layer_statistics: Layer-wise information statistics
        save_path: Directory to save plots
        logger: Optional logger
    """
    try:
        if logger:
            logger.info("Starting plot_info_vs_ei_depth_per_dendritic_layer...")
            logger.info(f"Received {len(layer_statistics)} layer statistics")

        # Organize data by dendritic branch layer instead of EI layer
        dendritic_layers_data = {}

        # Determine dynamic max branch index for correct depth mapping
        branch_indices = []
        for lname in layer_statistics.keys():
            if "branch_layers." in lname:
                try:
                    branch_indices.append(
                        int(lname.split("branch_layers.")[1].split(".")[0])
                    )
                except (IndexError, ValueError):
                    continue
        max_branch_idx = max(branch_indices) if branch_indices else -1

        # Extract data from layer_statistics
        for layer_name, stats in layer_statistics.items():
            if logger:
                logger.debug(f"Processing layer: {layer_name}")

            try:
                # Try to parse layer name - handle different formats
                parts = layer_name.split(".")
                if logger:
                    logger.debug(f"Layer parts: {parts}")

                # Determine EI layer index and cell type
                ei_layer_idx = None
                cell_type = None
                branch_layer_idx = 0  # Default to 0 for single branch layer

                # Format 1: "core_network.layers.0.excitatory_cells.branch_layers.1"
                if "layers" in parts and "branch_layers" in parts:
                    idx = parts.index("layers")
                    ei_layer_idx = int(parts[idx + 1])
                    cell_type = parts[idx + 2]
                    branch_idx = parts.index("branch_layers")
                    branch_layer_idx = int(parts[branch_idx + 1])

                # Format 2: "core_network.layers.0.excitatory_cells" (without branch_layers)
                elif "layers" in parts:
                    idx = parts.index("layers")
                    ei_layer_idx = int(parts[idx + 1])
                    cell_type = parts[idx + 2]
                    # Assume this is the main output layer (branch_layer 2)
                    branch_layer_idx = 2

                else:
                    # Skip if we can't parse the format
                    if logger:
                        logger.debug(f"Skipping unparseable layer: {layer_name}")
                    continue

                # Normalize cell type
                if "excitatory" in cell_type.lower():
                    cell_type = "excitatory_cells"
                elif "inhibitory" in cell_type.lower():
                    cell_type = "inhibitory_cells"
                else:
                    continue

                # Create key for dendritic layer
                dendritic_key = f"branch_layer_{branch_layer_idx}_{cell_type}"

                if dendritic_key not in dendritic_layers_data:
                    dendritic_layers_data[dendritic_key] = {}

                dendritic_layers_data[dendritic_key][ei_layer_idx] = stats

            except (ValueError, IndexError) as e:
                if logger:
                    logger.debug(f"Error parsing layer {layer_name}: {e}")
                continue

        if not dendritic_layers_data:
            if logger:
                logger.warning("No valid dendritic layer data found for EI depth plots")
            return

        # Create plots for each metric type (basic, pairwise, conditional)
        metric_types = [
            (
                "basic_mi",
                [
                    "I(E;C)",
                    "I(I;C)",
                    "I(Vb;C)",
                    "I(Vout;C)",
                    "I(E,I;C)",
                    "I(E_lin;C)",
                    "I(I_lin;C)",
                    "I(Vb_lin;C)",
                    "I(E_lin,I_lin;C)",
                ],
            ),
            (
                "pairwise_mi",
                [
                    "I(E;I)",
                    "I(E;Vout)",
                    "I(I;Vout)",
                    "I(E;Vb)",
                    "I(I;Vb)",
                    "I(Vb;Vout)",
                    "I(E_lin;I_lin)",
                    "I(E_lin;Vout)",
                    "I(I_lin;Vout)",
                    "I(E_lin;Vb_lin)",
                    "I(I_lin;Vb_lin)",
                ],
            ),
            (
                "conditional_mi",
                [
                    "I(E;I|C)",
                    "I(E;Vout|C)",
                    "I(I;Vout|C)",
                    "I(E;Vb|C)",
                    "I(I;Vb|C)",
                    "I(Vb;Vout|C)",
                    "I(E_lin;I_lin|C)",
                    "I(E_lin;Vout|C)",
                    "I(I_lin;Vout|C)",
                ],
            ),
        ]

        # Define colors for different branch layers
        branch_colors = {
            0: "#1f77b4",  # Blue for Soma/Branch Layer 0
            1: "#ff7f0e",  # Orange for Branch Layer 1
            2: "#2ca02c",  # Green for Branch Layer 2
        }

        if logger:
            logger.info(
                f"Dendritic layers data keys: {list(dendritic_layers_data.keys())}"
            )
            logger.info(f"Going to generate plots for metric types: {metric_types}")

        for metric_category, metric_names in metric_types:
            # Create separate plots for excitatory and inhibitory
            for cell_type in ["excitatory_cells", "inhibitory_cells"]:
                # Dynamically size grid based on number of metrics
                n_metrics = len(metric_names)
                n_cols = 3
                n_rows = (n_metrics + n_cols - 1) // n_cols

                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows)
                )
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                # Track if any plots were created
                plots_created = []

                # Plot all metrics
                for i, metric_name in enumerate(metric_names):
                    if i >= len(axes):
                        break

                    ax = axes[i]
                    data_plotted = False

                    # Plot each dendritic branch layer as a different colored line
                    for branch_idx in range(3):  # Assuming 3 dendritic layers (0, 1, 2)
                        dendritic_key = f"branch_layer_{branch_idx}_{cell_type}"

                        if dendritic_key not in dendritic_layers_data:
                            continue

                        ei_data = dendritic_layers_data[dendritic_key]

                        # Sort by EI layer index
                        ei_layers = sorted(ei_data.keys())

                        if len(ei_layers) < 2:  # Need at least 2 points
                            continue

                        # Collect values across EI layers using flattened format
                        values = []
                        errors = []

                        for ei_idx in ei_layers:
                            if ei_idx in ei_data:
                                layer_stats = ei_data[ei_idx]
                                # Use flattened format: metric_name + "_mean" and "_std"
                                mean_key = f"{metric_name}_mean"
                                std_key = f"{metric_name}_std"

                                if mean_key in layer_stats:
                                    values.append(layer_stats[mean_key])
                                    # Use std/5 for smaller error bars like in layer_comparison
                                    errors.append(layer_stats.get(std_key, 0) / 5.0)
                                else:
                                    values.append(0)
                                    errors.append(0)
                            else:
                                values.append(0)
                                errors.append(0)

                        # Only plot if we have non-zero values
                        if any(v != 0 for v in values):
                            color = branch_colors.get(branch_idx, "#808080")
                            # Fix branch labeling - higher branch_idx means more distal
                            # Map branch_idx to correct distal depth dynamically
                            # Highest branch_idx is closest to soma (Distal Layer 1)
                            if max_branch_idx >= 0:
                                distal_number = (max_branch_idx - branch_idx) + 1
                                if (
                                    not somatic_synapses
                                    and branch_idx == max_branch_idx
                                    and "synthetic_soma" in layer_statistics
                                ):
                                    # When somatic_synapses=False, we explicitly include a synthetic Soma elsewhere
                                    branch_label = f"Distal Layer {distal_number}"
                                else:
                                    branch_label = (
                                        "Soma"
                                        if distal_number == 0
                                        else f"Distal Layer {distal_number}"
                                    )
                                # distal_number never 0 here; Soma is provided via synthetic_soma
                            else:
                                # Fallback
                                branch_label = f"Distal Layer {branch_idx}"

                            # Plot line with error bars
                            ax.errorbar(
                                ei_layers,
                                values,
                                yerr=errors,
                                marker="o",
                                linewidth=2,
                                markersize=6,
                                label=branch_label,
                                color=color,
                                capsize=3,
                            )

                            # Fill between for variance (like layer_comparison)
                            ax.fill_between(
                                ei_layers,
                                [v - e for v, e in zip(values, errors)],
                                [v + e for v, e in zip(values, errors)],
                                alpha=0.2,
                                color=color,
                            )

                            data_plotted = True

                    # Customize subplot if data was plotted
                    if data_plotted:
                        ax.set_xlabel("EI Network Layer")
                        ax.set_ylabel(f"{metric_name} (bits)")
                        ax.set_title(f"{metric_name}")
                        ax.set_xticks(ei_layers)

                        # Create custom labels for layers
                        layer_labels = []
                        for idx, layer_idx in enumerate(ei_layers):
                            if idx == 0:
                                layer_labels.append("Input")
                            elif idx == len(ei_layers) - 1:
                                layer_labels.append("Output")
                            else:
                                layer_labels.append(f"Layer {layer_idx}")

                        ax.set_xticklabels(layer_labels, rotation=0)
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc="best")
                        plots_created.append(i)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            f"No data for {metric_name}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )

                # Remove empty subplots and finalize the figure
                if plots_created:
                    # Remove unused subplots
                    for i in range(len(metric_names), 4):
                        fig.delaxes(axes[i])

                    cell_type_name = cell_type.replace("_cells", "").title()
                    fig.suptitle(
                        f'{cell_type_name} Networks: {metric_category.replace("_", " ").title()} vs EI Network Layer',
                        fontsize=16,
                    )
                    plt.tight_layout()

                    # Save plot
                    filename = f"ei_depth_comparison_{cell_type}_{metric_category}.png"
                    plot_path = os.path.join(save_path, filename)
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    if logger:
                        logger.info(f"Saved EI depth comparison plot to {plot_path}")
                else:
                    plt.close(fig)

    except Exception as e:
        if logger:
            logger.error(f"Error generating EI depth comparison plots: {e}")
        else:
            print(f"Error generating EI depth comparison plots: {e}")


def plot_fully_aggregated_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
    plot_variance: bool = False,
):
    """Generate fully aggregated plots when both per_layer and per_einet analysis are disabled.

    This creates plots that:
    - Aggregate across all layers (Layer 0 + Layer 1)
    - Aggregate across E/I networks (Excitatory + Inhibitory)
    - Show dendritic depth (Soma, Layer 1, Layer 2) on x-axis
    - Display one line representing the overall network

    Args:
        plot_variance: If True, plot variance (std^2) instead of mean values
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    plot_type = "variance" if plot_variance else "mean"
    logger.info(f"Generating fully aggregated dendritic depth plots ({plot_type})...")
    logger.info(f"Input layer_statistics keys: {list(layer_statistics.keys())}")

    # Debug: Show sample data structure
    if layer_statistics:
        sample_key = next(iter(layer_statistics.keys()))
        sample_data = layer_statistics[sample_key]
        logger.info(
            f"Sample layer data keys for '{sample_key}': {list(sample_data.keys()) if isinstance(sample_data, dict) else 'Not a dict'}"
        )

    # Aggregate data across all layers and E/I networks
    aggregated_data = {}

    # Determine dynamic mapping for dendritic depth (Soma -> proximal -> distal)
    # Collect all branch indices to compute max index
    branch_indices = []
    for lname in layer_statistics.keys():
        if "branch_layers." in lname:
            try:
                branch_indices.append(
                    int(lname.split("branch_layers.")[1].split(".")[0])
                )
            except (IndexError, ValueError):
                continue
    max_branch_idx = max(branch_indices) if branch_indices else -1

    # First, collect all depth labels that exist across all entries
    all_depths = set()
    for layer_name, layer_data in layer_statistics.items():
        if not isinstance(layer_data, dict):
            continue

        # Helper to register a metric entry under a given depth label
        def register_metric(
            depth_label: str, metric_name: str, value: float, std_val: float
        ):
            all_depths.add(depth_label)
            if depth_label not in aggregated_data:
                aggregated_data[depth_label] = {}
            if metric_name not in aggregated_data[depth_label]:
                aggregated_data[depth_label][metric_name] = {"values": [], "stds": []}

            # Store the value directly (either mean or variance depending on plot_variance)
            aggregated_data[depth_label][metric_name]["values"].append(float(value))
            aggregated_data[depth_label][metric_name]["stds"].append(float(std_val))

        # Case 1: synthetic soma (added when somatic_synapses=False)
        if layer_name == "synthetic_soma":
            if plot_variance:
                # For variance plots, look for _variance keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_variance"):
                        metric_name = depth_key.replace("_variance", "")
                        var_val = layer_data.get(f"{metric_name}_variance", 0)
                        register_metric(
                            "Soma", metric_name, var_val, 0
                        )  # No std for variance plots
            else:
                # For mean plots, look for _mean keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_mean"):
                        metric_name = depth_key.replace("_mean", "")
                        mean_val = layer_data.get(f"{metric_name}_mean", 0)
                        std_val = layer_data.get(f"{metric_name}_std", 0)
                        register_metric("Soma", metric_name, mean_val, std_val)
            continue

        # Case 2: actual dendritic branch layers
        if "branch_layers." in layer_name:
            try:
                branch_idx = int(layer_name.split("branch_layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                continue

            # Map branch index to distal depth dynamically.
            # Highest branch index is closest to soma (depth 1), lowest is most distal.
            # Distal Layer numbering starts at 1.
            if max_branch_idx >= 0:
                distal_number = (max_branch_idx - branch_idx) + 1
                depth_label = f"Distal Layer {distal_number}"
            else:
                # Fallback if we couldn't determine max index
                depth_label = f"Distal Layer {branch_idx}"

            if plot_variance:
                # For variance plots, look for _variance keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_variance"):
                        metric_name = depth_key.replace("_variance", "")
                        var_val = layer_data.get(f"{metric_name}_variance", 0)
                        register_metric(
                            depth_label, metric_name, var_val, 0
                        )  # No std for variance plots
            else:
                # For mean plots, look for _mean keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_mean"):
                        metric_name = depth_key.replace("_mean", "")
                        mean_val = layer_data.get(f"{metric_name}_mean", 0)
                        std_val = layer_data.get(f"{metric_name}_std", 0)
                        register_metric(depth_label, metric_name, mean_val, std_val)

    # Compute final aggregated statistics
    final_aggregated = {}
    for depth_label, depth_data in aggregated_data.items():
        final_aggregated[depth_label] = {}
        for metric_name, metric_data in depth_data.items():
            # Average across all layers and networks
            final_aggregated[depth_label][f"{metric_name}_mean"] = np.mean(
                metric_data["values"]
            )
            # Combine standard deviations (root mean square)
            final_aggregated[depth_label][f"{metric_name}_std"] = np.sqrt(
                np.mean(np.array(metric_data["stds"]) ** 2)
            )

    logger.info(f"Final aggregated data depths: {list(final_aggregated.keys())}")
    if final_aggregated:
        sample_depth = next(iter(final_aggregated.keys()))
        logger.info(
            f"Sample depth '{sample_depth}' metrics: {list(final_aggregated[sample_depth].keys())}"
        )

    # Sort depths in logical order: Soma first, then Distal Layer 1..N
    def depth_sort_key(label: str) -> int:
        if label == "Soma":
            return 0
        if label.startswith("Distal Layer"):
            try:
                return int(label.split()[-1])
            except ValueError:
                return 999
        return 999

    sorted_depths = sorted(all_depths, key=depth_sort_key)

    # Define metric categories with paired comparisons (network-learned vs LDA)
    # Each entry: (title, [(metric_name, color, label), ...])
    metric_pairs = [
        (
            "basic_mi",
            [
                (
                    "I(E;C) vs I(E_lin;C)",
                    [
                        ("I(E;C)", "#1f77b4", "Network"),
                        ("I(E_lin;C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(I;C) vs I(I_lin;C)",
                    [
                        ("I(I;C)", "#1f77b4", "Network"),
                        ("I(I_lin;C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(Vb;C) vs I(Vb_lin;C)",
                    [
                        ("I(Vb;C)", "#1f77b4", "Network"),
                        ("I(Vb_lin;C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(Vout;C) vs I(Vout_lin;C)",
                    [
                        ("I(Vout;C)", "#1f77b4", "Network"),
                        ("I(Vout_lin;C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(E,I;C) vs I(E_lin,I_lin;C)",
                    [
                        ("I(E,I;C)", "#1f77b4", "Network"),
                        ("I(E_lin,I_lin;C)", "#ff7f0e", "LDA"),
                    ],
                ),
            ],
        ),
        (
            "pairwise_mi",
            [
                (
                    "I(E;I) vs I(E_lin;I_lin)",
                    [
                        ("I(E;I)", "#1f77b4", "Network"),
                        ("I(E_lin;I_lin)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(E;Vout) vs I(E_lin;Vout_lin)",
                    [
                        ("I(E;Vout)", "#1f77b4", "Network"),
                        ("I(E_lin;Vout_lin)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(I;Vout) vs I(I_lin;Vout_lin)",
                    [
                        ("I(I;Vout)", "#1f77b4", "Network"),
                        ("I(I_lin;Vout_lin)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(E;Vb) vs I(E_lin;Vb_lin)",
                    [
                        ("I(E;Vb)", "#1f77b4", "Network"),
                        ("I(E_lin;Vb_lin)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(I;Vb) vs I(I_lin;Vb_lin)",
                    [
                        ("I(I;Vb)", "#1f77b4", "Network"),
                        ("I(I_lin;Vb_lin)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(Vb;Vout) vs I(Vb_lin;Vout_lin)",
                    [
                        ("I(Vb;Vout)", "#1f77b4", "Network"),
                        ("I(Vb_lin;Vout_lin)", "#ff7f0e", "LDA"),
                    ],
                ),
            ],
        ),
        (
            "conditional_mi",
            [
                (
                    "I(E;I|C) vs I(E_lin;I_lin|C)",
                    [
                        ("I(E;I|C)", "#1f77b4", "Network"),
                        ("I(E_lin;I_lin|C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(E;Vout|C) vs I(E_lin;Vout_lin|C)",
                    [
                        ("I(E;Vout|C)", "#1f77b4", "Network"),
                        ("I(E_lin;Vout_lin|C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(I;Vout|C) vs I(I_lin;Vout_lin|C)",
                    [
                        ("I(I;Vout|C)", "#1f77b4", "Network"),
                        ("I(I_lin;Vout_lin|C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(E;Vb|C) vs I(E_lin;Vb_lin|C)",
                    [
                        ("I(E;Vb|C)", "#1f77b4", "Network"),
                        ("I(E_lin;Vb_lin|C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(I;Vb|C) vs I(I_lin;Vb_lin|C)",
                    [
                        ("I(I;Vb|C)", "#1f77b4", "Network"),
                        ("I(I_lin;Vb_lin|C)", "#ff7f0e", "LDA"),
                    ],
                ),
                (
                    "I(Vb;Vout|C) vs I(Vb_lin;Vout_lin|C)",
                    [
                        ("I(Vb;Vout|C)", "#1f77b4", "Network"),
                        ("I(Vb_lin;Vout_lin|C)", "#ff7f0e", "LDA"),
                    ],
                ),
            ],
        ),
    ]

    # Generate plots for each metric category with paired comparisons
    for metric_category, metric_groups in metric_pairs:
        n_plots = len(metric_groups)
        n_cols = 3  # Use 3 columns for better layout
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, (plot_title, metrics_list) in enumerate(metric_groups):
            if idx >= len(axes):
                break
            ax = axes[idx]
            has_data = False

            # Plot each metric in this comparison group (network vs LDA)
            for metric, color, label in metrics_list:
                depths = []
                values = []
                errors = []

                for depth_label in sorted_depths:
                    if depth_label in final_aggregated:
                        depth_stats = final_aggregated[depth_label]
                        if f"{metric}_mean" in depth_stats:
                            depths.append(depth_label)
                            values.append(depth_stats[f"{metric}_mean"])
                            errors.append(depth_stats.get(f"{metric}_std", 0) / 5.0)

                if depths and values:
                    has_data = True
                    x_pos = np.arange(len(depths))

                    ax.errorbar(
                        x_pos,
                        values,
                        yerr=errors,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=label,
                        color=color,
                        capsize=3,
                    )

                    ax.fill_between(
                        x_pos,
                        [v - e for v, e in zip(values, errors)],
                        [v + e for v, e in zip(values, errors)],
                        alpha=0.15,
                        color=color,
                    )

            if has_data:
                ax.set_xlabel("Dendritic Depth")
                if plot_variance:
                    ax.set_ylabel("MI variance (bits^2)")
                    ax.set_title(f"{plot_title} (variance)")
                else:
                    ax.set_ylabel("MI (bits)")
                    ax.set_title(plot_title)
                ax.set_xticks(range(len(sorted_depths)))
                ax.set_xticklabels(sorted_depths, rotation=0)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
            else:
                data_type = "variance" if plot_variance else "data"
                ax.text(
                    0.5,
                    0.5,
                    f"No {data_type} for {plot_title}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        # Remove empty subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        plot_type = "Variance" if plot_variance else "Analysis"
        fig.suptitle(
            f"Information {plot_type}: {metric_category.replace('_', ' ').title()} (Network vs LDA)",
            fontsize=16,
        )
        plt.tight_layout()

        suffix = "_variance" if plot_variance else ""
        filename = f"aggregated_{metric_category}_comparison{suffix}_depth.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved aggregated comparison plot: {filename}")


def plot_information_variance_analysis(
    variance_input: dict,
    save_path: str,
    somatic_synapses: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Simple wrapper that calls plot_fully_aggregated_analysis with variance=True.

    This function exists for backward compatibility with the analysis pipeline.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Extract layer_variances if it's wrapped in variance_analysis dict
    if isinstance(variance_input, dict) and "layer_variances" in variance_input:
        layer_statistics = variance_input.get("layer_variances", {})
    else:
        layer_statistics = variance_input

    # Call the enhanced aggregated analysis function with variance=True
    plot_fully_aggregated_analysis(
        layer_statistics=layer_statistics,
        save_path=save_path,
        logger=logger,
        somatic_synapses=somatic_synapses,
        plot_variance=True,
    )


def plot_lda_vs_trained_performance(
    comparison_results: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot comparison of network performance with trained weights vs LDA-optimal weights.

    Args:
        comparison_results: Dictionary from InformationAnalyzer.compare_lda_vs_trained_performance
            containing 'trained_accuracy', 'lda_accuracy', 'accuracy_difference'
        save_path: Directory path where the plot will be saved
        logger: Optional logger for output messages
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    trained_acc = comparison_results.get("trained_accuracy", 0.0)
    lda_acc = comparison_results.get("lda_accuracy", 0.0)
    diff = comparison_results.get("accuracy_difference", 0.0)

    if lda_acc is None:
        logger.warning(
            "LDA accuracy not available, skipping performance comparison plot"
        )
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Bar chart comparing accuracies
    x_pos = [0, 1]
    accuracies = [trained_acc * 100, lda_acc * 100]
    colors = ["#1f77b4", "#ff7f0e"]
    labels = ["Trained Weights", "LDA Weights"]

    bars = ax.bar(x_pos, accuracies, color=colors, width=0.6, edgecolor="black")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(
            f"{acc:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Network Performance: Trained vs LDA Weights", fontsize=14)

    # Add difference annotation
    ax.annotate(
        f"Δ = {diff * 100:+.2f}%",
        xy=(0.5, max(accuracies) + 3),
        xytext=(0.5, max(accuracies) + 5),
        ha="center",
        fontsize=11,
        color="green" if diff >= 0 else "red",
    )

    # Set y-axis limits
    ax.set_ylim(0, max(accuracies) + 10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, "lda_vs_trained_performance.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved LDA vs trained performance plot: {filepath}")
    logger.info(f"Trained accuracy: {trained_acc:.4f}, LDA accuracy: {lda_acc:.4f}")
