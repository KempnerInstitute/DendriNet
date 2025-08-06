"""
Clean modular information analysis plotting for dendritic modeling.

This module provides specialized plotting classes for information-theoretic analysis.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from dendritic_modeling import logger

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

        variables = ["E", "I", "Vout"]
        if any("Vinf" in key for key in pairwise_mi.keys()):
            variables.append("Vinf")

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
        metrics_to_plot = ["I(E;C)_mean", "I(I;C)_mean", "I(Vout;C)_mean"]
        metric_labels = ["I(E;C)", "I(I;C)", "I(Vout;C)"]
        colors = [
            self.get_color("excitatory"),
            self.get_color("inhibitory"),
            self.get_color("combined"),
        ]

        x = np.arange(len(layer_names))
        width = 0.25

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

            converted_results = convert_layer_names(layer_results)
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
            for key in ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E,I;C)"]:
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
            fig, axes = self.create_subplots(2, 2, figsize=(15, 10))

            # Define metric patterns for each type
            if metric_type == "basic_mi":
                metric_patterns = ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E,I;C)"]
            elif metric_type == "pairwise_mi":
                metric_patterns = ["I(E;I)", "I(E;Vout)", "I(I;Vout)"]
            elif metric_type == "conditional_mi":
                metric_patterns = ["I(E;I|C)", "I(E;Vout|C)", "I(I;Vout|C)"]
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

            # Plot each metric
            for idx, metric in enumerate(metrics[:4]):  # Limit to 4 subplots
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


def plot_information_analysis(results: dict[str, Any], **kwargs):
    """Convenience function for information analysis plotting."""
    plotter = InformationAnalysisPlotter()
    if "filename" in kwargs and "filename_prefix" not in kwargs:
        kwargs["filename_prefix"] = kwargs.pop("filename")
    return plotter.plot_information_metrics(results, **kwargs)


def plot_per_layer_information(layer_results: dict[str, dict[str, Any]], **kwargs):
    """Convenience function for per-layer information plotting."""
    plotter = InformationAnalysisPlotter()
    return plotter.plot_per_layer_information(layer_results, **kwargs)
