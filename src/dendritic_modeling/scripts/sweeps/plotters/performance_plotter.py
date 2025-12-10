"""Performance metric plotters."""

import logging
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.metrics import get_metric_label
from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


class PerformancePlotter(BasePlotter):
    """Generates performance-related plots."""

    PERFORMANCE_METRICS: ClassVar[list[str]] = [
        "test_accuracy",
        "valid_accuracy",
        "train_accuracy",
        "test_auc",
        "test_categorical_loglikelihood",
    ]

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all performance plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        plots_dir = output_dir / "performance"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Generate E/I ratio plots if applicable
        if "ei_ratio" in data.columns:
            plot_paths.extend(self.plot_metrics_vs_ei_ratio(data, plots_dir))
            plot_paths.extend(self.plot_performance_heatmaps(data, plots_dir))

        # Generate parameter scaling plots if applicable
        if "nparams" in data.columns:
            plot_paths.extend(self.plot_metrics_vs_nparams(data, plots_dir))

        return plot_paths

    def plot_metrics_vs_ei_ratio(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """
        Generate metric vs E/I ratio line plots with error bars.

        Args:
            data: Aggregated data with ei_ratio column
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        for metric in self.PERFORMANCE_METRICS:
            # Handle both aggregated (_mean) and non-aggregated (no suffix) data
            mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else metric

            if mean_col not in data.columns:
                continue

            try:
                fig, ax = plt.subplots(figsize=self.default_figsize)

                # Plot by network category
                self.plot_by_category(
                    ax=ax,
                    data=data,
                    x_col="ei_ratio",
                    y_col=metric,
                    category_col="network_category",
                    show_error=True,
                )

                # Style axes
                self.setup_axes_style(
                    ax=ax,
                    xlabel="E/I Ratio",
                    ylabel=get_metric_label(metric),
                    title=f"{get_metric_label(metric)} vs E/I Ratio",
                    use_log_x=True,
                    grid=True,
                )

                ax.legend(fontsize=10)
                fig.tight_layout()

                plot_path = output_dir / f"{metric}_vs_ei_ratio.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {metric} vs E/I plot: {e}")

        logger.info(f"Generated {len(plot_paths)} E/I ratio plots")
        return plot_paths

    def plot_performance_heatmaps(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """
        Generate performance heatmaps (metric vs EE vs IE).

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        if "ee_value" not in data.columns or "ie_value" not in data.columns:
            return plot_paths

        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        for metric in ["test_accuracy", "valid_accuracy"]:
            # Handle both aggregated (_mean) and non-aggregated (no suffix) data
            mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else metric

            if mean_col not in data.columns:
                continue

            try:
                # Create pivot table
                pivot = data.pivot_table(
                    values=mean_col,
                    index="ie_value",
                    columns="ee_value",
                    aggfunc="mean",
                )

                if pivot.empty:
                    continue

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    pivot,
                    cmap="viridis",
                    annot=True,
                    fmt=".3f",
                    cbar_kws={"label": get_metric_label(metric)},
                    ax=ax,
                )

                ax.set_xlabel("EE Synapses per Branch", fontsize=12)
                ax.set_ylabel("IE Synapses per Branch", fontsize=12)
                ax.set_title(f"{get_metric_label(metric)} Heatmap", fontsize=14)

                fig.tight_layout()
                plot_path = heatmap_dir / f"{metric}_heatmap.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {metric} heatmap: {e}")

        logger.info(f"Generated {len(plot_paths)} performance heatmaps")
        return plot_paths

    def plot_metrics_vs_nparams(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """
        Generate metric vs parameter count plots.

        Args:
            data: Aggregated data with nparams column
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        param_dir = output_dir / "parameter_scaling"
        param_dir.mkdir(parents=True, exist_ok=True)

        for metric in self.PERFORMANCE_METRICS:
            # Handle both aggregated (_mean) and non-aggregated (no suffix) data
            mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else metric
            std_col = f"{metric}_std" if f"{metric}_std" in data.columns else None

            if mean_col not in data.columns:
                continue

            try:
                fig, ax = plt.subplots(figsize=self.default_figsize)

                if "network_category" not in data.columns:
                    data["network_category"] = "default"

                categories = data["network_category"].unique()

                for category in categories:
                    cat_data = data[data["network_category"] == category].sort_values(
                        "nparams"
                    )

                    if cat_data.empty:
                        continue

                    color = self.category_colors.get(category, "black")
                    marker = self.category_markers.get(category, "o")

                    y_values = cat_data[mean_col]
                    yerr = cat_data[std_col] if std_col in cat_data.columns else None

                    ax.plot(
                        cat_data["nparams"],
                        y_values,
                        marker=marker,
                        color=color,
                        linestyle="-",
                        alpha=0.8,
                        label=category,
                    )

                    if yerr is not None:
                        ax.fill_between(
                            cat_data["nparams"],
                            y_values - yerr,
                            y_values + yerr,
                            color=color,
                            alpha=0.2,
                        )

                self.setup_axes_style(
                    ax=ax,
                    xlabel="Parameter Count",
                    ylabel=get_metric_label(metric),
                    title=f"{get_metric_label(metric)} vs Parameter Count",
                    use_log_x=True,
                    grid=True,
                )

                if len(categories) > 1:
                    ax.legend(fontsize=10)

                fig.tight_layout()
                plot_path = param_dir / f"{metric}_vs_nparams.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {metric} vs nparams plot: {e}")

        logger.info(f"Generated {len(plot_paths)} parameter scaling plots")
        return plot_paths
