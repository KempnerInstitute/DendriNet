"""Information theory plotters (inspired by comprehensive_analysis.py)."""

import logging
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.metrics import get_metric_label
from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


class InformationPlotter(BasePlotter):
    """Generates information theory plots."""

    # Metric groups for organized plotting
    METRIC_GROUPS: ClassVar[dict[str, dict]] = {
        "class_information": {
            "metrics": ["mi_E_C", "mi_I_C", "mi_V_C"],
            "labels": ["I(E;C)", "I(I;C)", "I(V;C)"],
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],  # Red, Teal, Blue
            "title": "Information with Class Labels",
            "ylabel": "Mutual Information (bits)",
        },
        "component_information": {
            "metrics": ["mi_E_I", "mi_E_V", "mi_I_V"],
            "labels": ["I(E;I)", "I(E;V)", "I(I;V)"],
            "colors": ["#8B5A96", "#FFA726", "#9C27B0"],  # Purple, Orange, Purple
            "title": "Component Information",
            "ylabel": "Mutual Information (bits)",
        },
        "conditional_information": {
            "metrics": ["mi_E_I_given_C", "mi_E_V_given_C", "mi_I_V_given_C"],
            "labels": ["I(E;I|C)", "I(E;V|C)", "I(I;V|C)"],
            "colors": [
                "#2E7D32",
                "#558B2F",
                "#827717",
            ],  # Dark green, Forest green, Olive
            "title": "Conditional Information",
            "ylabel": "Mutual Information (bits)",
        },
    }

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all information theory plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        info_dir = output_dir / "information"
        info_dir.mkdir(parents=True, exist_ok=True)

        # Check if we have any information metrics
        has_mi = any(col for col in data.columns if "mi_" in col)
        if not has_mi:
            logger.info("No information metrics found, skipping information plots")
            return plot_paths

        # Generate comprehensive multi-panel plot
        if "ei_ratio" in data.columns:
            panel_path = self.plot_information_panels(data, info_dir)
            if panel_path:
                plot_paths.append(panel_path)

        # Generate individual metric group plots
        plot_paths.extend(self.plot_metric_groups(data, info_dir))

        # Generate layer-wise information plots
        plot_paths.extend(self.plot_layerwise_information(data, info_dir))

        return plot_paths

    def plot_information_panels(self, data: pd.DataFrame, output_dir: Path) -> Path:
        """
        Generate comprehensive multi-panel information plot.

        This creates a 2x3 grid showing:
        - Row 1: Class information, Component information, Conditional information
        - Row 2: Layer-wise I(V;C), Shunting advantage, Information variance

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            Path to saved plot
        """
        try:
            fig, axes = self.create_multi_panel_figure(
                nrows=2,
                ncols=3,
                figsize=(18, 12),
                title="Comprehensive Information Analysis",
            )
            axes = axes.flatten()

            # Panel 1-3: Metric groups
            for idx, (_group_name, group_info) in enumerate(self.METRIC_GROUPS.items()):
                if idx < 3:
                    self._plot_metric_group_panel(data, axes[idx], group_info)

            # Panel 4: Layer-wise I(V;C)
            self._plot_layerwise_ivc_panel(data, axes[3])

            # Panel 5: Shunting advantage
            self._plot_shunting_advantage_panel(data, axes[4])

            # Panel 6: Information summary statistics
            self._plot_information_summary_panel(data, axes[5])

            fig.tight_layout()
            plot_path = output_dir / "information_panels.png"
            return self.save_figure(fig, plot_path)

        except Exception as e:
            logger.error(f"Error generating information panels: {e}")
            return None

    def _plot_metric_group_panel(
        self, data: pd.DataFrame, ax: plt.Axes, group_info: dict
    ):
        """Plot a single metric group panel."""
        metrics = group_info["metrics"]
        labels = group_info["labels"]
        colors = group_info["colors"]

        # Check which metrics are available
        available = [
            (m, lbl, c)
            for m, lbl, c in zip(metrics, labels, colors)
            if f"{m}_mean" in data.columns
        ]

        if not available:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(group_info["title"])
            return

        # Plot each metric
        for metric, lbl, color in available:
            label = lbl
            mean_col = (
                f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"
            )

            # Group by ei_ratio and plot
            if "ei_ratio" in data.columns:
                grouped = data.groupby("ei_ratio")[mean_col].mean().sort_index()
                ax.plot(
                    grouped.index,
                    grouped.values,
                    marker="o",
                    color=color,
                    label=label,
                    linewidth=2,
                )

        self.setup_axes_style(
            ax=ax,
            xlabel="E/I Ratio",
            ylabel=group_info["ylabel"],
            title=group_info["title"],
            use_log_x=True,
            grid=True,
        )
        ax.legend(fontsize=9)

    def _plot_layerwise_ivc_panel(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot layer-wise I(V;C) across depths."""
        # Find layer MI columns
        layer_cols = [
            col
            for col in data.columns
            if "layer_mi_V_C_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            ax.text(
                0.5,
                0.5,
                "No layer data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Layer-wise I(V;C)")
            return

        # Extract depths
        depths = []
        for col in layer_cols:
            try:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                depths.append(depth)
            except (ValueError, IndexError):
                pass

        depths = sorted(set(depths))

        if not depths:
            ax.text(
                0.5,
                0.5,
                "No depth data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Plot I(V;C) for each depth
        for depth in depths:
            col = f"layer_mi_V_C_depth{depth}_mean"
            if col in data.columns:
                if "ei_ratio" in data.columns:
                    grouped = data.groupby("ei_ratio")[col].mean().sort_index()
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        marker="o",
                        label=f"Layer {depth}",
                        linewidth=2,
                    )

        self.setup_axes_style(
            ax=ax,
            xlabel="E/I Ratio",
            ylabel="I(V;C) (bits)",
            title="Layer-wise Information with Class",
            use_log_x=True,
            grid=True,
        )
        ax.legend(fontsize=8, ncol=2)

    def _plot_shunting_advantage_panel(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot shunting advantage (difference in metrics)."""
        metric = "test_accuracy"
        mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"

        if mean_col not in data.columns or "network_category" not in data.columns:
            ax.text(
                0.5,
                0.5,
                "No shunting data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Shunting Advantage")
            return

        # Calculate difference
        shunting_data = data[data["network_category"] == "shunting"]
        no_shunting_data = data[data["network_category"] == "no_shunting"]

        if shunting_data.empty or no_shunting_data.empty:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Merge on ei_ratio
        merged = pd.merge(
            shunting_data[["ei_ratio", mean_col]],
            no_shunting_data[["ei_ratio", mean_col]],
            on="ei_ratio",
            suffixes=("_shunt", "_no_shunt"),
        )

        merged["advantage"] = (
            merged[f"{mean_col}_shunt"] - merged[f"{mean_col}_no_shunt"]
        )

        ax.plot(
            merged["ei_ratio"],
            merged["advantage"],
            marker="o",
            color="#2E7D32",
            linewidth=2,
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        self.setup_axes_style(
            ax=ax,
            xlabel="E/I Ratio",
            ylabel="Accuracy Difference (Shunting - No Shunting)",
            title="Shunting Advantage",
            use_log_x=True,
            grid=True,
        )

    def _plot_information_summary_panel(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot summary statistics of information metrics."""
        # Collect all MI metrics
        mi_metrics = [
            col.replace("_mean", "")
            for col in data.columns
            if "mi_" in col and col.endswith("_mean")
        ]

        if not mi_metrics:
            ax.text(
                0.5,
                0.5,
                "No MI metrics",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Information Summary")
            return

        # Compute mean and std across all ei_ratios for each metric
        means = []
        stds = []
        labels = []

        for metric in mi_metrics[:10]:  # Limit to avoid overcrowding
            mean_col = (
                f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"
            )
            if mean_col in data.columns:
                values = data[mean_col].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    stds.append(values.std())
                    labels.append(get_metric_label(metric))

        if means:
            x_pos = np.arange(len(labels))
            ax.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7, color="#45B7D1")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Mean MI (bits)", fontsize=10)
            ax.set_title("Information Metric Summary", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

    def plot_metric_groups(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate individual plots for each metric group.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        if "ei_ratio" not in data.columns or "network_category" not in data.columns:
            return plot_paths

        for group_name, group_info in self.METRIC_GROUPS.items():
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Plot for shunting and no_shunting separately
                for use_shunting, ax, title_suffix in [
                    (True, ax1, "Shunting"),
                    (False, ax2, "No Shunting"),
                ]:
                    category = "shunting" if use_shunting else "no_shunting"
                    cat_data = data[data["network_category"] == category]

                    if cat_data.empty:
                        ax.text(
                            0.5,
                            0.5,
                            f"No {category} data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"{group_info['title']} - {title_suffix}")
                        continue

                    # Plot each metric
                    for metric, label, color in zip(
                        group_info["metrics"],
                        group_info["labels"],
                        group_info["colors"],
                    ):
                        mean_col = (
                            f"{metric}_mean"
                            if f"{metric}_mean" in data.columns
                            else "metric"
                        )
                        if mean_col in cat_data.columns:
                            grouped = (
                                cat_data.groupby("ei_ratio")[mean_col]
                                .mean()
                                .sort_index()
                            )
                            ax.plot(
                                grouped.index,
                                grouped.values,
                                marker="o",
                                color=color,
                                label=label,
                                linewidth=2,
                                markersize=6,
                            )

                    self.setup_axes_style(
                        ax=ax,
                        xlabel="E/I Ratio",
                        ylabel=group_info["ylabel"],
                        title=f"{group_info['title']} - {title_suffix}",
                        use_log_x=True,
                        grid=True,
                    )
                    ax.legend(fontsize=10)

                fig.tight_layout()
                plot_path = output_dir / f"{group_name}.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {group_name} plot: {e}")

        logger.info(f"Generated {len(plot_paths)} metric group plots")
        return plot_paths

    def plot_layerwise_information(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """
        Generate layer-wise information plots and heatmaps.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        layer_dir = output_dir / "layer_information"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Generate I(V;C) heatmap by layer
        heatmap_path = self._plot_ivc_heatmap_by_layer(data, layer_dir)
        if heatmap_path:
            plot_paths.append(heatmap_path)

        # Generate first vs last layer comparison
        comparison_paths = self._plot_first_last_layer_comparison(data, layer_dir)
        plot_paths.extend(comparison_paths)

        logger.info(f"Generated {len(plot_paths)} layer-wise information plots")
        return plot_paths

    def _plot_ivc_heatmap_by_layer(self, data: pd.DataFrame, output_dir: Path) -> Path:
        """Generate I(V;C) heatmap across layers and E/I ratios."""
        # Find layer MI columns for I(V;C)
        ivc_cols = [
            col
            for col in data.columns
            if "layer_mi_V_C_depth" in col and col.endswith("_mean")
        ]

        if not ivc_cols:
            logger.info("No layer I(V;C) data for heatmap")
            return None

        try:
            # Extract layer data
            layer_data = []
            for col in ivc_cols:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                for _, row in data.iterrows():
                    if not pd.isna(row[col]) and "ei_ratio" in row:
                        layer_data.append(
                            {
                                "layer": depth,
                                "ei_ratio": row["ei_ratio"],
                                "use_shunting": row.get("network_category")
                                == "shunting",
                                "ivc_value": row[col],
                            }
                        )

            if not layer_data:
                return None

            layer_df = pd.DataFrame(layer_data)

            # Create dual heatmaps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Calculate global vmin/vmax for consistent scaling
            vmin, vmax = layer_df["ivc_value"].min(), layer_df["ivc_value"].max()

            for use_shunting, ax, title in [
                (True, ax1, "Shunting"),
                (False, ax2, "Non-Shunting"),
            ]:
                subset = layer_df[layer_df["use_shunting"] == use_shunting]

                if subset.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {title} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"I(V;C) Heatmap - {title}")
                    continue

                pivot = subset.pivot_table(
                    values="ivc_value",
                    index="layer",
                    columns="ei_ratio",
                    aggfunc="mean",
                )

                sns.heatmap(
                    pivot,
                    cmap="viridis",
                    annot=True,
                    fmt=".2f",
                    cbar_kws={"label": "I(V;C) (bits)"},
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                )

                ax.set_xlabel("E/I Ratio", fontsize=12)
                ax.set_ylabel("Layer Index", fontsize=12)
                ax.set_title(f"I(V;C) Heatmap - {title}", fontsize=14)

            fig.suptitle("Information with Class by Layer and E/I Ratio", fontsize=16)
            fig.tight_layout()

            plot_path = output_dir / "ivc_heatmap_by_layer.png"
            return self.save_figure(fig, plot_path)

        except Exception as e:
            logger.error(f"Error generating I(V;C) heatmap: {e}")
            return None

    def _plot_first_last_layer_comparison(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Generate first vs last layer comparison plots."""
        plot_paths = []

        # Find all layer MI columns
        layer_cols = [
            col
            for col in data.columns
            if "layer_mi_" in col and "_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            return plot_paths

        # Extract depths
        depths = set()
        for col in layer_cols:
            try:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                depths.add(depth)
            except (ValueError, IndexError):
                pass

        depths = sorted(depths)
        if len(depths) < 2:
            logger.info(f"Not enough layers ({len(depths)}) for first/last comparison")
            return plot_paths

        first_layer, last_layer = depths[0], depths[-1]
        logger.info(
            f"Comparing layer {first_layer} (first) vs layer {last_layer} (last)"
        )

        # Generate comparison plots for each metric group
        for group_name, group_info in self.METRIC_GROUPS.items():
            try:
                plot_path = self._plot_metric_group_first_last(
                    data, first_layer, last_layer, group_name, group_info, output_dir
                )
                if plot_path:
                    plot_paths.append(plot_path)
            except Exception as e:
                logger.error(
                    f"Error generating first/last comparison for {group_name}: {e}"
                )

        return plot_paths

    def _plot_metric_group_first_last(
        self,
        data: pd.DataFrame,
        first_layer: int,
        last_layer: int,
        group_name: str,
        group_info: dict,
        output_dir: Path,
    ) -> Path:
        """Plot first vs last layer comparison for a metric group."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for layer_idx, ax, title in [
            (first_layer, ax1, f"Layer {first_layer} (First)"),
            (last_layer, ax2, f"Layer {last_layer} (Last)"),
        ]:
            # Plot each metric for both shunting conditions
            for metric, label, color in zip(
                group_info["metrics"],
                group_info["labels"],
                group_info["colors"],
            ):
                # Get base metric name (e.g., "E_C" from "mi_E_C")
                base_metric = metric.replace("mi_", "")
                layer_col = f"layer_mi_{base_metric}_depth{layer_idx}_mean"

                if layer_col not in data.columns:
                    continue

                # Plot for both shunting conditions
                for use_shunting, linestyle, alpha in [
                    (True, "-", 0.9),
                    (False, "--", 0.6),
                ]:
                    category = "shunting" if use_shunting else "no_shunting"
                    cat_data = data[data["network_category"] == category]

                    if cat_data.empty or "ei_ratio" not in cat_data.columns:
                        continue

                    grouped = (
                        cat_data.groupby("ei_ratio")[layer_col].mean().sort_index()
                    )
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        color=color,
                        linestyle=linestyle,
                        label=f"{label} ({'S' if use_shunting else 'NS'})",
                        linewidth=2,
                        marker="o" if use_shunting else "s",
                        markersize=5,
                        alpha=alpha,
                    )

            self.setup_axes_style(
                ax=ax,
                xlabel="E/I Ratio",
                ylabel=group_info["ylabel"],
                title=title,
                use_log_x=True,
                grid=True,
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        fig.suptitle(f"{group_info['title']} - First vs Last Layer", fontsize=16)
        fig.tight_layout()

        plot_path = output_dir / f"{group_name}_first_last_layer.png"
        return self.save_figure(fig, plot_path)

    def _plot_shunting_advantage_panel(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot shunting advantage across metrics."""
        # Calculate shunting advantage for test accuracy
        self._plot_shunting_advantage_for_metric(data, ax, "test_accuracy")

    def _plot_shunting_advantage_for_metric(
        self, data: pd.DataFrame, ax: plt.Axes, metric: str
    ):
        """Plot shunting advantage for a specific metric."""
        mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"

        if mean_col not in data.columns or "network_category" not in data.columns:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Shunting Advantage")
            return

        # Separate shunting and no_shunting data
        shunting = data[data["network_category"] == "shunting"]
        no_shunting = data[data["network_category"] == "no_shunting"]

        if shunting.empty or no_shunting.empty or "ei_ratio" not in data.columns:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Merge on ei_ratio
        merged = pd.merge(
            shunting[["ei_ratio", mean_col]],
            no_shunting[["ei_ratio", mean_col]],
            on="ei_ratio",
            suffixes=("_shunt", "_no_shunt"),
        )

        merged["advantage"] = (
            merged[f"{mean_col}_shunt"] - merged[f"{mean_col}_no_shunt"]
        )
        merged = merged.sort_values("ei_ratio")

        ax.plot(
            merged["ei_ratio"],
            merged["advantage"],
            marker="o",
            color="#2E7D32",
            linewidth=2,
            markersize=6,
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.fill_between(
            merged["ei_ratio"], 0, merged["advantage"], alpha=0.3, color="#2E7D32"
        )

        self.setup_axes_style(
            ax=ax,
            xlabel="E/I Ratio",
            ylabel=f"{get_metric_label(metric)} Difference",
            title="Shunting Advantage (Shunting - Non-Shunting)",
            use_log_x=True,
            grid=True,
        )
