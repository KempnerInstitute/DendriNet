"""Information theory plotters (inspired by comprehensive_analysis.py)."""

import logging
import traceback
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
            "metrics": ["mi_E_C", "mi_I_C", "mi_Vb_C", "mi_V_C"],
            "labels": ["I(E;C)", "I(I;C)", "I(Vb;C)", "I(V;C)"],
            "colors": [
                "#FF6B6B",
                "#4ECDC4",
                "#FFB347",
                "#45B7D1",
            ],  # Red, Teal, Orange, Blue
            "title": "Information with Class Labels",
            "ylabel": "Mutual Information (bits)",
        },
        "component_information": {
            "metrics": ["mi_E_I", "mi_E_Vb", "mi_I_Vb", "mi_E_V", "mi_I_V", "mi_Vb_V"],
            "labels": ["I(E;I)", "I(E;Vb)", "I(I;Vb)", "I(E;V)", "I(I;V)", "I(Vb;V)"],
            "colors": [
                "#8B5A96",
                "#E91E63",
                "#9C27B0",
                "#FFA726",
                "#FF9800",
                "#FF5722",
            ],  # Purple, Pink, Purple, Orange, Orange, Red-orange
            "title": "Component Information",
            "ylabel": "Mutual Information (bits)",
        },
        "conditional_information": {
            "metrics": [
                "mi_E_I_given_C",
                "mi_E_Vb_given_C",
                "mi_I_Vb_given_C",
                "mi_E_V_given_C",
                "mi_I_V_given_C",
                "mi_Vb_V_given_C",
            ],
            "labels": [
                "I(E;I|C)",
                "I(E;Vb|C)",
                "I(I;Vb|C)",
                "I(E;V|C)",
                "I(I;V|C)",
                "I(Vb;V|C)",
            ],
            "colors": [
                "#2E7D32",
                "#388E3C",
                "#558B2F",
                "#689F38",
                "#7CB342",
                "#827717",
            ],  # Dark green through olive gradient
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

        sweep_types = self.detect_sweep_types(data)
        if "ei" in sweep_types:
            try:
                plot_paths.extend(self._create_information_ei_plots(data, info_dir))
            except Exception as e:
                logger.error(f"Error generating E/I information plots: {e}")
                traceback.print_exc()

        if "stoch1" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_heatmaps(
                        data, info_dir, "stimulus_duration", "max_gain_factor"
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error generating stochastic dataset information plots: {e}"
                )
                traceback.print_exc()

        if "stoch2" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_heatmaps(
                        data, info_dir, "stimulus_duration", "max_gain_tau_ratio"
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error generating stochastic GV dataset information plots: {e}"
                )
                traceback.print_exc()

        return plot_paths

    def _create_information_ei_plots(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        plot_paths = []
        # Generate comprehensive multi-panel plot
        panel_path = self.plot_information_panels_ei(data, output_dir)
        if panel_path:
            plot_paths.append(panel_path)

        # Generate individual metric group plots
        plot_paths.extend(self.plot_metric_groups_ei(data, output_dir))

        # Generate layer-wise information plots
        plot_paths.extend(self.plot_layerwise_information_ei(data, output_dir))

        # Generate legacy information plots
        plot_paths.extend(self._generate_information_plots_ei(data, output_dir))

        return plot_paths

    def plot_information_panels_ei(self, data: pd.DataFrame, output_dir: Path) -> Path:
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

    def plot_metric_groups_ei(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
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

    def plot_layerwise_information_ei(
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
                # Invert y-axis to place origin at lower left
                ax.invert_yaxis()

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

    def _generate_information_plots_ei(
        self, valid_data: pd.DataFrame, plots_dir: Path
    ) -> list[Path]:
        """Generate information theory plots matching old analysis format"""
        plot_paths = []

        if "ei_ratio" not in valid_data.columns:
            return plot_paths

        info_plot_dir = plots_dir / "information_theory_plots"

        # Collect global and layer-specific MI metrics
        global_mi_metrics = []
        layer_mi_metrics = []

        for col in valid_data.columns:
            if any(
                mi_pattern in col.lower()
                for mi_pattern in ["mi_", "mutual_information"]
            ):
                # Extract the metric name
                if "_mean" in col:
                    metric_name = col.replace("_mean", "")
                    if "layer_mi_" in metric_name:
                        # This is a layer-specific metric
                        base_metric = metric_name.replace("layer_mi_", "").split(
                            "_depth"
                        )[0]
                        if base_metric not in layer_mi_metrics:
                            layer_mi_metrics.append(base_metric)
                    else:
                        # This is a global metric
                        if metric_name not in global_mi_metrics:
                            global_mi_metrics.append(metric_name)

        if not global_mi_metrics and not layer_mi_metrics:
            logger.info("No mutual information metrics found")
            return plot_paths

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            logger.info("No shunting/no_shunting data for information analysis")
            return plot_paths

        # Define metric groups matching old analysis exactly
        metric_groups = {
            "class_information": {
                "metrics": ["mi_E_C", "mi_I_C", "mi_Vb_C", "mi_V_C"],
                "labels": ["I(E;C)", "I(I;C)", "I(Vb;C)", "I(V;C)"],
                "colors": [
                    self.info_colors["excitatory"],
                    self.info_colors["inhibitory"],
                    "#FFB347",  # Orange for Vb
                    self.info_colors["combined"],
                ],
                "title": "Information with Class Labels",
                "ylabel": "Mutual Information (bits)",
            },
            "component_information": {
                "metrics": [
                    "mi_E_I",
                    "mi_E_Vb",
                    "mi_I_Vb",
                    "mi_E_V",
                    "mi_I_V",
                    "mi_Vb_V",
                ],
                "labels": [
                    "I(E;I)",
                    "I(E;Vb)",
                    "I(I;Vb)",
                    "I(E;V)",
                    "I(I;V)",
                    "I(Vb;V)",
                ],
                "colors": [
                    "purple",
                    "#E91E63",
                    "#9C27B0",
                    "orange",
                    "#FF9800",
                    "brown",
                ],
                "title": "Component Information",
                "ylabel": "Mutual Information (bits)",
            },
            "conditional_information": {
                "metrics": [
                    "mi_E_I_given_C",
                    "mi_E_Vb_given_C",
                    "mi_I_Vb_given_C",
                    "mi_E_V_given_C",
                    "mi_I_V_given_C",
                    "mi_Vb_V_given_C",
                ],
                "labels": [
                    "I(E;I|C)",
                    "I(E;Vb|C)",
                    "I(I;Vb|C)",
                    "I(E;V|C)",
                    "I(I;V|C)",
                    "I(Vb;V|C)",
                ],
                "colors": [
                    "darkgreen",
                    "#388E3C",
                    "forestgreen",
                    "#689F38",
                    "olive",
                    "#827717",
                ],
                "title": "Conditional Information",
                "ylabel": "Mutual Information (bits)",
            },
        }

        # Generate global information plots grouped by metric type (matching old format)
        for group_name, group_info in metric_groups.items():
            metrics = group_info["metrics"]
            labels = group_info["labels"]
            metric_colors = group_info["colors"]
            group_title = group_info["title"]
            ylabel = group_info["ylabel"]

            # Check which metrics are available
            available_metrics = []
            available_labels = []
            available_colors = []

            for metric, label, color in zip(metrics, labels, metric_colors):
                mean_col = f"{metric}_mean"
                if mean_col in valid_data.columns:
                    available_metrics.append(metric)
                    available_labels.append(label)
                    available_colors.append(color)

            if not available_metrics:
                logger.info(f"No {group_name} metrics found")
                continue

            try:
                # Create separate plots for shunting vs no shunting (matching old analysis)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # First pass: collect all y-values to determine global y-axis range
                all_y_values = []
                plot_data = {}

                for use_shunting in [True, False]:
                    category = "shunting" if use_shunting else "no_shunting"
                    shunt_data = valid_data[valid_data["network_category"] == category]
                    plot_data[use_shunting] = {}

                    if not shunt_data.empty:
                        for metric in available_metrics:
                            mean_col = f"{metric}_mean"
                            # Group by E/I ratio and calculate statistics
                            grouped = shunt_data.groupby("ei_ratio")[mean_col].mean()
                            plot_data[use_shunting][metric] = grouped
                            all_y_values.extend(grouped.values)

                # Calculate global y-axis range
                if all_y_values:
                    global_y_min = min(all_y_values)
                    global_y_max = max(all_y_values)
                    y_margin = (global_y_max - global_y_min) * 0.1
                    final_y_min = global_y_min - y_margin
                    final_y_max = global_y_max + y_margin
                else:
                    final_y_min, final_y_max = 0, 1

                # Second pass: create the actual plots
                for use_shunting, ax, title_suffix in [
                    (True, ax1, "Shunting"),
                    (False, ax2, "No Shunting"),
                ]:
                    if use_shunting not in plot_data or not plot_data[use_shunting]:
                        ax.text(
                            0.5,
                            0.5,
                            f"No data for\\n{title_suffix}",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                        )
                        ax.set_title(f"{group_title}\\n{title_suffix} Networks")
                        ax.set_ylim(final_y_min, final_y_max)
                        continue

                    # Plot each metric
                    for metric, label, color in zip(
                        available_metrics, available_labels, available_colors
                    ):
                        if metric in plot_data[use_shunting]:
                            grouped = plot_data[use_shunting][metric]
                            ax.plot(
                                grouped.index,
                                grouped.values,
                                marker="o",
                                color=color,
                                label=label,
                                linewidth=2,
                                markersize=6,
                            )

                    ax.legend(fontsize=10)
                    ax.set_ylim(final_y_min, final_y_max)

                    self.setup_axes_style(
                        ax=ax,
                        xlabel="E/I Ratio",
                        ylabel=ylabel,
                        title=f"{group_title}\\n{title_suffix} Networks",
                        use_log_x=True,
                        grid=True,
                    )

                plt.suptitle(f"Branch Information Analysis: {group_title}", fontsize=16)
                plt.tight_layout()

                plot_path = info_plot_dir / f"{group_name}_branch_info.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {group_name} plot: {e}")

        # # Generate layer-specific information plots and heatmaps
        # layer_plot_dir = os.path.join(plots_dir, "layer_information_plots")
        # os.makedirs(layer_plot_dir, exist_ok=True)

        # # Create I(V;C) heatmap by layer (matching old analysis)
        # plot_paths.extend(
        #     self._create_ivc_heatmap_by_layer(valid_data, layer_plot_dir, base_name)
        # )

        # # Create first vs last layer comparison plots
        # plot_paths.extend(
        #     self._create_first_last_layer_plots(valid_data, layer_plot_dir, base_name)
        # )

        logger.info(
            f"Created {len(plot_paths)} information theory plots (global + layer-specific + heatmaps)"
        )
        return plot_paths

    def _create_information_heatmaps(
        self, data: pd.DataFrame, output_dir: Path, colx: str, coly: str
    ) -> list[Path]:
        plot_paths = []

        # Generate aggregate heatmaps for each metric group
        plot_paths.extend(
            self.plot_metric_groups_aggregate_heatmaps(data, output_dir, colx, coly)
        )

        # Generate layerwise heatmaps for each metric
        plot_paths.extend(
            self.plot_metric_layerwise_heatmaps(data, output_dir, colx, coly)
        )

        return plot_paths

    def plot_metric_groups_aggregate_heatmaps(
        self, data: pd.DataFrame, output_dir: Path, colx: str, coly: str
    ) -> list[Path]:
        """
        Generate heatmap plots for each metric group.

        Creates one figure per metric group with:
        - 2 columns (shunting vs non-shunting)
        - N rows where N = number of metrics in the group

        Each subplot shows a heatmap with colx on x-axis and coly on y-axis.

        Args:
            data: Aggregated data
            output_dir: Output directory
            colx: Column name for x-axis
            coly: Column name for y-axis

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Validate required columns
        if colx not in data.columns or coly not in data.columns:
            logger.warning(f"Missing required columns: {colx} or {coly}")
            return plot_paths

        if "network_category" not in data.columns:
            logger.warning("Missing 'network_category' column")
            return plot_paths

        # Generate one figure per metric group
        for group_name, group_info in self.METRIC_GROUPS.items():
            try:
                metrics = group_info["metrics"]
                labels = group_info["labels"]

                # Find available metrics
                available_metrics = []
                available_labels = []

                for metric, label in zip(metrics, labels):
                    mean_col = f"{metric}_mean"
                    if mean_col in data.columns:
                        available_metrics.append(metric)
                        available_labels.append(label)

                if not available_metrics:
                    logger.info(f"No available metrics for {group_name}, skipping")
                    continue

                # Create figure with nrows = number of metrics, ncols = 2
                nrows = len(available_metrics)
                fig, axes = plt.subplots(
                    nrows, 2, figsize=(18, 6 * nrows), sharex=True, sharey=True
                )

                # Handle single row case
                if nrows == 1:
                    axes = axes.reshape(1, -1)

                # Collect all values for consistent color scaling
                all_values = []
                plot_data = {}

                for use_shunting in [True, False]:
                    category = "shunting" if use_shunting else "no_shunting"
                    cat_data = data[data["network_category"] == category]
                    plot_data[use_shunting] = {}

                    if not cat_data.empty:
                        for metric in available_metrics:
                            mean_col = f"{metric}_mean"
                            values = cat_data[mean_col].dropna()
                            all_values.extend(values.tolist())

                # Calculate global vmin/vmax for consistent scaling
                if all_values:
                    vmin, vmax = min(all_values), max(all_values)
                else:
                    vmin, vmax = 0, 1

                # Create heatmaps for each metric and shunting condition
                for metric_idx, (metric, label) in enumerate(
                    zip(available_metrics, available_labels)
                ):
                    mean_col = f"{metric}_mean"

                    for col_idx, (use_shunting, title_suffix) in enumerate(
                        [
                            (True, "Shunting"),
                            (False, "Non-Shunting"),
                        ]
                    ):
                        ax = axes[metric_idx, col_idx]
                        category = "shunting" if use_shunting else "no_shunting"
                        cat_data = data[data["network_category"] == category]

                        if cat_data.empty or mean_col not in cat_data.columns:
                            ax.text(
                                0.5,
                                0.5,
                                f"No {category} data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"{label} - {title_suffix}")
                            continue

                        # Create pivot table for heatmap
                        try:
                            pivot = cat_data.pivot_table(
                                values=mean_col,
                                index=coly,
                                columns=colx,
                                aggfunc="mean",
                            )

                            if pivot.empty:
                                ax.text(
                                    0.5,
                                    0.5,
                                    "No data to pivot",
                                    ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                )
                                ax.set_title(f"{label} - {title_suffix}")
                                continue

                            # Create heatmap
                            sns.heatmap(
                                pivot,
                                cmap="viridis",
                                annot=True,
                                fmt=".3f",
                                cbar_kws={"label": group_info["ylabel"]},
                                ax=ax,
                                vmin=vmin,
                                vmax=vmax,
                            )
                            # Invert y-axis to place origin at lower left
                            ax.invert_yaxis()

                            ax.set_xlabel(colx, fontsize=12)
                            if col_idx == 0:  # Only label y-axis on left column
                                ax.set_ylabel(coly, fontsize=12)
                            else:
                                ax.set_ylabel("")
                            ax.set_title(f"{label} - {title_suffix}", fontsize=12)

                        except Exception as e:
                            logger.error(
                                f"Error creating heatmap for {metric} ({title_suffix}): {e}"
                            )
                            ax.text(
                                0.5,
                                0.5,
                                f"Error: {str(e)[:50]}",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"{label} - {title_suffix}")

                fig.suptitle(f"{group_info['title']} Heatmaps", fontsize=16, y=0.995)
                fig.tight_layout(rect=[0, 0, 1, 0.98])

                plot_path = output_dir / f"{group_name}_heatmaps.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {group_name} heatmaps: {e}")
                traceback.print_exc()

        logger.info(f"Generated {len(plot_paths)} metric group heatmap figures")
        return plot_paths

    def plot_metric_layerwise_heatmaps(
        self, data: pd.DataFrame, output_dir: Path, colx: str, coly: str
    ) -> list[Path]:
        """
        Generate layerwise heatmap plots for each metric.

        Creates one figure per metric with:
        - 2 columns (shunting vs non-shunting)
        - N rows where N = number of depths/layers found in the data

        Each subplot shows a heatmap with colx on x-axis and coly on y-axis.

        Args:
            data: Aggregated data
            output_dir: Output directory
            colx: Column name for x-axis
            coly: Column name for y-axis

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Validate required columns
        if colx not in data.columns or coly not in data.columns:
            logger.warning(f"Missing required columns: {colx} or {coly}")
            return plot_paths

        if "network_category" not in data.columns:
            logger.warning("Missing 'network_category' column")
            return plot_paths

        # Find all layer MI columns
        layer_cols = [
            col
            for col in data.columns
            if "layer_mi_" in col and "_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            logger.info("No layer MI columns found for heatmaps")
            return plot_paths

        # Extract all unique depths
        depths = set()
        for col in layer_cols:
            try:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                depths.add(depth)
            except (ValueError, IndexError):
                pass

        depths = sorted(depths)

        if not depths:
            logger.info("No valid depths found in layer columns")
            return plot_paths

        logger.info(f"Found {len(depths)} depths: {depths}")

        # Collect all metrics from all metric groups
        all_metrics = []
        metric_to_group_info = {}

        for _group_name, group_info in self.METRIC_GROUPS.items():
            for metric, label in zip(group_info["metrics"], group_info["labels"]):
                # Get base metric name (e.g., "E_C" from "mi_E_C")
                base_metric = metric.replace("mi_", "")

                # Check if we have layer data for this metric
                has_layer_data = any(
                    f"layer_mi_{base_metric}_depth{depth}_mean" in data.columns
                    for depth in depths
                )

                if has_layer_data:
                    if metric not in all_metrics:
                        all_metrics.append(metric)
                        metric_to_group_info[metric] = {
                            "base_metric": base_metric,
                            "label": label,
                            "group_title": group_info["title"],
                            "ylabel": group_info["ylabel"],
                        }

        if not all_metrics:
            logger.info("No metrics with layer data found")
            return plot_paths

        # Generate one figure per metric
        for metric in all_metrics:
            try:
                metric_info = metric_to_group_info[metric]
                base_metric = metric_info["base_metric"]
                label = metric_info["label"]
                group_title = metric_info["group_title"]
                ylabel = metric_info["ylabel"]

                # Find which depths have data for this metric
                available_depths = []
                for depth in depths:
                    layer_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                    if layer_col in data.columns:
                        available_depths.append(depth)

                if not available_depths:
                    logger.info(f"No layer data available for metric {metric}")
                    continue

                nrows = len(available_depths)
                fig, axes = plt.subplots(
                    nrows, 2, figsize=(18, 6 * nrows), sharex=True, sharey=True
                )

                # Handle single row case
                if nrows == 1:
                    axes = axes.reshape(1, -1)

                # Collect all values for consistent color scaling across all depths
                all_values = []

                for use_shunting in [True, False]:
                    category = "shunting" if use_shunting else "no_shunting"
                    cat_data = data[data["network_category"] == category]

                    if not cat_data.empty:
                        for depth in available_depths:
                            layer_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                            if layer_col in cat_data.columns:
                                values = cat_data[layer_col].dropna()
                                all_values.extend(values.tolist())

                # Calculate global vmin/vmax for consistent scaling
                if all_values:
                    vmin, vmax = min(all_values), max(all_values)
                else:
                    vmin, vmax = 0, 1

                # Create heatmaps for each depth and shunting condition
                for row_idx, depth in enumerate(available_depths):
                    layer_col = f"layer_mi_{base_metric}_depth{depth}_mean"

                    for col_idx, (use_shunting, title_suffix) in enumerate(
                        [
                            (True, "Shunting"),
                            (False, "Non-Shunting"),
                        ]
                    ):
                        ax = axes[row_idx, col_idx]
                        category = "shunting" if use_shunting else "no_shunting"
                        cat_data = data[data["network_category"] == category]

                        if cat_data.empty or layer_col not in cat_data.columns:
                            ax.text(
                                0.5,
                                0.5,
                                f"No {category} data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"Layer {depth} - {title_suffix}")
                            continue

                        # Create pivot table for heatmap
                        try:
                            pivot = cat_data.pivot_table(
                                values=layer_col,
                                index=coly,
                                columns=colx,
                                aggfunc="mean",
                            )

                            if pivot.empty:
                                ax.text(
                                    0.5,
                                    0.5,
                                    "No data to pivot",
                                    ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                )
                                ax.set_title(f"Layer {depth} - {title_suffix}")
                                continue

                            # Create heatmap
                            sns.heatmap(
                                pivot,
                                cmap="viridis",
                                annot=True,
                                fmt=".3f",
                                cbar_kws={"label": ylabel},
                                ax=ax,
                                vmin=vmin,
                                vmax=vmax,
                            )
                            # Invert y-axis to place origin at lower left
                            ax.invert_yaxis()

                            ax.set_xlabel(colx, fontsize=12)
                            if col_idx == 0:  # Only label y-axis on left column
                                ax.set_ylabel(f"{coly}\n(Layer {depth})", fontsize=12)
                            else:
                                ax.set_ylabel("")
                            ax.set_title(f"Layer {depth} - {title_suffix}", fontsize=12)

                        except Exception as e:
                            logger.error(
                                f"Error creating heatmap for {metric} layer {depth} ({title_suffix}): {e}"
                            )
                            ax.text(
                                0.5,
                                0.5,
                                f"Error: {str(e)[:50]}",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"Layer {depth} - {title_suffix}")

                fig.suptitle(
                    f"{label} ({group_title}) - Layerwise Heatmaps",
                    fontsize=16,
                    y=0.995,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.98])

                # Create safe filename from metric name
                safe_metric_name = metric.replace("mi_", "").replace("_", "-")
                plot_path = output_dir / f"{safe_metric_name}_layerwise_heatmaps.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating layerwise heatmaps for {metric}: {e}")
                traceback.print_exc()

        logger.info(f"Generated {len(plot_paths)} layerwise heatmap figures")
        return plot_paths
