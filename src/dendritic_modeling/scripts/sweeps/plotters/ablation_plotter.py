"""Ablation analysis plotters."""

import itertools
import logging
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


class AblationPlotter(BasePlotter):
    """Generates ablation analysis plots."""

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all ablation analysis plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        ablation_dir = output_dir / "ablation"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Check for ablation data
        ablation_cols = [col for col in data.columns if "ablation_" in col]
        if not ablation_cols:
            logger.info("No ablation data found, skipping ablation plots")
            return plot_paths

        # Generate layer contribution plots
        # plot_paths.extend(self._plot_layer_contributions(data, ablation_dir))
        # Generate ablation analysis plots
        plot_paths.extend(self._generate_ablation_plots(data, ablation_dir))

        return plot_paths

    def _plot_layer_contributions(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Plot contribution of each layer when ablated."""
        plot_paths = []

        # Find layer ablation columns
        layer_cols = [
            col
            for col in data.columns
            if "ablation_" in col and "_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            return plot_paths

        # Extract ablation types and depths
        ablation_types = set()
        depths = set()

        for col in layer_cols:
            try:
                parts = col.split("_")
                # Get ablation type (e.g., "all", "excitation", "inhibition")
                if len(parts) > 1:
                    ablation_types.add(parts[1])

                # Get depth
                if "depth" in col:
                    depth_part = next((p for p in parts if p.startswith("depth")), None)
                    if depth_part:
                        depth = int(depth_part.replace("depth", ""))
                    depths.add(depth)
            except (ValueError, IndexError):
                pass

        depths = sorted(depths)

        if not depths or not ablation_types:
            return plot_paths

        # Plot accuracy drop per layer for each ablation type
        try:
            fig, ax = plt.subplots(figsize=self.default_figsize)

            for ablation_type in sorted(ablation_types):
                layer_accuracies = []

                for depth in depths:
                    matching_cols = [
                        col
                        for col in layer_cols
                        if f"ablation_{ablation_type}_" in col
                        and f"depth{depth}" in col
                        and "accuracy" in col
                    ]

                    if matching_cols:
                        acc = data[matching_cols[0]].mean()
                        layer_accuracies.append(acc)
                    else:
                        layer_accuracies.append(None)

                if any(acc is not None for acc in layer_accuracies):
                    ax.plot(
                        depths,
                        layer_accuracies,
                        marker="o",
                        label=ablation_type.replace("_", " ").title(),
                        linewidth=2,
                        markersize=6,
                    )

            self.setup_axes_style(
                ax=ax,
                xlabel="Layer Index",
                ylabel="Accuracy After Ablation",
                title="Layer Contribution (Ablation Impact)",
                grid=True,
            )
            ax.legend(fontsize=10)

            fig.tight_layout()
            plot_path = output_dir / "layer_contribution.png"
            plot_paths.append(self.save_figure(fig, plot_path))

        except Exception as e:
            logger.error(f"Error generating layer contribution plot: {e}")

        logger.info(f"Generated {len(plot_paths)} ablation plots")
        return plot_paths

    def _generate_ablation_plots(
        self, valid_data: pd.DataFrame, plots_dir: Path
    ) -> list[Path]:
        """Generate ablation analysis plots matching old analysis format"""
        plot_paths = []

        # ablation_plot_dir = plots_dir / "ablation_analysis_plots"

        # Find ablation columns (look for both column name patterns)
        ablation_cols = [
            col
            for col in valid_data.columns
            if col.startswith("ablation_")
            or "ablation" in col.lower()
            or "contribution" in col.lower()
        ]

        if not ablation_cols:
            logger.info("No ablation columns found")
            return plot_paths

        # Extract ablation types and metrics from column structure
        ablation_types = []
        metrics = []
        depths = []
        layer_cols = []

        for col in ablation_cols:
            if "_depth" in col and "_mean" in col:
                # Pattern: ablation_type_metric_depthX_mean
                parts = col.replace("_mean", "").split("_")
                if len(parts) >= 4:
                    ablation_type = parts[1]
                    metric = parts[2]
                    depth_part = parts[3]
                    if depth_part.startswith("depth"):
                        try:
                            depth = int(depth_part.replace("depth", ""))
                            if ablation_type not in ablation_types:
                                ablation_types.append(ablation_type)
                            if metric not in metrics:
                                metrics.append(metric)
                            if depth not in depths:
                                depths.append(depth)
                        except ValueError:
                            continue
            elif "_mean" in col:
                # Check if this could be a layer-agnostic ablation column
                layer_cols.append(col)

        depths.sort()

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            logger.info("No shunting/no_shunting data for ablation analysis")
            return plot_paths

        sweep_types = self.detect_sweep_types(valid_data)

        # Generate E/I ratio plots by depth (current approach)
        for ablation_type, metric in itertools.product(ablation_types, metrics):
            if "ei" in sweep_types:
                try:
                    plot_paths.extend(
                        self._create_ablation_ei_plots(
                            valid_data,
                            plots_dir,
                            ablation_type,
                            metric,
                            depths,
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating E/I ablation plot for {ablation_type}_{metric}: {e}"
                    )
                    traceback.print_exc()

            if "stoch1" in sweep_types:
                try:
                    plot_paths.extend(
                        self._create_ablation_heatmaps(
                            valid_data,
                            plots_dir,
                            ablation_type,
                            metric,
                            depths,
                            "stimulus_duration",
                            "max_gain_factor",
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating stochastic ablation plot for {ablation_type}_{metric}: {e}"
                    )
                    traceback.print_exc()

            if "stoch2" in sweep_types:
                try:
                    plot_paths.extend(
                        self._create_ablation_heatmaps(
                            valid_data,
                            plots_dir,
                            ablation_type,
                            metric,
                            depths,
                            "stimulus_duration",
                            "max_gain_tau_ratio",
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating stochastic ablation plot for {ablation_type}_{metric}: {e}"
                    )
                    traceback.print_exc()

        logger.info(f"Created {len(plot_paths)} ablation analysis plots")
        return plot_paths

    def _create_ablation_ei_plots(
        self,
        valid_data: pd.DataFrame,
        output_dir: Path,
        ablation_type: str,
        metric: str,
        depths: list[int],
    ) -> list[Path]:
        """Create ablation vs E/I ratio plots by depth"""
        plot_paths = []

        # Find columns for this ablation type and metric
        ablation_cols_filtered = [
            col
            for col in valid_data.columns
            if f"ablation_{ablation_type}_{metric}_" in col and "_mean" in col
        ]

        if not ablation_cols_filtered:
            return plot_paths

        # Create separate plots for each network category
        categories = ["shunting", "no_shunting"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"{ablation_type.title()} Ablation: {metric.title()} vs E/I Ratio by Depth",
            fontsize=16,
        )

        for idx, category in enumerate(categories):
            ax = axes[idx]
            category_data = valid_data[valid_data["network_category"] == category]

            if category_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No {category} data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"{category.title()}")
                continue

            # Plot lines for each depth
            for depth in depths:
                depth_col = f"ablation_{ablation_type}_{metric}_depth{depth}_mean"
                if depth_col not in category_data.columns:
                    continue

                # Sort by E/I ratio for proper line plotting
                sorted_data = category_data.sort_values("ei_ratio")

                # Filter out non-numeric or missing values
                plot_data = sorted_data[[depth_col, "ei_ratio"]].dropna()
                if plot_data.empty:
                    continue

                try:
                    # Convert to numeric if needed
                    y_vals = pd.to_numeric(
                        plot_data[depth_col], errors="coerce"
                    ).dropna()
                    x_vals = plot_data.loc[y_vals.index, "ei_ratio"]

                    if len(y_vals) > 0:
                        ax.plot(
                            x_vals,
                            y_vals,
                            marker="o",
                            label=f"Depth {depth}",
                            alpha=0.8,
                            linewidth=2,
                        )
                except Exception as e:
                    logger.warning(f"Could not plot depth {depth} for {category}: {e}")
                    continue

            self.setup_axes_style(
                ax=ax,
                xlabel="E/I Ratio",
                ylabel=f"{metric.title()}",
                title=f"{category.title()}",
                use_log_x=True,
                grid=True,
            )
            ax.legend()

        fig.tight_layout()

        plot_path = output_dir / f"{ablation_type}_{metric}_ablation.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _create_ablation_heatmaps(
        self,
        valid_data: pd.DataFrame,
        output_dir: Path,
        ablation_type: str,
        metric: str,
        depths: list[int],
        colx: str,
        coly: str,
    ) -> list[Path]:
        """Create ablation heatmaps

        Args:
            valid_data: DataFrame with ablation data
            output_dir: Output directory for plots
            ablation_type: Type of ablation (e.g., 'layer', 'synapse')
            metric: Metric to plot (e.g., 'contribution', 'accuracy')
            depths: List of depths to create heatmaps for
            colx: Column name for x-axis of heatmap
            coly: Column name for y-axis of heatmap
        """
        plot_paths = []

        # Find columns for this ablation type and metric
        ablation_cols_filtered = [
            col
            for col in valid_data.columns
            if (f"ablation_{ablation_type}_{metric}_" in col and "_mean" in col)
        ]

        if not ablation_cols_filtered:
            return plot_paths

        # Validate that colx and coly exist in the data
        if colx not in valid_data.columns:
            logger.warning(f"Column {colx} not found in data")
            return plot_paths
        if coly not in valid_data.columns:
            logger.warning(f"Column {coly} not found in data")
            return plot_paths

        # Create separate plots for each network category
        categories = ["shunting", "no_shunting"]

        # Filter depths to only those with actual data
        valid_depths = []
        for depth in depths:
            depth_col = f"ablation_{ablation_type}_{metric}_depth{depth}_mean"
            if depth_col in valid_data.columns:
                # Check if there's actual valid data (not just NaN/empty)
                for category in categories:
                    category_data = valid_data[
                        valid_data["network_category"] == category
                    ]
                    if not category_data.empty:
                        plot_data = category_data[[depth_col, colx, coly]].dropna()
                        if not plot_data.empty:
                            # Convert to numeric and check for valid values
                            numeric_values = pd.to_numeric(
                                plot_data[depth_col], errors="coerce"
                            ).dropna()
                            if len(numeric_values) > 0:
                                valid_depths.append(depth)
                                break

        if not valid_depths:
            logger.warning(f"No valid depth data found for {ablation_type}_{metric}")
            return plot_paths

        # Create subplot layout: len(valid_depths) rows x 2 columns
        fig, axes = plt.subplots(
            len(valid_depths), 2, figsize=(15, 5 + 6 * len(valid_depths))
        )
        fig.suptitle(
            f"{ablation_type.title()} Ablation: {metric.title()} Heatmap ({colx} vs {coly})",
            fontsize=16,
        )

        # Handle case where there's only one depth (axes becomes 1D)
        if len(valid_depths) == 1:
            axes = axes.reshape(1, -1)

        # Collect all values for consistent color scaling
        all_values = []
        for depth in valid_depths:
            depth_col = f"ablation_{ablation_type}_{metric}_depth{depth}_mean"
            for category in categories:
                category_data = valid_data[valid_data["network_category"] == category]
                if not category_data.empty:
                    values = pd.to_numeric(
                        category_data[depth_col], errors="coerce"
                    ).dropna()
                    all_values.extend(values.tolist())

        # Calculate global vmin/vmax for consistent scaling
        if all_values:
            vmin, vmax = min(all_values), max(all_values)
        else:
            vmin, vmax = 0, 1

        for depth_idx, depth in enumerate(valid_depths):
            depth_col = f"ablation_{ablation_type}_{metric}_depth{depth}_mean"

            for cat_idx, category in enumerate(categories):
                ax = axes[depth_idx, cat_idx]
                category_data = valid_data[valid_data["network_category"] == category]

                if category_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {category} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Depth {depth} - {category.title()}")
                    continue

                # Prepare data for heatmap
                plot_data = category_data[[depth_col, colx, coly]].dropna()
                if plot_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for depth {depth}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Depth {depth} - {category.title()}")
                    continue

                try:
                    # Convert metric values to numeric
                    plot_data[depth_col] = pd.to_numeric(
                        plot_data[depth_col], errors="coerce"
                    )
                    plot_data = plot_data.dropna()

                    if plot_data.empty:
                        ax.text(
                            0.5,
                            0.5,
                            f"No valid data for depth {depth}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"Depth {depth} - {category.title()}")
                        continue

                    # Create pivot table for heatmap
                    pivot_data = plot_data.pivot_table(
                        values=depth_col,
                        index=coly,
                        columns=colx,
                        aggfunc="mean",
                    )

                    if pivot_data.empty:
                        ax.text(
                            0.5,
                            0.5,
                            f"No data to pivot for depth {depth}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"Depth {depth} - {category.title()}")
                        continue

                    # Create heatmap
                    sns.heatmap(
                        pivot_data,
                        cmap="viridis",
                        annot=True,
                        fmt=".3f",
                        cbar_kws={"label": f"{metric.title()}"},
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    # Invert y-axis to place origin at lower left
                    ax.invert_yaxis()

                    ax.set_xlabel(colx, fontsize=12)
                    ax.set_ylabel(coly, fontsize=12)
                    ax.set_title(f"Depth {depth} - {category.title()}", fontsize=12)

                except Exception as e:
                    logger.warning(
                        f"Could not create heatmap for depth {depth}, category {category}: {e}"
                    )
                    ax.text(
                        0.5,
                        0.5,
                        f"Error plotting depth {depth}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Depth {depth} - {category.title()}")
                    continue

        fig.tight_layout()

        plot_path = (
            output_dir / f"{ablation_type}_{metric}_heatmap_{colx}_vs_{coly}.png"
        )
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths
