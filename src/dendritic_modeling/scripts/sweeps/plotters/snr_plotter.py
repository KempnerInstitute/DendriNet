"""SNR analysis plotters."""

import logging
from pathlib import Path
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


SNR_PREFIX_MAP = {
    "exc": "Excitation",
    "inh": "Inhibition",
    "upstream": "Upstream",
    "vout": "Vout"
}


class SNRPlotter(BasePlotter):
    """Generates SNR analysis plots."""

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all SNR analysis plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        snr_dir = output_dir / "snr"
        snr_dir.mkdir(parents=True, exist_ok=True)

        # Check for SNR data
        snr_cols = [col for col in data.columns if "snr_" in col]
        if not snr_cols:
            logger.info("No SNR data found, skipping SNR plots")
            return plot_paths

        # Generate SNR analysis plots
        plot_paths.extend(self._generate_snr_plots(data, snr_dir))

        return plot_paths
    
    def _generate_snr_plots(
        self, valid_data: pd.DataFrame, plots_dir: str
    ) -> list[str]:
        """Generate SNR analysis plots"""
        plot_paths = []

        # Find SNR columns (look for both column name patterns)
        snr_cols = [
            col
            for col in valid_data.columns
            if "snr_" in col
        ]

        if not snr_cols:
            print("No SNR columns found")
            return plot_paths

        prefixes = []
        depths = []

        for col in snr_cols:
            col: str
            if "std" not in col:
                # Pattern: snr_prefix_d'^2_mean_depthX_mean
                parts = col.replace("snr_", "").replace("d'^2_mean_depth", "").split("_")
                prefix = parts[0]
                depth = int(parts[1])

                if prefix not in prefixes:
                    prefixes.append(prefix)
                if depth not in depths:
                    depths.append(depth)

        depths.sort()

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            print("No shunting/no_shunting data for ablation analysis")
            return plot_paths
        
        sweep_types = self.detect_sweep_types(valid_data)
        # Generate E/I ratio plots by depth (current approach)
        for prefix in prefixes:
            if "ei" in sweep_types:
                try:
                    plot_paths.extend(
                        self._create_snr_ei_plots(
                            valid_data,
                            plots_dir,
                            prefix,
                            depths
                        )
                    )
                except Exception as e:
                    print(
                        f"Error generating E/I SNR plot for {prefix}: {e}"
                    )
            if "stoch1" in sweep_types:
                try:
                    plot_paths.extend(
                        self._create_snr_heatmaps(
                            valid_data,
                            plots_dir,
                            prefix,
                            depths,
                            "stimulus_duration",
                            "max_gain_factor"
                        )
                    )
                except Exception as e:
                    print(
                        f"Error generating stochastic SNR plot for {prefix}: {e}"
                    )
            
            if "stoch2" in sweep_types:
                try:
                    plot_paths.extend(
                        self._create_snr_heatmaps(
                            valid_data,
                            plots_dir,
                            prefix,
                            depths,
                            "stimulus_duration",
                            "max_gain_tau_ratio"
                        )
                    )
                except Exception as e:
                    print(
                        f"Error generating stochastic GV SNR plot for {prefix}: {e}"
                    )

        print(f"Created {len(plot_paths)} SNR analysis plots")
        return plot_paths

    def _create_snr_ei_plots(
        self,
        valid_data: pd.DataFrame,
        output_dir: str,
        prefix: str,
        depths: list[int]
    ) -> list[str]:
        """Create SNR vs E/I ratio plots by depth"""
        plot_paths = []

        # Find snr columns for this state variable
        snr_cols_filtered = [
            col
            for col in valid_data.columns
            if f"snr_{prefix}_" in col and "std" not in col
        ]

        if not snr_cols_filtered:
            return plot_paths

        # Create separate plots for each network category
        categories = ["shunting", "no_shunting"]

        d_prime_sq_str = r"d'$^2$"

        # Create color mapping for depths using a colormap
        # Normalize depths to [0, 1] range using linspace
        if len(depths) > 1:
            # Create normalized values from 0 to 1
            normalized_depths = np.linspace(0, 0.6, len(depths))
        else:
            normalized_depths = [0.5]  # Middle of colormap for single depth
        
        # Choose a colormap (viridis, plasma, inferno, tab10, etc.)
        # cmap = plt.get_cmap("gnuplot2")
        
        # # Create depth -> color mapping
        # depth_colors = {
        #     depth: cmap(norm_val) for depth, norm_val in zip(depths, normalized_depths)
        # }

        cmap = plt.get_cmap("tab10")
        depth_colors = {
            depth: cmap(depth / 9.0) for depth in depths
        }

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"{SNR_PREFIX_MAP[prefix]} Disciminality: {d_prime_sq_str} vs E/I Ratio by Depth",
            fontsize=16,
        )
        depths = deepcopy(depths)
        depths.reverse()

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
                depth_col = f"snr_{prefix}_d'^2_mean_depth{depth}_mean"
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
                            color=depth_colors[depth],
                            alpha=0.8,
                            linewidth=2,
                        )
                except Exception as e:
                    print(f"Warning: Could not plot depth {depth} for {category}: {e}")
                    continue

            self.setup_axes_style(
                ax=ax,
                xlabel="E/I Ratio",
                ylabel=f"{SNR_PREFIX_MAP[prefix]} {d_prime_sq_str}",
                title=f"{category.title()}",
                use_log_x=True,
                grid=True
            )
            ax.legend()

        fig.tight_layout()
        
        plot_path = output_dir / f"{prefix}_snr_vs_ei_ratio.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _create_snr_heatmaps(
        self,
        valid_data: pd.DataFrame,
        output_dir: str,
        prefix: str,
        depths: list[int],
        colx: str,
        coly: str,
    ) -> list[str]:
        """
        Create SNR heatmaps

        Args:
            valid_data: DataFrame with SNR data
            output_dir: Output directory for plots
            prefix: Prefix of SNR column (e.g., 'exc', 'inh', 'upstream', 'vout')
            depths: List of depths to create heatmaps for
            colx: Column name for x-axis of heatmap
            coly: Column name for y-axis of heatmap
        """
        plot_paths = []

        # Find snr columns for this state variable
        snr_cols_filtered = [
            col
            for col in valid_data.columns
            if f"snr_{prefix}_" in col and "std" not in col
        ]

        if not snr_cols_filtered:
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

        d_prime_sq_str = r"d'$^2$"

        # Create subplot layout: len(depths) rows x 2 columns
        fig, axes = plt.subplots(len(depths), 2, figsize=(15, 3 + 6 * len(depths)))
        fig.suptitle(
            f"{SNR_PREFIX_MAP[prefix]} Disciminality: {d_prime_sq_str} Heatmap ({colx} vs {coly})",
            fontsize=16,
        )

        # Handle case where there's only one depth (axes becomes 1D)
        if len(depths) == 1:
            axes = axes.reshape(1, -1)

        # Collect all values for consistent color scaling
        all_values = []
        for depth in depths:
            depth_col = f"snr_{prefix}_d'^2_mean_depth{depth}_mean"
            if depth_col in valid_data.columns:
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

        for depth_idx, depth in enumerate(depths):
            depth_col = f"snr_{prefix}_d'^2_mean_depth{depth}_mean"
            
            if depth_col not in valid_data.columns:
                # Fill empty subplots if depth column doesn't exist
                for cat_idx in range(2):
                    ax = axes[depth_idx, cat_idx]
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for depth {depth}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Depth {depth} - {categories[cat_idx].title()}")
                continue

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
                        cbar_kws={"label": f"{SNR_PREFIX_MAP[prefix]} {d_prime_sq_str}"},
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax
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
        
        plot_path = output_dir / f"{prefix}_snr_heatmap_{colx}_vs_{coly}.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths