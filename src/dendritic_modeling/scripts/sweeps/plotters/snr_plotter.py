"""SNR analysis plotters."""

import logging
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


SNR_PREFIX_MAP = {
    "exc": "Excitation",
    "inh": "Inhibition",
    "upstream": "Upstream",
    "vout": "Vout",
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
        self, valid_data: pd.DataFrame, plots_dir: Path
    ) -> list[Path]:
        """Generate SNR analysis plots"""
        plot_paths = []

        # Find SNR columns (look for both column name patterns)
        snr_cols = [col for col in valid_data.columns if "snr_" in col]

        if not snr_cols:
            logger.info("No SNR columns found")
            return plot_paths

        prefixes = []
        depths = []

        for col in snr_cols:
            col: str
            if "std" not in col:
                # Pattern: snr_prefix_d'^2_mean_depthX_mean
                parts = (
                    col.replace("snr_", "").replace("d'^2_mean_depth", "").split("_")
                )
                prefix = parts[0]
                depth = int(parts[1])

                if prefix not in prefixes:
                    prefixes.append(prefix)
                if depth not in depths:
                    depths.append(depth)

        depths.sort()

        if not valid_data["network_category"].isin(["shunting", "no_shunting"]).any():
            logger.info("No shunting/no_shunting data for SNR analysis")
            return plot_paths

        # Generate E/I ratio plots by depth (current approach)
        for prefix in prefixes:
            try:
                plot_paths.extend(
                    self._create_snr_ei_plots(valid_data, plots_dir, prefix, depths)
                )
            except Exception as e:
                logger.error(f"Error generating E/I SNR plot for {prefix}: {e}")

        logger.info(f"Created {len(plot_paths)} SNR analysis plots")
        return plot_paths

    def _create_snr_ei_plots(
        self,
        valid_data: pd.DataFrame,
        output_dir: Path,
        prefix: str,
        depths: list[int],
    ) -> list[Path]:
        """Create SNR vs E/I ratio plots by depth"""
        plot_paths = []

        # Find columns for this ablation type and metric
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

        # Create color mapping for depths using tab10 colormap
        cmap = plt.get_cmap("tab10")
        depth_colors = {depth: cmap(depth / 9.0) for depth in depths}

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
                    logger.warning(f"Could not plot depth {depth} for {category}: {e}")
                    continue

            self.setup_axes_style(
                ax=ax,
                xlabel="E/I Ratio",
                ylabel=f"{SNR_PREFIX_MAP[prefix]} {d_prime_sq_str}",
                title=f"{category.title()}",
                use_log_x=True,
                grid=True,
            )
            ax.legend()

        fig.tight_layout()

        plot_path = output_dir / f"{prefix}_snr_vs_ei_ratio.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths
