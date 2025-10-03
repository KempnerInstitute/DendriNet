"""Ablation analysis plotters."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
        plot_paths.extend(self._plot_layer_contributions(data, ablation_dir))

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
