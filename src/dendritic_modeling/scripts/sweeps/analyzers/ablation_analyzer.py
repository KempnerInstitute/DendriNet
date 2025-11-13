"""Ablation analysis sweep analyzer."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..plotters import PerformancePlotter
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class AblationSweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for ablation analysis sweeps."""

    def __init__(self):
        """Initialize ablation sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        Detect if this is an ablation analysis sweep.

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """
        from ..collectors import DataCollector

        # Quick check: look for ablation-related data in first few results
        collector = DataCollector(results_dir)
        config_dirs = collector._get_config_dirs()

        if not config_dirs:
            return False

        # Check first config for ablation data
        first_config = config_dirs[0]
        ablation_file = first_config / "ablation_analysis" / "final"

        return ablation_file.exists() or ablation_file.with_suffix(".json").exists()

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for ablation sweeps."""
        return [
            "ee_value",
            "ie_value",
            "use_shunting",
            "network_category",
            "ablation_type",
        ]

    def get_plot_types(self) -> list[str]:
        """Get plot types specific to ablation sweeps."""
        return [
            "layer_contribution",
            "ablation_impact",
            "excitation_vs_inhibition_contribution",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all ablation-specific plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        ablation_dir = output_dir / "plots" / "ablation_analysis"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Check for ablation data
        ablation_cols = [col for col in data.columns if "ablation_" in col]
        if not ablation_cols:
            logger.info("No ablation data found")
            return plot_paths

        # Generate layer contribution plots
        logger.info("Generating layer contribution plots...")
        plot_paths.extend(self._plot_layer_contributions(data, ablation_dir))

        # Generate excitation vs inhibition comparison
        plot_paths.extend(self._plot_exc_inh_contributions(data, ablation_dir))

        # Also generate standard performance plots
        plot_paths.extend(self.performance_plotter.generate_all(data, output_dir))

        return plot_paths

    def _plot_layer_contributions(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Plot contribution of each layer when ablated."""
        plot_paths = []

        # Find layer ablation columns by depth
        layer_cols = [
            col
            for col in data.columns
            if "ablation_" in col and "_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            return plot_paths

        # Extract depths and ablation types
        depths = set()
        ablation_types = set()

        for col in layer_cols:
            try:
                # Parse column name: ablation_{type}_{metric}_depth{N}_mean
                parts = col.split("_")
                if "depth" in col:
                    depth_part = next((p for p in parts if p.startswith("depth")), None)
                    if depth_part:
                        depth = int(depth_part.replace("depth", ""))
                    depths.add(depth)

                # Extract ablation type (e.g., "all_synapses", "excitation", "inhibition")
                ablation_type = parts[1] if len(parts) > 1 else "unknown"
                ablation_types.add(ablation_type)
            except (ValueError, IndexError):
                pass

        depths = sorted(depths)

        if not depths:
            return plot_paths

        # Plot accuracy drop per layer for each ablation type
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            for ablation_type in sorted(ablation_types):
                layer_accuracies = []

                for depth in depths:
                    # Find accuracy column for this layer and ablation type
                    matching_cols = [
                        col
                        for col in layer_cols
                        if f"ablation_{ablation_type}_" in col
                        and f"depth{depth}" in col
                        and "accuracy" in col
                    ]

                    if matching_cols:
                        col = matching_cols[0]
                        acc = data[col].mean()
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

            self.performance_plotter.setup_axes_style(
                ax=ax,
                xlabel="Layer Index",
                ylabel="Accuracy After Ablation",
                title="Layer Contribution Analysis (Ablation Impact)",
                grid=True,
            )
            ax.legend(fontsize=10)

            fig.tight_layout()
            plot_path = output_dir / "layer_contribution_by_ablation.png"
            plot_paths.append(self.performance_plotter.save_figure(fig, plot_path))

        except Exception as e:
            logger.error(f"Error generating layer contribution plot: {e}")

        logger.info(f"Generated {len(plot_paths)} layer contribution plots")
        return plot_paths

    def _plot_exc_inh_contributions(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Plot excitatory vs inhibitory contributions."""
        plot_paths = []

        # Look for excitation and inhibition ablation columns
        exc_cols = [
            col
            for col in data.columns
            if "ablation_excitation_" in col and col.endswith("_mean")
        ]
        inh_cols = [
            col
            for col in data.columns
            if "ablation_inhibition_" in col and col.endswith("_mean")
        ]

        if not exc_cols or not inh_cols:
            return plot_paths

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # For each metric (accuracy, auc, etc.)
            metrics = set()
            for col in exc_cols:
                # Extract metric name
                parts = (
                    col.replace("ablation_excitation_", "")
                    .replace("_mean", "")
                    .split("_depth")[0]
                )
                metrics.add(parts)

            for metric in metrics:
                # Find excitation and inhibition columns for this metric
                exc_pattern = f"ablation_excitation_{metric}"
                inh_pattern = f"ablation_inhibition_{metric}"

                exc_matching = [col for col in exc_cols if exc_pattern in col]
                inh_matching = [col for col in inh_cols if inh_pattern in col]

                if exc_matching and inh_matching:
                    exc_values = [data[col].mean() for col in exc_matching]
                    inh_values = [data[col].mean() for col in inh_matching]

                    # Plot comparison
                    x_pos = range(len(exc_values))
                    width = 0.35

                    ax.bar(
                        [x - width / 2 for x in x_pos],
                        exc_values,
                        width,
                        label="Excitation Ablated",
                        alpha=0.8,
                    )
                    ax.bar(
                        [x + width / 2 for x in x_pos],
                        inh_values,
                        width,
                        label="Inhibition Ablated",
                        alpha=0.8,
                    )

            self.performance_plotter.setup_axes_style(
                ax=ax,
                xlabel="Layer Index",
                ylabel="Performance After Ablation",
                title="Excitatory vs Inhibitory Contribution",
                grid=True,
            )
            ax.legend(fontsize=10)

            fig.tight_layout()
            plot_path = output_dir / "excitation_vs_inhibition_contribution.png"
            plot_paths.append(self.performance_plotter.save_figure(fig, plot_path))

        except Exception as e:
            logger.error(f"Error generating E/I contribution comparison: {e}")

        return plot_paths
