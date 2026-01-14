"""Multiplicative gain analysis plotters."""

import logging
from pathlib import Path
import traceback

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)

METRIC_MAPPING = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "categorical_loglikelihood": "Log Likelihood",
    "mse": "Mean Squared Error",
    "cosine_similarity": "Cosine Similarity"
}


class MultiplicativeGainPlotter(BasePlotter):
    """Generates multiplicative gain analysis plots."""

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all multiplicative gain analysis plots.
        """

        plot_paths = []
        gain_dir = output_dir / "multiplicative_gain"
        gain_dir.mkdir(parents=True, exist_ok=True)

        # Check for multiplicative gain data
        gain_cols = [col for col in data.columns if "mult_gain_" in col]
        if not gain_cols:
            logger.info("No multiplicative gain data found, skipping multiplicative gain plots")
            return plot_paths

        # Generate multiplicative gain analysis plots
        plot_paths.extend(self._generate_gain_plots(data, gain_dir))

        return plot_paths

    def _generate_gain_plots(self, data: pd.DataFrame, plots_dir: Path) -> list[Path]:
        plot_paths = []

        gain_cols: list[str] = [col for col in data.columns if "mult_gain_" in col]

        gain_factors = []
        metrics = []

        for col in gain_cols:
            if "_mean" in col:
                parts = col.replace("mult_gain_", "").replace("_mean", "").split("_gf")

                metric = parts[0]
                gain_factor = float(parts[-1])

                if metric not in metrics:
                    metrics.append(metric)
                if gain_factor not in gain_factors:
                    gain_factors.append(gain_factor)
        
        gain_factors.sort()

        sweep_types = self.detect_sweep_types(data)

        if "ei" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="ei_ratio",
                        var_name="E/I Ratio",
                        plots_dir=plots_dir,
                        metrics=metrics,
                        gain_factors=gain_factors
                    )
                )
            except Exception as e:
                logger.error(f"Error generating E/I gain plot for {metric}: {e}")
                traceback.print_exc()
        
        if "stim_duration" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="stimulus_duration",
                        var_name="Stimulus Duration",
                        plots_dir=plots_dir,
                        metrics=metrics,
                        gain_factors=gain_factors
                    )
                )
            except Exception as e:
                logger.error(f"Error generating stimulus duration gain plot for {metric}: {e}")
                traceback.print_exc()
        
        if "max_gf" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="max_gain_factor",
                        var_name="Max Gain Factor",
                        plots_dir=plots_dir,
                        metrics=metrics,
                        gain_factors=gain_factors
                    )
                )
            except Exception as e:
                logger.error(f"Error generating max gain factor gain plot for {metric}: {e}")
                traceback.print_exc()
        
        if "nparams" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data,
                        "nparams",
                        plots_dir,
                        metrics,
                        gain_factors
                    )
                )
            except Exception as e:
                logger.error(f"Error generating nparams gain plot for {metric}: {e}")
                traceback.print_exc()

        if "stoch1" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_2d_sweep(
                        data,
                        "stimulus_duration",
                        "max_gain_factor",
                        plots_dir,
                        metrics,
                        gain_factors
                    )
                )
            except Exception as e:
                logger.error(f"Error generating stochastic gain plot for {metric}: {e}")
                traceback.print_exc()
        
        if "stoch2" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_2d_sweep(
                        data,
                        "stimulus_duration",
                        "max_gain_tau_ratio",
                        plots_dir,
                        metrics,
                        gain_factors
                    )
                )
            except Exception as e:
                logger.error(f"Error generating stochastic GV gain plot for {metric}: {e}")
                traceback.print_exc()
        
        if "sh_ns_comp" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_sh_ns_comparison(
                        data,
                        plots_dir,
                        metrics,
                        gain_factors
                    )
                )
            except Exception as e:
                logger.error(f"Error generating shunting vs non-shunting gain plot for {metric}: {e}")
                traceback.print_exc()
        
        logger.info(f"Created {len(plot_paths)} multiplicative gain analysis plots")
        return plot_paths
    
    def _create_gain_plots_1d_sweep(
        self, 
        data: pd.DataFrame, 
        sweep_var: str,
        var_name: str,
        plots_dir: Path, 
        metrics: list[str], 
        gain_factors: list[float]
    ) -> list[Path]:
        """Create multiplicative gain vs sweep_var plots"""
        plot_paths = []

        plot_paths.extend(
            self._plot_gain_performance_1d_sweep(
                data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                plots_dir=plots_dir,
                metrics=metrics,
                gain_factors=gain_factors
            )
        )

        return plot_paths
    
    def _plot_gain_performance_1d_sweep(
        self, 
        data: pd.DataFrame, 
        sweep_var: str,
        var_name: str,
        plots_dir: Path, 
        metrics: list[str], 
        gain_factors: list[float]
    ) -> list[Path]:
        """Create multiplicative gain performance vs sweep_var plots"""
        plot_paths = []

        # Get unique values of sweep_var and sort them
        sweep_values = sorted(data[sweep_var].unique())

        for metric in metrics:
            heatmap_dict = {}
            vmin = float("inf")
            vmax = float("-inf")

            for category in ["shunting", "no_shunting"]:
                cat_data = data[data["network_category"] == category]

                # Create a 2D array for the heatmap: rows = gain_factors, columns = sweep_values
                heatmap_data = np.zeros((len(gain_factors), len(sweep_values)))
                
                # Fill the heatmap data
                for i, gf in enumerate(gain_factors):
                    for j, sweep_val in enumerate(sweep_values):
                        # Filter data for this sweep value
                        filtered_data = cat_data[cat_data[sweep_var] == sweep_val]
                        
                        # Extract the mean value for this gain factor
                        if len(filtered_data) > 0:
                            mean_val = filtered_data[f"mult_gain_{metric}_gf{gf}_mean"].iloc[0]
                            heatmap_data[i, j] = mean_val
                        else:
                            heatmap_data[i, j] = np.nan
                
                heatmap_dict[category] = heatmap_data
                vmin = min(vmin, np.min(heatmap_data))
                vmax = max(vmax, np.max(heatmap_data))
            
            # Create the heatmap
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
            fig.suptitle(
                f"Multiplicative Gain: {METRIC_MAPPING[metric]} vs {var_name}",
                fontsize=16,
            )
            
            def create_heatmap(
                ax: plt.Axes,
                data: np.ndarray,
                cmap: str,
                cbar_label: str,
                ax_title: str,
                vmin: float,
                vmax: float,
                center: Optional[float] = None
            ):
                sns.heatmap(
                    data,
                    xticklabels=sweep_values,
                    yticklabels=[f"{gf:.3f}" for gf in gain_factors],
                    cmap=cmap,
                    annot=True,
                    fmt=".3f",
                    cbar_kws={'label': cbar_label},
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    center=center
                )
                ax.invert_yaxis()
                ax.set_xlabel(var_name, fontsize=12)
                ax.set_ylabel('Gain Factor', fontsize=12)
                ax.set_title(ax_title, fontsize=14)

            create_heatmap(
                ax=axes[0],
                data=heatmap_dict["shunting"],
                cmap="viridis",
                cbar_label=METRIC_MAPPING[metric],
                ax_title="Shunting",
                vmin=vmin,
                vmax=vmax
            )
            create_heatmap(
                ax=axes[2],
                data=heatmap_dict["no_shunting"],
                cmap="viridis",
                cbar_label=METRIC_MAPPING[metric],
                ax_title="No Shunting",
                vmin=vmin,
                vmax=vmax
            )

            diff_data = heatmap_dict["shunting"] - heatmap_dict["no_shunting"]
            max_abs = np.max(np.abs(diff_data))
            create_heatmap(
                ax=axes[1],
                data=diff_data,
                cmap="seismic_r",
                cbar_label=f"{METRIC_MAPPING[metric]} Difference",
                ax_title="Difference (Shunting - No Shunting)",
                vmin=-max_abs,
                vmax=max_abs
            )
            
            fig.tight_layout()
            
            plot_path = plots_dir / f"gain_{metric}_vs_{sweep_var}_heatmap.png"
            plot_paths.append(self.save_figure(fig, plot_path))
        
        return plot_paths

    def _create_gain_plots_2d_sweep(
        self, 
        data: pd.DataFrame, 
        sweep_var1: str,
        sweep_var2: str,
        plots_dir: Path, 
        metrics: list[str], 
        gain_factors: list[float]
    ) -> list[Path]:
        """Create multiplicative gain vs 2D sweep plots"""
        plot_paths = []
        
        return plot_paths

    def _create_gain_plots_sh_ns_comparison(
        self, 
        data: pd.DataFrame, 
        plots_dir: Path, 
        metrics: list[str], 
        gain_factors: list[float]
    ) -> list[Path]:
        """Create multiplicative gain shunting vs non-shunting comparison plots"""
        plot_paths = []

        for metric in metrics:
            plot_paths.extend(
                self._plot_gain_performance_sh_ns_comparison(
                    data=data,
                    plots_dir=plots_dir,
                    metric=metric,
                    gain_factors=gain_factors
                )
            )

        return plot_paths

    def _plot_gain_performance_sh_ns_comparison(
        self, 
        data: pd.DataFrame, 
        plots_dir: Path, 
        metric: str, 
        gain_factors: list[float]
    ) -> list[Path]:
        """Create multiplicative gain performance shunting vs non-shunting comparison plots"""
        plot_paths = []

        categories = ["shunting", "no_shunting"]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(
            f"Multiplicative Gain: {METRIC_MAPPING[metric]} vs Gain Factor",
            fontsize=16,
        )

        for category in categories:
            cat_data = data[data["network_category"] == category]

            mean_data = []
            std_data = []

            for gf in gain_factors:
                mean_data.append(cat_data[f"mult_gain_{metric}_gf{gf}_mean"].iloc[0])
                std_data.append(cat_data[f"mult_gain_{metric}_gf{gf}_std"].iloc[0])
            mean_data = np.array(mean_data)
            std_data = np.array(std_data)
            
            color = self.category_colors.get(category, "black")
            marker = self.category_markers.get(category, "o")

            ax.plot(gain_factors, mean_data, color=color, marker=marker, label=category)
            ax.fill_between(
                gain_factors, 
                mean_data - std_data, 
                mean_data + std_data, 
                color=color, 
                alpha=0.2
            )
        
        ax.set_xscale("log")
        self.setup_axes_style(
            ax=ax,
            xlabel="Gain Factor",
            ylabel=METRIC_MAPPING[metric],
            grid=True
        )

        ax.legend(fontsize=10)
        fig.tight_layout()

        plot_path = plots_dir / f"gain_{metric}_vs_sh_vs_ns.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths


