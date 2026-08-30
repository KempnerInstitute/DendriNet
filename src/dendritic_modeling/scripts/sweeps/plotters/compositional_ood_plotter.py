"""Compositional OOD metric plotters."""

import logging
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.metrics import get_metric_label
from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


class CompositionalOODPlotter(BasePlotter):
    """Generates compositional OOD performance plots."""

    COMP_OOD_METRICS: ClassVar[list[str]] = [
        "comp_ood_accuracy",
        "comp_ood_auc",
        "comp_ood_categorical_loglikelihood",
        "comp_ood_mse",
        "comp_ood_cosine_similarity",
        "comp_ood_pred_label_mi_bits",
    ]

    HEATMAP_METRICS: ClassVar[list[str]] = [
        "comp_ood_accuracy",
        "comp_ood_auc",
    ]

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all compositional OOD plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        ood_cols = [col for col in data.columns if "comp_ood_" in col]
        if not ood_cols:
            logger.info(
                "No compositional OOD data found, skipping compositional OOD plots"
            )
            return plot_paths

        plots_dir = output_dir / "compositional_ood"
        plots_dir.mkdir(parents=True, exist_ok=True)

        sweep_types = self.detect_sweep_types(data)

        if "ei" in sweep_types:
            plot_paths.extend(
                self.plot_compositional_ood_1d_sweep(
                    data=data,
                    sweep_var="ei_ratio",
                    var_name="E/I Ratio",
                    logscale=True,
                    output_dir=plots_dir,
                )
            )

        if "stim_duration" in sweep_types:
            plot_paths.extend(
                self.plot_compositional_ood_1d_sweep(
                    data=data,
                    sweep_var="stimulus_duration",
                    var_name="Stimulus Duration",
                    logscale=True,
                    output_dir=plots_dir,
                )
            )

        if "max_gf" in sweep_types:
            plot_paths.extend(
                self.plot_compositional_ood_1d_sweep(
                    data=data,
                    sweep_var="max_gain_factor",
                    var_name="Maximum Gain",
                    logscale=True,
                    output_dir=plots_dir,
                )
            )

        if "fixed_gf" in sweep_types:
            plot_paths.extend(
                self.plot_compositional_ood_1d_sweep(
                    data=data,
                    sweep_var="fixed_gain_factor",
                    var_name="Fixed Gain Factor",
                    logscale=True,
                    output_dir=plots_dir,
                )
            )

        if "stoch1" in sweep_types:
            plot_paths.extend(
                self.plot_compositional_ood_heatmaps(
                    data,
                    plots_dir,
                    "stimulus_duration",
                    "max_gain_factor",
                )
            )

        if "stoch2" in sweep_types:
            plot_paths.extend(
                self.plot_compositional_ood_heatmaps(
                    data,
                    plots_dir,
                    "stimulus_duration",
                    "max_gain_tau_ratio",
                )
            )

        if "nparams" in sweep_types:
            plot_paths.extend(self.plot_metrics_vs_nparams(data, plots_dir))

        return plot_paths

    def plot_compositional_ood_1d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate metric vs sweep_var line plots with error bars.

        Args:
            data: Aggregated data with sweep_var column
            sweep_var: Sweep variable column name
            var_name: Name of the sweep variable
            logscale: Whether to use log scale for the x-axis
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        for metric in self.COMP_OOD_METRICS:
            mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else metric

            if mean_col not in data.columns:
                continue

            try:
                fig, ax = plt.subplots(figsize=self.default_figsize)

                self.plot_by_category(
                    ax=ax,
                    data=data,
                    x_col=sweep_var,
                    y_col=metric,
                    category_col="network_category",
                    show_error=True,
                )

                self.setup_axes_style(
                    ax=ax,
                    xlabel=var_name,
                    ylabel=get_metric_label(metric),
                    title=f"{get_metric_label(metric)} vs {var_name}",
                    use_log_x=logscale,
                    grid=True,
                )

                ax.legend(fontsize=10)
                fig.tight_layout()

                plot_path = output_dir / f"{metric}_vs_{sweep_var}.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {metric} vs {var_name} plot: {e}")

        logger.info(f"Generated {len(plot_paths)} {var_name} compositional OOD plots")
        return plot_paths

    def plot_compositional_ood_heatmaps(
        self, data: pd.DataFrame, output_dir: Path, colx: str, coly: str
    ) -> list[Path]:
        """
        Generate compositional OOD heatmaps.

        Args:
            data: Aggregated data
            output_dir: Output directory
            colx: Column name for x-axis
            coly: Column name for y-axis

        Returns:
            List of plot paths
        """
        plot_paths = []

        if colx not in data.columns or coly not in data.columns:
            return plot_paths

        if "network_category" not in data.columns:
            logger.warning(
                "Missing 'network_category' column for compositional OOD heatmaps"
            )
            return plot_paths

        possible_categories = [
            "dendritic_shunting",
            "dendritic_additive",
            "dendritic_normalized_additive",
            "flat_shunting",
            "flat_additive",
            "flat_normalized_additive",
            "dendritic_mlp",
            "flat_mlp",
            "point_mlp",
            "ss_mlp",
            "ss_mlp_flat",
            "active_param_mlp",
            "total_param_mlp",
        ]
        categories = [
            cat
            for cat in possible_categories
            if cat in data["network_category"].unique()
        ]

        for metric in self.HEATMAP_METRICS:
            mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else metric

            if mean_col not in data.columns:
                continue

            try:
                all_values = []
                for category in categories:
                    category_data = data[data["network_category"] == category]
                    if not category_data.empty and mean_col in category_data.columns:
                        values = category_data[mean_col].dropna()
                        all_values.extend(values.tolist())

                if all_values:
                    vmin, vmax = min(all_values), max(all_values)
                else:
                    vmin, vmax = 0, 1

                fig, axes = plt.subplots(1, 3, figsize=(24, 8))
                fig.suptitle(
                    f"{get_metric_label(metric)} Heatmap ({colx} vs {coly})",
                    fontsize=16,
                )

                pivot_tables = {}

                for idx, category in enumerate(categories):
                    ax = axes[idx]
                    category_data = data[data["network_category"] == category]

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

                    pivot = category_data.pivot_table(
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
                        ax.set_title(f"{category.title()}")
                        continue

                    pivot_tables[category] = pivot

                    sns.heatmap(
                        pivot,
                        cmap="viridis",
                        annot=True,
                        fmt=".3f",
                        cbar_kws={"label": get_metric_label(metric)},
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.invert_yaxis()

                    ax.set_xlabel(colx, fontsize=12)
                    ax.set_ylabel(coly, fontsize=12)
                    ax.set_title(f"{category.title()}", fontsize=14)

                ax_diff = axes[2]
                if (
                    "dendritic_shunting" in pivot_tables
                    and "dendritic_additive" in pivot_tables
                ):
                    shunt_pivot = pivot_tables["dendritic_shunting"]
                    no_shunt_pivot = pivot_tables["dendritic_additive"]

                    common_index = shunt_pivot.index.union(no_shunt_pivot.index)
                    common_cols = shunt_pivot.columns.union(no_shunt_pivot.columns)

                    shunt_aligned = shunt_pivot.reindex(
                        index=common_index, columns=common_cols
                    )
                    no_shunt_aligned = no_shunt_pivot.reindex(
                        index=common_index, columns=common_cols
                    )

                    diff_pivot = shunt_aligned - no_shunt_aligned

                    diff_values = diff_pivot.values.flatten()
                    diff_values = diff_values[~pd.isna(diff_values)]

                    if len(diff_values) > 0:
                        abs_max = max(abs(diff_values.min()), abs(diff_values.max()))
                        diff_vmin, diff_vmax = -abs_max, abs_max
                    else:
                        diff_vmin, diff_vmax = -1, 1

                    sns.heatmap(
                        diff_pivot,
                        cmap="seismic_r",
                        annot=True,
                        fmt=".3f",
                        cbar_kws={"label": f"{get_metric_label(metric)} Difference"},
                        ax=ax_diff,
                        vmin=diff_vmin,
                        vmax=diff_vmax,
                        center=0,
                    )
                    ax_diff.invert_yaxis()

                    ax_diff.set_xlabel(colx, fontsize=12)
                    ax_diff.set_ylabel(coly, fontsize=12)
                    ax_diff.set_title(
                        "Difference (Shunting - Non-Shunting)", fontsize=14
                    )
                else:
                    ax_diff.text(
                        0.5,
                        0.5,
                        "Cannot calculate difference\n(missing data)",
                        ha="center",
                        va="center",
                        transform=ax_diff.transAxes,
                    )
                    ax_diff.set_title(
                        "Difference (Shunting - Non-Shunting)", fontsize=14
                    )

                fig.tight_layout()
                plot_path = output_dir / f"{metric}_heatmap.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {metric} heatmap: {e}")

        logger.info(f"Generated {len(plot_paths)} compositional OOD heatmaps")
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

        for metric in self.COMP_OOD_METRICS:
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
                    linestyle = self.category_linestyles.get(category, "-")

                    y_values = cat_data[mean_col]
                    yerr = cat_data[std_col] if std_col in cat_data.columns else None

                    ax.plot(
                        cat_data["nparams"],
                        y_values,
                        marker=marker,
                        color=color,
                        linestyle=linestyle,
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
