"""Plotting functions for analysis results and performance metrics."""

import glob
import json
import logging
import os
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dendritic_modeling.plotting.styles import (
    bar_kwargs,
    indexed_hatch,
    indexed_linestyle,
    indexed_style,
    split_style,
)

from .plotting_utils import (
    convert_layer_names,
    create_figure,
    create_subplots,
    handle_plot_error,
    save_plot,
    setup_basic_plot,
)

logger = logging.getLogger(__name__)


def plot_layer_contributions(results: dict[str, Any], **kwargs):
    """Plot layer contributions showing how each layer contributes to model performance."""
    logger.info("Plotting layer contributions")
    try:
        fig, ax = create_figure(figsize=(10, 6))

        if results and isinstance(results, dict):
            # Extract layer names and their contribution values
            layers = []
            contributions = []

            for layer_name, metrics in results.items():
                layers.append(layer_name)
                # Try to extract a meaningful contribution metric
                if isinstance(metrics, dict):
                    # Look for common contribution metrics
                    if "contribution" in metrics:
                        contributions.append(metrics["contribution"])
                    elif "accuracy_drop" in metrics:
                        contributions.append(metrics["accuracy_drop"])
                    elif "performance_drop" in metrics:
                        contributions.append(metrics["performance_drop"])
                    else:
                        # Use first numeric value found
                        for _key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                contributions.append(value)
                                break
                        else:
                            contributions.append(0.0)
                elif isinstance(metrics, (int, float)):
                    contributions.append(metrics)
                else:
                    contributions.append(0.0)

            if layers and contributions:
                # Create bar plot
                bars = ax.bar(range(len(layers)), contributions)
                ax.set_xticks(range(len(layers)))
                ax.set_xticklabels(layers, rotation=45, ha="right")
                ax.set_ylabel("Contribution")
                ax.set_title("Layer Contributions to Model Performance")

                # Add value labels on bars
                for bar, value in zip(bars, contributions):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

                plt.tight_layout()
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No layer contribution data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Layer Contributions")
        else:
            ax.text(
                0.5,
                0.5,
                "No data provided",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Layer Contributions")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "layer_contributions")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax
    except Exception as e:
        return handle_plot_error("layer contribution plotting", e)


def plot_ablation_results(results: dict, **kwargs):
    """Plot ablation study results showing performance impact of removing components."""
    logger.info("Plotting ablation analysis results")
    try:
        fig, ax = create_figure(figsize=(10, 6))
        if results:
            # Handle nested structure from ablation analysis
            if isinstance(next(iter(results.values())), dict) and "depth" not in next(
                iter(results.values())
            ):
                # This is the nested structure: ablation_type -> layer -> metrics
                # Flatten to show average performance drop per ablation type
                keys = []
                values = []
                for ablation_type, layer_data in results.items():
                    keys.append(ablation_type.replace("_", " ").title())
                    # Calculate average accuracy drop across layers
                    acc_drops = []
                    for _layer_name, metrics in layer_data.items():
                        if "accuracy_drop" in metrics:
                            drop_val = metrics["accuracy_drop"]
                            # Ensure we have a scalar value
                            if isinstance(drop_val, (list, np.ndarray)):
                                drop_val = np.mean(drop_val) if len(drop_val) > 0 else 0
                            elif hasattr(drop_val, "item"):  # torch tensor
                                drop_val = drop_val.item()
                            acc_drops.append(float(drop_val))
                    avg_drop = np.mean(acc_drops) if acc_drops else 0
                    values.append(avg_drop)
            else:
                # Simple structure: key -> value
                keys = list(results.keys())
                values = [
                    (
                        next(iter(result.values()))
                        if isinstance(result, dict) and result
                        else result if isinstance(result, (int, float)) else 0
                    )
                    for result in results.values()
                ]

            ax.bar(keys, values)
            setup_basic_plot(
                ax,
                "Ablation Study Results",
                "Ablation Type",
                "Average Performance Drop",
            )
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(
                0.5,
                0.5,
                "Ablation Results\n(No data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Ablation Results")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "ablation_results")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax
    except Exception as e:
        return handle_plot_error("ablation plotting", e)


def plot_layer_ablation(
    layer_results: dict,
    baselines: dict,
    save_path: Optional[str] = None,
    filename: str = "final",
    somatic_synapses: bool = True,
) -> Optional[list[plt.Figure]]:
    """
    Plot layer ablation results showing accuracy/log-likelihood decrease vs depth.

    Args:
        layer_results: Dictionary containing ablation results by type and layer
        save_path: Path to save plots. If None, returns list of figures
        filename: Base filename for saved plots

    Returns:
        List of figures if save_path is None, otherwise None
    """
    if save_path is None:
        fig_list = []

    metric_mapping = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "categorical_loglikelihood": "Log Likelihood",
        "mse": "MSE",
        "cosine_similarity": "Cosine Similarity",
        "pred_label_mi_bits": "I(C; Ĉ) [bits]",
    }

    def has_metric_nested(dictionary, metric):
        """
        Check if key exists at any level in a nested dictionary
        """
        if isinstance(dictionary, dict):
            key = f"{metric}_drop"
            if key in dictionary:
                return True
            return any(
                has_metric_nested(value, metric) for value in dictionary.values()
            )
        elif isinstance(dictionary, (list, tuple)):
            return any(has_metric_nested(item, metric) for item in dictionary)
        return False

    metrics = []
    for metric in metric_mapping.keys():
        if has_metric_nested(layer_results, metric):
            metrics.append(metric)

    base_targets = ["excitation", "inhibition", "all_synapses", "upstream"]
    method_to_targets = {
        method: targets
        for method, targets in layer_results.items()
        if isinstance(targets, dict)
    }

    if not method_to_targets:
        logger.warning("No recognizable ablation keys found for plotting")
        return fig_list if save_path is None else None

    def _method_sort_key(m: str) -> tuple[int, str]:
        return (0 if m == "lesion" else 1, m)

    for method in sorted(method_to_targets.keys(), key=_method_sort_key):
        target_results = method_to_targets[method]
        bases_present = [b for b in base_targets if b in target_results]
        if not bases_present:
            continue

        # Use consistent depth order across ablation types for this method.
        depths_set = set()
        for base in bases_present:
            type_dict: dict = target_results.get(base, {})
            for layer_dict in type_dict.values():
                if "depth" in layer_dict:
                    depths_set.add(layer_dict["depth"])
        depths = sorted(depths_set)
        if not depths:
            continue

        # Analyzer results use soma-relative depths: zero is the soma and
        # positive values are dendritic stages moving distally.  Do not infer
        # the soma from the minimum *observed* depth because synapse ablations
        # legitimately omit depth zero when somatic synapses are disabled.
        depth_labels = ["Soma" if depth == 0 else f"Distal {depth}" for depth in depths]

        width = 0.8 / max(len(bases_present), 1)

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 8))

            for i, base in enumerate(bases_present):
                type_dict: dict = target_results.get(base, {})

                # depth -> list[values] (handle duplicates safely)
                depth_to_vals: dict[int, list[float]] = {}
                for layer_dict in type_dict.values():
                    d = layer_dict.get("depth")
                    if d is None:
                        continue
                    v = layer_dict.get(f"{metric}_drop", 0.0)
                    if isinstance(v, list):
                        v = float(np.mean(v)) if v else 0.0
                    depth_to_vals.setdefault(int(d), []).append(float(v))

                decreases = [
                    float(np.mean(depth_to_vals.get(d, [0.0]))) for d in depths
                ]

                x = np.array(depths) + width * (i - (len(bases_present) - 1) / 2)
                ax.bar(
                    x,
                    decreases,
                    width=width,
                    label=base.replace("_", " ").title(),
                    **bar_kwargs(indexed_style(i).color, indexed_hatch(i)),
                )

            ax.set_xticks(depths)
            ax.set_xticklabels(depth_labels, rotation=45)
            ax.set_xlabel("Branch Depth (Soma to Distal)")
            ax.set_ylabel(f"{metric_mapping[metric]} Drop")

            title = f"{metric_mapping[metric]} Drop vs. Branch Depth"
            title += f"\nMethod: {method.replace('_', ' ').title()}"
            title += f"\n(Baseline: {float(baselines.get(metric, 0.0)):.3f})"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)

            if save_path is not None:
                plot_path = os.path.join(save_path, f"{filename}_{method}_{metric}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                fig_list.append(fig)

    if save_path is None:
        return fig_list


def plot_per_layer_ablation(layer_results: dict[str, dict[str, dict]], **kwargs):
    """Plot comprehensive per-layer ablation results similar to information analysis."""
    logger.info("Per-layer ablation plotting - creating comprehensive visualizations")
    try:
        save_path = kwargs.get("save_path")
        filename_prefix = kwargs.get("filename_prefix", "ablation_per_layer")

        if not layer_results:
            logger.warning("No layer ablation results provided")
            return None

        # Convert layer names and organize by depth
        layer_data_by_depth = {}

        # Use convert_layer_names for consistency with information analysis
        # First restructure the data to work with convert_layer_names
        sample_ablation = next(iter(layer_results.values()))
        temp_layer_results = {}
        for layer_name, layer_data in sample_ablation.items():
            temp_layer_results[layer_name] = layer_data

        # Convert layer names using the standard function
        convert_layer_names(
            temp_layer_results, somatic_synapses=kwargs.get("somatic_synapses", True)
        )

        # Create mapping from original technical names to display names
        all_depths = set()
        for _ablation_type, ablation_data in layer_results.items():
            if ablation_data:
                for _layer_name, layer_metrics in ablation_data.items():
                    if "depth" in layer_metrics:
                        all_depths.add(layer_metrics["depth"])

        all_depths = sorted(all_depths)  # CORRECT: Lower depth = soma, higher = distal

        # Create consistent naming
        for ablation_type, ablation_data in layer_results.items():
            if not ablation_data:
                continue

            # Create a consistent structure for all depths
            layers_info = []
            for depth in all_depths:
                # Depth values are soma-relative, so a missing zero must not
                # cause the first dendritic stage to be displayed as soma.
                display_name = "Soma" if depth == 0 else f"Distal Layer {depth}"

                # Find the layer data for this depth
                layer_metrics = None
                for _layer_name, metrics in ablation_data.items():
                    if metrics.get("depth") == depth:
                        layer_metrics = metrics
                        break

                # If no data for this depth, create empty metrics
                if layer_metrics is None:
                    layer_metrics = {"depth": depth}

                layers_info.append(
                    {"depth": depth, "name": display_name, "metrics": layer_metrics}
                )

            layer_data_by_depth[ablation_type] = layers_info

        plots = []

        # Create separate plots for each metric type
        metric_types = ["accuracy_drop", "auc_drop", "categorical_loglikelihood_drop"]

        for metric_type in metric_types:
            if any(
                any(metric_type in layer["metrics"] for layer in layers)
                for layers in layer_data_by_depth.values()
            ):
                fig, ax = create_figure(figsize=(14, 8))

                # Plot data for each ablation type
                ablation_types = list(layer_data_by_depth.keys())
                n_layers = len(layer_data_by_depth[ablation_types[0]])
                x_pos = np.arange(n_layers) * 1.5  # Add spacing between layer groups
                width = 0.3  # Slightly wider bars for better visibility

                colors = ["red", "blue", "green", "orange", "purple"]

                for i, ablation_type in enumerate(ablation_types):
                    layers = layer_data_by_depth[ablation_type]
                    layer_names = [layer["name"] for layer in layers]
                    values = [layer["metrics"].get(metric_type, 0) for layer in layers]

                    bars = ax.bar(
                        x_pos + i * width,
                        values,
                        width,
                        label=ablation_type.replace("_", " ").title(),
                        **bar_kwargs(
                            colors[i % len(colors)], indexed_hatch(i), alpha=0.8
                        ),
                    )

                    # Add value labels on bars
                    for bar, val in zip(bars, values):
                        if val > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                bar.get_height() + 0.001,
                                f"{val:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )

                # Customize plot
                ax.set_xlabel("Layer (Soma to Distal)", fontsize=12)
                ax.set_ylabel(f'{metric_type.replace("_", " ").title()}', fontsize=12)
                ax.set_title(
                    f'Ablation Effects: {metric_type.replace("_", " ").title()} by Layer',
                    fontsize=14,
                )
                ax.set_xticks(x_pos + width * (len(ablation_types) - 1) / 2)
                ax.set_xticklabels(layer_names)
                ax.legend()
                ax.grid(True, alpha=0.3)

                plots.append(fig)

                # Save individual plot
                if save_path:
                    clean_metric = metric_type.replace(" ", "_").replace(" drop", "")
                    save_plot(fig, save_path, f"{filename_prefix}_{clean_metric}")

        logger.info(f"Generated {len(plots)} ablation plots")
        return plots

    except Exception as e:
        logger.error(f"Error in per-layer ablation plotting: {e}")
        return handle_plot_error("per-layer ablation plotting", e)


def plot_pruning_performance_comparison(
    performance_data: dict,
    save_path: Optional[str] = None,
    filename: str = "pruning_performance_comparison",
) -> plt.Figure:
    """
    Create a comprehensive bar plot comparing performance before/after pruning.

    Args:
        performance_data: Dict containing performance metrics for different stages:
            {
                'before_pruning': {...},
                'best_performance': {...},
                'last_epoch': {...},
                'after_pruning': {...}
            }
        save_path: Directory to save the plot
        filename: Base filename for the plot

    Returns:
        matplotlib Figure object
    """
    try:
        logger.info("Creating pruning performance comparison plot")

        if not performance_data:
            logger.warning("No performance data provided")
            return None

        # Extract available metrics from the data
        all_metrics = set()
        stages = ["before_pruning", "best_performance", "last_epoch", "after_pruning"]
        available_stages = []

        for stage in stages:
            if performance_data.get(stage):
                available_stages.append(stage)
                all_metrics.update(performance_data[stage].keys())

        if not available_stages or not all_metrics:
            logger.warning("No valid performance data found")
            return None

        # Remove any non-numeric metrics
        numeric_metrics = []
        for metric in all_metrics:
            try:
                # Check if all values for this metric are numeric
                values = []
                for stage in available_stages:
                    if metric in performance_data[stage]:
                        val = performance_data[stage][metric]
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            values.append(val)
                if values:  # Only include if we have at least one valid value
                    numeric_metrics.append(metric)
            except (KeyError, TypeError, ValueError):
                continue

        if not numeric_metrics:
            logger.warning("No numeric metrics found")
            return None

        # Create the plot
        n_metrics = len(numeric_metrics)
        n_stages = len(available_stages)

        fig, axes = create_subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        # Color scheme for different stages
        colors = {
            "before_pruning": "#e74c3c",  # Red
            "best_performance": "#2ecc71",  # Green
            "last_epoch": "#f39c12",  # Orange
            "after_pruning": "#3498db",  # Blue
        }

        # Stage labels for display
        stage_labels = {
            "before_pruning": "Before\nPruning",
            "best_performance": "Best\nPerformance",
            "last_epoch": "Last\nEpoch",
            "after_pruning": "After\nPruning",
        }

        np.arange(n_stages)
        width = 0.6

        for i, metric in enumerate(numeric_metrics):
            ax = axes[i]

            # Collect values for this metric across stages
            values = []
            stage_names = []
            bar_colors = []

            for stage in available_stages:
                if metric in performance_data[stage]:
                    val = performance_data[stage][metric]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)
                        stage_names.append(stage_labels[stage])
                        bar_colors.append(colors[stage])
                    else:
                        values.append(0)
                        stage_names.append(stage_labels[stage])
                        bar_colors.append("#95a5a6")  # Gray for missing data
                else:
                    values.append(0)
                    stage_names.append(stage_labels[stage])
                    bar_colors.append("#95a5a6")  # Gray for missing data

            # Create the bar plot
            bars = ax.bar(
                range(len(values)), values, width, color=bar_colors, alpha=0.8
            )

            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(values) * 0.01,
                        f"{value:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            # Formatting
            ax.set_xlabel("Training Stage")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(range(len(stage_names)))
            ax.set_xticklabels(stage_names, rotation=0, ha="center")
            ax.grid(True, alpha=0.3, axis="y")

            # Set y-axis limits for better visualization
            if values and max(values) > 0:
                ax.set_ylim(0, max(values) * 1.1)

        plt.suptitle("Performance Comparison: Training vs Pruning Effects", fontsize=16)
        plt.tight_layout()

        # Save the plot
        if save_path:
            save_plot(fig, save_path, filename)

        return fig

    except Exception as e:
        logger.error(f"Error in plot_pruning_performance_comparison: {e}")
        return handle_plot_error("pruning performance comparison plotting", e)


def plot_weight_distributions(weight_data: dict[str, list[np.ndarray]], **kwargs):
    """Plot weight distributions for different layer types."""
    logger.info("Plotting weight distributions")
    try:
        fig, ax = create_figure(figsize=(10, 6))

        if weight_data:
            for index, (name, weights) in enumerate(weight_data.items()):
                if weights:
                    # Flatten all weight arrays; outlines in their own line
                    # styles keep overlapping histograms apart without colour.
                    flat_weights = np.concatenate([w.flatten() for w in weights])
                    ax.hist(
                        flat_weights,
                        bins=50,
                        histtype="step",
                        linewidth=2,
                        linestyle=indexed_linestyle(index),
                        color=indexed_style(index).color,
                        label=name,
                    )
            setup_basic_plot(ax, "Weight Distributions", "Weight Value", "Frequency")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Weight Distributions\n(No data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Weight Distributions")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "weight_distributions")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax
    except Exception as e:
        return handle_plot_error("weight distribution plotting", e)


_TURNOVER_PATHWAY_STYLES = {
    "exc": "-",
    "inh": "--",
    "rec_exc": "-.",
    "rec_inh": ":",
}
_TURNOVER_COUNT_STYLES = {
    "formed": ("o", "#0072B2"),
    "eliminated": ("v", "#D55E00"),
}


def plot_synapse_turnover(results: dict) -> plt.Figure:
    """Plot contacts formed and eliminated per pathway between snapshots.

    ``results`` is the output of ``SynapticTurnoverAnalyzer.results()``.
    Pathways differ by line style and counts by marker and colour, so the
    panels read without colour.
    """
    epochs = results["snapshot_epochs"][1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mode, title in (
        (axes[0], "incremental", "Since the previous snapshot"),
        (axes[1], "cumulative", "Since the first snapshot"),
    ):
        for pathway, linestyle in _TURNOVER_PATHWAY_STYLES.items():
            counts = results[mode].get(pathway)
            if counts is None:
                continue
            for count, (marker, color) in _TURNOVER_COUNT_STYLES.items():
                ax.plot(
                    epochs,
                    counts[count],
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                    label=f"{pathway} {count}",
                )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Contacts")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_layer_synapse_statistics(results: dict) -> plt.Figure:
    """Plot each branch layer's cumulative turnover against epoch.

    ``results`` is the output of ``SynapticTurnoverAnalyzer.results()``; one
    panel per branch layer shows the contacts formed and eliminated since the
    first snapshot for each pathway. Pathways differ by line style and counts
    by marker and colour.
    """
    epochs = results["snapshot_epochs"][1:]
    names = list(results["modules"])
    n_cols = min(3, max(len(names), 1))
    n_rows = max(1, int(np.ceil(len(names) / n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for ax, name in zip(axes.flat, names):
        module = results["modules"][name]
        for pathway, counts in module["cumulative"].items():
            linestyle = _TURNOVER_PATHWAY_STYLES.get(pathway, "-")
            for count, (marker, color) in _TURNOVER_COUNT_STYLES.items():
                ax.plot(
                    epochs,
                    counts[count],
                    linestyle=linestyle,
                    marker=marker,
                    color=color,
                    markersize=3,
                    label=f"{pathway} {count}",
                )
        ax.set_title(f"{name}\n{module['population']}", fontsize=7)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Contacts since first snapshot", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, frameon=False)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


_CENSUS_PATHWAY_STYLES = {
    "exc": ("-", "o", "#0072B2"),
    "inh": ("--", "v", "#D55E00"),
    "rec_exc": ("-.", "s", "#009E73"),
    "rec_inh": (":", "D", "#CC79A7"),
}


def _census_branch_counts(census: dict, name: str, pathway: str) -> list[int] | None:
    module = census["modules"].get(name)
    if module is None or pathway not in module["pathways"]:
        return None
    return module["pathways"][pathway]["realized_per_branch"]


def plot_active_synapses_per_branch(
    census: dict, before: dict | None = None
) -> plt.Figure:
    """Plot the distribution of realized contacts per branch, per branch layer.

    ``census`` (and the optional pre-pruning ``before``) come from
    :func:`~dendritic_modeling.analysis.utils.connectivity_census.connectivity_census`.
    Pathways differ by line style and colour; the pre-pruning distributions
    are drawn thinner.
    """
    names = list(census["modules"])
    n_cols = min(3, max(len(names), 1))
    n_rows = max(1, int(np.ceil(len(names) / n_cols)))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.0 * n_rows), squeeze=False
    )
    for ax, name in zip(axes.flat, names):
        for pathway, (linestyle, _marker, color) in _CENSUS_PATHWAY_STYLES.items():
            for phase, source, width in (
                ("after", census, 1.8),
                ("before", before, 0.8),
            ):
                if source is None:
                    continue
                counts = _census_branch_counts(source, name, pathway)
                if not counts:
                    continue
                bins = np.arange(0, max(counts) + 2) - 0.5
                label = pathway if before is None else f"{pathway} ({phase} pruning)"
                ax.hist(
                    counts,
                    bins=bins,
                    histtype="step",
                    linestyle=linestyle,
                    linewidth=width,
                    color=color,
                    label=label,
                )
        module = census["modules"][name]
        ax.set_title(
            f"{name}\n{module['population']}, depth {module['soma_relative_depth']}",
            fontsize=7,
        )
        ax.set_xlabel("Realized contacts per branch", fontsize=7)
        ax.set_ylabel("Branches", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, frameon=False)
    for ax in list(axes.flat)[len(names) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_active_synapses_summary_by_layer(
    census: dict, before: dict | None = None
) -> plt.Figure:
    """Plot the mean and SD of realized contacts per branch against depth.

    One panel per population label; branch layers of the same population and
    soma-relative depth are pooled. Pathways differ by marker, line style and
    colour; pre-pruning values are drawn with hollow markers.
    """
    populations = sorted({m["population"] for m in census["modules"].values()})
    fig, axes = plt.subplots(
        1, len(populations), figsize=(4.5 * len(populations), 3.4), squeeze=False
    )
    for ax, population in zip(axes[0], populations):
        for pathway, (linestyle, marker, color) in _CENSUS_PATHWAY_STYLES.items():
            for phase, source in (("after", census), ("before", before)):
                if source is None:
                    continue
                pooled: dict[int, list[int]] = {}
                for name, module in source["modules"].items():
                    counts = _census_branch_counts(source, name, pathway)
                    if module["population"] != population or not counts:
                        continue
                    pooled.setdefault(module["soma_relative_depth"], []).extend(counts)
                if not pooled:
                    continue
                depths = sorted(pooled)
                means = [float(np.mean(pooled[d])) for d in depths]
                stds = [float(np.std(pooled[d])) for d in depths]
                ax.errorbar(
                    depths,
                    means,
                    yerr=stds,
                    linestyle=linestyle if phase == "after" else ":",
                    marker=marker,
                    markerfacecolor=color if phase == "after" else "none",
                    color=color,
                    capsize=2,
                    label=pathway if before is None else f"{pathway} ({phase} pruning)",
                )
        ax.set_title(population, fontsize=8)
        ax.set_xlabel("Soma-relative depth (0 = soma)", fontsize=7)
        ax.set_ylabel("Realized contacts per branch", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, frameon=False)
    fig.tight_layout()
    return fig


def plot_performance_metrics(performance_data: dict, **kwargs):
    """Plot performance metrics as a bar chart."""
    try:
        fig, ax = create_figure(figsize=(10, 6))

        if performance_data:
            # Handle nested dictionary structure from PerformanceAnalyzer
            flattened_metrics = {}
            for metric_name, metric_data in performance_data.items():
                if isinstance(metric_data, dict):
                    # Handle nested structure (metric -> dataset -> value)
                    for dataset, value in metric_data.items():
                        if isinstance(value, (int, float)):
                            flattened_metrics[f"{metric_name}_{dataset}"] = value
                elif isinstance(metric_data, (int, float)):
                    # Handle direct values
                    flattened_metrics[metric_name] = metric_data

            if flattened_metrics:
                metrics = list(flattened_metrics.keys())
                values = list(flattened_metrics.values())
                ax.bar(metrics, values)
                setup_basic_plot(ax, "Performance Metrics", "Metric", "Value")
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid performance data found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Performance Metrics")
        else:
            ax.text(
                0.5,
                0.5,
                "No performance data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Metrics")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "performance_metrics")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax

    except Exception as e:
        return handle_plot_error("performance metrics plotting", e)


def plot_performance_evolution(
    performance_dir: str,
    save_path: Optional[str] = None,
    filename: str = "performance_evolution",
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """Generate performance evolution plots from per-epoch performance files."""
    try:
        plots = {}

        # Look for performance files in the directory
        if not os.path.exists(performance_dir):
            logger.warning(f"Performance directory does not exist: {performance_dir}")
            return {}

        # Find all performance files (assuming JSON format with epoch data)
        perf_files = glob.glob(os.path.join(performance_dir, "*performance*.json"))

        if not perf_files:
            logger.warning(f"No performance files found in {performance_dir}")
            # Create placeholder plot
            fig, ax = create_figure(figsize=(12, 8))
            ax.text(
                0.5,
                0.5,
                "No performance data found in directory",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Evolution Over Time")
            plots["evolution"] = (fig, ax)
            return plots

        # Load and parse performance data
        all_data = {}
        epochs = []

        for file_path in sorted(perf_files):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                # Extract epoch number from filename or data
                if "epoch" in data:
                    epoch = data["epoch"]
                else:
                    # Try to extract from filename
                    match = re.search(r"epoch_(\d+)", file_path)
                    if match:
                        epoch = int(match.group(1))
                    else:
                        continue

                epochs.append(epoch)

                # Store metrics
                for metric, value in data.items():
                    if metric != "epoch" and isinstance(value, (int, float)):
                        if metric not in all_data:
                            all_data[metric] = []
                        all_data[metric].append(value)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not parse performance file {file_path}: {e}")
                continue

        if not all_data or not epochs:
            logger.warning("No valid performance data found")
            fig, ax = create_figure(figsize=(12, 8))
            ax.text(
                0.5,
                0.5,
                "No valid performance data found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Evolution Over Time")
            plots["evolution"] = (fig, ax)
            return plots

        # Sort by epoch
        sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
        epochs = [epochs[i] for i in sorted_indices]
        for metric in all_data:
            all_data[metric] = [all_data[metric][i] for i in sorted_indices]

        # Create evolution plot
        fig, ax = create_figure(figsize=(12, 8))

        # Plot each metric with its own marker and line style
        for index, (metric, values) in enumerate(all_data.items()):
            if len(values) == len(epochs):
                ax.plot(
                    epochs,
                    values,
                    **indexed_style(index).kwargs(),
                    label=metric,
                    linewidth=2,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance Metric Value")
        ax.set_title("Performance Evolution Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plots["evolution"] = (fig, ax)

        if save_path:
            save_plot(fig, save_path, filename)

        return plots

    except Exception as e:
        logger.warning(f"Error in performance evolution plotting: {e}")
        return {}


# Training and Sweep Plotting Functions
def plot_loss_curves(
    train_losses: list[float],
    valid_losses: list[float],
    loss_name: str,
    save_dir: Optional[str] = None,
    filename_prefix: str = "",
    train_losses_base: Optional[list[float]] = None,
    train_losses_reg: Optional[list[float]] = None,
):
    """Plot training and validation loss curves with optional regularization breakdown."""
    try:
        fig, ax = create_figure(figsize=(12, 8))

        epochs = range(1, len(train_losses) + 1)

        # Every curve has its own line style besides its colour.
        train_color = split_style("train").color

        # Plot main losses
        ax.plot(
            epochs,
            train_losses,
            color=train_color,
            linestyle="-",
            linewidth=2,
            label="Training Loss (Total)",
        )

        if valid_losses:
            epochs_val = range(1, len(valid_losses) + 1)
            ax.plot(
                epochs_val,
                valid_losses,
                color=split_style("valid").color,
                linestyle="--",
                linewidth=2,
                label="Validation Loss",
            )

        # Plot separate components if available
        if train_losses_base is not None:
            ax.plot(
                epochs,
                train_losses_base,
                color=train_color,
                linestyle="-.",
                alpha=0.7,
                linewidth=1.5,
                label="Training Loss (Base)",
            )

        if train_losses_reg is not None and any(reg > 1e-6 for reg in train_losses_reg):
            ax.plot(
                epochs,
                train_losses_reg,
                color=indexed_style(2).color,
                linestyle=":",
                alpha=0.8,
                linewidth=1.5,
                label="Regularization Loss",
            )

        setup_basic_plot(ax, f"{loss_name} Loss Curves", "Epoch", "Loss")
        ax.legend()

        # Add text box with final values if regularization is present
        if train_losses_base is not None and train_losses_reg is not None:
            final_total = train_losses[-1]
            final_base = train_losses_base[-1]
            final_reg = train_losses_reg[-1]
            final_valid = valid_losses[-1] if valid_losses else 0

            textstr = f"Final Values:\nTrain (Total): {final_total:.4f}\nTrain (Base): {final_base:.4f}\nReg: {final_reg:.4f}"
            if valid_losses:
                textstr += f"\nValid: {final_valid:.4f}"

            props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
            ax.text(
                0.02,
                0.98,
                textstr,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=props,
            )

        if save_dir:
            save_plot(fig, save_dir, f"{filename_prefix}loss_curves")

        return fig, ax

    except Exception as e:
        return handle_plot_error("loss curves plotting", e)


def plot_branch_sweep_results(results_csv_path, split="test"):
    """Plot branch sweep results."""
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Branch Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs number of branches
    axes[0, 0].plot(split_df["n_branches"], split_df["accuracy"], "o-", color="blue")
    setup_basic_plot(
        axes[0, 0], "Accuracy vs Number of Branches", "Number of Branches", "Accuracy"
    )

    # Plot loss vs number of branches
    axes[0, 1].plot(split_df["n_branches"], split_df["loss"], "o-", color="red")
    setup_basic_plot(
        axes[0, 1], "Loss vs Number of Branches", "Number of Branches", "Loss"
    )

    # Plot precision vs number of branches
    axes[1, 0].plot(split_df["n_branches"], split_df["precision"], "o-", color="green")
    setup_basic_plot(
        axes[1, 0], "Precision vs Number of Branches", "Number of Branches", "Precision"
    )

    # Plot recall vs number of branches
    axes[1, 1].plot(split_df["n_branches"], split_df["recall"], "o-", color="orange")
    setup_basic_plot(
        axes[1, 1], "Recall vs Number of Branches", "Number of Branches", "Recall"
    )

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"branch_sweep_results_{split}.png"
    )
    save_plot(fig, os.path.dirname(results_csv_path), f"branch_sweep_results_{split}")

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes


def plot_ei_sweep_results(results_csv_path, ei_ratio_range=None, split="test"):
    """Plot E/I ratio sweep results."""

    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"E/I Ratio Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs E/I ratio
    axes[0, 0].plot(split_df["ei_ratio"], split_df["accuracy"], "o-", color="blue")
    setup_basic_plot(axes[0, 0], "Accuracy vs E/I Ratio", "E/I Ratio", "Accuracy")

    # Plot loss vs E/I ratio
    axes[0, 1].plot(split_df["ei_ratio"], split_df["loss"], "o-", color="red")
    setup_basic_plot(axes[0, 1], "Loss vs E/I Ratio", "E/I Ratio", "Loss")

    # Plot precision vs E/I ratio
    axes[1, 0].plot(split_df["ei_ratio"], split_df["precision"], "o-", color="green")
    setup_basic_plot(axes[1, 0], "Precision vs E/I Ratio", "E/I Ratio", "Precision")

    # Plot recall vs E/I ratio
    axes[1, 1].plot(split_df["ei_ratio"], split_df["recall"], "o-", color="orange")
    setup_basic_plot(axes[1, 1], "Recall vs E/I Ratio", "E/I Ratio", "Recall")

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"ei_sweep_results_{split}.png"
    )
    save_plot(fig, os.path.dirname(results_csv_path), f"ei_sweep_results_{split}")

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes


def plot_noise_sweep_branch_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for branch networks."""

    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Noise Sweep Results - {noise_type.title()} Noise (Branch Networks)",
        fontsize=16,
    )

    # Plot accuracy vs noise level
    for index, n_branches in enumerate(noise_df["n_branches"].unique()):
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 0].plot(
            branch_data["noise_level"],
            branch_data["accuracy"],
            **indexed_style(index).kwargs(),
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[0, 0], "Accuracy vs Noise Level", "Noise Level", "Accuracy")
    axes[0, 0].legend()

    # Plot loss vs noise level
    for index, n_branches in enumerate(noise_df["n_branches"].unique()):
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 1].plot(
            branch_data["noise_level"],
            branch_data["loss"],
            **indexed_style(index).kwargs(),
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[0, 1], "Loss vs Noise Level", "Noise Level", "Loss")
    axes[0, 1].legend()

    # Plot precision vs noise level
    for index, n_branches in enumerate(noise_df["n_branches"].unique()):
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 0].plot(
            branch_data["noise_level"],
            branch_data["precision"],
            **indexed_style(index).kwargs(),
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[1, 0], "Precision vs Noise Level", "Noise Level", "Precision")
    axes[1, 0].legend()

    # Plot recall vs noise level
    for index, n_branches in enumerate(noise_df["n_branches"].unique()):
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 1].plot(
            branch_data["noise_level"],
            branch_data["recall"],
            **indexed_style(index).kwargs(),
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[1, 1], "Recall vs Noise Level", "Noise Level", "Recall")
    axes[1, 1].legend()

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path),
        f"noise_sweep_branch_results_{noise_type}.png",
    )
    save_plot(
        fig,
        os.path.dirname(results_csv_path),
        f"noise_sweep_branch_results_{noise_type}",
    )

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes


def plot_noise_sweep_ei_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for E/I networks."""

    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Noise Sweep Results - {noise_type.title()} Noise (E/I Networks)", fontsize=16
    )

    # Plot accuracy vs noise level
    for index, ei_ratio in enumerate(noise_df["ei_ratio"].unique()):
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 0].plot(
            ei_data["noise_level"],
            ei_data["accuracy"],
            **indexed_style(index).kwargs(),
            label=f"E/I_var = {ei_ratio}",
        )
    setup_basic_plot(axes[0, 0], "Accuracy vs Noise Level", "Noise Level", "Accuracy")
    axes[0, 0].legend()

    # Plot loss vs noise level
    for index, ei_ratio in enumerate(noise_df["ei_ratio"].unique()):
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 1].plot(
            ei_data["noise_level"],
            ei_data["loss"],
            **indexed_style(index).kwargs(),
            label=f"E/I_var = {ei_ratio}",
        )
    setup_basic_plot(axes[0, 1], "Loss vs Noise Level", "Noise Level", "Loss")
    axes[0, 1].legend()

    # Plot precision vs noise level
    for index, ei_ratio in enumerate(noise_df["ei_ratio"].unique()):
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 0].plot(
            ei_data["noise_level"],
            ei_data["precision"],
            **indexed_style(index).kwargs(),
            label=f"E/I_var = {ei_ratio}",
        )
    setup_basic_plot(axes[1, 0], "Precision vs Noise Level", "Noise Level", "Precision")
    axes[1, 0].legend()

    # Plot recall vs noise level
    for index, ei_ratio in enumerate(noise_df["ei_ratio"].unique()):
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 1].plot(
            ei_data["noise_level"],
            ei_data["recall"],
            **indexed_style(index).kwargs(),
            label=f"E/I_var = {ei_ratio}",
        )
    setup_basic_plot(axes[1, 1], "Recall vs Noise Level", "Noise Level", "Recall")
    axes[1, 1].legend()

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"noise_sweep_ei_results_{noise_type}.png"
    )
    save_plot(
        fig, os.path.dirname(results_csv_path), f"noise_sweep_ei_results_{noise_type}"
    )

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes
