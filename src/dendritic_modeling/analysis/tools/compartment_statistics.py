"""
Unified Weight Analysis Module.

This module provides comprehensive analysis of synaptic weights across the dendritic network,
including both global statistics and hierarchical per-layer analysis with plotting capabilities.
Provides comprehensive weight analysis with both global and hierarchical statistics.
"""

import os

import traceback
import logging
import time
from typing import Any, Optional

import numpy as np
import torch
import pandas as pd
import seaborn as sns
from scipy.stats import differential_entropy
from math import exp

import matplotlib.pyplot as plt

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import CompartmentStatisticsAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    TopKLinear,
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict

from dendritic_modeling.plotting.visualizations.synaptic_weight_plots import (
    plot_synaptic_weight_analysis,
)
from dendritic_modeling.plotting.visualizations.plotting_utils import (
    convert_layer_names,
    get_color_scheme,
    setup_basic_plot,
)

logger = logging.getLogger(__name__)


class CompartmentStatisticsAnalyzer(AbstractAnalyzer):
    """
    Unified analyzer for comprehensive synaptic weight analysis.

    Provides both global weight statistics and hierarchical per-layer analysis
    with enhanced plotting capabilities including min/max and dendritic strength plots.
    """

    def __init__(self, params: CompartmentStatisticsAnalysisParams):
        """Initialize the unified compartment statistics analyzer."""
        super().__init__("CompartmentStatisticsAnalyzer")

        self.synapse_weights = getattr(params, "synapse_weights", True)
        self.branch_weights = getattr(params, "branch_weights", True)
        self.inputs = getattr(params, "inputs", True)
        self.activations = getattr(params, "activations", True)

        # Enhanced analysis modes
        self.global_analysis = getattr(params, "global_analysis", True)
        self.layer_analysis = getattr(params, "layer_analysis", True)
        self.branch_analysis = getattr(params, "branch_analysis", True)

        self.compute_mean = getattr(params, "compute_mean", True)
        self.compute_variance = getattr(params, "compute_variance", True)
        self.compute_percentiles = getattr(params, "compute_percentiles", True)
        self.compute_min_max = getattr(params, "compute_min_max", True)

        self.compute_entropy = getattr(params, "compute_entropy", True)
        self.dequantize = getattr(params, "dequantize", True)
        # self.n_neighbors = getattr(params, "n_neighbors", 10)

        self.n_samples = getattr(params, "n_samples", None)
        self.epsilon = torch.finfo(torch.float32).eps
    
    def collect_raw_data(self, model: BaseModel, x: torch.Tensor):
        """Collect raw data from the model."""
        self.raw_dict = {}
        handles = []
        for key, module in model.named_modules():
            if (
                isinstance(module, DendriticBranchLayer)
                and module.branch_excitation is not None
                and module.branch_inhibition is not None
            ):
                self.raw_dict[key] = {}
                for branch_idx in range(module.n_branches):
                    self.raw_dict[key][branch_idx] = {"depth": module.layer_idx}
                
                if self.synapse_weights:
                    exc_weights = module.branch_excitation.pruned_weight().detach().cpu()
                    inh_weights = module.branch_inhibition.pruned_weight().detach().cpu()
                    for branch_idx in range(module.n_branches):
                        exc_w = exc_weights[branch_idx]
                        exc_w = exc_w[exc_w > self.epsilon]
                        self.raw_dict[key][branch_idx]["exc_weights"] = exc_w

                        inh_w = inh_weights[branch_idx]
                        inh_w = inh_w[inh_w > self.epsilon]
                        self.raw_dict[key][branch_idx]["inh_weights"] = inh_w
                
                if self.branch_weights and hasattr(module, "branches_to_output"):
                    branch_weights = module.branches_to_output.weight().detach().cpu()
                    for branch_idx in range(module.n_branches):
                        branch_w = branch_weights[branch_idx]
                        branch_w = branch_w[branch_w > self.epsilon]
                        self.raw_dict[key][branch_idx]["branch_weights"] = branch_w
                
                if self.inputs or self.activations:
                    handle = module.branch_excitation.register_forward_hook(
                        lambda module, input, output, layer_key=key: self.forward_hook(
                            module, input, output, layer_key, "exc"
                        )
                    )
                    handles.append(handle)
                    handle = module.branch_inhibition.register_forward_hook(
                        lambda module, input, output, layer_key=key: self.forward_hook(
                            module, input, output, layer_key, "inh"
                        )
                    )
                    handles.append(handle)
        
        if self.inputs or self.activations:
            if self.n_samples is not None:
                x = x[torch.randperm(x.shape[0])[:self.n_samples]]
            with torch.no_grad():
                _ = model(x)
            self.remove_forward_hooks(handles)
    
    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self, 
        module: TopKLinear, 
        input: tuple[torch.Tensor, ...], 
        output: torch.Tensor,
        key: str,
        synapse_type: str
    ):
        if self.inputs:
            weight_mask = module.weight_mask()
            
            for branch_idx in range(module.out_features):
                branch_mask = weight_mask[branch_idx]
                branch_x = input[0][..., branch_mask > self.epsilon].detach().cpu()
                self.raw_dict[key][branch_idx][f"{synapse_type}_inputs"] = branch_x
    
        if self.activations:
            for branch_idx in range(module.out_features):
                branch_activation = output[..., branch_idx].detach().cpu()
                self.raw_dict[key][branch_idx][
                    f"{synapse_type}_activations"
                ] = branch_activation

    def _compute_statistics(
        self, 
        data_tensor: torch.Tensor,
        prefix: str,
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        """Compute comprehensive statistics for a weight tensor."""

        if self.compute_mean:
            stats_dict[f"{prefix}_mean"] = data_tensor.mean().item()

        if self.compute_variance:
            stats_dict[f"{prefix}_var"] = data_tensor.var().item()
            stats_dict[f"{prefix}_std"] = data_tensor.std().item()

        if self.compute_min_max:
            stats_dict[f"{prefix}_min"] = data_tensor.min().item()
            stats_dict[f"{prefix}_max"] = data_tensor.max().item()

        if self.compute_percentiles:
            stats_dict[f"{prefix}_q1"] = data_tensor.quantile(0.25).item()
            stats_dict[f"{prefix}_q2"] = data_tensor.median().item()
            stats_dict[f"{prefix}_q3"] = data_tensor.quantile(0.75).item()

        return stats_dict
    
    def _compute_branch_statistics(self):
        branch_statistics = {}
        for layer_key, layer_data in self.raw_dict.items():
            branch_statistics[layer_key] = {}
            layer_data: dict[int, dict[str, torch.Tensor]]
            for branch_idx, branch_data in layer_data.items():
                branch_stats = {}

                for prefix, data_tensor in branch_data.items():
                    if prefix == "depth":
                        branch_stats[prefix] = data_tensor  # data_tensor is the depth index
                    else:
                        branch_stats = self._compute_statistics(
                            data_tensor, prefix, branch_stats
                        )
                    
                    if self.compute_entropy:
                        if self.dequantize and (
                            "inputs" in prefix 
                            or "activations" in prefix
                        ):
                            u = torch.unique(data_tensor.flatten())
                            step = torch.median(u[1:] - u[:-1])
                            data_tensor = data_tensor + ((torch.rand_like(data_tensor) - 0.5) * step)
                        if "inputs" in prefix:
                            branch_stats[f"{prefix}_entropy"] = differential_entropy(
                                data_tensor.flatten(0,-2).mean(dim=-1).numpy(),
                                method="vasicek"
                            )
                        elif "activations" in prefix:
                            branch_stats[f"{prefix}_entropy"] = differential_entropy(
                                data_tensor.flatten(0,-1).numpy(),
                                method="vasicek"
                            )
                        if "inputs" in prefix or "activations" in prefix:
                            if np.isnan(branch_stats[f"{prefix}_entropy"]) or np.isinf(branch_stats[f"{prefix}_entropy"]):
                                print(branch_stats, flush=True)
                                print(data_tensor, flush=True)
                branch_statistics[layer_key][branch_idx] = branch_stats
        
        return branch_statistics

    def _compute_layer_statistics(
        self, 
        branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute layer statistics from branch statistics."""
        layer_statistics = {}
        for layer_key, layer_data in branch_statistics.items():
            layer_aggregated: dict[str, list[float]] = {}

            for branch_data in layer_data.values():
                for field, value in branch_data.items():
                    if field not in layer_aggregated.keys() and field != "depth":
                        layer_aggregated[field] = []
                    if field != "depth":
                        layer_aggregated[field].append(value)
            
            layer_stats = {"depth": branch_data["depth"]}
            for prefix, values in layer_aggregated.items():
                data_tensor = torch.tensor(values)
                layer_stats = self._compute_statistics(
                    data_tensor, prefix, layer_stats
                )
                
            layer_statistics[layer_key] = layer_stats
        return layer_statistics

    def _compute_global_statistics(
        self,
        branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute global statistics from branch statistics."""
        global_aggregated: dict[str, list[float]] = {}
        for layer_data in branch_statistics.values():
            for branch_data in layer_data.values():
                for field, value in branch_data.items():
                    if field not in global_aggregated.keys() and field != "depth":
                        global_aggregated[field] = []
                    if field != "depth":
                        global_aggregated[field].append(value)
        
        global_statistics = {}
        for prefix, values in global_aggregated.items():
            data_tensor = torch.tensor(values)
            global_statistics = self._compute_statistics(
                data_tensor, prefix, global_statistics
            )
        return global_statistics
    
    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False
    ):
        """
        Perform comprehensive weight analysis including global and hierarchical statistics.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            save_path: Path to save results (optional)
            filename: Filename for saved results
            training: If True, only save data (no plots). If False, generate plots.

        Returns:
            Dictionary containing all weight analysis results
        """
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            self.logger.error(
                "Model must have ExcitationInhibitionNetwork for weight analysis"
            )
            return None

        self.logger.info("Starting comprehensive excitation-inhibition analysis...")        
        start_time = time.time()

        model.to(device)
        x, _ = test_dataset[:]
        x: torch.Tensor = x.to(device)
        
        self.collect_raw_data(model, x)
        branch_statistics = self._compute_branch_statistics()

        results = {}

        if self.global_analysis:
            global_statistics = self._compute_global_statistics(branch_statistics)
            results["global_statistics"] = global_statistics
        
        if self.layer_analysis:
            layer_statistics = self._compute_layer_statistics(branch_statistics)
            results["layer_statistics"] = layer_statistics
        
        if self.branch_analysis:
            results["branch_statistics"] = branch_statistics

        elapsed_time = time.time() - start_time
        self.logger.info(f"Weight analysis completed in {elapsed_time:.2f} seconds")

        # Save results and generate plots
        if save_path is not None:
            self.logger.info(
                f"Saving weight analysis results to {save_path}/{filename}"
            )
            save_dict(results, save_path, filename)

            plot_synapse_weight_comparison(
                fields=["weights", "inputs", "activations"], 
                summ_stats=["mean", "q2", "std"], 
                branch_statistics=branch_statistics, 
                save_path=save_path
            )
            plot_ratio_comparison(
                branch_statistics=branch_statistics, 
                save_path=save_path
            )

            # Only generate plots during final analysis (not during training)
            if not training:
                self.logger.info("Starting plot generation...")
                self._generate_all_plots(results, save_path, model)
                self.logger.info("Plot generation completed!")
            else:
                self.logger.info("Training mode: plots skipped, only data saved")

        return results

    def _generate_all_plots(
        self, results: dict[str, Any], save_path: str, model: BaseModel
    ):
        """Generate all weight analysis plots."""
        self.logger.info("Generating weight analysis plots...")
        self.logger.info(f"Results keys: {list(results.keys())}")
        self.logger.info(f"Save path: {save_path}")

        try:
            # Get somatic synapses setting
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            self.logger.info(f"Somatic synapses: {somatic_synapses}")

            # Generate hierarchical plots (original synaptic_weight_expectation plots)
            if "layer_statistics" in results:
                self.logger.info("Generating hierarchical weight plots...")
                self._plot_hierarchical_weights(
                    results["layer_statistics"], save_path, somatic_synapses
                )
                self.logger.info("Hierarchical plots completed")
            else:
                self.logger.warning("No layer_statistics found in results")

            # Generate enhanced plots (min/max and dendritic strength)
            if "enhanced_layer_statistics" in results:
                self.logger.info("Generating enhanced statistics plots...")
                self._plot_enhanced_statistics(
                    results["enhanced_layer_statistics"], save_path, somatic_synapses
                )
                self.logger.info("Enhanced plots completed")
            else:
                self.logger.warning("No enhanced_layer_statistics found in results")

            # Generate global distribution plots
            if "global_statistics" in results:
                self.logger.info("Generating global distribution plots...")
                self._plot_global_distributions(results["global_statistics"], save_path)
                self.logger.info("Global plots completed")
            else:
                self.logger.warning("No global_statistics found in results")

        except Exception as e:
            self.logger.error(f"Error generating weight plots: {e}")

            traceback.print_exc()

    def _plot_hierarchical_weights(
        self, layer_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate all original synaptic weight plots plus the simplified ones."""

        # Generate all the original plots that synaptic_weight_expectation analyzer created
        plot_synaptic_weight_analysis(
            layer_stats,
            save_path=save_path,
            somatic_synapses=somatic_synapses,
        )

    def _plot_enhanced_statistics(
        self, enhanced_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate enhanced plots for min/max and dendritic strength."""
        try:

            # Convert layer names for proper ordering
            converted_results = convert_layer_names(
                enhanced_stats, somatic_synapses=somatic_synapses
            )
            layer_names = list(converted_results.keys())
            colors = get_color_scheme()

            # Create 2-panel plot for enhanced statistics
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Collect data
            depths = []
            min_weights = []
            max_weights = []
            dendritic_strengths = []

            for layer_name in layer_names:
                layer_data = converted_results[layer_name]
                if (
                    "layer_weight_min" in layer_data
                    and "layer_weight_max" in layer_data
                ):
                    depths.append(layer_name)
                    min_weights.append(layer_data["layer_weight_min"])
                    max_weights.append(layer_data["layer_weight_max"])
                    dendritic_strengths.append(
                        layer_data.get("dendritic_strength", 0.0)
                    )

            if depths:
                x_pos = np.arange(len(depths))
                width = 0.35

                # Panel 1: Min/Max weights per layer
                ax1 = axes[0]
                ax1.bar(
                    x_pos - width / 2,
                    min_weights,
                    width,
                    label="Min Weight",
                    color=colors["excitatory"],
                    alpha=0.7,
                )
                ax1.bar(
                    x_pos + width / 2,
                    max_weights,
                    width,
                    label="Max Weight",
                    color=colors["inhibitory"],
                    alpha=0.7,
                )

                setup_basic_plot(
                    ax1,
                    "Min/Max Weights per Layer",
                    "Layer (Soma → Distal)",
                    "Weight Value",
                    grid=True,
                )
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(depths, rotation=45)
                ax1.legend()

                # Panel 2: Dendritic strength per layer
                ax2 = axes[1]
                ax2.bar(
                    x_pos,
                    dendritic_strengths,
                    width * 2,
                    color=colors["mixed"],
                    alpha=0.7,
                )

                setup_basic_plot(
                    ax2,
                    "Total Dendritic Strength per Layer",
                    "Layer (Soma → Distal)",
                    "Total Weight Sum",
                    grid=True,
                )
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(depths, rotation=45)

            plt.suptitle(
                "Enhanced Weight Analysis: Min/Max and Dendritic Strength", fontsize=16
            )
            plt.tight_layout()

            # Save plot
            filename = os.path.join(save_path, "enhanced_weight_analysis.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Saved enhanced weight analysis plot: {filename}")

        except Exception as e:
            self.logger.error(f"Error generating enhanced plots: {e}")

    def _plot_global_distributions(self, global_stats: dict[str, Any], save_path: str):
        """Generate global weight distribution plots."""
        try:
            # Create histogram plots for global distributions
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            weight_types = [
                "excitatory_weights",
                "inhibitory_weights",
                "branch_weights",
            ]
            titles = [
                "Excitatory Weight Distribution",
                "Inhibitory Weight Distribution",
                "Branch Weight Distribution",
            ]

            for i, (weight_type, title) in enumerate(zip(weight_types, titles)):
                if weight_type in global_stats:
                    stats = global_stats[weight_type]
                    ax = axes[i]

                    # Create a simple bar plot of statistics
                    metrics = ["mean", "std", "min", "max", "median"]
                    values = [stats.get(metric, 0) for metric in metrics]

                    ax.bar(metrics, values, alpha=0.7)
                    ax.set_title(title)
                    ax.set_ylabel("Weight Value")
                    ax.tick_params(axis="x", rotation=45)
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"No {weight_type.replace('_', ' ')}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )
                    axes[i].set_title(titles[i])

            plt.suptitle("Global Weight Statistics", fontsize=16)
            plt.tight_layout()

            # Save plot
            filename = os.path.join(save_path, "global_weight_distributions.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Saved global weight distributions plot: {filename}")

        except Exception as e:
            self.logger.error(f"Error generating global distribution plots: {e}")



def plot_synapse_weight_comparison(
    fields: list[str],
    summ_stats: list[str], 
    branch_statistics: dict[str, dict[int, dict[str, float]]], 
    save_path: str
):
    plot_data: dict[str, dict[str, list[float]]] = {}
    for layer_key, layer_data in branch_statistics.items():
        for branch_data in layer_data.values():
            depth = branch_data["depth"]
            if f"depth {depth}" not in plot_data.keys():
                plot_data[f"depth {depth}"] = {}
            for field in fields:
                for summ_stat in summ_stats:
                    if f"exc_{field}_{summ_stat}" not in plot_data[f"depth {depth}"].keys():
                        plot_data[f"depth {depth}"][f"exc_{field}_{summ_stat}"] = []
                    if f"inh_{field}_{summ_stat}" not in plot_data[f"depth {depth}"].keys():
                        plot_data[f"depth {depth}"][f"inh_{field}_{summ_stat}"] = []

                    plot_data[f"depth {depth}"][f"exc_{field}_{summ_stat}"].append(branch_data[f"exc_{field}_{summ_stat}"])
                    plot_data[f"depth {depth}"][f"inh_{field}_{summ_stat}"].append(branch_data[f"inh_{field}_{summ_stat}"])
    
    fig, axes = plt.subplots(len(fields), len(summ_stats), figsize=(6 * len(summ_stats), 4 + 6 * len(fields)))
    for i, field in enumerate(fields):
        for j, summ_stat in enumerate(summ_stats):
            for depth, data in plot_data.items():
                axes[i, j].scatter(
                    data[f"exc_{field}_{summ_stat}"], 
                    data[f"inh_{field}_{summ_stat}"], 
                    label=depth.title(), 
                    marker="o", s=5, alpha=0.5
                )
            axes[i, j].legend()
            axes[i, j].set_xlabel(f"Excitatory {field.title()} {summ_stat.title()}")
            axes[i, j].set_ylabel(f"Inhibitory {field.title()} {summ_stat.title()}")
            axes[i, j].grid(True)
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, "exc_inh_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_ratio_comparison(
    branch_statistics: dict[str, dict[int, dict[str, float]]], 
    save_path: str
):
    agg_data: dict[str, dict[str, list[float]]] = {}
    for layer_key, layer_data in branch_statistics.items():
        for branch_data in layer_data.values():
            depth = branch_data["depth"]
            if f"depth {depth}" not in agg_data.keys():
                agg_data[f"depth {depth}"] = {}
            
            if "ei_expected_act_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_expected_act_ratio"] = []
            if "ei_act_mean_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_act_mean_ratio"] = []

            if "ei_acts_eff_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_acts_eff_std_ratio"] = []
            if "ei_acts_emp_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_acts_emp_std_ratio"] = []
            
            ei_expected_act_ratio = (
                branch_data["exc_inputs_mean"] * branch_data["exc_weights_mean"]
                ) / (
                branch_data["inh_inputs_mean"] * branch_data["inh_weights_mean"] + 1e-8
            )
            agg_data[f"depth {depth}"][f"ei_expected_act_ratio"].append(ei_expected_act_ratio)
            
            ei_act_mean_ratio = branch_data["exc_activations_mean"] / (
                branch_data["inh_activations_mean"] + 1e-8
            )
            agg_data[f"depth {depth}"][f"ei_act_mean_ratio"].append(ei_act_mean_ratio)
            

            ei_acts_entropy_diff = branch_data["exc_activations_entropy"] - branch_data["inh_activations_entropy"]
            ei_acts_eff_std_ratio = exp(ei_acts_entropy_diff)
            agg_data[f"depth {depth}"][f"ei_acts_eff_std_ratio"].append(ei_acts_eff_std_ratio)
            agg_data[f"depth {depth}"][f"ei_acts_emp_std_ratio"].append(
                branch_data["exc_activations_std"] / (branch_data["inh_activations_std"] + 1e-8)
            )            
    

    indeps = ["ei_expected_act_ratio", "ei_act_mean_ratio"]
    deps = ["ei_acts_eff_std_ratio", "ei_acts_emp_std_ratio"]
    
    # Build long-form DataFrame from agg_data
    rows = []
    for depth, data in agg_data.items():
        for dep in deps:          # y
            for indep in indeps:  # x
                x = np.asarray(data[indep]); y = np.asarray(data[dep])
                rows.append(pd.DataFrame({
                    'x': x, 'y': y, 'depth': depth,
                    'row': dep, 'col': indep
                }))
    df = pd.concat(rows, ignore_index=True)

    # Labels
    xlabels = {
        "ei_expected_act_ratio": r"$\frac{X_{exc}}{X_{inh}}*\frac{W_{exc}}{W_{inh}}$",
        "ei_act_mean_ratio":     r"$\frac{E}{I}$",
    }
    ylabels = {
        "ei_acts_eff_std_ratio": r"$\sqrt{\frac{N(E)}{N(I)}}$",
        "ei_acts_emp_std_ratio": r"$\frac{\sigma_{E}}{\sigma_{I}}$",
    }

    # Scatter faceted by (row=dep, col=indep) with hue=depth
    g = sns.relplot(
        data=df, x='x', y='y', hue='depth',
        row='row', col='col', kind='scatter',
        row_order=deps, col_order=indeps,
        facet_kws=dict(sharex="col", sharey="row"),
        height=4, aspect=1.2, s=10, alpha=0.5
    )

    # Add one global regression line per facet and annotate slope/intercept
    def add_regline_and_stats(data: pd.DataFrame, color: Optional[str] = None, **kws):
        ax = plt.gca()
        x = data['x'].to_numpy()
        y = data['y'].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            slope, intercept = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(x[m].min(), x[m].max(), 200)
            ax.plot(xx, slope * xx + intercept, color='k', lw=2)
            ax.text(0.02, 0.98, f"y={slope:.2g}x+{intercept:.2g}",
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=9, color='k')

    g.map_dataframe(add_regline_and_stats)

    # Axis labels per row/col (only outer axes)
    for i, row_key in enumerate(deps):
        for j, col_key in enumerate(indeps):
            ax = g.axes[i, j]
            ax.grid(True)
            if i == len(deps) - 1:
                ax.set_xlabel(xlabels[col_key])
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(ylabels[row_key])
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    g.figure.savefig(os.path.join(save_path, "another_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(g.figure)

    # fig, axes = plt.subplots(2, 2, figsize=(12, 16), sharex="col", sharey="row")
    # for i, dep in enumerate(deps):
    #     for j, indep in enumerate(indeps):
    #         for depth, data in agg_data.items():
    #             axes[i, j].scatter(
    #                 data[indep], 
    #                 data[dep], 
    #                 label=depth.title(), 
    #                 marker="o", s=5, alpha=0.5
    #             )
    #         axes[i, j].legend()
    #         axes[i, j].grid(True)
    
    # axes[-1, 0].set_xlabel(r"$\frac{X_{exc}}{X_{inh}}*\frac{W_{exc}}{W_{inh}}$")
    # axes[-1, -1].set_xlabel(r"$\frac{E}{I}$")
    # axes[0, 0].set_ylabel(r"$\sqrt{\frac{N(E)}{N(I)}}$")
    # axes[1, 0].set_ylabel(r"$\frac{\sigma_{E}}{\sigma_{I}}$")
    # plt.tight_layout()
    # fig.savefig(os.path.join(save_path, "another_comparison.png"), dpi=300, bbox_inches="tight")
    # plt.close(fig)


__all__ = ["CompartmentStatisticsAnalyzer"]
