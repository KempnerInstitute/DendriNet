"""
Unified Weight Analysis Module.

This module provides comprehensive analysis of synaptic weights across the dendritic network,
including both global statistics and hierarchical per-layer analysis with plotting capabilities.
Provides comprehensive weight analysis with both global and hierarchical statistics.
"""

import os
import json

import traceback
import logging
import time
from typing import Any, Optional, Union

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
    ParametricActivation,
    DendriticBranchLayer,
    ExcitationInhibitionNetwork
)
from dendritic_modeling.utils import save_dict

from dendritic_modeling.plotting.visualizations.synaptic_weight_plots import (
    plot_synaptic_weight_analysis,
)
from dendritic_modeling.plotting.visualizations.plotting_utils import (
    convert_layer_names,
    setup_basic_plot,
)

logger = logging.getLogger(__name__)


class CompartmentSNRAnalyzer(AbstractAnalyzer):
    """
    Unified analyzer for comprehensive synaptic weight analysis.

    Provides both global weight statistics and hierarchical per-layer analysis
    with enhanced plotting capabilities including min/max and dendritic strength plots.
    """

    def __init__(self, params: CompartmentStatisticsAnalysisParams):
        """Initialize the unified compartment statistics analyzer."""
        super().__init__("CompartmentSNRAnalyzer")
        # Enhanced analysis modes
        self.global_analysis = getattr(params, "global_analysis", True)
        self.layer_analysis = getattr(params, "layer_analysis", True)
        self.branch_analysis = getattr(params, "branch_analysis", True)

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

                handle = module.reactivation.register_forward_hook(
                    lambda module, input, output, layer_key=key: self.forward_hook(
                        module, input, output, layer_key, "vinf"
                    )
                )
                handles.append(handle)

                handle = module.reactivation.register_forward_hook(
                    lambda module, input, output, layer_key=key: self.forward_hook(
                        module, input, output, layer_key, "vout"
                    )
                )
                handles.append(handle)
        
        with torch.no_grad():
            _ = model(x)
        self.remove_forward_hooks(handles)
    
    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self, 
        module: Union[TopKLinear, ParametricActivation, torch.nn.Identity], 
        input: tuple[torch.Tensor, ...], 
        output: torch.Tensor,
        key: str,
        activation_type: str
    ):
        for branch_idx in range(output.shape[-1]):
            if activation_type == "vinf":
                branch_activation = input[0][..., branch_idx].detach().cpu()
            else:  # "exc", "inh", "vout"
                branch_activation = output[..., branch_idx].detach().cpu()
            
            self.raw_dict[key][branch_idx][
                activation_type
            ] = branch_activation

    def _compute_statistics(
        self, 
        data_tensor: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        """Compute comprehensive statistics for an activation tensor."""
        total_mean = data_tensor.mean().item()
        total_var = data_tensor.var().item()

        n_samples = data_tensor.shape[0]
        var_E_C = 0

        for label in torch.unique(labels):
            class_data = data_tensor[labels == label]
            
            class_mean = class_data.mean().item()
            class_prob = class_data.shape[0] / n_samples
            var_E_C += class_prob * (class_mean - total_mean)**2

        stats_dict[f"{prefix}_d'^2"] = var_E_C / total_var

        return stats_dict
    
    def _compute_branch_statistics(
        self, 
        model: BaseModel, 
        test_dataset: torch.utils.data.Dataset, 
        device: str = "cpu"
    ):
        """Compute branch statistics from model and test dataset."""
        model.to(device)
        x, labels = test_dataset[:]
        x: torch.Tensor = x.to(device)
        labels: torch.Tensor = labels.cpu()
        
        self.collect_raw_data(model, x)
        
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
                            data_tensor, labels, prefix, branch_stats
                        )
                
                branch_statistics[layer_key][branch_idx] = branch_stats
        
        return branch_statistics

    def _layer_aggregation(
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
            for field, values in layer_aggregated.items():
                data_tensor = torch.tensor(values)
                layer_stats[f"{field}_mean"] = data_tensor.mean().item()
                layer_stats[f"{field}_std"] = data_tensor.std().item()
            
            layer_statistics[layer_key] = layer_stats
        return layer_statistics

    def _global_aggregation(
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
        for field, values in global_aggregated.items():
            data_tensor = torch.tensor(values)
            global_statistics[f"{field}_mean"] = data_tensor.mean().item()
            global_statistics[f"{field}_std"] = data_tensor.std().item()
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

        branch_statistics = self._compute_branch_statistics(
            model = model,
            test_dataset = test_dataset,
            device = device
        )

        results = {}

        if self.global_analysis:
            global_statistics = self._global_aggregation(branch_statistics)
            results["global_statistics"] = global_statistics
        
        if self.layer_analysis:
            layer_statistics = self._layer_aggregation(branch_statistics)
            results["layer_statistics"] = layer_statistics
        
        if self.branch_analysis:
            results["branch_statistics"] = branch_statistics

        elapsed_time = time.time() - start_time
        self.logger.info(f"SNR analysis completed in {elapsed_time:.2f} seconds")

        # Save results and generate plots
        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")

            self.logger.info(
                f"Saving SNR analysis results to {save_path}/{filename}"
            )
            save_dict(results, save_path, f"{filename}.json")

            if not training:
                epochs_data, epoch_numbers = self.load_epochs_data(save_path)
                if self.global_analysis:
                    if epochs_data:
                        plot_global_vs_training(
                            epochs_data["global_statistics"], epoch_numbers, save_path
                        )

                if self.layer_analysis:
                    if epochs_data:
                        pass
                
                if self.branch_analysis:
                    if epochs_data:
                        pass
                    
                    plot_branch_final(results["branch_statistics"], save_path)




        return results

    def _initialize_agg_structure(self, data: dict) -> dict:
        """
        Recursively initialize an aggregate dictionary structure with empty lists at leaf nodes.
        
        Args:
            data: The dictionary structure to replicate
            
        Returns:
            Dictionary with same structure but empty lists at leaf nodes
        """
        agg_structure = {}
        for key, value in data.items():
            if isinstance(value, dict):
                agg_structure[key] = self._initialize_agg_structure(value)
            else:
                # Leaf node - initialize as empty list
                if key == "depth":
                    agg_structure[key] = data[key]
                else:
                    agg_structure[key] = []
        return agg_structure

    def _populate_agg_dict(self, agg_dict: dict, epoch_data: dict):
        """
        Recursively populate aggregate dictionary by appending leaf values from epoch_data.
        
        Args:
            agg_dict: The aggregate dictionary to populate (modified in place)
            epoch_data: The epoch data dictionary to extract values from
        """
        for key, value in epoch_data.items():
            if isinstance(value, dict):
                if key in agg_dict:
                    self._populate_agg_dict(agg_dict[key], value)
            else:
                # Leaf node - append value to list
                if key in agg_dict and key != "depth":
                    agg_dict[key].append(value)

    def load_epochs_data(self, save_path: str):
        training_path = os.path.join(save_path, "epochs")

        if not os.path.exists(training_path):
            return None, None 
        
        agg_dict_initialized = False
        agg_dict = {}
        
        # Collect all epoch files and sort by epoch number
        epoch_files = []
        for filename in os.listdir(training_path):
            # Check if file matches the "epoch#" naming convention
            if filename.startswith("epoch") and filename.endswith(".json"):
                # Extract the number part (everything between "epoch" and ".json")
                epoch_number_str = filename[5:-5]  # Remove "epoch" prefix and ".json" suffix
                try:
                    epoch_number = int(epoch_number_str)
                    epoch_files.append((epoch_number, filename))
                except ValueError:
                    # Skip files that don't have numeric epoch numbers
                    continue
        
        # Sort by epoch number to process in order
        epoch_files.sort(key=lambda x: x[0])
        
        # Iterate through epoch files in sorted order
        epoch_numbers = []
        for epoch_number, filename in epoch_files:
            epoch_numbers.append(epoch_number)
            # Load the JSON file
            file_path = os.path.join(training_path, filename)
            with open(file_path, "r") as f:
                epoch_data = json.load(f)
            
            if not agg_dict_initialized:
                # Initialize aggregate dictionary with same structure, empty lists at leaves
                agg_dict = self._initialize_agg_structure(epoch_data)
                agg_dict_initialized = True
            
            # Populate aggregate dictionary by appending values from this epoch
            self._populate_agg_dict(agg_dict, epoch_data)
        
        return agg_dict, epoch_numbers
    

def plot_global_vs_training(global_data: dict, epoch_numbers: list, save_path: str):
    """
    Plot mean +/- 1 standard deviation for all activation types across training epochs.
    
    Args:
        global_data: Aggregated dictionary with lists of values for each metric
        epoch_numbers: List of epoch numbers corresponding to the data
        save_path: Path to save the plot
    """
    activation_types = ["exc", "inh", "vinf", "vout"]
    colors = {
        "exc": "#e74c3c",
        "inh": "#3498db",
        "vinf": "#9b59b6",
        "vout": "#2c3e50",
    }
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    
    for act_type in activation_types:
        mean_key = f"{act_type}_d'^2_mean"
        std_key = f"{act_type}_d'^2_std"
        
        # Extract mean and std lists from global_data
        if (
            mean_key in global_data 
            and std_key in global_data
        ):
            mean_values = global_data[mean_key]
            std_values = global_data[std_key]
            
            # Convert to numpy arrays for easier manipulation
            mean_array = np.array(mean_values)
            std_array = np.array(std_values)
            
            # Plot mean line
            ax.plot(
                epoch_numbers, mean_array, 
                label=act_type, color=colors[act_type], linewidth=2
            )
            
            # Plot mean +/- std as filled region
            ax.fill_between(
                epoch_numbers,
                mean_array - std_array,
                mean_array + std_array,
                alpha=0.2,
                color=colors[act_type]
            )
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(r"d'$^2$", fontsize=12)
    ax.set_title("Global Statistics: Mean ± 1 Standard Deviation Across Training", fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, "global_vs_training.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_branch_final(branch_data: dict, save_path: str):
    """
    Plot the final branch statistics.
    
    Args:
        branch_data: The branch data dictionary to plot
        save_path: Path to save the plot
    """
    agg_data: dict[str, dict[str, list[float]]] = {}

    for layer_key, layer_data in branch_data.items():
        for branch_data in layer_data.values():
            depth = branch_data["depth"]
            if f"depth {depth}" not in agg_data.keys():
                agg_data[f"depth {depth}"] = {}
            
            if "exc" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"exc"] = []
            if "inh" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"inh"] = []
            if "vinf" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"vinf"] = []
            if "vout" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"vout"] = []
            
            for activation_type in ["exc", "inh", "vinf", "vout"]:
                agg_data[f"depth {depth}"][activation_type].append(branch_data[f"{activation_type}_d'^2"])
    
    indeps = ["exc", "inh", "vinf", "vout"]
    deps = ["exc", "inh", "vinf", "vout"]
    
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
    labels = {
        "exc": r"$d'^2_{E}$",
        "inh": r"$d'^2_{I}$",
        "vinf": r"$d'^2_{V_{\infty}}$",
        "vout": r"$d'^2_{V_{out}}$",
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
                ax.set_xlabel(labels[col_key])
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(labels[row_key])
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    g.figure.savefig(os.path.join(save_path, "branch_final.png"), dpi=300, bbox_inches="tight")
    plt.close(g.figure)


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

    

def plot_ratio_pairwise_comparison(
    branch_statistics: dict[str, dict[int, dict[str, float]]], 
    weights_ratio: bool = False,
    inputs_ratio: bool = False,
    activations_ratio: bool = False,
    save_path: str = None
):
    agg_data: dict[str, dict[str, list[float]]] = {}
    for layer_key, layer_data in branch_statistics.items():
        for branch_data in layer_data.values():
            depth = branch_data["depth"]
            if f"depth {depth}" not in agg_data.keys():
                agg_data[f"depth {depth}"] = {}
            
            if weights_ratio and "ei_w_mean_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_w_mean_ratio"] = []
            if weights_ratio and "ei_w_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_w_std_ratio"] = []
            
            if inputs_ratio and "ei_x_mean_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_x_mean_ratio"] = []
            if inputs_ratio and "ei_x_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_x_std_ratio"] = []
            
            if weights_ratio and inputs_ratio and "ei_wx_mean_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_wx_mean_ratio"] = []
            if weights_ratio and inputs_ratio and "ei_wx_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_wx_std_ratio"] = []
            
            if activations_ratio and "ei_a_mean_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_a_mean_ratio"] = []
            if activations_ratio and "ei_a_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"][f"ei_a_std_ratio"] = []
            
            
            if weights_ratio:
                ei_w_mean_ratio = branch_data["exc_weights_mean"] / (branch_data["inh_weights_mean"] + 1e-8)
                agg_data[f"depth {depth}"][f"ei_w_mean_ratio"].append(ei_w_mean_ratio)
                
                ei_w_std_ratio = branch_data["exc_weights_std"] / (branch_data["inh_weights_std"] + 1e-8)
                agg_data[f"depth {depth}"][f"ei_w_std_ratio"].append(ei_w_std_ratio)
            
            if inputs_ratio:
                ei_x_mean_ratio = branch_data["exc_inputs_mean"] / (branch_data["inh_inputs_mean"] + 1e-8)
                agg_data[f"depth {depth}"][f"ei_x_mean_ratio"].append(ei_x_mean_ratio)
                
                ei_x_std_ratio = branch_data["exc_inputs_std"] / (branch_data["inh_inputs_std"] + 1e-8)
                agg_data[f"depth {depth}"][f"ei_x_std_ratio"].append(ei_x_std_ratio)

            if weights_ratio and inputs_ratio:
                ei_wx_mean_ratio = ei_w_mean_ratio * ei_x_mean_ratio
                agg_data[f"depth {depth}"][f"ei_wx_mean_ratio"].append(ei_wx_mean_ratio)
                
                ei_wx_std_ratio = ei_w_std_ratio * ei_x_std_ratio
                agg_data[f"depth {depth}"][f"ei_wx_std_ratio"].append(ei_wx_std_ratio)

            if activations_ratio:
                ei_a_mean_ratio = branch_data["exc_activations_mean"] / (branch_data["inh_activations_mean"] + 1e-8)
                agg_data[f"depth {depth}"][f"ei_a_mean_ratio"].append(ei_a_mean_ratio)
                
                ei_a_std_ratio = branch_data["exc_activations_std"] / (branch_data["inh_activations_std"] + 1e-8)
                agg_data[f"depth {depth}"][f"ei_a_std_ratio"].append(ei_a_std_ratio)
    

    available = []
    if weights_ratio:
        available.extend(["ei_w_mean_ratio", "ei_w_std_ratio"])
    if inputs_ratio:
        available.extend(["ei_x_mean_ratio", "ei_x_std_ratio"])
    if weights_ratio and inputs_ratio:
        available.extend(["ei_wx_mean_ratio", "ei_wx_std_ratio"])
    if activations_ratio:
        available.extend(["ei_a_mean_ratio", "ei_a_std_ratio"])
    
    # Build long-form DataFrame from agg_data
    rows = []
    for depth, data in agg_data.items():
        for dep in available:          # y
            for indep in available:  # x
                x = np.asarray(data[indep]); y = np.asarray(data[dep])
                rows.append(pd.DataFrame({
                    'x': x, 'y': y, 'depth': depth,
                    'row': dep, 'col': indep
                }))
    df = pd.concat(rows, ignore_index=True)

    label_map = {
        "ei_w_mean_ratio": r"$\frac{\bar W_E}{\bar W_I}$",
        "ei_w_std_ratio":  r"$\frac{\sigma(W_E)}{\sigma(W_I)}$",
        "ei_x_mean_ratio": r"$\frac{\bar X_E}{\bar X_I}$",
        "ei_x_std_ratio":  r"$\frac{\sigma(X_E)}{\sigma(X_I)}$",
        "ei_wx_mean_ratio": r"$\frac{\bar W_E}{\bar W_I}\cdot\frac{\bar X_E}{\bar X_I}$",
        "ei_wx_std_ratio":  r"$\frac{\sigma(W_E)}{\sigma(W_I)}\cdot\frac{\sigma(X_E)}{\sigma(X_I)}$",
        "ei_a_mean_ratio": r"$\frac{\bar A_E}{\bar A_I}$",
        "ei_a_std_ratio":  r"$\frac{\sigma(A_E)}{\sigma(A_I)}$",
    }

    # Scatter faceted by (row=dep, col=indep) with hue=depth
    g = sns.relplot(
        data=df, x='x', y='y', hue='depth',
        row='row', col='col', kind='scatter',
        row_order=available, col_order=available,
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
    for i, row_key in enumerate(available):
        for j, col_key in enumerate(available):
            ax = g.axes[i, j]
            ax.grid(True)
            if i == len(available) - 1:
                ax.set_xlabel(label_map[col_key])
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(label_map[row_key])
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    g.figure.savefig(os.path.join(save_path, "ratio_pairwise_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(g.figure)

    


__all__ = ["CompartmentSNRAnalyzer"]
