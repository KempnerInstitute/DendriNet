"""
Ablation analysis module.

This module contains classes for evaluating model performance by systematically
disabling or ablating different network components to understand their contributions.
"""

from copy import deepcopy
from typing import Callable, Optional
import os
import json
import re

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config.analysis import AblationAnalysisParams
from dendritic_modeling.models import BaseModel, Regressor, Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.architectures import DendriticBranchLayer
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_ablation_results,
    plot_layer_ablation,
)
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict


class AblationAnalyzer(AbstractAnalyzer):
    """Analyzer for performing ablation studies on model components."""

    def __init__(self, params: AblationAnalysisParams):
        """Initialize with ablation parameters."""
        super().__init__("AblationAnalyzer")
        self.layer_ablation = getattr(params, "layer_ablation", False)
        self.compartment_ablation = getattr(params, "compartment_ablation", False)
        self.ablate_all_synapses = getattr(params, "ablate_all_synapses", False)
        self.ablate_excitation = getattr(params, "ablate_excitation", False)
        self.ablate_inhibition = getattr(params, "ablate_inhibition", False)
        self.ablate_upstream = getattr(params, "ablate_upstream", False)
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", False)
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)

    def ablate_layer_and_evaluate(
        self,
        test_dataset: torch.utils.data.Dataset,
        model: BaseModel,
        module: DendriticBranchLayer,
        module_name: str,
        ablation_type: str,
        baseline_dict: dict,
        results_dict: dict,
        state_dict: dict,
    ):
        """Ablate a single layer and evaluate its impact."""
        results_dict[ablation_type][module_name] = {}
        results_dict[ablation_type][module_name]["depth"] = module.layer_idx

        # Check if this is a soma layer without somatic synapses (using config)
        somatic_synapses_enabled = getattr(model.core_network, "somatic_synapses", True)
        is_soma_without_synapses = (
            module.layer_idx == 0 and not somatic_synapses_enabled
        )

        if is_soma_without_synapses:
            # For soma without synapses, ablation has no effect (zero drop)
            def _set_zero_metric(metric_key: str):
                results_dict[ablation_type][module_name][f"{metric_key} drop"] = 0.0

            if self.accuracy:
                _set_zero_metric("accuracy")
            if self.auc:
                _set_zero_metric("auc")
            if self.categorical_ll:
                _set_zero_metric("categorical_loglikelihood")
            if self.mse:
                _set_zero_metric("mse")
            if self.cosine_similarity:
                _set_zero_metric("cosine_similarity")
        else:
            # Normal ablation for layers with actual synapses
            if ablation_type in ["all synapses", "excitation"]:
                if module.branch_excitation is not None and hasattr(
                    module.branch_excitation, "pre_w"
                ):
                    module.branch_excitation.pre_w.data.fill_(-1000)
            if ablation_type in ["all synapses", "inhibition"]:
                if module.branch_inhibition is not None and hasattr(
                    module.branch_inhibition, "pre_w"
                ):
                    module.branch_inhibition.pre_w.data.fill_(-1000)
            if ablation_type == "upstream":
                if hasattr(module, "branches_to_output") and hasattr(
                    module.branches_to_output, "log_weight"
                ):
                    module.branches_to_output.log_weight.data.fill_(-1000)

            def _evaluate_metric(metric_func: Callable, metric_key: str):
                metric_score = metric_func(
                    model, test_ds=test_dataset, move_device=False
                )[-1]
                results_dict[ablation_type][module_name][f"{metric_key} drop"] = (
                    baseline_dict[metric_key] - metric_score
                )

            if self.accuracy:
                _evaluate_metric(metric_func=evaluate_accuracy, metric_key="accuracy")
            if self.auc:
                _evaluate_metric(metric_func=evaluate_auc, metric_key="auc")
            if self.categorical_ll:
                _evaluate_metric(
                    metric_func=evaluate_categorical_loglikelihood,
                    metric_key="categorical_loglikelihood",
                )
            if self.mse:
                _evaluate_metric(metric_func=evaluate_mse, metric_key="mse")
            if self.cosine_similarity:
                _evaluate_metric(
                    metric_func=evaluate_cosine_similarity,
                    metric_key="cosine_similarity",
                )

        model.load_state_dict(state_dict)

        return results_dict

    def ablate_compartment_and_evaluate(
        self,
        test_dataset: torch.utils.data.Dataset,
        model: BaseModel,
        module: DendriticBranchLayer,
        module_name: str,
        ablation_type: str,
        baseline_dict: dict,
        results_dict: dict,
        state_dict: dict,
    ):
        """Ablate a single compartment and evaluate its impact."""
        results_dict[ablation_type][module_name] = {}
        results_dict[ablation_type][module_name]["depth"] = module.layer_idx

        if module.branch_excitation is not None:
            n_branches = module.branch_excitation.pre_w.shape[0]
        elif module.branch_inhibition is not None:
            n_branches = module.branch_inhibition.pre_w.shape[0]
        elif hasattr(module, "branches_to_output"):
            n_branches = module.branches_to_output.log_weight.shape[0]
        else:
            raise ValueError(
                f"Number of compartments cannot be determined for {module_name}"
            )

        if self.accuracy:
            accuracy_list = []
        if self.auc:
            auc_list = []
        if self.categorical_ll:
            loglikelihood_list = []
        if self.mse:
            mse_list = []
        if self.cosine_similarity:
            cosine_similarity_list = []

        def _evaluate_metric(metric_func: Callable, metric_key: str, metric_list: list):
            metric_score = metric_func(model, test_ds=test_dataset, move_device=False)[
                -1
            ]
            metric_list.append(baseline_dict[metric_key] - metric_score)

        for branch_ix in range(n_branches):
            if ablation_type in ["all synapses", "excitation"]:
                module.branch_excitation.pre_w.data[branch_ix, :] = -1000
            if ablation_type in ["all synapses", "inhibition"]:
                module.branch_inhibition.pre_w.data[branch_ix, :] = -1000
            if ablation_type == "upstream":
                module.branches_to_output.log_weight.data[branch_ix, :] = -1000

            if self.accuracy:
                _evaluate_metric(
                    metric_func=evaluate_accuracy,
                    metric_key="accuracy",
                    metric_list=accuracy_list,
                )
            if self.auc:
                _evaluate_metric(
                    metric_func=evaluate_auc, metric_key="auc", metric_list=auc_list
                )
            if self.categorical_ll:
                _evaluate_metric(
                    metric_func=evaluate_categorical_loglikelihood,
                    metric_key="categorical_loglikelihood",
                    metric_list=loglikelihood_list,
                )
            if self.mse:
                _evaluate_metric(
                    metric_func=evaluate_mse, metric_key="mse", metric_list=mse_list
                )
            if self.cosine_similarity:
                _evaluate_metric(
                    metric_func=evaluate_cosine_similarity,
                    metric_key="cosine similarity",
                    metric_list=cosine_similarity_list,
                )

            model.load_state_dict(state_dict)

        if self.accuracy:
            results_dict[ablation_type][module_name]["accuracy drop"] = accuracy_list
        if self.auc:
            results_dict[ablation_type][module_name]["auc drop"] = auc_list
        if self.categorical_ll:
            results_dict[ablation_type][module_name][
                "categorical_loglikelihood drop"
            ] = loglikelihood_list
        if self.mse:
            results_dict[ablation_type][module_name]["mse drop"] = mse_list
        if self.cosine_similarity:
            results_dict[ablation_type][module_name]["cosine similarity drop"] = (
                cosine_similarity_list
            )

        return results_dict

    def _plot_ablation_results(
        self,
        results: dict,
        save_path: str,
        filename: str,
        somatic_synapses: bool = True,
    ):
        """Generate comprehensive ablation plots similar to information analysis."""
        try:
            # Plot layer ablation results if available
            if results.get("layer"):
                plot_layer_ablation(
                    layer_results=results["layer"],
                    baselines=results["baseline"],
                    save_path=save_path,
                    filename=filename,
                    somatic_synapses=somatic_synapses,
                )

                if os.path.exists(os.path.join(save_path, "epochs")):
                    print("Creating video for ablation vs training", flush=True)
                    create_video_layer_ablation_vs_training(
                        save_path=save_path,
                        metric="accuracy",
                        video_format="mp4",
                    )

            # Plot compartment ablation results if available
            if results.get("compartment"):
                plot_ablation_results(
                    results["compartment"],
                    save_path=save_path,
                    filename=f"{filename}_compartment",
                )
        except ImportError as e:
            print(f"Warning: Ablation plotting functions not found: {e}")
        except Exception as e:
            print(f"Warning: Could not generate ablation plots: {e}")

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False
    ):
        """Perform ablation analysis and return results."""
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        model = model.to(device)
        model.eval()
        inputs, labels = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        test_dataset = TensorDataset(inputs, labels)

        state_dict = deepcopy(model.state_dict())

        # Calculate baseline performance using correct function signatures
        baselines = {}
        if self.accuracy:
            if isinstance(model, Classifier):
                baselines["accuracy"] = evaluate_accuracy(
                    classifier=model, test_ds=test_dataset, move_device=False
                )[-1]
            else:
                self.accuracy = False

        if self.auc:
            if isinstance(model, Classifier):
                baselines["auc"] = evaluate_auc(
                    classifier=model, test_ds=test_dataset, move_device=False
                )[-1]
            else:
                self.auc = False

        if self.categorical_ll:
            if isinstance(model, Classifier):
                baselines["categorical_loglikelihood"] = evaluate_categorical_loglikelihood(
                    classifier=model, test_ds=test_dataset, move_device=False
                )[-1]
            else:
                self.categorical_ll = False

        if self.mse:
            if isinstance(model, Regressor):
                baselines["mse"] = evaluate_mse(
                    regressor=model, test_ds=test_dataset, move_device=False
                )[-1]
            else:
                self.mse = False

        if self.cosine_similarity:
            if isinstance(model, Regressor):
                baselines["cosine similarity"] = evaluate_cosine_similarity(
                    regressor=model, test_ds=test_dataset, move_device=False
                )[-1]
            else:
                self.cosine_similarity = False

        if self.layer_ablation:
            layer_results = {}
            if self.ablate_all_synapses:
                layer_results["all synapses"] = {}
            if self.ablate_excitation:
                layer_results["excitation"] = {}
            if self.ablate_inhibition:
                layer_results["inhibition"] = {}
            if self.ablate_upstream:
                layer_results["upstream"] = {}

        if self.compartment_ablation:
            compartment_results = {}
            if self.ablate_all_synapses:
                compartment_results["all synapses"] = {}
            if self.ablate_excitation:
                compartment_results["excitation"] = {}
            if self.ablate_inhibition:
                compartment_results["inhibition"] = {}
            if self.ablate_upstream:
                compartment_results["upstream"] = {}

        for _name, module in model.named_modules():
            if (
                isinstance(module, DendriticBranchLayer)
                and module.branch_excitation is not None
                and module.branch_inhibition is not None
            ):
                if self.ablate_all_synapses:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="all synapses",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="all synapses",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )

                if self.ablate_excitation:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="excitation",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="excitation",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )

                if self.ablate_inhibition:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="inhibition",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="inhibition",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )
                if self.ablate_upstream and hasattr(module, "branches_to_output"):
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="upstream",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="upstream",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )

        results = {"baseline": baselines}
        if self.layer_ablation:
            results["layer"] = layer_results
        if self.compartment_ablation:
            results["compartment"] = compartment_results

        # Add synthetic soma entry when somatic_synapses=False for uniform plotting
        somatic_synapses_enabled = getattr(model.core_network, "somatic_synapses", True)
        if not somatic_synapses_enabled and "layer" in results:
            # Add soma entry with zero values for all ablation types
            for ablation_type in results["layer"]:
                soma_entry = {"depth": 0}
                if self.accuracy:
                    soma_entry["accuracy drop"] = 0.0
                if self.auc:
                    soma_entry["auc drop"] = 0.0
                if self.categorical_ll:
                    soma_entry["categorical_loglikelihood drop"] = 0.0
                if self.mse:
                    soma_entry["mse drop"] = 0.0
                if self.cosine_similarity:
                    soma_entry["cosine_similarity drop"] = 0.0
                # Add soma entry to the results
                results["layer"][ablation_type]["synthetic_soma"] = soma_entry

        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")
            
            save_dict(results, save_path, f"{filename}.json")

            if not training:
                # Generate ablation plots
                self._plot_ablation_results(
                    results, save_path, filename, somatic_synapses_enabled
                )
        else:
            return results


def create_video_layer_ablation_vs_training(
    save_path: str,
    metric: str,
    video_format: str = "mp4",
):
    """
    Create a video showing layer ablation results evolving over training epochs.
    
    The video will be 20 seconds long, with fps automatically calculated based on
    the number of epochs.
    
    Args:
        save_path: Path to the directory containing the epochs folder
        metric: Metric to visualize (e.g., 'accuracy', 'auc', 'categorical_loglikelihood', 'mse', 'cosine_similarity')
        video_format: Video format ('mp4' or 'gif')
    """
    epochs_dir = os.path.join(save_path, "epochs")
    
    if not os.path.exists(epochs_dir):
        raise ValueError(f"Epochs directory not found: {epochs_dir}")
    
    # Collect and sort epoch JSON files
    epoch_files = []
    for filename in os.listdir(epochs_dir):
        if filename.startswith("epoch") and filename.endswith(".json"):
            # Extract epoch number from filename (e.g., "epoch0.json" -> 0)
            match = re.match(r"epoch(\d+)\.json", filename)
            if match:
                epoch_number = int(match.group(1))
                epoch_files.append((epoch_number, filename))
    
    if not epoch_files:
        raise ValueError(f"No epoch JSON files found in {epochs_dir}")
    
    # Sort by epoch number
    epoch_files.sort(key=lambda x: x[0])
    
    # Load all epoch data
    epoch_data_list = []
    for epoch_number, filename in epoch_files:
        file_path = os.path.join(epochs_dir, filename)
        with open(file_path, "r") as f:
            data = json.load(f)
            epoch_data_list.append((epoch_number, data))
    
    if not epoch_data_list:
        raise ValueError("No valid epoch data loaded")
    
    num_epochs = len(epoch_data_list)
    fps = int(num_epochs / 30.0)
    fps = min(max(fps, 2), 10)
    
    # Determine available metrics and ablation types from first epoch
    first_epoch_data = epoch_data_list[0][1]
    if "layer" not in first_epoch_data:
        raise ValueError("No layer ablation data found in epoch files")
    
    layer_results = first_epoch_data["layer"]
    # baselines = first_epoch_data.get("baseline", {})
    
    # Determine available metrics
    metric_mapping = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "categorical_loglikelihood": "Log Likelihood",
        "mse": "MSE",
        "cosine_similarity": "Cosine Similarity",
    }
    
    def has_metric_nested(dictionary, metric):
        """Check if key exists at any level in a nested dictionary"""
        if isinstance(dictionary, dict):
            key = f"{metric} drop"
            if key in dictionary:
                return True
            return any(has_metric_nested(value, metric) for value in dictionary.values())
        elif isinstance(dictionary, (list, tuple)):
            return any(has_metric_nested(item, metric) for item in dictionary)
        return False
    
    # Validate that the requested metric exists
    if metric not in metric_mapping:
        raise ValueError(
            f"Invalid metric '{metric}'. Must be one of: {list(metric_mapping.keys())}"
        )
    
    if not has_metric_nested(layer_results, metric):
        available_metrics = []
        for m in metric_mapping.keys():
            if has_metric_nested(layer_results, m):
                available_metrics.append(m)
        raise ValueError(
            f"Metric '{metric}' not found in ablation data. "
            f"Available metrics: {available_metrics}"
        )
    
    # Determine ablation types
    synapse_ablation_types = []
    if "excitation" in layer_results:
        synapse_ablation_types.append("excitation")
    if "inhibition" in layer_results:
        synapse_ablation_types.append("inhibition")
    if "all synapses" in layer_results:
        synapse_ablation_types.append("all synapses")
    
    if not synapse_ablation_types:
        raise ValueError("No ablation types found in data")
    
    # Create a video for the specified metric
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine depth range from first epoch (assume consistent across epochs)
    first_layer_results = epoch_data_list[0][1]["layer"]
    first_type_dict = first_layer_results[synapse_ablation_types[0]]
    depths = sorted(set(layer_dict["depth"] for layer_dict in first_type_dict.values()))
    
    # Create depth labels
    depth_labels = []
    for depth in depths:
        if depth == min(depths):
            depth_labels.append("Soma")
        else:
            distal_layer = depth - min(depths)
            depth_labels.append(f"Distal {distal_layer}")
    
    # Calculate global min and max values across all epochs for consistent y-axis
    all_values = []
    for epoch_number, epoch_data in epoch_data_list:
        layer_results = epoch_data["layer"]
        for ablation_type in synapse_ablation_types:
            type_dict = layer_results.get(ablation_type, {})
            for layer_dict in type_dict.values():
                metric_key = f"{metric} drop"
                decrease_value = layer_dict.get(metric_key, 0.0)
                # Handle list values (from compartment ablation)
                if isinstance(decrease_value, list):
                    if decrease_value:
                        all_values.extend([float(v) for v in decrease_value])
                else:
                    all_values.append(float(decrease_value))
    
    if not all_values:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = min(all_values)
        y_max = max(all_values)
        # Add small padding (5% of range)
        y_range = y_max - y_min
        if y_range == 0:
            y_min -= 0.1
            y_max += 0.1
        else:
            padding = y_range * 0.05
            # y_min = max(0.0, y_min - padding)  # Don't go below 0 for drop values
            y_min = y_min - padding
            y_max = y_max + padding
    
    # Create evenly spaced y-ticks rounded to 1 decimal place
    num_ticks = 6  # Number of ticks desired
    y_ticks = np.linspace(y_min, y_max, num_ticks)
    y_ticks = np.round(y_ticks, 1)  # Round to 1 decimal place
    y_ticks = np.unique(y_ticks)  # Remove duplicates
    
    # Set up plot elements that won't change
    ax.set_xticks(depths)
    ax.set_xticklabels(depth_labels, rotation=45)
    ax.set_xlabel("Branch Depth (Soma → Distal)")
    ax.set_ylabel(f"{metric_mapping[metric]} Drop Decrease")
    # ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    
    # Store bar containers for updating
    n_ablation_types = len(synapse_ablation_types)
    width = 0.25
    
    def animate(frame):
        """Animation function that updates the plot for each epoch"""
        # Clear previous bars but keep axes setup
        ax.clear()
        
        # Reload axes setup
        ax.set_xticks(depths)
        ax.set_xticklabels(depth_labels, rotation=45)
        ax.set_xlabel("Branch Depth (Soma → Distal)")
        ax.set_ylabel(f"{metric_mapping[metric]} Drop Decrease")
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        ax.grid(True, axis="y", alpha=0.3)
        
        # Get data for this frame
        epoch_number, epoch_data = epoch_data_list[frame]
        layer_results = epoch_data["layer"]
        baselines = epoch_data.get("baseline", {})
        
        # Prepare data for plotting (matching plot_layer_ablation logic exactly)
        results = {}
        for ablation_type in synapse_ablation_types:
            data_dict = {"depth": [], "decrease": []}
            type_dict = layer_results.get(ablation_type, {})
            for layer_dict in type_dict.values():
                data_dict["depth"].append(layer_dict["depth"])
                metric_key = f"{metric} drop"
                decrease_value = layer_dict.get(metric_key, 0.0)
                # Handle list values (from compartment ablation)
                if isinstance(decrease_value, list):
                    decrease_value = np.mean(decrease_value) if decrease_value else 0.0
                data_dict["decrease"].append(float(decrease_value))
            results[ablation_type] = data_dict
        
        # Use consistent depths (from first epoch) for x-axis positioning
        # Match values to depths in correct order
        for ablation_type in synapse_ablation_types:
            if ablation_type in results:
                # Create mapping from depth to decrease value
                depth_to_decrease = dict(zip(results[ablation_type]["depth"], results[ablation_type]["decrease"]))
                # Map decreases to consistent depth order
                ordered_decreases = [depth_to_decrease.get(d, 0.0) for d in depths]
                results[ablation_type]["decrease"] = ordered_decreases
        
        # Plot bars for each ablation type (matching plot_layer_ablation exactly)
        for i, ablation_type in enumerate(synapse_ablation_types):
            if ablation_type in results:
                x = np.array(depths) + width * (i - (n_ablation_types - 1) / 2)
                ax.bar(
                    x, 
                    results[ablation_type]["decrease"], 
                    width=width, 
                    label=ablation_type
                )
        
        # Update title with epoch number and baseline
        baseline_value = baselines.get(metric, 0.0)
        title = f"{metric_mapping[metric]} Drop Decrease vs. Branch Depth"
        title += f"\nEpoch {epoch_number} (Baseline: {baseline_value:.3f})"
        ax.set_title(title)
        ax.legend()
        
        return []
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, 
        animate, 
        frames=len(epoch_data_list),
        interval=1000 / fps,  # Convert fps to interval in milliseconds
        repeat=True,
        blit=True
    )
    
    # Save video
    video_filename = f"layer_ablation_{metric}_evolution"
    print(f"Creating {num_epochs}-frame video at {fps:.2f} fps (20 seconds total)")
    if video_format == "mp4":
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            video_path = os.path.join(save_path, f"{video_filename}.mp4")
            ani.save(video_path, writer=writer)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error saving MP4 video: {e}")
            print("Falling back to GIF format...")
            writer = animation.PillowWriter(fps=fps)
            video_path = os.path.join(save_path, f"{video_filename}.gif")
            ani.save(video_path, writer=writer)
            print(f"GIF saved to {video_path}")
    else:
        writer = animation.PillowWriter(fps=fps)
        video_path = os.path.join(save_path, f"{video_filename}.gif")
        ani.save(video_path, writer=writer)
        print(f"GIF saved to {video_path}")
    
    plt.close(fig)

