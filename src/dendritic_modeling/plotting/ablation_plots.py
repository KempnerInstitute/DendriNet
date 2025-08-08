"""
Ablation analysis plotting utilities.

This module contains plotting functions for visualizing ablation study results.
"""
import os
from typing import Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_layer_ablation(
    layer_results: dict,
    save_path: Optional[str] = None,
    filename: str = "final"
) -> Optional[List[plt.Figure]]:
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
        "accuracy drop": "Accuracy Drop",
        "auc drop": "AUC Drop",
        "log likelihood drop": "Log Likelihood Drop",
        "mse drop": "MSE Drop",
        "cosine similarity drop": "Cosine Similarity Drop"
    }

    def has_key_nested(dictionary, key):
        """
        Check if key exists at any level in a nested dictionary
        """
        if isinstance(dictionary, dict):
            if key in dictionary:
                return True
            return any(has_key_nested(value, key) for value in dictionary.values())
        elif isinstance(dictionary, (list, tuple)):
            return any(has_key_nested(item, key) for item in dictionary)
        return False

    metrics = []
    for metric in metric_mapping.keys():
        if has_key_nested(layer_results, metric):
            metrics.append(metric)

    synapse_ablation_types = []
    if "excitation" in layer_results:
        synapse_ablation_types.append("excitation")
    if "inhibition" in layer_results:
        synapse_ablation_types.append("inhibition")
    if "all synapses" in layer_results:
        synapse_ablation_types.append("all synapses")
    
    for metric in metrics:
        results = {}
        for ablation_type in synapse_ablation_types:
            data_dict = {"depth": [], "decrease": []}
            type_dict: dict = layer_results[ablation_type]
            for layer_dict in type_dict.values():
                data_dict["depth"].append(layer_dict["depth"])
                data_dict["decrease"].append(layer_dict[metric])
            results[ablation_type] = data_dict
                
        fig, ax = plt.subplots(figsize=(10, 8))
        n_ablation_types = len(synapse_ablation_types)
        depths = results[synapse_ablation_types[0]]["depth"]
        width = 0.25
        for i, ablation_type in enumerate(synapse_ablation_types):
            x = np.array(depths) + width * (i - (n_ablation_types - 1) / 2)
            ax.bar(
                x,
                results[ablation_type]["decrease"],
                width=width,
                label=ablation_type
            )
        ax.set_xticks(depths)
        ax.set_xlabel("Depth")
        ax.set_ylabel(f"{metric_mapping[metric]} Decrease")
        ax.set_title(f"{metric_mapping[metric]} Decrease vs. Depth")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        
        if save_path is not None:
            plot_path = os.path.join(
                save_path, f"{filename}_{metric}.png"
            )
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            fig_list.append(fig)
    
    if save_path is None:
        return fig_list


def plot_compartment_ablation(
    compartment_results: dict,
    save_path: Optional[str] = None,
    filename: str = "compartment_ablation"
) -> Optional[List[plt.Figure]]:
    """
    Plot compartment-wise ablation results.
    
    Args:
        compartment_results: Dictionary containing compartment ablation results
        save_path: Path to save plots. If None, returns list of figures
        filename: Base filename for saved plots
        
    Returns:
        List of figures if save_path is None, otherwise None
    """
    if save_path is None:
        fig_list = []
    
    for ablation_type, layers in compartment_results.items():
        fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4 * len(layers)))
        if len(layers) == 1:
            axes = [axes]
        
        for idx, (layer_name, layer_data) in enumerate(layers.items()):
            ax = axes[idx]
            
            if "accuracy drop" in layer_data:
                drops = layer_data["accuracy drop"]
                ax.bar(range(len(drops)), drops)
                ax.set_xlabel("Compartment Index")
                ax.set_ylabel("Accuracy Drop")
                ax.set_title(f"{layer_name} - {ablation_type} ablation")
                ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is not None:
            plot_path = os.path.join(
                save_path, f"{filename}_{ablation_type}.png"
            )
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            fig_list.append(fig)
    
    if save_path is None:
        return fig_list


def plot_ablation_summary(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    filename: str = "ablation_summary"
) -> Optional[plt.Figure]:
    """
    Create a summary plot of all ablation results.
    
    Args:
        results: Dictionary containing all ablation results
        save_path: Path to save plot. If None, returns figure
        filename: Filename for saved plot
        
    Returns:
        Figure if save_path is None, otherwise None
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot layer ablation summary if available
    if "layer" in results:
        ax = axes[0]
        layer_results = results["layer"]
        
        # Extract data for plotting
        ablation_types = []
        total_drops = []
        
        for ablation_type, layers in layer_results.items():
            if ablation_type in ["excitation", "inhibition", "all synapses"]:
                ablation_types.append(ablation_type)
                
                # Sum accuracy drops across all layers
                total_drop = sum(
                    layer_data.get("accuracy drop", 0)
                    for layer_data in layers.values()
                )
                total_drops.append(total_drop)
        
        ax.bar(ablation_types, total_drops)
        ax.set_xlabel("Ablation Type")
        ax.set_ylabel("Total Accuracy Drop")
        ax.set_title("Layer Ablation Summary")
        ax.grid(True, axis="y", alpha=0.3)
    
    # Plot compartment ablation summary if available
    if "compartment" in results:
        ax = axes[1]
        compartment_results = results["compartment"]
        
        # Similar summary for compartment ablation
        ablation_types = []
        avg_drops = []
        
        for ablation_type, layers in compartment_results.items():
            if ablation_type in ["excitation", "inhibition", "all synapses"]:
                ablation_types.append(ablation_type)
                
                # Average accuracy drops across all compartments
                all_drops = []
                for layer_data in layers.values():
                    if "accuracy drop" in layer_data:
                        all_drops.extend(layer_data["accuracy drop"])
                
                avg_drop = np.mean(all_drops) if all_drops else 0
                avg_drops.append(avg_drop)
        
        ax.bar(ablation_types, avg_drops)
        ax.set_xlabel("Ablation Type")
        ax.set_ylabel("Average Accuracy Drop")
        ax.set_title("Compartment Ablation Summary")
        ax.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        plot_path = os.path.join(save_path, f"{filename}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        return fig 