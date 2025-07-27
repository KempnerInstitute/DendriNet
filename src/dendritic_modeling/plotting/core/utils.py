"""
Plotting utilities and helper functions.

This module provides common plotting utilities used across different visualization modules.
"""

import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

# Optional networkx import for network utilities
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from dendritic_modeling import logger
from dendritic_modeling.plotting.visualizations.analysis import einet_activations_todict

def visualize_dendritic_tree(G, mode="info"):
    """
    Visualize the dendritic tree structure.

    Args:
        G: NetworkX graph object
        mode: Visualization mode ("info", "simple", etc.)
    """
    if not NETWORKX_AVAILABLE:
        logging.warning("NetworkX is not available. Cannot visualize dendritic tree.")
        return

    if G is None:
        logging.warning("No graph provided for visualization.")
        return

    plt.figure(figsize=(12, 8))

    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw different node types with different colors
    layer_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "layer"]
    branch_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "branch"]

    # Draw nodes
    if layer_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=layer_nodes,
            node_color="lightblue",
            node_size=1000,
            label="Layers",
        )

    if branch_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=branch_nodes,
            node_color="lightcoral",
            node_size=500,
            label="Branches",
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Dendritic Tree Structure")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()

def plot_activation_tuning(einet, input_list, save_root):
    """
    Plot activation tuning curves for EINet.

    Args:
        einet: EINet model
        input_list: list of input tensors
        save_root: Root directory to save plots
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    einet = einet.to(device)

    # Get activations for each input
    activations_dict = einet_activations_todict(einet, input_list)

    # Plot tuning curves for each layer
    for layer_key in activations_dict.keys():
        layer_data = activations_dict[layer_key]

        # Process excitatory cells
        if "exc cells" in layer_data:
            exc_data = layer_data["exc cells"]

            for soma_key in exc_data.keys():
                soma_data = exc_data[soma_key]

                # Plot tuning for each layer in this soma
                for level_key in soma_data.keys():
                    if level_key == "sum weights":
                        continue

                    level_data = soma_data[level_key]

                    for branch_key in level_data.keys():
                        branch_data = level_data[branch_key]
                        activation_list = branch_data.get("activation list", [])

                        if activation_list:
                            # Create tuning curve plot
                            fig, ax = plt.subplots(figsize=(10, 6))

                            means = []
                            stds = []

                            for activations in activation_list:
                                act_array = activations.detach().cpu().numpy()
                                means.append(np.mean(act_array))
                                stds.append(np.std(act_array))

                            x_labels = [f"Input {i}" for i in range(len(means))]
                            x_pos = np.arange(len(x_labels))

                            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                            ax.set_xlabel("Input Type")
                            ax.set_ylabel("Mean Activation")
                            ax.set_title(
                                f"Tuning Curve: {layer_key} {soma_key} {level_key} {branch_key}"
                            )
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(x_labels, rotation=45)

                            # Save plot
                            save_dir = os.path.join(
                                save_root,
                                layer_key,
                                "exc_cells",
                                soma_key,
                                level_key,
                                branch_key,
                            )
                            os.makedirs(save_dir, exist_ok=True)

                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(save_dir, "tuning_curve.png"),
                                dpi=300,
                                bbox_inches="tight",
                            )
                            plt.close()

def plot_voltage_before_activation(einet, input_list, save_root=None):
    """
    Plot voltage distributions before activation functions.

    Args:
        einet: EINet model
        input_list: list of input tensors
        save_root: Root directory to save plots (optional)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    einet = einet.to(device)

    # This would require model modifications to capture pre-activation voltages
    # For now, create a simple analysis of the activations

    activations_dict = einet_activations_todict(einet, input_list)

    for layer_key in activations_dict.keys():
        layer_data = activations_dict[layer_key]

        if "exc cells" in layer_data:
            exc_data = layer_data["exc cells"]

            # Collect all activation data for this layer
            all_activations = []

            for soma_key in exc_data.keys():
                soma_data = exc_data[soma_key]

                for level_key in soma_data.keys():
                    if level_key == "sum weights":
                        continue

                    level_data = soma_data[level_key]

                    for branch_key in level_data.keys():
                        branch_data = level_data[branch_key]
                        activation_list = branch_data.get("activation list", [])

                        for activations in activation_list:
                            all_activations.extend(
                                activations.detach().cpu().numpy().flatten()
                            )

            if all_activations:
                # Plot voltage distribution
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Histogram of activations
                axes[0].hist(all_activations, bins=50, alpha=0.7, density=True)
                axes[0].set_xlabel("Activation Value")
                axes[0].set_ylabel("Density")
                axes[0].set_title(f"{layer_key} - Activation Distribution")
                axes[0].grid(True, alpha=0.3)

                # Box plot of activations by input type
                activation_by_input = []
                for i, _input_tensor in enumerate(input_list):
                    # Get activations for this specific input
                    layer_acts = []
                    for soma_key in exc_data.keys():
                        soma_data = exc_data[soma_key]
                        for level_key in soma_data.keys():
                            if level_key == "sum weights":
                                continue
                            level_data = soma_data[level_key]
                            for branch_key in level_data.keys():
                                branch_data = level_data[branch_key]
                                activation_list = branch_data.get("activation list", [])
                                if i < len(activation_list):
                                    layer_acts.extend(
                                        activation_list[i]
                                        .detach()
                                        .cpu()
                                        .numpy()
                                        .flatten()
                                    )
                    activation_by_input.append(layer_acts)

                if activation_by_input:
                    axes[1].boxplot(
                        activation_by_input,
                        labels=[f"Input {i}" for i in range(len(activation_by_input))],
                    )
                    axes[1].set_xlabel("Input Type")
                    axes[1].set_ylabel("Activation Value")
                    axes[1].set_title(f"{layer_key} - Activations by Input")
                    axes[1].grid(True, alpha=0.3)

                plt.tight_layout()

                if save_root:
                    save_dir = os.path.join(save_root, layer_key)
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(
                        os.path.join(save_dir, "voltage_analysis.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()
                else:
                    plt.show()

def plot_depth_info(branch_info):
    """
    Plot information content vs depth analysis.

    Args:
        branch_info: Dictionary containing branch information analysis
    """
    if not branch_info:
        print("No branch information provided")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Information Content vs Network Depth", fontsize=16)

    # Extract depth information
    depths = branch_info.get("depths", [])
    mutual_info = branch_info.get("mutual_info", [])
    entropy = branch_info.get("entropy", [])

    if depths and mutual_info:
        axes[0].plot(depths, mutual_info, "o-", color="blue", linewidth=2)
        axes[0].set_title("Mutual Information vs Depth")
        axes[0].set_xlabel("Network Depth")
        axes[0].set_ylabel("Mutual Information (bits)")
        axes[0].grid(True, alpha=0.3)

    if depths and entropy:
        axes[1].plot(depths, entropy, "o-", color="red", linewidth=2)
        axes[1].set_title("Entropy vs Depth")
        axes[1].set_xlabel("Network Depth")
        axes[1].set_ylabel("Entropy (bits)")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
