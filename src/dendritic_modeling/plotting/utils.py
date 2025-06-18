"""
Plotting utility functions.

This module contains utility functions for visualization including CSV export and tree visualization.
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from typing import List

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX is not available. Some plotting functions will be disabled. "
                   "To fix this, try: pip install networkx or conda install networkx")
    nx = None

from typing import Dict, List, Optional, Tuple, Any

from .analysis import dendrinet_activations_todict, einet_activations_todict
from .network import add_colorbars, shape_helper


def reverse_layer_indices(layer_indices: List[int]) -> List[int]:
    """
    Reverse layer indices for soma-to-distal visualization.
    
    Args:
        layer_indices: Original layer indices
        
    Returns:
        Reversed indices where last layer (soma) becomes 0
    """
    num_layers = len(layer_indices)
    return [num_layers - 1 - idx for idx in layer_indices]


def format_layer_labels(layer_indices: List[int]) -> List[str]:
    """
    Format layer indices as labels with special handling for soma.
    
    Args:
        layer_indices: Layer indices (0-based)
        
    Returns:
        List of formatted labels
    """
    return ["Soma" if idx == 0 else str(idx) for idx in layer_indices]


def dendrinet_activations_to_csv(
    dendrinet,
    exc_input_list,
    inh_input_list,
    save_root,
    filename="activation_components",
):
    """Export DendriNet activations to CSV files."""
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    n_branch_layers = dendrinet.n_branch_layers

    output_list = [None] * len(exc_input_list)
    for i in range(n_branch_layers + 1):
        branches_per_soma = int(layer_sizes[i] / n_soma)

        chunk_list = []

        for j in range(len(exc_input_list)):
            output_list[j] = dendrinet.branch_layers[i](
                exc_input_list[j], inh_input_list[j], output_list[j]
            )
            chunk_list.append(output_list[j].chunk(n_soma, dim=-1))

        level = (
            "soma layer"
            if (i == n_branch_layers)
            else f"branch layer {n_branch_layers - i}"
        )

        for n in range(n_soma):
            for b in range(branches_per_soma):
                activation_list = []
                for k in range(len(exc_input_list)):
                    chunk = chunk_list[k][n][:, b]
                    activation_list.append(chunk.detach().cpu().numpy())

                soma_branch = f"soma{n}_branch{b}"
                
                # Create directory
                save_dir = os.path.join(save_root, level, soma_branch)
                os.makedirs(save_dir, exist_ok=True)
                
                # Export to CSV
                csv_path = os.path.join(save_dir, f"{filename}.csv")
                
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    writer.writerow([f'digit_{i}' for i in range(len(activation_list))])
                    
                    # Write data
                    max_len = max(len(act) for act in activation_list)
                    for i in range(max_len):
                        row = []
                        for act in activation_list:
                            if i < len(act):
                                row.append(act[i])
                            else:
                                row.append('')
                        writer.writerow(row)


def einet_activations_to_csv(
    einet, input_list, save_root, filename="activation_components"
):
    """Export EINet activations to CSV files."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        # Export inhibitory cell activations
        dendrinet_activations_to_csv(
            dendrinet=inh_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
            save_root=os.path.join(save_root, f"ei_layer{i+1}", "inhibitory_cells"),
            filename=filename,
        )

        # Update inhibitory inputs
        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], inh_input_list[j]))
        inh_input_list = inh_temp

        # Export excitatory cell activations
        dendrinet_activations_to_csv(
            dendrinet=exc_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
            save_root=os.path.join(save_root, f"ei_layer{i+1}", "excitatory_cells"),
            filename=filename,
        )

        # Update excitatory inputs
        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], inh_input_list[j]))
        exc_input_list = exc_temp


def build_dendritic_tree_graph(model, branch_info=None, ablation_info=None):
    """
    Build a NetworkX graph representing the dendritic tree structure.
    
    Args:
        model: Model with dendritic structure
        branch_info: Information about branches (optional)
        ablation_info: Information about ablations (optional)
        
    Returns:
        G: NetworkX graph object
    """
    if not NETWORKX_AVAILABLE:
        logging.warning("NetworkX is not available. Cannot build dendritic tree graph.")
        return None
        
    G = nx.DiGraph()
    
    # Add nodes based on model structure
    if hasattr(model, 'net') and hasattr(model.net, 'layers'):
        for layer_idx, layer in enumerate(model.net.layers):
            # Add layer node
            layer_name = f'layer_{layer_idx}'
            G.add_node(layer_name, node_type='layer', layer=layer_idx)
            
            # Add connections between layers
            if layer_idx > 0:
                prev_layer = f'layer_{layer_idx-1}'
                G.add_edge(prev_layer, layer_name)
            
            # Add branch nodes if available
            if hasattr(layer, 'excitatory_cells'):
                exc_cell = layer.excitatory_cells
                if hasattr(exc_cell, 'branch_layers'):
                    for branch_idx, branch in enumerate(exc_cell.branch_layers):
                        branch_name = f'layer_{layer_idx}_branch_{branch_idx}'
                        G.add_node(branch_name, node_type='branch', 
                                  layer=layer_idx, branch=branch_idx)
                        G.add_edge(layer_name, branch_name)
    
    return G


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
    layer_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'layer']
    branch_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'branch']
    
    # Draw nodes
    if layer_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=layer_nodes, 
                             node_color='lightblue', node_size=1000, 
                             label='Layers')
    
    if branch_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=branch_nodes, 
                             node_color='lightcoral', node_size=500, 
                             label='Branches')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Dendritic Tree Structure')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()


def plot_activation_tuning(einet, input_list, save_root):
    """
    Plot activation tuning curves for EINet.
    
    Args:
        einet: EINet model
        input_list: List of input tensors
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
        if 'exc cells' in layer_data:
            exc_data = layer_data['exc cells']
            
            for soma_key in exc_data.keys():
                soma_data = exc_data[soma_key]
                
                # Plot tuning for each layer in this soma
                for level_key in soma_data.keys():
                    if level_key == 'sum weights':
                        continue
                        
                    level_data = soma_data[level_key]
                    
                    for branch_key in level_data.keys():
                        branch_data = level_data[branch_key]
                        activation_list = branch_data.get('activation list', [])
                        
                        if activation_list:
                            # Create tuning curve plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            means = []
                            stds = []
                            
                            for activations in activation_list:
                                act_array = activations.detach().cpu().numpy()
                                means.append(np.mean(act_array))
                                stds.append(np.std(act_array))
                            
                            x_labels = [f'Input {i}' for i in range(len(means))]
                            x_pos = np.arange(len(x_labels))
                            
                            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                            ax.set_xlabel('Input Type')
                            ax.set_ylabel('Mean Activation')
                            ax.set_title(f'Tuning Curve: {layer_key} {soma_key} {level_key} {branch_key}')
                            ax.set_xticks(x_pos)
                            ax.set_xticklabels(x_labels, rotation=45)
                            
                            # Save plot
                            save_dir = os.path.join(save_root, layer_key, 'exc_cells', 
                                                  soma_key, level_key, branch_key)
                            os.makedirs(save_dir, exist_ok=True)
                            
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_dir, 'tuning_curve.png'), 
                                      dpi=300, bbox_inches='tight')
                            plt.close()


def plot_voltage_before_activation(einet, input_list, save_root=None):
    """
    Plot voltage distributions before activation functions.
    
    Args:
        einet: EINet model
        input_list: List of input tensors
        save_root: Root directory to save plots (optional)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    einet = einet.to(device)
    
    # This would require model modifications to capture pre-activation voltages
    # For now, create a simple analysis of the activations
    
    activations_dict = einet_activations_todict(einet, input_list)
    
    for layer_key in activations_dict.keys():
        layer_data = activations_dict[layer_key]
        
        if 'exc cells' in layer_data:
            exc_data = layer_data['exc cells']
            
            # Collect all activation data for this layer
            all_activations = []
            
            for soma_key in exc_data.keys():
                soma_data = exc_data[soma_key]
                
                for level_key in soma_data.keys():
                    if level_key == 'sum weights':
                        continue
                        
                    level_data = soma_data[level_key]
                    
                    for branch_key in level_data.keys():
                        branch_data = level_data[branch_key]
                        activation_list = branch_data.get('activation list', [])
                        
                        for activations in activation_list:
                            all_activations.extend(activations.detach().cpu().numpy().flatten())
            
            if all_activations:
                # Plot voltage distribution
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogram of activations
                axes[0].hist(all_activations, bins=50, alpha=0.7, density=True)
                axes[0].set_xlabel('Activation Value')
                axes[0].set_ylabel('Density')
                axes[0].set_title(f'{layer_key} - Activation Distribution')
                axes[0].grid(True, alpha=0.3)
                
                # Box plot of activations by input type
                activation_by_input = []
                for i, input_tensor in enumerate(input_list):
                    # Get activations for this specific input
                    layer_acts = []
                    for soma_key in exc_data.keys():
                        soma_data = exc_data[soma_key]
                        for level_key in soma_data.keys():
                            if level_key == 'sum weights':
                                continue
                            level_data = soma_data[level_key]
                            for branch_key in level_data.keys():
                                branch_data = level_data[branch_key]
                                activation_list = branch_data.get('activation list', [])
                                if i < len(activation_list):
                                    layer_acts.extend(activation_list[i].detach().cpu().numpy().flatten())
                    activation_by_input.append(layer_acts)
                
                if activation_by_input:
                    axes[1].boxplot(activation_by_input, labels=[f'Input {i}' for i in range(len(activation_by_input))])
                    axes[1].set_xlabel('Input Type')
                    axes[1].set_ylabel('Activation Value')
                    axes[1].set_title(f'{layer_key} - Activations by Input')
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_root:
                    save_dir = os.path.join(save_root, layer_key)
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, 'voltage_analysis.png'), 
                              dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()


def split_MNIST_inputs(inputs, labels):
    """
    Split MNIST inputs by digit class.
    
    Args:
        inputs: Input tensor
        labels: Label tensor
        
    Returns:
        List of input tensors, one for each digit class
    """
    input_list = []
    for digit in range(10):
        digit_mask = (labels == digit)
        digit_inputs = inputs[digit_mask]
        input_list.append(digit_inputs)
    
    return input_list


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
    fig.suptitle('Information Content vs Network Depth', fontsize=16)
    
    # Extract depth information
    depths = branch_info.get('depths', [])
    mutual_info = branch_info.get('mutual_info', [])
    entropy = branch_info.get('entropy', [])
    
    if depths and mutual_info:
        axes[0].plot(depths, mutual_info, 'o-', color='blue', linewidth=2)
        axes[0].set_title('Mutual Information vs Depth')
        axes[0].set_xlabel('Network Depth')
        axes[0].set_ylabel('Mutual Information (bits)')
        axes[0].grid(True, alpha=0.3)
    
    if depths and entropy:
        axes[1].plot(depths, entropy, 'o-', color='red', linewidth=2)
        axes[1].set_title('Entropy vs Depth')
        axes[1].set_xlabel('Network Depth')
        axes[1].set_ylabel('Entropy (bits)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 