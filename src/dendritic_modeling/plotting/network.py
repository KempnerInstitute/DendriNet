"""
Network visualization and plotting functions.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plotWithColor(x, colors, cmap="viridis", ax=None, colorbar=True, fig=None, **kwargs):
    """
    Plot data with colored points based on values.
    
    Args:
        x: Data to plot
        colors: Values to determine colors
        cmap: Colormap to use
        ax: Axis to plot on (optional)
        colorbar: Whether to add colorbar
        fig: Figure for colorbar (required if colorbar=True and ax is provided)
        **kwargs: Additional arguments passed to scatter plot
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    scatter = ax.scatter(range(len(x)), x, c=colors, cmap=cmap, **kwargs)
    
    if colorbar:
        if fig is None:
            fig = plt.gcf()
        fig.colorbar(scatter, ax=ax)
    
    return scatter


def add_colorbars(fig, scatters, labels):
    """
    Add multiple colorbars to a figure.
    
    Args:
        fig: Figure object
        scatters: List of scatter plot objects
        labels: List of labels for each colorbar
    """
    for i, (scatter, label) in enumerate(zip(scatters, labels)):
        cbar = fig.colorbar(scatter, ax=fig.axes, shrink=0.6, aspect=20)
        cbar.set_label(label)


def plot_dendrinet_params(model, save_path=None, show=True):
    """
    Plot DendriNet parameters including weights and biases.
    
    Args:
        model: DendriNet model
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Get model parameters
    params = dendrinet_params_todict(model)
    
    # Create subplots for different parameter types
    n_layers = len(params['layer_weights'])
    n_branches = len(params['branch_weights'][0]) if params['branch_weights'] else 0
    
    # Adjust figure size based on network structure
    fig_height = max(12, 4 + 2 * n_layers)
    fig_width = max(15, 5 + 2 * n_branches)
    
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    fig.suptitle('DendriNet Parameters', fontsize=16)
    
    # Plot layer weights
    if params['layer_weights']:
        for i, weights in enumerate(params['layer_weights']):
            im1 = axes[0, 0].imshow(weights, aspect='auto', cmap='RdBu_r')
            axes[0, 0].set_title(f'Layer Weights (Layer {i})')
            axes[0, 0].set_xlabel('Input Features')
            axes[0, 0].set_ylabel('Output Features')
        fig.colorbar(im1, ax=axes[0, 0])
    
    # Plot branch weights
    if params['branch_weights']:
        branch_data = np.concatenate([w.flatten() for layer_branches in params['branch_weights'] 
                                    for w in layer_branches])
        axes[0, 1].hist(branch_data, bins=50, alpha=0.7, color='blue')
        axes[0, 1].set_title('Branch Weights Distribution')
        axes[0, 1].set_xlabel('Weight Value')
        axes[0, 1].set_ylabel('Frequency')
    
    # Plot biases
    if params['layer_biases']:
        bias_data = np.concatenate([b.flatten() for b in params['layer_biases']])
        axes[1, 0].hist(bias_data, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Layer Biases Distribution')
        axes[1, 0].set_xlabel('Bias Value')
        axes[1, 0].set_ylabel('Frequency')
    
    # Plot output weights
    if params['output_weights'] is not None:
        im4 = axes[1, 1].imshow(params['output_weights'], aspect='auto', cmap='RdBu_r')
        axes[1, 1].set_title('Output Layer Weights')
        axes[1, 1].set_xlabel('Input Features')
        axes[1, 1].set_ylabel('Output Classes')
        fig.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_einet_params(model, save_path=None, show=True):
    """
    Plot EINet parameters including excitatory and inhibitory weights.
    
    Args:
        model: EINet model
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Get model parameters
    params = einet_params_todict(model)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EINet Parameters', fontsize=16)
    
    # Plot excitatory weights
    if params['excitatory_weights']:
        exc_data = np.concatenate([w.flatten() for w in params['excitatory_weights']])
        axes[0, 0].hist(exc_data, bins=50, alpha=0.7, color='red', label='Excitatory')
        axes[0, 0].set_title('Excitatory Weights Distribution')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Frequency')
    
    # Plot inhibitory weights
    if params['inhibitory_weights']:
        inh_data = np.concatenate([w.flatten() for w in params['inhibitory_weights']])
        axes[0, 1].hist(inh_data, bins=50, alpha=0.7, color='blue', label='Inhibitory')
        axes[0, 1].set_title('Inhibitory Weights Distribution')
        axes[0, 1].set_xlabel('Weight Value')
        axes[0, 1].set_ylabel('Frequency')
    
    # Plot E/I ratio across layers
    if params['ei_ratios']:
        axes[1, 0].bar(range(len(params['ei_ratios'])), params['ei_ratios'], alpha=0.7, color='purple')
        axes[1, 0].set_title('E/I Ratios by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('E/I Ratio')
    
    # Plot biases
    if params['biases']:
        bias_data = np.concatenate([b.flatten() for b in params['biases']])
        axes[1, 1].hist(bias_data, bins=50, alpha=0.7, color='green')
        axes[1, 1].set_title('Biases Distribution')
        axes[1, 1].set_xlabel('Bias Value')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_branch_weights(model, layer_idx=0, save_path=None, show=True):
    """
    Plot branch weights for a specific layer.
    
    Args:
        model: Model with branch structure
        layer_idx: Index of layer to plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Extract branch weights for specified layer
    if hasattr(model, 'net') and hasattr(model.net, 'layers'):
        layer = model.net.layers[layer_idx]
        if hasattr(layer, 'excitatory_cells') and hasattr(layer.excitatory_cells, 'branch_layers'):
            branch_layers = layer.excitatory_cells.branch_layers
            
            n_branches = len(branch_layers)
            fig, axes = plt.subplots(1, n_branches, figsize=(5*n_branches, 4))
            if n_branches == 1:
                axes = [axes]
            
            fig.suptitle(f'Branch Weights - Layer {layer_idx}', fontsize=16)
            
            for i, branch in enumerate(branch_layers):
                weights = branch.weight.detach().cpu().numpy()
                im = axes[i].imshow(weights, aspect='auto', cmap='RdBu_r')
                axes[i].set_title(f'Branch {i}')
                axes[i].set_xlabel('Input Features')
                axes[i].set_ylabel('Output Features')
                fig.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if show:
                plt.show()
            else:
                plt.close()


def plot_excitation_inhibition_weights(model, layer_idx=0, save_path=None, show=True):
    """
    Plot excitatory and inhibitory weights separately.
    
    Args:
        model: EINet model
        layer_idx: Index of layer to plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if hasattr(model, 'net') and hasattr(model.net, 'layers'):
        layer = model.net.layers[layer_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Excitatory vs Inhibitory Weights - Layer {layer_idx}', fontsize=16)
        
        # Plot excitatory weights
        if hasattr(layer, 'excitatory_weights'):
            exc_weights = layer.excitatory_weights.detach().cpu().numpy()
            im1 = axes[0].imshow(exc_weights, aspect='auto', cmap='Reds')
            axes[0].set_title('Excitatory Weights')
            axes[0].set_xlabel('Input Features')
            axes[0].set_ylabel('Excitatory Neurons')
            fig.colorbar(im1, ax=axes[0])
        
        # Plot inhibitory weights
        if hasattr(layer, 'inhibitory_weights'):
            inh_weights = layer.inhibitory_weights.detach().cpu().numpy()
            im2 = axes[1].imshow(inh_weights, aspect='auto', cmap='Blues')
            axes[1].set_title('Inhibitory Weights')
            axes[1].set_xlabel('Input Features')
            axes[1].set_ylabel('Inhibitory Neurons')
            fig.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()


def build_dendritic_tree_graph(model, layer_idx=0):
    """
    Build a NetworkX graph representing the dendritic tree structure.
    
    Args:
        model: Model with dendritic structure
        layer_idx: Index of layer to visualize
        
    Returns:
        G: NetworkX graph object
        pos: Node positions for plotting
    """
    G = nx.DiGraph()
    
    if hasattr(model, 'net') and hasattr(model.net, 'layers'):
        layer = model.net.layers[layer_idx]
        
        if hasattr(layer, 'excitatory_cells') and hasattr(layer.excitatory_cells, 'branch_layers'):
            branch_layers = layer.excitatory_cells.branch_layers
            n_branches = len(branch_layers)
            
            # Add soma node
            G.add_node('soma', node_type='soma')
            
            # Add branch nodes
            for i in range(n_branches):
                branch_name = f'branch_{i}'
                G.add_node(branch_name, node_type='branch')
                G.add_edge(branch_name, 'soma')
            
            # Create hierarchical layout
            pos = {}
            pos['soma'] = (0, 0)
            
            # Position branches in a circle around soma
            for i in range(n_branches):
                angle = 2 * np.pi * i / n_branches
                radius = 2
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[f'branch_{i}'] = (x, y)
    
    return G, pos


def visualize_dendritic_tree(model, layer_idx=0, save_path=None, show=True):
    """
    Visualize the dendritic tree structure of a layer.
    
    Args:
        model: Model with dendritic structure
        layer_idx: Index of layer to visualize
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    G, pos = build_dendritic_tree_graph(model, layer_idx)
    
    if len(G.nodes) == 0:
        print(f"No dendritic structure found in layer {layer_idx}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw nodes with different colors for different types
    soma_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'soma']
    branch_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'branch']
    
    nx.draw_networkx_nodes(G, pos, nodelist=soma_nodes, node_color='red', 
                          node_size=1000, ax=ax, label='Soma')
    nx.draw_networkx_nodes(G, pos, nodelist=branch_nodes, node_color='blue', 
                          node_size=500, ax=ax, label='Branches')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    ax.set_title(f'Dendritic Tree Structure - Layer {layer_idx}', fontsize=16)
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


# Gradient plotting functions
def plot_dendrinet_gradients(model, save_path=None, show=True):
    """
    Plot DendriNet gradients.
    
    Args:
        model: DendriNet model
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    gradients = dendrinet_gradients_todict(model)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DendriNet Gradients', fontsize=16)
    
    # Plot layer weight gradients
    if gradients['layer_weight_grads']:
        grad_data = np.concatenate([g.flatten() for g in gradients['layer_weight_grads'] if g is not None])
        if len(grad_data) > 0:
            axes[0, 0].hist(grad_data, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('Layer Weight Gradients')
            axes[0, 0].set_xlabel('Gradient Value')
            axes[0, 0].set_ylabel('Frequency')
    
    # Plot branch weight gradients
    if gradients['branch_weight_grads']:
        branch_grad_data = []
        for layer_grads in gradients['branch_weight_grads']:
            for grad in layer_grads:
                if grad is not None:
                    branch_grad_data.extend(grad.flatten())
        if len(branch_grad_data) > 0:
            axes[0, 1].hist(branch_grad_data, bins=50, alpha=0.7, color='red')
            axes[0, 1].set_title('Branch Weight Gradients')
            axes[0, 1].set_xlabel('Gradient Value')
            axes[0, 1].set_ylabel('Frequency')
    
    # Plot bias gradients
    if gradients['layer_bias_grads']:
        bias_grad_data = np.concatenate([g.flatten() for g in gradients['layer_bias_grads'] if g is not None])
        if len(bias_grad_data) > 0:
            axes[1, 0].hist(bias_grad_data, bins=50, alpha=0.7, color='green')
            axes[1, 0].set_title('Layer Bias Gradients')
            axes[1, 0].set_xlabel('Gradient Value')
            axes[1, 0].set_ylabel('Frequency')
    
    # Plot output weight gradients
    if gradients['output_weight_grads'] is not None:
        axes[1, 1].hist(gradients['output_weight_grads'].flatten(), bins=50, alpha=0.7, color='purple')
        axes[1, 1].set_title('Output Weight Gradients')
        axes[1, 1].set_xlabel('Gradient Value')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


# Helper functions that need to be imported from utils
def shape_helper(tensor):
    """Helper function to get tensor shape and statistics."""
    if tensor is None:
        return None
    return {
        'shape': tensor.shape,
        'mean': tensor.mean().item() if tensor.numel() > 0 else 0,
        'std': tensor.std().item() if tensor.numel() > 0 else 0,
        'min': tensor.min().item() if tensor.numel() > 0 else 0,
        'max': tensor.max().item() if tensor.numel() > 0 else 0,
    }


# Import these functions from plot_utils when creating this module
def dendrinet_params_todict(model):
    """Extract DendriNet parameters to dictionary."""
    # This would contain the actual implementation from plot_utils
    # Placeholder for now
    return {
        'layer_weights': [],
        'branch_weights': [],
        'layer_biases': [],
        'output_weights': None,
    }


def einet_params_todict(model):
    """Extract EINet parameters to dictionary."""
    # This would contain the actual implementation from plot_utils
    # Placeholder for now
    return {
        'excitatory_weights': [],
        'inhibitory_weights': [],
        'ei_ratios': [],
        'biases': [],
    }


def dendrinet_gradients_todict(model):
    """Extract DendriNet gradients to dictionary."""
    # This would contain the actual implementation from plot_utils
    # Placeholder for now
    return {
        'layer_weight_grads': [],
        'branch_weight_grads': [],
        'layer_bias_grads': [],
        'output_weight_grads': None,
    } 