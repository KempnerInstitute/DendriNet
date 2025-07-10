"""
Network visualization and plotting functions.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX is not available. Some plotting functions will be disabled. "
                   "To fix this, try: pip install networkx or conda install networkx")
    nx = None

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

def add_colorbars(
    fig,
    ax_iterable,
    im_iterable,
    position,
    size,
    pad,
):
    """
    Add multiple colorbars to a figure.

    Args:
        fig: Figure object
        ax_iterable: Iterable of axes
        im_iterable: Iterable of image objects
        position: Position for colorbar
        size: Size of colorbar
        pad: Padding for colorbar
    """
    for ax, im in zip(ax_iterable, im_iterable):
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=20)

    # Legacy compatibility - simple version
    if isinstance(ax_iterable, list) and len(ax_iterable) == 1:
        fig.colorbar(im_iterable[0], ax=ax_iterable[0])

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

def plot_einet_params(model=None, einet=None, save_path=None, save_root=None, save_in_dir=False, filename=None, show=True):
    """
    Plot EINet parameters including excitatory and inhibitory weights.

    Args:
        model: EINet model (alternative: use einet parameter)
        einet: EINet model (alternative: use model parameter)
        save_path: Path to save the plot (optional)
        save_root: Root directory for saving (alternative to save_path)
        save_in_dir: Whether to save in directory structure
        filename: Custom filename for saving
        show: Whether to display the plot
    """
    # Handle parameter flexibility
    if einet is not None:
        model = einet
    if model is None:
        raise ValueError("Must provide either 'model' or 'einet' parameter")

    # Handle save path flexibility
    if save_root is not None and save_path is None:
        if save_in_dir:
            os.makedirs(save_root, exist_ok=True)
            if filename:
                save_path = os.path.join(save_root, f"{filename}.png")
            else:
                save_path = os.path.join(save_root, "einet_params.png")
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
    Build a NetworkX graph representation of the dendritic tree structure.

    Args:
        model: Model with dendritic structure
        layer_idx: Index of layer to visualize

    Returns:
        G: NetworkX graph object
        pos: Node positions for plotting
    """
    if not NETWORKX_AVAILABLE:
        logging.warning("NetworkX is not available. Cannot build dendritic tree graph.")
        return None, None

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
    if not NETWORKX_AVAILABLE:
        logging.warning("NetworkX is not available. Cannot visualize dendritic tree.")
        return

    G, pos = build_dendritic_tree_graph(model, layer_idx)

    if G is None or len(G.nodes) == 0:
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

# Actual implementations from plot_utils
def dendrinet_params_todict(dendrinet, logspace=False):
    """Extract DendriNet parameters to dictionary."""
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    n_layers = dendrinet.n_branch_layers + 1

    plot_dict = {}
    for j in range(n_soma):
        plot_dict[f"soma{j}"] = {}

    for i in range(n_layers):
        level = (
            "soma layer"
            if i == dendrinet.n_branch_layers
            else f"branch layer {dendrinet.n_branch_layers - i}"
        )

        layer = dendrinet.branch_layers[i]
        branches_per_soma = int(layer_sizes[i] / n_soma)

        input_inh = layer.input_inhibitory
        input_exc = layer.input_excitatory

        if input_inh:
            if logspace:
                inhibitory_synapse_pruned = layer.branch_inhibition.log_pruned_weight()
            else:
                inhibitory_synapse_pruned = layer.branch_inhibition.pruned_weight()
            inhibitory_synapse_mask = layer.branch_inhibition.weight_mask()

            inhibitory_synapse_pruned = inhibitory_synapse_pruned.chunk(n_soma, dim=0)
            inhibitory_synapse_mask = inhibitory_synapse_mask.chunk(n_soma, dim=0)

        if input_exc:
            if logspace:
                excitatory_synapse_pruned = layer.branch_excitation.log_pruned_weight()
            else:
                excitatory_synapse_pruned = layer.branch_excitation.pruned_weight()
            excitatory_synapse_mask = layer.branch_excitation.weight_mask()

            excitatory_synapse_pruned = excitatory_synapse_pruned.chunk(n_soma, dim=0)
            excitatory_synapse_mask = excitatory_synapse_mask.chunk(n_soma, dim=0)

        for j in range(n_soma):
            plot_dict[f"soma{j}"][level] = {}
            for k in range(branches_per_soma):
                plot_dict[f"soma{j}"][level][f"branch{k}"] = {}
                plot_dict[f"soma{j}"][level][f"branch{k}"]["inh syn"] = {}
                plot_dict[f"soma{j}"][level][f"branch{k}"]["exc syn"] = {}

                plot_dict[f"soma{j}"][level][f"branch{k}"]["inh syn"]["pruned"] = (
                    inhibitory_synapse_pruned[j][k] if input_inh else None
                )
                plot_dict[f"soma{j}"][level][f"branch{k}"]["inh syn"]["mask"] = (
                    inhibitory_synapse_mask[j][k] if input_inh else None
                )

                plot_dict[f"soma{j}"][level][f"branch{k}"]["exc syn"]["pruned"] = (
                    excitatory_synapse_pruned[j][k] if input_exc else None
                )
                plot_dict[f"soma{j}"][level][f"branch{k}"]["exc syn"]["mask"] = (
                    excitatory_synapse_mask[j][k] if input_exc else None
                )

    if dendrinet.input_inhibitory:
        if logspace:
            (
                sum_excitatory_synapse_pruned,
                sum_inhibitory_synapse_pruned,
            ) = dendrinet.log_sum_weights(pruned=True)
        else:
            (
                sum_excitatory_synapse_pruned,
                sum_inhibitory_synapse_pruned,
            ) = dendrinet.sum_weights(pruned=True)
    else:
        if logspace:
            sum_excitatory_synapse_pruned = dendrinet.log_sum_weights(pruned=True)
        else:
            sum_excitatory_synapse_pruned = dendrinet.sum_weights(pruned=True)
        sum_inhibitory_synapse_pruned = [None] * n_soma

    for j in range(n_soma):
        plot_dict[f"soma{j}"]["sum weights"] = {}
        plot_dict[f"soma{j}"]["sum weights"]["inh"] = sum_inhibitory_synapse_pruned[j]
        plot_dict[f"soma{j}"]["sum weights"]["exc"] = sum_excitatory_synapse_pruned[j]

    return plot_dict

def einet_params_todict(einet, logspace=False):
    """Extract EINet parameters to dictionary."""
    einet = einet.to("cpu")

    plot_dict = {}

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dict[f"eilayer{i+1}"] = {}

        plot_dict[f"eilayer{i+1}"]["inh cells"] = dendrinet_params_todict(
            dendrinet=inh_dendrinet,
            logspace=logspace,
        )
        plot_dict[f"eilayer{i+1}"]["exc cells"] = dendrinet_params_todict(
            dendrinet=exc_dendrinet,
            logspace=logspace,
        )

    return plot_dict

def dendrinet_gradients_todict(dendrinet):
    """Extract DendriNet gradients to dictionary."""
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma

    n_layers = dendrinet.n_branch_layers + 1

    plot_dict = {}
    for n in range(n_soma):
        plot_dict[f"soma{n}"] = {}

    for i in range(n_layers):
        level = (
            "soma layer"
            if i == dendrinet.n_branch_layers
            else f"branch layer {dendrinet.n_branch_layers - i}"
        )

        layer = dendrinet.branch_layers[i]
        branches_per_soma = int(layer_sizes[i] / n_soma)

        inh_grad = layer.input_inhibitory
        exc_grad = layer.input_excitatory

        if inh_grad:
            inhibitory_synapse_grad = layer.branch_inhibition.pre_w.grad
            inhibitory_synapse_grad = inhibitory_synapse_grad.chunk(n_soma, dim=0)

        if exc_grad:
            excitatory_synapse_grad = layer.branch_excitation.pre_w.grad
            excitatory_synapse_grad = excitatory_synapse_grad.chunk(n_soma, dim=0)

        for j in range(n_soma):
            plot_dict[f"soma{j}"][level] = {}
            for k in range(branches_per_soma):
                plot_dict[f"soma{j}"][level][f"branch{k}"] = {}
                plot_dict[f"soma{j}"][level][f"branch{k}"]["inh syn"] = {}
                plot_dict[f"soma{j}"][level][f"branch{k}"]["exc syn"] = {}

                plot_dict[f"soma{j}"][level][f"branch{k}"]["inh syn"]["grad"] = (
                    inhibitory_synapse_grad[j][k] if inh_grad else None
                )
                plot_dict[f"soma{j}"][level][f"branch{k}"]["exc syn"]["grad"] = (
                    excitatory_synapse_grad[j][k] if exc_grad else None
                )

    return plot_dict

def einet_gradients_todict(model, inputs, labels):
    """Extract EINet gradients to dictionary."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    loss = model(inputs.to(device), labels.to(device))
    loss = loss.mean(dim=0)
    model.zero_grad()
    loss.backward()

    einet = model.net

    plot_dict = {}

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dict[f"eilayer{i+1}"] = {}

        plot_dict[f"eilayer{i+1}"]["inh cells"] = dendrinet_gradients_todict(
            dendrinet=inh_dendrinet,
        )
        plot_dict[f"eilayer{i+1}"]["exc cells"] = dendrinet_gradients_todict(
            dendrinet=exc_dendrinet,
        )

    model.eval()

    return plot_dict

def plot_branch_gradients(
    save_path,
    branches_to_output,
    save_in_dir=False,
    filename="image",
):
    """Plot branch gradients."""
    branch_grad = branches_to_output.grad_block()

    fig, ax = plt.subplots()
    fig.suptitle("Upstream Branch Gradients")
    maxabs = torch.max(branch_grad.abs()).item() + 1e-9
    norm = TwoSlopeNorm(vcenter=0, vmin=-maxabs, vmax=maxabs)
    im = ax.imshow(
        branch_grad.detach().cpu().numpy(),
        cmap="PRGn",
        norm=norm,
    )

    add_colorbars(fig, [ax], [im], "right", "5%", 0.05)

    if save_in_dir:
        save_path = os.path.join(save_path, "branch_grads")
        file_path = os.path.join(save_path, f"{filename}.jpeg")
    else:
        file_path = os.path.join(save_path, "branch_grads.jpeg")

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    fig.savefig(file_path)
    plt.close(fig)

def plot_excitation_inhibition_gradients(
    soma_branch,
    save_path,
    excitatory_synapse_grad,
    excitatory_synapse_mask,
    inhibitory_synapse_grad=None,
    inhibitory_synapse_mask=None,
    log_m_grad=None,
    log_b_grad=None,
    reshape_fn=None,
    reshape_exc_syn=False,
    reshape_inh_syn=False,
    save_in_dir=False,
    filename="image",
):
    """Plot excitatory and inhibitory synapse gradients."""
    if excitatory_synapse_grad is not None and inhibitory_synapse_grad is not None:
        return

    nrows, ncols = 0, 3
    height_ratios = []

    if excitatory_synapse_grad is not None:
        nrows += 1
        if reshape_exc_syn:
            height_ratios.append(10)
        else:
            height_ratios.append(1)

    if inhibitory_synapse_grad is not None:
        nrows += 1
        if reshape_inh_syn:
            height_ratios.append(10)
        else:
            height_ratios.append(1)

    if log_m_grad is not None:
        nrows += 1
        height_ratios.append(1)

    fig = plt.figure(
        constrained_layout=True,
    )
    fig.suptitle(f"Synapses on {soma_branch}")
    gs = GridSpec(nrows, ncols, height_ratios=height_ratios, figure=fig)

    ax_list = []
    im_list = []

    if excitatory_synapse_grad is not None:
        (
            excitatory_synapse_grad,
            excitatory_synapse_mask,
        ) = shape_helper(
            reshape_exc_syn,
            reshape_fn,
            (excitatory_synapse_grad, excitatory_synapse_mask),
        )

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("exc syn grad")
        maxabs = torch.max(excitatory_synapse_grad.abs()).item() + 1e-9
        norm = TwoSlopeNorm(vcenter=0, vmin=-maxabs, vmax=maxabs)
        im1 = ax1.imshow(
            excitatory_synapse_grad.detach().cpu().numpy(),
            cmap="PRGn",
            norm=norm,
        )
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("exc syn mask")
        im2 = ax2.imshow(
            excitatory_synapse_mask.detach().cpu().numpy(),
            cmap="Greys",
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax_list.append(ax1)
        im_list.append(im1)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title("active exc syn grad dist")
        ax3.hist(
            excitatory_synapse_grad[excitatory_synapse_mask > 0]
            .flatten()
            .detach()
            .cpu()
            .numpy(),
            bins=20,
            color="0.4",
        )

    if inhibitory_synapse_grad is not None:
        (
            inhibitory_synapse_grad,
            inhibitory_synapse_mask,
        ) = shape_helper(
            reshape_inh_syn,
            reshape_fn,
            (inhibitory_synapse_grad, inhibitory_synapse_mask),
        )

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title("inh syn grad")
        maxabs = torch.max(inhibitory_synapse_grad.abs()).item() + 1e-9
        norm = TwoSlopeNorm(vcenter=0, vmin=-maxabs, vmax=maxabs)
        im4 = ax4.imshow(
            inhibitory_synapse_grad.detach().cpu().numpy(),
            cmap="PRGn",
            norm=norm,
        )
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title("inh syn mask")
        im5 = ax5.imshow(
            inhibitory_synapse_mask.detach().cpu().numpy(),
            cmap="Greys",
        )
        ax5.set_xticks([])
        ax5.set_yticks([])

        ax_list.append(ax4)
        im_list.append(im4)

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title("active inh syn grad dist")
        ax6.hist(
            inhibitory_synapse_grad[inhibitory_synapse_mask > 0]
            .flatten()
            .detach()
            .cpu()
            .numpy(),
            bins=20,
            color="0.4",
        )

    if len(ax_list) > 0:
        add_colorbars(fig, ax_list, im_list, "right", "5%", 0.05)

    if log_m_grad is not None:
        ax5 = fig.add_subplot(gs[-1, 0])
        # optional param gradient

    if save_in_dir:
        save_path = os.path.join(save_path, soma_branch)
        file_path = os.path.join(save_path, f"{filename}.jpeg")
    else:
        file_path = os.path.join(save_path, f"{soma_branch}.jpeg")

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    fig.savefig(file_path)
    plt.close(fig)
