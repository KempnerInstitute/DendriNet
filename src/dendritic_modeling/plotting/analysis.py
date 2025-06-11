"""
Analysis-related plotting functions for activation patterns, tuning curves, and statistical analysis.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_dendrinet_activations(model, x, layer_idx=None, save_path=None, show=True):
    """
    Plot DendriNet activations for different layers and branches.
    
    Args:
        model: DendriNet model
        x: Input data
        layer_idx: Specific layer to plot (None for all layers)
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Get activations
    activations = dendrinet_activations_todict(model, x)
    
    if layer_idx is not None:
        # Plot specific layer
        if layer_idx < len(activations['layer_activations']):
            layer_acts = activations['layer_activations'][layer_idx]
            branch_acts = activations['branch_activations'][layer_idx]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'DendriNet Activations - Layer {layer_idx}', fontsize=16)
            
            # Plot layer activations
            im1 = axes[0].imshow(layer_acts.T, aspect='auto', cmap='viridis')
            axes[0].set_title('Layer Activations')
            axes[0].set_xlabel('Samples')
            axes[0].set_ylabel('Neurons')
            fig.colorbar(im1, ax=axes[0])
            
            # Plot branch activations (if available)
            if branch_acts:
                # Concatenate all branch activations
                all_branch_acts = np.concatenate([ba for ba in branch_acts], axis=1)
                im2 = axes[1].imshow(all_branch_acts.T, aspect='auto', cmap='plasma')
                axes[1].set_title('Branch Activations')
                axes[1].set_xlabel('Samples')
                axes[1].set_ylabel('Branch Neurons')
                fig.colorbar(im2, ax=axes[1])
    else:
        # Plot all layers
        n_layers = len(activations['layer_activations'])
        fig, axes = plt.subplots(n_layers, 2, figsize=(15, 4*n_layers))
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('DendriNet Activations - All Layers', fontsize=16)
        
        for i, (layer_acts, branch_acts) in enumerate(zip(
            activations['layer_activations'], activations['branch_activations']
        )):
            # Plot layer activations
            im1 = axes[i, 0].imshow(layer_acts.T, aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'Layer {i} Activations')
            axes[i, 0].set_xlabel('Samples')
            axes[i, 0].set_ylabel('Neurons')
            fig.colorbar(im1, ax=axes[i, 0])
            
            # Plot branch activations
            if branch_acts:
                all_branch_acts = np.concatenate([ba for ba in branch_acts], axis=1)
                im2 = axes[i, 1].imshow(all_branch_acts.T, aspect='auto', cmap='plasma')
                axes[i, 1].set_title(f'Layer {i} Branch Activations')
                axes[i, 1].set_xlabel('Samples')
                axes[i, 1].set_ylabel('Branch Neurons')
                fig.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_einet_activations(model, x, layer_idx=None, save_path=None, show=True):
    """
    Plot EINet activations showing excitatory and inhibitory responses.
    
    Args:
        model: EINet model
        x: Input data
        layer_idx: Specific layer to plot (None for all layers)
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Get activations
    activations = einet_activations_todict(model, x)
    
    if layer_idx is not None:
        # Plot specific layer
        if layer_idx < len(activations['excitatory_activations']):
            exc_acts = activations['excitatory_activations'][layer_idx]
            inh_acts = activations['inhibitory_activations'][layer_idx]
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'EINet Activations - Layer {layer_idx}', fontsize=16)
            
            # Plot excitatory activations
            im1 = axes[0].imshow(exc_acts.T, aspect='auto', cmap='Reds')
            axes[0].set_title('Excitatory Activations')
            axes[0].set_xlabel('Samples')
            axes[0].set_ylabel('Excitatory Neurons')
            fig.colorbar(im1, ax=axes[0])
            
            # Plot inhibitory activations
            im2 = axes[1].imshow(inh_acts.T, aspect='auto', cmap='Blues')
            axes[1].set_title('Inhibitory Activations')
            axes[1].set_xlabel('Samples')
            axes[1].set_ylabel('Inhibitory Neurons')
            fig.colorbar(im2, ax=axes[1])
    else:
        # Plot all layers
        n_layers = len(activations['excitatory_activations'])
        fig, axes = plt.subplots(n_layers, 2, figsize=(15, 4*n_layers))
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('EINet Activations - All Layers', fontsize=16)
        
        for i, (exc_acts, inh_acts) in enumerate(zip(
            activations['excitatory_activations'], activations['inhibitory_activations']
        )):
            # Plot excitatory activations
            im1 = axes[i, 0].imshow(exc_acts.T, aspect='auto', cmap='Reds')
            axes[i, 0].set_title(f'Layer {i} Excitatory')
            axes[i, 0].set_xlabel('Samples')
            axes[i, 0].set_ylabel('Exc. Neurons')
            fig.colorbar(im1, ax=axes[i, 0])
            
            # Plot inhibitory activations
            im2 = axes[i, 1].imshow(inh_acts.T, aspect='auto', cmap='Blues')
            axes[i, 1].set_title(f'Layer {i} Inhibitory')
            axes[i, 1].set_xlabel('Samples')
            axes[i, 1].set_ylabel('Inh. Neurons')
            fig.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def activation_boxplots(activations_dict, layer_idx=0, save_path=None, show=True):
    """
    Create boxplots of activation distributions.
    
    Args:
        activations_dict: Dictionary containing activation data
        layer_idx: Layer index to plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Activation Distributions - Layer {layer_idx}', fontsize=16)
    
    # Layer activations boxplot
    if 'layer_activations' in activations_dict:
        layer_acts = activations_dict['layer_activations'][layer_idx]
        axes[0].boxplot([layer_acts[:, i] for i in range(min(10, layer_acts.shape[1]))])
        axes[0].set_title('Layer Activation Distributions')
        axes[0].set_xlabel('Neuron Index')
        axes[0].set_ylabel('Activation Value')
    
    # Branch activations boxplot
    if 'branch_activations' in activations_dict:
        branch_acts = activations_dict['branch_activations'][layer_idx]
        if branch_acts:
            all_branch_acts = np.concatenate([ba for ba in branch_acts], axis=1)
            axes[1].boxplot([all_branch_acts[:, i] for i in range(min(10, all_branch_acts.shape[1]))])
            axes[1].set_title('Branch Activation Distributions')
            axes[1].set_xlabel('Branch Neuron Index')
            axes[1].set_ylabel('Activation Value')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def activation_means(activations_dict, save_path=None, show=True):
    """
    Plot mean activations across layers.
    
    Args:
        activations_dict: Dictionary containing activation data
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Mean Activations Across Layers', fontsize=16)
    
    # Layer activation means
    if 'layer_activations' in activations_dict:
        layer_means = [np.mean(acts) for acts in activations_dict['layer_activations']]
        axes[0].bar(range(len(layer_means)), layer_means, alpha=0.7, color='blue')
        axes[0].set_title('Mean Layer Activations')
        axes[0].set_xlabel('Layer Index')
        axes[0].set_ylabel('Mean Activation')
    
    # Branch activation means
    if 'branch_activations' in activations_dict:
        branch_means = []
        for branch_acts in activations_dict['branch_activations']:
            if branch_acts:
                all_branch_acts = np.concatenate([ba for ba in branch_acts], axis=1)
                branch_means.append(np.mean(all_branch_acts))
            else:
                branch_means.append(0)
        
        axes[1].bar(range(len(branch_means)), branch_means, alpha=0.7, color='red')
        axes[1].set_title('Mean Branch Activations')
        axes[1].set_xlabel('Layer Index')
        axes[1].set_ylabel('Mean Activation')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_activation_tuning(model, x, y, layer_idx=0, n_neurons=10, save_path=None, show=True):
    """
    Plot activation tuning curves for specific neurons.
    
    Args:
        model: Model to analyze
        x: Input data
        y: Target labels
        layer_idx: Layer to analyze
        n_neurons: Number of neurons to plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Get activations
    with model.eval():
        activations = model(x)
        if hasattr(model, 'get_layer_activations'):
            layer_acts = model.get_layer_activations(x)[layer_idx]
        else:
            # Fallback - just use the final output if no layer access
            layer_acts = activations
    
    # Create tuning curves
    n_classes = len(np.unique(y))
    n_neurons = min(n_neurons, layer_acts.shape[1])
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    fig.suptitle(f'Activation Tuning Curves - Layer {layer_idx}', fontsize=16)
    
    for i in range(n_neurons):
        # Get activations for this neuron
        neuron_acts = layer_acts[:, i]
        
        # Calculate mean activation for each class
        class_means = []
        class_stds = []
        for class_id in range(n_classes):
            class_mask = (y == class_id)
            class_acts = neuron_acts[class_mask]
            class_means.append(np.mean(class_acts))
            class_stds.append(np.std(class_acts))
        
        # Plot tuning curve
        axes[i].bar(range(n_classes), class_means, yerr=class_stds, 
                   alpha=0.7, capsize=5)
        axes[i].set_title(f'Neuron {i}')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Mean Activation')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_voltage_before_activation(model, x, layer_idx=0, save_path=None, show=True):
    """
    Plot voltage distributions before activation functions.
    
    Args:
        model: Model to analyze
        x: Input data
        layer_idx: Layer to analyze
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # This would require model modifications to capture pre-activation voltages
    # For now, create a placeholder visualization
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Voltage Distributions Before Activation - Layer {layer_idx}', fontsize=16)
    
    # Placeholder data - in real implementation, would capture actual pre-activation values
    voltages = np.random.normal(0, 1, (x.shape[0], 100))  # Placeholder
    
    # Plot voltage distribution
    axes[0].hist(voltages.flatten(), bins=50, alpha=0.7, color='purple')
    axes[0].set_title('Voltage Distribution')
    axes[0].set_xlabel('Voltage')
    axes[0].set_ylabel('Frequency')
    
    # Plot voltage vs sample
    mean_voltages = np.mean(voltages, axis=1)
    axes[1].plot(mean_voltages, alpha=0.7, color='orange')
    axes[1].set_title('Mean Voltage vs Sample')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Mean Voltage')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_einet_profiles(model, x, save_path=None, show=True):
    """
    Plot E/I profiles across layers.
    
    Args:
        model: EINet model
        x: Input data
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    # Get E/I activations
    activations = einet_activations_todict(model, x)
    
    n_layers = len(activations['excitatory_activations'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('E/I Network Profiles', fontsize=16)
    
    # Calculate E/I ratios
    ei_ratios = []
    exc_means = []
    inh_means = []
    
    for i in range(n_layers):
        exc_acts = activations['excitatory_activations'][i]
        inh_acts = activations['inhibitory_activations'][i]
        
        exc_mean = np.mean(exc_acts)
        inh_mean = np.mean(inh_acts)
        
        exc_means.append(exc_mean)
        inh_means.append(inh_mean)
        
        if inh_mean > 0:
            ei_ratios.append(exc_mean / inh_mean)
        else:
            ei_ratios.append(float('inf'))
    
    # Plot E/I ratios
    axes[0].bar(range(n_layers), ei_ratios, alpha=0.7, color='purple')
    axes[0].set_title('E/I Ratios by Layer')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('E/I Ratio')
    
    # Plot excitatory means
    axes[1].bar(range(n_layers), exc_means, alpha=0.7, color='red')
    axes[1].set_title('Mean Excitatory Activity')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean Activation')
    
    # Plot inhibitory means
    axes[2].bar(range(n_layers), inh_means, alpha=0.7, color='blue')
    axes[2].set_title('Mean Inhibitory Activity')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('Mean Activation')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_depth_info(info_data, save_path=None, show=True):
    """
    Plot information content vs depth analysis.
    
    Args:
        info_data: Dictionary containing depth information data
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Information Content vs Network Depth', fontsize=16)
    
    # Plot mutual information vs depth
    if 'mutual_info' in info_data:
        depths = info_data.get('depths', range(len(info_data['mutual_info'])))
        mutual_info = info_data['mutual_info']
        
        axes[0].plot(depths, mutual_info, 'o-', color='blue', linewidth=2)
        axes[0].set_title('Mutual Information vs Depth')
        axes[0].set_xlabel('Network Depth')
        axes[0].set_ylabel('Mutual Information (bits)')
        axes[0].grid(True, alpha=0.3)
    
    # Plot entropy vs depth
    if 'entropy' in info_data:
        depths = info_data.get('depths', range(len(info_data['entropy'])))
        entropy = info_data['entropy']
        
        axes[1].plot(depths, entropy, 'o-', color='red', linewidth=2)
        axes[1].set_title('Entropy vs Depth')
        axes[1].set_xlabel('Network Depth')
        axes[1].set_ylabel('Entropy (bits)')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


# Actual implementations from plot_utils
def dendrinet_activations_todict(
    dendrinet,
    exc_input_list,
    inh_input_list,
):
    """Extract DendriNet activations to dictionary."""
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    n_branch_layers = dendrinet.n_branch_layers

    plot_dict = {}
    for n in range(n_soma):
        plot_dict[f"soma{n}"] = {}

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
            plot_dict[f"soma{n}"][level] = {}
            for b in range(branches_per_soma):
                activation_list = []
                for k in range(len(exc_input_list)):
                    chunk = chunk_list[k][n][:, b]
                    activation_list.append(chunk)

                plot_dict[f"soma{n}"][level][f"branch{b}"] = {}
                plot_dict[f"soma{n}"][level][f"branch{b}"][
                    "activation list"
                ] = activation_list

    return plot_dict


def einet_activations_todict(einet, input_list):
    """Extract EINet activations to dictionary."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)

    plot_dict = {}

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dict[f"eilayer{i+1}"] = {}

        plot_dict[f"eilayer{i+1}"]["inh cells"] = dendrinet_activations_todict(
            dendrinet=inh_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], inh_input_list[j]))
        inh_input_list = inh_temp

        plot_dict[f"eilayer{i+1}"]["exc cells"] = dendrinet_activations_todict(
            dendrinet=exc_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
        )

        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], inh_input_list[j]))
        exc_input_list = exc_temp

    return plot_dict 