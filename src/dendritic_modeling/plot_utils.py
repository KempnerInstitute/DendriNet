"""
plot_utils.py
=============
This module contains utility functions for plotting dendritic model parameters.
"""

import os

import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_activation_tuning(einet, input_list, save_root):
    """
    Quantifies and plots the sensitivity of activations to different digits 
    per layer in the Excitation-Inhibition Network.

    Args:
        einet (Net): The trained network.
        input_list (list): List of input tensors, each corresponding to a digit.
        save_root (str): Directory to save the plots.
    """
    import seaborn as sns

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    einet = einet.to(device)
    einet.eval()

    os.makedirs(save_root, exist_ok=True)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)

    # Per layer storage for activations
    activation_data = {}

    with torch.no_grad():
        for layer_idx in range(len(einet.layers)):
            inh_dendrinet = einet.layers[layer_idx].inhibitory_cells
            exc_dendrinet = einet.layers[layer_idx].excitatory_cells

            # Process inhibitory cells
            inh_activations = []
            for digit_idx, inputs in enumerate(exc_input_list):
                activations = inh_dendrinet(inputs, inh_input_list[digit_idx])
                inh_activations.append(activations.cpu().numpy())
            activation_data[f"layer{layer_idx+1}_inh"] = np.stack(inh_activations)

            # Process excitatory cells
            exc_activations = []
            for digit_idx, inputs in enumerate(exc_input_list):
                activations = exc_dendrinet(inputs, inh_input_list[digit_idx])
                exc_activations.append(activations.cpu().numpy())
            activation_data[f"layer{layer_idx+1}_exc"] = np.stack(exc_activations)

            # Update inhibitory input for next layer
            inh_input_list = [inh_dendrinet(inputs, None) for inputs in exc_input_list]
            exc_input_list = [exc_dendrinet(inputs, inh_input) for inputs, inh_input in zip(exc_input_list, inh_input_list)]

    # Plot tuning per layer
    for layer_name, activations in activation_data.items():
        mean_activations = np.mean(activations, axis=2)  # Average over activations per neuron
        var_activations = np.var(activations, axis=2)    # Variance over activations per neuron

        # Plot heatmap of variance (tuning)
        plt.figure(figsize=(10, 6))
        sns.heatmap(var_activations, cmap="viridis", annot=False)
        plt.title(f"Activation Tuning (Variance) for {layer_name}")
        plt.xlabel("Digit (0-9)")
        plt.ylabel("Neuron Index")
        save_path = os.path.join(save_root, f"{layer_name}_tuning.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved tuning plot for {layer_name} at {save_path}")
        
        
def plot_voltage_before_activation(einet, input_list, save_root=None):
    """
    Plots the voltage before feeding it to the activation function 
    for each layer in the Excitation-Inhibition Network.

    Args:
        einet (Net): The neural network model.
        input_list (list): List of input tensors for visualization.
        save_root (str): Directory to save the plots. If None, plots are displayed.
    """
    # Ensure save directory exists
    if save_root is not None:
        os.makedirs(save_root, exist_ok=True)
    
    # Set model to evaluation mode
    einet.eval()

    with torch.no_grad():
        for idx, input_tensor in enumerate(input_list):
            input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor

            # Forward pass and extract voltages
            einet(input_tensor)  # Pass input directly without keyword

            # Collect voltages
            inhibition_v_inf = einet.inhibition_v_inf.cpu().numpy()
            dendrite_branch_v_inf = einet.dendrite_branch_v_inf.cpu().numpy()
            dendrite_v_inf = einet.dendrite_v_inf.cpu().numpy()
            soma_v_inf = einet.soma_v_inf.cpu().numpy()

            # Plot each voltage map
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(inhibition_v_inf.reshape(1, -1), aspect='auto', cmap='viridis')
            axes[0].set_title(r"Inhibition Voltage $V_{\text{inf}}$")

            axes[1].imshow(dendrite_branch_v_inf.reshape(1, -1), aspect='auto', cmap='viridis')
            axes[1].set_title(r"Dendrite Branch Voltage $V_{\text{inf}}$")

            axes[2].imshow(dendrite_v_inf.reshape(1, -1), aspect='auto', cmap='viridis')
            axes[2].set_title(r"Dendrite Voltage $V_{\text{inf}}$")

            axes[3].imshow(soma_v_inf.reshape(1, -1), aspect='auto', cmap='viridis')
            axes[3].set_title(r"Soma Voltage $V_{\text{inf}}$")

            plt.suptitle(f"Voltage Before Activation - Input {idx + 1}")

            # Save or show the plot
            if save_root is not None:
                save_path = os.path.join(save_root, f"voltage_before_activation_{idx + 1}.png")
                plt.savefig(save_path)
                plt.close(fig)
                print(f"Saved voltage plot for input {idx + 1} at {save_path}")
            else:
                plt.show()

#@title Function to plot an image with the scale color bar
def plotWithColor(data, title=''):
    if torch.is_tensor(data):
        data=data.cpu().numpy()
        ax = plt.subplot()
        im = ax.imshow(data)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)

        if len(title)>0:
            plt.title(title)

            plt.show()


def plot_NLL_loss_curves(
    train_losses, 
    valid_losses, 
    num_epochs,
    save_path = None,
):
    Train_losses = np.array(train_losses)
    Train_losses[np.where(Train_losses > 5)] = 5
    
    Valid_losses = np.array(valid_losses)
    Valid_losses[np.where(Valid_losses > 5)] = 5
    
    fig = plt.figure()
    
    plt.plot(range(1,num_epochs+1), Train_losses, '0.4', label = 'training')
    plt.plot(range(1,num_epochs+1), Valid_losses, 'b', label = 'validation')
    
    plt.xlabel('Epochs')
    plt.xticks(range(0,num_epochs+1,int(num_epochs // 10)))
    plt.ylabel('Negative Log Likelihood')
    plt.title('Loss Curves')
    plt.legend()

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok = True)
        fig.savefig(os.path.join(save_path, 'nll_loss_curves.jpeg'))
        plt.close(fig)
    else:
        return fig


def add_colorbars(
    fig, ax_iterable, im_iterable, position, size, pad,
    ):
    for ax, im in zip(ax_iterable, im_iterable):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position = position, size = size, pad = pad)
        fig.colorbar(im, cax = cax)


def shape_helper(condition, reshape_fn, tensors):
    if condition and reshape_fn is not None:
        return [None if tensor is None else reshape_fn(tensor) 
                for tensor in tensors]
    else:
        ret = []
        for tensor in tensors:
            if tensor is None:
                ret.append(None)
            elif tensor.dim() == 1:
                ret.append(tensor.view(1,-1))
            else:
                ret.append(tensor)
        return ret


# def shape_helper(condition, reshape_fn, weight, pruned, mask = None):
#     if condition and reshape_fn is not None:
#         return (reshape_fn(weight), 
#                 reshape_fn(pruned), 
#                 None if mask is None else reshape_fn(mask))
#     elif weight.dim() == 1:
#         return (weight.view(1,-1), 
#                 pruned.view(1,-1), 
#                 None if mask is None else mask.view(1,-1))


def plot_branch_weights(
    save_path, 
    branches_to_output, 
    logspace = False,
    save_in_dir = False,
    filename = 'image',
    ):
    if logspace:
        block = branches_to_output.log_block()
        title = 'Upstream Branch Log Weights'
    else:
        block = branches_to_output.block()
        title = 'Upstream Branch Weights'
    
    fig, ax = plt.subplots()
    fig.suptitle(title)
    im = ax.imshow(block.detach().numpy(), cmap = 'viridis')

    add_colorbars(fig, [ax], [im], 'right', '5%', 0.05)

    if save_in_dir:
        save_path = os.path.join(save_path, 'branch_weights')
        file_path = os.path.join(save_path, f'{filename}.jpeg')
    else:
        file_path = os.path.join(save_path, 'branch_weights.jpeg')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)    
    
    fig.savefig(file_path)
    plt.close(fig)


def plot_excitation_inhibition_weights(
    soma_branch, save_path, 
    excitatory_synapse_weight, 
    excitatory_synapse_pruned, 
    excitatory_synapse_mask = None,
    inhibitory_synapse_weight = None, 
    inhibitory_synapse_pruned = None, 
    inhibitory_synapse_mask = None,
    reshape_fn = None, 
    reshape_exc_syn = False, 
    reshape_inh_syn = False,
    logspace = False,
    reactivation_curve = None, 
    linspace = None,
    save_in_dir = False,
    filename = 'image',
    ):

    (
        excitatory_synapse_weight, 
        excitatory_synapse_pruned, 
        excitatory_synapse_mask
     ) = shape_helper(
        reshape_exc_syn, reshape_fn,
        (excitatory_synapse_weight,  
        excitatory_synapse_pruned,
        excitatory_synapse_mask),
    )

    nrows, ncols = 2, 2
    height_ratios = []

    if reshape_exc_syn:
        height_ratios.append(10)
    else:
        height_ratios.append(1)

    if excitatory_synapse_mask is not None:
        ncols += 1

    if inhibitory_synapse_weight is not None:
        nrows += 1

        if reshape_inh_syn:
            height_ratios.append(10)
        else:
            height_ratios.append(1)
    
    
    height_ratios.append(10)

    # plotting
    fig = plt.figure(
        # figsize = (8,6),
        constrained_layout = True,
    )
    fig.suptitle(f'Synapses on {soma_branch}')

    gs = GridSpec(nrows, ncols, height_ratios = height_ratios, figure = fig)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('exc syn log weight' if logspace else 'exc syn weight')
    im1 = ax1.imshow(excitatory_synapse_weight.detach().numpy(), 
                     cmap = 'viridis')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('exc syn log pruned' if logspace else 'exc syn pruned')
    im2 = ax2.imshow(excitatory_synapse_pruned.detach().numpy(), 
                     cmap = 'viridis')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax_list = [ax1, ax2]
    im_list = [im1, im2]

    if excitatory_synapse_mask is not None:
        ax3 = fig.add_subplot(gs[0,2])
        ax3.set_title('exc syn mask')
        im3 = ax3.imshow(excitatory_synapse_mask.detach().numpy(), 
                         cmap = 'Greys')
        ax3.set_xticks([])
        ax3.set_yticks([])
        # ax_list.append(ax3)
        # im_list.append(im3)


    if inhibitory_synapse_weight is not None:
        (
            inhibitory_synapse_weight, 
            inhibitory_synapse_pruned, 
            inhibitory_synapse_mask 
        ) = shape_helper(
            reshape_inh_syn, 
            reshape_fn,
            (inhibitory_synapse_weight, 
            inhibitory_synapse_pruned, 
            inhibitory_synapse_mask),
        )

        ax4 = fig.add_subplot(gs[1,0])
        ax4.set_title('inh syn log weight' if logspace else 'inh syn weight')
        im4 = ax4.imshow(inhibitory_synapse_weight.detach().numpy(), 
                         cmap = 'viridis')
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5 = fig.add_subplot(gs[1,1])
        ax5.set_title('inh syn log pruned' if logspace else 'inh syn pruned')
        im5 = ax5.imshow(inhibitory_synapse_pruned.detach().numpy(), 
                         cmap = 'viridis')
        ax5.set_xticks([])
        ax5.set_yticks([])

        ax_list.append(ax4)
        ax_list.append(ax5)
        im_list.append(im4)
        im_list.append(im5)

        if inhibitory_synapse_mask is not None:
            ax6 = fig.add_subplot(gs[1,2])
            ax6.set_title('inh syn mask')
            im6 = ax6.imshow(inhibitory_synapse_mask.detach().numpy(), 
                             cmap = 'Greys')
            ax6.set_xticks([])
            ax6.set_yticks([])

            # ax_list.append(ax6)
            # im_list.append(im6)

    add_colorbars(fig, ax_list, im_list, 'right', '5%', 0.05)

    if excitatory_synapse_mask is not None:
        ax7 = fig.add_subplot(gs[-1,0])
        ax7.set_title('exc syn log pruned dist' if logspace else 'exc syn pruned dist')
        # ax7.hist(
        #     excitatory_synapse_weight.flatten().detach().numpy(), 
        #     bins = 10, alpha = 0.5, color = '0.4', label = 'weight',
        #     )
        ax7.hist(
            excitatory_synapse_pruned[
                excitatory_synapse_mask > 0
            ].flatten().detach().numpy(),
            bins = 10, 
            alpha = 0.5, 
            color = 'blue', 
            # label = 'pruned',
        )
        # ax7.legend()

    if inhibitory_synapse_mask is not None:
        ax8 = fig.add_subplot(gs[-1,1])
        ax8.set_title('inh syn log pruned dist' if logspace else 'inh syn pruned dist')
        # ax8.hist(
        #     inhibitory_synapse_weight.flatten().detach().numpy(), 
        #     bins = 10, alpha = 0.5, color = '0.4', label = 'weight',
        #     )
        ax8.hist(
            inhibitory_synapse_pruned[
                inhibitory_synapse_mask > 0
            ].flatten().detach().numpy(),
            bins = 10, 
            alpha = 0.5, 
            color = 'blue', 
            # label = 'pruned',
        )
        # ax8.legend()

    if reactivation_curve is not None:
        ax9 = fig.add_subplot(gs[-1,2])
        ax9.set_title('reactivation curve')
        ax9.plot(linspace.detach().numpy(), 
                 reactivation_curve.detach().numpy(), 
                 color = 'k')
        ax9.yaxis.tick_right()

    if save_in_dir:
        save_path = os.path.join(save_path, soma_branch)
        file_path = os.path.join(save_path, f'{filename}.jpeg')
    else:
        file_path = os.path.join(save_path, f'{soma_branch}.jpeg')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    
    fig.savefig(file_path)
    plt.close(fig)


def plot_dendrinet_params(
        dendrinet, 
        save_root, 
        reshape_fn = None, 
        reshape_exc_syn = False, 
        reshape_inh_syn = False, 
        logspace = False,
        save_in_dir = False,
        filename = 'image',
    ):
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    input_inh = dendrinet.input_inhibitory
    somatic_synapses = dendrinet.somatic_synapses

    linspace = torch.linspace(0,1,50)

    if somatic_synapses:
        n_layers = dendrinet.n_branch_layers + 1
    else:
        n_layers = dendrinet.n_branch_layers

    for i in range(n_layers):
        level = ('soma' 
                 if i == dendrinet.n_branch_layers 
                 else f'branch layer {dendrinet.n_branch_layers - i}')

        if somatic_synapses and i == n_layers - 1:
            layer = dendrinet.soma_layer
            branches_per_soma = 1
        else:
            layer = dendrinet.branch_layers[i]
            branches_per_soma = int(layer_sizes[i] / n_soma)

        if layer.input_branches:
            plot_branch_weights(
                save_path = os.path.join(save_root, level),
                branches_to_output = layer.branches_to_output,
                logspace = logspace,
                save_in_dir = save_in_dir,
                filename = filename,
            )

        if input_inh:
            if logspace:
                (
                    excitatory_synapse_weight, 
                    inhibitory_synapse_weight
                ) = layer.get_log_weights()
                (
                    excitatory_synapse_pruned, 
                    inhibitory_synapse_pruned 
                ) = layer.get_log_pruned_weights()
            else:
                (
                    excitatory_synapse_weight, 
                    inhibitory_synapse_weight
                ) = layer.get_weights()
                (
                    excitatory_synapse_pruned, 
                    inhibitory_synapse_pruned 
                ) = layer.get_pruned_weights()
            (
                excitatory_synapse_mask, 
                inhibitory_synapse_mask
            ) = layer.get_mask()

            inhibitory_synapse_weight = inhibitory_synapse_weight.chunk(n_soma, 
                                                                        dim = 0)
            inhibitory_synapse_mask = inhibitory_synapse_mask.chunk(n_soma, 
                                                                    dim = 0)
            inhibitory_synapse_pruned = inhibitory_synapse_pruned.chunk(n_soma, 
                                                                        dim = 0)

        else:
            if logspace:
                excitatory_synapse_weight = layer.get_log_weights()
                excitatory_synapse_pruned = layer.get_log_pruned_weights()
            else:
                excitatory_synapse_weight = layer.get_weights()
                excitatory_synapse_pruned = layer.get_pruned_weights()
            excitatory_synapse_mask = layer.get_mask()

        
        excitatory_synapse_weight = excitatory_synapse_weight.chunk(n_soma, 
                                                                    dim = 0)
        excitatory_synapse_mask = excitatory_synapse_mask.chunk(n_soma, 
                                                                dim = 0)
        excitatory_synapse_pruned = excitatory_synapse_pruned.chunk(n_soma, 
                                                                    dim = 0)

        if layer.reactivate:
            reactivation_curve = layer.reactivation(
                linspace.expand(layer_sizes[i],-1).t()
            ).t()
            reactivation_curve = reactivation_curve.chunk(n_soma, dim = 0)

        for j in range(n_soma):
            for k in range(branches_per_soma):
                exc_syn_weight = excitatory_synapse_weight[j][k]
                exc_syn_mask = excitatory_synapse_mask[j][k]
                exc_syn_pruned = excitatory_synapse_pruned[j][k]
                
                inh_syn_weight = (inhibitory_synapse_weight[j][k] 
                                  if input_inh 
                                  else None)
                inh_syn_mask = (inhibitory_synapse_mask[j][k] 
                                if input_inh 
                                else None)
                inh_syn_pruned = (inhibitory_synapse_pruned[j][k] 
                                  if input_inh 
                                  else None)

                react_curve = reactivation_curve[j][k] if layer.reactivate else None

                plot_excitation_inhibition_weights(
                    soma_branch = f'soma{j}_branch{k}', 
                    save_path = os.path.join(save_root, level),
                    excitatory_synapse_weight = exc_syn_weight,
                    excitatory_synapse_pruned = exc_syn_pruned,
                    excitatory_synapse_mask = exc_syn_mask,
                    inhibitory_synapse_weight = inh_syn_weight,
                    inhibitory_synapse_pruned = inh_syn_pruned,
                    inhibitory_synapse_mask = inh_syn_mask,
                    reshape_fn = reshape_fn, 
                    reshape_exc_syn = reshape_exc_syn, 
                    reshape_inh_syn = reshape_inh_syn,
                    logspace = logspace,
                    reactivation_curve = react_curve, 
                    linspace = linspace,
                    save_in_dir = save_in_dir,
                    filename = filename,
                )

    if input_inh:
        if logspace:
            ( 
                excitatory_synapse_weight_total, 
                inhibitory_synapse_weight_total 
            ) = dendrinet.log_sum_weights(pruned = False)
            ( excitatory_synapse_pruned_total, 
            inhibitory_synapse_pruned_total 
            ) = dendrinet.log_sum_weights(pruned = True)
        else:
            ( 
                excitatory_synapse_weight_total, 
                inhibitory_synapse_weight_total 
            ) = dendrinet.sum_weights(pruned = False)
            ( excitatory_synapse_pruned_total, 
            inhibitory_synapse_pruned_total 
            ) = dendrinet.sum_weights(pruned = True)
    else:
        if logspace:
            excitatory_synapse_weight_total = dendrinet.log_sum_weights(pruned = False)
            excitatory_synapse_pruned_total = dendrinet.log_sum_weights(pruned = True)
        else:
            excitatory_synapse_weight_total = dendrinet.sum_weights(pruned = False)
            excitatory_synapse_pruned_total = dendrinet.sum_weights(pruned = True)
        (
            inhibitory_synapse_weight_total, 
            inhibitory_synapse_pruned_total
        ) = [None] * n_soma, [None] * n_soma

    for i in range(n_soma):
        plot_excitation_inhibition_weights(
            soma_branch = f'soma{i} all branches', 
            save_path = os.path.join(save_root, 'weight_totals'),
            excitatory_synapse_weight = excitatory_synapse_weight_total[i],
            excitatory_synapse_pruned = excitatory_synapse_pruned_total[i],
            excitatory_synapse_mask = None,
            inhibitory_synapse_weight = inhibitory_synapse_weight_total[i],
            inhibitory_synapse_pruned = inhibitory_synapse_pruned_total[i],
            inhibitory_synapse_mask = None,
            reshape_fn = reshape_fn, 
            reshape_exc_syn = reshape_exc_syn, 
            reshape_inh_syn = reshape_inh_syn,
            logspace = logspace,
            reactivation_curve = None, 
            linspace = None,
            save_in_dir = save_in_dir,
            filename = filename,
        )


def plot_einet_params(
    einet, 
    save_root, 
    reshape_fn = None, 
    reshape_exc_syn = False, 
    logspace = False,
    save_in_dir = False,
    filename = 'image',
    ):
    einet = einet.to('cpu')

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_params(
            dendrinet = inh_dendrinet, 
            save_root = os.path.join(save_root, 
                                     f'ei_layer{i+1}', 
                                     'inhibitory cells'),
            reshape_fn = reshape_fn, 
            reshape_exc_syn = reshape_exc_syn if i == 0 else False, 
            reshape_inh_syn = False,
            logspace = logspace,
            save_in_dir = save_in_dir,
            filename = filename,
        )
        plot_dendrinet_params(
            dendrinet = exc_dendrinet, 
            save_root = os.path.join(save_root, 
                                     f'ei_layer{i+1}', 
                                     'excitatory cells'),
            reshape_fn = reshape_fn, 
            reshape_exc_syn = reshape_exc_syn if i == 0 else False, 
            reshape_inh_syn = False,
            logspace = logspace,
            save_in_dir = save_in_dir,
            filename = filename,
        )


def plot_eng_mod10_einet_params(
    einet, 
    save_root, 
    reshape_fn = None, 
    logspace = False,
    save_in_dir = False,
    filename = 'image',
    ):
    einet = einet.to('cpu')

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_params(
            dendrinet = inh_dendrinet, 
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory cells'),
            reshape_fn = reshape_fn, 
            reshape_exc_syn = False if i == 2 else True, 
            reshape_inh_syn = False,
            logspace = logspace,
            save_in_dir = save_in_dir,
            filename = filename,
        )
        plot_dendrinet_params(
            dendrinet = exc_dendrinet, 
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory cells'),
            reshape_fn = reshape_fn, 
            reshape_exc_syn = False if i == 2 else True, 
            reshape_inh_syn = False,
            logspace = logspace,
            save_in_dir = save_in_dir,
            filename = filename,
        )


def plot_branch_gradients(
    save_path, 
    branches_to_output, 
    save_in_dir = False,
    filename = 'image',
    ):
    branch_grad = branches_to_output.grad_block()

    fig, ax = plt.subplots()
    fig.suptitle('Upstream Branch Gradients')
    maxabs = torch.max(branch_grad.abs()).item() + 1e-9
    norm = TwoSlopeNorm(vcenter = 0, vmin = -maxabs, vmax = maxabs)
    im = ax.imshow(
        branch_grad.detach().cpu().numpy(), 
        cmap = 'PRGn', norm = norm,
    )

    add_colorbars(fig, [ax], [im], 'right', '5%', 0.05)

    if save_in_dir:
        save_path = os.path.join(save_path, 'branch_grads')
        file_path = os.path.join(save_path, f'{filename}.jpeg')
    else:
        file_path = os.path.join(save_path, 'branch_grads.jpeg')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)    
    
    fig.savefig(file_path)
    plt.close(fig)


def plot_excitation_inhibition_gradients(
    soma_branch, save_path, 
    excitatory_synapse_grad, 
    excitatory_synapse_mask,
    inhibitory_synapse_grad = None, 
    inhibitory_synapse_mask = None,
    log_m_grad = None,
    log_b_grad = None,
    reshape_fn = None, 
    reshape_exc_syn = False, 
    reshape_inh_syn = False,
    save_in_dir = False,
    filename = 'image',
    ):
    (
        excitatory_synapse_grad, 
        excitatory_synapse_mask,
    ) = shape_helper(
        reshape_exc_syn, 
        reshape_fn, 
        (excitatory_synapse_grad, excitatory_synapse_mask),  
    )

    nrows, ncols = 1, 3
    height_ratios = []

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

    # plotting
    fig = plt.figure(
        # figsize = (8,6),
        constrained_layout = True,
    )
    fig.suptitle(f'Synapses on {soma_branch}')

    gs = GridSpec(nrows, ncols, height_ratios = height_ratios, figure = fig)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('exc syn grad')
    maxabs = torch.max(excitatory_synapse_grad.abs()).item() + 1e-9
    norm = TwoSlopeNorm(vcenter = 0, vmin = -maxabs, vmax = maxabs)
    im1 = ax1.imshow(
        excitatory_synapse_grad.detach().cpu().numpy(), 
        cmap = 'PRGn', norm = norm,
    )
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('exc syn mask')
    im2 = ax2.imshow(
        excitatory_synapse_mask.detach().cpu().numpy(), 
        cmap = 'Greys',
    )
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax_list = [ax1]
    im_list = [im1]

    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('exc syn grad dist')
    ax3.hist(
        excitatory_synapse_grad.flatten().detach().cpu().numpy(),
        bins = 20, color = '0.4',
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

        ax4 = fig.add_subplot(gs[1,0])
        ax4.set_title('inh syn grad')
        maxabs = torch.max(inhibitory_synapse_grad.abs()).item() + 1e-9
        norm = TwoSlopeNorm(vcenter = 0, vmin = -maxabs, vmax = maxabs)
        im4 = ax4.imshow(
            inhibitory_synapse_grad.detach().cpu().numpy(), 
            cmap = 'PRGn', norm = norm,
        )
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5 = fig.add_subplot(gs[1,1])
        ax5.set_title('inh syn mask')
        im5 = ax5.imshow(
            inhibitory_synapse_mask.detach().cpu().numpy(), 
            cmap = 'Greys',
        )
        ax5.set_xticks([])
        ax5.set_yticks([])

        ax_list.append(ax4)
        im_list.append(im4)

        ax6 = fig.add_subplot(gs[1,2])
        ax6.set_title('inh syn grad dist')
        ax6.hist(
            inhibitory_synapse_grad.flatten().detach().cpu().numpy(),
            bins = 20, color = '0.4',
        )

    add_colorbars(fig, ax_list, im_list, 'right', '5%', 0.05)

    if log_m_grad is not None:
        ax5 = fig.add_subplot(gs[-1,0])
        # ax9.set_title('reactivation curve')
        # ax9.plot(linspace.detach().numpy(), 
        #          reactivation_curve.detach().numpy(), 
        #          color = 'k')
        # ax9.yaxis.tick_right()

        # ax6 = fig.add_subplot(gs[-1,1])

    if save_in_dir:
        save_path = os.path.join(save_path, soma_branch)
        file_path = os.path.join(save_path, f'{filename}.jpeg')
    else:
        file_path = os.path.join(save_path, f'{soma_branch}.jpeg')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    
    fig.savefig(file_path)
    plt.close(fig)


def plot_dendrinet_gradients(
    dendrinet, 
    save_root, 
    reshape_fn = None, 
    reshape_exc_syn = False, 
    reshape_inh_syn = False, 
    save_in_dir = False,
    filename = 'image',
):
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    somatic_synapses = dendrinet.somatic_synapses

    if somatic_synapses:
        n_layers = dendrinet.n_branch_layers + 1
    else:
        n_layers = dendrinet.n_branch_layers

    for i in range(n_layers):
        level = ('soma' 
                 if i == dendrinet.n_branch_layers 
                 else f'branch layer {dendrinet.n_branch_layers - i}')

        if somatic_synapses and i == n_layers - 1:
            layer = dendrinet.soma_layer
            branches_per_soma = 1
        else:
            layer = dendrinet.branch_layers[i]
            branches_per_soma = int(layer_sizes[i] / n_soma)

        if layer.input_branches:
            if layer.branches_to_output.log_weight.requires_grad:
                branch_grad = True
            else:
                branch_grad = False
        else:
            branch_grad = False
        
        inh_grad = layer.input_inhibitory

        if layer.reactivate:
            if layer.log_m.requires_grad:
                reactivate_grad = True
            else:
                reactivate_grad = False
        else:
            reactivate_grad = False

        if branch_grad:
            plot_branch_gradients(
                save_path = os.path.join(save_root, level),
                branches_to_output = layer.branches_to_output,
                save_in_dir = save_in_dir,
                filename = filename,
            )

        if inh_grad:
            inhibitory_synapse_grad = layer.branch_inhibition.pre_w.grad
            inhibitory_synapse_grad = inhibitory_synapse_grad.chunk(n_soma,
                                                                    dim = 0)
            inhibitory_synapse_mask = layer.branch_inhibition.weight_mask()
            inhibitory_synapse_mask = inhibitory_synapse_mask.chunk(n_soma,
                                                                    dim = 0)

        excitatory_synapse_grad = layer.branch_excitation.pre_w.grad
        excitatory_synapse_grad = excitatory_synapse_grad.chunk(n_soma,
                                                                dim = 0)
        excitatory_synapse_mask = layer.branch_excitation.weight_mask()
        excitatory_synapse_mask = excitatory_synapse_mask.chunk(n_soma,
                                                                dim = 0)

        if reactivate_grad:
            log_m_grad = layer.log_m.grad.chunk(n_soma, dim = 0)
            log_b_grad = layer.log_b.grad.chunk(n_soma, dim = 0)

        for j in range(n_soma):
            for k in range(branches_per_soma):
                exc_syn_grad = excitatory_synapse_grad[j][k]
                exc_syn_mask = excitatory_synapse_mask[j][k]
                
                inh_syn_grad = (inhibitory_synapse_grad[j][k]
                                if inh_grad else None)
                inh_syn_mask = (inhibitory_synapse_mask[j][k]
                                if inh_grad else None)
                
                logm_grad = log_m_grad[j][k] if reactivate_grad else None
                logb_grad = log_b_grad[j][k] if reactivate_grad else None

                plot_excitation_inhibition_gradients(
                    soma_branch = f'soma{j}_branch{k}',
                    save_path = os.path.join(save_root, level),
                    excitatory_synapse_grad = exc_syn_grad,
                    excitatory_synapse_mask = exc_syn_mask,
                    inhibitory_synapse_grad = inh_syn_grad,
                    inhibitory_synapse_mask = inh_syn_mask,
                    log_m_grad = logm_grad,
                    log_b_grad = logb_grad,
                    reshape_fn = reshape_fn,
                    reshape_exc_syn = reshape_exc_syn,
                    reshape_inh_syn = reshape_inh_syn,
                    save_in_dir = save_in_dir,
                    filename = filename,
                )


def plot_einet_gradients(
    model, 
    inputs,
    labels,
    save_root, 
    reshape_fn = None, 
    reshape_exc_syn = False, 
    save_in_dir = False,
    filename = 'image',
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()

    loss = model(inputs.to(device), labels.to(device))
    loss = loss.mean(dim = 0)
    model.zero_grad()
    loss.backward()

    einet = model.net

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_gradients(
            dendrinet = inh_dendrinet,
            save_root = os.path.join(save_root,
                                     f'ei_layer{i+1}',
                                     'inhibitory cells'),
            reshape_fn = reshape_fn,
            reshape_exc_syn = reshape_exc_syn if i == 0 else False,
            reshape_inh_syn = False,
            save_in_dir = save_in_dir,
            filename = filename,
        )
        plot_dendrinet_gradients(
            dendrinet = exc_dendrinet,
            save_root = os.path.join(save_root,
                                     f'ei_layer{i+1}',
                                     'excitatory cells'),
            reshape_fn = reshape_fn,
            reshape_exc_syn = reshape_exc_syn if i == 0 else False,
            reshape_inh_syn = False,
            save_in_dir = save_in_dir,
            filename = filename,
        )
    
    model.eval()


def activation_boxplots(
    activation_list, 
    soma_branch, 
    save_path,
    save_in_dir = False,
    filename = 'image',
    ):
    activation_list = [act.detach().cpu().numpy() for act in activation_list]

    fig = plt.figure()
    plt.boxplot(activation_list)
    plt.title(f'Activation Distribution for {soma_branch}')
    plt.gca().set_xticklabels([i for i in range(10)])
    plt.xlabel('Input Digit')
    plt.ylabel('Values')

    if save_in_dir:
        save_path = os.path.join(save_path, soma_branch)
        file_path = os.path.join(save_path, f'{filename}.jpeg')
    else:
        file_path = os.path.join(save_path, f'{soma_branch}.jpeg')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)

    fig.savefig(file_path)
    plt.close(fig)


def activation_means(
    means, 
    xticks, 
    xtick_labels, 
    save_path,
    save_in_dir = False,
    filename = 'image',
    ):
    means = means.t().detach().cpu().numpy()
    x = means.shape[0]
    y = means.shape[1]
    if x/y > 1:
        figsize = (6, 6*x/y)
    else:
        figsize = (8,6)

    fig, ax = plt.subplots(figsize = figsize)

    im = ax.imshow(means, cmap = 'viridis')

    add_colorbars(fig, [ax], [im], 'right', '5%', 0.05)

    ax.set_yticks([i for i in range(10)])
    #ax.set_xticklabels([i for i in range(10)])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation = 45)

    if save_in_dir:
        save_path = os.path.join(save_path, 'activation_means')
        file_path = os.path.join(save_path, f'{filename}.jpeg')
    else:
        file_path = os.path.join(save_path, 'activation_means.jpeg')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    
    fig.savefig(file_path)
    plt.close(fig)


def plot_dendrinet_activations(
    dendrinet, 
    exc_input_list, 
    inh_input_list, 
    save_root,
    save_in_dir = False,
    filename = 'image',
    ):
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    n_branch_layers = dendrinet.n_branch_layers
    somatic_synapses = dendrinet.somatic_synapses

    output_list = [None] * len(exc_input_list)
    for i in range(n_branch_layers+1):
        try:
            branches_per_soma = int(layer_sizes[i] / n_soma)
        except:
            branches_per_soma = 1

        chunk_list = []

        for j in range(len(exc_input_list)):
            if i == n_branch_layers:
                output_list[j] = dendrinet.soma_layer(
                    exc_input_list[j], inh_input_list[j], output_list[j])
            else:
                output_list[j] = dendrinet.branch_layers[i](
                    exc_input_list[j], inh_input_list[j], output_list[j])
            # list, tuples, tensors -> numbers, somas, branches 
            chunk_list.append(output_list[j].chunk(n_soma, dim = -1))

        level = ('soma' 
                 if (i == n_branch_layers) 
                 else f'branch layer {n_branch_layers - i}')

        means = torch.zeros((n_soma, branches_per_soma, len(exc_input_list)))

        for n in range(n_soma):
            for b in range(branches_per_soma):
                activation_list = []
                for k in range(len(exc_input_list)):
                    chunk = chunk_list[k][n][:,b]
                    activation_list.append(chunk)
                    means[n,b,k] = chunk.mean(dim = 0)
                
                soma_branch = f'soma{n}_branch{b}'
                activation_boxplots(
                    activation_list = activation_list,
                    soma_branch = soma_branch,
                    save_path = os.path.join(save_root, level),
                    save_in_dir = save_in_dir,
                    filename = filename,
                )

        xticks = [branches_per_soma * n for n in range(n_soma)]
        xtick_labels = [f'soma{n}' for n in range(n_soma)]
        activation_means(
            means = means.flatten(0,1),
            xticks = xticks,
            xtick_labels = xtick_labels,
            save_path = os.path.join(save_root, level),
            save_in_dir = save_in_dir,
            filename = filename,
        )


def plot_einet_activations(
    einet, 
    input_list, 
    save_root,
    save_in_dir = False,
    filename = 'image',
    ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)
    
    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_activations(
            dendrinet = inh_dendrinet, 
            exc_input_list = exc_input_list, 
            inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory_cells'),
            save_in_dir = save_in_dir,
            filename = filename,
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], 
                                          inh_input_list[j]))
        inh_input_list = inh_temp

        plot_dendrinet_activations(
            dendrinet = exc_dendrinet, 
            exc_input_list = exc_input_list, 
            inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory_cells'),
            save_in_dir = save_in_dir,
            filename = filename,
        )

        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], 
                                          inh_input_list[j]))
        exc_input_list = exc_temp


def dendrinet_activations_to_csv(
        dendrinet, exc_input_list, inh_input_list, save_root, filename = 'activation_components',
    ):
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    n_branch_layers = dendrinet.n_branch_layers

    digit_data = []
    compartment_data = []
    exc_mean_data = []
    exc_sdev_data = []
    inh_mean_data = []
    inh_sdev_data = []
    vinf_mean_data = []
    vinf_sdev_data = []
    vout_mean_data = []
    vout_sdev_data = []

    branch_input = [None] * len(exc_input_list)
    for i in range(n_branch_layers+1):

        if i == n_branch_layers:
            layer = dendrinet.soma_layer
        else:
            layer = dendrinet.branch_layers[i]
        
        if hasattr(layer, 'branch_excitation'):
            branch_excitation = layer.branch_excitation
            input_excitatory = True
        else:
            input_excitatory = False
        
        if hasattr(layer, 'branch_inhibition'):
            branch_inhibition = layer.branch_inhibition
            input_inhibitory = True
        else:
            input_inhibitory = False
        
        reactivate = layer.reactivate

        try:
            branches_per_soma = int(layer_sizes[i] / n_soma)
        except:
            branches_per_soma = 1

        exc_chunks = []
        if input_inhibitory:
            inh_chunks = []
        vinf_chunks = []
        if reactivate:
            vout_chunks = []

        for j in range(len(exc_input_list)):
            numer = 0
            denom = 1

            if input_excitatory:
                exc = branch_excitation(exc_input_list[j])
                numer = numer + exc
                denom = denom + exc

            if layer.input_branches:
                numer = numer + layer.branches_to_output(branch_input[j])
                denom = denom + layer.branches_to_output.sum_conductances()

            if input_inhibitory:
                inh = branch_inhibition(inh_input_list[j])
                denom = denom + inh

            vinf = numer / denom

            if reactivate:
                vout = layer.reactivation(vinf)

            branch_input[j] = vout if reactivate else vinf
            # list, tuples, tensors -> numbers, somas, branches 
            exc_chunks.append(exc.chunk(n_soma, dim = -1))
            if input_inhibitory:
                inh_chunks.append(inh.chunk(n_soma, dim = -1))
            vinf_chunks.append(vinf.chunk(n_soma, dim = -1))
            if reactivate:
                vout_chunks.append(vout.chunk(n_soma, dim = -1))

        for n in range(n_soma):
            for b in range(branches_per_soma):
                for k in range(len(exc_input_list)):
                    
                    if input_excitatory:
                        exc = exc_chunks[k][n][:,b].detach().cpu()
                        exc_mean_data.append(exc.mean(dim = 0).item())
                        exc_sdev_data.append(exc.std(dim = 0).item())
                    else:
                        exc_mean_data.append(None)
                        exc_sdev_data.append(None)

                    if input_inhibitory:
                        inh = inh_chunks[k][n][:,b].detach().cpu()
                        inh_mean_data.append(inh.mean(dim = 0).item())
                        inh_sdev_data.append(inh.std(dim = 0).item())
                    else:
                        inh_mean_data.append(None)
                        inh_sdev_data.append(None)
                    
                    vinf = vinf_chunks[k][n][:,b].detach().cpu()
                    vinf_mean_data.append(vinf.mean(dim = 0).item())
                    vinf_sdev_data.append(vinf.std(dim = 0).item())

                    if reactivate:
                        vout = vout_chunks[k][n][:,b].detach().cpu()
                        vout_mean_data.append(vout.mean(dim = 0).item())
                        vout_sdev_data.append(vout.std(dim = 0).item())
                    else:
                        vout_mean_data.append(None)
                        vout_sdev_data.append(None)

                    digit_data.append(k)
                    
                    compartment = (f'soma{n}' 
                                   if i == n_branch_layers 
                                   else f'soma{n}_branch{b}')
                    compartment_data.append(compartment)

    digit = pd.Series(digit_data)
    compartment = pd.Series(compartment_data)
    
    g_exc_mean = pd.Series(exc_mean_data)
    g_exc_sdev = pd.Series(exc_sdev_data)
    
    g_inh_mean = pd.Series(inh_mean_data)
    g_inh_sdev = pd.Series(inh_sdev_data)
    
    v_inf_mean = pd.Series(vinf_mean_data)
    v_inf_sdev = pd.Series(vinf_sdev_data)
    
    v_out_mean = pd.Series(vout_mean_data)
    v_out_sdev = pd.Series(vout_sdev_data)

    df = pd.DataFrame({
        'digit' : digit, 'compartment' : compartment,
        'g_exc_mean' : g_exc_mean, 'g_exc_sdev' : g_exc_sdev,
        'g_inh_mean' : g_inh_mean, 'g_inh_sdev' : g_inh_sdev,
        'v_inf_mean' : v_inf_mean, 'v_inf_sdev' : v_inf_sdev,
        'v_out_mean' : v_out_mean, 'v_out_sdev' : v_out_sdev,
    })

    os.makedirs(save_root, exist_ok = True)
    df.to_csv(os.path.join(save_root, f'{filename}.csv'))


def einet_activations_to_csv(einet, input_list, save_root, filename = 'activation_components'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)
    
    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        dendrinet_activations_to_csv(
            dendrinet = inh_dendrinet, 
            exc_input_list = exc_input_list, 
            inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory_cells'),
            filename = filename,
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], 
                                          inh_input_list[j]))
        inh_input_list = inh_temp

        dendrinet_activations_to_csv(
            dendrinet = exc_dendrinet, 
            exc_input_list = exc_input_list, 
            inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory_cells'),
            filename = filename,
        )

        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], inh_input_list[j]))
<<<<<<< HEAD
        exc_input_list = exc_temp
=======
        exc_input_list = exc_temp




def build_dendritic_tree_graph(model, branch_info=None, ablation_info=None):
    """
    Creates a networkx DiGraph where each node corresponds to a (layer_idx, branch_id).
    We'll store 'info' and 'ablation' as node attributes if provided.


    """
    G = nx.DiGraph()
    for i, layer in enumerate(model.branch_layers):
        out_dim = layer.branch_excitation.out_features  # or layer.output_dim
        for b_id in range(out_dim):
            node_id = (i, b_id)
            G.add_node(node_id)
            # store info
            if branch_info and i in branch_info:

                G.nodes[node_id]['info'] = branch_info[i]
            if ablation_info and i in ablation_info:
                G.nodes[node_id]['ablation'] = ablation_info[i]
            else:
                G.nodes[node_id]['ablation'] = 0

    for i in range(model.n_branch_layers - 1):
        out_dim_i = model.branch_layers[i].branch_excitation.out_features
        out_dim_next = model.branch_layers[i+1].branch_excitation.out_features
        for b1 in range(out_dim_i):
            for b2 in range(out_dim_next):
                G.add_edge((i,b1), (i+1, b2))

    if hasattr(model, 'soma_layer'):
        i_soma = model.n_branch_layers
        out_dim_soma = model.soma_layer.branch_excitation.out_features \
                       if model.soma_layer.input_excitatory else model.n_soma
        for b_id in range(out_dim_soma):
            node_id = (i_soma, b_id)
            G.add_node(node_id)
            if branch_info and i_soma in branch_info:
                G.nodes[node_id]['info'] = branch_info[i_soma]
            else:
                G.nodes[node_id]['info'] = 0
            G.nodes[node_id]['ablation'] = 0
        out_dim_last = model.branch_layers[-1].branch_excitation.out_features
        for b1 in range(out_dim_last):
            for b2 in range(out_dim_soma):
                G.add_edge((model.n_branch_layers-1,b1), (i_soma, b2))

    return G

def visualize_dendritic_tree(G, mode='info'):
    """
    G: networkx DiGraph
    mode: 'info' or 'ablation' or any node attribute.
    We'll do a simple layout, color by the chosen attribute.
    """
    pos = nx.spring_layout(G, seed=42)
    node_vals = []
    for n in G.nodes():
        val = G.nodes[n].get(mode, 0)
        node_vals.append(val)

    plt.figure(figsize=(8,6))
    nc = nx.draw_networkx_nodes(G, pos, node_color=node_vals, 
                                cmap='viridis', node_size=400)
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_labels(G, pos, font_color='white')
    plt.colorbar(nc, label=mode)
    plt.title(f"Dendritic Tree colored by {mode}")
    plt.axis('off')
    plt.show()


def plot_depth_info(branch_info):
    """
    If branch_info is {layer_idx: MI}, we can just do a line plot vs layer_idx.
    """
    layers = sorted(branch_info.keys())
    values = [branch_info[l] for l in layers]
    plt.figure()
    plt.plot(layers, values, marker='o')
    plt.xlabel("Layer index (depth)")
    plt.ylabel("Information (MI)")
    plt.title("Branch Info by Depth")
    plt.show()

>>>>>>> 36b16bfaa7370a8eab620257f5609e43d9a2426e
