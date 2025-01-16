"""
plot_utils.py
=============
This module contains utility functions for plotting dendritic model parameters.
"""

import os

import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def shape_helper(condition, reshape_fn, weight, pruned, mask = None):
    if condition and reshape_fn is not None:
        return (reshape_fn(weight), 
                reshape_fn(pruned), 
                None if mask is None else reshape_fn(mask))
    elif weight.dim() == 1:
        return (weight.view(1,-1), 
                pruned.view(1,-1), 
                None if mask is None else mask.view(1,-1))


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
        excitatory_synapse_weight,  
        excitatory_synapse_pruned,
        excitatory_synapse_mask,
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
    fig = plt.figure()
    fig.suptitle(f'Synapses on {soma_branch}')

    gs = GridSpec(nrows, ncols, height_ratios = height_ratios)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('exc syn log weight' if logspace else 'exc syn weight')
    im1 = ax1.imshow(excitatory_synapse_weight.detach().numpy(), 
                     cmap = 'viridis')

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('exc syn log pruned' if logspace else 'exc syn pruned')
    im2 = ax2.imshow(excitatory_synapse_pruned.detach().numpy(), 
                     cmap = 'viridis')

    ax_list = [ax1, ax2]
    im_list = [im1, im2]

    if excitatory_synapse_mask is not None:
        ax3 = fig.add_subplot(gs[0,2])
        ax3.set_title('exc syn mask')
        im3 = ax3.imshow(excitatory_synapse_mask.detach().numpy(), 
                         cmap = 'Greys')
        ax_list.append(ax3)
        im_list.append(im3)


    if inhibitory_synapse_weight is not None:
        (
            inhibitory_synapse_weight, 
            inhibitory_synapse_pruned, 
            inhibitory_synapse_mask 
        ) = shape_helper(
            reshape_inh_syn, 
            reshape_fn,
            inhibitory_synapse_weight, 
            inhibitory_synapse_pruned, 
            inhibitory_synapse_mask,
        )

        ax4 = fig.add_subplot(gs[1,0])
        ax4.set_title('inh syn log weight' if logspace else 'inh syn weight')
        im4 = ax4.imshow(inhibitory_synapse_weight.detach().numpy(), 
                         cmap = 'viridis')

        ax5 = fig.add_subplot(gs[1,1])
        ax5.set_title('inh syn log pruned' if logspace else 'inh syn pruned')
        im5 = ax5.imshow(inhibitory_synapse_pruned.detach().numpy(), 
                         cmap = 'viridis')

        ax_list.append(ax4)
        ax_list.append(ax5)
        im_list.append(im4)
        im_list.append(im5)

        if inhibitory_synapse_mask is not None:
            ax6 = fig.add_subplot(gs[1,2])
            ax6.set_title('inh syn mask')
            im6 = ax6.imshow(inhibitory_synapse_mask.detach().numpy(), 
                             cmap = 'Greys')

            ax_list.append(ax6)
            im_list.append(im6)

    add_colorbars(fig, ax_list, im_list, 'right', '5%', 0.05)

    ax7 = fig.add_subplot(gs[-1,0])
    ax7.set_title('exc syn dist')
    ax7.hist(
        excitatory_synapse_weight.flatten().detach().numpy(), 
        bins = 10, alpha = 0.5, color = '0.4', label = 'weight',
        )
    # use same bin distribution
    ax7.hist(
        excitatory_synapse_pruned[
            excitatory_synapse_pruned > 0
        ].flatten().detach().numpy(),
        bins = 10, 
        alpha = 0.5, 
        color = 'blue', 
        label = 'pruned',
    )
    ax7.legend()

    if inhibitory_synapse_weight is not None:
        ax8 = fig.add_subplot(gs[-1,1])
        ax8.set_title('inh syn dist')
        ax8.hist(
            inhibitory_synapse_weight.flatten().detach().numpy(), 
            bins = 10, alpha = 0.5, color = '0.4', label = 'weight',
            )
        ax8.hist(
            inhibitory_synapse_pruned[
                inhibitory_synapse_pruned > 0
            ].flatten().detach().numpy(),
            bins = 10, 
            alpha = 0.5, 
            color = 'blue', 
            label = 'pruned',
        )
        ax8.legend()

    if reactivation_curve is not None:
        ax9 = fig.add_subplot(gs[-1,2])
        ax9.set_title('reactivation curve')
        ax9.plot(linspace.detach().numpy(), 
                 reactivation_curve.detach().numpy(), 
                 color = 'k')

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
            if somatic_synapses and i == n_branch_layers:
                output_list[j] = dendrinet.soma_layer(exc_input_list[j], 
                                                      inh_input_list[j], 
                                                      output_list[j])
            elif not somatic_synapses and i == n_branch_layers:
                output_list[j] = dendrinet.soma_layer(output_list[j])
            else:
                output_list[j] = dendrinet.branch_layers[i](exc_input_list[j],
                                                            inh_input_list[j], 
                                                            output_list[j])
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
    input_inhibitory = dendrinet.input_inhibitory
    n_branch_layers = dendrinet.n_branch_layers
    # reactivate = dendrinet.reactivate
    # somatic_synapses = dendrinet.somatic_synapses

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
        
        branch_excitation = layer.branch_excitation
        if input_inhibitory:
            branch_inhibition = layer.branch_inhibition
        
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
            exc = branch_excitation(exc_input_list[j])

            numer = exc
            denom = exc + 1

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

                    exc = exc_chunks[k][n][:,b].detach().cpu()
                    exc_mean_data.append(exc.mean(dim = 0).item())
                    exc_sdev_data.append(exc.std(dim = 0).item())

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