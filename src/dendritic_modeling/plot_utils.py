"""
plot_utils.py
=============
This module contains utility functions for plotting dendritic model parameters.
"""

import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx

from dendritic_modeling.utils import split_MNIST_inputs


def plotWithColor(data, title=''):
    """
    Function to plot an image with the scale color bar.
    """
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
    title = None,
):
    """
    Plots the training and validation negative log-likelihood loss curves
    versus epoch. Clamps high loss values at 5 for clarity.
    """
    Train_losses = np.array(train_losses)
    Train_losses[np.where(Train_losses > 5)] = 5
    
    Valid_losses = np.array(valid_losses)
    Valid_losses[np.where(Valid_losses > 5)] = 5
    
    fig = plt.figure()
    
    plt.plot(range(1,num_epochs+1), Train_losses, '0.4', label = 'training')
    plt.plot(range(1,num_epochs+1), Valid_losses, 'b', label = 'validation')
    
    plt.xlabel('Epochs')
    plt.xticks(range(0,num_epochs+1,int(num_epochs // 10) if num_epochs>=10 else 1))
    plt.ylabel('Negative Log Likelihood')
    if title is None:
        plt.title('Loss Curves')
    else:
        plt.title(title)
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
    """
    Adds colorbars to a set of subplots.
    """
    for ax, im in zip(ax_iterable, im_iterable):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position = position, size = size, pad = pad)
        fig.colorbar(im, cax = cax)


def shape_helper(condition, reshape_fn, tensors):
    """
    Helper to optionally reshape a list of tensors if condition is True,
    using reshape_fn, or leave them as is otherwise.
    """
    if condition and reshape_fn is not None:
        return [None if tensor is None else reshape_fn(tensor) for tensor in tensors]
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


def plot_branch_weights(
    save_path, 
    branches_to_output, 
    logspace = False,
    save_in_dir = False,
    filename = 'image',
    ):
    """
    Plots the block weights in a branches_to_output layer (BlockLinear).
    """
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
    """
    Plots excitatory/inhibitory synapse weights, pruned, and mask, plus optional
    reactivation curve, for a single soma/branch.
    """
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

    fig = plt.figure(constrained_layout = True)
    fig.suptitle(f'Synapses on {soma_branch}')

    gs = GridSpec(nrows, ncols, height_ratios = height_ratios, figure = fig)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('exc syn log weight' if logspace else 'exc syn weight')
    im1 = ax1.imshow(excitatory_synapse_weight.detach().numpy(), cmap = 'viridis')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('exc syn log pruned' if logspace else 'exc syn pruned')
    im2 = ax2.imshow(excitatory_synapse_pruned.detach().numpy(), cmap = 'viridis')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax_list = [ax1, ax2]
    im_list = [im1, im2]

    if excitatory_synapse_mask is not None:
        ax3 = fig.add_subplot(gs[0,2])
        ax3.set_title('exc syn mask')
        im3 = ax3.imshow(excitatory_synapse_mask.detach().numpy(), cmap = 'Greys')
        ax3.set_xticks([])
        ax3.set_yticks([])

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
        im4 = ax4.imshow(inhibitory_synapse_weight.detach().numpy(), cmap = 'viridis')
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5 = fig.add_subplot(gs[1,1])
        ax5.set_title('inh syn log pruned' if logspace else 'inh syn pruned')
        im5 = ax5.imshow(inhibitory_synapse_pruned.detach().numpy(), cmap = 'viridis')
        ax5.set_xticks([])
        ax5.set_yticks([])

        ax_list.append(ax4)
        ax_list.append(ax5)
        im_list.append(im4)
        im_list.append(im5)

        if inhibitory_synapse_mask is not None:
            ax6 = fig.add_subplot(gs[1,2])
            ax6.set_title('inh syn mask')
            im6 = ax6.imshow(inhibitory_synapse_mask.detach().numpy(), cmap = 'Greys')
            ax6.set_xticks([])
            ax6.set_yticks([])

    add_colorbars(fig, ax_list, im_list, 'right', '5%', 0.05)

    if excitatory_synapse_mask is not None:
        ax7 = fig.add_subplot(gs[-1,0])
        ax7.set_title('exc syn log pruned dist' if logspace else 'exc syn pruned dist')
        ax7.hist(
            excitatory_synapse_pruned[
                excitatory_synapse_mask > 0
            ].flatten().detach().numpy(),
            bins = 10, 
            alpha = 0.5, 
            color = 'blue',
        )

    if inhibitory_synapse_mask is not None:
        ax8 = fig.add_subplot(gs[-1,1])
        ax8.set_title('inh syn log pruned dist' if logspace else 'inh syn pruned dist')
        ax8.hist(
            inhibitory_synapse_pruned[
                inhibitory_synapse_mask > 0
            ].flatten().detach().numpy(),
            bins = 10, 
            alpha = 0.5, 
            color = 'blue',
        )

    if reactivation_curve is not None:
        ax9 = fig.add_subplot(gs[-1,2])
        ax9.set_title('reactivation curve')
        ax9.plot(linspace.detach().numpy(), reactivation_curve.detach().numpy(), color = 'k')
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
    """
    Plots parameter info for a single dendrinet, including excitatory/inhibitory
    branches, possible reactivation curves, etc.
    """
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    input_inh = dendrinet.input_inhibitory
    somatic_synapses = dendrinet.somatic_synapses
    reactivate = dendrinet.reactivate

    linspace = torch.linspace(0,1,50)

    n_layers = dendrinet.n_branch_layers + 1

    for i in range(n_layers):
        level = ('soma' 
                 if i == dendrinet.n_branch_layers 
                 else f'branch layer {dendrinet.n_branch_layers - i}')

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

        else:
            if logspace:
                excitatory_synapse_weight = layer.get_log_weights()
                excitatory_synapse_pruned = layer.get_log_pruned_weights()
            else:
                excitatory_synapse_weight = layer.get_weights()
                excitatory_synapse_pruned = layer.get_pruned_weights()
            excitatory_synapse_mask = layer.get_mask()

        excitatory_synapse_weight = excitatory_synapse_weight.chunk(n_soma, dim = 0)
        excitatory_synapse_mask = excitatory_synapse_mask.chunk(n_soma, dim = 0)
        excitatory_synapse_pruned = excitatory_synapse_pruned.chunk(n_soma, dim = 0)

        if input_inh:
            inhibitory_synapse_weight = inhibitory_synapse_weight.chunk(n_soma, dim = 0)
            inhibitory_synapse_mask = inhibitory_synapse_mask.chunk(n_soma, dim = 0)
            inhibitory_synapse_pruned = inhibitory_synapse_pruned.chunk(n_soma, dim = 0)

        # Instead of dendrinet.reactivation_layers, use dendrinet.activation_layers
        if reactivate:
            # gather the appropriate activation layer
            activation_layer_i = dendrinet.activation_layers[i]
            # build reactivation curves across the entire layer's dimension
            reactivation_curve = activation_layer_i(
                linspace.expand(layer_sizes[i], -1).t()
            ).t()
            reactivation_curve = reactivation_curve.chunk(n_soma, dim=0)

        for j in range(n_soma):
            branches_to_soma = int(layer_sizes[i] / n_soma)
            for k in range(branches_to_soma):
                exc_syn_weight = excitatory_synapse_weight[j][k]
                exc_syn_mask = excitatory_synapse_mask[j][k]
                exc_syn_pruned = excitatory_synapse_pruned[j][k]
                
                if input_inh:
                    inh_syn_weight = inhibitory_synapse_weight[j][k]
                    inh_syn_mask = inhibitory_synapse_mask[j][k]
                    inh_syn_pruned = inhibitory_synapse_pruned[j][k]
                else:
                    inh_syn_weight = None
                    inh_syn_mask = None
                    inh_syn_pruned = None

                if reactivate:
                    # pick the portion from reactivation_curve for this branch
                    react_curve = reactivation_curve[j][k]
                else:
                    react_curve = None

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
                exc_total, 
                inh_total
            ) = dendrinet.log_sum_weights(pruned = False)
            (
                exc_total_pruned, 
                inh_total_pruned
            ) = dendrinet.log_sum_weights(pruned = True)
        else:
            (
                exc_total, 
                inh_total
            ) = dendrinet.sum_weights(pruned = False)
            (
                exc_total_pruned, 
                inh_total_pruned
            ) = dendrinet.sum_weights(pruned = True)
    else:
        if logspace:
            exc_total = dendrinet.log_sum_weights(pruned = False)
            exc_total_pruned = dendrinet.log_sum_weights(pruned = True)
            inh_total = [None]*dendrinet.n_soma
            inh_total_pruned = [None]*dendrinet.n_soma
        else:
            exc_total = dendrinet.sum_weights(pruned = False)
            exc_total_pruned = dendrinet.sum_weights(pruned = True)
            inh_total = [None]*dendrinet.n_soma
            inh_total_pruned = [None]*dendrinet.n_soma

    for i in range(dendrinet.n_soma):
        plot_excitation_inhibition_weights(
            soma_branch = f'soma{i} all branches', 
            save_path = os.path.join(save_root, 'weight_totals'),
            excitatory_synapse_weight = exc_total[i],
            excitatory_synapse_pruned = exc_total_pruned[i],
            excitatory_synapse_mask = None,
            inhibitory_synapse_weight = inh_total[i],
            inhibitory_synapse_pruned = inh_total_pruned[i],
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
    """
    Plots param info for an ExcitationInhibitionNetwork with multiple layers.
    """
    einet = einet.to('cpu')

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_params(
            dendrinet = inh_dendrinet, 
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory cells'),
            reshape_fn = reshape_fn, 
            reshape_exc_syn = reshape_exc_syn if i == 0 else False, 
            reshape_inh_syn = False,
            logspace = logspace,
            save_in_dir = save_in_dir,
            filename = filename,
        )
        plot_dendrinet_params(
            dendrinet = exc_dendrinet, 
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory cells'),
            reshape_fn = reshape_fn, 
            reshape_exc_syn = reshape_exc_syn if i == 0 else False, 
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
    """
    Plots gradient color map for a BlockLinear layer.
    """
    branch_grad = branches_to_output.grad_block()

    fig, ax = plt.subplots()
    fig.suptitle('Upstream Branch Gradients')
    maxabs = torch.max(branch_grad.abs()).item() + 1e-9
    norm = TwoSlopeNorm(vcenter = 0, vmin = -maxabs, vmax = maxabs)
    im = ax.imshow(branch_grad.detach().cpu().numpy(), cmap = 'PRGn', norm = norm)

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
    """
    Plots gradient color maps for excitatory/inhibitory weights, plus optional
    mask and reactivation parameter grads.
    """
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

    fig = plt.figure(constrained_layout = True)
    fig.suptitle(f'Synapses on {soma_branch}')

    gs = GridSpec(nrows, ncols, height_ratios = height_ratios, figure = fig)

    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title('exc syn grad')
    maxabs = torch.max(excitatory_synapse_grad.abs()).item() + 1e-9
    norm = TwoSlopeNorm(vcenter=0, vmin=-maxabs, vmax=maxabs)
    im1 = ax1.imshow(excitatory_synapse_grad.detach().cpu().numpy(), cmap='PRGn', norm=norm)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('exc syn mask')
    im2 = ax2.imshow(excitatory_synapse_mask.detach().cpu().numpy(), cmap='Greys')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax_list = [ax1]
    im_list = [im1]

    ax3 = fig.add_subplot(gs[0,2])
    ax3.set_title('exc syn grad dist')
    ax3.hist(excitatory_synapse_grad.flatten().detach().cpu().numpy(), bins=20, color='0.4')

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
        norm = TwoSlopeNorm(vcenter=0, vmin=-maxabs, vmax=maxabs)
        im4 = ax4.imshow(inhibitory_synapse_grad.detach().cpu().numpy(), cmap='PRGn', norm=norm)
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5 = fig.add_subplot(gs[1,1])
        ax5.set_title('inh syn mask')
        im5 = ax5.imshow(inhibitory_synapse_mask.detach().cpu().numpy(), cmap='Greys')
        ax5.set_xticks([])
        ax5.set_yticks([])

        ax_list.append(ax4)
        im_list.append(im4)

        ax6 = fig.add_subplot(gs[1,2])
        ax6.set_title('inh syn grad dist')
        ax6.hist(inhibitory_synapse_grad.flatten().detach().cpu().numpy(), bins=20, color='0.4')

    add_colorbars(fig, ax_list, im_list, 'right', '5%', 0.05)

    if log_m_grad is not None:
        ax5 = fig.add_subplot(gs[-1,0])
        # optionally plot reactivation param gradient
        # code can be added here if needed

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
    """
    Plots gradient color maps for a DendriNet (exc + inh branches, etc.).
    """
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    reactivate = dendrinet.reactivate

    n_layers = dendrinet.n_branch_layers + 1

    for i in range(n_layers):
        level = ('soma' if i == dendrinet.n_branch_layers else f'branch layer {dendrinet.n_branch_layers - i}')
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

        if branch_grad:
            plot_branch_gradients(
                save_path = os.path.join(save_root, level),
                branches_to_output = layer.branches_to_output,
                save_in_dir = save_in_dir,
                filename = filename,
            )

        if inh_grad:
            inhibitory_synapse_grad = layer.branch_inhibition.pre_w.grad
            inhibitory_synapse_grad = inhibitory_synapse_grad.chunk(n_soma, dim = 0)
            inhibitory_synapse_mask = layer.branch_inhibition.weight_mask()
            inhibitory_synapse_mask = inhibitory_synapse_mask.chunk(n_soma, dim = 0)

        excitatory_synapse_grad = layer.branch_excitation.pre_w.grad
        excitatory_synapse_grad = excitatory_synapse_grad.chunk(n_soma, dim = 0)
        excitatory_synapse_mask = layer.branch_excitation.weight_mask()
        excitatory_synapse_mask = excitatory_synapse_mask.chunk(n_soma, dim = 0)

        for j in range(n_soma):
            for k in range(branches_per_soma):
                exc_syn_grad = excitatory_synapse_grad[j][k]
                exc_syn_mask = excitatory_synapse_mask[j][k]
                
                if inh_grad:
                    inh_syn_grad = inhibitory_synapse_grad[j][k]
                    inh_syn_mask = inhibitory_synapse_mask[j][k]
                else:
                    inh_syn_grad = None
                    inh_syn_mask = None

                plot_excitation_inhibition_gradients(
                    soma_branch = f'soma{j}_branch{k}',
                    save_path = os.path.join(save_root, level),
                    excitatory_synapse_grad = exc_syn_grad,
                    excitatory_synapse_mask = exc_syn_mask,
                    inhibitory_synapse_grad = inh_syn_grad,
                    inhibitory_synapse_mask = inh_syn_mask,
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
    """
    Plots gradient color maps for each layer in an ExcitationInhibitionNetwork
    after a forward/backward pass on the data.
    """
    from torch.utils.data import DataLoader
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
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory cells'),
            reshape_fn = reshape_fn,
            reshape_exc_syn = reshape_exc_syn if i == 0 else False,
            reshape_inh_syn = False,
            save_in_dir = save_in_dir,
            filename = filename,
        )
        plot_dendrinet_gradients(
            dendrinet = exc_dendrinet,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory cells'),
            reshape_fn = reshape_fn,
            reshape_exc_syn = reshape_exc_syn if i == 0 else False,
            reshape_inh_syn = False,
            save_in_dir = save_in_dir,
            filename = filename,
        )
    
    model.eval()