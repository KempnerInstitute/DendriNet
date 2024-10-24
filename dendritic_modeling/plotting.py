import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd

import os


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


def add_colorbars(
    fig, ax_iterable, im_iterable, position, size, pad,
    ):
    for ax, im in zip(ax_iterable, im_iterable):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position = position, size = size, pad = pad)
        fig.colorbar(im, cax = cax)


def shape_helper(condition, reshape_fn, weight, pruned, mask = None):
    if condition and reshape_fn is not None:
        return reshape_fn(weight), reshape_fn(pruned), None if mask is None else reshape_fn(mask)
    elif weight.dim() == 1:
        return weight.view(1,-1), pruned.view(1,-1), None if mask is None else mask.view(1,-1)


def plot_excitation_inhibition_weights(
    soma_branch, save_path, 
    excitatory_synapse_weight, excitatory_synapse_pruned, excitatory_synapse_mask = None,
    inhibitory_synapse_weight = None, inhibitory_synapse_pruned = None, inhibitory_synapse_mask = None,
    reshape_fn = None, reshape_exc_syn = False, reshape_inh_syn = False, logspace = False,
    reactivation_curve = None, linspace = None,
    ):

    excitatory_synapse_weight, excitatory_synapse_pruned, excitatory_synapse_mask = shape_helper(
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
    im1 = ax1.imshow(excitatory_synapse_weight.detach().numpy(), cmap = 'viridis')

    ax2 = fig.add_subplot(gs[0,1])
    ax2.set_title('exc syn log pruned' if logspace else 'exc syn pruned')
    im2 = ax2.imshow(excitatory_synapse_pruned.detach().numpy(), cmap = 'viridis')

    ax_list = [ax1, ax2]
    im_list = [im1, im2]

    if excitatory_synapse_mask is not None:
        ax3 = fig.add_subplot(gs[0,2])
        ax3.set_title('exc syn mask')
        im3 = ax3.imshow(excitatory_synapse_mask.detach().numpy(), cmap = 'Greys')
        ax_list.append(ax3)
        im_list.append(im3)


    if inhibitory_synapse_weight is not None:
        inhibitory_synapse_weight, inhibitory_synapse_pruned, inhibitory_synapse_mask = shape_helper(
            reshape_inh_syn, reshape_fn,
            inhibitory_synapse_weight, 
            inhibitory_synapse_pruned, 
            inhibitory_synapse_mask,
        )

        ax4 = fig.add_subplot(gs[1,0])
        ax4.set_title('inh syn log weight' if logspace else 'inh syn weight')
        im4 = ax4.imshow(inhibitory_synapse_weight.detach().numpy(), cmap = 'viridis')

        ax5 = fig.add_subplot(gs[1,1])
        ax5.set_title('inh syn log pruned' if logspace else 'inh syn pruned')
        im5 = ax5.imshow(inhibitory_synapse_pruned.detach().numpy(), cmap = 'viridis')

        ax_list.append(ax4)
        ax_list.append(ax5)
        im_list.append(im4)
        im_list.append(im5)

        if inhibitory_synapse_mask is not None:
            ax6 = fig.add_subplot(gs[1,2])
            ax6.set_title('inh syn mask')
            im6 = ax6.imshow(inhibitory_synapse_mask.detach().numpy(), cmap = 'Greys')

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
        excitatory_synapse_pruned[excitatory_synapse_pruned > 0].flatten().detach().numpy(),
        bins = 10, alpha = 0.5, color = 'blue', label = 'pruned',
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
            inhibitory_synapse_pruned[inhibitory_synapse_pruned > 0].flatten().detach().numpy(),
            bins = 10, alpha = 0.5, color = 'blue', label = 'pruned',
        )
        ax8.legend()

    if reactivation_curve is not None:
        ax9 = fig.add_subplot(gs[-1,2])
        ax9.set_title('reactivation curve')
        ax9.plot(linspace.detach().numpy(), reactivation_curve.detach().numpy(), color = 'k')

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    
    fig.savefig(os.path.join(save_path, f'{soma_branch}.jpeg'))
    plt.close(fig)


def plot_dendrinet_params(
        dendrinet, save_root, 
        reshape_fn = None, reshape_exc_syn = False, reshape_inh_syn = False, logspace = False,
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
        if somatic_synapses and i == n_layers - 1:
            layer = dendrinet.soma_layer
            branches_per_soma = 1
        else:
            layer = dendrinet.branch_layers[i]
            branches_per_soma = int(layer_sizes[i] / n_soma)

        if input_inh:
            if logspace:
                excitatory_synapse_weight, inhibitory_synapse_weight = layer.get_log_weights()
                excitatory_synapse_pruned, inhibitory_synapse_pruned = layer.get_log_pruned_weights()
            else:
                excitatory_synapse_weight, inhibitory_synapse_weight = layer.get_weights()
                excitatory_synapse_pruned, inhibitory_synapse_pruned = layer.get_pruned_weights()
            excitatory_synapse_mask, inhibitory_synapse_mask = layer.get_mask()

            inhibitory_synapse_weight = inhibitory_synapse_weight.chunk(n_soma, dim = 0)
            inhibitory_synapse_mask = inhibitory_synapse_mask.chunk(n_soma, dim = 0)
            inhibitory_synapse_pruned = inhibitory_synapse_pruned.chunk(n_soma, dim = 0)

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

        if layer.reactivate:
            reactivation_curve = layer.reactivation(linspace.expand(layer_sizes[i],-1).t()).t()
            reactivation_curve = reactivation_curve.chunk(n_soma, dim = 0)

        level = 'soma' if i == dendrinet.n_branch_layers else f'branch layer {dendrinet.n_branch_layers - i}'

        for j in range(n_soma):
            for k in range(branches_per_soma):
                exc_syn_weight = excitatory_synapse_weight[j][k]
                exc_syn_mask = excitatory_synapse_mask[j][k]
                exc_syn_pruned = excitatory_synapse_pruned[j][k]
                
                inh_syn_weight = inhibitory_synapse_weight[j][k] if input_inh else None
                inh_syn_mask = inhibitory_synapse_mask[j][k] if input_inh else None
                inh_syn_pruned = inhibitory_synapse_pruned[j][k] if input_inh else None

                react_curve = reactivation_curve[j][k] if layer.reactivate else None

                plot_excitation_inhibition_weights(
                    soma_branch = f'soma{j}_branch{k}', save_path = os.path.join(save_root, level),
                    excitatory_synapse_weight = exc_syn_weight,
                    excitatory_synapse_pruned = exc_syn_pruned,
                    excitatory_synapse_mask = exc_syn_mask,
                    inhibitory_synapse_weight = inh_syn_weight,
                    inhibitory_synapse_pruned = inh_syn_pruned,
                    inhibitory_synapse_mask = inh_syn_mask,
                    reshape_fn = reshape_fn, reshape_exc_syn = reshape_exc_syn, reshape_inh_syn = reshape_inh_syn,
                    logspace = logspace, reactivation_curve = react_curve, linspace = linspace,
                )

    if input_inh:
        if logspace:
            excitatory_synapse_weight_total, inhibitory_synapse_weight_total = dendrinet.log_sum_weights(pruned = False)
            excitatory_synapse_pruned_total, inhibitory_synapse_pruned_total = dendrinet.log_sum_weights(pruned = True)
        else:
            excitatory_synapse_weight_total, inhibitory_synapse_weight_total = dendrinet.sum_weights(pruned = False)
            excitatory_synapse_pruned_total, inhibitory_synapse_pruned_total = dendrinet.sum_weights(pruned = True)
    else:
        if logspace:
            excitatory_synapse_weight_total = dendrinet.log_sum_weights(pruned = False)
            excitatory_synapse_pruned_total = dendrinet.log_sum_weights(pruned = True)
        else:
            excitatory_synapse_weight_total = dendrinet.sum_weights(pruned = False)
            excitatory_synapse_pruned_total = dendrinet.sum_weights(pruned = True)
        inhibitory_synapse_weight_total, inhibitory_synapse_pruned_total = [None] * n_soma, [None] * n_soma

    for i in range(n_soma):
        plot_excitation_inhibition_weights(
            soma_branch = f'soma{i} all branches', save_path = os.path.join(save_root, 'weight_totals'),
            excitatory_synapse_weight = excitatory_synapse_weight_total[i],
            excitatory_synapse_pruned = excitatory_synapse_pruned_total[i],
            excitatory_synapse_mask = None,
            inhibitory_synapse_weight = inhibitory_synapse_weight_total[i],
            inhibitory_synapse_pruned = inhibitory_synapse_pruned_total[i],
            inhibitory_synapse_mask = None,
            reshape_fn = reshape_fn, reshape_exc_syn = reshape_exc_syn, reshape_inh_syn = reshape_inh_syn,
            logspace = logspace, reactivation_curve = None, linspace = None,
        )


def plot_einet_params(
        einet, save_root, reshape_fn = None, reshape_exc_syn = False, logspace = False,
    ):
    einet = einet.to('cpu')

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_params(
            dendrinet = inh_dendrinet, save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory cells'),
            reshape_fn = reshape_fn, reshape_exc_syn = reshape_exc_syn if i == 0 else False, reshape_inh_syn = False,
            logspace = logspace,
        )
        plot_dendrinet_params(
            dendrinet = exc_dendrinet, save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory cells'),
            reshape_fn = reshape_fn, reshape_exc_syn = reshape_exc_syn if i == 0 else False, reshape_inh_syn = False,
            logspace = logspace,
        )


def plot_eng_mod10_einet_params(einet, save_root, reshape_fn = None, logspace = False):
    einet = einet.to('cpu')

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_params(
            dendrinet = inh_dendrinet, save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory cells'),
            reshape_fn = reshape_fn, reshape_exc_syn = False if i == 2 else True, reshape_inh_syn = False,
            logspace = logspace,
        )
        plot_dendrinet_params(
            dendrinet = exc_dendrinet, save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory cells'),
            reshape_fn = reshape_fn, reshape_exc_syn = False if i == 2 else True, reshape_inh_syn = False,
            logspace = logspace,
        )



def activation_boxplots(activation_list, title, fname, save_path):
    activation_list = [act.detach().cpu().numpy() for act in activation_list]

    fig = plt.figure()
    plt.boxplot(activation_list)
    plt.title(title)
    plt.gca().set_xticklabels([i for i in range(10)])
    plt.xlabel('Input Digit')
    plt.ylabel('Values')

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)

    fig.savefig(os.path.join(save_path, fname))
    plt.close(fig)


def activation_means(means, xticks, xtick_labels, save_path):
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

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    
    fig.savefig(os.path.join(save_path, 'activation_means.jpeg'))
    plt.close(fig)


def plot_dendrinet_activations(
        dendrinet, exc_input_list, inh_input_list, save_root,
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
                output_list[j] = dendrinet.soma_layer(exc_input_list[j], inh_input_list[j], output_list[j])
            elif not somatic_synapses and i == n_branch_layers:
                output_list[j] = dendrinet.soma_layer(output_list[j])
            else:
                output_list[j] = dendrinet.branch_layers[i](exc_input_list[j], inh_input_list[j], output_list[j])
            # list, tuples, tensors -> numbers, somas, branches 
            chunk_list.append(output_list[j].chunk(n_soma, dim = -1))

        level = 'soma' if (i == n_branch_layers) else f'branch layer {n_branch_layers - i}'

        means = torch.zeros((n_soma, branches_per_soma, len(exc_input_list)))

        for n in range(n_soma):
            for b in range(branches_per_soma):
                activation_list = []
                for k in range(len(exc_input_list)):
                    chunk = chunk_list[k][n][:,b]
                    activation_list.append(chunk)
                    means[n,b,k] = chunk.mean(dim = 0)
                
                fname = f'soma{n}_branch{b}'
                activation_boxplots(
                    activation_list = activation_list,
                    title = f'Activation Distribution for {fname}',
                    fname = fname, save_path = os.path.join(save_root, level),
                )

        xticks = [branches_per_soma * n for n in range(n_soma)]
        xtick_labels = [f'soma{n}' for n in range(n_soma)]
        activation_means(means.flatten(0,1), xticks, xtick_labels, os.path.join(save_root, level))


def plot_einet_activations(einet, input_list, save_root):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)
    
    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dendrinet_activations(
            dendrinet = inh_dendrinet, exc_input_list = exc_input_list, inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory_cells'),
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], inh_input_list[j]))
        inh_input_list = inh_temp

        plot_dendrinet_activations(
            dendrinet = exc_dendrinet, exc_input_list = exc_input_list, inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory_cells'),
        )

        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], inh_input_list[j]))
        exc_input_list = exc_temp



def dendrinet_activations_to_csv(
        dendrinet, exc_input_list, inh_input_list, save_root,
    ):
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    input_inhibitory = dendrinet.input_inhibitory
    n_branch_layers = dendrinet.n_branch_layers
    # reactivate = dendrinet.reactivate
    # somatic_synapses = dendrinet.somatic_synapses

    digit_data = []
    compartment_data = []
    exc_data = []
    inh_data = []
    vinf_data = []
    vout_data = []

    branch_input = [None] * len(exc_input_list)
    for i in range(n_branch_layers+1):

        if i == n_branch_layers:
            layer = dendrinet.soma_layer
        else:
            layer = dendrinet.branch_layers[i]
        
        branch_excitation = layer.branch_excitation
        if input_inhibitory:
            branch_inhibition = layer.branch_inhibition

        try:
            branches_per_soma = int(layer_sizes[i] / n_soma)
        except:
            branches_per_soma = 1

        exc_chunks = []
        if input_inhibitory:
            inh_chunks = []
        vinf_chunks = []
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
            vout = layer.reactivation(vinf)

            branch_input[j] = vout
            # list, tuples, tensors -> numbers, somas, branches 
            exc_chunks.append(exc.chunk(n_soma, dim = -1))
            if input_inhibitory:
                inh_chunks.append(inh.chunk(n_soma, dim = -1))
            vinf_chunks.append(vinf.chunk(n_soma, dim = -1))
            vout_chunks.append(vout.chunk(n_soma, dim = -1))

        for n in range(n_soma):
            for b in range(branches_per_soma):
                for k in range(len(exc_input_list)):

                    exc = exc_chunks[k][n][:,b]
                    inh = inh_chunks[k][n][:,b] if input_inhibitory else [None] * exc.shape[0]
                    vinf = vinf_chunks[k][n][:,b]
                    vout = vinf_chunks[k][n][:,b]

                    digit_data += [k] * exc.shape[0]
                    
                    compartment = f'soma{n}' if i == n_branch_layers else f'soma{n}_branch{b}'
                    compartment_data += [compartment] * exc.shape[0]

                    exc_data.append(exc)
                    inh_data.append(inh)
                    vinf_data.append(vinf)
                    vout_data.append(vout)

    digit = pd.Series(digit_data)
    compartment = pd.Series(compartment_data)
    g_exc = pd.Series(torch.cat(exc_data, dim = 0).detach().cpu().numpy())
    g_inh = pd.Series(torch.cat(inh_data, dim = 0).detach().cpu().numpy() if input_inhibitory else inh_data)
    v_inf = pd.Series(torch.cat(vinf_data, dim = 0).detach().cpu().numpy())
    v_out = pd.Series(torch.cat(vout_data, dim = 0).detach().cpu().numpy())

    df = pd.DataFrame({
        'digit' : digit, 'compartment' : compartment,
        'g_exc' : g_exc, 'g_inh' : g_inh,
        'v_inf' : v_inf, 'v_out' : v_out,
    })

    os.makedirs(save_root, exist_ok = True)
    df.to_csv(os.path.join(save_root, 'activation_components.csv'))


def einet_activations_to_csv(einet, input_list, save_root):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)
    
    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        dendrinet_activations_to_csv(
            dendrinet = inh_dendrinet, exc_input_list = exc_input_list, inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'inhibitory_cells'),
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], inh_input_list[j]))
        inh_input_list = inh_temp

        dendrinet_activations_to_csv(
            dendrinet = exc_dendrinet, exc_input_list = exc_input_list, inh_input_list = inh_input_list,
            save_root = os.path.join(save_root, f'ei_layer{i+1}', 'excitatory_cells'),
        )

        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], inh_input_list[j]))
        exc_input_list = exc_temp



#@title dendrite weights
# plt.rcParams["figure.figsize"] = (20,3)

# print("inputs_to_inhibition")
# plt.xlabel("pixel")
# plt.ylabel("inh unit")
# plotWithColor(net.inputs_to_inhibition.weight.data)

# print("inputs_to_inhibition mask")
# plt.xlabel("pixel")
# plt.ylabel("inh unit")
# plotWithColor(net.inputs_to_inhibition.mask.data)

# plt.rcParams["figure.figsize"] = (3,20)

# print("inhibition_to_dendrite_branch")
# plt.ylabel("inh unit")
# plt.xlabel("dendrite")
# plotWithColor(net.inhibition_to_dendrite_branch.weight.data.t())

# print("inhibition_to_dendrite_branch net")
# plt.ylabel("inh unit net")
# plt.xlabel("dendrite")
# plotWithColor(net.inhibition_to_dendrite_branch.net_weight.data.t())

# print("inhibition_to_dendrite_branch mask")
# plt.ylabel("inh unit mask")
# plt.xlabel("dendrite")
# plotWithColor(net.inhibition_to_dendrite_branch.mask.data.t())

# print("inhibition_to_dendrite")
# plt.ylabel("inh unit")
# plt.xlabel("dendrite")
# plotWithColor(net.inhibition_to_dendrite.weight.data.t())

# print("inhibition_to_dendrite net")
# plt.ylabel("inh unit net")
# plt.xlabel("dendrite")
# plotWithColor(net.inhibition_to_dendrite.net_weight.data.t())

# print("inhibition_to_dendrite mask")
# plt.ylabel("inh unit mask")
# plt.xlabel("dendrite")
# plotWithColor(net.inhibition_to_dendrite.mask.data.t())




# #@title inhibition weights
# plt.rcParams["figure.figsize"] = (20,6)

# print("inputs_to_dendrite_branch")
# plt.xlabel("pixel")
# plt.ylabel("branch")
# plotWithColor(net.inputs_to_dendrite_branch.weight.data)

# print("inputs_to_dendrite_branch net")
# plt.xlabel("pixel")
# plt.ylabel("branch net")
# plotWithColor(net.inputs_to_dendrite_branch.net_weight.data)

# print("inputs_to_dendrite_branch mask")
# plt.xlabel("pixel")
# plt.ylabel("branch")
# plotWithColor(net.inputs_to_dendrite_branch.mask.data)

# print("inputs_to_dendrite")
# plt.xlabel("pixel")
# plt.ylabel("dendrite")
# plotWithColor(net.inputs_to_dendrite.weight.data)

# print("inputs_to_dendrite mask")
# plt.xlabel("pixel")
# plt.ylabel("dendrite")
# plotWithColor(net.inputs_to_dendrite.mask.data)

# print("dendrite_branch_to_dendrite")
# plt.xlabel("dendrite branch")
# plt.ylabel("dendrite")
# plotWithColor(net.dendrite_branch_to_dendrite.weight.data)

# print("dendrite_to_soma")
# plt.xlabel("dendrite")
# plt.ylabel("soma")
# plotWithColor(net.dendrite_to_soma.weight.data)



# plotWithColor(net.dendrites_to_soma.mask.data.cpu())



# labels_sort, labels_sort_ind=torch.sort(labels)

# images_sort=images[labels_sort_ind]
# plt.plot(labels_sort.cpu().numpy())

# images_solo=images_raw.view(-1,28*28).cuda()
# labels_solo_sort, labels_solo_sort_ind=torch.sort(labels_raw)
# images_solo_sort=images_solo[labels_solo_sort_ind]
# plt.plot(labels_solo_sort.cpu().numpy())



# with torch.no_grad():
#     output=net(transform_images(images_solo_sort))



# plt.rcParams["figure.figsize"] = (20,3)
# plotWithColor(net.inhibition_act.t(), 'inhibition_act')
# plotWithColor(net.inhibition_out.t(), 'inhibition_out')



# plotWithColor(net.dendrite_branch_inhibition, 'dendrite_branch_inhibition')
# plotWithColor(net.dendrite_branch_excitation, 'dendrite_branch_excitation')
# plotWithColor(net.dendrite_branch_act, 'dendrite_branch_act')
# plotWithColor(net.dendrite_branch_out, 'dendrite_branch_out')



# plotWithColor(net.dendrite_inhibition.t(), 'dendrite_inhibition')
# plotWithColor(net.dendrite_excitation.t(), 'dendrite_excitation')
# plotWithColor(net.dendrite_depolarization.t(), 'dendrite_depolarization')
# plotWithColor(net.dendrite_act.t(), 'dendrite_act')
# plotWithColor(net.dendrite_out.t(), 'dendrite_out')



# plt.rcParams["figure.figsize"] = (10,3)
# plotWithColor(net.soma_act.t(), 'soma_act')
# # plotWithColor(net.soma_out.t())



#@title The shape of the total excitatory input to each main dendrite of each soma

#fig.suptitle('The shape of the total excitatory each main dendrite of each soma', fontsize=16)
# for soma_counter in range(soma_n):
#     fig, axs = plt.subplots(1, dendrite_per_soma)
#     fig.set_figwidth(20)

# for dendrite_counter in range(dendrite_per_soma):
#     imAvg=0*net.inputs_to_dendrite_branch.net_weight.data[0,:].reshape(56,28)
#     for counter in range(dendrite_branch_per_dendrite):
#         start_dendrite_branch=soma_counter*dendrite_branch_per_soma +  dendrite_counter*dendrite_branch_per_dendrite
#         imAvg+=net.inputs_to_dendrite_branch.net_weight.data[start_dendrite_branch+counter,:].reshape(56,28)
#     axs[dendrite_counter].imshow(imAvg.cpu().numpy())
#     axs[dendrite_counter].set_xlabel(dendrite_counter)
# axs[0].set_ylabel(soma_counter)




#@title The shape of the total excitatory input to all the dendrites of each soma
# fig, axs = plt.subplots(1, 10)
# fig.set_figwidth(20)
# fig.suptitle('The shape of the total excitatory input to each soma', fontsize=16)

# for soma_counter in range(10):
#     imAvg=0*net.inputs_to_dendrite_branch.net_weight.data[0,:].reshape(56,28);
#     for counter in range(soma_counter*dendrite_branch_per_soma, (soma_counter+1)*dendrite_branch_per_soma):
#         imAvg+=net.inputs_to_dendrite_branch.net_weight.data[counter,:].reshape(56,28)
#     axs[soma_counter].imshow(imAvg.cpu().numpy())




#@title The shape of the total inhibitory input directly to the main dendrites of each soma
# inhIndices=net.inhibition_to_dendrite.topk_indices.flatten()
# fig, axs = plt.subplots(1, soma_n)
# fig.set_figwidth(20)
# for soma_index in range(soma_n):
#     imAvg=0*net.inputs_to_inhibition.weight.data[0,:].reshape(28,56)
#     for counter in range(soma_index*dendrite_per_soma, (soma_index+1)*dendrite_per_soma):
#         imAvg+=net.inputs_to_inhibition.net_weight.data[inhIndices[counter],:].reshape(56,28)
#     fig.suptitle('The shape of the total inhibitory input directly onto dendrites of each soma', fontsize=16)
#     axs[soma_index].imshow(imAvg.cpu().numpy())

# inhIndices=net.inhibition_to_dendrite_branch.topk_indices.flatten()
# fig, axs = plt.subplots(1, soma_n)
# fig.set_figwidth(20)
# for soma_index in range(soma_n):
#     imAvg=0*net.inputs_to_inhibition.weight.data[0,:].reshape(56,28);
#     for counter in range(soma_index*dendrite_branch_per_soma, (soma_index+1)*dendrite_branch_per_soma):
#         imAvg+=net.inputs_to_inhibition.net_weight.data[inhIndices[counter],:].reshape(56,28)
#     fig.suptitle('The shape of the total inhibitory input to all dendrite branches of each soma', fontsize=16)
#     axs[soma_index].imshow(imAvg.cpu().numpy())



#@title The shape of the input to each inhibitory cell
# plt.rcParams["figure.figsize"] = (20,9)
# from math import ceil

# inhIndices=net.inhibition_to_dendrite_branch.topk_indices.flatten()
# fig, axs = plt.subplots(ceil(inhibitory_units_n/10), 10)
# fig.set_figwidth(20)
# row_counter=0
# for inh_index in range(inhibitory_units_n):
#     fig.suptitle('The shape of the input to each inhibitory cell', fontsize=16)
#     axs[row_counter, inh_index % 10].imshow(net.inputs_to_inhibition.net_weight.data[inh_index,:].reshape(56,28).cpu().numpy())
#     if torch.where(inhIndices==inh_index)[0].numel()==0:
#         axs[row_counter, inh_index % 10].set_title('not used')
#     else:
#         axs[row_counter, inh_index % 10].set_title('used')

#     if inh_index % 10==9:
#         row_counter+=1



#@title The shape of the excitatory and inhibitory input to each dendrite of each soma
# inhIndices=net.inhibition_to_dendrite_branch.topk_indices.flatten()

# plt.rcParams["figure.figsize"] = (20,3.5)
# for soma_index in range(soma_n):
#     fig, axs = plt.subplots(2, dendrite_branch_per_soma)
#     fig.set_figwidth(20)
#     offset=soma_index*dendrite_branch_per_soma
#     for counter in range(dendrite_branch_per_soma):
#         fig.suptitle(soma_index, fontsize=16)
#         axs[0, counter].imshow(net.inputs_to_dendrite_branch.net_weight.data[offset+counter,:].reshape(56,28).cpu().numpy())
#         axs[1, counter].imshow(net.inputs_to_inhibition.weight.data[inhIndices[offset+counter],:].reshape(56,28).cpu().numpy())
#         axs[1, counter].set_title(inhIndices[offset+counter].item())
# axs[0, 0].set_ylabel('exc')
# axs[1, 0].set_ylabel('inh')









