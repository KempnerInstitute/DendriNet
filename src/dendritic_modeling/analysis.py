# branch_analysis.py

import torch
import numpy as np
import math
from copy import deepcopy
from torch.utils.data import DataLoader

from dendritic_modeling.training import TrainerMLE
from dendritic_modeling.utils import accuracy_score

###############################################################################
# 1. Collecting Branch Outputs
###############################################################################
def collect_branch_outputs(model, dataset, batch_size=64, device='cpu'):
    """
    Returns a dictionary: 
        branch_outputs[ (layer_idx, branch_idx) ] = [list of scalar outputs for each sample]
    If your code has multiple outputs per branch, you can store vectors or do something else.
    
    We assume:
    - model is an instance of DendriNet or something that 
      can expose internal dendritic branches.
    - each DendriticBranchLayer can return a 'branch_voltage' or something 
    
    Implementation approach:
    1) We define forward hooks or override the forward method to store each branch output.
       or
    2) We provide a special function "model.forward_with_branch_outputs(...)"
       that returns the branch outputs as well as the final output.
    """
    model.eval()
    branch_outputs = {}

    hooks = []

    def make_hook(layer_idx, branch_outputs_dict):
        def hook(module, input, output):
            # 'output' here is the final voltage or activation from that branch layer
            # But we need to know the 'branch_idx' if multiple branches are in that layer
            # In your code, each DendriticBranchLayer might represent multiple branches 
            # or just one set of 'output_dim'. We'll assume each is "one branch for each out neuron".
            # We'll store them as: branch_outputs[(layer_idx, i)] = ...
            # if 'output' is shape [batch_size, output_dim], we might store 
            # the mean across output_dim, or store them all. We'll do a mean as an example:
            
            # shape: [batch_size, output_dim]
            # We'll do a mean across output_dim to get a single scalar (just for demonstration).
            mean_per_sample = output.mean(dim=1).detach().cpu().numpy()
            if (layer_idx) not in branch_outputs_dict:
                branch_outputs_dict[layer_idx] = []
            branch_outputs_dict[layer_idx].append(mean_per_sample)
        return hook

    # Register the hook on each branch layer:
    # e.g. "model.branch_layers" is a ModuleList of DendriticBranchLayer
    for idx, layer in enumerate(model.branch_layers):
        h = layer.register_forward_hook(make_hook(idx, branch_outputs))
        hooks.append(h)

    soma_hook = model.soma_layer.register_forward_hook(make_hook('soma', branch_outputs))
    hooks.append(soma_hook)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x, y in loader:
        x = x.to(device)
        with torch.no_grad():
            _ = model(x)  # triggers hooks

    for h in hooks:
        h.remove()

    final_branch_outputs = {}
    for k, list_of_arrays in branch_outputs.items():
        final_branch_outputs[k] = np.concatenate(list_of_arrays, axis=0)  # shape [N,], single scalar each sample
        # or shape [N, output_dim] if we didn't do the .mean(dim=1)

    return final_branch_outputs


###############################################################################
# 2. Estimating "Information" for each branch
###############################################################################
def compute_branch_information(branch_outputs, labels, bins=10):
    """
    Example: we estimate mutual information I(BranchOutput; Label)
    by binning the branch output and label.

    Inputs:
    - branch_outputs: dict { layer_idx: array of shape [N,] } for each sample
    - labels: array shape [N,]
    - bins: int, number of bins for the branch output if you do naive binning

    Returns: info_dict: { layer_idx: estimated_mutual_info_value }

    This is a naive histogram approach: discretize branch outputs, compute p(b, y).
    More sophisticated methods might be better.
    """
    info_dict = {}
    labels = labels.numpy() if torch.is_tensor(labels) else labels
    N = len(labels)

    unique_labels = np.unique(labels)

    for layer_idx, out_vals in branch_outputs.items():
        hist, xedges, yedges = np.histogram2d(out_vals, labels, bins=[bins, len(unique_labels)])
        pxy = hist / hist.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)

        # compute MI = sum p(x,y) log [ p(x,y)/(p(x)p(y)) ]
        eps = 1e-12
        ratio = pxy / (px @ py + eps)
        valid = pxy > 0
        mi = np.sum(pxy[valid] * np.log2(ratio[valid] + eps))
        info_dict[layer_idx] = mi

    return info_dict


###############################################################################
# 3. Ablation: Zero Out the Exc & Inh Weights for One Branch
###############################################################################
def ablate_branch(model, layer_idx, branch_idx=None):
    """
    Zero out the excitatory & inhibitory synapses that feed that layer.

    If layer_idx is an integer indexing model.branch_layers,
    we zero that layer's branch_excitation and branch_inhibition pre_w, 
    only for excitatory input (NOT upstream branch inputs).
    """
    layer = None
    if layer_idx == 'soma':
        layer = model.soma_layer
    else:
        layer = model.branch_layers[layer_idx]

    if layer.input_excitatory:
        with torch.no_grad():
            layer.branch_excitation.pre_w[:] = 0.0

    if layer.input_inhibitory:
        with torch.no_grad():
            layer.branch_inhibition.pre_w[:] = 0.0


def restore_branch(model, layer_idx, exc_backup, inh_backup):
    """
    Put back the excitatory & inhibitory weights from backups.
    """
    layer = None
    if layer_idx == 'soma':
        layer = model.soma_layer
    else:
        layer = model.branch_layers[layer_idx]

    if layer.input_excitatory:
        with torch.no_grad():
            layer.branch_excitation.pre_w[:] = exc_backup

    if layer.input_inhibitory:
        with torch.no_grad():
            layer.branch_inhibition.pre_w[:] = inh_backup


###############################################################################
# 4. Measuring performance drop after ablating each branch
###############################################################################
def measure_performance_drop(model, layer_idx, dataset_dict, 
                             trainer_config=None, device='cpu'):
    """
    1. Backup the current excitatory/inhibitory pre_w for that layer
    2. Zero them
    3. Measure new performance (accuracy or cross-entropy) on train/valid/test
    4. Restore
    5. Return the difference

    dataset_dict = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
    trainer_config is some dict specifying batch_size, etc.

    """
    layer = None
    if layer_idx == 'soma':
        layer = model.soma_layer
    else:
        layer = model.branch_layers[layer_idx]

    exc_backup = None
    inh_backup = None
    if layer.input_excitatory:
        exc_backup = layer.branch_excitation.pre_w.clone()
    if layer.input_inhibitory:
        inh_backup = layer.branch_inhibition.pre_w.clone()

    batch_size = trainer_config.get('batch_size', 64)
    train_loader = DataLoader(dataset_dict['train'], batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset_dict['valid'], batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset_dict['test'],  batch_size=batch_size, shuffle=False)
    model.eval()

    with torch.no_grad():
        train_acc_orig = accuracy_score(model, *next(iter(train_loader))).item()
        valid_acc_orig = accuracy_score(model, *next(iter(valid_loader))).item()
        test_acc_orig  = accuracy_score(model, *next(iter(test_loader))).item()

    ablate_branch(model, layer_idx)

    with torch.no_grad():
        train_acc_ablat = accuracy_score(model, *next(iter(train_loader))).item()
        valid_acc_ablat = accuracy_score(model, *next(iter(valid_loader))).item()
        test_acc_ablat  = accuracy_score(model, *next(iter(test_loader))).item()

    restore_branch(model, layer_idx, exc_backup, inh_backup)

    return {
        'train_perf_drop': train_acc_orig - train_acc_ablat,
        'valid_perf_drop': valid_acc_orig - valid_acc_ablat,
        'test_perf_drop':  test_acc_orig  - test_acc_ablat,
        'train_acc_orig': train_acc_orig,
        'valid_acc_orig': valid_acc_orig,
        'test_acc_orig':  test_acc_orig,
        'train_acc_ablat': train_acc_ablat,
        'valid_acc_ablat': valid_acc_ablat,
        'test_acc_ablat':  test_acc_ablat
    }


###############################################################################
# 5. Putting It All Together:
#    Gathering info, ablating each branch, collecting "info" & "info_loss"
###############################################################################
def gather_dendritic_analysis(model, train_ds, valid_ds, test_ds, 
                              trainer_config=None, 
                              device='cpu', 
                              bins=10):
    """
    1. Collect branch outputs over the *train* set (or combined train+valid) 
       to estimate mutual information with the label.
    2. For each layer, do ablation => measure performance drop
    3. Return a structure like:
       {
         layer_idx: {
           'info': <float>,
           'perf_drop': <float or dict of (train, valid, test)>,
           ...
         },
         ...
       }
    """
    model.to(device)
    model.eval()

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=trainer_config.get('batch_size',64), shuffle=False)
    all_labels = []
    for x, y in loader:
        all_labels.append(y)
    all_labels = torch.cat(all_labels, dim=0)

    branch_outs = collect_branch_outputs(model, train_ds, 
                                         batch_size=trainer_config.get('batch_size',64),
                                         device=device)
    info_dict = compute_branch_information(branch_outs, all_labels, bins=bins)

    dataset_dict = {'train': train_ds, 'valid': valid_ds, 'test': test_ds}
    results = {}
    layer_indices = list(range(len(model.branch_layers))) + ['soma']

    for lidx in layer_indices:
        perf_drop = measure_performance_drop(model, lidx, dataset_dict, 
                                             trainer_config=trainer_config, device=device)
        info_val = info_dict[lidx] if lidx in info_dict else None
        results[lidx] = {
            'info': info_val,
            'perf_drop': perf_drop
        }

    return results