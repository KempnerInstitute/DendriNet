
import torch
import numpy as np
from torch.utils.data import DataLoader

################################################################################
# 1) RECORD BRANCH ACTIVATIONS
################################################################################
def record_branch_activations(model, dataset, batch_size=64, device='cpu'):
    """
    For a given DendriNetWithOutputs, forward the entire dataset and record 
    the dendritic branch activations at each layer.

    Returns:
       branch_activations: dict {layer_idx: tensor of shape [N, output_dim]}
       all_labels: tensor of shape [N]

    Requirements:
       model.forward_with_branch_outputs(x) -> (final, [layer1_act, layer2_act, ..., soma_act])
    """
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # We'll store each layer's activation in a list, then cat them.
    # branch_activations[layer_idx] = list of [batch_size, layer_output_dim]
    branch_activations = {}
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            # forward
            final_output, per_layer_acts = model.forward_with_branch_outputs(x_batch)

            # per_layer_acts is a list, length = n_branch_layers + 1 (soma)
            for layer_idx, layer_act in enumerate(per_layer_acts):
                if layer_idx not in branch_activations:
                    branch_activations[layer_idx] = []
                branch_activations[layer_idx].append(layer_act.cpu())

            all_labels.append(y_batch.cpu())

    for layer_idx in branch_activations:
        branch_activations[layer_idx] = torch.cat(branch_activations[layer_idx], dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return branch_activations, all_labels


################################################################################
# 2) SIMPLE MUTUAL INFORMATION
################################################################################
def compute_MI(activations, labels, n_bins=10):
    """
    Estimate the mutual information I(activations; labels).
    Here, we do a 1D approach:
      - if activations is [N], or [N,1], we bin them
      - build joint histogram with labels
      - compute I(X;Y) = sum p(x,y) log [ p(x,y)/(p(x)p(y)) ]

    If activations has shape [N, out_dim] with out_dim>1, 
    you might do a dimension reduction or loop over dims. 
    For simplicity, we'll do mean across dim if >1.
    """
    if activations.ndim > 1 and activations.shape[1] > 1:
        # e.g. reduce dimension
        activations = activations.mean(dim=1)

    N = len(activations)
    # bin the activation
    act_min, act_max = activations.min().item(), activations.max().item()
    if act_min == act_max:
        return 0.0

    edges = torch.linspace(act_min, act_max, n_bins+1)
    bin_indices = torch.bucketize(activations, edges) - 1  # [0..n_bins-1]

    num_classes = int(labels.max().item() + 1)
    joint_hist = torch.zeros((n_bins, num_classes), dtype=torch.float)
    for b, c in zip(bin_indices, labels):
        joint_hist[b, c] += 1

    joint_prob = joint_hist / joint_hist.sum()
    act_prob = joint_prob.sum(dim=1, keepdim=True)   # shape [n_bins,1]
    label_prob = joint_prob.sum(dim=0, keepdim=True) # shape [1,num_classes]

    eps = 1e-12
    ratio = (joint_prob+eps) / ((act_prob+eps)*(label_prob+eps))
    mi = (joint_prob * ratio.log2()).sum()
    return mi.item()

################################################################################
# 3) COMPUTE BRANCH INFO
################################################################################
def analyze_branch_information(model, dataset, 
                               batch_size=64, device='cpu', n_bins=10):
    """
    1) Gathers branch activations
    2) For each branch (layer_idx), compute MI with the label
    3) Return dict {layer_idx: MI_value}
    """
    branch_acts, labels = record_branch_activations(model, dataset, batch_size, device)
    branch_info = {}
    for layer_idx, act in branch_acts.items():
        mi_val = compute_MI(act, labels, n_bins=n_bins)
        branch_info[layer_idx] = mi_val
    return branch_info


################################################################################
# 4) ABLATION
################################################################################
def measure_performance(model, dataset, batch_size=64, device='cpu'):
    """
    Evaluate model's classification accuracy (or NLL) on the dataset.
    Modify as needed for your model's output.
    """
    from torch.utils.data import DataLoader

    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(x_batch)  # if EINet -> could be distribution or logits
            # if it's distribution with .logits
            if hasattr(output, 'logits'):
                preds = output.logits.argmax(dim=-1)
            else:
                # assume raw logits
                preds = output.argmax(dim=-1)

            total_correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_correct / total

def ablate_branch(model, layer_idx, branch_ids, mode='direct'):
    """
    Zero out excitatory/inhibitory weights for branch(es) at layer_idx.
    - mode='direct': only zero out excit/inhib pre_w for these branches
    - mode='full': also cut upstream if desired

    Adjust to match your DendriticBranchLayer structure. This is conceptual:
      model.branch_layers[layer_idx] -> e.g. has .branch_excitation.pre_w, .branch_inhibition.pre_w
    """
    layer = model.branch_layers[layer_idx]
    
    if mode == 'direct':
        if layer.input_excitatory:
            for b_id in branch_ids:
                layer.branch_excitation.pre_w.data[b_id, :] = 0.
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            for b_id in branch_ids:
                layer.branch_inhibition.pre_w.data[b_id, :] = 0.

    elif mode == 'full':
        if layer.input_excitatory:
            for b_id in branch_ids:
                layer.branch_excitation.pre_w.data[b_id, :] = 0.
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            for b_id in branch_ids:
                layer.branch_inhibition.pre_w.data[b_id, :] = 0.
        # plus zero out blockLinear if used
        if layer.input_branches and hasattr(layer, 'branches_to_output'):
            # ex: zero out the relevant rows/cols in layer.branches_to_output.log_weight

            pass

def analyze_branch_ablation(model, dataset, layer_idx, branch_ids,
                            mode='direct', device='cpu'):
    """
    1) measure baseline accuracy
    2) for each branch_id in branch_ids, ablate, measure accuracy drop
    3) restore weights
    """
    baseline_acc = measure_performance(model, dataset, device=device)
    results = {}
    # store backup
    original = {}
    for n, p in model.named_parameters():
        original[n] = p.detach().clone()

    for b_id in branch_ids:
        ablate_branch(model, layer_idx, [b_id], mode=mode)
        acc_after = measure_performance(model, dataset, device=device)
        delta = baseline_acc - acc_after
        results[b_id] = (acc_after, delta)

        # restore
        for n, p in model.named_parameters():
            p.data.copy_(original[n])
    
    return baseline_acc, results