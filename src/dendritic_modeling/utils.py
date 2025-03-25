"""
utils.py
========
This module contains utility functions for data loading, data splitting, 
and model evaluation.
"""

import os
import json
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
import random
from omegaconf import DictConfig, ListConfig
from sklearn.metrics import accuracy_score
import logging
import functools


class Shaper(object):
    """
    A utility class for reshaping input data to a specified shape.
    """
    def __init__(self, shape):
        self.shape = shape
    
    def reshape(self, x):
        return x.view(*self.shape)

def convert_to_serializable(obj):
    """
    Recursively convert OmegaConf objects to standard Python data types.
    """
    if isinstance(obj, DictConfig):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def set_seed(seed: int):
    """
    Set the seed for the random number generator.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_dict(dict_obj: dict, save_path: str, fname: str):
    dict_obj = convert_to_serializable(dict_obj)
    full_path = os.path.join(save_path, fname)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    with open(full_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)


def accuracy_score(classifier: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = classifier.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)
    yhat = classifier.predict(inputs)
    accuracy = torch.mean((yhat == labels).float()).item()
    return accuracy


def evaluate_accuracy(
    classifier: torch.nn.Module, 
    train_ds: Optional[torch.utils.data.Dataset] = None, 
    valid_ds: Optional[torch.utils.data.Dataset] = None, 
    test_ds: Optional[torch.utils.data.Dataset] = None,
    save_path: Optional[str] = None,
    filename: Optional[str] = 'accuracy',
):
    train_acc = accuracy_score(classifier, *train_ds[:]) if train_ds else None
    valid_acc = accuracy_score(classifier, *valid_ds[:]) if valid_ds else None
    test_acc  = accuracy_score(classifier, *test_ds[:]) if test_ds else None

    if save_path is not None:
        save_dict(
            {
                'train accuracy': train_acc,
                'valid accuracy': valid_acc,
                'test accuracy': test_acc
            },
            save_path,
            filename
        )
        
    return train_acc, valid_acc, test_acc


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
    we might do a dimension reduction or loop over dims. 
    For simplicity, we'll do mean across dim if >1.
    """
    if activations.ndim > 1 and activations.shape[1] > 1:
        activations = activations.mean(dim=1)

    N = len(activations)
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
                preds = output.argmax(dim=-1)

            total_correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return total_correct / total


def ablate_branch(model, layer_idx, branch_ids=None, mode='direct'):
    """
    Zero out excitatory/inhibitory weights for branch(es) at layer_idx.
    - If branch_ids is None, ablate ALL dendrites at this layer.
    - If branch_ids is a list of indices, ablate only those branches.
    - 'mode=direct' zeros only excit/inhib pre_w, not blockLinear or upstream.

    This ensures we don't remove signals from previous layers, so deeper layers
    can still matter after ablating a lower layer.

    We need to adjust to match DendriticBranchLayer structure. Typically:
      model.branch_layers[layer_idx] -> has .branch_excitation and .branch_inhibition
    """
    layer = model.branch_layers[layer_idx]
    
    if branch_ids is None:
        if layer.input_excitatory:
            layer.branch_excitation.pre_w.data.zero_()
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            layer.branch_inhibition.pre_w.data.zero_()

    elif mode == 'direct':
        if layer.input_excitatory:
            for b_id in branch_ids:
                layer.branch_excitation.pre_w.data[b_id, :] = 0.
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            for b_id in branch_ids:
                layer.branch_inhibition.pre_w.data[b_id, :] = 0.

    elif mode == 'full':
        if layer.input_excitatory:
            if branch_ids is None:
                layer.branch_excitation.pre_w.data.zero_()
            else:
                for b_id in branch_ids:
                    layer.branch_excitation.pre_w.data[b_id, :] = 0.
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            if branch_ids is None:
                layer.branch_inhibition.pre_w.data.zero_()
            else:
                for b_id in branch_ids:
                    layer.branch_inhibition.pre_w.data[b_id, :] = 0.
        # We do NOT zero out blockLinear or anything that kills upstream input.
        # That way deeper layers still get signals from earlier layers.


def analyze_branch_ablation(model, dataset, layer_idx, branch_ids,
                            mode='direct', device='cpu'):
    """
    1) measure baseline accuracy
    2) for each branch_id in branch_ids, ablate, measure accuracy drop
    3) restore weights

    If we pass branch_ids=None, it ablates the entire layer. 
    """
    baseline_acc = measure_performance(model, dataset, device=device)
    results = {}
    original = {}
    for n, p in model.named_parameters():
        original[n] = p.detach().clone()

    # If branch_ids is None, we do a single ablation of the whole layer
    # If it's a list, we do one ablation per branch in that list
    if branch_ids is None:
        ablate_branch(model, layer_idx, branch_ids=None, mode=mode)
        acc_after = measure_performance(model, dataset, device=device)
        delta = baseline_acc - acc_after
        results["all_branches"] = (acc_after, delta)

        for n, p in model.named_parameters():
            p.data.copy_(original[n])
    else:
        # ablate each branch ID separately
        for b_id in branch_ids:
            ablate_branch(model, layer_idx, [b_id], mode=mode)
            acc_after = measure_performance(model, dataset, device=device)
            delta = baseline_acc - acc_after
            results[b_id] = (acc_after, delta)

            # restore
            for n, p in model.named_parameters():
                p.data.copy_(original[n])
    
    return baseline_acc, results





def compute_entropy(prob_dist):
    """
    Given a discrete probability distribution (torch tensor),
    compute the Shannon entropy: H = - sum(p * log2 p).
    """
    prob_dist = prob_dist.clamp(min=1e-12)
    return -(prob_dist * prob_dist.log2()).sum(dim=-1)

def compute_mutual_information(joint_prob, marginal_x, marginal_y):
    """
    MI(X,Y) = sum_{x,y} p(x,y) log( p(x,y) / [p(x)p(y)] ).
    All inputs are torch tensors of probabilities.
    """
    joint_prob = joint_prob.clamp(min=1e-12)
    marginal_x = marginal_x.clamp(min=1e-12)
    marginal_y = marginal_y.clamp(min=1e-12)
    
    ratio = joint_prob / (marginal_x[:,None] * marginal_y[None,:])
    return (joint_prob * ratio.log2()).sum()



def estimate_hidden_output_MI(model, data_loader, n_bins=10):
    """
    Discretize a hidden layer's activation into n_bins and estimate
    mutual information with the class label. This is a rough approximation:
    1. Collect hidden activation for each sample & label
    2. Bin them to get histogram p(hidden_bin, label)
    3. Compare to p(hidden_bin)*p(label)
    """
    model.eval()
    hidden_acts = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            # forward pass
            # let's extract the second-to-last layer's output or any hidden layer
            # For demonstration, assume "model.net.layers[-1].excitatory_cells"
            # or we adapt the model forward pass to return hidden acts
            activations = model.net.layers[-1].excitatory_cells.forward(x)
            hidden_acts.append(activations.cpu())
            labels.append(y.cpu())

    hidden_acts = torch.cat(hidden_acts, dim=0)  # shape [N, n_excitatory]
    labels = torch.cat(labels, dim=0)           # shape [N]

    # We do this for a single dimension or average across dimensions
    # for simplicity, let's just do dimension 0:
    h0 = hidden_acts[:, 0]
    min_val, max_val = h0.min(), h0.max()

    bin_edges = torch.linspace(min_val, max_val, n_bins+1)
    bin_indices = torch.bucketize(h0, bin_edges) - 1  # in [0, n_bins-1]
    

    n_classes = labels.max().item() + 1
    joint_hist = torch.zeros((n_bins, n_classes), dtype=torch.float)
    for b, c in zip(bin_indices, labels):
        joint_hist[b, c] += 1
    joint_prob = joint_hist / joint_hist.sum()

    marginal_hidden = joint_prob.sum(dim=1)  # sum across classes
    marginal_label = joint_prob.sum(dim=0)   # sum across bins


    mi_est = compute_mutual_information(joint_prob, marginal_hidden, marginal_label)
    return mi_est.item()

class MLPClassifier(nn.Module):
    """
    A simple MLP classifier for decoding layer activations.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


def train_classifier(model, train_loader, valid_loader=None, epochs=50, lr=0.001, device='cuda'):
    """
    Train an MLP classifier on activation data.
    
    Parameters
    ----------
    model : nn.Module
        The classifier model to train
    train_loader : DataLoader
        DataLoader with training data
    valid_loader : DataLoader, optional
        DataLoader with validation data
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    device : str
        Device to use for training
        
    Returns
    -------
    float
        Final validation accuracy
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_valid_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        if valid_loader is not None:
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            valid_acc = accuracy_score(all_targets, all_preds)
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return evaluate_classifier(model, valid_loader, device)


def evaluate_classifier(model, data_loader, device='cuda'):
    """
    Evaluate a classifier on the given data.
    
    Parameters
    ----------
    model : nn.Module
        The classifier model to evaluate
    data_loader : DataLoader
        DataLoader with evaluation data
    device : str
        Device to use for evaluation
        
    Returns
    -------
    float
        Classification accuracy
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return accuracy_score(all_targets, all_preds)


def get_layer_activations(model, data_loader, device='cuda'):
    """
    Get activations for each layer in the model.
    
    Parameters
    ----------
    model : DendriNetWithOutputs
        The dendritic network model
    data_loader : DataLoader
        DataLoader with input data
    device : str
        Device to use
        
    Returns
    -------
    dict
        Dictionary mapping layer indices to (activations, labels) tuples
    """
    activations_per_layer = {}
    all_labels = []
    first_batch = True
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            
            # Get activations from all layers
            _, layer_activations = model.forward_with_branch_outputs(inputs)
            
            # Store activations and labels
            for i, act in enumerate(layer_activations):
                act_flat = act.view(act.size(0), -1).cpu()
                
                if first_batch:
                    activations_per_layer[i] = act_flat
                else:
                    activations_per_layer[i] = torch.cat([activations_per_layer[i], act_flat], dim=0)
            
            if first_batch:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)
                
            first_batch = False
    
    # Create tuples of (activations, labels) for each layer
    return {layer_idx: (acts, all_labels) for layer_idx, acts in activations_per_layer.items()}


def get_ablated_activations(model, data_loader, ablation_type, device='cuda'):
    """
    Get activations with specific inputs ablated.
    
    Parameters
    ----------
    model : DendriNetWithOutputs
        The dendritic network model
    data_loader : DataLoader
        DataLoader with input data
    ablation_type : str
        Type of ablation: 'inhibitory' or 'excitatory'
    device : str
        Device to use
        
    Returns
    -------
    dict
        Dictionary mapping layer indices to list of ablated activations
    """
    all_activations = {}
    all_labels = []
    first_batch = True
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            
            # Save original parameters
            original_params = {}
            for layer_idx, layer in enumerate(model.branch_layers):
                if ablation_type == 'inhibitory' and hasattr(layer, 'branch_inhibition'):
                    original_params[layer_idx] = layer.branch_inhibition.pre_w.data.clone()
                elif ablation_type == 'excitatory' and hasattr(layer, 'branch_excitation'):
                    original_params[layer_idx] = layer.branch_excitation.pre_w.data.clone()
            
            # For each layer, ablate that layer and get activations
            for layer_idx in range(len(model.branch_layers)):
                # Ablate the specified input type for this layer
                if ablation_type == 'inhibitory' and hasattr(model.branch_layers[layer_idx], 'branch_inhibition'):
                    model.branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(float('-inf'))
                elif ablation_type == 'excitatory' and hasattr(model.branch_layers[layer_idx], 'branch_excitation'):
                    model.branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(float('-inf'))
                
                # Get activations with this layer's inputs ablated
                _, layer_activations = model.forward_with_branch_outputs(inputs)
                
                # Restore the original parameters
                if ablation_type == 'inhibitory' and hasattr(model.branch_layers[layer_idx], 'branch_inhibition'):
                    model.branch_layers[layer_idx].branch_inhibition.pre_w.data.copy_(original_params[layer_idx])
                elif ablation_type == 'excitatory' and hasattr(model.branch_layers[layer_idx], 'branch_excitation'):
                    model.branch_layers[layer_idx].branch_excitation.pre_w.data.copy_(original_params[layer_idx])
                
                # Store activations for this layer
                for i, act in enumerate(layer_activations):
                    act_flat = act.view(act.size(0), -1).cpu()
                    
                    if first_batch:
                        if i not in all_activations:
                            all_activations[i] = {}
                        all_activations[i][layer_idx] = act_flat
                    else:
                        all_activations[i][layer_idx] = torch.cat([all_activations[i][layer_idx], act_flat], dim=0)
            
            if first_batch:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)
                
            first_batch = False
    
    # Create tuples of (activations, labels) for each layer
    return {layer_idx: {ablated_idx: (acts, all_labels) for ablated_idx, acts in layer_dict.items()} 
            for layer_idx, layer_dict in all_activations.items()}


def analyze_layerwise_ablation(model, train_data, valid_data, test_data, save_path, device='cuda'):
    """
    Run comprehensive ablation analysis on the dendritic network.
    
    Parameters
    ----------
    model : DendriNetWithOutputs
        The dendritic network model
    train_data : Dataset
        Training dataset
    valid_data : Dataset
        Validation dataset
    test_data : Dataset
        Test dataset
    save_path : str
        Path to save results
    device : str
        Device to use
        
    Returns
    -------
    dict
        Dictionary with ablation results
    """
    logging.info("Starting layer-wise ablation analysis...")
    
    # Check for required model attributes
    if not hasattr(model, 'forward_with_branch_outputs'):
        logging.info("Model does not have forward_with_branch_outputs method. Skipping analysis.")
        return {}
    
    # Create dataloaders with smaller batch size to avoid memory issues
    batch_size = 32
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Get baseline model performance
    baseline_acc = measure_performance(model, test_data, device=device)
    logging.info(f"Baseline accuracy on test set: {baseline_acc:.4f}")
    
    # Save directory for results
    save_dir = os.path.join(save_path, "layerwise_ablation")
    os.makedirs(save_dir, exist_ok=True)
    
    # Get number of layers in the model
    num_layers = len(model.branch_layers)
    logging.info(f"Model has {num_layers} branch layers")
    
    # Initialize results dictionary
    results = {
        'layer_idx': [],
        'baseline_acc': [],
        'inh_ablated_acc': [],
        'exc_ablated_acc': []
    }
    
    # Store original parameters for restoration
    original_params = {}
    for n, p in model.named_parameters():
        original_params[n] = p.detach().clone()
    
    # Process from top layers (higher indices) to bottom layers (lower indices)
    for layer_idx in reversed(range(num_layers)):
        logging.info(f"Analyzing layer {layer_idx}...")
        
        # 1. Measure baseline accuracy for this layer by ablating all other direct inputs
        for i in range(num_layers):
            if i != layer_idx:
                # Temporarily ablate other layers
                if hasattr(model.branch_layers[i], 'branch_excitation'):
                    model.branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)  # Very negative value to ensure zero after exp()
                if hasattr(model.branch_layers[i], 'branch_inhibition'):
                    model.branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)
        
        # Measure accuracy with only this layer active
        baseline_layer_acc = measure_performance(model, test_data, device=device)
        
        # 2. Now ablate inhibitory inputs for this layer
        if hasattr(model.branch_layers[layer_idx], 'branch_inhibition'):
            model.branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
            inh_ablated_acc = measure_performance(model, test_data, device=device)
        else:
            inh_ablated_acc = baseline_layer_acc  # No inhibitory inputs to ablate
        
        # Restore inhibitory inputs
        for n, p in model.named_parameters():
            p.data.copy_(original_params[n])
            
        # Re-ablate other layers
        for i in range(num_layers):
            if i != layer_idx:
                if hasattr(model.branch_layers[i], 'branch_excitation'):
                    model.branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)
                if hasattr(model.branch_layers[i], 'branch_inhibition'):
                    model.branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)
        
        # 3. Ablate excitatory inputs for this layer
        if hasattr(model.branch_layers[layer_idx], 'branch_excitation'):
            model.branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
            exc_ablated_acc = measure_performance(model, test_data, device=device)
        else:
            exc_ablated_acc = baseline_layer_acc  # No excitatory inputs to ablate
        
        # Restore all parameters
        for n, p in model.named_parameters():
            p.data.copy_(original_params[n])
        
        # Store results
        results['layer_idx'].append(layer_idx)
        results['baseline_acc'].append(baseline_layer_acc)
        results['inh_ablated_acc'].append(inh_ablated_acc)
        results['exc_ablated_acc'].append(exc_ablated_acc)
        
        logging.info(f"Layer {layer_idx}: baseline={baseline_layer_acc:.4f}, "
                    f"inh_ablated={inh_ablated_acc:.4f}, exc_ablated={exc_ablated_acc:.4f}")
    
    # Plot the results
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(results['layer_idx'], results['baseline_acc'], 'o-', label='Baseline')
        plt.plot(results['layer_idx'], results['inh_ablated_acc'], 's-', label='Inhibitory Ablated')
        plt.plot(results['layer_idx'], results['exc_ablated_acc'], '^-', label='Excitatory Ablated')
        
        plt.xlabel('Layer Index (higher = closer to input)')
        plt.ylabel('Classification Accuracy')
        plt.title('Layer-wise Ablation Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'layerwise_ablation.png'), dpi=300)
        plt.close()
        
        # Save detailed plot showing accuracy differences
        plt.figure(figsize=(10, 6))
        
        # Calculate differences
        inh_diff = [b - i for b, i in zip(results['baseline_acc'], results['inh_ablated_acc'])]
        exc_diff = [b - e for b, e in zip(results['baseline_acc'], results['exc_ablated_acc'])]
        
        plt.bar(np.array(results['layer_idx']) - 0.2, inh_diff, width=0.4, label='Inhibitory Contribution', color='orange')
        plt.bar(np.array(results['layer_idx']) + 0.2, exc_diff, width=0.4, label='Excitatory Contribution', color='blue')
        
        plt.xlabel('Layer Index (higher = closer to input)')
        plt.ylabel('Accuracy Drop when Ablated')
        plt.title('Contribution of Inhibitory and Excitatory Inputs per Layer')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'layerwise_contributions.png'), dpi=300)
        plt.close()
        
        # Save as CSV
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(save_dir, 'layerwise_results.csv'), index=False)
    except Exception as e:
        logging.error(f"Error creating plots: {e}")
    
    logging.info(f"Ablation analysis results saved to {save_dir}")
    
    return results

def safe_backward_hook(hook_fn):
    """
    Decorator for backward hooks to ensure they return None when no inputs require gradients.
    This prevents the "Backward hook for Modules where no input requires gradient" error.
    
    Args:
        hook_fn: The original hook function to decorate
        
    Returns:
        A decorated hook function that properly handles the case when no inputs require gradients
    """
    @functools.wraps(hook_fn)
    def safe_hook_fn(module, grad_input, grad_output):
        # Check if any grad_input requires gradient
        requires_grad = False
        if grad_input is not None:
            for g in grad_input:
                if g is not None:
                    requires_grad = True
                    break
                    
        # Check parameters too (some hooks need this)
        if not requires_grad:
            for p in module.parameters(recurse=True):
                if p.requires_grad:
                    requires_grad = True
                    break
        
        # If nothing requires gradients, return None
        if not requires_grad:
            return None
            
        # Otherwise, call the original hook function
        try:
            result = hook_fn(module, grad_input, grad_output)
            return result
        except Exception as e:
            # If anything goes wrong, return the original grad_input as a fallback
            print(f"Warning: Hook error: {e}")
            return grad_input
            
    return safe_hook_fn

def safe_register_hooks(module):
    """
    Recursively register safe hooks on a module and all its submodules.
    This safely handles the replacement of existing hooks to ensure PyTorch
    doesn't try to use both regular and full backward hooks on the same module.
    
    Args:
        module: The module to register safe hooks on
    """
    # First, check if module already has backward hooks
    has_backward_hooks = bool(module._backward_hooks)
    has_full_backward_hooks = bool(module._backward_pre_hooks)
    
    # Remove all existing hooks to prevent conflicts
    if has_backward_hooks:
        module._backward_hooks.clear()
    if has_full_backward_hooks:
        module._backward_pre_hooks.clear()
    
    # Register a new safe full backward hook (preferred)
    module.register_full_backward_hook(safe_hook_fn)
    
    # Process child modules
    for child in module.children():
        safe_register_hooks(child)

def register_safe_hook(module, hook_fn):
    """
    Register a safe hook on a module
    
    Args:
        module: The module to register the hook on
        hook_fn: The hook function to register
    """
    # First, check if module already has hooks
    has_backward_hooks = bool(module._backward_hooks)
    has_full_backward_hooks = bool(module._backward_pre_hooks)
    
    # Remove all existing hooks to prevent conflicts
    if has_backward_hooks:
        module._backward_hooks.clear()
    if has_full_backward_hooks:
        module._backward_pre_hooks.clear()
        
    # Register the new safe hook
    safe_hook = safe_backward_hook(hook_fn)
    module.register_full_backward_hook(safe_hook)

# Simple helper function to use for safe hooks
def safe_hook_fn(module, grad_input, grad_output):
    """Simple safe hook function that just returns grad_input unchanged"""
    return grad_input

class SafeHookModule(nn.Module):
    """
    A wrapper module that ensures all backward hooks are safely registered
    
    Args:
        module: The module to wrap
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        # Apply safe hooks to the wrapped module
        safe_register_hooks(module)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def analyze_dendritic_information(model, dataset, device='cuda', mlp_hidden_dims=[100, 50], 
                                 training_epochs=20, batch_size=64, use_excitatory=True, 
                                 use_inhibitory=True):
    """
    Analyze information content in dendritic branches by training an MLP classifier on branch activations.
    
    This function:
    1. Extracts w_i * x_i vectors from each branch at different depths
    2. Trains MLP classifiers to predict input labels from these vectors
    3. Reports classification accuracy as a measure of information content
    
    Parameters
    ----------
    model : nn.Module
        The dendritic network model
    dataset : torch.utils.data.Dataset
        Dataset containing inputs and labels
    device : str
        Device to use for computation
    mlp_hidden_dims : list
        Hidden layer dimensions for the MLP classifier
    training_epochs : int
        Number of epochs to train each MLP
    batch_size : int
        Batch size for training
    use_excitatory : bool
        Whether to use excitatory branch activations
    use_inhibitory : bool
        Whether to use inhibitory branch activations
    
    Returns
    -------
    dict
        Dictionary containing results for each branch depth
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import Adam
    
    # Set model to evaluation mode
    model.eval()
    
    # Check if model has the branch structure we need
    if not hasattr(model, 'net'):
        raise ValueError("Model does not have a 'net' attribute")
    
    # Check if model has forward_with_branch_outputs method
    if not hasattr(model, 'forward_with_branch_outputs') and not hasattr(model.net, 'forward_with_branch_outputs'):
        print("Model does not have forward_with_branch_outputs method. Will use hooks as fallback.")
        
        # Try to enable capturing activations across the model as fallback
        try:
            activation_captures = {}
            
            def make_activation_hook(name, capture_dict):
                def hook_fn(module, input, output):
                    # If output is a tuple, take the first item
                    if isinstance(output, tuple):
                        output = output[0]
                    # Store activation with the module name
                    capture_dict[name] = output.detach().cpu()
                return hook_fn
            
            # Register hooks on all modules that might be branches
            hooks = []
            for name, module in model.named_modules():
                if any(branch_term in name.lower() for branch_term in ['branch', 'dendrit', 'excit', 'inhib']):
                    hook = module.register_forward_hook(make_activation_hook(name, activation_captures))
                    hooks.append(hook)
            
            print(f"Registered {len(hooks)} activation capture hooks on branch-related modules")
        except Exception as e:
            print(f"Warning: Could not set up activation capture hooks: {e}")
            activation_captures = None
    else:
        print("Model has forward_with_branch_outputs method. Will use it to extract branch activations.")
    
    # Create a dataloader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Get number of classes from the dataset
    # Try multiple approaches to handle different dataset formats
    if hasattr(dataset, 'targets'):
        # Standard torchvision datasets have a 'targets' attribute
        targets = dataset.targets
        if hasattr(targets, 'numpy'):
            targets = targets.numpy()
        num_classes = len(set(targets))
    elif isinstance(dataset, torch.utils.data.Dataset):
        # If it's a Dataset but doesn't have targets attribute, try to get all labels
        all_targets = []
        for _, label in data_loader:
            if isinstance(label, torch.Tensor):
                all_targets.append(label)
            elif isinstance(label, tuple) and len(label) > 0:
                # Handle tuple case (when label might be tuple of tensors)
                all_targets.append(label[0])
        if all_targets:
            all_targets = torch.cat(all_targets)
            num_classes = len(torch.unique(all_targets))
        else:
            # Default if we can't determine
            num_classes = 10
    else:
        # Default to a reasonable number if can't determine
        num_classes = 10
        print("Warning: Could not determine number of classes from dataset. Using default value of 10.")
    
    # Store results
    results = {
        "branch_depths": [],
        "exc_accuracy": [],
        "inh_accuracy": [],
        "combined_accuracy": []
    }
    
    # Collect all branch activations using forward_with_branch_outputs
    all_branch_activations = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            
            # Use forward_with_branch_outputs if available
            if hasattr(model, 'forward_with_branch_outputs'):
                _, branch_acts = model.forward_with_branch_outputs(inputs)
                all_branch_activations.append(branch_acts)
            elif hasattr(model.net, 'forward_with_branch_outputs'):
                _, branch_acts = model.net.forward_with_branch_outputs(inputs)
                all_branch_activations.append(branch_acts)
            else:
                # Fallback - just do a regular forward pass to populate hooks
                _ = model(inputs)
            
            all_labels.append(labels)
    
    # Concatenate labels
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # If we collected branch activations via forward_with_branch_outputs
    if all_branch_activations:
        print(f"Collected branch activations from {len(all_branch_activations)} batches")
        
        # For each depth, concatenate across batches
        # branch_acts from forward_with_branch_outputs is a list [batch_size, branch_features] for each depth
        depth_acts = [[] for _ in range(len(all_branch_activations[0]))]
        
        for batch_acts in all_branch_activations:
            for depth_idx, depth_act in enumerate(batch_acts):
                # Move to CPU and store
                depth_acts[depth_idx].append(depth_act.cpu())
        
        # Now process each depth
        for depth_idx, acts_list in enumerate(depth_acts):
            # Skip if no activations
            if not acts_list:
                continue
                
            # Concatenate across batches
            depth_act_tensor = torch.cat(acts_list, dim=0)
            depth_name = f"Depth {depth_idx+1}"
            results["branch_depths"].append(depth_name)
            
            print(f"Processing branch depth {depth_idx+1} with activation shape: {depth_act_tensor.shape}")
            
            # If tensor has more than 2 dimensions, flatten except for batch dim
            if depth_act_tensor.dim() > 2:
                orig_shape = depth_act_tensor.shape
                depth_act_tensor = depth_act_tensor.reshape(orig_shape[0], -1)
                print(f"  Reshaped from {orig_shape} to {depth_act_tensor.shape}")
            
            # Check for NaN or Inf
            if torch.isnan(depth_act_tensor).any() or torch.isinf(depth_act_tensor).any():
                print(f"  Warning: Found NaN or Inf in activations!")
                depth_act_tensor = torch.nan_to_num(depth_act_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Split tensor into excitatory/inhibitory portions if relevant
            # For simplicity, assume first half is excitatory, second half is inhibitory
            # This is a simplified approach - adjust based on your actual model structure
            feature_dim = depth_act_tensor.shape[1]
            split_point = feature_dim // 2
            
            # Excitatory features
            if use_excitatory and split_point > 0:
                exc_data = depth_act_tensor[:, :split_point].numpy()
                exc_accuracy = train_and_evaluate_classifier(
                    exc_data, all_labels, 
                    hidden_dims=mlp_hidden_dims,
                    num_classes=num_classes,
                    epochs=training_epochs,
                    device=device
                )
                print(f"  Excitatory information accuracy: {exc_accuracy:.4f}")
                results["exc_accuracy"].append(exc_accuracy)
            else:
                results["exc_accuracy"].append(None)
            
            # Inhibitory features
            if use_inhibitory and split_point < feature_dim:
                inh_data = depth_act_tensor[:, split_point:].numpy()
                inh_accuracy = train_and_evaluate_classifier(
                    inh_data, all_labels, 
                    hidden_dims=mlp_hidden_dims,
                    num_classes=num_classes,
                    epochs=training_epochs,
                    device=device
                )
                print(f"  Inhibitory information accuracy: {inh_accuracy:.4f}")
                results["inh_accuracy"].append(inh_accuracy)
            else:
                results["inh_accuracy"].append(None)
            
            # Full activation tensor for combined
            combined_data = depth_act_tensor.numpy()
            combined_accuracy = train_and_evaluate_classifier(
                combined_data, all_labels, 
                hidden_dims=mlp_hidden_dims,
                num_classes=num_classes,
                epochs=training_epochs,
                device=device
            )
            print(f"  Combined information accuracy: {combined_accuracy:.4f}")
            results["combined_accuracy"].append(combined_accuracy)
    
    else:
        # Legacy approach if forward_with_branch_outputs isn't available or didn't work
        print("Warning: Could not collect branch activations using forward_with_branch_outputs.")
        print("Checking if model has branch_layers structure...")
        
        # Get branch layers
        branch_layers = []
        if hasattr(model.net, 'branch_layers'):
            branch_layers = model.net.branch_layers
        elif hasattr(model.net, 'layers'):
            # EINet structure has layers with excitatory_cells and inhibitory_cells
            for layer in model.net.layers:
                if hasattr(layer, 'excitatory_cells') and hasattr(layer.excitatory_cells, 'branch_layers'):
                    branch_layers.extend(layer.excitatory_cells.branch_layers)
        
        if not branch_layers:
            print("WARNING: Could not find branch layers in the model. No information to analyze.")
            # Return empty results with at least one entry to avoid plotting errors
            results["branch_depths"].append("No branches found")
            results["exc_accuracy"].append(None)
            results["inh_accuracy"].append(None)
            results["combined_accuracy"].append(None)
            return results
        
        print(f"Found {len(branch_layers)} branch layers for information analysis")
        
        # Continue with the rest of the original implementation...
        # (keeping the original code from here for legacy support)
        
        # For each branch depth
        for depth_idx, layer in enumerate(branch_layers):
            print(f"Analyzing information content at branch depth {depth_idx+1}...")
            depth_name = f"Depth {depth_idx+1}"
            results["branch_depths"].append(depth_name)
            
            # Check for branch excitation and inhibition
            has_branch_excitation = hasattr(layer, 'branch_excitation')
            has_branch_inhibition = hasattr(layer, 'branch_inhibition')
            
            # Collect activations (w_i * x_i) for each branch
            exc_activations = []
            inh_activations = []
            all_labels = []
            
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    
                    # Forward pass to get branch activations
                    _ = model(inputs)  # This populates activations in branches
                    
                    # Extract weighted activations (w_i * x_i)
                    if has_branch_excitation and use_excitatory:
                        # Extract branch excitation activations
                        # For each sample, get the raw activations before any nonlinearity
                        exc_act = None  # Initialize to None
                        success = False
                        
                        # Try different attribute approaches to access branch activations
                        try:
                            if hasattr(layer, 'last_exc_input'):
                                # If model stores activations directly
                                exc_act = layer.last_exc_input.detach().cpu()
                                success = True
                                print(f"  Found excitatory activations through last_exc_input: shape {exc_act.shape}")
                            elif hasattr(layer.branch_excitation, 'last_input') and hasattr(layer.branch_excitation, 'pruned_weight'):
                                # Calculate w_i * x_i from raw inputs and weights
                                x_i = layer.branch_excitation.last_input.detach().cpu()
                                w_i = layer.branch_excitation.pruned_weight().detach().cpu()
                                print(f"  Found excitatory inputs x_i: {x_i.shape}, weights w_i: {w_i.shape}")
                                
                                # For each neuron, compute the weighted activation
                                exc_act = torch.matmul(x_i, w_i.t())  # Shape: [batch_size, num_neurons]
                                success = True
                                print(f"  Computed excitatory activations via matmul: shape {exc_act.shape}")
                            elif hasattr(layer, 'forward_excitatory'):
                                # Try direct forward pass of the excitatory component
                                dummy_input = torch.zeros(inputs.shape).to(device)
                                with torch.no_grad():
                                    exc_act = layer.forward_excitatory(dummy_input).detach().cpu()
                                success = True
                                print(f"  Computed excitatory activations via forward_excitatory: shape {exc_act.shape}")
                            else:
                                # Try to access any direct attributes that might have activations
                                print(f"  Warning: Could not find standard excitatory activation attributes for layer {depth_idx}")
                                for attr_name in dir(layer):
                                    if 'excitatory' in attr_name.lower() and not attr_name.startswith('_'):
                                        attr = getattr(layer, attr_name)
                                        if isinstance(attr, torch.Tensor) and attr.dim() > 1:
                                            exc_act = attr.detach().cpu()
                                            success = True
                                            print(f"  Found alternative excitatory activation via {attr_name}: shape {exc_act.shape}")
                                            break
                        except Exception as e:
                            print(f"  Error extracting excitatory activations: {e}")
                        
                        if exc_act is not None and success:
                            # Check for NaN or infinite values
                            if torch.isnan(exc_act).any() or torch.isinf(exc_act).any():
                                print(f"  Warning: Found NaN or Inf in excitatory activations!")
                                # Try to clean bad values
                                exc_act = torch.nan_to_num(exc_act, nan=0.0, posinf=0.0, neginf=0.0)
                            exc_activations.append(exc_act)
                        else:
                            print(f"  Warning: Failed to get excitatory activations for layer {depth_idx}")
                    
                    if has_branch_inhibition and use_inhibitory:
                        # Extract branch inhibition activations
                        inh_act = None  # Initialize to None
                        success = False
                        
                        # Try different attribute approaches to access branch activations
                        try:
                            if hasattr(layer, 'last_inh_input'):
                                # If model stores activations directly
                                inh_act = layer.last_inh_input.detach().cpu()
                                success = True
                                print(f"  Found inhibitory activations through last_inh_input: shape {inh_act.shape}")
                            elif hasattr(layer.branch_inhibition, 'last_input') and hasattr(layer.branch_inhibition, 'pruned_weight'):
                                # Calculate w_i * x_i from raw inputs and weights
                                x_i = layer.branch_inhibition.last_input.detach().cpu()
                                w_i = layer.branch_inhibition.pruned_weight().detach().cpu()
                                print(f"  Found inhibitory inputs x_i: {x_i.shape}, weights w_i: {w_i.shape}")
                                
                                # For each neuron, compute the weighted activation
                                inh_act = torch.matmul(x_i, w_i.t())  # Shape: [batch_size, num_neurons]
                                success = True
                                print(f"  Computed inhibitory activations via matmul: shape {inh_act.shape}")
                            elif hasattr(layer, 'forward_inhibitory'):
                                # Try direct forward pass of the inhibitory component
                                dummy_input = torch.zeros(inputs.shape).to(device)
                                with torch.no_grad():
                                    inh_act = layer.forward_inhibitory(dummy_input).detach().cpu()
                                success = True
                                print(f"  Computed inhibitory activations via forward_inhibitory: shape {inh_act.shape}")
                            else:
                                # Try to access any direct attributes that might have activations
                                print(f"  Warning: Could not find standard inhibitory activation attributes for layer {depth_idx}")
                                for attr_name in dir(layer):
                                    if 'inhibitory' in attr_name.lower() and not attr_name.startswith('_'):
                                        attr = getattr(layer, attr_name)
                                        if isinstance(attr, torch.Tensor) and attr.dim() > 1:
                                            inh_act = attr.detach().cpu()
                                            success = True
                                            print(f"  Found alternative inhibitory activation via {attr_name}: shape {inh_act.shape}")
                                            break
                        except Exception as e:
                            print(f"  Error extracting inhibitory activations: {e}")
                        
                        if inh_act is not None and success:
                            # Check for NaN or infinite values
                            if torch.isnan(inh_act).any() or torch.isinf(inh_act).any():
                                print(f"  Warning: Found NaN or Inf in inhibitory activations!")
                                # Try to clean bad values
                                inh_act = torch.nan_to_num(inh_act, nan=0.0, posinf=0.0, neginf=0.0)
                            inh_activations.append(inh_act)
                        else:
                            print(f"  Warning: Failed to get inhibitory activations for layer {depth_idx}")
                    
                    all_labels.append(labels)
            
            # Concatenate activations and labels
            all_labels = torch.cat(all_labels, dim=0).numpy()
            
            # Train MLP classifier on excitatory activations
            if exc_activations and use_excitatory:
                exc_data = torch.cat(exc_activations, dim=0).numpy()
                exc_accuracy = train_and_evaluate_classifier(
                    exc_data, all_labels, 
                    hidden_dims=mlp_hidden_dims,
                    num_classes=num_classes,
                    epochs=training_epochs,
                    device=device
                )
                print(f"  Excitatory information accuracy: {exc_accuracy:.4f}")
                results["exc_accuracy"].append(exc_accuracy)
            else:
                results["exc_accuracy"].append(None)
            
            # Train MLP classifier on inhibitory activations
            if inh_activations and use_inhibitory:
                inh_data = torch.cat(inh_activations, dim=0).numpy()
                inh_accuracy = train_and_evaluate_classifier(
                    inh_data, all_labels, 
                    hidden_dims=mlp_hidden_dims,
                    num_classes=num_classes,
                    epochs=training_epochs,
                    device=device
                )
                print(f"  Inhibitory information accuracy: {inh_accuracy:.4f}")
                results["inh_accuracy"].append(inh_accuracy)
            else:
                results["inh_accuracy"].append(None)
            
            # Train MLP classifier on combined activations
            if exc_activations and inh_activations and use_excitatory and use_inhibitory:
                exc_data = torch.cat(exc_activations, dim=0).numpy()
                inh_data = torch.cat(inh_activations, dim=0).numpy()
                
                # Concatenate excitatory and inhibitory activations
                combined_data = np.concatenate([exc_data, inh_data], axis=1)
                
                combined_accuracy = train_and_evaluate_classifier(
                    combined_data, all_labels, 
                    hidden_dims=mlp_hidden_dims,
                    num_classes=num_classes,
                    epochs=training_epochs,
                    device=device
                )
                print(f"  Combined information accuracy: {combined_accuracy:.4f}")
                results["combined_accuracy"].append(combined_accuracy)
            else:
                results["combined_accuracy"].append(None)
    
    # Make sure we have at least one entry to avoid plotting errors
    if not results["branch_depths"]:
        print("WARNING: No branch depths analyzed. Adding a placeholder entry.")
        results["branch_depths"].append("No valid branches")
        results["exc_accuracy"].append(None)
        results["inh_accuracy"].append(None)
        results["combined_accuracy"].append(None)
    
    return results


def train_and_evaluate_classifier(X, y, hidden_dims=[100, 50], num_classes=10, 
                                 epochs=20, batch_size=64, device='cuda'):
    """
    Train and evaluate an MLP classifier on the given data.
    
    Parameters
    ----------
    X : numpy.ndarray
        Input features
    y : numpy.ndarray
        Target labels
    hidden_dims : list
        Hidden layer dimensions
    num_classes : int
        Number of classes
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    device : str
        Device to use for training
    
    Returns
    -------
    float
        Classification accuracy
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import Adam
    import torch.nn.functional as F
    
    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create MLP classifier
    input_dim = X.shape[1]
    model = MLPClassifier(input_dim, hidden_dims, num_classes)
    model = model.to(device)
    
    # Set up optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model state
    model.load_state_dict(best_model_state)
    
    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    final_accuracy = correct / total
    return final_accuracy