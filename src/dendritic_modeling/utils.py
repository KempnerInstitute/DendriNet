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
import random
from omegaconf import DictConfig, ListConfig
from sklearn.metrics import accuracy_score
import logging
import traceback
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


def analyze_branch_ablation(model, dataset, layer_idx=None, branch_ids=None,
                            mode='direct', device='cpu'):
    """
    1) measure baseline accuracy
    2) for each branch_id in branch_ids, ablate, measure accuracy drop
    3) restore weights

    If we pass branch_ids=None, it ablates the entire layer. 
    
    For dendritic order analysis, if layer_idx and branch_ids are None, 
    analyze ablation by dendritic order (primary, secondary, etc.).
    """
    net = model.net if hasattr(model, 'net') else model
    net = net.to(device)  # Ensure model is on the correct device
    
    baseline_acc = measure_performance(net, dataset, device=device)
    results = {}
    original = {}
    for n, p in net.named_parameters():
        original[n] = p.detach().clone()

    # Handle dendritic order analysis (when both layer_idx and branch_ids are None)
    if layer_idx is None and branch_ids is None:
        # Try to determine dendritic orders from the model
        n_orders = 0
        if hasattr(net, 'excitatory_branch_factors'):
            n_orders = len(net.excitatory_branch_factors)
        elif hasattr(net, 'branch_layers'):
            n_orders = len(net.branch_layers)
        else:
            # Try to infer from layers if it's an EINet
            if hasattr(net, 'layers'):
                n_orders = len(net.layers)
                
        if n_orders == 0:
            raise ValueError("Could not determine number of dendritic orders")
            
        # Initialize order-based results
        results = {
            "order": list(range(1, n_orders + 1)),
            "exc_drop": [],
            "inh_drop": []
        }
        
        # For each order, ablate and measure performance
        for order in range(1, n_orders + 1):
            # Ablate excitatory branches at this order
            if hasattr(net, 'branch_layers') and order-1 < len(net.branch_layers):
                layer = net.branch_layers[order-1]
                if hasattr(layer, 'branch_excitation') and hasattr(layer.branch_excitation, 'pre_w'):
                    # Save original weights
                    original_weights = layer.branch_excitation.pre_w.data.clone()
                    
                    # Ablate
                    layer.branch_excitation.pre_w.data.zero_()
                    exc_acc = measure_performance(net, dataset, device=device)
                    exc_drop = baseline_acc - exc_acc
                    results["exc_drop"].append(exc_drop)
                    
                    # Restore
                    layer.branch_excitation.pre_w.data.copy_(original_weights)
                    
                    # Ablate inhibitory branches at this order
                    if hasattr(layer, 'branch_inhibition') and hasattr(layer.branch_inhibition, 'pre_w'):
                        # Save original weights
                        original_weights = layer.branch_inhibition.pre_w.data.clone()
                        
                        # Ablate
                        layer.branch_inhibition.pre_w.data.zero_()
                        inh_acc = measure_performance(net, dataset, device=device)
                        inh_drop = baseline_acc - inh_acc
                        results["inh_drop"].append(inh_drop)
                        
                        # Restore
                        layer.branch_inhibition.pre_w.data.copy_(original_weights)
        
        return results

    # If branch_ids is None, we do a single ablation of the whole layer
    # If it's a list, we do one ablation per branch in that list
    if branch_ids is None:
        ablate_branch(net, layer_idx, branch_ids=None, mode=mode)
        acc_after = measure_performance(net, dataset, device=device)
        delta = baseline_acc - acc_after
        results["all_branches"] = (acc_after, delta)

        for n, p in net.named_parameters():
            p.data.copy_(original[n])
    else:
        # ablate each branch ID separately
        for b_id in branch_ids:
            ablate_branch(net, layer_idx, [b_id], mode=mode)
            acc_after = measure_performance(net, dataset, device=device)
            delta = baseline_acc - acc_after
            results[b_id] = (acc_after, delta)

            # restore
            for n, p in net.named_parameters():
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

def analyze_dendritic_information(model, dataset, device='cuda', 
                                 mlp_hidden_dims=[100, 50], training_epochs=20,
                                 batch_size=64, use_excitatory=True, use_inhibitory=True,
                                 use_post_shunting=False):
    """
    Analyze information content in dendritic branches by training classifiers on branch activations.
    
    Parameters:
    -----------
    model : nn.Module
        The dendritic network model with forward_with_branch_outputs method
    dataset : Dataset
        Dataset to analyze
    device : str
        Device to use ('cuda' or 'cpu')
    mlp_hidden_dims : list
        Hidden layer dimensions for MLP classifier
    training_epochs : int
        Number of epochs to train classifiers
    batch_size : int
        Batch size for dataloaders
    use_excitatory : bool
        Whether to analyze excitatory branches
    use_inhibitory : bool
        Whether to analyze inhibitory branches
    use_post_shunting : bool
        If True, analyze post-shunting branch outputs instead of pre-shunting inputs
        
    Returns:
    --------
    dict
        Dictionary with branch depths and corresponding accuracies
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import logging
    import traceback
    
    # Ensure model is on the right device
    model = model.to(device)
    
    # Check if model has the right method
    if hasattr(model, 'forward_with_branch_outputs'):
        forward_func = model.forward_with_branch_outputs
    elif hasattr(model, 'net') and hasattr(model.net, 'forward_with_branch_outputs'):
        forward_func = model.net.forward_with_branch_outputs
        model.net = model.net.to(device)  # Ensure model.net is on the right device
    else:
        logging.error("Model does not have forward_with_branch_outputs method")
        return {
            "branch_depths": [],
            "exc_accuracy": [],
            "inh_accuracy": [],
            "combined_accuracy": []
        }
    
    # Get number of branch depths for debugging
    n_branch_depths = 0
    if hasattr(model, 'net'):
        if hasattr(model.net, 'excitatory_branch_factors'):
            n_branch_depths = len(model.net.excitatory_branch_factors)
            logging.info(f"Model has {n_branch_depths} branch depths from excitatory_branch_factors: {model.net.excitatory_branch_factors}")
        elif hasattr(model.net, 'layers') and len(model.net.layers) > 0:
            if hasattr(model.net.layers[0], 'excitatory_cells') and hasattr(model.net.layers[0].excitatory_cells, 'branch_layers'):
                n_branch_depths = len(model.net.layers[0].excitatory_cells.branch_layers)
                logging.info(f"Model has {n_branch_depths} branch depths from layers[0].excitatory_cells.branch_layers")
    
    # Run a small test batch to understand activation structure before full processing
    logging.info("Running test batch to understand activation structure...")
    test_batch_size = min(8, len(dataset))
    test_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    test_inputs, test_labels = next(iter(test_loader))
    test_inputs = test_inputs.to(device)
    
    test_activation_format = None
    with torch.no_grad():
        try:
            # Determine which forward function to use
            if hasattr(model, 'forward_with_branch_outputs'):
                forward_func = model.forward_with_branch_outputs
            elif hasattr(model, 'net') and hasattr(model.net, 'forward_with_branch_outputs'):
                forward_func = model.net.forward_with_branch_outputs
            else:
                logging.error("Model does not have forward_with_branch_outputs method")
                return {
                    "branch_depths": [],
                    "exc_accuracy": [],
                    "inh_accuracy": [],
                    "combined_accuracy": []
                }
                
            # Execute the forward pass on test batch
            _, test_acts = forward_func(test_inputs)
            
            logging.info(f"Test batch activation type: {type(test_acts)}")
            
            # Analyze list format
            if isinstance(test_acts, list):
                logging.info(f"List length: {len(test_acts)}")
                if len(test_acts) > 0:
                    logging.info(f"First element type: {type(test_acts[0])}")
                    
                    if isinstance(test_acts[0], list):
                        logging.info(f"Nested list detected with {len(test_acts)} branch levels")
                        logging.info(f"First list has {len(test_acts[0])} components")
                        
                        if len(test_acts[0]) > 0:
                            for i, component in enumerate(test_acts[0]):
                                if isinstance(component, torch.Tensor):
                                    logging.info(f"Component {i} is tensor with shape {component.shape}")
                                else:
                                    logging.info(f"Component {i} is type {type(component)}")
                    
                    elif isinstance(test_acts[0], torch.Tensor):
                        logging.info(f"List of tensors detected")
                        for i, tensor in enumerate(test_acts):
                            logging.info(f"Tensor {i} shape: {tensor.shape}")
            
            # Analyze dict format
            elif isinstance(test_acts, dict):
                logging.info(f"Dictionary format detected with keys: {list(test_acts.keys())}")
                for k, v in test_acts.items():
                    if isinstance(v, torch.Tensor):
                        logging.info(f"Key {k}: tensor shape {v.shape}")
                    else:
                        logging.info(f"Key {k}: type {type(v)}")
            
            logging.info("End of test activation analysis")
            
        except Exception as e:
            logging.error(f"Error in test activation analysis: {str(e)}")
            logging.error(traceback.format_exc())
    
    # Create dataloader with smaller batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract all inputs and labels
    all_inputs = []
    all_labels = []
    
    logging.info("Collecting inputs and labels...")
    for inputs, labels in dataloader:
        # Move inputs to the right device
        inputs = inputs.to(device)
        
        # Store labels on CPU
        all_inputs.append(inputs.cpu().detach())
        all_labels.append(labels)
    
    # Concatenate all inputs and labels
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Get branch outputs for all inputs
    branch_activations = []
    activation_format = None  # Will store the format type
    
    logging.info(f"Processing {len(all_inputs)} inputs with batch size {batch_size}")
    
    # Debug: Test the output format with a single batch first
    try:
        sample_inputs = all_inputs[:min(batch_size, len(all_inputs))].to(device)
        with torch.no_grad():
            _, sample_acts = forward_func(sample_inputs)
            logging.info(f"Sample activation type: {type(sample_acts)}")
            if isinstance(sample_acts, list):
                logging.info(f"Activation list length: {len(sample_acts)}")
                if len(sample_acts) > 0:
                    logging.info(f"First element type: {type(sample_acts[0])}")
                    if isinstance(sample_acts[0], torch.Tensor):
                        logging.info(f"First tensor shape: {sample_acts[0].shape}")
                    elif isinstance(sample_acts[0], (list, tuple)):
                        logging.info(f"First sublist length: {len(sample_acts[0])}")
                        if len(sample_acts[0]) > 0:
                            logging.info(f"First sublist element type: {type(sample_acts[0][0])}")
            elif isinstance(sample_acts, dict):
                logging.info(f"Dict keys: {list(sample_acts.keys())}")
    except Exception as e:
        logging.error(f"Error in sample activation check: {str(e)}")
    
    with torch.no_grad():
        # Process in batches to avoid OOM
        for batch_idx in range(0, len(all_inputs), batch_size):
            batch_inputs = all_inputs[batch_idx:batch_idx+batch_size].to(device)
            
            # Forward pass with branch outputs
            try:
                _, batch_acts = forward_func(batch_inputs)
                
                # Determine activation format based on the first batch
                if activation_format is None:
                    if isinstance(batch_acts, list):
                        if len(batch_acts) > 0:
                            if isinstance(batch_acts[0], (list, tuple)):
                                # Format: [layer_idx][exc/inh_idx]
                                activation_format = "nested_list"
                                logging.info("Detected nested list activation format")
                                logging.info(f"Found {len(batch_acts)} branch levels")
                                if len(batch_acts[0]) > 0:
                                    logging.info(f"Each level has {len(batch_acts[0])} components")
                            elif isinstance(batch_acts[0], torch.Tensor):
                                # Format: [layer_idx] -> tensor
                                activation_format = "tensor_list"
                                logging.info("Detected tensor list activation format")
                                logging.info(f"Tensor shapes: {[t.shape for t in batch_acts]}")
                    elif isinstance(batch_acts, dict):
                        # Format: batch_acts[layer_idx] -> tensor
                        activation_format = "dict"
                        logging.info("Detected dictionary activation format")
                
                # Process based on determined format
                if activation_format == "nested_list":
                    # Convert branch activations to lists of tensors and move to CPU
                    batch_cpu_acts = []
                    for acts in batch_acts:
                        batch_cpu_acts.append([act.cpu().detach() for act in acts])
                    branch_activations.append(batch_cpu_acts)
                elif activation_format == "tensor_list":
                    # Store each tensor as is
                    batch_cpu_acts = [act.cpu().detach() for act in batch_acts]
                    branch_activations.append(batch_cpu_acts)
                elif activation_format == "dict":
                    # Store the dictionary with CPU tensors
                    batch_cpu_acts = {k: v.cpu().detach() for k, v in batch_acts.items()}
                    branch_activations.append(batch_cpu_acts)
                else:
                    # Unknown format, but still try to process it
                    logging.warning(f"Unknown activation format: {type(batch_acts)}")
                    # If it's a list of tensors but didn't match our criteria
                    if isinstance(batch_acts, list):
                        # Try to convert everything to CPU tensors
                        batch_cpu_acts = []
                        for act in batch_acts:
                            if isinstance(act, torch.Tensor):
                                batch_cpu_acts.append(act.cpu().detach())
                            elif isinstance(act, (list, tuple)):
                                batch_cpu_acts.append([a.cpu().detach() if isinstance(a, torch.Tensor) else a for a in act])
                            else:
                                batch_cpu_acts.append(act)
                        branch_activations.append(batch_cpu_acts)
                    else:
                        # Just store as is and hope for the best
                        branch_activations.append(batch_acts)
            except Exception as e:
                logging.error(f"Error in forward_with_branch_outputs: {str(e)}")
                logging.error(traceback.format_exc())
                # Return empty results if error
                return {
                    "branch_depths": [],
                    "exc_accuracy": [],
                    "inh_accuracy": [],
                    "combined_accuracy": []
                }
    
    # Log branch activation structure
    logging.info(f"Collected {len(branch_activations)} batches of activations")
    if len(branch_activations) > 0:
        if activation_format == "nested_list":
            logging.info(f"Number of layers: {len(branch_activations[0])}")
            if len(branch_activations[0]) > 0:
                logging.info(f"Layer 0 components: {len(branch_activations[0][0])}")
        elif activation_format == "tensor_list":
            logging.info(f"Number of tensors: {len(branch_activations[0])}")
        elif activation_format == "dict":
            logging.info(f"Dictionary keys: {list(branch_activations[0].keys())}")
    
    # Initialize results
    results = {
        "branch_depths": [],
        "exc_accuracy": [],
        "inh_accuracy": [],
        "combined_accuracy": []
    }
    
    # Simple MLP classifier
    class BranchClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dims, num_classes):
            super().__init__()
            self.layers = nn.ModuleList()
            
            # Input layer
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
            
            # Hidden layers
            for i in range(len(hidden_dims)-1):
                self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                
            # Output layer
            self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
            
        def forward(self, x):
            for i, layer in enumerate(self.layers[:-1]):
                x = F.relu(layer(x))
            x = self.layers[-1](x)
            return x
    
    # Train classifier function
    def train_classifier(train_data, valid_data, input_dim, num_classes):
        # Create classifier
        classifier = BranchClassifier(input_dim, mlp_hidden_dims, num_classes)
        classifier = classifier.to(device)
        
        # Create dataloaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=batch_size)
        
        # Define optimizer and loss
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
            # Train
            for epoch in range(training_epochs):
                classifier.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = classifier(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            classifier.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = classifier(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
    
    # Extract branch activations for each depth and train classifiers
    num_classes = len(torch.unique(all_labels))
    
    # Separate train/valid sets
    train_size = int(0.8 * len(all_labels))
    valid_size = len(all_labels) - train_size
    
    train_indices = torch.randperm(len(all_labels))[:train_size]
    valid_indices = torch.randperm(len(all_labels))[train_size:train_size+valid_size]
    
    train_labels = all_labels[train_indices]
    valid_labels = all_labels[valid_indices]
    
    # For post-shunting analysis, we need to access the model attributes to determine if it uses shunting
    use_shunting = True  # Default, assuming shunting is being used
    try:
        if hasattr(model, 'net') and hasattr(model.net, 'use_shunting'):
            use_shunting = model.net.use_shunting
        elif hasattr(model, 'use_shunting'):
            use_shunting = model.use_shunting
    except Exception as e:
        logging.warning(f"Couldn't determine use_shunting status: {e}")
    
    # Handle different activation formats
    if len(branch_activations) == 0:
        logging.error("No branch activations collected!")
        return results
    
    # For tensor_list format
    if activation_format == "tensor_list":
        logging.info("Processing tensor_list format")
        
        # First, determine how many branch layers we have
        n_levels = len(branch_activations[0])
        logging.info(f"Found {n_levels} branch levels in tensor_list format")
        
        # Create ordered branch depth names for all levels
        for i in range(n_levels):
            results["branch_depths"].append(f"Order {i+1}")
            logging.info(f"Added Order {i+1} to branch_depths")
        
        # Analyze each depth level
        for depth_idx in range(n_levels):
            try:
                logging.info(f"Processing depth {depth_idx}")
                # Collect activations for this depth
                depth_acts = []
                for sample_idx in range(len(all_inputs)):
                    batch_idx = sample_idx // batch_size
                    if batch_idx < len(branch_activations) and depth_idx < len(branch_activations[batch_idx]):
                        # Check if branch_activations contains list or tensor
                        if isinstance(branch_activations[batch_idx][depth_idx], list) and len(branch_activations[batch_idx][depth_idx]) >= 3:
                            # It's a list with separate components [exc, inh, combined]
                            depth_acts.append(branch_activations[batch_idx][depth_idx])
                        else:
                            # It's a tensor, we need to create synthetic differences
                            depth_acts.append(branch_activations[batch_idx][depth_idx])
                
                if depth_acts:
                    if isinstance(depth_acts[0], list) and len(depth_acts[0]) >= 3:
                        # We have component list format - extract components separately
                        exc_components = []
                        inh_components = []
                        combined_components = []
                        
                        for act in depth_acts:
                            if act[0] is not None and isinstance(act[0], torch.Tensor):
                                exc_components.append(act[0])
                            if act[1] is not None and isinstance(act[1], torch.Tensor):
                                inh_components.append(act[1])
                            if act[2] is not None and isinstance(act[2], torch.Tensor):
                                combined_components.append(act[2])
                        
                        # Train classifier on each component separately
                        exc_acc = None
                        inh_acc = None
                        combined_acc = None
                        
                        # Process excitatory components
                        if exc_components and len(exc_components) > 0:
                            try:
                                exc_tensor = torch.cat(exc_components, dim=0)
                                train_data = TensorDataset(exc_tensor[train_indices], train_labels)
                                valid_data = TensorDataset(exc_tensor[valid_indices], valid_labels)
                                exc_acc = train_classifier(train_data, valid_data, exc_tensor.shape[1], num_classes)
                                logging.info(f"Order {depth_idx+1} excitatory accuracy: {exc_acc:.4f}")
                            except Exception as e:
                                logging.error(f"Error processing excitatory component at depth {depth_idx}: {e}")
                                exc_acc = None
                        
                        # Process inhibitory components
                        if inh_components and len(inh_components) > 0:
                            try:
                                inh_tensor = torch.cat(inh_components, dim=0)
                                train_data = TensorDataset(inh_tensor[train_indices], train_labels)
                                valid_data = TensorDataset(inh_tensor[valid_indices], valid_labels)
                                inh_acc = train_classifier(train_data, valid_data, inh_tensor.shape[1], num_classes)
                                logging.info(f"Order {depth_idx+1} inhibitory accuracy: {inh_acc:.4f}")
                            except Exception as e:
                                logging.error(f"Error processing inhibitory component at depth {depth_idx}: {e}")
                                inh_acc = None
                        
                        # Process combined components
                        if combined_components and len(combined_components) > 0:
                            try:
                                combined_tensor = torch.cat(combined_components, dim=0)
                                train_data = TensorDataset(combined_tensor[train_indices], train_labels)
                                valid_data = TensorDataset(combined_tensor[valid_indices], valid_labels)
                                combined_acc = train_classifier(train_data, valid_data, combined_tensor.shape[1], num_classes)
                                logging.info(f"Order {depth_idx+1} combined accuracy: {combined_acc:.4f}")
                            except Exception as e:
                                logging.error(f"Error processing combined component at depth {depth_idx}: {e}")
                                combined_acc = None
                        
                        # Store the component values directly
                        results["exc_accuracy"].append(exc_acc)
                        results["inh_accuracy"].append(inh_acc)
                        results["combined_accuracy"].append(combined_acc)
                        else:
                        # Process single tensor activations
                        # Concatenate all activations
                        if all(isinstance(act, torch.Tensor) for act in depth_acts):
                            depth_acts = torch.cat(depth_acts, dim=0)
                            
                            # Create datasets
                            train_data = TensorDataset(depth_acts[train_indices], train_labels)
                            valid_data = TensorDataset(depth_acts[valid_indices], valid_labels)
                            
                            # Train classifier
                            accuracy = train_classifier(train_data, valid_data, depth_acts.shape[1], num_classes)
                            logging.info(f"Order {depth_idx+1} accuracy: {accuracy:.4f}")
                            
                                results["exc_accuracy"].append(accuracy)
                            results["inh_accuracy"].append(None if use_post_shunting else accuracy)
                                results["combined_accuracy"].append(accuracy)
                            else:
                            # Handle mixed types
                            logging.warning(f"Mixed activation types at depth {depth_idx}")
                            results["exc_accuracy"].append(None)
                            results["inh_accuracy"].append(None)
                            results["combined_accuracy"].append(None)
                    else:
                        # No activations for this depth
                        results["exc_accuracy"].append(None)
                        results["inh_accuracy"].append(None)
                        results["combined_accuracy"].append(None)
            except Exception as e:
                logging.error(f"Error analyzing depth {depth_idx}: {str(e)}")
                logging.error(traceback.format_exc())
                results["exc_accuracy"].append(None)
                results["inh_accuracy"].append(None)
                results["combined_accuracy"].append(None)
        
        # Log summary before returning
        logging.info(f"Final results summary for tensor_list format:")
        logging.info(f"Found {len(results['branch_depths'])} branch depths: {results['branch_depths']}")
        logging.info(f"Excitatory accuracies: {results['exc_accuracy']}")
        logging.info(f"Inhibitory accuracies: {results['inh_accuracy']}")
        logging.info(f"Combined accuracies: {results['combined_accuracy']}")
        
        return results
    
    # For dictionary format
    elif activation_format == "dict":
        logging.info("Processing dictionary format")
        layer_keys = list(branch_activations[0].keys())
        logging.info(f"Dict has {len(layer_keys)} keys: {layer_keys}")
        
        # Sort keys if they are numeric to ensure correct order
        if all(isinstance(k, (int, float)) for k in layer_keys):
            layer_keys = sorted(layer_keys)
            logging.info(f"Sorted keys: {layer_keys}")
        
        for key in layer_keys:
            depth_name = f"Order {key if isinstance(key, (int, str)) else 'Unknown'}"
            results["branch_depths"].append(depth_name)
            logging.info(f"Added {depth_name} to branch_depths")
            
            try:
                # Collect activations for this key
                key_acts = []
                for sample_idx in range(len(all_inputs)):
                    batch_idx = sample_idx // batch_size
                    if batch_idx < len(branch_activations) and key in branch_activations[batch_idx]:
                        key_acts.append(branch_activations[batch_idx][key])
                
                if key_acts:
                    # Concatenate all activations
                    key_acts = torch.cat(key_acts, dim=0)
                    
                    # Create datasets
                    train_data = TensorDataset(key_acts[train_indices], train_labels)
                    valid_data = TensorDataset(key_acts[valid_indices], valid_labels)
                    
                    # Train classifier
                    accuracy = train_classifier(train_data, valid_data, key_acts.shape[1], num_classes)
                    logging.info(f"Key {key} accuracy: {accuracy:.4f}")
                    
                    # Store the same accuracy for all categories
                    # For dict format, assume these are already post-shunting if use_post_shunting is True
                    results["exc_accuracy"].append(accuracy)
                    results["inh_accuracy"].append(None if use_post_shunting else accuracy)  
                    results["combined_accuracy"].append(accuracy)
                else:
                    # No activations for this key
                    results["exc_accuracy"].append(None)
                    results["inh_accuracy"].append(None)
                    results["combined_accuracy"].append(None)
            except Exception as e:
                logging.error(f"Error analyzing branches for key {key}: {str(e)}")
                logging.error(traceback.format_exc())
                results["exc_accuracy"].append(None)
                results["inh_accuracy"].append(None)
                results["combined_accuracy"].append(None)
        
        # Log summary before returning
        logging.info(f"Final results summary for dict format:")
        logging.info(f"Found {len(results['branch_depths'])} branch depths: {results['branch_depths']}")
        logging.info(f"Excitatory accuracies: {results['exc_accuracy']}")
        logging.info(f"Inhibitory accuracies: {results['inh_accuracy']}")
        logging.info(f"Combined accuracies: {results['combined_accuracy']}")
        
        return results
    
    # For nested_list format
    elif activation_format == "nested_list":
        logging.info("Processing nested_list format")
        n_depths = len(branch_activations[0])
        logging.info(f"Found {n_depths} branch depths in nested_list format")
        
        # VERY IMPORTANT - Initialize branch depth names!
        # This is the line that's missing in the current implementation
        for depth_idx in range(n_depths):
            results["branch_depths"].append(f"Order {depth_idx+1}")
        
        for depth_idx, depth_acts in enumerate(branch_activations[0]):
            logging.info(f"Processing branch depth {depth_idx+1}")
            
            # Extract excitatory, inhibitory, and combined activations for this depth
            exc_accuracy = None
            inh_accuracy = None
            combined_accuracy = None
            
            if use_excitatory and len(depth_acts) > 0:
                try:
                    logging.info(f"Processing excitatory at depth {depth_idx}")
                    if use_post_shunting:
                        # For post-shunting branch outputs
                        post_acts = []
                        for sample_idx in range(len(all_inputs)):
                            batch_idx = sample_idx // batch_size
                            if batch_idx < len(branch_activations) and depth_idx < len(branch_activations[batch_idx]):
                                if isinstance(branch_activations[batch_idx][depth_idx], (list, tuple)) and len(branch_activations[batch_idx][depth_idx]) > 2:
                                    # If third element contains post-shunting outputs
                                    post_acts.append(branch_activations[batch_idx][depth_idx][2])
                                else:
                                    # Manually compute if not available
                                    if len(branch_activations[batch_idx][depth_idx]) >= 2:
                                        exc_inp = branch_activations[batch_idx][depth_idx][0]
                                        inh_inp = branch_activations[batch_idx][depth_idx][1]
                                        
                                        # Apply shunting formula
                                        if use_shunting:
                                            # Shunting: E_ex/(E_ex+E_in+1)
                                            post_output = exc_inp / (exc_inp + inh_inp + 1.0)
                                        else:
                                            # Subtraction: E_ex - E_in
                                            post_output = exc_inp - inh_inp
                                        post_acts.append(post_output)
                        
                        if post_acts:
                            try:
                            post_acts = torch.cat(post_acts, dim=0)
                            
                            # Create train/valid datasets for post-shunting outputs
                            train_post_data = TensorDataset(post_acts[train_indices], train_labels)
                            valid_post_data = TensorDataset(post_acts[valid_indices], valid_labels)
                            
                            # Train classifier on post-shunting outputs
                            exc_accuracy = train_classifier(train_post_data, valid_post_data, 
                                                        post_acts.shape[1], num_classes)
                            logging.info(f"Post-shunting accuracy at depth {depth_idx}: {exc_accuracy:.4f}")
                            except Exception as e:
                                logging.error(f"Error processing post-shunting activations at depth {depth_idx}: {e}")
                                logging.error(traceback.format_exc())
                                exc_accuracy = None
                    else:
                        # For pre-shunting excitatory activations
                        exc_acts = []
                        for sample_idx in range(len(all_inputs)):
                            batch_idx = sample_idx // batch_size
                            if batch_idx < len(branch_activations) and depth_idx < len(branch_activations[batch_idx]):
                                exc_acts.append(branch_activations[batch_idx][depth_idx][0])
                        
                        if exc_acts:
                            try:
                            exc_acts = torch.cat(exc_acts, dim=0)
                            
                            # Create train/valid datasets
                            train_exc_data = TensorDataset(exc_acts[train_indices], train_labels)
                            valid_exc_data = TensorDataset(exc_acts[valid_indices], valid_labels)
                            
                            # Train classifier
                            exc_accuracy = train_classifier(train_exc_data, valid_exc_data, 
                                                        exc_acts.shape[1], num_classes)
                            logging.info(f"Excitatory accuracy at depth {depth_idx}: {exc_accuracy:.4f}")
                            except Exception as e:
                                logging.error(f"Error processing excitatory activations at depth {depth_idx}: {e}")
                                logging.error(traceback.format_exc())
                                exc_accuracy = None
                except Exception as e:
                    logging.error(f"Error analyzing excitatory branches at depth {depth_idx}: {str(e)}")
                    logging.error(traceback.format_exc())
            
            if use_inhibitory and len(depth_acts) > 1 and not use_post_shunting:
                # Only analyze inhibitory inputs separately if not using post-shunting
                try:
                    logging.info(f"Processing inhibitory at depth {depth_idx}")
                    # Collect inhibitory activations
                    inh_acts = []
                    for sample_idx in range(len(all_inputs)):
                        batch_idx = sample_idx // batch_size
                        if batch_idx < len(branch_activations) and depth_idx < len(branch_activations[batch_idx]):
                            inh_acts.append(branch_activations[batch_idx][depth_idx][1])
                    
                    if inh_acts:
                        try:
                        inh_acts = torch.cat(inh_acts, dim=0)
                        
                        # Create train/valid datasets
                        train_inh_data = TensorDataset(inh_acts[train_indices], train_labels)
                        valid_inh_data = TensorDataset(inh_acts[valid_indices], valid_labels)
                        
                        # Train classifier
                        inh_accuracy = train_classifier(train_inh_data, valid_inh_data, 
                                                    inh_acts.shape[1], num_classes)
                        logging.info(f"Inhibitory accuracy at depth {depth_idx}: {inh_accuracy:.4f}")
                        except Exception as e:
                            logging.error(f"Error processing inhibitory activations at depth {depth_idx}: {e}")
                            logging.error(traceback.format_exc())
                            inh_accuracy = None
                except Exception as e:
                    logging.error(f"Error analyzing inhibitory branches at depth {depth_idx}: {str(e)}")
                    logging.error(traceback.format_exc())
                    inh_accuracy = None
            
            # If both excitatory and inhibitory activations are available and not using post-shunting
            if exc_accuracy is not None and inh_accuracy is not None and not use_post_shunting:
                try:
                    logging.info(f"Processing combined at depth {depth_idx}")
                    # Collect combined activations
                    exc_acts = []
                    inh_acts = []
                    
                    for sample_idx in range(len(all_inputs)):
                        batch_idx = sample_idx // batch_size
                        if batch_idx < len(branch_activations) and depth_idx < len(branch_activations[batch_idx]):
                            exc_acts.append(branch_activations[batch_idx][depth_idx][0])
                            inh_acts.append(branch_activations[batch_idx][depth_idx][1])
                    
                    if exc_acts and inh_acts:
                        try:
                        exc_acts = torch.cat(exc_acts, dim=0)
                        inh_acts = torch.cat(inh_acts, dim=0)
                        
                        # Concatenate excitatory and inhibitory activations
                        combined_acts = torch.cat([exc_acts, inh_acts], dim=1)
                        
                        # Create train/valid datasets
                        train_combined_data = TensorDataset(combined_acts[train_indices], train_labels)
                        valid_combined_data = TensorDataset(combined_acts[valid_indices], valid_labels)
                        
                        # Train classifier
                        combined_accuracy = train_classifier(train_combined_data, valid_combined_data, 
                                                        combined_acts.shape[1], num_classes)
                        logging.info(f"Combined accuracy at depth {depth_idx}: {combined_accuracy:.4f}")
                        except Exception as e:
                            logging.error(f"Error processing combined activations at depth {depth_idx}: {e}")
                            logging.error(traceback.format_exc())
                            combined_accuracy = None
                except Exception as e:
                    logging.error(f"Error analyzing combined branches at depth {depth_idx}: {str(e)}")
                    logging.error(traceback.format_exc())
                    combined_accuracy = None
            
            # For post-shunting analysis, combined_accuracy is the same as exc_accuracy
            if use_post_shunting and exc_accuracy is not None:
                combined_accuracy = exc_accuracy
                inh_accuracy = None  # Not applicable for post-shunting
            
            # Store results
            results["exc_accuracy"].append(exc_accuracy)
            results["inh_accuracy"].append(inh_accuracy)
            results["combined_accuracy"].append(combined_accuracy)
        
        # Log summary before returning
        logging.info(f"Final results summary for nested_list format:")
        logging.info(f"Found {len(results['branch_depths'])} branch depths: {results['branch_depths']}")
        logging.info(f"Excitatory accuracies: {results['exc_accuracy']}")
        logging.info(f"Inhibitory accuracies: {results['inh_accuracy']}")
        logging.info(f"Combined accuracies: {results['combined_accuracy']}")
        
        return results
    
    # If unable to determine format, return empty results
    logging.warning("Unknown activation format. Returning empty results.")
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

def analyze_layerwise_ablation(model, dataset, train_data=None, valid_data=None, test_data=None, save_path=None, device='cuda'):
    """
    Run comprehensive ablation analysis on the dendritic network by layers.
    
    Parameters
    ----------
    model : DendriNetWithOutputs
        The dendritic network model
    dataset : Dataset
        Dataset to use for evaluation
    train_data : Dataset, optional
        Training dataset
    valid_data : Dataset, optional
        Validation dataset
    test_data : Dataset, optional
        Test dataset
    save_path : str, optional
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
    net = model.net if hasattr(model, 'net') else model
    if not hasattr(net, 'forward_with_branch_outputs'):
        logging.info("Model does not have forward_with_branch_outputs method. Skipping analysis.")
        return {}
    
    # Create dataloaders with smaller batch size to avoid memory issues
    batch_size = 32
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # If test_data is not provided but dataset is, use dataset
    if test_data is None:
        test_data = dataset
    
    # Get baseline model performance
    baseline_acc = measure_performance(net, test_data, device=device)
    logging.info(f"Baseline accuracy on test set: {baseline_acc:.4f}")
    
    # Save directory for results
    if save_path:
        save_dir = os.path.join(save_path, "layerwise_ablation")
        os.makedirs(save_dir, exist_ok=True)
    
    # Get number of layers in the model
    num_layers = 0
    if hasattr(net, 'branch_layers'):
        num_layers = len(net.branch_layers)
    elif hasattr(net, 'layers'):
        num_layers = len(net.layers)
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
    for n, p in net.named_parameters():
        original_params[n] = p.detach().clone()
    
    # Process from top layers (higher indices) to bottom layers (lower indices)
    for layer_idx in reversed(range(num_layers)):
        logging.info(f"Analyzing layer {layer_idx}...")
        
        # 1. Measure baseline accuracy for this layer by ablating all other direct inputs
        for i in range(num_layers):
            if i != layer_idx:
                # Temporarily ablate other layers
                if hasattr(net, 'branch_layers'):
                    if hasattr(net.branch_layers[i], 'branch_excitation'):
                        net.branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)  # Very negative value for zero activation
                    if hasattr(net.branch_layers[i], 'branch_inhibition'):
                        net.branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)
                elif hasattr(net, 'layers'):
                    # For EINet structure
                    if hasattr(net.layers[i], 'excitatory_cells') and hasattr(net.layers[i].excitatory_cells, 'branch_excitation'):
                        net.layers[i].excitatory_cells.branch_excitation.pre_w.data.fill_(-1000.0)
                    if hasattr(net.layers[i], 'inhibitory_cells') and hasattr(net.layers[i].inhibitory_cells, 'branch_inhibition'):
                        net.layers[i].inhibitory_cells.branch_inhibition.pre_w.data.fill_(-1000.0)
        
        # Measure accuracy with only this layer active
        baseline_layer_acc = measure_performance(net, test_data, device=device)
        
        # 2. Now ablate inhibitory inputs for this layer
        inh_ablated_acc = None
        if hasattr(net, 'branch_layers'):
            if hasattr(net.branch_layers[layer_idx], 'branch_inhibition'):
                net.branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
                inh_ablated_acc = measure_performance(net, test_data, device=device)
        elif hasattr(net, 'layers'):
            if hasattr(net.layers[layer_idx], 'inhibitory_cells') and hasattr(net.layers[layer_idx].inhibitory_cells, 'branch_inhibition'):
                net.layers[layer_idx].inhibitory_cells.branch_inhibition.pre_w.data.fill_(-1000.0)
                inh_ablated_acc = measure_performance(net, test_data, device=device)
        
        if inh_ablated_acc is None:
            inh_ablated_acc = baseline_layer_acc  # No inhibitory inputs to ablate
        
        # Restore parameters
        for n, p in net.named_parameters():
            p.data.copy_(original_params[n])
            
        # Re-ablate other layers
        for i in range(num_layers):
            if i != layer_idx:
                if hasattr(net, 'branch_layers'):
                    if hasattr(net.branch_layers[i], 'branch_excitation'):
                        net.branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)
                    if hasattr(net.branch_layers[i], 'branch_inhibition'):
                        net.branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)
                elif hasattr(net, 'layers'):
                    if hasattr(net.layers[i], 'excitatory_cells') and hasattr(net.layers[i].excitatory_cells, 'branch_excitation'):
                        net.layers[i].excitatory_cells.branch_excitation.pre_w.data.fill_(-1000.0)
                    if hasattr(net.layers[i], 'inhibitory_cells') and hasattr(net.layers[i].inhibitory_cells, 'branch_inhibition'):
                        net.layers[i].inhibitory_cells.branch_inhibition.pre_w.data.fill_(-1000.0)
        
        # 3. Ablate excitatory inputs for this layer
        exc_ablated_acc = None
        if hasattr(net, 'branch_layers'):
            if hasattr(net.branch_layers[layer_idx], 'branch_excitation'):
                net.branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
                exc_ablated_acc = measure_performance(net, test_data, device=device)
        elif hasattr(net, 'layers'):
            if hasattr(net.layers[layer_idx], 'excitatory_cells') and hasattr(net.layers[layer_idx].excitatory_cells, 'branch_excitation'):
                net.layers[layer_idx].excitatory_cells.branch_excitation.pre_w.data.fill_(-1000.0)
                exc_ablated_acc = measure_performance(net, test_data, device=device)
        
        if exc_ablated_acc is None:
            exc_ablated_acc = baseline_layer_acc  # No excitatory inputs to ablate
        
        # Restore all parameters
        for n, p in net.named_parameters():
            p.data.copy_(original_params[n])
        
        # Store results
        results['layer_idx'].append(layer_idx)
        results['baseline_acc'].append(baseline_layer_acc)
        results['inh_ablated_acc'].append(inh_ablated_acc)
        results['exc_ablated_acc'].append(exc_ablated_acc)
        
        logging.info(f"Layer {layer_idx}: baseline={baseline_layer_acc:.4f}, "
                    f"inh_ablated={inh_ablated_acc:.4f}, exc_ablated={exc_ablated_acc:.4f}")
    
    # Plot the results
    if save_path:
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
    
    logging.info(f"Ablation analysis results saved to {save_dir if save_path else 'not saved'}")
    
    return results