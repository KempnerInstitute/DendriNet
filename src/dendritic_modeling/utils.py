"""
utils.py
========
This module contains utility functions for data loading, data splitting, 
and model evaluation.
"""

import os
import json
import shutil
from copy import deepcopy

import numpy as np
import torch
import torchvision
from torchvision import transforms
from omegaconf import OmegaConf, DictConfig, ListConfig



class Data(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset for handling input features and labels.
    
    Parameters
    ----------
    x : Tensor or array-like 
        Input features.
    y : Tensor or array-like 
        Corresponding labels.


    Examples
    --------
    >>> dataset = Data(x_data, y_data)
    >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    >>> for inputs, labels in dataloader:
    ...     # Your training code here
    ...     pass
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.y.shape[0]


class Shaper(object):
    """
    A utility class for reshaping input data to a specified shape.

    The class takes an intended shape upon initialization and proivdes a method
    to reshape any given tensor to that shape.

    Parameters
    ----------
    shape : tuple of int
        The intended shape to which the input tensor should be reshaped.

    Methods
    -------
    reshape(x)
        Reshapes the input tensor `x` to the specified shape.

    Examples
    --------

    >>> shaper = Shaper(shape = (32, 1, 28, 28))
    >>> x = torch.randn(32, 784)
    >>> x_reshaped = shaper.reshape(x)
    >>> x_reshaped.shape
    torch.Size([32, 1, 28, 28])
    """

    def __init__(self, shape):
        self.shape = shape
    
    def reshape(self, x):
        return x.view(*self.shape)


def convert_to_serializable(obj):
    """
    Recursively convert OmegaConf objects to standard Python data types.
    
    Parameters
    ----------
    obj : Any
        The object to convert. Can be a DictConfig, ListConfig, dict, list, or primitive type.
    
    Returns
    -------
    Any
        A JSON-serializable version of the object.
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

def save_dict(dict_obj, save_path, fname):
    # Convert OmegaConf configurations to standard dictionary
    dict_obj = convert_to_serializable(dict_obj)

    # Define the full path for the JSON file
    full_path = os.path.join(save_path, fname)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)
    
    # Save the converted dictionary as JSON
    with open(full_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)



def load_MNIST(train_valid_split = 1, cache_dir=None):
    """
    Load the MNIST dataset from torchvision.datasets.MNIST with optional 
    caching and train-validation split.

    Parameters
    ----------
    train_valid_split : float, optional
        Fraction of the training data to use for validation. Defaults to 1 
        (no validation split).

    cache_dir : str, optional
        Path to the directory where the dataset should be cached. 

    Returns
    -------

    data : dict
        A dictionary containing the training, validation, and test datasets.


    Examples
    --------

    >>> data = load_MNIST(train_valid_split = 0.8)
    >>> train_data = data['train']
    >>> valid_data = data['valid']
    >>> test_data = data['test']
    """
    
    
    if cache_dir is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(root, 'data', 'mnist')
    else:
        data_path = os.path.join(cache_dir, 'data', 'mnist')


    if os.path.exists(data_path):
        train_data = Data(
            torch.load(os.path.join(data_path, 'train_inputs.pt'), 
                       weights_only = True),
            torch.load(os.path.join(data_path, 'train_labels.pt'), 
                       weights_only = True),
        )
        test_data = Data(
            torch.load(os.path.join(data_path, 'test_inputs.pt'), 
                       weights_only = True),
            torch.load(os.path.join(data_path, 'test_labels.pt'), 
                       weights_only = True),
        )

    else:
        os.makedirs(data_path, exist_ok = True)

        transform = transforms.ToTensor()

        def loader_saver(train = True):
            dataset = torchvision.datasets.MNIST(root = data_path, 
                                                 train = train, 
                                                 download = True)
            imgs, labels = zip(*(dataset[i] for i in range(len(dataset))))
            
            imgs = torch.stack([transform(img).flatten() for img in imgs], 
                               dim = 0)
            labels = torch.tensor(labels)
                        
            name = 'train' if train else 'test'
            torch.save(imgs, os.path.join(data_path, f'{name}_inputs.pt'))
            torch.save(labels, os.path.join(data_path, f'{name}_labels.pt'))
            
            return Data(imgs, labels)

        train_data = loader_saver(train = True)
        test_data = loader_saver(train = False)

        shutil.rmtree(os.path.join(data_path, 'MNIST'))

    data = {}

    if train_valid_split < 1:
        train_ix = torch.randperm(len(train_data))[
            :int(train_valid_split*len(train_data))
        ]
        train_mask = torch.zeros((len(train_data),)).bool()
        train_mask[train_ix] = True

        valid_data = Data(*train_data[~train_mask])
        train_data = Data(*train_data[train_mask])

        data['valid'] = valid_data

    data['train'] = train_data
    data['test'] = test_data

    return data


def load_MNIST_modulo10(shuffle_iterations = 2, 
                        train_valid_split = 1,
                        cache_dir = None):
    
    data = load_MNIST(train_valid_split, cache_dir)

    for key, ds in data.items():
        ix = torch.randperm(len(ds))
        ix0 = ix[:int(len(ds)/2)]
        ix1 = ix[int(len(ds)/2):]

        input0, label0 = ds[ix0]
        input1, label1 = ds[ix1]
        assert(label0.shape[0] == label1.shape[0])

        inputs = []
        labels = []
        for _ in range(shuffle_iterations):
            inputs.append(torch.cat([input0, input1], dim = -1))
            labels.append((label0 + label1) % 10)

            ix0 = torch.randperm(input0.shape[0])
            input0 = input0[ix0]
            label0 = label0[ix0]

            ix1 = torch.randperm(input1.shape[0])
            input1 = input1[ix1]
            label1 = label1[ix1]
        
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)

        data[key] = Data(inputs, labels)

    return data


def load_MNIST_task_switch(train_valid_split = 1, cache_dir = None):
    data = load_MNIST(train_valid_split, cache_dir)

    for key, ds in data.items():
        input0, label0 = ds[:]
        input1 = deepcopy(input0)

        input0[:,0] = 1
        input0[:,1] = 0

        input1[:,0] = 0
        input1[:,1] = 1
        label1 = 9 - label0

        inputs = torch.cat([input0, input1], dim = 0)
        labels = torch.cat([label0, label1], dim = 0)
        
        data[key] = Data(inputs, labels)
    
    return data


def load_MNIST_task_switch0(train_valid_split = 1, cache_dir = None):
    data = load_MNIST(train_valid_split, cache_dir)

    for key, ds in data.items():
        raw_input, raw_label = ds[:]
        raw_input[:,:10] = 0

        inputs = []
        labels = []

        for i in range(10):
            task_input = deepcopy(raw_input)
            task_input[:,i] = 1

            task_label = (raw_label + i) % 10

            inputs.append(task_input)
            labels.append(task_label)
        
        inputs = torch.cat(inputs, dim = 0)
        labels = torch.cat(labels, dim = 0)

        data[key] = Data(inputs, labels)

    return data


def split_MNIST_inputs(inputs, labels):
    """
    Split the input features and labels into separate lists based on unique
    labels.

    Parameters
    ----------
    inputs : Tensor
        The input data tesnor with shape `(num_samples, ...)`.

    labels : Tensor
        The corresponding labels tensor with shape `(num_samples,)`.

    Returns
    -------
    input_list : list of Tensor
        A list of tensors where each tensor contains all samples with the same
        label.

    Examples
    --------
    >>> inputs = torch.randn(1000, 784)
    >>> labels = torch.randint(0, 10, (1000,))
    >>> input_list = split_MNIST_inputs(inputs, labels)
    >>> len(input_list)
    10
    """
    unique_label = torch.unique(labels)

    input_list = []

    for unique in unique_label:
        ix = torch.where(labels == unique, True, False)
        input_list.append(inputs[ix])

    return input_list


def accuracy_score(classifier, inputs, labels):
    """
    Computes the accuracy score of a classifier on a given set of inputs and
    labels. 

    Parameters
    ----------
    classifier : nn.Module
        The classifier pytorch model with a `predict` method that takes input
        data and returns predicted labels.

    inputs : Tensor
        The input data tensor with shape `(num_samples, ...)`.

    labels : Tensor
    """

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifier = classifier.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    yhat = classifier.predict(inputs)

    accuracy = (yhat == labels).float().mean(dim = -1)
    return accuracy



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
        # do the direct ablation plus any 'branches_to_output'
        # or block connections that feed from upstream. 
        # Implementation depends on your actual code structure.
        if layer.input_excitatory:
            for b_id in branch_ids:
                layer.branch_excitation.pre_w.data[b_id, :] = 0.
        if layer.input_inhibitory and hasattr(layer, 'branch_inhibition'):
            for b_id in branch_ids:
                layer.branch_inhibition.pre_w.data[b_id, :] = 0.
        # plus zero out blockLinear if used
        if layer.input_branches and hasattr(layer, 'branches_to_output'):
            # ex: zero out the relevant rows/cols in layer.branches_to_output.log_weight
            # or do something that cuts the entire sub-tree. 
            # This is a conceptual placeholder.
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