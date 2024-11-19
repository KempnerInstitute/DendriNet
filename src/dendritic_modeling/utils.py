"""
utils.py
========
This module contains utility functions for data loading, data splitting, 
and model evaluation.
"""

import os
import json
import shutil

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