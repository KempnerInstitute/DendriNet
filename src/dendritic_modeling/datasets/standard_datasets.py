"""
Standard dataset loading utilities for dendritic modeling.

This module contains functions for loading and preprocessing standard datasets
like MNIST, CIFAR-10, and ImageNet, as well as synthetic datasets for theoretical validation.
"""

import os
from typing import Any, Optional, Union

import math
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

from .synthetic_datasets import get_synthetic_datasets


def _get_config_value(
    config: Union[dict, object], key: str, default: Any = None
) -> Any:
    """Safely get a value from a config object, handling both dict and attribute access."""
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)


def get_data_directory(data_type, task_data_path=None):
    """
    Get or create a data directory for the specified data type.

    Args:
        data_type (str): Type of data (e.g., 'mnist', 'cifar')
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        str: Path to the data directory
    """
    if task_data_path and task_data_path.strip():
        data_path = os.path.join(task_data_path, "datasets", data_type)
    else:
        # Default fallback to './data' directory
        data_path = os.path.join("./data", data_type)

    # Ensure the directory exists
    os.makedirs(data_path, exist_ok=True)

    return data_path


def load_mnist_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
):
    """
    Load the MNIST dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    mnist_dir = get_data_directory("mnist", task_data_path)

    tf_list = [transforms.ToTensor()]
    if normalize:
        tf_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(tf_list)

    train_dataset = datasets.MNIST(
        root=mnist_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=mnist_dir, train=False, download=True, transform=transform
    )

    train_size = int(len(train_dataset) * train_valid_split)
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    x_train = torch.stack([img for img, _ in train_dataset])
    y_train = torch.tensor([label for _, label in train_dataset])

    # Handle empty validation dataset when train_valid_split=1
    if len(valid_dataset) > 0:
        x_valid = torch.stack([img for img, _ in valid_dataset])
        y_valid = torch.tensor([label for _, label in valid_dataset])
    else:
        # Create empty tensors with correct shape
        sample_img, _ = train_dataset[0]
        x_valid = torch.empty(0, *sample_img.shape)
        y_valid = torch.empty(0, dtype=torch.long)

    x_test = torch.stack([img for img, _ in test_dataset])
    y_test = torch.tensor([label for _, label in test_dataset])

    if flatten:
        x_train = x_train.view(x_train.shape[0], -1)
        if len(x_valid) > 0:
            x_valid = x_valid.view(x_valid.shape[0], -1)
        else:
            x_valid = torch.empty(0, x_train.shape[1])
        x_test = x_test.view(x_test.shape[0], -1)

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    valid_ds = torch.utils.data.TensorDataset(x_valid, y_valid)
    test_ds = torch.utils.data.TensorDataset(x_test, y_test)

    return train_ds, valid_ds, test_ds


def load_mnist_modulo10(
    shuffle_iterations=2,
    train_valid_split=1,
    flatten=False,
    normalize=False,
    task_data_path=None,
):
    """
    Load MNIST dataset with modulo 10 operation on labels.

    Args:
        shuffle_iterations (int): Number of shuffle iterations
        train_valid_split (float): Proportion of training data to use for training
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        dict: Dictionary containing train, valid, and test datasets
    """
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        task_data_path=task_data_path,
    )
    data = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    for key, ds in data.items():
        ix = torch.randperm(len(ds))
        ix0 = ix[: int(len(ds) / 2)]
        ix1 = ix[int(len(ds) / 2) :]

        input0, label0 = ds[ix0]
        input1, label1 = ds[ix1]
        assert label0.shape[0] == label1.shape[0]

        inputs = []
        labels = []
        for _ in range(shuffle_iterations):
            inputs.append(torch.cat([input0, input1], dim=-1))
            labels.append((label0 + label1) % 10)

            ix0 = torch.randperm(input0.shape[0])
            input0 = input0[ix0]
            label0 = label0[ix0]

            ix1 = torch.randperm(input1.shape[0])
            input1 = input1[ix1]
            label1 = label1[ix1]

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        data[key] = TensorDataset(inputs, labels)

    return data


def load_cifar10_as_datasets(
    train_valid_split=1.0,
    flatten=False,
    normalize=False,
    task_data_path=None,
):
    """
    Load the CIFAR-10 dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    tf_list = [transforms.ToTensor()]
    if normalize:
        tf_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )

    transform = transforms.Compose(tf_list)

    cifar_dir = get_data_directory("cifar", task_data_path)

    train_dataset = datasets.CIFAR10(
        root=cifar_dir, train=True, download=True, transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=cifar_dir, train=False, download=True, transform=transform
    )

    def convert_to_tensor(dataset):
        data = []
        labels = []
        for img, label in dataset:
            data.append(img)
            labels.append(label)
        data_tensor = torch.stack(data)
        label_tensor = torch.tensor(labels)
        if flatten:
            data_tensor = data_tensor.view(data_tensor.size(0), -1)
        return data_tensor, label_tensor

    if train_valid_split < 1.0:
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size]
        )

        x_train, y_train = convert_to_tensor(train_dataset)
        x_valid, y_valid = convert_to_tensor(valid_dataset)
        x_test, y_test = convert_to_tensor(test_dataset)

        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        valid_ds = torch.utils.data.TensorDataset(x_valid, y_valid)
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    else:
        x_train, y_train = convert_to_tensor(train_dataset)
        x_test, y_test = convert_to_tensor(test_dataset)
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        # Create a separate validation dataset with the same data as train
        # to avoid reference issues
        valid_ds = torch.utils.data.TensorDataset(x_train.clone(), y_train.clone())
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)

    return train_ds, valid_ds, test_ds


class PoissonGeneratorDataset(Dataset):
    """Dataset that applies Poisson noise to base dataset samples."""

    def __init__(
        self,
        base_inputs: torch.Tensor,
        base_labels: torch.Tensor,
        multiplicative_gain: bool = True,
        fixed_gain_factor: Optional[float] = None,
        max_gain_factor: Optional[float] = None,
        uniform_gain: bool = True,
        poisson_sampling: bool = True,
        stimulus_duration: float = 1.0,
        max_gain_tau_ratio: Optional[float] = None
    ):
        stimulus_duration = float(stimulus_duration)
        
        if fixed_gain_factor:
            fixed_gain_factor = float(fixed_gain_factor)
        elif max_gain_factor:
            max_gain_factor = float(max_gain_factor)
        elif max_gain_tau_ratio:
            max_gain_factor = float(max_gain_tau_ratio) * stimulus_duration

        self.base_inputs = base_inputs
        self.base_labels = base_labels
        self.multiplicative_gain = multiplicative_gain
        self.fixed_gain_factor = fixed_gain_factor
        self.max_gain_factor = max_gain_factor
        self.uniform_gain = uniform_gain
        self.poisson_sampling = poisson_sampling
        self.stimulus_duration = stimulus_duration
        self.tau = stimulus_duration

        self._fixed_gf = None
        self._max_gf = None
    
    def _set_gain_factor(self, gain_factor: float | torch.Tensor):
        self._fixed_gf = gain_factor
    
    def _set_max_gain_factor(self, max_gain_factor: float | torch.Tensor):
        self._max_gf = max_gain_factor

    def __len__(self):
        return len(self.base_inputs)

    def __getitem__(self, idx):
        input = self.base_inputs[idx]
        label = self.base_labels[idx]
        
        if self._fixed_gf is not None:
            input = input * self._fixed_gf
            self._fixed_gf = None
        
        elif self._max_gf is not None:
            input = input * (
                (torch.rand_like(input) * (self._max_gf - 1)) + 1
            )
            # input = input * torch.exp(
            #     ((torch.rand_like(input) * 2) - 1
            #     ) * math.log(self.mgf)
            # )
            self._max_gf = None

        elif self.multiplicative_gain:
            if self.fixed_gain_factor is not None:
                gain_factor = self.fixed_gain_factor
            
            elif self.uniform_gain:
                gain_factor = (
                    torch.rand(input.shape[0]) * \
                        (self.max_gain_factor - 1)
                ) + 1
                while gain_factor.ndim < input.ndim:
                    gain_factor = gain_factor[..., None]
            
            else:
                # gain_factor = torch.exp(
                #     ((torch.rand(input.shape[0]) * 2) - 1
                #     ) * math.log(self.max_gain_factor)
                # )
                # gain_factor = torch.exp(
                #     torch.rand(input.shape[0]) * math.log(self.max_gain_factor)
                # )
                gain_factor = (
                    torch.rand_like(input) * (self.max_gain_factor - 1)
                ) + 1
                # gain_factor = torch.exp(
                #     ((torch.rand_like(input) * 2) - 1
                #     ) * math.log(self.max_gain_factor)
                # )
                    
            input = gain_factor * input

        if self.poisson_sampling:
            noisy_input = torch.poisson(self.tau * input)
            noisy_input = noisy_input / self.tau
            return noisy_input, label
        else:
            return input, label


def load_poisson_generator_datasets(
    base_dataset: str,
    multiplicative_gain: bool = True,
    fixed_gain_factor: Optional[float] = None,
    max_gain_factor: Optional[float] = None,
    uniform_gain: bool = True,
    poisson_sampling: bool = True,
    stimulus_duration: float = 1.0,
    max_gain_tau_ratio: Optional[float] = None,
    train_valid_split: float = 0.8,
    flatten: bool = False,
    task_data_path: Optional[str] = None,
):
    """
    Generate a dataset of Poisson-distributed samples from a base dataset.

    Args:
        base_dataset (str): The base dataset to sample from
        stimulus_duration (float): The stimulus duration for the Poisson distribution
        fixed_gain_factor (float): The fixed gain factor for the Poisson distribution
        max_gain_factor (float): The maximum gain factor for the Poisson distribution
        max_gain_tau_ratio (float): The maximum gain tau ratio for the Poisson distribution
        uniform_gain (bool): Whether to use uniform gain for the Poisson distribution
        train_valid_split (float): The proportion of samples to use for training vs validation
        flatten (bool): Whether to flatten the samples
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    # Load the base dataset
    if base_dataset == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            task_data_path=task_data_path,
        )
    elif base_dataset == "cifar10":
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            task_data_path=task_data_path,
        )
    else:
        raise ValueError(f"Unknown base dataset: {base_dataset}")

    train_ds = PoissonGeneratorDataset(
        *train_ds[:], 
        multiplicative_gain=multiplicative_gain,
        fixed_gain_factor=fixed_gain_factor,
        max_gain_factor=max_gain_factor, 
        uniform_gain=uniform_gain,
        poisson_sampling=poisson_sampling,
        stimulus_duration=stimulus_duration,
        max_gain_tau_ratio=max_gain_tau_ratio
    )
    valid_ds = PoissonGeneratorDataset(
        *valid_ds[:], 
        multiplicative_gain=multiplicative_gain,
        fixed_gain_factor=fixed_gain_factor,
        max_gain_factor=max_gain_factor, 
        uniform_gain=uniform_gain,
        poisson_sampling=poisson_sampling,
        stimulus_duration=stimulus_duration,
        max_gain_tau_ratio=max_gain_tau_ratio
    )
    test_ds = PoissonGeneratorDataset(
        *test_ds[:], 
        multiplicative_gain=multiplicative_gain,
        fixed_gain_factor=fixed_gain_factor,
        max_gain_factor=max_gain_factor, 
        uniform_gain=uniform_gain,
        poisson_sampling=poisson_sampling,
        stimulus_duration=stimulus_duration,
        max_gain_tau_ratio=max_gain_tau_ratio
    )

    return train_ds, valid_ds, test_ds


def load_imagenet_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
):
    """
    Load the ImageNet dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    # Check if we have a direct path to existing ImageNet data
    if task_data_path and os.path.exists(os.path.join(task_data_path, "train")):
        # Use the direct path if train directory exists
        imagenet_dir = task_data_path
    else:
        # Use the standard get_data_directory function
        imagenet_dir = get_data_directory("imagenet", task_data_path)

    # ImageNet preprocessing
    tf_list = [transforms.ToTensor()]
    if normalize:
        # ImageNet normalization values
        tf_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    # Add resize transform for ImageNet
    tf_list.insert(0, transforms.Resize((224, 224)))
    transform = transforms.Compose(tf_list)

    try:
        # Try to load ImageNet datasets
        train_dataset = datasets.ImageNet(
            root=imagenet_dir, split="train", transform=transform
        )
        test_dataset = datasets.ImageNet(
            root=imagenet_dir, split="val", transform=transform
        )
    except Exception as e:
        # If ImageNet is not available, create a warning and use a subset of data
        print(f"Warning: Could not load ImageNet from {imagenet_dir}: {e}")
        print("Creating dummy ImageNet-like dataset for testing...")

        # Create dummy data with ImageNet dimensions
        if flatten:
            input_dim = 224 * 224 * 3  # 150528
        else:
            input_dim = (3, 224, 224)

        # Create synthetic data with ImageNet-like properties
        train_size = 100000  # Smaller for testing
        test_size = 10000

        if flatten:
            train_data = torch.randn(train_size, input_dim)
            test_data = torch.randn(test_size, input_dim)
        else:
            train_data = torch.randn(train_size, 3, 224, 224)
            test_data = torch.randn(test_size, 3, 224, 224)

        train_labels = torch.randint(0, 1000, (train_size,))
        test_labels = torch.randint(0, 1000, (test_size,))

        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)

    # Split training data into train and validation
    if hasattr(train_dataset, "__len__"):
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, valid_size]
        )
    else:
        # Fallback for when we can't determine size
        valid_dataset = test_dataset

    # Convert to tensor datasets if needed and flatten if requested
    def process_dataset(dataset):
        if flatten and hasattr(dataset, "__getitem__"):
            # Convert to flattened tensor dataset
            data_list = []
            label_list = []
            # Sample a subset for memory efficiency
            subset_size = min(len(dataset), 50000)  # Limit size for memory
            for i in range(subset_size):
                data, label = dataset[i]
                if isinstance(data, torch.Tensor):
                    if len(data.shape) > 1:  # If not already flattened
                        data = data.view(-1)
                data_list.append(data)
                label_list.append(label)

            if data_list:
                data_tensor = torch.stack(data_list)
                label_tensor = torch.tensor(label_list)
                return TensorDataset(data_tensor, label_tensor)

        return dataset

    if flatten:
        train_dataset = process_dataset(train_dataset)
        valid_dataset = process_dataset(valid_dataset)
        test_dataset = process_dataset(test_dataset)

    return train_dataset, valid_dataset, test_dataset


def get_unified_datasets(task_cfg):
    """
    Get datasets based on configuration.

    Args:
        task_cfg: Task configuration (must include train_valid_split)

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    dataset_name = _get_config_value(task_cfg, "dataset")
    task_data_path = _get_config_value(task_cfg, "data_path", None)
    p = _get_config_value(task_cfg, "parameters", {})

    # If task_data_path is empty string, set it to None
    if task_data_path == "":
        task_data_path = None

    train_valid_split = _get_config_value(task_cfg, "train_valid_split", 1.0)
    flatten = p.get("flatten", True)
    normalize = p.get("normalize", False)
    base_dataset = p.get("base_dataset", "mnist")
    poisson_sampling = p.get("poisson_sampling", True)
    stimulus_duration = p.get("stimulus_duration", 1.0)
    multiplicative_gain = p.get("multiplicative_gain", True)
    fixed_gain_factor = p.get("fixed_gain_factor", None)
    max_gain_factor = p.get("max_gain_factor", None)
    max_gain_tau_ratio = p.get("max_gain_tau_ratio", None)
    uniform_gain = p.get("uniform_gain", True)

    # Handle standard datasets directly
    if dataset_name == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
        )
        return train_ds, valid_ds, test_ds
    elif dataset_name == "cifar10":
        return load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
        )
    elif dataset_name == "poisson_generator":
        return load_poisson_generator_datasets(
            base_dataset=base_dataset,
            multiplicative_gain=multiplicative_gain,
            fixed_gain_factor=fixed_gain_factor,
            max_gain_factor=max_gain_factor,
            uniform_gain=uniform_gain,
            poisson_sampling=poisson_sampling,
            stimulus_duration=stimulus_duration,
            max_gain_tau_ratio=max_gain_tau_ratio,
            train_valid_split=train_valid_split,
            flatten=flatten,
            task_data_path=task_data_path
        )
    elif dataset_name == "imagenet":
        return load_imagenet_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
        )
    elif dataset_name == "mnist_modulo10":
        data = load_mnist_modulo10(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
        )
        train_ds = data["train"]
        if "valid" in data:
            valid_ds = data["valid"]
        else:
            valid_ds = data["test"]
        test_ds = data["test"]
        return train_ds, valid_ds, test_ds
    elif dataset_name in [
        "correlated_gaussian",
        "nonlinear_interaction",
        "linear_separable",
        "nonlinear_separable",
        "block_correlated",
        "balanced_information",
    ]:
        # Handle synthetic datasets for theoretical validation
        dataset_config = p.copy()
        dataset_config["type"] = dataset_name
        return get_synthetic_datasets(dataset_config)
    else:
        # For synthetic/experimental datasets, delegate to the experiments folder
        try:
            # Add experiments directory to path
            experiments_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                ),
                "experiments",
            )
            import sys

            if experiments_path not in sys.path:
                sys.path.insert(0, experiments_path)

            from data_generation.synthetic_datasets import (
                get_unified_datasets as get_experimental_datasets,
            )

            return get_experimental_datasets(task_cfg)
        except ImportError as e:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. Synthetic datasets not available: {e}"
            )
