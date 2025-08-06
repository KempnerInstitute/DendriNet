"""
Standard dataset loading utilities for dendritic modeling.

This module contains functions for loading and preprocessing standard datasets
like MNIST, CIFAR-10, and ImageNet.
"""

import os

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms


# Create a centralized function to get or create a data directory
def get_data_directory(
    data_type, custom_path=None, output_dir=None, task_data_path=None, outputs_dir=None
):
    """
    Get or create a data directory for the specified data type.

    Priority order:
    1. task_data_path (from task.data_path config)
    2. outputs_dir (from outputs.dir config)
    3. custom_path (legacy parameter)
    4. output_dir (legacy parameter)
    5. Default 'data' directory

    Args:
        data_type (str): Type of data (e.g., 'mnist', 'cifar')
        custom_path (str, optional): Custom path provided by the user (legacy)
        output_dir (str, optional): Output directory from config to store datasets (legacy)
        task_data_path (str, optional): Path from task.data_path config (highest priority)
        outputs_dir (str, optional): Directory from outputs.dir config (second priority)

    Returns:
        str: Path to the data directory
    """
    # Priority 1: task.data_path
    if task_data_path and task_data_path.strip():
        data_path = os.path.join(task_data_path, "datasets", data_type)
    # Priority 2: outputs.dir
    elif outputs_dir and outputs_dir.strip():
        data_path = os.path.join(outputs_dir, "datasets", data_type)
    # Priority 3: custom_path (legacy)
    elif custom_path:
        data_path = os.path.join(custom_path, data_type)
    # Priority 4: output_dir (legacy)
    elif output_dir:
        data_path = os.path.join(output_dir, "datasets", data_type)
    else:
        # Priority 5: Default fallback to 'data' directory
        data_path = os.path.join("data", data_type)

    # Ensure the directory exists
    os.makedirs(data_path, exist_ok=True)

    return data_path


def load_mnist_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    data_path=None,
    output_dir=None,
    task_data_path=None,
    outputs_dir=None,
):
    """
    Load the MNIST dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        data_path (str, optional): Path to the data directory
        output_dir (str, optional): Output directory from config to store datasets

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    mnist_dir = get_data_directory(
        "mnist", data_path, output_dir, task_data_path, outputs_dir
    )

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
    cache_dir=None,
    output_dir=None,
    task_data_path=None,
    outputs_dir=None,
):
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        data_path=cache_dir,  # Legacy parameter
        output_dir=output_dir,  # Legacy parameter
        task_data_path=task_data_path,
        outputs_dir=outputs_dir,
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
    data_path=None,
    output_dir=None,
    task_data_path=None,
    outputs_dir=None,
):
    """
    Load the CIFAR-10 dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        data_path (str, optional): Path to the data directory
        output_dir (str, optional): Output directory from config to store datasets

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    tf_list = [transforms.ToTensor()]
    if normalize:
        tf_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )

    transform = transforms.Compose(tf_list)

    cifar_dir = get_data_directory(
        "cifar", data_path, output_dir, task_data_path, outputs_dir
    )

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


def load_imagenet_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    data_path=None,
    output_dir=None,
    task_data_path=None,
    outputs_dir=None,
):
    """
    Load the ImageNet dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        data_path (str, optional): Path to the data directory (legacy)
        output_dir (str, optional): Output directory from config to store datasets (legacy)
        task_data_path (str, optional): Path from task.data_path config
        outputs_dir (str, optional): Directory from outputs.dir config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    # Check if we have a direct path to existing ImageNet data
    if task_data_path and os.path.exists(os.path.join(task_data_path, "train")):
        # Use the direct path if train directory exists
        imagenet_dir = task_data_path
    else:
        # Use the standard get_data_directory function
        imagenet_dir = get_data_directory(
            "imagenet", data_path, output_dir, task_data_path, outputs_dir
        )

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


def get_unified_datasets(task_cfg, train_cfg, output_dir=None, outputs_cfg=None):
    """
    Get datasets based on configuration.

    Args:
        task_cfg: Task configuration
        train_cfg: Training configuration
        output_dir: Output directory from config to store datasets (legacy)
        outputs_cfg: Outputs configuration containing dir

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    try:
        dataset_name = task_cfg["dataset"]
        task_data_path = task_cfg.get("data_path", None)
        p = task_cfg.get("parameters", {})
    except TypeError:
        dataset_name = task_cfg.dataset
        task_data_path = getattr(task_cfg, "data_path", None)
        p = task_cfg.parameters

    # Extract outputs directory from outputs config
    try:
        outputs_dir = outputs_cfg.get("dir", None) if outputs_cfg else None
    except (TypeError, AttributeError):
        outputs_dir = getattr(outputs_cfg, "dir", None) if outputs_cfg else None

    # If task_data_path is empty string, set it to None
    if task_data_path == "":
        task_data_path = None

    # If outputs_dir is empty string, set it to None
    if outputs_dir == "":
        outputs_dir = None

    # Legacy fallback: If both task_data_path and outputs_dir are None, and output_dir is not provided,
    # use the outputs directory in project root instead of current working directory
    if task_data_path is None and outputs_dir is None and output_dir is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        outputs_dir = os.path.join(project_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

    train_valid_split = getattr(train_cfg, "train_valid_split", 1.0)
    flatten = p.get("flatten", True)
    normalize = p.get("normalize", False)

    # Handle standard datasets directly
    if dataset_name == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            data_path=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        return train_ds, valid_ds, test_ds
    elif dataset_name == "cifar10":
        return load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            data_path=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
    elif dataset_name == "imagenet":
        return load_imagenet_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            data_path=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
    elif dataset_name == "mnist_modulo10":
        data = load_mnist_modulo10(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            cache_dir=None,  # Legacy parameter
            output_dir=output_dir,  # Legacy parameter
            task_data_path=task_data_path,
            outputs_dir=outputs_dir,
        )
        train_ds = data["train"]
        if "valid" in data:
            valid_ds = data["valid"]
        else:
            valid_ds = data["test"]
        test_ds = data["test"]
        return train_ds, valid_ds, test_ds
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
                get_unified_datasets as get_synthetic_datasets,
            )

            return get_synthetic_datasets(task_cfg, train_cfg, output_dir, outputs_cfg)
        except ImportError as e:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. Synthetic datasets not available: {e}"
            )
