"""
Standard dataset loading utilities for dendritic modeling.

This module contains functions for loading and preprocessing standard datasets
like MNIST, CIFAR-10, and ImageNet, as well as synthetic datasets for theoretical validation.
"""

import importlib
import logging
import os
import pickle
import shutil
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
from math import log
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from .registry import build_registered_dataset
from .synthetic_datasets import get_synthetic_datasets

_ds_logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX systems
    fcntl = None


def _apply_label_noise(
    dataset: TensorDataset,
    noise_rate: float,
    num_classes: int,
    seed: int = 0,
) -> TensorDataset:
    """Randomly flip a fraction of training labels to random classes.

    Args:
        dataset: TensorDataset with (inputs, labels).
        noise_rate: Fraction of labels to randomize (0 = clean, 1 = fully random).
        num_classes: Number of classes for random label replacement.
        seed: Random seed for reproducibility.

    Returns:
        New TensorDataset with corrupted labels.
    """
    if noise_rate <= 0.0:
        return dataset
    x, y = dataset.tensors
    n = len(y)
    n_noisy = int(n * noise_rate)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng)[:n_noisy]
    noisy_labels = torch.randint(
        0, num_classes, (n_noisy,), generator=rng, dtype=y.dtype
    )
    y_noisy = y.clone()
    y_noisy[indices] = noisy_labels
    actual_flipped = int((y_noisy[indices] != y[indices]).sum().item())
    _ds_logger.info(
        f"Label noise: flipped {actual_flipped}/{n} labels "
        f"(requested rate={noise_rate:.1%}, effective={actual_flipped / n:.1%})"
    )
    return TensorDataset(x, y_noisy)


def _infer_num_classes(dataset: TensorDataset) -> int:
    """Infer number of classes from dataset labels.

    Returns 0 for empty datasets. Raises ValueError if labels are
    not integer-typed (float labels are ambiguous for classification).
    """
    _, y = dataset.tensors
    if y.numel() == 0:
        return 0
    if y.is_floating_point():
        raise ValueError(
            f"Cannot infer num_classes from float labels (dtype={y.dtype}). "
            "Label noise requires integer classification labels."
        )
    return int(y.max().item()) + 1


def _get_config_value(
    config: Union[dict, object], key: str, default: Any = None
) -> Any:
    """Safely get a value from a config object, handling both dict and attribute access."""
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)


def _dataset_to_tensors(
    dataset: Dataset,
    *,
    flatten: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a finite dataset using its declared length.

    Direct iteration over torch Dataset objects relies on ``__getitem__`` raising
    ``IndexError`` out of range. Some wrappers and test doubles do not enforce
    that contract, so explicit indexing by ``len(dataset)`` is safer.
    """
    data = []
    labels = []
    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        if not torch.is_tensor(label):
            label = torch.as_tensor(label)
        data.append(img)
        labels.append(label)

    data_tensor = torch.stack(data)
    label_tensor = torch.stack(labels)
    if flatten:
        data_tensor = data_tensor.view(data_tensor.size(0), -1)
    return data_tensor, label_tensor


@contextmanager
def _dataset_download_lock(root: str):
    """Serialize dataset repair/download work for a shared cache root."""
    if fcntl is None:
        yield
        return

    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, ".download.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _is_recoverable_torchvision_error(exc: BaseException) -> bool:
    """Return True for missing/corrupted-cache errors that merit a clean redownload."""
    if isinstance(exc, (AssertionError, EOFError, pickle.UnpicklingError)):
        return True

    if not isinstance(exc, (RuntimeError, OSError, ValueError)):
        return False

    message = str(exc).lower()
    recoverable_patterns = (
        "dataset not found",
        "file not found or corrupted",
        "pickle data was truncated",
        "truncated",
        "corrupt",
        "corrupted",
        "invalid magic number",
        "pascalvincent",
        "no such file",
    )
    return any(pattern in message for pattern in recoverable_patterns)


def _clear_dataset_cache(root: str) -> None:
    """Remove cached dataset payload while preserving the download lock file."""
    if not os.path.isdir(root):
        return

    for entry in os.listdir(root):
        if entry == ".download.lock":
            continue
        entry_path = os.path.join(root, entry)
        try:
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        except FileNotFoundError:
            continue


def _load_torchvision_train_test_pair(dataset_cls, *, root, transform, dataset_name):
    """Load a torchvision dataset pair with cache repair and serialized download."""
    with _dataset_download_lock(root):
        try:
            train_dataset = dataset_cls(
                root=root, train=True, download=False, transform=transform
            )
            test_dataset = dataset_cls(
                root=root, train=False, download=False, transform=transform
            )
            return train_dataset, test_dataset
        except Exception as exc:
            if not _is_recoverable_torchvision_error(exc):
                raise

            _ds_logger.warning(
                "Recovering %s cache at %s after load failure: %s",
                dataset_name,
                root,
                exc,
            )
            _clear_dataset_cache(root)
            train_dataset = dataset_cls(
                root=root, train=True, download=True, transform=transform
            )
            test_dataset = dataset_cls(
                root=root, train=False, download=True, transform=transform
            )
            return train_dataset, test_dataset


def _split_generator(seed: int | None = 0) -> torch.Generator:
    """Return a CPU generator for reproducible train/validation splits."""
    return torch.Generator().manual_seed(int(seed or 0))


def _normalize_dataset_index(idx: Any, length: int) -> tuple[Any, bool]:
    """Normalize scalar/slice/batch indices into scalar int or list[int]."""
    if isinstance(idx, slice):
        return list(range(*idx.indices(length))), False

    if isinstance(idx, torch.Tensor):
        if idx.ndim == 0:
            return int(idx.item()), True
        return [int(i) for i in idx.detach().cpu().reshape(-1).tolist()], False

    if isinstance(idx, np.ndarray):
        if idx.ndim == 0:
            return int(idx.item()), True
        return [int(i) for i in idx.reshape(-1).tolist()], False

    if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes)):
        return [int(i) for i in idx], False

    return int(idx), True


class SliceSafeDataset(Dataset):
    """
    Dataset adapter that adds robust list/slice/tensor indexing support.

    Scalar access is delegated unchanged. Batch-style indexing is materialized by
    stacking scalar reads, which keeps behavior consistent for wrapped datasets.
    """

    def __init__(self, base_dataset: Dataset):
        super().__init__()
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        normalized_idx, is_scalar = _normalize_dataset_index(
            idx, len(self.base_dataset)
        )

        if is_scalar:
            return self.base_dataset[normalized_idx]

        samples = [self.base_dataset[i] for i in normalized_idx]
        if len(samples) == 0:
            return torch.empty((0,)), torch.empty((0,), dtype=torch.long)

        xs, ys = zip(*samples)
        xs = [x if torch.is_tensor(x) else torch.as_tensor(x) for x in xs]
        ys = [y if torch.is_tensor(y) else torch.as_tensor(y) for y in ys]
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def _ensure_slice_safe_dataset(dataset: Dataset) -> Dataset:
    if isinstance(dataset, (TensorDataset, SliceSafeDataset)):
        return dataset
    return SliceSafeDataset(dataset)


def _ensure_slice_safe_triplet(
    train_ds: Dataset, valid_ds: Dataset, test_ds: Dataset
) -> tuple[Dataset, Dataset, Dataset]:
    return (
        _ensure_slice_safe_dataset(train_ds),
        _ensure_slice_safe_dataset(valid_ds),
        _ensure_slice_safe_dataset(test_ds),
    )


class PairedArithmeticDataset(Dataset):
    """Pair two examples and predict an arithmetic composition of their labels.

    Inputs are emitted either as flattened concatenations or as side-by-side images.
    Optional context can be appended as a one-hot vector (flattened mode) or as
    constant context planes (image mode). This keeps the task usable both for
    plain MLP-style inputs and for future conv-stem experiments.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        *,
        task_mode: str = "sum_mod10",
        flatten: bool = True,
        include_context: bool = False,
        context_dim: int = 2,
        context_seed: int = 0,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.labels = labels
        self.left_indices = left_indices.long()
        self.right_indices = right_indices.long()
        self.task_mode = str(task_mode)
        self.flatten = bool(flatten)
        self.include_context = bool(include_context)
        self.context_dim = int(context_dim)

        if self.include_context:
            generator = torch.Generator().manual_seed(int(context_seed))
            self.context = torch.randint(
                0,
                self.context_dim,
                (len(self.left_indices),),
                generator=generator,
                dtype=torch.long,
            )
        else:
            self.context = None

    def __len__(self) -> int:
        return int(self.left_indices.numel())

    def _compose_input(
        self, left_x: torch.Tensor, right_x: torch.Tensor, context_idx: int | None
    ) -> torch.Tensor:
        if self.flatten:
            pair_input = torch.cat([left_x.reshape(-1), right_x.reshape(-1)], dim=0)
            if context_idx is not None:
                ctx = torch.zeros(self.context_dim, dtype=pair_input.dtype)
                ctx[context_idx] = 1.0
                pair_input = torch.cat([pair_input, ctx], dim=0)
            return pair_input

        # Side-by-side image layout for future convolutional frontends.
        if left_x.ndim >= 2:
            pair_input = torch.cat([left_x, right_x], dim=-1)
        else:
            pair_input = torch.cat([left_x, right_x], dim=0)

        if context_idx is None:
            return pair_input

        if pair_input.ndim == 3:
            _, height, width = pair_input.shape
            ctx_planes = torch.zeros(
                self.context_dim, height, width, dtype=pair_input.dtype
            )
            ctx_planes[context_idx].fill_(1.0)
            return torch.cat([pair_input, ctx_planes], dim=0)

        ctx = torch.zeros(self.context_dim, dtype=pair_input.dtype)
        ctx[context_idx] = 1.0
        return torch.cat([pair_input.reshape(-1), ctx], dim=0)

    def _compose_label(
        self,
        left_label: torch.Tensor,
        right_label: torch.Tensor,
        context_idx: int | None,
    ) -> torch.Tensor:
        yl = int(left_label.item())
        yr = int(right_label.item())

        if self.task_mode == "sum_mod10":
            target = (yl + yr) % 10
        elif self.task_mode == "diff_mod10":
            target = (yl - yr) % 10
        elif self.task_mode == "context_sum_diff":
            if context_idx is None:
                raise ValueError("context_sum_diff requires context indices.")
            target = (yl + yr) % 10 if context_idx == 0 else (yl - yr) % 10
        else:
            raise ValueError(f"Unknown paired arithmetic mode: {self.task_mode}")

        return torch.tensor(target, dtype=torch.long)

    def __getitem__(self, idx):
        li = int(self.left_indices[idx].item())
        ri = int(self.right_indices[idx].item())
        context_idx = (
            int(self.context[idx].item()) if self.context is not None else None
        )

        left_x = self.inputs[li]
        right_x = self.inputs[ri]
        left_y = self.labels[li]
        right_y = self.labels[ri]

        x = self._compose_input(left_x, right_x, context_idx)
        y = self._compose_label(left_y, right_y, context_idx)
        return x, y


def _make_pair_indices(
    n_examples: int,
    *,
    shuffle_iterations: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic left/right pair indices across repeated shuffles."""
    half = n_examples // 2
    generator = torch.Generator().manual_seed(int(seed))
    left_batches = []
    right_batches = []

    for _ in range(max(1, int(shuffle_iterations))):
        perm = torch.randperm(n_examples, generator=generator)
        left_batches.append(perm[:half])
        right_batches.append(perm[half : half + half])

    return torch.cat(left_batches, dim=0), torch.cat(right_batches, dim=0)


def _build_paired_arithmetic_dataset(
    dataset: TensorDataset,
    *,
    shuffle_iterations: int = 1,
    task_mode: str = "sum_mod10",
    flatten: bool = True,
    include_context: bool = False,
    context_seed: int = 0,
    pair_seed: int = 0,
) -> Dataset:
    """Convert a TensorDataset into a paired arithmetic-composition dataset."""
    inputs, labels = dataset.tensors
    left_indices, right_indices = _make_pair_indices(
        len(dataset), shuffle_iterations=shuffle_iterations, seed=pair_seed
    )
    return SliceSafeDataset(
        PairedArithmeticDataset(
            inputs,
            labels,
            left_indices,
            right_indices,
            task_mode=task_mode,
            flatten=flatten,
            include_context=include_context,
            context_seed=context_seed,
        )
    )


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
    split_seed: int = 0,
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

    train_dataset, test_dataset = _load_torchvision_train_test_pair(
        datasets.MNIST,
        root=mnist_dir,
        transform=transform,
        dataset_name="MNIST",
    )

    train_size = int(len(train_dataset) * train_valid_split)
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, valid_size],
        generator=_split_generator(split_seed),
    )

    x_train, y_train = _dataset_to_tensors(train_dataset)

    # Handle empty validation dataset when train_valid_split=1
    if len(valid_dataset) > 0:
        x_valid, y_valid = _dataset_to_tensors(valid_dataset)
    else:
        # Create empty tensors with correct shape
        sample_img, _ = train_dataset[0]
        x_valid = torch.empty(0, *sample_img.shape)
        y_valid = torch.empty(0, dtype=torch.long)

    x_test, y_test = _dataset_to_tensors(test_dataset)

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


def load_fashion_mnist_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
    split_seed: int = 0,
):
    """
    Load the Fashion-MNIST dataset and split into train, validation, and test sets.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    fmnist_dir = get_data_directory("fashion_mnist", task_data_path)

    tf_list = [transforms.ToTensor()]
    if normalize:
        tf_list.append(transforms.Normalize((0.2860,), (0.3530,)))
    transform = transforms.Compose(tf_list)

    train_dataset, test_dataset = _load_torchvision_train_test_pair(
        datasets.FashionMNIST,
        root=fmnist_dir,
        transform=transform,
        dataset_name="FashionMNIST",
    )

    train_size = int(len(train_dataset) * train_valid_split)
    valid_size = len(train_dataset) - train_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, valid_size],
        generator=_split_generator(split_seed),
    )

    x_train, y_train = _dataset_to_tensors(train_dataset)

    if len(valid_dataset) > 0:
        x_valid, y_valid = _dataset_to_tensors(valid_dataset)
    else:
        sample_img, _ = train_dataset[0]
        x_valid = torch.empty(0, *sample_img.shape)
        y_valid = torch.empty(0, dtype=torch.long)

    x_test, y_test = _dataset_to_tensors(test_dataset)

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
    split_seed: int = 0,
    pair_seed: int = 0,
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
        split_seed=split_seed,
    )
    data = {"train": train_ds, "valid": valid_ds, "test": test_ds}

    effective_pair_seed = int(pair_seed) + 1009 * max(1, int(shuffle_iterations))
    for split_idx, (key, ds) in enumerate(data.items()):
        generator = torch.Generator().manual_seed(effective_pair_seed + split_idx)
        ix = torch.randperm(len(ds), generator=generator)
        n_pairs = len(ds) // 2
        ix0 = ix[:n_pairs]
        ix1 = ix[n_pairs : n_pairs + n_pairs]

        input0, label0 = ds[ix0]
        input1, label1 = ds[ix1]

        inputs = []
        labels = []
        for _ in range(shuffle_iterations):
            inputs.append(torch.cat([input0, input1], dim=-1))
            labels.append((label0 + label1) % 10)

            ix0 = torch.randperm(input0.shape[0], generator=generator)
            input0 = input0[ix0]
            label0 = label0[ix0]

            ix1 = torch.randperm(input1.shape[0], generator=generator)
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
    split_seed: int = 0,
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
    train_dataset, test_dataset = _load_torchvision_train_test_pair(
        datasets.CIFAR10,
        root=cifar_dir,
        transform=transform,
        dataset_name="CIFAR10",
    )

    if train_valid_split < 1.0:
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, valid_size],
            generator=_split_generator(split_seed),
        )

        x_train, y_train = _dataset_to_tensors(train_dataset, flatten=flatten)
        x_valid, y_valid = _dataset_to_tensors(valid_dataset, flatten=flatten)
        x_test, y_test = _dataset_to_tensors(test_dataset, flatten=flatten)

        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        valid_ds = torch.utils.data.TensorDataset(x_valid, y_valid)
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)
    else:
        x_train, y_train = _dataset_to_tensors(train_dataset, flatten=flatten)
        x_test, y_test = _dataset_to_tensors(test_dataset, flatten=flatten)
        train_ds = torch.utils.data.TensorDataset(x_train, y_train)
        # Create a separate validation dataset with the same data as train
        # to avoid reference issues
        valid_ds = torch.utils.data.TensorDataset(x_train.clone(), y_train.clone())
        test_ds = torch.utils.data.TensorDataset(x_test, y_test)

    return train_ds, valid_ds, test_ds


def load_cifar10_modulo10(
    shuffle_iterations=1,
    train_valid_split=1.0,
    flatten=True,
    normalize=False,
    task_data_path=None,
    pair_seed: int = 0,
    split_seed: int = 0,
):
    """Load paired CIFAR-10 where the label is the sum of two digits mod 10."""
    train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
        train_valid_split=train_valid_split,
        flatten=False,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )

    return {
        "train": _build_paired_arithmetic_dataset(
            train_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="sum_mod10",
            flatten=flatten,
            include_context=False,
            pair_seed=pair_seed,
        ),
        "valid": _build_paired_arithmetic_dataset(
            valid_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="sum_mod10",
            flatten=flatten,
            include_context=False,
            pair_seed=pair_seed + 1,
        ),
        "test": _build_paired_arithmetic_dataset(
            test_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="sum_mod10",
            flatten=flatten,
            include_context=False,
            pair_seed=pair_seed + 2,
        ),
    }


def load_double_cifar10_contextual_mod10(
    shuffle_iterations=1,
    train_valid_split=1.0,
    flatten=True,
    normalize=False,
    task_data_path=None,
    pair_seed: int = 0,
    context_seed: int = 123,
    split_seed: int = 0,
):
    """Load paired CIFAR-10 with a context-controlled sum-or-difference label."""
    train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
        train_valid_split=train_valid_split,
        flatten=False,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )

    return {
        "train": _build_paired_arithmetic_dataset(
            train_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="context_sum_diff",
            flatten=flatten,
            include_context=True,
            pair_seed=pair_seed,
            context_seed=context_seed,
        ),
        "valid": _build_paired_arithmetic_dataset(
            valid_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="context_sum_diff",
            flatten=flatten,
            include_context=True,
            pair_seed=pair_seed + 1,
            context_seed=context_seed + 1,
        ),
        "test": _build_paired_arithmetic_dataset(
            test_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="context_sum_diff",
            flatten=flatten,
            include_context=True,
            pair_seed=pair_seed + 2,
            context_seed=context_seed + 2,
        ),
    }


def load_double_mnist_contextual_mod10(
    shuffle_iterations=1,
    train_valid_split=1.0,
    flatten=True,
    normalize=False,
    task_data_path=None,
    pair_seed: int = 0,
    context_seed: int = 123,
    split_seed: int = 0,
):
    """Load paired MNIST with a context-controlled sum-or-difference label."""
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(
        train_valid_split=train_valid_split,
        flatten=False,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )

    return {
        "train": _build_paired_arithmetic_dataset(
            train_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="context_sum_diff",
            flatten=flatten,
            include_context=True,
            pair_seed=pair_seed,
            context_seed=context_seed,
        ),
        "valid": _build_paired_arithmetic_dataset(
            valid_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="context_sum_diff",
            flatten=flatten,
            include_context=True,
            pair_seed=pair_seed + 1,
            context_seed=context_seed + 1,
        ),
        "test": _build_paired_arithmetic_dataset(
            test_ds,
            shuffle_iterations=shuffle_iterations,
            task_mode="context_sum_diff",
            flatten=flatten,
            include_context=True,
            pair_seed=pair_seed + 2,
            context_seed=context_seed + 2,
        ),
    }


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
        max_gain_tau_ratio: Optional[float] = None,
    ):
        stimulus_duration = float(stimulus_duration)

        if fixed_gain_factor is not None:
            fixed_gain_factor = float(fixed_gain_factor)
        elif max_gain_factor is not None:
            max_gain_factor = float(max_gain_factor)
        elif max_gain_tau_ratio is not None:
            max_gain_factor = float(max_gain_tau_ratio) * stimulus_duration

        if (
            multiplicative_gain
            and fixed_gain_factor is None
            and max_gain_factor is None
        ):
            if uniform_gain:
                max_gain_factor = 1.0
            else:
                raise ValueError(
                    "PoissonGeneratorDataset with multiplicative_gain=True and "
                    "uniform_gain=False requires fixed_gain_factor, "
                    "max_gain_factor, or max_gain_tau_ratio."
                )
        if max_gain_factor is not None and max_gain_factor < 1.0:
            raise ValueError(
                f"max_gain_factor must be >= 1.0 when provided, got {max_gain_factor}"
            )

        self.base_inputs = base_inputs
        self.base_labels = base_labels
        self.multiplicative_gain = multiplicative_gain
        self.fixed_gain_factor = fixed_gain_factor
        self.max_gain_factor = max_gain_factor
        self.log_max_gain_factor = None
        if isinstance(max_gain_factor, float):
            self.log_max_gain_factor = log(max_gain_factor)
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

    @staticmethod
    def _is_batched_index(idx) -> bool:
        if isinstance(idx, slice):
            return True
        if isinstance(idx, (list, tuple, np.ndarray)):
            return True
        if torch.is_tensor(idx):
            return idx.ndim > 0
        return False

    @staticmethod
    def _sample_uniform_gain(
        input: torch.Tensor,
        log_max_gain_factor: float,
        *,
        batched: bool,
    ) -> torch.Tensor:
        if batched and input.ndim > 0:
            shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        else:
            shape = ()

        log_gain_factor = torch.rand(
            shape, dtype=input.dtype, device=input.device
        ) * log_max_gain_factor
        return torch.exp(log_gain_factor)

    def __len__(self):
        return len(self.base_inputs)

    def __getitem__(self, idx):
        batched = self._is_batched_index(idx)
        input = self.base_inputs[idx]
        label = self.base_labels[idx]

        if self._fixed_gf is not None:
            input = input * self._fixed_gf
            self._fixed_gf = None

        elif self._max_gf is not None:
            input = input * ((torch.rand_like(input) * (self._max_gf - 1)) + 1)
            # input = input * torch.exp(
            #     ((torch.rand_like(input) * 2) - 1
            #     ) * math.log(self.mgf)
            # )
            self._max_gf = None

        elif self.multiplicative_gain:
            if self.fixed_gain_factor is not None:
                gain_factor = self.fixed_gain_factor

            elif self.uniform_gain:
                gain_factor = self._sample_uniform_gain(
                    input, self.log_max_gain_factor, batched=batched
                )

            else:
                gain_factor = torch.exp(
                    torch.rand_like(input) * self.log_max_gain_factor
                )

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
    split_seed: int = 0,
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
            split_seed=split_seed,
        )
    elif base_dataset == "cifar10":
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=False,
            task_data_path=task_data_path,
            split_seed=split_seed,
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
        max_gain_tau_ratio=max_gain_tau_ratio,
    )
    valid_ds = PoissonGeneratorDataset(
        *valid_ds[:],
        multiplicative_gain=multiplicative_gain,
        fixed_gain_factor=fixed_gain_factor,
        max_gain_factor=max_gain_factor,
        uniform_gain=uniform_gain,
        poisson_sampling=poisson_sampling,
        stimulus_duration=stimulus_duration,
        max_gain_tau_ratio=max_gain_tau_ratio,
    )
    test_ds = PoissonGeneratorDataset(
        *test_ds[:],
        multiplicative_gain=multiplicative_gain,
        fixed_gain_factor=fixed_gain_factor,
        max_gain_factor=max_gain_factor,
        uniform_gain=uniform_gain,
        poisson_sampling=poisson_sampling,
        stimulus_duration=stimulus_duration,
        max_gain_tau_ratio=max_gain_tau_ratio,
    )

    return train_ds, valid_ds, test_ds


def load_imagenet_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
    train_samples_per_class=None,
    val_samples_per_class=None,
    split_seed: int = 0,
    allow_fake_data: bool = False,
    transform_preset: str = "legacy_resize",
    allow_flatten_materialize: bool = False,
):
    """
    Load the ImageNet dataset and split into train, validation, and test sets.

    If the environment variable ``IMAGENET_LOCAL_DIR`` is set and points to a
    valid directory, it overrides *task_data_path*.  This allows SLURM scripts
    to copy ImageNet to node-local NVMe (``$SLURM_TMPDIR``) for faster I/O
    without editing the YAML config.

    Args:
        train_valid_split (float): Proportion of training data to use for training (vs validation)
        flatten (bool): Whether to flatten the images
        normalize (bool): Whether to normalize the images
        task_data_path (str, optional): Path from task.data_path config
        train_samples_per_class (int, optional): Limit training samples per class.
        val_samples_per_class (int, optional): Limit validation/test samples per class.
        allow_fake_data (bool): If True, missing ImageNet falls back to a small
            FakeData smoke dataset. Real training should leave this False.
        transform_preset (str): "legacy_resize" preserves the historical
            Resize(224,224) transform; "pretrained" uses train/eval transforms
            appropriate for pretrained torchvision backbones.
        allow_flatten_materialize (bool): Explicit opt-in for the memory-heavy
            ImageNet flatten path.

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    # Allow SLURM scripts to redirect to node-local NVMe for faster I/O.
    local_override = os.environ.get("IMAGENET_LOCAL_DIR")
    if local_override and os.path.isdir(local_override):
        _ds_logger.info(
            "IMAGENET_LOCAL_DIR set — using local data path: %s (was: %s)",
            local_override,
            task_data_path,
        )
        task_data_path = local_override

    def _resolve_imagefolder_roots(root: str | None) -> tuple[str | None, str | None]:
        """Return (train_root, val_root) for ImageFolder-style layouts."""
        if not root or not os.path.isdir(root):
            return None, None

        candidates = [
            (os.path.join(root, "train"), os.path.join(root, "val")),
            (os.path.join(root, "ILSVRC2012_img_train"), os.path.join(root, "val")),
        ]
        for train_root, val_root in candidates:
            if os.path.isdir(train_root) and os.path.isdir(val_root):
                return train_root, val_root
        return None, None

    def _is_official_imagenet_root(root: str | None) -> bool:
        """Return True when the root looks like an official torchvision ImageNet root."""
        if not root or not os.path.isdir(root):
            return False

        required_any = (
            os.path.exists(os.path.join(root, "meta.bin"))
            or os.path.exists(os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz"))
            or os.path.isdir(os.path.join(root, "ILSVRC2012_devkit_t12"))
        )
        has_train = os.path.isdir(
            os.path.join(root, "ILSVRC2012_img_train")
        ) or os.path.isdir(os.path.join(root, "train"))
        has_val = os.path.isdir(os.path.join(root, "val")) or os.path.isdir(
            os.path.join(root, "ILSVRC2012_img_val")
        )
        return required_any and has_train and has_val

    def _class_subset_indices(
        dataset: Dataset,
        max_per_class: int | None,
        split_name: str,
        *,
        seed: int,
    ) -> list[int] | None:
        """Return indices for a fixed number of samples per class."""
        if max_per_class is None:
            return None
        if max_per_class <= 0:
            raise ValueError(
                f"{split_name}_samples_per_class must be positive, got {max_per_class}"
            )

        targets = getattr(dataset, "targets", None)
        if targets is None:
            _ds_logger.warning(
                "Dataset %s has no targets attribute; cannot apply %s_samples_per_class=%s",
                type(dataset).__name__,
                split_name,
                max_per_class,
            )
            return None

        per_class_indices: dict[int, list[int]] = {}
        for idx, target in enumerate(targets):
            per_class_indices.setdefault(int(target), []).append(idx)

        generator = _split_generator(seed)
        selected_indices = []
        for label in sorted(per_class_indices):
            indices = per_class_indices[label]
            order = torch.randperm(len(indices), generator=generator).tolist()
            selected_indices.extend(indices[i] for i in order[:max_per_class])
        selected_indices.sort()

        _ds_logger.info(
            "Subsetting %s split to %d samples across %d classes (%d per class cap)",
            split_name,
            len(selected_indices),
            len(per_class_indices),
            max_per_class,
        )
        return selected_indices

    def _subset_by_indices(dataset: Dataset, indices: list[int] | None) -> Dataset:
        return Subset(dataset, indices) if indices is not None else dataset

    def _split_train_valid_pair(
        train_dataset: Dataset,
        train_eval_dataset: Dataset,
    ) -> tuple[Dataset, Dataset]:
        train_size = int(len(train_dataset) * train_valid_split)
        valid_size = len(train_dataset) - train_size
        indices = torch.randperm(
            len(train_dataset), generator=_split_generator(split_seed)
        ).tolist()
        train_indices = indices[:train_size]
        valid_indices = indices[train_size : train_size + valid_size]
        return Subset(train_dataset, train_indices), Subset(
            train_eval_dataset, valid_indices
        )

    if flatten and not allow_flatten_materialize:
        raise ValueError(
            "ImageNet with flatten=True materializes up to 50k 224x224x3 "
            "samples and can consume tens of GB. Use flatten=False for "
            "AlexNet/pretrained runs, or set "
            "data.dataset_params.imagenet.allow_flatten_materialize=true "
            "for an explicit debugging-only opt-in."
        )

    # Check if we have a direct path to existing ImageNet data
    train_root, val_root = None, None
    if _is_official_imagenet_root(task_data_path):
        imagenet_dir = task_data_path
    else:
        train_root, val_root = _resolve_imagefolder_roots(task_data_path)
        if train_root is not None:
            imagenet_dir = task_data_path
        else:
            # Use the standard get_data_directory function
            imagenet_dir = get_data_directory("imagenet", task_data_path)
            if _is_official_imagenet_root(imagenet_dir):
                train_root, val_root = None, None
            else:
                train_root, val_root = _resolve_imagefolder_roots(imagenet_dir)

    def _append_normalize(tf_list: list) -> list:
        if normalize:
            tf_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )
        return tf_list

    preset = str(transform_preset or "legacy_resize").strip().lower()
    preset = preset.replace("-", "_")
    if preset in {"legacy", "legacy_resize", "resize", "resize_224"}:
        transform = transforms.Compose(
            _append_normalize([transforms.Resize((224, 224)), transforms.ToTensor()])
        )
        train_transform = transform
        eval_transform = transform
    elif preset in {"pretrained", "torchvision", "imagenet", "standard"}:
        train_transform = transforms.Compose(
            _append_normalize(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        )
        eval_transform = transforms.Compose(
            _append_normalize(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
        )
    else:
        raise ValueError(
            "Unknown ImageNet transform_preset "
            f"{transform_preset!r}; expected 'legacy_resize' or 'pretrained'."
        )

    fake_train_transform = train_transform
    fake_eval_transform = eval_transform
    if flatten:
        fake_train_transform = transforms.Compose(
            [train_transform, transforms.Lambda(lambda x: x.view(-1))]
        )
        fake_eval_transform = transforms.Compose(
            [eval_transform, transforms.Lambda(lambda x: x.view(-1))]
        )

    try:
        if train_root is not None:
            _ds_logger.info(
                "Loading ImageNet-style folder tree via ImageFolder: train=%s val=%s",
                train_root,
                val_root,
            )
            train_dataset = datasets.ImageFolder(
                root=train_root, transform=train_transform
            )
            train_eval_dataset = datasets.ImageFolder(
                root=train_root, transform=eval_transform
            )
            test_dataset = datasets.ImageFolder(root=val_root, transform=eval_transform)
        else:
            # Fall back to torchvision's official ImageNet loader when a devkit-based
            # root is available.
            train_dataset = datasets.ImageNet(
                root=imagenet_dir, split="train", transform=train_transform
            )
            train_eval_dataset = datasets.ImageNet(
                root=imagenet_dir, split="train", transform=eval_transform
            )
            test_dataset = datasets.ImageNet(
                root=imagenet_dir, split="val", transform=eval_transform
            )
    except Exception as e:
        if not allow_fake_data:
            raise RuntimeError(
                "Could not load ImageNet. Set data.base_dir or "
                "IMAGENET_LOCAL_DIR to a valid ImageNet root. For smoke tests "
                "only, set data.dataset_params.imagenet.allow_fake_data=true."
            ) from e

        # If ImageNet is not available, fall back to a small lazy synthetic dataset
        # so smoke tests do not allocate tens of GB of host memory.
        _ds_logger.warning(
            "Could not load ImageNet from %s; using FakeData because "
            "allow_fake_data=True: %s",
            imagenet_dir,
            e,
        )

        train_dataset = datasets.FakeData(
            size=512,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=fake_train_transform,
        )
        train_eval_dataset = datasets.FakeData(
            size=512,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=fake_eval_transform,
        )
        test_dataset = datasets.FakeData(
            size=128,
            image_size=(3, 224, 224),
            num_classes=1000,
            transform=fake_eval_transform,
        )

    train_indices = _class_subset_indices(
        train_dataset,
        train_samples_per_class,
        split_name="train",
        seed=int(split_seed),
    )
    train_dataset = _subset_by_indices(train_dataset, train_indices)
    train_eval_dataset = _subset_by_indices(train_eval_dataset, train_indices)
    test_indices = _class_subset_indices(
        test_dataset,
        val_samples_per_class,
        split_name="val",
        seed=int(split_seed) + 1,
    )
    test_dataset = _subset_by_indices(test_dataset, test_indices)

    # Split training data into train and validation
    if hasattr(train_dataset, "__len__"):
        train_dataset, valid_dataset = _split_train_valid_pair(
            train_dataset,
            train_eval_dataset,
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

    registered = build_registered_dataset(dataset_name, task_cfg)
    if registered is not None:
        return registered

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
    split_seed = int(p.get("split_seed", p.get("seed", 0)) or 0)

    label_noise_rate = p.get("label_noise_rate", 0.0)
    label_noise_seed = p.get("label_noise_seed", 0)

    def _maybe_inject_label_noise(train_ds, valid_ds, test_ds):
        """Apply label noise to training set only if configured."""
        if label_noise_rate > 0.0 and isinstance(train_ds, TensorDataset):
            num_classes = _infer_num_classes(train_ds)
            if num_classes > 0:
                train_ds = _apply_label_noise(
                    train_ds, label_noise_rate, num_classes, seed=label_noise_seed
                )
        return train_ds, valid_ds, test_ds

    # Handle standard datasets directly
    if dataset_name == "mnist":
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            split_seed=split_seed,
        )
        return _maybe_inject_label_noise(train_ds, valid_ds, test_ds)
    elif dataset_name == "fashion_mnist":
        train_ds, valid_ds, test_ds = load_fashion_mnist_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            split_seed=split_seed,
        )
        return _maybe_inject_label_noise(train_ds, valid_ds, test_ds)
    elif dataset_name == "cifar10":
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            split_seed=split_seed,
        )
        return _maybe_inject_label_noise(train_ds, valid_ds, test_ds)
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
            task_data_path=task_data_path,
            split_seed=split_seed,
        )
    elif dataset_name == "imagenet":
        return load_imagenet_as_datasets(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            train_samples_per_class=p.get("train_samples_per_class"),
            val_samples_per_class=p.get("val_samples_per_class"),
            split_seed=split_seed,
            allow_fake_data=bool(p.get("allow_fake_data", False)),
            transform_preset=p.get("transform_preset", "legacy_resize"),
            allow_flatten_materialize=bool(p.get("allow_flatten_materialize", False)),
        )
    elif dataset_name == "mnist_modulo10":
        data = load_mnist_modulo10(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            shuffle_iterations=p.get("shuffle_iterations", 2),
            split_seed=split_seed,
            pair_seed=p.get("pair_seed", 0),
        )
        train_ds = data["train"]
        if "valid" in data:
            valid_ds = data["valid"]
        else:
            valid_ds = data["test"]
        test_ds = data["test"]
        return train_ds, valid_ds, test_ds
    elif dataset_name == "cifar10_modulo10":
        data = load_cifar10_modulo10(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            shuffle_iterations=p.get("shuffle_iterations", 1),
            pair_seed=p.get("pair_seed", 0),
            split_seed=split_seed,
        )
        train_ds = data["train"]
        valid_ds = data["valid"]
        test_ds = data["test"]
        return _ensure_slice_safe_triplet(train_ds, valid_ds, test_ds)
    elif dataset_name == "double_cifar10_contextual_mod10":
        data = load_double_cifar10_contextual_mod10(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            shuffle_iterations=p.get("shuffle_iterations", 1),
            pair_seed=p.get("pair_seed", 0),
            context_seed=p.get("context_seed", 123),
            split_seed=split_seed,
        )
        train_ds = data["train"]
        valid_ds = data["valid"]
        test_ds = data["test"]
        return _ensure_slice_safe_triplet(train_ds, valid_ds, test_ds)
    elif dataset_name == "double_mnist_contextual_mod10":
        data = load_double_mnist_contextual_mod10(
            train_valid_split=train_valid_split,
            flatten=flatten,
            normalize=normalize,
            task_data_path=task_data_path,
            shuffle_iterations=p.get("shuffle_iterations", 1),
            pair_seed=p.get("pair_seed", 0),
            context_seed=p.get("context_seed", 123),
            split_seed=split_seed,
        )
        train_ds = data["train"]
        valid_ds = data["valid"]
        test_ds = data["test"]
        return _ensure_slice_safe_triplet(train_ds, valid_ds, test_ds)
    elif dataset_name in [
        "correlated_gaussian",
        "nonlinear_interaction",
        "linear_separable",
        "nonlinear_separable",
        "block_correlated",
        "balanced_information",
        "contextual_stream_gain_shift",
    ]:
        # Handle synthetic datasets for theoretical validation
        dataset_config = p.copy()
        dataset_config["type"] = dataset_name
        return get_synthetic_datasets(dataset_config)
    elif dataset_name in (
        "sequential_mnist",
        "copy_task",
        "adding_problem",
        "variable_delay_dms",
        "multi_frequency",
        "hierarchical_temporal",
        "context_dependent_decision",
        "context_dependent_decision_hard",
        "context_dependent_decision_positive",
        "context_dependent_decision_hard_positive",
        "switching_context_decision",
        "switching_context_decision_positive",
        "timescale_generalization",
        "timescale_generalization_positive",
        "odr_generalization",
        "odr_generalization_positive",
        "romo_delay_comparison",
        "romo_delay_comparison_positive",
        "gain_modulated_contextual_comparison",
        "gain_modulated_contextual_comparison_positive",
        "lorenz",
        "lorenz_sequence",
        "switching_lds",
        "multi_sine_forecast",
        "stringer_v1",
    ):
        from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets

        return get_sequence_datasets(
            dataset_name=dataset_name,
            data_path=task_data_path,
            train_valid_split=train_valid_split,
            **p,
        )
    else:
        # For synthetic/experimental datasets, delegate to experiments synthetic datasets.
        # Keep this robust to both layouts:
        #   <repo>/experiments/experiments/data_generation/synthetic_datasets.py
        #   <repo>/experiments/data_generation/synthetic_datasets.py
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        experiments_root = os.path.join(repo_root, "experiments")

        import sys

        # Try canonical package imports first.
        import_candidates = [
            "experiments.data_generation.synthetic_datasets",
            "data_generation.synthetic_datasets",
        ]
        # Ensure both candidate roots are importable regardless launch cwd/PYTHONPATH.
        path_candidates = [
            experiments_root,
            os.path.join(experiments_root, "experiments"),
        ]
        for candidate_path in path_candidates:
            if os.path.isdir(candidate_path) and candidate_path not in sys.path:
                sys.path.insert(0, candidate_path)

        last_error: Exception | None = None
        get_experimental_datasets = None
        for module_name in import_candidates:
            try:
                module = importlib.import_module(module_name)
                get_experimental_datasets = module.get_unified_datasets
                break
            except Exception as e:  # pragma: no cover - fallback path
                last_error = e

        if get_experimental_datasets is None:
            raise ValueError(
                f"Unknown dataset name: {dataset_name}. Synthetic datasets not available: {last_error}"
            )

        train_ds, valid_ds, test_ds = get_experimental_datasets(task_cfg, task_cfg)
        return _ensure_slice_safe_triplet(train_ds, valid_ds, test_ds)
