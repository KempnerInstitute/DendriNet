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
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms

from .imagenet import (
    IMAGENET_FLATTEN_MATERIALIZATION_LIMIT as IMAGENET_FLATTEN_MATERIALIZATION_LIMIT,
    _apply_imagenet_local_override as _apply_imagenet_local_override,
    _build_imagenet_transforms as _build_imagenet_transforms,
    _class_subset_indices as _class_subset_indices,
    _copy_dataset_with_transform as _copy_dataset_with_transform,
    _is_official_imagenet_root as _is_official_imagenet_root,
    _load_imagenet_dataset_triplet as _load_imagenet_dataset_triplet,
    _materialize_flattened_tensor_dataset as _materialize_flattened_tensor_dataset,
    _normalized_imagenet_loader_backend as _normalized_imagenet_loader_backend,
    _normalized_imagenet_preset_name as _normalized_imagenet_preset_name,
    _prepare_imagenet_dataset_triplet as _prepare_imagenet_dataset_triplet,
    _resolve_imagefolder_roots as _resolve_imagefolder_roots,
    _resolve_imagenet_data_location as _resolve_imagenet_data_location,
    _split_train_valid_pair as _split_train_valid_pair,
    _subset_by_indices as _subset_by_indices,
    load_imagenet_as_datasets,
)
from .paths import get_data_directory, split_generator as _split_generator
from .poisson import PoissonGeneratorDataset as PoissonGeneratorDataset
from .registry import build_registered_dataset
from .synthetic_datasets import get_synthetic_datasets

_ds_logger = logging.getLogger(__name__)

_SEQUENCE_DATASET_NAMES = (
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
)

_THEORETICAL_SYNTHETIC_DATASET_NAMES = (
    "correlated_gaussian",
    "nonlinear_interaction",
    "linear_separable",
    "nonlinear_separable",
    "block_correlated",
    "balanced_information",
    "contextual_stream_gain_shift",
)

_EXPERIMENTAL_DATASET_IMPORT_CANDIDATES = (
    "experiments.data_generation.synthetic_datasets",
    "data_generation.synthetic_datasets",
)

_DatasetTriplet = tuple[Dataset, Dataset, Dataset]


@dataclass(frozen=True)
class _UnifiedDatasetOptions:
    dataset_name: str
    task_data_path: str | None
    parameters: dict[str, Any]
    train_valid_split: float
    flatten: bool
    normalize: bool
    base_dataset: str
    poisson_sampling: bool
    stimulus_duration: float
    multiplicative_gain: bool
    fixed_gain_factor: Any
    max_gain_factor: Any
    max_gain_tau_ratio: Any
    uniform_gain: bool
    split_seed: int
    label_noise_rate: float
    label_noise_seed: int


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


def _maybe_apply_label_noise_to_triplet(
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    *,
    label_noise_rate: float,
    label_noise_seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Apply label noise to the training TensorDataset only when configured."""
    if label_noise_rate > 0.0 and isinstance(train_ds, TensorDataset):
        num_classes = _infer_num_classes(train_ds)
        if num_classes > 0:
            train_ds = _apply_label_noise(
                train_ds,
                label_noise_rate,
                num_classes,
                seed=label_noise_seed,
            )
    return train_ds, valid_ds, test_ds


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


def _dataset_mapping_to_triplet(
    data: dict[str, Dataset],
    *,
    valid_fallback_to_test: bool = False,
    slice_safe: bool = False,
) -> tuple[Dataset, Dataset, Dataset]:
    """Convert a dataset mapping into a train/valid/test triplet."""
    train_ds = data["train"]
    if valid_fallback_to_test:
        valid_ds = data["valid"] if "valid" in data else data["test"]
    else:
        valid_ds = data["valid"]
    test_ds = data["test"]

    if slice_safe:
        return _ensure_slice_safe_triplet(train_ds, valid_ds, test_ds)
    return train_ds, valid_ds, test_ds


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


def _load_standard_vision_dataset_triplet(
    *,
    dataset_name: str,
    train_valid_split: float,
    flatten: bool,
    normalize: bool,
    task_data_path: str | None,
    split_seed: int,
    label_noise_rate: float,
    label_noise_seed: int,
) -> tuple[Dataset, Dataset, Dataset] | None:
    """Load MNIST/Fashion-MNIST/CIFAR-10 with shared post-processing."""
    loaders = {
        "mnist": load_mnist_as_datasets,
        "fashion_mnist": load_fashion_mnist_as_datasets,
        "cifar10": load_cifar10_as_datasets,
    }
    loader = loaders.get(dataset_name)
    if loader is None:
        return None

    train_ds, valid_ds, test_ds = loader(
        train_valid_split=train_valid_split,
        flatten=flatten,
        normalize=normalize,
        task_data_path=task_data_path,
        split_seed=split_seed,
    )
    return _maybe_apply_label_noise_to_triplet(
        train_ds,
        valid_ds,
        test_ds,
        label_noise_rate=label_noise_rate,
        label_noise_seed=label_noise_seed,
    )


def _load_sequence_dataset_triplet(
    *,
    dataset_name: str,
    data_path: str | None,
    train_valid_split: float,
    parameters: dict[str, Any],
) -> tuple[Dataset, Dataset, Dataset]:
    """Load a sequence dataset through the sequence dataset factory."""
    from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets

    return get_sequence_datasets(
        dataset_name=dataset_name,
        data_path=data_path,
        train_valid_split=train_valid_split,
        **parameters,
    )


def _load_theoretical_synthetic_dataset_triplet(
    *,
    dataset_name: str,
    parameters: dict[str, Any],
) -> tuple[Dataset, Dataset, Dataset]:
    """Load built-in theoretical synthetic datasets by injecting the type key."""
    dataset_config = parameters.copy()
    dataset_config["type"] = dataset_name
    return get_synthetic_datasets(dataset_config)


def _load_experimental_dataset_triplet(
    *,
    dataset_name: str,
    task_cfg: Any,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load a dataset from the optional experiments package fallback."""
    # For synthetic/experimental datasets, delegate to experiments synthetic datasets.
    # Keep this robust to both layouts:
    #   <repo>/experiments/experiments/data_generation/synthetic_datasets.py
    #   <repo>/experiments/data_generation/synthetic_datasets.py
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    experiments_root = os.path.join(repo_root, "experiments")

    import sys

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
    for module_name in _EXPERIMENTAL_DATASET_IMPORT_CANDIDATES:
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


def _resolve_unified_dataset_options(task_cfg: Any) -> _UnifiedDatasetOptions:
    dataset_name = _get_config_value(task_cfg, "dataset")
    task_data_path = _get_config_value(task_cfg, "data_path", None)
    parameters = _get_config_value(task_cfg, "parameters", {})
    if task_data_path == "":
        task_data_path = None

    return _UnifiedDatasetOptions(
        dataset_name=dataset_name,
        task_data_path=task_data_path,
        parameters=parameters,
        train_valid_split=_get_config_value(task_cfg, "train_valid_split", 1.0),
        flatten=parameters.get("flatten", True),
        normalize=parameters.get("normalize", False),
        base_dataset=parameters.get("base_dataset", "mnist"),
        poisson_sampling=parameters.get("poisson_sampling", True),
        stimulus_duration=parameters.get("stimulus_duration", 1.0),
        multiplicative_gain=parameters.get("multiplicative_gain", True),
        fixed_gain_factor=parameters.get("fixed_gain_factor", None),
        max_gain_factor=parameters.get("max_gain_factor", None),
        max_gain_tau_ratio=parameters.get("max_gain_tau_ratio", None),
        uniform_gain=parameters.get("uniform_gain", True),
        split_seed=int(parameters.get("split_seed", parameters.get("seed", 0)) or 0),
        label_noise_rate=parameters.get("label_noise_rate", 0.0),
        label_noise_seed=parameters.get("label_noise_seed", 0),
    )


def _load_poisson_generator_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    return load_poisson_generator_datasets(
        base_dataset=options.base_dataset,
        multiplicative_gain=options.multiplicative_gain,
        fixed_gain_factor=options.fixed_gain_factor,
        max_gain_factor=options.max_gain_factor,
        uniform_gain=options.uniform_gain,
        poisson_sampling=options.poisson_sampling,
        stimulus_duration=options.stimulus_duration,
        max_gain_tau_ratio=options.max_gain_tau_ratio,
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        task_data_path=options.task_data_path,
        split_seed=options.split_seed,
    )


def _load_imagenet_from_options(options: _UnifiedDatasetOptions) -> _DatasetTriplet:
    p = options.parameters
    return load_imagenet_as_datasets(
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=options.task_data_path,
        train_samples_per_class=p.get("train_samples_per_class"),
        val_samples_per_class=p.get("val_samples_per_class"),
        split_seed=options.split_seed,
        transform_preset=p.get("transform_preset", "legacy_resize"),
        allow_flatten_materialize=bool(p.get("allow_flatten_materialize", False)),
        loader_backend=p.get("loader_backend", "auto"),
    )


def _load_mnist_modulo10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    task_data_path = options.task_data_path
    p = options.parameters
    data = load_mnist_modulo10(
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=task_data_path,
        shuffle_iterations=p.get("shuffle_iterations", 2),
        split_seed=options.split_seed,
        pair_seed=p.get("pair_seed", 0),
    )
    return _dataset_mapping_to_triplet(data, valid_fallback_to_test=True)


def _load_cifar10_modulo10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    p = options.parameters
    data = load_cifar10_modulo10(
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=options.task_data_path,
        shuffle_iterations=p.get("shuffle_iterations", 1),
        pair_seed=p.get("pair_seed", 0),
        split_seed=options.split_seed,
    )
    return _dataset_mapping_to_triplet(data, slice_safe=True)


def _load_double_cifar10_contextual_mod10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    p = options.parameters
    data = load_double_cifar10_contextual_mod10(
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=options.task_data_path,
        shuffle_iterations=p.get("shuffle_iterations", 1),
        pair_seed=p.get("pair_seed", 0),
        context_seed=p.get("context_seed", 123),
        split_seed=options.split_seed,
    )
    return _dataset_mapping_to_triplet(data, slice_safe=True)


def _load_double_mnist_contextual_mod10_from_options(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet:
    p = options.parameters
    data = load_double_mnist_contextual_mod10(
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=options.task_data_path,
        shuffle_iterations=p.get("shuffle_iterations", 1),
        pair_seed=p.get("pair_seed", 0),
        context_seed=p.get("context_seed", 123),
        split_seed=options.split_seed,
    )
    return _dataset_mapping_to_triplet(data, slice_safe=True)


_SPECIAL_DATASET_LOADERS: dict[
    str, Callable[[_UnifiedDatasetOptions], _DatasetTriplet]
] = {
    "poisson_generator": _load_poisson_generator_from_options,
    "imagenet": _load_imagenet_from_options,
    "mnist_modulo10": _load_mnist_modulo10_from_options,
    "cifar10_modulo10": _load_cifar10_modulo10_from_options,
    "double_cifar10_contextual_mod10": (
        _load_double_cifar10_contextual_mod10_from_options
    ),
    "double_mnist_contextual_mod10": _load_double_mnist_contextual_mod10_from_options,
}


def _load_special_dataset_triplet(
    options: _UnifiedDatasetOptions,
) -> _DatasetTriplet | None:
    dataset_name = options.dataset_name
    p = options.parameters

    loader = _SPECIAL_DATASET_LOADERS.get(dataset_name)
    if loader is not None:
        return loader(options)

    if dataset_name in _THEORETICAL_SYNTHETIC_DATASET_NAMES:
        return _load_theoretical_synthetic_dataset_triplet(
            dataset_name=dataset_name,
            parameters=p,
        )

    if dataset_name in _SEQUENCE_DATASET_NAMES:
        return _load_sequence_dataset_triplet(
            dataset_name=dataset_name,
            data_path=options.task_data_path,
            train_valid_split=options.train_valid_split,
            parameters=p,
        )

    return None


def get_unified_datasets(task_cfg):
    """
    Get datasets based on configuration.

    Args:
        task_cfg: Task configuration (must include train_valid_split)

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    options = _resolve_unified_dataset_options(task_cfg)
    dataset_name = options.dataset_name
    task_data_path = options.task_data_path

    registered = build_registered_dataset(dataset_name, task_cfg)
    if registered is not None:
        return registered

    # Handle standard datasets directly
    standard_vision_datasets = _load_standard_vision_dataset_triplet(
        dataset_name=dataset_name,
        train_valid_split=options.train_valid_split,
        flatten=options.flatten,
        normalize=options.normalize,
        task_data_path=task_data_path,
        split_seed=options.split_seed,
        label_noise_rate=options.label_noise_rate,
        label_noise_seed=options.label_noise_seed,
    )
    if standard_vision_datasets is not None:
        return standard_vision_datasets

    special_datasets = _load_special_dataset_triplet(options)
    if special_datasets is not None:
        return special_datasets

    return _load_experimental_dataset_triplet(
        dataset_name=dataset_name,
        task_cfg=task_cfg,
    )
