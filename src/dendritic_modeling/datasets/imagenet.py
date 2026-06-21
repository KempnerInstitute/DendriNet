"""ImageNet dataset loading and preprocessing helpers."""

import logging
import os
from copy import copy
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from .paths import get_data_directory, split_generator

_logger = logging.getLogger(__name__)

IMAGENET_FLATTEN_MATERIALIZATION_LIMIT = 50_000


def _resolve_imagefolder_roots(root: str | None) -> tuple[str | None, str | None]:
    """Return ``(train_root, val_root)`` for ImageFolder-style ImageNet layouts."""
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


def _normalized_imagenet_loader_backend(loader_backend: str | None) -> str:
    """Normalize ImageNet loader backend aliases."""
    backend = str(loader_backend or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "default": "auto",
        "torchvision": "torchvision",
        "imagenet": "torchvision",
        "official": "torchvision",
        "image_folder": "imagefolder",
        "imagefolder": "imagefolder",
        "folder": "imagefolder",
    }
    if backend not in aliases:
        raise ValueError(
            "Unknown ImageNet loader_backend "
            f"{loader_backend!r}; expected 'auto', 'torchvision', or 'imagefolder'."
        )
    return aliases[backend]


def _is_official_imagenet_root(root: str | None) -> bool:
    """Return True when a root looks like a torchvision ImageNet root."""
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


def _apply_imagenet_local_override(task_data_path: str | None) -> str | None:
    """Apply ``IMAGENET_LOCAL_DIR`` when it points to an existing directory."""
    local_override = os.environ.get("IMAGENET_LOCAL_DIR")
    if local_override and os.path.isdir(local_override):
        _logger.info(
            "IMAGENET_LOCAL_DIR set — using local data path: %s (was: %s)",
            local_override,
            task_data_path,
        )
        return local_override
    return task_data_path


def _resolve_imagenet_data_location(
    task_data_path: str | None,
    loader_backend: str | None = "auto",
) -> tuple[str, str | None, str | None]:
    """Resolve ImageNet root plus optional ImageFolder train/val roots."""
    backend = _normalized_imagenet_loader_backend(loader_backend)
    task_data_path = _apply_imagenet_local_override(task_data_path)

    train_root, val_root = None, None
    if backend == "torchvision" and task_data_path and task_data_path.strip():
        return str(task_data_path), train_root, val_root

    if backend == "imagefolder":
        train_root, val_root = _resolve_imagefolder_roots(task_data_path)
        if train_root is not None:
            return str(task_data_path), train_root, val_root
    elif _is_official_imagenet_root(task_data_path):
        return str(task_data_path), train_root, val_root

    if backend != "torchvision":
        train_root, val_root = _resolve_imagefolder_roots(task_data_path)
    if train_root is not None:
        return str(task_data_path), train_root, val_root

    imagenet_dir = get_data_directory("imagenet", task_data_path)
    if backend == "imagefolder":
        train_root, val_root = _resolve_imagefolder_roots(imagenet_dir)
        return imagenet_dir, train_root, val_root

    if _is_official_imagenet_root(imagenet_dir):
        return imagenet_dir, None, None

    if backend != "torchvision":
        train_root, val_root = _resolve_imagefolder_roots(imagenet_dir)
    return imagenet_dir, train_root, val_root


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
        _logger.warning(
            "Dataset %s has no targets attribute; cannot apply %s_samples_per_class=%s",
            type(dataset).__name__,
            split_name,
            max_per_class,
        )
        return None

    per_class_indices: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        per_class_indices.setdefault(int(target), []).append(idx)

    generator = split_generator(seed)
    selected_indices = []
    for label in sorted(per_class_indices):
        indices = per_class_indices[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected_indices.extend(indices[i] for i in order[:max_per_class])
    selected_indices.sort()

    _logger.info(
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
    *,
    train_valid_split: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    train_size = int(len(train_dataset) * train_valid_split)
    valid_size = len(train_dataset) - train_size
    indices = torch.randperm(
        len(train_dataset), generator=split_generator(split_seed)
    ).tolist()
    train_indices = indices[:train_size]
    valid_indices = indices[train_size : train_size + valid_size]
    return Subset(train_dataset, train_indices), Subset(
        train_eval_dataset, valid_indices
    )


def _append_imagenet_normalize(tf_list: list, *, normalize: bool) -> list:
    """Append ImageNet normalization to a transform list when requested."""
    if normalize:
        tf_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return tf_list


def _normalized_imagenet_preset_name(transform_preset: str | None) -> str:
    return str(transform_preset or "legacy_resize").strip().lower().replace("-", "_")


def _build_imagenet_transforms(
    *,
    normalize: bool,
    transform_preset: str,
) -> tuple[Any, Any]:
    """Build train/eval transforms for ImageNet without changing preset semantics."""
    preset = _normalized_imagenet_preset_name(transform_preset)
    if preset in {"legacy", "legacy_resize", "resize", "resize_224"}:
        transform = transforms.Compose(
            _append_imagenet_normalize(
                [transforms.Resize((224, 224)), transforms.ToTensor()],
                normalize=normalize,
            )
        )
        return transform, transform

    if preset in {"pretrained", "torchvision", "imagenet", "standard"}:
        train_transform = transforms.Compose(
            _append_imagenet_normalize(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ],
                normalize=normalize,
            )
        )
        eval_transform = transforms.Compose(
            _append_imagenet_normalize(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ],
                normalize=normalize,
            )
        )
        return train_transform, eval_transform

    raise ValueError(
        "Unknown ImageNet transform_preset "
        f"{transform_preset!r}; expected 'legacy_resize' or 'pretrained'."
    )


def _copy_dataset_with_transform(dataset: Dataset, transform) -> Dataset:
    """Return a shallow dataset view with the same samples and a new transform."""
    dataset_view = copy(dataset)
    dataset_view.transform = transform

    transforms_obj = getattr(dataset_view, "transforms", None)
    if transforms_obj is not None and hasattr(transforms_obj, "transform"):
        dataset_view.transforms = copy(transforms_obj)
        dataset_view.transforms.transform = transform

    return dataset_view


def _load_imagenet_dataset_triplet(
    *,
    imagenet_dir: str,
    train_root: str | None,
    val_root: str | None,
    train_transform,
    eval_transform,
    loader_backend: str | None = "auto",
) -> tuple[Dataset, Dataset, Dataset]:
    """Load ImageNet train/eval/test datasets from ImageFolder or ImageNet roots."""
    backend = _normalized_imagenet_loader_backend(loader_backend)
    if backend == "imagefolder" and train_root is None:
        raise RuntimeError(
            "ImageNet loader_backend='imagefolder' requires train/val "
            "class-directory roots."
        )
    if train_root is not None:
        _logger.info(
            "Loading ImageNet-style folder tree via ImageFolder: train=%s val=%s",
            train_root,
            val_root,
        )
        train_dataset = datasets.ImageFolder(root=train_root, transform=train_transform)
        return (
            train_dataset,
            _copy_dataset_with_transform(train_dataset, eval_transform),
            datasets.ImageFolder(root=val_root, transform=eval_transform),
        )

    train_dataset = datasets.ImageNet(
        root=imagenet_dir, split="train", transform=train_transform
    )
    return (
        train_dataset,
        _copy_dataset_with_transform(train_dataset, eval_transform),
        datasets.ImageNet(root=imagenet_dir, split="val", transform=eval_transform),
    )


def _prepare_imagenet_dataset_triplet(
    *,
    train_dataset: Dataset,
    train_eval_dataset: Dataset,
    test_dataset: Dataset,
    train_valid_split: float,
    train_samples_per_class: int | None,
    val_samples_per_class: int | None,
    split_seed: int,
    flatten: bool,
) -> tuple[Dataset, Dataset, Dataset]:
    """Apply ImageNet subsetting, validation split, and optional flattening."""
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

    if hasattr(train_dataset, "__len__"):
        train_dataset, valid_dataset = _split_train_valid_pair(
            train_dataset,
            train_eval_dataset,
            train_valid_split=train_valid_split,
            split_seed=split_seed,
        )
    else:
        valid_dataset = test_dataset

    if flatten:
        train_dataset = _materialize_flattened_tensor_dataset(train_dataset)
        valid_dataset = _materialize_flattened_tensor_dataset(valid_dataset)
        test_dataset = _materialize_flattened_tensor_dataset(test_dataset)

    return train_dataset, valid_dataset, test_dataset


def _materialize_flattened_tensor_dataset(
    dataset: Dataset,
    *,
    max_samples: int = IMAGENET_FLATTEN_MATERIALIZATION_LIMIT,
) -> Dataset:
    """Convert an indexable image dataset into a capped flattened TensorDataset."""
    if not hasattr(dataset, "__getitem__"):
        return dataset

    data_list = []
    label_list = []
    subset_size = min(len(dataset), max_samples)
    for i in range(subset_size):
        data, label = dataset[i]
        if isinstance(data, torch.Tensor) and len(data.shape) > 1:
            data = data.view(-1)
        data_list.append(data)
        label_list.append(label)

    if not data_list:
        return dataset

    data_tensor = torch.stack(data_list)
    label_tensor = torch.tensor(label_list)
    return TensorDataset(data_tensor, label_tensor)


def load_imagenet_as_datasets(
    train_valid_split=0.8,
    flatten=False,
    normalize=False,
    task_data_path=None,
    train_samples_per_class=None,
    val_samples_per_class=None,
    split_seed: int = 0,
    transform_preset: str = "legacy_resize",
    allow_flatten_materialize: bool = False,
    loader_backend: str = "auto",
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
        transform_preset (str): "legacy_resize" preserves the historical
            Resize(224,224) transform; "pretrained" uses train/eval transforms
            appropriate for pretrained torchvision backbones.
        allow_flatten_materialize (bool): Explicit opt-in for the memory-heavy
            ImageNet flatten path.
        loader_backend (str): "auto" preserves the historical official-root
            preference; "imagefolder" forces train/val class-directory loading;
            "torchvision" forces torchvision.datasets.ImageNet.

    Returns:
        tuple: (train_dataset, valid_dataset, test_dataset)
    """
    if flatten and not allow_flatten_materialize:
        raise ValueError(
            "ImageNet with flatten=True materializes up to 50k 224x224x3 "
            "samples and can consume tens of GB. Use flatten=False for "
            "AlexNet/pretrained runs, or set "
            "data.dataset_params.imagenet.allow_flatten_materialize=true "
            "for an explicit debugging-only opt-in."
        )

    loader_backend = _normalized_imagenet_loader_backend(loader_backend)
    imagenet_dir, train_root, val_root = _resolve_imagenet_data_location(
        task_data_path,
        loader_backend=loader_backend,
    )

    train_transform, eval_transform = _build_imagenet_transforms(
        normalize=normalize,
        transform_preset=transform_preset,
    )

    try:
        train_dataset, train_eval_dataset, test_dataset = (
            _load_imagenet_dataset_triplet(
                imagenet_dir=imagenet_dir,
                train_root=train_root,
                val_root=val_root,
                train_transform=train_transform,
                eval_transform=eval_transform,
                loader_backend=loader_backend,
            )
        )
    except Exception as e:
        raise RuntimeError(
            "Could not load ImageNet. Set data.base_dir or IMAGENET_LOCAL_DIR "
            "to a valid ImageNet root."
        ) from e

    return _prepare_imagenet_dataset_triplet(
        train_dataset=train_dataset,
        train_eval_dataset=train_eval_dataset,
        test_dataset=test_dataset,
        train_valid_split=train_valid_split,
        train_samples_per_class=train_samples_per_class,
        val_samples_per_class=val_samples_per_class,
        split_seed=split_seed,
        flatten=flatten,
    )


__all__ = [
    "IMAGENET_FLATTEN_MATERIALIZATION_LIMIT",
    "_apply_imagenet_local_override",
    "_build_imagenet_transforms",
    "_class_subset_indices",
    "_copy_dataset_with_transform",
    "_is_official_imagenet_root",
    "_load_imagenet_dataset_triplet",
    "_materialize_flattened_tensor_dataset",
    "_normalized_imagenet_loader_backend",
    "_normalized_imagenet_preset_name",
    "_prepare_imagenet_dataset_triplet",
    "_resolve_imagefolder_roots",
    "_resolve_imagenet_data_location",
    "_split_train_valid_pair",
    "_subset_by_indices",
    "load_imagenet_as_datasets",
]
