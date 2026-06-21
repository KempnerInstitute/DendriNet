"""ImageNet dataset construction helpers."""

import logging
from copy import copy

from torch.utils.data import Dataset
from torchvision import datasets

from .imagenet_paths import _normalized_imagenet_loader_backend

_logger = logging.getLogger(__name__)


def _copy_dataset_with_transform(dataset: Dataset, transform) -> Dataset:
    """Return a shallow dataset view with the same samples and a new transform."""
    dataset_view = copy(dataset)
    dataset_view.transform = transform

    transforms_obj = getattr(dataset_view, "transforms", None)
    if transforms_obj is not None and hasattr(transforms_obj, "transform"):
        dataset_view.transforms = copy(transforms_obj)
        dataset_view.transforms.transform = transform

    return dataset_view


def _load_imagefolder_dataset_triplet(
    *,
    train_root: str,
    val_root: str,
    train_transform,
    eval_transform,
) -> tuple[Dataset, Dataset, Dataset]:
    if not train_root or not val_root:
        raise RuntimeError(
            "ImageFolder ImageNet loading requires both train and val "
            "class-directory roots."
        )

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


def _load_torchvision_imagenet_dataset_triplet(
    *,
    imagenet_dir: str,
    train_transform,
    eval_transform,
) -> tuple[Dataset, Dataset, Dataset]:
    train_dataset = datasets.ImageNet(
        root=imagenet_dir, split="train", transform=train_transform
    )
    return (
        train_dataset,
        _copy_dataset_with_transform(train_dataset, eval_transform),
        datasets.ImageNet(root=imagenet_dir, split="val", transform=eval_transform),
    )


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
    if backend == "imagefolder" and (not train_root or not val_root):
        raise RuntimeError(
            "ImageNet loader_backend='imagefolder' requires train/val "
            "class-directory roots."
        )
    if train_root is not None:
        return _load_imagefolder_dataset_triplet(
            train_root=train_root,
            val_root=val_root,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
    return _load_torchvision_imagenet_dataset_triplet(
        imagenet_dir=imagenet_dir,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


__all__ = [
    "_copy_dataset_with_transform",
    "_load_imagenet_dataset_triplet",
]
