"""ImageNet protocol partitions without constructing a model or replacement workflow.

These subset loaders preserve the full-dataset loader's split and transform
semantics while avoiding scans of unused train/test trees. They find the data
as training does: ``data.base_dir``, else ``$DENDRITIC_DATA_DIR``, else the
cache (:func:`~dendritic_modeling.datasets.paths.resolve_data_root`).
"""

from torch.utils.data import Dataset

from dendritic_modeling.config.conversion import (
    dataset_specific_params as _get_dataset_specific_params,
)
from dendritic_modeling.datasets.parameters import effective_processing

from .imagenet_loading import (
    _copy_dataset_with_transform,
    _imagefolder_with_index_cache,
)
from .imagenet_paths import (
    _normalized_imagenet_loader_backend,
    _resolve_imagenet_data_location,
)
from .imagenet_splits import _split_train_valid_pair
from .imagenet_transforms import _build_imagenet_transforms
from .paths import resolve_data_root


def build_protocol_test_dataset(config) -> tuple[Dataset, dict]:
    """Build the runs' own test split (official ImageNet validation set).

    Mirrors ``setup_environment`` -> ``get_unified_datasets`` ->
    ``load_imagenet_as_datasets`` for the TEST member only: the test split
    is the validation root with the eval transform, is untouched by
    ``split_seed`` (which only partitions train/valid), and here carries no
    per-class subsetting.  Loading only the test member avoids the 1.28M
    file train-tree scan that the full triplet loader performs.
    """

    data_config = config.data
    params = _get_dataset_specific_params(data_config)
    if str(data_config.dataset_name).lower() != "imagenet":
        raise ValueError(
            f"protocol config uses dataset {data_config.dataset_name!r}, "
            "expected 'imagenet'"
        )
    processing = effective_processing("imagenet", data_config.processing, params)
    if processing["flatten"]:
        raise ValueError("protocol config unexpectedly sets flatten=true")
    if params.get("val_samples_per_class") is not None:
        raise ValueError(
            "protocol config subsets the test split "
            "(val_samples_per_class is set); refusing to guess"
        )
    normalize = processing["normalize"]
    transform_preset = params.get("transform_preset", "legacy_resize")
    loader_backend = _normalized_imagenet_loader_backend(
        params.get("loader_backend", "auto")
    )
    imagenet_dir, train_root, val_root = _resolve_imagenet_data_location(
        resolve_data_root(data_config.base_dir),
        loader_backend=loader_backend,
    )
    _, eval_transform = _build_imagenet_transforms(
        normalize=normalize,
        transform_preset=transform_preset,
    )
    if train_root is not None:
        test_dataset: Dataset = _imagefolder_with_index_cache(val_root, eval_transform)
        backend_used = "imagefolder"
    else:
        from torchvision import datasets as tv_datasets

        test_dataset = tv_datasets.ImageNet(
            root=imagenet_dir, split="val", transform=eval_transform
        )
        backend_used = "torchvision"
    protocol = {
        "dataset": "imagenet",
        "split": "test == official ILSVRC2012 validation set",
        "imagenet_dir": str(imagenet_dir),
        "loader_backend": backend_used,
        "transform_preset": str(transform_preset),
        "eval_transform": (
            "Resize(256) -> CenterCrop(224) -> ToTensor -> "
            "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
            if transform_preset == "pretrained" and normalize
            else repr(eval_transform)
        ),
        "normalize": normalize,
        "n_samples": len(test_dataset),
    }
    return test_dataset, protocol


def build_protocol_train_partition(config) -> tuple[Dataset, Dataset, dict]:
    """Build the runs' own train/valid partition of the ImageNet train set.

    Mirrors ``setup_environment`` -> ``get_unified_datasets`` ->
    ``load_imagenet_as_datasets`` for the TRAIN members: the train tree with
    the preset's TRAIN transform, an eval-transform view of the same samples,
    and the runs' 0.8/0.2 partition through the exact
    ``_split_train_valid_pair(split_generator(split_seed))`` code path, so
    the returned train subset is sample-for-sample the partition the
    dendritic runs trained on and the returned valid subset is their
    model-selection split (eval transform, no augmentation).
    """

    data_config = config.data
    params = _get_dataset_specific_params(data_config)
    if str(data_config.dataset_name).lower() != "imagenet":
        raise ValueError(
            f"protocol config uses dataset {data_config.dataset_name!r}, "
            "expected 'imagenet'"
        )
    processing = effective_processing("imagenet", data_config.processing, params)
    if processing["flatten"]:
        raise ValueError("protocol config unexpectedly sets flatten=true")
    if params.get("train_samples_per_class") is not None:
        raise ValueError(
            "protocol config subsets the train split "
            "(train_samples_per_class is set); refusing to guess"
        )
    normalize = processing["normalize"]
    transform_preset = params.get("transform_preset", "legacy_resize")
    loader_backend = _normalized_imagenet_loader_backend(
        params.get("loader_backend", "auto")
    )
    imagenet_dir, train_root, _ = _resolve_imagenet_data_location(
        resolve_data_root(data_config.base_dir),
        loader_backend=loader_backend,
    )
    train_transform, eval_transform = _build_imagenet_transforms(
        normalize=normalize,
        transform_preset=transform_preset,
    )
    if train_root is not None:
        train_dataset: Dataset = _imagefolder_with_index_cache(
            train_root, train_transform
        )
        backend_used = "imagefolder"
    else:
        from torchvision import datasets as tv_datasets

        train_dataset = tv_datasets.ImageNet(
            root=imagenet_dir, split="train", transform=train_transform
        )
        backend_used = "torchvision"
    train_eval_dataset = _copy_dataset_with_transform(train_dataset, eval_transform)
    raw_split_seed = getattr(config.experiment, "split_seed", None)
    if raw_split_seed is None:
        raw_split_seed = getattr(config.experiment, "seed", 0) or 0
    split_seed = int(raw_split_seed)
    train_valid_split = float(config.experiment.train_valid_split)
    train_subset, valid_subset = _split_train_valid_pair(
        train_dataset,
        train_eval_dataset,
        train_valid_split=train_valid_split,
        split_seed=split_seed,
    )
    info = {
        "dataset": "imagenet",
        "split": "the runs' own train partition of the ImageNet train set",
        "imagenet_dir": str(imagenet_dir),
        "loader_backend": backend_used,
        "transform_preset": str(transform_preset),
        "train_transform": (
            "RandomResizedCrop(224) -> RandomHorizontalFlip -> ToTensor -> "
            "Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])"
            if transform_preset == "pretrained" and normalize
            else repr(train_transform)
        ),
        "normalize": normalize,
        "train_valid_split": train_valid_split,
        "split_seed": split_seed,
        "split_code_path": "dendritic_modeling.datasets.imagenet_splits."
        "_split_train_valid_pair(split_generator(split_seed))",
        "n_train": len(train_subset),
        "n_valid": len(valid_subset),
    }
    return train_subset, valid_subset, info
