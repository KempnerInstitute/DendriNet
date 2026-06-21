"""Factory for sequence dataset train/valid/test triplets."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import random_split

from dendritic_modeling.datasets.registry import build_registered_dataset
from dendritic_modeling.datasets.sequence.builders import (
    get_registered_sequence_dataset_names,
    get_sequence_dataset_builder,
    has_sequence_dataset_builder,
    register_sequence_dataset_builder,
    unregister_sequence_dataset_builder,
)


def _split_train_valid(
    train_full: Any,
    test_ds: Any,
    *,
    train_valid_split: float,
    split_seed: int,
) -> tuple[Any, Any, Any]:
    n_train = int(len(train_full) * train_valid_split)
    n_valid = len(train_full) - n_train
    if n_valid > 0:
        train_ds, valid_ds = random_split(
            train_full,
            [n_train, n_valid],
            generator=torch.Generator().manual_seed(split_seed),
        )
    else:
        train_ds = train_full
        valid_ds = test_ds
    return train_ds, valid_ds, test_ds


def _registered_sequence_dataset_task_config(
    *,
    dataset_name: str,
    data_path: str | None,
    train_valid_split: float,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build the config payload used by external sequence dataset registries."""
    return {
        "dataset": dataset_name,
        "data_path": data_path,
        "train_valid_split": train_valid_split,
        "parameters": dict(kwargs),
    }


def get_sequence_datasets(
    dataset_name: str,
    data_path: str | None = None,
    train_valid_split: float = 0.9,
    **kwargs: Any,
) -> tuple[Any, Any, Any]:
    """Factory function for sequence datasets.

    Args:
        dataset_name: Name of the dataset.
        data_path: Path for datasets that need file storage.
        train_valid_split: Fraction of training data to use for training.
        **kwargs: Dataset-specific parameters.

    Returns:
        Tuple of (train_ds, valid_ds, test_ds).
    """
    registered = build_registered_dataset(
        dataset_name,
        _registered_sequence_dataset_task_config(
            dataset_name=dataset_name,
            data_path=data_path,
            train_valid_split=train_valid_split,
            kwargs=kwargs,
        ),
    )
    if registered is not None:
        return registered

    builder = get_sequence_dataset_builder(dataset_name)

    build = builder(dataset_name, data_path, kwargs)
    if build.split_datasets is not None:
        return build.split_datasets
    if build.train_full is None or build.test_ds is None:
        raise RuntimeError(
            f"Sequence dataset builder returned no datasets: {dataset_name}"
        )

    split_seed = int(kwargs.get("split_seed", kwargs.get("seed", 42)) or 0)
    return _split_train_valid(
        build.train_full,
        build.test_ds,
        train_valid_split=train_valid_split,
        split_seed=split_seed,
    )


__all__ = [
    "get_registered_sequence_dataset_names",
    "get_sequence_dataset_builder",
    "get_sequence_datasets",
    "has_sequence_dataset_builder",
    "register_sequence_dataset_builder",
    "unregister_sequence_dataset_builder",
]
