"""Dataset-loader registry for project extensions.

Built-in datasets continue through the established loaders in
``standard_datasets`` and ``sequence_datasets``.  This registry gives new
datasets a config-only entry point without editing those long dispatch chains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from dendritic_modeling.plugins import load_plugins

DatasetTriplet = tuple[Dataset, Dataset, Dataset]
DatasetBuilder = Callable[[Any], DatasetTriplet]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a registered dataset loader."""

    name: str
    builder: DatasetBuilder
    description: str = ""
    sequence: bool = False
    # Training strategy the dataset's training split is built for, if any.
    required_strategy: str | None = None
    # data.processing options (flatten, normalize) the builder applies; None
    # leaves them unchecked.
    processing: frozenset[str] | None = None


_DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def _builtin_dataset_names() -> set[str]:
    """Names the built-in loaders handle (imported lazily: they import torch data)."""
    from dendritic_modeling.datasets import standard_datasets
    from dendritic_modeling.datasets.sequence.builders.registry import (
        SEQUENCE_DATASET_BUILDERS,
    )

    return {
        *standard_datasets._STANDARD_VISION_DATASET_LOADERS,
        *standard_datasets._SPECIAL_DATASET_LOADERS,
        *standard_datasets._THEORETICAL_SYNTHETIC_DATASET_NAMES,
        *SEQUENCE_DATASET_BUILDERS,
    }


def register_dataset(
    name: str,
    builder: DatasetBuilder,
    *,
    aliases: tuple[str, ...] | list[str] = (),
    description: str = "",
    sequence: bool = False,
    required_strategy: str | None = None,
    processing: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    allow_override: bool = False,
) -> None:
    """Register a dataset builder.

    Builders receive the original task/data config object and must return
    ``(train_ds, valid_ds, test_ds)``.
    """
    names = [name, *aliases]
    if not names or any(not str(item).strip() for item in names):
        raise ValueError("dataset name and aliases must be non-empty")
    builtin = _builtin_dataset_names()
    for raw_name in names:
        key = str(raw_name).lower()
        if key in builtin and not allow_override:
            raise ValueError(
                f"Dataset '{key}' is a built-in dataset; pass allow_override=True "
                "to replace its loader"
            )
        if key in _DATASET_REGISTRY and not allow_override:
            raise ValueError(f"Dataset '{key}' is already registered")
        _DATASET_REGISTRY[key] = DatasetSpec(
            name=key,
            builder=builder,
            description=description,
            sequence=sequence,
            required_strategy=required_strategy,
            processing=None if processing is None else frozenset(processing),
        )


def unregister_dataset(name: str) -> None:
    """Remove a registered dataset if present."""
    _DATASET_REGISTRY.pop(str(name).lower(), None)


def get_dataset_spec(name: str) -> DatasetSpec | None:
    """Return a registered dataset spec, if any."""
    key = str(name).lower()
    if key not in _DATASET_REGISTRY:
        load_plugins()
    return _DATASET_REGISTRY.get(key)


def reject_unsupported_training_strategy(dataset_name: str, strategy: str) -> None:
    """Raise when a registered dataset requires a different training strategy."""
    spec = get_dataset_spec(dataset_name)
    if spec is None or spec.required_strategy is None:
        return
    if str(strategy).strip().lower() != spec.required_strategy:
        raise ValueError(
            f"dataset {spec.name!r} must be trained with "
            f"training.main.strategy: {spec.required_strategy}, got {strategy!r}"
        )


def get_registered_dataset_names(*, sequence: bool | None = None) -> list[str]:
    """Return registered dataset names, optionally filtered by sequence flag."""
    if sequence is None:
        return sorted(_DATASET_REGISTRY)
    return sorted(
        name for name, spec in _DATASET_REGISTRY.items() if spec.sequence == sequence
    )


def build_registered_dataset(name: str, task_cfg: Any) -> DatasetTriplet | None:
    """Build a registered dataset, or ``None`` if the name is unregistered."""
    spec = get_dataset_spec(name)
    if spec is None:
        return None
    return spec.builder(task_cfg)


def _build_contextual_remapping(task_cfg: Any) -> DatasetTriplet:
    # Imported lazily: the module loads torchvision only when the task is used.
    from .contextual_remapping import build_contextual_remapping_datasets

    return build_contextual_remapping_datasets(task_cfg)


# Its training split serves presentations through ``.stream``; indexing it
# returns uncorrupted context-0 items meant only for shape inference.
register_dataset(
    "contextual_remapping",
    _build_contextual_remapping,
    description=(
        "EMNIST Balanced with 32 cyclic context label shifts and per-presentation "
        "gain/noise corruption"
    ),
    required_strategy="presentation_stream",
    processing=frozenset(),
)


__all__ = [
    "DatasetBuilder",
    "DatasetSpec",
    "DatasetTriplet",
    "build_registered_dataset",
    "get_dataset_spec",
    "get_registered_dataset_names",
    "register_dataset",
    "reject_unsupported_training_strategy",
    "unregister_dataset",
]
