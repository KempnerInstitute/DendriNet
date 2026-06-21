"""Shared helpers for sequence dataset builders."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SequenceDatasetBuild:
    """Datasets returned by a sequence dataset builder.

    Builders either return an explicit train/valid/test triplet or a full
    training set plus a test set for the public factory to split.
    """

    train_full: Any | None = None
    test_ds: Any | None = None
    split_datasets: tuple[Any, Any, Any] | None = None


SequenceDatasetBuilder = Callable[
    [str, str | None, dict[str, Any]], SequenceDatasetBuild
]


def test_size(n_train: int, minimum: int) -> int:
    return max(n_train // 5, minimum)


def deferred_split(train_full: Any, test_ds: Any) -> SequenceDatasetBuild:
    return SequenceDatasetBuild(train_full=train_full, test_ds=test_ds)


def explicit_split(
    train_ds: Any,
    valid_ds: Any,
    test_ds: Any,
) -> SequenceDatasetBuild:
    return SequenceDatasetBuild(split_datasets=(train_ds, valid_ds, test_ds))


def positive_input_encoding(dataset_name: str, kwargs: dict[str, Any]) -> bool:
    return dataset_name.endswith("_positive") or kwargs.get(
        "positive_input_encoding", False
    )


def has_explicit_splits(kwargs: dict[str, Any]) -> bool:
    return any(
        isinstance(kwargs.get(split_name), dict)
        for split_name in ("train", "valid", "test")
    )


def build_explicit_or_deferred_split(
    *,
    dataset_cls: Callable[..., Any],
    kwargs: dict[str, Any],
    common_defaults: dict[str, Any],
    n_train: int,
    min_test_size: int,
) -> SequenceDatasetBuild:
    if has_explicit_splits(kwargs):

        def build_split(split_name: str, default_n_samples: int) -> Any:
            split_kwargs = dict(kwargs.get(split_name, {}))
            split_params = {
                **common_defaults,
                **{
                    key: value
                    for key, value in split_kwargs.items()
                    if key != "n_samples"
                },
            }
            split_n = split_kwargs.get("n_samples", default_n_samples)
            return dataset_cls(n_samples=split_n, **split_params)

        return explicit_split(
            build_split("train", n_train),
            build_split("valid", test_size(n_train, min_test_size)),
            build_split("test", test_size(n_train, min_test_size)),
        )

    return deferred_split(
        dataset_cls(n_samples=n_train, **common_defaults),
        dataset_cls(n_samples=test_size(n_train, min_test_size), **common_defaults),
    )


__all__ = [
    "SequenceDatasetBuild",
    "SequenceDatasetBuilder",
    "build_explicit_or_deferred_split",
    "deferred_split",
    "explicit_split",
    "has_explicit_splits",
    "positive_input_encoding",
    "test_size",
]
