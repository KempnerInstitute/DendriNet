"""Builders for memory-oriented sequence datasets."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    deferred_split,
    test_size,
)
from dendritic_modeling.datasets.sequence.memory_tasks import (
    AddingProblemDataset,
    CopyTaskDataset,
    SequentialMNISTDataset,
    VariableDelayDMSDataset,
)


def build_sequential_mnist(
    _dataset_name: str,
    data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    mode = kwargs.get("mode", "pixel")
    normalize = kwargs.get("normalize", True)
    return deferred_split(
        SequentialMNISTDataset(
            mode=mode,
            train=True,
            data_path=data_path,
            normalize=normalize,
        ),
        SequentialMNISTDataset(
            mode=mode,
            train=False,
            data_path=data_path,
            normalize=normalize,
        ),
    )


def build_copy_task(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "seq_len": kwargs.get("seq_len", 10),
        "delay": kwargs.get("delay", 50),
        "n_symbols": kwargs.get("n_symbols", 8),
    }
    return deferred_split(
        CopyTaskDataset(n_samples=n_train, **common),
        CopyTaskDataset(n_samples=test_size(n_train, 1000), **common),
    )


def build_adding_problem(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {"seq_len": kwargs.get("seq_len", 200)}
    return deferred_split(
        AddingProblemDataset(n_samples=n_train, **common),
        AddingProblemDataset(n_samples=test_size(n_train, 1000), **common),
    )


def build_variable_delay_dms(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "stimulus_dim": kwargs.get("stimulus_dim", 16),
        "stimulus_duration": kwargs.get("stimulus_duration", 10),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "probe_duration": kwargs.get("probe_duration", 10),
    }
    return deferred_split(
        VariableDelayDMSDataset(n_samples=n_train, **common),
        VariableDelayDMSDataset(n_samples=test_size(n_train, 1000), **common),
    )


MEMORY_SEQUENCE_BUILDERS = {
    "adding_problem": build_adding_problem,
    "copy_task": build_copy_task,
    "sequential_mnist": build_sequential_mnist,
    "variable_delay_dms": build_variable_delay_dms,
}


__all__ = [
    "MEMORY_SEQUENCE_BUILDERS",
    "build_adding_problem",
    "build_copy_task",
    "build_sequential_mnist",
    "build_variable_delay_dms",
]
