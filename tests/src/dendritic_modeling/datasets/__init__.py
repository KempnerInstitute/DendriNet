"""
Datasets module for dendritic modeling.

This module contains various synthetic and real dataset classes and utilities.
"""

from .registry import (
    DatasetSpec,
    get_dataset_spec,
    get_registered_dataset_names,
    register_dataset,
    unregister_dataset,
)
from .sequence_datasets import (
    AddingProblemDataset,
    CopyTaskDataset,
    GainModulatedContextualComparisonDataset,
    LorenzSequenceDataset,
    MultiFrequencyDataset,
    OculomotorDelayedResponseDataset,
    RomoDelayComparisonDataset,
    SequentialMNISTDataset,
    StringerV1Dataset,
    SwitchingLDSDataset,
    VariableDelayDMSDataset,
    get_sequence_datasets,
)
from .standard_datasets import (
    get_data_directory,
    get_unified_datasets,
    load_cifar10_as_datasets,
    load_cifar10_modulo10,
    load_double_cifar10_contextual_mod10,
    load_double_mnist_contextual_mod10,
    load_imagenet_as_datasets,
    load_mnist_as_datasets,
    load_mnist_modulo10,
)

__all__ = [
    "AddingProblemDataset",
    "CopyTaskDataset",
    "DatasetSpec",
    "GainModulatedContextualComparisonDataset",
    "LorenzSequenceDataset",
    "MultiFrequencyDataset",
    "OculomotorDelayedResponseDataset",
    "RomoDelayComparisonDataset",
    "SequentialMNISTDataset",
    "StringerV1Dataset",
    "SwitchingLDSDataset",
    "VariableDelayDMSDataset",
    "get_data_directory",
    "get_dataset_spec",
    "get_registered_dataset_names",
    "get_sequence_datasets",
    "get_unified_datasets",
    "load_cifar10_as_datasets",
    "load_cifar10_modulo10",
    "load_double_cifar10_contextual_mod10",
    "load_double_mnist_contextual_mod10",
    "load_imagenet_as_datasets",
    "load_mnist_as_datasets",
    "load_mnist_modulo10",
    "register_dataset",
    "unregister_dataset",
]
