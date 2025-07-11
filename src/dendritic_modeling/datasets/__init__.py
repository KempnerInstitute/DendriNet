"""
Datasets module for dendritic modeling.

This module contains various synthetic and real dataset classes and utilities.
"""

from dendritic_modeling.datasets.synthetic_correlated import (
    SyntheticCorrelatedDataset,
    load_synthetic_correlated,
)
from dendritic_modeling.datasets.synthetic_datasets import (
    CenterSurroundDataset,
    CompositionalMNIST,
    ContextualFigureGroundMNIST,
    InfoShuntingDataset,
    LineDataset,
    MultiTaskLineDataset,
    MultiXORDataset,
    NoisyLineDataset,
    OrientationBarsDataset,
    OrthoNet,
    OrthoNetDataset,
    OrthoNetRepeat,
    PoissonGeneratorDataset,
    RandomFlipMNISTDataset,
    SparseMNISTDataset,
    apply_random_flip,
    get_data_directory,
    get_unified_datasets,
    load_cifar10_as_datasets,
    load_mnist_as_datasets,
    load_mnist_modulo10,
    load_poisson_generator_datasets,
)

__all__ = [
    "CenterSurroundDataset",
    "CompositionalMNIST",
    "ContextualFigureGroundMNIST",
    "InfoShuntingDataset",
    "LineDataset",
    "MultiTaskLineDataset",
    "MultiXORDataset",
    "NoisyLineDataset",
    "OrientationBarsDataset",
    "OrthoNet",
    "OrthoNetDataset",
    "OrthoNetRepeat",
    "PoissonGeneratorDataset",
    "RandomFlipMNISTDataset",
    "SparseMNISTDataset",
    "SyntheticCorrelatedDataset",
    "apply_random_flip",
    "get_data_directory",
    "get_unified_datasets",
    "load_cifar10_as_datasets",
    "load_mnist_as_datasets",
    "load_mnist_modulo10",
    "load_poisson_generator_datasets",
    "load_synthetic_correlated",
]
