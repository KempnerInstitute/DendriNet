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
    # Dataset classes
    LineDataset,
    MultiTaskLineDataset,
    MultiXORDataset,
    NoisyLineDataset,
    OrientationBarsDataset,
    # Network models
    OrthoNet,
    OrthoNetDataset,
    OrthoNetRepeat,
    PoissonGeneratorDataset,
    RandomFlipMNISTDataset,
    SparseMNISTDataset,
    # Utility functions
    applyRandomFlip,
    # Data loading functions
    get_data_directory,
    get_unified_datasets,
    load_cifar10_as_datasets,
    load_mnist_as_datasets,
    load_MNIST_modulo10,
    load_poisson_generator_datasets,
)

__all__ = [
    # Dataset classes
    "LineDataset",
    "CenterSurroundDataset",
    "OrientationBarsDataset",
    "MultiXORDataset",
    "InfoShuntingDataset",
    "NoisyLineDataset",
    "SparseMNISTDataset",
    "ContextualFigureGroundMNIST",
    "MultiTaskLineDataset",
    "CompositionalMNIST",
    "RandomFlipMNISTDataset",
    "PoissonGeneratorDataset",
    "OrthoNetDataset",
    "SyntheticCorrelatedDataset",
    # Network models
    "OrthoNet",
    "OrthoNetRepeat",
    # Data loading functions
    "get_data_directory",
    "load_mnist_as_datasets",
    "load_MNIST_modulo10",
    "load_cifar10_as_datasets",
    "load_poisson_generator_datasets",
    "get_unified_datasets",
    "load_synthetic_correlated",
    # Utility functions
    "applyRandomFlip"
]
