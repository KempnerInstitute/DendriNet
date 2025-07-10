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
    apply_random_flip,
    # Data loading functions
    get_data_directory,
    get_unified_datasets,
    load_cifar10_as_datasets,
    load_mnist_as_datasets,
    load_mnist_modulo10,
    load_poisson_generator_datasets,
)

__all__ = [
    # Utility functions
    "apply_random_flip"
    "CenterSurroundDataset"
    "CompositionalMNIST"
    "ContextualFigureGroundMNIST"
    # Data loading functions
    "get_data_directory"
    "get_unified_datasets"
    "InfoShuntingDataset"
    # Dataset classes
    "LineDataset"
    "load_cifar10_as_datasets"
    "load_mnist_as_datasets"
    "load_mnist_modulo10"
    "load_poisson_generator_datasets"
    "load_synthetic_correlated"
    "MultiTaskLineDataset"
    "MultiXORDataset"
    "NoisyLineDataset"
    "OrientationBarsDataset"
    # Network models
    "OrthoNet"
    "OrthoNetDataset"
    "OrthoNetRepeat"
    "PoissonGeneratorDataset"
    "RandomFlipMNISTDataset"
    "SparseMNISTDataset"
    "SyntheticCorrelatedDataset"
]
