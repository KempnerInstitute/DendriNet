"""
Datasets module for dendritic modeling.

This module contains various synthetic and real dataset classes and utilities.
"""

from dendritic_modeling.datasets.synthetic_datasets import (
    # Dataset classes
    LineDataset,
    CenterSurroundDataset,
    OrientationBarsDataset,
    MultiXORDataset,
    InfoShuntingDataset,
    NoisyLineDataset,
    SparseMNISTDataset,
    ContextualFigureGroundMNIST,
    MultiTaskLineDataset,
    CompositionalMNIST,
    RandomFlipMNISTDataset,
    PoissonGeneratorDataset,
    OrthoNetDataset,
    # Network models
    OrthoNet,
    OrthoNetRepeat,
    # Data loading functions
    get_data_directory,
    load_mnist_as_datasets,
    load_MNIST_modulo10,
    load_cifar10_as_datasets,
    load_poisson_generator_datasets,
    get_unified_datasets,
    # Utility functions
    applyRandomFlip
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
    # Utility functions
    "applyRandomFlip"
] 