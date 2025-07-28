"""
Datasets module for dendritic modeling.

This module contains various synthetic and real dataset classes and utilities.
"""


from .standard_datasets import (
    get_data_directory,
    load_mnist_as_datasets,
    load_mnist_modulo10,
    load_cifar10_as_datasets,
    load_imagenet_as_datasets,
)

__all__ = [
    # Dataset classes (alphabetically sorted)
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
    # Utility functions (alphabetically sorted)
    "apply_random_flip",
    "get_data_directory",
    "get_unified_datasets",
    "load_cifar10_as_datasets",
    "load_imagenet_as_datasets",
    "load_mnist_as_datasets",
    "load_mnist_modulo10",
    "load_poisson_generator_datasets",

]
