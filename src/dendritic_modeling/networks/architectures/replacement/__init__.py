"""Input and output adapters for dendritic cores placed inside other networks.

A dendritic core placed at a boundary of another network uses the contract
``input_adapter -> dendritic core -> output_adapter``. PopulationNetwork, the
spatial dendritic convolution and the dendritic AlexNet use these adapters.
"""

from dendritic_modeling.networks.architectures.replacement.adapters import (
    NonNegativeInputAdapter,
    PopulationToFeatureReduction,
    PositiveFeatureScale,
    PositiveThresholdReLU,
    SpatialNonNegativeInputAdapter,
    make_nonmixing_output_adapter,
    transformed_feature_dim,
    validate_input_transform,
)

__all__ = [
    "NonNegativeInputAdapter",
    "PopulationToFeatureReduction",
    "PositiveFeatureScale",
    "PositiveThresholdReLU",
    "SpatialNonNegativeInputAdapter",
    "make_nonmixing_output_adapter",
    "transformed_feature_dim",
    "validate_input_transform",
]
