"""Shared replacement-cell components.

Replacement cells define the reusable computation that sits between a source
model and a target contract:

``input_adapter -> dendritic core -> output_adapter``.

Domain-specific code is still responsible for placement.  Vision models split
backbones into prefix/core/suffix segments; transformer models patch modules
inside decoder blocks.
"""

from dendritic_modeling.networks.architectures.replacement.adapters import (
    NonNegativeInputAdapter,
    PositiveFeatureScale,
    SpatialNonNegativeInputAdapter,
    transformed_feature_dim,
    validate_input_transform,
)
from dendritic_modeling.networks.architectures.replacement.cells import (
    LayerwiseTokenReplacementStack,
    TokenReplacementCell,
    make_linear_output_projection,
)

__all__ = [
    "LayerwiseTokenReplacementStack",
    "NonNegativeInputAdapter",
    "PositiveFeatureScale",
    "SpatialNonNegativeInputAdapter",
    "TokenReplacementCell",
    "make_linear_output_projection",
    "transformed_feature_dim",
    "validate_input_transform",
]
