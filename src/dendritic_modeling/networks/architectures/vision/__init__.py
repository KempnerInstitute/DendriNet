"""Config-driven dendritic vision architectures."""

from dendritic_modeling.networks.architectures.vision.dendritic_alexnet import (
    DendriticAlexNet,
)
from dendritic_modeling.networks.architectures.vision.feature_map_core import (
    FlattenedSpatialCore,
)

__all__ = ["DendriticAlexNet", "FlattenedSpatialCore"]
