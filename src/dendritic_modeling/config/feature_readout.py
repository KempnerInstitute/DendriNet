"""Explicit normalization and affine readout of fixed feature maps."""

import math
from dataclasses import dataclass, field

from dendritic_modeling.config.affine_mlp import AffineMLPConfig
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.keys import build_nested_sections


@dataclass
class FeatureReadoutConfig(BaseConfig):
    """A conventional control, not an E/I conductance neuron.

    ``input_shape`` excludes the batch and starts with channels. The readout
    reuses AffineMLPConfig rather than introducing another MLP implementation.
    """

    input_shape: list[int] = field(default_factory=list)
    readout: AffineMLPConfig = field(default_factory=AffineMLPConfig)
    normalization: str = "none"
    groups: int | None = None
    divisor_offset: float | None = None
    normalization_epsilon: float = 1e-5
    normalization_affine: bool = True

    def __post_init__(self):
        build_nested_sections(self, section="feature_readout")

    def validate(self) -> None:
        if not self.input_shape or any(
            type(n) is not int or n < 1 for n in self.input_shape
        ):
            raise ValueError("input_shape must contain positive integers")
        if not isinstance(self.readout, AffineMLPConfig):
            raise TypeError("readout must be an AffineMLPConfig")
        self.readout.validate()
        if self.readout.mode != "dense":
            raise ValueError("feature_readout currently supports dense readouts only")
        if self.readout.dimensions[0] != math.prod(self.input_shape):
            raise ValueError("readout input dimension must equal product(input_shape)")
        if self.normalization not in {"none", "group_mean", "layer_norm", "group_norm"}:
            raise ValueError("unknown feature normalization")
        if self.normalization in {"group_mean", "group_norm"}:
            if type(self.groups) is not int or self.groups < 1:
                raise ValueError("group normalization requires positive integer groups")
            if self.input_shape[0] % self.groups:
                raise ValueError("groups must divide the channel count")
        elif self.groups is not None:
            raise ValueError("groups only applies to group_mean or group_norm")
        if self.normalization == "group_mean":
            if (
                isinstance(self.divisor_offset, bool)
                or not isinstance(self.divisor_offset, (int, float))
                or not math.isfinite(self.divisor_offset)
                or self.divisor_offset <= 0
            ):
                raise ValueError("group_mean requires a finite positive divisor_offset")
        elif self.divisor_offset is not None:
            raise ValueError("divisor_offset only applies to group_mean")
        if type(self.normalization_affine) is not bool:
            raise ValueError("normalization_affine must be a bool")
        if (
            isinstance(self.normalization_epsilon, bool)
            or not isinstance(self.normalization_epsilon, (int, float))
            or not math.isfinite(self.normalization_epsilon)
            or self.normalization_epsilon <= 0
        ):
            raise ValueError("normalization_epsilon must be finite and positive")
        if self.normalization not in {"layer_norm", "group_norm"} and (
            self.normalization_epsilon != 1e-5 or not self.normalization_affine
        ):
            raise ValueError("epsilon and affine options only apply to Layer/GroupNorm")
