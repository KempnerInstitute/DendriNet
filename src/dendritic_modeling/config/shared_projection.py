"""Configuration for sparse projections reused by native shunting banks."""

import math
from dataclasses import dataclass, field

from dendritic_modeling.config.base import BaseConfig


@dataclass
class SharedProjectionShuntingConfig(BaseConfig):
    """Explicit support and initial transition scales; no inferred budget.

    Inputs must be finite and nonnegative. Supports may overlap; disjointness
    is a separate assumption of some approximation results. All groups have
    the same number of contacts and initially use the same positive nodes.
    Only ordinary autograd/backpropagation is supported by this composition.
    """

    input_dim: int = 1
    output_dim: int = 1
    supports: list[list[int]] = field(default_factory=lambda: [[0]])
    nodes: list[float] = field(default_factory=lambda: [1.0])
    epsilon: float = 1e-8
    dtype: str = "float32"
    train_projection: bool = True
    train_conductance: bool = True
    train_output: bool = True

    def validate(self) -> None:
        for name in ("input_dim", "output_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.supports or not self.supports[0]:
            raise ValueError("supports must contain nonempty rows")
        size = len(self.supports[0])
        for row in self.supports:
            if len(row) != size or any(
                type(i) is not int or not 0 <= i < self.input_dim for i in row
            ):
                raise ValueError("supports must have equal length and valid indices")
            if len(set(row)) != size:
                raise ValueError("support indices must be unique within each row")
        if not self.nodes or any(
            isinstance(z, bool) or not math.isfinite(z) or z <= 0 for z in self.nodes
        ):
            raise ValueError("nodes must be finite and positive")
        if (
            isinstance(self.epsilon, bool)
            or not math.isfinite(self.epsilon)
            or self.epsilon < 0
        ):
            raise ValueError("epsilon must be finite and nonnegative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        for name in ("train_projection", "train_conductance", "train_output"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
