"""Configuration for ordinary dense and sparse affine MLP controls."""

import math
from dataclasses import dataclass, field

from dendritic_modeling.config.base import BaseConfig


@dataclass
class AffineMLPConfig(BaseConfig):
    """Explicit dimensions include input and output; no budget is inferred.

    Dense weights use Kaiming normal initialization after constructing all
    layers. Sparse weights use normal initialization with retained fan-in
    variance two. These policies are independent of the chosen activation.
    """

    dimensions: list[int] = field(default_factory=list)
    mode: str = "dense"
    initialization: str = "kaiming_normal"
    fanin: list[int] = field(default_factory=list)
    connectivity_seeds: list[int] = field(default_factory=list)
    activation: str = "relu"
    negative_slope: float = 0.1
    activate_output: bool = True
    # None omits dropout modules; 0.0 retains them and their checkpoint layout.
    dropout: float | None = None
    output_weight_scale: float = 1.0
    projection_backend: str = "eager"
    workspace_mb: float = 128.0
    index_dtype: str = "int32"

    def validate(self) -> None:
        if len(self.dimensions) < 2 or any(
            type(value) is not int or value < 1 for value in self.dimensions
        ):
            raise ValueError("dimensions must contain at least two positive integers")
        if self.mode not in {"dense", "fixed", "topk"}:
            raise ValueError("mode must be dense, fixed, or topk")
        if self.initialization not in {
            "kaiming_normal",
            "kaiming_uniform_xavier_readout",
        }:
            raise ValueError("unknown affine MLP initialization")
        if self.initialization != "kaiming_normal" and self.mode != "dense":
            raise ValueError("kaiming_uniform_xavier_readout requires dense mode")
        count = len(self.dimensions) - 1
        if self.mode == "dense":
            if self.fanin or self.connectivity_seeds:
                raise ValueError("dense mode does not use fanin or connectivity_seeds")
        else:
            if len(self.fanin) != count or any(
                type(k) is not int or not 1 <= k <= width
                for k, width in zip(self.fanin, self.dimensions[:-1])
            ):
                raise ValueError("fanin must specify one valid input count per layer")
            if self.mode == "fixed":
                if len(self.connectivity_seeds) != count or any(
                    type(seed) is not int for seed in self.connectivity_seeds
                ):
                    raise ValueError(
                        "fixed mode requires one connectivity seed per layer"
                    )
            elif self.connectivity_seeds:
                raise ValueError(
                    "topk selects by weight and does not use connectivity_seeds"
                )
        if self.activation not in {"relu", "leaky_relu"}:
            raise ValueError("activation must be relu or leaky_relu")
        if not math.isfinite(self.negative_slope) or self.negative_slope < 0:
            raise ValueError("negative_slope must be finite and nonnegative")
        if self.dropout is not None and not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be None or between zero and one")
        if not math.isfinite(self.output_weight_scale):
            raise ValueError("output_weight_scale must be finite")
        if self.index_dtype not in {"int32", "int64"}:
            raise ValueError("index_dtype must be int32 or int64")
        if not math.isfinite(self.workspace_mb) or self.workspace_mb <= 0:
            raise ValueError("workspace_mb must be finite and positive")
