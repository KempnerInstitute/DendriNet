"""Opt-in paired-input E/I routing controls; independent of neuronal defaults."""

import math
from dataclasses import dataclass, field
from numbers import Real

from dendritic_modeling.config.base import BaseConfig


@dataclass
class RoutedEIConfig(BaseConfig):
    """One bank of routed branches followed by an unconstrained affine readout.

    The standard model wrapper receives concatenated ``[E, I]`` with
    ``2 * features`` channels. Direct callers may pass the streams separately.
    This is a rate-model control, not a Dale-constrained output population.
    No instance is created unless ``core.type: routed_ei`` is selected.
    """

    features: int = 1
    branches: int = 1
    output_dim: int = 1
    mechanism: str = "shunting"
    routing: str = "learned"
    fixed_assignment: list[int] = field(default_factory=list)
    inhibitory_support: str = "matched"
    temperature: float = 1.0
    hard_routing: bool = True
    leak_conductance: float = 1.0
    epsilon: float = 1e-8
    # None uses the existing robust L1 normalization. A floor explicitly selects
    # the clamped-sum convention used by some reduced-model historical runs.
    weight_normalization_floor: float | None = None
    route_init_std: float = 0.05
    weight_init_std: float = 0.04
    branch_center: float = 0.0
    branch_scale: float = 1.0
    branch_divisor: float = 1.0
    # A signed control gate, not the unit-range neuronal ParametricTanh.
    branch_activation: str = "none"
    trainable_activation: bool = False
    activation_slope_raw: float = 0.5413249
    activation_offset: float = 0.0

    def validate(self) -> None:
        for name in ("features", "branches", "output_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"routed_ei.{name} must be a positive integer")
        if self.mechanism not in {"shunting", "additive"}:
            raise ValueError("routed_ei.mechanism must be shunting or additive")
        if self.routing not in {"learned", "fixed"}:
            raise ValueError("routed_ei.routing must be learned or fixed")
        if self.inhibitory_support not in {"matched", "global_mean"}:
            raise ValueError("inhibitory_support must be matched or global_mean")
        if self.routing == "fixed":
            if len(self.fixed_assignment) != self.features or any(
                type(i) is not int or not 0 <= i < self.branches
                for i in self.fixed_assignment
            ):
                raise ValueError("fixed_assignment needs one valid branch per feature")
        elif self.fixed_assignment:
            raise ValueError("fixed_assignment requires routing: fixed")
        if type(self.hard_routing) is not bool:
            raise ValueError("hard_routing must be a bool")
        if self.branch_activation not in {"none", "tanh"}:
            raise ValueError("branch_activation must be none or tanh")
        if type(self.trainable_activation) is not bool:
            raise ValueError("trainable_activation must be a bool")
        if self.branch_activation == "none" and (
            self.trainable_activation
            or self.activation_slope_raw != 0.5413249
            or self.activation_offset != 0.0
        ):
            raise ValueError("activation parameters require branch_activation: tanh")
        positive = {
            "temperature",
            "leak_conductance",
            "branch_scale",
            "branch_divisor",
        }
        nonnegative = {"epsilon", "route_init_std", "weight_init_std"}
        if self.weight_normalization_floor is not None:
            positive.add("weight_normalization_floor")
        for name in (
            positive
            | nonnegative
            | {"branch_center", "activation_slope_raw", "activation_offset"}
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(f"routed_ei.{name} must be finite and real")
            if name in positive and value <= 0:
                raise ValueError(f"routed_ei.{name} must be positive")
            if name in nonnegative and value < 0:
                raise ValueError(f"routed_ei.{name} must be nonnegative")
        if self.mechanism == "additive":
            if self.inhibitory_support != "matched":
                raise ValueError("global_mean inhibition is a shunting-only control")
            if self.leak_conductance != 1.0 or self.epsilon != 1e-8:
                raise ValueError(
                    "additive does not read leak_conductance or epsilon; leave their defaults"
                )


__all__ = ["RoutedEIConfig"]
