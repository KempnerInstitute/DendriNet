"""Configuration dataclasses for population-network layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.population_routing import projection_delays
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    PopulationDefinitionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_constants import (
    _DELAYED,
    _INPUT_SOURCE,
    _SAME_STEP,
)

TARGET_LEVEL_NAMES = ("soma", "primary", "distal")


# Seeds that training fills in automatically; they configure nothing while the
# transfer layer is off.
_TRANSFER_SEED_KEYS = frozenset({"split_seed", "seed"})


def transfer_options(transfer_params: dict[str, Any] | None) -> set[str]:
    """Configured transfer option names, excluding automatically filled seeds."""
    return set(transfer_params or {}) - _TRANSFER_SEED_KEYS


@dataclass
class PopulationProjectionConfig(BaseConfig):
    """Directed connection between one source and one target population."""

    source: str = _INPUT_SOURCE
    target: str = ""
    pathway: str = ""  # inferred from source polarity when omitted
    timing: str = _SAME_STEP  # same_step | delayed
    enabled: bool = True
    probability: float | None = None
    seed: int | None = None
    # Dendritic levels of the target this projection may contact: "soma",
    # "primary" (children of the soma), "distal" (outermost level) or depths
    # from the soma (0 = soma). None contacts every level.
    target_levels: list[str | int] | None = None
    # Whole network updates before source activity reaches this projection.
    # Write a common delay or one delay per target level, distal through soma.
    delay_steps: int | None = None
    delay_steps_by_level: list[int] | None = None

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.target = str(self.target)
        self.pathway = str(self.pathway).lower()
        self.timing = str(self.timing).lower()
        if not self.source:
            raise ValueError("connection source must be non-empty")
        if not self.target:
            raise ValueError("connection target must be non-empty")
        if self.timing not in {_SAME_STEP, _DELAYED}:
            raise ValueError(
                "connection timing must be 'same_step' or 'delayed', "
                f"got {self.timing!r}"
            )
        projection_delays(self.timing, self.delay_steps, self.delay_steps_by_level)
        if self.probability is not None:
            self.probability = float(self.probability)
            if self.probability < 0.0 or self.probability > 1.0:
                raise ValueError(
                    f"connection probability must be in [0, 1], got {self.probability}"
                )
        if self.target_levels is not None:
            levels = list(self.target_levels)
            if not levels:
                raise ValueError("target_levels must be None or non-empty")
            for level in levels:
                if isinstance(level, bool) or not (
                    level in TARGET_LEVEL_NAMES
                    or (isinstance(level, int) and level >= 0)
                ):
                    raise ValueError(
                        "target_levels entries must be 'soma', 'primary', 'distal' "
                        f"or a depth >= 0 from the soma, got {level!r}"
                    )
            self.target_levels = levels

    def target_depths(self, n_levels: int) -> set[int] | None:
        """Allowed depths from the soma (0 = soma) for a target with n_levels."""
        if self.target_levels is None:
            return None
        depths = set()
        for level in self.target_levels:
            depth = {"soma": 0, "primary": 1, "distal": n_levels - 1}.get(level, level)
            if not 0 <= int(depth) < n_levels:
                raise ValueError(
                    f"target level {level!r} does not exist on a target with "
                    f"{n_levels} levels (connection {self.source}->{self.target})"
                )
            depths.add(int(depth))
        return depths


@dataclass
class PopulationLayerConfig(BaseConfig):
    """One layer containing multiple named dendritic populations."""

    name: str = "layer0"
    populations: list[PopulationDefinitionConfig | dict[str, Any]] = field(
        default_factory=list
    )
    connections: list[PopulationProjectionConfig | dict[str, Any]] = field(
        default_factory=list
    )
    population_defaults: dict[str, Any] = field(default_factory=dict)
    readout_population: str | None = None
    recurrent: bool = False
    dt: float = 1.0
    connection_seed: int | None = None
    store_routing: bool = False

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("layer name must be non-empty")
        if not self.populations:
            raise ValueError("population-network layer requires populations")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")


@dataclass
class PopulationNetworkConfig(BaseConfig):
    """Stack of population-network dendritic layers."""

    layers: list[PopulationLayerConfig | dict[str, Any]] = field(default_factory=list)
    input_dim: int = 64
    # Adapter applied before optional TransferLayer/input projection. Identity
    # preserves historical behavior; signed_split maps arbitrary hidden states
    # to concatenated nonnegative positive/negative channels.
    input_transform: str = "identity"
    use_transfer: bool = False
    transfer_params: dict[str, Any] = field(default_factory=dict)
    input_projection_dims: list[int] = field(default_factory=list)
    # Identity preserves the historical signed Linear projection. Softplus
    # represents positive afferent weights and requires nonnegative inputs.
    input_projection_weight_transform: str = "identity"
    readout_layer: str | None = None
    readout_population: str | None = None
    output_mode: str = "last"  # recurrent only: last | mean | all
    store_routing: bool = False
    # Compile each layer's step with torch.compile (opt-in). Recurrent
    # networks call it once per timestep, so it fuses the many small
    # per-step kernels; values change at floating-point reassociation level.
    compile_step: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if not isinstance(
            self.input_projection_weight_transform, str
        ) or self.input_projection_weight_transform not in {"identity", "softplus"}:
            raise ValueError(
                "input_projection_weight_transform must be identity or softplus"
            )
        if (
            not self.input_projection_dims
            and self.input_projection_weight_transform != "identity"
        ):
            raise ValueError(
                "input_projection_weight_transform requires input_projection_dims"
            )
        if not self.layers:
            raise ValueError("population network requires at least one layer")
        for hidden_dim in self.input_projection_dims:
            if int(hidden_dim) <= 0:
                raise ValueError(
                    f"input_projection_dims entries must be positive, got {hidden_dim}"
                )
        if self.output_mode not in {"last", "mean", "all"}:
            raise ValueError(
                f"output_mode must be one of ('last', 'mean', 'all'), got {self.output_mode!r}"
            )
        if transfer_options(self.transfer_params) and not self.use_transfer:
            raise ValueError(
                "population_network.transfer_params is applied only when "
                "population_network.use_transfer is true; enable use_transfer "
                "or remove transfer_params"
            )


__all__ = [
    "PopulationLayerConfig",
    "PopulationNetworkConfig",
    "PopulationProjectionConfig",
    "transfer_options",
]
