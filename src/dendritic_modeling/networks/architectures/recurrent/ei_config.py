"""
Configuration dataclasses for unified E-I layers/networks with optional recurrence.
"""

from dataclasses import dataclass, field
from typing import Optional

from dendritic_modeling.config.base import BaseConfig


@dataclass
class PopulationConfig(BaseConfig):
    """Configuration for one population (E or I)."""

    n_neurons: int = 64
    branch_factors: list[int] = field(default_factory=lambda: [1])

    ff_excitatory_synapses: int = 8
    ff_inhibitory_synapses: int = 4
    rec_excitatory_synapses: int = 0
    rec_inhibitory_synapses: int = 0

    topk_init_method: str = "xavier_normal"
    weight_transform: str = "exp"
    use_shunting: bool = True
    reactivate: bool = True
    reactivation_type: str = "param_tanh"
    reactivation_init_m: float = 1.0
    reactivation_init_b: float = 0.5

    # Per-level taus. Empty list -> auto-generated (distal slower, proximal faster).
    level_taus: list[float] = field(default_factory=list)
    tau_base: float = 50.0
    tau_ratio: float = 3.0  # Geometric ratio between adjacent dendritic levels
    epsilon: float = 1e-8

    # Additional DendriticBranchLayer parameters
    somatic_synapses: bool = True  # Whether soma level gets direct synaptic input
    dbl_init_method: str = "analytical_expectation"
    efficient_blocklinear: bool = False

    def __post_init__(self):
        if self.n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0, got {self.n_neurons}")
        if not self.branch_factors:
            raise ValueError("branch_factors must be non-empty")
        if any(b < 1 for b in self.branch_factors):
            raise ValueError(
                f"branch_factors must all be >= 1, got {self.branch_factors}"
            )
        for name in (
            "ff_excitatory_synapses",
            "ff_inhibitory_synapses",
            "rec_excitatory_synapses",
            "rec_inhibitory_synapses",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.tau_base <= 0:
            raise ValueError(f"tau_base must be > 0, got {self.tau_base}")
        if self.tau_ratio <= 0:
            raise ValueError(f"tau_ratio must be > 0, got {self.tau_ratio}")
        if self.level_taus and any(t <= 0 for t in self.level_taus):
            raise ValueError(f"level_taus must all be > 0, got {self.level_taus}")


@dataclass
class EILayerConfig(BaseConfig):
    """Configuration for one unified E-I layer."""

    excitatory: PopulationConfig = field(default_factory=PopulationConfig)
    inhibitory: Optional[PopulationConfig] = field(
        default_factory=lambda: PopulationConfig(
            n_neurons=16,
            branch_factors=[1],
            ff_excitatory_synapses=8,
            ff_inhibitory_synapses=4,
            rec_excitatory_synapses=0,
            rec_inhibitory_synapses=0,
        )
    )

    # Set/overwritten by network at build time.
    excitatory_input_dim: int = 64
    inhibitory_input_dim: Optional[int] = None

    recurrent: bool = False
    dt: float = 1.0

    def __post_init__(self):
        if self.excitatory_input_dim <= 0:
            raise ValueError(
                f"excitatory_input_dim must be > 0, got {self.excitatory_input_dim}"
            )
        if self.inhibitory_input_dim is not None and self.inhibitory_input_dim <= 0:
            raise ValueError(
                f"inhibitory_input_dim must be > 0 or None, got {self.inhibitory_input_dim}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")


@dataclass
class EINetworkConfig(BaseConfig):
    """Configuration for a stack of unified E-I layers."""

    layers: list[EILayerConfig] = field(default_factory=lambda: [EILayerConfig()])
    input_dim: int = 64
    input_projection_dims: list[int] = field(default_factory=list)
    output_mode: str = "last"  # recurrent only: "last" | "all" | "mean"
    store_routing: bool = False

    def __post_init__(self):
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if self.output_mode not in ("last", "all", "mean"):
            raise ValueError(
                f"output_mode must be one of ('last','all','mean'), got {self.output_mode}"
            )
