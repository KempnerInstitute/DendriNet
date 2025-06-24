from dataclasses import dataclass, field
from typing import List, Optional

from dendritic_modeling.config.base import BaseConfig


@dataclass
class TransferFunctionParams(BaseConfig):
    input_mode: int = 0
    independent_pathways: bool = False
    excitatory_dim: Optional[int] = None
    inhibitory_dim: Optional[int] = None


@dataclass
class EINetParameters(BaseConfig):
    input_dim: int = 784
    transfer_params: TransferFunctionParams = field(
        default_factory=TransferFunctionParams
    )
    excitatory_layer_sizes: List[int] = field(default_factory=lambda: [10])
    inhibitory_layer_sizes: List[int] = field(default_factory=lambda: [20])
    excitatory_branch_factors: List[int] = field(default_factory=lambda: [2, 2])
    inhibitory_branch_factors: List[int] = field(default_factory=lambda: [])
    ee_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [24])
    ei_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [200])
    ie_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [1])
    ii_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [])
    reactivate: bool = True
    reactivation_init_m: float = 1.5
    reactivation_init_b: float = 0.5
    somatic_synapses: bool = True
    topk_init_method: str = "xavier_normal"
    topk_noise_level: float = 0.0
    topk_type: str = "standard"  # can be "standard", "stochastic", or "variance"
    topk_temperature: float = 0.5
    topk_ultrafast: bool = False
    topk_strategy: str = "conductance_dynamic"
    use_shunting: bool = True
    reactivation_type: str = "param_tanh"
    reactivation_strategy: str = "inverse"
    blocklinear_strategy: str = "conductance_dynamic"
    print_hooks: bool = False  # Whether to print gradient statistics in hooks
    excitatory_target_density: float = 0.1
    inhibitory_target_density: float = 0.1
    use_noise: bool = True
    sigma: float = 0.05
    rewiring_mode: str = "global"
    rewire_frequency: int = 1
    synapses_per_branch: Optional[int] = None
    freeze_excitatory_connectivity: bool = False
    freeze_inhibitory_connectivity: bool = False
    init_method: str = "xavier_normal"
    weight_threshold: float = 1e-6


@dataclass
class FeedForwardParameters(BaseConfig):
    input_dim: int = 784
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10


@dataclass
class BaseModelConfig(BaseConfig):
    output_layer: bool = False
    output_dim: int = 10


@dataclass
class NetworkConfig(BaseConfig):
    type: str = "EINet"
    parameters: EINetParameters = field(default_factory=EINetParameters)


@dataclass
class ModelConfig(BaseConfig):
    task: str = "classification"
    base: BaseModelConfig = field(default_factory=BaseModelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
