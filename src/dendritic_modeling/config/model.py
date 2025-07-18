from dataclasses import dataclass, field
from typing import Optional

from dendritic_modeling.config.base import BaseConfig


@dataclass
class TransferFunctionParams(BaseConfig):
    input_mode: int = 0
    independent_pathways: bool = False
    excitatory_dim: Optional[int] = None
    inhibitory_dim: Optional[int] = None
    output_activation: Optional[str] = None  # Options: "sigmoid", "relu", "tanh", "softplus", None


@dataclass
class EINetParameters(BaseConfig):
    input_dim: int = 784
    transfer_params: TransferFunctionParams = field(
        default_factory=TransferFunctionParams
    )
    excitatory_layer_sizes: list[int] = field(default_factory=lambda: [10])
    inhibitory_layer_sizes: list[int] = field(default_factory=lambda: [20])
    excitatory_branch_factors: list[int] = field(default_factory=lambda: [2, 2])
    inhibitory_branch_factors: list[int] = field(default_factory=list)
    ee_synapses_per_branch_per_layer: list[int] = field(default_factory=lambda: [24])
    ei_synapses_per_branch_per_layer: list[int] = field(default_factory=lambda: [200])
    ie_synapses_per_branch_per_layer: list[int] = field(default_factory=lambda: [1])
    ii_synapses_per_branch_per_layer: list[int] = field(default_factory=list)
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
    efficient_blocklinear: bool = False  # use memory-efficient implementation
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
    adaptive_initialization: bool = True  # use network-aware weight scaling


@dataclass
class CoreNetworkConfig(BaseConfig):
    type: str = "EINet"
    parameters: EINetParameters = field(default_factory=EINetParameters)


@dataclass
class EncoderNetworkParameters(BaseConfig):
    input_dim: int = None
    input_shape: tuple[int, int, int] = None
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    encoder_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    encoder_strides: list[int] = field(default_factory=lambda: [2, 2])
    latent_dim: int = None
    latent_shape: tuple[int, int, int] = None
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [64, 128])
    activation: str = "relu"


@dataclass
class EncoderNetworkConfig(BaseConfig):
    type: str = "MLPAutoencoder"
    parameters: EncoderNetworkParameters = field(
        default_factory=EncoderNetworkParameters
    )


@dataclass
class DecoderNetworkParameters(BaseConfig):
    input_dim: int = None
    hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    activation: str = "relu"
    output_dim: int = None


@dataclass
class DecoderNetworkConfig(BaseConfig):
    type: str = "MLP"
    parameters: DecoderNetworkParameters = field(
        default_factory=DecoderNetworkParameters
    )


@dataclass
class ModelConfig(BaseConfig):
    task: str = "classification"
    encoder_network: EncoderNetworkConfig = field(default_factory=EncoderNetworkConfig)
    core_network: CoreNetworkConfig = field(default_factory=CoreNetworkConfig)
    decoder_network: DecoderNetworkConfig = field(default_factory=DecoderNetworkConfig)
