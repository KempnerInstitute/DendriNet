from dataclasses import dataclass, field
from typing import Optional

from dendritic_modeling.config.base import BaseConfig


@dataclass
class TransferFunctionConfig(BaseConfig):
    """Transfer function parameters."""

    input_mode: int = 1
    independent_pathways: bool = False
    excitatory_dim: Optional[int] = None
    inhibitory_dim: Optional[int] = None
    output_activation: Optional[str] = None


@dataclass
class ArchitectureConfig(BaseConfig):
    """Core network architecture configuration."""

    excitatory_layer_sizes: list[int] = field(default_factory=lambda: [20])
    inhibitory_layer_sizes: list[int] = field(default_factory=list)
    excitatory_branch_factors: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
    inhibitory_branch_factors: list[int] = field(default_factory=lambda: [1])

    # Inhibitory network type and MLP parameters
    inhibitory_network_type: str = "dendritic"  # Options: "dendritic", "mlp"
    mlp_inhibitory_network_params: dict = field(
        default_factory=lambda: {"hidden_dims": [200], "activation": "relu"}
    )


@dataclass
class ConnectivityConfig(BaseConfig):
    """Synaptic connectivity configuration."""

    ee_synapses_per_branch_per_layer: list[int] = field(default_factory=lambda: [40])
    ei_synapses_per_branch_per_layer: list[int] = field(default_factory=lambda: [0])
    ie_synapses_per_branch_per_layer: list[int] = field(default_factory=lambda: [10])
    ii_synapses_per_branch_per_layer: list[int] = field(default_factory=lambda: [0])


@dataclass
class MorphologyConfig(BaseConfig):
    """Morphological features configuration."""

    somatic_synapses: bool = False
    use_shunting: bool = True
    weight_transform: str = "softplus"
    dbl_init_method: str = "analytical_expectation"


@dataclass
class SparsityConfig(BaseConfig):
    """Sparsity configuration."""

    init_method: str = "xavier_normal"
    noise_level: float = 0.0
    type: str = "standard"
    temperature: float = 0.0
    gradient_scaling: str = "none"

    deepst: dict = field(
        default_factory=lambda: {
            "excitatory_target_density": 0.1,
            "inhibitory_target_density": 0.1,
            "use_noise": True,
            "sigma": 0.05,
            "rewiring_mode": "global",
            "rewire_frequency": 1,
            "synapses_per_branch": None,
            "freeze_excitatory_connectivity": False,
            "freeze_inhibitory_connectivity": False,
            "init_method": "xavier_normal",
            "weight_threshold": 1e-6,
        }
    )


@dataclass
class ReactivationConfig(BaseConfig):
    """Reactivation function configuration."""

    enabled: bool = True
    type: str = "param_tanh"
    init_m: Optional[float] = None
    init_b: Optional[float] = None
    gradient_scaling: str = "none"


@dataclass
class BlockLinearConfig(BaseConfig):
    """BlockLinear configuration."""

    gradient_scaling: str = "none"
    efficient: bool = False


@dataclass
class ImplementationConfig(BaseConfig):
    """Implementation optimization settings."""

    adaptive_initialization: bool = False
    print_hooks: bool = False


@dataclass
class EncoderParamsConfig(BaseConfig):
    """Encoder network parameters."""

    input_dim: Optional[int] = None
    input_shape: list[int] = field(default_factory=lambda: [3, 32, 32])
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    encoder_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    encoder_strides: list[int] = field(default_factory=lambda: [2, 2])
    latent_dim: Optional[int] = None
    latent_shape: Optional[list[int]] = None
    decoder_hidden_dims: list[int] = field(default_factory=lambda: [64, 128])
    activation: str = "relu"


@dataclass
class EncoderConfig(BaseConfig):
    """Encoder network configuration."""

    type: str = "identity"
    load_save_root: str = "./trained_encoder_networks/"
    params: EncoderParamsConfig = field(default_factory=EncoderParamsConfig)


@dataclass
class CoreConfig(BaseConfig):
    """Core network configuration."""

    type: str = "EINet"

    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    transfer: TransferFunctionConfig = field(default_factory=TransferFunctionConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    sparsity: SparsityConfig = field(
        default_factory=SparsityConfig
    )  # Renamed from topk
    reactivation: ReactivationConfig = field(default_factory=ReactivationConfig)
    blocklinear: BlockLinearConfig = field(default_factory=BlockLinearConfig)
    implementation: ImplementationConfig = field(default_factory=ImplementationConfig)


@dataclass
class DecoderParamsConfig(BaseConfig):
    """Decoder network parameters."""

    input_dim: int = 20
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    activation: str = "relu"
    output_dim: int = 10


@dataclass
class DecoderConfig(BaseConfig):
    """Decoder network configuration."""

    type: str = "MLP"
    params: DecoderParamsConfig = field(default_factory=DecoderParamsConfig)


@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration."""

    task: str = "classification"

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
