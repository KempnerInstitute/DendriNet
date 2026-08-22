from dataclasses import dataclass, field
from typing import Any, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.recurrent import RecurrentConfig


@dataclass
class TransferFunctionConfig(BaseConfig):
    """Transfer function parameters."""

    input_mode: int = 1
    independent_pathways: bool = False
    excitatory_dim: Optional[int] = None
    inhibitory_dim: Optional[int] = None
    output_activation: Optional[str] = None
    inhibitory_mode: str = "first"  # "none", "first", "all"
    force_projection: bool = False  # confound-control: force output projection even when dims already match
    zero_pad_output: bool = False   # pad EINet output with zeros to match suffix input dim (no learned projection)


@dataclass
class ArchitectureConfig(BaseConfig):
    """Core network architecture configuration."""

    excitatory_layer_sizes: list[int] = field(default_factory=lambda: [20])
    inhibitory_layer_sizes: list[int] = field(default_factory=list)
    excitatory_branch_factors: list[int] = field(default_factory=lambda: [3, 3, 3, 3])
    inhibitory_branch_factors: list[int] = field(default_factory=lambda: [1])
    input_projection_dims: list[int] = field(default_factory=list)

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

    # Recurrent synapse counts (per branch, list per layer; 0 = no recurrence)
    rec_ee_synapses_per_branch: list[int] = field(default_factory=lambda: [0])
    rec_ie_synapses_per_branch: list[int] = field(default_factory=lambda: [0])
    rec_ei_synapses_per_branch: list[int] = field(default_factory=lambda: [0])
    rec_ii_synapses_per_branch: list[int] = field(default_factory=lambda: [0])


@dataclass
class MorphologyConfig(BaseConfig):
    """Morphological features configuration."""

    somatic_synapses: bool = False
    use_shunting: bool = True
    use_additive_normalization: bool = False  # Normalize voltage in additive mode
    weight_transform: str = "softplus"
    dbl_init_method: str = "analytical_expectation"


@dataclass
class SparsityConfig(BaseConfig):
    """Sparsity configuration."""

    init_method: str = "xavier_normal"
    noise_level: float = 0.0
    type: str = "standard"
    weight_norm_order: Optional[int] = None
    gamma: float = 1.0
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
    init_m: float = 1.5
    init_b: float = 0.5
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
    store_routing: bool = False


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
    router_mode: str = "fixed"
    pathway_groups: list[list[int]] = field(default_factory=list)
    shared_indices: list[int] = field(default_factory=list)
    n_pathways: int = 2
    pathway_dim: Optional[int] = None
    pathway_dims: list[int] = field(default_factory=list)
    router_activation: str = "none"
    learned_router_temperature: float = 1.0
    learned_router_init_scale: float = 4.0


@dataclass
class EncoderConfig(BaseConfig):
    """Encoder network configuration."""

    type: str = "identity"
    load_save_root: str = "./trained_encoder_networks/"
    params: EncoderParamsConfig = field(default_factory=EncoderParamsConfig)


@dataclass
class SpatialConfig(BaseConfig):
    """Spatial convolution parameters for dendritic conv cores.

    Used by ``hierarchical_dendritic_conv`` to configure the unfold operation
    that extracts local patches from spatial feature maps.
    """

    kernel_size: int = 3
    stride: int = 1
    padding: int = 0


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
    # Spatial convolution params for hierarchical_dendritic_conv cores.
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    # DEPRECATED: Legacy recurrent config. Use unified EI config classes instead.
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    # Optional config payload for the new unified E-I architecture.
    # Kept as a plain dict to avoid coupling config package to architecture modules.
    unified_ei: dict[str, Any] = field(default_factory=dict)
    # Optional recurrent mode for EINet-family cores.
    # When enabled, factory can translate EINet structured config into recurrent EINetwork.
    recurrent_ei: dict[str, Any] = field(default_factory=dict)
    # Optional config payload for baseline recurrent cores (gru/lstm/vanilla_rnn).
    baseline_rnn: dict[str, Any] = field(default_factory=dict)


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
class ReplacementRegionConfig(BaseConfig):
    """Specifies the contiguous layer range to replace with a core network."""

    start: str = ""  # first replaced layer (inclusive)
    end: str = ""  # last replaced layer (inclusive)


@dataclass
class SegmentTrainabilityConfig(BaseConfig):
    """Trainability policy for an encoder or decoder segment.

    ``mode`` must be ``"frozen"`` or ``"trainable"``.
    Learning rates are controlled via ``training.main.common.param_groups``
    (``encoder_lr`` for the encoder, ``decoder_lr`` for the decoder).
    """

    mode: str = "frozen"


@dataclass
class TrainabilityConfig(BaseConfig):
    """Trainability policies for encoder and decoder segments."""

    encoder: SegmentTrainabilityConfig = field(
        default_factory=lambda: SegmentTrainabilityConfig(mode="frozen")
    )
    decoder: SegmentTrainabilityConfig = field(
        default_factory=lambda: SegmentTrainabilityConfig(mode="trainable")
    )


@dataclass
class PretrainedReplacementConfig(BaseConfig):
    """Single-source config for replacing layers in a pretrained backbone.

    When ``enabled=true``, ``model.encoder`` and ``model.decoder`` are
    ignored — the encoder and decoder are built from the backbone split.

    Example::

        pretrained_replacement:
          backbone: "alexnet"
          weights: "IMAGENET1K_V1"
          input_shape: [3, 224, 224]
          replace:
            start: "classifier.4"
            end: "classifier.5"
          omit_layers: ["classifier.0", "classifier.3"]
          trainability:
            encoder:
              mode: "frozen"
            decoder:
              mode: "trainable"

    Learning rates are controlled via ``training.main.common.param_groups``
    (``encoder_lr`` for the encoder, ``decoder_lr`` for the decoder).
    """

    enabled: bool = False
    backbone: str = "alexnet"
    weights: Optional[str] = "IMAGENET1K_V1"
    input_shape: list[int] = field(default_factory=lambda: [3, 224, 224])
    replace: ReplacementRegionConfig = field(default_factory=ReplacementRegionConfig)
    omit_layers: list[str] = field(default_factory=list)
    trainability: TrainabilityConfig = field(default_factory=TrainabilityConfig)
    reinit_decoder: bool = False  # if True, reset decoder weights to random (simulates replacing all layers)


@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration."""

    task: str = "classification"

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # New unified pretrained replacement config
    pretrained_replacement: PretrainedReplacementConfig = field(
        default_factory=PretrainedReplacementConfig
    )
