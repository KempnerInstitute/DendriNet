from dataclasses import dataclass, field
from typing import Any, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.recurrent import RecurrentConfig


@dataclass
class TransferFunctionConfig(BaseConfig):
    """Transfer function parameters."""

    input_mode: int = 1
    input_transform: Optional[str] = None
    independent_pathways: bool = False
    excitatory_dim: Optional[int] = None
    inhibitory_dim: Optional[int] = None
    split_strategy: str = "random"
    split_seed: Optional[int] = None
    output_activation: Optional[str] = None
    inhibitory_mode: str = "first"  # "none", "first", "all"
    allow_direct_inhibitory_stream: bool = False
    input_mode1_build_inhibitory_population: bool = False
    require_explicit_direct_inhibitory_stream: bool = False


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
class StructuredConnectivityConfig(BaseConfig):
    """Optional hard topology constraints for synaptic pathways."""

    enabled: bool = False
    seed: Optional[int] = None
    method: str = "bernoulli"
    probability: Optional[float] = None
    indegree: Optional[int] = None
    distance_kernel: str = "none"
    distance_sigma: float = 1.0
    spatial: dict[str, Any] = field(default_factory=dict)
    pathways: dict[str, Any] = field(default_factory=dict)


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
    structured: StructuredConnectivityConfig = field(
        default_factory=StructuredConnectivityConfig
    )


@dataclass
class MorphologyConfig(BaseConfig):
    """Morphological features configuration."""

    somatic_synapses: bool = False
    use_shunting: bool = True  # Legacy fallback for core.type == "einet"
    # Legacy per-sample, across-output z-score for raw additive voltage.
    use_additive_normalization: bool = False
    # Explicit fair-comparison controls. ``raw`` preserves the historical
    # additive equation; the other modes are active only when use_shunting=false.
    additive_mode: str = "raw"
    additive_tangent_n0: Optional[float] = None
    additive_tangent_t0: Optional[float] = None
    # Only used when an EINet-style config is translated into the recurrent
    # unified E-I path. Feedforward models ignore this flag.
    allow_self_recurrence: bool = True
    # Keep the structured-config default aligned with legacy BP configs.
    # Experiments that need a stricter positive-conductance regime can still
    # request "exp" explicitly.
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
    ultrafast: bool = (
        True  # ultrafast stochastic-topk default (set False for rank-probabilistic)
    )
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
            "credit_trace_decay": 0.95,
            "credit_candidate_pool_size": 8,
            "credit_turnover_fraction": 0.1,
            "credit_weak_active_pool_fraction": 0.5,
            "credit_min_observations": 5,
            "credit_swap_margin": 0.0,
            "credit_force_turnover": False,
            "credit_selection": "credit",
            "credit_warmup_steps": 5,
        }
    )
    indexed: dict = field(
        default_factory=lambda: {
            "candidate_size": None,
            "selection": "standard",
            "seed": None,
            "output_chunk_size": 2048,
            "rewire_frequency": 100,
            "rewire_quantile": 0.05,
            "rewire_until_step": None,
        }
    )
    dense_to_sparse: dict = field(
        default_factory=lambda: {
            "initial_density": 1.0,
            "initial_k": None,
            "start_step": 0,
            "end_step": 1000,
            "update_interval": 1,
            "schedule": "cubic",
            "freeze_on_end": False,
            "advance_on_forward": True,
            "prune_metric": "weight",
        }
    )


@dataclass
class ReactivationConfig(BaseConfig):
    """Reactivation function configuration."""

    enabled: bool = True
    type: str = "param_tanh"
    # Alias for paper-facing configs. If provided, this maps to ``type``.
    dendritic_activation: Optional[str] = None
    # init_policy selects how the parametric gate ``(m, b)`` is initialized.
    #
    # Primary policies:
    # - "analytical": initialize from the selected DBL init rule.
    # - "fixed": use init_m and init_b exactly as provided.
    # - "empirical": robust median/MAD calibration from sampled voltages.
    # - "occupancy_quantile": generalized q_low/q_high -> r_low/r_high rule.
    # - "analytical_slope_occupancy_center": keep analytical m, use empirical b.
    # - "occupancy_slope_analytical_center": use empirical m, keep analytical b.
    # Legacy alias: "quantile" is accepted and normalized to "occupancy_quantile".
    init_policy: str = "analytical"
    init_m: float = 1.5
    init_b: float = 0.5
    occupancy_quantile_low: Optional[float] = None
    occupancy_quantile_high: Optional[float] = None
    occupancy_target_low: Optional[float] = None
    occupancy_target_high: Optional[float] = None
    calibration_min_quantile_width: float = 1e-3
    calibration_max_m: float = 50.0
    calibration_revert_on_invalid: bool = True
    # Scale factor used by the "empirical" policy when mapping a robust
    # voltage spread estimate (MAD * 1.4826) to the initial gate slope.
    sigma_aware_k: float = 0.25
    gradient_scaling: str = "none"


@dataclass
class BlockLinearConfig(BaseConfig):
    """BlockLinear configuration."""

    gradient_scaling: str = "none"
    efficient: bool = False


@dataclass
class ImplementationConfig(BaseConfig):
    """Implementation optimization settings."""

    # Enabled by default for shunting models, but use the center-preserving
    # policy so conductance scaling does not move the initial shunting gate
    # operating point. "legacy_scale" is kept only for reproducing older sweeps.
    adaptive_initialization: bool = True
    adaptive_initialization_policy: str = "preserve_shunting_center"
    adaptive_target_conductance: float = 5.0
    initial_child_conductance: float = 1.0
    print_hooks: bool = False
    store_routing: bool = False
    compile_forward: bool = False


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

    type: str = "einet"
    # Optional base seed for component-keyed parameter initialization. Training
    # setup fills this from experiment.model_seed; ``None`` preserves direct
    # construction through the historical global RNG stream.
    initialization_seed: Optional[int] = None

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
    # Optional receptor/conductance-style synapse-type payload. Kept plain so
    # architecture modules can evolve without coupling the config package.
    synapse_types: dict[str, Any] = field(default_factory=dict)
    # Optional recurrent population dynamics payload, e.g. LIF soma settings.
    dynamics: dict[str, Any] = field(default_factory=dict)
    # Optional dendritic spike/plateau payload for recurrent cores.
    dendritic_spikes: dict[str, Any] = field(default_factory=dict)
    # Optional delayed soma-to-dendrite feedback payload for recurrent cores.
    soma_feedback: dict[str, Any] = field(default_factory=dict)
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
    # Sparse point-state control with a prespecified multiset of leak time
    # constants. Kept plain to avoid coupling typed config to architecture code.
    heterogeneous_leak_ctrnn: dict[str, Any] = field(default_factory=dict)
    # Fixed Legendre Delay Network memory with sparse learned input/readout maps.
    legendre_memory: dict[str, Any] = field(default_factory=dict)
    # Canonical population/layer/connection schema for feedforward or recurrent
    # named-population dendritic networks.
    population_network: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderParamsConfig(BaseConfig):
    """Decoder network parameters."""

    input_dim: int = 20
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    activation: str = "relu"
    output_dim: int = 10
    input_normalization: str = "none"
    input_norm_eps: float = 1e-5
    output_init: str = "default"


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
    """Config for replacing vision-backbone modules with a dendritic core.

    When ``enabled=true``, ``model.encoder`` and ``model.decoder`` are
    ignored — the encoder and decoder are built from the backbone split.
    ``vision_replacement`` is the preferred config name; the older
    ``pretrained_replacement`` field remains supported as an alias.

    Example::

        vision_replacement:
          backbone: "alexnet"
          weights: "IMAGENET1K_V1"
          input_shape: [3, 224, 224]
          target_modules: ["classifier.4", "classifier.5"]
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
    # ``configured`` constructs model.core as the replacement. The
    # ``pretrained_span`` control retains the original modules and weights in
    # the declared span while using the same replacement training pipeline.
    core_source: str = "configured"
    input_shape: list[int] = field(default_factory=lambda: [3, 224, 224])
    replace: ReplacementRegionConfig = field(default_factory=ReplacementRegionConfig)
    target_modules: list[str] = field(default_factory=list)
    omit_layers: list[str] = field(default_factory=list)
    trainability: TrainabilityConfig = field(default_factory=TrainabilityConfig)


@dataclass
class TransformerReplacementConfig(BaseConfig):
    """Config for replacing transformer decoder-block modules with DendriNet.

    This is the transformer analogue of ``pretrained_replacement``.  The target
    model/layers live here; the dendritic replacement itself is described by
    the shared ``model.core`` section.
    """

    enabled: bool = False
    model_name: str = ""
    layers: list[int] = field(default_factory=list)
    target_module: str = "mlp"
    layers_attr: Optional[str] = None
    input_transform: str = "signed_split"
    preserve_device_dtype: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    replacement_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration."""

    task: str = "classification"
    # Identity decoders historically received a learned positive scale for each
    # output. Keep that behavior by default, while allowing experiments whose
    # somatic outputs must be the literal class logits to disable it.
    learned_output_scale: bool = True
    # Fixed positive global temperature applied after the decoder. Unlike the
    # historical learned per-output scale, this cannot mix classes or change
    # the argmax. The default preserves all existing behavior.
    fixed_output_scale: float = 1.0
    # Explicit calibration mode for identity/class-aligned readouts:
    # ``per_class`` (legacy vector), ``global`` (one positive scalar), or
    # ``fixed`` (no learned calibration). None preserves learned_output_scale.
    output_scale_mode: Optional[str] = None

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # Vision backbone replacement. ``pretrained_replacement`` is kept as a
    # compatibility alias for older configs.
    vision_replacement: PretrainedReplacementConfig = field(
        default_factory=PretrainedReplacementConfig
    )
    pretrained_replacement: PretrainedReplacementConfig = field(
        default_factory=PretrainedReplacementConfig
    )
    transformer_replacement: TransformerReplacementConfig = field(
        default_factory=TransformerReplacementConfig
    )
