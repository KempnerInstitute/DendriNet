from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from dendritic_modeling.config.affine_mlp import AffineMLPConfig
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.choices import normalize_gradient_scaling
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.extensions import extension_default
from dendritic_modeling.config.keys import check_section_keys
from dendritic_modeling.config.shared_projection import SharedProjectionShuntingConfig
from dendritic_modeling.config.sparse_backends import (
    normalize_indexed_projection_options,
)
from dendritic_modeling.config.structured_connectivity import (
    GLOBAL_RNG_SEED,
    validate_structured_connectivity,
)
from dendritic_modeling.config.synapse_payloads import DeepSTConfig, DenseToSparseConfig


@dataclass
class TransferFunctionConfig(BaseConfig):
    """The transfer layer: how the core input becomes E and I input streams.

    The excitatory stream drives the first layer's excitatory synapses. The
    inhibitory stream is the first layer's feedforward inhibitory input: its
    inhibitory cells read it, and its excitatory cells read it directly when
    the layer has no inhibitory cells.
    """

    # Only "identity" is accepted; other input adapters belong to
    # population_network.input_transform or spatial.input_transform.
    input_transform: Optional[str] = None
    # False duplicates the input into both streams; True splits the input
    # channels between them (excitatory_dim + inhibitory_dim = input size).
    independent_pathways: bool = False
    excitatory_dim: Optional[int] = None
    inhibitory_dim: Optional[int] = None
    split_strategy: str = "random"
    split_seed: Optional[int] = None
    # Applied to both streams: none, relu, sigmoid, softplus or tanh.
    output_activation: Optional[str] = None


#: Options of the MLP inhibitory cells (inhibitory_network_type: mlp).
MLP_INHIBITORY_NETWORK_DEFAULTS = {"hidden_dims": [200], "activation": "relu"}


def resolve_mlp_inhibitory_network_params(
    params: Mapping[str, Any] | None, *, section: str
) -> dict[str, Any]:
    """Return the MLP inhibitory-cell options with unset keys at their defaults."""
    written = check_section_keys(
        dict(params or {}), MLP_INHIBITORY_NETWORK_DEFAULTS, section=section
    )
    return {**MLP_INHIBITORY_NETWORK_DEFAULTS, **written}


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
        default_factory=lambda: dict(MLP_INHIBITORY_NETWORK_DEFAULTS)
    )

    def __post_init__(self) -> None:
        self.mlp_inhibitory_network_params = resolve_mlp_inhibitory_network_params(
            self.mlp_inhibitory_network_params,
            section="model.core.architecture.mlp_inhibitory_network_params",
        )


@dataclass
class StructuredConnectivityConfig(BaseConfig):
    """Optional hard topology constraints for synaptic pathways."""

    enabled: bool = False
    seed: Optional[int] = None
    method: str = "bernoulli"
    probability: Optional[float] = None
    indegree: Optional[int] = None
    # Kernel of ``method: distance`` rules that do not write one. Before
    # config_version 3 it was "none" (a uniform base_probability), unlike
    # the gaussian of population rules; upgraded files write "none".
    distance_kernel: str = "gaussian"
    distance_sigma: float = 1.0
    spatial: dict[str, Any] = field(default_factory=dict)
    pathways: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        section = "model.core.connectivity.structured"
        validate_structured_connectivity(self, section=section)
        seeds = [("", self.seed)] + [
            (f".pathways.{name}", to_plain_dict(entry or {}).get("seed"))
            for name, entry in to_plain_dict(self.pathways or {}).items()
        ]
        for where, seed in seeds:
            if seed == GLOBAL_RNG_SEED:
                raise ValueError(
                    f"{section}{where}.seed={GLOBAL_RNG_SEED!r} applies to "
                    "population rules; training seeds this block from "
                    "experiment.topology_seed, so write an integer seed or none"
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
    leak_conductance: float = 1.0
    learnable_leak: bool = False
    # "none", "float16" or "bfloat16": opt-in rounding of dendritic signals and
    # conductances (indexed_release shunting populations only).
    signal_rounding: str = "none"
    additive_mode: str = "raw"
    additive_tangent_n0: Optional[float] = None
    additive_tangent_t0: Optional[float] = None
    # Options of an additive rule that a plugin provides (additive_mode names it).
    additive_options: dict[str, Any] = field(default_factory=dict)
    # Optional anchors indexed by soma-relative depth (soma=0). These are
    # useful when a tangent-matched control spans several operating regimes.
    additive_tangent_n0_by_depth: list[float] = field(default_factory=list)
    additive_tangent_t0_by_depth: list[float] = field(default_factory=list)
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

    # Constructor initializer of the sparse weights. Dendritic branch layers
    # overwrite those weights with morphology.dbl_init_method, so this value
    # only changes the discarded constructor draw (and, without an
    # initialization seed, later draws of the global random stream).
    # ``type: indexed_rewire`` gives a regrown contact the weight of the
    # contact it replaces; only an IndexedRewireLinear built directly with
    # rewire_init_policy="fresh" draws regrown weights with this initializer.
    init_method: str = "xavier_normal"
    noise_level: float = 0.0
    type: str = "standard"
    weight_norm_order: Optional[int] = None
    gamma: float = 1.0
    # Stochastic selection (type stochastic, indexed_dynamic selection
    # stochastic): noise or sampling temperature, and the ultrafast noisy
    # top-K versus rank-probabilistic sampling.
    temperature: float = 0.5
    ultrafast: bool = True
    gradient_scaling: str = "none"
    # Input statistic of type "variance": "absolute" or scale-free "relative".
    variance_score: str = "absolute"

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
            # Log-space regrowth: every registered method regrows at
            # weight_threshold; only null (uniform regrowth) differs.
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
            "index_dtype": "int64",
            "workspace_mb": None,
            "cache_transformed_weights": False,
            "recompute_backward": False,
            "projection_backend": "eager",
            "persistent_indices": True,
            "init_mode": "per_rank",
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

    def __post_init__(self) -> None:
        """Validate the nested indexed-runtime block before it is translated.

        This block intentionally remains dictionary-backed because several
        architecture families consume it. Validating it here prevents a typo
        from being silently ignored by their ``dict.get`` compatibility paths.
        """
        self.gradient_scaling = normalize_gradient_scaling(
            self.gradient_scaling,
            "topk",
            field="model.core.sparsity.gradient_scaling",
        )
        if not isinstance(self.indexed, Mapping):
            raise TypeError("model.core.sparsity.indexed must be a mapping")

        indexed = dict(self.indexed)
        allowed = {
            "release_p_min",
            "release_p_max",
            "release_weight_scale",
            "release_impact_decay",
            "release_impact_metric",
            "release_max_conductance",
            "release_track_source_activity",
            "candidate_size",
            "selection",
            "seed",
            "output_chunk_size",
            "index_dtype",
            "workspace_mb",
            "cache_transformed_weights",
            "recompute_backward",
            "projection_backend",
            "persistent_indices",
            "init_mode",
            "rewire_frequency",
            "rewire_quantile",
            "rewire_until_step",
        }
        unknown = sorted(set(indexed) - allowed)
        if unknown:
            if "comparison_backend" in unknown:
                raise ValueError(
                    "model.core.sparsity.indexed.comparison_backend is not a "
                    "runtime option and was previously ignored; select "
                    "projection_backend: eager, recompute, triton, "
                    "triton_transposed, triton_transposed_atomic, "
                    "triton_transposed_bf16_compat, triton_fused, triton_ell, "
                    "cuda, or auto"
                )
            raise ValueError(
                "Unknown model.core.sparsity.indexed option(s): " + ", ".join(unknown)
            )

        for name, payload_cls in (
            ("deepst", DeepSTConfig),
            ("dense_to_sparse", DenseToSparseConfig),
        ):
            payload = getattr(self, name)
            if not isinstance(payload, Mapping):
                raise TypeError(f"model.core.sparsity.{name} must be a mapping")
            check_section_keys(
                payload,
                {item.name for item in fields(payload_cls)},
                section=f"model.core.sparsity.{name}",
            )

        indexed["projection_backend"] = normalize_indexed_projection_options(
            indexed.get("projection_backend", "eager"),
            recompute_backward=bool(indexed.get("recompute_backward", False)),
        )
        self.indexed = indexed


@dataclass
class ReactivationConfig(BaseConfig):
    """Reactivation function configuration."""

    enabled: bool = True
    type: str = "param_tanh"
    # Optional soma-output-stage override: when set, only the final branch
    # layer of each excitatory DendriNet uses this activation type (with
    # soma_init_m / soma_init_b), while branches keep ``type``. ``null``
    # preserves the historical uniform gate exactly; inhibitory populations
    # always keep the base gate.
    soma_type: Optional[str] = None
    soma_init_m: float = 1.0
    soma_init_b: float = 0.0
    # Alias of ``type``. If provided, this maps to ``type``.
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
    memory_efficient: bool = False

    def __post_init__(self) -> None:
        from dendritic_modeling.config.reactivation import (
            normalize_reactivation_init_policy,
        )

        self.gradient_scaling = normalize_gradient_scaling(
            self.gradient_scaling,
            "reactivation",
            field="model.core.reactivation.gradient_scaling",
        )
        self.init_policy = normalize_reactivation_init_policy(self.init_policy)
        self._resolve_dendritic_activation_alias()

    def _resolve_dendritic_activation_alias(self) -> None:
        """Fold ``dendritic_activation`` into ``enabled`` and ``type``."""
        if self.dendritic_activation is None:
            return
        alias = str(self.dendritic_activation).strip().lower()
        field = "model.core.reactivation.dendritic_activation"
        if alias in {"none", "identity"}:
            self.enabled = False
        else:
            if not self.enabled:
                raise ValueError(
                    f"{field}={self.dendritic_activation!r} conflicts with "
                    "enabled: false; set one of them"
                )
            if self.type != "param_tanh" and self.type.lower() != alias:
                raise ValueError(
                    f"{field}={self.dendritic_activation!r} conflicts with "
                    f"type={self.type!r}; set only type"
                )
            self.type = alias
        self.dendritic_activation = None


@dataclass
class BlockLinearConfig(BaseConfig):
    """BlockLinear configuration."""

    gradient_scaling: str = "none"
    efficient: bool = False

    def __post_init__(self) -> None:
        self.gradient_scaling = normalize_gradient_scaling(
            self.gradient_scaling,
            "blocklinear",
            field="model.core.blocklinear.gradient_scaling",
        )


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
    initial_conductance_scale: float = 1.0
    initial_child_conductance_log_sd: float = 0.0
    print_hooks: bool = False
    store_routing: bool = False
    compile_forward: bool = False


@dataclass
class EncoderParamsConfig(BaseConfig):
    """Encoder network parameters."""

    input_dim: Optional[int] = None
    input_shape: list[int] = field(default_factory=lambda: [3, 32, 32])
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
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
    # Non-negative input contract for positive-conductance spatial cores.
    # ``signed_split`` emits concatenated positive and negative channels.
    input_transform: str = "identity"
    # Fixed positive drive calibration applied after the non-negative input
    # transform. Pretrained conv feature maps exceed the [0, 1] activity range
    # the dendritic conductance initialization assumes; a boundary-specific
    # scale keeps the shunting denominator near its design operating point.
    input_scale: float = 1.0
    # Optional channel-preserving adapter at the dendritic-convolution output.
    # Threshold-ReLU modes can match a downstream sparse activation contract
    # without introducing cross-channel mixing.
    output_adapter_mode: str = "identity"
    output_initial_scale: float = 1.0
    output_initial_threshold: float = 0.25


@dataclass
class CoreConfig(BaseConfig):
    """Core network configuration."""

    type: str = "einet"
    # Opt-in ordinary dense/fixed/Top-K MLP control; existing defaults unchanged.
    affine_mlp: Optional[AffineMLPConfig] = None
    shared_projection_shunting: Optional[SharedProjectionShuntingConfig] = None
    # Optional explicit learned-scalar budget for fixed-output sparse point
    # controls.  Structured dendritic configurations normally leave this
    # unset, in which case ``sparse_active_matched_point`` derives its budget
    # from the referenced E/I network.
    target_active_parameters: Optional[int] = None
    # Declares a biologically constrained E/I network: building the core fails
    # unless it has nonnegative synaptic weights, separate excitation and
    # inhibition (Dale's law), a rectified input and nonnegative population
    # outputs. See networks.architectures.biological_constraints. Only
    # dendritic E/I core types can declare it.
    biological_neuron: bool = False
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
    # Config-driven end-to-end vision architectures assembled from dendritic
    # convolutional and feedforward blocks.
    vision: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoderParamsConfig(BaseConfig):
    """Decoder network parameters."""

    input_dim: int = 20
    hidden_dims: list[int] = field(default_factory=lambda: [16])
    activation: str = "relu"
    # Identity decoders: None unless written; then it must equal the core output.
    output_dim: Optional[int] = 10
    input_normalization: str = "none"
    input_norm_eps: float = 1e-5
    output_init: str = "default"


@dataclass
class DecoderConfig(BaseConfig):
    """Decoder network configuration."""

    type: str = "MLP"
    params: DecoderParamsConfig = field(default_factory=DecoderParamsConfig)


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


OUTPUT_SCALE_MODES = ("per_class", "global", "fixed")


def resolve_output_scale_mode(output_scale_mode: Optional[str]) -> str:
    """Return the classifier output-scale mode; unset (None) is ``per_class``."""
    if output_scale_mode is None:
        return "per_class"
    mode = str(output_scale_mode).strip().lower()
    if mode not in OUTPUT_SCALE_MODES:
        raise ValueError(
            f"model.output_scale_mode must be one of {list(OUTPUT_SCALE_MODES)}, "
            f"got {output_scale_mode!r}"
        )
    return mode


def fold_learned_output_scale(
    learned_output_scale: Optional[bool], output_scale_mode: Optional[str]
) -> Optional[str]:
    """Translate the retired ``learned_output_scale`` into ``output_scale_mode``.

    False was ``output_scale_mode: fixed`` and true the default. A value that
    contradicts a written mode raises.
    """
    if learned_output_scale is None:
        return output_scale_mode
    if output_scale_mode is None:
        return None if learned_output_scale else "fixed"
    mode = resolve_output_scale_mode(output_scale_mode)
    if bool(learned_output_scale) != (mode != "fixed"):
        raise ValueError(
            f"model.learned_output_scale={learned_output_scale} contradicts "
            f"model.output_scale_mode={mode!r}, which "
            f"{'fixes' if mode == 'fixed' else 'learns'} the output scale; set "
            "only output_scale_mode"
        )
    return output_scale_mode


@dataclass
class ModelConfig(BaseConfig):
    """Complete model configuration."""

    task: str = "classification"
    # Optional strict warm start for the complete constructed model. A hash can
    # be required so delayed scheduler jobs cannot silently consume a changed
    # checkpoint.
    initial_checkpoint: Optional[str] = None
    initial_checkpoint_sha256: Optional[str] = None
    initial_checkpoint_strict: bool = True
    # Fixed positive temperature that multiplies the decoder output of every
    # feedforward classifier, whatever the decoder type.
    fixed_output_scale: float = 1.0
    # Learned scale of a feedforward classifier with an identity decoder:
    # ``per_class`` (one positive scale per output), ``global`` (one positive
    # scalar) or ``fixed`` (none). Other decoders learn no scale, so a written
    # learned mode raises for them. None (the default) is ``per_class`` for an
    # identity decoder and no scale otherwise. Replaces learned_output_scale
    # (retired 2026-10-08; false was ``fixed``).
    output_scale_mode: Optional[str] = None

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    # Slots for sections that plugins provide (dendritic_modeling.config.extensions).
    # ``pretrained_replacement`` is the older spelling of ``vision_replacement``.
    vision_replacement: Any = field(
        default_factory=extension_default("model.vision_replacement")
    )
    pretrained_replacement: Any = field(
        default_factory=extension_default("model.pretrained_replacement")
    )
    # The library's section, unless a plugin provides an extended one.
    transformer_replacement: Any = field(
        default_factory=extension_default(
            "model.transformer_replacement", TransformerReplacementConfig
        )
    )

    def __post_init__(self) -> None:
        resolve_output_scale_mode(self.output_scale_mode)
