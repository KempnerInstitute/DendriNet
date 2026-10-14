from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.choices import normalize_choice
from dendritic_modeling.config.extensions import extension_default
from dendritic_modeling.config.keys import build_nested_sections, check_section_keys
from dendritic_modeling.config.multi_stage_training import MultiStageStrategyConfig
from dendritic_modeling.config.regularization import (
    PruningConfig,
    RegularizationConfig,
    SelectiveConfig as _SelectiveConfig,
)

# Backward-compatible re-export: downstream code imports SelectiveConfig
# from config.training in several places.
SelectiveConfig = _SelectiveConfig

# Values accepted by the local-rule trainer (training.strategies.local_learning).
LOCAL_RULE_VARIANTS = ("3f", "4f", "5f", "stdp")
LOCAL_RULE_STDP_ALIASES = {
    "spike_timing": "stdp",
    "spike_timing_dependent_plasticity": "stdp",
    "trace_stdp": "stdp",
}
# "<rule>_vh" enables strictly local voltage homeostasis with these settings.
LOCAL_RULE_VH_ALIASES = {
    f"{rule}{sep}vh": rule for rule in ("3f", "4f", "5f") for sep in ("_", "+")
}
VOLTAGE_HOMEOSTASIS_ALIAS_WEIGHT = 0.02
VOLTAGE_HOMEOSTASIS_ALIAS_TARGET = 0.5
LOCAL_ERROR_MODES = ("auto", "mse", "ce", "bce")
LOCAL_ERROR_BROADCAST_MODES = (
    "scalar",
    "per_soma",
    "per_soma_shared",
    "per_soma_tree",
    "per_soma_shuffled",
    "local_mismatch",
    "low_rank",
    "path_transport",
    "pathway_vector",
)
LOCAL_BROADCAST_BANDWIDTHS = ("full", "sign_only", "quantized", "sparse_topk")
LOCAL_ENCODER_UPDATE_MODES = ("backprop", "none")
LOCAL_DECODER_UPDATE_MODES = ("backprop", "local", "none")
LOCAL_SCHEDULE_KEYS = {
    "start_epoch",
    "end_epoch",
    "update_reactivation",
    "encoder_update_mode",
    "decoder_update_mode",
}


@dataclass
class ParamGroupsConfig(BaseConfig):
    """Parameter groups with different learning rates."""

    lr: float = 0.01
    split_params: bool = True
    topk_lr: float = 0.001
    blocklinear_lr: float = 0.0001
    reactivation_lr: float = 0.0001
    decoder_lr: float = 0.001
    # Optional LR for only the first decoder Linear weight. This supports
    # width-scaled updates without slowing decoder biases or the output head.
    decoder_input_lr: Optional[float] = None
    encoder_lr: Optional[float] = None  # None = use lr (param_groups.lr)


GRAD_CLIP_MODES = ("value", "norm")
# Strategies whose training loops clip the global gradient norm; the others
# clip each gradient element by value.
_NORM_CLIPPING_STRATEGIES = frozenset({"recurrent"})


def default_grad_clip_mode(strategy: Any, distributed_mode: Any = None) -> str:
    """Return the clipping operator a strategy applies when none is configured."""
    if str(distributed_mode or "").strip().lower() == "fsdp":
        return "norm"
    if str(strategy or "").strip().lower() in _NORM_CLIPPING_STRATEGIES:
        return "norm"
    return "value"


BEST_STATE_DEVICES = ("model", "cpu")
BEST_CHECKPOINT_SAVES = ("every_improvement", "end", "never")


@dataclass
class CommonTrainerConfig(BaseConfig):
    """Common training parameters shared across strategies."""

    RENAMED_FIELDS: ClassVar[dict[str, tuple[str, str]]] = {
        "weight_decay_rate": ("synapse_decay_rate", "2026-10-08"),
    }

    epochs: int = 1000
    batch_size: int = 512
    shuffle: bool = True
    # Gradient clipping threshold; 0 or None disables clipping.
    grad_clip_value: Optional[float] = 2.0
    # "value" clamps each gradient element to +-grad_clip_value; "norm" rescales
    # all gradients so their global L2 norm is at most grad_clip_value. None
    # uses the strategy's operator: norm for strategy recurrent and for FSDP,
    # value otherwise. Training records the resolved mode in config.json.
    grad_clip_mode: Optional[str] = None
    loss_function: str = "ce"
    early_stopping: bool = True
    patience: int = 500
    load_best_state_dict: bool = True
    # Where the best-validation weights are kept during training: "model"
    # (the model's device) or "cpu" (host memory; the same values without a
    # second copy on the GPU).
    best_state_device: str = "model"
    # When <prefix>best_model.pt is written: "every_improvement", "end" (once,
    # when the training phase ends) or "never".
    save_best_checkpoint: str = "every_improvement"
    plot_losses: bool = True
    suppress_prints: bool = False
    print_every: int = 10
    # DataLoader controls. None keeps the trainer's automatic behavior.
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    in_order: Optional[bool] = None
    use_amp: bool = False
    lr_schedule: str = "none"
    lr_warmup_epochs: int = 0
    # Zero NaN and infinite gradient entries before the optimizer step (after
    # AMP unscaling, so the scaler still skips overflowing steps).
    enable_nan_checking: bool = False
    profile_dataloader: bool = False
    # Record compact decoder-input, logit, and gradient statistics for the
    # first FSDP optimizer step. Disabled by default to avoid extra reductions.
    log_first_step_diagnostics: bool = False
    # Data-driven reactivation initialization can retain intermediate voltage
    # tensors for every dendritic layer. These controls bound that one-time
    # calibration for deep or spatial models without changing its estimator.
    reactivation_initialization_batch_size: int = 256
    reactivation_initialization_num_batches: int = 3
    # None iterates to convergence with the package default cap.
    reactivation_initialization_max_iterations: Optional[int] = None
    # Sparse-synapse maintenance applied before every optimizer step: active
    # sparse synapse weights decay by lr * synapse_decay_rate, and with
    # weight_boosting unselected candidates are boosted by the same factor.
    # This is separate from optimizer.weight_decay (L2 / AdamW decay) and
    # regularization.l2_weight (an L2 term in the loss).
    synapse_decay_rate: float = 0.01
    weight_boosting: bool = False
    # Periodic reactivation recalibration during training.
    # 0 = disabled (default). N > 0 = recompute (m, b) from the current
    # voltage distribution every N epochs using the same calibration rule
    # as the configured reactivation init policy.
    recalibrate_reactivation_every: int = 0
    # How reactivation parameters (m, b) are updated during training:
    # "backprop" = by the optimizer or local rule (default); "quantile" = only
    # by periodic recalibration (requires recalibrate_reactivation_every > 0);
    # "frozen" = not at all.
    reactivation_update_mode: str = "backprop"
    # Calibration rule of periodic refreshes: "from_init_policy" (each layer's
    # initialization rule; "occupancy_quantile" under
    # reactivation_update_mode="quantile"), "occupancy_quantile",
    # "median_mad" or "mean_std".
    reactivation_recalibration_mode: str = "from_init_policy"
    # First epoch (counted from 1) at whose start periodic refreshes may run.
    # 1 refreshes before any training; 2 keeps the initialization through
    # epoch 1.
    reactivation_recalibration_start_epoch: int = 1
    # Number of training batches used to estimate the live voltage
    # distribution for a periodic (m, b) refresh.
    reactivation_recalibration_num_batches: int = 3
    # EMA-style blend factor for periodic reactivation refreshes.
    # 1.0 = hard overwrite with the freshly calibrated (m, b).
    # 0 < alpha < 1 = blend the current parameters toward the calibrated target.
    reactivation_recalibration_ema_alpha: float = 1.0

    param_groups: ParamGroupsConfig = field(default_factory=ParamGroupsConfig)
    # Cellwise impact rewiring of indexed_release synapses after every update
    # (synapse.impact_rewiring.ImpactRewiringConfig fields, plus
    # freeze_last_epochs). Empty disables it. Requires a population_network core.
    impact_rewiring: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.grad_clip_mode is not None:
            self.grad_clip_mode = normalize_choice(
                self.grad_clip_mode,
                GRAD_CLIP_MODES,
                field="training.<phase>.common.grad_clip_mode",
            )
        if self.weight_boosting and float(self.synapse_decay_rate) <= 0:
            raise ValueError(
                "training.<phase>.common.weight_boosting scales candidates by the "
                "synapse_decay_rate factor, so it has no effect with "
                f"synapse_decay_rate={self.synapse_decay_rate}; set a positive "
                "rate or weight_boosting: false"
            )
        if self.grad_clip_value is not None and float(self.grad_clip_value) < 0:
            raise ValueError(
                "training.<phase>.common.grad_clip_value must be >= 0 (0 or null "
                f"disables clipping), got {self.grad_clip_value!r}"
            )
        if self.impact_rewiring:
            self._check_impact_rewiring()
        self.best_state_device = normalize_choice(
            self.best_state_device,
            BEST_STATE_DEVICES,
            field="training.<phase>.common.best_state_device",
        )
        self.save_best_checkpoint = normalize_choice(
            self.save_best_checkpoint,
            BEST_CHECKPOINT_SAVES,
            field="training.<phase>.common.save_best_checkpoint",
        )
        from dendritic_modeling.config.reactivation import validate_reactivation_updates

        (
            self.reactivation_update_mode,
            self.reactivation_recalibration_mode,
        ) = validate_reactivation_updates(
            update_mode=self.reactivation_update_mode,
            recalibration_mode=self.reactivation_recalibration_mode,
            recalibrate_every=self.recalibrate_reactivation_every,
            start_epoch=self.reactivation_recalibration_start_epoch,
            num_batches=self.reactivation_recalibration_num_batches,
            ema_alpha=self.reactivation_recalibration_ema_alpha,
        )

    def _check_impact_rewiring(self) -> None:
        """Validate impact-rewiring options at load instead of at trainer setup."""
        from dendritic_modeling.config.impact_rewiring import ImpactRewiringConfig

        section = "training.<phase>.common.impact_rewiring"
        options = dict(self.impact_rewiring)
        derived = sorted({"until_step", "seed"} & set(options))
        if derived:
            raise ValueError(
                f"{section}.{derived[0]} is derived; use freeze_last_epochs or the "
                "experiment topology seed"
            )
        freeze = options.pop("freeze_last_epochs", 0)
        if isinstance(freeze, bool) or not isinstance(freeze, int) or freeze < 0:
            raise ValueError(f"{section}.freeze_last_epochs must be an integer >= 0")
        allowed = {item.name for item in fields(ImpactRewiringConfig)}
        check_section_keys(options, allowed - {"until_step", "seed"}, section=section)
        ImpactRewiringConfig(**options)


@dataclass
class StandardStrategyConfig(BaseConfig):
    """Standard backpropagation strategy configuration."""

    # Uses common config only


# Activations a voltage-stabilization warm-up drives: the gate input V
# ("vinf"), its output r ("vout"), both, or the gate-output occupancy.
STABILIZE_MODES = ("vinf", "vout", "both", "gate_occupancy")


def _check_warmup_strategy(config, strategy: str) -> None:
    section = f"training.main.strategies.{strategy}"
    if int(config.pretrain_epochs) < 1:
        raise ValueError(
            f"{section}.pretrain_epochs={config.pretrain_epochs}: the warm-up runs "
            "at least one epoch; use strategy 'standard' to train without it"
        )
    if not hasattr(config, "stabilize_mode"):
        return
    config.stabilize_mode = normalize_choice(
        config.stabilize_mode, STABILIZE_MODES, field=f"{section}.stabilize_mode"
    )
    if not 0.0 <= float(config.target_saturation) <= 1.0:
        raise ValueError(
            f"{section}.target_saturation is a fraction of gate outputs, got "
            f"{config.target_saturation}"
        )


@dataclass
class VoltageStabilizationStrategyConfig(BaseConfig):
    """Voltage stabilization strategy configuration.

    Only explicitly set fields reach the trainer, so these defaults equal the
    runtime defaults of ``VoltageStabilizationTrainer``.
    """

    # Warm-up epochs (at least 1) before standard training.
    pretrain_epochs: int = 1
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    # Fraction of gate outputs near either rail (gate_occupancy), in [0, 1].
    target_saturation: float = 0.2
    loss_metric: str = "mse"

    def __post_init__(self) -> None:
        _check_warmup_strategy(self, "voltage_stabilization")


@dataclass
class HomeostaticControlStrategyConfig(BaseConfig):
    """Homeostatic control strategy configuration.

    Defaults equal the runtime defaults of ``HomeostaticControlTrainer``.
    """

    # Warm-up epochs (at least 1) before standard training.
    pretrain_epochs: int = 1
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    # Fraction of gate outputs near either rail (gate_occupancy), in [0, 1].
    target_saturation: float = 0.2
    loss_metric: str = "mse"

    def __post_init__(self) -> None:
        _check_warmup_strategy(self, "homeostatic_control")


@dataclass
class EIEquilibriumStrategyConfig(BaseConfig):
    """E/I equilibrium warm-up configuration.

    The warm-up drives each branch's excitatory conductance output toward its
    inhibitory one before standard training. Defaults equal the runtime
    defaults of ``ExcitationInhibitionEquilibriumTrainer``.
    """

    # Warm-up epochs (at least 1) before standard training.
    pretrain_epochs: int = 1
    freeze_reactivation: bool = False
    loss_metric: str = "mse"

    def __post_init__(self) -> None:
        _check_warmup_strategy(self, "ei_equilibrium")


@dataclass
class RecurrentStrategyConfig(BaseConfig):
    """Recurrent trainer-specific overrides of ``training.main.common``.

    ``None`` inherits the common setting of the same name (``grad_clip_value``,
    ``use_amp``, ``lr_schedule`` and ``lr_warmup_epochs``). A set value takes
    precedence. Because unset fields stay ``None``, a saved config.json replays
    with the same precedence.
    """

    RENAMED_FIELDS: ClassVar[dict[str, tuple[str, str]]] = {
        "grad_clip": ("grad_clip_value", "2026-10-10"),
    }

    grad_clip_value: Optional[float] = None
    use_amp: Optional[bool] = None
    lr_schedule: Optional[str] = None
    lr_warmup_epochs: Optional[int] = None


@dataclass
class PresentationStreamStrategyConfig(BaseConfig):
    """Epochs of freshly sampled presentations (training.strategies.presentation_stream).

    The learning rate follows the common ``lr_schedule`` and
    ``lr_warmup_epochs`` per optimizer step: ``cosine`` falls from the initial
    rate at the first step to ``lr_terminal_multiplier`` times it at the last
    step of all ``epochs``, and during the first ``lr_warmup_epochs`` (fractions
    allowed) it is multiplied by a linear ramp. ``cosine`` needs
    ``lr_terminal_multiplier`` below 1, and a warmup needs ``cosine``.
    """

    presentations_per_epoch: int = 65536
    lr_terminal_multiplier: float = 1.0
    sample_seed_offset: int = 4100
    noise_seed_offset: int = 9100
    selection_metric: str = "accuracy"
    selection_variant: Optional[str] = None  # "noisy" if present, else the only set
    evaluation_batch_size: int = 512
    require_finite: bool = True
    metrics_file: str = "metrics.jsonl"
    # Train only the first N epochs of the full schedule; 0 runs all epochs.
    stop_after_epochs: int = 0


@dataclass
class FreezeLayersStrategyConfig(BaseConfig):
    """Freeze layers strategy configuration.

    Defaults equal the runtime defaults of ``LayerWiseTrainer``.
    """

    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_layer: int = 5
    final_tune_epochs: int = 10


@dataclass
class FreezeBranchesStrategyConfig(BaseConfig):
    """Freeze branches strategy configuration.

    ``pretrain_epochs`` equals the ``TwoStepTrainer`` runtime default.
    """

    pretrain_epochs: int = 5
    # Not implemented by the two-step trainers: values other than these
    # defaults raise ValueError instead of being ignored.
    reverse_training: bool = False
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10


@dataclass
class FreezeBranchKLStrategyConfig(BaseConfig):
    """Freeze branch KL strategy configuration.

    ``pretrain_epochs`` equals the ``TwoStepTrainerWithKL`` runtime default.
    """

    pretrain_epochs: int = 5
    # Not implemented (see FreezeBranchesStrategyConfig); non-defaults raise.
    reverse_training: bool = False
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10
    kl_weight: float = 0.1


@dataclass
class TrainingStrategiesConfig(BaseConfig):
    """Training strategies configuration."""

    standard: StandardStrategyConfig = field(default_factory=StandardStrategyConfig)
    multi_stage: MultiStageStrategyConfig = field(
        default_factory=MultiStageStrategyConfig
    )
    recurrent: RecurrentStrategyConfig = field(default_factory=RecurrentStrategyConfig)
    voltage_stabilization: VoltageStabilizationStrategyConfig = field(
        default_factory=VoltageStabilizationStrategyConfig
    )
    homeostatic_control: HomeostaticControlStrategyConfig = field(
        default_factory=HomeostaticControlStrategyConfig
    )
    ei_equilibrium: EIEquilibriumStrategyConfig = field(
        default_factory=EIEquilibriumStrategyConfig
    )
    freeze_layers: FreezeLayersStrategyConfig = field(
        default_factory=FreezeLayersStrategyConfig
    )
    freeze_branches: FreezeBranchesStrategyConfig = field(
        default_factory=FreezeBranchesStrategyConfig
    )
    freeze_branch_kl: FreezeBranchKLStrategyConfig = field(
        default_factory=FreezeBranchKLStrategyConfig
    )
    presentation_stream: PresentationStreamStrategyConfig = field(
        default_factory=PresentationStreamStrategyConfig
    )
    # Settings of trainer strategies that plugins provide, by strategy name
    # (see dendritic_modeling.config.extensions).
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig(BaseConfig):
    """Configuration for optimizer selection and parameters."""

    name: str = "adam"  # Options: "adam", "sgd", "rmsprop", "adamw", "muonh"
    lr: float = 0.001
    weight_decay: float = 0.0  # Built-in L2 regularization (global)
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW
    eps: float = 1e-8  # For Adam/AdamW
    # ``None`` preserves PyTorch's automatic implementation choice.  An
    # explicit boolean is useful for prospectively matched experiments whose
    # optimizer execution contract requires the foreach path to be fixed.
    foreach: bool | None = None
    # MuonH options (Puro-2B, arXiv 2608.27370): Muon (momentum +
    # Newton-Schulz orthogonalized updates for 2D matrices) with a hyperball
    # projection back to each matrix's initial Frobenius norm, and an AdamW
    # fallback for non-2D / gate-like parameters.  Read ONLY when
    # ``name == "muonh"``; every other optimizer ignores these fields, so
    # existing configs are behaviorally untouched.
    muon_lr_multiplier: float = 10.0  # effective Muon LR = lr * multiplier
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_hyperball: bool = True
    # 2D matrices with min(shape) < muon_min_dim (e.g. [N, 1] single-contact
    # columns) fall back to AdamW alongside vectors and scalars.
    muon_min_dim: int = 2
    # First N distributed steps assert bitwise-identical Muon parameters
    # across ranks (Muon must run after the DDP gradient all-reduce).
    muon_rank_consistency_check_steps: int = 1


# Strategies whose trainers read training.main.learning_strategy_config
# (multi_stage hands it to local_ca stages without their own block). Other
# strategies do not receive it.
LOCAL_RULE_STRATEGIES = frozenset({"local_ca", "multi_stage"})


@dataclass
class MainTrainingConfig(BaseConfig):
    """Main network training configuration."""

    strategy: str = "standard"

    common: CommonTrainerConfig = field(default_factory=CommonTrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    strategies: TrainingStrategiesConfig = field(
        default_factory=TrainingStrategiesConfig
    )
    learning_strategy_config: Optional["LocalRuleConfig"] = field(
        default_factory=lambda: None
    )


@dataclass
class ThreeFactorConfig(BaseConfig):
    """Parameters for 3-factor (3F) base learning rule.

    The 3F rule uses: presynaptic * postsynaptic * error
    """

    # Compartmental dynamics: "auto" gives each layer the derivatives of its
    # own integration, conductance-aware for shunting layers and additive for
    # additive layers. "conductance" or "additive" must match every layer, or
    # raise, unless allow_mismatched_dynamics_mode is true: then that mode's
    # derivatives apply to every layer, a deliberate rule variant (for
    # example a fairness control) whose updates are not the derivatives of
    # the forward pass of the mismatched layers.
    dynamics_mode: str = "auto"  # "auto", "conductance", "additive"
    allow_mismatched_dynamics_mode: bool = False
    use_conductance_scaling: bool = True  # Scale by R_tot = 1/g_tot
    # False replaces the driving forces (E_rev - V) with the threshold factors
    # (V - theta) for excitation and (theta - V) for inhibition.
    use_driving_force: bool = True
    # Threshold of the use_driving_force=false variant and of the inhibitory
    # factor in additive input_dependent mode; not a reversal potential.
    theta: float = 0.0
    # The branch forward fixes the reversal potentials: excitation drives the
    # compartment toward 1 and inhibition shunts toward 0. Other values raise.
    e_rev_exc: float = 1.0
    e_rev_inh: float = 0.0

    # Additive gain control mode (only used when dynamics_mode is "additive").
    # "none"            : post_factor = e_n  (current default, no local modulation)
    # "input_dependent" : post_factor = e_n * pseudo_R * pseudo_drive
    #                     pseudo_R = 1/(1 + |activations|), pseudo_drive = (v_mean - v_n + 1)
    # "running_stats"   : post_factor = e_n * (1 / running_std_n)
    additive_gain_mode: str = "none"
    additive_stats_ema_alpha: float = 0.01  # EMA rate for running_stats mode

    def __post_init__(self) -> None:
        self.dynamics_mode = normalize_choice(
            self.dynamics_mode,
            ("auto", "conductance", "additive"),
            field="three_factor.dynamics_mode",
            aliases={"legacy": "conductance"},
        )
        if self.allow_mismatched_dynamics_mode and self.dynamics_mode == "auto":
            raise ValueError(
                "three_factor.allow_mismatched_dynamics_mode needs dynamics_mode "
                "'conductance' or 'additive'"
            )
        if (float(self.e_rev_exc), float(self.e_rev_inh)) != (1.0, 0.0):
            raise ValueError(
                "three_factor.e_rev_exc and e_rev_inh must be 1.0 and 0.0, the "
                "reversal potentials of the branch-layer forward equations; other "
                "values give updates that are not derived from the model (got "
                f"{self.e_rev_exc!r} and {self.e_rev_inh!r})"
            )
        if str(self.additive_gain_mode).strip().lower() == "learned_gain":
            raise ValueError(
                "three_factor.additive_gain_mode 'learned_gain' was retired on "
                "2026-10-08: its gain never received a gradient, so it scaled "
                "every post factor by sigmoid(0) = 0.5. Use 'none', or replay the "
                "run at the library revision that produced it."
            )
        self.additive_gain_mode = normalize_choice(
            self.additive_gain_mode,
            ("none", "input_dependent", "running_stats"),
            field="three_factor.additive_gain_mode",
        )


@dataclass
class FourFactorConfig(BaseConfig):
    """Parameters for 4-factor (4F) morphology correlation.

    Adds morphology factor rho to 3F rule.
    """

    # Morphology correlation mode
    rho_mode: str = "pearson"  # "pearson", "dot", "none"
    # Estimator for single-sample scenarios
    rho_estimator: str = "ema"  # "ema", "augment", "ema+augment"
    # EMA smoothing rate for rho statistics
    ema_alpha: float = 0.05
    # Layer-wise scaling: rho / sqrt(layer_depth)
    layer_wise_rho_scale: float = 1.0
    # Augmentation parameters (for rho_estimator="augment")
    augment_k: int = 8  # Number of augmented samples
    augment_noise_sigma: float = 0.01  # Noise level

    def __post_init__(self) -> None:
        self.rho_mode = normalize_choice(
            self.rho_mode, ("pearson", "dot", "none"), field="four_factor.rho_mode"
        )
        self.rho_estimator = normalize_choice(
            self.rho_estimator,
            ("ema", "augment", "ema+augment"),
            field="four_factor.rho_estimator",
        )


@dataclass
class FiveFactorConfig(BaseConfig):
    """Parameters for 5-factor (5F) conditional information proxy.

    Adds information factor φ to 4F rule.
    """

    # Information proxy mode
    phi_mode: str = "conditional"  # "conditional", "variance"
    # Estimator for single-sample scenarios
    phi_estimator: str = "conditional_ema"  # "conditional_ema", "variance_ema"
    # Ridge regularization for conditional regression
    phi_ridge_lambda: float = 1e-3
    # Layer-wise scaling: φ / sqrt(layer_depth)
    layer_wise_phi_scale: float = 1.0
    # Stability clamp for the phi confidence factor
    phi_clamp_min: float = 0.25
    phi_clamp_max: float = 4.0

    def __post_init__(self) -> None:
        self.phi_mode = normalize_choice(
            self.phi_mode, ("conditional", "variance"), field="five_factor.phi_mode"
        )
        self.phi_estimator = normalize_choice(
            self.phi_estimator,
            ("conditional_ema", "variance_ema"),
            field="five_factor.phi_estimator",
        )


@dataclass
class MorphologyAwareConfig(BaseConfig):
    """Parameters for morphology-aware extensions.

    These extensions use dendritic tree topology explicitly.
    """

    # Path-integrated propagation: approximate ∏ R_k g_k attenuation
    use_path_propagation: bool = False
    path_factor_mode: str = "per_branch"  # "per_branch", "scalar_mean"

    # Branch-specific depth modulation: rho_j = rho_base / (depth_j + alpha)
    # "centrality" uses inverse branch depth normalized to unit mean.
    morphology_modulator_mode: str = "none"  # "none", "depth", "centrality"
    morphology_depth_offset: float = 1.0  # alpha parameter

    # Dendritic normalization: Δg_j ← Δg_j / (Σ_k g_k + ε)
    use_dendritic_normalization: bool = False

    # Apical vs basal branch differentiation
    use_branch_type_rules: bool = False
    apical_branch_scale: float = 1.0  # Scale for apical (feedback) branches
    basal_branch_scale: float = 1.0  # Scale for basal (feedforward) branches
    use_branch_length_modulation: bool = False

    # Router/pathway-derived branch roles (preferred over heuristic branch types)
    use_branch_role_rules: bool = False
    branch_role_source: str = "router_pathways"  # "router_pathways"
    specialized_branch_scale: float = 1.0
    mixed_branch_scale: float = 1.0
    branch_role_power: float = 1.0
    branch_role_alignment_weight: float = 0.0

    def __post_init__(self) -> None:
        self.path_factor_mode = normalize_choice(
            self.path_factor_mode,
            ("per_branch", "scalar_mean"),
            field="morphology_aware.path_factor_mode",
        )
        self.morphology_modulator_mode = normalize_choice(
            self.morphology_modulator_mode,
            ("none", "depth", "centrality"),
            field="morphology_aware.morphology_modulator_mode",
        )
        self.branch_role_source = normalize_choice(
            self.branch_role_source,
            ("router_pathways",),
            field="morphology_aware.branch_role_source",
        )


@dataclass
class HSICConfig(BaseConfig):
    """Parameters for HSIC auxiliary loss."""

    enabled: bool = False
    weight: float = 0.0
    self_weight: float = 1.0  # Self-decorrelation
    target_weight: float = 1.0  # Target-correlation
    target_source: str = "labels"  # "labels" or "logits"
    # Kernel configuration
    kernel: str = "linear"  # "linear", "rbf", "polynomial"
    sigma: float = 1.0  # RBF bandwidth
    degree: int = 2  # Polynomial degree
    coef0: float = 1.0  # Polynomial coefficient
    # Stability
    grad_clip_value: float = 0.0
    warmup_epochs: int = 0
    apply_last_layer_only: bool = False

    def __post_init__(self) -> None:
        self.target_source = normalize_choice(
            self.target_source, ("labels", "logits"), field="hsic.target_source"
        )
        self.kernel = normalize_choice(
            self.kernel,
            ("linear", "rbf", "polynomial"),
            field="hsic.kernel",
            aliases={"gaussian": "rbf", "poly": "polynomial"},
        )


@dataclass
class STDPConfig(BaseConfig):
    """Parameters for spike-timing / trace-based local plasticity.

    The rule is implemented at the same TopK synapse sites used by LocalCA. It
    can run alone via ``rule_variant: "stdp"`` or as an auxiliary update on top
    of 3F/4F/5F by setting ``stdp.enabled: true``.
    """

    enabled: bool = False
    # Pathways are TopK inputs recorded on each DendriticBranchLayer:
    # "exc", "inh", "rec_exc", "rec_inh".
    apply_to: list[str] = field(default_factory=lambda: ["exc", "rec_exc"])
    activity_mode: str = "relu"  # "relu", "identity", "binary", "abs"
    pre_threshold: float = 0.0
    post_threshold: float = 0.0
    tau_pre: float = 20.0
    tau_post: float = 20.0
    a_plus: float = 1.0
    a_minus: float = 0.5
    learning_rate_scale: float = 1.0
    inhibitory_update_sign: float = -1.0
    use_error_modulation: bool = False
    error_modulation_mode: str = "scalar_abs"  # "scalar_abs", "scalar_signed"
    clamp_update: float = 0.0
    detach_traces: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.apply_to, str):
            self.apply_to = [self.apply_to]
        elif self.apply_to is None:
            self.apply_to = ["exc", "rec_exc"]
        self.apply_to = [
            normalize_choice(
                pathway,
                ("exc", "inh", "rec_exc", "rec_inh", "all"),
                field="stdp.apply_to",
            )
            for pathway in self.apply_to
        ]
        self.activity_mode = normalize_choice(
            self.activity_mode,
            ("relu", "identity", "binary", "abs"),
            field="stdp.activity_mode",
            aliases={"absolute": "abs"},
        )
        self.error_modulation_mode = normalize_choice(
            self.error_modulation_mode,
            ("scalar_abs", "scalar_signed"),
            field="stdp.error_modulation_mode",
        )


@dataclass
class InhibitoryHomeostasisConfig(BaseConfig):
    """Parameters for local inhibitory homeostasis updates.

    These auxiliary updates are applied only to inhibitory synapses and are
    designed to steer branches toward stable shunting regimes using local state.
    """

    enabled: bool = False
    mode: str = "r_tot"  # "r_tot", "voltage"
    weight: float = 0.0
    target_r_tot: float = 0.35
    target_voltage: float = 0.15

    def __post_init__(self) -> None:
        self.mode = normalize_choice(
            self.mode, ("r_tot", "voltage"), field="inhibitory_homeostasis.mode"
        )


@dataclass
class LocalVoltageHomeostasisConfig(BaseConfig):
    """Parameters for strictly local voltage-centering updates.

    These auxiliary updates use only branch-local voltages and the local
    sensitivity terms already available to LocalCA. They are intended as a
    local counterpart to the separate voltage-stabilization pretraining
    strategy, not as a replacement for the task-driven credit signal.
    """

    enabled: bool = False
    weight: float = 0.0
    target_voltage: float = 0.5


@dataclass
class LocalGateHomeostasisConfig(BaseConfig):
    """Parameters for strictly local gate-occupancy updates.

    These updates act only on reactivation parameters and use only each
    layer's own gate outputs plus local activation derivatives. This is the
    local-learning counterpart to the global gate-occupancy warmup strategy.
    """

    enabled: bool = False
    center_weight: float = 0.0
    saturation_weight: float = 0.0
    target_mean: float = 0.5
    target_saturation: float = 0.2
    low_threshold: float = 0.1
    high_threshold: float = 0.9
    saturation_slope: float = 20.0


@dataclass
class LocalRuleConfig(BaseConfig):
    """Configuration for local learning rules.

    Organizes parameters by learning rule method for clarity.
    Each rule (3F/4F/5F/STDP) has its own nested configuration.
    """

    # ========== CORE CONFIGURATION ==========
    rule_variant: str = "3f"  # "3f", "4f", "5f", "stdp" (also accepts *_vh aliases)
    # The rules derive their updates from the branch-layer forward equations,
    # so local_ca raises for cores with typed synapses, a spiking soma,
    # dendritic spikes, soma feedback or the parallel_readout/all_active_star
    # cross-level modes, whose dynamics those equations omit. True applies the
    # rules anyway, as a rule variant whose updates ignore those features.
    allow_unmodeled_forward_features: bool = False
    # Path-based errors (error_broadcast_mode path_transport/pathway_vector and
    # morphology_aware.use_path_propagation) follow each compartment's path to
    # one soma; on models without that path (several trees, explicit
    # inhibitory cells, populations) they raise. True uses the scalar error,
    # or the unit path factor, where no path exists.
    allow_scalar_fallback: bool = False

    # ========== ERROR SIGNAL ==========
    # "auto" takes the rule from the configured loss (ce/cat_nll, bce, mse)
    # and raises for any other loss; "ce", "bce" and "mse" select it.
    error_mode: str = "auto"
    # Between-neuron production of the per-soma error, orthogonal to
    # error_broadcast_mode (which distributes it within each arbor).
    # "decoder": exact readout Jacobian (delta_core = J^T delta_out).
    # "dfa": fixed random feedback matrix in place of the decoder Jacobian,
    #        matching the SomaDFA construction (seed/scale conventions).
    soma_error_source: str = "decoder"
    dfa_feedback_seed: Optional[int] = None  # None -> trainer seed
    dfa_feedback_scale: float = 1.0
    error_broadcast_mode: str = (
        "scalar"  # "scalar", legacy "per_soma", "per_soma_shared", "per_soma_shuffled", "local_mismatch", "low_rank", "path_transport", "pathway_vector"
    )
    # Seed of the randomized feedback controls (per_soma_shuffled, low_rank);
    # None derives it from the run seed, so replicates differ.
    broadcast_seed: Optional[int] = None
    error_noise_sigma: float = (
        0.0  # Additive Gaussian noise on broadcast error (0 = off)
    )
    broadcast_rank: int = 4  # Rank for low-rank vectorized broadcast
    broadcast_init_scale: float = 1.0  # Scale for low-rank random broadcast
    pathway_broadcast_residual: float = 0.25  # Blend with direct local error
    pathway_activity_gate_strength: float = 1.0  # Modulate by local pathway activity

    # ========== BROADCAST BANDWIDTH CONTROL ==========
    # Reduce broadcast precision to test low-bandwidth biological plausibility
    broadcast_bandwidth: str = "full"  # "full", "sign_only", "quantized", "sparse_topk"
    broadcast_bits: int = (
        8  # Quantization bits (2, 4, 8); only used when bandwidth="quantized"
    )
    broadcast_topk_fraction: float = (
        0.3  # Fraction of neurons to keep; only for "sparse_topk"
    )

    # ========== RULE-SPECIFIC PARAMETERS ==========
    # 3-factor rule parameters
    three_factor: ThreeFactorConfig = field(default_factory=ThreeFactorConfig)

    # 4-factor rule parameters (includes 3F)
    four_factor: FourFactorConfig = field(default_factory=FourFactorConfig)

    # 5-factor rule parameters (includes 4F)
    five_factor: FiveFactorConfig = field(default_factory=FiveFactorConfig)

    # Morphology-aware extensions
    morphology_aware: MorphologyAwareConfig = field(
        default_factory=MorphologyAwareConfig
    )

    # HSIC auxiliary loss
    hsic: HSICConfig = field(default_factory=HSICConfig)

    # Trace-based spike-timing plasticity rule
    stdp: STDPConfig = field(default_factory=STDPConfig)

    # Local inhibitory homeostasis
    inhibitory_homeostasis: InhibitoryHomeostasisConfig = field(
        default_factory=InhibitoryHomeostasisConfig
    )

    # Strictly local voltage-centering auxiliary
    voltage_homeostasis: LocalVoltageHomeostasisConfig = field(
        default_factory=LocalVoltageHomeostasisConfig
    )

    # Strictly local gate-occupancy auxiliary
    gate_homeostasis: LocalGateHomeostasisConfig = field(
        default_factory=LocalGateHomeostasisConfig
    )

    # ========== PARAMETER UPDATE CONTROL ==========
    update_inactive_weights: bool = False
    update_reactivation: bool = True
    # LocalCA-only policy for explicit inhibitory DendriNet populations.
    # "local_ca": update I-cell dendrites with the same local rule machinery
    #             used for E-cell dendrites. This is the matched/comparable mode.
    # "freeze": skip LocalCA updates for explicit I-cell dendrites while still
    #           updating I-to-E synapses on excitatory dendrites.
    explicit_inhibitory_update_mode: str = "local_ca"
    # Deprecated compatibility shim. If set by old YAMLs, it is converted to
    # explicit_inhibitory_update_mode in __post_init__.
    update_explicit_inhibitory_cells: Optional[bool] = None
    encoder_update_mode: str = "none"  # "backprop", "none"
    decoder_update_mode: str = "backprop"  # "backprop", "local", "none"

    # ========== TRAINING SCHEDULE ==========
    freeze_encoder_epochs: int = 0
    freeze_reactivation_epochs: int = 0
    freeze_decoder_epochs: int = 0
    training_schedule: Optional[list[dict[str, Any]]] = None

    # ========== OPTIMIZATION ==========
    clip_grad_value: float = 5.0
    normalize_by_batch: bool = True

    def __post_init__(self) -> None:
        build_nested_sections(self, section="learning_strategy_config")

        normalized_source = str(self.soma_error_source).strip().lower()
        if normalized_source not in {"decoder", "dfa"}:
            raise ValueError(
                "soma_error_source must be 'decoder' or 'dfa', got "
                f"{self.soma_error_source!r}"
            )
        self.soma_error_source = normalized_source
        if self.dfa_feedback_scale <= 0:
            raise ValueError("dfa_feedback_scale must be positive")

        raw_rule = str(self.rule_variant).strip().lower()
        vh_base = LOCAL_RULE_VH_ALIASES.get(raw_rule.replace("-", "_"))
        if vh_base is not None:
            self.rule_variant = vh_base
            homeostasis = self.voltage_homeostasis
            if not homeostasis.enabled:
                homeostasis.enabled = True
                if homeostasis.weight == 0.0:
                    homeostasis.weight = VOLTAGE_HOMEOSTASIS_ALIAS_WEIGHT
        else:
            self.rule_variant = normalize_choice(
                raw_rule,
                LOCAL_RULE_VARIANTS,
                field="rule_variant",
                aliases=LOCAL_RULE_STDP_ALIASES,
            )
        if self.rule_variant == "stdp":
            self.stdp.enabled = True

        self.error_mode = normalize_choice(
            self.error_mode, LOCAL_ERROR_MODES, field="error_mode"
        )
        self.error_broadcast_mode = normalize_choice(
            self.error_broadcast_mode,
            LOCAL_ERROR_BROADCAST_MODES,
            field="error_broadcast_mode",
        )
        self.broadcast_bandwidth = normalize_choice(
            self.broadcast_bandwidth,
            LOCAL_BROADCAST_BANDWIDTHS,
            field="broadcast_bandwidth",
        )
        if int(self.broadcast_bits) < 1:
            raise ValueError("broadcast_bits must be at least 1")
        if not 0.0 < float(self.broadcast_topk_fraction) <= 1.0:
            raise ValueError("broadcast_topk_fraction must be in (0, 1]")
        if int(self.broadcast_rank) < 1:
            raise ValueError("broadcast_rank must be at least 1")
        self.encoder_update_mode = normalize_choice(
            self.encoder_update_mode,
            LOCAL_ENCODER_UPDATE_MODES,
            field="encoder_update_mode",
        )
        self.decoder_update_mode = normalize_choice(
            self.decoder_update_mode,
            LOCAL_DECODER_UPDATE_MODES,
            field="decoder_update_mode",
        )
        if self.training_schedule is not None:
            self.training_schedule = [
                _normalize_schedule_entry(entry, index)
                for index, entry in enumerate(self.training_schedule)
            ]

        mode = self._normalize_explicit_inhibitory_update_mode(
            self.explicit_inhibitory_update_mode
        )
        legacy_update = self.update_explicit_inhibitory_cells
        if legacy_update is not None:
            legacy_mode = "local_ca" if bool(legacy_update) else "freeze"
            if mode != "local_ca" and mode != legacy_mode:
                raise ValueError(
                    "Conflicting explicit inhibitory update settings: "
                    f"explicit_inhibitory_update_mode={mode!r} but "
                    f"update_explicit_inhibitory_cells={legacy_update!r}"
                )
            mode = legacy_mode

        self.explicit_inhibitory_update_mode = mode
        self.update_explicit_inhibitory_cells = mode == "local_ca"

    @staticmethod
    def _normalize_explicit_inhibitory_update_mode(mode: Any) -> str:
        if isinstance(mode, bool):
            return "local_ca" if mode else "freeze"
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "local_ca": "local_ca",
            "same_as_excitatory": "local_ca",
            "same_local_rule": "local_ca",
            "update": "local_ca",
            "train": "local_ca",
            "true": "local_ca",
            "freeze": "freeze",
            "none": "freeze",
            "off": "freeze",
            "false": "freeze",
        }
        if normalized not in aliases:
            allowed = "'local_ca' or 'freeze'"
            raise ValueError(
                f"Invalid explicit_inhibitory_update_mode={mode!r}; expected {allowed}"
            )
        return aliases[normalized]


def _normalize_schedule_entry(entry: Any, index: int) -> dict[str, Any]:
    """Validate one local-rule training_schedule entry (an inclusive epoch range)."""
    field = f"training_schedule[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{field} must be a mapping")
    unknown = sorted(set(entry) - LOCAL_SCHEDULE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown {field} option(s): {', '.join(unknown)}; valid keys: "
            + ", ".join(sorted(LOCAL_SCHEDULE_KEYS))
        )
    normalized = dict(entry)
    for key in ("start_epoch", "end_epoch"):
        if key in normalized:
            value = normalized[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field}.{key} must be an integer, got {value!r}")
    start = normalized.get("start_epoch", 1)
    end = normalized.get("end_epoch")
    if end is not None and end < start:
        raise ValueError(f"{field}: end_epoch {end} is before start_epoch {start}")
    if "encoder_update_mode" in normalized:
        normalized["encoder_update_mode"] = normalize_choice(
            normalized["encoder_update_mode"],
            LOCAL_ENCODER_UPDATE_MODES,
            field=f"{field}.encoder_update_mode",
        )
    if "decoder_update_mode" in normalized:
        normalized["decoder_update_mode"] = normalize_choice(
            normalized["decoder_update_mode"],
            LOCAL_DECODER_UPDATE_MODES,
            field=f"{field}.decoder_update_mode",
        )
    return normalized


ENCODER_PRETRAINING_LOSSES = ("mse", "log_mse", "bce", "vae_elbo")


@dataclass
class EncoderPretrainingConfig(BaseConfig):
    """Reconstruction pretraining of an autoencoder encoder.

    ``training.encoder_pretraining`` runs before the main phase when
    ``model.encoder`` is an autoencoder (``mlpautoencoder``,
    ``cnnautoencoder`` or a variational form) and ``epochs`` is positive: the
    whole autoencoder learns to reconstruct the training inputs, keeping the
    state with the lowest validation loss when ``early_stopping`` is on. Its
    encoder is the model's encoder in the main phase. With ``freeze_encoder``
    the encoder receives no gradients and stays in evaluation mode (a
    variational encoder returns its latent mean); otherwise it trains with
    the model at ``training.main.common.param_groups.encoder_lr``. The section
    has no effect with other encoders, and a positive ``epochs`` then raises.
    """

    epochs: int = 0
    batch_size: int = 256
    # "mse", "log_mse", "bce", or "vae_elbo" (variational autoencoders only).
    loss_function: str = "mse"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    early_stopping: bool = True
    patience: int = 10
    # Gradient clipping threshold (elementwise); null disables clipping.
    grad_clip_value: Optional[float] = None
    freeze_encoder: bool = True

    def __post_init__(self) -> None:
        section = "training.encoder_pretraining"
        for name, minimum in (("epochs", 0), ("batch_size", 1), ("patience", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f"{section}.{name} must be an integer >= {minimum}, got {value!r}"
                )
        if not float(self.learning_rate) > 0:
            raise ValueError(f"{section}.learning_rate must be positive")
        if float(self.weight_decay) < 0:
            raise ValueError(f"{section}.weight_decay must be >= 0")
        if self.grad_clip_value is not None and not float(self.grad_clip_value) > 0:
            raise ValueError(
                f"{section}.grad_clip_value must be positive, or null to disable "
                "clipping"
            )
        self.loss_function = normalize_choice(
            self.loss_function,
            ENCODER_PRETRAINING_LOSSES,
            field=f"{section}.loss_function",
        )
        self.optimizer = str(self.optimizer).strip().lower()


@dataclass
class TrainingConfig(BaseConfig):
    """Complete training configuration."""

    main: MainTrainingConfig = field(default_factory=MainTrainingConfig)
    encoder_pretraining: EncoderPretrainingConfig = field(
        default_factory=EncoderPretrainingConfig
    )
    # Slots for sections that plugins provide (dendritic_modeling.config.extensions).
    transformer_replacement: Any = field(
        default_factory=extension_default("training.transformer_replacement")
    )
    vision_replacement: Any = field(
        default_factory=extension_default("training.vision_replacement")
    )
