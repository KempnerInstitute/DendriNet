"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""

import dataclasses
import difflib
import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional

from omegaconf import OmegaConf

from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.config.affine_mlp import AffineMLPConfig
from dendritic_modeling.config.analysis import (
    AnalysisConfig,
    AnalysisRuntimeConfig,
    CorrelationAnalysisParams,
    EvaluationRuntimeConfig,
    InformationAnalysisParams,
)
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.biological import resolve_biological_neuron
from dendritic_modeling.config.composition import apply_overrides, load_composed_yaml
from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_dict
from dendritic_modeling.config.core_sections import reject_unread_core_sections
from dendritic_modeling.config.extensions import (
    apply_config_migrations,
    build_extension_section,
    config_section_builder,
    registered_named_sections,
)
from dendritic_modeling.config.inhibitory_wiring import check_retired_inhibitory_routing
from dendritic_modeling.config.io_params import (
    IDENTITY_TYPES,
    is_saved_record,
    reject_unread_params,
)
from dendritic_modeling.config.keys import (
    METADATA_SECTION,
    build_section,
    check_section_keys,
    drop_retired_values_in,
    field_default,
    to_mapping,
)
from dendritic_modeling.config.legacy import (
    CORRELATION_FLAT_PARAMS,
    INFORMATION_FLAT_PARAMS,
    migrate_flat_params,
)
from dendritic_modeling.config.local_learning import (
    build_local_rule_config as _build_local_rule_config,
)
from dendritic_modeling.config.model import (
    ArchitectureConfig,
    BlockLinearConfig,
    ConnectivityConfig,
    CoreConfig,
    DecoderConfig,
    DecoderParamsConfig,
    EncoderConfig,
    EncoderParamsConfig,
    ImplementationConfig,
    ModelConfig,
    MorphologyConfig,
    ReactivationConfig,
    SparsityConfig,
    SpatialConfig,
    TransferFunctionConfig,
    TransformerReplacementConfig,
    fold_learned_output_scale,
)
from dendritic_modeling.config.model_aliases import get_core_morphology_alias_overrides
from dendritic_modeling.config.reactivation import (
    DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY,
    normalize_reactivation_init_policy,
    warn_unused_gate_init,
)
from dendritic_modeling.config.recurrence import (
    _STRUCTURED_EI_TYPES,
    is_recurrent_core_config,
)
from dendritic_modeling.config.recurrent_payloads import check_recurrent_payloads
from dendritic_modeling.config.regularization import PruningConfig, RegularizationConfig
from dendritic_modeling.config.shared_projection import SharedProjectionShuntingConfig
from dendritic_modeling.config.structured_connectivity import canonical_rule
from dendritic_modeling.config.training import (
    LOCAL_RULE_STRATEGIES,
    CommonTrainerConfig,
    EIEquilibriumStrategyConfig,
    EncoderPretrainingConfig,
    FreezeBranchesStrategyConfig,
    FreezeBranchKLStrategyConfig,
    FreezeLayersStrategyConfig,
    HomeostaticControlStrategyConfig,
    LocalRuleConfig,
    MainTrainingConfig,
    MultiStageStrategyConfig,
    OptimizerConfig,
    ParamGroupsConfig,
    PresentationStreamStrategyConfig,
    RecurrentStrategyConfig,
    StandardStrategyConfig,
    TrainingConfig,
    TrainingStrategiesConfig,
    VoltageStabilizationStrategyConfig,
)
from dendritic_modeling.config.validation import validate_loaded_config
from dendritic_modeling.plugins import load_plugins

logger = logging.getLogger(__name__)


@dataclass
class CheckpointingConfig(BaseConfig):
    """Periodic training checkpoints, written under the run directory."""

    enabled: bool = False
    save_every_n_epochs: int = 10

    def __post_init__(self) -> None:
        interval = self.save_every_n_epochs
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError(
                "experiment.checkpointing.save_every_n_epochs must be an integer "
                f">= 1, got {interval!r}"
            )


@dataclass
class ExperimentConfig(BaseConfig):
    """Basic experiment settings."""

    seed: int = 42
    # Independent confirmatory RNG streams. ``None`` preserves legacy config
    # ergonomics and resolves to ``seed`` at runtime after CLI overrides.
    dataset_seed: Optional[int] = None
    split_seed: Optional[int] = None
    model_seed: Optional[int] = None
    topology_seed: Optional[int] = None
    loader_seed: Optional[int] = None
    evaluation_seed: Optional[int] = None
    probe_seed: Optional[int] = None
    strict_deterministic: bool = False
    # Optional reviewer-facing test seal. Legacy runs retain unrestricted test
    # access; confirmatory harnesses can expose only an explicitly capped audit
    # set and disable every analysis path that could adapt to test examples.
    # allow_test_data: false is part of the seal; max_test_samples bounds the
    # test split size whenever it is set.
    sealed_test: bool = False
    allow_test_data: bool = True
    max_test_samples: Optional[int] = None
    # Opt-in content hashes over the realized train/validation/test datasets.
    # Confirmatory generated-data runs use this to prove which exact tensors
    # reached the training process without serializing the examples twice.
    record_dataset_fingerprints: bool = False
    train_valid_split: float = 0.8
    fast_mode: bool = False
    # None = derived from fast_mode (false: deterministic, no cuDNN benchmark,
    # no TF32, "highest"; true: the opposite and "high"). Explicit values win;
    # training records the resolved values in config.json.
    deterministic: Optional[bool] = None
    cudnn_benchmark: Optional[bool] = None
    allow_tf32: Optional[bool] = None
    float32_matmul_precision: Optional[str] = None

    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    def __post_init__(self) -> None:
        if not self.allow_test_data and not self.sealed_test:
            raise ValueError(
                "experiment.allow_test_data: false takes effect only with "
                "sealed_test: true; set sealed_test: true to seal the test split, "
                "or remove allow_test_data"
            )
        if self.max_test_samples is not None and int(self.max_test_samples) < 0:
            raise ValueError("experiment.max_test_samples must be nonnegative")
        split = self.train_valid_split
        if isinstance(split, bool) or not 0.0 <= float(split) <= 1.0:
            raise ValueError(
                "experiment.train_valid_split is the fraction of the training data "
                f"used for training and must lie in [0, 1], got {split!r}"
            )


@dataclass
class DataProcessingConfig(BaseConfig):
    """Data processing parameters."""

    # Unset (None) leaves each loader's default: vision and derived datasets
    # flatten and do not normalize; sequential_mnist and stringer_v1 normalize.
    flatten: Optional[bool] = None
    normalize: Optional[bool] = None
    # Fraction of training labels replaced by random classes (0 = clean). Only
    # the mnist, fashion_mnist, cifar10 and cifar100 loaders apply it.
    label_noise_rate: float = 0.0

    def __post_init__(self) -> None:
        rate = self.label_noise_rate
        if isinstance(rate, bool) or not 0.0 <= float(rate) <= 1.0:
            raise ValueError(
                "data.processing.label_noise_rate must lie in [0, 1], " f"got {rate!r}"
            )


@dataclass
class DataConfig(BaseConfig):
    """Data configuration including dataset selection and processing."""

    dataset_name: str = "cifar10"
    base_dir: str = ""

    processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    # Options of each dataset, keyed by dataset name; only the block named by
    # dataset_name is read. Each loader owns its defaults and rejects keys it
    # does not read (see dendritic_modeling.datasets.parameters).
    dataset_params: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class OutputsConfig(BaseConfig):
    """Output configuration."""

    run_name: str = "ei_sweep"
    results_dir: str = "results"
    # Use ``results_dir`` itself as the run directory. This is intended for
    # manifest-controlled runs whose output path has already been atomically
    # claimed; the legacy timestamped behavior remains the default.
    exact_run_dir: bool = False


DISTRIBUTED_MODES = ("none", "ddp", "fsdp")
FSDP_SHARDING_STRATEGIES = ("FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD")
FSDP_BACKWARD_PREFETCH = ("BACKWARD_PRE", "BACKWARD_POST")
FSDP_DTYPE_NAMES = ("bf16", "bfloat16", "fp16", "float16", "half", "fp32", "float32")


def _choice(
    value: Any, choices: tuple[str, ...], path: str, *, upper: bool = False
) -> str:
    """Return ``value`` normalized to one of ``choices`` or raise naming ``path``."""
    text = str(value).strip()
    normalized = text.upper() if upper else text.lower()
    if normalized in choices:
        return normalized
    hint = difflib.get_close_matches(normalized, choices, n=1)
    raise ValueError(
        f"{path} must be one of {', '.join(choices)}, got {value!r}"
        + (f" (did you mean {hint[0]!r}?)" if hint else "")
    )


@dataclass
class FSDPConfig(BaseConfig):
    """FSDP-specific tuning knobs.

    These only take effect when ``distributed.mode`` is ``"fsdp"``. The model
    reaches FSDP on the CPU and each wrapped unit moves to its GPU while it is
    sharded.
    """

    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: bool = True
    cpu_offload: bool = False
    # "none": the whole model is one FSDP unit; "indexed_synapses": one unit
    # per indexed sparse synapse module (bounds parameter all-gathers for
    # contact-heavy models); "size": PyTorch's size-based policy, wrapping
    # submodules with at least ``min_num_params`` parameters.
    auto_wrap_policy: str = "none"
    min_num_params: int = 1000000
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    sync_module_states: bool = True
    use_orig_params: bool = True
    backward_prefetch: str = "BACKWARD_PRE"
    gradient_checkpointing: bool = False
    # Readout-only mode for contact-heavy recurrent models. Freezing before
    # FSDP wrapping avoids core gradients and optimizer state while preserving
    # the full dendritic forward dynamics.
    freeze_core: bool = False
    # Cache deterministic frozen-core outputs once per local dataset shard.
    # Later epochs train the readout without repeating the recurrent forward.
    cache_frozen_core_outputs: bool = False
    # Optional local batch size used after the frozen states are cached.
    cached_core_batch_size: Optional[int] = None
    # Mixed-precision dtypes (used when mixed_precision=True). The single knob
    # `mixed_precision_dtype` sets the parameter, gradient-reduction and buffer
    # dtypes together; the granular fields override it one by one. An unset
    # param_dtype follows the knob, and an unset reduce_dtype or buffer_dtype
    # follows param_dtype. fp16 parameters or reductions need loss scaling
    # (training.main.common.use_amp: true). buffer_dtype "keep" leaves buffers
    # in their own dtype; otherwise FSDP casts floating buffers for the whole
    # run without a full-precision copy (in-place statistics lose precision,
    # checkpoints store the rounded values).
    mixed_precision_dtype: str = "bf16"
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    buffer_dtype: Optional[str] = None

    def __post_init__(self) -> None:
        self.sharding_strategy = _choice(
            self.sharding_strategy,
            FSDP_SHARDING_STRATEGIES,
            "distributed.fsdp.sharding_strategy",
            upper=True,
        )
        if self.backward_prefetch is not None:
            self.backward_prefetch = _choice(
                self.backward_prefetch,
                FSDP_BACKWARD_PREFETCH,
                "distributed.fsdp.backward_prefetch",
                upper=True,
            )
        self.mixed_precision_dtype = _choice(
            self.mixed_precision_dtype,
            FSDP_DTYPE_NAMES,
            "distributed.fsdp.mixed_precision_dtype",
        )
        for name, choices in (
            ("param_dtype", FSDP_DTYPE_NAMES),
            ("reduce_dtype", FSDP_DTYPE_NAMES),
            ("buffer_dtype", (*FSDP_DTYPE_NAMES, "keep")),
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _choice(value, choices, f"distributed.fsdp.{name}"))
        policy = str(self.auto_wrap_policy).strip().lower()
        if policy not in ("none", "indexed_synapses", "size"):
            raise ValueError(
                "distributed.fsdp.auto_wrap_policy must be 'none', "
                f"'indexed_synapses' or 'size', got {self.auto_wrap_policy!r}"
            )
        self.auto_wrap_policy = policy
        if policy != "size" and int(self.min_num_params) != 1000000:
            raise ValueError(
                "distributed.fsdp.min_num_params is read only by "
                "auto_wrap_policy: size; it would be ignored with "
                f"auto_wrap_policy: {policy}"
            )
        if int(self.min_num_params) < 1:
            raise ValueError("distributed.fsdp.min_num_params must be positive")


@dataclass
class DistributedConfig(BaseConfig):
    """Distributed training configuration.

    ``mode`` declares the runtime; the training scripts check it against their
    launch and record the runtime they used:

        - ``"none"`` — single-GPU, plain ``python`` (default)
        - ``"ddp"``  — DistributedDataParallel via ``torchrun``
        - ``"fsdp"`` — FullyShardedDataParallel via ``torchrun``

    ``fsdp`` contains FSDP-specific tuning knobs (only used when mode=fsdp).

    Node and GPU counts are **not** part of training config — they are
    scheduler/launcher concerns.  Use ``sbatch --nodes=`` and
    ``--gpus-per-node=`` (or the script's CLI argument) to control resources.
    """

    mode: str = "none"  # none | ddp | fsdp
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    # Seconds a collective may wait before the process group aborts the run;
    # null keeps the entry point's default (PyTorch's 600 for DDP, 1800 for
    # FSDP). Rank-0-only work (training-time analyses, calibration) runs
    # while the other ranks wait in a collective, so long phases need more.
    timeout_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        self.mode = _choice(self.mode or "none", DISTRIBUTED_MODES, "distributed.mode")
        if self.timeout_seconds is not None and int(self.timeout_seconds) <= 0:
            raise ValueError(
                "distributed.timeout_seconds must be a positive number of "
                f"seconds, got {self.timeout_seconds!r}"
            )


#: Version of the configuration format. Version 3 (2026-10-13) is the meaning
#: this library gives every key. Files without ``config_version`` load with
#: that meaning; ``dendritic-upgrade-config`` rewrites files written for
#: earlier versions, whose omitted keys could mean something else, and marks
#: them ``config_version: 3``. Runs record the version in ``config.json``.
#:
#: Version 3 changed two omitted structured-connectivity keys: a ``distance``
#: rule without ``distance_kernel`` is gaussian in structured E/I cores as it
#: is in population rules (it was "none"), and training seeds an unseeded
#: population or unified E/I rule from the topology seed (its masks drew from
#: the global random stream).
CONFIG_VERSION = 3

#: Top-level key in which ``dendritic-upgrade-config`` lists what it could not
#: rewrite. A file with it is only partly upgraded and does not load.
UNRESOLVED_MIGRATION_KEY = "unresolved_migration"


def _reject_partial_migration(conf: Mapping[str, Any]) -> None:
    """Refuse a file that ``dendritic-upgrade-config`` upgraded only in part."""
    if UNRESOLVED_MIGRATION_KEY not in conf:
        return
    items = conf[UNRESOLVED_MIGRATION_KEY]
    items = items if isinstance(items, list) else [items]
    raise ValueError(
        "This configuration was only partly upgraded: dendritic-upgrade-config "
        "could not rewrite "
        + ("; ".join(str(item) for item in items) or "some of its settings")
        + f". Resolve each item, then delete {UNRESOLVED_MIGRATION_KEY}."
    )


def _check_config_version(value: Any) -> int:
    """Validate ``config_version``; files without it have the current meaning."""
    if value is None:
        return CONFIG_VERSION
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"config_version must be an integer, got {value!r}")
    if value > CONFIG_VERSION:
        raise ValueError(
            f"config_version {value} is newer than this library's configuration "
            f"format ({CONFIG_VERSION}); use a newer DendriNet"
        )
    if value < CONFIG_VERSION:
        raise ValueError(
            f"config_version {value} configurations predate the current meaning "
            f"of some keys; rewrite this file with dendritic-upgrade-config, "
            f"which writes config_version: {CONFIG_VERSION}"
        )
    return value


@dataclass
class Config:
    """Main configuration class with clean structure."""

    # The configuration format a file is written for (see CONFIG_VERSION).
    config_version: int = CONFIG_VERSION
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    # Annotations from people and tools (study contracts, sweep identifiers).
    # Holds the YAML ``metadata:`` mapping and every top-level key that starts
    # with an underscore. The library records these values and never reads them.
    metadata: dict[str, Any] = field(default_factory=dict)


# Legacy recurrent_ei spellings that typed configs can never reach: the typed
# sections always carry the canonical field, which takes precedence.
_SHADOWED_RECURRENT_EI_KEYS = {
    "rec_excitatory_synapses": "connectivity.rec_ee_synapses_per_branch",
    "rec_excitatory_synapses_per_layer": "connectivity.rec_ee_synapses_per_branch",
    "rec_inhibitory_synapses": "connectivity.rec_ie_synapses_per_branch",
    "rec_inhibitory_synapses_per_layer": "connectivity.rec_ie_synapses_per_branch",
    "inh_rec_excitatory_synapses": "connectivity.rec_ei_synapses_per_branch",
    "inh_rec_excitatory_synapses_per_layer": "connectivity.rec_ei_synapses_per_branch",
    "inh_rec_inhibitory_synapses": "connectivity.rec_ii_synapses_per_branch",
    "inh_rec_inhibitory_synapses_per_layer": "connectivity.rec_ii_synapses_per_branch",
    "input_projection_dims": "architecture.input_projection_dims",
    "store_routing": "implementation.store_routing",
    "soma_feedback": "soma_feedback",
}


def _reject_shadowed_legacy_recurrent_keys(core_dict: dict[str, Any]) -> None:
    """Fail on legacy recurrent spellings that typed defaults would shadow."""
    recurrent_ei = _to_plain_dict(core_dict.get("recurrent_ei", {}) or {})
    shadowed = [
        f"model.core.recurrent_ei.{key} -> model.core.{replacement}"
        for key, replacement in _SHADOWED_RECURRENT_EI_KEYS.items()
        if key in recurrent_ei
    ]
    dynamics = _to_plain_dict(
        core_dict.get("dynamics", core_dict.get("spiking", {})) or {}
    )
    for name in ("soma_feedback", "dendritic_spikes"):
        if name in dynamics:
            shadowed.append(f"model.core.dynamics.{name} -> model.core.{name}")
    if shadowed:
        raise ValueError(
            "Legacy recurrent fields are ignored by typed configs because the "
            "canonical typed field always takes precedence. Move them to the "
            "canonical location: " + "; ".join(shadowed)
        )


def _field_names(config_cls: type) -> set[str]:
    return {item.name for item in fields(config_cls)}


def _resolve_alias(
    mapping: dict[str, Any], alias: str, canonical: str, *, section: str
) -> None:
    """Move a deprecated spelling onto its canonical key.

    Giving both spellings raises, because one of them would be ignored.
    """
    if alias not in mapping:
        return
    value = mapping.pop(alias)
    if canonical in mapping:
        raise ValueError(
            f"{section}.{alias} is a deprecated spelling of {section}.{canonical}; "
            f"set only {section}.{canonical}"
        )
    logger.warning(
        "%s.%s is deprecated; use %s.%s instead.", section, alias, section, canonical
    )
    mapping[canonical] = value


_CORE_ALIASES = {"ei_unified": "unified_ei", "spiking": "dynamics"}


def _build_core_config(core_data: Any) -> CoreConfig:
    """Build a typed CoreConfig from possibly nested dict/OmegaConf payload."""
    core_dict = to_mapping(core_data, section="model.core")
    for alias, canonical in _CORE_ALIASES.items():
        _resolve_alias(core_dict, alias, canonical, section="model.core")
    _reject_shadowed_legacy_recurrent_keys(core_dict)
    core_dict = check_section_keys(
        core_dict, _field_names(CoreConfig), section="model.core"
    )
    check_recurrent_payloads(core_dict)
    core_type = str(core_dict.get("type", "einet"))
    lower_core_type = core_type.lower()
    if lower_core_type in _STRUCTURED_EI_TYPES:
        check_retired_inhibitory_routing(
            core_dict, recurrent=is_recurrent_core_config(core_dict)
        )
    if lower_core_type in {"ei_net", "unified_einet"}:
        logger.warning(
            "model.core.type=%r is deprecated; use 'unified_ei' instead.",
            core_type,
        )
    connectivity_dict = to_mapping(
        core_dict.get("connectivity"), section="model.core.connectivity"
    )
    _resolve_alias(
        connectivity_dict,
        "structured_connectivity",
        "structured",
        section="model.core.connectivity",
    )
    structured_root = connectivity_dict.get("structured")
    if structured_root is not None and not is_dataclass(structured_root):
        # Rule-key aliases (p_connect, in_degree, kernel, sigma) load under the
        # typed field they stand for, as they do in pathway entries.
        connectivity_dict["structured"] = canonical_rule(
            to_mapping(structured_root, section="model.core.connectivity.structured"),
            section="model.core.connectivity.structured",
        )
    morphology_dict = to_mapping(
        core_dict.get("morphology"), section="model.core.morphology"
    )
    # Type aliases fix these morphology flags at build time. Record the implied
    # value for omitted fields. A conflicting written value would be overridden
    # (and recorded wrongly in config.json), so it raises; saved records from
    # versions that stored the dataclass default instead keep loading and get
    # the build-time alias warning.
    saved_record = _field_names(CoreConfig) <= set(core_dict)
    if lower_core_type == "dendritic_alexnet" and not saved_record:
        written_sizes = [
            key
            for key in ("excitatory_layer_sizes", "inhibitory_layer_sizes")
            if key in to_mapping(core_dict.get("architecture"), section="")
        ]
        if written_sizes:
            raise ValueError(
                "model.core.type='dendritic_alexnet' sets each block's population "
                "sizes from vision.channels, vision.classifier_widths and "
                "vision.inhibitory_ratio, so "
                + ", ".join(f"model.core.architecture.{key}" for key in written_sizes)
                + " would be ignored; remove them"
            )
    for key, alias_value in get_core_morphology_alias_overrides(
        lower_core_type
    ).items():
        written = morphology_dict.get(key)
        if written is not None and written != alias_value and not saved_record:
            raise ValueError(
                f"model.core.type={core_type!r} fixes morphology.{key}="
                f"{alias_value!r}, but the config sets {written!r}. Remove "
                "the field, or use type 'einet' to choose it freely."
            )
        morphology_dict.setdefault(key, alias_value)
    sparsity_dict = to_mapping(core_dict.get("sparsity"), section="model.core.sparsity")
    _resolve_alias(
        sparsity_dict, "annealed_topk", "dense_to_sparse", section="model.core.sparsity"
    )
    reactivation_dict = to_mapping(
        core_dict.get("reactivation"), section="model.core.reactivation"
    )
    additive_aliases = {
        "dendritic_additive",
        "flat_additive",
        "dendritic_normalized_additive",
        "flat_normalized_additive",
    }
    is_explicit_additive_einet = (
        lower_core_type in {"einet", "ei_net", "unified_einet"}
        and morphology_dict.get("use_shunting") is False
    )
    if "init_policy" not in reactivation_dict and (
        lower_core_type in additive_aliases or is_explicit_additive_einet
    ):
        reactivation_dict["init_policy"] = DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY
    elif "init_policy" in reactivation_dict:
        reactivation_dict["init_policy"] = normalize_reactivation_init_policy(
            reactivation_dict.get("init_policy")
        )
    gate_value_keys = ("init_m", "init_b") + (
        ("soma_init_m", "soma_init_b") if reactivation_dict.get("soma_type") else ()
    )
    if saved_record:
        # A saved record lists every field, so only a value other than the
        # default shows that the start values were written.
        gate_value_keys = tuple(
            key
            for key in gate_value_keys
            if key in reactivation_dict
            and reactivation_dict[key] != field_default(ReactivationConfig, key)
        )
    warn_unused_gate_init(
        reactivation_dict,
        where="model.core.reactivation",
        policy_key="init_policy",
        value_keys=gate_value_keys,
    )

    def typed(name: str, config_cls: type, value: Any = None) -> Any:
        value = core_dict.get(name) if value is None else value
        return build_section(config_cls, value, section=f"model.core.{name}")

    def plain(name: str) -> dict[str, Any]:
        return to_mapping(core_dict.get(name), section=f"model.core.{name}")

    affine_mlp = core_dict.get("affine_mlp")
    core = CoreConfig(
        type=core_type,
        affine_mlp=None if affine_mlp is None else typed("affine_mlp", AffineMLPConfig),
        shared_projection_shunting=(
            None
            if core_dict.get("shared_projection_shunting") is None
            else typed("shared_projection_shunting", SharedProjectionShuntingConfig)
        ),
        target_active_parameters=(
            None
            if core_dict.get("target_active_parameters") is None
            else int(core_dict["target_active_parameters"])
        ),
        biological_neuron=resolve_biological_neuron(
            core_type, core_dict.get("biological_neuron")
        ),
        initialization_seed=(
            None
            if core_dict.get("initialization_seed") is None
            else int(core_dict["initialization_seed"])
        ),
        architecture=typed("architecture", ArchitectureConfig),
        connectivity=typed("connectivity", ConnectivityConfig, connectivity_dict),
        transfer=typed("transfer", TransferFunctionConfig),
        morphology=typed("morphology", MorphologyConfig, morphology_dict),
        sparsity=typed("sparsity", SparsityConfig, sparsity_dict),
        reactivation=typed("reactivation", ReactivationConfig, reactivation_dict),
        blocklinear=typed("blocklinear", BlockLinearConfig),
        implementation=typed("implementation", ImplementationConfig),
        spatial=typed("spatial", SpatialConfig),
        synapse_types=plain("synapse_types"),
        dynamics=plain("dynamics"),
        dendritic_spikes=plain("dendritic_spikes"),
        soma_feedback=plain("soma_feedback"),
        unified_ei=plain("unified_ei"),
        recurrent_ei=plain("recurrent_ei"),
        baseline_rnn=plain("baseline_rnn"),
        heterogeneous_leak_ctrnn=plain("heterogeneous_leak_ctrnn"),
        legendre_memory=plain("legendre_memory"),
        population_network=plain("population_network"),
        vision=plain("vision"),
    )
    reject_unread_core_sections(core, core_dict)
    return core


def _build_vision_replacement_section(data: Any, *, written_as: str) -> Any:
    """Build ``model.vision_replacement`` (also written ``pretrained_replacement``)."""
    builder = config_section_builder("model.vision_replacement")
    if builder is not None:
        return builder(data)
    return build_extension_section(written_as, data)


def _build_transformer_replacement_config(
    tr_data: Any,
) -> TransformerReplacementConfig:
    """Build a typed TransformerReplacementConfig from dict/OmegaConf."""
    tr_dict = _to_plain_dict(tr_data)
    allowed = {item.name for item in fields(TransformerReplacementConfig)}
    allowed.update({"backbone", "layer_indices", "mlp_attr"})
    unknown = set(tr_dict) - allowed
    if unknown and tr_dict.get("enabled", False):
        raise ValueError(
            f"model.transformer_replacement fields {sorted(unknown)} are not library "
            "fields; they need the DendriNet plugin that extends this section"
        )
    if unknown:
        logger.info(
            "Ignoring the settings %s of the disabled model.transformer_replacement",
            sorted(unknown),
        )
    return TransformerReplacementConfig(
        enabled=bool(tr_dict.get("enabled", False)),
        model_name=str(tr_dict.get("model_name", tr_dict.get("backbone", ""))),
        layers=[
            int(idx) for idx in tr_dict.get("layers", tr_dict.get("layer_indices", []))
        ],
        target_module=str(tr_dict.get("target_module", tr_dict.get("mlp_attr", "mlp"))),
        layers_attr=tr_dict.get("layers_attr"),
        input_transform=str(tr_dict.get("input_transform", "signed_split")),
        preserve_device_dtype=bool(tr_dict.get("preserve_device_dtype", True)),
        model_kwargs=_to_plain_dict(tr_dict.get("model_kwargs", {})),
        replacement_kwargs=_to_plain_dict(tr_dict.get("replacement_kwargs", {})),
    )


def _build_param_groups_config(
    param_groups_data: Any, *, section: str = "training.main.common.param_groups"
) -> ParamGroupsConfig:
    """Build a typed ParamGroupsConfig from dict/OmegaConf payload."""
    return build_section(ParamGroupsConfig, param_groups_data, section=section)


def _build_common_trainer_config(
    common_data: Any, *, section: str = "training.main.common"
) -> CommonTrainerConfig:
    """Build a typed CommonTrainerConfig with typed param groups.

    Learning-rate keys written directly under ``common`` are a deprecated
    spelling of the ``common.param_groups`` keys and are moved there.
    """
    if isinstance(common_data, CommonTrainerConfig):
        return common_data
    common_dict = to_mapping(common_data, section=section)
    param_groups_section = f"{section}.param_groups"
    param_groups_dict = to_mapping(
        common_dict.get("param_groups"), section=param_groups_section
    )
    for key in sorted(set(common_dict) & _field_names(ParamGroupsConfig)):
        value = common_dict.pop(key)
        if key in param_groups_dict and param_groups_dict[key] != value:
            raise ValueError(
                f"{section}.{key}={value!r} conflicts with "
                f"{param_groups_section}.{key}={param_groups_dict[key]!r}; set only "
                f"{param_groups_section}.{key}"
            )
        logger.warning(
            "%s.%s is deprecated; use %s.%s instead.",
            section,
            key,
            param_groups_section,
            key,
        )
        param_groups_dict[key] = value
    common_dict["param_groups"] = _build_param_groups_config(
        param_groups_dict, section=param_groups_section
    )
    adaptive = common_dict.pop("enable_adaptive_clipping", None)
    if adaptive:
        # It switched the standard trainer to global-norm clipping.
        mode = common_dict.get("grad_clip_mode")
        if mode is not None and str(mode).strip().lower() != "norm":
            raise ValueError(
                f"{section}.enable_adaptive_clipping=true selects norm clipping, "
                f"which contradicts {section}.grad_clip_mode={mode!r}; set only "
                "grad_clip_mode"
            )
        logger.warning(
            "%s.enable_adaptive_clipping is deprecated; it selects global-norm "
            "clipping, so use %s.grad_clip_mode: norm.",
            section,
            section,
        )
        common_dict["grad_clip_mode"] = "norm"
    return build_section(CommonTrainerConfig, common_dict, section=section)


def _build_training_strategies_config(
    strategies_data: Any, *, section: str = "training.strategies"
) -> TrainingStrategiesConfig:
    """Build typed nested strategy configs from dict/OmegaConf payload."""
    strategies_dict = to_mapping(strategies_data, section=section)
    strategy_classes = {
        "standard": StandardStrategyConfig,
        "multi_stage": MultiStageStrategyConfig,
        "recurrent": RecurrentStrategyConfig,
        "voltage_stabilization": VoltageStabilizationStrategyConfig,
        "homeostatic_control": HomeostaticControlStrategyConfig,
        "ei_equilibrium": EIEquilibriumStrategyConfig,
        "freeze_layers": FreezeLayersStrategyConfig,
        "freeze_branches": FreezeBranchesStrategyConfig,
        "freeze_branch_kl": FreezeBranchKLStrategyConfig,
        "presentation_stream": PresentationStreamStrategyConfig,
    }
    # ``local_ca`` is reported as unsupported by the callers before this point.
    # Settings of plugin strategies are written next to the built-in blocks or,
    # in saved configs, under ``extensions``.
    strategies_dict = dict(strategies_dict)
    plugin_blocks = dict(
        to_mapping(
            strategies_dict.pop("extensions", None), section=f"{section}.extensions"
        )
    )
    for name in list(strategies_dict):
        if name not in strategy_classes and name != "local_ca":
            if config_section_builder(f"training.strategies.{name}") is not None:
                if name in plugin_blocks:
                    raise ValueError(
                        f"{section}.{name} is written both directly and under "
                        f"{section}.extensions; keep one"
                    )
                plugin_blocks[name] = strategies_dict.pop(name)
    known = sorted(strategy_classes) + registered_named_sections("training.strategies")
    unknown = sorted(
        (set(strategies_dict) - set(strategy_classes) - {"local_ca"})
        | {
            name
            for name in plugin_blocks
            if config_section_builder(f"training.strategies.{name}") is None
        }
    )
    if unknown:
        details = []
        for name in unknown:
            matches = difflib.get_close_matches(
                str(name), sorted(strategy_classes), n=1
            )
            details.append(
                f"{name}" + (f" (did you mean {matches[0]!r}?)" if matches else "")
            )
        raise ValueError(
            f"Unknown {section} block(s): " + ", ".join(details) + ". Known "
            "strategy blocks: " + ", ".join(known)
        )

    built: dict[str, Any] = {}
    extensions = {
        name: build_extension_section(f"training.strategies.{name}", value)
        for name, value in sorted(plugin_blocks.items())
    }
    for name, config_cls in strategy_classes.items():
        raw_value = strategies_dict.get(name)
        if raw_value is None:
            continue
        if isinstance(raw_value, config_cls):
            built[name] = raw_value
        else:
            built[name] = build_section(
                config_cls, raw_value, section=f"{section}.{name}"
            )

    return TrainingStrategiesConfig(**built, extensions=extensions)


def _build_main_training_config(main_data: Any) -> MainTrainingConfig:
    """Build a typed MainTrainingConfig with typed common/optimizer config."""
    section = "training.main"
    main_dict = _map_main_training_config(main_data)
    raw_strategies = main_dict.get("strategies", {})
    explicit_strategy_overrides = (
        {}
        if isinstance(raw_strategies, TrainingStrategiesConfig)
        else to_mapping(raw_strategies, section=f"{section}.strategies")
    )
    main_dict["common"] = _build_common_trainer_config(
        main_dict.get("common", {}), section=f"{section}.common"
    )
    for name, config_cls in (
        ("regularization", RegularizationConfig),
        ("pruning", PruningConfig),
    ):
        if name in main_dict:
            main_dict[name] = build_section(
                config_cls, main_dict[name], section=f"{section}.{name}"
            )
    strategies_data = main_dict.get("strategies")
    if strategies_data is not None and not isinstance(
        strategies_data, TrainingStrategiesConfig
    ):
        strategies_dict = to_mapping(strategies_data, section=f"{section}.strategies")
        if "local_ca" in strategies_dict:
            logger.warning(
                "training.main.strategies.local_ca is no longer supported; "
                "use training.main.learning_strategy_config instead."
            )
        main_dict["strategies"] = _build_training_strategies_config(
            strategies_dict, section=f"{section}.strategies"
        )
    main_config = MainTrainingConfig(**main_dict)
    _warn_unread_local_rule_config(main_config)
    # Preserve the raw user-provided strategy block so downstream config
    # preparation can distinguish "explicitly set to the default value" from
    # "not specified at all".
    main_config._explicit_strategy_overrides = explicit_strategy_overrides
    return main_config


def _warn_unread_local_rule_config(main_config: MainTrainingConfig) -> None:
    """Warn when a non-default local-rule block reaches a strategy that ignores it.

    Sweeps often keep one block across their backpropagation and local-rule
    arms, so this is a warning rather than an error.
    """
    rule_config = main_config.learning_strategy_config
    strategy = str(main_config.strategy).lower()
    if rule_config is None or strategy in LOCAL_RULE_STRATEGIES:
        return
    if rule_config == LocalRuleConfig():
        return
    logger.warning(
        "training.main.learning_strategy_config has no effect with strategy %r; "
        "only %s read it.",
        strategy,
        " and ".join(sorted(LOCAL_RULE_STRATEGIES)),
    )


def _build_transformer_replacement_training_config(training_data: Any) -> Any:
    return build_extension_section("training.transformer_replacement", training_data)


def _build_vision_replacement_training_config(training_data: Any) -> Any:
    return build_extension_section("training.vision_replacement", training_data)


def _reject_explicit_learned_scale_without_wrapper(
    learned_output_scale: Any, model_dict: dict[str, Any], core_dict: dict[str, Any]
) -> None:
    """Reject an explicit learned output scale that no wrapper would create.

    Only the feedforward classification wrapper owns output-scale parameters.
    The retired ``learned_output_scale: true`` is checked here, where an
    explicitly written value is still distinguishable from the default.
    """
    if not bool(learned_output_scale):
        return
    mode = model_dict.get("output_scale_mode")
    if mode is not None and str(mode).strip().lower() == "fixed":
        return
    task = str(model_dict.get("task", "classification"))
    if task == "regression" or is_recurrent_core_config(core_dict):
        raise ValueError(
            "model.learned_output_scale=true (retired) is not applied by recurrent or "
            "regression models, which return the decoder output unscaled; set "
            "output_scale_mode: fixed instead"
        )


def _check_encoder_decoder_params(
    encoder_dict: dict[str, Any], decoder_dict: dict[str, Any]
) -> None:
    """Reject unread encoder/decoder params; leave identity output_dim unset.

    An identity decoder checks ``output_dim`` against the core output only
    when the user wrote it, so an unwritten (or saved-record) value is cleared.
    """
    encoder_params = to_mapping(
        encoder_dict.get("params"), section="model.encoder.params"
    )
    reject_unread_params(
        "encoder",
        encoder_dict.get("type", field_default(EncoderConfig, "type")),
        encoder_params,
        EncoderParamsConfig,
    )
    decoder_type = decoder_dict.get("type", field_default(DecoderConfig, "type"))
    decoder_params = to_mapping(
        decoder_dict.get("params"), section="model.decoder.params"
    )
    reject_unread_params("decoder", decoder_type, decoder_params, DecoderParamsConfig)
    if str(decoder_type).strip().lower() in IDENTITY_TYPES and (
        "output_dim" not in decoder_params
        or is_saved_record(decoder_params, DecoderParamsConfig)
    ):
        decoder_params["output_dim"] = None
        decoder_dict["params"] = decoder_params


def _build_model_config(model_data: Any) -> ModelConfig:
    """Build a typed ModelConfig from possibly nested dict/OmegaConf payload."""
    model_dict = to_mapping(model_data, section="model")
    learned_output_scale = model_dict.pop("learned_output_scale", None)
    model_dict = check_section_keys(
        model_dict, _field_names(ModelConfig), section="model"
    )
    encoder_dict = to_mapping(model_dict.get("encoder"), section="model.encoder")
    decoder_dict = to_mapping(model_dict.get("decoder"), section="model.decoder")
    _check_encoder_decoder_params(encoder_dict, decoder_dict)

    core_dict = to_mapping(model_dict.get("core"), section="model.core")
    transfer_dict = to_mapping(core_dict.get("transfer"), section="model.core.transfer")
    legacy_force_projection = bool(transfer_dict.pop("force_projection", False))
    legacy_zero_pad = bool(transfer_dict.pop("zero_pad_output", False))
    if legacy_force_projection:
        raise ValueError(
            "Legacy core.transfer.force_projection=true cannot be migrated "
            "losslessly; retain the historical source revision for that projection "
            "control or provide a canonical replacement configuration"
        )
    if legacy_zero_pad:
        logger.warning(
            "Ignoring model.core.transfer.zero_pad_output: no model of this "
            "installation reads it."
        )
    core_dict["transfer"] = transfer_dict

    pr_config = _build_vision_replacement_section(
        model_dict.get("pretrained_replacement", {}),
        written_as="model.pretrained_replacement",
    )
    vr_config = _build_vision_replacement_section(
        model_dict.get("vision_replacement", {}),
        written_as="model.vision_replacement",
    )
    tr_config = build_extension_section(
        "model.transformer_replacement",
        model_dict.get("transformer_replacement", {}),
        default=_build_transformer_replacement_config,
    )
    _reject_explicit_learned_scale_without_wrapper(
        learned_output_scale, model_dict, core_dict
    )
    if learned_output_scale is not None:
        logger.warning(
            "model.learned_output_scale was retired on 2026-10-08: "
            "output_scale_mode alone selects the output scale (false is "
            "output_scale_mode: fixed)."
        )
    output_scale_mode = fold_learned_output_scale(
        None if learned_output_scale is None else bool(learned_output_scale),
        model_dict.get("output_scale_mode"),
    )

    return ModelConfig(
        task=str(model_dict.get("task", "classification")),
        initial_checkpoint=model_dict.get("initial_checkpoint"),
        initial_checkpoint_sha256=model_dict.get("initial_checkpoint_sha256"),
        initial_checkpoint_strict=bool(
            model_dict.get("initial_checkpoint_strict", True)
        ),
        fixed_output_scale=float(model_dict.get("fixed_output_scale", 1.0)),
        output_scale_mode=output_scale_mode,
        encoder=build_section(EncoderConfig, encoder_dict, section="model.encoder"),
        core=_build_core_config(core_dict),
        decoder=build_section(DecoderConfig, decoder_dict, section="model.decoder"),
        vision_replacement=vr_config,
        pretrained_replacement=pr_config,
        transformer_replacement=tr_config,
    )


def _build_data_config(data_data: Any) -> DataConfig:
    """Build a typed DataConfig while preserving custom dataset sections."""
    data_dict = check_section_keys(
        to_mapping(data_data, section="data"),
        _field_names(DataConfig),
        section="data",
    )
    return DataConfig(
        dataset_name=str(data_dict.get("dataset_name", "cifar10")),
        base_dir=str(data_dict.get("base_dir", "")),
        processing=_build_data_processing_config(data_dict.get("processing", {})),
        dataset_params=_build_dataset_params(data_dict.get("dataset_params")),
    )


def _build_data_processing_config(processing_data: Any) -> DataProcessingConfig:
    """Build typed data-processing config from a config-like payload."""
    return build_section(
        DataProcessingConfig, processing_data, section="data.processing"
    )


def _build_dataset_params(dataset_params: Any) -> dict[str, dict[str, Any]]:
    """Return ``data.dataset_params`` as plain per-dataset mappings."""
    blocks = to_mapping(dataset_params, section="data.dataset_params")
    return {
        str(name): to_mapping(block, section=f"data.dataset_params.{name}")
        for name, block in blocks.items()
    }


def _collect_metadata(conf: dict[str, Any]) -> dict[str, Any]:
    """Remove and return the ``metadata`` section and underscore-prefixed keys."""
    metadata = to_mapping(conf.pop(METADATA_SECTION, None), section=METADATA_SECTION)
    for key in [key for key in conf if isinstance(key, str) and key.startswith("_")]:
        value = conf.pop(key)
        if key in metadata and metadata[key] != value:
            raise ValueError(
                f"Top-level {key} conflicts with {METADATA_SECTION}.{key}; keep one"
            )
        metadata[key] = value
    return metadata


def _config_from_mapping(conf_data: Any) -> Config:
    """Build and validate a typed Config from a loaded YAML mapping."""
    load_plugins()  # plugins register the sections they provide
    conf = _migrated(to_mapping(conf_data, section=""))
    _reject_partial_migration(conf)
    metadata = _collect_metadata(conf)
    # A top-level ``fsdp`` block is handled (and rejected) by the distributed
    # builder, which explains where FSDP options moved.
    conf = check_section_keys(conf, _field_names(Config) | {"fsdp"}, section="")
    config_version = _check_config_version(conf.get("config_version"))

    distributed_config = _build_distributed_config(conf)
    mapped_analysis = _map_analysis_config(
        conf.get("analysis", {}),
        distributed_mode=distributed_config.mode,
    )
    mapped_training = _map_training_config(conf.get("training", {}))
    training_config = TrainingConfig(
        main=_build_main_training_config(mapped_training.get("main", {})),
        encoder_pretraining=build_section(
            EncoderPretrainingConfig,
            mapped_training.get("encoder_pretraining", {}),
            section="training.encoder_pretraining",
        ),
        transformer_replacement=_build_transformer_replacement_training_config(
            mapped_training.get("transformer_replacement", {})
        ),
        vision_replacement=_build_vision_replacement_training_config(
            mapped_training.get("vision_replacement", {})
        ),
    )
    config = Config(
        config_version=config_version,
        experiment=build_section(
            ExperimentConfig, conf.get("experiment"), section="experiment"
        ),
        data=_build_data_config(conf.get("data", {})),
        model=_build_model_config(conf.get("model", {})),
        training=training_config,
        analysis=AnalysisConfig(**mapped_analysis),
        outputs=build_section(OutputsConfig, conf.get("outputs"), section="outputs"),
        distributed=distributed_config,
        metadata=metadata,
    )
    validate_loaded_config(config)
    return config


def _migrated(conf: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the configuration migrations that plugins register."""
    migrated, notes = apply_config_migrations(conf)
    for note in notes:
        logger.warning("Migrated a historical configuration key: %s", note)
    return migrated


def config_from_mapping(data: Mapping[str, Any]) -> Config:
    """Build and validate a Config from an in-memory mapping.

    The mapping has the layout of a config file (``experiment``, ``data``,
    ``model``, ...); it is checked exactly as :func:`load_config` checks a file.
    """
    return _config_from_mapping(data)


def load_config(
    path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None
) -> Config:
    """Load a configuration file, or return the validated defaults.

    The file may ``extends`` a parent file (resolved relative to the file that
    names it). A sweep file is accepted as well; its ``base_config`` section is
    loaded. Every section is checked against the keys the library reads:
    unknown keys raise :class:`~dendritic_modeling.config.keys.UnknownConfigKeyError`,
    and keys retired from earlier library versions warn (see
    :data:`~dendritic_modeling.config.keys.RETIRED_KEYS`). Each call parses the
    file again, so edits are always picked up.

    ``overrides`` maps dotted paths (``"training.main.common.epochs"``) to
    values. They are applied to the composed file before validation, so an
    override is checked exactly like a value written in the file.
    """
    if not path and not overrides:
        config = Config()
        validate_loaded_config(config)
        return config
    conf_data: Any = {}
    if path:
        raw_conf, _sources = load_composed_yaml(path)
        conf_data = raw_conf["base_config"] if "base_config" in raw_conf else raw_conf
    if overrides:
        conf_data = apply_overrides(conf_data, overrides)
    return _config_from_mapping(conf_data)


def load_analysis_config(
    path: str,
    *,
    distributed_mode: str = "none",
) -> AnalysisConfig:
    """Load a standalone, typed ``analysis:`` YAML profile.

    Checkpoint analyses often need to reuse an immutable training config while
    applying a newer measurement profile.  Keeping the analysis profile in a
    separate YAML avoids copying or mutating the scientific training config.
    The input may either contain an ``analysis:`` root or be the analysis
    mapping itself.
    """

    load_plugins()  # plugins register the analysis sections they provide
    raw_conf = OmegaConf.load(path)
    analysis_data = raw_conf.get("analysis", raw_conf)
    if not isinstance(analysis_data, Mapping):
        raise ValueError(f"Analysis profile must be a mapping: {path}")
    analysis_data = _migrated({"analysis": analysis_data}).get("analysis", {})
    mapped = _map_analysis_config(
        analysis_data,
        distributed_mode=distributed_mode,
    )
    return AnalysisConfig(**mapped)


def _clear_load_config_cache() -> None:
    """Deprecated no-op: ``load_config`` no longer caches parsed files."""
    warnings.warn(
        "load_config.cache_clear() is deprecated and does nothing; load_config "
        "parses the file on every call",
        DeprecationWarning,
        stacklevel=2,
    )


load_config.cache_clear = _clear_load_config_cache  # type: ignore[attr-defined]


def _build_distributed_config(conf_data: Mapping[str, Any]) -> DistributedConfig:
    """Build DistributedConfig from the ``distributed:`` YAML block.

    Expected layout::

        distributed:
          mode: ddp
          fsdp:
            sharding_strategy: FULL_SHARD

    Raises ``ValueError`` if a legacy top-level ``fsdp:`` block is present:
    no builder reads it, so its settings would be ignored (they belong under
    ``distributed:``; ``dendritic-upgrade-config`` reports such blocks).
    """
    if "fsdp" in conf_data:
        raise ValueError(
            "Top-level 'fsdp:' block is not read, so its settings would be "
            "ignored. Move them under 'distributed.fsdp:' and set "
            "'distributed.mode: fsdp' (dendritic-upgrade-config reports such blocks). "
            "See docs/distributed_training.rst."
        )
    dist_raw = to_mapping(conf_data.get("distributed"), section="distributed")
    return build_section(DistributedConfig, dist_raw, section="distributed")


def _default_analysis_runtime_config(distributed_mode: str) -> AnalysisRuntimeConfig:
    """Return the implicit shared analysis runtime for a distributed mode."""
    if str(distributed_mode).lower() == "none":
        default_runtime = EvaluationRuntimeConfig(mode="materialize")
        return AnalysisRuntimeConfig(
            training=default_runtime,
            final=EvaluationRuntimeConfig(mode="materialize"),
        )
    return AnalysisRuntimeConfig()


_LEGACY_ANALYSIS_ALIASES = {
    "spike_trains": "spike_train",
}


def _analysis_field_mapping() -> dict[str, str]:
    """Return the analysis field of every accepted section name."""
    # Plugin analyzers have plugin sections (config.extensions), not fields.
    fields = _field_names(AnalysisConfig)
    mapping = {
        name: spec.config_field
        for name, spec in get_registered_analyzers().items()
        if spec.config_field in fields
    }
    mapping.update(_LEGACY_ANALYSIS_ALIASES)
    mapping.update(AnalysisConfig.RENAMED_SECTIONS)
    return mapping


def _build_legacy_performance_runtime(
    runtime_data: Any, *, section: str
) -> tuple[EvaluationRuntimeConfig | None, Any]:
    """Build a legacy performance runtime section and return its split list."""
    if runtime_data is None:
        return None, None
    runtime_dict = to_mapping(runtime_data, section=section)
    splits = runtime_dict.pop("splits", None)
    return build_section(EvaluationRuntimeConfig, runtime_dict, section=section), splits


def _migrate_legacy_performance_runtime(
    performance: dict[str, Any],
    mapped: dict[str, Any],
    distributed_mode: str,
    *,
    has_runtime: bool,
) -> None:
    """Move pre-2026-03-16 performance runtime blocks to ``analysis.runtime``.

    Their split lists move to ``training_splits``/``final_splits``. The runtime
    settings are used only when no ``analysis.runtime`` section is given.
    """
    section = "analysis.performance"
    training_runtime, training_splits = _build_legacy_performance_runtime(
        performance.pop("training_runtime", None), section=f"{section}.training_runtime"
    )
    final_runtime, final_splits = _build_legacy_performance_runtime(
        performance.pop("final_runtime", None), section=f"{section}.final_runtime"
    )
    legacy_runtime = training_runtime is not None or final_runtime is not None
    if legacy_runtime and has_runtime:
        logger.warning(
            "Ignoring %s.training_runtime/final_runtime: analysis.runtime is set "
            "and takes precedence.",
            section,
        )
    elif legacy_runtime:
        default_runtime = _default_analysis_runtime_config(distributed_mode)
        mapped["runtime"] = AnalysisRuntimeConfig(
            training=training_runtime or default_runtime.training,
            final=final_runtime or default_runtime.final,
        )
    if training_splits is not None and "training_splits" not in performance:
        performance["training_splits"] = list(training_splits)
    if final_splits is not None and "final_splits" not in performance:
        performance["final_splits"] = list(final_splits)


def _map_analysis_config(
    analysis_data: Any,
    distributed_mode: str = "none",
) -> dict:
    """Map analysis YAML sections to typed ``AnalysisConfig`` fields.

    A section may be written under its registry name or its field name, but
    not both. Section names, section keys and parameter keys are all checked.
    """
    data = dict(to_mapping(analysis_data, section="analysis"))
    # Sections of plugin analyzers are written next to the built-in ones or,
    # in saved configs, under ``extensions``.
    plugin_sections = dict(
        to_mapping(data.pop("extensions", None), section="analysis.extensions")
    )
    for key in list(data):
        if config_section_builder(f"analysis.{key}") is not None:
            if key in plugin_sections:
                raise ValueError(
                    f"analysis.{key} is written both directly and under "
                    "analysis.extensions; keep one"
                )
            plugin_sections[key] = data.pop(key)
    unknown_plugin = sorted(
        key
        for key in plugin_sections
        if config_section_builder(f"analysis.{key}") is None
    )
    if unknown_plugin:
        raise ValueError(
            "analysis.extensions names analyzers no installed plugin provides: "
            + ", ".join(unknown_plugin)
        )

    field_names = _field_names(AnalysisConfig) - {"extensions"}
    aliases = _analysis_field_mapping()
    spellings: dict[str, str] = {}
    sections: dict[str, Any] = {}
    unknown: dict[str, Any] = {}
    for key, value in data.items():
        field_name = (
            "runtime"
            if key == "runtime"
            else aliases.get(key) or (key if key in field_names else None)
        )
        if field_name is None:
            unknown[key] = value
            continue
        if field_name in sections:
            raise ValueError(
                f"analysis.{spellings[field_name]} and analysis.{key} configure "
                "the same analysis; keep one"
            )
        spellings[field_name] = key
        sections[field_name] = value
    check_section_keys(
        unknown, set(aliases) | field_names | {"runtime"}, section="analysis"
    )
    renamed = sorted(
        f"{key} -> {AnalysisConfig.RENAMED_SECTIONS[key]}"
        for key in spellings.values()
        if key in AnalysisConfig.RENAMED_SECTIONS
    )
    if renamed:
        logger.warning(
            "analysis sections were renamed on 2026-10-08 to their analyzer "
            "names; the old names still load: %s.",
            ", ".join(renamed),
        )

    mapped: dict[str, Any] = {}
    if "performance" in sections:
        performance = to_mapping(
            sections["performance"], section="analysis.performance"
        )
        _migrate_legacy_performance_runtime(
            performance,
            mapped,
            distributed_mode,
            has_runtime="runtime" in sections,
        )
        sections["performance"] = performance

    default_runtime = _default_analysis_runtime_config(distributed_mode)
    if "runtime" in sections:
        runtime = check_section_keys(
            to_mapping(sections.pop("runtime"), section="analysis.runtime"),
            _field_names(AnalysisRuntimeConfig),
            section="analysis.runtime",
        )
        for phase in ("training", "final"):
            # A partial phase overrides only the keys it names; the rest keep
            # the implicit default of the distributed mode.
            default_phase = dataclasses.asdict(getattr(default_runtime, phase))
            runtime[phase] = {
                **default_phase,
                **to_mapping(runtime.get(phase), section=f"analysis.runtime.{phase}"),
            }
        mapped["runtime"] = build_section(
            AnalysisRuntimeConfig, runtime, section="analysis.runtime"
        )
    mapped.setdefault("runtime", default_runtime)

    for field_name, params_cls, table in (
        ("information", InformationAnalysisParams, INFORMATION_FLAT_PARAMS),
        ("correlation", CorrelationAnalysisParams, CORRELATION_FLAT_PARAMS),
    ):
        if field_name not in sections:
            continue
        section_data = to_mapping(
            sections[field_name], section=f"analysis.{field_name}"
        )
        if "params" in section_data:
            params = to_mapping(
                section_data["params"], section=f"analysis.{field_name}.params"
            )
            # Retired list entries go before flat and grouped values are compared.
            params = drop_retired_values_in(
                params, section=f"analysis.{field_name}.params"
            )
            migrate_flat_params(
                params,
                table,
                params_cls=params_cls,
                section=f"analysis.{field_name}.params",
            )
            section_data["params"] = params
        sections[field_name] = section_data

    if sections:
        typed = build_section(AnalysisConfig, sections, section="analysis")
        for field_name in sections:
            mapped[field_name] = getattr(typed, field_name)
    mapped["extensions"] = {
        key: build_extension_section(f"analysis.{key}", value)
        for key, value in sorted(plugin_sections.items())
    }
    return mapped


def _map_training_config(training_data: Any) -> dict:
    """Check ``training`` keys and resolve deprecated spellings."""
    mapped = to_mapping(training_data, section="training")
    mapped = check_section_keys(
        mapped, _field_names(TrainingConfig), section="training"
    )
    if "main" in mapped:
        mapped["main"] = _map_main_training_config(mapped["main"])
    return mapped


def _map_main_training_config(main_data: Any) -> dict[str, Any]:
    """Check ``training.main`` keys and build its typed leaf sections.

    Idempotent: sections that are already typed are left unchanged.
    """
    section = "training.main"
    mapped_main = check_section_keys(
        to_mapping(main_data, section=section),
        _field_names(MainTrainingConfig),
        section=section,
    )
    _hoist_legacy_main_learning_strategy_config(mapped_main)
    _coerce_main_learning_strategy_config(mapped_main)
    _coerce_main_optimizer_config(mapped_main)
    return mapped_main


def _hoist_legacy_main_learning_strategy_config(main_data: dict[str, Any]) -> None:
    """Move legacy ``training.main.common.learning_strategy_config`` if present."""
    common = main_data.get("common")
    if not isinstance(common, Mapping) or "learning_strategy_config" not in common:
        return
    common = dict(common)
    legacy = common.pop("learning_strategy_config")
    main_data["common"] = common
    if main_data.get("learning_strategy_config") is not None:
        raise ValueError(
            "learning_strategy_config is set under both training.main and "
            "training.main.common; keep only training.main.learning_strategy_config"
        )
    main_data["learning_strategy_config"] = legacy
    logger.warning(
        "learning_strategy_config found under training.main.common — "
        "hoisting to training.main. Please update your config."
    )


def _coerce_main_learning_strategy_config(main_data: dict[str, Any]) -> None:
    """Build LocalRuleConfig for a mapping-valued main learning strategy config."""
    strategy_config = main_data.get("learning_strategy_config")
    if strategy_config is None or isinstance(strategy_config, LocalRuleConfig):
        return
    section = "training.main.learning_strategy_config"
    main_data["learning_strategy_config"] = _build_local_rule_config(
        to_mapping(strategy_config, section=section), section=section
    )


def _coerce_main_optimizer_config(main_data: dict[str, Any]) -> None:
    """Build OptimizerConfig for a mapping-valued main optimizer config."""
    optimizer = main_data.get("optimizer")
    if optimizer is None or isinstance(optimizer, OptimizerConfig):
        return
    main_data["optimizer"] = build_section(
        OptimizerConfig, optimizer, section="training.main.optimizer"
    )
