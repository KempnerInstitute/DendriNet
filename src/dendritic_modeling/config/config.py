"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""

import logging
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Optional

from omegaconf import OmegaConf

from dendritic_modeling.config.analysis import (
    AnalysisConfig,
    AnalysisRuntimeConfig,
    EvaluationRuntimeConfig,
    LocalRuleComponentAnalysisConfig,
    LocalRuleComponentAnalysisParams,
    PerformanceAnalysisConfig,
    PerformanceAnalysisParams,
    SpikeTrainAnalysisConfig,
    SpikeTrainAnalysisParams,
)
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.legacy import normalize_transfer_config
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
    PretrainedReplacementConfig,
    ReactivationConfig,
    ReplacementRegionConfig,
    SegmentTrainabilityConfig,
    SparsityConfig,
    SpatialConfig,
    StructuredConnectivityConfig,
    TrainabilityConfig,
    TransferFunctionConfig,
    TransformerReplacementConfig,
)
from dendritic_modeling.config.reactivation import (
    DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY,
    normalize_reactivation_init_policy,
)
from dendritic_modeling.config.recurrent import RecurrentConfig
from dendritic_modeling.config.training import (
    CommonTrainerConfig,
    EncoderTrainingConfig,
    FreezeBranchesStrategyConfig,
    FreezeBranchKLStrategyConfig,
    FreezeLayersStrategyConfig,
    HomeostaticControlStrategyConfig,
    LocalRuleConfig,
    MainTrainingConfig,
    MultiStageStrategyConfig,
    OptimizerConfig,
    ParamGroupsConfig,
    RecurrentStrategyConfig,
    StandardStrategyConfig,
    TrainingConfig,
    TrainingStrategiesConfig,
    TransformerReplacementTrainingConfig,
    VoltageStabilizationStrategyConfig,
)
from dendritic_modeling.config.validation import validate_loaded_config

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(BaseConfig):
    """Basic experiment settings."""

    seed: int = 42
    train_valid_split: float = 0.8
    fast_mode: bool = False
    deterministic: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    float32_matmul_precision: str = "highest"
    enable_profiling: bool = False
    profiling_output_dir: str = "results/profiling/"
    enable_hooks: bool = True

    checkpointing: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "save_every_n_epochs": 10,
            "checkpoint_dir": "results/checkpoints/",
        }
    )


@dataclass
class DataProcessingConfig(BaseConfig):
    """Data processing parameters."""

    flatten: bool = True
    normalize: bool = False
    label_noise_rate: float = (
        0.0  # Fraction of training labels to randomize (0 = clean)
    )


@dataclass
class DatasetParamsConfig(BaseConfig):
    """Dataset-specific configuration parameters."""

    # MNIST variants
    mnist_modulo10: dict[str, Any] = field(
        default_factory=lambda: {"shuffle_iterations": 2}
    )
    cifar10_modulo10: dict[str, Any] = field(
        default_factory=lambda: {"shuffle_iterations": 1, "pair_seed": 0}
    )
    double_mnist_contextual_mod10: dict[str, Any] = field(
        default_factory=lambda: {
            "shuffle_iterations": 1,
            "pair_seed": 0,
            "context_seed": 123,
        }
    )
    double_cifar10_contextual_mod10: dict[str, Any] = field(
        default_factory=lambda: {
            "shuffle_iterations": 1,
            "pair_seed": 0,
            "context_seed": 123,
        }
    )
    random_flip_mnist: dict[str, Any] = field(
        default_factory=lambda: {"flip_probability": 0.5}
    )

    # Synthetic datasets
    poisson_generator: dict[str, Any] = field(
        default_factory=lambda: {
            "base_dataset": "mnist",
            "multiplicative_gain": True,
            "fixed_gain_factor": None,
            "max_gain_factor": None,
            "max_gain_tau_ratio": None,
            "uniform_gain": True,
            "poisson_sampling": True,
            "stimulus_duration": 1.0,
        }
    )

    multixor: dict[str, Any] = field(default_factory=lambda: {"n_bits": 8})
    orientation_bars: dict[str, Any] = field(
        default_factory=lambda: {"n_orient_classes": 8}
    )
    orthonet: dict[str, Any] = field(
        default_factory=lambda: {
            "input_dim": 784,
            "n_layers": 5,
            "nonlinearity": "sigmoid",
            "repeat_layers": False,
            "noise_std": 0.1,
        }
    )
    contextual_stream_gain_shift: dict[str, Any] = field(
        default_factory=lambda: {
            "n_samples": 4000,
            "stream_dim": 64,
            "n_classes": 2,
            "signal_strength": 0.35,
            "relevant_noise_std": 0.08,
            "irrelevant_noise_std": 0.18,
            "train_gain_relevant_min": 0.9,
            "train_gain_relevant_max": 1.1,
            "train_gain_irrelevant_min": 0.8,
            "train_gain_irrelevant_max": 1.2,
            "test_gain_relevant": 1.0,
            "test_gain_irrelevant": 3.0,
            "test_irrelevant_alignment_alpha": 1.0,
            "ood_mode": "irrelevant",
            "valid_split_mode": "ood",
            "context_signal_scale": 1.0,
        }
    )
    imagenet: dict[str, Any] = field(
        default_factory=lambda: {
            "train_samples_per_class": None,
            "val_samples_per_class": None,
        }
    )


@dataclass
class DataConfig(BaseConfig):
    """Data configuration including dataset selection and processing."""

    dataset_name: str = "cifar10"
    base_dir: str = ""

    processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    dataset_params: DatasetParamsConfig = field(default_factory=DatasetParamsConfig)


@dataclass
class WandbConfig(BaseConfig):
    """Weights & Biases logging configuration."""

    use_wandb: bool = False
    entity: str = ""
    project: str = "dendritic-sweeps"
    group: str = "unified_sweep"
    tags: list[str] = field(default_factory=lambda: ["debug"])


@dataclass
class OutputsConfig(BaseConfig):
    """Output configuration."""

    run_name: str = "ei_sweep"
    results_dir: str = "results"


@dataclass
class FSDPConfig(BaseConfig):
    """FSDP-specific tuning knobs.

    These only take effect when ``distributed.mode`` is ``"fsdp"``.
    """

    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: bool = True
    cpu_offload: bool = False
    min_num_params: int = 1000000
    reduce_communication_overhead: bool = True
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    sync_module_states: bool = True
    use_orig_params: bool = True
    backward_prefetch: str = "BACKWARD_PRE"
    gradient_checkpointing: bool = False
    # Mixed-precision dtype control (only used when mixed_precision=True). The
    # single knob `mixed_precision_dtype` sets param/reduce/buffer together; the
    # granular fields override it individually (None = follow the knob). Default
    # "fp16" preserves historical behavior; set "bf16" (or reduce_dtype="fp32")
    # to avoid fp16 gradient-all-reduce overflow on bf16-capable GPUs.
    mixed_precision_dtype: str = "fp16"
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    buffer_dtype: Optional[str] = None


@dataclass
class DistributedConfig(BaseConfig):
    """Distributed training configuration.

    ``mode`` selects the runtime:

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


@dataclass
class Config:
    """Main configuration class with clean structure."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


def _to_plain_dict(data: Any) -> dict[str, Any]:
    """Convert OmegaConf/dict-like objects to a plain Python dict."""
    return to_plain_dict(data)


def _build_core_config(core_data: Any) -> CoreConfig:
    """Build a typed CoreConfig from possibly nested dict/OmegaConf payload."""
    core_dict = _to_plain_dict(core_data)
    core_type = str(core_dict.get("type", "einet"))
    lower_core_type = core_type.lower()
    if lower_core_type in {"ei_net", "unified_einet"}:
        logger.warning(
            "model.core.type=%r is deprecated; use 'unified_ei' instead.",
            core_type,
        )
    unified_ei = core_dict.get("unified_ei", core_dict.get("ei_unified", {}))
    connectivity_dict = _to_plain_dict(core_dict.get("connectivity", {}))
    structured_alias_present = "structured_connectivity" in connectivity_dict
    structured_alias = connectivity_dict.pop("structured_connectivity", {})
    if structured_alias_present:
        logger.warning(
            "model.core.connectivity.structured_connectivity is deprecated; "
            "use model.core.connectivity.structured instead."
        )
    structured_connectivity = connectivity_dict.pop("structured", structured_alias)
    morphology_dict = _to_plain_dict(core_dict.get("morphology", {}))
    sparsity_dict = _to_plain_dict(core_dict.get("sparsity", {}))
    annealed_topk_alias = sparsity_dict.pop("annealed_topk", None)
    if annealed_topk_alias is not None:
        logger.warning(
            "model.core.sparsity.annealed_topk is deprecated; use "
            "model.core.sparsity.dense_to_sparse instead."
        )
        sparsity_dict.setdefault("dense_to_sparse", annealed_topk_alias)
    reactivation_dict = _to_plain_dict(core_dict.get("reactivation", {}))
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

    return CoreConfig(
        type=core_type,
        architecture=ArchitectureConfig(
            **_to_plain_dict(core_dict.get("architecture", {}))
        ),
        connectivity=ConnectivityConfig(
            **connectivity_dict,
            structured=StructuredConnectivityConfig(
                **_to_plain_dict(structured_connectivity)
            ),
        ),
        transfer=TransferFunctionConfig(
            **normalize_transfer_config(core_dict.get("transfer", {}))
        ),
        morphology=MorphologyConfig(**morphology_dict),
        sparsity=SparsityConfig(**sparsity_dict),
        reactivation=ReactivationConfig(**reactivation_dict),
        blocklinear=BlockLinearConfig(
            **_to_plain_dict(core_dict.get("blocklinear", {}))
        ),
        implementation=ImplementationConfig(
            **_to_plain_dict(core_dict.get("implementation", {}))
        ),
        spatial=SpatialConfig(**_to_plain_dict(core_dict.get("spatial", {}))),
        synapse_types=_to_plain_dict(core_dict.get("synapse_types", {})),
        dynamics=_to_plain_dict(
            core_dict.get("dynamics", core_dict.get("spiking", {}))
        ),
        dendritic_spikes=_to_plain_dict(core_dict.get("dendritic_spikes", {})),
        soma_feedback=_to_plain_dict(core_dict.get("soma_feedback", {})),
        recurrent=RecurrentConfig(**_to_plain_dict(core_dict.get("recurrent", {}))),
        unified_ei=_to_plain_dict(unified_ei),
        recurrent_ei=_to_plain_dict(core_dict.get("recurrent_ei", {})),
        baseline_rnn=_to_plain_dict(core_dict.get("baseline_rnn", {})),
        population_network=_to_plain_dict(core_dict.get("population_network", {})),
    )


def _build_pretrained_replacement_config(
    pr_data: Any,
) -> PretrainedReplacementConfig:
    """Build a typed PretrainedReplacementConfig from dict/OmegaConf."""
    pr_dict = _to_plain_dict(pr_data)
    replace_dict = _to_plain_dict(pr_dict.get("replace", {}))
    train_dict = _to_plain_dict(pr_dict.get("trainability", {}))

    encoder_dict = _to_plain_dict(train_dict.get("encoder", {}))
    decoder_dict = _to_plain_dict(train_dict.get("decoder", {}))
    trainability = TrainabilityConfig(
        encoder=SegmentTrainabilityConfig(
            mode=str(encoder_dict.get("mode", "frozen")),
        ),
        decoder=SegmentTrainabilityConfig(
            mode=str(decoder_dict.get("mode", "trainable")),
        ),
    )

    return PretrainedReplacementConfig(
        enabled=bool(pr_dict.get("enabled", False)),
        backbone=str(pr_dict.get("backbone", "alexnet")),
        weights=pr_dict.get("weights", "IMAGENET1K_V1"),
        input_shape=list(pr_dict.get("input_shape", [3, 224, 224])),
        replace=ReplacementRegionConfig(
            start=str(replace_dict.get("start", "")),
            end=str(replace_dict.get("end", "")),
        ),
        omit_layers=list(pr_dict.get("omit_layers", [])),
        trainability=trainability,
    )


def _build_transformer_replacement_config(
    tr_data: Any,
) -> TransformerReplacementConfig:
    """Build a typed TransformerReplacementConfig from dict/OmegaConf."""
    tr_dict = _to_plain_dict(tr_data)
    if "backbone" in tr_dict and "model_name" not in tr_dict:
        logger.warning(
            "model.transformer_replacement.backbone is deprecated; use "
            "model.transformer_replacement.model_name instead."
        )
    if "layer_indices" in tr_dict and "layers" not in tr_dict:
        logger.warning(
            "model.transformer_replacement.layer_indices is deprecated; use "
            "model.transformer_replacement.layers instead."
        )
    if "mlp_attr" in tr_dict and "target_module" not in tr_dict:
        logger.warning(
            "model.transformer_replacement.mlp_attr is deprecated; use "
            "model.transformer_replacement.target_module instead."
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


def _build_param_groups_config(param_groups_data: Any) -> ParamGroupsConfig:
    """Build a typed ParamGroupsConfig from dict/OmegaConf payload."""
    param_groups_dict = _to_plain_dict(param_groups_data)
    return ParamGroupsConfig(**param_groups_dict)


def _build_common_trainer_config(common_data: Any) -> CommonTrainerConfig:
    """Build a typed CommonTrainerConfig with typed param groups."""
    common_dict = _to_plain_dict(common_data)
    param_group_keys = set(ParamGroupsConfig.__dataclass_fields__.keys())
    param_groups_dict = _to_plain_dict(common_dict.get("param_groups", {}))

    # Backward compatibility: historical configs sometimes put LR knobs
    # directly under `training.*.common` instead of under `param_groups`.
    for key in list(common_dict.keys()):
        if key in param_group_keys:
            param_groups_dict.setdefault(key, common_dict.pop(key))

    common_dict["param_groups"] = _build_param_groups_config(param_groups_dict)
    return CommonTrainerConfig(
        **_filter_dataclass_kwargs(CommonTrainerConfig, common_dict)
    )


def _build_training_strategies_config(strategies_data: Any) -> TrainingStrategiesConfig:
    """Build typed nested strategy configs from dict/OmegaConf payload."""
    strategies_dict = _to_plain_dict(strategies_data)
    strategy_classes = {
        "standard": StandardStrategyConfig,
        "multi_stage": MultiStageStrategyConfig,
        "recurrent": RecurrentStrategyConfig,
        "voltage_stabilization": VoltageStabilizationStrategyConfig,
        "homeostatic_control": HomeostaticControlStrategyConfig,
        "freeze_layers": FreezeLayersStrategyConfig,
        "freeze_branches": FreezeBranchesStrategyConfig,
        "freeze_branch_kl": FreezeBranchKLStrategyConfig,
    }

    built: dict[str, Any] = {}
    for name, config_cls in strategy_classes.items():
        raw_value = strategies_dict.get(name)
        if raw_value is None:
            continue
        if isinstance(raw_value, config_cls):
            built[name] = raw_value
        else:
            built[name] = config_cls(**_to_plain_dict(raw_value))

    return TrainingStrategiesConfig(**built)


def _build_encoder_training_config(encoder_data: Any) -> EncoderTrainingConfig:
    """Build a typed EncoderTrainingConfig with typed common config."""
    encoder_dict = _to_plain_dict(encoder_data)
    encoder_dict["common"] = _build_common_trainer_config(
        encoder_dict.get("common", {})
    )
    if (
        "learning_strategy_config" in encoder_dict
        and encoder_dict["learning_strategy_config"] is not None
        and not isinstance(encoder_dict["learning_strategy_config"], LocalRuleConfig)
    ):
        encoder_dict["learning_strategy_config"] = _build_local_rule_config(
            _to_plain_dict(encoder_dict["learning_strategy_config"])
        )
    strategies_data = encoder_dict.get("strategies")
    if strategies_data is not None and not isinstance(
        strategies_data, TrainingStrategiesConfig
    ):
        strategies_dict = _to_plain_dict(strategies_data)
        if "local_ca" in strategies_dict:
            logger.warning(
                "training.encoder.strategies.local_ca is no longer supported; "
                "use training.encoder.learning_strategy_config instead."
            )
        encoder_dict["strategies"] = _build_training_strategies_config(strategies_dict)
    return EncoderTrainingConfig(**encoder_dict)


def _build_main_training_config(main_data: Any) -> MainTrainingConfig:
    """Build a typed MainTrainingConfig with typed common/optimizer config."""
    main_dict = _to_plain_dict(main_data)
    explicit_strategy_overrides = _to_plain_dict(main_dict.get("strategies", {}))
    main_dict["common"] = _build_common_trainer_config(main_dict.get("common", {}))
    optimizer_data = main_dict.get("optimizer")
    if optimizer_data is not None and not isinstance(optimizer_data, OptimizerConfig):
        main_dict["optimizer"] = OptimizerConfig(**_to_plain_dict(optimizer_data))
    if (
        "learning_strategy_config" in main_dict
        and main_dict["learning_strategy_config"] is not None
        and not isinstance(main_dict["learning_strategy_config"], LocalRuleConfig)
    ):
        main_dict["learning_strategy_config"] = _build_local_rule_config(
            _to_plain_dict(main_dict["learning_strategy_config"])
        )
    strategies_data = main_dict.get("strategies")
    if strategies_data is not None and not isinstance(
        strategies_data, TrainingStrategiesConfig
    ):
        strategies_dict = _to_plain_dict(strategies_data)
        if "local_ca" in strategies_dict:
            logger.warning(
                "training.main.strategies.local_ca is no longer supported; "
                "use training.main.learning_strategy_config instead."
            )
        main_dict["strategies"] = _build_training_strategies_config(strategies_dict)
    main_config = MainTrainingConfig(**main_dict)
    # Preserve the raw user-provided strategy block so downstream config
    # preparation can distinguish "explicitly set to the default value" from
    # "not specified at all".
    main_config._explicit_strategy_overrides = explicit_strategy_overrides
    return main_config


def _build_transformer_replacement_training_config(
    training_data: Any,
) -> TransformerReplacementTrainingConfig:
    """Build typed config for transformer replacement distillation."""
    training_dict = _to_plain_dict(training_data)
    return TransformerReplacementTrainingConfig(
        **_filter_dataclass_kwargs(
            TransformerReplacementTrainingConfig,
            training_dict,
        )
    )


def _build_model_config(model_data: Any) -> ModelConfig:
    """Build a typed ModelConfig from possibly nested dict/OmegaConf payload."""
    model_dict = _to_plain_dict(model_data)
    encoder_dict = _to_plain_dict(model_dict.get("encoder", {}))
    decoder_dict = _to_plain_dict(model_dict.get("decoder", {}))

    pr_data = model_dict.get("pretrained_replacement", {})
    pr_config = _build_pretrained_replacement_config(pr_data)
    tr_data = model_dict.get("transformer_replacement", {})
    tr_config = _build_transformer_replacement_config(tr_data)

    return ModelConfig(
        task=str(model_dict.get("task", "classification")),
        encoder=EncoderConfig(
            type=str(encoder_dict.get("type", "identity")),
            load_save_root=str(
                encoder_dict.get("load_save_root", "./trained_encoder_networks/")
            ),
            params=EncoderParamsConfig(
                **_to_plain_dict(encoder_dict.get("params", {}))
            ),
        ),
        core=_build_core_config(model_dict.get("core", {})),
        decoder=DecoderConfig(
            type=str(decoder_dict.get("type", "MLP")),
            params=DecoderParamsConfig(
                **_to_plain_dict(decoder_dict.get("params", {}))
            ),
        ),
        pretrained_replacement=pr_config,
        transformer_replacement=tr_config,
    )


def _build_data_config(data_data: Any) -> DataConfig:
    """Build a typed DataConfig while preserving custom dataset sections."""
    data_dict = _to_plain_dict(data_data)
    processing = DataProcessingConfig(**_to_plain_dict(data_dict.get("processing", {})))

    dataset_params = data_dict.get("dataset_params")
    if dataset_params is None:
        dataset_params = DatasetParamsConfig()
    elif hasattr(dataset_params, "_metadata"):
        dataset_params = OmegaConf.to_container(dataset_params, resolve=True)

    return DataConfig(
        dataset_name=str(data_dict.get("dataset_name", "cifar10")),
        base_dir=str(data_dict.get("base_dir", "")),
        processing=processing,
        dataset_params=dataset_params,
    )


def _filter_dataclass_kwargs(config_cls: type, raw_data: Any) -> dict[str, Any]:
    """Drop unknown keys when loading historical configs into dataclasses."""
    data = raw_data or {}
    if isinstance(data, Mapping):
        pass  # dict, OmegaConf DictConfig, etc.
    elif hasattr(data, "_metadata"):
        data = OmegaConf.to_container(data, resolve=True) or {}
    else:
        return {}
    allowed = {f.name for f in fields(config_cls)}
    return {key: value for key, value in data.items() if key in allowed}


@lru_cache(maxsize=10)
def _load_config_cached(path: Optional[str] = None) -> Config:
    """Load configuration from a file or use defaults.

    This internal helper is cached because YAML parsing and typed conversion are
    relatively expensive. Public callers receive a deepcopy via ``load_config``
    so runtime mutations cannot leak into later loads of the same path.
    """
    if path:
        raw_conf = OmegaConf.load(path)

        # Handle both regular configs and sweep configs (which have base_config)
        if "base_config" in raw_conf:
            # This is a sweep config file, use base_config
            conf_data = raw_conf["base_config"]
        else:
            # This is a regular config file
            conf_data = raw_conf

        # Handle analysis config field mapping
        analysis_data = conf_data.get("analysis", {})
        distributed_config = _build_distributed_config(conf_data)
        mapped_analysis = _map_analysis_config(
            analysis_data,
            distributed_mode=distributed_config.mode,
        )

        # Handle training config field mapping
        training_data = conf_data.get("training", {})
        mapped_training = _map_training_config(training_data)

        # Explicitly construct nested training configs
        encoder_config = _build_encoder_training_config(
            mapped_training.get("encoder", {})
        )
        main_config = _build_main_training_config(mapped_training.get("main", {}))
        if "transformer_distillation" in mapped_training:
            logger.warning(
                "training.transformer_distillation is deprecated; use "
                "training.transformer_replacement instead."
            )
        transformer_training = _build_transformer_replacement_training_config(
            mapped_training.get(
                "transformer_replacement",
                mapped_training.get("transformer_distillation", {}),
            )
        )
        training_config = TrainingConfig(
            encoder=encoder_config,
            main=main_config,
            transformer_replacement=transformer_training,
        )

        config = Config(
            experiment=ExperimentConfig(
                **_filter_dataclass_kwargs(
                    ExperimentConfig, conf_data.get("experiment", {})
                )
            ),
            data=_build_data_config(conf_data.get("data", {})),
            model=_build_model_config(conf_data.get("model", {})),
            training=training_config,
            analysis=AnalysisConfig(**mapped_analysis),
            wandb=WandbConfig(
                **_filter_dataclass_kwargs(WandbConfig, conf_data.get("wandb", {}))
            ),
            outputs=OutputsConfig(**conf_data.get("outputs", {})),
            distributed=distributed_config,
        )
        validate_loaded_config(config)
        return config
    config = Config()
    validate_loaded_config(config)
    return config


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a file or defaults as an independent object."""
    return deepcopy(_load_config_cached(path))


# Preserve the lru_cache testing/debug interface that callers already use.
load_config.cache_clear = _load_config_cached.cache_clear  # type: ignore[attr-defined]
load_config.cache_info = _load_config_cached.cache_info  # type: ignore[attr-defined]


def _build_distributed_config(conf_data: dict) -> DistributedConfig:
    """Build DistributedConfig from the ``distributed:`` YAML block.

    Expected layout::

        distributed:
          mode: ddp
          fsdp:
            sharding_strategy: FULL_SHARD

    Raises ``ValueError`` if a legacy top-level ``fsdp:`` block is present
    (it must be nested under ``distributed:`` now).
    """
    # Reject legacy top-level fsdp: block.
    if "fsdp" in conf_data and "distributed" not in conf_data:
        raise ValueError(
            "Top-level 'fsdp:' block is no longer supported. "
            "Move FSDP settings under 'distributed.fsdp:' and set "
            "'distributed.mode: fsdp'. See docs/configuration_guide.rst."
        )
    if "fsdp" in conf_data and "fsdp" not in conf_data.get("distributed", {}):
        logger.warning(
            "Ignoring top-level 'fsdp:' block. FSDP settings must be under "
            "'distributed.fsdp:'. See docs/configuration_guide.rst."
        )

    dist_raw = dict(conf_data.get("distributed", {}))
    fsdp_raw = dict(dist_raw.pop("fsdp", {}))
    fsdp_config = FSDPConfig(**_filter_dataclass_kwargs(FSDPConfig, fsdp_raw))
    return DistributedConfig(
        **_filter_dataclass_kwargs(DistributedConfig, dist_raw),
        fsdp=fsdp_config,
    )


def _default_analysis_runtime_config(distributed_mode: str) -> AnalysisRuntimeConfig:
    """Return the implicit shared analysis runtime for a distributed mode."""
    if str(distributed_mode).lower() == "none":
        default_runtime = EvaluationRuntimeConfig(mode="materialize")
        return AnalysisRuntimeConfig(
            training=default_runtime,
            final=EvaluationRuntimeConfig(mode="materialize"),
        )
    return AnalysisRuntimeConfig()


def _map_analysis_config(
    analysis_data: dict,
    distributed_mode: str = "none",
) -> dict:
    """Map analysis config fields to expected AnalysisConfig structure."""
    mapped = {}

    field_mapping = {
        "performance": "performance_analysis",
        "information": "information_analysis",
        "correlation": "correlation_analysis",
        "ablation": "ablation_analysis",
        "synapse_turnover": "synapse_turnover_analysis",
        "noise_perturbation": "noise_perturbation_analysis",
        "weight": "weight_analysis",
        "compartment_statistics": "compartment_statistics_analysis",
        "compartment_snr": "compartment_snr_analysis",
        "multiplicative_gain": "multiplicative_gain_analysis",
        "receptive_field": "receptive_field_analysis",
        "reactivation_dynamics": "reactivation_dynamics_analysis",
        "single_layer_contribution": "single_layer_contribution",
        "synaptic_activation": "synaptic_activation_analysis",
        "branch_activation": "branch_activation_analysis",
        "local_rule_components": "local_rule_component_analysis",
        "routing": "routing_analysis",
        "inhibitory_specialization": "inhibitory_specialization_analysis",
        "jacobian_spectrum": "jacobian_spectrum_analysis",
        "dendritic_timetraces": "dendritic_timetraces_analysis",
        "spike_train": "spike_train_analysis",
        "spike_trains": "spike_train_analysis",
        "adversarial_robustness": "adversarial_robustness_analysis",
    }

    for new_name, old_name in field_mapping.items():
        if new_name in analysis_data:
            mapped[old_name] = analysis_data[new_name]
        elif old_name in analysis_data:
            mapped[old_name] = analysis_data[old_name]

    if "runtime" in analysis_data and isinstance(analysis_data["runtime"], Mapping):
        runtime_data = dict(analysis_data["runtime"])
        default_runtime = _default_analysis_runtime_config(distributed_mode)
        if "training" in runtime_data and isinstance(runtime_data["training"], Mapping):
            tr = dict(runtime_data["training"])
            tr.pop("splits", None)
            tr.pop("persistent_workers", None)
            tr.pop("prefetch_factor", None)
            runtime_data["training"] = EvaluationRuntimeConfig(**tr)
        elif "training" not in runtime_data:
            runtime_data["training"] = default_runtime.training
        if "final" in runtime_data and isinstance(runtime_data["final"], Mapping):
            fr = dict(runtime_data["final"])
            fr.pop("splits", None)
            fr.pop("persistent_workers", None)
            fr.pop("prefetch_factor", None)
            runtime_data["final"] = EvaluationRuntimeConfig(**fr)
        elif "final" not in runtime_data:
            runtime_data["final"] = default_runtime.final
        mapped["runtime"] = AnalysisRuntimeConfig(**runtime_data)

    # Instantiate PerformanceAnalysisConfig and migrate legacy performance runtime
    # config into the shared top-level analysis.runtime surface.
    if "performance_analysis" in mapped and isinstance(
        mapped["performance_analysis"], Mapping
    ):
        pa = dict(mapped["performance_analysis"])
        if "params" in pa and isinstance(pa["params"], Mapping):
            pa["params"] = PerformanceAnalysisParams(**pa["params"])

        legacy_training_runtime = None
        legacy_final_runtime = None
        legacy_training_splits = None
        legacy_final_splits = None
        if "training_runtime" in pa and isinstance(pa["training_runtime"], Mapping):
            tr = dict(pa["training_runtime"])
            legacy_training_splits = tr.pop("splits", None)
            tr.pop("persistent_workers", None)
            tr.pop("prefetch_factor", None)
            legacy_training_runtime = EvaluationRuntimeConfig(**tr)
        if "final_runtime" in pa and isinstance(pa["final_runtime"], Mapping):
            fr = dict(pa["final_runtime"])
            legacy_final_splits = fr.pop("splits", None)
            fr.pop("persistent_workers", None)
            fr.pop("prefetch_factor", None)
            legacy_final_runtime = EvaluationRuntimeConfig(**fr)

        if "runtime" not in mapped and (
            legacy_training_runtime is not None or legacy_final_runtime is not None
        ):
            mapped["runtime"] = AnalysisRuntimeConfig(
                training=legacy_training_runtime
                or _default_analysis_runtime_config(distributed_mode).training,
                final=legacy_final_runtime
                or _default_analysis_runtime_config(distributed_mode).final,
            )

        if legacy_training_splits is not None and "training_splits" not in pa:
            pa["training_splits"] = list(legacy_training_splits)
        if legacy_final_splits is not None and "final_splits" not in pa:
            pa["final_splits"] = list(legacy_final_splits)

        pa.pop("training_runtime", None)
        pa.pop("final_runtime", None)
        mapped["performance_analysis"] = PerformanceAnalysisConfig(**pa)

    if "runtime" not in mapped:
        mapped["runtime"] = _default_analysis_runtime_config(distributed_mode)

    if "local_rule_component_analysis" in mapped and isinstance(
        mapped["local_rule_component_analysis"], Mapping
    ):
        lrc = dict(mapped["local_rule_component_analysis"])
        if "params" in lrc and isinstance(lrc["params"], Mapping):
            lrc["params"] = LocalRuleComponentAnalysisParams(**lrc["params"])
        mapped["local_rule_component_analysis"] = LocalRuleComponentAnalysisConfig(
            **lrc
        )

    if "spike_train_analysis" in mapped and isinstance(
        mapped["spike_train_analysis"], Mapping
    ):
        spike = dict(mapped["spike_train_analysis"])
        if "params" in spike and isinstance(spike["params"], Mapping):
            spike["params"] = SpikeTrainAnalysisParams(**spike["params"])
        mapped["spike_train_analysis"] = SpikeTrainAnalysisConfig(**spike)

    return mapped


def _map_training_config(training_data: dict) -> dict:
    """Map training config fields and handle nested learning_strategy_config."""
    # Convert OmegaConf to regular dict
    mapped = (
        OmegaConf.to_container(training_data, resolve=True)
        if hasattr(training_data, "_metadata")
        else dict(training_data)
    )

    # Handle nested learning_strategy_config and optimizer in main config
    if "main" in mapped and isinstance(mapped["main"], dict):
        main_data = dict(mapped["main"])

        # Backward compatibility: hoist learning_strategy_config from common
        # to main if it was placed there by legacy configs.
        if "common" in main_data and isinstance(main_data["common"], dict):
            common = main_data["common"]
            if "learning_strategy_config" in common:
                main_data.setdefault(
                    "learning_strategy_config", common.pop("learning_strategy_config")
                )
                logger.warning(
                    "learning_strategy_config found under training.main.common — "
                    "hoisting to training.main. Please update your config."
                )

        # Build learning_strategy_config if present
        if (
            "learning_strategy_config" in main_data
            and main_data["learning_strategy_config"] is not None
        ):
            if isinstance(main_data["learning_strategy_config"], dict):
                main_data["learning_strategy_config"] = _build_local_rule_config(
                    main_data["learning_strategy_config"]
                )

        # Build OptimizerConfig if present as dict
        if "optimizer" in main_data and isinstance(main_data["optimizer"], dict):
            from dendritic_modeling.config.training import OptimizerConfig

            main_data["optimizer"] = OptimizerConfig(**main_data["optimizer"])

        mapped["main"] = main_data

    return mapped
