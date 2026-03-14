"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""

from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Optional

from omegaconf import OmegaConf

from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.config.base import BaseConfig
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
    TrainabilityConfig,
    TransferFunctionConfig,
)
from dendritic_modeling.config.recurrent import RecurrentConfig
from dendritic_modeling.config.training import (
    CommonTrainerConfig,
    EncoderTrainingConfig,
    FiveFactorConfig,
    FourFactorConfig,
    HSICConfig,
    InhibitoryHomeostasisConfig,
    LocalRuleConfig,
    MainTrainingConfig,
    MorphologyAwareConfig,
    OptimizerConfig,
    ParamGroupsConfig,
    ThreeFactorConfig,
    TrainingConfig,
)


@dataclass
class ExperimentConfig(BaseConfig):
    """Basic experiment settings."""

    seed: int = 42
    train_valid_split: float = 0.8
    enable_amp: bool = False
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
    enabled: bool = True


@dataclass
class OutputsConfig(BaseConfig):
    """Output configuration."""

    run_name: str = "ei_sweep"
    results_dir: str = "results"


@dataclass
class FSDPConfig(BaseConfig):
    """Fully Sharded Data Parallel configuration."""

    use_fsdp: bool = False
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
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)


def _to_plain_dict(data: Any) -> dict[str, Any]:
    """Convert OmegaConf/dict-like objects to a plain Python dict."""
    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)
    if hasattr(data, "_metadata"):
        converted = OmegaConf.to_container(data, resolve=True)
        return converted if isinstance(converted, dict) else {}
    if hasattr(data, "asdict") and callable(data.asdict):
        converted = data.asdict()
        return converted if isinstance(converted, dict) else {}
    if hasattr(data, "__dict__"):
        return {
            key: value for key, value in vars(data).items() if not key.startswith("_")
        }
    return {}


def _build_core_config(core_data: Any) -> CoreConfig:
    """Build a typed CoreConfig from possibly nested dict/OmegaConf payload."""
    core_dict = _to_plain_dict(core_data)
    unified_ei = core_dict.get("unified_ei", core_dict.get("ei_unified", {}))
    return CoreConfig(
        type=str(core_dict.get("type", "EINet")),
        architecture=ArchitectureConfig(
            **_to_plain_dict(core_dict.get("architecture", {}))
        ),
        connectivity=ConnectivityConfig(
            **_to_plain_dict(core_dict.get("connectivity", {}))
        ),
        transfer=TransferFunctionConfig(
            **_to_plain_dict(core_dict.get("transfer", {}))
        ),
        morphology=MorphologyConfig(**_to_plain_dict(core_dict.get("morphology", {}))),
        sparsity=SparsityConfig(**_to_plain_dict(core_dict.get("sparsity", {}))),
        reactivation=ReactivationConfig(
            **_to_plain_dict(core_dict.get("reactivation", {}))
        ),
        blocklinear=BlockLinearConfig(
            **_to_plain_dict(core_dict.get("blocklinear", {}))
        ),
        implementation=ImplementationConfig(
            **_to_plain_dict(core_dict.get("implementation", {}))
        ),
        recurrent=RecurrentConfig(**_to_plain_dict(core_dict.get("recurrent", {}))),
        unified_ei=_to_plain_dict(unified_ei),
        recurrent_ei=_to_plain_dict(core_dict.get("recurrent_ei", {})),
        baseline_rnn=_to_plain_dict(core_dict.get("baseline_rnn", {})),
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
    return CommonTrainerConfig(**common_dict)


def _build_encoder_training_config(encoder_data: Any) -> EncoderTrainingConfig:
    """Build a typed EncoderTrainingConfig with typed common config."""
    encoder_dict = _to_plain_dict(encoder_data)
    encoder_dict["common"] = _build_common_trainer_config(
        encoder_dict.get("common", {})
    )
    return EncoderTrainingConfig(**encoder_dict)


def _build_main_training_config(main_data: Any) -> MainTrainingConfig:
    """Build a typed MainTrainingConfig with typed common/optimizer config."""
    main_dict = _to_plain_dict(main_data)
    main_dict["common"] = _build_common_trainer_config(main_dict.get("common", {}))
    optimizer_data = main_dict.get("optimizer")
    if optimizer_data is not None and not isinstance(optimizer_data, OptimizerConfig):
        main_dict["optimizer"] = OptimizerConfig(**_to_plain_dict(optimizer_data))
    return MainTrainingConfig(**main_dict)


def _build_model_config(model_data: Any) -> ModelConfig:
    """Build a typed ModelConfig from possibly nested dict/OmegaConf payload."""
    model_dict = _to_plain_dict(model_data)
    encoder_dict = _to_plain_dict(model_dict.get("encoder", {}))
    decoder_dict = _to_plain_dict(model_dict.get("decoder", {}))

    pr_data = model_dict.get("pretrained_replacement", {})
    pr_config = _build_pretrained_replacement_config(pr_data)

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
    if not isinstance(data, dict):
        return {}
    allowed = {f.name for f in fields(config_cls)}
    return {key: value for key, value in data.items() if key in allowed}


@lru_cache(maxsize=10)
def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a file or use defaults."""
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
        mapped_analysis = _map_analysis_config(analysis_data)

        # Handle training config field mapping
        training_data = conf_data.get("training", {})
        mapped_training = _map_training_config(training_data)

        # Explicitly construct nested training configs
        encoder_config = _build_encoder_training_config(
            mapped_training.get("encoder", {})
        )
        main_config = _build_main_training_config(mapped_training.get("main", {}))
        training_config = TrainingConfig(encoder=encoder_config, main=main_config)

        return Config(
            experiment=ExperimentConfig(**conf_data.get("experiment", {})),
            data=_build_data_config(conf_data.get("data", {})),
            model=_build_model_config(conf_data.get("model", {})),
            training=training_config,
            analysis=AnalysisConfig(**mapped_analysis),
            wandb=WandbConfig(
                **_filter_dataclass_kwargs(WandbConfig, conf_data.get("wandb", {}))
            ),
            outputs=OutputsConfig(**conf_data.get("outputs", {})),
            fsdp=FSDPConfig(**conf_data.get("fsdp", {})),
        )
    return Config()


def _map_analysis_config(analysis_data: dict) -> dict:
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
        "single_layer_contribution": "single_layer_contribution",
        "synaptic_activation": "synaptic_activation_analysis",
        "branch_activation": "branch_activation_analysis",
        "routing": "routing_analysis",
        "adversarial_robustness": "adversarial_robustness_analysis",
    }

    for new_name, old_name in field_mapping.items():
        if new_name in analysis_data:
            mapped[old_name] = analysis_data[new_name]
        elif old_name in analysis_data:
            mapped[old_name] = analysis_data[old_name]

    return mapped


def _build_local_rule_config(lsc_data: dict) -> LocalRuleConfig:
    """Build LocalRuleConfig from dict data with nested configs."""
    lsc_kwargs = dict(lsc_data)

    # Handle nested three_factor config
    if "three_factor" in lsc_kwargs and isinstance(lsc_kwargs["three_factor"], dict):
        lsc_kwargs["three_factor"] = ThreeFactorConfig(**lsc_kwargs["three_factor"])

    # Handle nested four_factor config
    if "four_factor" in lsc_kwargs and isinstance(lsc_kwargs["four_factor"], dict):
        lsc_kwargs["four_factor"] = FourFactorConfig(**lsc_kwargs["four_factor"])

    # Handle nested five_factor config
    if "five_factor" in lsc_kwargs and isinstance(lsc_kwargs["five_factor"], dict):
        lsc_kwargs["five_factor"] = FiveFactorConfig(**lsc_kwargs["five_factor"])

    # Handle nested morphology_aware config
    if "morphology_aware" in lsc_kwargs and isinstance(
        lsc_kwargs["morphology_aware"], dict
    ):
        lsc_kwargs["morphology_aware"] = MorphologyAwareConfig(
            **lsc_kwargs["morphology_aware"]
        )

    # Handle nested hsic config
    if "hsic" in lsc_kwargs and isinstance(lsc_kwargs["hsic"], dict):
        lsc_kwargs["hsic"] = HSICConfig(**lsc_kwargs["hsic"])

    # Handle nested inhibitory homeostasis config
    if "inhibitory_homeostasis" in lsc_kwargs and isinstance(
        lsc_kwargs["inhibitory_homeostasis"], dict
    ):
        lsc_kwargs["inhibitory_homeostasis"] = InhibitoryHomeostasisConfig(
            **lsc_kwargs["inhibitory_homeostasis"]
        )

    return LocalRuleConfig(**lsc_kwargs)


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
