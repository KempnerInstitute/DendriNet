"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from omegaconf import OmegaConf

from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.model import ModelConfig
from dendritic_modeling.config.training import (
    EncoderTrainingConfig,
    FiveFactorConfig,
    FourFactorConfig,
    HSICConfig,
    LocalRuleConfig,
    MainTrainingConfig,
    MorphologyAwareConfig,
    ThreeFactorConfig,
    TrainingConfig,
)


@dataclass
class ExperimentConfig(BaseConfig):
    """Basic experiment settings."""

    seed: int = 42
    train_valid_split: float = 0.8
    enable_amp: bool = False
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
            "stimulus_duration": 1.0,
            "max_gain_factor": None,
            "max_gain_tau_ratio": None,
            "uniform_gain": True
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
    log_freq: int = 100
    log_gradients: bool = False
    log_parameters: bool = False
    log_activations: bool = False


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
        encoder_config = EncoderTrainingConfig(**mapped_training.get("encoder", {}))
        main_config = MainTrainingConfig(**mapped_training.get("main", {}))
        training_config = TrainingConfig(encoder=encoder_config, main=main_config)

        data_raw = dict(conf_data.get("data", {}))
        if "processing" in data_raw and not isinstance(data_raw["processing"], DataProcessingConfig):
            data_raw["processing"] = DataProcessingConfig(**data_raw["processing"])

        return Config(
            experiment=ExperimentConfig(**conf_data.get("experiment", {})),
            data=DataConfig(**data_raw),
            model=ModelConfig(**conf_data.get("model", {})),
            training=training_config,
            analysis=AnalysisConfig(**mapped_analysis),
            wandb=WandbConfig(**conf_data.get("wandb", {})),
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
        "receptive_field": "receptive_field_analysis",
        "single_layer_contribution": "single_layer_contribution",
        "synaptic_activation": "synaptic_activation_analysis",
        "branch_activation": "branch_activation_analysis",
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

    return LocalRuleConfig(**lsc_kwargs)


def _map_training_config(training_data: dict) -> dict:
    """Map training config fields and handle nested learning_strategy_config."""
    # Convert OmegaConf to regular dict
    mapped = (
        OmegaConf.to_container(training_data, resolve=True)
        if hasattr(training_data, "_metadata")
        else dict(training_data)
    )

    # Handle nested learning_strategy_config in main config
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

        mapped["main"] = main_data

    return mapped
