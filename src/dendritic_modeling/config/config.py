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
from dendritic_modeling.config.training import TrainingConfig


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
        default_factory=lambda: {"base_dataset": "mnist", "rate_multiplier": 1.0}
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

        # Handle analysis config field mapping
        analysis_data = raw_conf.get("analysis", {})
        mapped_analysis = _map_analysis_config(analysis_data)

        return Config(
            experiment=ExperimentConfig(**raw_conf.get("experiment", {})),
            data=DataConfig(**raw_conf.get("data", {})),
            model=ModelConfig(**raw_conf.get("model", {})),
            training=TrainingConfig(**raw_conf.get("training", {})),
            analysis=AnalysisConfig(**mapped_analysis),
            wandb=WandbConfig(**raw_conf.get("wandb", {})),
            outputs=OutputsConfig(**raw_conf.get("outputs", {})),
            fsdp=FSDPConfig(**raw_conf.get("fsdp", {})),
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
