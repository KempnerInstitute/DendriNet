"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf as om


class BaseConfig:
    @classmethod
    def load(cls, path: str):
        conf = om.load(path)
        return cls(**conf)

    def save(self, path: str):
        om.save(config=self, f=path)

    def asdict(self):
        return asdict(self)


@dataclass
class EINetParameters(BaseConfig):
    input_dim: int = 784
    excitatory_layer_sizes: List[int] = field(default_factory=lambda: [10])
    inhibitory_layer_sizes: List[int] = field(default_factory=lambda: [20])
    excitatory_branch_factors: List[int] = field(default_factory=lambda: [2, 2])
    inhibitory_branch_factors: List[int] = field(default_factory=lambda: [])
    ee_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [24])
    ei_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [200])
    ie_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [1])
    ii_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [])
    reactivate: bool = True
    reactivation_init_m: float = 1.5
    reactivation_init_b: float = 0.5
    somatic_synapses: bool = True
    topk_init_method: str = "xavier_normal"
    topk_noise_level: float = 0.0
    topk_type: str = "standard"  # can be "standard", "stochastic", or "variance"
    topk_temperature: float = 0.5
    topk_ultrafast: bool = False
    topk_strategy: str = "conductance_dynamic"
    use_shunting: bool = True
    reactivation_type: str = "param_tanh"
    reactivation_strategy: str = "inverse"
    blocklinear_strategy: str = "conductance_dynamic"
    print_hooks: bool = False  # Whether to print gradient statistics in hooks
    input_mode: int = 0  # Input mode (0=original, 1=exc=x+inh=x, 2=input_net)
    input_net_dict: Dict[str, Any] = field(
        default_factory=dict
    )  # Input network configuration
    enable_branch_outputs: bool = True
    output_layer: bool = False
    output_dim: int = 10
    excitatory_target_density: float = 0.1
    inhibitory_target_density: float = 0.1
    use_noise: bool = True
    sigma: float = 0.05
    rewiring_mode: str = "global"
    synapses_per_branch: Optional[int] = None
    freeze_excitatory_connectivity: bool = False
    freeze_inhibitory_connectivity: bool = False
    init_method: str = "xavier_normal"
    weight_threshold: float = 1e-6


@dataclass
class FeedForwardParameters(BaseConfig):
    input_dim: int = 784
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10


@dataclass
class ModelNetworkConfig(BaseConfig):
    type: str = "EINet"
    parameters: EINetParameters = field(default_factory=EINetParameters)


@dataclass
class ModelConfig(BaseConfig):
    task: str = "classification"
    probabilistic: bool = True
    network: ModelNetworkConfig = field(default_factory=ModelNetworkConfig)


@dataclass
class TrainConfig(BaseConfig):
    learning_strategy: str = "mle"
    run_name: str = "exp_run"
    seed: int = 0
    train_valid_split: float = 0.8
    optim_dict: Dict[str, Any] = field(
        default_factory=dict
    )  # Optimizer / param group settings
    weight_decay_rate: float = 0.1
    epochs: int = 50
    early_stopping: bool = True
    patience: int = 10
    batch_size: int = 128
    grad_clip_value: float = 5.0
    epochs_per_layer: int = 50
    final_tune_epochs: int = (
        10  # Number of epochs for final fine-tuning after layer-by-layer training
    )
    pretrain_epochs: int = 50
    input_net_training: dict = field(default_factory=dict)

    # Newly added fields for "voltage_stabilization" strategy:
    warmup_epochs: int = 5  # how many epochs to do MSE warmup
    stability_target_voltage: float = 0.5  # the target "voltage" used in warmup


@dataclass
class VisualizationConfig(BaseConfig):
    enabled: bool = False
    visualize_epochs: List[int] = field(default_factory=lambda: [])
    plots: List[str] = field(default_factory=lambda: ["branch_info", "ablation"])
    parameters: Dict = field(default_factory=dict)


@dataclass
class AnalysisConfig(BaseConfig):
    ei_info_analysis: Dict = field(default_factory=dict)
    synapse_turnover_analysis: Dict = field(default_factory=dict)
    noise_perturbation_analysis: Dict = field(default_factory=dict)
    single_layer_contribution: Dict = field(default_factory=dict)
    synaptic_ablation: Dict = field(default_factory=dict)
    branch_information: Dict = field(default_factory=dict)


@dataclass
class WandbConfig(BaseConfig):
    entity: str = ""
    project: str = ""
    group: str = ""
    tags: List[str] = field(default_factory=list)
    use_wandb: bool = False


@dataclass
class TaskConfig(BaseConfig):
    dataset: str = "mnist"
    data_path: str = ""
    parameters: Dict = field(default_factory=dict)


@dataclass
class OutputsConfig(BaseConfig):
    dir: str = "outputs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def load_config(path: str = None) -> Config:
    """Load configuration from a file or use defaults."""
    if path:
        raw_conf = om.load(path)
        return Config(
            model=ModelConfig(**raw_conf.get("model", {})),
            train=TrainConfig(**raw_conf.get("train", {})),
            wandb=WandbConfig(**raw_conf.get("wandb", {})),
            visualization=VisualizationConfig(**raw_conf.get("visualization", {})),
            analysis=AnalysisConfig(**raw_conf.get("analysis", {})),
            task=TaskConfig(**raw_conf.get("task", {})),
            outputs=OutputsConfig(**raw_conf.get("outputs", {})),
        )
    return Config()
