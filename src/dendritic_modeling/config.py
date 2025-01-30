"""
config.py
=========
Contains data classes and a loader for experiment configurations using OmegaConf.
"""

from dataclasses import dataclass, field
from typing import List
from omegaconf import OmegaConf as om

@dataclass
class EINetParameters:
    """
    Example parameters for the ExcitationInhibitionNetwork.
    (Now with local_learning removed.)
    """
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
    somatic_synapses: bool = True
    topk_init_method: str = "xavier_normal"
    use_shunting: bool = True

    init_gain: float = 1.0
    init_inhib_offset: float = 1.3
    reactivation_initial_m: float = 2.0
    reactivation_initial_b: float = 0.5
    reactivation_type: str = "param_tanh"
    train_reactivation_b_only: bool = False

    gradient_strategy: str = "block_conductance_dynamic"
    gradient_scale_factor: float = 2.0
    weight_decay_rate: float = 0.1

    enable_branch_outputs: bool = True
    output_layer: bool = False
    output_dim: int = 10


@dataclass
class ModelNetworkConfig:
    """
    Container for specifying the network type and parameters.
    """
    type: str = "EINet"
    parameters: EINetParameters = field(default_factory=EINetParameters)

@dataclass
class ModelConfig:
    """
    Model-level config. 
    """
    task: str = "classification"
    probabilistic: bool = True
    network: ModelNetworkConfig = field(default_factory=ModelNetworkConfig)

@dataclass
class TrainConfig:
    """
    Training hyperparameters for whichever strategy.
    """
    learning_strategy: str = "two_step" 
    run_name: str = "mnist_einet"
    seed: int = 0
    train_valid_split: float = 0.8
    lr: float = 0.01
    epochs: int = 50
    batch_size: int = 1000
    grad_clip_value: float = 5.0
    epochs_per_layer: int = 50

@dataclass
class VisualizationConfig:
    enabled: bool = True
    save_path: str = "/n/holylabs/LABS/kempner_dev/Users/hsafaai/results/dendritic_modeling/results"
    plots: List[str] = field(default_factory=lambda: ["branch_info", "ablation"])

@dataclass
class WandbConfig:
    entity: str = ""
    project: str = ""
    group: str = ""
    tags: List[str] = field(default_factory=list)
    # If you have a 'use_wandb' or something
    use_wandb: bool = False

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

def load_config(path: str):
    """
    Loads the config YAML into our dataclasses.
    """
    from omegaconf import OmegaConf
    base = Config()
    if not path:
        return base

    loaded = OmegaConf.load(path)

    # Merge 'model' block
    if "model" in loaded:
        # read model.task, model.probabilistic, etc.
        if "task" in loaded["model"]:
            base.model.task = loaded["model"]["task"]
        if "probabilistic" in loaded["model"]:
            base.model.probabilistic = loaded["model"]["probabilistic"]

        if "network" in loaded["model"]:
            net_type = loaded["model"]["network"].get("type", "EINet")
            base.model.network.type = net_type
            if "parameters" in loaded["model"]["network"]:
                params = loaded["model"]["network"]["parameters"]
                # We'll manually map them to EINetParameters
                for k, v in params.items():
                    setattr(base.model.network.parameters, k, v)

    # Merge 'train' block
    if "train" in loaded:
        for k, v in loaded["train"].items():
            if hasattr(base.train, k):
                setattr(base.train, k, v)

    # Merge 'wandb'
    if "wandb" in loaded:
        for k, v in loaded["wandb"].items():
            if hasattr(base.wandb, k):
                setattr(base.wandb, k, v)

    # Merge 'visualization'
    if "visualization" in loaded:
        for k, v in loaded["visualization"].items():
            if hasattr(base.visualization, k):
                setattr(base.visualization, k, v)

    return base