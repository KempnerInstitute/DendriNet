from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import os
import yaml

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
    inhibitory_layer_sizes: List[int] = field(default_factory=lambda: [10])
    excitatory_branch_factors: List[int] = field(default_factory=lambda: [2,2])
    inhibitory_branch_factors: List[int] = field(default_factory=lambda: [])
    ee_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [24])
    ei_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [1])
    ie_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [1])
    ii_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [1])
    reactivate: bool = True
    somatic_synapses: bool = True
    topk_init_method: str = "xavier_normal"
    use_shunting: bool = True
    reactivation_type: str = "param_tanh"
    reactivation_strategy: str = "inverse"
    blocklinear_strategy: str = "block_conductance_dynamic"
    enable_branch_outputs: bool = False
    output_layer: bool = False
    output_dim: int = 10
    local_loss_weight: float = 0.0
    learning_strategy: str = "mle"

    # 0=original, 1=direct, 2=two-headed MLP
    input_mode: int = 0

    # Dictionary describing how to build the input network if input_mode=2
    input_net_dict: Dict[str, Any] = field(default_factory=dict)


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
    run_name: str = "mnist_einet"
    seed: int = 0
    train_valid_split: float = 0.8
    lr: float = 0.01
    weight_decay_rate: float = 0.1
    epochs: int = 50
    batch_size: int = 128
    grad_clip_value: float = 5.0
    epochs_per_layer: int = 50
    pretrain_epochs: int = 50
    
    # Parameters for mlp_only strategy
    mlp_epochs: int = 20
    full_epochs: int = 30
    
    # Parameters for dendrinet_from_mlp strategy
    dendrinet_epochs: int = 30
    freeze_mlp: bool = False
    
    # Input network training configuration
    input_net_training: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VisualizationConfig(BaseConfig):
    enabled: bool = False
    visualize_epochs: List[int] = field(default_factory=lambda: [])
    plots: List[str] = field(default_factory=lambda: ["branch_info", "ablation"])
    branch_ablation: bool = True
    layer_ablation: bool = True

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
    enable_ablation: bool = True
    ablation_batch_size: int = 64
    enable_info_estimation: bool = True
    info_estimation_epochs: int = 20

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig) 
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def load_config(config_path: str) -> Config:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Config object with all configuration parameters
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create model config
    network_params = ModelNetworkConfig(
        type=config_dict['model']['network']['type'],
        parameters=EINetParameters(**config_dict['model']['network']['parameters'])
    )
    
    model_config = ModelConfig(
        task=config_dict['model']['task'],
        probabilistic=config_dict['model']['probabilistic'],
        network=network_params
    )
    
    # Create train config
    train_config = TrainConfig(
        learning_strategy=config_dict['train']['learning_strategy'],
        run_name=config_dict['train']['run_name'],
        seed=config_dict['train']['seed'],
        train_valid_split=config_dict['train']['train_valid_split'],
        lr=config_dict['train']['lr'],
        epochs=config_dict['train']['epochs'],
        batch_size=config_dict['train']['batch_size'],
        grad_clip_value=config_dict['train']['grad_clip_value'],
        epochs_per_layer=config_dict['train'].get('epochs_per_layer', 50),
        pretrain_epochs=config_dict['train'].get('pretrain_epochs', 50),
        weight_decay_rate=config_dict['train'].get('weight_decay_rate', 0.1)
    )
    
    # Add input_net_training if present
    if 'input_net_training' in config_dict['train']:
        train_config.input_net_training = config_dict['train']['input_net_training']
    
    # Create task config
    task_config = TaskConfig(
        dataset=config_dict['task']['dataset'],
        data_path=config_dict['task']['data_path'],
        parameters=config_dict['task'].get('parameters', {})
    )
    
    # Create outputs config
    outputs_config = OutputsConfig(
        dir=config_dict['outputs']['dir']
    )
    
    # Add optional output parameters if present
    if 'enable_ablation' in config_dict['outputs']:
        outputs_config.enable_ablation = config_dict['outputs']['enable_ablation']
    
    if 'ablation_batch_size' in config_dict['outputs']:
        outputs_config.ablation_batch_size = config_dict['outputs']['ablation_batch_size']
        
    if 'enable_info_estimation' in config_dict['outputs']:
        outputs_config.enable_info_estimation = config_dict['outputs']['enable_info_estimation']
        
    if 'info_estimation_epochs' in config_dict['outputs']:
        outputs_config.info_estimation_epochs = config_dict['outputs']['info_estimation_epochs']
    
    # Create visualization config
    vis_config = VisualizationConfig(
        enabled=config_dict['visualization']['enabled'],
        visualize_epochs=config_dict['visualization'].get('visualize_epochs', []),
        plots=config_dict['visualization'].get('plots', [])
    )
    
    # Create wandb config if present
    wandb_config = None
    if 'wandb' in config_dict:
        wandb_config = WandbConfig(
            entity=config_dict['wandb']['entity'],
            project=config_dict['wandb']['project'],
            group=config_dict['wandb']['group'],
            tags=config_dict['wandb'].get('tags', []),
            use_wandb=config_dict['wandb'].get('use_wandb', False)
        )
    
    # Create main config
    config = Config(
        model=model_config,
        train=train_config,
        wandb=wandb_config,
        visualization=vis_config,
        task=task_config,
        outputs=outputs_config
    )
    
    return config