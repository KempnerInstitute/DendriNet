"""
Optimizer Factory
================

Factory functions for creating optimizers with flexible configuration.
"""

from typing import Dict, Any, Iterator
import torch
import torch.optim as optim

from dendritic_modeling.config.training import OptimizerConfig


def create_optimizer(
    model_parameters: Iterator[torch.nn.Parameter],
    config: OptimizerConfig
) -> torch.optim.Optimizer:
    """
    Create an optimizer based on configuration.
    
    Args:
        model_parameters: Iterator over model parameters to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If optimizer name is not supported
    """
    optimizer_name = config.name.lower()
    
    # Prepare parameters dictionary
    params = {
        'lr': config.lr,
        'weight_decay': config.weight_decay
    }
    
    if optimizer_name == "adam":
        params.update({
            'betas': config.betas,
            'eps': config.eps
        })
        return optim.Adam(model_parameters, **params)
        
    elif optimizer_name == "adamw":
        params.update({
            'betas': config.betas,
            'eps': config.eps
        })
        return optim.AdamW(model_parameters, **params)
        
    elif optimizer_name == "sgd":
        params.update({
            'momentum': config.momentum
        })
        return optim.SGD(model_parameters, **params)
        
    elif optimizer_name == "rmsprop":
        params.update({
            'eps': config.eps,
            'momentum': config.momentum
        })
        return optim.RMSprop(model_parameters, **params)
        
    else:
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop"]
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Choose from {valid_optimizers}"
        )


def create_parameter_groups(
    model: torch.nn.Module,
    config: OptimizerConfig,
    param_groups_config: Dict[str, Any] = None
) -> list:
    """
    Create parameter groups for differential learning rates.
    
    Args:
        model: The model to create parameter groups for
        config: Optimizer configuration
        param_groups_config: Optional configuration for parameter groups
        
    Returns:
        List of parameter groups
    """
    if param_groups_config is None or not param_groups_config.get('split_params', False):
        # Return all parameters with default config
        return [{'params': model.parameters(), 'lr': config.lr}]
    
    # Create differential parameter groups
    param_groups = []
    
    # Default group
    default_params = []
    
    # Special groups for different component types
    topk_params = []
    blocklinear_params = []
    reactivation_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if 'topk' in name.lower():
            topk_params.append(param)
        elif 'blocklinear' in name.lower():
            blocklinear_params.append(param)
        elif 'reactivation' in name.lower():
            reactivation_params.append(param)
        elif 'decoder' in name.lower():
            decoder_params.append(param)
        else:
            default_params.append(param)
    
    # Add groups if they have parameters
    if default_params:
        param_groups.append({
            'params': default_params,
            'lr': config.lr
        })
    
    if topk_params:
        param_groups.append({
            'params': topk_params,
            'lr': param_groups_config.get('topk_lr', config.lr)
        })
    
    if blocklinear_params:
        param_groups.append({
            'params': blocklinear_params,
            'lr': param_groups_config.get('blocklinear_lr', config.lr)
        })
    
    if reactivation_params:
        param_groups.append({
            'params': reactivation_params,
            'lr': param_groups_config.get('reactivation_lr', config.lr)
        })
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': param_groups_config.get('decoder_lr', config.lr)
        })
    
    return param_groups


def validate_optimizer_config(config: OptimizerConfig) -> None:
    """
    Validate optimizer configuration.
    
    Args:
        config: Optimizer configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    valid_optimizers = ["adam", "adamw", "sgd", "rmsprop"]
    if config.name.lower() not in valid_optimizers:
        raise ValueError(
            f"Invalid optimizer: {config.name}. "
            f"Choose from {valid_optimizers}"
        )
    
    if config.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {config.lr}")
    
    if config.weight_decay < 0:
        raise ValueError(f"Weight decay must be non-negative, got {config.weight_decay}")
    
    if config.name.lower() in ["adam", "adamw"]:
        if len(config.betas) != 2 or not all(0 <= b < 1 for b in config.betas):
            raise ValueError(f"Invalid betas for {config.name}: {config.betas}")
        
        if config.eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {config.eps}")
    
    if config.name.lower() in ["sgd", "rmsprop"]:
        if config.momentum < 0:
            raise ValueError(f"Momentum must be non-negative, got {config.momentum}")