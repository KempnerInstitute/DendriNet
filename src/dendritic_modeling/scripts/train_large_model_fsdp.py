#!/usr/bin/env python
"""
train_large_model_fsdp.py
=========================
Training script for large dendritic models using FSDP.

This script is designed for models that cannot fit on a single GPU
and require model parallelism through FSDP.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=4 train_large_model_fsdp.py configs/large_model.yaml
    
    # Multi-node with SLURM
    srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE train_large_model_fsdp.py configs/large_model.yaml
"""

import os
import argparse
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from typing import Dict, Any

from dendritic_modeling import logger
from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer
from dendritic_modeling.training.fsdp_utils import (
    estimate_memory_requirements,
    get_recommended_gpu_count,
    get_fsdp_config,
)
from dendritic_modeling.utils import set_seed


def setup_distributed():
    """Initialize distributed environment."""
    # Get rank and world size from torchrun
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device
    torch.cuda.set_device(rank)
    
    return rank, world_size


def create_large_model(config: Dict[str, Any]):
    """Create a large dendritic model from configuration."""
    model_config = config['model']
    
    # Log model size estimates
    if 'core_network' in model_config:
        core_params = model_config['core_network']['parameters']
        if all(k in core_params for k in ['excitatory_layer_sizes', 'inhibitory_layer_sizes']):
            # Estimate memory requirements
            all_neurons = (
                [core_params.get('input_dim', 784)] + 
                core_params['excitatory_layer_sizes']
            )
            
            # Estimate synapses per neuron from configuration
            ee_synapses = core_params.get('ee_synapses_per_branch_per_layer', [100])
            branches = core_params.get('excitatory_branch_factors', [1])
            
            memory_est = estimate_memory_requirements(
                num_neurons=all_neurons,
                num_synapses_per_neuron=ee_synapses,
                num_branches=branches,
                dtype=torch.float16 if config.get('mixed_precision', True) else torch.float32
            )
            
            logger.info(f"Estimated model size: {memory_est['total_params']:,} parameters")
            logger.info(f"Estimated memory requirement: {memory_est['total_gb']:.1f} GB")
            
            # Get GPU recommendation
            gpu_memory = 80  # H100 memory
            recommended_gpus = get_recommended_gpu_count(gpu_memory, memory_est['total_gb'])
            logger.info(f"Recommended number of GPUs: {recommended_gpus}")
    
    # Create encoder
    encoder_config = model_config['encoder_network']
    if encoder_config['type'].lower() == 'identity':
        encoder_network = Identity()
    else:
        encoder_network = get_architecture(
            encoder_config['type'],
            **encoder_config.get('parameters', {})
        )
    
    # Create core network
    core_config = model_config['core_network']
    core_network = get_architecture(
        core_config['type'],
        **core_config['parameters']
    )
    
    # Create decoder
    decoder_config = model_config['decoder_network']
    if decoder_config['type'].lower() == 'identity':
        decoder_network = Identity()
    else:
        decoder_params = decoder_config.get('parameters', {}).copy()
        if hasattr(core_network, 'output_dim'):
            decoder_params['input_dim'] = core_network.output_dim
        elif hasattr(core_network, 'excitatory_layer_sizes'):
            decoder_params['input_dim'] = core_network.excitatory_layer_sizes[-1]
        
        decoder_network = get_architecture(
            decoder_config['type'],
            **decoder_params
        )
    
    # Create model
    if model_config['task'] == 'classification':
        model = Classifier(
            encoder_network=encoder_network,
            net=core_network,
            decoder_network=decoder_network
        )
    else:
        model = Regressor(
            encoder_network=encoder_network,
            net=core_network,
            decoder_network=decoder_network
        )
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train large dendritic models with FSDP')
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='Use mixed precision training')
    parser.add_argument('--cpu_offload', action='store_true',
                        help='Offload parameters to CPU to save GPU memory')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for saving checkpoints')
    parser.add_argument('--model_size', type=str, default='large',
                        choices=['small', 'large', 'xlarge'],
                        help='Model size category for FSDP configuration')
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size = setup_distributed()
    
    # Load configuration
    config = load_config(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Set seed
    set_seed(config_dict['train'].get('seed', 0) + rank)
    
    # Log setup info
    if rank == 0:
        logger.info(f"Starting FSDP training on {world_size} GPUs")
        logger.info(f"Configuration: {args.config}")
        logger.info(f"Mixed precision: {args.mixed_precision}")
        logger.info(f"CPU offload: {args.cpu_offload}")
    
    # Create model
    model = create_large_model(config_dict)
    
    # Load datasets
    task_config = config_dict['task']
    train_config = config_dict['train']
    
    train_ds, valid_ds, test_ds = get_unified_datasets(
        dataset_name=task_config['dataset'],
        data_path=task_config.get('data_path', './data'),
        train_valid_split=train_config.get('train_valid_split', 0.8),
        flatten=task_config.get('flatten', False),
        normalize=task_config.get('normalize', True),
        num_classes=task_config.get('num_classes', 10),
        train_size=task_config.get('train_size'),
        valid_size=task_config.get('valid_size'),
        test_size=task_config.get('test_size')
    )
    
    # Get FSDP configuration
    fsdp_config = get_fsdp_config(
        model_size=args.model_size,
        mixed_precision=args.mixed_precision,
        cpu_offload=args.cpu_offload,
        sharding_strategy="FULL_SHARD"  # Best for large models
    )
    
    # Create optimizer configuration
    optimizer_config = train_config['main']['param_groups']
    if optimizer_config.get('split_params', False):
        # Complex optimizer setup handled by FSDPTrainer
        optimizer_params = optimizer_config
    else:
        optimizer_params = {
            'lr': optimizer_config.get('lr', 0.001),
            'weight_decay': train_config['main'].get('weight_decay_rate', 0.0)
        }
    
    # Create optimizer
    # Note: We create optimizer after FSDP wrapping in the trainer
    optimizer = None
    
    # Get trainer configuration
    trainer_config = train_config['main']['trainer'].copy()
    
    # Create FSDP trainer
    trainer = FSDPTrainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        checkpoint_dir=args.checkpoint_dir or config_dict['outputs']['dir'],
        optimizer=optimizer,  # Will be created after FSDP wrapping
        **trainer_config
    )
    
    # Create optimizer after model is wrapped (done inside trainer)
    # This is handled internally by FSDPTrainer
    
    # Run training
    if rank == 0:
        logger.info("Starting training...")
    
    results = trainer.train(model, train_ds, valid_ds)
    
    # Log results
    if rank == 0:
        logger.info(f"Training completed!")
        logger.info(f"Best epoch: {results['best_epoch']}")
        logger.info(f"Best loss: {results['best_loss']:.4f}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main() 