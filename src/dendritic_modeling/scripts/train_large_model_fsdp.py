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
import numpy as np

from dendritic_modeling import logger
from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.utils.data import set_seed
from dendritic_modeling.utils.gpu_estimation import GPUMemoryEstimator
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.training.fsdp_utils import (
    get_fsdp_config,
    estimate_memory_requirements,
    get_recommended_gpu_count,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint
)
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer


def setup_distributed():
    """Initialize distributed environment."""
    # Check if we're in a SLURM environment without torchrun
    if 'SLURM_PROCID' in os.environ and 'RANK' not in os.environ:
        # Set up environment variables from SLURM
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        
        # Set the environment variables that PyTorch expects
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # MASTER_ADDR and MASTER_PORT should already be set by SLURM script
        if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
            raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set in SLURM environment")
    else:
        # Get rank and world size from torchrun
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device using local rank (rank within the node)
    torch.cuda.set_device(local_rank)
    
    return rank, world_size


def create_large_model(config):
    """Create a large dendritic model from configuration."""
    model_config = config.model
    
    # Get input dimension from task config
    task_params = config.task.parameters or {}
    if config.task.dataset == 'mnist':
        input_dim = 784  # 28x28 flattened
    elif config.task.dataset == 'cifar10':
        input_dim = 3072  # 32x32x3 flattened
    elif config.task.dataset == 'imagenet':
        # Check if images are flattened
        if task_params.get('flatten', False):
            input_dim = 150528  # 224x224x3 flattened
        else:
            # For non-flattened, the encoder should handle the shape
            input_dim = task_params.get('input_dim', 150528)
    else:
        input_dim = task_params.get('input_dim', 784)
    
    # Create encoder
    encoder_config = model_config.encoder_network
    if encoder_config.type.lower() == 'identity':
        encoder = Identity(input_dim)
    else:
        encoder_params = dict(encoder_config.parameters) if encoder_config.parameters else {}
        encoder_params['input_dim'] = input_dim
        encoder_network = get_architecture(encoder_config.type, encoder_params)
        encoder = encoder_network.encoder if hasattr(encoder_network, 'encoder') else encoder_network
    
    # Create core network
    core_config = model_config.core_network
    core_params = dict(core_config.parameters) if core_config.parameters else {}
    if hasattr(encoder, 'output_dim'):
        core_params['input_dim'] = encoder.output_dim
    core_network = get_architecture(core_config.type, core_params)
    
    # Create decoder
    decoder_config = model_config.decoder_network
    if decoder_config.type.lower() == 'identity':
        decoder = Identity(core_network.output_dim if hasattr(core_network, 'output_dim') else core_network.excitatory_layer_sizes[-1])
    else:
        decoder_params = dict(decoder_config.parameters) if decoder_config.parameters else {}
        if hasattr(core_network, 'output_dim'):
            decoder_params['input_dim'] = core_network.output_dim
        elif hasattr(core_network, 'excitatory_layer_sizes'):
            decoder_params['input_dim'] = core_network.excitatory_layer_sizes[-1]
        decoder = get_architecture(decoder_config.type, decoder_params)
    
    # Create model based on task
    if model_config.task == 'classification':
        model = Classifier(encoder, core_network, decoder)
    elif model_config.task == 'regression':
        model = Regressor(encoder, core_network, decoder)
    else:
        raise ValueError(f"Unknown task: {model_config.task}")
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
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
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config.train.seed + rank)
    
    # Log setup info
    if rank == 0:
        logger.info(f"Starting FSDP training on {world_size} GPUs")
        logger.info(f"Model size: {args.model_size}")
        logger.info(f"Mixed precision: {args.mixed_precision}")
    
    # Create model
    model = create_large_model(config)
    
    # Comprehensive GPU memory estimation (only on rank 0)
    if rank == 0:
        # Get batch size from config
        batch_size = train_config.main.trainer.batch_size
        
        # Get input dimensions
        task_params = config.task.parameters or {}
        if config.task.dataset == 'mnist':
            input_shape = (784,)
        elif config.task.dataset == 'cifar10':
            input_shape = (3072,)
        elif config.task.dataset == 'imagenet':
            if task_params.get('flatten', False):
                input_shape = (150528,)
            else:
                input_shape = (3, 224, 224)
        else:
            input_dim = task_params.get('input_dim', 784)
            input_shape = (input_dim,)
        
        # Get precision and optimizer settings
        dtype = 'fp16' if args.mixed_precision else 'fp32'
        optimizer_type = train_config.main.param_groups.get('optimizer', 'adam')
        
        # Create GPU memory estimator
        estimator = GPUMemoryEstimator(dtype=dtype, optimizer=optimizer_type)
        
        # Print comprehensive estimation report
        estimator.print_estimation_report(model, batch_size, input_shape, config)
        
        # Check if current setup is adequate
        memory_breakdown = estimator.estimate_memory_breakdown(
            model, batch_size, input_shape, use_fsdp=True, num_gpus=world_size
        )
        
        if torch.cuda.is_available():
            current_gpu = torch.cuda.get_device_properties(0)
            current_memory_gb = current_gpu.total_memory / (1024**3)
            
            if memory_breakdown.total > current_memory_gb * 0.85:
                logger.warning("\n⚠️  WARNING: Model may not fit with current GPU setup!")
                logger.warning(f"  Required: {memory_breakdown.total:.2f}GB per GPU")
                logger.warning(f"  Available: {current_memory_gb * 0.85:.2f}GB (with safety margin)")
                
                if args.cpu_offload:
                    logger.info("  CPU offloading is enabled, which will help reduce GPU memory usage.")
                else:
                    logger.warning("  Consider:")
                    logger.warning("    1. Using more GPUs")
                    logger.warning("    2. Enabling CPU offloading with --cpu_offload")
                    logger.warning("    3. Reducing batch size")
                    logger.warning("    4. Using gradient checkpointing")
        
        logger.info("")  # Empty line for spacing
    
    # Load datasets
    task_config = config.task
    train_config = config.train
    
    # Convert configs to dict format for get_unified_datasets
    task_dict = {
        'dataset': task_config.dataset,
        'data_path': task_config.data_path,
        'parameters': task_config.parameters or {}
    }
    train_dict = {
        'train_valid_split': train_config.train_valid_split
    }
    
    train_ds, valid_ds, test_ds = get_unified_datasets(task_dict, train_dict, outputs_cfg=config.outputs)
    
    # Setup analysis manager (only on rank 0 to avoid duplicate file writes)
    analysis_manager = None
    # Only create AnalysisManager if running on a single process. Running evaluation that
    # performs forward passes on just one rank while the remaining ranks sit idle causes
    # mismatched collective operations in FSDP and eventually NCCL watchdog timeouts.
    # Therefore, we disable analysis when world_size > 1 (i.e. multi-GPU or multi-node).
    if world_size == 1 and hasattr(config, 'analysis'):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = getattr(config.outputs, 'run_name', 'fsdp_run')
        run_id = f"{run_name}_{timestamp}"
        run_save_path = os.path.join(config.outputs.dir, run_id)
        os.makedirs(run_save_path, exist_ok=True)
        
        encoded_data = {
            "train": train_ds,
            "valid": valid_ds,
            "test": test_ds,
        }
        analysis_manager = AnalysisManager(
            model=model,
            data=encoded_data,
            analysis_config=config.analysis,
            save_root=run_save_path,
            device=f'cuda:{local_rank}',
            train_config=train_config,
        )
        logger.info(f"Created analysis manager with output directory: {run_save_path}")
    
    # Create data loaders with FSDP-aware sampler
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False) if valid_ds else None
    
    batch_size = train_config.main.trainer.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, sampler=valid_sampler, num_workers=4, pin_memory=True) if valid_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) if test_ds else None
    
    # Get FSDP configuration
    fsdp_config = get_fsdp_config(
        model_size=args.model_size,
        mixed_precision=args.mixed_precision,
        cpu_offload=args.cpu_offload,
        sharding_strategy="FULL_SHARD"  # Best for large models
    )
    
    # Get optimizer configuration from config
    optimizer_config = {
        'lr': train_config.main.param_groups.lr,
        'weight_decay': train_config.main.weight_decay_rate
    }
    if hasattr(train_config.main.param_groups, 'split_params') and train_config.main.param_groups.split_params:
        optimizer_config.update({
            'topk_lr': train_config.main.param_groups.topk_lr,
            'blocklinear_lr': train_config.main.param_groups.blocklinear_lr,
            'reactivation_lr': train_config.main.param_groups.reactivation_lr,
            'decoder_lr': train_config.main.param_groups.decoder_lr
        })
    
    # Get trainer configuration
    trainer_config = {
        'epochs': train_config.main.trainer.epochs,
        'loss_function': train_config.main.trainer.loss_function,
        'early_stopping': train_config.main.trainer.early_stopping,
        'patience': train_config.main.trainer.patience,
        'grad_clip_value': train_config.main.trainer.grad_clip_value,
        'print_every': train_config.main.trainer.print_every,
        'suppress_prints': train_config.main.trainer.suppress_prints,
        'plot_losses': train_config.main.trainer.plot_losses,
        'save_checkpoints': getattr(train_config.main.trainer, 'save_checkpoints', True)
    }
    
    # Create FSDP trainer
    save_checkpoints = trainer_config.pop('save_checkpoints', True)  # Remove from trainer_config
    
    # Set save_path for loss plotting (use the analysis manager's save path if available)
    if rank == 0 and analysis_manager is not None:
        trainer_config['save_path'] = analysis_manager.save_root
    else:
        trainer_config['save_path'] = args.checkpoint_dir or config.outputs.dir
    
    trainer = FSDPTrainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        checkpoint_dir=args.checkpoint_dir or config.outputs.dir,
        save_every_n_epochs=getattr(train_config.main.trainer, 'save_every_n_epochs', 1),
        save_checkpoints=save_checkpoints,
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        **trainer_config
    )
    
    # Run training
    if rank == 0:
        logger.info("Starting training...")
    
    results = trainer.train(model, train_loader, valid_loader)
    
    # Log results
    if rank == 0:
        logger.info(f"Training completed!")
        if 'best_valid_loss' in results and results['best_valid_loss'] is not None:
            logger.info(f"Best validation loss: {results['best_valid_loss']:.4f}")
        if 'final_train_loss' in results and results['final_train_loss'] is not None:
            logger.info(f"Final train loss: {results['final_train_loss']:.4f}")
        if 'final_valid_loss' in results and results['final_valid_loss'] is not None:
            logger.info(f"Final validation loss: {results['final_valid_loss']:.4f}")
    
    # Final analysis run (single-process only)
    if world_size == 1 and analysis_manager is not None:
        try:
            logger.info("Running final analysis...")
            analysis_manager.run_analysis(filename="final", training=False)
            logger.info("Final analysis completed")
        except Exception as e:
            logger.warning(f"Final analysis failed: {e}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    main() 