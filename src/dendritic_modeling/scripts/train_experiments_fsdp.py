#!/usr/bin/env python
"""
Large Model FSDP Training Script
================================

This script enables FSDP (Fully Sharded Data Parallel) training for large dendritic models
that don't fit on a single GPU.

Usage:
    torchrun --nproc_per_node=<num_gpus> train_experiments_fsdp.py <config_path>

Example:
    torchrun --nproc_per_node=4 train_experiments_fsdp.py configs/large_model_config.yaml
"""

import os
import argparse
import logging
from dataclasses import asdict
from datetime import datetime

# Set NCCL environment variables to fix communication issues
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30 minutes timeout
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault(
    "NCCL_SOCKET_IFNAME", "ib0,eth0"
)  # Try InfiniBand first, then Ethernet
os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand if available
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "2")  # Enable GPU Direct RDMA

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from dendritic_modeling import logger, logger_manager

# WandB import (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.utils.data import set_seed, save_dict
from dendritic_modeling.utils.gpu_estimation import GPUMemoryEstimator
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.training.fsdp_utils import get_fsdp_config
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer
from dendritic_modeling.training import CustomWeightDecayOptimizer
from dendritic_modeling.plotting.analysis import plot_information_analysis


def setup_wandb(config, rank, world_size):
    """Initialize WandB logging (only on rank 0)."""
    if rank != 0 or not WANDB_AVAILABLE:
        return None

    wandb_config = config.wandb
    if not wandb_config.use_wandb:
        return None

    # Initialize wandb
    run = wandb.init(
        project=wandb_config.project or "dendritic-modeling",
        entity=wandb_config.entity or None,
        name=getattr(wandb_config, "name", None) or config.train.run_name,
        tags=wandb_config.tags,
        notes=getattr(wandb_config, "notes", None),
        config={
            # Model config
            "model_type": config.model.core_network.type,
            "dataset": config.task.dataset,
            "batch_size": config.train.main.trainer.batch_size,
            "total_batch_size": config.train.main.trainer.batch_size * world_size,
            "epochs": config.train.main.trainer.epochs,
            "learning_rate": config.train.main.param_groups.lr,
            "weight_decay": config.train.main.weight_decay_rate,
            "world_size": world_size,
            # Model architecture details
            "excitatory_layers": config.model.core_network.parameters.get(
                "excitatory_layer_sizes", []
            ),
            "inhibitory_layers": config.model.core_network.parameters.get(
                "inhibitory_layer_sizes", []
            ),
            "excitatory_branches": config.model.core_network.parameters.get(
                "excitatory_branch_factors", []
            ),
            "inhibitory_branches": config.model.core_network.parameters.get(
                "inhibitory_branch_factors", []
            ),
        },
    )

    logger.info(f"WandB initialized: {run.url}")
    return run


def setup_distributed():
    """Initialize distributed environment."""
    # Check if we're in a SLURM environment without torchrun
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        # Set up environment variables from SLURM
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])

        # Set the environment variables that PyTorch expects
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # MASTER_ADDR and MASTER_PORT should already be set by SLURM script
        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            raise RuntimeError(
                "MASTER_ADDR and MASTER_PORT must be set in SLURM environment"
            )
    else:
        # Get rank and world size from torchrun
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed training
    # Use GLOO backend if specified (more robust for network issues)
    backend = os.environ.get("PYTORCH_BACKEND", "nccl")
    if backend == "gloo":
        logger.info("Using Gloo backend for distributed training")
    dist.init_process_group(backend=backend)

    # Set device using local rank (rank within the node)
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def create_large_model(config):
    """Create a large dendritic model from configuration."""
    model_config = config.model

    # Get input dimension from task config
    task_params = config.task.parameters or {}
    if config.task.dataset == "mnist":
        input_dim = 784  # 28x28 flattened
    elif config.task.dataset == "cifar10":
        input_dim = 3072  # 32x32x3 flattened
    elif config.task.dataset == "imagenet":
        # Check if images are flattened
        if task_params.get("flatten", False):
            input_dim = 150528  # 224x224x3 flattened
        else:
            # For non-flattened, the encoder should handle the shape
            input_dim = task_params.get("input_dim", 150528)
    else:
        input_dim = task_params.get("input_dim", 784)

    # Create encoder
    encoder_config = model_config.encoder_network
    if encoder_config.type.lower() == "identity":
        encoder = Identity(input_dim)
    else:
        encoder_params = (
            dict(encoder_config.parameters) if encoder_config.parameters else {}
        )
        encoder_params["input_dim"] = input_dim
        encoder_network = get_architecture(encoder_config.type, encoder_params)
        encoder = (
            encoder_network.encoder
            if hasattr(encoder_network, "encoder")
            else encoder_network
        )

    # Create core network
    core_config = model_config.core_network
    core_params = dict(core_config.parameters) if core_config.parameters else {}
    if hasattr(encoder, "output_dim"):
        core_params["input_dim"] = encoder.output_dim
    core_network = get_architecture(core_config.type, core_params)

    # Create decoder
    decoder_config = model_config.decoder_network
    if decoder_config.type.lower() == "identity":
        decoder = Identity(
            core_network.output_dim
            if hasattr(core_network, "output_dim")
            else core_network.excitatory_layer_sizes[-1]
        )
    else:
        decoder_params = (
            dict(decoder_config.parameters) if decoder_config.parameters else {}
        )
        if hasattr(core_network, "output_dim"):
            decoder_params["input_dim"] = core_network.output_dim
        elif hasattr(core_network, "excitatory_layer_sizes"):
            decoder_params["input_dim"] = core_network.excitatory_layer_sizes[-1]
        decoder = get_architecture(decoder_config.type, decoder_params)

    # Create model based on task
    if model_config.task == "classification":
        model = Classifier(encoder, core_network, decoder)
    elif model_config.task == "regression":
        model = Regressor(encoder, core_network, decoder)
    else:
        raise ValueError(f"Unknown task: {model_config.task}")

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model


def setup_data_loaders(config, rank, world_size):
    """Setup distributed data loaders."""
    # Get datasets
    train_ds, valid_ds, test_ds = get_unified_datasets(
        task_cfg=config.task,
        train_cfg=config.train,
        output_dir=config.outputs.dir,
    )
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_ds, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    valid_sampler = DistributedSampler(
        valid_ds, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    batch_size = config.train.main.trainer.batch_size
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, valid_loader, test_ds


def main(config_path):
    """Main FSDP training function."""
    # Setup distributed environment
    rank, local_rank, world_size = setup_distributed()
    
    if rank == 0:
        logger.info(f"Starting FSDP training on {world_size} GPUs")
    
    # Load config
    config = load_config(config_path)
    
    # Set seed
    set_seed(config.train.seed)
    
    # Setup output directory (only on rank 0)
    if rank == 0:
        run_id = f"{config.train.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_save_path = os.path.join(config.outputs.dir, run_id)
        os.makedirs(run_save_path, exist_ok=True)
        
        # Set up logging
        logger_manager.set_log_directory(run_save_path)
        log_file = os.path.join(run_save_path, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")
        
        # Save config
        save_dict(asdict(config), run_save_path, "config.json")
    else:
        run_save_path = None
    
    # Setup WandB (only rank 0)
    wandb_run = setup_wandb(config, rank, world_size) if rank == 0 else None
    
    # Create model
    model = create_large_model(config)
    
    # Setup data loaders
    train_loader, valid_loader, test_ds = setup_data_loaders(config, rank, world_size)
    
    # Create FSDP config
    trainer_config = config.train.main.trainer
    fsdp_config = get_fsdp_config(
        mixed_precision=getattr(trainer_config, 'use_mixed_precision', True),
        cpu_offload=getattr(trainer_config, 'cpu_offload', False),
        sharding_strategy="FULL_SHARD"
    )
    
    # Create optimizer config
    optimizer_config = {
        "lr": config.train.main.param_groups.lr,
        "weight_decay": config.train.main.weight_decay_rate,
        "split_params": getattr(config.train.main.param_groups, 'split_params', False),
    }
    
    # Setup analysis manager (only on rank 0)
    analysis_manager = None
    if rank == 0:
        analysis_manager = AnalysisManager(
            model=model,
            data={"train": train_ds, "valid": valid_ds, "test": test_ds},
            analysis_config=config.analysis,
            save_root=run_save_path,
            device=f"cuda:{local_rank}",
            train_config=config.train,
        )
    
    # Create FSDP trainer
    trainer = FSDPTrainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        checkpoint_dir=run_save_path if rank == 0 else None,
        save_every_n_epochs=getattr(trainer_config, 'save_every_n_epochs', None),
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        wandb_run=wandb_run,
        epochs=trainer_config.epochs,
        loss_function=getattr(trainer_config, 'loss_function', 'cat_nll'),
        grad_clip_value=getattr(trainer_config, 'grad_clip_value', 5.0),
        use_amp=getattr(trainer_config, 'use_amp', False),
        early_stopping=getattr(trainer_config, 'early_stopping', False),
        patience=getattr(trainer_config, 'patience', 10),
        print_every=getattr(trainer_config, 'print_every', 1),
        save_checkpoints=True,
    )
    
    if rank == 0:
        logger.info("Starting FSDP training...")
    
    # Train the model
    results = trainer.train(model, train_loader, valid_loader)
    
    if rank == 0:
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {results.get('final_train_loss', 'N/A')}")
        logger.info(f"Final validation loss: {results.get('final_valid_loss', 'N/A')}")
        
        # Save final results
        save_dict(results, run_save_path, "training_results.json")
        
        # Run final analysis
        if analysis_manager:
            try:
                analysis_manager.run_analysis(filename="final", training=False)
                logger.info("Final analysis completed")
            except Exception as e:
                logger.error(f"Final analysis failed: {e}")
    
    # Clean up distributed
    if wandb_run:
        wandb_run.finish()
    
    dist.destroy_process_group()
    
    if rank == 0:
        logger.info("FSDP training script completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP training for large dendritic models")
    parser.add_argument("config", help="Path to the configuration YAML file")
    args = parser.parse_args()
    
    main(args.config)
