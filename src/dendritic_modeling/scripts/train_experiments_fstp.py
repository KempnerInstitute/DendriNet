#!/usr/bin/env python
"""
Large Model FSDP Training Script
================================

This script enables FSDP (Fully Sharded Data Parallel) training for large dendritic models
that don't fit on a single GPU.

Usage:
    torchrun --nproc_per_node=<num_gpus> train_experiments_fstp.py <config_path>

Example:
    torchrun --nproc_per_node=4 train_experiments_fstp.py configs/large_model_config.yaml
"""

import os
import argparse

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

from dendritic_modeling import logger

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
from dendritic_modeling.utils.data import set_seed
from dendritic_modeling.utils.gpu_estimation import GPUMemoryEstimator
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.training.fsdp_utils import (
    get_fsdp_config,
)
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer
from dendritic_modeling.plotting.visualizations.analysis import plot_information_analysis


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
        name=getattr(wandb_config, "name", None) or config.outputs.run_name,
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

    return rank, world_size


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


def main(config_path):
    config = load_config(config_path)
    from .check_ei_config_detailed import check_ei_config
    if not check_ei_config(config):
        raise ValueError('EI config validation failed')
    # FSDP setup
    dist.init_process_group(backend='nccl')
    # Model setup with FSDP wrap
    model = create_large_model(config)
    # Training loop with distributed
    # ...
    # Run analysis similarly
    analysis_manager = AnalysisManager(...)
    run_analysis(...)
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
