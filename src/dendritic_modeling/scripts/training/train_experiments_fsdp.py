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

import argparse
import logging
import os
from dataclasses import asdict
from datetime import datetime

os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_TIMEOUT", "1800")  # 30 minutes timeout
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault(
    "NCCL_SOCKET_FAMILY", "AF_INET"
)  # Force IPv4 to fix socket errors
os.environ.setdefault("NCCL_SOCKET_IFNAME", "ib0,eth0")
os.environ.setdefault("NCCL_IB_DISABLE", "0")  # Enable InfiniBand if available
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "2")  # Enable GPU Direct RDMA

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.training.train_encoder_network import (
    load_train_encoder_network,
)
from dendritic_modeling.training.fsdp_utils import get_fsdp_config
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer
from dendritic_modeling.utils.general import save_dict, set_seed
from dendritic_modeling.utils.logging_config import LoggerManager

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()

# WandB import (optional)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")

# torchinfo import (optional)
try:
    import torchinfo

    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    logger.warning("torchinfo not available. Model summary will be skipped.")


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
            "model_type": config.model.core.type,
            "dataset": config.data.dataset_name,
            "batch_size": config.training.main.common.batch_size,
            "total_batch_size": config.training.main.common.batch_size * world_size,
            "epochs": config.training.main.common.epochs,
            "learning_rate": config.training.main.common.param_groups.lr,
            "weight_decay": config.training.main.common.weight_decay_rate,
            "world_size": world_size,
            # Model architecture details
            "excitatory_layers": config.model.core.architecture.excitatory_layer_sizes,
            "inhibitory_layers": config.model.core.architecture.inhibitory_layer_sizes,
            "excitatory_branches": config.model.core.architecture.excitatory_branch_factors,
            "inhibitory_branches": config.model.core.architecture.inhibitory_branch_factors,
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


def print_model_parameter_summary(model, sample_input=None):
    """Print detailed model parameter summary using torchinfo if available."""
    logger.info("Network architecture summary before training:")
    try:
        if TORCHINFO_AVAILABLE and sample_input is not None:
            torchinfo.summary(
                model,
                input_size=(100, *sample_input.shape),
                col_names=["input_size", "output_size", "num_params"],
                col_width=25,
                row_settings=["depth"],
                depth=7,
            )
            logger.info("Model summary generated (see console output)")
        else:
            if not TORCHINFO_AVAILABLE:
                logger.warning(
                    "torchinfo is not available. Model summary not generated."
                )
            else:
                logger.warning("No sample input provided. Using basic parameter count.")

            # Basic parameter breakdown
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")
        # Fallback to basic count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")


def create_large_model(config):
    """Create a large dendritic model from configuration."""
    model_config = config.model

    # Get input dimension from data config
    data_params = config.data.processing if hasattr(config.data, "processing") else {}
    if config.data.dataset_name == "mnist":
        input_dim = 784  # 28x28 flattened
    elif config.data.dataset_name == "cifar10":
        input_dim = 3072  # 32x32x3 flattened
    elif config.data.dataset_name == "imagenet":
        # Check if images are flattened
        if getattr(data_params, "flatten", False):
            input_dim = 150528  # 224x224x3 flattened
        else:
            # For non-flattened, the encoder should handle the shape
            input_dim = 150528
    else:
        input_dim = 784

    # Create encoder
    encoder_config = model_config.encoder
    if encoder_config.type.lower() == "identity":
        encoder = Identity(input_dim)
    else:
        encoder_params = dict(encoder_config.params) if encoder_config.params else {}
        encoder_params["input_dim"] = input_dim
        encoder_network = get_architecture(encoder_config.type, encoder_params)
        encoder = (
            encoder_network.encoder
            if hasattr(encoder_network, "encoder")
            else encoder_network
        )

    # Create core network
    core_config = model_config.core
    if hasattr(encoder, "output_dim"):
        core_network = get_architecture(
            core_config.type, core_config, input_dim=encoder.output_dim
        )
    else:
        core_network = get_architecture(
            core_config.type, core_config, input_dim=input_dim
        )

    # Create decoder
    decoder_config = model_config.decoder
    if decoder_config.type.lower() == "identity":
        decoder = Identity(
            core_network.output_dim
            if hasattr(core_network, "output_dim")
            else core_network.excitatory_layer_sizes[-1]
        )
    else:
        decoder_params = dict(decoder_config.params) if decoder_config.params else {}
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

    # Log model info with detailed breakdown
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
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )

    valid_sampler = DistributedSampler(
        valid_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Create data loaders
    batch_size = config.training.main.common.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, valid_loader, test_ds, train_ds, valid_ds


def main(config_path):
    """Main FSDP training function."""
    # Setup distributed environment
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        logger.info(f"Starting FSDP training on {world_size} GPUs")

    # Load config
    config = load_config(config_path)

    # Set seed
    set_seed(config.experiment.seed)

    # Setup output directory (only on rank 0)
    if rank == 0:
        run_id = f"{config.outputs.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    train_loader, valid_loader, test_ds, train_ds, valid_ds = setup_data_loaders(
        config, rank, world_size
    )

    # Only rank 0 needs to handle encoding for analysis.
    # Extract the encoder that create_large_model() already built into the model
    # so that load_train_encoder_network() can use it (e.g., for pre-training
    # BaseAutoencoder encoders).  For Identity and plain encoders (including
    # PretrainedEncoder) it returns immediately with datasets unchanged.
    if rank == 0:
        logger.info("Encoding datasets for analysis...")
        encoder_network = model.encoder_network

        encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds = (
            load_train_encoder_network(
                encoder_network=encoder_network,
                encoder_train_config=config.training.encoder,
                train_ds=train_ds,
                valid_ds=valid_ds,
                test_ds=test_ds,
                encoder_network_config=config.model.encoder,
                task_config=config.data,
            )
        )
    else:
        encoded_train_ds = encoded_valid_ds = encoded_test_ds = None
        encoder_network = Identity(1)

    # Generate detailed parameter summary with real sample input
    if rank == 0:  # Only on rank 0 to avoid duplicate output
        try:
            sample_input = train_ds[0][0]  # Get sample input from dataset
            print_model_parameter_summary(model, sample_input)
        except Exception as e:
            logger.warning(f"Could not generate model summary with sample input: {e}")
            print_model_parameter_summary(model)  # Fallback without sample input

    # Create FSDP config from config file or defaults
    trainer_config = getattr(config.training.main, "trainer", None)

    # Use FSDP config from config file if available
    if hasattr(config, "fsdp") and config.fsdp.use_fsdp:
        logger.info("Using FSDP configuration from config file")
        fsdp_config = get_fsdp_config(
            mixed_precision=config.fsdp.mixed_precision,
            cpu_offload=config.fsdp.cpu_offload,
            sharding_strategy=config.fsdp.sharding_strategy,
        )
    else:
        # Fall back to default method
        logger.info("Using default FSDP configuration")
        fsdp_config = get_fsdp_config(
            mixed_precision=(
                getattr(trainer_config, "use_mixed_precision", False)
                if trainer_config
                else False
            ),
            cpu_offload=(
                getattr(trainer_config, "cpu_offload", False)
                if trainer_config
                else False
            ),
            sharding_strategy="FULL_SHARD",
        )

    # Create optimizer config
    optimizer_config = {
        "lr": config.training.main.common.param_groups.lr,
        "weight_decay": config.training.main.common.weight_decay_rate,
        "split_params": getattr(
            config.training.main.common.param_groups, "split_params", False
        ),
    }

    # Setup analysis manager (only on rank 0)
    analysis_manager = None
    if rank == 0:
        logger.info("Setting up AnalysisManager...")

        # Set the model's encoder network
        if isinstance(encoder_network, Identity):
            model.encoder_network = encoder_network
            logger.info("Set model encoder to Identity network")
        elif hasattr(encoder_network, "encoder"):
            model.encoder_network = Identity(encoder_network.encoder.output_dim)
            logger.info(
                f"Set model encoder to Identity with output_dim: {encoder_network.encoder.output_dim}"
            )
        else:
            model.encoder_network = encoder_network
            logger.info("Set model encoder to %s", type(encoder_network).__name__)

        encoded_data = {
            "train": encoded_train_ds,
            "valid": encoded_valid_ds,
            "test": encoded_test_ds,
        }
        analysis_manager = AnalysisManager(
            model=model,
            data=encoded_data,
            analysis_config=config.analysis,
            save_root=run_save_path,
            device=f"cuda:{local_rank}",
            train_config=config.training,
        )
        logger.info(f"AnalysisManager created: {type(analysis_manager)}")
        logger.info(
            f"Performance analysis enabled: {config.analysis.performance_analysis.enabled}"
        )
        logger.info(
            f"Performance analysis training: {config.analysis.performance_analysis.training}"
        )

    # Create FSDP trainer
    trainer = FSDPTrainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        checkpoint_dir=run_save_path if rank == 0 else None,
        save_every_n_epochs=getattr(trainer_config, "save_every_n_epochs", None),
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        wandb_run=wandb_run,
        epochs=trainer_config.epochs,
        loss_function=getattr(trainer_config, "loss_function", "cat_nll"),
        grad_clip_value=getattr(trainer_config, "grad_clip_value", 5.0),
        use_amp=getattr(config.train, "enable_amp", False),
        early_stopping=getattr(trainer_config, "early_stopping", False),
        patience=getattr(trainer_config, "patience", 10),
        print_every=getattr(trainer_config, "print_every", 1),
        save_checkpoints=True,
        # Add plotting functionality similar to regular trainer
        plot_losses=getattr(trainer_config, "plot_losses", False),
        save_path=run_save_path if rank == 0 else None,
    )

    if rank == 0:
        logger.info("Starting FSDP training...")
        logger.info(
            f"Plot losses enabled: {getattr(trainer_config, 'plot_losses', False)}"
        )
        logger.info(f"Save path for plots: {run_save_path}")

    # Train the model
    results = trainer.train(model, train_loader, valid_loader)

    # Fix for Identity encoder
    if rank == 0:
        if isinstance(encoder_network, Identity):
            model.encoder_network = encoder_network
        elif hasattr(encoder_network, "encoder"):
            model.encoder_network = Identity(encoder_network.encoder.output_dim)
        else:
            model.encoder_network = encoder_network

    if rank == 0:
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {results.get('final_train_loss', 'N/A')}")
        logger.info(f"Final validation loss: {results.get('final_valid_loss', 'N/A')}")

        # Save final results
        save_dict(results, run_save_path, "training_results.json")

        # Run final analysis - this will automatically handle performance analysis and figure generation
        if analysis_manager:
            try:
                logger.info(
                    "Running final analysis (includes performance metrics plotting)..."
                )

                # Use unwrapped model for analysis if FSDP-wrapped
                if hasattr(model, "module"):
                    unwrapped_model = model.module
                    # Ensure the unwrapped model is on the correct device
                    unwrapped_model = unwrapped_model.to(f"cuda:{local_rank}")
                    analysis_manager.model = unwrapped_model

                analysis_manager.run_analysis(filename="final", training=False)
                logger.info(
                    "Final analysis completed - figures saved to performance folder"
                )
            except Exception as e:
                logger.error(f"Final analysis failed: {e}")

    # Clean up distributed
    if wandb_run:
        wandb_run.finish()

    dist.destroy_process_group()

    if rank == 0:
        logger.info("FSDP training script completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FSDP training for large dendritic models"
    )
    parser.add_argument("config", help="Path to the configuration YAML file")
    args = parser.parse_args()

    main(args.config)
