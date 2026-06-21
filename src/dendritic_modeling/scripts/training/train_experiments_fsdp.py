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
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    is_data_driven_reactivation_policy,
    normalize_reactivation_init_policy,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils import config_utils as _config_utils
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _config_to_plain_dict,
    _get_dataset_specific_params,
    initialize_model,
)
from dendritic_modeling.scripts.training.train_encoder_network import (
    load_train_encoder_network,
)
from dendritic_modeling.training.fsdp_utils import get_fsdp_config, save_fsdp_checkpoint
from dendritic_modeling.training.strategies.fsdp_standard import (
    FSDPTrainer,
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.utils.general import save_dict, set_seed
from dendritic_modeling.utils.logging_config import LoggerManager

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()
canonicalize_model_core_flags = getattr(
    _config_utils, "canonicalize_model_core_flags", None
)
canonicalize_rnn_core_flags = getattr(
    _config_utils, "canonicalize_rnn_core_flags", None
)

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


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed environment via ``torchrun``.

    ``torchrun`` sets ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``
    automatically.  When these are absent the function raises an error
    because FSDP requires a distributed process group.

    Returns:
        (rank, local_rank, world_size)
    """
    if "RANK" not in os.environ:
        raise RuntimeError(
            "FSDP requires torchrun.  Launch with:\n"
            "  torchrun --nproc_per_node=<N> train_experiments_fsdp.py <config>"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

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

    vision_replacement = getattr(model_config, "vision_replacement", None)
    pretrained_replacement = getattr(model_config, "pretrained_replacement", None)
    replacement_enabled = (
        vision_replacement is not None and getattr(vision_replacement, "enabled", False)
    ) or (
        pretrained_replacement is not None
        and getattr(pretrained_replacement, "enabled", False)
    )
    if replacement_enabled:
        model, _ = initialize_model(model_config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model created from vision/pretrained replacement with %s total parameters",
            f"{total_params:,}",
        )
        logger.info("Trainable parameters: %s", f"{trainable_params:,}")
        return model

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


def _get_analysis_encoder_network(encoder_network):
    """Return the encoder module that analysis paths should see."""
    if isinstance(encoder_network, Identity):
        return encoder_network
    if hasattr(encoder_network, "encoder"):
        return Identity(encoder_network.encoder.output_dim)
    return encoder_network


def _dataloader_kwargs_from_common(common_config) -> dict:
    """Build validated DataLoader kwargs from training.main.common."""
    available_cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except OSError:
            pass

    num_workers = getattr(common_config, "num_workers", None)
    if num_workers is None:
        num_workers = min(4, available_cpus)
    else:
        num_workers = max(0, int(num_workers))
        num_workers = min(num_workers, available_cpus)

    pin_memory_cfg = getattr(common_config, "pin_memory", None)
    pin_memory = (
        bool(pin_memory_cfg)
        if pin_memory_cfg is not None
        else torch.cuda.is_available() and num_workers > 0
    )

    persistent_cfg = getattr(common_config, "persistent_workers", None)
    if persistent_cfg is None:
        persistent_workers = num_workers > 0
    else:
        persistent_workers = bool(persistent_cfg)
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0")

    prefetch_factor = getattr(common_config, "prefetch_factor", None)
    if prefetch_factor is not None and num_workers == 0:
        raise ValueError("prefetch_factor requires num_workers > 0")

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return kwargs


def setup_data_loaders(config, rank, world_size):
    """Setup distributed data loaders."""
    data_config = config.data
    base_dir = data_config.base_dir or ""
    dataset_specific_params = _get_dataset_specific_params(data_config)
    processing_params = _config_to_plain_dict(getattr(data_config, "processing", None))

    task_cfg = type(
        "TaskConfig",
        (),
        {
            "dataset": data_config.dataset_name,
            "data_path": base_dir or None,
            "train_valid_split": config.experiment.train_valid_split,
            "parameters": {
                **processing_params,
                **dataset_specific_params,
                "label_noise_seed": config.experiment.seed,
                "split_seed": config.experiment.seed,
            },
        },
    )()

    train_ds, valid_ds, test_ds = get_unified_datasets(task_cfg=task_cfg)

    # Create distributed sampler for training only.
    # Validation uses a non-padded sampler: each rank gets a disjoint
    # shard *without* padding, and the trainer reduces exact counts
    # (loss_sum, correct, total_examples) via all-reduce SUM.
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(config.experiment.seed),
    )

    # PaddedDistributedEvalSampler gives each rank an equal number of
    # indices (required by FSDP all-gather) but marks the extras as
    # padding (-1 sentinels).  The trainer's _epoch_valid_fsdp excludes
    # padding from the reduction, so validation is numerically identical
    # to a non-distributed run over the full dataset.
    valid_sampler = PaddedDistributedEvalSampler(
        valid_ds,
        num_replicas=world_size,
        rank=rank,
    )

    # Create data loaders
    batch_size = config.training.main.common.batch_size
    loader_kwargs = _dataloader_kwargs_from_common(config.training.main.common)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        **loader_kwargs,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        sampler=valid_sampler,
        **loader_kwargs,
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
        run_save_path = os.path.join(config.outputs.results_dir, run_id)
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
        config_payload = asdict(config)
        if canonicalize_model_core_flags is not None:
            config_payload = canonicalize_model_core_flags(config_payload)
        if canonicalize_rnn_core_flags is not None:
            config_payload = canonicalize_rnn_core_flags(config_payload)
        save_dict(config_payload, run_save_path, "config.json")
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

    # Encode datasets for analysis (rank 0 only).
    # Note: FSDP does NOT support autoencoder pretraining.  If the
    # encoder is a BaseAutoencoder, create_large_model() already
    # collapsed it to its encoder sub-module, so
    # load_train_encoder_network() will return immediately.
    # To pretrain an autoencoder encoder, run it *before* FSDP
    # training with the standard single-GPU script, then point
    # the config at the saved checkpoint.
    if rank == 0:
        from dendritic_modeling.networks.architectures.classical.autoencoder import (
            BaseAutoencoder,
        )

        encoder_network = model.encoder_network
        if isinstance(encoder_network, BaseAutoencoder):
            logger.warning(
                "BaseAutoencoder pretraining is not supported inside the "
                "FSDP script.  Pretrain the encoder separately, then load "
                "its checkpoint.  Skipping encoder pretraining."
            )
            encoded_train_ds = train_ds
            encoded_valid_ds = valid_ds
            encoded_test_ds = test_ds
        else:
            logger.info("Encoding datasets for analysis...")
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

    # Training common config is the source of truth for epochs, loss, etc.
    train_common = config.training.main.common

    # Read FSDP tuning knobs from config.distributed.fsdp.
    fsdp_cfg = config.distributed.fsdp
    logger.info("Using FSDP configuration from config file")
    fsdp_config = get_fsdp_config(
        mixed_precision=fsdp_cfg.mixed_precision,
        cpu_offload=fsdp_cfg.cpu_offload,
        sharding_strategy=fsdp_cfg.sharding_strategy,
        fsdp_config_dict=vars(fsdp_cfg) if hasattr(fsdp_cfg, "__dict__") else None,
    )

    # Create optimizer config
    optimizer_config = {
        "lr": train_common.param_groups.lr,
        "weight_decay": getattr(train_common, "weight_decay_rate", 0.0),
        "split_params": getattr(train_common.param_groups, "split_params", False),
    }

    # Setup analysis manager (only on rank 0)
    analysis_manager = None
    if rank == 0:
        logger.info("Setting up AnalysisManager...")

        # Set the model's encoder network
        model.encoder_network = _get_analysis_encoder_network(encoder_network)
        logger.info("Set model encoder to %s", type(model.encoder_network).__name__)

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
        )
        logger.info(f"AnalysisManager created: {type(analysis_manager)}")
        logger.info(
            f"Performance analysis enabled: {config.analysis.performance_analysis.enabled}"
        )
        logger.info(
            f"Performance analysis training: {config.analysis.performance_analysis.training}"
        )

        # Parity with the single-GPU launcher: if the config requests
        # data-driven reactivation calibration, run it here before training
        # starts. Under FSDP the model is already sharded by this point;
        # calibration runs on rank 0's shard-view of the model and writes
        # back to the ParametricTanh parameters, which then get
        # synchronized across ranks via FSDP's standard parameter-sharing
        # on the first forward pass. We log a warning so users can spot
        # divergence if it occurs.
        from dendritic_modeling.scripts.training.train_experiments import (
            _maybe_calibrate_data_driven_reactivation,
        )

        requested_policies = set()
        try:
            reactivation_cfg = config.model.core.reactivation
            init_policy = normalize_reactivation_init_policy(
                getattr(reactivation_cfg, "init_policy", "analytical")
            )
            if is_data_driven_reactivation_policy(init_policy):
                requested_policies.add(init_policy)
        except AttributeError:
            logger.debug(
                "Reactivation config path missing (config.model.core.reactivation); "
                "skipping config-level data-driven init policy detection."
            )

        for module in model.modules():
            layer_policy = getattr(module, "reactivation_init_policy", None)
            if layer_policy is None:
                continue
            layer_policy = normalize_reactivation_init_policy(layer_policy)
            if is_data_driven_reactivation_policy(layer_policy):
                requested_policies.add(layer_policy)

        if requested_policies:
            logger.warning(
                "FSDP + init_policy=%r: running calibration on rank 0 only. "
                "Parameter writes may not propagate perfectly across ranks "
                "depending on sharding strategy. Verify with the "
                "ReactivationDynamicsAnalyzer if in doubt.",
                ", ".join(sorted(requested_policies)),
            )
            _maybe_calibrate_data_driven_reactivation(
                config=config,
                model=model,
                train_dataset=encoded_train_ds,
                is_main=True,
            )

    # Create FSDP trainer
    trainer = FSDPTrainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        checkpoint_dir=run_save_path if rank == 0 else None,
        save_every_n_epochs=getattr(train_common, "save_every_n_epochs", None),
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        wandb_run=wandb_run,
        epochs=train_common.epochs,
        loss_function=getattr(train_common, "loss_function", "cat_nll"),
        grad_clip_value=getattr(train_common, "grad_clip_value", 5.0),
        use_amp=getattr(train_common, "use_amp", False),
        early_stopping=getattr(train_common, "early_stopping", False),
        patience=getattr(train_common, "patience", 10),
        print_every=getattr(train_common, "print_every", 1),
        save_checkpoints=True,
        plot_losses=getattr(train_common, "plot_losses", False),
        save_path=run_save_path if rank == 0 else None,
        seed=int(config.experiment.seed),
    )

    if rank == 0:
        logger.info("Starting FSDP training...")
        logger.info(
            f"Plot losses enabled: {getattr(train_common, 'plot_losses', False)}"
        )
        logger.info(f"Save path for plots: {run_save_path}")

    # Train the model
    results = trainer.train(model, train_loader, valid_loader)

    if rank == 0:
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {results.get('final_train_loss', 'N/A')}")
        logger.info(f"Final validation loss: {results.get('final_valid_loss', 'N/A')}")

        # Save final results
        save_dict(results, run_save_path, "training_results.json")

    # --- Final analysis: gather full state dict, reload into plain model ---
    # FSDP shards parameters across ranks; analysis on the sharded model
    # produces incorrect results.  The safe pattern is:
    # 1. Save a full (unsharded) checkpoint from the FSDP model (all ranks
    #    participate in the all-gather, rank 0 saves).
    # 2. On rank 0, create a fresh non-FSDP model and load that checkpoint.
    # 3. Run analysis on the plain model.
    fsdp_model = trainer.fsdp_model
    full_ckpt_path = os.path.join(
        run_save_path or "/tmp", "checkpoints", "final_full.pt"
    )
    if rank == 0:
        os.makedirs(os.path.dirname(full_ckpt_path), exist_ok=True)

    # All ranks participate in the full state-dict gather.
    save_fsdp_checkpoint(
        fsdp_model,
        trainer.optimizer,
        trainer.epochs - 1,
        full_ckpt_path,
        rank=rank,
    )
    if rank == 0:
        logger.info(f"Full checkpoint saved to {full_ckpt_path}")

    if rank == 0 and analysis_manager:
        try:
            logger.info("Running final analysis on non-FSDP model...")
            # Create a fresh non-FSDP model and load the gathered state dict
            plain_model = create_large_model(config)
            ckpt = torch.load(full_ckpt_path, map_location=f"cuda:{local_rank}")
            plain_model.load_state_dict(ckpt["model_state_dict"])
            plain_model = plain_model.to(f"cuda:{local_rank}")
            plain_model.eval()

            # Set the encoder for analysis
            plain_model.encoder_network = _get_analysis_encoder_network(encoder_network)

            analysis_manager.model = plain_model
            analysis_manager.run_analysis(filename="final", training=False)
            logger.info(
                "Final analysis completed - figures saved to performance folder"
            )
        except Exception as e:
            logger.error(f"Final analysis failed: {e}")
            import traceback

            logger.error(traceback.format_exc())

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
