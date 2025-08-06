"""
Setup utilities for dendritic modeling training experiments.

This module contains helper functions for setting up the training environment,
including model initialization, dataset loading, and logging configuration.
"""

import logging
import os
from dataclasses import asdict
from datetime import datetime

import torch
import wandb
from omegaconf import OmegaConf

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.config import ModelConfig
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.utils import set_seed


def initialize_model(model_config: ModelConfig):
    """Initialize the model based on configuration."""
    encoder_network_config = model_config.encoder_network
    core_network_config = model_config.core_network
    decoder_network_config = model_config.decoder_network

    encoder_network = get_architecture(
        encoder_network_config.type,
        OmegaConf.to_container(encoder_network_config.parameters, resolve=True),
    )
    if isinstance(encoder_network, Identity):
        encoder = encoder_network
    else:
        encoder = encoder_network.encoder

    core_network_config.parameters.input_dim = encoder.output_dim
    core_network = get_architecture(
        core_network_config.type,
        OmegaConf.to_container(core_network_config.parameters, resolve=True),
    )

    decoder_network_config.parameters.input_dim = core_network.output_dim
    decoder_network = get_architecture(
        decoder_network_config.type,
        OmegaConf.to_container(decoder_network_config.parameters, resolve=True),
    )

    if model_config.task == "classification":
        model = Classifier(encoder, core_network, decoder_network)
    elif model_config.task == "regression":
        model = Regressor(encoder, core_network, decoder_network)
    else:
        raise ValueError(f"Invalid task: {model_config.task}")

    logger.info(f"Initialized model: {type(model).__name__}")

    return model, encoder_network


def setup_environment(
    model_config,
    train_config,
    task_config,
    wandb_config,
    outputs_config,
    run_config=None,
):
    """
    Set up the environment for training.

    Returns:
        Tuple of (run_save_path, train_ds, valid_ds, test_ds, model, encoder_network)
    """
    # Set random seed
    set_seed(train_config.seed)

    # Create run directory using run_config.dir if available, fallback to outputs_config.dir
    output_dir = (
        run_config.dir
        if run_config and hasattr(run_config, "dir")
        else outputs_config.dir
    )

    # Check if we're in a sweep context by looking for sweep-specific directory patterns
    # Sweep directories typically end with config_X where X is a number
    is_sweep_context = output_dir and (
        # Check if path ends with config_X pattern (sweep context)
        (
            str(output_dir).split("/")[-1].startswith("config_")
            and str(output_dir).split("/")[-1].replace("config_", "").isdigit()
        )
        or
        # Or if "sweep" or "unified_" is in the path
        "sweep" in str(output_dir).lower()
        or "unified_" in str(output_dir)
    )

    if is_sweep_context:
        # We're in a sweep - use the provided directory directly
        run_save_path = output_dir
        logger.info(f"Using sweep output directory directly: {run_save_path}")
    else:
        # Standard single experiment - create timestamped directory
        run_id = f"{train_config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_save_path = os.path.join(output_dir, run_id)
        logger.info(f"Created timestamped experiment directory: {run_save_path}")

    # Set the logger manager to use the same directory for the main log file
    logger_manager.set_log_directory(run_save_path)

    log_file = os.path.join(run_save_path, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    if hasattr(wandb_config, "use_wandb") and wandb_config.use_wandb:
        try:
            wandb.init(
                entity=wandb_config.entity,
                project=wandb_config.project,
                group=wandb_config.group,
                tags=wandb_config.tags,
                name=train_config.run_name,
                config=asdict(train_config),
            )
            logger.info("Successfully initialized W&B")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B logging")
            wandb_config.use_wandb = False

    # Get datasets using a shared dataset directory (not the individual run directory)
    # This prevents each config from saving its own copy of the dataset
    if hasattr(outputs_config, "dataset_dir") and outputs_config.dataset_dir:
        # Use explicitly configured dataset directory
        shared_dataset_dir = outputs_config.dataset_dir
    elif hasattr(outputs_config, "dir") and outputs_config.dir:
        # Use outputs directory as fallback
        shared_dataset_dir = outputs_config.dir
    else:
        # Use parent of output_dir as final fallback
        shared_dataset_dir = os.path.dirname(output_dir)

    unified_datasets = get_unified_datasets(
        task_cfg=task_config,
        train_cfg=train_config,
        output_dir=shared_dataset_dir,  # Use shared directory for datasets
        outputs_cfg=None,  # No longer using outputs_cfg
    )
    train_ds, valid_ds, test_ds = unified_datasets

    logger.info(
        f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples"
    )

    input_sample: torch.Tensor = train_ds[0][0]
    model_config.encoder_network.parameters.input_dim = input_sample.numel()
    model_config.encoder_network.parameters.input_shape = tuple(input_sample.shape)

    model, encoder_network = initialize_model(model_config)

    return run_save_path, train_ds, valid_ds, test_ds, model, encoder_network


def save_model(model: torch.nn.Module, save_path: str, filename: str):
    """Save a model to the specified path."""
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    torch.save(model.state_dict(), full_path)
    logger.info(f"Saved model to {full_path}")
    return full_path


def load_model(model: torch.nn.Module, load_path: str):
    """Load a model from the specified path."""
    model.load_state_dict(torch.load(load_path))
    logger.info(f"Loaded model from {load_path}")
    return model
