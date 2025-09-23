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

from dendritic_modeling.config import DataConfig, ModelConfig, TrainingConfig
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.utils import set_seed
from dendritic_modeling.utils.logging_config import LoggerManager

logger = logging.getLogger(__name__)
logger_manager = LoggerManager()


def initialize_model(model_config: ModelConfig):
    """Initialize the model based on configuration."""
    encoder_config = model_config.encoder
    core_config = model_config.core
    decoder_config = model_config.decoder

    # Build encoder parameters dict
    encoder_params = {}
    if encoder_config.type.lower() == "identity":
        encoder_params["input_dim"] = encoder_config.params.input_dim or 784
    else:
        encoder_params = OmegaConf.to_container(encoder_config.params, resolve=True)

    # Create encoder
    encoder_network = get_architecture(encoder_config.type, encoder_params)
    if isinstance(encoder_network, Identity):
        encoder = encoder_network
    else:
        encoder = encoder_network.encoder

    # Build core network parameters dict by flattening the structured config
    architecture_dict = OmegaConf.to_container(core_config.architecture, resolve=True)
    connectivity_dict = OmegaConf.to_container(core_config.connectivity, resolve=True)
    transfer_dict = OmegaConf.to_container(core_config.transfer, resolve=True)
    morphology_dict = OmegaConf.to_container(core_config.morphology, resolve=True)
    implementation_dict = OmegaConf.to_container(
        core_config.implementation, resolve=True
    )

    core_params = {
        "input_dim": encoder.output_dim,
        **architecture_dict,
        **connectivity_dict,
        "transfer_params": transfer_dict,
        **morphology_dict,
        **implementation_dict,
    }

    # Add sparsity parameters
    sparsity_dict = OmegaConf.to_container(core_config.sparsity, resolve=True)
    for key, value in sparsity_dict.items():
        if key == "deepst":
            for deepst_key, deepst_value in value.items():
                core_params[deepst_key] = deepst_value
        else:
            core_params[f"topk_{key}"] = value

    # Add reactivation parameters
    reactivation_dict = OmegaConf.to_container(core_config.reactivation, resolve=True)
    core_params["reactivate"] = reactivation_dict.pop("enabled", True)
    for key, value in reactivation_dict.items():
        if key == "type":
            core_params["reactivation_type"] = value
        elif key.startswith("init_"):
            core_params[f"reactivation_{key}"] = value
        else:
            core_params[f"reactivation_{key}"] = value

    # Add blocklinear parameters
    blocklinear_dict = OmegaConf.to_container(core_config.blocklinear, resolve=True)
    for key, value in blocklinear_dict.items():
        if key == "efficient":
            core_params["efficient_blocklinear"] = value
        else:
            core_params[f"blocklinear_{key}"] = value

    # Create core network
    core_network = get_architecture(core_config.type, core_params)

    # Build decoder parameters
    decoder_params_dict = OmegaConf.to_container(decoder_config.params, resolve=True)
    decoder_params = {
        "input_dim": core_network.output_dim,
        **decoder_params_dict,
    }

    # Create decoder
    decoder_network = get_architecture(decoder_config.type, decoder_params)

    # Create model based on task
    if model_config.task == "classification":
        model = Classifier(encoder, core_network, decoder_network)
    elif model_config.task == "regression":
        model = Regressor(encoder, core_network, decoder_network)
    else:
        raise ValueError(f"Invalid task: {model_config.task}")

    logger.info(f"Initialized model: {type(model).__name__}")

    return model, encoder_network


def setup_environment(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
    wandb_config,
    outputs_config,
    experiment_config,
):
    """Setup training environment."""

    # Set seed
    seed = experiment_config.seed
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Setup output directory
    output_dir = outputs_config.results_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    run_name = outputs_config.run_name
    run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_save_path = os.path.join(output_dir, run_id)
    os.makedirs(run_save_path, exist_ok=True)
    logger.info(f"Created run directory: {run_save_path}")

    # Set the logger manager to use the same directory for the main log file
    logger_manager.set_log_directory(run_save_path)

    log_file = os.path.join(run_save_path, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # Setup wandb if enabled
    if wandb_config.use_wandb:
        try:
            wandb.init(
                entity=wandb_config.entity,
                project=wandb_config.project,
                group=wandb_config.group,
                tags=wandb_config.tags,
                name=run_name,
                config={
                    "model": asdict(model_config),
                    "training": asdict(training_config),
                    "data": asdict(data_config),
                    "experiment": asdict(experiment_config),
                },
            )
            logger.info("Successfully initialized W&B")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B logging")
            wandb_config.use_wandb = False

    base_dir = data_config.base_dir or output_dir
    dataset_specific_params = {}
    if hasattr(data_config.dataset_params, data_config.dataset_name):
        dataset_specific_params = getattr(
            data_config.dataset_params, data_config.dataset_name, {}
        )

    task_cfg = type(
        "TaskConfig",
        (),
        {
            "dataset": data_config.dataset_name,
            "data_path": os.path.join(base_dir, data_config.dataset_name)
            if base_dir
            else None,
            "parameters": {
                **data_config.processing.__dict__,
                **dataset_specific_params,
            },
        },
    )()

    train_cfg = type(
        "TrainConfig",
        (),
        {
            "seed": experiment_config.seed,
            "train_valid_split": experiment_config.train_valid_split,
        },
    )()

    unified_datasets = get_unified_datasets(
        task_cfg=task_cfg,
        train_cfg=train_cfg,
    )
    train_ds, valid_ds, test_ds = unified_datasets

    logger.info(
        f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples"
    )

    # Update encoder input dimension
    input_sample: torch.Tensor = train_ds[0][0]
    model_config.encoder.params.input_dim = input_sample.numel()
    model_config.encoder.params.input_shape = list(input_sample.shape)

    # Initialize model
    model, encoder_network = initialize_model(model_config)

    return run_save_path, train_ds, valid_ds, test_ds, model, encoder_network
