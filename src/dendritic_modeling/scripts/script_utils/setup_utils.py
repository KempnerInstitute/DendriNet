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

    # Create core network
    core_network = get_architecture(
        core_config.type, core_config, input_dim=encoder.output_dim
    )

    # Build decoder parameters
    decoder_params_dict = OmegaConf.to_container(decoder_config.params, resolve=True)
    # Filter out None values to allow auto-setting
    decoder_params_dict = {
        k: v for k, v in decoder_params_dict.items() if v is not None
    }
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

    # Check if this is part of a sweep (has _sweep_config_id)
    # If so, output_dir should already point to the correct location
    # Otherwise, create a timestamped run directory
    if hasattr(outputs_config, "_sweep_config_id") or (
        hasattr(outputs_config, "run_name")
        and "/config_" in str(outputs_config.results_dir)
    ):
        # Running as part of a sweep - use output_dir directly
        run_save_path = output_dir
        logger.info(f"Sweep mode detected - using directory: {run_save_path}")
    else:
        # Standalone run - create timestamped directory
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
            "train_valid_split": experiment_config.train_valid_split,
            "parameters": {
                **data_config.processing.__dict__,
                **dataset_specific_params,
            },
        },
    )()

    unified_datasets = get_unified_datasets(
        task_cfg=task_cfg,
    )
    train_ds, valid_ds, test_ds = unified_datasets

    logger.info(
        f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples"
    )

    # Update encoder input dimension (only if params exists)
    input_sample: torch.Tensor = train_ds[0][0]
    if (
        hasattr(model_config.encoder, "params")
        and model_config.encoder.params is not None
    ):
        model_config.encoder.params.input_dim = input_sample.numel()
        model_config.encoder.params.input_shape = list(input_sample.shape)
    else:
        # Create params if it doesn't exist (e.g., for identity encoder)
        model_config.encoder.params = {
            "input_dim": input_sample.numel(),
            "input_shape": list(input_sample.shape),
        }

    # Initialize model
    model, encoder_network = initialize_model(model_config)

    # Apply LDA initialization if dbl_init_method is "lda_analytical"
    # LDA initialization requires data, so it must happen after model construction
    morphology_config = getattr(model_config.core, "morphology", None)
    dbl_init_method = (
        getattr(morphology_config, "dbl_init_method", None)
        if morphology_config
        else None
    )

    if dbl_init_method == "lda_analytical":
        lda_init_config = getattr(model_config.core, "lda_init", None)
        n_samples = (
            getattr(lda_init_config, "n_samples", 1000) if lda_init_config else 1000
        )
        adaptive = (
            getattr(lda_init_config, "adaptive", True) if lda_init_config else True
        )

        logger.info(
            "Applying LDA-based weight initialization (dbl_init_method='lda_analytical')..."
        )
        model = apply_lda_initialization(
            model=model,
            train_ds=train_ds,
            encoder_network=encoder_network,
            n_samples=n_samples,
            adaptive=adaptive,
        )
        logger.info("LDA initialization completed")

    return run_save_path, train_ds, valid_ds, test_ds, model, encoder_network


def apply_lda_initialization(
    model,
    train_ds: torch.utils.data.Dataset,
    encoder_network=None,
    n_samples: int = 1000,
    adaptive: bool = True,
    device: str = "cpu",
):
    """Apply LDA-based initialization to all dendritic branch layers.

    This function uses labeled training data to initialize weights with
    LDA-optimal directions that maximize class separability.

    Args:
        model: The model to initialize
        train_ds: Training dataset with (input, label) pairs
        encoder_network: Optional encoder to preprocess inputs
        n_samples: Number of samples to use for LDA fitting
        adaptive: Use network-aware scaling
        device: Device to run computations on

    Returns:
        Model with LDA-initialized weights
    """
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
        initialize_model_with_lda,
    )

    # Collect samples from training data
    n_samples = min(n_samples, len(train_ds))
    indices = torch.randperm(len(train_ds))[:n_samples]

    inputs = []
    labels = []
    for idx in indices:
        x, y = train_ds[idx]
        inputs.append(x)
        labels.append(y)

    X = torch.stack(inputs)
    y = (
        torch.tensor(labels)
        if not isinstance(labels[0], torch.Tensor)
        else torch.stack(labels)
    )

    # Encode inputs if encoder is available
    if encoder_network is not None:
        encoder_network.eval()
        with torch.no_grad():
            X = X.to(device)
            encoder_network = encoder_network.to(device)
            X = encoder_network(X)

    # Apply LDA initialization
    model = initialize_model_with_lda(
        model=model,
        X=X,
        y=y,
        device=device,
        n_samples=n_samples,
        adaptive=adaptive,
    )

    return model
