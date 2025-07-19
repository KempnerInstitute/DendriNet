#!/usr/bin/env python
"""
train_experiments.py
====================
Main training script for dendritic modeling experiments.

This script handles:
- Configuration loading and setup
- Model initialization and training
- Encoder network pretraining (for dendritic models)
- Visualization and analysis
- Results saving and logging

Usage:
    python train_experiments.py <config_path> [options]

Options:
    --device: Device to use (cuda or cpu)
    --wandb: Enable W&B logging
    --save_model: Save the trained model
    --save_results: Save training results
    --run_analysis: Run post-training analysis
"""

from __future__ import annotations

# Suppress TensorFlow warnings for cleaner output
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "2"  # Suppress TensorFlow INFO and WARNING messages
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations warnings

import argparse
import logging
from dataclasses import asdict
from datetime import datetime

import torch

try:
    import torchinfo

    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    torchinfo = None

import wandb
from omegaconf import OmegaConf

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config import (
    ModelConfig,
    load_config,
)
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import BaseModel, Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.train_encoder_network import load_train_encoder_network
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.utils import save_dict, validate_config, set_seed
# from .check_ei_config_detailed import estimate_ei_model_detailed  # Not currently used
from dendritic_modeling.plotting.analysis import plot_information_analysis


def initialize_model(model_config: ModelConfig):
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


def save_model(model: torch.nn.Module, save_path: str, filename: str):
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    torch.save(model.state_dict(), full_path)
    logger.info(f"Saved model to {full_path}")
    return full_path


def load_model(model: torch.nn.Module, load_path: str):
    model.load_state_dict(torch.load(load_path))
    logger.info(f"Loaded model from {load_path}")
    return model


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
    """
    # Set random seed
    set_seed(train_config.seed)

    # Create run directory using run_config.dir if available, fallback to outputs_config.dir
    output_dir = (
        run_config.dir
        if run_config and hasattr(run_config, "dir")
        else outputs_config.dir
    )
    run_id = f"{train_config.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_save_path = os.path.join(output_dir, run_id)

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

    # Get datasets using the correct output directory
    unified_datasets = get_unified_datasets(
        task_cfg=task_config,
        train_cfg=train_config,
        output_dir=output_dir,
        outputs_cfg=None,  # No longer using outputs_cfg
    )
    train_ds, valid_ds, test_ds = unified_datasets

    train_ds: torch.utils.data.Dataset
    valid_ds: torch.utils.data.Dataset
    test_ds: torch.utils.data.Dataset

    logger.info(
        f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples"
    )

    input_sample: torch.Tensor = train_ds[0][0]
    model_config.encoder_network.parameters.input_dim = input_sample.numel()
    model_config.encoder_network.parameters.input_shape = tuple(input_sample.shape)

    model, encoder_network = initialize_model(model_config)

    return run_save_path, train_ds, valid_ds, test_ds, model, encoder_network


def run_analysis(
    model: BaseModel,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    run_save_path: str,
    analysis_manager: AnalysisManager | None = None,
    training: bool = False,
):
    """Run analysis using the AnalysisManager."""
    logger.info("Running analyses...")
    if analysis_manager is not None:
        results = analysis_manager.run_analysis()
        if not training:
            # Call plotting functions on results
            pass # Placeholder for plotting functions


def main(config_path: str, args=None):
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Validate configuration before proceeding
    logger.info("Validating configuration...")

    # Convert config to dict for validation
    config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config

    if not validate_config(config_dict):
        logger.error("Configuration validation failed. Please fix the errors above.")
        raise ValueError("Invalid configuration")

    logger.info("Configuration validation passed")

    if args:
        if args.learning_strategy:
            config.train.learning_strategy = args.learning_strategy
            logger.info(f"Override learning strategy: {args.learning_strategy}")
        if args.output_dir:
            # Override run_config.dir if it exists, otherwise create run_config
            if hasattr(config, "run_config") and config.run_config:
                config.run_config.dir = args.output_dir
            else:
                # For backwards compatibility, still update outputs.dir if run_config doesn't exist
                config.outputs.dir = args.output_dir
            logger.info(f"Override output directory: {args.output_dir}")
        if args.run_name:
            config.train.run_name = args.run_name
            logger.info(f"Override run name: {args.run_name}")

    model_config = config.model
    train_config = config.train
    task_config = config.task
    wandb_config = config.wandb
    analysis_config = config.analysis
    outputs_config = config.outputs
    run_config = getattr(config, "run_config", None)

    # EI config validation removed - let model initialization handle validation

    (run_save_path, train_ds, valid_ds, test_ds, model, encoder_network) = (
        setup_environment(
            model_config=model_config,
            train_config=train_config,
            task_config=task_config,
            wandb_config=wandb_config,
            outputs_config=outputs_config,
            run_config=run_config,
        )
    )

    logger.info("Network architecture summary before training:")
    try:
        sample_input: torch.Tensor = train_ds[0][0]
        if TORCHINFO_AVAILABLE:
            summary = torchinfo.summary(
                model,
                input_size=(100, *sample_input.shape),
                col_names=["input_size", "output_size", "num_params"],
                col_width=25,
                row_settings=["depth"],
                depth=7,
            )
            logger.info("Model summary generated (see console output)")
            if wandb.run:
                wandb.log(
                    {
                        "total_params": summary.total_params,
                        "trainable_params": summary.trainable_params,
                    }
                )
        else:
            logger.warning("torchinfo is not available. Model summary not generated.")
    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")

    save_dict(asdict(config), run_save_path, "config.json")

    (encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds) = (
        load_train_encoder_network(
            encoder_network=encoder_network,
            encoder_train_config=train_config.encoder,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            encoder_network_config=model_config.encoder_network,
            task_config=task_config,
        )
    )

    encoded_data = {
        "train": encoded_train_ds,
        "valid": encoded_valid_ds,
        "test": encoded_test_ds,
    }
    analysis_manager = AnalysisManager(
        model=model,
        data=encoded_data,
        analysis_config=analysis_config,
        save_root=run_save_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_config=train_config,
    )

    # Fix for Identity encoder
    if isinstance(encoder_network, Identity):
        model.encoder_network = encoder_network
    else:
        model.encoder_network = Identity(encoder_network.encoder.output_dim)

    main_net_save_path = os.path.join(run_save_path, "_main_network")
    os.makedirs(main_net_save_path, exist_ok=True)

    main_train_config = train_config.main
    main_train_config.trainer.save_path = main_net_save_path
    main_trainer_config_dict = OmegaConf.to_container(
        main_train_config.trainer, resolve=True
    )

    # Add checkpoint configuration if specified
    if (
        hasattr(main_train_config.trainer, "save_every_n_epochs")
        and main_train_config.trainer.save_every_n_epochs is not None
    ):
        main_trainer_config_dict["save_every_n_epochs"] = (
            main_train_config.trainer.save_every_n_epochs
        )
        main_trainer_config_dict["checkpoint_dir"] = run_save_path

    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(
            model.get_param_groups(main_train_config.param_groups)
        ),
        weight_decay=main_train_config.weight_decay_rate,
    )

    main_trainer = get_trainer(
        strategy=main_train_config.learning_strategy,
        optimizer=optimizer,
        trainer_config_dict=main_trainer_config_dict,
        analysis_manager=analysis_manager,
    )
    main_trainer.train(
        model=model, train_data=encoded_train_ds, valid_data=encoded_valid_ds
    )

    nparams = 0
    for param in model.parameters():
        nparams += param.numel()
    if wandb.run is not None:
        wandb.log({"nparams": nparams})
    save_dict({"nparams": nparams}, run_save_path, "nparams")

    run_analysis(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        run_save_path=run_save_path,
        analysis_manager=analysis_manager,
    )

    # Fix for Identity encoder
    if isinstance(encoder_network, Identity):
        model.encoder_network = encoder_network
    else:
        model.encoder_network = encoder_network.encoder
    save_model(model, run_save_path, "final_model.pt")

    if (
        hasattr(wandb_config, "use_wandb")
        and wandb_config.use_wandb
        and wandb.run is not None
    ):
        wandb.finish()

    logger.info("Training experiment completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dendritic modeling training and evaluation."
    )
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument(
        "--output_dir", help="Override output directory specified in config."
    )
    parser.add_argument("--run_name", help="Override run name specified in config.")
    parser.add_argument(
        "--learning_strategy",
        choices=[
            "mle",
            "freeze_branches",
            "freeze_branches_reversed",
            "freeze_branches_with_pretraining",
            "freeze_branches_reversed_with_pretraining",
            "two_step",
            "two_step_kl",
            "voltage_stabilization",
        ],
        help="Override learning strategy specified in config.",
    )
    args = parser.parse_args()
    main(config_path=args.config, args=args)
