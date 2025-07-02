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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations warnings

import argparse
import logging
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Union

import torch
try:
    import torchinfo
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    torchinfo = None
from omegaconf import OmegaConf

import wandb
from dendritic_modeling import logger
from dendritic_modeling import logger_manager
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config import (
    load_config, 
    Config,
    ModelConfig, 
    TrainConfig,
    VisualizationConfig,
    OutputsConfig,
    WandbConfig,
    TaskConfig
)
from dendritic_modeling.training import (
    CustomWeightDecayOptimizer,
    get_trainer,
    extract_weights_from_simple_mlp
)
from dendritic_modeling.models import (
    BaseModel,
    Classifier,
    Regressor
)
from dendritic_modeling.networks import (
    Identity,
    BaseNetwork
)
from dendritic_modeling.networks.architectures.classical.autoencoder import BaseAutoencoder
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.utils import (
    save_dict,
    set_seed
)
from dendritic_modeling.analysis.managers import PlotManager
from dendritic_modeling.networks.architectures.factory import get_architecture


def initialize_model(model_config: ModelConfig):
    encoder_network_config = model_config.encoder_network
    core_network_config = model_config.core_network
    decoder_network_config = model_config.decoder_network

    encoder_network = get_architecture(
        encoder_network_config.type,
        OmegaConf.to_container(encoder_network_config.parameters, resolve=True)
    )
    if isinstance(encoder_network, Identity):
        encoder = encoder_network
    else:
        encoder = encoder_network.encoder
    
    if isinstance(encoder.output_dim, tuple):
        encoder_output_dim = encoder.output_dim[0]
    else:
        encoder_output_dim = encoder.output_dim

    core_network_config.parameters.input_dim = encoder_output_dim
    core_network = get_architecture(
        core_network_config.type,
        OmegaConf.to_container(core_network_config.parameters, resolve=True)
    )

    decoder_network_config.parameters.input_dim = core_network.output_dim
    decoder_network = get_architecture(
        decoder_network_config.type,
        OmegaConf.to_container(decoder_network_config.parameters, resolve=True)
    )

    if model_config.task == "classification":
        model = Classifier(
            encoder,
            core_network,
            decoder_network
        )
    elif model_config.task == "regression":
        model = Regressor(
            encoder,
            core_network,
            decoder_network
        )
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
    model_config: ModelConfig,
    train_config: TrainConfig,
    task_config: TaskConfig,
    wandb_config: WandbConfig,
    outputs_config: OutputsConfig
):
    run_name = outputs_config.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"

    run_save_path = os.path.join(outputs_config.dir, run_id)
    os.makedirs(run_save_path, exist_ok=True)
    logger.info(f"Created output directory: {run_save_path}")
    
    # Set the logger manager to use the same directory for the main log file
    logger_manager.set_log_directory(run_save_path)

    log_file = os.path.join(run_save_path, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    seed = train_config.seed
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    if hasattr(wandb_config, "use_wandb") and wandb_config.use_wandb:
        try:
            wandb_config = {
                "model": asdict(model_config),
                "train": asdict(train_config),
                "task": asdict(task_config),
            }
            tags = wandb_config.tags if hasattr(wandb_config, "tags") else None
            wandb.init(
                project=wandb_config.project
                if hasattr(wandb_config, "project")
                else "dendritic-modeling",
                entity=wandb_config.entity if hasattr(wandb_config, "entity") else None,
                group=wandb_config.group if hasattr(wandb_config, "group") else None,
                name=run_id,
                config=wandb_config,
                tags=tags,
            )
            logger.info("Successfully initialized W&B")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B logging")
            wandb_config.use_wandb = False

    logger.info(f"Loading dataset: {task_config.dataset}")
    if not hasattr(task_config, "dataset") or not task_config.dataset:
        raise ValueError("No dataset specified in configuration")

    train_ds, valid_ds, test_ds = get_unified_datasets(
        task_cfg=task_config,
        train_cfg=train_config,
        output_dir=outputs_config.dir,
        outputs_cfg=outputs_config
    )
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


def train_encoder_network(
    encoder_network: Union[BaseNetwork, BaseAutoencoder],
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    run_save_path: str,
    train_config: TrainConfig
) -> tuple[
    BaseNetwork, 
    torch.utils.data.Dataset, 
    torch.utils.data.Dataset, 
    torch.utils.data.Dataset
]:
    if isinstance(encoder_network, Identity):
        return encoder_network, train_ds, valid_ds, test_ds
    else:
        encoder_net_save_path = os.path.join(run_save_path, "_encoder_network")
        os.makedirs(encoder_net_save_path, exist_ok=True)

        encoder_net_trainer_config = train_config.encoder_net_trainer
        encoder_net_trainer_config.save_path = encoder_net_save_path

        encoder_net_trainer_config_dict = OmegaConf.to_container(
            encoder_net_trainer_config, 
            resolve=True
        )

        train_input, train_label = train_ds[:]
        valid_input, valid_label = valid_ds[:]
        test_input, test_label = test_ds[:]

        train_input: torch.Tensor
        valid_input: torch.Tensor
        test_input: torch.Tensor
        train_label: torch.Tensor
        valid_label: torch.Tensor
        test_label: torch.Tensor

        auto_train_ds = torch.utils.data.TensorDataset(
            train_input, train_input
        )
        auto_valid_ds = torch.utils.data.TensorDataset(
            valid_input, valid_input
        )
        
        optimizer = torch.optim.Adam(
            encoder_network.parameters(),
            lr=train_config.optim_dict.get("encoder_net_lr", 0.001)
        )
        
        encoder_net_trainer = get_trainer(
            strategy="standard",
            optimizer=optimizer,
            trainer_config_dict=encoder_net_trainer_config_dict,
            analysis_manager=None
        )
        encoder_net_trainer.train(
            model=encoder_network,
            train_data=auto_train_ds,
            valid_data=auto_valid_ds
        )
        encoder_network = encoder_network.cpu()

        with torch.no_grad():
            encoded_train_input = encoder_network.encoder(train_input)
            encoded_valid_input = encoder_network.encoder(valid_input)
            encoded_test_input = encoder_network.encoder(test_input)

        encoded_train_ds = torch.utils.data.TensorDataset(
            encoded_train_input, train_label
        )
        encoded_valid_ds = torch.utils.data.TensorDataset(
            encoded_valid_input, valid_label
        )
        encoded_test_ds = torch.utils.data.TensorDataset(
            encoded_test_input, test_label
        )

        for param in encoder_network.parameters():
            param.requires_grad = False

        return (
            encoder_network.encoder, 
            encoded_train_ds, 
            encoded_valid_ds, 
            encoded_test_ds
        )


def run_analysis(
    model: BaseModel,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    visualization_cfg: VisualizationConfig,
    run_save_path: str,
    analysis_manager: Optional[AnalysisManager] = None,
):
    data = {"train": train_ds, "valid": valid_ds, "test": test_ds}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_visualization_enabled = (
        visualization_cfg
        and hasattr(visualization_cfg, "enabled")
        and visualization_cfg.enabled
    )
    if is_visualization_enabled:
        logger.info("Visualization is enabled. Setting up PlotManager...")

        # Create a class to hold args in the format expected by PlotManager
        class Args:
            def __init__(self, viz_cfg, save_path):
                self.visualization = viz_cfg
                self.save_path = save_path
                self.file_name = "final"
                self.shape = None
                self.train = None

        # Create args and data dictionary for PlotManager
        args = Args(visualization_cfg, run_save_path)

        # Create PlotManager instance
        plot_manager = PlotManager(model, data, args, device=device)

        # Get the list of plots to generate
        plots = visualization_cfg.plots if hasattr(visualization_cfg, "plots") else []
        logger.info(f"Generating the following plots: {plots}")

        # Run each requested plot
        for plot_name in plots:
            try:
                logger.info(f"Generating plot: {plot_name}")
                plot_manager.run_plot(plot_name)
            except Exception as e:
                logger.error(f"Error generating plot '{plot_name}': {e}")
                logger.error(traceback.format_exc())
    else:
        logger.info("Visualization is disabled. Skipping plot generation.")

    logger.info("Running analyses...")
    if analysis_manager is not None:
        analysis_manager.run_analysis()


def main(config_path: str, args=None):
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    if args:
        if args.learning_strategy:
            config.train.learning_strategy = args.learning_strategy
            logger.info(f"Override learning strategy: {args.learning_strategy}")
        if args.output_dir:
            config.outputs.dir = args.output_dir
            logger.info(f"Override output directory: {args.output_dir}")
        if args.run_name:
            config.train.run_name = args.run_name
            logger.info(f"Override run name: {args.run_name}")

    model_config = config.model
    train_config = config.train
    task_config = config.task
    wandb_config = config.wandb
    visualization_config = config.visualization
    analysis_config = config.analysis
    outputs_config = config.outputs

    (
        run_save_path,
        train_ds,
        valid_ds,
        test_ds,
        model,
        encoder_network
    ) = setup_environment(
        model_config=model_config,
        train_config=train_config,
        task_config=task_config,
        wandb_config=wandb_config,
        outputs_config=outputs_config
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


    (
        encoder, 
        encoded_train_ds, 
        encoded_valid_ds, 
        encoded_test_ds
    ) = train_encoder_network(
        encoder_network=encoder_network,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        run_save_path=run_save_path,
        train_config=train_config
    )

    encoded_data = {
        "train": encoded_train_ds,
        "valid": encoded_valid_ds,
        "test": encoded_test_ds
    }
    analysis_manager = AnalysisManager(
        model=model,
        data=encoded_data,
        analysis_config=analysis_config,
        save_root=run_save_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_config=train_config
    )

    model.encoder_network = Identity(encoder.output_dim)

    main_net_save_path = os.path.join(run_save_path, "_main_network")
    os.makedirs(main_net_save_path, exist_ok=True)

    main_trainer_config = train_config.main_trainer
    main_trainer_config.save_path = main_net_save_path
    main_trainer_config_dict = OmegaConf.to_container(
        main_trainer_config, resolve=True
    )

    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(
            model.get_param_groups(train_config.optim_dict)
        ),
        weight_decay=train_config.weight_decay_rate
    )

    main_trainer = get_trainer(
        strategy=train_config.learning_strategy,
        optimizer=optimizer,
        trainer_config_dict=main_trainer_config_dict,
        analysis_manager=analysis_manager
    )
    main_trainer.train(
        model=model,
        train_data=encoded_train_ds,
        valid_data=encoded_valid_ds
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
        visualization_cfg=visualization_config,
        run_save_path=run_save_path,
        analysis_manager=analysis_manager
    )

    model.encoder_network = encoder
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
