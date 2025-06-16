#!/usr/bin/env python
"""
train_experiments.py
====================
Main training script for dendritic modeling experiments.

This script handles:
- Configuration loading and setup
- Model initialization and training
- Input network pretraining (for dendritic models)
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
from typing import Optional

import numpy as np
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
    VisualizationConfig
)
from dendritic_modeling.training import (
    CustomWeightDecayOptimizer,
    get_trainer,
    extract_weights_from_simple_mlp
)
from dendritic_modeling.models import (
    BaseModel,
    Classifier,
    DeepstRegressor,
    ProbabilisticClassifier,
    Regressor,
    SimpleMLP
)
from dendritic_modeling.networks import (
    ExcitationInhibitionNetwork,
    FeatureExtractorMLP,
    MatchedParamMLP,
    SigmoidInputNetTransform
)
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.utils import (
    save_dict,
    set_seed
)
from dendritic_modeling.analysis.managers import PlotManager


def initialize_model(model_cfg: ModelConfig, visualization_enabled=False):
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters
    # Branch outputs are now automatically enabled when needed for analysis
    enable_branch_outputs = visualization_enabled
    learning_strategy = getattr(net_params, "learning_strategy", "mle")

    if enable_branch_outputs:
        logger.info("EINet branch outputs enabled for visualization/analysis")

    if task == "classification":
        if net_type == "MLP":
            logger.info(
                f"Running feedforward with {'probabilistic' if probabilistic else 'simple'} classifier"
            )
            net_params_dict = OmegaConf.to_container(net_params, resolve=True)
            net = MatchedParamMLP(**net_params_dict)
            output_dim = net_params_dict.get("output_dim", 10)
            if probabilistic:
                model = ProbabilisticClassifier(net, output_dim)
            else:
                model = Classifier(net)
            # Commenting out to preserve gradient scaling hooks
            # safe_register_hooks(model)
            return model
        elif net_type == "EINet":
            logger.info(
                f"Running EINet with {'probabilistic' if probabilistic else 'simple'} classifier"
            )
            net_params_dict = OmegaConf.to_container(net_params, resolve=True)
            net_params_dict["enable_branch_outputs"] = enable_branch_outputs
            net_params_dict["learning_strategy"] = learning_strategy
            net = ExcitationInhibitionNetwork(**net_params_dict)
            # Commenting out to preserve gradient scaling hooks
            # safe_register_hooks(net)
            output_dim = net_params_dict.get("output_dim", 10)
            if probabilistic:
                model = ProbabilisticClassifier(net, output_dim)
            else:
                model = Classifier(net)
            # Commenting out to preserve gradient scaling hooks
            # safe_register_hooks(model)
            return model
        else:
            raise ValueError(f"Invalid network type: {net_type}")
    elif task == "regression":
        if net_type == "MLP":
            logger.info(f"Running feedforward with regressor")
            net_params_dict = OmegaConf.to_container(net_params, resolve=True)
            net = MatchedParamMLP(**net_params_dict)
            output_dim = net_params_dict.get("output_dim", 10)
            model = Regressor(net, output_dim)
            # Commenting out to preserve gradient scaling hooks
            # safe_register_hooks(model)
            return model
        elif net_type == "EINet":
            logger.info(f"Running EINet with regressor")
            net_params_dict = OmegaConf.to_container(net_params, resolve=True)
            net_params_dict["enable_branch_outputs"] = enable_branch_outputs
            net_params_dict["learning_strategy"] = learning_strategy
            net = ExcitationInhibitionNetwork(**net_params_dict)
            # Commenting out to preserve gradient scaling hooks
            # safe_register_hooks(net)
            output_dim = net_params_dict.get("output_dim", 10)
            rewire_frequency = net_params_dict.get("rewire_frequency", 1)
            model = DeepstRegressor(net, output_dim, rewire_frequency=rewire_frequency)
            # Commenting out to preserve gradient scaling hooks
            # safe_register_hooks(model)
            return model
        else:
            raise ValueError(f"Invalid network type: {net_type}")
    else:
        raise ValueError(f"Invalid task: {task}")


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


def setup_training_environment(config: Config):
    model_cfg = config.model
    train_cfg = config.train
    trainer_cfg = train_cfg.trainer
    task_cfg = config.task
    wandb_cfg = config.wandb
    outputs_cfg = config.outputs
    visualization_cfg = config.visualization
    analysis_cfg = config.analysis

    run_name = train_cfg.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"

    run_save_path = os.path.join(outputs_cfg.dir, run_id)
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

    seed = train_cfg.seed
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    if hasattr(wandb_cfg, "use_wandb") and wandb_cfg.use_wandb:
        try:
            wandb_config = {
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "task": asdict(task_cfg),
            }
            tags = wandb_cfg.tags if hasattr(wandb_cfg, "tags") else None
            wandb.init(
                project=wandb_cfg.project
                if hasattr(wandb_cfg, "project")
                else "dendritic-modeling",
                entity=wandb_cfg.entity if hasattr(wandb_cfg, "entity") else None,
                group=wandb_cfg.group if hasattr(wandb_cfg, "group") else None,
                name=run_id,
                config=wandb_config,
                tags=tags,
            )
            logger.info("Successfully initialized W&B")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B logging")
            wandb_cfg.use_wandb = False

    logger.info(f"Loading dataset: {task_cfg.dataset}")
    if not hasattr(task_cfg, "dataset") or not task_cfg.dataset:
        raise ValueError("No dataset specified in configuration")

    train_ds, valid_ds, test_ds = get_unified_datasets(
        task_cfg=task_cfg,
        train_cfg=train_cfg,
        output_dir=outputs_cfg.dir,
        outputs_cfg=outputs_cfg
    )
    train_ds: torch.utils.data.Dataset
    valid_ds: torch.utils.data.Dataset
    test_ds: torch.utils.data.Dataset

    logger.info(
        f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples"
    )

    if hasattr(train_ds, "data") and len(train_ds.data) > 0:
        input_dim = np.prod(train_ds.data[0].shape)
    elif hasattr(train_ds, "tensors") and len(train_ds.tensors) > 0:
        input_dim = train_ds.tensors[0].shape[1]
    elif hasattr(train_ds, "__getitem__"):
        x, _ = train_ds[0]
        if isinstance(x, torch.Tensor):
            input_dim = x.numel()
        else:
            input_dim = np.prod(x.shape)
    else:
        logger.warning(
            "Could not automatically determine input dimension. Using default 784."
        )
        input_dim = 784

    try:
        model_cfg.network.parameters.input_dim = input_dim
    except Exception:
        model_cfg.network.parameters["input_dim"] = input_dim
    logger.info(f"Set input dimension to {input_dim}")

    visualization_enabled = (
        visualization_cfg.enabled if hasattr(visualization_cfg, "enabled") else False
    )
    model = initialize_model(model_cfg, visualization_enabled)
    logger.info(f"Initialized model: {type(model).__name__}")

    analysis_manager = AnalysisManager(
        model=model,
        data={"train": train_ds, "valid": valid_ds, "test": test_ds},
        analysis_config=analysis_cfg,
        save_root=run_save_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        train_config=asdict(train_cfg),
    )

    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.get_param_groups(train_cfg.optim_dict)),
        weight_decay=train_cfg.weight_decay_rate,
    )

    learning_strategy = train_cfg.learning_strategy.lower()
    trainer_cfg.save_path = run_save_path
    trainer_cfg_dict = OmegaConf.to_container(trainer_cfg, resolve=True)
    trainer = get_trainer(
        strategy=learning_strategy,
        optimizer=optimizer,
        trainer_config_dict=trainer_cfg_dict,
        analysis_manager=analysis_manager
    )
    logger.info(f"Created trainer with learning strategy: {learning_strategy}")

    return run_save_path, model, train_ds, valid_ds, test_ds, trainer, analysis_manager


def train_sequential_input_net(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    run_save_path: str,
    analysis_manager: Optional[AnalysisManager] = None,
):
    logger.info("Starting sequential input network training")
    if hasattr(train_ds, "data"):
        input_dim = np.prod(train_ds.data[0].shape)
    else:
        input_dim = train_ds[:][0].shape[-1]
    input_net_dict = getattr(model_cfg.network.parameters, "input_net_dict", {})
    hidden_dims = input_net_dict.get("hidden_dims", [128, 64])
    output_dim = input_net_dict.get("output_dim", 64)
    input_net_training = getattr(train_cfg, "input_net_training", {})
    train_input_net = input_net_training.get("train_input_net", True)
    input_net_epochs = input_net_training.get("input_net_epochs", 20)
    input_net_patience = input_net_training.get("input_net_patience", 10)
    if not train_input_net:
        logger.info("Skipping input network training (train_input_net=False)")
        return initialize_model(model_cfg, visualization_enabled=False)
    train_heads_independently = input_net_training.get(
        "train_heads_independently", False
    )
    if train_heads_independently:
        logger.info("Training excitatory and inhibitory heads independently")
        return train_independent_input_nets(
            model_cfg,
            train_cfg,
            train_ds,
            valid_ds,
            test_ds,
            input_dim,
            hidden_dims,
            output_dim,
            input_net_epochs,
            run_save_path,
            analysis_manager=analysis_manager,
        )
    else:
        logger.info("Training a single input network for both heads")
        return train_single_input_net(
            model_cfg,
            train_cfg,
            train_ds,
            valid_ds,
            test_ds,
            input_dim,
            hidden_dims,
            output_dim,
            input_net_epochs,
            input_net_patience,
            run_save_path,
            analysis_manager=analysis_manager,
        )


def train_independent_input_nets(
    model_cfg,
    train_cfg,
    train_ds,
    valid_ds,
    test_ds,
    input_dim,
    hidden_dims,
    output_dim,
    input_net_epochs,
    run_save_path,
    analysis_manager: Optional[AnalysisManager] = None,
):
    final_output_dim = 10
    excit_head_dir = os.path.join(run_save_path, "excit_head_model")
    os.makedirs(excit_head_dir, exist_ok=True)
    logger.info("Training excitatory head")
    excit_feature_extractor = SimpleMLP(input_dim, hidden_dims, output_dim)
    excit_feature_extractor, excit_metrics = train_mlp_with_classification_head(
        excit_feature_extractor,
        num_features=output_dim,
        num_classes=final_output_dim,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        epochs=input_net_epochs,
        batch_size=train_cfg.batch_size,
        lr=train_cfg.optim_dict.get("input_net_lr", 0.001),
        run_save_path=excit_head_dir,
    )
    save_model(excit_feature_extractor, run_save_path, "excit_feature_extractor.pt")
    logger.info(
        f"Excitatory head metrics: train={excit_metrics[0]:.4f}, valid={excit_metrics[1]:.4f}, test={excit_metrics[2]:.4f}"
    )
    if wandb.run is not None:
        wandb.log(
            {
                "excit_head_train_acc": excit_metrics[0],
                "excit_head_valid_acc": excit_metrics[1],
                "excit_head_test_acc": excit_metrics[2],
            }
        )
    inhib_head_dir = os.path.join(run_save_path, "inhib_head_model")
    os.makedirs(inhib_head_dir, exist_ok=True)
    logger.info("Training inhibitory head")
    inhib_feature_extractor = SimpleMLP(input_dim, hidden_dims, output_dim)
    inhib_feature_extractor, inhib_metrics = train_mlp_with_classification_head(
        inhib_feature_extractor,
        num_features=output_dim,
        num_classes=final_output_dim,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        epochs=input_net_epochs,
        batch_size=train_cfg.batch_size,
        lr=train_cfg.optim_dict.get("input_net_lr", 0.001),
        run_save_path=inhib_head_dir,
    )
    save_model(inhib_feature_extractor, run_save_path, "inhib_feature_extractor.pt")
    logger.info(
        f"Inhibitory head metrics: train={inhib_metrics[0]:.4f}, valid={inhib_metrics[1]:.4f}, test={inhib_metrics[2]:.4f}"
    )
    if wandb.run is not None:
        wandb.log(
            {
                "inhib_head_train_acc": inhib_metrics[0],
                "inhib_head_valid_acc": inhib_metrics[1],
                "inhib_head_test_acc": inhib_metrics[2],
            }
        )
    model = initialize_model(model_cfg, visualization_enabled=False)
    analysis_manager.model = model
    logger.info("Creating input network with sigmoid activation")
    new_input_net = SigmoidInputNetTransform(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        network_type=model_cfg.network.parameters.input_net_dict.get(
            "type", "MLPInputTransform"
        ),
    )
    logger.info("Transferring weights to input network")
    extract_weights_from_simple_mlp(excit_feature_extractor, new_input_net)
    if hasattr(model, "net") and hasattr(model.net, "input_net"):
        model.net.input_net = new_input_net
    elif hasattr(model, "network") and hasattr(model.network, "input_net"):
        model.network.input_net = new_input_net
    else:
        logger.warning(
            "Could not set input network in model - model structure may be incompatible"
        )
    save_model(new_input_net, run_save_path, "input_net_with_sigmoid.pt")
    logger.info("Training full model with pretrained input networks")
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.get_param_groups(train_cfg.optim_dict)),
        weight_decay=train_cfg.weight_decay_rate,
    )
    trainer = get_trainer(train_cfg.learning_strategy, optimizer)
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_clip_value=train_cfg.grad_clip_value,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager=analysis_manager,
        plot_losses=True,
        save_path=run_save_path,
    )
    save_model(model, run_save_path, "final_model.pt")
    return model


def train_single_input_net(
    model_cfg,
    train_cfg,
    train_ds,
    valid_ds,
    test_ds,
    input_dim,
    hidden_dims,
    output_dim,
    input_net_epochs,
    input_net_patience,
    run_save_path,
    analysis_manager: Optional[AnalysisManager] = None,
):
    final_output_dim = 10
    mlp_model_dir = os.path.join(run_save_path, "mlp_model")
    os.makedirs(mlp_model_dir, exist_ok=True)
    logger.info("Training MLP feature extractor")
    feature_extractor = FeatureExtractorMLP(
        input_dim=input_dim, hidden_dims=hidden_dims, output_dim=final_output_dim
    )
    clf = ProbabilisticClassifier(
        net=feature_extractor, output_dim=final_output_dim, use_output_layer=False
    )
    trainer = TrainerMLE(
        optimizer=torch.optim.Adam(
            clf.parameters(), lr=train_cfg.optim_dict.get("input_net_lr", 0.001)
        )
    )
    trainer.train(
        model=clf,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=input_net_epochs,
        batch_size=train_cfg.batch_size,
        early_stopping=True,
        patience=input_net_patience,
        plot_losses=True,
        save_path=mlp_model_dir,
    )

    # feature_extractor = SimpleMLP(input_dim, hidden_dims, output_dim)
    # feature_extractor, metrics = train_mlp_with_classification_head(
    #     feature_extractor,
    #     num_features=output_dim,
    #     num_classes=final_output_dim,
    #     train_ds=train_ds,
    #     valid_ds=valid_ds,
    #     test_ds=test_ds,
    #     epochs=input_net_epochs,
    #     batch_size=train_cfg.batch_size,
    #     lr=train_cfg.optim_dict.get('input_net_lr', 0.001),
    #     run_save_path=mlp_model_dir,
    # )
    save_model(clf.net, run_save_path, "feature_extractor.pt")
    # logger.info(
    #     f"Feature extractor metrics: train={metrics[0]:.4f}, valid={metrics[1]:.4f}, test={metrics[2]:.4f}"
    # )
    # if wandb.run is not None:
    #     wandb.log(
    #         {
    #             "feature_extractor_train_acc": metrics[0],
    #             "feature_extractor_valid_acc": metrics[1],
    #             "feature_extractor_test_acc": metrics[2],
    #         }
    #     )
    model = initialize_model(model_cfg, visualization_enabled=False)
    analysis_manager.model = model
    logger.info("Creating input network with sigmoid activation")
    # new_input_net = SigmoidInputNetTransform(
    #     input_dim=input_dim,
    #     hidden_dims=hidden_dims,
    #     output_dim=output_dim,
    #     network_type=model_cfg.network.parameters.input_net_dict.get(
    #         "type", "MLPInputTransform"
    #     ),
    # )
    logger.info("Transferring weights to input network")
    if hasattr(model, "net"):
        if hasattr(model.net, "input_net"):
            if hasattr(model.net.input_net, "input_transform"):
                model.net.input_net.input_transform.load_state_dict(
                    clf.net.state_dict()
                )
                model.net.input_net.input_transform.set_readout(False)
                model.net.input_net.input_transform.disable_grad()
    # # extract_weights_from_simple_mlp(feature_extractor, new_input_net)
    # if hasattr(model, "net") and hasattr(model.net, "input_net"):
    #     model.net.input_net = new_input_net
    # elif hasattr(model, "network") and hasattr(model.network, "input_net"):
    #     model.network.input_net = new_input_net
    else:
        logger.warning(
            "Could not set input network in model - model structure may be incompatible"
        )
    # save_model(new_input_net, run_save_path, "input_net_with_sigmoid.pt")
    logger.info("Training full model with pretrained input network")
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.get_param_groups(train_cfg.optim_dict)),
        weight_decay=train_cfg.weight_decay_rate,
    )
    trainer = get_trainer(train_cfg.learning_strategy, optimizer)
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_clip_value=train_cfg.grad_clip_value,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager=analysis_manager,
        plot_losses=True,
        save_path=run_save_path,
    )
    save_model(model, run_save_path, "final_model.pt")
    return model


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


def train_and_evaluate_main(config_path: str, args=None):
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

    model_cfg = config.model
    train_cfg = config.train
    trainer_cfg = train_cfg.trainer
    wandb_cfg = config.wandb
    visualization_cfg = config.visualization
    train_cfg.task = model_cfg.task

    (
        run_save_path,
        model,
        train_ds,
        valid_ds,
        test_ds,
        trainer,
        analysis_manager
    ) = setup_training_environment(config)

    logger.info("Network architecture summary before training:")
    try:
        sample_input, _ = train_ds[0]
        sample_input_dim = sample_input.numel()
        if TORCHINFO_AVAILABLE:
            summary = torchinfo.summary(
                model,
                input_size=(trainer_cfg.batch_size, sample_input_dim),
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

    input_mode = getattr(model_cfg.network.parameters, "input_mode", 0)
    input_net_training = getattr(train_cfg, "input_net_training", {})
    training_mode = input_net_training.get("training_mode", "joint")
    sequential_training = input_mode == 2 and training_mode == "sequential"

    if sequential_training:
        logger.info("Using sequential training mode")
        model = train_sequential_input_net(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            run_save_path=run_save_path,
            analysis_manager=analysis_manager
        )
    else:
        logger.info("Using standard training mode")
        trainer.train(model, train_ds, valid_ds)

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
        visualization_cfg=visualization_cfg,
        run_save_path=run_save_path,
        analysis_manager=analysis_manager,
    )

    if (
        hasattr(wandb_cfg, "use_wandb")
        and wandb_cfg.use_wandb
        and wandb.run is not None
    ):
        wandb.finish()

    logger.info("Training experiment completed successfully")


def main(config_path, args=None):
    train_and_evaluate_main(config_path, args=args)


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
