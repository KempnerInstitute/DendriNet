"""
train_experiments.py
--------------------
Main script that reads config_exp.yaml, builds the model,
chooses a training strategy, trains, and evaluates using
a unified data-loading function from synthetic_datasets.py.

Learning Strategies:
- mle: Standard Maximum Likelihood Estimation (default)
- freeze_layers: Train one layer at a time, freezing others
- two_step: Two-phase training with pretraining and main training
- mlp_only: Train only the MLP component, then train the full model
- dendrinet_from_mlp: Train MLP first, then use it to initialize and train the dendrinet

Example usage:
python scripts/train_experiments.py .vscode/config_exp.yaml --learning_strategy dendrinet_from_mlp
"""

import os
import argparse
import logging
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import time
from copy import deepcopy
import json
import yaml

import wandb
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchinfo

from dendritic_modeling import logger
from dendritic_modeling.learning_strategies import (
    get_trainer, CustomWeightDecayOptimizer, 
    train_simple_mlp, train_mlp_with_classification_head,
    transfer_mlp_weights_to_input_net, extract_weights_from_simple_mlp
)
from dendritic_modeling.models import (
    ProbabilisticClassifier, Classifier, 
    SimpleMLP, SimpleClassifier, MLPWithClassificationHead
)
from dendritic_modeling.networks import (
    ExcitationInhibitionNetwork, MLPExcInhNetwork, 
    InputNetTransform, SigmoidInputNetTransform
)
from dendritic_modeling.config import load_config
from dendritic_modeling.visualization_manager import PlotManager
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import (
    evaluate_accuracy, save_dict, 
    analyze_branch_ablation, analyze_layerwise_ablation,
    safe_register_hooks
)


def initialize_model(model_cfg, visualization_enabled=False):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.
    model_cfg is from config.model, with .task, .probabilistic, .network, etc.
    visualization_enabled is used to determine if branch outputs should be enabled.
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters.copy() if hasattr(model_cfg.network.parameters, 'copy') else deepcopy(model_cfg.network.parameters)
    
    # Only enable branch outputs if visualization is enabled or it's explicitly set to True in config
    config_branch_outputs = getattr(net_params, 'enable_branch_outputs', False)
    enable_branch_outputs = config_branch_outputs and visualization_enabled
    
    # Log the branch outputs setting
    if config_branch_outputs and not visualization_enabled:
        logger.info("Branch outputs disabled despite config setting because visualization is disabled")
    
    learning_strategy = getattr(net_params, 'learning_strategy', 'mle')

    if task == 'classification':
        if net_type == 'MLP':
            logger.info(f"Running feedforward with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = MLPExcInhNetwork(**asdict(net_params))
            output_dim = getattr(net_params, 'output_dim', 10)
            if probabilistic:
                model = ProbabilisticClassifier(net, output_dim)
            else:
                model = Classifier(net)
                
            # Apply safe hooks
            safe_register_hooks(model)
            return model
                
        elif net_type == 'EINet':
            logger.info(f"Running EINet with {'probabilistic' if probabilistic else 'simple'} classifier")
            logger.info(f"EINet branch outputs enabled: {enable_branch_outputs}")
            
            # Convert dataclass to dictionary and set enable_branch_outputs
            net_params_dict = asdict(net_params)
            net_params_dict['enable_branch_outputs'] = enable_branch_outputs
            net_params_dict['learning_strategy'] = learning_strategy
            
            net = ExcitationInhibitionNetwork(**net_params_dict)
            
            # Apply safe hooks to the network
            safe_register_hooks(net)

            output_dim = getattr(net_params, 'output_dim', 10)
            
            if probabilistic:
                model = ProbabilisticClassifier(net, output_dim)
            else:
                model = Classifier(net)
                
            # Apply safe hooks to the full model
            safe_register_hooks(model)
            return model
        else:
            raise ValueError(f"Invalid network type: {net_type}")
    else:
        raise ValueError(f"Invalid task: {task}")


def save_model(model, save_path, filename):
    """
    Save a model's state dictionary to disk.
    Uses the existing output directory structure.
    """
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    torch.save(model.state_dict(), full_path)
    logger.info(f"Saved model to {full_path}")
    return full_path


def load_model(model, load_path):
    """
    Load a model's state dictionary from disk.
    """
    model.load_state_dict(torch.load(load_path))
    logger.info(f"Loaded model from {load_path}")
    return model


def setup_training_environment(config, args=None):
    """
    Sets up the training environment based on configuration.
    
    Args:
        config: The loaded configuration object
        args: Optional command-line arguments that can override config
        
    Returns:
        A tuple of (run_save_path, model, train_ds, valid_ds, test_ds, optimizer, trainer)
    """
    # Extract configuration sections
    model_cfg = config.model
    train_cfg = config.train
    task_cfg = config.task
    wandb_cfg = config.wandb
    outputs_cfg = config.outputs
    visualization_cfg = config.visualization
    
    # Set up run ID and save path
    run_name = train_cfg.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"
    
    # Create output directory
    run_save_path = os.path.join(outputs_cfg.dir, run_id)
    os.makedirs(run_save_path, exist_ok=True)
    logger.info(f"Created output directory: {run_save_path}")
    
    # Set up file logging
    log_file = os.path.join(run_save_path, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")
    
    # Set random seed for reproducibility
    seed = train_cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Set random seed to {seed}")
    
    # Initialize W&B if enabled
    if hasattr(wandb_cfg, 'use_wandb') and wandb_cfg.use_wandb:
        try:
            wandb_config = {
                "model": asdict(model_cfg),
                "train": asdict(train_cfg),
                "task": asdict(task_cfg),
            }
            
            # Get tags if available
            tags = wandb_cfg.tags if hasattr(wandb_cfg, 'tags') else None
            
            wandb.init(
                project=wandb_cfg.project if hasattr(wandb_cfg, 'project') else "dendritic-modeling",
                entity=wandb_cfg.entity if hasattr(wandb_cfg, 'entity') else None,
                group=wandb_cfg.group if hasattr(wandb_cfg, 'group') else None,
                name=run_id,
                config=wandb_config,
                tags=tags,
            )
            logger.info("Successfully initialized W&B")
        except Exception as e:
            logger.error(f"Error initializing W&B: {e}")
            logger.error("Continuing without W&B logging")
            wandb_cfg.use_wandb = False
    
    # Load dataset
    logger.info(f"Loading dataset: {task_cfg.dataset}")
    
    # Verify dataset configuration
    if not hasattr(task_cfg, 'dataset') or not task_cfg.dataset:
        raise ValueError("No dataset specified in configuration")
    
    # Load datasets
    try:
        train_ds, valid_ds, test_ds = get_unified_datasets(task_cfg, train_cfg)
        logger.info(f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples")
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise
    
    # Determine input dimension from dataset
    if hasattr(train_ds, 'data'):
        input_dim = np.prod(train_ds.data[0].shape)
    else:
        input_dim = 784  # Default to MNIST dimensions
    
    # Update input dimension in model config
    model_cfg.network.parameters.input_dim = input_dim
    logger.info(f"Set input dimension to {input_dim}")
    
    # Get visualization setting
    visualization_enabled = visualization_cfg.enabled if hasattr(visualization_cfg, 'enabled') else False
    
    # Initialize model
    model = initialize_model(model_cfg, visualization_enabled)
    logger.info(f"Initialized model: {type(model).__name__}")
    
    # Create optimizer
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr),
        weight_decay=train_cfg.weight_decay_rate,
    )
    
    # Get learning strategy
    learning_strategy = train_cfg.learning_strategy.lower()
    
    # Create trainer
    trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)
    logger.info(f"Created trainer with learning strategy: {learning_strategy}")
    
    return run_save_path, model, train_ds, valid_ds, test_ds, optimizer, trainer


def train_and_evaluate(
    model, train_ds, valid_ds, test_ds, trainer, 
    train_cfg, visualization_cfg, run_save_path
):
    """
    Train the model and evaluate its performance.
    
    Args:
        model: The initialized model
        train_ds, valid_ds, test_ds: The datasets
        trainer: The training strategy
        train_cfg: Training configuration
        visualization_cfg: Visualization configuration
        run_save_path: Directory to save results
        
    Returns:
        The results dictionary from training
    """
    # Get learning strategy
    learning_strategy = train_cfg.learning_strategy.lower()
    
    # Train the model using the appropriate strategy
    if learning_strategy in ["two_step", "two_step_kl"]:
        logger.info(f"Training with {learning_strategy} strategy: {train_cfg.pretrain_epochs} pretrain epochs + {train_cfg.epochs - train_cfg.pretrain_epochs} main epochs")
        results = trainer.train(
            model=model,
            train_data=train_ds,
            valid_data=valid_ds,
            pretrain_epochs=train_cfg.pretrain_epochs,
            maintrain_epochs=train_cfg.epochs - train_cfg.pretrain_epochs,
            batch_size=train_cfg.batch_size,
            grad_clip_value=train_cfg.grad_clip_value,
            shuffle=True,
            load_best_state_dict=True,
            plot_losses=True,
            save_path=run_save_path
        )
    elif learning_strategy == "freeze_layers":
        logger.info(f"Training with freeze_layers strategy: {train_cfg.epochs_per_layer} epochs per layer")
        results = trainer.train_freeze_layers(
            model=model,
            train_data=train_ds,
            valid_data=valid_ds,
            epochs_per_layer=train_cfg.epochs_per_layer,
            batch_size=train_cfg.batch_size,
            grad_clip_value=train_cfg.grad_clip_value,
            shuffle=True,
            plot_losses=True,
            save_path=run_save_path
        )
    else:
        logger.info(f"Training with standard approach: {train_cfg.epochs} epochs")
        results = trainer.train(
            model=model,
            train_data=train_ds,
            valid_data=valid_ds,
            epochs=train_cfg.epochs,
            batch_size=train_cfg.batch_size,
            grad_clip_value=train_cfg.grad_clip_value,
            shuffle=True,
            load_best_state_dict=True,
            plot_losses=True,
            save_path=run_save_path
        )
    
    # Save the final model
    save_model(model, run_save_path, "final_model.pt")
    
    # Plot loss curves if we have valid loss history
    train_losses = results.get('train_losses', [])
    valid_losses = results.get('valid_losses', [])
    
    if len(train_losses) > 0:
        logger.info("Plotting loss curves")
        # Handle case where epochs don't match losses (can happen with freeze_layers)
        if len(train_losses) != train_cfg.epochs:
            logger.warning(f"Number of loss points ({len(train_losses)}) doesn't match epochs ({train_cfg.epochs})")
            plot_epochs = len(train_losses)
        else:
            plot_epochs = train_cfg.epochs
            
        plot_NLL_loss_curves(train_losses, valid_losses, plot_epochs, run_save_path)
    
    # Evaluate final model performance
    logger.info("Evaluating final model performance")
    train_acc, valid_acc, test_acc = evaluate_accuracy(
        model, train_ds, valid_ds, test_ds, 
        save_path=os.path.join(run_save_path, 'performance'), 
        filename="final",
    )
    logger.info(f"Final accuracy: train={train_acc:.4f}, valid={valid_acc:.4f}, test={test_acc:.4f}")
    
    # Log to W&B if initialized
    if wandb.run is not None:
        wandb.log({
            "final_train_acc": train_acc,
            "final_valid_acc": valid_acc,
            "final_test_acc": test_acc
        })
    
    # Create visualizations if enabled
    if hasattr(visualization_cfg, 'enabled') and visualization_cfg.enabled:
        logger.info("Creating visualizations")
        try:
            plot_manager = PlotManager(
                model=model,
                train_data=train_ds,
                valid_data=valid_ds,
                test_data=test_ds,
                save_path=os.path.join(run_save_path, 'visualizations'),
                plots=visualization_cfg.plots
            )
            plot_manager.create_visualizations()
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    return results


def train_sequential_input_net(model_cfg, train_cfg, task_cfg, train_ds, valid_ds, test_ds, run_save_path):
    """
    Train a model with sequential input network training.
    
    Args:
        model_cfg: Model configuration
        train_cfg: Training configuration
        task_cfg: Task configuration
        train_ds, valid_ds, test_ds: The datasets
        run_save_path: Directory to save results
        
    Returns:
        The trained model
    """
    logger.info("Starting sequential input network training")
    
    # Get input dimension from dataset
    if hasattr(train_ds, 'data'):
        input_dim = np.prod(train_ds.data[0].shape)
    else:
        input_dim = 784  # Default to MNIST dimensions
    
    # Get input network configuration
    input_net_dict = getattr(model_cfg.network.parameters, 'input_net_dict', {})
    hidden_dims = input_net_dict.get('hidden_dims', [128, 64])
    output_dim = input_net_dict.get('output_dim', 64)
    
    # Get input network training parameters
    input_net_training = getattr(train_cfg, 'input_net_training', {})
    train_input_net = input_net_training.get('train_input_net', True)
    input_net_epochs = input_net_training.get('input_net_epochs', 20)
    
    # Skip if we don't need to train the input network
    if not train_input_net:
        logger.info("Skipping input network training (train_input_net=False)")
        return initialize_model(model_cfg, visualization_enabled=False)
    
    # Get training strategy for independent heads
    train_heads_independently = input_net_training.get('train_heads_independently', False)
    
    if train_heads_independently:
        logger.info("Training excitatory and inhibitory heads independently")
        return train_independent_input_nets(
            model_cfg, train_cfg, train_ds, valid_ds, test_ds, 
            input_dim, hidden_dims, output_dim, input_net_epochs, run_save_path
        )
    else:
        logger.info("Training a single input network for both heads")
        return train_single_input_net(
            model_cfg, train_cfg, train_ds, valid_ds, test_ds,
            input_dim, hidden_dims, output_dim, input_net_epochs, run_save_path
        )


def train_independent_input_nets(
    model_cfg, train_cfg, train_ds, valid_ds, test_ds, 
    input_dim, hidden_dims, output_dim, input_net_epochs, run_save_path
):
    """
    Train independent input networks for excitatory and inhibitory paths.
    
    Args:
        model_cfg, train_cfg: Configurations
        train_ds, valid_ds, test_ds: Datasets
        input_dim, hidden_dims, output_dim: Network dimensions
        input_net_epochs: Number of epochs for input network training
        run_save_path: Directory to save results
        
    Returns:
        The final trained model
    """
    # For classification, final output dimension is number of classes
    final_output_dim = 10  # Default for MNIST
    
    # Create directories for model saving
    excit_head_dir = os.path.join(run_save_path, "excit_head_model")
    os.makedirs(excit_head_dir, exist_ok=True)
    
    # Train excitatory head
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
        lr=train_cfg.lr,
        run_save_path=excit_head_dir
    )
    
    # Save excitatory feature extractor
    save_model(excit_feature_extractor, run_save_path, "excit_feature_extractor.pt")
    logger.info(f"Excitatory head metrics: train={excit_metrics[0]:.4f}, valid={excit_metrics[1]:.4f}, test={excit_metrics[2]:.4f}")
    
    # Log to W&B if initialized
    if wandb.run is not None:
        wandb.log({
            "excit_head_train_acc": excit_metrics[0],
            "excit_head_valid_acc": excit_metrics[1],
            "excit_head_test_acc": excit_metrics[2]
        })
    
    # Train inhibitory head
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
        lr=train_cfg.lr,
        run_save_path=inhib_head_dir
    )
    
    # Save inhibitory feature extractor
    save_model(inhib_feature_extractor, run_save_path, "inhib_feature_extractor.pt")
    logger.info(f"Inhibitory head metrics: train={inhib_metrics[0]:.4f}, valid={inhib_metrics[1]:.4f}, test={inhib_metrics[2]:.4f}")
    
    # Log to W&B if initialized
    if wandb.run is not None:
        wandb.log({
            "inhib_head_train_acc": inhib_metrics[0],
            "inhib_head_valid_acc": inhib_metrics[1],
            "inhib_head_test_acc": inhib_metrics[2]
        })
    
    # Initialize the full model
    model = initialize_model(model_cfg, visualization_enabled=False)
    
    # Create a new input network with sigmoid activation
    logger.info("Creating input network with sigmoid activation")
    new_input_net = SigmoidInputNetTransform(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        network_type=model_cfg.network.parameters.input_net_dict.get("type", "MLPInputTransform")
    )
    
    # Transfer weights from trained feature extractors
    logger.info("Transferring weights to input network")
    extract_weights_from_simple_mlp(excit_feature_extractor, new_input_net)
    # No need to call extract_weights again for inhibitory head since it transfers both at once
    
    # Set the input network in the model
    if hasattr(model, 'net') and hasattr(model.net, 'input_net'):
        model.net.input_net = new_input_net
    elif hasattr(model, 'network') and hasattr(model.network, 'input_net'):
        model.network.input_net = new_input_net
    else:
        logger.warning("Could not set input network in model - model structure may be incompatible")
    
    # Save the input network
    save_model(new_input_net, run_save_path, "input_net_with_sigmoid.pt")
    
    # Train the full model
    logger.info("Training full model with pretrained input networks")
    
    # Create optimizer for full model
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr),
        weight_decay=train_cfg.weight_decay_rate,
    )
    
    # Create trainer
    trainer = get_trainer(train_cfg.learning_strategy, optimizer)
    
    # Train the model
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_clip_value=train_cfg.grad_clip_value,
        shuffle=True,
        load_best_state_dict=True,
        plot_losses=True,
        save_path=run_save_path
    )
    
    # Save the final model
    save_model(model, run_save_path, "full_model.pt")
    
    return model


def train_single_input_net(
    model_cfg, train_cfg, train_ds, valid_ds, test_ds,
    input_dim, hidden_dims, output_dim, input_net_epochs, run_save_path
):
    """
    Train a single input network for both excitatory and inhibitory paths.
    
    Args:
        model_cfg, train_cfg: Configurations
        train_ds, valid_ds, test_ds: Datasets
        input_dim, hidden_dims, output_dim: Network dimensions
        input_net_epochs: Number of epochs for input network training
        run_save_path: Directory to save results
        
    Returns:
        The final trained model
    """
    # For classification, final output dimension is number of classes
    final_output_dim = 10  # Default for MNIST
    
    # Create directory for model saving
    mlp_model_dir = os.path.join(run_save_path, "mlp_model")
    os.makedirs(mlp_model_dir, exist_ok=True)
    
    # Train MLP feature extractor
    logger.info("Training MLP feature extractor")
    feature_extractor = SimpleMLP(input_dim, hidden_dims, output_dim)
    feature_extractor, metrics = train_mlp_with_classification_head(
        feature_extractor,
        num_features=output_dim,
        num_classes=final_output_dim,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        epochs=input_net_epochs,
        batch_size=train_cfg.batch_size,
        lr=train_cfg.lr,
        run_save_path=mlp_model_dir
    )
    
    # Save feature extractor
    save_model(feature_extractor, run_save_path, "feature_extractor.pt")
    logger.info(f"Feature extractor metrics: train={metrics[0]:.4f}, valid={metrics[1]:.4f}, test={metrics[2]:.4f}")
    
    # Log to W&B if initialized
    if wandb.run is not None:
        wandb.log({
            "feature_extractor_train_acc": metrics[0],
            "feature_extractor_valid_acc": metrics[1],
            "feature_extractor_test_acc": metrics[2]
        })
    
    # Initialize the full model
    model = initialize_model(model_cfg, visualization_enabled=False)
    
    # Create a new input network with sigmoid activation
    logger.info("Creating input network with sigmoid activation")
    new_input_net = SigmoidInputNetTransform(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        network_type=model_cfg.network.parameters.input_net_dict.get("type", "MLPInputTransform")
    )
    
    # Transfer weights from trained feature extractor
    logger.info("Transferring weights to input network")
    extract_weights_from_simple_mlp(feature_extractor, new_input_net)
    
    # Set the input network in the model
    if hasattr(model, 'net') and hasattr(model.net, 'input_net'):
        model.net.input_net = new_input_net
    elif hasattr(model, 'network') and hasattr(model.network, 'input_net'):
        model.network.input_net = new_input_net
    else:
        logger.warning("Could not set input network in model - model structure may be incompatible")
    
    # Save the input network
    save_model(new_input_net, run_save_path, "input_net_with_sigmoid.pt")
    
    # Train the full model
    logger.info("Training full model with pretrained input network")
    
    # Create optimizer for full model
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr),
        weight_decay=train_cfg.weight_decay_rate,
    )
    
    # Create trainer
    trainer = get_trainer(train_cfg.learning_strategy, optimizer)
    
    # Train the model
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_clip_value=train_cfg.grad_clip_value,
        shuffle=True,
        load_best_state_dict=True,
        plot_losses=True,
        save_path=run_save_path
    )
    
    # Save the final model
    save_model(model, run_save_path, "full_model.pt")
    
    return model


def run_analysis(model, train_ds, valid_ds, test_ds, outputs_cfg, visualization_cfg, run_save_path):
    """
    Run additional analyses on the trained model if configured to do so.
    
    Args:
        model: The trained model
        train_ds, valid_ds, test_ds: The datasets
        outputs_cfg: Output configuration
        visualization_cfg: Visualization configuration
        run_save_path: Directory to save results
    """
    # Determine if visualization is enabled
    is_visualization_enabled = (visualization_cfg and 
                               hasattr(visualization_cfg, 'enabled') and 
                               visualization_cfg.enabled)
    
    # Run ablation analysis if enabled
    if is_visualization_enabled and hasattr(outputs_cfg, 'enable_ablation') and outputs_cfg.enable_ablation:
        # Check if ablation plots are requested
        ablation_plots = []
        if hasattr(visualization_cfg, 'plots'):
            ablation_plots = [p for p in visualization_cfg.plots if 'ablation' in p]
            
        if ablation_plots:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            batch_size = outputs_cfg.ablation_batch_size if hasattr(outputs_cfg, 'ablation_batch_size') else 64
            
            # Run branch-level ablation
            if 'branch_ablation' in ablation_plots:
                logger.info("Running branch-level ablation analysis")
                try:
                    for layer_idx in range(len(model.net.excitatory_layer_sizes)):
                        for branch_idx in range(model.net.excitatory_branch_factors[layer_idx]):
                            analyze_branch_ablation(
                                model, train_ds, valid_ds, test_ds,
                                layer_idx, branch_idx,
                                device=device,
                                save_path=os.path.join(run_save_path, 'ablation')
                            )
                except Exception as e:
                    logger.error(f"Error in branch ablation analysis: {e}")
            
            # Run layer-level ablation
            if 'layer_ablation' in ablation_plots:
                logger.info("Running layer-level ablation analysis")
                try:
                    analyze_layerwise_ablation(
                        model, train_ds, valid_ds, test_ds,
                        save_path=os.path.join(run_save_path, 'ablation'),
                        device=device
                    )
                except Exception as e:
                    logger.error(f"Error in layer ablation analysis: {e}")


def main(config_path, args=None):
    """
    Main function that runs the training experiment.
    
    Args:
        config_path: Path to the configuration YAML file
        args: Optional command-line arguments that can override config
        
    Returns:
        The trained model and results
    """
    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Apply command-line overrides
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
    
    # Extract key configuration sections
    model_cfg = config.model
    train_cfg = config.train
    task_cfg = config.task
    wandb_cfg = config.wandb
    outputs_cfg = config.outputs
    visualization_cfg = config.visualization
    
    # Set up training environment
    run_save_path, model, train_ds, valid_ds, test_ds, optimizer, trainer = setup_training_environment(config, args)
    
    # Generate and display model architecture summary BEFORE training
    logger.info("Network architecture summary before training:")
    try:
        sample_input, _ = train_ds[0]
        sample_input_dim = sample_input.numel()
        summary = torchinfo.summary(
            model,
            input_size=(train_cfg.batch_size, sample_input_dim),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25,
            row_settings=["depth"],
            depth=7
        )
        # Print summary at INFO level for visibility
        logger.info("\n" + str(summary))
        if wandb.run:
            wandb.log({
                "total_params": summary.total_params,
                "trainable_params": summary.trainable_params
            })
    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")
    
    # Save configuration
    save_dict(asdict(config), run_save_path, "config.json")
    
    # Check for sequential training mode
    input_mode = getattr(model_cfg.network.parameters, 'input_mode', 0)
    input_net_training = getattr(train_cfg, 'input_net_training', {})
    training_mode = input_net_training.get('training_mode', 'joint')
    sequential_training = (input_mode == 2 and training_mode == 'sequential')
    
    # Train model with appropriate strategy
    if sequential_training:
        logger.info("Using sequential training mode")
        model = train_sequential_input_net(
            model_cfg, train_cfg, task_cfg, 
            train_ds, valid_ds, test_ds, run_save_path
        )
        # Evaluate final model
        results = {
            "train_acc": evaluate_accuracy(model, train_ds),
            "valid_acc": evaluate_accuracy(model, valid_ds),
            "test_acc": evaluate_accuracy(model, test_ds)
        }
    else:
        logger.info("Using standard training mode")
        results = train_and_evaluate(
            model, train_ds, valid_ds, test_ds, trainer,
            train_cfg, visualization_cfg, run_save_path
        )
    
    # Run additional analyses
    run_analysis(model, train_ds, valid_ds, test_ds, 
                outputs_cfg, visualization_cfg, run_save_path)
    
    # Close W&B if initialized
    if hasattr(wandb_cfg, 'use_wandb') and wandb_cfg.use_wandb and wandb.run is not None:
        wandb.finish()
    
    logger.info("Training experiment completed successfully")
    
    return model, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", help="Override output directory specified in config.")
    parser.add_argument("--run_name", help="Override run name specified in config.")
    parser.add_argument("--learning_strategy", 
                       choices=["mle", "freeze_layers", "two_step", "two_step_kl"],
                       help="Override learning strategy specified in config.")
    
    args = parser.parse_args()
    main(config_path=args.config, args=args)