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
python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml --learning_strategy dendrinet_from_mlp
"""
# python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml

import os
import argparse
import logging
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import inspect

import wandb
import torch
import torchinfo
import numpy as np
import random

from dendritic_modeling import logger
from dendritic_modeling.learning_strategies import get_trainer, CustomWeightDecayOptimizer
from dendritic_modeling.models import (ProbabilisticClassifier, Classifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, MLPExcInhNetwork)
from dendritic_modeling.config import load_config
from dendritic_modeling.visualization_manager import PlotManager
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import evaluate_accuracy, save_dict


def initialize_model(model_cfg):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.
    model_cfg is from config.model, with .task, .probabilistic, .network, etc.
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters
    enable_branch_outputs = net_params.pop('enable_branch_outputs', False)
    learning_strategy = net_params.pop('learning_strategy', 'mle')

    if task == 'classification':
        if net_type == 'MLP':
            logger.info(f"Running feedforward with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = MLPExcInhNetwork(**net_params)
            output_dim = net_params.get('output_dim', 10)
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)
        elif net_type == 'EINet':
            logger.info(f"Running EINet with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = ExcitationInhibitionNetwork(
                enable_branch_outputs=enable_branch_outputs,
                learning_strategy=learning_strategy,
                **net_params,
            )

            output_dim = net_params['excitatory_layer_sizes'][-1]
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)
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

def add_network_methods():
    """
    Add methods to ExcitationInhibitionNetwork class for loading and freezing weights.
    These are added dynamically to avoid modifying the original class.
    """
    # These methods are now directly implemented in the ExcitationInhibitionNetwork class
    pass

def main(config_path: str):
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration sections
    model_cfg = config.model
    train_cfg = config.train
    task_cfg = config.task
    wandb_cfg = config.wandb
    outputs_cfg = config.outputs
    visualization_cfg = config.visualization
    
    # Set up logging
    run_name = train_cfg.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"
    
    # Create output directory
    run_save_path = os.path.join(outputs_cfg.dir, run_id)
    os.makedirs(run_save_path, exist_ok=True)
    
    # Set up file logging - Fix for AttributeError
    log_file = os.path.join(run_save_path, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")
    
    # Set random seed
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    random.seed(train_cfg.seed)
    
    # Initialize W&B
    if wandb_cfg.use_wandb:
        wandb_config = {
            "model": asdict(model_cfg),
            "train": asdict(train_cfg),
            "task": asdict(task_cfg),
        }
        wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            group=wandb_cfg.group,
            name=run_id,
            config=wandb_config,
            tags=wandb_cfg.tags,
        )
    
    # Load dataset
    # The function signature shows it expects task_cfg and train_cfg objects
    logger.info(f"get_unified_datasets signature: {inspect.signature(get_unified_datasets)}")
    
    # Pass the entire configuration objects
    train_ds, valid_ds, test_ds = get_unified_datasets(
        task_cfg,
        train_cfg
    )
    
    # Get input dimension from dataset
    if hasattr(train_ds, 'data'):
        input_dim = np.prod(train_ds.data[0].shape)
    else:
        # Default to MNIST dimensions
        input_dim = 784
    
    # Update input dimension in model config
    model_cfg.network.parameters['input_dim'] = input_dim
    
    # Get learning strategy
    learning_strategy = train_cfg.learning_strategy.lower()
    
    # Get input network configuration and training parameters
    input_net_dict = model_cfg.network.parameters.get('input_net_dict', {})
    input_mode = model_cfg.network.parameters.get('input_mode', 0)
    
    # Get input network training parameters from train config
    input_net_training = getattr(train_cfg, 'input_net_training', {})
    training_mode = input_net_training.get('training_mode', 'joint')
    sequential_training = (input_mode == 2 and training_mode == 'sequential')
    
    # Add network methods
    add_network_methods()
    
    # Handle sequential training for input_net if configured
    if sequential_training:
        logger.info("Using sequential training mode for input network and dendrinet")
        
        # Get sequential training parameters from train.input_net_training
        train_input_net = input_net_training.get('train_input_net', True)
        input_net_epochs = input_net_training.get('input_net_epochs', 20)
        freeze_input_net = input_net_training.get('freeze_input_net', True)
        
        if train_input_net:
            # First, train just the input network
            logger.info("Training input network...")
            
            # Create a standalone input network model for classification
            input_net_model_cfg = model_cfg.copy()
            input_net_model_cfg.network.type = "MLP"
            input_net_clf = initialize_model(input_net_model_cfg)
            
            # Create optimizer and trainer for input network
            input_net_optimizer = CustomWeightDecayOptimizer(
                model=input_net_clf, 
                optimizer=torch.optim.Adam(input_net_clf.parameters(), lr=train_cfg.lr), 
                weight_decay=train_cfg.weight_decay_rate,
            )
            input_net_trainer = get_trainer("mle", input_net_optimizer, suppress_prints=False, print_every=10)
            
            # Train the input network model
            input_net_results = input_net_trainer.train(
                model=input_net_clf,
                train_data=train_ds,
                valid_data=valid_ds,
                epochs=input_net_epochs,
                batch_size=train_cfg.batch_size,
                grad_clip_value=train_cfg.grad_clip_value,
                shuffle=True,
                load_best_state_dict=True,
                plot_losses=True,
                save_path=os.path.join(run_save_path, "input_net_model")
            )
            
            # Save the trained input network model
            input_net_save_path = save_model(input_net_clf, run_save_path, "input_net_model.pt")
            
            # Evaluate input network performance
            input_net_train_acc, input_net_valid_acc, input_net_test_acc = evaluate_accuracy(
                input_net_clf, train_ds, valid_ds, test_ds, 
                save_path=os.path.join(run_save_path, 'performance'), 
                filename="input_net_final",
            )
            logger.info(f"Input network train acc={input_net_train_acc:.3f}, valid acc={input_net_valid_acc:.3f}, test acc={input_net_test_acc:.3f}")
            
            if wandb_cfg.use_wandb:
                wandb.log({
                    "input_net_train_acc": input_net_train_acc,
                    "input_net_valid_acc": input_net_valid_acc,
                    "input_net_test_acc": input_net_test_acc
                })
            
            # Now initialize the full model
            logger.info("Initializing full model with trained input network...")
            model = initialize_model(model_cfg)
            
            # Extract the input network from the trained model and load it into the full model
            if hasattr(input_net_clf, 'net'):
                input_net_state = input_net_clf.net.state_dict()
            elif hasattr(input_net_clf, 'network'):
                input_net_state = input_net_clf.network.state_dict()
            else:
                input_net_state = input_net_clf.state_dict()
            
            # Load the input network weights into the full model
            if hasattr(model, 'net'):
                model.net.load_input_net_weights(input_net_state)
                if freeze_input_net:
                    model.net.freeze_input_net_weights()
            elif hasattr(model, 'network'):
                model.network.load_input_net_weights(input_net_state)
                if freeze_input_net:
                    model.network.freeze_input_net_weights()
            
            # Create optimizer for full model
            optimizer = CustomWeightDecayOptimizer(
                model=model, 
                optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr), 
                weight_decay=train_cfg.weight_decay_rate,
            )
            
            # Create trainer for full model
            trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)
            
            # Train the full model
            logger.info("Training dendritic network with fixed input network...")
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
            
            # Save the trained full model
            save_model(model, run_save_path, "full_model.pt")
            
            # Evaluate full model performance
            train_acc, valid_acc, test_acc = evaluate_accuracy(
                model, train_ds, valid_ds, test_ds, 
                save_path=os.path.join(run_save_path, 'performance'), 
                filename="full_model_final",
            )
            logger.info(f"Full model train acc={train_acc:.3f}, valid acc={valid_acc:.3f}, test acc={test_acc:.3f}")
            
            if wandb_cfg.use_wandb:
                wandb.log({
                    "full_model_train_acc": train_acc,
                    "full_model_valid_acc": valid_acc,
                    "full_model_test_acc": test_acc
                })
        
        else:
            # Use pre-trained weights for input network
            logger.info("Using pre-trained input network weights")
            
            # Initialize the full model
            model = initialize_model(model_cfg)
            
            # Load pretrained weights if specified
            pretrained_options = input_net_training.get('pretrained_options', {})
            source = pretrained_options.get('source', 'none')
            weights_path = pretrained_options.get('weights_path', '')
            pytorch_model = pretrained_options.get('pytorch_model', '')
            
            # Apply pretrained weights to the input network
            if hasattr(model, 'net') and hasattr(model.net, 'input_net'):
                model.net.input_net.load_pretrained_weights(
                    source=source,
                    weights_path=weights_path,
                    pytorch_model=pytorch_model
                )
            elif hasattr(model, 'network') and hasattr(model.network, 'input_net'):
                model.network.input_net.load_pretrained_weights(
                    source=source,
                    weights_path=weights_path,
                    pytorch_model=pytorch_model
                )
            
            # Freeze input network weights if specified
            if freeze_input_net:
                if hasattr(model, 'net'):
                    model.net.freeze_input_net_weights()
                elif hasattr(model, 'network'):
                    model.network.freeze_input_net_weights()
            
            # Create optimizer for full model
            optimizer = CustomWeightDecayOptimizer(
                model=model, 
                optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr), 
                weight_decay=train_cfg.weight_decay_rate,
            )
            
            # Create trainer for full model
            trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)
            
            # Train the full model
            logger.info("Training dendritic network with fixed input network...")
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
            
            # Save the trained full model
            save_model(model, run_save_path, "full_model.pt")
            
            # Evaluate full model performance
            train_acc, valid_acc, test_acc = evaluate_accuracy(
                model, train_ds, valid_ds, test_ds, 
                save_path=os.path.join(run_save_path, 'performance'), 
                filename="full_model_final",
            )
            logger.info(f"Full model train acc={train_acc:.3f}, valid acc={valid_acc:.3f}, test acc={test_acc:.3f}")
            
            if wandb_cfg.use_wandb:
                wandb.log({
                    "full_model_train_acc": train_acc,
                    "full_model_valid_acc": valid_acc,
                    "full_model_test_acc": test_acc
                })
    
    else:
        # Standard training (joint or other learning strategies)
        logger.info(f"Using standard training with learning strategy: {learning_strategy}")
        
        # Initialize model
        model = initialize_model(model_cfg)
        
        # Create optimizer
        optimizer = CustomWeightDecayOptimizer(
            model=model, 
            optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr), 
            weight_decay=train_cfg.weight_decay_rate,
        )
        
        # Create trainer
        trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)
        
        # Train model
        if learning_strategy in ["two_step", "two_step_kl"]:
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
            results = trainer.train(
                model=model,
                train_data=train_ds,
                valid_data=valid_ds,
                epochs=train_cfg.epochs_per_layer,
                batch_size=train_cfg.batch_size,
                grad_clip_value=train_cfg.grad_clip_value,
                shuffle=True,
                load_best_state_dict=True,
                plot_losses=True,
                save_path=run_save_path
            )
        else:
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
        
        # Save model
        save_model(model, run_save_path, "model.pt")
        
        # Evaluate model
        train_acc, valid_acc, test_acc = evaluate_accuracy(
            model, train_ds, valid_ds, test_ds, 
            save_path=os.path.join(run_save_path, 'performance'), 
            filename="final",
        )
        logger.info(f"Model train acc={train_acc:.3f}, valid acc={valid_acc:.3f}, test acc={test_acc:.3f}")
        
        if wandb_cfg.use_wandb:
            wandb.log({
                "train_acc": train_acc,
                "valid_acc": valid_acc,
                "test_acc": test_acc
            })
    
    # Save configuration
    save_dict(asdict(config), os.path.join(run_save_path, "config.json"))
    
    # Visualize model if enabled
    if visualization_cfg.enabled:
        logger.info("Creating visualizations...")
        plot_manager = PlotManager(
            model=model,
            train_data=train_ds,
            valid_data=valid_ds,
            test_data=test_ds,
            save_path=os.path.join(run_save_path, 'visualizations'),
            plots=visualization_cfg.plots
        )
        plot_manager.create_visualizations()
    
    # Close W&B
    if wandb_cfg.use_wandb:
        wandb.finish()
    
    return model, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dendritic modeling training and evaluation.')
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument("--output_dir", default="results", help="Output directory.")
    parser.add_argument("--run_name", default="example", help="Run name (used for W&B run name).")
    parser.add_argument("--learning_strategy", 
                       choices=["mle", "freeze_layers", "two_step", "two_step_kl"],
                       help="Learning strategy to use. Overrides the config file setting.")
    
    # Input network options
    parser.add_argument("--input_mode", type=int, choices=[0, 1, 2], 
                       help="Input mode (0=original, 1=exc=x+inh=x, 2=input_net)")
    parser.add_argument("--training_mode", choices=["joint", "sequential"],
                       help="Training mode for input_net (joint or sequential)")
    parser.add_argument("--train_input_net", action='store_true', 
                       help="Train the input network in sequential mode")
    parser.add_argument("--input_net_epochs", type=int, 
                       help="Number of epochs for input network training")
    parser.add_argument("--freeze_input_net", action='store_true', 
                       help="Freeze input network weights during dendrinet training")
    
    # Pre-trained options
    parser.add_argument("--pretrained_source", choices=["file", "pytorch", "none"],
                       help="Source of pre-trained weights")
    parser.add_argument("--weights_path", 
                       help="Path to weights file if pretrained_source is 'file'")
    parser.add_argument("--pytorch_model", 
                       help="Name of PyTorch model if pretrained_source is 'pytorch'")
    
    args = parser.parse_args()

    # load configuration
    config = load_config(args.config)

    # Override configuration with command line arguments
    if args.output_dir:
        config.outputs.dir = args.output_dir

    if args.run_name:
        config.train.run_name = args.run_name
        
    if args.learning_strategy:
        config.train.learning_strategy = args.learning_strategy
    
    # Update input network configuration if provided
    if args.input_mode is not None:
        config.model.network.parameters['input_mode'] = args.input_mode
    
    # Only update input_net_dict if input_mode is 2
    if config.model.network.parameters.get('input_mode') == 2:
        # Ensure input_net_dict exists
        if 'input_net_dict' not in config.model.network.parameters:
            config.model.network.parameters['input_net_dict'] = {}
        
        # Update training mode
        if args.training_mode:
            config.model.network.parameters['input_net_dict']['training_mode'] = args.training_mode
        
        # Ensure sequential_options exists if needed
        if config.model.network.parameters['input_net_dict'].get('training_mode') == 'sequential':
            if 'sequential_options' not in config.model.network.parameters['input_net_dict']:
                config.model.network.parameters['input_net_dict']['sequential_options'] = {}
            
            # Update sequential options
            if args.train_input_net:
                config.model.network.parameters['input_net_dict']['sequential_options']['train_input_net'] = True
            
            if args.input_net_epochs:
                config.model.network.parameters['input_net_dict']['sequential_options']['input_net_epochs'] = args.input_net_epochs
            
            if args.freeze_input_net:
                config.model.network.parameters['input_net_dict']['sequential_options']['freeze_input_net'] = True
        
        # Update pretrained options if provided
        if args.pretrained_source:
            if 'pretrained_options' not in config.model.network.parameters['input_net_dict']:
                config.model.network.parameters['input_net_dict']['pretrained_options'] = {}
            
            config.model.network.parameters['input_net_dict']['pretrained_options']['source'] = args.pretrained_source
            
            if args.weights_path:
                config.model.network.parameters['input_net_dict']['pretrained_options']['weights_path'] = args.weights_path
            
            if args.pytorch_model:
                config.model.network.parameters['input_net_dict']['pretrained_options']['pytorch_model'] = args.pytorch_model

    main(config_path=args.config)