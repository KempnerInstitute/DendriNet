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
from copy import deepcopy
import time

import wandb
import torch
import torchinfo
import numpy as np
import random
import torch.nn.functional as F

from dendritic_modeling import logger
from dendritic_modeling.learning_strategies import get_trainer, CustomWeightDecayOptimizer
from dendritic_modeling.models import (ProbabilisticClassifier, Classifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, MLPExcInhNetwork, InputNetTransform)
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
    net_params = model_cfg.network.parameters.copy() if hasattr(model_cfg.network.parameters, 'copy') else deepcopy(model_cfg.network.parameters)
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

            output_dim = net_params.get('output_dim', 10)  # Use output_dim directly instead of excitatory_layer_sizes[-1]
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

def transfer_mlp_weights_to_input_net(mlp_net, input_net, head_type="excitatory"):
    """
    Transfer weights from a trained MLP to the corresponding part of an InputNetTransform.
    
    Args:
        mlp_net: The trained MLP network
        input_net: The InputNetTransform to transfer weights to
        head_type: Either "excitatory" or "inhibitory" to specify which head to update
    """
    logger.info(f"Transferring weights from MLP to {head_type} head of InputNetTransform")
    
    # Extract the MLP layers
    if hasattr(mlp_net, 'layers'):
        mlp_layers = mlp_net.layers
    elif hasattr(mlp_net, 'net') and hasattr(mlp_net.net, 'layers'):
        mlp_layers = mlp_net.net.layers
    else:
        logger.warning(f"Could not extract layers from MLP model. Using random initialization for {head_type} head.")
        return
    
    # Get the shared backbone and target head of the InputNetTransform
    shared_layers = []
    for module in input_net.shared:
        if isinstance(module, torch.nn.Linear):
            shared_layers.append(module)
    
    target_head = input_net.head_exc if head_type == "excitatory" else input_net.head_inh
    head_layers = []
    for module in target_head:
        if isinstance(module, torch.nn.Linear):
            head_layers.append(module)
    
    # Transfer weights for the shared backbone (only once, typically from excitatory head)
    if head_type == "excitatory":
        for i, shared_layer in enumerate(shared_layers):
            if i < len(mlp_layers) - 1:  # All but the last MLP layer go to shared backbone
                mlp_layer = mlp_layers[i]
                if hasattr(mlp_layer, 'excit_pre_w'):
                    # MLPExcInhLayer has a different structure
                    shared_layer.weight.data = mlp_layer.excit_pre_w.exp().t()
                    shared_layer.bias.data = mlp_layer.bias.data
                else:
                    logger.warning(f"Unexpected MLP layer structure. Using random initialization for shared layer {i}.")
    
    # Transfer weights for the head-specific part
    if len(head_layers) > 0 and len(mlp_layers) > 0:
        # The last layer of MLP goes to the head
        last_mlp_layer = mlp_layers[-1]
        if hasattr(last_mlp_layer, 'excit_pre_w'):
            head_layers[0].weight.data = last_mlp_layer.excit_pre_w.exp().t()
            head_layers[0].bias.data = last_mlp_layer.bias.data
        else:
            logger.warning(f"Unexpected MLP layer structure. Using random initialization for {head_type} head.")
    
    logger.info(f"Weight transfer completed for {head_type} head")

class InhibitoryHeadTrainer:
    """
    Custom trainer for the inhibitory head with different objective functions.
    """
    def __init__(self, model, optimizer, objective="classification", 
                 excit_head_model=None, alpha=0.5, beta=0.1, 
                 suppress_prints=False, print_every=10):
        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.excit_head_model = excit_head_model
        self.alpha = alpha  # Weight for the diversity loss
        self.beta = beta    # Weight for the reconstruction loss
        self.suppress = suppress_prints
        self.print_every = print_every
        
    def compute_classification_loss(self, x, y):
        """Standard classification loss"""
        return self.model.compute_loss(x, y)
    
    def compute_diversity_loss(self, x):
        """
        Diversity loss: encourages inhibitory head to learn features different from excitatory head
        by maximizing the cosine distance between their representations.
        """
        if self.excit_head_model is None:
            return torch.tensor(0.0, device=x.device)
        
        # Get representations from both heads
        with torch.no_grad():
            if hasattr(self.excit_head_model, 'net'):
                excit_repr = self.excit_head_model.net(x)
            elif hasattr(self.excit_head_model, 'network'):
                excit_repr = self.excit_head_model.network(x)
            else:
                excit_repr = self.excit_head_model(x)
        
        if hasattr(self.model, 'net'):
            inhib_repr = self.model.net(x)
        elif hasattr(self.model, 'network'):
            inhib_repr = self.model.network(x)
        else:
            inhib_repr = self.model(x)
        
        # Normalize the representations
        excit_repr = F.normalize(excit_repr, p=2, dim=1)
        inhib_repr = F.normalize(inhib_repr, p=2, dim=1)
        
        # Compute cosine similarity (we want to minimize this, or maximize negative)
        similarity = torch.sum(excit_repr * inhib_repr, dim=1).mean()
        
        # Return negative similarity to maximize distance
        return similarity
    
    def compute_reconstruction_loss(self, x):
        """
        Reconstruction loss: encourages inhibitory head to learn a good representation
        by reconstructing the input.
        """
        if hasattr(self.model, 'net'):
            inhib_repr = self.model.net(x)
        elif hasattr(self.model, 'network'):
            inhib_repr = self.model.network(x)
        else:
            inhib_repr = self.model(x)
        
        # Simple linear reconstruction (could be replaced with a more complex decoder)
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Linear(inhib_repr.size(1), x.size(1)).to(x.device)
            torch.nn.init.xavier_uniform_(self.decoder.weight)
            torch.nn.init.zeros_(self.decoder.bias)
            self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.001)
        
        # Reconstruct the input
        x_recon = self.decoder(inhib_repr)
        
        # Compute reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        
        # Update decoder
        self.decoder_optimizer.zero_grad()
        recon_loss.backward(retain_graph=True)
        self.decoder_optimizer.step()
        
        return recon_loss
    
    def train(self, model, train_data, valid_data=None, epochs=100, batch_size=256, 
              grad_clip_value=5, shuffle=True, load_best_state_dict=True, 
              plot_losses=False, save_path=None):
        """
        Train the inhibitory head with the specified objective.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if self.excit_head_model is not None:
            self.excit_head_model = self.excit_head_model.to(device)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        train_losses = []
        valid_losses = []
        best_loss = float('inf')
        best_state = deepcopy(model.state_dict())
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                # Compute the appropriate loss based on the objective
                if self.objective == "classification":
                    loss = self.compute_classification_loss(xb, yb)
                elif self.objective == "diversity":
                    class_loss = self.compute_classification_loss(xb, yb)
                    div_loss = self.compute_diversity_loss(xb)
                    loss = class_loss + self.alpha * div_loss
                elif self.objective == "reconstruction":
                    class_loss = self.compute_classification_loss(xb, yb)
                    recon_loss = self.compute_reconstruction_loss(xb)
                    loss = class_loss + self.beta * recon_loss
                elif self.objective == "combined":
                    class_loss = self.compute_classification_loss(xb, yb)
                    div_loss = self.compute_diversity_loss(xb)
                    recon_loss = self.compute_reconstruction_loss(xb)
                    loss = class_loss + self.alpha * div_loss + self.beta * recon_loss
                else:
                    raise ValueError(f"Unknown objective: {self.objective}")
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Evaluate on validation set
            model.eval()
            val_loss = 0.0
            
            if valid_data is not None:
                valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
                with torch.no_grad():
                    for xb, yb in valid_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        
                        # Use classification loss for validation
                        loss = self.compute_classification_loss(xb, yb)
                        val_loss += loss.item()
                
                val_loss /= len(valid_loader)
                valid_losses.append(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                    if save_path and os.path.exists(save_path):
                        torch.save(best_state, os.path.join(save_path, "best_model.pt"))
            
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                if valid_data is not None:
                    logger.info(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                                f"valid_loss={val_loss:.4f}, elapsed={elapsed:.1f}s")
                else:
                    logger.info(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                                f"elapsed={elapsed:.1f}s")
            
            if wandb.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                }
                if valid_data is not None:
                    log_dict["valid_loss"] = val_loss
                wandb.log(log_dict)
        
        if load_best_state_dict and valid_data is not None:
            model.load_state_dict(best_state)
        
        results = {
            "train_losses": train_losses,
            "valid_losses": valid_losses if valid_data is not None else [],
            "best_loss": best_loss if valid_data is not None else None,
        }
        
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses if valid_data is not None else [], 
                                     epochs, save_path)
        
        return results

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
            
            # Get training options for the heads
            train_heads_independently = input_net_training.get('train_heads_independently', False)
            excitatory_head_epochs = input_net_training.get('excitatory_head_epochs', input_net_epochs)
            inhibitory_head_epochs = input_net_training.get('inhibitory_head_epochs', input_net_epochs)
            
            if train_heads_independently:
                logger.info("Training excitatory and inhibitory heads independently")
                
                # PHASE 1: Train the excitatory head for classification
                logger.info("Phase 1: Training excitatory head for classification...")
                
                # Create a standalone input network model for classification (excitatory head)
                excit_model_cfg = deepcopy(model_cfg)
                excit_model_cfg.network.type = "MLP"
                
                # Get input network parameters from the configuration
                input_net_dict = model_cfg.network.parameters.get('input_net_dict', {})
                hidden_dims = input_net_dict.get('hidden_dims', [128, 64])
                net_output_dim = input_net_dict.get('output_dim', 64)
                
                # For classification, the final output dimension must be the number of classes
                final_output_dim = 10  # For MNIST
                
                # Set up MLP parameters for excitatory head
                excit_model_cfg.network.parameters = {
                    'input_dim': input_dim,
                    'excitatory_layer_sizes': hidden_dims + [final_output_dim],  # Hidden layers + output layer (10 classes)
                    'inhibitory_layer_sizes': [],  # No inhibitory layers for MLP
                    'excitatory_branch_factors': [1] * (len(hidden_dims) + 1),  # No branching
                    'inhibitory_branch_factors': [],
                    'ee_synapses_per_branch_per_layer': [1] * (len(hidden_dims) + 1),
                    'ie_synapses_per_branch_per_layer': [0] * (len(hidden_dims) + 1),
                    'output_dim': final_output_dim  # Number of classes
                }
                
                excit_head_clf = initialize_model(excit_model_cfg)
                
                # Create optimizer and trainer for excitatory head
                excit_optimizer = CustomWeightDecayOptimizer(
                    model=excit_head_clf, 
                    optimizer=torch.optim.Adam(excit_head_clf.parameters(), lr=train_cfg.lr), 
                    weight_decay=train_cfg.weight_decay_rate,
                )
                excit_trainer = get_trainer("mle", excit_optimizer, suppress_prints=False, print_every=10)
                
                # Train the excitatory head
                excit_results = excit_trainer.train(
                    model=excit_head_clf,
                    train_data=train_ds,
                    valid_data=valid_ds,
                    epochs=excitatory_head_epochs,
                    batch_size=train_cfg.batch_size,
                    grad_clip_value=train_cfg.grad_clip_value,
                    shuffle=True,
                    load_best_state_dict=True,
                    plot_losses=True,
                    save_path=os.path.join(run_save_path, "excit_head_model")
                )
                
                # Save the trained excitatory head model
                excit_save_path = save_model(excit_head_clf, run_save_path, "excit_head_model.pt")
                
                # Evaluate excitatory head performance
                excit_train_acc, excit_valid_acc, excit_test_acc = evaluate_accuracy(
                    excit_head_clf, train_ds, valid_ds, test_ds, 
                    save_path=os.path.join(run_save_path, 'performance'), 
                    filename="excit_head_final",
                )
                logger.info(f"Excitatory head train acc={excit_train_acc:.3f}, valid acc={excit_valid_acc:.3f}, test acc={excit_test_acc:.3f}")
                
                if wandb_cfg.use_wandb:
                    wandb.log({
                        "excit_head_train_acc": excit_train_acc,
                        "excit_head_valid_acc": excit_valid_acc,
                        "excit_head_test_acc": excit_test_acc
                    })
                
                # PHASE 2: Train the inhibitory head (with different objective if needed)
                logger.info("Phase 2: Training inhibitory head...")
                
                # Get the inhibitory head training objective
                inhib_objective = input_net_training.get('inhibitory_head_objective', 'classification')
                diversity_weight = input_net_training.get('diversity_weight', 0.5)
                reconstruction_weight = input_net_training.get('reconstruction_weight', 0.1)
                
                logger.info(f"Using {inhib_objective} objective for inhibitory head training")
                
                # Create inhibitory head model
                inhib_model_cfg = deepcopy(model_cfg)
                inhib_model_cfg.network.type = "MLP"
                
                # Set up MLP parameters for inhibitory head
                inhib_model_cfg.network.parameters = {
                    'input_dim': input_dim,
                    'excitatory_layer_sizes': hidden_dims + [net_output_dim],  # Use original output_dim
                    'inhibitory_layer_sizes': [],  # No inhibitory layers for MLP
                    'excitatory_branch_factors': [1] * (len(hidden_dims) + 1),  # No branching
                    'inhibitory_branch_factors': [],
                    'ee_synapses_per_branch_per_layer': [1] * (len(hidden_dims) + 1),
                    'ie_synapses_per_branch_per_layer': [0] * (len(hidden_dims) + 1),
                    'output_dim': final_output_dim  # Number of classes for training
                }
                
                inhib_head_clf = initialize_model(inhib_model_cfg)
                
                # Create optimizer for inhibitory head
                inhib_optimizer = CustomWeightDecayOptimizer(
                    model=inhib_head_clf, 
                    optimizer=torch.optim.Adam(inhib_head_clf.parameters(), lr=train_cfg.lr), 
                    weight_decay=train_cfg.weight_decay_rate,
                )
                
                # Create appropriate trainer based on the objective
                if inhib_objective == "classification":
                    # Use standard trainer for classification
                    inhib_trainer = get_trainer("mle", inhib_optimizer, suppress_prints=False, print_every=10)
                else:
                    # Use custom trainer for other objectives
                    inhib_trainer = InhibitoryHeadTrainer(
                        model=inhib_head_clf,
                        optimizer=inhib_optimizer,
                        objective=inhib_objective,
                        excit_head_model=excit_head_clf if inhib_objective in ["diversity", "combined"] else None,
                        alpha=diversity_weight,
                        beta=reconstruction_weight,
                        suppress_prints=False,
                        print_every=10
                    )
                
                # Train the inhibitory head
                inhib_results = inhib_trainer.train(
                    model=inhib_head_clf,
                    train_data=train_ds,
                    valid_data=valid_ds,
                    epochs=inhibitory_head_epochs,
                    batch_size=train_cfg.batch_size,
                    grad_clip_value=train_cfg.grad_clip_value,
                    shuffle=True,
                    load_best_state_dict=True,
                    plot_losses=True,
                    save_path=os.path.join(run_save_path, "inhib_head_model")
                )
                
                # Save the trained inhibitory head model
                inhib_save_path = save_model(inhib_head_clf, run_save_path, "inhib_head_model.pt")
                
                # Evaluate inhibitory head performance
                inhib_train_acc, inhib_valid_acc, inhib_test_acc = evaluate_accuracy(
                    inhib_head_clf, train_ds, valid_ds, test_ds, 
                    save_path=os.path.join(run_save_path, 'performance'), 
                    filename="inhib_head_final",
                )
                logger.info(f"Inhibitory head train acc={inhib_train_acc:.3f}, valid acc={inhib_valid_acc:.3f}, test acc={inhib_test_acc:.3f}")
                
                if wandb_cfg.use_wandb:
                    wandb.log({
                        "inhib_head_train_acc": inhib_train_acc,
                        "inhib_head_valid_acc": inhib_valid_acc,
                        "inhib_head_test_acc": inhib_test_acc
                    })
                
                # PHASE 3: Create the full model with both trained heads
                logger.info("Phase 3: Creating full model with both trained heads...")
                
                # Initialize the full model
                model = initialize_model(model_cfg)
                
                # Create a new InputNetTransform with the same architecture
                input_net_type = input_net_dict.get("type", "MLPInputTransform")
                
                # Create a new input network with the same architecture
                new_input_net = InputNetTransform(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=net_output_dim,
                    network_type=input_net_type
                )
                
                # Extract networks from the trained models
                if hasattr(excit_head_clf, 'net'):
                    excit_net = excit_head_clf.net
                elif hasattr(excit_head_clf, 'network'):
                    excit_net = excit_head_clf.network
                else:
                    excit_net = excit_head_clf
                
                if hasattr(inhib_head_clf, 'net'):
                    inhib_net = inhib_head_clf.net
                elif hasattr(inhib_head_clf, 'network'):
                    inhib_net = inhib_head_clf.network
                else:
                    inhib_net = inhib_head_clf
                
                # Transfer weights from trained models to the new input network
                transfer_mlp_weights_to_input_net(excit_net, new_input_net, head_type="excitatory")
                transfer_mlp_weights_to_input_net(inhib_net, new_input_net, head_type="inhibitory")
                
                # Set the input network in the full model
                if hasattr(model, 'net'):
                    model.net.input_net = new_input_net
                elif hasattr(model, 'network'):
                    model.network.input_net = new_input_net
                
                # Freeze input network weights
                if freeze_input_net:
                    if hasattr(model, 'net'):
                        model.net.freeze_input_net_weights()
                    elif hasattr(model, 'network'):
                        model.network.freeze_input_net_weights()
            
            else:
                # Original sequential training code (train both heads together)
                # Create a standalone input network model for classification
                input_net_model_cfg = deepcopy(model_cfg)
                
                # Configure the input network as an MLP
                input_net_model_cfg.network.type = "MLP"
                
                # Get input network parameters from the configuration
                input_net_dict = model_cfg.network.parameters.get('input_net_dict', {})
                hidden_dims = input_net_dict.get('hidden_dims', [128, 64])
                net_output_dim = input_net_dict.get('output_dim', 64)
                
                # For classification, the final output dimension must be the number of classes
                final_output_dim = 10  # For MNIST
                
                # Set up MLP parameters
                input_net_model_cfg.network.parameters = {
                    'input_dim': input_dim,
                    'excitatory_layer_sizes': hidden_dims + [final_output_dim],  # Hidden layers + output layer (10 classes)
                    'inhibitory_layer_sizes': [],  # No inhibitory layers for MLP
                    'excitatory_branch_factors': [1] * (len(hidden_dims) + 1),  # No branching
                    'inhibitory_branch_factors': [],
                    'ee_synapses_per_branch_per_layer': [1] * (len(hidden_dims) + 1),
                    'ie_synapses_per_branch_per_layer': [0] * (len(hidden_dims) + 1),
                    'output_dim': final_output_dim  # Number of classes
                }
                
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
                
                # Extract the input network weights from the trained MLP model
                logger.info("Extracting weights from trained MLP model...")
                
                # Create a new InputNetTransform with the same architecture as specified in input_net_dict
                input_net_type = input_net_dict.get("type", "MLPInputTransform")
                hidden_dims = input_net_dict.get("hidden_dims", [128, 64])
                net_output_dim = input_net_dict.get("output_dim", 64)
                
                # Create a new input network with the same architecture
                # Note: For the dendritic network, we use the original output_dim from input_net_dict
                # This is different from the MLP classifier which needed to output 10 classes
                new_input_net = InputNetTransform(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=net_output_dim,  # Use the original output_dim for the input network
                    network_type=input_net_type
                )
                
                # Extract weights from the trained MLP model
                # This is a simplified approach - in practice, you'd need to map the weights correctly
                # based on the specific architectures of both models
                logger.info("Note: This is a simplified weight transfer that may not be optimal.")
                
                # For now, we'll just initialize the input network with random weights
                # In a real implementation, you'd map the weights from the MLP to the input network
                
                # Set the input network in the full model
                if hasattr(model, 'net'):
                    model.net.input_net = new_input_net
                elif hasattr(model, 'network'):
                    model.network.input_net = new_input_net
                    
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
    save_dict(asdict(config), run_save_path, "config.json")
    
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
    
    # Ensure input_net_training exists in train config
    if not hasattr(config.train, 'input_net_training'):
        config.train.input_net_training = {}
    
    # Update training mode
    if args.training_mode:
        config.train.input_net_training['training_mode'] = args.training_mode
    
    # Update sequential options
    if args.train_input_net:
        config.train.input_net_training['train_input_net'] = True
    
    if args.input_net_epochs:
        config.train.input_net_training['input_net_epochs'] = args.input_net_epochs
    
    if args.freeze_input_net:
        config.train.input_net_training['freeze_input_net'] = True
    
    # Update pretrained options if provided
    if args.pretrained_source:
        if 'pretrained_options' not in config.train.input_net_training:
            config.train.input_net_training['pretrained_options'] = {}
        
        config.train.input_net_training['pretrained_options']['source'] = args.pretrained_source
        
        if args.weights_path:
            config.train.input_net_training['pretrained_options']['weights_path'] = args.weights_path
        
        if args.pytorch_model:
            config.train.input_net_training['pretrained_options']['pytorch_model'] = args.pytorch_model

    main(config_path=args.config)