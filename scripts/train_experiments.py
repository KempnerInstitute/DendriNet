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
import pandas as pd
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

from dendritic_modeling import logger
from dendritic_modeling.learning_strategies import get_trainer, CustomWeightDecayOptimizer
from dendritic_modeling.models import (ProbabilisticClassifier, Classifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork, MLPExcInhNetwork, InputNetTransform)
from dendritic_modeling.config import load_config
from dendritic_modeling.visualization_manager import PlotManager as CorePlotManager
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import (
    Shaper, 
    analyze_branch_ablation,
    analyze_layerwise_ablation,
    save_dict, 
    evaluate_accuracy, 
    record_branch_activations,
    compute_MI,
    SafeHookModule,
    safe_register_hooks
)


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
                model = ProbabilisticClassifier(net, output_dim)
            else:
                model = Classifier(net)
                
            # Apply safe hooks
            safe_register_hooks(model)
            return model
                
        elif net_type == 'EINet':
            logger.info(f"Running EINet with {'probabilistic' if probabilistic else 'simple'} classifier")
            net = ExcitationInhibitionNetwork(
                enable_branch_outputs=enable_branch_outputs,
                learning_strategy=learning_strategy,
                **net_params,
            )
            
            # Apply safe hooks to the network
            safe_register_hooks(net)

            output_dim = net_params.get('output_dim', 10)  # Use output_dim directly instead of excitatory_layer_sizes[-1]
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

# Add a custom InputNetTransform with sigmoid activation
class SigmoidInputNetTransform(InputNetTransform):
    """
    Extension of InputNetTransform that applies a sigmoid activation to the output,
    ensuring values are in the range [0, 1] before entering the dendrinet.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, network_type="MLPInputTransform"):
        super().__init__(input_dim, hidden_dims, output_dim, network_type)
        
    def forward(self, x):
        # Call the parent's forward method to get outputs
        excit_out, inhib_out = InputNetTransform.forward(self, x)
        
        # Apply sigmoid activation to ensure outputs are in [0, 1]
        excit_out = torch.sigmoid(excit_out)
        inhib_out = torch.sigmoid(inhib_out)
        
        return excit_out, inhib_out

# Add a standard PyTorch MLP for better MNIST classification
class SimpleMLP(torch.nn.Module):
    """
    A simple PyTorch MLP that's optimized for MNIST classification.
    This will be used to demonstrate good performance on MNIST.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(h_dim))
            prev_dim = h_dim
        
        # Create output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        # Ensure input is flattened
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return self.network(x)

# Simple classifier that wraps the simple MLP
class SimpleClassifier(torch.nn.Module):
    """
    A simple classifier that wraps our SimpleMLP for better performance.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = SimpleMLP(input_dim, hidden_dims, output_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.net(x)
    
    def compute_loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)

# Add a function to train the simple MLP
def train_simple_mlp(input_dim, hidden_dims, output_dim, train_ds, valid_ds, test_ds, 
                     epochs=20, batch_size=128, lr=0.001, run_save_path=None):
    """
    Train a simple MLP model on MNIST with better performance.
    Returns the trained model and its accuracy.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create the model
    model = SimpleClassifier(input_dim, hidden_dims, output_dim)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)
    
    # Training loop
    best_valid_acc = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = model.criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                outputs = model(xb)
                loss = model.criterion(outputs, yb)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += yb.size(0)
                valid_correct += predicted.eq(yb).sum().item()
        
        valid_loss /= len(valid_loader)
        valid_acc = valid_correct / valid_total
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())
            
            if run_save_path:
                save_path = os.path.join(run_save_path, "best_simple_mlp.pt")
                torch.save(best_state, save_path)
        
        # Log progress
        logger.info(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Evaluate on test set
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            outputs = model(xb)
            _, predicted = outputs.max(1)
            test_total += yb.size(0)
            test_correct += predicted.eq(yb).sum().item()
    
    test_acc = test_correct / test_total
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    # Return the model and its accuracy
    return model, (train_acc, best_valid_acc, test_acc)

# Add a function to extract weights from the simple MLP for the input network
def extract_weights_from_simple_mlp(simple_mlp, input_net_transform):
    """
    Extract weights from a SimpleMLP model and transfer them to an InputNetTransform.
    
    Args:
        simple_mlp: The trained SimpleMLP model
        input_net_transform: The InputNetTransform to transfer weights to
    """
    logger.info("Transferring weights from SimpleMLP to InputNetTransform")
    
    # Get network modules
    mlp_modules = list(simple_mlp.network.children())
    mlp_linear_modules = [m for m in mlp_modules if isinstance(m, torch.nn.Linear)]
    
    # Get shared backbone modules of InputNetTransform
    shared_modules = []
    for module in input_net_transform.shared:
        if isinstance(module, torch.nn.Linear):
            shared_modules.append(module)
    
    # Get head modules
    exc_head_modules = []
    for module in input_net_transform.head_exc:
        if isinstance(module, torch.nn.Linear):
            exc_head_modules.append(module)
    
    inh_head_modules = []
    for module in input_net_transform.head_inh:
        if isinstance(module, torch.nn.Linear):
            inh_head_modules.append(module)
    
    # Transfer weights for the shared backbone (from early layers of MLP)
    for i, shared_module in enumerate(shared_modules):
        if i < len(mlp_linear_modules) - 1:  # All but the last MLP layer
            mlp_layer = mlp_linear_modules[i]
            shared_module.weight.data = mlp_layer.weight.data.clone()
            shared_module.bias.data = mlp_layer.bias.data.clone()
            logger.info(f"Transferred weights for shared layer {i}")
    
    # Transfer weights for heads (from last layer of MLP)
    if len(mlp_linear_modules) > 0 and len(exc_head_modules) > 0:
        # Use the second-to-last layer for the heads (since the last layer is the classifier)
        if len(mlp_linear_modules) >= 2:
            source_layer = mlp_linear_modules[-2]
        else:
            source_layer = mlp_linear_modules[0]
        
        # Transfer to excitatory head
        exc_head_modules[0].weight.data = source_layer.weight.data[:exc_head_modules[0].weight.shape[0], :].clone()
        exc_head_modules[0].bias.data = source_layer.bias.data[:exc_head_modules[0].bias.shape[0]].clone()
        logger.info("Transferred weights to excitatory head")
        
        # Transfer to inhibitory head
        inh_head_modules[0].weight.data = source_layer.weight.data[:inh_head_modules[0].weight.shape[0], :].clone()
        inh_head_modules[0].bias.data = source_layer.bias.data[:inh_head_modules[0].bias.shape[0]].clone()
        logger.info("Transferred weights to inhibitory head")
    
    logger.info("Weight transfer from SimpleMLP to InputNetTransform completed")

class InhibitoryHeadTrainer:
    """
    Custom trainer for the inhibitory head with different objective functions.
    """
    def __init__(self, model, optimizer, objective="classification", 
                 excit_head_model=None, alpha=0.5,
                 suppress_prints=False, print_every=10):
        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.excit_head_model = excit_head_model
        self.alpha = alpha  # Weight for the diversity loss
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
                elif self.objective == "combined":
                    class_loss = self.compute_classification_loss(xb, yb)
                    div_loss = self.compute_diversity_loss(xb)
                    loss = class_loss + self.alpha * div_loss
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

class MLPWithClassificationHead(torch.nn.Module):
    """
    A wrapper around an MLP that adds a classification head for training,
    but allows accessing the feature extractor part separately.
    """
    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = torch.nn.Linear(num_features, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classification_head(features)
    
    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def compute_loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)

# Train an MLP with a classification head for feature learning
def train_mlp_with_classification_head(feature_extractor, num_features, num_classes, 
                                     train_ds, valid_ds, test_ds, 
                                     epochs=20, batch_size=128, lr=0.001, run_save_path=None):
    """
    Train an MLP with a classification head, then return the trained feature extractor part.
    
    Args:
        feature_extractor: The feature extraction part of the MLP
        num_features: The output dimension of the feature extractor
        num_classes: The number of classes for classification
        train_ds, valid_ds, test_ds: The datasets
        epochs, batch_size, lr: Training parameters
        run_save_path: Where to save the model and logs
        
    Returns:
        The trained feature extractor
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create the model with classification head
    model = MLPWithClassificationHead(feature_extractor, num_features, num_classes)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)
    
    # Training loop
    best_valid_acc = 0.0
    best_state = None
    best_epoch = 0
    
    # Create the save directory if it doesn't exist
    if run_save_path:
        # Ensure the main directory exists
        os.makedirs(run_save_path, exist_ok=True)
        logger.info(f"Will save best model to {run_save_path} at the end of training")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = model.criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                outputs = model(xb)
                loss = model.criterion(outputs, yb)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += yb.size(0)
                valid_correct += predicted.eq(yb).sum().item()
        
        valid_loss /= len(valid_loader)
        valid_acc = valid_correct / valid_total
        
        # Track best model but don't save yet
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            logger.info(f"New best model found at epoch {epoch} with validation accuracy: {valid_acc:.4f}")
        
        # Log progress
        logger.info(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "mlp_train_loss": train_loss,
                "mlp_train_acc": train_acc,
                "mlp_valid_loss": valid_loss,
                "mlp_valid_acc": valid_acc,
            })
    
    # Save only the best model at the end of training
    if best_state and run_save_path:
        save_path = os.path.join(run_save_path, "best_mlp_with_classification.pt")
        torch.save(best_state, save_path)
        logger.info(f"Saved best model (from epoch {best_epoch}) to {save_path}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        logger.info(f"Loaded best model from epoch {best_epoch} with validation accuracy: {best_valid_acc:.4f}")
    
    # Evaluate on test set
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            outputs = model(xb)
            _, predicted = outputs.max(1)
            test_total += yb.size(0)
            test_correct += predicted.eq(yb).sum().item()
    
    test_acc = test_correct / test_total
    logger.info(f"Test accuracy with classification head: {test_acc:.4f}")
    
    # Return the feature extractor part (without the classification head)
    return model.feature_extractor, (train_acc, best_valid_acc, test_acc)

# Update the wrapper class to work correctly
class PlotManager:
    """
    A wrapper around the core PlotManager class that adapts to the interface
    expected by train_experiments.py.
    """
    def __init__(self, model, train_data, valid_data, test_data, save_path, plots):
        self.model = model
        # Create data dictionary as expected by CorePlotManager
        self.data = {
            'train': train_data,
            'valid': valid_data,
            'test': test_data
        }
        self.save_path = save_path
        self.plots = plots
        
        # Create a simple args object that mimics the expected structure
        class PlotManagerArgs:
            def __init__(self, save_path, file_name, plots):
                self.save_path = save_path
                self.file_name = "analysis"
                self.visualization = type('obj', (object,), {
                    'enabled': True,
                    'plots': plots
                })
        
        # Create args for CorePlotManager
        plot_args = PlotManagerArgs(
            save_path=os.path.join(save_path, 'visualizations'),
            file_name="analysis",
            plots=plots
        )
        
        # Create the core PlotManager with the correct parameters
        self.core_manager = CorePlotManager(
            model=model,
            data=self.data,
            args=plot_args,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

# Add an option to skip ablation analysis
def run_ablation_analysis(model, train_ds, valid_ds, test_ds, save_path, device='cuda', enable_ablation=True, batch_size=64):
    """
    Run branch and layerwise ablation analysis explicitly, without relying on the visualization system.
    
    Args:
        model: The model to analyze
        train_ds, valid_ds, test_ds: Datasets for evaluation
        save_path: Directory where to save results
        device: Device to run analysis on
        enable_ablation: Set to False to skip ablation analysis
        batch_size: Batch size for evaluation
    """
    if not enable_ablation:
        print("\n" + "="*80)
        print("ABLATION ANALYSIS SKIPPED (disabled in config)")
        print("="*80 + "\n")
        logger.info("Ablation analysis skipped (disabled in config)")
        return None
        
    print("\n" + "="*80)
    print("STARTING ABLATION ANALYSIS")
    print("="*80)
    logger.info("Running explicit ablation analysis...")
    
    # Create ablation results directory
    ablation_dir = os.path.join(save_path, 'ablation_results')
    os.makedirs(ablation_dir, exist_ok=True)
    print(f"Results will be saved to: {ablation_dir}")
    
    # Use smaller batch size for ablation to reduce memory usage
    print(f"Preparing dataloaders with batch size {batch_size}...")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Dataset sizes - Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")
    
    # Ensure model is on the correct device
    print(f"Moving model to {device}...")
    model = model.to(device)
    
    # Run branch ablation analysis
    try:
        print("\n" + "-"*60)
        print("RUNNING BRANCH ABLATION ANALYSIS")
        print("-"*60)
        logger.info("Running branch ablation analysis...")
        
        print("This may take a few minutes as it tests ablating different branches...")
        start_time = time.time()
        branch_ablation_results = analyze_branch_ablation(
            model=model, 
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device
        )
        end_time = time.time()
        print(f"Branch ablation took {end_time - start_time:.2f} seconds")
        
        # Save branch ablation results
        branch_save_path = os.path.join(ablation_dir, 'branch_ablation_results.json')
        save_dict(branch_ablation_results, os.path.dirname(branch_save_path), os.path.basename(branch_save_path))
        print(f"✓ Branch ablation completed successfully")
        print(f"✓ Saved branch ablation results to {branch_save_path}")
        logger.info(f"Saved branch ablation results to {branch_save_path}")
    except Exception as e:
        print(f"✗ Error in branch ablation analysis: {str(e)}")
        logger.error(f"Error in branch ablation analysis: {str(e)}")
    
    # Run layerwise ablation analysis
    try:
        print("\n" + "-"*60)
        print("RUNNING LAYERWISE ABLATION ANALYSIS")
        print("-"*60)
        logger.info("Running layerwise ablation analysis...")
        
        print("Testing performance impact of ablating entire layers...")
        start_time = time.time()
        layerwise_ablation_results = analyze_layerwise_ablation(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            device=device
        )
        end_time = time.time()
        print(f"Layerwise ablation took {end_time - start_time:.2f} seconds")
        
        # Save layerwise ablation results
        layer_save_path = os.path.join(ablation_dir, 'layerwise_ablation_results.json')
        save_dict(layerwise_ablation_results, os.path.dirname(layer_save_path), os.path.basename(layer_save_path))
        print(f"✓ Layerwise ablation completed successfully")
        print(f"✓ Saved layerwise ablation results to {layer_save_path}")
        logger.info(f"Saved layerwise ablation results to {layer_save_path}")
    except Exception as e:
        print(f"✗ Error in layerwise ablation analysis: {str(e)}")
        logger.error(f"Error in layerwise ablation analysis: {str(e)}")
    
    # Record branch activations
    try:
        print("\n" + "-"*60)
        print("RECORDING BRANCH ACTIVATIONS")
        print("-"*60)
        logger.info("Recording branch activations...")
        
        print("Measuring activations across different branches...")
        start_time = time.time()
        activation_results = record_branch_activations(
            model=model,
            data_loader=valid_loader,
            num_samples=100,  # Limit to 100 samples for speed
            device=device
        )
        end_time = time.time()
        print(f"Recording branch activations took {end_time - start_time:.2f} seconds")
        
        # Save activation results
        activation_save_path = os.path.join(ablation_dir, 'branch_activations.json')
        save_dict(activation_results, os.path.dirname(activation_save_path), os.path.basename(activation_save_path))
        print(f"✓ Branch activation recording completed successfully")
        print(f"✓ Saved branch activation results to {activation_save_path}")
        logger.info(f"Saved branch activation results to {activation_save_path}")
    except Exception as e:
        print(f"✗ Error in recording branch activations: {str(e)}")
        logger.error(f"Error in recording branch activations: {str(e)}")
    
    # Compute mutual information if possible
    try:
        print("\n" + "-"*60)
        print("COMPUTING MUTUAL INFORMATION")
        print("-"*60)
        logger.info("Computing mutual information...")
        
        print("Calculating mutual information between branches and output...")
        start_time = time.time()
        mi_results = compute_MI(
            model=model,
            data_loader=valid_loader,
            device=device
        )
        end_time = time.time()
        print(f"Computing mutual information took {end_time - start_time:.2f} seconds")
        
        # Save MI results
        mi_save_path = os.path.join(ablation_dir, 'mutual_information.json')
        save_dict(mi_results, os.path.dirname(mi_save_path), os.path.basename(mi_save_path))
        print(f"✓ Mutual information calculation completed successfully")
        print(f"✓ Saved mutual information results to {mi_save_path}")
        logger.info(f"Saved mutual information results to {mi_save_path}")
    except Exception as e:
        print(f"✗ Error in computing mutual information: {str(e)}")
        logger.error(f"Error in computing mutual information: {str(e)}")
    
    print("\n" + "="*60)
    print("ABLATION ANALYSIS COMPLETED")
    print("="*60 + "\n")
    
    return {
        'branch_ablation': branch_ablation_results if 'branch_ablation_results' in locals() else None,
        'layerwise_ablation': layerwise_ablation_results if 'layerwise_ablation_results' in locals() else None,
        'branch_activations': activation_results if 'activation_results' in locals() else None,
        'mutual_information': mi_results if 'mi_results' in locals() else None
    }

# Add a function for information estimation using MLP decoders
def run_information_estimation(model, train_ds, valid_ds, test_ds, save_path, device='cuda', batch_size=64, epochs=20):
    """
    Run information estimation analysis by training MLP decoders on branch activations.
    This measures how much task-relevant information is contained in each branch/layer.
    
    Args:
        model: The model to analyze
        train_ds, valid_ds, test_ds: Datasets for evaluation
        save_path: Directory where to save results
        device: Device to run analysis on
        batch_size: Batch size for evaluation
        epochs: Number of epochs to train MLP classifiers
    """
    print("\n" + "="*80)
    print("STARTING INFORMATION ESTIMATION ANALYSIS")
    print("="*80)
    logger.info("Running information estimation analysis...")
    
    # Create results directory
    info_dir = os.path.join(save_path, 'information_analysis')
    os.makedirs(info_dir, exist_ok=True)
    print(f"Results will be saved to: {info_dir}")
    print(f"Will train MLP classifiers for {epochs} epochs")
    
    # Prepare dataloaders
    print(f"Preparing dataloaders with batch size {batch_size}...")
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Dataset sizes - Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")
    
    # Ensure model is on the correct device
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Helper function to train MLP classifier
    def train_mlp_classifier(train_features, train_labels, valid_features, valid_labels, hidden_dim=100, lr=0.001, epochs=epochs):
        """
        Train MLP classifier for estimating information content.
        
        Args:
            train_features: Training set features [n_samples, n_features]
            train_labels: Training set labels [n_samples]
            valid_features: Validation set features [n_samples, n_features]
            valid_labels: Validation set labels [n_samples]
            hidden_dim: Hidden dimension size
            lr: Learning rate
            epochs: Number of training epochs
            
        Returns:
            Accuracy of the trained model on validation set
        """
        input_dim = train_features.shape[1]
        num_classes = len(torch.unique(train_labels))
        
        # Create simple MLP classifier
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        # Create model, optimizer, loss
        model = SimpleMLP(input_dim, hidden_dim, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Create datasets and dataloaders
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        valid_dataset = torch.utils.data.TensorDataset(valid_features, valid_labels)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
        
        # Train
        best_valid_acc = 0.0
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            valid_acc = correct / total if total > 0 else 0.0
            
            # Save best model
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_state = deepcopy(model.state_dict())
        
        # Load best model
        if best_state:
            model.load_state_dict(best_state)
        
        # Final evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0
    
    # Check if the model has branch layers
    if not hasattr(model, 'net') or not hasattr(model.net, 'branch_layers'):
        print("Model does not have branch layers. Skipping information estimation.")
        return None
    
    net = model.net
    
    # Results dictionaries
    layerwise_results = {
        'layer_indices': [],
        'excitatory_accuracy': [],
        'inhibitory_accuracy': [],
        'combined_accuracy': []
    }
    
    branchwise_results = {
        'branch_indices': [],
        'layer_indices': [],
        'excitatory_accuracy': [],
        'inhibitory_accuracy': [],
        'combined_accuracy': []
    }
    
    print("\n" + "-"*60)
    print("COLLECTING BRANCH ACTIVATIONS")
    print("-"*60)
    
    # Collect all inputs and branch activations
    train_inputs_list = []
    train_labels_list = []
    
    valid_inputs_list = []
    valid_labels_list = []
    
    # Hook to store branch inputs
    branch_inputs = {}
    
    def hook_fn(layer_idx, input, output):
        if layer_idx not in branch_inputs:
            branch_inputs[layer_idx] = {
                'excitatory_inputs': [],
                'inhibitory_inputs': []
            }
        
        # Store excitatory input if it exists
        if hasattr(net.branch_layers[layer_idx], 'excitatory_input'):
            branch_inputs[layer_idx]['excitatory_inputs'].append(
                net.branch_layers[layer_idx].excitatory_input.detach().cpu()
            )
        
        # Store inhibitory input if it exists
        if hasattr(net.branch_layers[layer_idx], 'inhibitory_input'):
            branch_inputs[layer_idx]['inhibitory_inputs'].append(
                net.branch_layers[layer_idx].inhibitory_input.detach().cpu()
            )
        
        return output
    
    # Register hooks for each branch layer
    hooks = []
    for layer_idx, layer in enumerate(net.branch_layers):
        hook = layer.register_forward_hook(lambda m, i, o, idx=layer_idx: hook_fn(idx, i, o))
        hooks.append(hook)
    
    # Collect data from training set
    print("Collecting activations from training set...")
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        train_inputs_list.append(inputs.detach().cpu())
        train_labels_list.append(labels.detach().cpu())
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = model(inputs)
    
    # Collect data from validation set
    print("Collecting activations from validation set...")
    for inputs, labels in tqdm(valid_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        valid_inputs_list.append(inputs.detach().cpu())
        valid_labels_list.append(labels.detach().cpu())
        # Forward pass to trigger hooks
        with torch.no_grad():
            _ = model(inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate collected data
    train_inputs = torch.cat(train_inputs_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)
    
    valid_inputs = torch.cat(valid_inputs_list, dim=0)
    valid_labels = torch.cat(valid_labels_list, dim=0)
    
    # Process each layer
    for layer_idx in sorted(branch_inputs.keys()):
        print(f"\n" + "-"*60)
        print(f"ANALYZING LAYER {layer_idx}")
        print("-"*60)
        
        layer = net.branch_layers[layer_idx]
        layer_info = branch_inputs[layer_idx]
        
        # Concatenate collected activations
        if layer_info['excitatory_inputs']:
            exc_inputs_train = torch.cat(layer_info['excitatory_inputs'][:len(train_loader)], dim=0)
            exc_inputs_valid = torch.cat(layer_info['excitatory_inputs'][len(train_loader):], dim=0)
        else:
            print(f"  No excitatory inputs for layer {layer_idx}")
            exc_inputs_train = None
            exc_inputs_valid = None
        
        if layer_info['inhibitory_inputs']:
            inh_inputs_train = torch.cat(layer_info['inhibitory_inputs'][:len(train_loader)], dim=0)
            inh_inputs_valid = torch.cat(layer_info['inhibitory_inputs'][len(train_loader):], dim=0)
        else:
            print(f"  No inhibitory inputs for layer {layer_idx}")
            inh_inputs_train = None
            inh_inputs_valid = None
        
        # If no inputs, skip this layer
        if exc_inputs_train is None and inh_inputs_train is None:
            print(f"  Layer {layer_idx} has no inputs. Skipping.")
            continue
        
        print(f"Layer {layer_idx} - Processing...")
        
        # Determine number of branches in this layer
        n_branches = 0
        if hasattr(layer, 'branch_excitation'):
            n_branches = layer.branch_excitation.pre_w.shape[0] if hasattr(layer.branch_excitation, 'pre_w') else 0
        elif hasattr(layer, 'branch_inhibition'):
            n_branches = layer.branch_inhibition.pre_w.shape[0] if hasattr(layer.branch_inhibition, 'pre_w') else 0
        
        print(f"  Layer {layer_idx} has {n_branches} branches")
        
        # 1. LAYERWISE ANALYSIS
        # ------------------------------
        print(f"  Performing layerwise analysis...")
        
        # Extract weighted inputs for each branch
        exc_branch_features_train = []
        exc_branch_features_valid = []
        inh_branch_features_train = []
        inh_branch_features_valid = []
        combined_branch_features_train = []
        combined_branch_features_valid = []
        
        for branch_idx in range(n_branches):
            # Get branch weights
            if hasattr(layer, 'branch_excitation') and exc_inputs_train is not None:
                exc_w = layer.branch_excitation.pre_w[branch_idx].cpu()
                exc_weighted_train = (exc_inputs_train @ exc_w).unsqueeze(1)  # [n_samples, 1]
                exc_weighted_valid = (exc_inputs_valid @ exc_w).unsqueeze(1)  # [n_samples, 1]
                
                exc_branch_features_train.append(exc_weighted_train)
                exc_branch_features_valid.append(exc_weighted_valid)
            else:
                exc_weighted_train = None
                exc_weighted_valid = None
            
            if hasattr(layer, 'branch_inhibition') and inh_inputs_train is not None:
                inh_w = layer.branch_inhibition.pre_w[branch_idx].cpu()
                inh_weighted_train = (inh_inputs_train @ inh_w).unsqueeze(1)  # [n_samples, 1]
                inh_weighted_valid = (inh_inputs_valid @ inh_w).unsqueeze(1)  # [n_samples, 1]
                
                inh_branch_features_train.append(inh_weighted_train)
                inh_branch_features_valid.append(inh_weighted_valid)
            else:
                inh_weighted_train = None
                inh_weighted_valid = None
            
            # Compute combined features (shunting formula)
            if exc_weighted_train is not None and inh_weighted_train is not None:
                # w_i * x_i / (w_i * x_i + inh_j * x_j + 1)
                combined_train = exc_weighted_train / (exc_weighted_train + inh_weighted_train + 1.0)
                combined_valid = exc_weighted_valid / (exc_weighted_valid + inh_weighted_valid + 1.0)
                
                combined_branch_features_train.append(combined_train)
                combined_branch_features_valid.append(combined_valid)
            
            # BRANCHWISE ANALYSIS
            # ------------------------------
            print(f"  Branch {branch_idx}: Performing branchwise analysis...")
            
            # For branch-wise analysis, we use the actual input features
            exc_acc = 0.0
            inh_acc = 0.0
            combined_acc = 0.0
            
            if exc_inputs_train is not None:
                print(f"    Training excitatory branch classifier...")
                exc_acc = train_mlp_classifier(
                    exc_inputs_train, train_labels,
                    exc_inputs_valid, valid_labels
                )
                print(f"    Excitatory branch accuracy: {exc_acc:.4f}")
            
            if inh_inputs_train is not None:
                print(f"    Training inhibitory branch classifier...")
                inh_acc = train_mlp_classifier(
                    inh_inputs_train, train_labels,
                    inh_inputs_valid, valid_labels
                )
                print(f"    Inhibitory branch accuracy: {inh_acc:.4f}")
            
            if exc_inputs_train is not None and inh_inputs_train is not None:
                # Concatenate excitatory and inhibitory inputs
                combined_inputs_train = torch.cat([exc_inputs_train, inh_inputs_train], dim=1)
                combined_inputs_valid = torch.cat([exc_inputs_valid, inh_inputs_valid], dim=1)
                
                print(f"    Training combined branch classifier...")
                combined_acc = train_mlp_classifier(
                    combined_inputs_train, train_labels,
                    combined_inputs_valid, valid_labels
                )
                print(f"    Combined branch accuracy: {combined_acc:.4f}")
            
            # Store branchwise results
            branchwise_results['branch_indices'].append(f"{layer_idx}-{branch_idx}")
            branchwise_results['layer_indices'].append(layer_idx)
            branchwise_results['excitatory_accuracy'].append(exc_acc)
            branchwise_results['inhibitory_accuracy'].append(inh_acc)
            branchwise_results['combined_accuracy'].append(combined_acc)
        
        # Continue with layerwise analysis
        if exc_branch_features_train:
            exc_features_train = torch.cat(exc_branch_features_train, dim=1)  # [n_samples, n_branches]
            exc_features_valid = torch.cat(exc_branch_features_valid, dim=1)
            
            print(f"  Training excitatory layer classifier...")
            exc_layer_acc = train_mlp_classifier(
                exc_features_train, train_labels, 
                exc_features_valid, valid_labels
            )
            print(f"  Excitatory layer accuracy: {exc_layer_acc:.4f}")
        else:
            exc_layer_acc = 0.0
        
        if inh_branch_features_train:
            inh_features_train = torch.cat(inh_branch_features_train, dim=1)  # [n_samples, n_branches]
            inh_features_valid = torch.cat(inh_branch_features_valid, dim=1)
            
            print(f"  Training inhibitory layer classifier...")
            inh_layer_acc = train_mlp_classifier(
                inh_features_train, train_labels,
                inh_features_valid, valid_labels
            )
            print(f"  Inhibitory layer accuracy: {inh_layer_acc:.4f}")
        else:
            inh_layer_acc = 0.0
        
        if combined_branch_features_train:
            combined_features_train = torch.cat(combined_branch_features_train, dim=1)  # [n_samples, n_branches]
            combined_features_valid = torch.cat(combined_branch_features_valid, dim=1)
            
            print(f"  Training combined layer classifier...")
            combined_layer_acc = train_mlp_classifier(
                combined_features_train, train_labels,
                combined_features_valid, valid_labels
            )
            print(f"  Combined layer accuracy: {combined_layer_acc:.4f}")
        else:
            combined_layer_acc = 0.0
        
        # Store layerwise results
        layerwise_results['layer_indices'].append(layer_idx)
        layerwise_results['excitatory_accuracy'].append(exc_layer_acc)
        layerwise_results['inhibitory_accuracy'].append(inh_layer_acc)
        layerwise_results['combined_accuracy'].append(combined_layer_acc)
    
    # Save results
    print("\nSaving information estimation results...")
    layerwise_path = os.path.join(info_dir, 'layerwise_information.json')
    save_dict(layerwise_results, os.path.dirname(layerwise_path), os.path.basename(layerwise_path))
    print(f"✓ Saved layerwise information results to {layerwise_path}")
    
    branchwise_path = os.path.join(info_dir, 'branchwise_information.json')
    save_dict(branchwise_results, os.path.dirname(branchwise_path), os.path.basename(branchwise_path))
    print(f"✓ Saved branchwise information results to {branchwise_path}")
    
    # Create layerwise plot
    print("Generating layerwise information plots...")
    plt.figure(figsize=(10, 6))
    x = np.arange(len(layerwise_results['layer_indices']))
    width = 0.25
    
    plt.bar(x - width, layerwise_results['excitatory_accuracy'], width, label='Excitatory')
    plt.bar(x, layerwise_results['inhibitory_accuracy'], width, label='Inhibitory')
    plt.bar(x + width, layerwise_results['combined_accuracy'], width, label='Combined')
    
    plt.xlabel('Layer Index')
    plt.ylabel('Information Content (Classifier Accuracy)')
    plt.title('Layer-wise Information Content')
    plt.xticks(x, layerwise_results['layer_indices'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(info_dir, 'layerwise_information_plot.png'), dpi=300)
    plt.close()
    
    # Create branchwise plot, grouped by layer
    if branchwise_results['branch_indices']:
        print("Generating branchwise information plots...")
        
        # Group branches by layer
        unique_layers = sorted(set(branchwise_results['layer_indices']))
        
        for layer_idx in unique_layers:
            # Filter branches for this layer
            layer_mask = [i for i, idx in enumerate(branchwise_results['layer_indices']) if idx == layer_idx]
            if not layer_mask:
                continue
            
            # Get branch indices, accuracies for this layer
            branches = [branchwise_results['branch_indices'][i].split('-')[1] for i in layer_mask]
            exc_accs = [branchwise_results['excitatory_accuracy'][i] for i in layer_mask]
            inh_accs = [branchwise_results['inhibitory_accuracy'][i] for i in layer_mask]
            combined_accs = [branchwise_results['combined_accuracy'][i] for i in layer_mask]
            
            plt.figure(figsize=(10, 6))
            x = np.arange(len(branches))
            width = 0.25
            
            plt.bar(x - width, exc_accs, width, label='Excitatory')
            plt.bar(x, inh_accs, width, label='Inhibitory')
            plt.bar(x + width, combined_accs, width, label='Combined')
            
            plt.xlabel('Branch Index')
            plt.ylabel('Information Content (Classifier Accuracy)')
            plt.title(f'Branch-wise Information Content - Layer {layer_idx}')
            plt.xticks(x, branches)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(info_dir, f'branchwise_information_layer{layer_idx}_plot.png'), dpi=300)
            plt.close()
    
    print("\n" + "="*60)
    print("INFORMATION ESTIMATION ANALYSIS COMPLETED")
    print("="*60 + "\n")
    
    return {
        'layerwise_information': layerwise_results,
        'branchwise_information': branchwise_results
    }

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
    
    # Set default values for new parameters if not present
    if not hasattr(outputs_cfg, 'enable_ablation'):
        outputs_cfg.enable_ablation = True
    
    if not hasattr(outputs_cfg, 'ablation_batch_size'):
        outputs_cfg.ablation_batch_size = 64
        
    if not hasattr(outputs_cfg, 'enable_info_estimation'):
        outputs_cfg.enable_info_estimation = True
        
    if not hasattr(outputs_cfg, 'info_estimation_epochs'):
        outputs_cfg.info_estimation_epochs = 20
    
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
        
        if train_input_net:
            # First, train just the input network
            logger.info("Training input network...")
            
            # Get training options for the heads
            train_heads_independently = input_net_training.get('train_heads_independently', False)
            
            if train_heads_independently:
                logger.info("Training excitatory and inhibitory heads independently")
                
                # Create directories for model saving
                excit_head_dir = os.path.join(run_save_path, "excit_head_model")
                os.makedirs(excit_head_dir, exist_ok=True)
                
                # PHASE 1: Train the excitatory head for classification
                logger.info("Phase 1: Training excitatory head for classification...")
                
                # Get input network parameters from the configuration
                input_net_dict = model_cfg.network.parameters.get('input_net_dict', {})
                hidden_dims = input_net_dict.get('hidden_dims', [128, 64])
                net_output_dim = input_net_dict.get('output_dim', 64)  # Feature dimension
                
                # For classification, the final output dimension must be the number of classes
                final_output_dim = 10  # For MNIST
                
                # Create the feature extractor part of the MLP (without classification head)
                excit_feature_extractor = SimpleMLP(input_dim, hidden_dims, net_output_dim)
                
                # Train the MLP with a classification head
                logger.info(f"Training excitatory MLP with classification head using hidden dims: {hidden_dims}")
                excit_feature_extractor, (excit_train_acc, excit_valid_acc, excit_test_acc) = train_mlp_with_classification_head(
                    excit_feature_extractor,
                    num_features=net_output_dim,
                    num_classes=final_output_dim,
                    train_ds=train_ds,
                    valid_ds=valid_ds,
                    test_ds=test_ds,
                    epochs=input_net_epochs,
                    batch_size=train_cfg.batch_size,
                    lr=train_cfg.lr,
                    run_save_path=excit_head_dir
                )
                
                # Save the trained feature extractor
                excit_save_path = save_model(excit_feature_extractor, run_save_path, "excit_feature_extractor.pt")
                
                logger.info(f"Excitatory head train acc={excit_train_acc:.3f}, valid acc={excit_valid_acc:.3f}, test acc={excit_test_acc:.3f}")
                
                if wandb_cfg.use_wandb:
                    wandb.log({
                        "excit_head_train_acc": excit_train_acc,
                        "excit_head_valid_acc": excit_valid_acc,
                        "excit_head_test_acc": excit_test_acc
                    })
                
                # Create directory for inhibitory head
                inhib_head_dir = os.path.join(run_save_path, "inhib_head_model")
                os.makedirs(inhib_head_dir, exist_ok=True)
                
                # PHASE 2: Train the inhibitory head with classification head
                logger.info("Phase 2: Training inhibitory head with classification head...")
                
                # Create the feature extractor part of the MLP (without classification head)
                inhib_feature_extractor = SimpleMLP(input_dim, hidden_dims, net_output_dim)
                
                # Train the MLP with a classification head
                logger.info(f"Training inhibitory MLP with classification head using hidden dims: {hidden_dims}")
                inhib_feature_extractor, (inhib_train_acc, inhib_valid_acc, inhib_test_acc) = train_mlp_with_classification_head(
                    inhib_feature_extractor,
                    num_features=net_output_dim,
                    num_classes=final_output_dim,
                    train_ds=train_ds,
                    valid_ds=valid_ds,
                    test_ds=test_ds,
                    epochs=input_net_epochs,
                    batch_size=train_cfg.batch_size,
                    lr=train_cfg.lr,
                    run_save_path=inhib_head_dir
                )
                
                # Save the trained feature extractor
                inhib_save_path = save_model(inhib_feature_extractor, run_save_path, "inhib_feature_extractor.pt")
                
                logger.info(f"Inhibitory head train acc={inhib_train_acc:.3f}, valid acc={inhib_valid_acc:.3f}, test acc={inhib_test_acc:.3f}")
                
                if wandb_cfg.use_wandb:
                    wandb.log({
                        "inhib_head_train_acc": inhib_train_acc,
                        "inhib_head_valid_acc": inhib_valid_acc,
                        "inhib_head_test_acc": inhib_test_acc
                    })
                
                # PHASE 3: Create the full model with both trained feature extractors
                logger.info("Phase 3: Creating full model with both trained feature extractors...")
                
                # Initialize the full model
                model = initialize_model(model_cfg)
                
                # Create a new input network with sigmoid activation
                logger.info("Creating input network with sigmoid activation for dendrinet input")
                new_input_net = SigmoidInputNetTransform(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=net_output_dim,
                    network_type=input_net_dict.get("type", "MLPInputTransform")
                )
                
                # Transfer weights from trained feature extractors to the new input network
                # This requires a custom function to map the weights from SimpleMLP to InputNetTransform
                logger.info("Transferring weights from trained feature extractors to input network...")
                
                # Custom function to transfer weights from SimpleMLP to InputNetTransform
                def transfer_simple_mlp_to_input_net(simple_mlp, input_net, head_type):
                    """Transfer weights from SimpleMLP to specific head of InputNetTransform"""
                    logger.info(f"Transferring weights to {head_type} head")
                    
                    # Get linear layers from SimpleMLP
                    mlp_layers = []
                    for module in simple_mlp.network:
                        if isinstance(module, torch.nn.Linear):
                            mlp_layers.append(module)
                    
                    # Get shared layers from InputNetTransform
                    shared_layers = []
                    for module in input_net.shared:
                        if isinstance(module, torch.nn.Linear):
                            shared_layers.append(module)
                    
                    # Get head layers
                    target_head = input_net.head_exc if head_type == "excitatory" else input_net.head_inh
                    head_layers = []
                    for module in target_head:
                        if isinstance(module, torch.nn.Linear):
                            head_layers.append(module)
                    
                    # Transfer weights for the shared backbone (only once, typically from excitatory head)
                    if head_type == "excitatory":
                        for i, shared_layer in enumerate(shared_layers):
                            if i < len(mlp_layers) - 1:  # All but the last MLP layer
                                mlp_layer = mlp_layers[i]
                                shared_layer.weight.data = mlp_layer.weight.data.clone()
                                shared_layer.bias.data = mlp_layer.bias.data.clone()
                                logger.info(f"Transferred weights for shared layer {i}")
                    
                    # Transfer weights for the head
                    if len(head_layers) > 0 and len(mlp_layers) > 0:
                        # Last MLP layer goes to the head
                        last_mlp_layer = mlp_layers[-1]
                        head_layers[0].weight.data = last_mlp_layer.weight.data.clone()
                        head_layers[0].bias.data = last_mlp_layer.bias.data.clone()
                        logger.info(f"Transferred weights to {head_type} head")
                
                # Transfer weights from both trained feature extractors
                transfer_simple_mlp_to_input_net(excit_feature_extractor, new_input_net, "excitatory")
                transfer_simple_mlp_to_input_net(inhib_feature_extractor, new_input_net, "inhibitory")
                
                # Set the input network in the full model
                if hasattr(model, 'net'):
                    model.net.input_net = new_input_net
                elif hasattr(model, 'network'):
                    model.network.input_net = new_input_net
                
                # Save the feature extractor and input network state before training dendrinet
                feature_extractor_path = save_model(excit_feature_extractor, run_save_path, "feature_extractor_final.pt")
                input_net_path = save_model(new_input_net, run_save_path, "input_net_with_sigmoid.pt")
                
                # Now train the dendrinet with the transferred weights
                logger.info("Training dendrinet with the transferred feature extractor weights...")
                
                # Create optimizer for full model
                optimizer = CustomWeightDecayOptimizer(
                    model=model, 
                    optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr), 
                    weight_decay=train_cfg.weight_decay_rate,
                )
                
                # Create trainer for full model
                trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)
                
                # Train the full model
                logger.info("Training dendritic network...")
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
                
                if wandb.run is not None:
                    wandb.log({
                        "full_model_train_acc": train_acc,
                        "full_model_valid_acc": valid_acc,
                        "full_model_test_acc": test_acc
                    })
            
            else:
                # Case where we train both heads together with a single classification head
                logger.info("Training a single MLP with classification head for both excitatory and inhibitory heads")
                
                # Create directory for MLP model
                mlp_model_dir = os.path.join(run_save_path, "mlp_model")
                os.makedirs(mlp_model_dir, exist_ok=True)
                logger.info(f"Created directory for MLP model: {mlp_model_dir}")
                
                # Get input network parameters
                input_net_dict = model_cfg.network.parameters.get('input_net_dict', {})
                hidden_dims = input_net_dict.get('hidden_dims', [128, 64])
                net_output_dim = input_net_dict.get('output_dim', 64)
                
                # For classification, the final output dimension must be the number of classes
                final_output_dim = 10  # For MNIST
                
                # Create the feature extractor part of the MLP (without classification head)
                feature_extractor = SimpleMLP(input_dim, hidden_dims, net_output_dim)
                
                # Train the MLP with a classification head
                logger.info(f"Training MLP with classification head using hidden dims: {hidden_dims}")
                feature_extractor, (train_acc, valid_acc, test_acc) = train_mlp_with_classification_head(
                    feature_extractor,
                    num_features=net_output_dim,
                    num_classes=final_output_dim,
                    train_ds=train_ds,
                    valid_ds=valid_ds,
                    test_ds=test_ds,
                    epochs=input_net_epochs,
                    batch_size=train_cfg.batch_size,
                    lr=train_cfg.lr,
                    run_save_path=mlp_model_dir
                )
                
                # Now initialize the full model
                logger.info("Initializing full model with trained feature extractor...")
                model = initialize_model(model_cfg)
                
                # Create a new input network with sigmoid activation
                logger.info("Creating input network with sigmoid activation for dendrinet input")
                new_input_net = SigmoidInputNetTransform(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=net_output_dim,
                    network_type=input_net_dict.get("type", "MLPInputTransform")
                )
                
                # Custom function to transfer weights from SimpleMLP to InputNetTransform
                def transfer_simple_mlp_to_input_net(simple_mlp, input_net):
                    """Transfer weights from SimpleMLP to both heads of InputNetTransform"""
                    logger.info("Transferring weights to both heads")
                    
                    # Get linear layers from SimpleMLP
                    mlp_layers = []
                    for module in simple_mlp.network:
                        if isinstance(module, torch.nn.Linear):
                            mlp_layers.append(module)
                    
                    # Get shared layers from InputNetTransform
                    shared_layers = []
                    for module in input_net.shared:
                        if isinstance(module, torch.nn.Linear):
                            shared_layers.append(module)
                    
                    # Get head layers
                    exc_layers = []
                    for module in input_net.head_exc:
                        if isinstance(module, torch.nn.Linear):
                            exc_layers.append(module)
                    
                    inh_layers = []
                    for module in input_net.head_inh:
                        if isinstance(module, torch.nn.Linear):
                            inh_layers.append(module)
                    
                    # Transfer weights for shared layers
                    for i, shared_layer in enumerate(shared_layers):
                        if i < len(mlp_layers) - 1:  # All but the last MLP layer
                            mlp_layer = mlp_layers[i]
                            shared_layer.weight.data = mlp_layer.weight.data.clone()
                            shared_layer.bias.data = mlp_layer.bias.data.clone()
                            logger.info(f"Transferred weights for shared layer {i}")
                    
                    # Transfer weights for both heads from the last MLP layer
                    if len(mlp_layers) > 0:
                        last_mlp_layer = mlp_layers[-1]
                        
                        # Transfer to excitatory head
                        if len(exc_layers) > 0:
                            exc_layers[0].weight.data = last_mlp_layer.weight.data.clone()
                            exc_layers[0].bias.data = last_mlp_layer.bias.data.clone()
                            logger.info("Transferred weights to excitatory head")
                        
                        # Transfer to inhibitory head
                        if len(inh_layers) > 0:
                            inh_layers[0].weight.data = last_mlp_layer.weight.data.clone()
                            inh_layers[0].bias.data = last_mlp_layer.bias.data.clone()
                            logger.info("Transferred weights to inhibitory head")
                
                # Transfer weights from the trained feature extractor to both heads
                transfer_simple_mlp_to_input_net(feature_extractor, new_input_net)
                
                # Set the input network in the full model
                if hasattr(model, 'net'):
                    model.net.input_net = new_input_net
                elif hasattr(model, 'network'):
                    model.network.input_net = new_input_net
                
                # Save the feature extractor and input network state before training dendrinet
                feature_extractor_path = save_model(feature_extractor, run_save_path, "feature_extractor_final.pt")
                input_net_path = save_model(new_input_net, run_save_path, "input_net_with_sigmoid.pt")
                
                # Now train the dendrinet with the transferred weights
                logger.info("Training dendrinet with the transferred feature extractor weights...")
                
                # Create optimizer for full model
                optimizer = CustomWeightDecayOptimizer(
                    model=model, 
                    optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr), 
                    weight_decay=train_cfg.weight_decay_rate,
                )
                
                # Create trainer for full model
                trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)
                
                # Train the full model
                logger.info("Training dendritic network...")
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
                
                if wandb.run is not None:
                    wandb.log({
                        "full_model_train_acc": train_acc,
                        "full_model_valid_acc": valid_acc,
                        "full_model_test_acc": test_acc
                    })
    
    else:
        # Standard training (joint or other learning strategies)
        print("\n" + "="*80)
        print(f"STANDARD TRAINING: {learning_strategy}")
        print("="*80)
        logger.info(f"Using standard training with learning strategy: {learning_strategy}")
        
        # Initialize model
        print("Initializing model...")
        model = initialize_model(model_cfg)
        
        # Create optimizer
        print("Creating optimizer...")
        optimizer = CustomWeightDecayOptimizer(
            model=model, 
            optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr), 
            weight_decay=train_cfg.weight_decay_rate,
        )
        
        # Create trainer
        print("Creating trainer...")
        trainer = get_trainer(learning_strategy, optimizer, suppress_prints=False, print_every=10)
        
        # Determine visualization epochs
        has_visualize_epochs = hasattr(visualization_cfg, 'visualize_epochs') and visualization_cfg.visualize_epochs
        visualize_at_specific_epochs = has_visualize_epochs and len(visualization_cfg.visualize_epochs) > 0
        
        # Get device for training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Define shape for visualization if needed
        shape = None
        if task_cfg.dataset in ["mnist", "mnist_switch2", "mnist_switch10"]:
            shape = (28, 28)
        elif task_cfg.dataset == "cifar10":
            shape = (32, 32, 3)
        elif task_cfg.dataset == "mnist_modulo10":
            shape = (28, 28, 2)
            
        print("\nStarting training...")
        print(f"Training parameters: epochs={train_cfg.epochs}, batch_size={train_cfg.batch_size}, learning_rate={train_cfg.lr}")
        print(f"Dataset: {task_cfg.dataset}, Train size: {len(train_ds)}, Valid size: {len(valid_ds)}, Test size: {len(test_ds)}")
        print("-"*80)
        
        # Initialize training tracking variables
        valid_losses = []
        train_losses = []
        best_loss = float('inf')
        best_state_dict = deepcopy(model.state_dict())
        
        # Train model one epoch at a time
        for epoch in range(1, train_cfg.epochs + 1):
            # Train for one epoch
            if learning_strategy in ["two_step", "two_step_kl"]:
                if epoch < train_cfg.pretrain_epochs:
                    logger.info(f"Pretrain epoch {epoch}/{train_cfg.pretrain_epochs}")
                else:
                    logger.info(f"Main train epoch {epoch-train_cfg.pretrain_epochs}/{train_cfg.epochs-train_cfg.pretrain_epochs}")
                
                results = trainer.train(
                    model=model,
                    train_data=train_ds,
                    valid_data=valid_ds,
                    pretrain_epochs=1 if epoch < train_cfg.pretrain_epochs else 0,
                    maintrain_epochs=0 if epoch < train_cfg.pretrain_epochs else 1,
                    batch_size=train_cfg.batch_size,
                    grad_clip_value=train_cfg.grad_clip_value,
                    shuffle=True,
                    load_best_state_dict=False,
                    plot_losses=False,
                    save_path=run_save_path
                )
            elif learning_strategy == "freeze_layers":
                # For freeze_layers, we'll use a slightly different approach
                # This is a simplified version, assuming one epoch per layer at a time
                layer_idx = epoch % model.net.num_layers if hasattr(model.net, 'num_layers') else 0
                logger.info(f"Training layer {layer_idx} for epoch {epoch}/{train_cfg.epochs}")
                
                # Freeze all layers except the current one
                if hasattr(model.net, 'freeze_all_layers_except'):
                    model.net.freeze_all_layers_except(layer_idx)
                
                results = trainer.train(
                    model=model,
                    train_data=train_ds,
                    valid_data=valid_ds,
                    epochs=1,
                    batch_size=train_cfg.batch_size,
                    grad_clip_value=train_cfg.grad_clip_value,
                    shuffle=True,
                    load_best_state_dict=False,
                    plot_losses=False,
                    save_path=run_save_path
                )
                
                # Unfreeze all layers after training
                if hasattr(model.net, 'unfreeze_all_layers'):
                    model.net.unfreeze_all_layers()
            else:
                results = trainer.train(
                    model=model,
                    train_data=train_ds,
                    valid_data=valid_ds,
                    epochs=1,
                    batch_size=train_cfg.batch_size,
                    grad_clip_value=train_cfg.grad_clip_value,
                    shuffle=True,
                    load_best_state_dict=False,
                    plot_losses=False,
                    save_path=run_save_path
                )
            
            # Track losses and best model
            if "train_losses" in results:
                train_losses.extend(results["train_losses"])
            elif "train losses" in results:
                train_losses.extend(results["train losses"])
                
            if "valid_losses" in results:
                valid_losses.extend(results["valid_losses"])
            elif "valid losses" in results:
                valid_losses.extend(results["valid losses"])
                
            current_loss = valid_losses[-1] if valid_losses else float('inf')
            
            if "best_loss" in results and results["best_loss"] < best_loss:
                best_loss = results["best_loss"]
                if "best_state_dict" in results:
                    best_state_dict = results["best_state_dict"]
                elif "best state dict" in results:
                    best_state_dict = results["best state dict"]
            
            # Check if this epoch is in the visualization list
            if epoch in visualization_cfg.visualize_epochs:
                logger.info(f"Epoch {epoch}/{train_cfg.epochs} complete: train_loss={train_losses[-1]:.4f}, valid_loss={valid_losses[-1]:.4f}")
                
                # Evaluate model accuracy at this epoch
                train_acc, valid_acc, test_acc = evaluate_accuracy(
                    model, train_ds, valid_ds, test_ds, 
                    save_path=os.path.join(run_save_path, 'performance'), 
                    filename=f"epoch{epoch}",
                )
                logger.info(f"  Accuracy: train={train_acc:.4f}, valid={valid_acc:.4f}, test={test_acc:.4f}")
                
                if wandb.run is not None:
                    wandb.log({
                        f"epoch_{epoch}_train_acc": train_acc,
                        f"epoch_{epoch}_valid_acc": valid_acc,
                        f"epoch_{epoch}_test_acc": test_acc
                    })
                
                # Create visualizations if enabled
                if visualization_cfg.enabled:
                    # Check if input_mode is 0 before attempting visualizations that might cause errors
                    input_mode = model_cfg.network.parameters.get('input_mode', 0)
                    
                    logger.info(f"Creating visualizations after training...")
                    # Create data dictionary for plot manager
                    data_dict = {"train": train_ds, "valid": valid_ds, "test": test_ds}
                    
                    # Initialize core plot manager with correct parameters
                    # The original PlotManager requires (model, data, args, device)
                    class PlotManagerArgs:
                        def __init__(self, save_path, file_name, visualization_cfg=None):
                            self.save_path = save_path
                            self.file_name = file_name
                            self.visualization = visualization_cfg
                            
                    # Create args object for PlotManager
                    plot_args = PlotManagerArgs(
                        save_path=os.path.join(run_save_path, 'visualizations'),
                        file_name=f"final",
                        visualization_cfg=visualization_cfg
                    )
                    
                    plot_manager = CorePlotManager(
                        model=model,
                        data=data_dict,
                        args=plot_args,
                        device=device
                    )
                    
                    if input_mode == 0:
                        # Run all plots except branch_info for input_mode=0
                        safe_plots = [p for p in visualization_cfg.plots if p != 'branch_info']
                        for plot_name in safe_plots:
                            logger.info(f"  Generating {plot_name} plot...")
                            try:
                                plot_manager.run_plot(plot_name)
                            except Exception as e:
                                logger.error(f"Error generating {plot_name} plot: {str(e)}")
                        
                        if 'branch_info' in visualization_cfg.plots:
                            logger.info(f"  Skipping branch_info plot for input_mode=0 to avoid errors")
                    else:
                        # For other input modes, run all visualizations as normal
                        for plot_name in visualization_cfg.plots:
                            logger.info(f"  Generating {plot_name} plot...")
                            try:
                                plot_manager.run_plot(plot_name)
                            except Exception as e:
                                logger.error(f"Error generating {plot_name} plot: {str(e)}")
            
            # Load best model at the end of training
            model.load_state_dict(best_state_dict)
            
            # Plot the final loss curves
            plot_NLL_loss_curves(train_losses, valid_losses, train_cfg.epochs, run_save_path)
            
            # Final evaluation
            logger.info("Evaluating final model performance...")
            train_acc, valid_acc, test_acc = evaluate_accuracy(
                model, train_ds, valid_ds, test_ds, 
                save_path=os.path.join(run_save_path, 'performance'), 
                filename="final",
            )
            logger.info(f"Final accuracy: train={train_acc:.4f}, valid={valid_acc:.4f}, test={test_acc:.4f}")
            
            if wandb.run is not None:
                wandb.log({
                    "final_train_acc": train_acc,
                    "final_valid_acc": valid_acc,
                    "final_test_acc": test_acc
                })
    
    # Save configuration
    save_dict(asdict(config), run_save_path, "config.json")
    
    # Close W&B
    if wandb_cfg.use_wandb:
        wandb.finish()
    
    # Run information estimation analysis if enabled
    if getattr(outputs_cfg, 'enable_info_estimation', True):
        info_results = run_information_estimation(
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            save_path=run_save_path,
            device=device,
            batch_size=getattr(outputs_cfg, 'ablation_batch_size', 64),
            epochs=getattr(outputs_cfg, 'info_estimation_epochs', 20)
        )
    else:
        print("\n" + "="*80)
        print("INFORMATION ESTIMATION ANALYSIS SKIPPED (disabled in config)")
        print("="*80 + "\n")
        logger.info("Information estimation analysis skipped (disabled in config)")
    
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
    
    # Pre-trained options
    parser.add_argument("--pretrained_source", choices=["file", "pytorch", "none"],
                       help="Source of pre-trained weights")
    parser.add_argument("--weights_path", 
                       help="Path to weights file if pretrained_source is 'file'")
    parser.add_argument("--pytorch_model", 
                       help="Name of PyTorch model if pretrained_source is 'pytorch'")
    
    # Information estimation and ablation options
    parser.add_argument("--enable_ablation", type=lambda x: (str(x).lower() == 'true'), 
                       help="Enable or disable ablation analysis")
    parser.add_argument("--enable_info_estimation", type=lambda x: (str(x).lower() == 'true'), 
                       help="Enable or disable information estimation")
    parser.add_argument("--info_estimation_epochs", type=int, 
                       help="Number of epochs for information estimation MLP training")
    parser.add_argument("--ablation_batch_size", type=int, 
                       help="Batch size for ablation and information estimation")
    
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
    
    # Update pretrained options if provided
    if args.pretrained_source:
        if 'pretrained_options' not in config.train.input_net_training:
            config.train.input_net_training['pretrained_options'] = {}
        
        config.train.input_net_training['pretrained_options']['source'] = args.pretrained_source
        
        if args.weights_path:
            config.train.input_net_training['pretrained_options']['weights_path'] = args.weights_path
        
        if args.pytorch_model:
            config.train.input_net_training['pretrained_options']['pytorch_model'] = args.pytorch_model
    
    # Update ablation and information estimation options
    if args.enable_ablation is not None:
        if not hasattr(config, 'outputs'):
            config.outputs = type('obj', (object,), {})
        config.outputs.enable_ablation = args.enable_ablation
        
    if args.enable_info_estimation is not None:
        if not hasattr(config, 'outputs'):
            config.outputs = type('obj', (object,), {})
        config.outputs.enable_info_estimation = args.enable_info_estimation
        
    if args.info_estimation_epochs is not None:
        if not hasattr(config, 'outputs'):
            config.outputs = type('obj', (object,), {})
        config.outputs.info_estimation_epochs = args.info_estimation_epochs
        
    if args.ablation_batch_size is not None:
        if not hasattr(config, 'outputs'):
            config.outputs = type('obj', (object,), {})
        config.outputs.ablation_batch_size = args.ablation_batch_size

    main(config_path=args.config)