"""
multistage_train.py
--------------------
Script for training a model in multiple stages:
1. Train an MLP on MNIST and report its performance
2. Use the trained MLP as input to dendrinet with fixed MLP weights
3. Train the entire model (MLP + dendrinet) end-to-end
"""

import os
import argparse
import yaml
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import copy

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.learning_strategies import get_trainer
from dendritic_modeling.models import ProbabilisticClassifier, Classifier
from dendritic_modeling.networks import (
    ExcitationInhibitionNetwork,
    MLPInputTransform,
    MLPExcInhNetwork
)
from dendritic_modeling.config import load_config
from dendritic_modeling.visualization_manager import PlotManager
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import evaluate_accuracy, save_dict


class SimpleMLP(nn.Module):
    """
    A simple MLP implementation for MNIST classification
    """
    def __init__(self, input_dim=784, hidden_dims=[128, 64], output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        in_features = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_features, hdim))
            layers.append(nn.ReLU())
            in_features = hdim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def get_features(self, x):
        """Extract features without classification"""
        return self.feature_extractor(x)


class TwoHeadAdapter(nn.Module):
    """
    Adapter that takes a pre-trained MLP feature extractor and adds two heads
    for excitatory and inhibitory inputs to dendrinet
    """
    def __init__(self, feature_extractor, feature_dim, output_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head_exc = nn.Sequential(nn.Linear(feature_dim, output_dim), nn.Sigmoid())
        self.head_inh = nn.Sequential(nn.Linear(feature_dim, output_dim), nn.Sigmoid())
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear) and m not in self.feature_extractor.modules():
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        excitatory_input = self.head_exc(features)
        inhibitory_input = self.head_inh(features)
        return excitatory_input, inhibitory_input


class MultistageModel(nn.Module):
    """
    Wrapper class that combines MLP feature extractor, two-head adapter, and dendrinet
    with options to freeze parts of the model
    """
    def __init__(self, feature_extractor, adapter, dendrinet, freeze_feature_extractor=False):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.adapter = adapter
        self.dendrinet = dendrinet
        
        if freeze_feature_extractor:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        excitatory_input, inhibitory_input = self.adapter(x)
        return self.dendrinet(excitatory_input, inhibitory_input)
    
    def unfreeze_all(self):
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True


def train_mlp(train_loader, valid_loader, model, device, epochs=20, lr=0.001, log_interval=100):
    """Train a simple MLP model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_model_state = None
    
    logger.info(f"Training MLP for {epochs} epochs")
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % log_interval == 0:
                logger.info(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | '
                           f'Loss: {running_loss/log_interval:.4f} | Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        logger.info(f'Validation | Loss: {val_loss/len(valid_loader):.4f} | Acc: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model, best_accuracy


def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    logger.info(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


def initialize_dendrinet(config, input_dim):
    """Initialize dendrinet model using configuration"""
    model_cfg = config.model
    net_params = model_cfg.network.parameters
    
    # Update input mode to use pre-trained MLP
    net_params['input_mode'] = 2
    
    # Create dendrinet
    net = ExcitationInhibitionNetwork(
        input_dim=input_dim,
        enable_branch_outputs=net_params.pop('enable_branch_outputs', False),
        learning_strategy=net_params.pop('learning_strategy', 'mle'),
        **net_params,
    )
    
    output_dim = net_params['excitatory_layer_sizes'][-1]
    if model_cfg.probabilistic:
        return ProbabilisticClassifier(net, output_dim)
    else:
        return Classifier(net)


def main(config_path):
    config = load_config(config_path)
    model_cfg = config.model
    train_cfg = config.train
    visual_cfg = config.visualization
    wandb_cfg = config.wandb
    outputs_cfg = config.outputs
    
    # Setup directories and logging
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    now = now.replace("_", "").replace("-", "")
    run_save_path = os.path.join(outputs_cfg.dir, f"multistage_{train_cfg.run_name}_{now}")
    os.makedirs(run_save_path, exist_ok=True)
    logger_manager.set_log_file(os.path.join(run_save_path, "dendritic_modeling.log"))
    logger.info(f"Saving results and visualizations to: {run_save_path}")
    
    # WandB setup if enabled
    if getattr(wandb_cfg, "use_wandb", False):
        wandb_dir = Path(run_save_path) / "wandb"
        wandb_run_name = f"multistage_{train_cfg.run_name}"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb_dict = asdict(config)
        wandb.init(
            dir=str(wandb_dir),
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            group=wandb_cfg.group,
            name=wandb_run_name,
            tags=wandb_cfg.tags,
            config=wandb_dict,
        )
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(train_cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(train_cfg.seed)
    
    # Load dataset
    dataset_info = get_unified_datasets(config.task)
    train_dataset, valid_dataset, test_dataset = dataset_info['train'], dataset_info['valid'], dataset_info['test']
    input_dim = dataset_info['input_dim']
    output_dim = dataset_info['output_dim']
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=train_cfg.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_cfg.batch_size, shuffle=False
    )
    
    # Get MLP hidden dimensions from config if available, otherwise use defaults
    mlp_hidden_dims = [128, 64]
    if model_cfg.network.parameters.get('mlp_transform_dict'):
        mlp_hidden_dims = model_cfg.network.parameters.mlp_transform_dict.get('hidden_dims', mlp_hidden_dims)
        mlp_output_dim = model_cfg.network.parameters.mlp_transform_dict.get('output_dim', 64)
    else:
        mlp_output_dim = 64
    
    logger.info(f"Using MLP with hidden dimensions: {mlp_hidden_dims}, output dim: {mlp_output_dim}")
    
    # ------------------------------------------------------------------------
    # Stage 1: Train MLP on MNIST and evaluate
    # ------------------------------------------------------------------------
    logger.info("=== Stage 1: Training standalone MLP ===")
    mlp_model = SimpleMLP(input_dim=input_dim, hidden_dims=mlp_hidden_dims, output_dim=output_dim)
    mlp_model.to(device)
    
    # Train the MLP
    mlp_model, mlp_accuracy = train_mlp(
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=mlp_model,
        device=device,
        epochs=train_cfg.epochs,
        lr=train_cfg.lr
    )
    
    # Evaluate MLP on test set
    test_accuracy = evaluate_model(mlp_model, test_loader, device)
    logger.info(f"MLP Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the MLP model
    mlp_save_path = os.path.join(run_save_path, "mlp_model.pt")
    torch.save(mlp_model.state_dict(), mlp_save_path)
    logger.info(f"MLP model saved to {mlp_save_path}")
    
    if getattr(wandb_cfg, "use_wandb", False):
        wandb.log({"mlp_test_accuracy": test_accuracy})
    
    # ------------------------------------------------------------------------
    # Stage 2: Use trained MLP as fixed input to dendrinet
    # ------------------------------------------------------------------------
    logger.info("\n=== Stage 2: Training dendrinet with fixed MLP ===")
    
    # Create adapter with frozen feature extractor
    adapter = TwoHeadAdapter(
        feature_extractor=mlp_model.feature_extractor,
        feature_dim=mlp_hidden_dims[-1],
        output_dim=mlp_output_dim
    )
    
    # Initialize dendrinet
    dendrinet_model = initialize_dendrinet(config, input_dim=mlp_output_dim)
    dendrinet_model.to(device)
    
    # Create multistage model with frozen feature extractor
    multistage_model = MultistageModel(
        feature_extractor=mlp_model.feature_extractor,
        adapter=adapter,
        dendrinet=dendrinet_model,
        freeze_feature_extractor=True
    )
    multistage_model.to(device)
    
    # Get trainer for dendrinet
    trainer = get_trainer(
        model=multistage_model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        run_save_path=os.path.join(run_save_path, "stage2")
    )
    
    # Train with fixed MLP
    logger.info("Training dendrinet with fixed MLP weights")
    stage2_results = trainer.train()
    
    # Evaluate on test set
    stage2_accuracy = evaluate_model(multistage_model, test_loader, device)
    logger.info(f"Stage 2 (Fixed MLP + Trained Dendrinet) Test Accuracy: {stage2_accuracy:.2f}%")
    
    # Save the stage 2 model
    stage2_save_path = os.path.join(run_save_path, "stage2_model.pt")
    torch.save(multistage_model.state_dict(), stage2_save_path)
    logger.info(f"Stage 2 model saved to {stage2_save_path}")
    
    if getattr(wandb_cfg, "use_wandb", False):
        wandb.log({"stage2_test_accuracy": stage2_accuracy})
    
    # ------------------------------------------------------------------------
    # Stage 3: End-to-end training of MLP + dendrinet
    # ------------------------------------------------------------------------
    logger.info("\n=== Stage 3: End-to-end training of MLP + dendrinet ===")
    
    # Unfreeze all parameters
    multistage_model.unfreeze_all()
    multistage_model.to(device)
    
    # Get new trainer for end-to-end training
    trainer = get_trainer(
        model=multistage_model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        run_save_path=os.path.join(run_save_path, "stage3")
    )
    
    # Train end-to-end
    logger.info("Training full model end-to-end")
    stage3_results = trainer.train()
    
    # Evaluate on test set
    stage3_accuracy = evaluate_model(multistage_model, test_loader, device)
    logger.info(f"Stage 3 (End-to-end) Test Accuracy: {stage3_accuracy:.2f}%")
    
    # Save the stage 3 model
    stage3_save_path = os.path.join(run_save_path, "stage3_model.pt")
    torch.save(multistage_model.state_dict(), stage3_save_path)
    logger.info(f"Stage 3 model saved to {stage3_save_path}")
    
    if getattr(wandb_cfg, "use_wandb", False):
        wandb.log({"stage3_test_accuracy": stage3_accuracy})
    
    # ------------------------------------------------------------------------
    # Print summary of results
    # ------------------------------------------------------------------------
    logger.info("\n=== Summary of Results ===")
    logger.info(f"Stage 1 (MLP only): {test_accuracy:.2f}%")
    logger.info(f"Stage 2 (Fixed MLP + Trained Dendrinet): {stage2_accuracy:.2f}%")
    logger.info(f"Stage 3 (End-to-end): {stage3_accuracy:.2f}%")
    
    # Save summary to file
    summary = {
        "mlp_accuracy": test_accuracy,
        "fixed_mlp_with_dendrinet_accuracy": stage2_accuracy,
        "end_to_end_accuracy": stage3_accuracy
    }
    save_dict(summary, os.path.join(run_save_path, "results_summary.json"))
    
    if getattr(wandb_cfg, "use_wandb", False):
        wandb.log(summary)
        wandb.finish()
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage training of MLP and dendrinet")
    parser.add_argument("config_path", type=str, help="Path to config file")
    args = parser.parse_args()
    
    main(args.config_path) 