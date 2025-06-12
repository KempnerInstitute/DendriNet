"""
Interactive branch analysis for dendritic models.

This module contains the InteractiveBranchAnalyzer class that provides
comprehensive analysis of branch-level information and ablation studies.
"""

import json
import logging
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.datasets import split_MNIST_inputs

logger = logging.getLogger(__name__)


class InteractiveBranchAnalyzer:
    """
    Comprehensive branch analysis for dendritic network models.
    
    This class provides methods for analyzing branch-level information content
    and performing ablation studies to understand the functional role of individual
    branches in dendritic networks.
    
    Args:
        device: Device to run analysis on ('cpu' or 'cuda')
    """
    
    def __init__(self, device="cpu"):
        self.device = device

    def analyze_branch_info(self, model, data, save_path, file_name=None, batch_size=64):
        """
        Analyze information content of individual branches using MLP decoding.
        
        This method trains small MLP classifiers on the activations of individual
        branches to quantify how much task-relevant information each branch contains.
        
        Args:
            model: The trained dendritic network model
            data: Dictionary containing 'train', 'valid', 'test' datasets
            save_path: Directory to save analysis results
            file_name: Optional filename prefix for saved files
            batch_size: Batch size for forward passes
            
        Returns:
            Dictionary containing analysis results
        """
        print("\n" + "=" * 80)
        print("STARTING BRANCH INFORMATION ANALYSIS")
        print("=" * 80)
        
        os.makedirs(save_path, exist_ok=True)
        
        if not hasattr(model, "net"):
            print("ERROR: No 'net' attribute found on model. Skipping branch_info analysis.")
            logging.error("No 'net' attribute found on model. Skipping branch_info analysis.")
            return {}
            
        net = model.net
        if not hasattr(net, "forward_with_branch_outputs"):
            print("ERROR: No forward_with_branch_outputs method found on net. Skipping branch_info analysis.")
            logging.error("No forward_with_branch_outputs method found on net. Skipping branch_info analysis.")
            return {}
            
        print(f"Model has {len(net.branch_layers)} branch layers")
        logging.info(f"Model has {len(net.branch_layers)} branch layers")
        print("Computing branch info using MLP decoding...")
        logging.info("Computing branch info using MLP decoding...")
        
        save_dir = os.path.join(save_path, "branch_info_plots")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")
        
        print("Loading datasets...")
        train_inputs, train_labels = data["train"][:]
        valid_inputs, valid_labels = data["valid"][:]
        test_inputs, test_labels = data["test"][:]
        print(f"Loaded datasets: train={len(train_inputs)}, valid={len(valid_inputs)}, test={len(test_inputs)} samples")
        
        layer_indices = []
        branch_indices = []
        layer_exc_accuracies = []
        layer_inh_accuracies = []
        layer_combined_accuracies = []
        
        use_shunting = getattr(model.net, "use_shunting", True)
        print(f"Using {'shunting' if use_shunting else 'subtraction'} for combining excitatory and inhibitory inputs")
        
        for layer_idx, layer in enumerate(net.branch_layers):
            print(f"\nProcessing layer {layer_idx} for branch_info...")
            logging.info(f"Processing layer {layer_idx} for branch_info...")
            
            if not hasattr(layer, "branch_excitation") and not hasattr(layer, "branch_inhibition"):
                print(f"  Layer {layer_idx} has no branch excitation or inhibition. Skipping.")
                logging.info(f"Layer {layer_idx} has no branch excitation or inhibition. Skipping.")
                continue
                
            n_branches = 0
            if hasattr(layer, "branch_excitation"):
                n_branches = layer.branch_excitation.pre_w.shape[0]
                print(f"  Layer {layer_idx} has branch_excitation with {n_branches} branches")
            elif hasattr(layer, "branch_inhibition"):
                n_branches = layer.branch_inhibition.pre_w.shape[0]
                print(f"  Layer {layer_idx} has branch_inhibition with {n_branches} branches")
                
            if n_branches == 0:
                print(f"  Layer {layer_idx} has no branches. Skipping.")
                logging.info(f"Layer {layer_idx} has no branches. Skipping.")
                continue
                
            print(f"  Layer {layer_idx} has {n_branches} branches")
            logging.info(f"Layer {layer_idx} has {n_branches} branches")
            
            print("  Collecting branch inputs from dataset...")
            exc_branch_inputs_all = []
            inh_branch_inputs_all = []
            
            net.eval()
            net = net.to(self.device)
            
            with torch.no_grad():
                for batch_start in range(0, len(train_inputs), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_inputs))
                    batch = train_inputs[batch_start:batch_end].to(self.device)
                    _ = net(batch)
                    
                    if hasattr(layer, "branch_excitation"):
                        if hasattr(layer, "excitatory_input"):
                            exc_inputs = layer.excitatory_input
                            exc_branch_inputs_all.append(exc_inputs.cpu())
                        else:
                            print("    Warning: layer has branch_excitation but no excitatory_input attribute")
                            
                    if hasattr(layer, "branch_inhibition"):
                        if hasattr(layer, "inhibitory_input"):
                            inh_inputs = layer.inhibitory_input
                            inh_branch_inputs_all.append(inh_inputs.cpu())
                        else:
                            print("    Warning: layer has branch_inhibition but no inhibitory_input attribute")
                            
            if torch.cuda.is_available():
                net = net.cuda()
                
            if exc_branch_inputs_all:
                exc_inputs = torch.cat(exc_branch_inputs_all, dim=0)
                print(f"  Collected excitatory inputs: {exc_inputs.shape}")
            else:
                print("  No excitatory inputs collected - this is expected with input_mode=0")
                print("  Skipping branch information analysis for this layer")
                continue
                
            if inh_branch_inputs_all:
                inh_inputs = torch.cat(inh_branch_inputs_all, dim=0)
                print(f"  Collected inhibitory inputs: {inh_inputs.shape}")
            else:
                inh_inputs = torch.zeros((len(train_inputs), 0))
                print("  No inhibitory inputs collected")
                
            layer_exc_acc = 0.0
            layer_inh_acc = 0.0
            layer_combined_acc = 0.0
            
            for branch_idx in range(n_branches):
                print(f"  Processing branch {branch_idx} in layer {layer_idx}...")
                logging.info(f"  Processing branch {branch_idx} in layer {layer_idx}")
                
                # Analyze excitatory component
                if hasattr(layer, "branch_excitation") and exc_inputs.shape[1] > 0:
                    branch_exc_weights = layer.branch_excitation.pre_w[branch_idx].cpu()
                    print(f"    Excitatory weights shape: {branch_exc_weights.shape}")
                    
                    branch_exc_activations = exc_inputs @ branch_exc_weights
                    print(f"    Excitatory activations shape: {branch_exc_activations.shape}")
                    
                    train_exc_dataset = torch.utils.data.TensorDataset(
                        branch_exc_activations.unsqueeze(1), train_labels
                    )
                    valid_exc_dataset = torch.utils.data.TensorDataset(
                        branch_exc_activations[:len(valid_inputs)].unsqueeze(1), valid_labels
                    )
                    
                    print(f"    Training excitatory classifier...")
                    exc_acc = self._train_mlp_classifier(train_exc_dataset, valid_exc_dataset)
                    print(f"    Branch {branch_idx} excitatory accuracy: {exc_acc:.4f}")
                    logging.info(f"    Branch {branch_idx} excitatory accuracy: {exc_acc:.4f}")
                    layer_exc_acc = max(layer_exc_acc, exc_acc)
                else:
                    print(f"    No excitatory inputs for branch {branch_idx}")
                    branch_exc_activations = None
                    exc_acc = 0.0
                    
                # Analyze inhibitory component
                if hasattr(layer, "branch_inhibition") and inh_inputs.shape[1] > 0:
                    branch_inh_weights = layer.branch_inhibition.pre_w[branch_idx].cpu()
                    print(f"    Inhibitory weights shape: {branch_inh_weights.shape}")
                    
                    branch_inh_activations = inh_inputs @ branch_inh_weights
                    print(f"    Inhibitory activations shape: {branch_inh_activations.shape}")
                    
                    train_inh_dataset = torch.utils.data.TensorDataset(
                        branch_inh_activations.unsqueeze(1), train_labels
                    )
                    valid_inh_dataset = torch.utils.data.TensorDataset(
                        branch_inh_activations[:len(valid_inputs)].unsqueeze(1), valid_labels
                    )
                    
                    print(f"    Training inhibitory classifier...")
                    inh_acc = self._train_mlp_classifier(train_inh_dataset, valid_inh_dataset)
                    print(f"    Branch {branch_idx} inhibitory accuracy: {inh_acc:.4f}")
                    logging.info(f"    Branch {branch_idx} inhibitory accuracy: {inh_acc:.4f}")
                    layer_inh_acc = max(layer_inh_acc, inh_acc)
                else:
                    print(f"    No inhibitory inputs for branch {branch_idx}")
                    branch_inh_activations = None
                    inh_acc = 0.0
                    
                # Analyze combined component
                if branch_exc_activations is not None and branch_inh_activations is not None:
                    if use_shunting:
                        branch_combined = branch_exc_activations / (
                            branch_exc_activations + branch_inh_activations + 1.0
                        )
                        print(f"    Using shunting combination")
                    else:
                        branch_combined = branch_exc_activations - branch_inh_activations
                        print(f"    Using subtraction combination")
                        
                    train_combined_dataset = torch.utils.data.TensorDataset(
                        branch_combined.unsqueeze(1), train_labels
                    )
                    valid_combined_dataset = torch.utils.data.TensorDataset(
                        branch_combined[:len(valid_inputs)].unsqueeze(1), valid_labels
                    )
                    
                    print(f"    Training combined classifier...")
                    combined_acc = self._train_mlp_classifier(train_combined_dataset, valid_combined_dataset)
                    print(f"    Branch {branch_idx} combined accuracy: {combined_acc:.4f}")
                    logging.info(f"    Branch {branch_idx} combined accuracy: {combined_acc:.4f}")
                    layer_combined_acc = max(layer_combined_acc, combined_acc)
                else:
                    print(f"    Could not compute combined inputs (missing excitatory or inhibitory)")
                    combined_acc = max(exc_acc, inh_acc)
                    
                branch_indices.append(f"{layer_idx}-{branch_idx}")
                
            layer_indices.append(layer_idx)
            layer_exc_accuracies.append(layer_exc_acc)
            layer_inh_accuracies.append(layer_inh_acc)
            layer_combined_accuracies.append(layer_combined_acc)
            
            print(f"  Layer {layer_idx} best accuracies: exc={layer_exc_acc:.4f}, inh={layer_inh_acc:.4f}, combined={layer_combined_acc:.4f}")
            
        # Generate visualization and save results
        if layer_indices:
            print("\nGenerating branch information plots...")
            plt.figure(figsize=(10, 6))
            plt.plot(layer_indices, layer_exc_accuracies, "o-", label="Excitatory")
            plt.plot(layer_indices, layer_inh_accuracies, "s-", label="Inhibitory")
            plt.plot(layer_indices, layer_combined_accuracies, "^-", label="Combined")
            plt.xlabel("Layer Index")
            plt.ylabel("Max Branch Classification Accuracy")
            plt.title("Branch Information Analysis - MLP Decoding Accuracy")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_filename = "branch_info_mlp_decoding.png"
            if file_name:
                plot_filename = f"{file_name}_{plot_filename}"
            plt.savefig(os.path.join(save_dir, plot_filename), dpi=300)
            plt.close()
            print(f"Saved plot to {os.path.join(save_dir, plot_filename)}")
            
            # Save results to CSV
            df = pd.DataFrame({
                "layer_idx": layer_indices,
                "excitatory_acc": layer_exc_accuracies,
                "inhibitory_acc": layer_inh_accuracies,
                "combined_acc": layer_combined_accuracies,
            })
            
            csv_filename = "branch_info_results.csv"
            if file_name:
                csv_filename = f"{file_name}_{csv_filename}"
            csv_path = os.path.join(save_dir, csv_filename)
            # Save as JSON instead of CSV
            json_path = csv_path.replace('.csv', '.json')
            results_dict = df.to_dict(orient='records')
            with open(json_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Saved results to {json_path}")
            
            # Print summary
            print("\nBranch Information Analysis Summary:")
            print("-" * 60)
            print(f"{'Layer':<8} {'Excitatory':<12} {'Inhibitory':<12} {'Combined':<12}")
            print("-" * 60)
            for i, layer_idx in enumerate(layer_indices):
                print(f"{layer_idx:<8} {layer_exc_accuracies[i]:<12.4f} {layer_inh_accuracies[i]:<12.4f} {layer_combined_accuracies[i]:<12.4f}")
            print("-" * 60)
        else:
            print("No valid layers processed. No plots generated.")
            
        print(f"Branch information analysis completed. Results saved to {save_dir}")
        logging.info(f"Saved branch info plots in {save_dir}")
        print("=" * 80 + "\n")
        
        return {
            "layer_indices": layer_indices,
            "excitatory_accuracies": layer_exc_accuracies,
            "inhibitory_accuracies": layer_inh_accuracies,
            "combined_accuracies": layer_combined_accuracies,
            "save_dir": save_dir
        }

    def analyze_branch_ablation(self, model, data, save_path, args=None):
        """
        Perform ablation studies on individual branches and layers.
        
        This method systematically ablates (zeros out) different branches and measures
        the impact on model performance to understand their functional importance.
        
        Args:
            model: The trained dendritic network model
            data: Dictionary containing 'train', 'valid', 'test' datasets
            save_path: Directory to save analysis results
            args: Configuration arguments for ablation settings
            
        Returns:
            Dictionary containing ablation results
        """
        print("\n" + "=" * 80)
        print("STARTING BRANCH ABLATION ANALYSIS") 
        print("=" * 80)
        
        os.makedirs(save_path, exist_ok=True)
        
        if not hasattr(model, "net"):
            print("ERROR: No 'net' attribute found on model. Skipping branch_ablation analysis.")
            logging.error("No 'net' attribute found on model. Skipping branch_ablation analysis.")
            return {}
            
        # TODO: Extract complete implementation from PlotManager._branch_ablation
        # For now, return placeholder to establish module structure
        print("Branch ablation analysis implementation pending...")
        return {"status": "branch_ablation_analysis_structure_created"}
        
    def _train_mlp_classifier(self, train_dataset, valid_dataset, hidden_dim=100, lr=0.001, epochs=20):
        """
        Train a simple MLP classifier for branch information analysis.
        
        This utility method trains a small neural network to decode task information
        from branch activations, providing a measure of information content.
        
        Args:
            train_dataset: Training dataset (TensorDataset)
            valid_dataset: Validation dataset (TensorDataset)
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            epochs: Number of training epochs
            
        Returns:
            Validation accuracy of trained classifier
        """
        input_dim = train_dataset.tensors[0].shape[1]
        num_classes = len(torch.unique(train_dataset.tensors[1]))

        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = SimpleMLP(input_dim, hidden_dim, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return correct / total if total > 0 else 0.0


__all__ = ["InteractiveBranchAnalyzer"] 