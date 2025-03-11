"""
visualization_manager.py
========================
Contains a PlotManager class that holds references to all visualization
and analysis functions to produce actual visualizations for "gradients", 
"branch_info", and "ablation" using plot_utils.py functions 
"""

import os
import logging

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dendritic_modeling.synthetic_datasets import split_MNIST_inputs
from dendritic_modeling.utils import (
    Shaper,
    analyze_branch_ablation,
    analyze_branch_information,
    analyze_layerwise_ablation,
    measure_performance
)
from dendritic_modeling.plot_utils import (
    plot_einet_params, 
    plot_einet_activations,
    plot_einet_gradients,
    plot_einet_profiles,
    einet_activations_to_csv,
    plot_NLL_loss_curves,
    plot_depth_info
)

class PlotManager:
    """
    A manager that stores all known plot/analysis methods in a dictionary.
    The main experiment code can call PlotManager.run_plot(plot_name),
    without needing to manually code each function call in train_experiments.py.
    """
    def __init__(self, model, data, shape, config, save_path, file_name, device='cpu'):

        self.model = model
        self.data = data
        self.shape = shape
        self.config = config
        self.device = device

        vis_cfg = self.config.visualization
        self.save_path = save_path
        self.file_name = file_name
        os.makedirs(self.save_path, exist_ok=True)

        self.plot_methods = {
            "weights": self._plot_weights,
            "activations": self._plot_activations,
            "gradients": self._plot_gradients,
            "profiles": self._plot_profiles,
            "branch_info": self._branch_info,
            "ablation": self._branch_ablation,
            "layerwise_ablation": self._branch_ablation,
            "branch_ablation": self._branch_ablation,
        }

    def run_plot(self, plot_name, epoch=None):

        if epoch is not None:
            logging.info(f"At epoch {epoch}: generating plot '{plot_name}'...")
        else:
            logging.info(f"Generating plot '{plot_name}'...")
        func = self.plot_methods.get(plot_name, None)
        if func is not None:
            func()
            if epoch is not None:
                logging.info(f"Plot '{plot_name}' saved for epoch {epoch}.")
            else:
                logging.info(f"Plot '{plot_name}' saved.")
        else:
            logging.info(f"Plot name '{plot_name}' not recognized in PlotManager.")

    def _plot_weights(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "param_visuals")
            os.makedirs(save_dir, exist_ok=True)
            if self.shape is not None:
                shaper = Shaper(shape=self.shape)
            plot_einet_params(
                einet=self.model.net,
                save_root=save_dir,
                reshape_fn=shaper.reshape if self.shape is not None else None,
                reshape_exc_syn=True,
                logspace=True,
                save_in_dir=True,
                filename=self.file_name
            )
        else:
            logging.info("No 'net' attribute found on model. Skipping _plot_weights.")

    def _plot_activations(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "activation_visuals")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data['train'][:]
            input_list = split_MNIST_inputs(train_input, train_label)
            plot_einet_activations(
                einet=self.model.net,
                input_list=input_list,
                save_root=save_dir,
                save_in_dir=True,
                filename=self.file_name                
            )
        else:
            logging.info("No 'net' attribute on model. Skipping _plot_activations.")

    def _plot_gradients(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "grad_visuals")
            os.makedirs(save_dir, exist_ok=True)
            train_input, train_label = self.data['train'][:]
            if self.shape is not None:
                shaper = Shaper(shape=self.shape)
            plot_einet_gradients(
                model=self.model,
                inputs=train_input,
                labels=train_label,
                save_root=save_dir,
                reshape_fn=shaper.reshape if self.shape is not None else None,
                reshape_exc_syn=True,
                save_in_dir=True,
                filename=self.file_name
            )
            logging.info(f"Saved gradient visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping gradient plotting.")
    
    def _plot_profiles(self):
        if hasattr(self.model, 'net'):
            save_dir = os.path.join(self.save_path, "profiles")
            os.makedirs(save_dir, exist_ok=True)
            if self.shape is not None:
                shaper = Shaper(shape=self.shape)
            plot_einet_profiles(
                model=self.model,
                train_data=self.data['train'],
                valid_data=self.data['valid'],
                n_tasks=1,
                logspace=True,
                reshape_fn=shaper.reshape if self.shape is not None else None,
                save_root=save_dir,
                save_in_dir=True,
                filename=self.file_name,
            )
            logging.info(f"Saved profile visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping profile plotting.")

    def _branch_info(self):
        if not hasattr(self.model, 'net'):
            logging.info("No 'net' attribute found. Skipping branch_info.")
            return
        net = self.model.net
        if not hasattr(net, 'forward_with_branch_outputs'):
            logging.info("No forward_with_branch_outputs found. Skipping branch_info analysis.")
            return
        
        logging.info("Computing branch info using MLP decoding...")
        batch_size = getattr(self.config.train, 'batch_size', 64)
        save_dir = os.path.join(self.save_path, "branch_info_plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # Get data
        train_inputs, train_labels = self.data['train'][:]
        valid_inputs, valid_labels = self.data['valid'][:]
        test_inputs, test_labels = self.data['test'][:]
        
        # Record branch activations
        net.eval()
        
        # For storing accuracies
        layer_indices = []
        exc_accuracies = []
        inh_accuracies = []
        combined_accuracies = []
        
        use_shunting = getattr(self.model.net, 'use_shunting', True)
        
        for layer_idx, layer in enumerate(net.branch_layers):
            logging.info(f"Processing layer {layer_idx} for branch_info...")
            
            # Skip if no branch excitation or inhibition
            if not hasattr(layer, 'branch_excitation') and not hasattr(layer, 'branch_inhibition'):
                continue
                
            # For each sample, get excitatory and inhibitory inputs to branches
            all_exc_branch_inputs = []
            all_inh_branch_inputs = []
            
            with torch.no_grad():
                for batch_start in range(0, len(train_inputs), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_inputs))
                    batch = train_inputs[batch_start:batch_end].to(self.device)
                    
                    # Forward pass to get activations
                    _ = net(batch)
                    
                    # Extract excitatory and inhibitory inputs to branches
                    if hasattr(layer, 'branch_excitation'):
                        exc_inputs = layer.branch_excitation.pre_w @ layer.excitatory_input
                        all_exc_branch_inputs.append(exc_inputs.cpu())
                    
                    if hasattr(layer, 'branch_inhibition'):
                        inh_inputs = layer.branch_inhibition.pre_w @ layer.inhibitory_input
                        all_inh_branch_inputs.append(inh_inputs.cpu())
            
            # Concat batches
            if all_exc_branch_inputs:
                exc_branch_inputs = torch.cat(all_exc_branch_inputs, dim=0)
            else:
                exc_branch_inputs = torch.zeros((len(train_inputs), 0))
                
            if all_inh_branch_inputs:
                inh_branch_inputs = torch.cat(all_inh_branch_inputs, dim=0)
            else:
                inh_branch_inputs = torch.zeros((len(train_inputs), 0))
            
            # Create datasets for MLP training
            # 1. Excitatory only
            if exc_branch_inputs.shape[1] > 0:
                train_exc_dataset = TensorDataset(exc_branch_inputs, train_labels)
                valid_exc_dataset = TensorDataset(exc_branch_inputs[:len(valid_inputs)], valid_labels)
                
                # Train MLP
                exc_acc = self._train_mlp_classifier(train_exc_dataset, valid_exc_dataset)
                logging.info(f"Layer {layer_idx} excitatory branch info accuracy: {exc_acc:.4f}")
            else:
                exc_acc = 0.0
            
            # 2. Inhibitory only
            if inh_branch_inputs.shape[1] > 0:
                train_inh_dataset = TensorDataset(inh_branch_inputs, train_labels)
                valid_inh_dataset = TensorDataset(inh_branch_inputs[:len(valid_inputs)], valid_labels)
                
                # Train MLP
                inh_acc = self._train_mlp_classifier(train_inh_dataset, valid_inh_dataset)
                logging.info(f"Layer {layer_idx} inhibitory branch info accuracy: {inh_acc:.4f}")
            else:
                inh_acc = 0.0
            
            # 3. Combined
            if exc_branch_inputs.shape[1] > 0 and inh_branch_inputs.shape[1] > 0:
                # Combine according to use_shunting parameter
                if use_shunting:
                    # Shunting: E_ex/(E_ex+E_in+1)
                    combined_inputs = exc_branch_inputs / (exc_branch_inputs + inh_branch_inputs + 1.0)
                else:
                    # Subtraction: E_ex - E_in
                    combined_inputs = exc_branch_inputs - inh_branch_inputs
                
                train_combined_dataset = TensorDataset(combined_inputs, train_labels)
                valid_combined_dataset = TensorDataset(combined_inputs[:len(valid_inputs)], valid_labels)
                
                # Train MLP
                combined_acc = self._train_mlp_classifier(train_combined_dataset, valid_combined_dataset)
                logging.info(f"Layer {layer_idx} combined branch info accuracy: {combined_acc:.4f}")
            else:
                combined_acc = max(exc_acc, inh_acc)  # Use better of the two if one is missing
            
            # Store results
            layer_indices.append(layer_idx)
            exc_accuracies.append(exc_acc)
            inh_accuracies.append(inh_acc)
            combined_accuracies.append(combined_acc)
        
        # Plot results
        if layer_indices:
            plt.figure(figsize=(10, 6))
            plt.plot(layer_indices, exc_accuracies, 'o-', label='Excitatory')
            plt.plot(layer_indices, inh_accuracies, 's-', label='Inhibitory')
            plt.plot(layer_indices, combined_accuracies, '^-', label='Combined')
            
            plt.xlabel('Layer Index')
            plt.ylabel('Classification Accuracy')
            plt.title('Branch Information Analysis - MLP Decoding Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'branch_info_mlp_decoding.png'), dpi=300)
            plt.close()
            
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame({
                'layer_idx': layer_indices,
                'excitatory_acc': exc_accuracies,
                'inhibitory_acc': inh_accuracies,
                'combined_acc': combined_accuracies
            })
            df.to_csv(os.path.join(save_dir, 'branch_info_results.csv'), index=False)
            
        logging.info(f"Saved branch info plots in {save_dir}")
    
    def _train_mlp_classifier(self, train_dataset, valid_dataset, hidden_dim=100, lr=0.001, epochs=20):
        """Helper method to train MLP classifier for branch_info analysis"""
        input_dim = train_dataset.tensors[0].shape[1]
        num_classes = len(torch.unique(train_dataset.tensors[1]))
        
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
        model = SimpleMLP(input_dim, hidden_dim, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64)
        
        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
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
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total if total > 0 else 0.0

    def _branch_ablation(self):
        """
        Performs three types of ablation analysis for each dendritic layer:
        1. Zero out all excitatory AND inhibitory inputs and measure accuracy drop
        2. Zero out only excitatory inputs and measure accuracy drop  
        3. Zero out only inhibitory inputs and measure accuracy drop
        """
        if not hasattr(self.model, 'net'):
            logging.info("No 'net' attribute found. Skipping branch_ablation.")
            return
        net = self.model.net
        
        logging.info("Performing branch ablation analysis...")
        
        # Get baseline performance
        from dendritic_modeling.utils import measure_performance
        baseline_acc = measure_performance(net, self.data['test'], device=self.device)
        logging.info(f"Branch ablation baseline acc on test = {baseline_acc:.4f}")
        
        # Store original parameters for restoration
        original = {}
        for n, p in net.named_parameters():
            original[n] = p.detach().clone()
        
        # Initialize results dictionary
        layer_indices = []
        both_ablated_acc = []
        exc_ablated_acc = []
        inh_ablated_acc = []
        
        # For each layer
        for layer_idx in range(len(net.branch_layers)):
            layer = net.branch_layers[layer_idx]
            logging.info(f"Ablating layer {layer_idx}...")
            
            # 1. Ablate both excitatory and inhibitory inputs
            if hasattr(layer, 'branch_excitation'):
                layer.branch_excitation.pre_w.data.zero_()
            if hasattr(layer, 'branch_inhibition'):
                layer.branch_inhibition.pre_w.data.zero_()
            
            # Measure accuracy
            both_acc = measure_performance(net, self.data['test'], device=self.device)
            logging.info(f"Layer {layer_idx} with both exc+inh ablated: acc={both_acc:.4f}, drop={baseline_acc-both_acc:.4f}")
            
            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original[n])
            
            # 2. Ablate only excitatory inputs
            if hasattr(layer, 'branch_excitation'):
                layer.branch_excitation.pre_w.data.zero_()
                
                # Measure accuracy
                exc_acc = measure_performance(net, self.data['test'], device=self.device)
                logging.info(f"Layer {layer_idx} with exc ablated: acc={exc_acc:.4f}, drop={baseline_acc-exc_acc:.4f}")
            else:
                exc_acc = baseline_acc  # No excitatory to ablate
            
            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original[n])
            
            # 3. Ablate only inhibitory inputs
            if hasattr(layer, 'branch_inhibition'):
                layer.branch_inhibition.pre_w.data.zero_()
                
                # Measure accuracy
                inh_acc = measure_performance(net, self.data['test'], device=self.device)
                logging.info(f"Layer {layer_idx} with inh ablated: acc={inh_acc:.4f}, drop={baseline_acc-inh_acc:.4f}")
            else:
                inh_acc = baseline_acc  # No inhibitory to ablate
            
            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original[n])
            
            # Store results
            layer_indices.append(layer_idx)
            both_ablated_acc.append(both_acc)
            exc_ablated_acc.append(exc_acc)
            inh_ablated_acc.append(inh_acc)
        
        # Create plots
        save_dir = os.path.join(self.save_path, "branch_ablation_plots")
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(layer_indices, both_ablated_acc, 'o-', label='Both Ablated')
        plt.plot(layer_indices, exc_ablated_acc, 's-', label='Excitatory Ablated')
        plt.plot(layer_indices, inh_ablated_acc, '^-', label='Inhibitory Ablated')
        plt.axhline(y=baseline_acc, color='k', linestyle='--', label='Baseline')
        
        plt.xlabel('Layer Index')
        plt.ylabel('Classification Accuracy')
        plt.title('Branch Ablation Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'branch_ablation_accuracy.png'), dpi=300)
        plt.close()
        
        # Also create a plot showing the accuracy drop
        plt.figure(figsize=(10, 6))
        plt.bar(np.array(layer_indices) - 0.2, [baseline_acc - acc for acc in both_ablated_acc], width=0.2, label='Both')
        plt.bar(np.array(layer_indices), [baseline_acc - acc for acc in exc_ablated_acc], width=0.2, label='Excitatory')
        plt.bar(np.array(layer_indices) + 0.2, [baseline_acc - acc for acc in inh_ablated_acc], width=0.2, label='Inhibitory')
        
        plt.xlabel('Layer Index')
        plt.ylabel('Accuracy Drop')
        plt.title('Branch Ablation - Accuracy Drop')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'branch_ablation_drop.png'), dpi=300)
        plt.close()
        
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame({
            'layer_idx': layer_indices,
            'baseline_acc': [baseline_acc] * len(layer_indices),
            'both_ablated_acc': both_ablated_acc,
            'exc_ablated_acc': exc_ablated_acc,
            'inh_ablated_acc': inh_ablated_acc,
            'both_drop': [baseline_acc - acc for acc in both_ablated_acc],
            'exc_drop': [baseline_acc - acc for acc in exc_ablated_acc],
            'inh_drop': [baseline_acc - acc for acc in inh_ablated_acc]
        })
        df.to_csv(os.path.join(save_dir, 'branch_ablation_results.csv'), index=False)
        
        logging.info(f"Saved branch ablation plots in {save_dir}")