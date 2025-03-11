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
        
        # For storing results
        layer_indices = []
        branch_indices = []
        layer_exc_accuracies = []
        layer_inh_accuracies = []
        layer_combined_accuracies = []
        
        use_shunting = getattr(self.model.net, 'use_shunting', True)
        
        # Iterate through each layer
        for layer_idx, layer in enumerate(net.branch_layers):
            logging.info(f"Processing layer {layer_idx} for branch_info...")
            
            # Skip if no branch excitation or inhibition
            if not hasattr(layer, 'branch_excitation') and not hasattr(layer, 'branch_inhibition'):
                logging.info(f"Layer {layer_idx} has no branch excitation or inhibition. Skipping.")
                continue
            
            # Number of branches in this layer
            n_branches = 0
            if hasattr(layer, 'branch_excitation'):
                n_branches = layer.branch_excitation.pre_w.shape[0]
            elif hasattr(layer, 'branch_inhibition'):
                n_branches = layer.branch_inhibition.pre_w.shape[0]
            
            if n_branches == 0:
                logging.info(f"Layer {layer_idx} has no branches. Skipping.")
                continue
                
            logging.info(f"Layer {layer_idx} has {n_branches} branches")
            
            # Collect inputs to each branch
            exc_branch_inputs_all = []
            inh_branch_inputs_all = []
            
            # Loop over data to collect branch inputs
            net.eval()
            with torch.no_grad():
                for batch_start in range(0, len(train_inputs), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_inputs))
                    batch = train_inputs[batch_start:batch_end].to(self.device)
                    
                    # Forward pass
                    _ = net(batch)
                    
                    # Extract branch inputs
                    if hasattr(layer, 'branch_excitation'):
                        exc_inputs = layer.excitatory_input  # [batch, exc_input_dim]
                        exc_branch_inputs_all.append(exc_inputs.cpu())
                    
                    if hasattr(layer, 'branch_inhibition'):
                        inh_inputs = layer.inhibitory_input  # [batch, inh_input_dim]
                        inh_branch_inputs_all.append(inh_inputs.cpu())
            
            # Concatenate batches
            if exc_branch_inputs_all:
                exc_inputs = torch.cat(exc_branch_inputs_all, dim=0)  # [n_samples, exc_input_dim]
            else:
                exc_inputs = torch.zeros((len(train_inputs), 0))
                
            if inh_branch_inputs_all:
                inh_inputs = torch.cat(inh_branch_inputs_all, dim=0)  # [n_samples, inh_input_dim]
            else:
                inh_inputs = torch.zeros((len(train_inputs), 0))
            
            # For layer-level accuracy measurement
            layer_exc_acc = 0.0
            layer_inh_acc = 0.0
            layer_combined_acc = 0.0
            
            # Process each branch
            for branch_idx in range(n_branches):
                logging.info(f"  Processing branch {branch_idx} in layer {layer_idx}")
                
                # 1. Excitatory inputs to this branch
                if hasattr(layer, 'branch_excitation') and exc_inputs.shape[1] > 0:
                    # Get weights for this branch [exc_input_dim]
                    branch_exc_weights = layer.branch_excitation.pre_w[branch_idx].cpu()
                    
                    # Compute weighted input for each sample [n_samples]
                    branch_exc_activations = exc_inputs @ branch_exc_weights
                    
                    # Create dataset
                    train_exc_dataset = TensorDataset(branch_exc_activations.unsqueeze(1), train_labels)
                    valid_exc_dataset = TensorDataset(branch_exc_activations[:len(valid_inputs)].unsqueeze(1), valid_labels)
                    
                    # Train classifier
                    exc_acc = self._train_mlp_classifier(train_exc_dataset, valid_exc_dataset)
                    logging.info(f"    Branch {branch_idx} excitatory accuracy: {exc_acc:.4f}")
                    
                    # Update layer accuracy (maximum across branches)
                    layer_exc_acc = max(layer_exc_acc, exc_acc)
                else:
                    branch_exc_activations = None
                    exc_acc = 0.0
                
                # 2. Inhibitory inputs to this branch
                if hasattr(layer, 'branch_inhibition') and inh_inputs.shape[1] > 0:
                    # Get weights for this branch [inh_input_dim]
                    branch_inh_weights = layer.branch_inhibition.pre_w[branch_idx].cpu()
                    
                    # Compute weighted input for each sample [n_samples]
                    branch_inh_activations = inh_inputs @ branch_inh_weights
                    
                    # Create dataset
                    train_inh_dataset = TensorDataset(branch_inh_activations.unsqueeze(1), train_labels)
                    valid_inh_dataset = TensorDataset(branch_inh_activations[:len(valid_inputs)].unsqueeze(1), valid_labels)
                    
                    # Train classifier
                    inh_acc = self._train_mlp_classifier(train_inh_dataset, valid_inh_dataset)
                    logging.info(f"    Branch {branch_idx} inhibitory accuracy: {inh_acc:.4f}")
                    
                    # Update layer accuracy (maximum across branches)
                    layer_inh_acc = max(layer_inh_acc, inh_acc)
                else:
                    branch_inh_activations = None
                    inh_acc = 0.0
                
                # 3. Combined inputs
                if branch_exc_activations is not None and branch_inh_activations is not None:
                    # Combine according to use_shunting
                    if use_shunting:
                        # Shunting: E_ex/(E_ex+E_in+1)
                        branch_combined = branch_exc_activations / (branch_exc_activations + branch_inh_activations + 1.0)
                    else:
                        # Subtraction: E_ex - E_in
                        branch_combined = branch_exc_activations - branch_inh_activations
                    
                    # Create dataset
                    train_combined_dataset = TensorDataset(branch_combined.unsqueeze(1), train_labels)
                    valid_combined_dataset = TensorDataset(branch_combined[:len(valid_inputs)].unsqueeze(1), valid_labels)
                    
                    # Train classifier
                    combined_acc = self._train_mlp_classifier(train_combined_dataset, valid_combined_dataset)
                    logging.info(f"    Branch {branch_idx} combined accuracy: {combined_acc:.4f}")
                    
                    # Update layer accuracy (maximum across branches)
                    layer_combined_acc = max(layer_combined_acc, combined_acc)
                else:
                    combined_acc = max(exc_acc, inh_acc)
                
                # Store branch results
                branch_indices.append(f"{layer_idx}-{branch_idx}")
            
            # Store layer-level results
            layer_indices.append(layer_idx)
            layer_exc_accuracies.append(layer_exc_acc)
            layer_inh_accuracies.append(layer_inh_acc)
            layer_combined_accuracies.append(layer_combined_acc)
        
        # Plot layer-level results
        if layer_indices:
            plt.figure(figsize=(10, 6))
            plt.plot(layer_indices, layer_exc_accuracies, 'o-', label='Excitatory')
            plt.plot(layer_indices, layer_inh_accuracies, 's-', label='Inhibitory')
            plt.plot(layer_indices, layer_combined_accuracies, '^-', label='Combined')
            
            plt.xlabel('Layer Index')
            plt.ylabel('Max Branch Classification Accuracy')
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
                'excitatory_acc': layer_exc_accuracies,
                'inhibitory_acc': layer_inh_accuracies,
                'combined_acc': layer_combined_accuracies
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
        Performs three types of ablation analysis for each branch in each dendritic layer:
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
        
        # Initialize results
        layer_indices = []
        layer_both_ablated_acc = []
        layer_exc_ablated_acc = []
        layer_inh_ablated_acc = []
        
        branch_indices = []
        branch_both_ablated_acc = []
        branch_exc_ablated_acc = []
        branch_inh_ablated_acc = []
        
        # For each layer
        for layer_idx in range(len(net.branch_layers)):
            layer = net.branch_layers[layer_idx]
            logging.info(f"Ablating layer {layer_idx}...")
            
            # Get number of branches
            n_branches = 0
            if hasattr(layer, 'branch_excitation'):
                n_branches = layer.branch_excitation.pre_w.shape[0]
            elif hasattr(layer, 'branch_inhibition'):
                n_branches = layer.branch_inhibition.pre_w.shape[0]
            
            if n_branches == 0:
                logging.info(f"Layer {layer_idx} has no branches. Skipping.")
                continue
                
            logging.info(f"Layer {layer_idx} has {n_branches} branches")
            
            # 1. Layer-level ablation: ablate all branches in this layer
            
            # 1.1 Ablate both excitatory and inhibitory inputs
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
            
            # 1.2 Ablate only excitatory inputs
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
            
            # 1.3 Ablate only inhibitory inputs
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
            
            # Store layer-level results
            layer_indices.append(layer_idx)
            layer_both_ablated_acc.append(both_acc)
            layer_exc_ablated_acc.append(exc_acc)
            layer_inh_ablated_acc.append(inh_acc)
            
            # 2. Branch-level ablation: ablate each branch individually
            
            for branch_idx in range(n_branches):
                logging.info(f"  Ablating branch {branch_idx} in layer {layer_idx}...")
                
                # 2.1 Ablate both excitatory and inhibitory inputs for this branch
                if hasattr(layer, 'branch_excitation'):
                    layer.branch_excitation.pre_w.data[branch_idx].zero_()
                if hasattr(layer, 'branch_inhibition'):
                    layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                
                # Measure accuracy
                branch_both_acc = measure_performance(net, self.data['test'], device=self.device)
                logging.info(f"    Branch {branch_idx} with both ablated: acc={branch_both_acc:.4f}, drop={baseline_acc-branch_both_acc:.4f}")
                
                # Restore parameters
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                
                # 2.2 Ablate only excitatory inputs for this branch
                if hasattr(layer, 'branch_excitation'):
                    layer.branch_excitation.pre_w.data[branch_idx].zero_()
                    
                    # Measure accuracy
                    branch_exc_acc = measure_performance(net, self.data['test'], device=self.device)
                    logging.info(f"    Branch {branch_idx} with exc ablated: acc={branch_exc_acc:.4f}, drop={baseline_acc-branch_exc_acc:.4f}")
                else:
                    branch_exc_acc = baseline_acc
                
                # Restore parameters
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                
                # 2.3 Ablate only inhibitory inputs for this branch
                if hasattr(layer, 'branch_inhibition'):
                    layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                    
                    # Measure accuracy
                    branch_inh_acc = measure_performance(net, self.data['test'], device=self.device)
                    logging.info(f"    Branch {branch_idx} with inh ablated: acc={branch_inh_acc:.4f}, drop={baseline_acc-branch_inh_acc:.4f}")
                else:
                    branch_inh_acc = baseline_acc
                
                # Restore parameters
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                
                # Store branch-level results
                branch_indices.append(f"{layer_idx}-{branch_idx}")
                branch_both_ablated_acc.append(branch_both_acc)
                branch_exc_ablated_acc.append(branch_exc_acc)
                branch_inh_ablated_acc.append(branch_inh_acc)
        
        # Create plots
        save_dir = os.path.join(self.save_path, "branch_ablation_plots")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Layer-level accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(layer_indices, layer_both_ablated_acc, 'o-', label='Both Ablated')
        plt.plot(layer_indices, layer_exc_ablated_acc, 's-', label='Excitatory Ablated')
        plt.plot(layer_indices, layer_inh_ablated_acc, '^-', label='Inhibitory Ablated')
        plt.axhline(y=baseline_acc, color='k', linestyle='--', label='Baseline')
        
        plt.xlabel('Layer Index')
        plt.ylabel('Classification Accuracy')
        plt.title('Layer-wise Ablation Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'layer_ablation_accuracy.png'), dpi=300)
        plt.close()
        
        # 2. Layer-level accuracy drop plot
        plt.figure(figsize=(10, 6))
        plt.bar(np.array(layer_indices) - 0.2, [baseline_acc - acc for acc in layer_both_ablated_acc], width=0.2, label='Both')
        plt.bar(np.array(layer_indices), [baseline_acc - acc for acc in layer_exc_ablated_acc], width=0.2, label='Excitatory')
        plt.bar(np.array(layer_indices) + 0.2, [baseline_acc - acc for acc in layer_inh_ablated_acc], width=0.2, label='Inhibitory')
        
        plt.xlabel('Layer Index')
        plt.ylabel('Accuracy Drop')
        plt.title('Layer-wise Ablation - Accuracy Drop')
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'layer_ablation_drop.png'), dpi=300)
        plt.close()
        
        # 3. Branch-level accuracy drop heatmap (if we have branch data)
        if branch_indices:
            # Parse branch indices to get layer and branch numbers
            layers = []
            branches = []
            for idx in branch_indices:
                l, b = map(int, idx.split('-'))
                layers.append(l)
                branches.append(b)
            
            # Create DataFrames for heatmaps
            import pandas as pd
            
            # Both ablated
            df_both = pd.DataFrame({
                'layer': layers,
                'branch': branches,
                'drop': [baseline_acc - acc for acc in branch_both_ablated_acc]
            })
            both_pivot = df_both.pivot_table(index='branch', columns='layer', values='drop', aggfunc='mean')
            
            # Excitatory ablated
            df_exc = pd.DataFrame({
                'layer': layers,
                'branch': branches,
                'drop': [baseline_acc - acc for acc in branch_exc_ablated_acc]
            })
            exc_pivot = df_exc.pivot_table(index='branch', columns='layer', values='drop', aggfunc='mean')
            
            # Inhibitory ablated
            df_inh = pd.DataFrame({
                'layer': layers,
                'branch': branches,
                'drop': [baseline_acc - acc for acc in branch_inh_ablated_acc]
            })
            inh_pivot = df_inh.pivot_table(index='branch', columns='layer', values='drop', aggfunc='mean')
            
            # Plot heatmaps
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Both ablated
            im0 = axes[0].imshow(both_pivot.values, cmap='viridis')
            axes[0].set_title('Both Ablated - Accuracy Drop')
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('Branch')
            axes[0].set_xticks(range(len(both_pivot.columns)))
            axes[0].set_xticklabels(both_pivot.columns)
            axes[0].set_yticks(range(len(both_pivot.index)))
            axes[0].set_yticklabels(both_pivot.index)
            plt.colorbar(im0, ax=axes[0])
            
            # Excitatory ablated
            im1 = axes[1].imshow(exc_pivot.values, cmap='viridis')
            axes[1].set_title('Excitatory Ablated - Accuracy Drop')
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Branch')
            axes[1].set_xticks(range(len(exc_pivot.columns)))
            axes[1].set_xticklabels(exc_pivot.columns)
            axes[1].set_yticks(range(len(exc_pivot.index)))
            axes[1].set_yticklabels(exc_pivot.index)
            plt.colorbar(im1, ax=axes[1])
            
            # Inhibitory ablated
            im2 = axes[2].imshow(inh_pivot.values, cmap='viridis')
            axes[2].set_title('Inhibitory Ablated - Accuracy Drop')
            axes[2].set_xlabel('Layer')
            axes[2].set_ylabel('Branch')
            axes[2].set_xticks(range(len(inh_pivot.columns)))
            axes[2].set_xticklabels(inh_pivot.columns)
            axes[2].set_yticks(range(len(inh_pivot.index)))
            axes[2].set_yticklabels(inh_pivot.index)
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'branch_ablation_heatmap.png'), dpi=300)
            plt.close()
        
        # Save all results as CSV
        import pandas as pd
        
        # Layer-level results
        df_layer = pd.DataFrame({
            'layer_idx': layer_indices,
            'baseline_acc': [baseline_acc] * len(layer_indices),
            'both_ablated_acc': layer_both_ablated_acc,
            'exc_ablated_acc': layer_exc_ablated_acc,
            'inh_ablated_acc': layer_inh_ablated_acc,
            'both_drop': [baseline_acc - acc for acc in layer_both_ablated_acc],
            'exc_drop': [baseline_acc - acc for acc in layer_exc_ablated_acc],
            'inh_drop': [baseline_acc - acc for acc in layer_inh_ablated_acc]
        })
        df_layer.to_csv(os.path.join(save_dir, 'layer_ablation_results.csv'), index=False)
        
        # Branch-level results
        if branch_indices:
            df_branch = pd.DataFrame({
                'branch_idx': branch_indices,
                'baseline_acc': [baseline_acc] * len(branch_indices),
                'both_ablated_acc': branch_both_ablated_acc,
                'exc_ablated_acc': branch_exc_ablated_acc,
                'inh_ablated_acc': branch_inh_ablated_acc,
                'both_drop': [baseline_acc - acc for acc in branch_both_ablated_acc],
                'exc_drop': [baseline_acc - acc for acc in branch_exc_ablated_acc],
                'inh_drop': [baseline_acc - acc for acc in branch_inh_ablated_acc]
            })
            df_branch.to_csv(os.path.join(save_dir, 'branch_ablation_results.csv'), index=False)
        
        logging.info(f"Saved branch ablation plots in {save_dir}")