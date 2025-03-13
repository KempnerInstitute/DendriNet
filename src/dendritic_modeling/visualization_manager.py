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
    def __init__(self, model, data, args, device='cpu'):

        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.save_path = args.save_path
        
        # Set up logging
        os.makedirs(self.save_path, exist_ok=True)
        logging.basicConfig(filename=os.path.join(self.save_path, 'visualization.log'),
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        vis_cfg = self.args.visualization
        self.file_name = args.file_name
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
            # Store the current plot name for reference in the plot method
            self.current_plot = plot_name
            func()
            self.current_plot = None
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
            if self.args.shape is not None:
                shaper = Shaper(shape=self.args.shape)
            plot_einet_params(
                einet=self.model.net,
                save_root=save_dir,
                reshape_fn=shaper.reshape if self.args.shape is not None else None,
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
            if self.args.shape is not None:
                shaper = Shaper(shape=self.args.shape)
            plot_einet_gradients(
                model=self.model,
                inputs=train_input,
                labels=train_label,
                save_root=save_dir,
                reshape_fn=shaper.reshape if self.args.shape is not None else None,
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
            if self.args.shape is not None:
                shaper = Shaper(shape=self.args.shape)
            plot_einet_profiles(
                model=self.model,
                train_data=self.data['train'],
                valid_data=self.data['valid'],
                n_tasks=1,
                logspace=True,
                reshape_fn=shaper.reshape if self.args.shape is not None else None,
                save_root=save_dir,
                save_in_dir=True,
                filename=self.file_name,
            )
            logging.info(f"Saved profile visualizations in {save_dir}")
        else:
            logging.info("No 'net' attribute found. Skipping profile plotting.")

    def _branch_info(self):
        print("\n" + "="*80)
        print("STARTING BRANCH INFORMATION ANALYSIS")
        print("="*80)
        
        if not hasattr(self.model, 'net'):
            print("ERROR: No 'net' attribute found on model. Skipping branch_info analysis.")
            logging.error("No 'net' attribute found on model. Skipping branch_info analysis.")
            return
        net = self.model.net
        if not hasattr(net, 'forward_with_branch_outputs'):
            print("ERROR: No forward_with_branch_outputs method found on net. Skipping branch_info analysis.")
            logging.error("No forward_with_branch_outputs method found on net. Skipping branch_info analysis.")
            return
        
        print(f"Model has {len(net.branch_layers)} branch layers")
        logging.info(f"Model has {len(net.branch_layers)} branch layers")
        
        print("Computing branch info using MLP decoding...")
        logging.info("Computing branch info using MLP decoding...")
        
        batch_size = getattr(self.args.train, 'batch_size', 64)
        save_dir = os.path.join(self.save_path, "branch_info_plots")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved to: {save_dir}")
        
        # Get data
        print("Loading datasets...")
        train_inputs, train_labels = self.data['train'][:]
        valid_inputs, valid_labels = self.data['valid'][:]
        test_inputs, test_labels = self.data['test'][:]
        print(f"Loaded datasets: train={len(train_inputs)}, valid={len(valid_inputs)}, test={len(test_inputs)} samples")
        
        # For storing results
        layer_indices = []
        branch_indices = []
        layer_exc_accuracies = []
        layer_inh_accuracies = []
        layer_combined_accuracies = []
        
        use_shunting = getattr(self.model.net, 'use_shunting', True)
        print(f"Using {'shunting' if use_shunting else 'subtraction'} for combining excitatory and inhibitory inputs")
        
        # Iterate through each layer
        for layer_idx, layer in enumerate(net.branch_layers):
            print(f"\nProcessing layer {layer_idx} for branch_info...")
            logging.info(f"Processing layer {layer_idx} for branch_info...")
            
            # Skip if no branch excitation or inhibition
            if not hasattr(layer, 'branch_excitation') and not hasattr(layer, 'branch_inhibition'):
                print(f"  Layer {layer_idx} has no branch excitation or inhibition. Skipping.")
                logging.info(f"Layer {layer_idx} has no branch excitation or inhibition. Skipping.")
                continue
            
            # Number of branches in this layer
            n_branches = 0
            if hasattr(layer, 'branch_excitation'):
                n_branches = layer.branch_excitation.pre_w.shape[0]
                print(f"  Layer {layer_idx} has branch_excitation with {n_branches} branches")
            elif hasattr(layer, 'branch_inhibition'):
                n_branches = layer.branch_inhibition.pre_w.shape[0]
                print(f"  Layer {layer_idx} has branch_inhibition with {n_branches} branches")
            
            if n_branches == 0:
                print(f"  Layer {layer_idx} has no branches. Skipping.")
                logging.info(f"Layer {layer_idx} has no branches. Skipping.")
                continue
                
            print(f"  Layer {layer_idx} has {n_branches} branches")
            logging.info(f"Layer {layer_idx} has {n_branches} branches")
            
            # Collect inputs to each branch
            print("  Collecting branch inputs from dataset...")
            exc_branch_inputs_all = []
            inh_branch_inputs_all = []
            
            # Loop over data to collect branch inputs
            net.eval()
            
            # Ensure model is on the correct device
            net = net.to(self.device)
            
            with torch.no_grad():
                for batch_start in range(0, len(train_inputs), batch_size):
                    batch_end = min(batch_start + batch_size, len(train_inputs))
                    batch = train_inputs[batch_start:batch_end].to(self.device)
                    
                    # Forward pass
                    _ = net(batch)
                    
                    # Extract branch inputs - safely check for attributes
                    if hasattr(layer, 'branch_excitation'):
                        # Check if excitatory_input exists, otherwise skip
                        if hasattr(layer, 'excitatory_input'):
                            exc_inputs = layer.excitatory_input  # [batch, exc_input_dim]
                            # Make sure to move to CPU before storing
                            exc_branch_inputs_all.append(exc_inputs.cpu())
                        else:
                            # Skip this batch if excitatory_input isn't available
                            print("    Warning: layer has branch_excitation but no excitatory_input attribute")
                    
                    if hasattr(layer, 'branch_inhibition'):
                        # Check if inhibitory_input exists, otherwise skip
                        if hasattr(layer, 'inhibitory_input'):
                            inh_inputs = layer.inhibitory_input  # [batch, inh_input_dim]
                            # Make sure to move to CPU before storing
                            inh_branch_inputs_all.append(inh_inputs.cpu())
                        else:
                            # Skip this batch if inhibitory_input isn't available
                            print("    Warning: layer has branch_inhibition but no inhibitory_input attribute")
            
            # After processing, move the network back to its original device if needed
            # This ensures the model is ready for training
            if torch.cuda.is_available():
                net = net.cuda()
            
            # Concatenate batches
            if exc_branch_inputs_all:
                exc_inputs = torch.cat(exc_branch_inputs_all, dim=0)  # [n_samples, exc_input_dim]
                print(f"  Collected excitatory inputs: {exc_inputs.shape}")
            else:
                print("  No excitatory inputs collected - this is expected with input_mode=0")
                print("  Skipping branch information analysis for this layer")
                continue  # Skip to the next layer
                
            if inh_branch_inputs_all:
                inh_inputs = torch.cat(inh_branch_inputs_all, dim=0)  # [n_samples, inh_input_dim]
                print(f"  Collected inhibitory inputs: {inh_inputs.shape}")
            else:
                inh_inputs = torch.zeros((len(train_inputs), 0))
                print("  No inhibitory inputs collected")
            
            # For layer-level accuracy measurement
            layer_exc_acc = 0.0
            layer_inh_acc = 0.0
            layer_combined_acc = 0.0
            
            # Process each branch
            for branch_idx in range(n_branches):
                print(f"  Processing branch {branch_idx} in layer {layer_idx}...")
                logging.info(f"  Processing branch {branch_idx} in layer {layer_idx}")
                
                # 1. Excitatory inputs to this branch
                if hasattr(layer, 'branch_excitation') and exc_inputs.shape[1] > 0:
                    # Get weights for this branch [exc_input_dim]
                    branch_exc_weights = layer.branch_excitation.pre_w[branch_idx].cpu()
                    print(f"    Excitatory weights shape: {branch_exc_weights.shape}")
                    
                    # Compute weighted input for each sample [n_samples]
                    branch_exc_activations = exc_inputs @ branch_exc_weights
                    print(f"    Excitatory activations shape: {branch_exc_activations.shape}")
                    
                    # Create dataset
                    train_exc_dataset = TensorDataset(branch_exc_activations.unsqueeze(1), train_labels)
                    valid_exc_dataset = TensorDataset(branch_exc_activations[:len(valid_inputs)].unsqueeze(1), valid_labels)
                    
                    # Train classifier
                    print(f"    Training excitatory classifier...")
                    exc_acc = self._train_mlp_classifier(train_exc_dataset, valid_exc_dataset)
                    print(f"    Branch {branch_idx} excitatory accuracy: {exc_acc:.4f}")
                    logging.info(f"    Branch {branch_idx} excitatory accuracy: {exc_acc:.4f}")
                    
                    # Update layer accuracy (maximum across branches)
                    layer_exc_acc = max(layer_exc_acc, exc_acc)
                else:
                    print(f"    No excitatory inputs for branch {branch_idx}")
                    branch_exc_activations = None
                    exc_acc = 0.0
                
                # 2. Inhibitory inputs to this branch
                if hasattr(layer, 'branch_inhibition') and inh_inputs.shape[1] > 0:
                    # Get weights for this branch [inh_input_dim]
                    branch_inh_weights = layer.branch_inhibition.pre_w[branch_idx].cpu()
                    print(f"    Inhibitory weights shape: {branch_inh_weights.shape}")
                    
                    # Compute weighted input for each sample [n_samples]
                    branch_inh_activations = inh_inputs @ branch_inh_weights
                    print(f"    Inhibitory activations shape: {branch_inh_activations.shape}")
                    
                    # Create dataset
                    train_inh_dataset = TensorDataset(branch_inh_activations.unsqueeze(1), train_labels)
                    valid_inh_dataset = TensorDataset(branch_inh_activations[:len(valid_inputs)].unsqueeze(1), valid_labels)
                    
                    # Train classifier
                    print(f"    Training inhibitory classifier...")
                    inh_acc = self._train_mlp_classifier(train_inh_dataset, valid_inh_dataset)
                    print(f"    Branch {branch_idx} inhibitory accuracy: {inh_acc:.4f}")
                    logging.info(f"    Branch {branch_idx} inhibitory accuracy: {inh_acc:.4f}")
                    
                    # Update layer accuracy (maximum across branches)
                    layer_inh_acc = max(layer_inh_acc, inh_acc)
                else:
                    print(f"    No inhibitory inputs for branch {branch_idx}")
                    branch_inh_activations = None
                    inh_acc = 0.0
                
                # 3. Combined inputs
                if branch_exc_activations is not None and branch_inh_activations is not None:
                    # Combine according to use_shunting
                    if use_shunting:
                        # Shunting: E_ex/(E_ex+E_in+1)
                        branch_combined = branch_exc_activations / (branch_exc_activations + branch_inh_activations + 1.0)
                        print(f"    Using shunting combination")
                    else:
                        # Subtraction: E_ex - E_in
                        branch_combined = branch_exc_activations - branch_inh_activations
                        print(f"    Using subtraction combination")
                    
                    # Create dataset
                    train_combined_dataset = TensorDataset(branch_combined.unsqueeze(1), train_labels)
                    valid_combined_dataset = TensorDataset(branch_combined[:len(valid_inputs)].unsqueeze(1), valid_labels)
                    
                    # Train classifier
                    print(f"    Training combined classifier...")
                    combined_acc = self._train_mlp_classifier(train_combined_dataset, valid_combined_dataset)
                    print(f"    Branch {branch_idx} combined accuracy: {combined_acc:.4f}")
                    logging.info(f"    Branch {branch_idx} combined accuracy: {combined_acc:.4f}")
                    
                    # Update layer accuracy (maximum across branches)
                    layer_combined_acc = max(layer_combined_acc, combined_acc)
                else:
                    print(f"    Could not compute combined inputs (missing excitatory or inhibitory)")
                    combined_acc = max(exc_acc, inh_acc)
                
                # Store branch results
                branch_indices.append(f"{layer_idx}-{branch_idx}")
            
            # Store layer-level results
            layer_indices.append(layer_idx)
            layer_exc_accuracies.append(layer_exc_acc)
            layer_inh_accuracies.append(layer_inh_acc)
            layer_combined_accuracies.append(layer_combined_acc)
            
            print(f"  Layer {layer_idx} best accuracies: exc={layer_exc_acc:.4f}, inh={layer_inh_acc:.4f}, combined={layer_combined_acc:.4f}")
        
        # Plot layer-level results
        if layer_indices:
            print("\nGenerating branch information plots...")
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
            print(f"Saved plot to {os.path.join(save_dir, 'branch_info_mlp_decoding.png')}")
            
            # Save as CSV
            import pandas as pd
            df = pd.DataFrame({
                'layer_idx': layer_indices,
                'excitatory_acc': layer_exc_accuracies,
                'inhibitory_acc': layer_inh_accuracies,
                'combined_acc': layer_combined_accuracies
            })
            csv_path = os.path.join(save_dir, 'branch_info_results.csv') 
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
            
            # Print summary table
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
        print("="*80 + "\n")
    
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
        print("\n" + "="*80)
        print("STARTING BRANCH ABLATION ANALYSIS")
        print("="*80)
        
        if not hasattr(self.model, 'net'):
            print("ERROR: No 'net' attribute found on model. Skipping branch_ablation analysis.")
            logging.error("No 'net' attribute found on model. Skipping branch_ablation analysis.")
            return
        net = self.model.net
        
        # Check for configuration options
        # First check if they are specified directly as config parameters
        do_layer_ablation = getattr(self.args.visualization, 'layer_ablation', True)
        do_branch_ablation = getattr(self.args.visualization, 'branch_ablation', True)
        
        # Then check if they are specified as plot types in the plots list
        plots_list = getattr(self.args.visualization, 'plots', [])
        if 'layer_ablation' in plots_list:
            do_layer_ablation = True
        if 'branch_ablation' in plots_list:
            do_branch_ablation = True
        
        # If the current function was called via a specific plot type, set the corresponding flag
        current_plot = getattr(self, 'current_plot', None)
        if current_plot == 'layer_ablation':
            do_layer_ablation = True
            do_branch_ablation = False
        elif current_plot == 'branch_ablation':
            do_branch_ablation = True
            do_layer_ablation = False
        
        if not (do_layer_ablation or do_branch_ablation):
            print("Both layer_ablation and branch_ablation are disabled in config. Skipping analysis.")
            logging.info("Both layer_ablation and branch_ablation are disabled in config. Skipping analysis.")
            return
        
        print(f"Model has {len(net.branch_layers)} branch layers")
        logging.info(f"Model has {len(net.branch_layers)} branch layers")
        
        print("Performing branch ablation analysis...")
        logging.info("Performing branch ablation analysis...")
        print(f"Layer ablation enabled: {do_layer_ablation}")
        print(f"Branch ablation enabled: {do_branch_ablation}")
        logging.info(f"Layer ablation enabled: {do_layer_ablation}")
        logging.info(f"Branch ablation enabled: {do_branch_ablation}")
        
        # Get baseline performance
        from dendritic_modeling.utils import measure_performance
        print("Measuring baseline performance...")
        baseline_acc = measure_performance(net, self.data['test'], device=self.device)
        print(f"Baseline accuracy on test set: {baseline_acc:.4f}")
        logging.info(f"Branch ablation baseline acc on test = {baseline_acc:.4f}")
        
        # Store original parameters for restoration
        print("Storing original model parameters...")
        original = {}
        for n, p in net.named_parameters():
            original[n] = p.detach().clone()
        
        # Initialize results
        layer_indices = []
        layer_both_ablated_acc = []
        layer_exc_ablated_acc = []
        layer_inh_ablated_acc = []
        
        branch_indices = []
        branch_layer_indices = []  # Store layer index for each branch
        branch_both_ablated_acc = []
        branch_exc_ablated_acc = []
        branch_inh_ablated_acc = []
        
        # For each layer
        for layer_idx in range(len(net.branch_layers)):
            layer = net.branch_layers[layer_idx]
            print(f"\nAblating layer {layer_idx}...")
            logging.info(f"Ablating layer {layer_idx}...")
            
            # Get number of branches
            n_branches = 0
            if hasattr(layer, 'branch_excitation'):
                n_branches = layer.branch_excitation.pre_w.shape[0]
                print(f"  Layer {layer_idx} has branch_excitation with {n_branches} branches")
            elif hasattr(layer, 'branch_inhibition'):
                n_branches = layer.branch_inhibition.pre_w.shape[0]
                print(f"  Layer {layer_idx} has branch_inhibition with {n_branches} branches")
            
            if n_branches == 0:
                print(f"  Layer {layer_idx} has no branches. Skipping.")
                logging.info(f"Layer {layer_idx} has no branches. Skipping.")
                continue
                
            print(f"  Layer {layer_idx} has {n_branches} branches")
            logging.info(f"Layer {layer_idx} has {n_branches} branches")
            
            # 1. Layer-level ablation: ablate all branches in this layer
            if do_layer_ablation:
                print("  Performing layer-level ablation...")
                
                # 1.1 Ablate both excitatory and inhibitory inputs
                if hasattr(layer, 'branch_excitation'):
                    layer.branch_excitation.pre_w.data.zero_()
                    print("    Zeroed excitatory weights")
                
                if hasattr(layer, 'branch_inhibition'):
                    layer.branch_inhibition.pre_w.data.zero_()
                    print("    Zeroed inhibitory weights")
                
                # Measure accuracy
                print("    Measuring accuracy with both ablated...")
                both_acc = measure_performance(net, self.data['test'], device=self.device)
                both_drop = baseline_acc - both_acc
                print(f"    Layer {layer_idx} with both ablated: acc={both_acc:.4f}, drop={both_drop:.4f}")
                logging.info(f"Layer {layer_idx} with both ablated: acc={both_acc:.4f}, drop={both_drop:.4f}")
                
                # Restore parameters
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                print("    Restored original parameters")
                
                # 1.2 Ablate only excitatory inputs
                if hasattr(layer, 'branch_excitation'):
                    layer.branch_excitation.pre_w.data.zero_()
                    print("    Zeroed excitatory weights only")
                    
                    # Measure accuracy
                    print("    Measuring accuracy with excitatory ablated...")
                    exc_acc = measure_performance(net, self.data['test'], device=self.device)
                    exc_drop = baseline_acc - exc_acc
                    print(f"    Layer {layer_idx} with exc ablated: acc={exc_acc:.4f}, drop={exc_drop:.4f}")
                    logging.info(f"Layer {layer_idx} with exc ablated: acc={exc_acc:.4f}, drop={exc_drop:.4f}")
                else:
                    print("    No excitatory weights to ablate")
                    exc_acc = baseline_acc  # No excitatory to ablate
                    exc_drop = 0.0
                
                # Restore parameters
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                print("    Restored original parameters")
                
                # 1.3 Ablate only inhibitory inputs
                if hasattr(layer, 'branch_inhibition'):
                    layer.branch_inhibition.pre_w.data.zero_()
                    print("    Zeroed inhibitory weights only")
                    
                    # Measure accuracy
                    print("    Measuring accuracy with inhibitory ablated...")
                    inh_acc = measure_performance(net, self.data['test'], device=self.device)
                    inh_drop = baseline_acc - inh_acc
                    print(f"    Layer {layer_idx} with inh ablated: acc={inh_acc:.4f}, drop={inh_drop:.4f}")
                    logging.info(f"Layer {layer_idx} with inh ablated: acc={inh_acc:.4f}, drop={inh_drop:.4f}")
                else:
                    print("    No inhibitory weights to ablate")
                    inh_acc = baseline_acc  # No inhibitory to ablate
                    inh_drop = 0.0
                
                # Restore parameters
                for n, p in net.named_parameters():
                    p.data.copy_(original[n])
                print("    Restored original parameters")
                
                # Store layer-level results
                layer_indices.append(layer_idx)
                layer_both_ablated_acc.append(both_acc)
                layer_exc_ablated_acc.append(exc_acc)
                layer_inh_ablated_acc.append(inh_acc)
            
            # 2. Branch-level ablation: ablate each branch individually
            if do_branch_ablation:
                print(f"  Performing branch-level ablation for {n_branches} branches...")
                
                for branch_idx in range(n_branches):
                    print(f"    Ablating branch {branch_idx}...")
                    
                    # 2.1 Ablate both excitatory and inhibitory inputs for this branch
                    exc_ablated = False
                    inh_ablated = False
                    
                    if hasattr(layer, 'branch_excitation'):
                        layer.branch_excitation.pre_w.data[branch_idx].zero_()
                        exc_ablated = True
                        print(f"      Zeroed excitatory weights for branch {branch_idx}")
                    
                    if hasattr(layer, 'branch_inhibition'):
                        layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                        inh_ablated = True
                        print(f"      Zeroed inhibitory weights for branch {branch_idx}")
                    
                    if not (exc_ablated or inh_ablated):
                        print(f"      No weights to ablate for branch {branch_idx}. Skipping.")
                        continue
                    
                    # Measure accuracy
                    print(f"      Measuring accuracy with both ablated...")
                    branch_both_acc = measure_performance(net, self.data['test'], device=self.device)
                    branch_both_drop = baseline_acc - branch_both_acc
                    print(f"      Branch {branch_idx} with both ablated: acc={branch_both_acc:.4f}, drop={branch_both_drop:.4f}")
                    logging.info(f"    Branch {branch_idx} with both ablated: acc={branch_both_acc:.4f}, drop={branch_both_drop:.4f}")
                    
                    # Restore parameters
                    for n, p in net.named_parameters():
                        p.data.copy_(original[n])
                    print(f"      Restored original parameters")
                    
                    # 2.2 Ablate only excitatory inputs for this branch
                    if hasattr(layer, 'branch_excitation'):
                        layer.branch_excitation.pre_w.data[branch_idx].zero_()
                        print(f"      Zeroed excitatory weights only for branch {branch_idx}")
                        
                        # Measure accuracy
                        print(f"      Measuring accuracy with excitatory ablated...")
                        branch_exc_acc = measure_performance(net, self.data['test'], device=self.device)
                        branch_exc_drop = baseline_acc - branch_exc_acc
                        print(f"      Branch {branch_idx} with exc ablated: acc={branch_exc_acc:.4f}, drop={branch_exc_drop:.4f}")
                        logging.info(f"    Branch {branch_idx} with exc ablated: acc={branch_exc_acc:.4f}, drop={branch_exc_drop:.4f}")
                    else:
                        print(f"      No excitatory weights to ablate for branch {branch_idx}")
                        branch_exc_acc = baseline_acc
                        branch_exc_drop = 0.0
                    
                    # Restore parameters
                    for n, p in net.named_parameters():
                        p.data.copy_(original[n])
                    print(f"      Restored original parameters")
                    
                    # 2.3 Ablate only inhibitory inputs for this branch
                    if hasattr(layer, 'branch_inhibition'):
                        layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                        print(f"      Zeroed inhibitory weights only for branch {branch_idx}")
                        
                        # Measure accuracy
                        print(f"      Measuring accuracy with inhibitory ablated...")
                        branch_inh_acc = measure_performance(net, self.data['test'], device=self.device)
                        branch_inh_drop = baseline_acc - branch_inh_acc
                        print(f"      Branch {branch_idx} with inh ablated: acc={branch_inh_acc:.4f}, drop={branch_inh_drop:.4f}")
                        logging.info(f"    Branch {branch_idx} with inh ablated: acc={branch_inh_acc:.4f}, drop={branch_inh_drop:.4f}")
                    else:
                        print(f"      No inhibitory weights to ablate for branch {branch_idx}")
                        branch_inh_acc = baseline_acc
                        branch_inh_drop = 0.0
                    
                    # Restore parameters
                    for n, p in net.named_parameters():
                        p.data.copy_(original[n])
                    print(f"      Restored original parameters")
                    
                    # Store branch-level results
                    branch_indices.append(f"{layer_idx}-{branch_idx}")
                    branch_layer_indices.append(layer_idx)
                    branch_both_ablated_acc.append(branch_both_acc)
                    branch_exc_ablated_acc.append(branch_exc_acc)
                    branch_inh_ablated_acc.append(branch_inh_acc)
        
        # Create plots
        save_dir = os.path.join(self.save_path, "branch_ablation_plots")
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nResults will be saved to: {save_dir}")
        
        if not (layer_indices or branch_indices):
            print("No valid layers or branches processed. No plots generated.")
            return
        
        print("Generating ablation plots...")
        
        # Layer-level results
        if do_layer_ablation and layer_indices:
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
            layer_csv = os.path.join(save_dir, 'layer_ablation_results.csv')
            df_layer.to_csv(layer_csv, index=False)
            print(f"Saved layer results to {layer_csv}")
            
            # Layer plots
            plt.figure(figsize=(12, 8))
            x = np.arange(len(layer_indices))
            width = 0.25
            
            plt.bar(x - width, [baseline_acc - acc for acc in layer_both_ablated_acc], width, label='Both Ablated')
            plt.bar(x, [baseline_acc - acc for acc in layer_exc_ablated_acc], width, label='Excitatory Ablated')
            plt.bar(x + width, [baseline_acc - acc for acc in layer_inh_ablated_acc], width, label='Inhibitory Ablated')
            
            plt.xlabel('Layer Index (lower = closer to soma)')
            plt.ylabel('Accuracy Drop')
            plt.title('Layer-wise Ablation Analysis')
            plt.xticks(x, layer_indices)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'layer_ablation_plot.png'), dpi=300)
            plt.close()
        
        # Branch-level results
        if do_branch_ablation and branch_indices:
            df_branch = pd.DataFrame({
                'branch_idx': branch_indices,
                'layer_idx': branch_layer_indices,
                'baseline_acc': [baseline_acc] * len(branch_indices),
                'both_ablated_acc': branch_both_ablated_acc,
                'exc_ablated_acc': branch_exc_ablated_acc,
                'inh_ablated_acc': branch_inh_ablated_acc,
                'both_drop': [baseline_acc - acc for acc in branch_both_ablated_acc],
                'exc_drop': [baseline_acc - acc for acc in branch_exc_ablated_acc],
                'inh_drop': [baseline_acc - acc for acc in branch_inh_ablated_acc]
            })
            branch_csv = os.path.join(save_dir, 'branch_ablation_results.csv')
            df_branch.to_csv(branch_csv, index=False)
            print(f"Saved branch results to {branch_csv}")
            
            # Calculate summary statistics per layer
            print("Calculating branch-level summary statistics by layer...")
            
            # Create dictionaries to store statistics
            layer_summary = {}
            unique_layer_indices = sorted(set(branch_layer_indices))
            
            for layer_idx in unique_layer_indices:
                # Filter branches for this layer
                layer_mask = [i for i, idx in enumerate(branch_layer_indices) if idx == layer_idx]
                
                if not layer_mask:
                    continue
                    
                # Calculate statistics for this layer's branches
                exc_accs = [branch_exc_ablated_acc[i] for i in layer_mask]
                inh_accs = [branch_inh_ablated_acc[i] for i in layer_mask]
                both_accs = [branch_both_ablated_acc[i] for i in layer_mask]
                
                exc_drops = [baseline_acc - acc for acc in exc_accs]
                inh_drops = [baseline_acc - acc for acc in inh_accs]
                both_drops = [baseline_acc - acc for acc in both_accs]
                
                layer_summary[layer_idx] = {
                    'n_branches': len(layer_mask),
                    'both_mean_acc': np.mean(both_accs),
                    'both_std_acc': np.std(both_accs),
                    'exc_mean_acc': np.mean(exc_accs),
                    'exc_std_acc': np.std(exc_accs),
                    'inh_mean_acc': np.mean(inh_accs),
                    'inh_std_acc': np.std(inh_accs),
                    'both_mean_drop': np.mean(both_drops),
                    'both_std_drop': np.std(both_drops),
                    'exc_mean_drop': np.mean(exc_drops),
                    'exc_std_drop': np.std(exc_drops),
                    'inh_mean_drop': np.mean(inh_drops),
                    'inh_std_drop': np.std(inh_drops),
                }
            
            # Create DataFrame for layer-wise branch statistics
            df_branch_summary = pd.DataFrame.from_dict(layer_summary, orient='index')
            df_branch_summary.index.name = 'layer_idx'
            df_branch_summary.reset_index(inplace=True)
            
            # Save to CSV
            branch_summary_csv = os.path.join(save_dir, 'branch_ablation_summary_by_layer.csv')
            df_branch_summary.to_csv(branch_summary_csv, index=False)
            print(f"Saved branch-level summary statistics by layer to {branch_summary_csv}")
            
            # Plot branch summary statistics
            if len(df_branch_summary) > 0:
                plt.figure(figsize=(12, 8))
                x = np.arange(len(df_branch_summary))
                width = 0.25
                
                plt.bar(x - width, df_branch_summary['both_mean_drop'], width, yerr=df_branch_summary['both_std_drop'], 
                        label='Both Ablated', capsize=5, color='purple', alpha=0.7)
                plt.bar(x, df_branch_summary['exc_mean_drop'], width, yerr=df_branch_summary['exc_std_drop'], 
                        label='Excitatory Ablated', capsize=5, color='red', alpha=0.7)
                plt.bar(x + width, df_branch_summary['inh_mean_drop'], width, yerr=df_branch_summary['inh_std_drop'], 
                        label='Inhibitory Ablated', capsize=5, color='blue', alpha=0.7)
                
                plt.xlabel('Layer Index (lower = closer to soma)')
                plt.ylabel('Mean Accuracy Drop (with std dev)')
                plt.title('Average Branch Ablation Effect by Layer')
                plt.xticks(x, df_branch_summary['layer_idx'])
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'branch_ablation_summary_plot.png'), dpi=300)
                plt.close()
        
        # Print summary tables
        if do_layer_ablation and layer_indices:
            print("\nLayer-wise Ablation Analysis Summary:")
            print("-" * 80)
            print(f"{'Layer':<8} {'Baseline':<10} {'Both Acc':<10} {'Both Drop':<10} {'Exc Acc':<10} {'Exc Drop':<10} {'Inh Acc':<10} {'Inh Drop':<10}")
            print("-" * 80)
            for i, layer_idx in enumerate(layer_indices):
                both_drop = baseline_acc - layer_both_ablated_acc[i]
                exc_drop = baseline_acc - layer_exc_ablated_acc[i]
                inh_drop = baseline_acc - layer_inh_ablated_acc[i]
                print(f"{layer_idx:<8} {baseline_acc:<10.4f} {layer_both_ablated_acc[i]:<10.4f} {both_drop:<10.4f} {layer_exc_ablated_acc[i]:<10.4f} {exc_drop:<10.4f} {layer_inh_ablated_acc[i]:<10.4f} {inh_drop:<10.4f}")
            print("-" * 80)
        
        if do_branch_ablation and len(df_branch_summary) > 0:
            print("\nBranch-level Summary Statistics by Layer:")
            print("-" * 100)
            print(f"{'Layer':<8} {'# Branches':<12} {'Both Mean':<12} {'Both Std':<12} {'Exc Mean':<12} {'Exc Std':<12} {'Inh Mean':<12} {'Inh Std':<12}")
            print("-" * 100)
            for _, row in df_branch_summary.iterrows():
                print(f"{int(row['layer_idx']):<8} {int(row['n_branches']):<12} {row['both_mean_drop']:<12.4f} {row['both_std_drop']:<12.4f} {row['exc_mean_drop']:<12.4f} {row['exc_std_drop']:<12.4f} {row['inh_mean_drop']:<12.4f} {row['inh_std_drop']:<12.4f}")
            print("-" * 100)
        
        print(f"Branch ablation analysis completed. Results saved to {save_dir}")
        logging.info(f"Saved branch ablation plots in {save_dir}")
        print("="*80 + "\n")