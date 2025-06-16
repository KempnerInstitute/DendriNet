"""
Classification utilities for neural network analysis.

This module provides MLP classifiers and training utilities for analyzing
neural network activations and performing classification tasks.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    # These are optional for some functions
    plt = None
    pd = None


class MLPClassifier(nn.Module):
    """
    A simple MLP classifier for decoding layer activations.
    
    Args:
        input_dim: Size of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output classes
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        return self.model(x)


def train_classifier(
    model: nn.Module, 
    train_loader: DataLoader, 
    valid_loader: Optional[DataLoader] = None, 
    epochs: int = 50, 
    lr: float = 0.001, 
    device: str = "cuda"
) -> float:
    """
    Train an MLP classifier on activation data.

    Parameters
    ----------
    model : nn.Module
        The classifier model to train
    train_loader : DataLoader
        DataLoader with training data
    valid_loader : DataLoader, optional
        DataLoader with validation data
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    device : str
        Device to use for training

    Returns
    -------
    float
        Final validation accuracy
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_valid_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        if valid_loader is not None:
            model.eval()
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            valid_acc = accuracy_score(all_targets, all_preds)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model_state = model.state_dict().copy()

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return evaluate_classifier(model, valid_loader, device) if valid_loader else 0.0


def evaluate_classifier(model: nn.Module, data_loader: DataLoader, device: str = "cuda") -> float:
    """
    Evaluate a classifier on the given data.

    Parameters
    ----------
    model : nn.Module
        The classifier model to evaluate
    data_loader : DataLoader
        DataLoader with evaluation data
    device : str
        Device to use for evaluation

    Returns
    -------
    float
        Classification accuracy
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return accuracy_score(all_targets, all_preds)


def get_layer_activations(model, data_loader: DataLoader, device: str = "cuda") -> Dict[int, Tuple]:
    """
    Get activations for each layer in the model.

    Parameters
    ----------
    model : DendriNetWithOutputs
        The dendritic network model
    data_loader : DataLoader
        DataLoader with input data
    device : str
        Device to use

    Returns
    -------
    dict
        Dictionary mapping layer indices to (activations, labels) tuples
    """
    activations_per_layer = {}
    all_labels = []
    first_batch = True

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            # Get activations from all layers
            _, layer_activations = model.forward_with_branch_outputs(inputs)

            # Store activations and labels
            for i, act in enumerate(layer_activations):
                act_flat = act.view(act.size(0), -1).cpu()

                if first_batch:
                    activations_per_layer[i] = act_flat
                else:
                    activations_per_layer[i] = torch.cat(
                        [activations_per_layer[i], act_flat], dim=0
                    )

            if first_batch:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)

            first_batch = False

    # Create tuples of (activations, labels) for each layer
    return {
        layer_idx: (acts, all_labels)
        for layer_idx, acts in activations_per_layer.items()
    }


def get_ablated_activations(model, data_loader: DataLoader, ablation_type: str, device: str = "cuda") -> Dict:
    """
    Get activations with specific inputs ablated.

    Parameters
    ----------
    model : DendriNetWithOutputs
        The dendritic network model
    data_loader : DataLoader
        DataLoader with input data
    ablation_type : str
        Type of ablation: 'inhibitory' or 'excitatory'
    device : str
        Device to use

    Returns
    -------
    dict
        Dictionary mapping layer indices to list of ablated activations
    """
    all_activations = {}
    all_labels = []
    first_batch = True

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)

            # Save original parameters
            original_params = {}
            for layer_idx, layer in enumerate(model.branch_layers):
                if ablation_type == "inhibitory" and hasattr(
                    layer, "branch_inhibition"
                ):
                    original_params[
                        layer_idx
                    ] = layer.branch_inhibition.pre_w.data.clone()
                elif ablation_type == "excitatory" and hasattr(
                    layer, "branch_excitation"
                ):
                    original_params[
                        layer_idx
                    ] = layer.branch_excitation.pre_w.data.clone()

            # For each layer, ablate that layer and get activations
            for layer_idx in range(len(model.branch_inputs)):
                # Ablate the specified input type for this layer
                if ablation_type == "inhibitory" and hasattr(
                    model.branch_layers[layer_idx], "branch_inhibition"
                ):
                    model.branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(
                        float("-inf")
                    )
                elif ablation_type == "excitatory" and hasattr(
                    model.branch_layers[layer_idx], "branch_excitation"
                ):
                    model.branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(
                        float("-inf")
                    )

                # Get activations with this layer's inputs ablated
                _, layer_activations = model.forward_with_branch_outputs(inputs)

                # Restore the original parameters
                if ablation_type == "inhibitory" and hasattr(
                    model.branch_layers[layer_idx], "branch_inhibition"
                ):
                    model.branch_layers[layer_idx].branch_inhibition.pre_w.data.copy_(
                        original_params[layer_idx]
                    )
                elif ablation_type == "excitatory" and hasattr(
                    model.branch_layers[layer_idx], "branch_excitation"
                ):
                    model.branch_layers[layer_idx].branch_excitation.pre_w.data.copy_(
                        original_params[layer_idx]
                    )

                # Store activations for this layer
                for i, act in enumerate(layer_activations):
                    act_flat = act.view(act.size(0), -1).cpu()

                    if first_batch:
                        if i not in all_activations:
                            all_activations[i] = {}
                        all_activations[i][layer_idx] = act_flat
                    else:
                        all_activations[i][layer_idx] = torch.cat(
                            [all_activations[i][layer_idx], act_flat], dim=0
                        )

            if first_batch:
                all_labels = labels
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)

            first_batch = False

    # Create tuples of (activations, labels) for each layer
    return {
        layer_idx: {
            ablated_idx: (acts, all_labels) for ablated_idx, acts in layer_dict.items()
        }
        for layer_idx, layer_dict in all_activations.items()
    }


def analyze_layerwise_ablation(
    model, 
    train_data, 
    valid_data, 
    test_data, 
    save_path: str, 
    device: str = "cuda"
) -> Dict:
    """
    Run comprehensive ablation analysis on the dendritic network.

    Parameters
    ----------
    model : DendriNetWithOutputs
        The dendritic network model
    train_data : Dataset
        Training dataset
    valid_data : Dataset
        Validation dataset
    test_data : Dataset
        Test dataset
    save_path : str
        Path to save results
    device : str
        Device to use

    Returns
    -------
    dict
        Dictionary with ablation results
    """
    logging.info("Starting layer-wise ablation analysis...")

    # Check for required model attributes
    if not hasattr(model, "forward_with_branch_outputs"):
        logging.info(
            "Model does not have forward_with_branch_outputs method. Skipping analysis."
        )
        return {}

    # Import performance measurement from current module (to avoid circular import)
    from ..utils import measure_performance

    # Create dataloaders with smaller batch size to avoid memory issues
    batch_size = 32
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Get baseline model performance
    baseline_acc = measure_performance(model, test_data, device=device)
    logging.info(f"Baseline accuracy on test set: {baseline_acc:.4f}")

    # Save directory for results
    save_dir = os.path.join(save_path, "layerwise_ablation")
    os.makedirs(save_dir, exist_ok=True)

    # Get number of layers in the model
    num_layers = len(model.branch_layers)
    logging.info(f"Model has {num_layers} branch layers")

    # Initialize results dictionary
    results = {
        "layer_idx": [],
        "baseline_acc": [],
        "inh_ablated_acc": [],
        "exc_ablated_acc": [],
    }

    # Store original parameters for restoration
    original_params = {}
    for n, p in model.named_parameters():
        original_params[n] = p.detach().clone()

    # Process from top layers (higher indices) to bottom layers (lower indices)
    for layer_idx in reversed(range(num_layers)):
        logging.info(f"Analyzing layer {layer_idx}...")

        # 1. Measure baseline accuracy for this layer by ablating all other direct inputs
        for i in range(num_layers):
            if i != layer_idx:
                # Temporarily ablate other layers
                if hasattr(model.branch_layers[i], "branch_excitation"):
                    model.branch_layers[i].branch_excitation.pre_w.data.fill_(
                        -1000.0
                    )  # Very negative value to ensure zero after exp()
                if hasattr(model.branch_layers[i], "branch_inhibition"):
                    model.branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)

        # Measure accuracy with only this layer active
        baseline_layer_acc = measure_performance(model, test_data, device=device)

        # 2. Now ablate inhibitory inputs for this layer
        if hasattr(model.branch_layers[layer_idx], "branch_inhibition"):
            model.branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
            inh_ablated_acc = measure_performance(model, test_data, device=device)
        else:
            inh_ablated_acc = baseline_layer_acc  # No inhibitory inputs to ablate

        # Restore inhibitory inputs
        for n, p in model.named_parameters():
            p.data.copy_(original_params[n])

        # Re-ablate other layers
        for i in range(num_layers):
            if i != layer_idx:
                if hasattr(model.branch_layers[i], "branch_excitation"):
                    model.branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)
                if hasattr(model.branch_layers[i], "branch_inhibition"):
                    model.branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)

        # 3. Ablate excitatory inputs for this layer
        if hasattr(model.branch_layers[layer_idx], "branch_excitation"):
            model.branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
            exc_ablated_acc = measure_performance(model, test_data, device=device)
        else:
            exc_ablated_acc = baseline_layer_acc  # No excitatory inputs to ablate

        # Restore all parameters
        for n, p in model.named_parameters():
            p.data.copy_(original_params[n])

        # Store results
        results["layer_idx"].append(layer_idx)
        results["baseline_acc"].append(baseline_layer_acc)
        results["inh_ablated_acc"].append(inh_ablated_acc)
        results["exc_ablated_acc"].append(exc_ablated_acc)

        logging.info(
            f"Layer {layer_idx}: baseline={baseline_layer_acc:.4f}, "
            f"inh_ablated={inh_ablated_acc:.4f}, exc_ablated={exc_ablated_acc:.4f}"
        )

    # Plot the results if matplotlib is available
    if plt is not None:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(results["layer_idx"], results["baseline_acc"], "o-", label="Baseline")
            plt.plot(
                results["layer_idx"],
                results["inh_ablated_acc"],
                "s-",
                label="Inhibitory Ablated",
            )
            plt.plot(
                results["layer_idx"],
                results["exc_ablated_acc"],
                "^-",
                label="Excitatory Ablated",
            )

            plt.xlabel("Layer Index (higher = closer to input)")
            plt.ylabel("Classification Accuracy")
            plt.title("Layer-wise Ablation Analysis")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "layerwise_ablation.png"), dpi=300)
            plt.close()

            # Save detailed plot showing accuracy differences
            plt.figure(figsize=(10, 6))

            # Calculate differences
            inh_diff = [
                b - i for b, i in zip(results["baseline_acc"], results["inh_ablated_acc"])
            ]
            exc_diff = [
                b - e for b, e in zip(results["baseline_acc"], results["exc_ablated_acc"])
            ]

            plt.bar(
                np.array(results["layer_idx"]) - 0.2,
                inh_diff,
                width=0.4,
                label="Inhibitory Contribution",
                color="orange",
            )
            plt.bar(
                np.array(results["layer_idx"]) + 0.2,
                exc_diff,
                width=0.4,
                label="Excitatory Contribution",
                color="blue",
            )

            plt.xlabel("Layer Index (higher = closer to input)")
            plt.ylabel("Accuracy Drop when Ablated")
            plt.title("Contribution of Inhibitory and Excitatory Inputs per Layer")
            plt.legend()
            plt.grid(True, axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "layerwise_contributions.png"), dpi=300)
            plt.close()

            # Save as JSON if pandas is available
            if pd is not None:
                df = pd.DataFrame(results)
                # Save as JSON instead of CSV
                json_path = os.path.join(save_dir, "layerwise_results.json")
                results_dict = df.to_dict(orient='records')
                with open(json_path, 'w') as f:
                    json.dump(results_dict, f, indent=2)
        except Exception as e:
            logging.error(f"Error creating plots: {e}")

    logging.info(f"Ablation analysis results saved to {save_dir}")

    return results


def train_and_evaluate_classifier(
    X: np.ndarray, 
    y: np.ndarray, 
    hidden_dims: List[int] = [100, 50], 
    num_classes: int = 10, 
    epochs: int = 20, 
    batch_size: int = 64, 
    device: str = "cuda"
) -> float:
    """
    Train and evaluate an MLP classifier on the given data.

    Parameters
    ----------
    X : numpy.ndarray
        Input features
    y : numpy.ndarray
        Target labels
    hidden_dims : list
        Hidden layer dimensions
    num_classes : int
        Number of classes
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    device : str
        Device to use for training

    Returns
    -------
    float
        Classification accuracy
    """
    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create MLP classifier
    input_dim = X.shape[1]
    model = MLPClassifier(input_dim, hidden_dims, num_classes)
    model = model.to(device)

    # Set up optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    final_accuracy = correct / total
    return final_accuracy 