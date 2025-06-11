"""
Training Factory and Utilities
===============================

This module contains factory functions for creating training strategies
and utility functions for training simple MLPs and weight transfer.
"""

import os
import time
from copy import deepcopy
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import wandb

from dendritic_modeling import logger
from dendritic_modeling.training.strategies import (
    TrainerMLE,
    TrainerRegression,
    LocalCreditAssignment,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
)


def get_trainer(strategy, optimizer, suppress_prints=False, print_every=10):
    """
    Factory function to create training strategies.
    
    Args:
        strategy: String specifying the training strategy
        optimizer: The optimizer to use
        suppress_prints: Whether to suppress print statements
        print_every: Print frequency for training logs
        
    Returns:
        An instance of the requested training strategy
    """
    strategy = strategy.lower()
    
    if strategy == "mle":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy in ["freeze_branches", "freeze_branches_reversed", 
                     "freeze_branches_with_pretraining", "freeze_branches_reversed_with_pretraining"]:
        # These strategies use TrainerMLE with specialized methods
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "local_credit_assignment":
        return LocalCreditAssignment(optimizer, suppress_prints, print_every)
    elif strategy == "two_step":
        return TwoStepTrainer(optimizer, suppress_prints, print_every)
    elif strategy == "two_step_kl":
        return TwoStepTrainerWithKL(optimizer, suppress_prints, print_every)
    elif strategy == "voltage_stabilization":
        return VoltageStabilizationTrainer(optimizer, suppress_prints, print_every)
    elif strategy == "mse":
        return TrainerRegression(optimizer, suppress_prints, print_every)
    else:
        logger.info(f"Unrecognized strategy '{strategy}', defaulting to 'mle'.")
        return TrainerMLE(optimizer, suppress_prints, print_every)


def train_simple_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    train_ds,
    valid_ds,
    test_ds,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.001,
    run_save_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Tuple[float, float, float]]:
    """
    Train a simple MLP model with better performance.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (number of classes)
        train_ds, valid_ds, test_ds: PyTorch datasets
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        run_save_path: Path to save model checkpoints
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Tuple of (model, (train_acc, valid_acc, test_acc))
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import here to avoid circular imports
    from dendritic_modeling.models import SimpleClassifier

    # Create the model
    model = SimpleClassifier(input_dim, hidden_dims, output_dim)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
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
                os.makedirs(run_save_path, exist_ok=True)
                save_path = os.path.join(run_save_path, "best_simple_mlp.pt")
                torch.save(best_state, save_path)

        # Log progress
        logger.info(
            f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

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

    return model, (train_acc, best_valid_acc, test_acc)


def train_mlp_with_classification_head(
    feature_extractor,
    num_features: int,
    num_classes: int,
    train_ds,
    valid_ds,
    test_ds,
    epochs: int = 20,
    patience: int = 10,
    batch_size: int = 128,
    lr: float = 0.001,
    run_save_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Tuple[float, float, float]]:
    """
    Train an MLP with a classification head, then return the trained feature extractor part.

    Args:
        feature_extractor: The feature extraction part of the MLP
        num_features: The output dimension of the feature extractor
        num_classes: The number of classes for classification
        train_ds, valid_ds, test_ds: The datasets
        epochs, batch_size, lr: Training parameters
        patience: Early stopping patience
        run_save_path: Where to save the model and logs
        device: Device to use ('cuda' or 'cpu')

    Returns:
        The trained feature extractor and performance metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import here to avoid circular imports
    from dendritic_modeling.models import MLPWithClassificationHead

    # Create the model with classification head
    model = MLPWithClassificationHead(feature_extractor, num_features, num_classes)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

    # Training loop
    best_valid_acc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    # Create the save directory if it doesn't exist
    if run_save_path:
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
            patience_counter = 0
            logger.info(
                f"New best model found at epoch {epoch} with validation accuracy: {valid_acc:.4f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation accuracy")
                break

        # Log progress
        logger.info(
            f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}"
        )

        # Log to wandb if available
        if wandb.run is not None:
            wandb.log(
                {
                    "mlp_train_loss": train_loss,
                    "mlp_train_acc": train_acc,
                    "mlp_valid_loss": valid_loss,
                    "mlp_valid_acc": valid_acc,
                }
            )

    # Save only the best model at the end of training
    if best_state and run_save_path:
        save_path = os.path.join(run_save_path, "best_mlp_with_classification.pt")
        torch.save(best_state, save_path)
        logger.info(f"Saved best model (from epoch {best_epoch}) to {save_path}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        logger.info(
            f"Loaded best model from epoch {best_epoch} with validation accuracy: {best_valid_acc:.4f}"
        )

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


def transfer_mlp_weights_to_input_net(mlp_net, input_net, head_type="excitatory"):
    """
    Transfer weights from a trained MLP to the corresponding part of an InputNetTransform.

    Args:
        mlp_net: The trained MLP network
        input_net: The InputNetTransform to transfer weights to
        head_type: Either "excitatory" or "inhibitory" to specify which head to update
    """
    logger.info(
        f"Transferring weights from MLP to {head_type} head of InputNetTransform"
    )

    # Extract the MLP layers
    if hasattr(mlp_net, "layers"):
        mlp_layers = mlp_net.layers
    elif hasattr(mlp_net, "net") and hasattr(mlp_net.net, "layers"):
        mlp_layers = mlp_net.net.layers
    else:
        logger.warning(
            f"Could not extract layers from MLP model. Using random initialization for {head_type} head."
        )
        return

    # Get the shared backbone and target head of the InputNetTransform
    shared_layers = []
    for module in input_net.shared:
        if isinstance(module, torch.nn.Linear):
            shared_layers.append(module)

    target_head = (
        input_net.head_exc if head_type == "excitatory" else input_net.head_inh
    )
    head_layers = []
    for module in target_head:
        if isinstance(module, torch.nn.Linear):
            head_layers.append(module)

    # Transfer weights for the shared backbone (only once, typically from excitatory head)
    if head_type == "excitatory":
        for i, shared_layer in enumerate(shared_layers):
            if (
                i < len(mlp_layers) - 1
            ):  # All but the last MLP layer go to shared backbone
                mlp_layer = mlp_layers[i]
                if hasattr(mlp_layer, "excit_pre_w"):
                    # MLPExcInhLayer has a different structure
                    shared_layer.weight.data = mlp_layer.excit_pre_w.exp().t()
                    shared_layer.bias.data = mlp_layer.bias.data
                else:
                    logger.warning(
                        f"Unexpected MLP layer structure. Using random initialization for shared layer {i}."
                    )

    # Transfer weights for the head-specific part
    if len(head_layers) > 0 and len(mlp_layers) > 0:
        # The last layer of MLP goes to the head
        last_mlp_layer = mlp_layers[-1]
        if hasattr(last_mlp_layer, "excit_pre_w"):
            head_layers[0].weight.data = last_mlp_layer.excit_pre_w.exp().t()
            head_layers[0].bias.data = last_mlp_layer.bias.data
        else:
            logger.warning(
                f"Unexpected MLP layer structure. Using random initialization for {head_type} head."
            )

    logger.info(f"Weight transfer completed for {head_type} head")


def extract_weights_from_simple_mlp(simple_mlp, input_net_transform):
    """
    Extract weights from a SimpleMLP model and transfer them to an InputNetTransform.

    Args:
        simple_mlp: The trained SimpleMLP model
        input_net_transform: The InputNetTransform to transfer weights to
    """
    logger.info("Transferring weights from SimpleMLP to InputNetTransform")

    # Get network modules
    mlp_modules = list(simple_mlp.net.network.children())
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
        exc_head_modules[0].weight.data = source_layer.weight.data[
            : exc_head_modules[0].weight.shape[0], :
        ].clone()
        exc_head_modules[0].bias.data = source_layer.bias.data[
            : exc_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to excitatory head")

        # Transfer to inhibitory head
        inh_head_modules[0].weight.data = source_layer.weight.data[
            : inh_head_modules[0].weight.shape[0], :
        ].clone()
        inh_head_modules[0].bias.data = source_layer.bias.data[
            : inh_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to inhibitory head")

    logger.info("Weight transfer from SimpleMLP to InputNetTransform completed") 