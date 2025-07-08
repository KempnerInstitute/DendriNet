"""
Evaluation Utilities
====================

This module contains utility functions for evaluating model performance
using various metrics including accuracy, log-likelihood, and MSE.
"""

from typing import Optional, Union, Tuple
import torch
from torch.distributions import Categorical

from dendritic_modeling.utils.data import save_dict


def accuracy_score(
    classifier: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Compute accuracy score for a classifier.
    
    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        
    Returns:
        Accuracy tensor
    """
    if move_device:
        classifier = classifier.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
    
    with torch.no_grad():
        yhat = classifier.predict(inputs)
        accuracy = torch.mean((yhat == labels).float(), dim=reduce_dim)
    
    return accuracy


def evaluate_accuracy(
    classifier: torch.nn.Module,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "accuracy",
) -> Tuple[Optional[Union[float, torch.Tensor]], Optional[Union[float, torch.Tensor]], Optional[Union[float, torch.Tensor]]]:
    """
    Evaluate accuracy on multiple datasets.
    
    Args:
        classifier: The classifier model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        
    Returns:
        Tuple of (train_acc, valid_acc, test_acc)
    """
    # Compute accuracy for each dataset
    train_acc = (
        accuracy_score(
            classifier,
            *train_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if train_ds
        else None
    )
    valid_acc = (
        accuracy_score(
            classifier,
            *valid_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if valid_ds
        else None
    )
    test_acc = (
        accuracy_score(
            classifier,
            *test_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if test_ds
        else None
    )

    # Convert single-element tensors to scalars
    if train_acc is not None and train_acc.numel() == 1:
        train_acc = train_acc.item()
    if valid_acc is not None and valid_acc.numel() == 1:
        valid_acc = valid_acc.item()
    if test_acc is not None and test_acc.numel() == 1:
        test_acc = test_acc.item()

    # Save results if path provided
    if save_path is not None:
        save_dict(
            {
                "train accuracy": train_acc.tolist()
                if isinstance(train_acc, torch.Tensor)
                else train_acc,
                "valid accuracy": valid_acc.tolist()
                if isinstance(valid_acc, torch.Tensor)
                else valid_acc,
                "test accuracy": test_acc.tolist()
                if isinstance(test_acc, torch.Tensor)
                else test_acc,
            },
            save_path,
            filename,
        )

    return train_acc, valid_acc, test_acc


def categorical_loglikelihood_score(
    classifier: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Compute categorical log-likelihood score for a classifier.
    
    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        
    Returns:
        Log-likelihood tensor
    """
    if move_device:
        classifier = classifier.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
    
    with torch.no_grad():
        logits = classifier(inputs)
        loglikelihood = torch.mean(
            Categorical(logits=logits).log_prob(labels),
            dim=reduce_dim
        )
    
    return loglikelihood


def evaluate_categorical_loglikelihood(
    classifier: torch.nn.Module,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "loglikelihood",
) -> Tuple[Optional[Union[float, torch.Tensor]], Optional[Union[float, torch.Tensor]], Optional[Union[float, torch.Tensor]]]:
    """
    Evaluate categorical log-likelihood on multiple datasets.
    
    Args:
        classifier: The classifier model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        
    Returns:
        Tuple of (train_ll, valid_ll, test_ll)
    """
    # Compute log-likelihood for each dataset
    train_ll = (
        categorical_loglikelihood_score(
            classifier,
            *train_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if train_ds
        else None
    )
    valid_ll = (
        categorical_loglikelihood_score(
            classifier,
            *valid_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if valid_ds
        else None
    )
    test_ll = (
        categorical_loglikelihood_score(
            classifier,
            *test_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if test_ds
        else None
    )

    # Convert single-element tensors to scalars
    if train_ll is not None and train_ll.numel() == 1:
        train_ll = train_ll.item()
    if valid_ll is not None and valid_ll.numel() == 1:
        valid_ll = valid_ll.item()
    if test_ll is not None and test_ll.numel() == 1:
        test_ll = test_ll.item()

    # Save results if path provided
    if save_path is not None:
        save_dict(
            {
                "train loglikelihood": train_ll.tolist()
                if isinstance(train_ll, torch.Tensor)
                else train_ll,
                "valid loglikelihood": valid_ll.tolist()
                if isinstance(valid_ll, torch.Tensor)
                else valid_ll,
                "test loglikelihood": test_ll.tolist()
                if isinstance(test_ll, torch.Tensor)
                else test_ll,
            },
            save_path,
            filename,
        )

    return train_ll, valid_ll, test_ll


def mse_score(
    regressor: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: Optional[int] = None,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Compute mean squared error score for a regressor.
    
    Args:
        regressor: The regressor model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean (not used in current implementation)
        move_device: Whether to move tensors to device
        device: Device to use for computation
        
    Returns:
        MSE tensor
    """
    if move_device:
        regressor = regressor.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
    
    with torch.no_grad():
        yhat = regressor.predict(inputs)
        mse = torch.mean((yhat - labels) ** 2)
    
    return mse


def evaluate_mse(
    regressor: torch.nn.Module,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "mse",
) -> Tuple[Optional[Union[float, torch.Tensor]], Optional[Union[float, torch.Tensor]], Optional[Union[float, torch.Tensor]]]:
    """
    Evaluate mean squared error on multiple datasets.
    
    Args:
        regressor: The regressor model
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results
        
    Returns:
        Tuple of (train_mse, valid_mse, test_mse)
    """
    # Compute MSE for each dataset
    train_mse = (
        mse_score(
            regressor,
            *train_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if train_ds
        else None
    )
    valid_mse = (
        mse_score(
            regressor,
            *valid_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if valid_ds
        else None
    )
    test_mse = (
        mse_score(
            regressor,
            *test_ds[:],
            reduce_dim=reduce_dim,
            move_device=move_device,
            device=device,
        )
        if test_ds
        else None
    )

    # Convert single-element tensors to scalars
    if train_mse is not None and train_mse.numel() == 1:
        train_mse = train_mse.item()
    if valid_mse is not None and valid_mse.numel() == 1:
        valid_mse = valid_mse.item()
    if test_mse is not None and test_mse.numel() == 1:
        test_mse = test_mse.item()

    # Save results if path provided
    if save_path is not None:
        save_dict(
            {
                "train mse": train_mse.tolist()
                if isinstance(train_mse, torch.Tensor)
                else train_mse,
                "valid mse": valid_mse.tolist()
                if isinstance(valid_mse, torch.Tensor)
                else valid_mse,
                "test mse": test_mse.tolist()
                if isinstance(test_mse, torch.Tensor)
                else test_mse,
            },
            save_path,
            filename,
        )

    return train_mse, valid_mse, test_mse 