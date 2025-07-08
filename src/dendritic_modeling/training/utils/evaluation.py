"""
Evaluation Utilities
====================

This module contains utility functions for evaluating model performance
using various metrics including accuracy, log-likelihood, and MSE.
"""

from typing import Optional, Union, Tuple
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from dendritic_modeling.utils.auc import roc_auc_score

from dendritic_modeling.models import (
    Classifier,
    Regressor
)
from dendritic_modeling.utils.data import save_dict

from typing import Union, Callable


def _compute_metric_score(
    computation_func: Callable,
    model: Union[Classifier, Regressor],
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.Tensor:
    """
    Generic function to compute any metric score with common boilerplate.
    
    Args:
        computation_func: Function that computes the specific metric
        model: The model (classifier or regressor)
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        
    Returns:
        Computed metric tensor
    """
    if move_device:
        model = model.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
    try:
        with torch.no_grad():
            return computation_func(model, inputs, labels, reduce_dim)
    except Exception as e:
        print(f"Error computing metric: {e}")
        nan_tensor = torch.tensor(torch.nan)
        nan_tensor = nan_tensor.to(inputs.device)
        return nan_tensor


def _evaluate_metric_generic(
    metric_func: Callable,
    metric_name: str,
    model: Union[Classifier, Regressor],
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = None,
    **metric_kwargs
) -> Tuple[
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]]
]:
    """
    Generic function to evaluate any metric on multiple datasets.
    
    Args:
        metric_func: The metric function to call (e.g., accuracy_score, auc_score)
        metric_name: Name of the metric for saving (e.g., "accuracy", "auc")
        model: The model (classifier or regressor)
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results (defaults to metric_name)
        **metric_kwargs: Additional keyword arguments for the metric function
        
    Returns:
        Tuple of (train_metric, valid_metric, test_metric)
    """
    if filename is None:
        filename = metric_name
    
    # Compute metric for each dataset
    datasets = [train_ds, valid_ds, test_ds]
    results = []
    
    for dataset in datasets:
        if dataset is not None:
            result: torch.Tensor = metric_func(
                model,
                *dataset[:],
                reduce_dim=reduce_dim,
                move_device=move_device,
                device=device,
                **metric_kwargs
            )
            # Convert single-element tensors to scalars
            if hasattr(result, 'numel') and result.numel() == 1:
                result = result.item()
        else:
            result = None
        results.append(result)

    train_metric, valid_metric, test_metric = results

    # Save results if path provided
    if save_path is not None:
        def _to_serializable(value):
            return value.tolist() if isinstance(value, torch.Tensor) else value

        save_dict(
            {
                f"train {metric_name}": _to_serializable(train_metric),
                f"valid {metric_name}": _to_serializable(valid_metric),
                f"test {metric_name}": _to_serializable(test_metric),
            },
            save_path,
            filename
        )

    return train_metric, valid_metric, test_metric


def _compute_accuracy(
    classifier: Classifier, 
    inputs: torch.Tensor,
    labels: torch.Tensor, 
    reduce_dim: int = 0
) -> torch.Tensor:
    yhat = classifier.predict(inputs)
    return torch.mean((yhat == labels).float(), dim=reduce_dim)


def accuracy_score(
    classifier: Classifier,
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
    return _compute_metric_score(
        computation_func=_compute_accuracy, model=classifier,
        inputs=inputs, labels=labels, reduce_dim=reduce_dim,
        move_device=move_device, device=device
    )


def evaluate_accuracy(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "accuracy",
) -> Tuple[
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]]
]:
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
    return _evaluate_metric_generic(
        metric_func=accuracy_score, metric_name="accuracy",
        model=classifier, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds,
        reduce_dim=reduce_dim, move_device=move_device, device=device,
        save_path=save_path, filename=filename
    )


def _compute_auc(
    classifier: Classifier, 
    inputs: torch.Tensor,
    labels: torch.Tensor, 
    reduce_dim: int = 0
) -> torch.Tensor:
    return roc_auc_score(labels, classifier(inputs), dim=reduce_dim)


def auc_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Compute AUC score for a classifier.
    
    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        
    Returns:
        AUC tensor
    """
    return _compute_metric_score(
        computation_func=_compute_auc, model=classifier,
        inputs=inputs, labels=labels, reduce_dim=reduce_dim,
        move_device=move_device, device=device
    )


def evaluate_auc(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "auc",
) -> Tuple[
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]]
]:
    """
    Evaluate AUC on multiple datasets.
    
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
        Tuple of (train_auc, valid_auc, test_auc)
    """
    return _evaluate_metric_generic(
        metric_func=auc_score, metric_name="auc",
        model=classifier, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds,
        reduce_dim=reduce_dim, move_device=move_device, device=device,
        save_path=save_path, filename=filename
    )


def _compute_loglikelihood(
    classifier: Classifier, 
    inputs: torch.Tensor,
    labels: torch.Tensor, 
    reduce_dim: int = 0
) -> torch.Tensor:
    return torch.mean(
        Categorical(logits=classifier(inputs)).log_prob(labels),
        dim=reduce_dim
    )
    

def categorical_loglikelihood_score(
    classifier: Classifier,
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
    return _compute_metric_score(
        computation_func=_compute_loglikelihood, model=classifier,
        inputs=inputs, labels=labels, reduce_dim=reduce_dim,
        move_device=move_device, device=device
    )


def evaluate_categorical_loglikelihood(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "loglikelihood",
) -> Tuple[
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]]
]:
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
    return _evaluate_metric_generic(
        metric_func=categorical_loglikelihood_score, metric_name="loglikelihood",
        model=classifier, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds,
        reduce_dim=reduce_dim, move_device=move_device, device=device,
        save_path=save_path, filename=filename
    )


def _compute_mse(
    regressor: Regressor, 
    inputs: torch.Tensor,
    labels: torch.Tensor, 
    reduce_dim: int = 0
) -> torch.Tensor:
    return torch.mean((regressor.predict(inputs) - labels) ** 2, dim=reduce_dim)
    

def mse_score(
    regressor: Regressor,
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
    return _compute_metric_score(
        computation_func=_compute_mse, model=regressor,
        inputs=inputs, labels=labels, reduce_dim=reduce_dim,
        move_device=move_device, device=device
    )

def evaluate_mse(
    regressor: Regressor,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "mse",
) -> Tuple[
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]]
]:
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
    return _evaluate_metric_generic(
        metric_func=mse_score, metric_name="mse",
        model=regressor, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds,
        reduce_dim=reduce_dim, move_device=move_device, device=device,
        save_path=save_path, filename=filename
    )


def _compute_cosine_similarity(
    regressor: Regressor, 
    inputs: torch.Tensor,
    labels: torch.Tensor, 
    reduce_dim: int = 0
) -> torch.Tensor:
    return torch.mean(
        F.cosine_similarity(regressor.predict(inputs), labels, dim=-1),
        dim=reduce_dim
    )
    

def cosine_similarity_score(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: Optional[int] = None,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Compute cosine similarity score for a regressor.
    
    Args:
        regressor: The regressor model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean (not used in current implementation)
        move_device: Whether to move tensors to device
        device: Device to use for computation
        
    Returns:
        Cosine similarity tensor
    """
    return _compute_metric_score(
        computation_func=_compute_cosine_similarity, model=regressor,
        inputs=inputs, labels=labels, reduce_dim=reduce_dim,
        move_device=move_device, device=device
    )


def evaluate_cosine_similarity(
    regressor: Regressor,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "cosine_similarity",
) -> Tuple[
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]], 
    Optional[Union[float, torch.Tensor]]
]:
    """
    Evaluate cosine similarity on multiple datasets.
    
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
        Tuple of (train_cosine_similarity, valid_cosine_similarity, test_cosine_similarity)
    """
    return _evaluate_metric_generic(
        metric_func=cosine_similarity_score, metric_name="cosine_similarity",
        model=regressor, train_ds=train_ds, valid_ds=valid_ds, test_ds=test_ds,
        reduce_dim=reduce_dim, move_device=move_device, device=device,
        save_path=save_path, filename=filename
    )
