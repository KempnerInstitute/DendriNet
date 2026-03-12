"""
Evaluation Utilities
====================

This module contains utility functions for evaluating model performance
using various metrics including accuracy, log-likelihood, and MSE.
"""

import inspect
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as functional
from torch.distributions import Categorical

from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.utils.general import save_dict
from dendritic_modeling.utils.math.auc import roc_auc_score


def _reduce_temporal(output: torch.Tensor) -> torch.Tensor:
    """Reduce [B, T, C] -> [B, C] using last timestep for evaluation metrics.

    No-op when output is already 2D. This ensures evaluation metrics (AUC,
    log-likelihood, MSE, cosine similarity) work with both feedforward and
    recurrent models.
    """
    if output.dim() == 3:
        return output[:, -1, :]
    return output


def _reduce_temporal_labels(labels: torch.Tensor, expected_dims: int) -> torch.Tensor:
    """Reduce temporal labels to match reduced outputs.

    For classification (expected_dims=1): [B, T] -> [B] (last timestep)
    For regression (expected_dims=2): [B, T, C] -> [B, C] (last timestep)
    No-op when labels already have the expected number of dimensions.
    """
    if labels.dim() > expected_dims:
        return labels[:, -1] if labels.dim() == 2 else labels[:, -1, :]
    return labels


def _flatten_labels_for_grouped_logits(
    labels: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Match labels to non-temporal grouped logits and flatten to [N_total]."""
    if labels.dim() == logits.dim() - 1 and labels.shape == logits.shape[:-1]:
        return labels.reshape(-1)

    # Support logits [G, B, C] with labels [B] by repeating labels over G.
    if logits.dim() >= 3 and labels.dim() == 1 and labels.shape[0] == logits.shape[-2]:
        expanded = labels
        for _ in range(logits.dim() - 2):
            expanded = expanded.unsqueeze(0)
        expanded = expanded.expand(*logits.shape[:-2], labels.shape[0])
        return expanded.reshape(-1)

    # Fallback: reduce extra axes in labels and flatten.
    if labels.dim() > 1:
        labels = _reduce_temporal_labels(labels, expected_dims=1)
    return labels.reshape(-1)


def _prepare_classification_logits_and_labels(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    seq_lengths: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize classifier outputs/labels for metric computation.

    - Recurrent [B, T, C] outputs are flattened over time when labels are
      provided per timestep as [B, T].
    - Otherwise recurrent outputs are reduced to the last timestep.
    - Grouped non-temporal outputs (e.g. [G, B, C]) are flattened to [N_total, C].
    """
    kwargs = {}
    if seq_lengths is not None:
        kwargs["seq_lengths"] = seq_lengths
    logits = classifier(inputs, **kwargs)
    labels = labels.long()

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    is_recurrent_core = bool(getattr(classifier, "_is_recurrent_core", False))
    if logits.dim() == 3 and is_recurrent_core:
        if labels.dim() == 2 and labels.shape[:2] == logits.shape[:2]:
            if seq_lengths is not None:
                time_index = torch.arange(
                    logits.shape[1], device=logits.device
                ).unsqueeze(0)
                valid_mask = time_index < seq_lengths.to(logits.device).unsqueeze(1)
                logits = logits[valid_mask]
                labels = labels.to(logits.device)[valid_mask]
            else:
                logits = logits.reshape(-1, logits.shape[-1])
                labels = labels.reshape(-1)
            return logits, labels.long()
        logits = _reduce_temporal(logits)
        labels = _reduce_temporal_labels(labels, expected_dims=1)
        return logits, labels.reshape(-1)

    if logits.dim() >= 3:
        labels_flat = _flatten_labels_for_grouped_logits(labels, logits)
        logits_flat = logits.reshape(-1, logits.shape[-1])
        return logits_flat, labels_flat

    if labels.dim() > 1:
        labels = _reduce_temporal_labels(labels, expected_dims=1)
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)
    return logits, labels


def _compute_metric_score(
    computation_func: Callable,
    model: Union[Classifier, Regressor],
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: Optional[torch.Tensor] = None,
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
        seq_lengths: Optional per-sample sequence lengths for variable-length sequences

    Returns:
        Computed metric tensor
    """
    # Early skip for inappropriate metrics on Classifier models
    if isinstance(model, Classifier) and computation_func.__name__ in [
        "_compute_mse",
        "_compute_cosine_similarity",
    ]:
        print(f"Skipping {computation_func.__name__} for Classifier model")
        return torch.tensor(float("nan"))

    if move_device:
        model = model.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        if seq_lengths is not None:
            seq_lengths = seq_lengths.to(device)
    try:
        with torch.no_grad():
            # Pass seq_lengths to classification metrics that support it
            sig = inspect.signature(computation_func)
            if "seq_lengths" in sig.parameters:
                return computation_func(
                    model, inputs, labels, reduce_dim, seq_lengths=seq_lengths
                )
            return computation_func(model, inputs, labels, reduce_dim)
    except Exception as e:
        print(f"Error computing metric in {computation_func.__name__}: {e}")
        print(f"  Input dtypes: inputs={inputs.dtype}, labels={labels.dtype}")
        print(f"  Input shapes: inputs={inputs.shape}, labels={labels.shape}")
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
    **metric_kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
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
            items = dataset[:]
            seq_lengths = items[2] if len(items) > 2 else None
            result: torch.Tensor = metric_func(
                model,
                items[0],
                items[1],
                reduce_dim=reduce_dim,
                move_device=move_device,
                device=device,
                seq_lengths=seq_lengths,
                **metric_kwargs,
            )
            # Convert single-element tensors to scalars
            if hasattr(result, "numel") and result.numel() == 1:
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
            filename,
        )

    return train_metric, valid_metric, test_metric


def _compute_accuracy(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits, labels = _prepare_classification_logits_and_labels(
        classifier, inputs, labels, seq_lengths=seq_lengths
    )
    yhat = logits.argmax(dim=-1).long().to(labels.device)
    correct = (yhat == labels.long()).float()
    if correct.numel() == 0:
        return torch.tensor(float("nan"), device=logits.device)
    return correct.mean()


def accuracy_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: Optional[torch.Tensor] = None,
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
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        Accuracy tensor
    """
    return _compute_metric_score(
        computation_func=_compute_accuracy,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
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
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
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
        metric_func=accuracy_score,
        metric_name="accuracy",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
    )


def _compute_auc(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _ = reduce_dim  # AUC is always computed over flattened sample axis.
    logits, labels = _prepare_classification_logits_and_labels(
        classifier, inputs, labels, seq_lengths=seq_lengths
    )
    labels = labels.long()
    unique_labels = torch.unique(labels)

    # AUC is undefined for a single-class target set.
    if unique_labels.numel() < 2:
        return torch.tensor(0.5, device=logits.device, dtype=torch.float32)

    # Allow datasets where only a subset of classes is present in labels.
    if logits.dim() == 2 and logits.shape[-1] != unique_labels.numel():
        class_indices = unique_labels.long()
        if class_indices.numel() <= logits.shape[-1]:
            logits = logits.index_select(-1, class_indices)
            label_map = {
                int(old_label.item()): new_idx
                for new_idx, old_label in enumerate(class_indices)
            }
            labels = labels.clone()
            for old_label, new_idx in label_map.items():
                labels[labels == old_label] = new_idx

    return roc_auc_score(labels, logits, dim=0)


def auc_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: Optional[torch.Tensor] = None,
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
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        AUC tensor
    """
    return _compute_metric_score(
        computation_func=_compute_auc,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
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
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
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
        metric_func=auc_score,
        metric_name="auc",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
    )


def _compute_loglikelihood(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits, labels = _prepare_classification_logits_and_labels(
        classifier, inputs, labels, seq_lengths=seq_lengths
    )
    log_probs = Categorical(logits=logits).log_prob(labels.long())
    if log_probs.numel() == 0:
        return torch.tensor(float("nan"), device=logits.device)
    if log_probs.dim() == 0:
        return log_probs
    if -log_probs.dim() <= reduce_dim < log_probs.dim():
        return torch.mean(log_probs, dim=reduce_dim)
    return torch.mean(log_probs)


def categorical_loglikelihood_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: Optional[torch.Tensor] = None,
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
        seq_lengths: Optional per-sample sequence lengths

    Returns:
        Log-likelihood tensor
    """
    return _compute_metric_score(
        computation_func=_compute_loglikelihood,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        seq_lengths=seq_lengths,
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
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
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
        metric_func=categorical_loglikelihood_score,
        metric_name="loglikelihood",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
    )


def _compute_mse(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
) -> torch.Tensor:
    # Early exit for Classifier models (safety check)
    if hasattr(regressor, "__class__") and "Classifier" in str(type(regressor)):
        return torch.tensor(float("nan"))
    predictions = _reduce_temporal(regressor.predict(inputs))
    labels = _reduce_temporal_labels(labels, expected_dims=2)
    return torch.mean((predictions - labels) ** 2, dim=reduce_dim)


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
        computation_func=_compute_mse,
        model=regressor,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
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
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
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
        metric_func=mse_score,
        metric_name="mse",
        model=regressor,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
    )


def _compute_cosine_similarity(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
) -> torch.Tensor:
    # Early exit for Classifier models (safety check)
    if hasattr(regressor, "__class__") and "Classifier" in str(type(regressor)):
        return torch.tensor(float("nan"))
    predictions = _reduce_temporal(regressor.predict(inputs))
    labels = _reduce_temporal_labels(labels, expected_dims=2)
    return torch.mean(
        functional.cosine_similarity(predictions, labels, dim=-1),
        dim=reduce_dim,
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
        computation_func=_compute_cosine_similarity,
        model=regressor,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
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
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
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
        metric_func=cosine_similarity_score,
        metric_name="cosine_similarity",
        model=regressor,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
    )
