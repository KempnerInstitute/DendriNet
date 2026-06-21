"""
Evaluation Utilities
====================

This module contains utility functions for evaluating model performance
using various metrics including accuracy, log-likelihood, and MSE.

Supports two evaluation modes:

* **materialize** - load the full dataset into memory (fast for small datasets
  like CIFAR-10/MNIST).
* **stream** - iterate batch-by-batch through a ``DataLoader`` (required for
  large datasets like ImageNet that would OOM if materialized).

An ``"auto"`` mode picks the right strategy based on estimated dataset size.
"""

import inspect
import logging
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as functional
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    estimate_dataset_size_mb,
    make_analysis_loader,
    should_materialize_dataset,
)
from dendritic_modeling.models import Classifier, RecurrentClassifier, Regressor
from dendritic_modeling.utils.general import save_dict
from dendritic_modeling.utils.math.auc import roc_auc_score

logger = logging.getLogger(__name__)


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


def _prepare_regression_predictions_and_labels(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    seq_lengths: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize regressor outputs/labels for metric computation.

    - Recurrent [B, T, C] outputs are flattened over valid timesteps when labels
      are provided per timestep as [B, T, C].
    - Otherwise recurrent outputs are reduced to the last timestep.
    """
    kwargs = {}
    if seq_lengths is not None:
        kwargs["seq_lengths"] = seq_lengths
    predictions = regressor.predict(inputs, **kwargs)

    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)

    is_recurrent_core = bool(getattr(regressor, "_is_recurrent_core", False))
    if predictions.dim() == 3 and is_recurrent_core:
        if labels.dim() == 3 and labels.shape[:2] == predictions.shape[:2]:
            if seq_lengths is not None:
                time_index = torch.arange(
                    predictions.shape[1], device=predictions.device
                ).unsqueeze(0)
                valid_mask = time_index < seq_lengths.to(predictions.device).unsqueeze(
                    1
                )
                predictions = predictions[valid_mask]
                labels = labels.to(predictions.device)[valid_mask]
            else:
                predictions = predictions.reshape(-1, predictions.shape[-1])
                labels = labels.reshape(-1, labels.shape[-1])
            return predictions, labels
        predictions = _reduce_temporal(predictions)
        labels = _reduce_temporal_labels(labels, expected_dims=2)
        return predictions, labels

    if labels.dim() > 2:
        labels = _reduce_temporal_labels(labels, expected_dims=2)
    return predictions, labels


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
        logger.debug("Skipping %s for Classifier model", computation_func.__name__)
        return torch.tensor(float("nan"))

    try:
        with torch.no_grad():
            if move_device:
                with analysis_device_context(model, device) as analysis_device:
                    inputs = inputs.to(analysis_device)
                    labels = labels.to(analysis_device)
                    if seq_lengths is not None:
                        seq_lengths = seq_lengths.to(analysis_device)
                    sig = inspect.signature(computation_func)
                    if "seq_lengths" in sig.parameters:
                        return computation_func(
                            model, inputs, labels, reduce_dim, seq_lengths=seq_lengths
                        )
                    return computation_func(model, inputs, labels, reduce_dim)
            # Pass seq_lengths to classification metrics that support it
            sig = inspect.signature(computation_func)
            if "seq_lengths" in sig.parameters:
                return computation_func(
                    model, inputs, labels, reduce_dim, seq_lengths=seq_lengths
                )
            return computation_func(model, inputs, labels, reduce_dim)
    except Exception as e:
        logger.warning(
            "Error computing metric in %s: %s; input dtypes: inputs=%s, "
            "labels=%s; input shapes: inputs=%s, labels=%s",
            computation_func.__name__,
            e,
            inputs.dtype,
            labels.dtype,
            tuple(inputs.shape),
            tuple(labels.shape),
        )
        nan_tensor = torch.tensor(torch.nan)
        nan_tensor = nan_tensor.to(inputs.device)
        return nan_tensor


def _should_materialize(
    dataset: torch.utils.data.Dataset,
    mode: str = "auto",
    threshold_mb: float = 1024,
) -> bool:
    """Decide whether to materialize *dataset* into a single tensor."""

    class _Runtime:
        def __init__(self):
            self.mode = mode
            self.materialize_threshold_mb = threshold_mb

    return should_materialize_dataset(dataset, _Runtime())


def _stream_model_outputs(
    model: Union[Classifier, Regressor],
    loader: DataLoader,
    device: str,
    max_batches: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Run *model* on batches from *loader*, accumulating outputs + labels on CPU.

    Model outputs (logits/predictions) are much smaller than raw inputs
    (e.g. Nx1000 floats vs Nx3x224x224 for ImageNet), so accumulating them
    on CPU is safe even for very large datasets.

    Returns (all_inputs_placeholder, all_labels, all_seq_lengths) where
    ``all_inputs_placeholder`` is a dummy since we only need labels+outputs
    for metrics.  We actually return (all_outputs, all_labels, seq_lengths).
    """
    all_outputs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_seq_lengths: list[torch.Tensor] = []
    batches_seen = 0

    with analysis_device_context(model, device) as analysis_device:
        with torch.no_grad():
            for batch in loader:
                if max_batches is not None and batches_seen >= max_batches:
                    break
                inputs = batch[0].to(analysis_device)
                labels = batch[1]
                seq_lengths = batch[2] if len(batch) > 2 else None

                outputs = model(
                    inputs,
                    **(
                        {"seq_lengths": seq_lengths.to(analysis_device)}
                        if seq_lengths is not None
                        and "seq_lengths" in inspect.signature(model.forward).parameters
                        else {}
                    ),
                )
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
                if seq_lengths is not None:
                    all_seq_lengths.append(seq_lengths.cpu())
                batches_seen += 1

    if not all_outputs:
        return torch.tensor([]), torch.tensor([]), None

    cat_outputs = torch.cat(all_outputs, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    cat_seq_lengths = torch.cat(all_seq_lengths, dim=0) if all_seq_lengths else None
    return cat_outputs, cat_labels, cat_seq_lengths


def _make_eval_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str = "cpu",
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = 2,
    **_extra,
) -> DataLoader:
    """Build a DataLoader suitable for evaluation (no shuffle, no drop_last)."""

    class _Runtime:
        def __init__(self):
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.persistent_workers = persistent_workers
            self.prefetch_factor = prefetch_factor

    return make_analysis_loader(
        dataset=dataset,
        runtime=_Runtime(),
        device=device,
    )


class _CachedOutputModel:
    """Wraps a model so that ``__call__`` / ``predict`` return pre-computed outputs.

    Preserves ``isinstance`` checks (Classifier / Regressor) and attribute
    access (``_is_recurrent_core``, etc.) by delegating to the wrapped model.
    """

    def __init__(self, real_model: torch.nn.Module, cached_outputs: torch.Tensor):
        # Store on the instance directly so __getattr__ doesn't intercept them.
        object.__setattr__(self, "_real_model", real_model)
        object.__setattr__(self, "_cached_outputs", cached_outputs)

    # --- calls that metric functions make ---
    def __call__(self, *_args, **_kwargs):
        return object.__getattribute__(self, "_cached_outputs")

    def forward(self, *_args, **_kwargs):
        return object.__getattribute__(self, "_cached_outputs")

    def predict(self, *_args, **_kwargs):
        cached = object.__getattribute__(self, "_cached_outputs")
        real = object.__getattribute__(self, "_real_model")
        # Classifier.predict() returns argmax of logits; Regressor.predict()
        # returns raw outputs.  Replicate that from cached forward outputs
        # so metrics that call .predict() (e.g. pred_label_mi) see correct
        # class indices rather than raw logits.
        if isinstance(real, (Classifier, RecurrentClassifier)):
            return cached.argmax(dim=-1)
        return cached

    # --- delegate everything else (isinstance, attributes) to real model ---
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real_model"), name)

    def __isinstance_check__(self, cls):
        return isinstance(object.__getattribute__(self, "_real_model"), cls)

    # Make isinstance() work by forwarding class checks.
    @property
    def __class__(self):
        return object.__getattribute__(self, "_real_model").__class__


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
    eval_mode: str = "auto",
    eval_batch_size: int = 256,
    eval_num_workers: int = 4,
    eval_pin_memory: bool = True,
    eval_persistent_workers: bool = False,
    eval_prefetch_factor: Optional[int] = 2,
    eval_max_batches: Optional[int] = None,
    eval_materialize_threshold_mb: float = 1024,
    **metric_kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Generic function to evaluate any metric on multiple datasets.

    Supports two modes:
    - **materialize**: load dataset into memory via ``dataset[:]`` (old behaviour,
      fast for small datasets).
    - **stream**: iterate batch-by-batch via DataLoader (memory-safe for large
      datasets like ImageNet).

    The ``"auto"`` mode (default) picks based on estimated dataset size.

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
        eval_mode: "auto", "materialize", or "stream"
        eval_batch_size: Batch size for streaming evaluation
        eval_num_workers: DataLoader workers for streaming evaluation
        eval_max_batches: Cap evaluation to this many batches (None = all)
        eval_materialize_threshold_mb: Size threshold for auto mode switch
        **metric_kwargs: Additional keyword arguments for the metric function

    Returns:
        Tuple of (train_metric, valid_metric, test_metric)
    """
    if filename is None:
        filename = metric_name

    datasets = [train_ds, valid_ds, test_ds]
    results = []

    # Check once whether metric_func accepts seq_lengths.
    _metric_accepts_seq_lengths = (
        "seq_lengths" in inspect.signature(metric_func).parameters
    )

    for dataset in datasets:
        if dataset is None:
            results.append(None)
            continue

        materialize = _should_materialize(
            dataset, mode=eval_mode, threshold_mb=eval_materialize_threshold_mb
        )

        if materialize:
            # --- fast path: full materialization (old behaviour) ---
            try:
                items = dataset[:]
            except Exception:
                # Some wrappers don't support slicing; fall back to streaming.
                materialize = False
            else:
                seq_lengths = items[2] if len(items) > 2 else None
                call_kwargs = dict(
                    reduce_dim=reduce_dim,
                    move_device=move_device,
                    device=device,
                    **metric_kwargs,
                )
                if _metric_accepts_seq_lengths:
                    call_kwargs["seq_lengths"] = seq_lengths
                result: torch.Tensor = metric_func(
                    model,
                    items[0],
                    items[1],
                    **call_kwargs,
                )
                if hasattr(result, "numel") and result.numel() == 1:
                    result = result.item()
                results.append(result)
                continue

        # --- streaming path: run model batch-by-batch, accumulate outputs on CPU,
        # then compute the metric once on the full accumulated data.
        # This is numerically identical to the materialize path because the
        # same metric function runs on the complete output+label tensors. ---
        est_mb = estimate_dataset_size_mb(dataset)
        logger.info(
            "Streaming evaluation for %s (est. %.0f MB, threshold %.0f MB, "
            "max_batches=%s)",
            metric_name,
            est_mb,
            eval_materialize_threshold_mb,
            eval_max_batches,
        )
        loader = _make_eval_loader(
            dataset,
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
            pin_memory=eval_pin_memory,
            device=device,
            persistent_workers=eval_persistent_workers,
            prefetch_factor=eval_prefetch_factor,
        )
        cat_outputs, cat_labels, cat_seq_lengths = _stream_model_outputs(
            model=model,
            loader=loader,
            device=device,
            max_batches=eval_max_batches,
        )
        if cat_outputs.numel() == 0:
            results.append(None)
            continue

        # The _compute_* functions expect (model, inputs, labels) and call
        # model(inputs) internally.  We wrap the model so forward() returns
        # cached outputs, avoiding a second forward pass.  The wrapper
        # preserves isinstance checks and attributes like _is_recurrent_core.
        cached_model = _CachedOutputModel(model, cat_outputs)

        stream_kwargs = dict(
            reduce_dim=reduce_dim,
            move_device=False,
            device="cpu",
            **metric_kwargs,
        )
        if _metric_accepts_seq_lengths:
            stream_kwargs["seq_lengths"] = cat_seq_lengths
        result = metric_func(
            cached_model,
            cat_outputs,  # inputs arg (ignored by cached forward)
            cat_labels,
            **stream_kwargs,
        )
        if hasattr(result, "numel") and result.numel() == 1:
            result = result.item()
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
    **kwargs,
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
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

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
        **kwargs,
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
    **kwargs,
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
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

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
        **kwargs,
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
    **kwargs,
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
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

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
        **kwargs,
    )


def _pred_label_mi_bits(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Discrete mutual information I(C; Ĉ) in bits for predicted classes.

    Notes
    -----
    This uses the empirical joint distribution of (C, Ĉ) and does not require
    any external dependencies (e.g. scikit-learn).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0
    if y_true.shape[0] != y_pred.shape[0]:
        return 0.0

    # Map labels to contiguous integer IDs
    _, t = np.unique(y_true, return_inverse=True)
    _, p = np.unique(y_pred, return_inverse=True)

    joint = np.zeros((int(t.max()) + 1, int(p.max()) + 1), dtype=np.float64)
    np.add.at(joint, (t, p), 1.0)
    n = float(joint.sum())
    if n <= 0:
        return 0.0

    joint /= n
    p_t = joint.sum(axis=1, keepdims=True)
    p_p = joint.sum(axis=0, keepdims=True)
    denom = p_t @ p_p

    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log2(joint[nz] / denom[nz])))
    return mi if np.isfinite(mi) else 0.0


def _compute_pred_label_mi(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    logits, normalized_labels = _prepare_classification_logits_and_labels(
        classifier,
        inputs,
        labels,
        seq_lengths=seq_lengths,
    )
    yhat = logits.argmax(dim=-1)
    y_true = normalized_labels.long().detach().cpu().numpy()
    y_pred = yhat.long().detach().cpu().numpy()
    mi = _pred_label_mi_bits(y_true, y_pred)
    return torch.tensor(mi, dtype=torch.float32, device=labels.device)


def pred_label_mi_score(
    classifier: Classifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Compute mutual information between predicted labels and
    ground truth labels for a classifier.

    Args:
        classifier: The classifier model
        inputs: Input tensor
        labels: Ground truth labels
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation

    Returns:
        Mutual information tensor
    """
    return _compute_metric_score(
        computation_func=_compute_pred_label_mi,
        model=classifier,
        inputs=inputs,
        labels=labels,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
    )


def evaluate_pred_label_mi(
    classifier: Classifier,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    valid_ds: Optional[torch.utils.data.Dataset] = None,
    test_ds: Optional[torch.utils.data.Dataset] = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: Optional[str] = None,
    filename: Optional[str] = "pred_label_mi",
    **kwargs,
) -> tuple[
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
    Optional[Union[float, torch.Tensor]],
]:
    """
    Evaluate mutual information between predicted labels and
    ground truth labels on multiple datasets.

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
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

    Returns:
        Tuple of (train_mi, valid_mi, test_mi)
    """
    return _evaluate_metric_generic(
        metric_func=pred_label_mi_score,
        metric_name="pred_label_mi",
        model=classifier,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        save_path=save_path,
        filename=filename,
        **kwargs,
    )


def _compute_mse(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Early exit for Classifier models (safety check)
    if hasattr(regressor, "__class__") and "Classifier" in str(type(regressor)):
        return torch.tensor(float("nan"))
    predictions, labels = _prepare_regression_predictions_and_labels(
        regressor, inputs, labels, seq_lengths=seq_lengths
    )
    return torch.mean((predictions - labels) ** 2, dim=reduce_dim)


def mse_score(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: Optional[int] = None,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seq_lengths: Optional[torch.Tensor] = None,
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
        seq_lengths: Optional per-sample sequence lengths

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
        seq_lengths=seq_lengths,
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
    **kwargs,
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
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

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
        **kwargs,
    )


def _compute_cosine_similarity(
    regressor: Regressor,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    reduce_dim: int = 0,
    seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Early exit for Classifier models (safety check)
    if hasattr(regressor, "__class__") and "Classifier" in str(type(regressor)):
        return torch.tensor(float("nan"))
    predictions, labels = _prepare_regression_predictions_and_labels(
        regressor, inputs, labels, seq_lengths=seq_lengths
    )
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
    seq_lengths: Optional[torch.Tensor] = None,
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
        seq_lengths: Optional per-sample sequence lengths

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
        seq_lengths=seq_lengths,
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
    **kwargs,
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
        **kwargs: Forwarded to ``_evaluate_metric_generic`` (eval_mode, etc.)

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
        **kwargs,
    )
