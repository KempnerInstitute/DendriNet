"""Evaluation and benchmarking utilities for transformer replacements."""

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.networks.architectures.transformer import (
    apply_transformer_replacement_config,
)
from dendritic_modeling.training._transformer_replacement.builders import (
    _make_synthetic_units,
)
from dendritic_modeling.training._transformer_replacement.checkpoints import (
    _load_replacement_state_dict,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    TransformerReplacementBenchmarkResult,
    _compute_distillation_loss,
    _resolve_device,
    _resolve_dtype,
    _to_plain_mapping,
)
from dendritic_modeling.training._transformer_replacement.data import (
    _build_hidden_cache_units_and_datasets,
    _build_synthetic_datasets,
    _read_token_source,
)
from dendritic_modeling.training.replacement_common import _write_metrics_json


def _evaluate_tensor_datasets(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    loss_name: str,
    cosine_weight: float,
    forward_modules: Sequence[nn.Module] | None = None,
) -> float:
    total = 0.0
    count = 0
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for dataset in datasets
    ]
    with torch.no_grad():
        for module, loader in zip(modules, loaders):
            module.eval()
            for x, target in loader:
                x = x.to(device=device, dtype=dtype)
                target = target.to(device=device, dtype=dtype)
                loss = _compute_distillation_loss(
                    module(x),
                    target,
                    loss_name=loss_name,
                    cosine_weight=cosine_weight,
                )
                total += float(loss.item())
                count += 1
    return total / max(count, 1)


def _compute_prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    pred_flat = prediction.reshape(-1, prediction.shape[-1]).float()
    target_flat = target.reshape(-1, target.shape[-1]).float()
    error = pred_flat - target_flat
    mse = float(error.square().mean().item())
    mae = float(error.abs().mean().item())
    target_power = float(target_flat.square().mean().item())
    target_mean = target_flat.mean(dim=0, keepdim=True)
    total_var = float((target_flat - target_mean).square().sum().item())
    sse = float(error.square().sum().item())
    cosine = float(F.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item())
    return {
        "mse": mse,
        "mae": mae,
        "relative_mse": mse / max(target_power, 1e-12),
        "cosine": cosine,
        "r2": 1.0 - sse / max(total_var, 1e-12),
    }


def _evaluate_tensor_dataset_metrics(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    predictions = []
    targets = []
    loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for dataset in datasets
    ]
    with torch.no_grad():
        for unit, loader in zip(units, loaders):
            unit.replacement.eval()
            for x, target in loader:
                x = x.to(device=device, dtype=dtype)
                predictions.append(unit.replacement(x).detach().cpu())
                targets.append(target.detach().cpu())
    if not predictions:
        raise ValueError("No benchmark batches were evaluated")
    return _compute_prediction_metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )


def _synthetic_teacher_parameter_count(config: Config) -> int | None:
    train_cfg = config.training.transformer_replacement
    if str(train_cfg.teacher_source).lower() != "synthetic_mlp":
        return None
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    intermediate_size = int(train_cfg.intermediate_size or hidden_size * 4)
    kind = str(teacher_cfg.get("kind", "gated_mlp")).lower()
    bias = bool(teacher_cfg.get("bias", False))
    if kind == "linear":
        count = hidden_size * hidden_size
        return count + (hidden_size if bias else 0)
    if kind == "mlp":
        count = hidden_size * intermediate_size + intermediate_size * hidden_size
        return count + (intermediate_size + hidden_size if bias else 0)
    if kind == "gated_mlp":
        count = 3 * hidden_size * intermediate_size
        return count + (2 * intermediate_size + hidden_size if bias else 0)
    return None


def _evaluate_lm_loss_on_tokens(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    max_batches: int,
) -> dict[str, float]:
    """Evaluate causal-LM loss on deterministic windows from ``tokens``."""
    model.eval()
    input_device = next(model.parameters()).device
    max_start = int(tokens.numel()) - int(sequence_length)
    if max_start <= 0:
        raise ValueError("Token source is shorter than sequence_length")

    losses = []
    n_tokens = 0
    with torch.no_grad():
        for batch_idx in range(max(1, int(max_batches))):
            starts = [
                (batch_idx * int(batch_size) + offset) * int(sequence_length)
                for offset in range(int(batch_size))
            ]
            starts = [start % max_start for start in starts]
            input_ids = torch.stack(
                [tokens[start : start + int(sequence_length)] for start in starts]
            ).to(device=input_device)
            output = model(input_ids=input_ids, labels=input_ids)
            loss = output.loss.detach().float()
            losses.append(loss.cpu())
            n_tokens += int(input_ids.numel())
    mean_loss = float(torch.stack(losses).mean().item())
    perplexity = float(torch.exp(torch.tensor(min(mean_loss, 50.0))).item())
    return {
        "lm_loss": mean_loss,
        "perplexity": perplexity,
        "n_tokens": float(n_tokens),
    }


def _benchmark_hf_text_replacement(
    config: Config,
    *,
    checkpoint_dir: str,
    device: torch.device,
) -> TransformerReplacementBenchmarkResult:
    """Evaluate original vs patched causal-LM loss on the configured text."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text benchmarking") from exc

    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    model_kwargs = _to_plain_mapping(model_cfg.model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name, **model_kwargs)
    if "device_map" not in model_kwargs:
        model = model.to(device=device)
    _, valid_tokens = _read_token_source(config)
    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )

    original_metrics = _evaluate_lm_loss_on_tokens(
        model,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
    )

    records = apply_transformer_replacement_config(
        model,
        config.model.transformer_replacement,
        config.model.core,
    )
    if not records:
        raise ValueError("No transformer replacement records were created")

    for record in records:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"layer_{record.layer_index}_replacement.pt",
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(
            checkpoint_path, map_location=next(model.parameters()).device
        )
        state_dict = payload.get("state_dict", payload)
        _load_replacement_state_dict(record.replacement, state_dict)

    patched_metrics = _evaluate_lm_loss_on_tokens(
        model,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
    )

    teacher_dense = sum(
        sum(param.numel() for param in record.original_mlp.parameters())
        for record in records
    )
    estimates = [record.replacement.parameter_estimate() for record in records]
    stored = sum(int(estimate["stored_total"]) for estimate in estimates)
    active = sum(int(estimate["active_total"]) for estimate in estimates)

    result = TransformerReplacementBenchmarkResult(
        teacher_source="hf_text",
        layer_indices=[int(record.layer_index) for record in records],
        checkpoint_dir=checkpoint_dir,
        initial=original_metrics,
        trained=patched_metrics,
        teacher_dense_params=teacher_dense,
        dendritic_stored_params=stored,
        dendritic_active_params=active,
        stored_reduction_factor=teacher_dense / max(stored, 1),
        active_reduction_factor=teacher_dense / max(active, 1),
        initial_label="original_teacher_model",
        trained_label="trained_replacement_model",
    )
    _write_metrics_json(
        result, output_dir=checkpoint_dir, filename="benchmark_metrics.json"
    )
    return result


def benchmark_saved_transformer_replacement(
    config: Config,
    *,
    checkpoint_dir: str | None = None,
) -> TransformerReplacementBenchmarkResult:
    """Compare untrained and trained dendritic replacements against the teacher.

    The benchmark reconstructs the configured teacher data source, evaluates the
    initial dendritic replacement, loads ``layer_<idx>_replacement.pt`` files,
    and evaluates the trained replacement on the same validation hidden states.
    """
    train_cfg = config.training.transformer_replacement
    if not train_cfg.enabled:
        raise ValueError("training.transformer_replacement.enabled must be true")

    device = _resolve_device(train_cfg.device)
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    seed = int(train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ckpt_dir = checkpoint_dir or train_cfg.save_dir
    source = str(train_cfg.teacher_source).lower()
    if source == "hf_text":
        return _benchmark_hf_text_replacement(
            config,
            checkpoint_dir=ckpt_dir,
            device=device,
        )

    if source == "synthetic_mlp":
        units = _make_synthetic_units(config, device=device, dtype=dtype)
        _, valid_datasets = _build_synthetic_datasets(
            units,
            config,
            device=device,
            dtype=dtype,
        )
    elif source == "hidden_cache":
        units, _, valid_datasets = _build_hidden_cache_units_and_datasets(
            config,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            "Post-training benchmark currently supports synthetic_mlp and "
            "hidden_cache teacher sources. Use hidden_cache for large HF models."
        )

    initial_metrics = _evaluate_tensor_dataset_metrics(
        units,
        valid_datasets,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=dtype,
    )

    for unit in units:
        checkpoint_path = os.path.join(
            ckpt_dir,
            f"layer_{unit.layer_index}_replacement.pt",
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload.get("state_dict", payload)
        _load_replacement_state_dict(unit.replacement, state_dict)

    trained_metrics = _evaluate_tensor_dataset_metrics(
        units,
        valid_datasets,
        batch_size=train_cfg.batch_size,
        device=device,
        dtype=dtype,
    )

    estimates = [unit.replacement.parameter_estimate() for unit in units]
    stored = sum(int(estimate["stored_total"]) for estimate in estimates)
    active = sum(int(estimate["active_total"]) for estimate in estimates)
    teacher_dense = _synthetic_teacher_parameter_count(config)
    if teacher_dense is not None:
        teacher_dense *= max(len(units), 1)

    result = TransformerReplacementBenchmarkResult(
        teacher_source=source,
        layer_indices=[unit.layer_index for unit in units],
        checkpoint_dir=ckpt_dir,
        initial=initial_metrics,
        trained=trained_metrics,
        teacher_dense_params=teacher_dense,
        dendritic_stored_params=stored,
        dendritic_active_params=active,
        stored_reduction_factor=(
            None if teacher_dense is None else teacher_dense / max(stored, 1)
        ),
        active_reduction_factor=(
            None if teacher_dense is None else teacher_dense / max(active, 1)
        ),
    )

    _write_metrics_json(result, output_dir=ckpt_dir, filename="benchmark_metrics.json")
    return result
