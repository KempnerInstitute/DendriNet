"""Training loops for transformer replacement experiments."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.networks.architectures.transformer import (
    apply_transformer_replacement_config,
)
from dendritic_modeling.training._transformer_replacement.builders import (
    _make_synthetic_units,
    _records_to_units,
)
from dendritic_modeling.training._transformer_replacement.calibration import (
    calibrate_transformer_reactivation,
    calibration_batch_limit,
    prefix_calibration_diagnostics,
    save_reactivation_calibration,
)
from dendritic_modeling.training._transformer_replacement.checkpoints import (
    _compact_sparse_bitmask_state_dict,
    _compact_sparse_index_state_dict,
    _load_layerwise_training_checkpoint,
    _load_replacement_record_checkpoints,
    _restore_best_replacement_checkpoint,
    _save_best_replacement_checkpoint,
    _save_layerwise_training_checkpoint,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    TransformerReplacementTrainingResult,
    _compute_distillation_loss,
    _make_tensor_loader,
    _move_replacement_batch_tensors,
    _replacement_forward_modules,
    _resolve_device,
    _resolve_dtype,
    resolve_configured_replacement_layers,
)
from dendritic_modeling.training._transformer_replacement.data import (
    _build_hidden_cache_units_and_datasets,
    _build_synthetic_datasets,
    _load_hf_causal_lm_for_joint_training,
    _load_hf_teacher_and_units,
    _MLPIOCapture,
    _read_token_source,
    _sample_token_batch,
)
from dendritic_modeling.training._transformer_replacement.distributed import (
    initialize_transformer_distributed,
    is_transformer_main_process,
    transformer_distributed_mean,
    transformer_process_rank,
    wrap_transformer_ddp,
)
from dendritic_modeling.training._transformer_replacement.evaluation import (
    _compute_prediction_metrics,
    _evaluate_lm_loss_on_tokens,
    _evaluate_tensor_datasets,
)
from dendritic_modeling.training._transformer_replacement.joint import (
    _joint_lm_distillation_loss,
    _resolve_joint_hidden_layers,
    _select_joint_trainable_parameters,
)
from dendritic_modeling.training._transformer_replacement.token_stream import (
    StreamingTokenBatchSource,
)
from dendritic_modeling.training.optimizers import create_optimizer
from dendritic_modeling.training.optimizers.custom import (
    apply_indexed_rewiring_after_step,
)
from dendritic_modeling.training.replacement_common import (
    ReplacementTrainingHistory,
    _clip_replacement_grad_norm,
    _infinite_loader,
    _make_cuda_grad_scaler,
    _should_run_periodic_step,
    _write_metrics_json,
)

logger = logging.getLogger(__name__)


def _save_token_stream_state(
    source: torch.Tensor | StreamingTokenBatchSource,
    save_dir: str,
    *,
    checkpoint: bool = False,
) -> None:
    if isinstance(source, StreamingTokenBatchSource):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        filename = (
            (
                "token_stream_checkpoint_state.json"
                if checkpoint
                else "token_stream_state.json"
            )
            if world_size == 1
            else (
                "token_stream_checkpoint_state" if checkpoint else "token_stream_state"
            )
            + f".rank{transformer_process_rank()}.json"
        )
        source.save_state(os.path.join(save_dir, filename))


def _restore_layerwise_training(
    config: Config,
    units: Sequence[DistillationUnit],
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    device: torch.device,
) -> tuple[int, ReplacementTrainingHistory | None]:
    resume_path = str(
        getattr(config.training.transformer_replacement, "resume_checkpoint", "") or ""
    )
    if not resume_path:
        return 0, None
    step, history = _load_layerwise_training_checkpoint(
        resume_path,
        units,
        optimizer,
        scaler,
        device=device,
    )
    logger.info("Resumed transformer replacement training at step %s", step)
    return step, history


def _maybe_save_layerwise_training(
    config: Config,
    units: Sequence[DistillationUnit],
    optimizer: torch.optim.Optimizer,
    scaler,
    history: ReplacementTrainingHistory,
    *,
    step: int,
    device: torch.device,
    token_source: torch.Tensor | StreamingTokenBatchSource | None = None,
) -> None:
    train_cfg = config.training.transformer_replacement
    every = int(getattr(train_cfg, "checkpoint_every", 0) or 0)
    if every <= 0 or not _should_run_periodic_step(
        step,
        every,
        int(train_cfg.max_steps),
    ):
        return
    if token_source is not None:
        _save_token_stream_state(
            token_source,
            train_cfg.save_dir,
            checkpoint=True,
        )
    _save_layerwise_training_checkpoint(
        os.path.join(train_cfg.save_dir, "training_checkpoint.pt"),
        units,
        optimizer,
        scaler,
        history,
        step=step,
        device=device,
    )


def _save_initial_best_replacement(
    config: Config,
    units: Sequence[DistillationUnit],
    history: ReplacementTrainingHistory,
) -> None:
    train_cfg = config.training.transformer_replacement
    if not bool(getattr(train_cfg, "restore_best_replacement", True)):
        return
    _save_best_replacement_checkpoint(
        train_cfg.save_dir,
        units,
        step=0,
        valid_loss=history.best_valid_loss,
    )


def _record_valid_and_maybe_save_best(
    config: Config,
    units: Sequence[DistillationUnit],
    history: ReplacementTrainingHistory,
    *,
    step: int,
    valid_loss: float,
) -> None:
    previous_best = float(history.best_valid_loss)
    history.record_valid_loss(step, valid_loss)
    train_cfg = config.training.transformer_replacement
    if (
        bool(getattr(train_cfg, "restore_best_replacement", True))
        and float(valid_loss) < previous_best
    ):
        _save_best_replacement_checkpoint(
            train_cfg.save_dir,
            units,
            step=step,
            valid_loss=valid_loss,
        )


def _restore_best_replacement(
    config: Config,
    units: Sequence[DistillationUnit],
    *,
    device: torch.device,
) -> None:
    train_cfg = config.training.transformer_replacement
    if not bool(getattr(train_cfg, "restore_best_replacement", True)):
        return
    restored = _restore_best_replacement_checkpoint(
        train_cfg.save_dir,
        units,
        device=device,
    )
    if restored is not None:
        step, valid_loss = restored
        logger.info(
            "Restored validation-best replacement checkpoint step=%s loss=%.6f",
            step,
            valid_loss,
        )


def _calibrate_tensor_replacements(
    units: Sequence[DistillationUnit],
    datasets: Sequence[TensorDataset],
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    train_cfg = config.training.transformer_replacement
    limit = calibration_batch_limit(train_cfg)
    diagnostics = {}
    for unit, dataset in zip(units, datasets):
        batch_size = max(1, int(train_cfg.batch_size))
        batches = [
            dataset.tensors[0][start : start + batch_size].to(
                device=device,
                dtype=dtype,
            )
            for start in range(0, min(len(dataset), limit * batch_size), batch_size)
        ]
        current = calibrate_transformer_reactivation(
            unit.replacement,
            batches,
            train_cfg,
            device=device,
        )
        diagnostics.update(
            prefix_calibration_diagnostics(current, f"layer_{unit.layer_index}")
        )
    save_reactivation_calibration(diagnostics, train_cfg.save_dir)


def _calibrate_hf_replacements(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    validation_tokens: torch.Tensor,
    capture: _MLPIOCapture,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    train_cfg = config.training.transformer_replacement
    limit = calibration_batch_limit(train_cfg)
    inputs: dict[int, list[torch.Tensor]] = {unit.layer_index: [] for unit in units}
    input_device = next(teacher.parameters()).device
    with torch.no_grad():
        for _ in range(limit):
            capture.clear()
            input_ids = _sample_token_batch(
                validation_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            teacher(input_ids=input_ids)
            for unit in units:
                inputs[unit.layer_index].append(
                    capture.records[unit.layer_index]["input"].to(
                        device=device,
                        dtype=dtype,
                    )
                )
    diagnostics = {}
    for unit in units:
        current = calibrate_transformer_reactivation(
            unit.replacement,
            inputs[unit.layer_index],
            train_cfg,
            device=device,
        )
        diagnostics.update(
            prefix_calibration_diagnostics(current, f"layer_{unit.layer_index}")
        )
    capture.clear()
    save_reactivation_calibration(diagnostics, train_cfg.save_dir)


def _calibrate_joint_replacements(
    student: nn.Module,
    validation_tokens: torch.Tensor,
    config: Config,
    *,
    device: torch.device,
) -> None:
    train_cfg = config.training.transformer_replacement
    batches = [
        _sample_token_batch(
            validation_tokens,
            batch_size=train_cfg.batch_size,
            sequence_length=train_cfg.sequence_length,
            device=device,
        )
        for _ in range(calibration_batch_limit(train_cfg))
    ]
    diagnostics = calibrate_transformer_reactivation(
        student,
        batches,
        train_cfg,
        device=device,
    )
    save_reactivation_calibration(diagnostics, train_cfg.save_dir)


def _save_result(
    result: TransformerReplacementTrainingResult,
    units: Sequence[DistillationUnit],
    *,
    save_replacements: bool,
    topology_encoding: str = "uint",
) -> None:
    if not is_transformer_main_process():
        return
    _write_metrics_json(result)
    initialization_diagnostics = {
        f"layer_{unit.layer_index}": dict(diagnostics)
        for unit in units
        if (
            diagnostics := getattr(
                unit.replacement,
                "teacher_topk_diagnostics",
                {},
            )
        )
    }
    if initialization_diagnostics:
        with open(
            os.path.join(result.save_dir, "teacher_topk_initialization.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(initialization_diagnostics, handle, indent=2)
            handle.write("\n")
    if not save_replacements:
        return
    if topology_encoding not in {"uint", "bitmask"}:
        raise ValueError("replacement_checkpoint_encoding must be 'uint' or 'bitmask'")
    compact_state = (
        _compact_sparse_bitmask_state_dict
        if topology_encoding == "bitmask"
        else _compact_sparse_index_state_dict
    )
    for unit in units:
        state_dict, index_encoding = compact_state(unit.replacement.state_dict())
        torch.save(
            {
                "layer_index": unit.layer_index,
                "state_dict": state_dict,
                "sparse_index_encoding": index_encoding,
                "parameter_estimate": unit.replacement.parameter_estimate(),
                "teacher_topk_initialization": getattr(
                    unit.replacement,
                    "teacher_topk_diagnostics",
                    {},
                ),
            },
            os.path.join(result.save_dir, f"layer_{unit.layer_index}_replacement.pt"),
        )


def _save_student_model(
    student: nn.Module,
    config: Config,
) -> None:
    """Save an opt-in reconstructible full student state dict."""
    train_cfg = config.training.transformer_replacement
    if not is_transformer_main_process() or not bool(
        getattr(train_cfg, "save_student_model", False)
    ):
        return
    model_cfg = config.model.transformer_replacement
    torch.save(
        {
            "state_dict": student.state_dict(),
            "model_name": str(model_cfg.model_name),
            "replacement_layers": resolve_configured_replacement_layers(config),
            "target_module": str(model_cfg.target_module),
            "student_initialization": str(train_cfg.student_initialization),
            "reconstruction": (
                "Load the base architecture, apply the same experiment YAML "
                "replacement config, then load this state_dict."
            ),
        },
        os.path.join(train_cfg.save_dir, "student_model.pt"),
    )


def _train_tensor_dataset_source(
    units: Sequence[DistillationUnit],
    train_datasets: Sequence[TensorDataset],
    valid_datasets: Sequence[TensorDataset],
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    train_cfg = config.training.transformer_replacement
    if not str(getattr(train_cfg, "resume_checkpoint", "") or ""):
        _calibrate_tensor_replacements(
            units,
            train_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    loaders = [
        _infinite_loader(
            _make_tensor_loader(
                dataset,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                train_cfg=train_cfg,
                device=device,
            )
        )
        for dataset in train_datasets
    ]
    forward_modules = _replacement_forward_modules(units, train_cfg, device)
    trainable_modules = nn.ModuleList(forward_modules)
    optimizer = create_optimizer(
        trainable_modules.parameters(), config.training.main.optimizer
    )
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)

    start_step, history = _restore_layerwise_training(
        config,
        units,
        optimizer,
        scaler,
        device=device,
    )
    if history is None:
        initial_valid = _evaluate_tensor_datasets(
            units,
            valid_datasets,
            batch_size=train_cfg.batch_size,
            device=device,
            dtype=dtype,
            loss_name=train_cfg.loss,
            cosine_weight=train_cfg.cosine_weight,
            forward_modules=forward_modules,
            train_cfg=train_cfg,
        )
        initial_valid = transformer_distributed_mean(initial_valid, device)
        history = ReplacementTrainingHistory.start(initial_valid)
        _save_initial_best_replacement(config, units, history)

    for step in range(start_step + 1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device, dtype=dtype)
        for module, loader in zip(forward_modules, loaders):
            module.train()
            x, target = next(loader)
            x, target = _move_replacement_batch_tensors(
                x,
                target,
                device=device,
                dtype=dtype,
            )
            with autocast("cuda", enabled=scaler.is_enabled()):
                prediction = module(x)
                loss = _compute_distillation_loss(
                    prediction,
                    target,
                    loss_name=train_cfg.loss,
                    cosine_weight=train_cfg.cosine_weight,
                )
            step_loss = step_loss + loss / max(len(units), 1)
        scaler.scale(step_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        apply_indexed_rewiring_after_step(trainable_modules, optimizer)
        history.record_train_loss(
            transformer_distributed_mean(
                float(step_loss.detach().item()),
                device,
            )
        )

        if _should_run_periodic_step(step, train_cfg.eval_every, train_cfg.max_steps):
            valid = _evaluate_tensor_datasets(
                units,
                valid_datasets,
                batch_size=train_cfg.batch_size,
                device=device,
                dtype=dtype,
                loss_name=train_cfg.loss,
                cosine_weight=train_cfg.cosine_weight,
                forward_modules=forward_modules,
                train_cfg=train_cfg,
            )
            valid = transformer_distributed_mean(valid, device)
            _record_valid_and_maybe_save_best(
                config,
                units,
                history,
                step=step,
                valid_loss=valid,
            )
        if _should_run_periodic_step(step, train_cfg.log_every, 1):
            logger.info(
                "transformer replacement distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                step,
                train_cfg.max_steps,
                history.train_losses[-1],
                history.valid_losses[-1],
            )
        _maybe_save_layerwise_training(
            config,
            units,
            optimizer,
            scaler,
            history,
            step=step,
            device=device,
        )

    _restore_best_replacement(config, units, device=device)
    result = TransformerReplacementTrainingResult(
        **history.as_result_kwargs(),
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(
        result,
        units,
        save_replacements=train_cfg.save_replacements,
        topology_encoding=str(train_cfg.replacement_checkpoint_encoding),
    )
    return result


def _evaluate_hf_text(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    tokens: torch.Tensor,
    capture: _MLPIOCapture,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int = 4,
    forward_modules: Sequence[nn.Module] | None = None,
) -> float:
    train_cfg = config.training.transformer_replacement
    input_device = next(teacher.parameters()).device
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch_index in range(max(1, n_batches)):
            capture.clear()
            input_ids = _deterministic_validation_token_batch(
                tokens,
                batch_index=batch_index,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            teacher(input_ids=input_ids)
            for unit, module in zip(units, modules):
                values = capture.records[unit.layer_index]
                x = values["input"].to(device=device, dtype=dtype)
                target = values["target"].to(device=device, dtype=dtype)
                module.eval()
                loss = _compute_distillation_loss(
                    module(x),
                    target,
                    loss_name=train_cfg.loss,
                    cosine_weight=train_cfg.cosine_weight,
                )
                total += float(loss.item())
                count += 1
    return total / max(count, 1)


def _deterministic_validation_token_batch(
    tokens: torch.Tensor,
    *,
    batch_index: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return fixed, non-random validation windows shared by every run cell."""
    max_start = int(tokens.numel()) - int(sequence_length)
    if max_start <= 0:
        raise ValueError("Validation token source is shorter than sequence_length")
    starts = [
        (int(batch_index) * int(batch_size) + offset) * int(sequence_length)
        for offset in range(int(batch_size))
    ]
    return torch.stack(
        [
            tokens[start % max_start : start % max_start + sequence_length]
            for start in starts
        ]
    ).to(device=device)


def _evaluate_hf_text_prediction_metrics(
    teacher: nn.Module,
    units: Sequence[DistillationUnit],
    tokens: torch.Tensor,
    capture: _MLPIOCapture,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_batches: int,
    forward_modules: Sequence[nn.Module] | None = None,
) -> dict[str, dict[str, float]]:
    """Measure held-out layer predictions on deterministic token windows."""
    train_cfg = config.training.transformer_replacement
    input_device = next(teacher.parameters()).device
    modules = (
        list(forward_modules)
        if forward_modules is not None
        else [unit.replacement for unit in units]
    )
    predictions: dict[int, list[torch.Tensor]] = {
        unit.layer_index: [] for unit in units
    }
    targets: dict[int, list[torch.Tensor]] = {unit.layer_index: [] for unit in units}
    with torch.no_grad():
        for batch_index in range(max(1, int(n_batches))):
            capture.clear()
            input_ids = _deterministic_validation_token_batch(
                tokens,
                batch_index=batch_index,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            teacher(input_ids=input_ids)
            for unit, module in zip(units, modules):
                values = capture.records[unit.layer_index]
                x = values["input"].to(device=device, dtype=dtype)
                target = values["target"].to(device=device, dtype=dtype)
                module.eval()
                predictions[unit.layer_index].append(module(x).detach().cpu())
                targets[unit.layer_index].append(target.detach().cpu())

    result = {}
    for unit in units:
        layer_index = unit.layer_index
        prediction = torch.cat(predictions[layer_index], dim=0)
        target = torch.cat(targets[layer_index], dim=0)
        metrics = _compute_prediction_metrics(prediction, target)
        metrics["n_tokens"] = float(prediction.numel() // prediction.shape[-1])
        result[f"layer_{layer_index}"] = metrics
    return result


def _train_hf_text_source(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    teacher, units = _load_hf_teacher_and_units(config, device=device, dtype=dtype)
    train_tokens, valid_tokens = _read_token_source(config)
    capture = _MLPIOCapture(
        teacher,
        [unit.layer_index for unit in units],
        layers_attr=model_cfg.layers_attr,
        mlp_attr=model_cfg.target_module,
    )
    if not str(getattr(train_cfg, "resume_checkpoint", "") or ""):
        _calibrate_hf_replacements(
            teacher,
            units,
            valid_tokens,
            capture,
            config,
            device=device,
            dtype=dtype,
        )
    forward_modules = _replacement_forward_modules(units, train_cfg, device)
    trainable_modules = nn.ModuleList(forward_modules)
    optimizer = create_optimizer(
        trainable_modules.parameters(), config.training.main.optimizer
    )
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)
    input_device = next(teacher.parameters()).device
    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )

    start_step, history = _restore_layerwise_training(
        config,
        units,
        optimizer,
        scaler,
        device=device,
    )
    if history is None:
        initial_valid = _evaluate_hf_text(
            teacher,
            units,
            valid_tokens,
            capture,
            config,
            device=device,
            dtype=dtype,
            n_batches=eval_batches,
            forward_modules=forward_modules,
        )
        initial_valid = transformer_distributed_mean(initial_valid, device)
        history = ReplacementTrainingHistory.start(initial_valid)
        _save_initial_best_replacement(config, units, history)

    try:
        for step in range(start_step + 1, int(train_cfg.max_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            capture.clear()
            input_ids = _sample_token_batch(
                train_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                device=input_device,
            )
            with torch.no_grad():
                teacher(input_ids=input_ids)

            step_loss = torch.zeros((), device=device, dtype=dtype)
            for unit, module in zip(units, forward_modules):
                values = capture.records[unit.layer_index]
                x = values["input"].to(device=device, dtype=dtype)
                target = values["target"].to(device=device, dtype=dtype)
                module.train()
                with autocast("cuda", enabled=scaler.is_enabled()):
                    loss = _compute_distillation_loss(
                        module(x),
                        target,
                        loss_name=train_cfg.loss,
                        cosine_weight=train_cfg.cosine_weight,
                    )
                step_loss = step_loss + loss / max(len(units), 1)

            scaler.scale(step_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            apply_indexed_rewiring_after_step(trainable_modules, optimizer)
            history.record_train_loss(
                transformer_distributed_mean(
                    float(step_loss.detach().item()),
                    device,
                )
            )

            if _should_run_periodic_step(
                step,
                train_cfg.eval_every,
                train_cfg.max_steps,
            ):
                valid = _evaluate_hf_text(
                    teacher,
                    units,
                    valid_tokens,
                    capture,
                    config,
                    device=device,
                    dtype=dtype,
                    n_batches=eval_batches,
                    forward_modules=forward_modules,
                )
                valid = transformer_distributed_mean(valid, device)
                _record_valid_and_maybe_save_best(
                    config,
                    units,
                    history,
                    step=step,
                    valid_loss=valid,
                )
                _save_token_stream_state(train_tokens, train_cfg.save_dir)
            if _should_run_periodic_step(step, train_cfg.log_every, 1):
                logger.info(
                    "hf_text dendritic distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                    step,
                    train_cfg.max_steps,
                    history.train_losses[-1],
                    history.valid_losses[-1],
                )
            _maybe_save_layerwise_training(
                config,
                units,
                optimizer,
                scaler,
                history,
                step=step,
                device=device,
                token_source=train_tokens,
            )
        _restore_best_replacement(config, units, device=device)
        layer_metrics = _evaluate_hf_text_prediction_metrics(
            teacher,
            units,
            valid_tokens,
            capture,
            config,
            device=device,
            dtype=dtype,
            n_batches=eval_batches,
            forward_modules=forward_modules,
        )
    finally:
        capture.close()

    _save_token_stream_state(train_tokens, train_cfg.save_dir)

    result = TransformerReplacementTrainingResult(
        **history.as_result_kwargs(),
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
        layer_metrics=layer_metrics,
    )
    _save_result(
        result,
        units,
        save_replacements=train_cfg.save_replacements,
        topology_encoding=str(train_cfg.replacement_checkpoint_encoding),
    )
    return result


def _train_joint_hf_text_source(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformerReplacementTrainingResult:
    """Train all patched dendritic MLPs through the full LM objective."""
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement

    train_tokens, valid_tokens = _read_token_source(config)
    teacher_needed = bool(
        float(train_cfg.kl_loss_weight) > 0 or float(train_cfg.hidden_loss_weight) > 0
    )
    teacher = (
        _load_hf_causal_lm_for_joint_training(
            config,
            device=device,
            dtype=dtype,
            initialization="pretrained",
        )
        if teacher_needed
        else None
    )
    if teacher is not None:
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    student = _load_hf_causal_lm_for_joint_training(
        config,
        device=device,
        dtype=dtype,
        initialization=str(getattr(train_cfg, "student_initialization", "pretrained")),
    )
    records = apply_transformer_replacement_config(
        student,
        model_cfg,
        config.model.core,
    )
    if not records:
        raise ValueError("No transformer replacement records were created")
    warm_start_dir = str(getattr(train_cfg, "warm_start_dir", "") or "")
    if warm_start_dir:
        logger.info(
            "Loading dendritic replacement warm-start from %s",
            warm_start_dir,
        )
        _load_replacement_record_checkpoints(
            records,
            warm_start_dir,
            device=device,
        )

    train_target = str(getattr(train_cfg, "train_target", "replacement_only"))
    trainable_params = _select_joint_trainable_parameters(
        student,
        records,
        train_target,
    )
    train_full_model = train_target.lower() in {"full_model", "all", "student"}
    use_gradient_checkpointing = bool(
        getattr(train_cfg, "gradient_checkpointing", False)
    )
    if use_gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    if use_gradient_checkpointing or train_full_model:
        student.train()
    else:
        student.eval()
    for record in records:
        record.replacement.train()

    _calibrate_joint_replacements(
        student,
        valid_tokens,
        config,
        device=device,
    )
    student_forward = wrap_transformer_ddp(student, train_cfg, device)

    optimizer = create_optimizer(trainable_params, config.training.main.optimizer)
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)

    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )
    initial_metrics = _evaluate_lm_loss_on_tokens(
        student_forward,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
    )
    initial_valid = float(initial_metrics["lm_loss"])
    initial_valid = transformer_distributed_mean(initial_valid, device)
    history = ReplacementTrainingHistory.start(initial_valid)
    hidden_layers = _resolve_joint_hidden_layers(config)
    input_device = next(student.parameters()).device

    for step in range(1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        input_ids = _sample_token_batch(
            train_tokens,
            batch_size=train_cfg.batch_size,
            sequence_length=train_cfg.sequence_length,
            device=input_device,
        )
        output_hidden = float(train_cfg.hidden_loss_weight) > 0
        teacher_output = None
        if teacher is not None:
            with torch.no_grad(), autocast("cuda", enabled=scaler.is_enabled()):
                teacher_output = teacher(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_hidden_states=output_hidden,
                )

        if use_gradient_checkpointing or train_full_model:
            student.train()
        else:
            student.eval()
        for record in records:
            record.replacement.train()
        with autocast("cuda", enabled=scaler.is_enabled()):
            student_output = student_forward(
                input_ids=input_ids,
                labels=input_ids,
                output_hidden_states=output_hidden,
            )
            loss, components = _joint_lm_distillation_loss(
                student_output,
                teacher_output,
                hidden_layers=hidden_layers,
                lm_loss_weight=float(train_cfg.lm_loss_weight),
                kl_loss_weight=float(train_cfg.kl_loss_weight),
                kl_temperature=float(train_cfg.kl_temperature),
                hidden_loss_weight=float(train_cfg.hidden_loss_weight),
            )

        scaler.scale(loss).backward()
        _clip_replacement_grad_norm(
            trainable_params,
            max_grad_norm=getattr(train_cfg, "max_grad_norm", None),
            scaler=scaler,
            optimizer=optimizer,
        )
        scaler.step(optimizer)
        scaler.update()
        apply_indexed_rewiring_after_step(student, optimizer)
        history.record_train_loss(
            transformer_distributed_mean(
                float(loss.detach().float().item()),
                device,
            )
        )

        if _should_run_periodic_step(step, train_cfg.eval_every, train_cfg.max_steps):
            valid_metrics = _evaluate_lm_loss_on_tokens(
                student_forward,
                valid_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                max_batches=eval_batches,
            )
            valid = float(valid_metrics["lm_loss"])
            valid = transformer_distributed_mean(valid, device)
            history.record_valid_loss(step, valid)
            _save_token_stream_state(train_tokens, train_cfg.save_dir)
            for record in records:
                record.replacement.train()

        if _should_run_periodic_step(step, train_cfg.log_every, 1):
            logger.info(
                (
                    "joint LM distillation step=%s/%s total=%.6f lm=%.6f "
                    "kl=%.6f hidden=%.6f valid_lm=%.6f"
                ),
                step,
                train_cfg.max_steps,
                components["total"],
                components["lm"],
                components["kl"],
                components["hidden"],
                history.valid_losses[-1],
            )

    units = _records_to_units(records)
    _save_token_stream_state(train_tokens, train_cfg.save_dir)
    result = TransformerReplacementTrainingResult(
        **history.as_result_kwargs(),
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(
        result,
        units,
        save_replacements=train_cfg.save_replacements,
        topology_encoding=str(train_cfg.replacement_checkpoint_encoding),
    )
    _save_student_model(student, config)
    return result


def _run_transformer_replacement_training(
    config: Config,
    *,
    distributed_context,
) -> TransformerReplacementTrainingResult:
    """Run the configured transformer replacement distillation experiment."""
    train_cfg = config.training.transformer_replacement
    if not train_cfg.enabled:
        raise ValueError("training.transformer_replacement.enabled must be true")
    mode = str(train_cfg.mode).lower()
    if mode not in {"layerwise_distillation", "joint_lm_distillation"}:
        raise ValueError(
            "mode must be 'layerwise_distillation' or 'joint_lm_distillation'"
        )
    if (
        mode == "layerwise_distillation"
        and str(train_cfg.train_target).lower() != "replacement_only"
    ):
        raise ValueError(
            "layerwise_distillation supports only train_target='replacement_only'; "
            "use mode='joint_lm_distillation' for train_target='full_model'"
        )

    seed = int(train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    seed += int(distributed_context.rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = (
        distributed_context.device
        if distributed_context.enabled
        else _resolve_device(train_cfg.device)
    )
    assert device is not None
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Using float32 for CPU transformer replacement training")
        dtype = torch.float32

    source = str(train_cfg.teacher_source).lower()
    default_layers = [0] if source in {"synthetic_mlp", "hidden_cache"} else []
    replacement_layers = resolve_configured_replacement_layers(
        config,
        default=default_layers,
    )
    if not replacement_layers:
        raise ValueError("At least one transformer replacement layer is required")
    if mode == "joint_lm_distillation":
        if str(getattr(train_cfg, "resume_checkpoint", "") or ""):
            raise ValueError(
                "resume_checkpoint currently supports layerwise distillation; "
                "joint/FSDP checkpointing remains a scale gate"
            )
        if source != "hf_text":
            raise ValueError(
                "joint_lm_distillation currently requires teacher_source='hf_text'"
            )
        return _train_joint_hf_text_source(config, device=device, dtype=dtype)

    if source == "synthetic_mlp":
        units = _make_synthetic_units(config, device=device, dtype=dtype)
        train_datasets, valid_datasets = _build_synthetic_datasets(
            units,
            config,
            device=device,
            dtype=dtype,
        )
        return _train_tensor_dataset_source(
            units,
            train_datasets,
            valid_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    if source == "hidden_cache":
        units, train_datasets, valid_datasets = _build_hidden_cache_units_and_datasets(
            config,
            device=device,
            dtype=dtype,
        )
        return _train_tensor_dataset_source(
            units,
            train_datasets,
            valid_datasets,
            config,
            device=device,
            dtype=dtype,
        )
    if source == "hf_text":
        return _train_hf_text_source(config, device=device, dtype=dtype)
    raise ValueError(
        "teacher_source must be 'synthetic_mlp', 'hidden_cache', or 'hf_text'"
    )


def run_transformer_replacement_training(
    config: Config,
) -> TransformerReplacementTrainingResult:
    """Run a transformer replacement experiment, optionally under torchrun DDP."""
    context = initialize_transformer_distributed(
        config.training.transformer_replacement
    )
    started = time.perf_counter()
    metrics_device = context.device or _resolve_device(
        config.training.transformer_replacement.device
    )
    use_cuda_metrics = metrics_device.type == "cuda"
    if use_cuda_metrics:
        torch.cuda.reset_peak_memory_stats()
    try:
        result = _run_transformer_replacement_training(
            config,
            distributed_context=context,
        )
        result.wall_time_seconds = time.perf_counter() - started
        if use_cuda_metrics:
            result.peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()
            result.peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved()
        if is_transformer_main_process():
            _write_metrics_json(result, output_dir=result.save_dir)
        return result
    finally:
        context.close()
