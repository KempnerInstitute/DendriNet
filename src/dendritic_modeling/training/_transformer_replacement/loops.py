"""Training loops for transformer replacement experiments."""

from __future__ import annotations

import logging
import os
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
from dendritic_modeling.training._transformer_replacement.checkpoints import (
    _load_replacement_record_checkpoints,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    TransformerReplacementTrainingResult,
    _compute_distillation_loss,
    _make_tensor_loader,
    _replacement_forward_modules,
    _resolve_device,
    _resolve_dtype,
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
from dendritic_modeling.training._transformer_replacement.evaluation import (
    _evaluate_lm_loss_on_tokens,
    _evaluate_tensor_datasets,
)
from dendritic_modeling.training._transformer_replacement.joint import (
    _joint_lm_distillation_loss,
    _resolve_joint_hidden_layers,
    _select_joint_trainable_parameters,
)
from dendritic_modeling.training.optimizers import create_optimizer
from dendritic_modeling.training.replacement_common import (
    _infinite_loader,
    _make_cuda_grad_scaler,
    _write_metrics_json,
)

logger = logging.getLogger(__name__)


def _save_result(
    result: TransformerReplacementTrainingResult,
    units: Sequence[DistillationUnit],
    *,
    save_replacements: bool,
) -> None:
    _write_metrics_json(result)
    if not save_replacements:
        return
    for unit in units:
        torch.save(
            {
                "layer_index": unit.layer_index,
                "state_dict": unit.replacement.state_dict(),
                "parameter_estimate": unit.replacement.parameter_estimate(),
            },
            os.path.join(result.save_dir, f"layer_{unit.layer_index}_replacement.pt"),
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
    best_valid = initial_valid
    best_step = 0
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]

    for step in range(1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device, dtype=dtype)
        for module, loader in zip(forward_modules, loaders):
            module.train()
            x, target = next(loader)
            x = x.to(device=device, dtype=dtype)
            target = target.to(device=device, dtype=dtype)
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
        train_losses.append(float(step_loss.detach().item()))

        if step % int(train_cfg.eval_every) == 0 or step == int(train_cfg.max_steps):
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
            valid_losses.append(valid)
            if valid < best_valid:
                best_valid = valid
                best_step = step
        if step % int(train_cfg.log_every) == 0 or step == 1:
            logger.info(
                "transformer replacement distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                step,
                train_cfg.max_steps,
                train_losses[-1],
                valid_losses[-1],
            )

    result = TransformerReplacementTrainingResult(
        train_losses=train_losses,
        valid_losses=valid_losses,
        initial_valid_loss=initial_valid,
        final_valid_loss=valid_losses[-1],
        best_valid_loss=best_valid,
        best_step=best_step,
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(result, units, save_replacements=train_cfg.save_replacements)
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
        for _ in range(max(1, n_batches)):
            capture.clear()
            input_ids = _sample_token_batch(
                tokens,
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
    forward_modules = _replacement_forward_modules(units, train_cfg, device)
    trainable_modules = nn.ModuleList(forward_modules)
    optimizer = create_optimizer(
        trainable_modules.parameters(), config.training.main.optimizer
    )
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)
    input_device = next(teacher.parameters()).device

    initial_valid = _evaluate_hf_text(
        teacher,
        units,
        valid_tokens,
        capture,
        config,
        device=device,
        dtype=dtype,
        forward_modules=forward_modules,
    )
    best_valid = initial_valid
    best_step = 0
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]

    try:
        for step in range(1, int(train_cfg.max_steps) + 1):
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
            train_losses.append(float(step_loss.detach().item()))

            if step % int(train_cfg.eval_every) == 0 or step == int(
                train_cfg.max_steps
            ):
                valid = _evaluate_hf_text(
                    teacher,
                    units,
                    valid_tokens,
                    capture,
                    config,
                    device=device,
                    dtype=dtype,
                    forward_modules=forward_modules,
                )
                valid_losses.append(valid)
                if valid < best_valid:
                    best_valid = valid
                    best_step = step
            if step % int(train_cfg.log_every) == 0 or step == 1:
                logger.info(
                    "hf_text dendritic distillation step=%s/%s train_loss=%.6f valid_loss=%.6f",
                    step,
                    train_cfg.max_steps,
                    train_losses[-1],
                    valid_losses[-1],
                )
    finally:
        capture.close()

    result = TransformerReplacementTrainingResult(
        train_losses=train_losses,
        valid_losses=valid_losses,
        initial_valid_loss=initial_valid,
        final_valid_loss=valid_losses[-1],
        best_valid_loss=best_valid,
        best_step=best_step,
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(result, units, save_replacements=train_cfg.save_replacements)
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
        _load_hf_causal_lm_for_joint_training(config, device=device, dtype=dtype)
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

    optimizer = create_optimizer(trainable_params, config.training.main.optimizer)
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)

    eval_batches = max(
        1, int(train_cfg.valid_samples) // max(1, int(train_cfg.batch_size))
    )
    initial_metrics = _evaluate_lm_loss_on_tokens(
        student,
        valid_tokens,
        batch_size=train_cfg.batch_size,
        sequence_length=train_cfg.sequence_length,
        max_batches=eval_batches,
    )
    initial_valid = float(initial_metrics["lm_loss"])
    best_valid = initial_valid
    best_step = 0
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]
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
            student_output = student(
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
        max_grad_norm = getattr(train_cfg, "max_grad_norm", None)
        if max_grad_norm is not None and float(max_grad_norm) > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, float(max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(float(loss.detach().float().item()))

        if step % int(train_cfg.eval_every) == 0 or step == int(train_cfg.max_steps):
            valid_metrics = _evaluate_lm_loss_on_tokens(
                student,
                valid_tokens,
                batch_size=train_cfg.batch_size,
                sequence_length=train_cfg.sequence_length,
                max_batches=eval_batches,
            )
            valid = float(valid_metrics["lm_loss"])
            valid_losses.append(valid)
            if valid < best_valid:
                best_valid = valid
                best_step = step
            for record in records:
                record.replacement.train()

        if step % int(train_cfg.log_every) == 0 or step == 1:
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
                valid_losses[-1],
            )

    units = _records_to_units(records)
    result = TransformerReplacementTrainingResult(
        train_losses=train_losses,
        valid_losses=valid_losses,
        initial_valid_loss=initial_valid,
        final_valid_loss=valid_losses[-1],
        best_valid_loss=best_valid,
        best_step=best_step,
        save_dir=train_cfg.save_dir,
        layer_indices=[unit.layer_index for unit in units],
    )
    _save_result(result, units, save_replacements=train_cfg.save_replacements)
    return result


def run_transformer_replacement_training(
    config: Config,
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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(train_cfg.device)
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Using float32 for CPU transformer replacement training")
        dtype = torch.float32

    source = str(train_cfg.teacher_source).lower()
    if mode == "joint_lm_distillation":
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
