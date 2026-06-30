"""Joint language-model training losses and parameter policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.config import Config
from dendritic_modeling.training.replacement_common import (
    _capture_requires_grad_states,
    _restore_requires_grad_states,
)


def _freeze_except_replacements(
    model: nn.Module,
    records: Sequence[Any],
) -> list[nn.Parameter]:
    states = _capture_requires_grad_states(model.parameters())
    try:
        for param in model.parameters():
            param.requires_grad_(False)
        trainable: list[nn.Parameter] = []
        seen: set[int] = set()
        for record in records:
            for param in record.replacement.parameters():
                param.requires_grad_(True)
                param_id = id(param)
                if param_id not in seen:
                    seen.add(param_id)
                    trainable.append(param)
        if not trainable:
            raise ValueError("No replacement parameters are trainable")
        return trainable
    except Exception:
        _restore_requires_grad_states(states)
        raise


def _select_joint_trainable_parameters(
    model: nn.Module,
    records: Sequence[Any],
    train_target: str,
) -> list[nn.Parameter]:
    normalized = str(train_target).lower()
    if normalized in {"replacement_only", "replacements_only", "dendritic_only"}:
        return _freeze_except_replacements(model, records)
    if normalized in {"full_model", "all", "student"}:
        params: list[nn.Parameter] = []
        for param in model.parameters():
            param.requires_grad_(True)
            params.append(param)
        if not params:
            raise ValueError("No student parameters are trainable")
        return params
    raise ValueError(
        "joint_lm_distillation train_target must be 'replacement_only' or "
        "'full_model'"
    )


def _resolve_joint_hidden_layers(
    config: Config,
) -> list[int]:
    train_cfg = config.training.transformer_replacement
    configured = list(getattr(train_cfg, "hidden_loss_layers", []) or [])
    if configured:
        return [int(layer) for layer in configured]
    return [
        int(layer)
        for layer in (
            train_cfg.layers or config.model.transformer_replacement.layers or []
        )
    ]


def _joint_lm_distillation_loss(
    student_output: Any,
    teacher_output: Any | None,
    *,
    hidden_layers: Sequence[int],
    lm_loss_weight: float,
    kl_loss_weight: float,
    kl_temperature: float,
    hidden_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if student_output.loss is None:
        raise ValueError("Student output did not include a language-modeling loss")

    total = student_output.loss * float(lm_loss_weight)
    components = {
        "lm": float(student_output.loss.detach().float().item()),
        "kl": 0.0,
        "hidden": 0.0,
    }

    if kl_loss_weight > 0:
        if teacher_output is None:
            raise ValueError("KL loss requires teacher_output")
        temperature = max(float(kl_temperature), 1e-6)
        student_logits = student_output.logits[:, :-1, :].float() / temperature
        teacher_logits = teacher_output.logits[:, :-1, :].float() / temperature
        vocab = student_logits.shape[-1]
        student_flat = student_logits.reshape(-1, vocab)
        teacher_flat = teacher_logits.reshape(-1, vocab)
        kl = F.kl_div(
            F.log_softmax(student_flat, dim=-1),
            F.softmax(teacher_flat, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        total = total + float(kl_loss_weight) * kl
        components["kl"] = float(kl.detach().item())

    if hidden_loss_weight > 0:
        if teacher_output is None:
            raise ValueError("Hidden loss requires teacher_output")
        if student_output.hidden_states is None or teacher_output.hidden_states is None:
            raise ValueError("Hidden loss requires output_hidden_states=True")
        hidden_terms = []
        for layer in hidden_layers:
            hidden_index = int(layer) + 1
            hidden_terms.append(
                F.mse_loss(
                    student_output.hidden_states[hidden_index].float(),
                    teacher_output.hidden_states[hidden_index].float(),
                )
            )
        if hidden_terms:
            hidden_loss = torch.stack(hidden_terms).mean()
            total = total + float(hidden_loss_weight) * hidden_loss
            components["hidden"] = float(hidden_loss.detach().item())

    components["total"] = float(total.detach().float().item())
    return total, components
