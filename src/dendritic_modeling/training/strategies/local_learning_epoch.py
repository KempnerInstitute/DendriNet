"""Epoch-level helpers for local learning training."""

from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.utils.hooks import remove_hook_handles


def _has_dendritic_branch_layers(model: BaseModel) -> bool:
    try:
        return any(isinstance(m, DendriticBranchLayer) for m in model.modules())
    except Exception:
        return False


def _supports_local_learning_train(model: BaseModel) -> bool:
    has_ei = hasattr(model, "core_network") and isinstance(
        model.core_network, ExcitationInhibitionNetwork
    )
    return has_ei or _has_dendritic_branch_layers(model)


def _resolve_epoch_update_modes(
    local_cfg,
    epoch_counter: int,
) -> tuple[bool, str, str]:
    update_reactivation = local_cfg.update_reactivation
    encoder_update_mode = local_cfg.encoder_update_mode
    decoder_update_mode = local_cfg.decoder_update_mode

    if epoch_counter <= max(0, int(local_cfg.freeze_encoder_epochs)):
        encoder_update_mode = "none"
    if epoch_counter <= max(0, int(local_cfg.freeze_reactivation_epochs)):
        update_reactivation = False
    if epoch_counter <= max(0, int(local_cfg.freeze_decoder_epochs)):
        decoder_update_mode = "none"

    if local_cfg.training_schedule:
        for sched in local_cfg.training_schedule:
            try:
                start_epoch = int(sched.get("start_epoch", 1))
                end_epoch = int(sched.get("end_epoch", 10**9))
            except Exception:
                start_epoch, end_epoch = 1, 10**9
            if start_epoch <= epoch_counter <= end_epoch:
                if "update_reactivation" in sched:
                    update_reactivation = bool(sched["update_reactivation"])
                if "encoder_update_mode" in sched:
                    encoder_update_mode = str(sched["encoder_update_mode"]).lower()
                if "decoder_update_mode" in sched:
                    decoder_update_mode = str(sched["decoder_update_mode"]).lower()
                break

    return update_reactivation, encoder_update_mode, decoder_update_mode


def _collect_topk_modules(model: BaseModel) -> list[TopKLinear]:
    return [m for m in model.modules() if isinstance(m, TopKLinear)]


def _enable_topk_mask_cache(topk_modules: list[TopKLinear]) -> None:
    for module in topk_modules:
        module.cache_mask = True


def _clear_topk_mask_cache(topk_modules: list[TopKLinear]) -> None:
    for module in topk_modules:
        module.cache_mask = False
        module._last_forward_weight_mask = None


def _remove_local_recorders(
    trainer: Any,
    handles: list[torch.utils.hooks.RemovableHandle],
) -> None:
    remove_forward_hooks = getattr(trainer, "remove_forward_hooks", None)
    if callable(remove_forward_hooks):
        remove_forward_hooks(handles)
    else:
        remove_hook_handles(handles)


def _prepare_hsic_target(
    local_cfg,
    y_batch: torch.Tensor,
    y_hat: torch.Tensor,
) -> Optional[torch.Tensor]:
    if not local_cfg.hsic.enabled:
        return None

    if local_cfg.hsic.target_source == "labels":
        if y_batch.dim() == 1 or (y_batch.dim() == 2 and y_batch.size(1) == 1):
            num_classes = y_hat.size(-1)
            return torch.nn.functional.one_hot(
                y_batch.view(-1),
                num_classes=num_classes,
            ).to(y_hat.dtype)
        return y_batch.to(y_hat.dtype)

    return y_hat.detach()


def _zero_optimizer_gradients(optimizer: torch.optim.Optimizer) -> None:
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()


def _collect_backprop_param_groups(
    model: BaseModel,
    encoder_update_mode: str,
    decoder_update_mode: str,
) -> list[list[torch.nn.Parameter]]:
    backprop_param_groups: list[list[torch.nn.Parameter]] = []
    if (
        encoder_update_mode == "backprop"
        and hasattr(model, "encoder_network")
        and model.encoder_network is not None
    ):
        enc_params = [p for p in model.encoder_network.parameters() if p.requires_grad]
        if enc_params:
            backprop_param_groups.append(enc_params)

    if hasattr(model, "decoder_network") and decoder_update_mode == "backprop":
        dec_params = [p for p in model.decoder_network.parameters() if p.requires_grad]
        if dec_params:
            backprop_param_groups.append(dec_params)

    return backprop_param_groups


def _accumulate_backprop_gradients(
    loss: torch.Tensor,
    backprop_param_groups: list[list[torch.nn.Parameter]],
) -> None:
    for group_idx, params in enumerate(backprop_param_groups):
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=group_idx < len(backprop_param_groups) - 1,
            allow_unused=True,
        )
        for param, grad_value in zip(params, grads):
            if grad_value is None:
                continue
            grad = grad_value.detach()
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad = param.grad + grad


def _apply_decoder_local_gradients(
    lin_mod: Any,
    h_in: Any,
    delta_out: torch.Tensor,
    normalize_by_batch: bool,
) -> None:
    if not isinstance(lin_mod, nn.Linear) or not isinstance(h_in, torch.Tensor):
        return

    e_out = delta_out.detach()  # [B, out]
    if normalize_by_batch and e_out.size(0) > 0:
        scale = 1.0 / float(e_out.size(0))
    else:
        scale = 1.0

    grad_w = scale * (e_out.t() @ h_in.detach())  # [out, in]
    if lin_mod.weight.grad is None:
        lin_mod.weight.grad = grad_w.to(lin_mod.weight.dtype)
    else:
        lin_mod.weight.grad = lin_mod.weight.grad + grad_w.to(lin_mod.weight.dtype)

    if lin_mod.bias is not None:
        grad_b = scale * e_out.sum(dim=0)
        if lin_mod.bias.grad is None:
            lin_mod.bias.grad = grad_b.to(lin_mod.bias.dtype)
        else:
            lin_mod.bias.grad = lin_mod.bias.grad + grad_b.to(lin_mod.bias.dtype)


def _clip_model_gradients(model: BaseModel, clip_grad_value: Any) -> None:
    if clip_grad_value and clip_grad_value > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)


def _forward_with_local_recorders(
    trainer: Any,
    model: BaseModel,
    x_batch: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    layer_records, handles = trainer._attach_local_recorders(model)
    topk_modules = _collect_topk_modules(model)
    _enable_topk_mask_cache(topk_modules)
    try:
        y_hat: torch.Tensor = model(x_batch)
    finally:
        _remove_local_recorders(trainer, handles)
        _clear_topk_mask_cache(topk_modules)

    return y_hat, layer_records


def _resolve_stdp_error_signals(
    local_cfg: Any,
    e_n: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    stdp_error_signal = e_n
    if str(getattr(local_cfg, "rule_variant", "")).lower() == "stdp":
        e_n = torch.zeros_like(e_n)
    return e_n, stdp_error_signal


def _apply_path_propagation_factor(
    local_cfg: Any,
    broadcast_state: Any,
    rec: dict[str, Any],
    e_n: torch.Tensor,
) -> torch.Tensor:
    if (
        local_cfg.morphology_aware.use_path_propagation
        and broadcast_state.mode != "path_transport"
    ):
        path_factor = rec.get("path_factor", 1.0)
        if isinstance(path_factor, torch.Tensor):
            return e_n * path_factor
    return e_n
