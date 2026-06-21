"""Forward-hook recorders for local credit assignment."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_pathways import (
    RECORDER_TOPK_PATHS,
    TopKGradientPath,
    iter_topk_path_modules,
)
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_modules_of_type,
    iter_named_modules_of_type,
    register_hook_groups,
)

logger = logging.getLogger(__name__)


def _population_label(module_name: str) -> str:
    return (
        "explicit_inhibitory" if ".inhibitory_cells." in module_name else "excitatory"
    )


def _branch_record(module_name: str, module: DendriticBranchLayer) -> dict[str, Any]:
    return {
        "layer": module,
        "module_name": module_name,
        "population": _population_label(module_name),
    }


def _record_topk_forward(
    rec: dict[str, Any],
    path: TopKGradientPath,
    layer: TopKLinear,
    inputs,
    outputs,
) -> None:
    """Record one TopK pathway forward pass for local learning."""
    rec[path.input_key] = inputs[0].detach()
    rec[path.output_key] = outputs.detach()
    rec[path.module_key] = layer
    try:
        cached_mask = getattr(layer, "_last_forward_weight_mask", None)
        if isinstance(cached_mask, torch.Tensor):
            rec[path.mask_key] = cached_mask.detach()
        else:
            rec[path.mask_key] = layer.weight_mask().detach()
    except Exception:
        rec[path.mask_key] = None


class LocalLearningRecorderMixin(ForwardHookRemovalMixin):
    def _attach_local_recorders(
        self, model: BaseModel
    ) -> tuple[list[dict[str, Any]], list[torch.utils.hooks.RemovableHandle]]:
        """Attach forward hooks to collect local signals per DendriticBranchLayer.

        Returns a tuple of (records, handles). Each record contains:
            - layer: module reference
            - v_n: pre-activation compartment voltage (inputs to reactivation)
            - v_out: post-reactivation output
            - x_exc, exc_out, exc_module, exc_mask (if present)
            - x_inh, inh_out, inh_module, inh_mask (if present)
            - x_rec_exc, rec_exc_out, rec_exc_module, rec_exc_mask (if present)
            - x_rec_inh, rec_inh_out, rec_inh_module, rec_inh_mask (if present)
            - x_blk (reshaped input), blk_out, blk_module (if present)
        """
        records: list[dict[str, Any]] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []

        # Cache the decoder input so local rules can map output error back to
        # soma space through the decoder Jacobian when the decoder is nonlinear.
        self._decoder_cache.clear()

        try:
            if hasattr(model, "decoder_network"):
                self._attach_decoder_recorders(model.decoder_network, handles)
            records.extend(self._attach_branch_recorders(model, handles))
            return records, handles
        except Exception:
            self.remove_forward_hooks(handles)
            raise

    def _attach_decoder_recorders(
        self,
        decoder_network: nn.Module,
        handles: list[torch.utils.hooks.RemovableHandle],
    ) -> None:
        def _decoder_pre_hook(mod, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                self._decoder_cache["decoder_input"] = inputs[0]

        handles.append(decoder_network.register_forward_pre_hook(_decoder_pre_hook))

        # Hook decoder input for local decoder updates (capture input to the last
        # Linear specifically).
        for module in iter_modules_of_type(decoder_network, nn.Linear):
            handles.append(self._register_decoder_linear_hook(module))

    def _register_decoder_linear_hook(
        self,
        module: nn.Linear,
    ) -> torch.utils.hooks.RemovableHandle:
        def _dec_in_hook(mod, inputs, outputs):
            self._decoder_cache["module"] = mod
            self._decoder_cache["input"] = inputs[0].detach()

        return module.register_forward_hook(_dec_in_hook)

    def _attach_branch_recorders(
        self,
        model: BaseModel,
        handles: list[torch.utils.hooks.RemovableHandle],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for module_name, module in iter_named_modules_of_type(
            model, DendriticBranchLayer
        ):
            rec = _branch_record(module_name, module)
            handles.append(self._register_reactivation_hook(module, rec))
            handles.extend(self._register_topk_hooks(module, rec))
            block_hook = self._register_block_output_hook(module, rec)
            if block_hook is not None:
                handles.append(block_hook)
            records.append(rec)
        return records

    def _register_reactivation_hook(
        self,
        module: DendriticBranchLayer,
        rec: dict[str, Any],
    ) -> torch.utils.hooks.RemovableHandle:
        def _react_hook(m, inputs, outputs, rec=rec):
            try:
                rec["v_n"] = inputs[0].detach()
            except Exception:
                rec["v_n"] = outputs.detach() if torch.is_tensor(outputs) else None
            rec["v_out"] = outputs.detach() if torch.is_tensor(outputs) else None

        return module.reactivation.register_forward_hook(_react_hook)

    def _register_topk_hooks(
        self,
        module: DendriticBranchLayer,
        rec: dict[str, Any],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        return register_hook_groups(
            iter_topk_path_modules(module, RECORDER_TOPK_PATHS),
            lambda path_layer: [
                self._register_topk_hook(rec, path_layer[0], path_layer[1])
            ],
        )

    def _register_topk_hook(
        self,
        rec: dict[str, Any],
        path: TopKGradientPath,
        topk_layer: TopKLinear,
    ) -> torch.utils.hooks.RemovableHandle:
        def _topk_hook(m, inputs, outputs, rec=rec, layer=topk_layer, path=path):
            _record_topk_forward(rec, path, layer, inputs, outputs)

        return topk_layer.register_forward_hook(_topk_hook)

    def _register_block_output_hook(
        self,
        module: DendriticBranchLayer,
        rec: dict[str, Any],
    ) -> torch.utils.hooks.RemovableHandle | None:
        blk_layer = getattr(module, "branches_to_output", None)
        if blk_layer is None:
            return None

        def _blk_hook(m, inputs, outputs, rec=rec, layer=blk_layer):
            rec["x_blk_raw"] = inputs[0].detach()
            rec["blk_out"] = outputs.detach()
            rec["blk_module"] = layer

        return blk_layer.register_forward_hook(_blk_hook)


__all__ = ["LocalLearningRecorderMixin"]
