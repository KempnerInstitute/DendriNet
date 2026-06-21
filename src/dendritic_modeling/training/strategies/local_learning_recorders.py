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

logger = logging.getLogger(__name__)


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


class LocalLearningRecorderMixin:
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

        if hasattr(model, "decoder_network"):

            def _decoder_pre_hook(mod, inputs):

                if inputs and isinstance(inputs[0], torch.Tensor):

                    self._decoder_cache["decoder_input"] = inputs[0]

            handles.append(
                model.decoder_network.register_forward_pre_hook(_decoder_pre_hook)
            )

            # Hook decoder input for local decoder updates (capture input to

            # the last Linear specifically).

            for m in model.decoder_network.modules():

                if isinstance(m, nn.Linear):

                    def _dec_in_hook(mod, inputs, outputs):

                        self._decoder_cache["module"] = mod

                        self._decoder_cache["input"] = inputs[0].detach()

                    handle = m.register_forward_hook(_dec_in_hook)

                    handles.append(handle)

        for module_name, module in model.named_modules():

            if not isinstance(module, DendriticBranchLayer):

                continue

            population = (
                "explicit_inhibitory"
                if ".inhibitory_cells." in module_name
                else "excitatory"
            )

            rec: dict[str, Any] = {
                "layer": module,
                "module_name": module_name,
                "population": population,
            }

            # Reactivation: capture pre-activation (voltage) and output

            def _react_hook(m, inputs, outputs, rec=rec):

                try:

                    rec["v_n"] = inputs[0].detach()

                except Exception:

                    rec["v_n"] = outputs.detach() if torch.is_tensor(outputs) else None

                rec["v_out"] = outputs.detach() if torch.is_tensor(outputs) else None

            handles.append(module.reactivation.register_forward_hook(_react_hook))

            for path, topk_layer in iter_topk_path_modules(module, RECORDER_TOPK_PATHS):

                def _topk_hook(
                    m, inputs, outputs, rec=rec, layer=topk_layer, path=path
                ):
                    _record_topk_forward(rec, path, layer, inputs, outputs)

                handles.append(topk_layer.register_forward_hook(_topk_hook))

            # Dendritic BlockLinear

            if getattr(module, "branches_to_output", None) is not None:

                blk_layer = module.branches_to_output

                def _blk_hook(m, inputs, outputs, rec=rec, layer=blk_layer):

                    rec["x_blk_raw"] = inputs[0].detach()

                    rec["blk_out"] = outputs.detach()

                    rec["blk_module"] = layer

                handles.append(blk_layer.register_forward_hook(_blk_hook))

            records.append(rec)

        return records, handles


__all__ = ["LocalLearningRecorderMixin"]
