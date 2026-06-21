"""Mixin methods for :mod:`dendritic_modeling.training.strategies.local_learning`.

This module is a mechanical split of ``LocalCreditAssignment`` helpers. Method
bodies are kept behavior-compatible with the original trainer; the main trainer
class owns orchestration and mutable state.
"""

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

logger = logging.getLogger(__name__)


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

            # Excitatory TopK

            if getattr(module, "branch_excitation", None) is not None:

                exc_layer: TopKLinear = module.branch_excitation

                def _exc_hook(m, inputs, outputs, rec=rec, layer=exc_layer):

                    rec["x_exc"] = inputs[0].detach()

                    rec["exc_out"] = outputs.detach()

                    rec["exc_module"] = layer

                    try:

                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)

                        if isinstance(cached_mask, torch.Tensor):

                            rec["exc_mask"] = cached_mask.detach()

                        else:

                            rec["exc_mask"] = layer.weight_mask().detach()

                    except Exception:

                        rec["exc_mask"] = None

                handles.append(exc_layer.register_forward_hook(_exc_hook))

            # Inhibitory TopK

            if getattr(module, "branch_inhibition", None) is not None:

                inh_layer: TopKLinear = module.branch_inhibition

                def _inh_hook(m, inputs, outputs, rec=rec, layer=inh_layer):

                    rec["x_inh"] = inputs[0].detach()

                    rec["inh_out"] = outputs.detach()

                    rec["inh_module"] = layer

                    try:

                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)

                        if isinstance(cached_mask, torch.Tensor):

                            rec["inh_mask"] = cached_mask.detach()

                        else:

                            rec["inh_mask"] = layer.weight_mask().detach()

                    except Exception:

                        rec["inh_mask"] = None

                handles.append(inh_layer.register_forward_hook(_inh_hook))

            # Recurrent excitatory TopK

            if getattr(module, "branch_recurrent", None) is not None:

                rec_exc_layer: TopKLinear = module.branch_recurrent

                def _rec_exc_hook(m, inputs, outputs, rec=rec, layer=rec_exc_layer):

                    rec["x_rec_exc"] = inputs[0].detach()

                    rec["rec_exc_out"] = outputs.detach()

                    rec["rec_exc_module"] = layer

                    try:

                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)

                        if isinstance(cached_mask, torch.Tensor):

                            rec["rec_exc_mask"] = cached_mask.detach()

                        else:

                            rec["rec_exc_mask"] = layer.weight_mask().detach()

                    except Exception:

                        rec["rec_exc_mask"] = None

                handles.append(rec_exc_layer.register_forward_hook(_rec_exc_hook))

            # Recurrent inhibitory TopK

            if getattr(module, "branch_rec_inhibition", None) is not None:

                rec_inh_layer: TopKLinear = module.branch_rec_inhibition

                def _rec_inh_hook(m, inputs, outputs, rec=rec, layer=rec_inh_layer):

                    rec["x_rec_inh"] = inputs[0].detach()

                    rec["rec_inh_out"] = outputs.detach()

                    rec["rec_inh_module"] = layer

                    try:

                        cached_mask = getattr(layer, "_last_forward_weight_mask", None)

                        if isinstance(cached_mask, torch.Tensor):

                            rec["rec_inh_mask"] = cached_mask.detach()

                        else:

                            rec["rec_inh_mask"] = layer.weight_mask().detach()

                    except Exception:

                        rec["rec_inh_mask"] = None

                handles.append(rec_inh_layer.register_forward_hook(_rec_inh_hook))

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
