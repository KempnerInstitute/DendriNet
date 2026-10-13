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
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.networks.structure import inhibitory_branch_layer_ids
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_pathways import (
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


def reject_unsupported_branch_layer(name: str, module: DendriticBranchLayer) -> None:
    """Raise for branch layers whose dynamics the local rules do not model."""
    if (
        module.topk_type == "indexed_release"
        or module.learnable_leak
        or module.leak_conductance != 1
        or type(module.reactivation).__name__ == "EndpointNormalizedTanh"
    ):
        raise ValueError(
            "analytic local updates do not support indexed_release synapses, a "
            f"configurable leak or endpoint_tanh at {name!r}; use backpropagation"
        )
    if not module.use_shunting and module.additive_mode != "raw":
        raise ValueError(
            "analytic local updates do not implement derivatives for "
            f"additive_mode={module.additive_mode!r} at {name!r}. "
            "Use backpropagation for tangent_matched/conductance_normalized "
            "controls; the local additive rule supports additive_mode='raw' "
            "including its legacy feature normalization."
        )


def reject_mismatched_dynamics_mode(
    model: nn.Module, dynamics_mode: str, *, allow_mismatch: bool = False
) -> None:
    """Raise when a forced three_factor.dynamics_mode disagrees with a layer.

    ``allow_mismatch`` (three_factor.allow_mismatched_dynamics_mode) accepts
    the mismatch as a deliberate rule variant.
    """
    if dynamics_mode == "auto" or allow_mismatch:
        return
    for name, layer in iter_named_modules_of_type(model, DendriticBranchLayer):
        integration = "conductance" if layer.use_shunting else "additive"
        if integration != dynamics_mode:
            raise ValueError(
                f"three_factor.dynamics_mode {dynamics_mode!r} would apply "
                f"{dynamics_mode} derivatives to {name}, which integrates "
                f"{'by shunting' if layer.use_shunting else 'additively'}, so its "
                "updates would not follow its forward pass; use 'auto', or set "
                "three_factor.allow_mismatched_dynamics_mode to apply them as a "
                "rule variant"
            )


def reject_unsupported_forward_paths(model: nn.Module, *, allow: bool = False) -> None:
    """Raise for cores whose forward equations the local rules do not model.

    ``allow`` (learning_strategy_config.allow_unmodeled_forward_features)
    applies the rules anyway, as a rule variant, and logs the features.
    """
    for name, module in model.named_modules():
        if not isinstance(module, StatefulDendriNet):
            continue
        features = [
            label
            for label, enabled in (
                ("typed synapses", module.synapse_types_enabled),
                ("a spiking soma", module.spiking_soma is not None),
                ("dendritic spikes", module.dendritic_spikes_enabled),
                ("soma feedback", module.soma_feedback_enabled),
                ("cross_level_mode 'parallel_readout'", module.is_parallel_readout),
                ("cross_level_mode 'all_active_star'", module.is_all_active_star),
            )
            if enabled
        ]
        if features and allow:
            logger.warning(
                "local_ca applies its rules to %s although it uses %s, whose "
                "dynamics the rules do not include "
                "(allow_unmodeled_forward_features).",
                name,
                ", ".join(features),
            )
        elif features:
            raise ValueError(
                "local_ca derives its updates from the branch-layer forward "
                f"equations, but {name} uses {', '.join(features)}, whose "
                "dynamics the local rules do not include; use backpropagation "
                "for this core, or set learning_strategy_config."
                "allow_unmodeled_forward_features: true to apply the rules as a "
                "variant that ignores them"
            )


_SYNAPSE_BANKS = {
    "exc": "branch_excitation",
    "inh": "branch_inhibition",
    "rec_exc": "branch_recurrent",
    "rec_inh": "branch_rec_inhibition",
}


def reject_unsupported_synapses(model: nn.Module, *, stdp_pathways: set[str]) -> None:
    """Raise for synapse banks whose local updates the rules do not implement.

    Row weight normalization couples the weights of a row, while the local
    rules apply the elementwise weight-transform derivative; STDP updates
    TopK banks only.
    """
    for name, layer in model.named_modules():
        if not isinstance(layer, DendriticBranchLayer):
            continue
        for pathway, attribute in _SYNAPSE_BANKS.items():
            bank = getattr(layer, attribute, None)
            if bank is None:
                continue
            if getattr(bank, "weight_norm_order", None) is not None:
                raise ValueError(
                    f"{name}.{attribute} normalizes its weight rows "
                    "(sparsity.weight_norm_order), which the local rules do not "
                    "model; set weight_norm_order to null or use backpropagation "
                    "or dfa"
                )
            if pathway in stdp_pathways and not isinstance(bank, TopKLinear):
                raise ValueError(
                    f"STDP updates TopK synapse banks, but {name}.{attribute} is a "
                    f"{type(bank).__name__}; remove {pathway!r} from stdp.apply_to "
                    "or use a TopK synapse type"
                )


def _branch_record(
    module_name: str, module: DendriticBranchLayer, inhibitory: set[int]
) -> dict[str, Any]:
    return {
        "layer": module,
        "module_name": module_name,
        "population": (
            "explicit_inhibitory" if id(module) in inhibitory else "excitatory"
        ),
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
        connection_indices = getattr(layer, "connection_indices", None)
        pre_weight = getattr(layer, "pre_w", None)
        if (
            isinstance(connection_indices, torch.Tensor)
            and isinstance(pre_weight, torch.Tensor)
            and connection_indices.shape == pre_weight.shape
        ):
            # Every compact parameter is active. A dense fixed-support mask can
            # be hundreds of GiB and is neither needed nor safe to construct.
            rec[path.mask_key] = None
            return
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
            component_model = self._unwrap_model(model)
            if hasattr(component_model, "decoder_network"):
                handles.extend(
                    self._attach_decoder_recorders(component_model.decoder_network)
                )
            branch_records, branch_handles = self._attach_branch_recorders(model)
            records.extend(branch_records)
            handles.extend(branch_handles)
            return records, handles
        except Exception:
            self.remove_forward_hooks(handles)
            raise

    def _attach_decoder_recorders(
        self,
        decoder_network: nn.Module,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _decoder_pre_hook(mod, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                self._decoder_cache["decoder_input"] = inputs[0]

        def _register_decoder_hook(target: nn.Module | None):
            if target is None:
                return [decoder_network.register_forward_pre_hook(_decoder_pre_hook)]
            return [self._register_decoder_linear_hook(target)]

        # Hook decoder input for local decoder updates (capture input to the last
        # Linear specifically).
        return register_hook_groups(
            (None, *iter_modules_of_type(decoder_network, nn.Linear)),
            _register_decoder_hook,
        )

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
    ) -> tuple[
        list[dict[str, Any]],
        list[torch.utils.hooks.RemovableHandle],
    ]:
        records: list[dict[str, Any]] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []
        inhibitory = inhibitory_branch_layer_ids(model)

        def _register_branch_hooks(module_entry: tuple[str, DendriticBranchLayer]):
            module_name, module = module_entry
            reject_unsupported_branch_layer(module_name, module)
            rec = _branch_record(module_name, module, inhibitory)
            records.append(rec)

            def _register_hook_group(hook_group: str):
                if hook_group == "reactivation":
                    return [self._register_reactivation_hook(module, rec)]
                if hook_group == "topk":
                    return self._register_topk_hooks(module, rec)
                block_hook = self._register_block_output_hook(module, rec)
                return [] if block_hook is None else [block_hook]

            return register_hook_groups(
                ("reactivation", "topk", "block"),
                _register_hook_group,
            )

        handles = register_hook_groups(
            iter_named_modules_of_type(model, DendriticBranchLayer),
            _register_branch_hooks,
        )
        return records, handles

    def _register_reactivation_hook(
        self,
        module: DendriticBranchLayer,
        rec: dict[str, Any],
    ) -> torch.utils.hooks.RemovableHandle:
        def _react_hook(m, inputs, outputs, rec=rec):
            if "v_out" in rec:
                raise ValueError(
                    f"local_ca records one forward step per branch layer, but "
                    f"{rec['module_name']} ran more than once in a forward pass "
                    "(a recurrent unroll); use backpropagation for this core"
                )
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


__all__ = [
    "LocalLearningRecorderMixin",
    "reject_mismatched_dynamics_mode",
    "reject_unsupported_branch_layer",
    "reject_unsupported_forward_paths",
    "reject_unsupported_synapses",
]
