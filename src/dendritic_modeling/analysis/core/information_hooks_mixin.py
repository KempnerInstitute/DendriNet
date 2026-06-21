"""Forward-hook collection helpers for information analysis."""

from __future__ import annotations

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    HookHandle,
    register_named_forward_hooks,
)


def _has_no_branch_synapses(module: DendriticBranchLayer) -> bool:
    return module.branch_excitation is None and module.branch_inhibition is None


def _zero_branch_currents_for_output(output: torch.Tensor) -> torch.Tensor:
    batch_size = output.shape[0]
    n_branches = output.shape[1] if len(output.shape) > 1 else 1
    return torch.zeros(batch_size, n_branches, device=output.device)


def _resolve_synaptic_branch_currents(
    module: DendriticBranchLayer,
    excitatory_input: torch.Tensor | None,
    inhibitory_input: torch.Tensor | None,
    output: torch.Tensor,
    cached_currents: object,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    has_no_synapses = _has_no_branch_synapses(module)

    if has_no_synapses:
        return (
            _zero_branch_currents_for_output(output),
            _zero_branch_currents_for_output(output),
            has_no_synapses,
        )

    if isinstance(cached_currents, dict):
        excitation_per_branch = cached_currents.get("excitation")
        inhibition_per_branch = cached_currents.get("inhibition")
        if excitation_per_branch is None:
            excitation_per_branch = torch.zeros_like(output)
        if inhibition_per_branch is None:
            inhibition_per_branch = torch.zeros_like(output)
        return excitation_per_branch, inhibition_per_branch, has_no_synapses

    excitation_per_branch = (
        module.branch_excitation(excitatory_input)
        if module.branch_excitation is not None and excitatory_input is not None
        else torch.zeros_like(output)
    )
    inhibition_per_branch = (
        module.branch_inhibition(inhibitory_input)
        if module.branch_inhibition is not None and inhibitory_input is not None
        else torch.zeros_like(output)
    )
    return excitation_per_branch, inhibition_per_branch, has_no_synapses


def _resolve_branch_input_current(
    module: DendriticBranchLayer,
    branch_input: torch.Tensor | None,
    cached_currents: object,
) -> torch.Tensor | None:
    if isinstance(cached_currents, dict):
        return cached_currents.get("branch_input")
    if branch_input is not None and module.input_branches:
        return module.branches_to_output(branch_input)
    return None


def _has_analyzable_branch_signals(module: DendriticBranchLayer) -> bool:
    has_synapses = (
        module.branch_excitation is not None or module.branch_inhibition is not None
    )
    return has_synapses or module.input_branches


def _enable_analysis_current_capture(module: DendriticBranchLayer) -> object:
    previous_store = getattr(module, "_store_analysis_currents", False)
    module._store_analysis_currents = True
    return previous_store


def _restore_analysis_current_capture(
    module: DendriticBranchLayer,
    previous_store: object,
) -> None:
    module._store_analysis_currents = previous_store
    if hasattr(module, "_last_analysis_currents"):
        delattr(module, "_last_analysis_currents")


def _prepare_branch_hook_capture(
    data_dict: dict,
    hook_modules: list[tuple[DendriticBranchLayer, object]],
    name: str,
    module: DendriticBranchLayer,
) -> None:
    data_dict[name] = {}
    module._name = name
    previous_store = _enable_analysis_current_capture(module)
    hook_modules.append((module, previous_store))


def _restore_branch_hook_capture(
    hook_modules: list[tuple[DendriticBranchLayer, object]],
) -> None:
    for module, previous_store in hook_modules:
        _restore_analysis_current_capture(module, previous_store)
    hook_modules.clear()


def _attach_analyzable_branch_hooks(
    *,
    model: BaseModel,
    data_dict: dict,
    hook_modules: list[tuple[DendriticBranchLayer, object]],
    hook,
) -> list[HookHandle]:
    try:
        return register_named_forward_hooks(
            model,
            DendriticBranchLayer,
            hook,
            predicate=lambda _name, module: _has_analyzable_branch_signals(module),
            prepare=lambda name, module: _prepare_branch_hook_capture(
                data_dict,
                hook_modules,
                name,
                module,
            ),
        )
    except Exception:
        _restore_branch_hook_capture(hook_modules)
        raise


def _store_branch_hook_data(
    data_dict: dict,
    module: DendriticBranchLayer,
    *,
    excitation_per_branch: torch.Tensor,
    inhibition_per_branch: torch.Tensor,
    branch_current: torch.Tensor | None,
    output: torch.Tensor,
    has_no_synapses: bool,
    excitatory_input: torch.Tensor | None,
    inhibitory_input: torch.Tensor | None,
    branch_input: torch.Tensor | None,
    compute_lda_weights: bool,
) -> None:
    layer_data = data_dict[module._name]
    layer_data["excitation"] = excitation_per_branch
    layer_data["inhibition"] = inhibition_per_branch
    layer_data["branch_input"] = branch_current
    layer_data["output"] = output
    layer_data["has_synapses"] = not has_no_synapses
    layer_data["has_exc_synapses"] = module.branch_excitation is not None
    layer_data["has_inh_synapses"] = (
        module.branch_inhibition is not None and inhibitory_input is not None
    )
    layer_data["has_branch_input"] = branch_input is not None and module.input_branches

    if compute_lda_weights:
        layer_data["raw_excitatory_input"] = excitatory_input
        layer_data["raw_inhibitory_input"] = inhibitory_input
        layer_data["raw_branch_input"] = branch_input
        layer_data["module"] = module


class InformationHooksMixin(ForwardHookRemovalMixin):
    """Collects branch-layer currents and outputs during model evaluation."""

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to the model."""
        # Initialize data_dict attribute and store somatic_synapses config
        self.data_dict = {}
        self._analysis_hook_modules = []
        self.somatic_synapses_enabled = getattr(
            model.core_network, "somatic_synapses", True
        )

        return _attach_analyzable_branch_hooks(
            model=model,
            data_dict=self.data_dict,
            hook_modules=self._analysis_hook_modules,
            hook=self.forward_hook,
        )

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        """Forward hook to collect branch data."""
        excitatory_input, inhibitory_input, branch_input = input

        cached_currents = getattr(module, "_last_analysis_currents", None)
        (
            excitation_per_branch,
            inhibition_per_branch,
            has_no_synapses,
        ) = _resolve_synaptic_branch_currents(
            module,
            excitatory_input,
            inhibitory_input,
            output,
            cached_currents,
        )
        branch_current = _resolve_branch_input_current(
            module,
            branch_input,
            cached_currents,
        )

        _store_branch_hook_data(
            self.data_dict,
            module,
            excitation_per_branch=excitation_per_branch,
            inhibition_per_branch=inhibition_per_branch,
            branch_current=branch_current,
            output=output,
            has_no_synapses=has_no_synapses,
            excitatory_input=excitatory_input,
            inhibitory_input=inhibitory_input,
            branch_input=branch_input,
            compute_lda_weights=bool(getattr(self, "compute_lda_weights", False)),
        )

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        super().remove_forward_hooks(handles)
        _restore_branch_hook_capture(getattr(self, "_analysis_hook_modules", []))


__all__ = ["InformationHooksMixin"]
