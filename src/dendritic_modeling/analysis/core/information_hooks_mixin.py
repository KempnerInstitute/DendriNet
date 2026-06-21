"""Forward-hook collection helpers for information analysis."""

from __future__ import annotations

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils.hooks import remove_hook_handles


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


class InformationHooksMixin:
    """Collects branch-layer currents and outputs during model evaluation."""

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to the model."""
        # Initialize data_dict attribute and store somatic_synapses config
        self.data_dict = {}
        self._analysis_hook_modules = []
        self.somatic_synapses_enabled = getattr(
            model.core_network, "somatic_synapses", True
        )

        handles = []
        try:
            for _name, module in model.named_modules():
                if isinstance(module, DendriticBranchLayer):
                    if _has_analyzable_branch_signals(module):
                        self.data_dict[_name] = {}
                        module._name = _name
                        previous_store = _enable_analysis_current_capture(module)
                        self._analysis_hook_modules.append((module, previous_store))
                        handle = module.register_forward_hook(self.forward_hook)
                        handles.append(handle)
            return handles
        except Exception:
            self.remove_forward_hooks(handles)
            raise

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

        # Store per-branch data
        self.data_dict[module._name]["excitation"] = excitation_per_branch
        self.data_dict[module._name]["inhibition"] = inhibition_per_branch
        self.data_dict[module._name]["branch_input"] = branch_current
        self.data_dict[module._name]["output"] = output
        self.data_dict[module._name]["has_synapses"] = not has_no_synapses
        self.data_dict[module._name]["has_exc_synapses"] = (
            module.branch_excitation is not None
        )
        self.data_dict[module._name]["has_inh_synapses"] = (
            module.branch_inhibition is not None and inhibitory_input is not None
        )
        self.data_dict[module._name]["has_branch_input"] = (
            branch_input is not None and module.input_branches
        )

        # Store raw inputs and module for LDA computation if needed
        if hasattr(self, "compute_lda_weights") and self.compute_lda_weights:
            self.data_dict[module._name]["raw_excitatory_input"] = excitatory_input
            self.data_dict[module._name]["raw_inhibitory_input"] = inhibitory_input
            self.data_dict[module._name]["raw_branch_input"] = branch_input
            self.data_dict[module._name]["module"] = module

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        remove_hook_handles(handles)
        for module, previous_store in getattr(self, "_analysis_hook_modules", []):
            _restore_analysis_current_capture(module, previous_store)
        self._analysis_hook_modules = []


__all__ = ["InformationHooksMixin"]
