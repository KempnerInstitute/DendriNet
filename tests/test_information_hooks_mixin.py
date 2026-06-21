"""Focused tests for information-analysis hook current resolution."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dendritic_modeling.analysis.core.information_hooks_mixin import (
    InformationHooksMixin,
    _enable_analysis_current_capture,
    _has_analyzable_branch_signals,
    _has_no_branch_synapses,
    _resolve_branch_input_current,
    _resolve_synaptic_branch_currents,
    _restore_analysis_current_capture,
)
from dendritic_modeling.networks import DendriticBranchLayer


class _Scale(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


def _branch_layer() -> DendriticBranchLayer:
    return DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=3,
        inhibitory_synapses_per_branch=2,
        use_shunting=False,
        reactivate=False,
        topk_noise_level=0.0,
        weight_transform="identity",
    )


class _InformationHookModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core_network = SimpleNamespace(somatic_synapses=True)
        self.layer_a = _branch_layer()
        self.layer_b = _branch_layer()


def test_resolve_synaptic_branch_currents_no_synapses_preserves_zero_behavior():
    module = SimpleNamespace(branch_excitation=None, branch_inhibition=None)
    output = torch.ones(2, 3, dtype=torch.float64)

    excitation, inhibition, has_no_synapses = _resolve_synaptic_branch_currents(
        module,
        excitatory_input=None,
        inhibitory_input=None,
        output=output,
        cached_currents=None,
    )

    assert has_no_synapses is True
    assert _has_no_branch_synapses(module) is True
    assert excitation.shape == (2, 3)
    assert inhibition.shape == (2, 3)
    assert excitation.dtype == torch.get_default_dtype()
    assert inhibition.dtype == torch.get_default_dtype()
    assert excitation.device == output.device
    assert inhibition.device == output.device
    assert torch.equal(excitation, torch.zeros(2, 3))
    assert torch.equal(inhibition, torch.zeros(2, 3))
    assert excitation is not inhibition


def test_resolve_synaptic_branch_currents_cached_currents_preserve_fallbacks():
    module = SimpleNamespace(
        branch_excitation=_Scale(2.0), branch_inhibition=_Scale(3.0)
    )
    output = torch.ones(2, 3, dtype=torch.float64)
    cached_excitation = torch.full((2, 3), 4.0, dtype=torch.float64)

    excitation, inhibition, has_no_synapses = _resolve_synaptic_branch_currents(
        module,
        excitatory_input=torch.ones(2, 3),
        inhibitory_input=torch.ones(2, 3),
        output=output,
        cached_currents={"excitation": cached_excitation},
    )

    assert has_no_synapses is False
    assert excitation is cached_excitation
    assert torch.equal(inhibition, torch.zeros_like(output))
    assert inhibition.dtype == output.dtype


def test_resolve_synaptic_branch_currents_recomputes_when_cache_is_absent():
    module = SimpleNamespace(
        branch_excitation=_Scale(2.0), branch_inhibition=_Scale(3.0)
    )
    excitatory_input = torch.tensor([[1.0, 2.0]])
    inhibitory_input = torch.tensor([[3.0, 4.0]])
    output = torch.zeros(1, 2)

    excitation, inhibition, has_no_synapses = _resolve_synaptic_branch_currents(
        module,
        excitatory_input=excitatory_input,
        inhibitory_input=inhibitory_input,
        output=output,
        cached_currents=None,
    )

    assert has_no_synapses is False
    assert torch.equal(excitation, excitatory_input * 2.0)
    assert torch.equal(inhibition, inhibitory_input * 3.0)


def test_resolve_synaptic_branch_currents_missing_input_uses_output_like_zero():
    module = SimpleNamespace(
        branch_excitation=_Scale(2.0), branch_inhibition=_Scale(3.0)
    )
    output = torch.ones(2, 3, dtype=torch.float64)

    excitation, inhibition, has_no_synapses = _resolve_synaptic_branch_currents(
        module,
        excitatory_input=None,
        inhibitory_input=None,
        output=output,
        cached_currents=None,
    )

    assert has_no_synapses is False
    assert torch.equal(excitation, torch.zeros_like(output))
    assert torch.equal(inhibition, torch.zeros_like(output))
    assert excitation.dtype == output.dtype
    assert inhibition.dtype == output.dtype


def test_resolve_branch_input_current_preserves_cache_precedence():
    module = SimpleNamespace(input_branches=True, branches_to_output=_Scale(5.0))
    branch_input = torch.tensor([[1.0, 2.0]])
    cached_branch_current = torch.tensor([[7.0, 8.0]])

    branch_current = _resolve_branch_input_current(
        module,
        branch_input=branch_input,
        cached_currents={"branch_input": cached_branch_current},
    )
    missing_cached_current = _resolve_branch_input_current(
        module,
        branch_input=branch_input,
        cached_currents={},
    )

    assert branch_current is cached_branch_current
    assert missing_cached_current is None


def test_resolve_branch_input_current_recomputes_only_for_branch_inputs():
    branch_input = torch.tensor([[1.0, 2.0]])
    module_with_branches = SimpleNamespace(
        input_branches=True,
        branches_to_output=_Scale(5.0),
    )
    module_without_branches = SimpleNamespace(
        input_branches=False,
        branches_to_output=_Scale(5.0),
    )

    branch_current = _resolve_branch_input_current(
        module_with_branches,
        branch_input=branch_input,
        cached_currents=None,
    )
    no_branch_current = _resolve_branch_input_current(
        module_without_branches,
        branch_input=branch_input,
        cached_currents=None,
    )

    assert torch.equal(branch_current, branch_input * 5.0)
    assert no_branch_current is None


def test_has_analyzable_branch_signals_matches_hook_registration_predicate():
    no_signals = SimpleNamespace(
        branch_excitation=None,
        branch_inhibition=None,
        input_branches=False,
    )
    with_excitation = SimpleNamespace(
        branch_excitation=_Scale(2.0),
        branch_inhibition=None,
        input_branches=False,
    )
    with_inhibition = SimpleNamespace(
        branch_excitation=None,
        branch_inhibition=_Scale(3.0),
        input_branches=False,
    )
    with_branch_input = SimpleNamespace(
        branch_excitation=None,
        branch_inhibition=None,
        input_branches=True,
    )

    assert _has_analyzable_branch_signals(no_signals) is False
    assert _has_analyzable_branch_signals(with_excitation) is True
    assert _has_analyzable_branch_signals(with_inhibition) is True
    assert _has_analyzable_branch_signals(with_branch_input) is True


def test_analysis_current_capture_lifecycle_preserves_previous_state():
    module = SimpleNamespace(_store_analysis_currents="previous")

    previous_store = _enable_analysis_current_capture(module)
    module._last_analysis_currents = {"excitation": torch.ones(1)}
    _restore_analysis_current_capture(module, previous_store)

    assert previous_store == "previous"
    assert module._store_analysis_currents == "previous"
    assert not hasattr(module, "_last_analysis_currents")


def test_information_hook_stores_resolved_currents_and_flags():
    module = SimpleNamespace(
        _name="layer0",
        branch_excitation=_Scale(2.0),
        branch_inhibition=None,
        input_branches=True,
        branches_to_output=_Scale(5.0),
    )
    hook = InformationHooksMixin()
    hook.data_dict = {"layer0": {}}

    excitatory_input = torch.tensor([[1.0, 2.0]])
    branch_input = torch.tensor([[3.0, 4.0]])
    output = torch.tensor([[9.0, 10.0]])

    hook.forward_hook(
        module,
        (excitatory_input, None, branch_input),
        output,
    )

    layer_data = hook.data_dict["layer0"]
    assert torch.equal(layer_data["excitation"], excitatory_input * 2.0)
    assert torch.equal(layer_data["inhibition"], torch.zeros_like(output))
    assert torch.equal(layer_data["branch_input"], branch_input * 5.0)
    assert layer_data["output"] is output
    assert layer_data["has_synapses"] is True
    assert layer_data["has_exc_synapses"] is True
    assert layer_data["has_inh_synapses"] is False
    assert layer_data["has_branch_input"] is True


def test_information_hook_preserves_lda_raw_input_capture():
    module = SimpleNamespace(
        _name="layer0",
        branch_excitation=_Scale(2.0),
        branch_inhibition=_Scale(3.0),
        input_branches=True,
        branches_to_output=_Scale(5.0),
    )
    hook = InformationHooksMixin()
    hook.compute_lda_weights = True
    hook.data_dict = {"layer0": {}}

    excitatory_input = torch.tensor([[1.0, 2.0]])
    inhibitory_input = torch.tensor([[4.0, 5.0]])
    branch_input = torch.tensor([[7.0, 8.0]])
    output = torch.tensor([[9.0, 10.0]])

    hook.forward_hook(
        module,
        (excitatory_input, inhibitory_input, branch_input),
        output,
    )

    layer_data = hook.data_dict["layer0"]
    assert layer_data["raw_excitatory_input"] is excitatory_input
    assert layer_data["raw_inhibitory_input"] is inhibitory_input
    assert layer_data["raw_branch_input"] is branch_input
    assert layer_data["module"] is module


def test_information_hooks_restore_state_after_failed_registration(monkeypatch):
    model = _InformationHookModel()
    hook = InformationHooksMixin()
    model.layer_a._store_analysis_currents = "previous_a"
    model.layer_b._store_analysis_currents = "previous_b"

    def _failing_register(*args, **kwargs):
        raise RuntimeError("information hook registration failed")

    monkeypatch.setattr(model.layer_b, "register_forward_hook", _failing_register)

    with pytest.raises(RuntimeError, match="information hook registration failed"):
        hook.attach_forward_hooks(model)

    assert sum(len(module._forward_hooks) for module in model.modules()) == 0
    assert model.layer_a._store_analysis_currents == "previous_a"
    assert model.layer_b._store_analysis_currents == "previous_b"
    assert hook._analysis_hook_modules == []
