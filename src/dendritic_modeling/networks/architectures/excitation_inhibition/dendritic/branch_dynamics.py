"""Forward current and voltage dynamics for dendritic branch layers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from dendritic_modeling.networks.activations.memory_efficient import (
    fused_shunt_reactivation,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_rule_override import (
    active_branch_rule_override,
    check_branch_rule_override_target,
)
from dendritic_modeling.networks.functional import (
    _raw_current_sum,
    _shunting_current_components,
)
from dendritic_modeling.networks.utils.signal_rounding import round_signal
from dendritic_modeling.plugins import load_plugins

#: The library's additive integration rules.
ADDITIVE_MODES = frozenset({"raw", "tangent_matched", "conductance_normalized"})


@dataclass(frozen=True)
class AdditiveRule:
    """An additive integration rule that a plugin provides.

    ``voltage(layer, *, excitation, inhibition, child_current, child_conductance,
    raw_voltage, inputs)`` returns the compartment voltage before reactivation.
    The currents are those of the built-in rules (``raw_voltage`` is
    ``excitation - inhibition + child_current``); ``inputs`` maps ``x``,
    ``inhibitory_input`` and ``branch_input`` to the layer's inputs on the
    feedforward path, and is None where voltages are computed from stored
    currents (recurrent and analysis paths), where a rule that needs the inputs
    must raise.

    ``options`` holds the rule's options and their defaults; a layer receives
    them as ``additive_options``. ``setup(layer, options)``, if given, runs once
    while the layer is constructed (after its leak parameter) to create the
    rule's parameters and buffers and to check the layer.
    """

    voltage: Callable[..., Any]
    options: Mapping[str, Any] = field(default_factory=dict)
    setup: Callable[[Any, dict[str, Any]], None] | None = None


_ADDITIVE_RULES: dict[str, AdditiveRule] = {}


def register_additive_rule(
    name: str, rule: AdditiveRule, *, allow_override: bool = False
) -> None:
    """Register an additive integration rule under ``additive_mode: <name>``."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("additive rule name must be non-empty")
    if key in ADDITIVE_MODES:
        raise ValueError(f"additive_mode {key!r} is built in")
    if key in _ADDITIVE_RULES and not allow_override:
        raise ValueError(
            f"additive rule {key!r} is already registered; pass allow_override=True "
            "to replace it"
        )
    if not isinstance(rule, AdditiveRule):
        raise TypeError("rule must be an AdditiveRule")
    _ADDITIVE_RULES[key] = rule


def unregister_additive_rule(name: str) -> None:
    """Remove a registered additive rule if present."""
    _ADDITIVE_RULES.pop(str(name).strip().lower(), None)


def get_additive_rule(name: str) -> AdditiveRule | None:
    """The registered rule for a non-built-in additive mode, or None."""
    key = str(name).strip().lower()
    if key not in ADDITIVE_MODES and key not in _ADDITIVE_RULES:
        load_plugins()
    return _ADDITIVE_RULES.get(key)


def available_additive_modes() -> frozenset[str]:
    """Built-in and registered additive modes, including those of plugins."""
    load_plugins()
    return ADDITIVE_MODES | frozenset(_ADDITIVE_RULES)


def normalize_additive_mode(mode: Any) -> str:
    """Return a validated additive control name."""
    normalized = str(mode).strip().lower()
    if normalized in ADDITIVE_MODES or normalized in _ADDITIVE_RULES:
        return normalized
    load_plugins()
    if normalized not in _ADDITIVE_RULES:
        choices = ", ".join(sorted(available_additive_modes()))
        raise ValueError(f"additive_mode must be one of {{{choices}}}, got {mode!r}")
    return normalized


def _diagnostic_synapse_forward(owner: Any, layer: Any, value: Any):
    """Run a synapse layer while retaining its exact realized forward mask."""
    if not getattr(owner, "_store_diagnostics", False) or not hasattr(
        layer, "cache_mask"
    ):
        return layer(value)

    previous_cache_mask = layer.cache_mask
    layer.cache_mask = True
    try:
        return layer(value)
    finally:
        layer.cache_mask = previous_cache_mask


def _realized_k_per_output(layer: Any) -> torch.Tensor | None:
    """Return the realized synapse count from the most recent forward."""
    if layer is None:
        return None

    mask = getattr(layer, "_last_forward_weight_mask", None)
    if mask is None:
        # DeepST exposes its realized mask directly rather than through the
        # TopK cache interface.
        mask = getattr(layer, "mask", None)
    if torch.is_tensor(mask) and mask.numel() > 0:
        return mask.detach().reshape(mask.shape[0], -1).ne(0).sum(dim=-1)

    # Fixed indexed layers always realize their K stored connections. This is
    # also a safe fallback for custom deterministic sparse layers that expose
    # K but not a dense mask.
    k = getattr(layer, "K", None)
    out_features = getattr(layer, "out_features", None)
    parameter = getattr(layer, "pre_w", None)
    if k is not None and out_features is not None:
        device = parameter.device if torch.is_tensor(parameter) else None
        return torch.full((int(out_features),), int(k), device=device, dtype=torch.long)
    return None


def _collect_realized_k(owner: Any, active_pathways: set[str]) -> dict[str, Any]:
    pathway_layers = {
        "E": getattr(owner, "branch_excitation", None),
        "I": getattr(owner, "branch_inhibition", None),
        "E_rec": getattr(owner, "branch_recurrent", None),
        "I_rec": getattr(owner, "branch_rec_inhibition", None),
    }
    return {
        name: (_realized_k_per_output(layer) if name in active_pathways else None)
        for name, layer in pathway_layers.items()
    }


def _diagnostic_component(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast one circuit component to the voltage shape and detach it."""
    if torch.is_tensor(value):
        value = value.to(device=reference.device, dtype=reference.dtype)
    else:
        value = reference.new_tensor(value)
    return (value + torch.zeros_like(reference)).detach()


def _store_branch_diagnostics(
    owner: Any,
    *,
    E: Any,
    inhibitory: Any,
    C: Any,
    G: Any,
    N: Any,
    T: Any,
    V: Any,
    realized_k: dict[str, Any] | None = None,
    shunt_denominator: Any = None,
) -> None:
    """Store a detached, code-faithful branch-circuit decomposition."""
    if not getattr(owner, "_store_diagnostics", False) or not torch.is_tensor(V):
        return

    realized_k = realized_k or {}
    detached_k = {
        name: value.detach() if torch.is_tensor(value) else value
        for name, value in realized_k.items()
    }
    owner._last_branch_diagnostics = {
        "E": _diagnostic_component(E, V),
        "I": _diagnostic_component(inhibitory, V),
        "C": _diagnostic_component(C, V),
        "G": _diagnostic_component(G, V),
        "N": _diagnostic_component(N, V),
        "T": _diagnostic_component(T, V),
        "V": V.detach(),
        "leak": _diagnostic_component(owner.effective_leak, V),
        "realized_K": detached_k,
        "mode": "shunting" if owner.use_shunting else owner.additive_mode,
    }

    # Private compatibility fields consumed by shunting feedback alignment.
    # Keep them unset for every additive control so additive FA behavior does
    # not change merely because richer diagnostics are available.
    if owner.use_shunting:
        g_total = 1 + T if shunt_denominator is None else shunt_denominator
        owner._diag_g_tot = _diagnostic_component(g_total, V)
        owner._diag_numerator = _diagnostic_component(N, V)
    else:
        owner._diag_g_tot = None
        owner._diag_numerator = None


def _tangent_matched_voltage(owner: Any, N: Any, T: Any):
    """Evaluate the affine tangent of ``N / (1 + T + epsilon)``."""
    if not owner.has_additive_operating_point:
        raise RuntimeError(
            "additive_mode='tangent_matched' requires a frozen operating "
            "point. Pass additive_tangent_n0/additive_tangent_t0 or call "
            "set_additive_operating_point(n0, t0) before forward()."
        )

    reference = N if torch.is_tensor(N) else T
    if torch.is_tensor(reference):
        n0 = owner._additive_tangent_n0.to(
            device=reference.device, dtype=reference.dtype
        )
        t0 = owner._additive_tangent_t0.to(
            device=reference.device, dtype=reference.dtype
        )
    else:
        n0 = float(owner._additive_tangent_n0.item())
        t0 = float(owner._additive_tangent_t0.item())

    d0 = 1 + t0 + owner.epsilon
    v0 = n0 / d0
    return v0 + ((N - n0) - v0 * (T - t0)) / d0


def compute_additive_control_voltage(
    owner: Any,
    *,
    E: Any,
    inhibitory: Any,
    C: Any,
    G: Any,
    raw_voltage: Any = None,
    inputs: Mapping[str, Any] | None = None,
):
    """Compute one explicitly named additive comparison control."""
    mode = owner.additive_mode
    if raw_voltage is None:
        raw_voltage = E - inhibitory + C
    if mode == "raw":
        return owner.normalize_additive_voltage(raw_voltage)
    rule = _ADDITIVE_RULES.get(mode)
    if rule is not None:
        return rule.voltage(
            owner,
            excitation=E,
            inhibition=inhibitory,
            child_current=C,
            child_conductance=G,
            raw_voltage=raw_voltage,
            inputs=inputs,
        )

    N = E + C
    T = E + inhibitory + G
    if mode == "conductance_normalized":
        return raw_voltage / (1 + T + owner.epsilon)
    if mode == "tangent_matched":
        return _tangent_matched_voltage(owner, N, T)
    # Construction validates this, but retain a local guard for custom owners.
    raise RuntimeError(f"Unsupported additive_mode: {mode!r}")


def forward_branch_dynamics(
    owner: Any,
    x,
    inhibitory_input=None,
    branch_input=None,
    recurrent_input=None,
    rec_inhibitory_input=None,
):
    """Run the branch-layer forward dynamics before returning reactivation output."""
    excitation = None
    inhibition = None
    branch_current = None
    E = 0
    inhibitory_total = 0
    C = 0
    G = 0
    raw_voltage = 0
    shunt_numerator = 0
    shunt_denominator = owner.effective_leak
    store_diagnostics = bool(getattr(owner, "_store_diagnostics", False))
    rounding = getattr(owner, "_signal_round_dtype", None)
    analysis_rule = active_branch_rule_override(owner)
    if analysis_rule is not None:
        check_branch_rule_override_target(owner)
    needs_components = (
        store_diagnostics
        or analysis_rule is not None
        or (not owner.use_shunting and owner.additive_mode != "raw")
    )
    active_pathways: set[str] = set()

    if owner.input_excitatory and owner.branch_excitation is not None:
        excitation = _diagnostic_synapse_forward(owner, owner.branch_excitation, x)
        if needs_components:
            E = E + excitation
        if not owner.use_shunting:
            raw_voltage = raw_voltage + excitation
        if owner.use_shunting:
            shunt_numerator = shunt_numerator + excitation
            shunt_denominator = shunt_denominator + excitation
        if store_diagnostics:
            active_pathways.add("E")

    if (
        owner.input_recurrent
        and owner.branch_recurrent is not None
        and recurrent_input is not None
    ):
        rec_excitation = _diagnostic_synapse_forward(
            owner, owner.branch_recurrent, recurrent_input
        )
        if needs_components:
            E = E + rec_excitation
        if not owner.use_shunting:
            raw_voltage = raw_voltage + rec_excitation
        if owner.use_shunting:
            shunt_numerator = shunt_numerator + rec_excitation
            shunt_denominator = shunt_denominator + rec_excitation
        if store_diagnostics:
            active_pathways.add("E_rec")

    if owner.input_branches and branch_input is not None:
        branch_current = owner.branches_to_output(branch_input)
        if needs_components:
            C = C + branch_current
        if not owner.use_shunting:
            raw_voltage = raw_voltage + branch_current
        if owner.use_shunting or owner.additive_mode != "raw" or store_diagnostics:
            G = owner.branches_to_output.sum_conductances()
        if owner.use_shunting:
            shunt_numerator = shunt_numerator + branch_current
            shunt_denominator = shunt_denominator + G

    if (
        owner.input_inhibitory
        and owner.branch_inhibition is not None
        and inhibitory_input is not None
    ):
        inhibition = _diagnostic_synapse_forward(
            owner, owner.branch_inhibition, inhibitory_input
        )
        if needs_components:
            inhibitory_total = inhibitory_total + inhibition
        if not owner.use_shunting:
            raw_voltage = raw_voltage - inhibition
        if owner.use_shunting:
            shunt_denominator = shunt_denominator + inhibition
        if store_diagnostics:
            active_pathways.add("I")

    if (
        owner.input_rec_inhibitory
        and owner.branch_rec_inhibition is not None
        and rec_inhibitory_input is not None
    ):
        rec_inhibition = _diagnostic_synapse_forward(
            owner, owner.branch_rec_inhibition, rec_inhibitory_input
        )
        if needs_components:
            inhibitory_total = inhibitory_total + rec_inhibition
        if not owner.use_shunting:
            raw_voltage = raw_voltage - rec_inhibition
        if owner.use_shunting:
            shunt_denominator = shunt_denominator + rec_inhibition
        if store_diagnostics:
            active_pathways.add("I_rec")

    fused_shunt = None
    if (
        owner.use_shunting
        and not store_diagnostics
        and not getattr(owner, "_store_analysis_currents", False)
        and analysis_rule is None
        and rounding is None
    ):
        fused_shunt = fused_shunt_reactivation(owner.reactivation)

    denominator = None
    voltage = None
    if fused_shunt is not None:
        denominator = shunt_denominator
    elif owner.use_shunting:
        denominator = shunt_denominator
        if analysis_rule is None:
            voltage = shunt_numerator / (denominator + owner.epsilon)
        else:
            voltage = analysis_rule.voltage(
                owner,
                shunt_numerator,
                denominator - 1,
                excitation=E,
                inhibition=inhibitory_total,
                child_current=C,
            )
    else:
        voltage = compute_additive_control_voltage(
            owner,
            E=E,
            inhibitory=inhibitory_total,
            C=C,
            G=G,
            raw_voltage=raw_voltage,
            inputs=(
                {
                    "x": x,
                    "inhibitory_input": inhibitory_input,
                    "branch_input": branch_input,
                }
                if owner.additive_mode in _ADDITIVE_RULES
                else None
            ),
        )
    if rounding is not None:
        voltage = round_signal(voltage, rounding)

    if store_diagnostics:
        if owner.use_shunting:
            N = shunt_numerator
            T = denominator - owner.effective_leak
        else:
            N = E + C
            T = E + inhibitory_total + G
        realized_k = _collect_realized_k(owner, active_pathways)
        _store_branch_diagnostics(
            owner,
            E=E,
            inhibitory=inhibitory_total,
            C=C,
            G=G,
            N=N,
            T=T,
            V=voltage,
            realized_k=realized_k,
            shunt_denominator=denominator,
        )

    if owner.training and "conductance_dynamic" in [
        owner.topk_strategy,
        owner.blocklinear_strategy,
    ]:
        owner.compute_grad_scales(
            g_total=denominator if owner.use_shunting else None,
            use_forward_hooks=bool(
                getattr(owner, "_forward_dynamic_grad_scaling", False)
            ),
        )

    if fused_shunt is not None:
        result = fused_shunt(shunt_numerator, denominator, owner.epsilon)
    else:
        result = round_signal(owner.reactivation(voltage), rounding)
    if getattr(owner, "_store_analysis_currents", False):
        owner._last_analysis_currents = {
            # Canonical stage-explicit analysis names.
            "pre_gate_excitation_current": excitation,
            "pre_gate_inhibition_current": inhibition,
            "pre_gate_upstream_current": branch_current,
            "pre_gate_voltage": voltage,
            "post_gate_output": result,
            # Compatibility aliases consumed by older analysis code.
            "excitation": excitation,
            "inhibition": inhibition,
            "branch_input": branch_current,
        }
    return result


def compute_branch_raw_currents(
    owner: Any,
    x,
    inhibitory_input=None,
    recurrent_input=None,
    rec_inhibitory_input=None,
):
    """Compute raw feedforward and recurrent branch currents."""
    store_diagnostics = bool(getattr(owner, "_store_diagnostics", False))
    active_pathways: set[str] = set()
    raw_E = 0
    if owner.input_excitatory and owner.branch_excitation is not None:
        raw_E = _diagnostic_synapse_forward(owner, owner.branch_excitation, x)
        if store_diagnostics:
            active_pathways.add("E")

    raw_E_rec = 0
    if (
        owner.input_recurrent
        and owner.branch_recurrent is not None
        and recurrent_input is not None
    ):
        raw_E_rec = _diagnostic_synapse_forward(
            owner, owner.branch_recurrent, recurrent_input
        )
        if store_diagnostics:
            active_pathways.add("E_rec")

    raw_I = 0
    if (
        owner.input_inhibitory
        and owner.branch_inhibition is not None
        and inhibitory_input is not None
    ):
        raw_I = _diagnostic_synapse_forward(
            owner, owner.branch_inhibition, inhibitory_input
        )
        if store_diagnostics:
            active_pathways.add("I")

    raw_I_rec = 0
    if (
        owner.input_rec_inhibitory
        and owner.branch_rec_inhibition is not None
        and rec_inhibitory_input is not None
    ):
        raw_I_rec = _diagnostic_synapse_forward(
            owner, owner.branch_rec_inhibition, rec_inhibitory_input
        )
        if store_diagnostics:
            active_pathways.add("I_rec")

    if store_diagnostics:
        owner._pending_realized_k = _collect_realized_k(owner, active_pathways)
    return raw_E, raw_E_rec, raw_I, raw_I_rec


def normalize_branch_additive_voltage(owner: Any, voltage):
    """Apply the legacy per-sample, across-output z-score to additive voltage."""
    if not owner.use_additive_normalization or not torch.is_tensor(voltage):
        return voltage
    centered = voltage - voltage.mean(dim=-1, keepdim=True)
    std = voltage.std(dim=-1, keepdim=True, unbiased=False)
    if getattr(owner, "_record_additive_scale", False):
        # Local learning rules that need this z-score's derivative read the
        # per-sample scale back (see LocalCA's additive_gain_mode).
        owner._last_additive_scale = std.detach()
    return centered / std.clamp_min(owner.epsilon)


def compute_branch_voltage_from_currents(
    owner: Any,
    trace_E,
    trace_E_rec,
    trace_I,
    trace_I_rec=0,
    trace_branch=0,
    branch_conductance=None,
    record_diagnostics=True,
):
    """Compute pre-reactivation voltage from integrated recurrent traces.

    ``branch_conductance`` is a denominator-only input for shunting. Raw
    additive voltage ignores it; explicitly normalized additive controls may
    use it through their named comparison rule. This asymmetry is useful for
    mechanism-isolation interventions but is not a same-physical-input model
    of additive and conductance-based neurons. ``record_diagnostics=False``
    leaves the layer's stored diagnostics untouched (for analyses that apply
    the rule to substituted currents).
    """
    store_diagnostics = record_diagnostics and bool(
        getattr(owner, "_store_diagnostics", False)
    )
    if active_branch_rule_override(owner) is not None:
        raise ValueError(
            "branch rule overrides apply to the feedforward forward pass; "
            "voltages computed from currents (recurrent and stateful paths, "
            "analyses that substitute currents) would ignore them"
        )
    denominator = None
    if owner.use_shunting:
        child_current = torch.is_tensor(trace_branch) or bool(trace_branch)
        if (
            child_current
            and branch_conductance is None
            and getattr(owner, "input_branches", False)
        ):
            raise ValueError(
                "a shunting layer given its child-branch current needs their "
                "coupling conductance (branches_to_output.sum_conductances()) in "
                "the denominator; pass branch_conductance"
            )
        N, denominator = _shunting_current_components(
            trace_E,
            trace_E_rec,
            trace_I,
            trace_I_rec,
            trace_branch,
            branch_conductance,
            owner.effective_leak,
        )
        T = denominator - owner.effective_leak
        voltage = N / (denominator + owner.epsilon)
    else:
        raw_voltage = _raw_current_sum(
            trace_E, trace_E_rec, trace_I, trace_I_rec, trace_branch
        )

        if store_diagnostics or owner.additive_mode != "raw":
            E = trace_E + trace_E_rec
            inhibitory_total = trace_I + trace_I_rec
            C = trace_branch
            G = 0 if branch_conductance is None else branch_conductance
        else:
            E = 0
            inhibitory_total = 0
            C = 0
            G = 0
        voltage = compute_additive_control_voltage(
            owner,
            E=E,
            inhibitory=inhibitory_total,
            C=C,
            G=G,
            raw_voltage=raw_voltage,
        )

    if store_diagnostics:
        if owner.use_shunting:
            E = trace_E + trace_E_rec
            inhibitory_total = trace_I + trace_I_rec
            C = trace_branch
            G = 0 if branch_conductance is None else branch_conductance
        else:
            N = E + C
            T = E + inhibitory_total + G
        _store_branch_diagnostics(
            owner,
            E=E,
            inhibitory=inhibitory_total,
            C=C,
            G=G,
            N=N,
            T=T,
            V=voltage,
            realized_k=getattr(owner, "_pending_realized_k", None),
            shunt_denominator=denominator,
        )
    return voltage, denominator


def compute_branch_output_from_currents(
    owner: Any,
    excitation,
    inhibition,
    branch_current=0,
    branch_conductance=None,
):
    """Branch output for given currents under the layer's own rule.

    Applies the layer's voltage rule (leak, integration mode and, for a
    shunting layer with child branches, their coupling conductance), its
    signal rounding and its reactivation, as the forward pass does. Analyses
    that substitute, condition or project currents use this instead of
    copying the equation; the layer's stored diagnostics are left untouched.
    """
    rounding = getattr(owner, "_signal_round_dtype", None)
    voltage, _ = compute_branch_voltage_from_currents(
        owner,
        trace_E=excitation,
        trace_E_rec=0,
        trace_I=inhibition,
        trace_branch=branch_current,
        branch_conductance=branch_conductance,
        record_diagnostics=False,
    )
    voltage = round_signal(voltage, rounding)
    return round_signal(owner.reactivation(voltage), rounding)


__all__ = [
    "ADDITIVE_MODES",
    "AdditiveRule",
    "available_additive_modes",
    "compute_additive_control_voltage",
    "compute_branch_output_from_currents",
    "compute_branch_raw_currents",
    "compute_branch_voltage_from_currents",
    "forward_branch_dynamics",
    "get_additive_rule",
    "normalize_additive_mode",
    "normalize_branch_additive_voltage",
    "register_additive_rule",
    "unregister_additive_rule",
]
