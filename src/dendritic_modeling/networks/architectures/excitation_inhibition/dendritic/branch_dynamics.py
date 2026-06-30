"""Forward current and voltage dynamics for dendritic branch layers."""

from __future__ import annotations

from typing import Any

import torch


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

    if owner.use_shunting:
        numerator = 0
        denominator = 1

        if owner.input_excitatory and owner.branch_excitation is not None:
            excitation = owner.branch_excitation(x)
            numerator = numerator + excitation
            denominator = denominator + excitation

        if (
            owner.input_recurrent
            and owner.branch_recurrent is not None
            and recurrent_input is not None
        ):
            rec_excitation = owner.branch_recurrent(recurrent_input)
            numerator = numerator + rec_excitation
            denominator = denominator + rec_excitation

        if owner.input_branches and branch_input is not None:
            branch_current = owner.branches_to_output(branch_input)
            numerator = numerator + branch_current
            conductance = owner.branches_to_output.sum_conductances()
            denominator = denominator + conductance

        if (
            owner.input_inhibitory
            and owner.branch_inhibition is not None
            and inhibitory_input is not None
        ):
            inhibition = owner.branch_inhibition(inhibitory_input)
            denominator = denominator + inhibition

        if (
            owner.input_rec_inhibitory
            and owner.branch_rec_inhibition is not None
            and rec_inhibitory_input is not None
        ):
            rec_inhibition = owner.branch_rec_inhibition(rec_inhibitory_input)
            denominator = denominator + rec_inhibition

        voltage = numerator / (denominator + owner.epsilon)

        if getattr(owner, "_store_diagnostics", False):
            owner._diag_g_tot = denominator.detach()
            owner._diag_numerator = (
                numerator.detach() if torch.is_tensor(numerator) else None
            )

    else:
        voltage = 0

        if owner.input_excitatory and owner.branch_excitation is not None:
            excitation = owner.branch_excitation(x)
            voltage = voltage + excitation

        if (
            owner.input_recurrent
            and owner.branch_recurrent is not None
            and recurrent_input is not None
        ):
            voltage = voltage + owner.branch_recurrent(recurrent_input)

        if owner.input_branches and branch_input is not None:
            branch_current = owner.branches_to_output(branch_input)
            voltage = voltage + branch_current

        if (
            owner.input_inhibitory
            and owner.branch_inhibition is not None
            and inhibitory_input is not None
        ):
            inhibition = owner.branch_inhibition(inhibitory_input)
            voltage = voltage - inhibition

        if (
            owner.input_rec_inhibitory
            and owner.branch_rec_inhibition is not None
            and rec_inhibitory_input is not None
        ):
            voltage = voltage - owner.branch_rec_inhibition(rec_inhibitory_input)

        voltage = owner.normalize_additive_voltage(voltage)

    if owner.training and "conductance_dynamic" in [
        owner.topk_strategy,
        owner.blocklinear_strategy,
    ]:
        owner.compute_grad_scales(g_total=denominator if owner.use_shunting else None)

    result = owner.reactivation(voltage)
    if getattr(owner, "_store_analysis_currents", False):
        owner._last_analysis_currents = {
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
    raw_E = 0
    if owner.input_excitatory and owner.branch_excitation is not None:
        raw_E = owner.branch_excitation(x)

    raw_E_rec = 0
    if (
        owner.input_recurrent
        and owner.branch_recurrent is not None
        and recurrent_input is not None
    ):
        raw_E_rec = owner.branch_recurrent(recurrent_input)

    raw_I = 0
    if (
        owner.input_inhibitory
        and owner.branch_inhibition is not None
        and inhibitory_input is not None
    ):
        raw_I = owner.branch_inhibition(inhibitory_input)

    raw_I_rec = 0
    if (
        owner.input_rec_inhibitory
        and owner.branch_rec_inhibition is not None
        and rec_inhibitory_input is not None
    ):
        raw_I_rec = owner.branch_rec_inhibition(rec_inhibitory_input)

    return raw_E, raw_E_rec, raw_I, raw_I_rec


def normalize_branch_additive_voltage(owner: Any, voltage):
    """Normalize additive voltages safely across output features."""
    if not owner.use_additive_normalization or not torch.is_tensor(voltage):
        return voltage
    centered = voltage - voltage.mean(dim=-1, keepdim=True)
    std = voltage.std(dim=-1, keepdim=True, unbiased=False)
    return centered / std.clamp_min(owner.epsilon)


def compute_branch_voltage_from_currents(
    owner: Any,
    trace_E,
    trace_E_rec,
    trace_I,
    trace_I_rec=0,
    trace_branch=0,
    branch_conductance=None,
):
    """Compute pre-reactivation voltage from integrated recurrent traces."""
    denominator = None
    if owner.use_shunting:
        numerator = trace_E + trace_E_rec
        if torch.is_tensor(trace_branch):
            numerator = numerator + trace_branch
        elif trace_branch:
            numerator = numerator + trace_branch

        denominator = 1 + trace_E
        if torch.is_tensor(trace_E_rec):
            denominator = denominator + trace_E_rec
        if torch.is_tensor(trace_I):
            denominator = denominator + trace_I
        if torch.is_tensor(trace_I_rec):
            denominator = denominator + trace_I_rec
        if branch_conductance is not None:
            denominator = denominator + branch_conductance
        voltage = numerator / (denominator + owner.epsilon)
    else:
        voltage = trace_E + trace_E_rec - trace_I - trace_I_rec
        if torch.is_tensor(trace_branch):
            voltage = voltage + trace_branch
        elif trace_branch:
            voltage = voltage + trace_branch
        voltage = owner.normalize_additive_voltage(voltage)
    return voltage, denominator


__all__ = [
    "compute_branch_raw_currents",
    "compute_branch_voltage_from_currents",
    "forward_branch_dynamics",
    "normalize_branch_additive_voltage",
]
