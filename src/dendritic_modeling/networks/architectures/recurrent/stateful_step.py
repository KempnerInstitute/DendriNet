"""Step-result helpers for stateful recurrent dendritic populations."""

from __future__ import annotations

import torch
from torch import Tensor

from dendritic_modeling.networks.architectures.recurrent.dendritic_spikes import (
    DendriticSpikeDynamics,
    previous_dendritic_spike_state,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState


def _stateful_level_keep_mask(
    active_level_silencing: Tensor | None,
    *,
    level_idx: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    """Resolve a per-level keep mask from the optional lesion mask."""
    if active_level_silencing is None:
        return None

    mask = active_level_silencing.to(device=device)
    if mask.dim() == 1:
        level_mask = mask[level_idx].expand(batch_size)
    elif mask.dim() == 2:
        level_mask = mask[:, level_idx]
    else:
        raise ValueError(
            "Level silencing mask must have shape [n_levels] or [batch, n_levels]"
        )

    keep = 1.0 - level_mask.to(dtype=dtype).clamp(0.0, 1.0)
    return keep.unsqueeze(-1)


def _apply_stateful_dendritic_spike_dynamics(
    *,
    dynamics: DendriticSpikeDynamics,
    level_idx: int,
    branch_voltage: Tensor,
    state: DendriNetState,
    keep_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply configured dendritic spike dynamics to one level."""
    prev_plateau = previous_dendritic_spike_state(
        state.dendritic_spike_plateau,
        level_idx,
        branch_voltage,
    )
    prev_refractory = previous_dendritic_spike_state(
        state.dendritic_spike_refractory,
        level_idx,
        branch_voltage,
    )
    return dynamics.apply(
        level_idx=level_idx,
        branch_voltage=branch_voltage,
        prev_plateau=prev_plateau,
        prev_refractory=prev_refractory,
        keep_mask=keep_mask,
    )


def _apply_stateful_soma_dynamics(
    *,
    spiking_soma: torch.nn.Module | None,
    soma_feedback_enabled: bool,
    soma_voltage: Tensor,
    state: DendriNetState,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Apply optional spiking soma dynamics to one stateful step."""
    if spiking_soma is None:
        v_soma = soma_voltage if soma_feedback_enabled else state.v_soma
        return (
            soma_voltage,
            v_soma,
            state.refractory_counter,
            state.spike_readout,
        )
    return spiking_soma(
        soma_voltage,
        state.v_soma,
        state.refractory_counter,
        state.spike_readout,
    )[:4]


def _build_stateful_next_state(
    *,
    trace_E: list[Tensor],
    trace_E_rec: list[Tensor],
    trace_I: list[Tensor],
    trace_I_rec: list[Tensor],
    trace_branch: list[Tensor],
    level_voltage_cache: list[Tensor],
    typed_traces: dict[str, list[Tensor]] | None,
    output: Tensor,
    v_soma: Tensor | None,
    refractory_counter: Tensor | None,
    spike_readout: Tensor | None,
    dendritic_spike_plateau: list[Tensor] | None,
    dendritic_spike_refractory: list[Tensor] | None,
    dendritic_spike_events: list[Tensor] | None,
    track_branch_voltage: bool,
    track_soma_output: bool,
) -> DendriNetState:
    """Build the next recurrent state from the tensors produced by one step."""
    return DendriNetState(
        trace_E=trace_E,
        trace_E_rec=trace_E_rec,
        trace_I=trace_I,
        trace_I_rec=trace_I_rec,
        trace_branch=trace_branch,
        branch_voltage=level_voltage_cache if track_branch_voltage else None,
        typed_traces=typed_traces,
        soma_output=output if track_soma_output else None,
        v_soma=v_soma,
        refractory_counter=refractory_counter,
        spike_readout=spike_readout,
        dendritic_spike_plateau=dendritic_spike_plateau,
        dendritic_spike_refractory=dendritic_spike_refractory,
        dendritic_spike_events=dendritic_spike_events,
    )


def _build_stateful_routing_info(
    *,
    trace_E: list[Tensor],
    trace_E_rec: list[Tensor],
    trace_I: list[Tensor],
    trace_I_rec: list[Tensor],
    level_voltage_cache: list[Tensor],
    n_levels: int,
) -> dict[str, Tensor]:
    """Build per-level routing diagnostics from completed step tensors."""
    eps = 1e-8
    level_voltages = []
    ff_rec_ratios = []
    for level_idx in range(n_levels):
        level_voltages.append(level_voltage_cache[level_idx].norm(dim=-1))

        ff_mag = trace_E[level_idx].abs().sum(dim=-1) + trace_I[level_idx].abs().sum(
            dim=-1
        )
        rec_mag = trace_E_rec[level_idx].abs().sum(dim=-1) + trace_I_rec[
            level_idx
        ].abs().sum(dim=-1)
        ff_rec_ratios.append(ff_mag / (ff_mag + rec_mag + eps))

    return {
        "level_contributions": torch.stack(level_voltages, dim=-1),
        "ff_rec_ratios": torch.stack(ff_rec_ratios, dim=-1),
    }


__all__ = [
    "_apply_stateful_dendritic_spike_dynamics",
    "_apply_stateful_soma_dynamics",
    "_build_stateful_next_state",
    "_build_stateful_routing_info",
    "_stateful_level_keep_mask",
]
