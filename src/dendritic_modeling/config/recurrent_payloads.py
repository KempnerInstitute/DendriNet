"""Key checks for the structured-EINet recurrent payloads of ``model.core``.

``dynamics``, ``dendritic_spikes``, ``soma_feedback`` and ``recurrent_ei`` are
plain mappings read by the recurrent EINet translators. Each option has a short
name in its own block; ``dynamics`` also accepts long ``spike_*`` spellings and
flat ``dendritic_spike_*`` / ``soma_feedback_*`` spellings of the other blocks.
These checks reject keys no translator reads and options given twice.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dendritic_modeling.config.keys import UnknownConfigKeyError, to_mapping

_DYNAMICS_LONG_NAMES = {
    "mode": "dynamics_mode",
    "threshold": "spike_threshold",
    "reset": "spike_reset",
    "tau": "spike_tau",
    "refractory_steps": "spike_refractory_steps",
    "surrogate_beta": "spike_surrogate_beta",
    "readout": "spike_readout",
    "readout_tau": "spike_readout_tau",
}
DENDRITIC_SPIKE_KEYS = (
    "enabled",
    "mode",
    "levels",
    "threshold",
    "plateau_amplitude",
    "plateau_tau",
    "refractory_steps",
    "surrogate_beta",
    "propagation",
)
SOMA_FEEDBACK_KEYS = (
    "enabled",
    "mode",
    "source",
    "levels",
    "strength",
    "learnable_strength",
    "per_level",
    "init_std",
    "reversal",
)
RECURRENT_EI_KEYS = (
    "enabled",
    "recurrent_layers",
    "dt",
    "tau_base",
    "tau_ratio",
    "level_taus",
    "learnable_tau",
    "epsilon",
    "output_mode",
    "use_transfer",
)
# Second spellings of other options, removed so each option has one name.
REPLACED_RECURRENT_EI_KEYS = {
    "recurrent_per_layer": "model.core.recurrent_ei.recurrent_layers",
    "transfer": "model.core.transfer",
    "transfer_params": "model.core.transfer",
}


def _flat_dendritic_spike_name(key: str) -> str:
    return "dendritic_spikes_enabled" if key == "enabled" else f"dendritic_spike_{key}"


def _flat_soma_feedback_name(key: str) -> str:
    return f"soma_feedback_{key}"


def reject_replaced_recurrent_ei_keys(recurrent_ei: Mapping[str, Any]) -> None:
    """Fail on ``recurrent_ei`` keys that were second spellings of other options."""
    if "transfer_inhibitory_mode" in recurrent_ei:
        raise ValueError(
            "model.core.recurrent_ei.transfer_inhibitory_mode was removed on "
            "2026-10-08 with transfer.inhibitory_mode: which cells read the "
            "transferred inhibitory stream follows from "
            "architecture.inhibitory_layer_sizes and the connectivity synapse "
            "counts. Remove the key."
        )
    replaced = [
        f"model.core.recurrent_ei.{key} -> {target}"
        for key, target in REPLACED_RECURRENT_EI_KEYS.items()
        if key in recurrent_ei
    ]
    if replaced:
        raise ValueError(
            "These recurrent_ei options were second spellings of other options "
            "and are no longer read; set them at the listed location: "
            + "; ".join(replaced)
        )


def _check(block: Mapping[str, Any], allowed, section: str) -> None:
    unknown = sorted(set(block) - set(allowed))
    if unknown:
        raise UnknownConfigKeyError(section, unknown, allowed)


def _reject_twice(block: Mapping[str, Any], first: str, second: str, where: str):
    if first in block and second in block:
        raise ValueError(
            f"{where} sets both {first} and {second}, which name the same option; "
            "keep one"
        )


def check_recurrent_payloads(core: Mapping[str, Any]) -> None:
    """Validate the recurrent payload mappings of one ``model.core`` mapping."""
    dynamics, spikes, feedback, recurrent = (
        to_mapping(core.get(name), section=f"model.core.{name}")
        for name in ("dynamics", "dendritic_spikes", "soma_feedback", "recurrent_ei")
    )

    flat_spikes = {_flat_dendritic_spike_name(k): k for k in DENDRITIC_SPIKE_KEYS}
    flat_feedback = {_flat_soma_feedback_name(k): k for k in SOMA_FEEDBACK_KEYS}
    allowed = (
        set(_DYNAMICS_LONG_NAMES)
        | set(_DYNAMICS_LONG_NAMES.values())
        | set(flat_spikes)
        | set(flat_feedback)
    )
    _check(dynamics, allowed, "model.core.dynamics")
    for short, long in _DYNAMICS_LONG_NAMES.items():
        _reject_twice(dynamics, short, long, "model.core.dynamics")

    _check(spikes, DENDRITIC_SPIKE_KEYS, "model.core.dendritic_spikes")
    _check(feedback, SOMA_FEEDBACK_KEYS, "model.core.soma_feedback")
    for flat, key in flat_spikes.items():
        if flat in dynamics and key in spikes:
            raise ValueError(
                f"model.core.dynamics.{flat} and model.core.dendritic_spikes.{key} "
                "name the same option; keep model.core.dendritic_spikes"
            )
    for flat, key in flat_feedback.items():
        if flat in dynamics and key in feedback:
            raise ValueError(
                f"model.core.dynamics.{flat} and model.core.soma_feedback.{key} "
                "name the same option; keep model.core.soma_feedback"
            )

    reject_replaced_recurrent_ei_keys(recurrent)
    _check(recurrent, RECURRENT_EI_KEYS, "model.core.recurrent_ei")


__all__ = [
    "DENDRITIC_SPIKE_KEYS",
    "RECURRENT_EI_KEYS",
    "REPLACED_RECURRENT_EI_KEYS",
    "SOMA_FEEDBACK_KEYS",
    "check_recurrent_payloads",
    "reject_replaced_recurrent_ei_keys",
]
