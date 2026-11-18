"""Shared constants for population-network routing."""

from __future__ import annotations

from dendritic_modeling.config.population_routing import (
    INPUT_ALIASES as _EXTERNAL_INPUT_ALIASES,
)

_INPUT_SOURCE = "input"
_INPUT_E_SOURCE = "input_e"
_INPUT_I_SOURCE = "input_i"
_EXTERNAL_INPUT_SOURCES = {_INPUT_SOURCE, _INPUT_E_SOURCE, _INPUT_I_SOURCE}
_SAME_STEP = "same_step"
_DELAYED = "delayed"
_FF_EXC = "ff_excitatory"
_FF_INH = "ff_inhibitory"
_REC_EXC = "rec_excitatory"
_REC_INH = "rec_inhibitory"

__all__ = [
    "_DELAYED",
    "_EXTERNAL_INPUT_ALIASES",
    "_EXTERNAL_INPUT_SOURCES",
    "_FF_EXC",
    "_FF_INH",
    "_INPUT_E_SOURCE",
    "_INPUT_I_SOURCE",
    "_INPUT_SOURCE",
    "_REC_EXC",
    "_REC_INH",
    "_SAME_STEP",
]
