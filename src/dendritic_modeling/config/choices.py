"""Validation of enumerated configuration values."""

from __future__ import annotations

import difflib
import logging
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

# Gradient-scaling strategies each component implements.
GRADIENT_SCALING_CHOICES = {
    "topk": ("none", "conductance_dynamic"),
    "blocklinear": ("none", "conductance_dynamic", "distal_upweight_by_idx"),
    "reactivation": ("none", "straight_through"),
}
# Earlier versions installed the straight-through gate gradient for any
# reactivation value other than "none"; these names were in use.
REACTIVATION_GRADIENT_SCALING_ALIASES = {
    "inverse": "straight_through",
    "conductance_dynamic": "straight_through",
    "voltage_dynamic": "straight_through",
    "scaled": "straight_through",
}
DEEPST_REWIRING_MODES = ("global", "local", "constant_branch")
DEEPST_PARAM_SPACES = ("log", "presigmoid", "uniform")


def normalize_choice(
    value: Any,
    choices: Iterable[str],
    *,
    field: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Return the canonical spelling of ``value`` or raise.

    Values are compared case-insensitively with ``-`` read as ``_``. Aliases map
    accepted alternative spellings onto canonical choices.
    """
    choices = set(choices)
    text = str(value).strip().lower().replace("-", "_")
    if aliases:
        text = aliases.get(text, text)
    if text in choices:
        return text
    matches = difflib.get_close_matches(text, sorted(choices), n=1)
    hint = f" (did you mean {matches[0]!r}?)" if matches else ""
    raise ValueError(
        f"{field}={value!r} is not supported{hint}; choose one of "
        + ", ".join(sorted(choices))
    )


def normalize_gradient_scaling(
    value: Any, component: str, *, field: str | None = None
) -> str:
    """Return the canonical gradient-scaling strategy of one component.

    ``component`` is ``"topk"``, ``"blocklinear"`` or ``"reactivation"``.
    """
    field = field or f"{component} gradient_scaling"
    raw = "none" if value is None else value
    aliases = (
        REACTIVATION_GRADIENT_SCALING_ALIASES if component == "reactivation" else None
    )
    canonical = normalize_choice(
        raw, GRADIENT_SCALING_CHOICES[component], field=field, aliases=aliases
    )
    if aliases and str(raw).strip().lower().replace("-", "_") in aliases:
        logger.warning("%s=%r is a deprecated name for 'straight_through'.", field, raw)
    return canonical


__all__ = [
    "DEEPST_PARAM_SPACES",
    "DEEPST_REWIRING_MODES",
    "GRADIENT_SCALING_CHOICES",
    "REACTIVATION_GRADIENT_SCALING_ALIASES",
    "normalize_choice",
    "normalize_gradient_scaling",
]
