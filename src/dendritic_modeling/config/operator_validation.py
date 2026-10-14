"""Config-time compatibility diagnostics for dendritic operator settings.

The dendritic operator space (input domain, weight domain, integration rule,
gates, morphology, topology, output contract) contains combinations that are
legal but subtle. This module surfaces those interactions as warnings at
model-build time so they are visible in every run log, without changing any
behavior: nothing here raises, and each distinct message is emitted once per
process to stay quiet inside sweeps.

Interactions that are genuinely invalid keep failing exactly where they always
did (for example the runtime non-negative-drive check for shunting networks);
this module only makes the run log explain the configuration earlier.
"""

from __future__ import annotations

import logging
from typing import Any

from dendritic_modeling.networks.utils.weight_transforms import (
    POSITIVE_WEIGHT_TRANSFORMS,
)

logger = logging.getLogger(__name__)

_TANH_FAMILY_GATES = {"param_tanh", "param_tanh_only_m"}
_emitted: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _emitted:
        _emitted.add(message)
        logger.warning("operator config: %s", message)


def _get(section: Any, key: str, default: Any = None) -> Any:
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def collect_operator_config_warnings(core_config: Any) -> list[str]:
    """Return diagnostic messages for one core configuration (no side effects)."""

    morphology = _get(core_config, "morphology", {})
    reactivation = _get(core_config, "reactivation", {})
    warnings: list[str] = []

    use_shunting = bool(_get(morphology, "use_shunting", True))
    weight_transform = str(_get(morphology, "weight_transform", "exp")).lower()
    gate_type = str(_get(reactivation, "type", "param_tanh")).lower()
    gate_enabled = bool(_get(reactivation, "enabled", True))
    init_policy = str(_get(reactivation, "init_policy", "analytical")).lower()
    soma_type = _get(reactivation, "soma_type", None)

    if use_shunting and weight_transform not in POSITIVE_WEIGHT_TRANSFORMS:
        warnings.append(
            "shunting with a signed weight transform "
            f"({weight_transform!r}) skips the non-negative drive validation; "
            "the shunting denominator is then not guaranteed positive."
        )
    if (
        gate_enabled
        and init_policy == "occupancy_quantile"
        and gate_type not in _TANH_FAMILY_GATES
    ):
        warnings.append(
            "occupancy_quantile calibration derives (m, b) from tanh occupancy "
            f"formulas but the gate type is {gate_type!r}; the fitted values "
            "are applied through the gate's initialize() interface."
        )
    if soma_type and not gate_enabled:
        warnings.append(
            "reactivation.soma_type is set but reactivation.enabled is false; "
            "the soma override has no effect."
        )
    return warnings


def log_operator_config_warnings(core_config: Any) -> None:
    """Emit each distinct diagnostic once per process."""

    for message in collect_operator_config_warnings(core_config):
        _warn_once(message)
