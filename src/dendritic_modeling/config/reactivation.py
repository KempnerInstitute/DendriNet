"""Shared reactivation initialization policy helpers."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from typing import Any

from dendritic_modeling.config.choices import normalize_choice

logger = logging.getLogger(__name__)

# Canonical default reactivation-init policy for additive cores that omit an
# explicit ``reactivation.init_policy``. Additive voltage is fragile under the
# analytical/fixed gate (depth- and normalization-dependent), so the data-driven
# occupancy_quantile calibration is the safe default. Route additive-default
# assignments through this constant rather than hardcoding the string.
DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY = "occupancy_quantile"

REACTIVATION_INIT_POLICY_ALIASES = {
    "analytical": "analytical",
    "auto": "analytical",
    "fixed": "fixed",
    "manual": "fixed",
    "empirical": "empirical",
    # Backward-compatible alias. The canonical public name is
    # occupancy_quantile because the rule maps voltage quantiles to target
    # reactivation occupancies.
    "quantile": "occupancy_quantile",
    "occupancy_quantile": "occupancy_quantile",
    # Deterministic center/slope decomposition of the analytical and occupancy
    # policies. These use no fitted mixing coefficient.
    "analytical_slope_occupancy_center": "analytical_slope_occupancy_center",
    "occupancy_slope_analytical_center": "occupancy_slope_analytical_center",
}


REACTIVATION_INIT_POLICIES = frozenset(REACTIVATION_INIT_POLICY_ALIASES.values())

# How training updates the gate parameters (m, b): by the optimizer or local
# rule ("backprop"), by periodic recalibration only ("quantile"), or not at
# all ("frozen").
REACTIVATION_UPDATE_MODES = ("backprop", "quantile", "frozen")
# Calibration rule of periodic refreshes; "from_init_policy" reuses each
# layer's initialization rule.
REACTIVATION_RECALIBRATION_MODES = (
    "from_init_policy",
    "occupancy_quantile",
    "median_mad",
    "mean_std",
)
# Deprecated spellings that earlier versions accepted.
REACTIVATION_UPDATE_MODE_ALIASES = {
    "bp": "backprop",
    "backpropagation": "backprop",
    "quantile_only": "quantile",
    "quantile_rule": "quantile",
    "freeze": "frozen",
}
REACTIVATION_RECALIBRATION_MODE_ALIASES = {
    "auto": "from_init_policy",
    "init_policy": "from_init_policy",
    "quantile": "occupancy_quantile",
    "empirical": "median_mad",
}


def normalize_reactivation_init_policy(policy: str | None) -> str:
    """Return the canonical reactivation-init policy name, or raise."""

    key = "analytical" if policy is None else str(policy).strip().lower()
    if key == "quantile":
        warnings.warn(
            "reactivation init policy 'quantile' is deprecated; use "
            "'occupancy_quantile' for the canonical occupancy-calibration rule.",
            DeprecationWarning,
            stacklevel=2,
        )
    return normalize_choice(
        key,
        REACTIVATION_INIT_POLICIES,
        field="reactivation init_policy",
        aliases=REACTIVATION_INIT_POLICY_ALIASES,
    )


_WARNED_UNUSED_GATE_INIT: set[str] = set()


def warn_unused_gate_init(
    settings: Mapping[str, Any],
    *,
    where: str,
    policy_key: str,
    value_keys: tuple[str, ...],
    default_policy: str = "analytical",
) -> None:
    """Warn once when gate start values are written under a policy that ignores them.

    Only ``init_policy: fixed`` reads the start slope and midpoint; the other
    policies set the gate from the initialization rule or from data.
    """
    written = [key for key in value_keys if key in settings]
    if not written:
        return
    policy = normalize_reactivation_init_policy(
        settings.get(policy_key, default_policy)
    )
    if policy == "fixed":
        return
    once = f"{where}:{','.join(written)}:{policy}"
    if once in _WARNED_UNUSED_GATE_INIT:
        return
    _WARNED_UNUSED_GATE_INIT.add(once)
    logger.warning(
        "%s in %s %s no effect: %s is %r, which sets the gate itself; only "
        "'fixed' uses written start values. Remove them or select 'fixed'.",
        ", ".join(written),
        where,
        "has" if len(written) == 1 else "have",
        policy_key,
        policy,
    )


def reactivation_policy_to_calibration_mode(policy: str | None) -> str | None:
    """Map a canonical init policy to its data-driven calibration mode."""

    canonical = normalize_reactivation_init_policy(policy)
    return {
        "empirical": "median_mad",
        "occupancy_quantile": "occupancy_quantile",
        "analytical_slope_occupancy_center": ("analytical_slope_occupancy_center"),
        "occupancy_slope_analytical_center": ("occupancy_slope_analytical_center"),
    }.get(canonical)


REACTIVATION_CALIBRATION_MODES = frozenset(
    {
        "median_mad",
        "mean_std",
        "occupancy_quantile",
        "analytical_slope_occupancy_center",
        "occupancy_slope_analytical_center",
    }
)


def resolve_reactivation_calibration_mode(mode: str) -> str:
    """Return a calibration mode named directly or through a data-driven policy."""

    key = str(mode).strip().lower()
    if key in REACTIVATION_CALIBRATION_MODES:
        return key
    calibration = reactivation_policy_to_calibration_mode(key)
    if calibration is None:
        raise ValueError(
            f"reactivation calibration mode {mode!r} is neither a calibration mode "
            f"({', '.join(sorted(REACTIVATION_CALIBRATION_MODES))}) nor a "
            "data-driven init policy"
        )
    return calibration


def is_data_driven_reactivation_policy(policy: str | None) -> bool:
    """Whether this init policy requires a post-build calibration pass."""

    return reactivation_policy_to_calibration_mode(policy) is not None


# Gate policies that leave the slope, the midpoint or both uncalibrated on
# measured voltages, with what each uses instead.
_UNCALIBRATED_GATE_PARTS = {
    "analytical": "the slope and midpoint of the initialization rule's prior",
    "fixed": "the written init_m and init_b",
    "analytical_slope_occupancy_center": "the slope of the initialization rule's prior",
    "occupancy_slope_analytical_center": (
        "the midpoint of the initialization rule's prior"
    ),
}


def warn_integration_gate_pairing(
    *,
    weight_transform: str | None,
    use_shunting: bool,
    additive_mode: str | None = "raw",
    additive_normalization: bool = False,
    reactivation_init_policy: str | None = None,
    reactivate: bool = True,
) -> str | None:
    """Warn about two allowed but fragile integration settings.

    * Shunting integration of signed weights (``weight_transform: identity``).
      Shunting divides by the total conductance (leak plus the summed synaptic
      and child-branch conductances); signed weights can make that total zero
      or negative, where the branch voltage diverges or changes sign.
    * Additive integration of positive conductances (``softplus`` or ``exp``
      weights) with a gate policy that does not calibrate both the gate's slope
      and its midpoint on measured voltages. The summed drive then has a
      nonzero mean and a scale set by fan-in, depth and the input distribution,
      so a gate taken from the initialization rule's prior or from fixed values
      can start saturated or flat. ``occupancy_quantile`` (the default for
      additive structured E/I and population-network cores; unified E/I cores
      default to ``analytical``) and ``empirical`` calibrate both. ``raw`` voltage with
      the across-feature normalization (``additive_normalization``) is
      standardized, and ``tangent_matched`` voltage is centred on its frozen
      operating point, so neither is flagged.

    Returns the warning text, also issued with ``warnings.warn``, or ``None``.
    """

    transform = str(weight_transform or "").lower()
    positive_transform = transform in ("softplus", "exp")
    if use_shunting and transform == "identity":
        message = (
            "Shunting integration with signed weights (weight_transform "
            "'identity'): the shunting denominator, leak plus the summed "
            "conductances, can reach zero or turn negative, where the branch "
            "voltage diverges or changes sign. Use a positive weight transform "
            "('softplus' or 'exp') for shunting, or additive integration for "
            "signed weights."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        return message
    if use_shunting or not reactivate or not positive_transform:
        return None
    mode = str(additive_mode or "raw").lower()
    if mode not in ("raw", "conductance_normalized"):
        return None
    if mode == "raw" and additive_normalization:
        return None
    policy = normalize_reactivation_init_policy(reactivation_init_policy)
    uses = _UNCALIBRATED_GATE_PARTS.get(policy)
    if uses is None:
        return None
    message = (
        f"Additive integration (additive_mode {mode!r}) of positive "
        f"conductances with reactivation init_policy {policy!r}, which uses "
        f"{uses}: the summed drive has a nonzero mean and a scale set by "
        "fan-in, depth and the input distribution, so the gate can start "
        f"saturated or flat. {DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY!r} "
        "and 'empirical' calibrate the slope and midpoint on measured "
        "voltages."
    )
    warnings.warn(message, UserWarning, stacklevel=2)
    return message


def collect_model_data_driven_reactivation_policies(model) -> tuple[set[str], bool]:
    """Return data-driven module policies and whether any module policy exists."""

    requested_policies: set[str] = set()
    saw_layer_policy = False
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return requested_policies, saw_layer_policy

    for module in modules():
        layer_policy = getattr(module, "reactivation_init_policy", None)
        if layer_policy is None:
            continue
        saw_layer_policy = True
        layer_policy = normalize_reactivation_init_policy(layer_policy)
        if is_data_driven_reactivation_policy(layer_policy):
            requested_policies.add(layer_policy)

    return requested_policies, saw_layer_policy


def _canonical_mode(value, choices, aliases, *, field: str, default: str) -> str:
    text = str(value if value is not None else "").strip().lower().replace("-", "_")
    text = text or default
    canonical = normalize_choice(text, choices, field=field, aliases=aliases)
    if text in aliases:
        warnings.warn(
            f"{field}={value!r} is deprecated; write {canonical!r}.",
            DeprecationWarning,
            stacklevel=3,
        )
    return canonical


def normalize_reactivation_update_mode(mode: Any) -> str:
    """Return the canonical ``reactivation_update_mode`` (None is backprop)."""
    return _canonical_mode(
        mode,
        REACTIVATION_UPDATE_MODES,
        REACTIVATION_UPDATE_MODE_ALIASES,
        field="reactivation_update_mode",
        default="backprop",
    )


def normalize_reactivation_recalibration_mode(mode: Any) -> str:
    """Return the canonical ``reactivation_recalibration_mode``."""
    return _canonical_mode(
        mode,
        REACTIVATION_RECALIBRATION_MODES,
        REACTIVATION_RECALIBRATION_MODE_ALIASES,
        field="reactivation_recalibration_mode",
        default="from_init_policy",
    )


def validate_reactivation_updates(
    *,
    update_mode: Any,
    recalibration_mode: Any,
    recalibrate_every: Any,
    start_epoch: Any,
    num_batches: Any,
    ema_alpha: Any,
) -> tuple[str, str]:
    """Check the training-time gate update options; return the canonical modes.

    ``quantile`` updates the gates only by periodic recalibration, so it needs
    ``recalibrate_reactivation_every > 0`` (without it the gates stay fixed,
    which is ``frozen``).
    """
    update = normalize_reactivation_update_mode(update_mode)
    recalibration = normalize_reactivation_recalibration_mode(recalibration_mode)
    every = int(recalibrate_every or 0)
    start_epoch = 1 if start_epoch is None else start_epoch
    num_batches = 3 if num_batches is None else num_batches
    ema_alpha = 1.0 if ema_alpha is None else ema_alpha
    if every < 0:
        raise ValueError(
            f"recalibrate_reactivation_every must be >= 0 (0 disables periodic "
            f"recalibration), got {recalibrate_every}"
        )
    if update == "quantile" and every == 0:
        raise ValueError(
            "reactivation_update_mode='quantile' updates the gates only by "
            "periodic recalibration, but recalibrate_reactivation_every is 0, so "
            "they would stay at their initial values. Set "
            "recalibrate_reactivation_every > 0, or write 'frozen'."
        )
    if int(start_epoch) < 1:
        raise ValueError(
            "reactivation_recalibration_start_epoch counts epochs from 1, got "
            f"{start_epoch}"
        )
    if int(num_batches) < 1:
        raise ValueError(
            "reactivation_recalibration_num_batches must be >= 1, got " f"{num_batches}"
        )
    if not 0.0 < float(ema_alpha) <= 1.0:
        raise ValueError(
            "reactivation_recalibration_ema_alpha must lie in (0, 1], got "
            f"{ema_alpha}"
        )
    return update, recalibration


__all__ = [
    "DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY",
    "REACTIVATION_INIT_POLICY_ALIASES",
    "REACTIVATION_RECALIBRATION_MODES",
    "REACTIVATION_UPDATE_MODES",
    "collect_model_data_driven_reactivation_policies",
    "is_data_driven_reactivation_policy",
    "normalize_reactivation_init_policy",
    "normalize_reactivation_recalibration_mode",
    "normalize_reactivation_update_mode",
    "reactivation_policy_to_calibration_mode",
    "validate_reactivation_updates",
    "warn_integration_gate_pairing",
    "warn_unused_gate_init",
]
