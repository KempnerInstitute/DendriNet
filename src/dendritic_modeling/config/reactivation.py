"""Shared reactivation initialization policy helpers."""

from __future__ import annotations

import warnings

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
}


def normalize_reactivation_init_policy(policy: str | None) -> str:
    """Return the canonical reactivation-init policy name."""

    key = "analytical" if policy is None else str(policy).strip().lower()
    if key == "quantile":
        warnings.warn(
            "reactivation init policy 'quantile' is deprecated; use "
            "'occupancy_quantile' for the canonical occupancy-calibration rule.",
            DeprecationWarning,
            stacklevel=2,
        )
    return REACTIVATION_INIT_POLICY_ALIASES.get(key, key)


def reactivation_policy_to_calibration_mode(policy: str | None) -> str | None:
    """Map a canonical init policy to its data-driven calibration mode."""

    canonical = normalize_reactivation_init_policy(policy)
    return {
        "empirical": "median_mad",
        "occupancy_quantile": "occupancy_quantile",
    }.get(canonical)


def is_data_driven_reactivation_policy(policy: str | None) -> bool:
    """Whether this init policy requires a post-build calibration pass."""

    return reactivation_policy_to_calibration_mode(policy) is not None


__all__ = [
    "REACTIVATION_INIT_POLICY_ALIASES",
    "is_data_driven_reactivation_policy",
    "normalize_reactivation_init_policy",
    "reactivation_policy_to_calibration_mode",
]
