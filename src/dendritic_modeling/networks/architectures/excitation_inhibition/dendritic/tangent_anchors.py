"""Shared validation for explicit, soma-first integration anchors."""

from __future__ import annotations

import math


def resolve_tangent_anchors_by_depth(
    n0_by_depth, t0_by_depth, *, n_levels, scalar_n0, scalar_t0, epsilon
):
    """Return validated anchors without calibrating or changing random state."""
    n0 = tuple(float(value) for value in n0_by_depth)
    t0 = tuple(float(value) for value in t0_by_depth)
    if bool(n0) != bool(t0):
        raise ValueError(
            "additive_tangent_n0_by_depth and additive_tangent_t0_by_depth "
            "must be provided together"
        )
    if not n0:
        return n0, t0
    if scalar_n0 is not None or scalar_t0 is not None:
        raise ValueError(
            "configure either scalar additive tangent anchors or "
            "soma-relative per-depth anchors, not both"
        )
    if len(n0) != n_levels or len(t0) != n_levels:
        raise ValueError(
            "soma-relative additive tangent anchors must each contain exactly "
            f"{n_levels} values, found n0={len(n0)}, t0={len(t0)}"
        )
    for name, values in (("n0", n0), ("t0", t0)):
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"additive_tangent_{name}_by_depth must be finite")
    if any(1 + value + epsilon <= 0 for value in t0):
        raise ValueError(
            "every per-depth tangent denominator must satisfy 1 + t0 + epsilon > 0"
        )
    return n0, t0
