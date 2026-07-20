"""Shunting occupancy calibration: median(E)=median(I)=1 and G_j budget."""

from __future__ import annotations

from math import isfinite
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_additive_relu_scale import (
    _median,
    _positive_finite,
    scale_synapse_params_,
)

_TARGET_MEDIAN = 1.0


def _bank_present(tensor: torch.Tensor | None) -> bool:
    return (
        tensor is not None
        and tensor.numel() > 0
        and bool(torch.isfinite(tensor).any())
    )


def _is_plain_relu_or_identity(reactivation: nn.Module) -> bool:
    """True for ``nn.ReLU`` or ``nn.Identity``, not ParametricReLU."""
    from dendritic_modeling.networks.activations.parametric import ParametricReLU

    if isinstance(reactivation, nn.Identity):
        return True
    return isinstance(reactivation, nn.ReLU) and not isinstance(
        reactivation, ParametricReLU
    )


def is_shunting_occupancy_scale_eligible(layer) -> bool:
    """Whether this branch layer should use shunting occupancy current scaling."""
    if not getattr(layer, "use_shunting", False):
        return False
    if not _is_plain_relu_or_identity(getattr(layer, "reactivation", None)):
        return False
    transform = str(getattr(layer, "weight_transform", "") or "").lower()
    return transform in {"softplus", "exp"}


def solve_shunting_occupancy_scales(
    E: torch.Tensor | None,
    I: torch.Tensor | None,
    *,
    block_size: int | None = None,
) -> dict[str, Any]:
    """Solve E/I median=1 scales and optional per-branch conductance target.

    Returns ``s_e``, ``s_i``, ``g_branch``, ``reverted``, ``revert_reason``.
    ``g_branch`` is ``median(E'+I'+1)/block_size`` when ``block_size`` is
    provided; otherwise ``None``.
    """
    has_e = _bank_present(E)
    has_i = _bank_present(I)
    none_result = {
        "s_e": None,
        "s_i": None,
        "g_branch": None,
        "median_pool": None,
    }

    if not has_e and not has_i and block_size is None:
        return {
            **none_result,
            "reverted": True,
            "revert_reason": "no_current_banks",
        }

    s_e = None
    s_i = None
    e_scaled = None
    i_scaled = None

    if has_e:
        m_e = _median(E)
        if not _positive_finite(m_e):
            return {
                **none_result,
                "reverted": True,
                "revert_reason": "nonpositive_median",
            }
        s_e = _TARGET_MEDIAN / m_e
        e_scaled = E.double() * s_e

    if has_i:
        m_i = _median(I)
        if not _positive_finite(m_i):
            return {
                **none_result,
                "reverted": True,
                "revert_reason": "nonpositive_median",
            }
        s_i = _TARGET_MEDIAN / m_i
        i_scaled = I.double() * s_i

    g_leak = 1.0

    g_branch = None
    median_pool = None
    if block_size is not None:
        n = max(int(block_size), 1)
        if e_scaled is not None and i_scaled is not None:
            # Align lengths by flattening independently then using broadcast via
            # median of the sum on paired samples when shapes match; otherwise
            # fall back to median(E')+median(I')+g_leak.
            if e_scaled.shape == i_scaled.shape:
                pool = e_scaled + i_scaled + g_leak
                median_pool = _median(pool)
            else:
                median_pool = _median(e_scaled) + _median(i_scaled) + g_leak
        elif e_scaled is not None:
            median_pool = _median(e_scaled + g_leak)
        elif i_scaled is not None:
            median_pool = _median(i_scaled + g_leak)
        else:
            median_pool = g_leak

        if not _positive_finite(float(median_pool)):
            return {
                "s_e": s_e,
                "s_i": s_i,
                "g_branch": None,
                "median_pool": median_pool,
                "reverted": True,
                "revert_reason": "nonpositive_median_pool",
            }
        g_branch = float(median_pool) / n

    return {
        "s_e": s_e,
        "s_i": s_i,
        "g_branch": g_branch,
        "median_pool": median_pool,
        "reverted": False,
        "revert_reason": "none",
    }


def calibrate_shunting_occupancy_currents(
    layer,
    *,
    E: torch.Tensor | None,
    I: torch.Tensor | None,
) -> dict[str, Any]:
    """Scale E/I to median 1 and constant-set branch couplings when present."""
    transform = str(getattr(layer, "weight_transform", "softplus") or "softplus").lower()
    has_branches = bool(getattr(layer, "input_branches", False)) and hasattr(
        layer, "branches_to_output"
    )
    block_size = (
        int(getattr(layer.branches_to_output, "block_size")) if has_branches else None
    )

    # Require the module to exist as well as samples for a bank to count.
    if getattr(layer, "branch_excitation", None) is None:
        E = None
    if getattr(layer, "branch_inhibition", None) is None:
        I = None

    result = solve_shunting_occupancy_scales(E, I, block_size=block_size)
    if result["reverted"]:
        return result

    if result["s_e"] is not None and layer.branch_excitation is not None:
        scale_synapse_params_(layer.branch_excitation, float(result["s_e"]), transform)
    if result["s_i"] is not None and layer.branch_inhibition is not None:
        scale_synapse_params_(layer.branch_inhibition, float(result["s_i"]), transform)

    if has_branches and result["g_branch"] is not None:
        g_j = float(result["g_branch"])
        if not _positive_finite(g_j):
            return {
                **result,
                "reverted": True,
                "revert_reason": "nonpositive_g_branch",
            }
        layer.branches_to_output.initialize(g_branch=g_j)

    return result


__all__ = [
    "calibrate_shunting_occupancy_currents",
    "is_shunting_occupancy_scale_eligible",
    "solve_shunting_occupancy_scales",
]
