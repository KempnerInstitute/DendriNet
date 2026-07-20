"""Median/MAD current scaling for additive ReLU occupancy calibration."""

from __future__ import annotations

from math import isfinite, log, sqrt
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.utils.weight_transforms import (
    apply_weight_transform,
    inverse_softplus,
)

_MAD_STD_FACTOR = 1.4826
_SCALE_FLOOR = 1e-8


def mad_std(x: torch.Tensor) -> float:
    """Return 1.4826 * MAD of a 1-D tensor."""
    flat = x.reshape(-1).double()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    median = float(torch.median(flat).item())
    mad = float(torch.median(torch.abs(flat - median)).item())
    return _MAD_STD_FACTOR * mad


def _median(x: torch.Tensor) -> float:
    flat = x.reshape(-1).double()
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return float("nan")
    return float(torch.median(flat).item())


def _positive_finite(value: float, floor: float = _SCALE_FLOOR) -> bool:
    return isfinite(value) and value > floor


def solve_additive_relu_scales(
    E: torch.Tensor | None,
    I: torch.Tensor | None,
    U: torch.Tensor | None,
    *,
    enforce_median_balance: bool,
) -> dict[str, Any]:
    """Solve positive scales for additive ReLU current calibration.

    Returns a dict with ``s_e``, ``s_i``, ``s_u`` (``None`` if unused),
    ``case``, ``reverted``, and ``revert_reason``.
    """
    has_e = E is not None and E.numel() > 0 and torch.isfinite(E).any()
    has_i = I is not None and I.numel() > 0 and torch.isfinite(I).any()
    has_u = U is not None and U.numel() > 0 and torch.isfinite(U).any()

    none_scales = {"s_e": None, "s_i": None, "s_u": None}

    if has_e and has_i and has_u:
        case = "e_i_u"
    elif has_e and has_i:
        case = "e_i"
    elif has_e and has_u:
        case = "e_u"
    elif has_e:
        case = "e"
    elif has_u:
        case = "u"
    else:
        return {
            **none_scales,
            "case": "none",
            "reverted": True,
            "revert_reason": "no_current_banks",
        }

    if not enforce_median_balance:
        # Per-bank MAD-std budgets: madstd(s_x X) = 1/sqrt(n_sources) so that
        # under independence Var(V) ≈ sum madstd^2 = 1.
        n_sources = int(has_e) + int(has_i) + int(has_u)
        target_mad = 1.0 / sqrt(n_sources)
        bank_tensors = {
            "s_e": E if has_e else None,
            "s_i": I if has_i else None,
            "s_u": U if has_u else None,
        }
        scales: dict[str, float | None] = {
            "s_e": None,
            "s_i": None,
            "s_u": None,
        }
        for key, tensor in bank_tensors.items():
            if tensor is None:
                continue
            bank_mad = mad_std(tensor)
            if not _positive_finite(bank_mad):
                return {
                    **none_scales,
                    "case": case,
                    "reverted": True,
                    "revert_reason": "tiny_madstd",
                }
            scales[key] = target_mad / bank_mad
        return {
            **scales,
            "case": case,
            "reverted": False,
            "revert_reason": "none",
        }

    if case == "e_i_u":
        m_e, m_i, m_u = _median(E), _median(I), _median(U)
        if not (
            _positive_finite(m_e) and _positive_finite(m_i) and _positive_finite(m_u)
        ):
            return {
                **none_scales,
                "case": case,
                "reverted": True,
                "revert_reason": "nonpositive_median",
            }
        unit = E + (m_e / m_u) * U - (2.0 * m_e / m_i) * I
        unit_mad = mad_std(unit)
        if not _positive_finite(unit_mad):
            return {
                **none_scales,
                "case": case,
                "reverted": True,
                "revert_reason": "tiny_madstd",
            }
        s_e = 1.0 / unit_mad
        return {
            "s_e": s_e,
            "s_i": s_e * (2.0 * m_e / m_i),
            "s_u": s_e * (m_e / m_u),
            "case": case,
            "reverted": False,
            "revert_reason": "none",
        }

    if case == "e_i":
        m_e, m_i = _median(E), _median(I)
        if not (_positive_finite(m_e) and _positive_finite(m_i)):
            return {
                **none_scales,
                "case": case,
                "reverted": True,
                "revert_reason": "nonpositive_median",
            }
        unit = E - (m_e / m_i) * I
        unit_mad = mad_std(unit)
        if not _positive_finite(unit_mad):
            return {
                **none_scales,
                "case": case,
                "reverted": True,
                "revert_reason": "tiny_madstd",
            }
        s_e = 1.0 / unit_mad
        return {
            "s_e": s_e,
            "s_i": s_e * (m_e / m_i),
            "s_u": None,
            "case": case,
            "reverted": False,
            "revert_reason": "none",
        }

    if case == "e_u":
        m_e, m_u = _median(E), _median(U)
        if not (_positive_finite(m_e) and _positive_finite(m_u)):
            return {
                **none_scales,
                "case": case,
                "reverted": True,
                "revert_reason": "nonpositive_median",
            }
        unit = E + (m_e / m_u) * U
        unit_mad = mad_std(unit)
        if not _positive_finite(unit_mad):
            return {
                **none_scales,
                "case": case,
                "reverted": True,
                "revert_reason": "tiny_madstd",
            }
        s_e = 1.0 / unit_mad
        return {
            "s_e": s_e,
            "s_i": None,
            "s_u": s_e * (m_e / m_u),
            "case": case,
            "reverted": False,
            "revert_reason": "none",
        }

    # Single-bank cases e / u
    bank = E if case == "e" else U
    unit_mad = mad_std(bank)
    if not _positive_finite(unit_mad):
        return {
            **none_scales,
            "case": case,
            "reverted": True,
            "revert_reason": "tiny_madstd",
        }
    s = 1.0 / unit_mad
    return {
        "s_e": s if case == "e" else None,
        "s_i": None,
        "s_u": s if case == "u" else None,
        "case": case,
        "reverted": False,
        "revert_reason": "none",
    }


def scale_synapse_params_(
    module: nn.Module,
    scale: float,
    weight_transform: str,
) -> None:
    """In-place scale a TopK/BlockLinear module so transformed weights *= scale."""
    scale = float(scale)
    if not isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale}")
    if abs(scale - 1.0) < 1e-15:
        return

    transform = str(weight_transform).lower()
    if hasattr(module, "pre_w"):
        param = module.pre_w
    elif hasattr(module, "log_weight"):
        param = module.log_weight
    else:
        raise TypeError(
            f"Module {type(module).__name__} has neither pre_w nor log_weight"
        )

    with torch.no_grad():
        if transform == "identity":
            param.data.mul_(scale)
        elif transform == "exp":
            param.data.add_(log(scale))
        elif transform == "softplus":
            transformed = apply_weight_transform(param.data, "softplus")
            param.data.copy_(inverse_softplus(transformed * scale))
        else:
            raise ValueError(
                f"Unsupported weight_transform for additive ReLU scaling: {transform}"
            )


def _is_plain_relu(reactivation: nn.Module) -> bool:
    """True for ``nn.ReLU`` but not parametric / custom ReLU subclasses we own."""
    from dendritic_modeling.networks.activations.parametric import ParametricReLU

    return isinstance(reactivation, nn.ReLU) and not isinstance(
        reactivation, ParametricReLU
    )


def is_additive_relu_scale_eligible(layer) -> bool:
    """Whether this branch layer should use additive ReLU current scaling."""
    if getattr(layer, "use_shunting", True):
        return False
    if not _is_plain_relu(getattr(layer, "reactivation", None)):
        return False
    transform = str(getattr(layer, "weight_transform", "") or "").lower()
    return transform in {"softplus", "exp", "identity"}


def calibrate_additive_relu_currents(
    layer,
    *,
    E: torch.Tensor | None,
    I: torch.Tensor | None,
    U: torch.Tensor | None,
    enforce_median_balance: bool | None = None,
) -> dict[str, Any]:
    """Solve and apply additive ReLU current scales on one branch layer."""
    transform = str(getattr(layer, "weight_transform", "identity") or "identity").lower()
    if enforce_median_balance is None:
        enforce_median_balance = transform in {"softplus", "exp"}

    result = solve_additive_relu_scales(
        E, I, U, enforce_median_balance=enforce_median_balance
    )
    if result["reverted"]:
        return result

    bank_scales = (
        ("branch_excitation", result["s_e"]),
        ("branch_inhibition", result["s_i"]),
        ("branches_to_output", result["s_u"]),
    )
    for attr, scale in bank_scales:
        if scale is None:
            continue
        mod = getattr(layer, attr, None)
        if mod is None:
            continue
        scale_synapse_params_(mod, float(scale), transform)

    return result


__all__ = [
    "calibrate_additive_relu_currents",
    "is_additive_relu_scale_eligible",
    "mad_std",
    "scale_synapse_params_",
    "solve_additive_relu_scales",
]
