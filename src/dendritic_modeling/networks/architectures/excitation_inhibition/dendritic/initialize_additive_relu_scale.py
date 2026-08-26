"""Median/MAD current scaling for additive ReLU occupancy calibration."""

from __future__ import annotations

from math import isfinite, log
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


def _solve_two_stage_mad_scales(
    E: torch.Tensor | None,
    I: torch.Tensor | None,
    U: torch.Tensor | None,
    *,
    case: str,
    has_e: bool,
    has_i: bool,
    has_u: bool,
) -> dict[str, Any]:
    """Two-stage MAD scales for identity additive ReLU occupancy.

    Stage 1 sets each bank madstd to 1. Stage 2 (multi-bank only) applies a
    global ``g`` so analytic ``madstd(V)≈1`` with ``V = E - I [+ U]``.
    """
    none_scales = {"s_e": None, "s_i": None, "s_u": None, "g": None}
    bank_tensors = {
        "s_e": E if has_e else None,
        "s_i": I if has_i else None,
        "s_u": U if has_u else None,
    }
    stage1: dict[str, float | None] = {
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
        stage1[key] = 1.0 / bank_mad

    n_sources = int(has_e) + int(has_i) + int(has_u)
    if n_sources <= 1:
        return {
            **stage1,
            "g": 1.0,
            "case": case,
            "reverted": False,
            "revert_reason": "none",
        }

    s_e = stage1["s_e"]
    s_i = stage1["s_i"]
    s_u = stage1["s_u"]
    if case == "e_i":
        assert s_e is not None and s_i is not None and E is not None and I is not None
        unit = s_e * E - s_i * I
    elif case == "e_u":
        assert s_e is not None and s_u is not None and E is not None and U is not None
        unit = s_e * E + s_u * U
    elif case == "e_i_u":
        assert (
            s_e is not None
            and s_i is not None
            and s_u is not None
            and E is not None
            and I is not None
            and U is not None
        )
        unit = s_e * E - s_i * I + s_u * U
    else:
        return {
            **none_scales,
            "case": case,
            "reverted": True,
            "revert_reason": "no_current_banks",
        }

    unit_mad = mad_std(unit)
    if not _positive_finite(unit_mad):
        return {
            **none_scales,
            "case": case,
            "reverted": True,
            "revert_reason": "tiny_madstd",
        }
    g = 1.0 / unit_mad
    return {
        "s_e": None if s_e is None else s_e * g,
        "s_i": None if s_i is None else s_i * g,
        "s_u": None if s_u is None else s_u * g,
        "g": g,
        "case": case,
        "reverted": False,
        "revert_reason": "none",
    }


def solve_additive_relu_scales(
    E: torch.Tensor | None,
    I: torch.Tensor | None,
    U: torch.Tensor | None,
    *,
    enforce_median_balance: bool,
) -> dict[str, Any]:
    """Solve positive scales for additive ReLU current calibration.

    When ``enforce_median_balance`` is False (identity), uses two-stage MAD
    equalization: per-bank madstd→1, then a global ``g`` so analytic
    ``madstd(V)≈1``. When True (softplus/exp), balances bank medians then
    scales so ``madstd(V)≈1``.

    Returns a dict with ``s_e``, ``s_i``, ``s_u`` (``None`` if unused),
    ``case``, ``reverted``, and ``revert_reason``. The identity path also
    includes ``g`` (``1.0`` for single-bank).
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
        return _solve_two_stage_mad_scales(
            E,
            I,
            U,
            case=case,
            has_e=has_e,
            has_i=has_i,
            has_u=has_u,
        )

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
