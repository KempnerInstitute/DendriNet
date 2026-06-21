from __future__ import annotations

import warnings
from math import atanh, isfinite, sqrt

import torch

from dendritic_modeling.config.reactivation import (
    is_data_driven_reactivation_policy,
    normalize_reactivation_init_policy,
    reactivation_policy_to_calibration_mode,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_expectations import (
    _aggregate_quantiles_from_chunks,
)
from dendritic_modeling.utils.hooks import (
    iter_named_modules_of_type,
    remove_hook_handles,
)


def _apply_reactivation_init(
    dbl,
    m_auto: float,
    b_auto: float,
) -> None:
    """
    Initialize the reactivation ``(m, b)`` from the layer's configured policy.

    ``analytical`` uses the values supplied by the selected DBL init rule.
    ``fixed`` uses the explicit ``reactivation_init_m`` / ``init_b`` values.
    Data-driven policies use the analytical fallback here and are overwritten
    later by calibration from sampled voltages.
    """
    initialize = getattr(dbl.reactivation, "initialize", None)
    if isinstance(dbl.reactivation, torch.nn.Identity) or not callable(initialize):
        return

    policy = normalize_reactivation_init_policy(
        getattr(dbl, "reactivation_init_policy", "analytical")
    )

    if policy == "fixed":
        initialize(dbl.reactivation_init_m, dbl.reactivation_init_b)
        return

    initialize(m_auto, b_auto)


def _apply_reactivation_calibration(
    layer,
    *,
    m_target: float,
    b_target: float,
    ema_alpha: float = 1.0,
) -> tuple[float, float]:
    """Write calibrated reactivation parameters, optionally with EMA blending."""

    alpha = float(max(0.0, min(1.0, ema_alpha)))
    react = layer.reactivation
    initialize = getattr(react, "initialize", None)
    if not callable(initialize):
        return _current_reactivation_state(layer)

    safe_m_target = max(float(m_target), 1e-8)
    target_b = float(b_target)

    with torch.no_grad():
        if alpha >= 1.0 - 1e-12:
            initialize(safe_m_target, target_b)
            return safe_m_target, target_b

        applied_m = safe_m_target
        applied_b = target_b

        if hasattr(react, "log_m") and torch.is_tensor(react.log_m):
            current_m = react.log_m.detach().exp()
            target_m = torch.full_like(current_m, safe_m_target)
            blended_m = torch.lerp(current_m, target_m, alpha).clamp_min(1e-8)
            react.log_m.data.copy_(blended_m.log())
            applied_m = float(blended_m.mean().item())
        else:
            initialize(safe_m_target, target_b)

        if hasattr(react, "b") and torch.is_tensor(react.b):
            current_b = react.b.detach()
            target_b_tensor = torch.full_like(current_b, target_b)
            blended_b = torch.lerp(current_b, target_b_tensor, alpha)
            react.b.data.copy_(blended_b)
            applied_b = float(blended_b.mean().item())
        elif hasattr(react, "fixed_b"):
            current_b = float(react.fixed_b)
            applied_b = (1.0 - alpha) * current_b + alpha * target_b
            react.fixed_b = float(applied_b)

    return applied_m, applied_b


def _current_reactivation_state(layer) -> tuple[float, float]:
    """Return the current scalar `(m, b)` state for a layer reactivation."""

    react = layer.reactivation
    current_m = float(layer.reactivation_init_m)
    current_b = float(layer.reactivation_init_b)

    if hasattr(react, "log_m") and torch.is_tensor(react.log_m):
        current_m = float(react.log_m.detach().exp().mean().item())
    if hasattr(react, "b") and torch.is_tensor(react.b):
        current_b = float(react.b.detach().mean().item())
    elif hasattr(react, "fixed_b"):
        current_b = float(react.fixed_b)

    return current_m, current_b


def _reactivation_calibration_safeguards(layer) -> tuple[float, float, bool]:
    """Read per-layer calibration safeguard settings with sane fallbacks."""

    min_width = float(
        getattr(layer, "reactivation_calibration_min_quantile_width", 1e-3)
    )
    max_m = float(getattr(layer, "reactivation_calibration_max_m", 50.0))
    revert_on_invalid = bool(
        getattr(layer, "reactivation_calibration_revert_on_invalid", True)
    )

    if not isfinite(min_width) or min_width <= 0.0:
        min_width = 1e-3
    if not isfinite(max_m) or max_m <= 0.0:
        max_m = 50.0

    return min_width, max_m, revert_on_invalid


def _resolve_occupancy_spec(
    layer, layer_name: str
) -> tuple[float, float, float, float]:
    """Return the quantile and target occupancy pair for one layer."""
    q_low = getattr(layer, "reactivation_occupancy_quantile_low", None)
    q_high = getattr(layer, "reactivation_occupancy_quantile_high", None)
    r_low = getattr(layer, "reactivation_occupancy_target_low", None)
    r_high = getattr(layer, "reactivation_occupancy_target_high", None)

    q_low = 0.10 if q_low is None else float(q_low)
    q_high = 0.90 if q_high is None else float(q_high)
    r_low = 0.10 if r_low is None else float(r_low)
    r_high = 0.90 if r_high is None else float(r_high)

    if not (0.0 < q_low < q_high < 1.0):
        raise ValueError(
            f"Invalid occupancy quantiles ({q_low}, {q_high}) for layer {layer_name}"
        )
    if not (0.0 < r_low < r_high < 1.0):
        raise ValueError(
            f"Invalid occupancy targets ({r_low}, {r_high}) for layer {layer_name}"
        )
    return q_low, q_high, r_low, r_high


def _fit_occupancy_quantile(
    *,
    layer,
    layer_name: str,
    v: torch.Tensor,
    flattened_chunks: list[torch.Tensor],
    sigma_mad_std: float,
    quantile_aggregation: str,
) -> dict[str, float | bool | str]:
    """Fit an occupancy-quantile reactivation gate for one layer."""
    q_low_p, q_high_p, r_low, r_high = _resolve_occupancy_spec(layer, layer_name)
    q_vals = torch.tensor([q_low_p, q_high_p], dtype=v.dtype, device=v.device)
    occ_q_vals, _ = _aggregate_quantiles_from_chunks(
        flattened_chunks, q_vals, aggregation=quantile_aggregation
    )
    q_low_v, q_high_v = [float(x.item()) for x in occ_q_vals]
    delta_q_raw = float(q_high_v - q_low_v)
    z_low = atanh(2.0 * r_low - 1.0)
    z_high = atanh(2.0 * r_high - 1.0)
    delta_z = z_high - z_low
    min_width, max_m, revert_on_invalid = _reactivation_calibration_safeguards(layer)
    current_m, current_b = _current_reactivation_state(layer)

    if not all(
        isfinite(val)
        for val in (q_low_v, q_high_v, delta_q_raw, z_low, z_high, delta_z)
    ):
        warnings.warn(
            f"Reactivation occupancy calibration for layer {layer_name} produced "
            "non-finite quantiles; reverting to the previous gate state.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "b": current_b,
            "m": current_m,
            "sigma": sigma_mad_std,
            "q_low_prob": q_low_p,
            "q_high_prob": q_high_p,
            "r_low": r_low,
            "r_high": r_high,
            "delta_q_raw": delta_q_raw,
            "delta_q_used": float("nan"),
            "m_raw": float("nan"),
            "m_clamped": False,
            "reverted": True,
            "revert_reason": "nonfinite_quantiles",
        }

    delta_q_used = max(delta_q_raw, min_width)
    if delta_q_raw < min_width:
        if revert_on_invalid:
            warnings.warn(
                f"Reactivation occupancy calibration for layer {layer_name} saw a tiny "
                f"quantile span ({delta_q_raw:.4e}); reverting to the previous "
                "gate state instead of forcing a saturated fit.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                "b": current_b,
                "m": current_m,
                "sigma": sigma_mad_std,
                "q_low_prob": q_low_p,
                "q_high_prob": q_high_p,
                "r_low": r_low,
                "r_high": r_high,
                "delta_q_raw": delta_q_raw,
                "delta_q_used": delta_q_used,
                "m_raw": float("nan"),
                "m_clamped": False,
                "reverted": True,
                "revert_reason": "tiny_quantile_span",
            }
        warnings.warn(
            f"Reactivation occupancy calibration for layer {layer_name} saw a tiny "
            f"quantile span ({delta_q_raw:.4e}); using floor {min_width:.4e}.",
            RuntimeWarning,
            stacklevel=2,
        )

    m_raw = delta_z / delta_q_used
    if not isfinite(m_raw) or m_raw <= 0.0:
        if revert_on_invalid:
            warnings.warn(
                f"Reactivation occupancy calibration for layer {layer_name} produced "
                f"invalid slope {m_raw}; reverting to the previous gate state.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                "b": current_b,
                "m": current_m,
                "sigma": sigma_mad_std,
                "q_low_prob": q_low_p,
                "q_high_prob": q_high_p,
                "r_low": r_low,
                "r_high": r_high,
                "delta_q_raw": delta_q_raw,
                "delta_q_used": delta_q_used,
                "m_raw": m_raw,
                "m_clamped": False,
                "reverted": True,
                "revert_reason": "invalid_slope",
            }
        m_raw = max_m

    if m_raw > max_m and revert_on_invalid:
        warnings.warn(
            f"Reactivation occupancy calibration for layer {layer_name} would clamp "
            f"slope from {m_raw:.4f} to {max_m:.4f}; reverting to the previous "
            "gate state instead of forcing a saturated fit.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "b": current_b,
            "m": current_m,
            "sigma": sigma_mad_std,
            "q_low_prob": q_low_p,
            "q_high_prob": q_high_p,
            "r_low": r_low,
            "r_high": r_high,
            "delta_q_raw": delta_q_raw,
            "delta_q_used": delta_q_used,
            "m_raw": m_raw,
            "m_clamped": False,
            "reverted": True,
            "revert_reason": "would_exceed_max_m",
        }

    m_fit = min(max(m_raw, 1e-6), max_m)
    m_clamped = abs(m_fit - m_raw) > 1e-12
    if m_clamped:
        warnings.warn(
            f"Reactivation occupancy calibration for layer {layer_name} clamped "
            f"slope from {m_raw:.4f} to {m_fit:.4f} (max_m={max_m:.4f}).",
            RuntimeWarning,
            stacklevel=2,
        )
    b_fit = q_low_v - (z_low / m_fit)
    sigma_fit = 0.5 * max(delta_q_raw, 0.0)
    return {
        "b": b_fit,
        "m": m_fit,
        "sigma": sigma_fit,
        "q_low_prob": q_low_p,
        "q_high_prob": q_high_p,
        "r_low": r_low,
        "r_high": r_high,
        "delta_q_raw": delta_q_raw,
        "delta_q_used": delta_q_used,
        "m_raw": m_raw,
        "m_clamped": m_clamped,
        "reverted": False,
        "revert_reason": "none",
    }


def calibrate_reactivation_from_data(
    model: torch.nn.Module,
    batches,
    k: float | None = None,
    device: torch.device | str | None = None,
    robust: bool = True,
    mode: str | None = "median_mad",
    quantile_aggregation: str = "global",
    ema_alpha: float = 1.0,
) -> dict[str, dict[str, float]]:
    """
    Calibrate the reactivation ``(m, b)`` of every DendriticBranchLayer in
    ``model`` from measured voltage statistics on ``batches``.

    For each branch layer whose reactivation is not an Identity, we run the
    supplied batches through ``model``, record the pre-reactivation voltage
    ``V`` via a forward pre-hook, and write back

    * When ``robust=True`` (default):
        b = median(V)
        m = k / (MAD(V) * 1.4826)
      The ``* 1.4826`` factor makes MAD a consistent estimator of the
      Gaussian std; in code the result is returned under the key
      ``"V_mad_std"``.

    * When ``robust=False``:
        b = mean(V)
        m = k / std(V)

    Robust statistics are used by default because DBL voltage distributions
    at init are often heavy-tailed — a handful of extreme samples otherwise
    push ``std(V)`` to nonsensical values (seen at init for shunting layers
    with large conductances). The robust scale is far less sensitive.

    Note: despite the shared ``sigma_aware_k`` parameter name, the
    calibrated slope is ``k / robust_scale``, not ``k / std``. This is
    intentional — see the docstring of ``ReactivationConfig`` in
    ``config/model.py`` for the user-facing description.

    When ``mode`` is a string, it forces the same calibration rule on every
    eligible layer. When ``mode is None``, the helper instead consults each
    layer's own ``reactivation_init_policy`` and calibrates only the layers
    whose policy is one of the data-driven modes.

    Args:
        model: any ``nn.Module`` whose subtree contains DendriticBranchLayers
            (e.g. a ``DendriNet``, ``ExcitationInhibitionNetwork``, or a full
            classifier with one wrapped inside).
        batches: iterable of input tensors or input-tuples that can be passed
            to ``model(...)``. Each element is either ``x`` or ``(x,)`` or a
            tuple forwarded with ``model(*elem)``.
        k: override for the slope constant; if ``None`` each layer uses its own
            ``reactivation_sigma_aware_k`` attribute.
        device: optional device to move each batch to before the forward pass.
        robust: if True, use median + MAD*1.4826; if False, use mean + std.
        ema_alpha: 1.0 applies the freshly calibrated values directly. Smaller
            values perform an EMA-style blend from the current parameters toward
            the calibrated target.

    Returns:
        A ``{layer_name: {"m": ..., "b": ..., "V_mean": ..., "V_std": ...,
        "V_median": ..., "V_mad_std": ..., "V_min": ..., "V_max": ...,
        "n": N}}`` dict for each calibrated branch layer, for diagnostics.
    """
    from .branch_layer import DendriticBranchLayer

    # Collect the branch layers we will calibrate, keyed by module name.
    targets: dict[str, DendriticBranchLayer] = {}
    for name, mod in iter_named_modules_of_type(model, DendriticBranchLayer):
        initialize = getattr(mod.reactivation, "initialize", None)
        if isinstance(mod.reactivation, torch.nn.Identity) or not callable(initialize):
            continue
        if mode is None:
            layer_policy = normalize_reactivation_init_policy(
                getattr(mod, "reactivation_init_policy", "analytical")
            )
            if not is_data_driven_reactivation_policy(layer_policy):
                continue
        targets[name] = mod

    if not targets:
        return {}

    # Accumulate voltage samples per layer (keep raw samples so we can compute
    # robust statistics like median and MAD; DendriticBranchLayers at init can
    # produce long-tailed voltage distributions that make naive mean/std a bad
    # choice).
    samples: dict[str, list[torch.Tensor]] = {}

    # Attach pre-hooks on each reactivation module. The hook's input is V.
    handles = []
    for name, layer in targets.items():

        def _make_hook(layer_name: str):
            def hook(_mod, inputs):
                v = inputs[0].detach().float().cpu()
                if layer_name not in samples:
                    samples[layer_name] = []
                samples[layer_name].append(v)

            return hook

        handles.append(layer.reactivation.register_forward_pre_hook(_make_hook(name)))

    # Forward passes in inference mode, no gradient, no autograd graph built.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                if isinstance(batch, (list, tuple)):
                    inputs = tuple(
                        (
                            b.to(device)
                            if (device is not None and torch.is_tensor(b))
                            else b
                        )
                        for b in batch
                    )
                    model(*inputs)
                else:
                    x = batch.to(device) if device is not None else batch
                    model(x)
    finally:
        remove_hook_handles(handles)
        if was_training:
            model.train()

    # Compute empirical (m, b) per layer and apply it.
    diagnostics: dict[str, dict[str, float]] = {}
    for name, layer in targets.items():
        chunks = samples.get(name, [])
        if not chunks:
            continue
        flattened_chunks = []
        for chunk in chunks:
            flat = chunk.reshape(-1).double()
            flat = flat[torch.isfinite(flat)]
            if flat.numel() > 0:
                flattened_chunks.append(flat)
        if not flattened_chunks:
            continue

        v = torch.cat(flattened_chunks)
        n = int(v.numel())
        if n == 0:
            continue

        mean = float(v.mean().item())
        variance = float(v.var(unbiased=False).item())
        sigma_mean_std = sqrt(max(variance, 0.0))
        vmin = float(v.min().item())
        vmax = float(v.max().item())

        # Always compute quantiles so we can report them for diagnosis even
        # when a non-quantile mode is selected.
        q_probs = torch.tensor([0.10, 0.50, 0.90], dtype=v.dtype, device=v.device)
        q_stats, quantile_sample_n = _aggregate_quantiles_from_chunks(
            flattened_chunks, q_probs, aggregation=quantile_aggregation
        )
        q10, q50, q90 = [float(x.item()) for x in q_stats]
        median = float(torch.median(v).item())
        mad = float(torch.median(torch.abs(v - median)).item())
        sigma_mad_std = 1.4826 * mad

        if mode is None:
            layer_policy = normalize_reactivation_init_policy(
                getattr(layer, "reactivation_init_policy", "analytical")
            )
            active_mode = reactivation_policy_to_calibration_mode(layer_policy)
            if active_mode is None:
                continue
        else:
            # Back-compat: robust=False still selects mean/std mode.
            active_mode = (
                (reactivation_policy_to_calibration_mode(mode) or str(mode).lower())
                if robust
                else "mean_std"
            )

        layer_k = (
            float(k)
            if k is not None
            else float(getattr(layer, "reactivation_sigma_aware_k", 0.25))
        )

        # Fallback m from robust scale (used when mode doesn't set m from quantiles).
        robust_m = (
            max(layer_k / sigma_mad_std, 1e-6)
            if sigma_mad_std > 0
            else max(layer_k, 1e-6)
        )

        occ_q_low = occ_q_high = occ_r_low = occ_r_high = None
        occ_delta_q_raw = occ_delta_q_used = occ_m_raw = None
        occ_m_clamped = False
        calibration_reverted = False

        if active_mode == "occupancy_quantile":
            occ_fit = _fit_occupancy_quantile(
                layer=layer,
                layer_name=name,
                v=v,
                flattened_chunks=flattened_chunks,
                sigma_mad_std=sigma_mad_std,
                quantile_aggregation=quantile_aggregation,
            )
            b_emp = float(occ_fit["b"])
            m_emp = float(occ_fit["m"])
            occ_q_low = float(occ_fit["q_low_prob"])
            occ_q_high = float(occ_fit["q_high_prob"])
            occ_r_low = float(occ_fit["r_low"])
            occ_r_high = float(occ_fit["r_high"])
            occ_delta_q_raw = float(occ_fit["delta_q_raw"])
            occ_delta_q_used = float(occ_fit["delta_q_used"])
            occ_m_raw = float(occ_fit["m_raw"])
            occ_m_clamped = bool(occ_fit["m_clamped"])
            calibration_reverted = bool(occ_fit["reverted"])
        elif active_mode == "median_mad":
            b_emp = median
            m_emp = robust_m
        elif active_mode == "mean_std":
            sigma_emp = sigma_mean_std
            b_emp = mean
            m_emp = (
                max(layer_k / sigma_emp, 1e-6) if sigma_emp > 0 else max(layer_k, 1e-6)
            )
        else:
            raise ValueError(
                f"calibrate_reactivation_from_data: unknown mode={active_mode!r} "
                f"(expected 'median_mad', 'mean_std', 'occupancy_quantile')"
            )

        applied_m, applied_b = _apply_reactivation_calibration(
            layer,
            m_target=m_emp,
            b_target=b_emp,
            ema_alpha=ema_alpha,
        )

        diagnostics[name] = {
            "m": m_emp,
            "b": b_emp,
            "m_applied": applied_m,
            "b_applied": applied_b,
            "ema_alpha": float(max(0.0, min(1.0, ema_alpha))),
            "mode": active_mode,
            "V_mean": mean,
            "V_std": sigma_mean_std,
            "V_median": median,
            "V_mad_std": sigma_mad_std,
            "V_q10": q10,
            "V_q50": q50,
            "V_q90": q90,
            "occupancy_q_low": occ_q_low,
            "occupancy_q_high": occ_q_high,
            "occupancy_r_low": occ_r_low,
            "occupancy_r_high": occ_r_high,
            "occupancy_delta_q_raw": occ_delta_q_raw,
            "occupancy_delta_q_used": occ_delta_q_used,
            "occupancy_m_raw": occ_m_raw,
            "occupancy_m_was_clamped": occ_m_clamped,
            "calibration_reverted": calibration_reverted,
            "calibration_revert_reason": (
                occ_fit.get("revert_reason")
                if active_mode == "occupancy_quantile"
                else "none"
            ),
            "V_min": vmin,
            "V_max": vmax,
            "n": n,
            "n_quantile_sample": quantile_sample_n,
            "quantile_aggregation": quantile_aggregation,
        }

    return diagnostics


def calibrate_reactivation_empirical(*args, **kwargs):
    """Backward-compatible alias for :func:`calibrate_reactivation_from_data`."""

    return calibrate_reactivation_from_data(*args, **kwargs)


__all__ = [
    "_apply_reactivation_calibration",
    "_apply_reactivation_init",
    "_current_reactivation_state",
    "_fit_occupancy_quantile",
    "_reactivation_calibration_safeguards",
    "_resolve_occupancy_spec",
    "calibrate_reactivation_empirical",
    "calibrate_reactivation_from_data",
]
