import warnings
from math import atanh, erf, exp, isfinite, log, sqrt

import torch
from scipy.stats import norm
from torch.nn.init import _no_grad_normal_

from dendritic_modeling.config.reactivation import (
    is_data_driven_reactivation_policy,
    normalize_reactivation_init_policy,
    reactivation_policy_to_calibration_mode,
)

MAX_CALIBRATION_QUANTILE_SAMPLES = 1_000_000


ADAPTIVE_INITIALIZATION_POLICY_ALIASES = {
    None: "preserve_shunting_center",
    "preserve": "preserve_shunting_center",
    "preserve_center": "preserve_shunting_center",
    "preserve_shunting": "preserve_shunting_center",
    "preserve_shunting_center": "preserve_shunting_center",
    "center_preserving": "preserve_shunting_center",
    "legacy": "legacy_scale",
    "legacy_scale": "legacy_scale",
    "scale": "legacy_scale",
    "scale_conductance": "legacy_scale",
}


def normalize_adaptive_initialization_policy(policy: str | None) -> str:
    """Return the canonical adaptive-initialization policy name."""

    key = None if policy is None else str(policy).strip().lower()
    canonical = ADAPTIVE_INITIALIZATION_POLICY_ALIASES.get(key, key)
    if canonical not in {"preserve_shunting_center", "legacy_scale"}:
        raise ValueError(
            "Unknown adaptive_initialization_policy "
            f"{policy!r}; expected 'preserve_shunting_center' or 'legacy_scale'."
        )
    return canonical


def _safe_quantile(
    values: torch.Tensor,
    probs: torch.Tensor,
    *,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Compute quantiles on a capped deterministic subsample when needed."""

    if values.numel() == 0:
        raise ValueError("Cannot compute quantiles of an empty tensor")

    if max_samples is None:
        max_samples = MAX_CALIBRATION_QUANTILE_SAMPLES

    sample_n = int(values.numel())
    if sample_n <= max_samples:
        return torch.quantile(values, probs), sample_n

    idx = torch.linspace(
        0,
        sample_n - 1,
        steps=max_samples,
        dtype=torch.long,
        device=values.device,
    )
    sampled = values.index_select(0, idx)
    return torch.quantile(sampled, probs), int(sampled.numel())


def _aggregate_quantiles_from_chunks(
    chunks: list[torch.Tensor],
    probs: torch.Tensor,
    *,
    aggregation: str = "global",
) -> tuple[torch.Tensor, int]:
    """Aggregate quantiles from voltage chunks.

    ``global`` matches the original behavior by pooling all chunks before
    fitting quantiles. ``chunk_median`` computes the requested quantiles for
    each chunk independently and then takes the median across chunks. The
    latter is useful for recurrent models where each hook invocation typically
    corresponds to a different timestep and we want a calibration that reflects
    typical temporal slices rather than the pooled multimodal distribution.
    """

    valid_chunks = []
    for chunk in chunks:
        flat = chunk.reshape(-1).double()
        flat = flat[torch.isfinite(flat)]
        if flat.numel() > 0:
            valid_chunks.append(flat)

    if not valid_chunks:
        raise ValueError("Cannot compute quantiles from empty calibration chunks")

    if aggregation == "global":
        pooled = torch.cat(valid_chunks)
        return _safe_quantile(pooled, probs)

    if aggregation == "chunk_median":
        per_chunk = []
        total_n = 0
        for chunk in valid_chunks:
            q_vals, sample_n = _safe_quantile(chunk, probs)
            per_chunk.append(q_vals)
            total_n += sample_n
        return torch.median(torch.stack(per_chunk, dim=0), dim=0).values, total_n

    raise ValueError(
        f"Unknown quantile aggregation mode {aggregation!r} "
        f"(expected 'global' or 'chunk_median')"
    )


def compute_expectation_truncated_log_normal(
    mean,
    std,
    upper_quantile,
):
    log_alpha = mean + std * norm.ppf(1 - upper_quantile).item()

    numerator = exp(mean + 0.5 * std**2) * (
        1 - erf((log_alpha - mean - std**2) / (std * sqrt(2)))
    )
    denominator = 1 - erf((log_alpha - mean) / (std * sqrt(2)))

    expectation = numerator / denominator
    return expectation


def compute_expectation_truncated_inverse_softplus_normal(
    mean,
    std,
    upper_quantile,
):
    z = mean + std * norm.ppf(1 - upper_quantile).item()

    normal = torch.distributions.Normal(mean, std)

    truncated_samples_list = []
    max_iter = 100
    iter_counter = 0
    done = False
    while not done:
        iter_counter += 1
        samples = normal.sample((10000,))
        truncated_samples = samples[samples > z]
        truncated_samples_list.append(truncated_samples)
        all_truncated_samples = torch.cat(truncated_samples_list)
        if (all_truncated_samples.shape[0] >= 10000) or (iter_counter >= max_iter):
            done = True

    transformed_samples = torch.log(1 + torch.exp(all_truncated_samples))
    return torch.mean(transformed_samples).item()


# -----------------------------
# Transform-aware expectation helpers
# -----------------------------


def _truncated_normal_moments(
    mean: float, std: float, upper_quantile: float
) -> tuple[float, float, float]:
    """
    Compute moments of X | X > alpha where X ~ Normal(mean, std^2),
    and alpha is the threshold corresponding to the given upper_quantile (top proportion kept).

    Returns:
        alpha: threshold
        mean_c: E[X | X > alpha]
        var_c: Var[X | X > alpha]
    """
    # Determine threshold alpha such that P(X > alpha) = upper_quantile.
    # Clamp to open interval (0, 1) to avoid +/-inf from norm.ppf at the edges
    # (e.g., K == in_features -> upper_quantile == 1.0).
    upper_quantile = min(max(float(upper_quantile), 1e-12), 1 - 1e-12)

    # Determine threshold alpha such that P(X > alpha) = upper_quantile
    # gamma = (alpha - mean) / std
    gamma = norm.ppf(1 - upper_quantile)
    alpha = mean + std * gamma

    # Avoid numerical issues for extreme quantiles
    Q = max(1e-12, 1 - norm.cdf(gamma))
    phi = norm.pdf(gamma)
    lam = phi / Q

    mean_c = mean + std * lam
    var_c = (std**2) * (1 + gamma * lam - lam**2)

    # Ensure variance is non-negative (can be slightly negative due to numerical errors)
    var_c = max(0.0, var_c)

    return alpha, mean_c, var_c


def _compute_expectation_truncated_softplus(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[softplus(X) | X > alpha] via delta method using truncated normal moments.
    softplus(x) = ln(1 + e^x),
    g'(x) = sigmoid(x), g''(x) = sigmoid(x) * (1 - sigmoid(x)).
    """
    _, mu_c, var_c = _truncated_normal_moments(mean, std, upper_quantile)
    # softplus at conditional mean
    from math import exp as mexp, log as mlog

    # Clip mu_c to prevent overflow
    mu_c = max(-50.0, min(50.0, mu_c))

    # Safe computation of softplus to avoid overflow
    if mu_c > 20:
        sp = mu_c  # For large x, softplus(x) ≈ x
    else:
        sp = mlog(1 + mexp(mu_c))

    # Safe computation of sigmoid to avoid overflow
    if mu_c > 0:
        sig = 1.0 / (1.0 + mexp(-mu_c))
    else:
        exp_mu = mexp(mu_c)
        sig = exp_mu / (1.0 + exp_mu)

    sp_corr = sp + 0.5 * sig * (1 - sig) * var_c
    return sp_corr


def _compute_expectation_truncated_relu(
    mean: float, std: float, upper_quantile: float
) -> float:
    """
    Approximate E[ReLU(X) | X > alpha]. Since conditioning ensures X > alpha,
    and for typical TopK alpha is positive, we use E[X | X > alpha] as a good approximation.
    """
    _, mu_c, _ = _truncated_normal_moments(mean, std, upper_quantile)
    # If alpha <= 0 but mu_c still small, ensure non-negativity
    return max(0.0, mu_c)


def _expected_weight_for_transform(
    transform: str, mean: float, std: float, upper_quantile: float
) -> float:
    """
    Compute/approximate E[g(X) | X > alpha] where g is the transform and X ~ N(mean, std^2).

    For 'exp' uses the existing closed-form truncated log-normal expectation.
    For 'softplus' and 'relu' uses truncated normal moments with a delta-method approximation.
    """
    t = (transform or "exp").lower()
    if t == "exp":
        return compute_expectation_truncated_log_normal(
            mean=mean, std=std, upper_quantile=upper_quantile
        )
    if t == "softplus":
        return _compute_expectation_truncated_softplus(mean, std, upper_quantile)
    if t == "relu":
        return _compute_expectation_truncated_relu(mean, std, upper_quantile)
    # Fallback to exp-style expectation if unknown
    return compute_expectation_truncated_log_normal(
        mean=mean, std=std, upper_quantile=upper_quantile
    )


def _mean_for_expected_weight(
    transform: str,
    std: float,
    upper_quantile: float,
    target_weight: float,
    *,
    reference_mean: float = 0.0,
) -> float:
    """Find a raw-weight mean whose transformed top-K expectation hits target."""

    target = max(float(target_weight), 1e-12)
    t = (transform or "exp").lower()

    if t == "exp":
        reference = _expected_weight_for_transform(
            transform=t,
            mean=reference_mean,
            std=std,
            upper_quantile=upper_quantile,
        )
        return reference_mean + log(target / max(reference, 1e-12))

    low, high = -80.0, 80.0
    for _ in range(96):
        mid = 0.5 * (low + high)
        value = _expected_weight_for_transform(
            transform=t,
            mean=mid,
            std=std,
            upper_quantile=upper_quantile,
        )
        if value < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _center_preserving_shunting_conductances(
    E_g_exc: float,
    E_g_inh: float,
    target_conductance: float,
) -> tuple[float, float]:
    """Reduce total conductance while preserving the shunting voltage center."""

    E_g_exc = max(float(E_g_exc), 0.0)
    E_g_inh = max(float(E_g_inh), 0.0)
    total = E_g_exc + E_g_inh
    if E_g_exc <= 0.0 or total <= 0.0:
        return E_g_exc, E_g_inh

    center = E_g_exc / (total + 1.0)
    if center <= 0.0 or center >= 1.0:
        return E_g_exc, E_g_inh

    target_total = min(total, max(float(target_conductance), 1e-12))
    min_total_for_center = center / (1.0 - center)
    target_total = max(target_total, min_total_for_center)
    target_total = min(target_total, total)

    target_exc = center * (target_total + 1.0)
    target_inh = target_total - target_exc
    return max(target_exc, 0.0), max(target_inh, 0.0)


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
    for name, mod in model.named_modules():
        if not isinstance(mod, DendriticBranchLayer):
            continue
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
        for h in handles:
            h.remove()
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
        current_layer = layer
        layer_name = name
        v_ref = v
        flattened_chunks_ref = flattened_chunks
        sigma_mad_std_ref = sigma_mad_std

        def _resolve_occupancy_spec(
            current_layer=current_layer,
            layer_name=layer_name,
        ) -> tuple[float, float, float, float]:
            q_low = getattr(current_layer, "reactivation_occupancy_quantile_low", None)
            q_high = getattr(
                current_layer, "reactivation_occupancy_quantile_high", None
            )
            r_low = getattr(current_layer, "reactivation_occupancy_target_low", None)
            r_high = getattr(current_layer, "reactivation_occupancy_target_high", None)

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
            v_ref=v_ref,
            flattened_chunks_ref=flattened_chunks_ref,
            current_layer=current_layer,
            layer_name=layer_name,
            sigma_mad_std_ref=sigma_mad_std_ref,
        ) -> dict[str, float | bool]:
            q_low_p, q_high_p, r_low, r_high = _resolve_occupancy_spec()
            q_vals = torch.tensor(
                [q_low_p, q_high_p], dtype=v_ref.dtype, device=v_ref.device
            )
            occ_q_vals, _ = _aggregate_quantiles_from_chunks(
                flattened_chunks_ref, q_vals, aggregation=quantile_aggregation
            )
            q_low_v, q_high_v = [float(x.item()) for x in occ_q_vals]
            delta_q_raw = float(q_high_v - q_low_v)
            z_low = atanh(2.0 * r_low - 1.0)
            z_high = atanh(2.0 * r_high - 1.0)
            delta_z = z_high - z_low
            min_width, max_m, revert_on_invalid = _reactivation_calibration_safeguards(
                current_layer
            )
            current_m, current_b = _current_reactivation_state(current_layer)

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
                    "sigma": sigma_mad_std_ref,
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
                        "sigma": sigma_mad_std_ref,
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
                        "sigma": sigma_mad_std_ref,
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
                    "sigma": sigma_mad_std_ref,
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

        occ_q_low = occ_q_high = occ_r_low = occ_r_high = None
        occ_delta_q_raw = occ_delta_q_used = occ_m_raw = None
        occ_m_clamped = False
        calibration_reverted = False

        if active_mode == "occupancy_quantile":
            occ_fit = _fit_occupancy_quantile()
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


def compute_network_scale_factor(dbl, branch_factors=None, target_conductance=5.0):
    """
    Compute a scale factor based on the total network architecture.
    This prevents overflow when many branches accumulate in shunting computation.

    Args:
        dbl: DendriticBranchLayer to initialize
        branch_factors: List of branch factors from the parent network
        target_conductance: Conductance budget used by legacy scale mode

    Returns a scale factor to reduce initial weights in log space.
    """
    # Calculate total branches
    total_branches = 1

    if branch_factors:
        for bf in branch_factors:
            total_branches *= bf
    else:
        # Conservative estimate if branch factors not provided
        total_branches = 100

    # Calculate approximate total conductance
    k_exc = dbl.branch_excitation.K if dbl.input_excitatory else 0
    k_rec = dbl.branch_recurrent.K if getattr(dbl, "input_recurrent", False) else 0
    k_inh = dbl.branch_inhibition.K if dbl.input_inhibitory else 0
    k_rec_inh = (
        dbl.branch_rec_inhibition.K
        if getattr(dbl, "input_rec_inhibitory", False)
        else 0
    )

    # Total approximate conductance with all weights at 1
    # Scale by sqrt of branches for more conservative scaling
    total_conductance = (k_exc + k_rec + k_inh + k_rec_inh) * sqrt(total_branches)

    # Target a reasonable initial total conductance
    if total_conductance > target_conductance:
        scale_factor = log(total_conductance / target_conductance)
    else:
        scale_factor = 0.0

    return scale_factor


def identity_weighttransform_dbl_init(dbl):
    """
    Initialize a dendritic branch layer.

    Args:
        dbl: The DendriticBranchLayer to initialize
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        if dbl.branch_excitation is not None:
            torch.nn.init.xavier_normal_(dbl.branch_excitation.pre_w)

        if dbl.branch_recurrent is not None:
            torch.nn.init.xavier_normal_(dbl.branch_recurrent.pre_w)

        if dbl.branch_inhibition is not None:
            torch.nn.init.xavier_normal_(dbl.branch_inhibition.pre_w)

        if dbl.branch_rec_inhibition is not None:
            torch.nn.init.xavier_normal_(dbl.branch_rec_inhibition.pre_w)

        if hasattr(dbl, "branches_to_output"):
            torch.nn.init.xavier_normal_(dbl.branches_to_output.log_weight)

        if dbl.reactivate:
            _apply_reactivation_init(dbl, m_auto=1.0, b_auto=0.0)


def analytical_expectation_dbl_init(dbl, adaptive=True, branch_factors=None):
    """
    Initialize a dendritic branch layer.

    Args:
        dbl: The DendriticBranchLayer to initialize
        adaptive: If True, use network-aware scaling to prevent overflow.
                 If False, use the original initialization method.
        branch_factors: List of branch factors from parent network (for adaptive mode)
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        E_w_exc = 0
        E_w_inh = 0
        exc_mean = 0
        inh_mean = 0
        exc_std = None
        inh_std = None
        exc_upper_quantile = None
        inh_upper_quantile = None
        weight_transform = dbl.weight_transform
        adaptive_policy = normalize_adaptive_initialization_policy(
            getattr(dbl, "adaptive_initialization_policy", "preserve_shunting_center")
        )
        adaptive_target_conductance = float(
            getattr(dbl, "adaptive_target_conductance", 5.0)
        )

        # Compute network-aware scale factor if adaptive mode is enabled
        network_scale_factor = (
            compute_network_scale_factor(
                dbl,
                branch_factors,
                target_conductance=adaptive_target_conductance,
            )
            if adaptive
            else 0.0
        )

        if dbl.input_excitatory:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_mean = 0
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))

            if not dbl.input_inhibitory and not dbl.input_branches:
                step = 0.01
                done = False
                while not done:
                    E_w_exc = _expected_weight_for_transform(
                        transform=weight_transform,
                        mean=exc_mean,
                        std=exc_std,
                        upper_quantile=exc_upper_quantile,
                    )
                    if 0.5 * dbl.branch_excitation.K * E_w_exc <= 1:
                        done = True
                    else:
                        exc_mean -= step

            else:
                E_w_exc = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=exc_mean,
                    std=exc_std,
                    upper_quantile=exc_upper_quantile,
                )

            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.input_inhibitory:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_mean = 0
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))

            if dbl.input_excitatory:
                ie_ratio = dbl.branch_inhibition.K / dbl.branch_excitation.K
                inh_std = inh_std * ie_ratio if ie_ratio < 1 else inh_std
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )

                step = 0.01
                done = False
                if E_w_exc >= ie_ratio * E_w_inh:
                    while not done:
                        E_w_inh = _expected_weight_for_transform(
                            transform=weight_transform,
                            mean=inh_mean,
                            std=inh_std,
                            upper_quantile=inh_upper_quantile,
                        )
                        if E_w_exc <= ie_ratio * E_w_inh:
                            done = True
                        else:
                            inh_mean += step
                else:
                    while not done:
                        E_w_inh = _expected_weight_for_transform(
                            transform=weight_transform,
                            mean=inh_mean,
                            std=inh_std,
                            upper_quantile=inh_upper_quantile,
                        )
                        if E_w_exc >= ie_ratio * E_w_inh:
                            done = True
                        else:
                            inh_mean -= step

            else:
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=inh_mean,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )

            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        E_g_exc_target = E_g_exc
        E_g_inh_target = E_g_inh

        if adaptive and network_scale_factor > 0.0:
            if adaptive_policy == "preserve_shunting_center" and dbl.use_shunting:
                E_g_exc_target, E_g_inh_target = (
                    _center_preserving_shunting_conductances(
                        E_g_exc,
                        E_g_inh,
                        adaptive_target_conductance,
                    )
                )
            else:
                scale = exp(-network_scale_factor)
                E_g_exc_target = E_g_exc * scale
                E_g_inh_target = E_g_inh * scale

        if dbl.input_excitatory:
            if (
                adaptive
                and adaptive_policy == "legacy_scale"
                and network_scale_factor > 0.0
            ):
                exc_mean_adjusted = exc_mean - network_scale_factor
            elif E_g_exc > 0.0 and E_g_exc_target != E_g_exc:
                target_w_exc = 2.0 * E_g_exc_target / dbl.branch_excitation.K
                exc_mean_adjusted = _mean_for_expected_weight(
                    weight_transform,
                    exc_std,
                    exc_upper_quantile,
                    target_w_exc,
                    reference_mean=exc_mean,
                )
            else:
                exc_mean_adjusted = exc_mean
            _no_grad_normal_(dbl.branch_excitation.pre_w, exc_mean_adjusted, exc_std)
            E_g_exc = E_g_exc_target

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            rec_mean_adjusted = -network_scale_factor if adaptive else 0.0
            _no_grad_normal_(dbl.branch_recurrent.pre_w, rec_mean_adjusted, rec_std)

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            rec_inh_mean_adjusted = -network_scale_factor if adaptive else 0.0
            _no_grad_normal_(
                dbl.branch_rec_inhibition.pre_w, rec_inh_mean_adjusted, rec_inh_std
            )

        if dbl.input_inhibitory:
            if (
                adaptive
                and adaptive_policy == "legacy_scale"
                and network_scale_factor > 0.0
            ):
                inh_mean_adjusted = inh_mean - network_scale_factor
            elif E_g_inh > 0.0 and E_g_inh_target != E_g_inh:
                target_w_inh = 2.0 * E_g_inh_target / dbl.branch_inhibition.K
                inh_mean_adjusted = _mean_for_expected_weight(
                    weight_transform,
                    inh_std,
                    inh_upper_quantile,
                    target_w_inh,
                    reference_mean=inh_mean,
                )
            else:
                inh_mean_adjusted = inh_mean
            _no_grad_normal_(dbl.branch_inhibition.pre_w, inh_mean_adjusted, inh_std)
            E_g_inh = E_g_inh_target

        if dbl.input_branches:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)
                # Map E_Vinf close to 0.5 to give strong initial branch
                # conductances. Clamp strictly below 0.5 to keep the
                # downstream conductance formula well-defined.
                if E_Vinf < 0.5:
                    E_Vinf = max(0.49, 0.5 * E_Vinf + 0.25)
                else:
                    E_Vinf = min(0.499, 0.5 * E_Vinf + 0.25)

                sum_g_branch = ((E_g_exc + E_g_inh + 1) * E_Vinf - E_g_exc) / (
                    0.5 - E_Vinf
                )
                g_branch = max(sum_g_branch / dbl.branches_to_output.block_size, 1e-6)
                m = 1.5
            else:
                # g_branch = E_g_exc + E_g_inh + 1
                g_branch = 1
                # Additive voltage V = E - I is centered near 0 (it is NOT
                # bounded in [0, 1] like the shunting voltage), so the gate
                # center is E[V] = E_g_exc - E_g_inh. The former `+ 0.5*g_branch`
                # term was a holdover from the shunting branch (whose V lives in
                # [0, 1] centered at 0.5) and pushed the additive non-leaf gate
                # center ~+0.5 std above the actual voltage bulk.
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1

            dbl.branches_to_output.initialize(g_branch=g_branch)
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)
                m = 1.5
            else:
                E_Vinf = 0
                m = 0.1

        _apply_reactivation_init(dbl, m_auto=m, b_auto=E_Vinf)


def ei_equivalence_dbl_init(dbl):
    """
    Initialize a dendritic branch layer.

    Args:
        dbl: The DendriticBranchLayer to initialize
    """
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        weight_transform = dbl.weight_transform

        if dbl.branch_excitation is not None:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            E_w_exc = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.branch_inhibition is not None:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        step = 0.01
        done = False
        if (E_g_exc < E_g_inh) and dbl.branch_excitation is not None:
            while not done:
                exc_std += step
                E_w_exc = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=0,
                    std=exc_std,
                    upper_quantile=exc_upper_quantile,
                )
                E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc
                if E_g_exc > E_g_inh:
                    done = True

        elif (E_g_exc > E_g_inh) and dbl.branch_inhibition is not None:
            while not done:
                inh_std += step
                E_w_inh = _expected_weight_for_transform(
                    transform=weight_transform,
                    mean=0,
                    std=inh_std,
                    upper_quantile=inh_upper_quantile,
                )
                E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh
                if E_g_inh >= E_g_exc:
                    done = True

        if dbl.branch_excitation is not None:
            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            _no_grad_normal_(dbl.branch_recurrent.pre_w, 0, rec_std)

        if dbl.branch_inhibition is not None:
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            _no_grad_normal_(dbl.branch_rec_inhibition.pre_w, 0, rec_inh_std)

        if dbl.input_branches:
            if dbl.use_shunting:
                g_branch = E_g_exc + E_g_inh + 1
                sum_g_branch = g_branch * dbl.branches_to_output.block_size
                E_Vinf = (E_g_exc + 0.5 * sum_g_branch) / (
                    E_g_exc + E_g_inh + sum_g_branch + 1
                )
                m = 1.5
            else:
                # g_branch = E_g_exc + E_g_inh + 1
                g_branch = 1
                # Additive voltage V = E - I is centered near 0 (it is NOT
                # bounded in [0, 1] like the shunting voltage), so the gate
                # center is E[V] = E_g_exc - E_g_inh. The former `+ 0.5*g_branch`
                # term was a holdover from the shunting branch (whose V lives in
                # [0, 1] centered at 0.5) and pushed the additive non-leaf gate
                # center ~+0.5 std above the actual voltage bulk.
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1

            dbl.branches_to_output.initialize(g_branch=g_branch)
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)
                m = 1.5
            else:
                E_Vinf = 0
                m = 0.1

        _apply_reactivation_init(dbl, m_auto=m, b_auto=E_Vinf)


def naive_dbl_init(dbl):
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        weight_transform = dbl.weight_transform

        if dbl.branch_excitation is not None:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

            E_w_exc = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            _no_grad_normal_(dbl.branch_recurrent.pre_w, 0, rec_std)

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            _no_grad_normal_(dbl.branch_rec_inhibition.pre_w, 0, rec_inh_std)

        if dbl.branch_inhibition is not None:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        if dbl.input_branches:
            g_branch = E_g_exc + E_g_inh + 1
            dbl.branches_to_output.initialize(g_branch=g_branch)

            if dbl.use_shunting:
                E_Vinf = (E_g_exc + 0.5 * g_branch) / (E_g_exc + E_g_inh + g_branch + 1)
                m = 1.5
            else:
                # Additive voltage V = E - I is centered near 0 (it is NOT
                # bounded in [0, 1] like the shunting voltage), so the gate
                # center is E[V] = E_g_exc - E_g_inh. The former `+ 0.5*g_branch`
                # term was a holdover from the shunting branch (whose V lives in
                # [0, 1] centered at 0.5) and pushed the additive non-leaf gate
                # center ~+0.5 std above the actual voltage bulk.
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1
        else:
            if dbl.use_shunting:
                E_Vinf = E_g_exc / (E_g_exc + E_g_inh + 1)
                m = 1.5
            else:
                E_Vinf = E_g_exc - E_g_inh
                m = 0.1

        _apply_reactivation_init(dbl, m_auto=m, b_auto=E_Vinf)


def default_dbl_init(dbl):
    from .branch_layer import DendriticBranchLayer

    assert isinstance(dbl, DendriticBranchLayer)

    with torch.no_grad():
        E_g_exc = 0
        E_g_inh = 0
        weight_transform = dbl.weight_transform

        if dbl.branch_excitation is not None:
            exc_upper_quantile = dbl.branch_excitation.K / dbl.excitatory_input_dim
            exc_std = sqrt(2 / (dbl.excitatory_input_dim + 1))
            _no_grad_normal_(dbl.branch_excitation.pre_w, 0, exc_std)

            E_w_exc = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=exc_std,
                upper_quantile=exc_upper_quantile,
            )
            E_g_exc = 0.5 * dbl.branch_excitation.K * E_w_exc

        if dbl.branch_recurrent is not None:
            rec_std = sqrt(2 / (dbl.recurrent_input_dim + 1))
            _no_grad_normal_(dbl.branch_recurrent.pre_w, 0, rec_std)

        if dbl.branch_rec_inhibition is not None:
            rec_inh_std = sqrt(2 / (dbl.rec_inhibitory_input_dim + 1))
            _no_grad_normal_(dbl.branch_rec_inhibition.pre_w, 0, rec_inh_std)

        if dbl.branch_inhibition is not None:
            inh_upper_quantile = dbl.branch_inhibition.K / dbl.inhibitory_input_dim
            inh_std = sqrt(2 / (dbl.inhibitory_input_dim + 1))
            _no_grad_normal_(dbl.branch_inhibition.pre_w, 0, inh_std)

            E_w_inh = _expected_weight_for_transform(
                transform=weight_transform,
                mean=0,
                std=inh_std,
                upper_quantile=inh_upper_quantile,
            )
            E_g_inh = 0.5 * dbl.branch_inhibition.K * E_w_inh

        if dbl.input_branches:
            if dbl.use_shunting:
                g_branch = E_g_exc + E_g_inh + 1
            else:
                g_branch = 1

            dbl.branches_to_output.initialize(g_branch=g_branch)

        if dbl.use_shunting:
            m_auto = 1.5
            b_auto = 0.5
        else:
            m_auto = 0.1
            b_auto = 0.0
        _apply_reactivation_init(dbl, m_auto=m_auto, b_auto=b_auto)
