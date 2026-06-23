from __future__ import annotations

from math import erf, exp, log, sqrt

import torch
from scipy.stats import norm

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


__all__ = [
    "ADAPTIVE_INITIALIZATION_POLICY_ALIASES",
    "MAX_CALIBRATION_QUANTILE_SAMPLES",
    "_aggregate_quantiles_from_chunks",
    "_compute_expectation_truncated_relu",
    "_compute_expectation_truncated_softplus",
    "_expected_weight_for_transform",
    "_mean_for_expected_weight",
    "_safe_quantile",
    "_truncated_normal_moments",
    "compute_expectation_truncated_inverse_softplus_normal",
    "compute_expectation_truncated_log_normal",
    "normalize_adaptive_initialization_policy",
]
