"""Soma-versus-distal targeting of a synapse route across dendritic levels.

A route's realized weights at dendritic level ``l`` form a tensor of shape
``(n_target, K_l, n_source)``: the ``K_l`` compartments of each target cell at
that level. The weight mass a source delivers to a target at a level is the
summed absolute weight over those compartments. Normalizing the masses over
levels gives the level fractions ``f_l(source, target)``, and the targeting
index is ``f_soma - f_distal`` (soma = last level, distal = first). A
source-target pair without realized weight at any level is disconnected and
is excluded from every statistic.

``level_weight_per_compartment`` divides each level's mass by ``K_l``; it is
reported as an architectural control, not as the targeting estimand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

EPS = 1e-12


@dataclass(frozen=True)
class TargetingClassLabels:
    """Names of the source classes of one targeting analysis."""

    soma: str
    distal: str
    mixed: str
    diffuse: str = "diffuse"
    disconnected: str = "disconnected"


def apply_weight_transform(pre_w: torch.Tensor, weight_transform: str) -> torch.Tensor:
    """Return synaptic weights from raw parameters under a weight transform."""
    transform = weight_transform.lower()
    if transform == "exp":
        return torch.exp(pre_w)
    if transform == "softplus":
        return torch.nn.functional.softplus(pre_w)
    if transform == "relu":
        return torch.relu(pre_w)
    if transform == "identity":
        return pre_w
    raise ValueError(f"Unsupported weight_transform: {weight_transform}")


def reshape_by_compartment(w: torch.Tensor, n_target: int) -> torch.Tensor | None:
    """Reshape ``(n_target * K_l, n_source)`` weights to ``(n_target, K_l, n_source)``."""
    out_features, _n_source = w.shape
    if n_target <= 0 or out_features % n_target != 0:
        return None
    branch_factor = out_features // n_target
    if branch_factor == 0:
        return None
    return w.reshape(n_target, branch_factor, -1)


def level_weight_mass(
    compartment_weights: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Return the weight mass each source delivers to each target per level."""
    return [None if w is None else w.abs().sum(dim=1) for w in compartment_weights]


def level_weight_per_compartment(
    compartment_weights: Sequence[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Return each level's weight mass divided by its compartment count."""
    return [
        None if w is None else w.abs().sum(dim=1) / float(w.shape[1])
        for w in compartment_weights
    ]


def targeting_index(
    per_level: Sequence[torch.Tensor | None],
    eps: float = EPS,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor] | None:
    """Return the targeting index, level fractions and connected pairs.

    ``per_level[l]`` has shape ``(n_target, n_source)``; levels given as
    ``None`` are skipped, and fewer than two present levels return ``None``.
    The index and the connected mask are ``(n_source, n_target)``; the level
    fractions keep the input orientation and are zero for disconnected pairs.
    """
    present = [level for level in per_level if level is not None]
    if len(present) < 2:
        return None
    stacked = torch.stack(present, dim=0)
    total = stacked.sum(dim=0)
    connected = total > eps
    safe_total = total.clamp(min=eps)
    level_fracs = [
        torch.where(connected, stacked[i] / safe_total, torch.zeros_like(total))
        for i in range(stacked.shape[0])
    ]
    index = (level_fracs[-1] - level_fracs[0]).T
    return index, level_fracs, connected.T


def classify_sources(
    index: torch.Tensor,
    hi: float,
    sigma: float,
    labels: TargetingClassLabels,
    connected: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Return each source's mean and spread of the index over its targets, and its class.

    Sources without a connected target are ``labels.disconnected`` with NaN
    statistics.
    """
    if connected is None:
        connected = torch.ones_like(index, dtype=torch.bool)
    mean = torch.full(
        (index.shape[0],), torch.nan, dtype=index.dtype, device=index.device
    )
    std = torch.full_like(mean, torch.nan)
    classes: list[str] = []
    for source in range(index.shape[0]):
        values = index[source][connected[source]]
        if values.numel() == 0:
            classes.append(labels.disconnected)
            continue
        mean[source] = values.mean()
        std[source] = values.std(unbiased=False)
        m = float(mean[source].item())
        s = float(std[source].item())
        if s > sigma:
            classes.append(labels.mixed)
        elif m > hi:
            classes.append(labels.soma)
        elif m < -hi:
            classes.append(labels.distal)
        else:
            classes.append(labels.diffuse)
    return mean, std, classes


def pair_level_stats(
    index: torch.Tensor,
    hi: float,
    connected: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return the percentages of connected pairs by their targeting index.

    Strong soma (``index >= hi``), strong distal (``index <= -hi``), balanced
    (``|index| < hi``) and any soma bias (``index > 0``).
    """
    if connected is None:
        connected = torch.ones_like(index, dtype=torch.bool)
    values = index[connected]
    n = values.numel()
    n_possible = index.numel()
    if n == 0:
        return {
            "strong_soma_pct": 0.0,
            "strong_distal_pct": 0.0,
            "balanced_pct": 0.0,
            "any_soma_bias_pct": 0.0,
            "n_pairs": 0,
            "n_possible_pairs": int(n_possible),
            "n_disconnected_pairs": int(n_possible),
        }
    return {
        "strong_soma_pct": float((values >= hi).float().mean().item() * 100.0),
        "strong_distal_pct": float((values <= -hi).float().mean().item() * 100.0),
        "balanced_pct": float(
            ((values > -hi) & (values < hi)).float().mean().item() * 100.0
        ),
        "any_soma_bias_pct": float((values > 0).float().mean().item() * 100.0),
        "n_pairs": int(n),
        "n_possible_pairs": int(n_possible),
        "n_disconnected_pairs": int(n_possible - n),
    }


def population_fraction_summary(
    level_fracs: Sequence[torch.Tensor],
    connected: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return the mean distal, middle and soma fractions over connected pairs.

    ``connected`` has the orientation of the level fractions.
    """
    if connected is None:
        connected = torch.ones_like(level_fracs[0], dtype=torch.bool)

    def _connected_mean(values: torch.Tensor) -> float:
        selected = values[connected]
        return float(selected.mean().item()) if selected.numel() else 0.0

    return {
        "frac_distal_pop": _connected_mean(level_fracs[0]),
        "frac_soma_pop": _connected_mean(level_fracs[-1]),
        "frac_middle_pop": (
            _connected_mean(torch.stack(list(level_fracs[1:-1]), dim=0).mean(dim=0))
            if len(level_fracs) > 2
            else 0.0
        ),
    }


def pair_stats_by_class(
    index: torch.Tensor,
    classes: list[str],
    class_order: Sequence[str],
    hi: float = 0.1,
    connected: torch.Tensor | None = None,
    count_key: str = "n_source",
) -> dict[str, dict[str, float]]:
    """Summarize the connected pairs' targeting index within each source class."""
    out: dict[str, dict[str, float]] = {}
    for cname in class_order:
        idx = [i for i, c in enumerate(classes) if c == cname]
        if not idx:
            out[cname] = {count_key: 0, "n_pairs": 0}
            continue
        class_values = index[idx]
        vals = (
            class_values.reshape(-1)
            if connected is None
            else class_values[connected[idx]]
        )
        if vals.numel() == 0:
            out[cname] = {count_key: len(idx), "n_pairs": 0}
            continue
        out[cname] = {
            count_key: len(idx),
            "n_pairs": int(vals.numel()),
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "q10": float(torch.quantile(vals, 0.10).item()),
            "q50": float(torch.quantile(vals, 0.50).item()),
            "q90": float(torch.quantile(vals, 0.90).item()),
            "strong_soma_pct": float((vals >= hi).float().mean().item() * 100.0),
            "strong_distal_pct": float((vals <= -hi).float().mean().item() * 100.0),
            "balanced_pct": float(
                ((vals > -hi) & (vals < hi)).float().mean().item() * 100.0
            ),
            "bimodality": float(
                2.0
                * min(
                    float((vals >= hi).float().mean().item()),
                    float((vals <= -hi).float().mean().item()),
                )
            ),
        }
    return out


__all__ = [
    "EPS",
    "TargetingClassLabels",
    "apply_weight_transform",
    "classify_sources",
    "level_weight_mass",
    "level_weight_per_compartment",
    "pair_level_stats",
    "pair_stats_by_class",
    "population_fraction_summary",
    "reshape_by_compartment",
    "targeting_index",
]
