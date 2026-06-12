"""Pure STDP helper functions for local credit assignment.

Extracted from ``LocalCreditAssignment`` so the trace/activity math and the
pathway-enable predicates are unit-testable in isolation. These functions hold
no trainer state: config is passed in explicitly as ``local_cfg`` and tensors
are detached where the originals detached.
"""

from __future__ import annotations

import logging
import math

import torch
from torch.nn import functional

logger = logging.getLogger(__name__)


def stdp_enabled(local_cfg) -> bool:
    """Return whether trace-based STDP updates should be applied."""
    stdp_cfg = getattr(local_cfg, "stdp", None)
    return bool(
        getattr(stdp_cfg, "enabled", False)
        or str(getattr(local_cfg, "rule_variant", "")).lower() == "stdp"
    )


def stdp_pathway_enabled(local_cfg, pathway: str) -> bool:
    """Return whether STDP is configured for a recorded synaptic pathway."""
    stdp_cfg = getattr(local_cfg, "stdp", None)
    apply_to = getattr(stdp_cfg, "apply_to", ("exc", "rec_exc"))
    if isinstance(apply_to, str):
        apply_to = (apply_to,)
    normalized = {str(item).strip().lower().replace("-", "_") for item in apply_to}
    return "all" in normalized or pathway in normalized


def stdp_activity(values: torch.Tensor, threshold: float, mode: str) -> torch.Tensor:
    """Convert raw local values to non-backpropagating STDP activities."""
    activity = values.detach()
    if activity.dim() != 2:
        activity = activity.reshape(activity.size(0), -1)
    centered = activity - float(threshold)
    mode = (mode or "relu").lower()
    if mode == "identity":
        return centered
    if mode == "binary":
        return (centered > 0).to(dtype=activity.dtype)
    if mode in {"abs", "absolute"}:
        return centered.abs()
    if mode != "relu":
        logger.warning("Unknown stdp.activity_mode=%r; using relu.", mode)
    return functional.relu(centered)


def stdp_decay(tau: float) -> float:
    """Single-step exponential decay for an STDP eligibility trace."""
    tau = max(float(tau), 1e-6)
    return math.exp(-1.0 / tau)


__all__ = [
    "stdp_activity",
    "stdp_decay",
    "stdp_enabled",
    "stdp_pathway_enabled",
]
