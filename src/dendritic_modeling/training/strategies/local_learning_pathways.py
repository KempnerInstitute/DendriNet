"""Shared pathway descriptors for local-learning synaptic updates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopKGradientPath:
    """Recorded TopK pathway keys used by local-learning dispatch."""

    pathway: str
    module_key: str
    input_key: str
    mask_key: str


EXCITATORY_TOPK_PATHS = (
    TopKGradientPath("exc", "exc_module", "x_exc", "exc_mask"),
    TopKGradientPath("rec_exc", "rec_exc_module", "x_rec_exc", "rec_exc_mask"),
)

INHIBITORY_TOPK_PATHS = (
    TopKGradientPath("inh", "inh_module", "x_inh", "inh_mask"),
    TopKGradientPath("rec_inh", "rec_inh_module", "x_rec_inh", "rec_inh_mask"),
)

STDP_TOPK_PATHS = (
    TopKGradientPath("exc", "exc_module", "x_exc", "exc_mask"),
    TopKGradientPath("inh", "inh_module", "x_inh", "inh_mask"),
    TopKGradientPath("rec_exc", "rec_exc_module", "x_rec_exc", "rec_exc_mask"),
    TopKGradientPath("rec_inh", "rec_inh_module", "x_rec_inh", "rec_inh_mask"),
)


__all__ = [
    "EXCITATORY_TOPK_PATHS",
    "INHIBITORY_TOPK_PATHS",
    "STDP_TOPK_PATHS",
    "TopKGradientPath",
]
