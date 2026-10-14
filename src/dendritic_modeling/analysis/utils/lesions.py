"""Raw parameter values that switch synapses off."""

from __future__ import annotations

from typing import Any


def lesion_raw_value(module: Any) -> float:
    """Return the raw weight that makes ``module``'s transformed weights zero.

    ``identity`` and ``relu`` weights are zero at a raw value of 0; ``exp`` and
    ``softplus`` weights at -1000 are zero in floating point. The transform is
    the synapse bank's own ``weight_transform``, so banks of one model with
    different transforms are lesioned correctly.
    """
    transform = str(getattr(module, "weight_transform", "") or "").lower()
    if transform in {"identity", "relu"}:
        return 0.0
    if transform in {"exp", "softplus"}:
        return -1000.0
    raise ValueError(
        f"cannot lesion {type(module).__name__} synapses with weight_transform "
        f"{transform!r}"
    )


__all__ = ["lesion_raw_value"]
