"""Options of cellwise impact rewiring (``training.<phase>.common.impact_rewiring``).

The rewiring maintenance lives with the indexed synapse layers, which import
these options; configuration loading validates them without importing torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Registered regrowth policies. Sources: "uniform" samples every legal unused
# source; "active" restricts sampling to sources whose tracked presynaptic
# activity exceeds ``min_source_activity`` (requires
# ``release_track_source_activity``). Weights: "cell_median" starts a new
# contact at the cell/sign median conductance; "initial" draws it from the
# bank's recorded initialization law (requires an initializer that records one,
# such as ``fan_in_lognormal``).
REGROWTH_SOURCE_POLICIES = ("uniform", "active")
REGROWTH_WEIGHT_POLICIES = ("cell_median", "initial")


@dataclass(frozen=True)
class ImpactRewiringConfig:
    """Schedule and limits of cellwise impact rewiring.

    At every ``frequency``-th optimizer step a contact is eligible when its
    impact is below ``threshold`` times its cell's mean impact and it is at
    least ``maturity_steps`` optimizer steps old. Each cell replaces at most
    ``floor(n_eligible * eligible_fraction)`` eligible contacts ("up to 10% of
    eligible contacts" at the default), and at most
    ``floor(K * segment_fraction)`` per dendritic segment. Because the per-cell
    limit is floored, a cell with fewer than ``1 / eligible_fraction``
    eligible contacts (fewer than 10 at the default) replaces none at that
    event; this is the specified behavior, not a rounding error.
    """

    frequency: int = 100
    maturity_steps: int = 100
    threshold: float = 0.1
    eligible_fraction: float = 0.1
    segment_fraction: float = 0.25
    until_step: int | None = None
    seed: int = 0
    regrowth_sources: str = "uniform"
    min_source_activity: float = 0.0
    regrowth_weight: str = "cell_median"

    def __post_init__(self):
        if self.frequency < 1 or self.maturity_steps < 1:
            raise ValueError("rewiring frequency and maturity must be positive")
        for value in (self.threshold, self.eligible_fraction, self.segment_fraction):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("rewiring fractions must be finite and in [0, 1]")
        if self.until_step is not None and self.until_step < 0:
            raise ValueError("until_step must be nonnegative")
        if self.regrowth_sources not in REGROWTH_SOURCE_POLICIES:
            raise ValueError(
                f"regrowth_sources must be one of {REGROWTH_SOURCE_POLICIES}, "
                f"got {self.regrowth_sources!r}"
            )
        if self.regrowth_weight not in REGROWTH_WEIGHT_POLICIES:
            raise ValueError(
                f"regrowth_weight must be one of {REGROWTH_WEIGHT_POLICIES}, "
                f"got {self.regrowth_weight!r}"
            )
        if not math.isfinite(self.min_source_activity) or self.min_source_activity < 0:
            raise ValueError("min_source_activity must be finite and nonnegative")
        if self.min_source_activity and self.regrowth_sources != "active":
            raise ValueError("min_source_activity requires regrowth_sources='active'")


__all__ = [
    "REGROWTH_SOURCE_POLICIES",
    "REGROWTH_WEIGHT_POLICIES",
    "ImpactRewiringConfig",
]
