"""Which core types can declare ``model.core.biological_neuron: true``.

The declaration states that a core is a biologically constrained E/I network.
The constraint set is checked on the built model by
:mod:`dendritic_modeling.networks.architectures.biological_constraints`; this
module only validates the declared value, without importing torch.
"""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.recurrence import (
    _POPULATION_NETWORK_TYPES,
    _STRUCTURED_EI_TYPES,
    _UNIFIED_EI_TYPES,
)

# Dendritic E/I core families whose built models the constraint check reads.
BIOLOGICAL_CORE_TYPES = frozenset(
    _STRUCTURED_EI_TYPES | _UNIFIED_EI_TYPES | _POPULATION_NETWORK_TYPES
)


def resolve_biological_neuron(core_type: Any, value: Any) -> bool:
    """Return the ``biological_neuron`` declaration of a core as a bool.

    ``None`` means not declared (configs saved before 2026-10-08 store null).
    Raises for a value that is not a bool, and for ``true`` on a core type the
    constraint set does not cover.
    """
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(
            f"model.core.biological_neuron must be true or false, got {value!r}"
        )
    lowered = str(core_type or "").strip().lower()
    if value and lowered not in BIOLOGICAL_CORE_TYPES:
        raise ValueError(
            f"model.core.biological_neuron: true is not available for core type "
            f"{core_type!r}. The biological constraint set covers dendritic E/I "
            "cores only: structured EINet, unified E/I and population-network "
            "types."
        )
    return value


__all__ = ["BIOLOGICAL_CORE_TYPES", "resolve_biological_neuron"]
