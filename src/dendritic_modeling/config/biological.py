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
# Explicitly signed/MLP aliases are conventional controls, not biological
# neuron recipes. Other architecture families also retain their own contracts.
BIOLOGICAL_DEFAULT_CORE_TYPES = BIOLOGICAL_CORE_TYPES - {
    "dendritic_signed",
    "flat_signed",
    "dendritic_mlp",
    "flat_mlp",
}


def resolve_biological_neuron(core_type: Any, value: Any) -> bool:
    """Return the ``biological_neuron`` declaration of a core as a bool.

    An omitted/null declaration enables the contract for standard dendritic
    E/I cores. Conventional controls and unrelated core types default off.
    The versioned upgrader writes false for historical omitted declarations.
    Raises for a value that is not a bool, and for ``true`` on a core type the
    constraint set does not cover.
    """
    lowered = str(core_type or "").strip().lower()
    if value is None:
        return lowered in BIOLOGICAL_DEFAULT_CORE_TYPES
    if not isinstance(value, bool):
        raise ValueError(
            f"model.core.biological_neuron must be true or false, got {value!r}"
        )
    if value and lowered not in BIOLOGICAL_CORE_TYPES:
        raise ValueError(
            f"model.core.biological_neuron: true is not available for core type "
            f"{core_type!r}. The biological constraint set covers dendritic E/I "
            "cores only: structured EINet, unified E/I and population-network "
            "types."
        )
    return value


def validate_input_source_identity(identity: Any, independent_pathways: Any) -> None:
    """Validate the explicit identity of external E/I sources."""
    if not isinstance(independent_pathways, bool):
        raise ValueError("independent_pathways must be true or false")
    if not isinstance(identity, str) or identity not in {"shared", "paired_afferents"}:
        raise ValueError("input_source_identity must be shared or paired_afferents")
    if independent_pathways and identity != "shared":
        raise ValueError(
            "paired_afferents requires independent_pathways: false; "
            "split channels already have distinct identities"
        )


def reference_core_config(config: Any) -> dict:
    """Copy a composite/control's internal EINet recipe with its old contract.

    These constructors do not declare a biological neuronal core merely by
    reusing EINet internally. Preserve any explicit declaration on the reference.
    """
    from dendritic_modeling.config.conversion import to_plain_dict

    result = to_plain_dict(config)
    if result.get("biological_neuron") is None:
        result["biological_neuron"] = False
    return result


__all__ = [
    "BIOLOGICAL_CORE_TYPES",
    "BIOLOGICAL_DEFAULT_CORE_TYPES",
    "resolve_biological_neuron",
]
