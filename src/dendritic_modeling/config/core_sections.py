"""Which ``model.core`` sections each core family reads.

``CoreConfig`` carries the sections of every family, and each builder reads
only its own. A section written explicitly for a core type that never reads it
would be accepted and ignored, so the loader rejects it unless it still holds
its default value (saved ``config.json`` files contain every section).

The names of the built-in core types live here, without torch, so the loader
can tell them from core types that plugins register (whose sections are not
checked); the architecture factory builds exactly these types.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable
from typing import Any

from dendritic_modeling.config.keys import field_default
from dendritic_modeling.config.recurrence import (
    _BASELINE_RNN_TYPES,
    _HETEROGENEOUS_LEAK_CTRNN_TYPES,
    _LEGENDRE_MEMORY_TYPES,
    _POPULATION_NETWORK_TYPES,
    _STRUCTURED_EI_TYPES,
    _UNIFIED_EI_TYPES,
)

logger = logging.getLogger(__name__)

# Fields every core family consumes (training setup propagates the seed; the
# biological declaration is validated for every core type).
COMMON_CORE_FIELDS = frozenset({"type", "initialization_seed", "biological_neuron"})

#: Spatial dendritic cores that flatten a feature map into a structured EINet.
SPATIAL_DENDRITIC_TYPES = frozenset(
    {"hierarchical_dendritic_tensor_map", "dendritic_tensor_map", "tensor_map"}
)
#: Shared-patch dendritic convolutions (``hierarchical_dendritic_conv``).
SPATIAL_DENDRITIC_CONV_TYPES = frozenset(
    {"hierarchical_dendritic_conv", "spatial_dendritic_conv"}
)
#: Shared-patch fixed-index affine controls.
SPATIAL_PATCH_POINT_TYPES = frozenset(
    {"sparse_active_matched_point_conv", "spatial_sparse_active_matched_point"}
)
#: Built-in core types besides the structured E/I, payload-family and spatial
#: types: controls, classical networks and dendritic AlexNet.
OTHER_BUILTIN_CORE_TYPES = frozenset(
    {
        "active_param_mlp",
        "affine_mlp",
        "routed_ei",
        "shared_projection_shunting",
        "alexnet",
        "cnn",
        "cnn_down",
        "cnn_downsample",
        "cnn_up",
        "cnn_upsample",
        "cnnautoencoder",
        "cnnvariationalautoencoder",
        "cnndownsample",
        "cnnupsample",
        "direct_active_matched_point",
        "dendritic_alexnet",
        "effectivematchedparammlp",
        "identity",
        "mlp",
        "mlpautoencoder",
        "mlpvariationalautoencoder",
        "none",
        "pathway_router",
        "point_mlp",
        "router",
        "sparse_active_matched_point",
        "ss_mlp",
        "ss_mlp_flat",
        "total_param_mlp",
    }
)
#: Every core type the architecture factory builds without a plugin.
BUILTIN_CORE_TYPES = frozenset(
    _STRUCTURED_EI_TYPES
    | _UNIFIED_EI_TYPES
    | _BASELINE_RNN_TYPES
    | _HETEROGENEOUS_LEAK_CTRNN_TYPES
    | _LEGENDRE_MEMORY_TYPES
    | _POPULATION_NETWORK_TYPES
    | SPATIAL_DENDRITIC_TYPES
    | SPATIAL_DENDRITIC_CONV_TYPES
    | SPATIAL_PATCH_POINT_TYPES
    | OTHER_BUILTIN_CORE_TYPES
)
#: The ``model.core.spatial`` options each core type reads (None: all of
#: them). Every other built-in core type reads none.
SPATIAL_OPTIONS_READ: dict[str, frozenset[str] | None] = {
    **dict.fromkeys(SPATIAL_DENDRITIC_CONV_TYPES, None),
    **dict.fromkeys(
        SPATIAL_PATCH_POINT_TYPES, frozenset({"kernel_size", "stride", "padding"})
    ),
}

# Families that read only their own payload section.
_PAYLOAD_FAMILIES: tuple[tuple[frozenset[str], str, str], ...] = (
    (frozenset({"routed_ei"}), "routed_ei", "routed_ei"),
    (
        frozenset({"shared_projection_shunting"}),
        "shared_projection_shunting",
        "shared_projection_shunting",
    ),
    (frozenset(_POPULATION_NETWORK_TYPES), "population_network", "population_network"),
    (frozenset(_UNIFIED_EI_TYPES), "unified_ei", "unified_ei"),
    (frozenset(_BASELINE_RNN_TYPES), "baseline_rnn", "baseline_rnn"),
    (
        frozenset(_HETEROGENEOUS_LEAK_CTRNN_TYPES),
        "heterogeneous_leak_ctrnn",
        "heterogeneous_leak_ctrnn",
    ),
    (frozenset(_LEGENDRE_MEMORY_TYPES), "legendre_memory", "legendre_memory"),
)

# Payload sections of other families, and the spatial convolution section.
# Structured EINet cores read none of them.
EINET_FOREIGN_PAYLOADS = frozenset(
    {
        "population_network",
        "unified_ei",
        "baseline_rnn",
        "heterogeneous_leak_ctrnn",
        "legendre_memory",
        "vision",
        "shared_projection_shunting",
        "affine_mlp",
        "routed_ei",
        "spatial",
    }
)

# Values that training setup writes into every core (topology seeds). They are
# ignored when deciding whether a section differs from its default.
_SETUP_FILLED_FIELDS = {
    "sparsity": (("indexed", "seed"),),
    "connectivity": (("structured", "seed"),),
    "transfer": (("split_seed",),),
}

_HINTS = {
    "population_network": (
        "population networks read only model.core.population_network; set "
        "per-population options in its population_defaults or populations"
    ),
    "unified_ei": "unified E/I cores read only model.core.unified_ei",
}


def core_family(core_type: Any) -> str | None:
    """Return the payload family of a core type, or None for section-based cores."""
    lowered = str(core_type or "").strip().lower()
    for types, family, _ in _PAYLOAD_FAMILIES:
        if lowered in types:
            return family
    return None


def _unread_spatial_options(core: Any, core_type: str) -> list[str]:
    """``spatial`` options with non-default values that ``core_type`` never reads."""
    read = SPATIAL_OPTIONS_READ.get(core_type, frozenset())
    if read is None:
        return []
    value = _comparable("spatial", getattr(core, "spatial", None))
    default = _comparable("spatial", field_default(type(core), "spatial"))
    if not isinstance(value, dict) or not isinstance(default, dict):
        return [] if value == default else ["spatial"]
    written = [
        key for key in value if key not in read and value[key] != default.get(key)
    ]
    if not read:
        return ["spatial"] if written else []
    return [f"spatial.{key}" for key in written]


def unread_core_sections(core: Any, explicit: Iterable[str]) -> list[str]:
    """Return explicitly written, non-default sections the core type never reads.

    For a core type that reads only some ``spatial`` options, the unread ones
    are returned as ``spatial.<option>``.
    """
    lowered = str(getattr(core, "type", "") or "").strip().lower()
    if (
        lowered in BUILTIN_CORE_TYPES
        and lowered != "routed_ei"
        and "routed_ei" in explicit
        and getattr(core, "routed_ei", None) is not None
    ):
        return ["routed_ei"]
    if (
        lowered in BUILTIN_CORE_TYPES
        and lowered != "shared_projection_shunting"
        and "shared_projection_shunting" in explicit
        and getattr(core, "shared_projection_shunting", None) is not None
    ):
        return ["shared_projection_shunting"]
    family = core_family(lowered)
    if family is None and lowered not in _STRUCTURED_EI_TYPES:
        # Other builders (MLP controls, spatial and vision cores) read sections
        # that are not yet declared per type, except ``spatial``, which only
        # the spatial convolution types read. The sections of core types that
        # plugins register are not checked.
        if lowered not in BUILTIN_CORE_TYPES or "spatial" not in set(explicit):
            return []
        return _unread_spatial_options(core, lowered)
    unread = []
    for name in sorted(set(explicit) - COMMON_CORE_FIELDS):
        if family is not None:
            consumed = name == family
        else:
            consumed = name not in EINET_FOREIGN_PAYLOADS
        if consumed:
            continue
        value = _comparable(name, getattr(core, name))
        if value != _comparable(name, field_default(type(core), name)):
            unread.append(name)
    return unread


def _comparable(name: str, value: Any) -> Any:
    """Return section content with setup-written seeds reset to their defaults."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        default = dataclasses.asdict(type(value)())
        value = dataclasses.asdict(value)
        for path in _SETUP_FILLED_FIELDS.get(name, ()):
            target, reference = value, default
            for key in path[:-1]:
                target, reference = target.get(key), reference.get(key)
                if not isinstance(target, dict) or not isinstance(reference, dict):
                    break
            else:
                target[path[-1]] = reference.get(path[-1])
    return value


#: ``CoreConfig`` fields added after a release had written saved records, with
#: the first release that has them. A record from an earlier release lacks them.
CORE_FIELDS_ADDED = {"shared_projection_shunting": "0.3.0", "routed_ei": "unreleased"}


def is_saved_core_record(keys: Iterable[str]) -> bool:
    """Whether a ``model.core`` mapping is a saved record (``config.json``).

    A run records every ``CoreConfig`` field of the release that wrote it, so a
    saved record lists every current field except those added since.
    """
    from dendritic_modeling.config.model import CoreConfig

    current = {item.name for item in dataclasses.fields(CoreConfig)}
    return current - set(CORE_FIELDS_ADDED) <= set(keys)


def reject_unread_core_sections(core: Any, explicit: Iterable[str]) -> None:
    """Raise when explicit ``model.core`` sections would be ignored by the builder.

    A core mapping that lists every ``CoreConfig`` field is a saved record
    (``config.json``). Its sections are compared with today's defaults, which
    may have changed since it was written, so unread sections in a record are
    reported as a warning instead of an error.
    """
    explicit = set(explicit)
    unread = unread_core_sections(core, explicit)
    if not unread:
        return
    family = core_family(getattr(core, "type", ""))
    paths = ", ".join(f"model.core.{name}" for name in unread)
    hint = _HINTS.get(family or "", "")
    message = f"model.core.type={core.type!r} does not read {paths}" + (
        f"; {hint}" if hint else ""
    )
    if is_saved_core_record(explicit):
        logger.warning("Ignoring sections of a saved config: %s.", message)
        return
    raise ValueError(message + ". Remove them.")


__all__ = [
    "BUILTIN_CORE_TYPES",
    "COMMON_CORE_FIELDS",
    "CORE_FIELDS_ADDED",
    "EINET_FOREIGN_PAYLOADS",
    "OTHER_BUILTIN_CORE_TYPES",
    "SPATIAL_DENDRITIC_CONV_TYPES",
    "SPATIAL_DENDRITIC_TYPES",
    "SPATIAL_OPTIONS_READ",
    "SPATIAL_PATCH_POINT_TYPES",
    "core_family",
    "is_saved_core_record",
    "reject_unread_core_sections",
    "unread_core_sections",
]
