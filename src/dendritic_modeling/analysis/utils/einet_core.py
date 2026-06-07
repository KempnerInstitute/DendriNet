"""Helpers for analysis tools that operate on dendritic E/I cores."""

from dendritic_modeling.networks import ExcitationInhibitionNetwork


def has_einet_core(model) -> bool:
    """Return whether a model exposes a plain or tensor-map-wrapped EINet core."""
    core = getattr(model, "core_network", None)
    if isinstance(core, ExcitationInhibitionNetwork):
        return True

    wrapped = getattr(core, "einet", None)
    return isinstance(wrapped, ExcitationInhibitionNetwork)
