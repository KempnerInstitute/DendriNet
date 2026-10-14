"""Seed derivation shared by configuration and component code.

The library's seed policy: a seed written in a config is used as written, and
an unset (``None``) seed is derived from one of the experiment streams in
``experiment`` (see ``utils.reproducibility.resolve_experiment_seeds``).
Derived seeds come from :func:`derive_seed`, so they depend only on the stream
seed and a stable name, never on build order or list positions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Optional

# Derived seeds fit in 31 bits so every consumer accepts them, including
# numpy.random.RandomState, which needs seeds below 2**32.
_SEED_MASK = (1 << 31) - 1


def derive_seed(base_seed: int, namespace: str) -> int:
    """Return a seed determined by ``base_seed`` and the name ``namespace``.

    Distinct namespaces give unrelated seeds for the same base seed. The offset
    is the library's ``stable_seed_offset`` hash, as used for component seeds.
    """
    from dendritic_modeling.utils.stable_hash import stable_seed_offset

    return (int(base_seed) + stable_seed_offset(namespace)) & _SEED_MASK


def seed_or_default(value: Optional[Any], default: int) -> int:
    """Return ``value`` as an int, or ``default`` when it is unset.

    Components use this where a seed may still be unset because they were built
    outside a training run, which fills unset seeds from the experiment streams.
    """
    return int(default) if value is None else int(value)


def with_default_seeds(params: Any, defaults: Mapping[str, int]) -> Any:
    """Return dataclass ``params`` with its unset seed fields set to ``defaults``.

    The input is not modified; a copy is returned when a field was unset.
    """
    changes = {
        name: int(default)
        for name, default in defaults.items()
        if getattr(params, name, None) is None
    }
    return dataclasses.replace(params, **changes) if changes else params


__all__ = ["derive_seed", "seed_or_default", "with_default_seeds"]
