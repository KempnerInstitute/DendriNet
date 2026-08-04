"""Direct sparse indices for morphology-aligned spatial connectivity.

The sampler in this module maps a dendritic tree onto a ``(C, H, W)`` input
without constructing a dense ``[n_branches, C * H * W]`` mask.  Every soma
owns the full image.  Moving from the soma toward the leaves recursively
partitions that image according to the configured branch factors and split
axes.  A branch samples exactly ``K`` unique input features from its rectangle.

No weights are shared between somas or spatial locations.  This is therefore
an anatomical sparse-connectivity prior, not a convolution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _resolve_index_dtype,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    _resolve_pathway_config,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset

SPATIAL_MORPHOLOGY_METHODS = frozenset({"spatial_morphology", "morphology_spatial"})
_AXIS_ALIASES = {
    "h": "height",
    "height": "height",
    "y": "height",
    "w": "width",
    "width": "width",
    "x": "width",
}


def uses_spatial_morphology(config, pathway: str) -> bool:
    """Return whether one configured pathway uses spatial image routing."""

    pathway_config = _resolve_pathway_config(config, pathway)
    method = str(pathway_config.get("method", "")).strip().lower()
    return method in SPATIAL_MORPHOLOGY_METHODS


def _positive_int_tuple(
    values: Sequence[int],
    *,
    name: str,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    """Normalize a positive integer sequence."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers")
    normalized = tuple(int(value) for value in values)
    if expected_length is not None and len(normalized) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, got {len(normalized)}"
        )
    if not normalized or any(value < 1 for value in normalized):
        raise ValueError(f"{name} values must all be positive")
    return normalized


def _normalize_split_axes(
    values: Sequence[str] | None,
    *,
    depth: int,
) -> tuple[str, ...]:
    """Return root-to-leaf spatial split axes."""

    if values is None:
        return tuple("height" if idx % 2 == 0 else "width" for idx in range(depth))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("spatial.split_axes must be a sequence")
    if len(values) != depth:
        raise ValueError(
            "spatial.split_axes must contain one root-to-leaf axis per "
            f"branch factor, got {len(values)} axes for depth {depth}"
        )
    axes = []
    for value in values:
        normalized = _AXIS_ALIASES.get(str(value).strip().lower())
        if normalized is None:
            choices = ", ".join(sorted(_AXIS_ALIASES))
            raise ValueError(
                f"Unknown spatial split axis {value!r}; choose from {choices}"
            )
        axes.append(normalized)
    return tuple(axes)


def _branch_path(
    local_branch_idx: int,
    branch_factors: tuple[int, ...],
) -> tuple[int, ...]:
    """Decode a root-to-branch path from a mixed-radix branch index."""

    digits = []
    remainder = int(local_branch_idx)
    for idx, factor in enumerate(branch_factors):
        suffix = math.prod(branch_factors[idx + 1 :])
        digit, remainder = divmod(remainder, suffix)
        if digit >= factor:
            raise RuntimeError("Decoded branch path exceeds its branch factor")
        digits.append(digit)
    return tuple(digits)


def _split_interval(
    lower: int,
    upper: int,
    *,
    factor: int,
    child: int,
) -> tuple[int, int]:
    """Select one integer child interval from an equal continuous partition."""

    length = upper - lower
    child_lower = lower + (length * child) // factor
    child_upper = lower + (length * (child + 1)) // factor
    if child_upper <= child_lower:
        raise ValueError(
            "Spatial morphology creates an empty region. Increase the input "
            "resolution or reduce branching along the corresponding axis."
        )
    return child_lower, child_upper


def _path_rectangle(
    *,
    height: int,
    width: int,
    path: tuple[int, ...],
    branch_factors: tuple[int, ...],
    split_axes: tuple[str, ...],
) -> tuple[int, int, int, int]:
    """Return ``(y0, y1, x0, x1)`` for one root-to-branch path."""

    y0, y1 = 0, height
    x0, x1 = 0, width
    for child, factor, axis in zip(path, branch_factors, split_axes):
        if axis == "height":
            y0, y1 = _split_interval(y0, y1, factor=factor, child=child)
        else:
            x0, x1 = _split_interval(x0, x1, factor=factor, child=child)
    return y0, y1, x0, x1


def _rectangle_candidates(
    *,
    channels: int,
    height: int,
    width: int,
    rectangle: tuple[int, int, int, int],
) -> torch.Tensor:
    """Return flattened CHW indices belonging to a rectangle."""

    y0, y1, x0, x1 = rectangle
    y = torch.arange(y0, y1, dtype=torch.long)
    x = torch.arange(x0, x1, dtype=torch.long)
    spatial = (y[:, None] * width + x[None, :]).reshape(-1)
    channel_offsets = torch.arange(channels, dtype=torch.long) * height * width
    return (channel_offsets[:, None] + spatial[None, :]).reshape(-1)


def _sample_unique_candidates(
    candidates: torch.Tensor,
    *,
    n_rows: int,
    synapses_per_branch: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample deterministic unique candidates without a dense row-wise mask."""

    population_size = int(candidates.numel())
    if synapses_per_branch > population_size:
        raise ValueError(
            f"Requested {synapses_per_branch} synapses from a spatial branch "
            f"region containing only {population_size} input features"
        )

    # Shuffle each region once, then give every row an independently seeded
    # affine traversal of that permutation. A step coprime to the population
    # size guarantees unique indices within each row for every valid K.
    permutation = torch.randperm(population_size, generator=generator)
    starts = torch.randint(
        population_size,
        (n_rows,),
        generator=generator,
        dtype=torch.long,
    )
    if population_size == 1:
        steps = torch.ones(n_rows, dtype=torch.long)
    else:
        steps = torch.randint(
            1,
            population_size,
            (n_rows,),
            generator=generator,
            dtype=torch.long,
        )
        invalid = torch.gcd(steps, torch.full_like(steps, population_size)) != 1
        while bool(invalid.any()):
            steps[invalid] = torch.randint(
                1,
                population_size,
                (int(invalid.sum().item()),),
                generator=generator,
                dtype=torch.long,
            )
            invalid = torch.gcd(steps, torch.full_like(steps, population_size)) != 1

    offsets = (
        starts[:, None]
        + steps[:, None] * torch.arange(synapses_per_branch, dtype=torch.long)
    ) % population_size
    return candidates[permutation[offsets]]


def _spatial_options(config: Mapping[str, object]) -> Mapping[str, object]:
    """Return the nested spatial options with a clear configuration error."""

    spatial = to_plain_dict(config.get("spatial", {}))
    if not spatial:
        raise ValueError(
            "method='spatial_morphology' requires a structured.spatial block"
        )
    return spatial


def sample_spatial_morphology_indices(
    *,
    input_shape: Sequence[int],
    out_features: int,
    in_features: int,
    synapses_per_branch: int,
    owner_count: int,
    branch_factors: Sequence[int],
    level_idx: int,
    split_axes: Sequence[str] | None = None,
    seed: int = 0,
    index_dtype: str | torch.dtype = "int64",
) -> torch.Tensor:
    """Generate compact branch indices for a full-image dendritic map.

    ``level_idx`` follows DendriNet construction order: zero is the distal
    level and ``len(branch_factors)`` is the soma. ``split_axes`` and
    ``branch_factors`` are specified in the opposite, anatomical order:
    root-to-leaf.
    """

    channels, height, width = _positive_int_tuple(
        input_shape,
        name="spatial.input_shape",
        expected_length=3,
    )
    if channels * height * width != int(in_features):
        raise ValueError(
            "spatial.input_shape does not match the pathway input dimension: "
            f"{channels}*{height}*{width} != {in_features}"
        )
    factors = _positive_int_tuple(branch_factors, name="branch_factors")
    axes = _normalize_split_axes(split_axes, depth=len(factors))
    if owner_count < 1:
        raise ValueError("owner_count must be positive")
    if level_idx < 0 or level_idx > len(factors):
        raise ValueError(f"level_idx must be in [0, {len(factors)}], got {level_idx}")
    if synapses_per_branch < 1:
        raise ValueError("synapses_per_branch must be positive")

    anatomical_depth = len(factors) - level_idx
    active_factors = factors[:anatomical_depth]
    active_axes = axes[:anatomical_depth]
    branches_per_owner = math.prod(active_factors)
    expected_out_features = owner_count * branches_per_owner
    if int(out_features) != expected_out_features:
        raise ValueError(
            "Spatial morphology does not match the DendriNet branch layout: "
            f"expected {expected_out_features} rows, got {out_features}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % ((1 << 63) - 1))
    indices = torch.empty(
        int(out_features),
        int(synapses_per_branch),
        dtype=torch.long,
    )
    owner_offsets = torch.arange(owner_count, dtype=torch.long) * branches_per_owner

    for local_branch_idx in range(branches_per_owner):
        path = _branch_path(local_branch_idx, active_factors)
        rectangle = _path_rectangle(
            height=height,
            width=width,
            path=path,
            branch_factors=active_factors,
            split_axes=active_axes,
        )
        candidates = _rectangle_candidates(
            channels=channels,
            height=height,
            width=width,
            rectangle=rectangle,
        )
        rows = owner_offsets + local_branch_idx
        indices[rows] = _sample_unique_candidates(
            candidates,
            n_rows=owner_count,
            synapses_per_branch=synapses_per_branch,
            generator=generator,
        )
    return indices.to(
        dtype=_resolve_index_dtype(index_dtype, in_features=int(in_features))
    )


def sample_configured_indices(
    config,
    *,
    pathway: str,
    out_features: int,
    in_features: int,
    synapses_per_branch: int,
    owner_count: int,
    branch_factors: Sequence[int],
    layer_idx: int = 0,
    level_idx: int = 0,
    index_dtype: str | torch.dtype = "int64",
) -> torch.Tensor | None:
    """Sample direct sparse indices for a configured structured pathway."""

    pathway_config = _resolve_pathway_config(config, pathway)
    if not pathway_config:
        return None
    method = str(pathway_config.get("method", "bernoulli")).strip().lower()
    if method not in SPATIAL_MORPHOLOGY_METHODS:
        return None

    spatial = _spatial_options(pathway_config)
    base_seed = int(pathway_config.get("seed", 0) or 0)
    seed = (
        base_seed
        + stable_seed_offset(
            "spatial_morphology",
            pathway,
            int(layer_idx),
            int(level_idx),
            int(out_features),
            int(in_features),
        )
    ) % ((1 << 63) - 1)
    return sample_spatial_morphology_indices(
        input_shape=spatial.get("input_shape", ()),
        out_features=out_features,
        in_features=in_features,
        synapses_per_branch=synapses_per_branch,
        owner_count=owner_count,
        branch_factors=branch_factors,
        level_idx=level_idx,
        split_axes=spatial.get("split_axes"),
        seed=seed,
        index_dtype=index_dtype,
    )


__all__ = [
    "SPATIAL_MORPHOLOGY_METHODS",
    "sample_configured_indices",
    "sample_spatial_morphology_indices",
    "uses_spatial_morphology",
]
