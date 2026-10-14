"""Utilities for structured synaptic connectivity masks."""

from __future__ import annotations

import torch

from dendritic_modeling.config.structured_connectivity import (
    DENSE_METHODS,
    DISTANCE_METHODS,
    GLOBAL_RNG_SEED,
    distance_base_probability,
    normalize_pathway_name,
    resolve_pathway_rule,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset


def validate_connection_mask(
    connection_mask: torch.Tensor | None,
    out_features: int,
    in_features: int,
    *,
    name: str = "connection_mask",
) -> torch.Tensor | None:
    """Validate and normalize a hard allowed-connection mask.

    Masks use the same orientation as linear weights: rows are postsynaptic
    output branches/neurons and columns are presynaptic input features.
    ``True`` means the connection is structurally allowed.
    """
    if connection_mask is None:
        return None

    mask = torch.as_tensor(connection_mask, dtype=torch.bool)
    expected_shape = (out_features, in_features)
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}"
        )
    if mask.numel() > 0 and bool((mask.sum(dim=1) == 0).any()):
        raise ValueError(f"{name} must allow at least one input per output row")
    return mask


def sample_probability_mask(
    out_features: int,
    in_features: int,
    probability: float | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample an allowed-connection mask from Bernoulli probabilities."""
    if isinstance(probability, torch.Tensor):
        probs = probability.to(dtype=torch.float32)
        if device is not None:
            probs = probs.to(device=device)
        if tuple(probs.shape) != (out_features, in_features):
            raise ValueError(
                "probability tensor must have shape "
                f"{(out_features, in_features)}, got {tuple(probs.shape)}"
            )
    else:
        probs = torch.full(
            (out_features, in_features),
            float(probability),
            dtype=torch.float32,
            device=device,
        )

    if bool(((probs < 0) | (probs > 1)).any()):
        raise ValueError("connection probabilities must be in [0, 1]")

    samples = torch.rand(
        out_features,
        in_features,
        generator=generator,
        device=probs.device,
    )
    mask = samples < probs
    empty_rows = mask.sum(dim=1) == 0
    repairable_rows = empty_rows & (probs.sum(dim=1) > 0)
    if bool(repairable_rows.any()):
        row_indices = repairable_rows.nonzero(as_tuple=False).flatten()
        for row_idx in row_indices:
            row_probs = probs[row_idx]
            selected = torch.multinomial(
                row_probs / row_probs.sum(),
                num_samples=1,
                generator=generator,
            )
            mask[row_idx, selected] = True
    return mask


def sample_fixed_indegree_mask(
    out_features: int,
    in_features: int,
    indegree: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Sample a mask with a fixed number of inputs per output row."""
    if indegree < 0 or indegree > in_features:
        raise ValueError(f"indegree must be in [0, {in_features}], got {indegree}")

    mask = torch.zeros(out_features, in_features, dtype=torch.bool, device=device)
    for out_idx in range(out_features):
        selected = torch.randperm(in_features, generator=generator, device=mask.device)[
            :indegree
        ]
        mask[out_idx, selected] = True
    return mask


def distance_kernel_probabilities(
    pre_positions: torch.Tensor,
    post_positions: torch.Tensor,
    *,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    base_probability: float = 1.0,
) -> torch.Tensor:
    """Build connection probabilities from pairwise pre/post distances."""
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if base_probability < 0 or base_probability > 1:
        raise ValueError("base_probability must be in [0, 1]")

    pre = torch.as_tensor(pre_positions, dtype=torch.float32)
    post = torch.as_tensor(post_positions, dtype=torch.float32, device=pre.device)
    if pre.dim() != 2 or post.dim() != 2:
        raise ValueError("pre_positions and post_positions must be 2D tensors")
    if pre.shape[1] != post.shape[1]:
        raise ValueError(
            "pre_positions and post_positions must use the same coordinate dimension"
        )

    distances = torch.cdist(post, pre)
    kernel_name = kernel.lower()
    if kernel_name == "gaussian":
        probs = torch.exp(-0.5 * (distances / sigma) ** 2)
    elif kernel_name == "exponential":
        probs = torch.exp(-distances / sigma)
    elif kernel_name == "none":
        probs = torch.ones_like(distances)
    else:
        raise ValueError(
            f"Unknown distance kernel '{kernel}'. "
            "Choose from: gaussian, exponential, none"
        )

    return (base_probability * probs).clamp(0.0, 1.0)


def sample_distance_kernel_mask(
    pre_positions: torch.Tensor,
    post_positions: torch.Tensor,
    *,
    kernel: str = "gaussian",
    sigma: float = 1.0,
    base_probability: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a mask using distance-dependent connection probabilities."""
    probs = distance_kernel_probabilities(
        pre_positions,
        post_positions,
        kernel=kernel,
        sigma=sigma,
        base_probability=base_probability,
    )
    return sample_probability_mask(
        out_features=probs.shape[0],
        in_features=probs.shape[1],
        probability=probs,
        generator=generator,
        device=probs.device,
    )


# Methods resolved as direct compact indices from an external manifest file
# (see ``index_manifest.py``).  Declared here — the import-graph root of the
# synapse package — so both the mask dispatch below and the indices dispatch
# in ``spatial_morphology.py`` agree on the names without an import cycle.
INDEX_MANIFEST_METHODS = frozenset({"index_manifest", "manifest"})


def _resolve_indexed(value, index: int | None):
    if isinstance(value, list):
        if not value:
            return None
        if index is None:
            return value[0]
        return value[index] if index < len(value) else value[-1]
    return value


def _level_feature_block_mask(
    *,
    out_features: int,
    in_features: int,
    ranges_by_level,
    level_idx: int,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Allow configured contiguous feature ranges at one dendritic level."""

    ranges = _resolve_indexed(ranges_by_level, level_idx)
    if ranges is None:
        raise ValueError("level_feature_blocks requires 'ranges_by_level'.")
    if (
        isinstance(ranges, (tuple, list))
        and len(ranges) == 2
        and all(isinstance(value, (int, float)) for value in ranges)
    ):
        ranges = [ranges]
    if not isinstance(ranges, (tuple, list)) or not ranges:
        raise ValueError(
            "Each ranges_by_level entry must contain at least one [start, stop] range."
        )

    allowed = torch.zeros(in_features, dtype=torch.bool, device=device)
    for bounds in ranges:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("Feature ranges must be [start, stop] pairs.")
        start, stop = int(bounds[0]), int(bounds[1])
        if start < 0 or stop > in_features or stop <= start:
            raise ValueError(
                "Feature ranges must satisfy 0 <= start < stop <= in_features; "
                f"got [{start}, {stop}] for in_features={in_features}."
            )
        allowed[start:stop] = True
    if not bool(allowed.any()):
        raise ValueError("level_feature_blocks must allow at least one feature.")
    return allowed.unsqueeze(0).expand(out_features, -1).clone()


def _inventory_feature_block_mask(
    *,
    out_features: int,
    in_features: int,
    inventory_counts,
    feature_ranges,
    tier_groups,
    owner_count: int,
    level_idx: int,
    n_levels: int | None,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Route a fixed branch inventory across the available dendritic stages."""

    counts = [int(value) for value in inventory_counts or []]
    ranges = list(feature_ranges or [])
    if not counts or len(counts) != len(ranges):
        raise ValueError(
            "inventory_feature_blocks requires equally sized nonempty "
            "'inventory_counts' and 'feature_ranges'."
        )
    if any(value <= 0 for value in counts):
        raise ValueError("inventory_counts entries must be positive.")
    if owner_count <= 0 or out_features % owner_count != 0:
        raise ValueError(
            "owner_count must be positive and divide out_features for "
            "inventory_feature_blocks."
        )
    if n_levels is None or n_levels < 2:
        raise ValueError(
            "inventory_feature_blocks requires n_levels including a soma level."
        )

    active_levels = n_levels - 1
    if active_levels > len(counts):
        raise ValueError(
            "inventory_feature_blocks requires at least one inventory tier per "
            "non-somatic level."
        )
    if level_idx >= active_levels:
        raise ValueError(
            "inventory_feature_blocks is intended for non-somatic levels; "
            "configure somatic_synapses=false."
        )
    if tier_groups is None:
        # Partition ordered inventory tiers over active stages. With tier
        # counts 4/2/2 this yields 8; 6+2; or 4+2+2 branches for one, two, or
        # three non-somatic stages, respectively.
        tier_start = round(level_idx * len(counts) / active_levels)
        tier_stop = round((level_idx + 1) * len(counts) / active_levels)
        level_tiers = list(range(tier_start, tier_stop))
    else:
        groups = [list(map(int, group)) for group in tier_groups]
        if len(groups) != active_levels:
            raise ValueError(
                "inventory_feature_blocks 'tier_groups' must contain one "
                "group per non-somatic level."
            )
        flattened = [tier for group in groups for tier in group]
        if flattened != list(range(len(counts))) or any(not group for group in groups):
            raise ValueError(
                "inventory_feature_blocks 'tier_groups' must be a nonempty, "
                "ordered partition of all inventory tier indices."
            )
        level_tiers = groups[level_idx]
    expected_per_owner = sum(counts[tier] for tier in level_tiers)
    rows_per_owner = out_features // owner_count
    if rows_per_owner != expected_per_owner:
        raise ValueError(
            "The morphology does not match the configured inventory at level "
            f"{level_idx}: expected {expected_per_owner} rows per owner, got "
            f"{rows_per_owner}."
        )

    per_owner_ranges = []
    for tier in level_tiers:
        bounds = ranges[tier]
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("feature_ranges entries must be [start, stop] pairs.")
        start, stop = int(bounds[0]), int(bounds[1])
        if start < 0 or stop > in_features or stop <= start:
            raise ValueError(
                "Feature ranges must satisfy 0 <= start < stop <= in_features; "
                f"got [{start}, {stop}] for in_features={in_features}."
            )
        per_owner_ranges.extend([(start, stop)] * counts[tier])

    mask = torch.zeros(out_features, in_features, dtype=torch.bool, device=device)
    for owner_idx in range(owner_count):
        row_offset = owner_idx * rows_per_owner
        for local_row, (start, stop) in enumerate(per_owner_ranges):
            mask[row_offset + local_row, start:stop] = True
    return mask


def _resolve_pathway_config(config, pathway: str) -> dict:
    """The rule of one pathway with canonical key names (``{}``: dense).

    See :func:`dendritic_modeling.config.structured_connectivity.resolve_pathway_rule`.
    """
    return resolve_pathway_rule(config, pathway)


def layer_seed_parts(layer_idx: int, network_layer_idx: int | None) -> tuple:
    """Seed components of a layer; a later population-network layer adds its index.

    Populations of a population network pass their position in the layer as
    ``layer_idx`` and the layer's index in the network as
    ``network_layer_idx``. The first network layer keeps the components of
    ``layer_idx`` alone, so its draws, and those of every EINet, are unchanged.
    """
    if not network_layer_idx:
        return (int(layer_idx),)
    return (int(layer_idx), "network_layer", int(network_layer_idx))


def sample_configured_mask(
    config,
    *,
    pathway: str,
    out_features: int,
    in_features: int,
    layer_idx: int = 0,
    level_idx: int = 0,
    n_levels: int | None = None,
    device: torch.device | str | None = None,
    network_layer_idx: int | None = None,
) -> torch.Tensor | None:
    """Sample a hard connectivity mask from a structured config block.

    Supported mask methods are ``"bernoulli"``/``"probability"``,
    ``"fixed_indegree"``, ``"distance"``, ``"level_feature_blocks"``, and
    ``"inventory_feature_blocks"``. ``"spatial_morphology"`` is handled
    separately as compact sparse indices and therefore returns no dense mask.
    Scalar or list-valued ``probability``/``indegree`` fields use repeat-last
    semantics by layer: the network layer ``network_layer_idx`` when given
    (population networks), otherwise ``layer_idx``.
    """
    list_idx = layer_idx if network_layer_idx is None else network_layer_idx
    cfg = _resolve_pathway_config(config, pathway)
    if not cfg:
        return None

    method = str(cfg.get("method", "bernoulli")).lower()
    if method in DENSE_METHODS:
        return None
    # Spatial morphology is represented directly as compact [out, K] indices.
    # Returning None here avoids materializing an ImageNet-scale Boolean mask.
    if method in {"spatial_morphology", "morphology_spatial"}:
        return None
    # External index manifests are likewise compact [out, K] indices, loaded
    # by ``sample_configured_indices``; they contribute no dense mask.
    if method in INDEX_MANIFEST_METHODS:
        return None

    if method in {"level_feature_blocks", "level_blocks"}:
        return _level_feature_block_mask(
            out_features=out_features,
            in_features=in_features,
            ranges_by_level=cfg.get("ranges_by_level", None),
            level_idx=level_idx,
            device=device,
        )
    if method in {"inventory_feature_blocks", "inventory_blocks"}:
        return _inventory_feature_block_mask(
            out_features=out_features,
            in_features=in_features,
            inventory_counts=cfg.get("inventory_counts", None),
            feature_ranges=cfg.get("feature_ranges", None),
            tier_groups=cfg.get("tier_groups", None),
            owner_count=int(cfg.get("owner_count", 0)),
            level_idx=level_idx,
            n_levels=n_levels,
            device=device,
        )

    seed = cfg.get("seed", None)
    generator = None
    # Without a seed, or with ``global_rng``, the mask draws from the global
    # random stream.
    if seed is not None and seed != GLOBAL_RNG_SEED:
        generator = torch.Generator()
        generator.manual_seed(
            (
                int(seed)
                + stable_seed_offset(
                    normalize_pathway_name(pathway),
                    *layer_seed_parts(layer_idx, network_layer_idx),
                    level_idx,
                    out_features,
                    in_features,
                )
            )
            % ((1 << 63) - 1)
        )

    if method in {"bernoulli", "probability", "probabilistic"}:
        probability = cfg.get("probability")
        if probability is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' requires "
                "'probability' for method='bernoulli'."
            )
        probability = _resolve_indexed(probability, list_idx)
        return sample_probability_mask(
            out_features,
            in_features,
            probability,
            generator=generator,
            device=device,
        )

    if method in {"fixed_indegree", "fixed_in_degree", "indegree"}:
        indegree = cfg.get("indegree")
        if indegree is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' requires "
                "'indegree' for method='fixed_indegree'."
            )
        indegree = int(_resolve_indexed(indegree, list_idx))
        return sample_fixed_indegree_mask(
            out_features,
            in_features,
            indegree,
            generator=generator,
            device=device,
        )

    if method in DISTANCE_METHODS:
        pre_positions = cfg.get("pre_positions", None)
        post_positions = cfg.get("post_positions", None)
        if pre_positions is None or post_positions is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' requires "
                "'pre_positions' and 'post_positions' for method='distance'."
            )
        probs = distance_kernel_probabilities(
            torch.as_tensor(pre_positions, dtype=torch.float32),
            torch.as_tensor(post_positions, dtype=torch.float32),
            kernel=str(cfg.get("distance_kernel", "gaussian")),
            sigma=float(cfg.get("distance_sigma", 1.0)),
            base_probability=distance_base_probability(
                cfg, section=f"structured connectivity pathway {pathway!r}"
            ),
        )
        return sample_probability_mask(
            out_features,
            in_features,
            probs,
            generator=generator,
            device=device,
        )

    raise ValueError(
        f"Unknown structured connectivity method '{method}'. "
        "Choose from: bernoulli, fixed_indegree, distance, "
        "level_feature_blocks, inventory_feature_blocks, none."
    )


__all__ = [
    "INDEX_MANIFEST_METHODS",
    "distance_kernel_probabilities",
    "normalize_pathway_name",
    "sample_configured_mask",
    "sample_distance_kernel_mask",
    "sample_fixed_indegree_mask",
    "sample_probability_mask",
    "validate_connection_mask",
]
