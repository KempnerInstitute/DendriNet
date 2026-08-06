"""Utilities for structured synaptic connectivity masks."""

from __future__ import annotations

import torch

from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping
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


_PATHWAY_ALIASES = {
    "ee": "ee",
    "e_to_e": "ee",
    "exc_to_exc": "ee",
    "excitatory_to_excitatory": "ee",
    "ie": "ie",
    "i_to_e": "ie",
    "inh_to_exc": "ie",
    "inhibitory_to_excitatory": "ie",
    "ei": "ei",
    "e_to_i": "ei",
    "exc_to_inh": "ei",
    "excitatory_to_inhibitory": "ei",
    "ii": "ii",
    "i_to_i": "ii",
    "inh_to_inh": "ii",
    "inhibitory_to_inhibitory": "ii",
    "rec_ee": "rec_ee",
    "recurrent_ee": "rec_ee",
    "rec_e_to_e": "rec_ee",
    "rec_ie": "rec_ie",
    "recurrent_ie": "rec_ie",
    "rec_i_to_e": "rec_ie",
    "rec_ei": "rec_ei",
    "recurrent_ei": "rec_ei",
    "rec_e_to_i": "rec_ei",
    "rec_ii": "rec_ii",
    "recurrent_ii": "rec_ii",
    "rec_i_to_i": "rec_ii",
}


def normalize_pathway_name(pathway: str) -> str:
    """Normalize public pathway aliases to canonical E/I pathway names."""
    normalized = str(pathway).lower()
    return _PATHWAY_ALIASES.get(normalized, normalized)


def _resolve_indexed(value, index: int | None):
    if isinstance(value, list):
        if not value:
            return None
        if index is None:
            return value[0]
        return value[index] if index < len(value) else value[-1]
    return value


def _resolve_pathway_config(config, pathway: str) -> dict:
    root = _to_plain_mapping(config)
    if not bool(root.get("enabled", False)):
        return {}

    root_defaults = {
        key: value for key, value in root.items() if key not in {"pathways", "enabled"}
    }
    pathways = _to_plain_mapping(root.get("pathways", {}))
    canonical_pathway = normalize_pathway_name(pathway)

    pathway_cfg = {}
    for key, value in pathways.items():
        if normalize_pathway_name(key) == canonical_pathway:
            pathway_cfg = _to_plain_mapping(value)
            break

    # A typed StructuredConnectivityConfig carries default fields such as
    # method="bernoulli" even when the user only configured selected pathways.
    # Treat global defaults as a real root-level rule only when they include
    # the parameters needed to sample a mask.
    root_has_rule = any(
        key in root_defaults and root_defaults.get(key) is not None
        for key in (
            "probability",
            "p_connect",
            "indegree",
            "in_degree",
            "pre_positions",
            "post_positions",
        )
    )
    if not pathway_cfg and not root_has_rule:
        return {}

    merged = dict(root_defaults)
    merged.update(pathway_cfg)
    return merged


def sample_configured_mask(
    config,
    *,
    pathway: str,
    out_features: int,
    in_features: int,
    layer_idx: int = 0,
    level_idx: int = 0,
    device: torch.device | str | None = None,
    n_soma: int | None = None,
    branch_factors=None,
) -> torch.Tensor | None:
    """Sample a hard connectivity mask from a structured config block.

    Supported methods are ``"bernoulli"``/``"probability"``,
    ``"fixed_indegree"``, ``"distance"``, and ``"nested_multixor_blocks"``.
    Scalar or list-valued ``probability``/``indegree`` fields use repeat-last
    semantics by layer. ``nested_multixor_blocks`` requires ``n_soma`` and
    ``branch_factors`` from the DendriNet build scope.
    """
    cfg = _resolve_pathway_config(config, pathway)
    if not cfg:
        return None

    method = str(cfg.get("method", "bernoulli")).lower()
    if method in {"none", "dense", "all"}:
        return None

    seed = cfg.get("seed", None)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(
            (
                int(seed)
                + stable_seed_offset(
                    normalize_pathway_name(pathway),
                    layer_idx,
                    level_idx,
                    out_features,
                    in_features,
                )
            )
            % ((1 << 63) - 1)
        )

    if method in {"bernoulli", "probability", "probabilistic"}:
        probability = cfg.get("probability", cfg.get("p_connect", None))
        if probability is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' requires "
                "'probability' for method='bernoulli'."
            )
        probability = _resolve_indexed(probability, layer_idx)
        return sample_probability_mask(
            out_features,
            in_features,
            probability,
            generator=generator,
            device=device,
        )

    if method in {"fixed_indegree", "fixed_in_degree", "indegree"}:
        indegree = cfg.get("indegree", cfg.get("in_degree", None))
        if indegree is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' requires "
                "'indegree' for method='fixed_indegree'."
            )
        indegree = int(_resolve_indexed(indegree, layer_idx))
        return sample_fixed_indegree_mask(
            out_features,
            in_features,
            indegree,
            generator=generator,
            device=device,
        )

    if method in {"distance", "distance_kernel"}:
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
            kernel=str(cfg.get("distance_kernel", cfg.get("kernel", "gaussian"))),
            sigma=float(cfg.get("distance_sigma", cfg.get("sigma", 1.0))),
            base_probability=float(
                cfg.get("base_probability", cfg.get("probability", 1.0))
            ),
        )
        return sample_probability_mask(
            out_features,
            in_features,
            probs,
            generator=generator,
            device=device,
        )

    if method in {"nested_multixor_blocks", "nested_multixor", "multixor_blocks"}:
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.nested_multixor_mask import (
            build_nested_multixor_connection_mask,
        )

        if n_soma is None or branch_factors is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' method "
                "'nested_multixor_blocks' requires n_soma and branch_factors "
                "from the DendriNet build scope."
            )
        task_branch_factors = cfg.get("task_branch_factors", None)
        if task_branch_factors is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' requires "
                "'task_branch_factors' for method='nested_multixor_blocks'."
            )
        dims_per_leaf = cfg.get("dims_per_leaf", None)
        if dims_per_leaf is None:
            raise ValueError(
                f"Structured connectivity pathway '{pathway}' requires "
                "'dims_per_leaf' for method='nested_multixor_blocks'."
            )
        return build_nested_multixor_connection_mask(
            out_features=out_features,
            in_features=in_features,
            level_idx=level_idx,
            n_soma=int(n_soma),
            branch_factors=branch_factors,
            task_branch_factors=task_branch_factors,
            dims_per_leaf=int(dims_per_leaf),
            auxiliary_root=bool(cfg.get("auxiliary_root", False)),
            include_internal_latents=bool(
                cfg.get("include_internal_latents", False)
            ),
            device=device,
        )

    raise ValueError(
        f"Unknown structured connectivity method '{method}'. "
        "Choose from: bernoulli, fixed_indegree, distance, "
        "nested_multixor_blocks, none."
    )


__all__ = [
    "distance_kernel_probabilities",
    "normalize_pathway_name",
    "sample_configured_mask",
    "sample_distance_kernel_mask",
    "sample_fixed_indegree_mask",
    "sample_probability_mask",
    "validate_connection_mask",
]
