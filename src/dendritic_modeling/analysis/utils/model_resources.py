"""Exact model-resource accounting for confirmatory experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import torch

from dendritic_modeling.analysis.utils.dendritic_depth import (
    soma_relative_dendritic_depth,
)
from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)

_SYNAPSE_PATHWAYS = {
    "branch_excitation": "ff_excitatory",
    "branch_inhibition": "ff_inhibitory",
    "branch_recurrent": "rec_excitatory",
    "branch_rec_inhibition": "rec_inhibitory",
}


@dataclass(frozen=True)
class SynapseResourceRecord:
    module: str
    pathway: str
    soma_relative_depth: int
    out_features: int
    in_features: int
    candidate_parameters: int
    candidate_slots: int
    active_synapses: int
    realized_k_min: int
    realized_k_max: int
    realized_k_mean: float
    selection_policy: str
    mask_source: str


@dataclass(frozen=True)
class StaticDendriticResourceSummary:
    """Config-derived resources for a regular dendritic population.

    This summary is intentionally limited to quantities that are determined
    exactly by morphology, pathway input widths, and retained contacts.  It
    does not infer trainable parameter counts for a particular sparse-layer
    implementation; checkpoint-backed analyses should use
    :func:`model_resource_summary` for those quantities.
    """

    level_compartments_per_soma: tuple[int, ...]
    nonsomatic_compartments_per_soma: int
    nonsomatic_compartments_total: int
    synaptic_compartments_total: int
    active_synapses_by_pathway: dict[str, int]
    active_synapses_total: int
    candidate_slots_by_pathway: dict[str, int]
    candidate_synapse_slots_total: int
    child_couplings_total: int
    reactivation_parameters_total: int


def _resource_integer(value: Any, *, name: str, allow_zero: bool) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be an integer, got {value!r}") from None
    if parsed != value:
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}, got {parsed}")
    return parsed


def dendritic_level_widths(branch_factors: Sequence[int]) -> tuple[int, ...]:
    """Return nonsomatic compartment counts per soma from proximal to distal."""

    if isinstance(branch_factors, (str, bytes)):
        raise TypeError("branch_factors must be a sequence of positive integers")
    factors = tuple(
        _resource_integer(value, name=f"branch_factors[{index}]", allow_zero=False)
        for index, value in enumerate(branch_factors)
    )
    if not factors:
        raise ValueError("branch_factors must contain at least one dendritic level")
    level_widths = []
    width = 1
    for factor in factors:
        width *= factor
        level_widths.append(width)
    return tuple(level_widths)


def static_dendritic_resource_summary(
    *,
    branch_factors: Sequence[int],
    n_somas: int,
    pathway_input_dims: Mapping[str, int],
    synapses_per_compartment: Mapping[str, int],
    somatic_synapses: bool = False,
    reactivation_scalars_per_compartment: int = 2,
    reactivation_includes_soma: bool = True,
) -> StaticDendriticResourceSummary:
    """Compute exact config-level resources for one regular population.

    ``candidate_slots`` counts structurally available input/output pairs, not
    necessarily trainable parameters.  For standard dense-candidate Top-K the
    two counts coincide; indexed and structured implementations can differ.
    """

    level_widths = dendritic_level_widths(branch_factors)
    soma_count = _resource_integer(n_somas, name="n_somas", allow_zero=False)
    reactivation_width = _resource_integer(
        reactivation_scalars_per_compartment,
        name="reactivation_scalars_per_compartment",
        allow_zero=True,
    )

    input_dims = {
        str(pathway): _resource_integer(
            value,
            name=f"pathway_input_dims[{pathway!r}]",
            allow_zero=True,
        )
        for pathway, value in pathway_input_dims.items()
    }
    retained = {
        str(pathway): _resource_integer(
            value,
            name=f"synapses_per_compartment[{pathway!r}]",
            allow_zero=True,
        )
        for pathway, value in synapses_per_compartment.items()
    }
    if not input_dims:
        raise ValueError("At least one synaptic pathway must be specified")
    if set(input_dims) != set(retained):
        raise ValueError(
            "pathway_input_dims and synapses_per_compartment must have identical keys"
        )
    for pathway in sorted(input_dims):
        if retained[pathway] > input_dims[pathway]:
            raise ValueError(
                f"Retained contacts for {pathway!r} exceed its input width: "
                f"{retained[pathway]} > {input_dims[pathway]}"
            )

    nonsomatic_per_soma = sum(level_widths)
    nonsomatic_total = nonsomatic_per_soma * soma_count
    synaptic_compartments = nonsomatic_total + (
        soma_count if bool(somatic_synapses) else 0
    )
    active_by_pathway = {
        pathway: synaptic_compartments * count for pathway, count in retained.items()
    }
    candidate_by_pathway = {
        pathway: synaptic_compartments * input_dims[pathway] for pathway in input_dims
    }
    reactivation_compartments = nonsomatic_total + (
        soma_count if bool(reactivation_includes_soma) else 0
    )
    return StaticDendriticResourceSummary(
        level_compartments_per_soma=level_widths,
        nonsomatic_compartments_per_soma=nonsomatic_per_soma,
        nonsomatic_compartments_total=nonsomatic_total,
        synaptic_compartments_total=synaptic_compartments,
        active_synapses_by_pathway=active_by_pathway,
        active_synapses_total=sum(active_by_pathway.values()),
        candidate_slots_by_pathway=candidate_by_pathway,
        candidate_synapse_slots_total=sum(candidate_by_pathway.values()),
        child_couplings_total=nonsomatic_total,
        reactivation_parameters_total=(reactivation_compartments * reactivation_width),
    )


def _indexed_record_counts(synapse: torch.nn.Module):
    """Snapshot-equivalent counts for indexed-sparse layers, without the snapshot.

    ``effective_synapse_snapshot`` was found -- by an allocator-trace replay of a
    full trainer run -- to dominate the PROCESS peak of training this repo's
    stress configs: for an indexed layer, its ``weight()``/``weight_mask()``
    calls each materialize a dense ``[out_features, in_features]`` matrix, and
    the snapshot holds ~5 of them at once (~45 GB at 144,000 x 16,000) to
    produce what the resource record actually consumes: five integers. That one
    reporting call was 6x the training step's own peak.

    Every count the record needs is index-arithmetic, independent of weight
    values, so this computes the SAME scatters the dense path performs -- in
    bounded row chunks -- rather than re-deriving the statistics analytically
    (which would have to re-prove semantics for duplicate indices, connection
    masks, and forbidden entries). Bit-exact by construction; verified against
    the snapshot path by check_resource_records_equiv.py.

    Returns ``None`` whenever the legacy path's richer semantics could apply
    (cached forward mask, a candidate_mask() method, stochastic selection, or
    ``DENDRINET_DENSE_RESOURCE_RECORDS=1``), so behaviour changes only where
    the two paths are provably identical.
    """
    import os

    from dendritic_modeling.analysis.utils.effective_synapses import (
        _selection_is_stochastic,
    )

    if os.environ.get("DENDRINET_DENSE_RESOURCE_RECORDS", "0") == "1":
        return None
    idx = getattr(synapse, "connection_indices", None)
    if not isinstance(idx, torch.Tensor) or idx.dim() != 2:
        return None
    if isinstance(getattr(synapse, "_last_forward_weight_mask", None), torch.Tensor):
        return None                       # snapshot would use the cached mask
    if callable(getattr(synapse, "candidate_mask", None)):
        return None                       # snapshot would call it; do not guess
    if _selection_is_stochastic(synapse):
        return None                       # legacy path raises the right error

    out_features = int(synapse.out_features)
    in_features = int(synapse.in_features)
    device = idx.device

    # active mask == scatter ones at indices, clamp, count -- exactly
    # weight_mask().clamp(0,1).ne(0), i.e. UNIQUE indices per row, chunked so no
    # more than ~256 MB of dense rows ever exists.
    realized_k = torch.empty(out_features, dtype=torch.long)
    rows_per_chunk = max(1, (256 * 2**20) // max(in_features * 4, 1))
    ones = None
    for start in range(0, out_features, rows_per_chunk):
        end = min(start + rows_per_chunk, out_features)
        block = torch.zeros(end - start, in_features, device=device)
        if ones is None or ones.shape[0] != end - start:
            ones = torch.ones(end - start, idx.shape[1], device=device)
        block.scatter_add_(1, idx[start:end].to(device=device, dtype=torch.long), ones)
        realized_k[start:end] = block.clamp_max(1).ne(0).sum(dim=-1).cpu()
    active_synapses = int(realized_k.sum().item())

    # candidate mask == ones * connection_mask with forbidden entries zeroed.
    connection_mask = getattr(synapse, "connection_mask", None)
    forbidden = getattr(synapse, "_forbidden_input_index_per_output", None)
    has_forbidden = isinstance(forbidden, torch.Tensor) and forbidden.numel() > 0
    if isinstance(connection_mask, torch.Tensor) and connection_mask.numel() > 0:
        candidate = int(connection_mask.ne(0).sum().item())
        if has_forbidden:
            valid = forbidden >= 0
            if bool(valid.any()):
                rows = torch.arange(connection_mask.shape[0],
                                    device=connection_mask.device)[valid.to(connection_mask.device)]
                cols = forbidden.to(connection_mask.device)[valid.to(connection_mask.device)]
                candidate -= int(connection_mask[rows, cols].ne(0).sum().item())
    else:
        candidate = out_features * in_features
        if has_forbidden:
            candidate -= int((forbidden >= 0).sum().item())

    rk = realized_k.to(dtype=torch.float32)
    return {
        "candidate_slots": candidate,
        "active_synapses": active_synapses,
        "realized_k_min": int(rk.min().item()),
        "realized_k_max": int(rk.max().item()),
        "realized_k_mean": float(rk.mean().item()),
        "selection_policy": str(getattr(synapse, "selection",
                                        type(synapse).__name__)).lower(),
        "mask_source": "deterministic_current",
    }


def synapse_resource_records(model: torch.nn.Module) -> list[SynapseResourceRecord]:
    """Return one realized-connectivity record per dendritic pathway."""

    records: list[SynapseResourceRecord] = []
    seen: set[int] = set()
    for branch_name, branch_layer in model.named_modules():
        if not isinstance(branch_layer, DendriticBranchLayer):
            continue
        depth = soma_relative_dendritic_depth(branch_layer)
        for attr, pathway in _SYNAPSE_PATHWAYS.items():
            synapse = getattr(branch_layer, attr, None)
            if synapse is None or id(synapse) in seen:
                continue
            seen.add(id(synapse))
            common = dict(
                module=f"{branch_name}.{attr}",
                pathway=pathway,
                soma_relative_depth=int(depth),
                candidate_parameters=int(
                    sum(parameter.numel() for parameter in synapse.parameters())
                ),
            )
            fast = _indexed_record_counts(synapse)
            if fast is not None:
                records.append(
                    SynapseResourceRecord(
                        out_features=int(synapse.out_features),
                        in_features=int(synapse.in_features),
                        **common,
                        **fast,
                    )
                )
                continue
            snapshot = effective_synapse_snapshot(synapse)
            realized_k = snapshot.realized_k.to(dtype=torch.float32)
            records.append(
                SynapseResourceRecord(
                    out_features=int(snapshot.active_mask.shape[0]),
                    in_features=int(snapshot.active_mask.shape[1]),
                    **common,
                    candidate_slots=int(snapshot.candidate_mask.ne(0).sum().item()),
                    active_synapses=int(snapshot.active_mask.ne(0).sum().item()),
                    realized_k_min=int(realized_k.min().item()),
                    realized_k_max=int(realized_k.max().item()),
                    realized_k_mean=float(realized_k.mean().item()),
                    selection_policy=snapshot.selection_policy,
                    mask_source=snapshot.mask_source,
                )
            )
    return records


def _tensor_tree_numel(value: Any, seen: set[int]) -> int:
    if isinstance(value, torch.Tensor):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return int(value.numel())
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _tensor_tree_numel(getattr(value, field.name), seen)
            for field in fields(value)
        )
    if isinstance(value, dict):
        return sum(_tensor_tree_numel(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_numel(item, seen) for item in value)
    return 0


def persistent_state_scalars(core: torch.nn.Module) -> int:
    """Count recurrent state scalars retained per sample between timesteps."""

    init_state = getattr(core, "init_state", None)
    if callable(init_state):
        state = init_state(
            batch_size=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        return _tensor_tree_numel(state, set())

    config = getattr(core, "config", None)
    cell_type = str(getattr(config, "cell_type", "")).lower()
    hidden_dim = int(getattr(config, "hidden_dim", 0) or 0)
    num_layers = int(getattr(config, "num_layers", 1) or 1)
    if cell_type in {"vanilla", "gru"}:
        return hidden_dim * num_layers
    if cell_type == "lstm":
        return 2 * hidden_dim * num_layers
    return 0


def model_resource_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Return auditable parameter, connectivity, and state accounting."""

    records = synapse_resource_records(model)
    total_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameters = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    synapse_candidate_parameters = int(
        sum(record.candidate_parameters for record in records)
    )
    core = getattr(model, "core_network", model)
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "synapse_candidate_parameters": synapse_candidate_parameters,
        "non_synaptic_parameters": total_parameters - synapse_candidate_parameters,
        "candidate_synapse_slots": int(
            sum(record.candidate_slots for record in records)
        ),
        "active_synapses": int(sum(record.active_synapses for record in records)),
        "persistent_state_scalars_per_sample": persistent_state_scalars(core),
        "synapses": [asdict(record) for record in records],
    }


__all__ = [
    "StaticDendriticResourceSummary",
    "SynapseResourceRecord",
    "dendritic_level_widths",
    "model_resource_summary",
    "persistent_state_scalars",
    "static_dendritic_resource_summary",
    "synapse_resource_records",
]
