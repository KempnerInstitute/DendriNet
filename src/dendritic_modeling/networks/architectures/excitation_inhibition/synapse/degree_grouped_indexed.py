"""Fixed-index sparse projections with an exact per-output degree vector."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    inverse_weight_transform,
)


class _IndexedDegreeGroup(nn.Module):
    """One uniform-degree subset used by :class:`DegreeGroupedIndexedLinear`."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_indices: torch.Tensor,
        connection_indices: torch.Tensor,
        degree: int,
        weight_transform: WeightTransformType,
        row_sum_init: float | None,
        output_chunk_size: int,
        index_dtype: str | torch.dtype,
        workspace_mb: float | None,
        cache_transformed_weights: bool,
        recompute_backward: bool,
        persistent_indices: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("output_indices", output_indices.to(torch.long))
        self.projection = IndexedSparseLinear(
            in_features=input_dim,
            out_features=int(output_indices.numel()),
            K=degree,
            connection_indices=connection_indices,
            init_method="xavier_normal",
            weight_transform=weight_transform,
            output_chunk_size=output_chunk_size,
            index_dtype=index_dtype,
            workspace_mb=workspace_mb,
            cache_transformed_weights=cache_transformed_weights,
            recompute_backward=recompute_backward,
            persistent_indices=persistent_indices,
        )
        if row_sum_init is not None:
            per_contact = float(row_sum_init) / float(degree)
            if per_contact <= 0.0 and weight_transform != "identity":
                raise ValueError(
                    "row_sum_init must be positive for a positive weight transform"
                )
            raw_value = inverse_weight_transform(
                torch.tensor(
                    per_contact,
                    dtype=self.projection.pre_w.dtype,
                    device=self.projection.pre_w.device,
                ),
                weight_transform,
            )
            with torch.no_grad():
                self.projection.pre_w.fill_(float(raw_value.item()))


def _validate_degree_vector(
    degrees: Sequence[int] | torch.Tensor,
    *,
    input_dim: int,
    output_dim: int,
) -> torch.Tensor:
    try:
        raw_degrees = torch.as_tensor(degrees)
    except (TypeError, ValueError) as exc:
        raise TypeError("degrees must contain integer values") from exc
    if raw_degrees.ndim != 1 or raw_degrees.numel() != output_dim:
        raise ValueError(
            f"degrees must have shape ({output_dim},), got {tuple(raw_degrees.shape)}"
        )
    if isinstance(degrees, torch.Tensor):
        integral_dtype = degrees.dtype != torch.bool and not (
            degrees.is_floating_point() or degrees.is_complex()
        )
        if not integral_dtype:
            raise TypeError("degrees must contain integer values")
    elif any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in degrees
    ):
        raise TypeError("degrees must contain integer values")
    degree_tensor = raw_degrees.to(dtype=torch.long)
    if bool((degree_tensor < 0).any()):
        raise ValueError("degrees must be non-negative")
    if bool((degree_tensor > input_dim).any()):
        raise ValueError(f"degrees cannot exceed the input dimension ({input_dim})")
    return degree_tensor.contiguous()


def _balanced_connection_rows(
    degrees: torch.Tensor,
    *,
    input_dim: int,
    seed: int,
) -> list[torch.Tensor]:
    """Build unique row supports with optimally balanced source usage.

    A seeded source permutation is traversed cyclically across all requested
    slots.  Each row contains at most ``input_dim`` consecutive permutation
    entries and therefore has no duplicate.  Globally, source outdegrees differ
    by at most one; all sources are covered whenever the contact budget permits.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    source_order = torch.randperm(input_dim, generator=generator)
    start = int(torch.randint(input_dim, (1,), generator=generator).item())
    cursor = start
    rows: list[torch.Tensor] = []
    for degree in degrees.tolist():
        if degree == 0:
            rows.append(torch.empty(0, dtype=torch.long))
            continue
        positions = (torch.arange(degree, dtype=torch.long) + cursor) % input_dim
        rows.append(source_order[positions])
        cursor += degree
    return rows


class DegreeGroupedIndexedLinear(nn.Module):
    """Sparse linear map with exact, potentially heterogeneous row degrees.

    The implementation composes the existing fixed-index sparse kernel by
    grouping output rows that share a degree.  It supports zero-degree rows,
    exact total contact budgets, deterministic topology, and guaranteed source
    coverage whenever the total number of contacts is at least ``input_dim``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degrees: Sequence[int] | torch.Tensor,
        *,
        seed: int,
        require_source_coverage: bool = False,
        weight_transform: WeightTransformType = "exp",
        row_sum_init: float | None = None,
        output_chunk_size: int = 2048,
        index_dtype: str | torch.dtype = "int64",
        workspace_mb: float | None = None,
        cache_transformed_weights: bool = False,
        recompute_backward: bool = False,
        persistent_indices: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or int(input_dim) != input_dim or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if (
            isinstance(output_dim, bool)
            or int(output_dim) != output_dim
            or output_dim < 1
        ):
            raise ValueError("output_dim must be a positive integer")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight_transform = weight_transform
        degree_tensor = _validate_degree_vector(
            degrees,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        total_contacts = int(degree_tensor.sum().item())
        if require_source_coverage and total_contacts < self.input_dim:
            raise ValueError(
                "source coverage requires at least input_dim total contacts; "
                f"got {total_contacts} < {self.input_dim}"
            )
        self.require_source_coverage = bool(require_source_coverage)
        self.register_buffer("degrees", degree_tensor)

        connection_rows = _balanced_connection_rows(
            degree_tensor,
            input_dim=self.input_dim,
            seed=int(seed),
        )
        self.groups = nn.ModuleList()
        for degree in sorted(set(degree_tensor.tolist())):
            if degree == 0:
                continue
            output_indices = torch.nonzero(
                degree_tensor == degree,
                as_tuple=False,
            ).flatten()
            connection_indices = torch.stack(
                [connection_rows[index] for index in output_indices.tolist()]
            )
            self.groups.append(
                _IndexedDegreeGroup(
                    input_dim=self.input_dim,
                    output_indices=output_indices,
                    connection_indices=connection_indices,
                    degree=degree,
                    weight_transform=weight_transform,
                    row_sum_init=row_sum_init,
                    output_chunk_size=output_chunk_size,
                    index_dtype=index_dtype,
                    workspace_mb=workspace_mb,
                    cache_transformed_weights=cache_transformed_weights,
                    recompute_backward=recompute_backward,
                    persistent_indices=persistent_indices,
                )
            )

        if self.require_source_coverage and bool((self.source_outdegrees() == 0).any()):
            raise RuntimeError("failed to construct the required source coverage")

    @property
    def total_contacts(self) -> int:
        return int(self.degrees.sum().item())

    def connection_indices_padded(self) -> torch.Tensor:
        """Return ``[output_dim, max_degree]`` indices with ``-1`` padding."""

        max_degree = int(self.degrees.max().item())
        padded = torch.full(
            (self.output_dim, max_degree),
            -1,
            dtype=torch.long,
            device=self.degrees.device,
        )
        for group in self.groups:
            degree = group.projection.K
            padded[group.output_indices, :degree] = (
                group.projection.connection_indices.to(padded.device)
            )
        return padded

    def source_outdegrees(self) -> torch.Tensor:
        """Count how often each source coordinate appears in the fixed support."""

        counts = torch.zeros(
            self.input_dim,
            dtype=torch.long,
            device=self.degrees.device,
        )
        for group in self.groups:
            indices = group.projection.connection_indices.to(counts.device).flatten()
            counts = counts + torch.bincount(indices, minlength=self.input_dim)
        return counts

    def connectivity_resource_counts(self) -> dict[str, int | float | str]:
        """Return exact compact connectivity counts for resource ledgers."""

        return {
            "out_features": self.output_dim,
            "in_features": self.input_dim,
            "candidate_parameters": self.total_contacts,
            "candidate_slots": self.output_dim * self.input_dim,
            "active_synapses": self.total_contacts,
            "realized_k_min": int(self.degrees.min().item()),
            "realized_k_max": int(self.degrees.max().item()),
            "realized_k_mean": float(self.degrees.float().mean().item()),
            "selection_policy": "indexed_fixed_degree_grouped",
            "mask_source": "seeded_balanced_indices",
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")
        if x.ndim < 1 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"input must end in dimension {self.input_dim}, got {tuple(x.shape)}"
            )
        output = x.new_zeros(*x.shape[:-1], self.output_dim)
        for group in self.groups:
            values = group.projection(x)
            output = torch.index_copy(
                output,
                -1,
                group.output_indices.to(output.device),
                values,
            )
        return output


__all__ = ["DegreeGroupedIndexedLinear"]
