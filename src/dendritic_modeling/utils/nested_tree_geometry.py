"""Shared proximal→distal tree geometry for nested MultiXOR-style tasks."""

from __future__ import annotations


def normalize_branch_factors(
    branch_factors: list[int] | tuple[int, ...],
    *,
    min_factor: int = 2,
) -> tuple[int, ...]:
    """Validate proximal→distal branch factors and return a normalized tuple."""
    if not branch_factors:
        raise ValueError("branch_factors must be a non-empty sequence of integers")
    normalized = tuple(int(b) for b in branch_factors)
    if any(b < min_factor for b in normalized):
        raise ValueError(
            f"each branch factor must be >= {min_factor} (proximal→distal), "
            f"got {branch_factors}"
        )
    return normalized


def n_leaves_from_branch_factors(branch_factors: tuple[int, ...]) -> int:
    """Return ``prod(branch_factors)``."""
    n_leaves = 1
    for b in branch_factors:
        n_leaves *= int(b)
    return n_leaves


def dendritic_layer_sizes(branch_factors: tuple[int, ...]) -> list[int]:
    """Return root→distal layer sizes ``[1, b0, b0*b1, …, n_leaves]``."""
    sizes = [1]
    for b in branch_factors:
        sizes.append(sizes[-1] * int(b))
    return sizes


def n_internal_nodes_from_branch_factors(
    branch_factors: tuple[int, ...],
    auxiliary_root: bool,
) -> int:
    """Count internal aux nodes (optionally including the root)."""
    sizes = dendritic_layer_sizes(branch_factors)
    n_internal = sum(sizes[1:-1])
    if auxiliary_root:
        n_internal += 1
    return int(n_internal)


def n_blocks_from_branch_factors(
    branch_factors: tuple[int, ...],
    *,
    auxiliary_root: bool = False,
) -> int:
    """Total population blocks: leaves + internal aux (+ optional root aux)."""
    return n_leaves_from_branch_factors(branch_factors) + (
        n_internal_nodes_from_branch_factors(
            branch_factors, auxiliary_root=auxiliary_root
        )
    )


def task_block_index_for_node(
    branch_factors: tuple[int, ...],
    *,
    depth_from_root: int,
    index_in_layer: int,
    auxiliary_root: bool = False,
) -> int | None:
    """Map a tree node ``(depth, index)`` to its contiguous input block index.

    Layout: leaf blocks ``0 … n_leaves-1``, then aux blocks distal→proximal
    (root aux last when ``auxiliary_root``). Root with ``auxiliary_root=False``
    returns ``None`` (no dedicated block).
    """
    sizes = dendritic_layer_sizes(branch_factors)
    depth = int(depth_from_root)
    idx = int(index_in_layer)
    if depth < 0 or depth >= len(sizes):
        raise ValueError(
            f"depth_from_root={depth} out of range for branch_factors={branch_factors}"
        )
    if idx < 0 or idx >= sizes[depth]:
        raise ValueError(
            f"index_in_layer={idx} out of range for depth {depth} "
            f"(layer size {sizes[depth]})"
        )

    n_leaves = sizes[-1]
    leaf_depth = len(sizes) - 1
    if depth == leaf_depth:
        return idx

    if depth == 0:
        if not auxiliary_root:
            return None
        return n_leaves + n_internal_nodes_from_branch_factors(
            branch_factors, auxiliary_root=True
        ) - 1

    # Aux section: levels leaf_depth-1, …, 1 (, 0) in that order.
    offset = n_leaves
    for level in range(leaf_depth - 1, depth, -1):
        offset += sizes[level]
    return offset + idx


def task_block_ranges(
    branch_factors: tuple[int, ...] | list[int],
    dims_per_leaf: int,
    *,
    auxiliary_root: bool = False,
) -> list[slice]:
    """Return column slices for each task block in input order."""
    bf = tuple(int(b) for b in branch_factors)
    n_blocks = n_blocks_from_branch_factors(bf, auxiliary_root=auxiliary_root)
    d = int(dims_per_leaf)
    if d < 1:
        raise ValueError(f"dims_per_leaf must be >= 1, got {dims_per_leaf}")
    return [slice(i * d, (i + 1) * d) for i in range(n_blocks)]


def blocks_at_depth(
    branch_factors: tuple[int, ...],
    depth_from_root: int,
    *,
    auxiliary_root: bool = False,
) -> list[int]:
    """Block indices for all nodes at ``depth_from_root`` (skip root if no aux)."""
    sizes = dendritic_layer_sizes(branch_factors)
    depth = int(depth_from_root)
    out: list[int] = []
    for i in range(sizes[depth]):
        block = task_block_index_for_node(
            branch_factors,
            depth_from_root=depth,
            index_in_layer=i,
            auxiliary_root=auxiliary_root,
        )
        if block is not None:
            out.append(block)
    return out


__all__ = [
    "blocks_at_depth",
    "dendritic_layer_sizes",
    "n_blocks_from_branch_factors",
    "n_internal_nodes_from_branch_factors",
    "n_leaves_from_branch_factors",
    "normalize_branch_factors",
    "task_block_index_for_node",
    "task_block_ranges",
]
