"""Nested MultiXOR task-aligned hard connectivity masks."""

from __future__ import annotations

import torch

from dendritic_modeling.utils.nested_tree_geometry import (
    blocks_at_depth,
    dendritic_layer_sizes,
    descendant_leaf_indices,
    effective_auxiliary_root,
    n_blocks_from_branch_factors,
    task_block_index_for_node,
    task_block_ranges,
)


def _as_bf_tuple(branch_factors) -> tuple[int, ...]:
    return tuple(int(b) for b in branch_factors)


def _distribute_children(n_model_children: int, n_task_children: int) -> list[int]:
    """Map each model child index to a task-child index.

    Ensures every task child is used at least once when
    ``n_model_children >= n_task_children``. When the model has fewer children,
    each model child still gets one task child (coverage of leftover task
    children is handled by assigning multiple blocks where needed).
    """
    if n_model_children < 1 or n_task_children < 1:
        raise ValueError(
            f"n_model_children and n_task_children must be >= 1, got "
            f"{n_model_children}, {n_task_children}"
        )
    if n_model_children >= n_task_children:
        # Spread so each task child appears ~equally often.
        return [i * n_task_children // n_model_children for i in range(n_model_children)]
    return list(range(n_model_children))


def _extra_task_children_for_coverage(
    n_model_children: int, n_task_children: int
) -> list[list[int]]:
    """When model has fewer children than task, leftover task children per model child."""
    extras: list[list[int]] = [[] for _ in range(n_model_children)]
    if n_model_children >= n_task_children:
        return extras
    for task_i in range(n_model_children, n_task_children):
        extras[task_i % n_model_children].append(task_i)
    return extras


def _blocks_for_task_node(
    task_bf: tuple[int, ...],
    *,
    depth_from_root: int,
    index_in_layer: int,
    auxiliary_root: bool,
    include_internal_latents: bool,
) -> list[int]:
    """Resolve allowed input blocks for a task node under the active latent mode."""
    if include_internal_latents:
        block = task_block_index_for_node(
            task_bf,
            depth_from_root=depth_from_root,
            index_in_layer=index_in_layer,
            auxiliary_root=auxiliary_root,
            include_internal_latents=True,
        )
        if block is None:
            raise RuntimeError(
                f"expected task block at depth={depth_from_root} "
                f"index={index_in_layer} for {task_bf}"
            )
        return [block]
    # Leaves-only: dedicated leaf block, or descendant leaf cover for internals.
    return descendant_leaf_indices(
        task_bf,
        depth_from_root=depth_from_root,
        index_in_layer=index_in_layer,
    )


def assign_compartment_blocks_per_soma(
    *,
    model_branch_factors: tuple[int, ...] | list[int],
    task_branch_factors: tuple[int, ...] | list[int],
    auxiliary_root: bool = False,
    include_internal_latents: bool = False,
) -> list[list[list[int]]]:
    """Assign task block indices to compartments for one soma.

    Returns ``assignments[level_from_distal][compartment_in_level] -> [block, ...]``.
    Distal-first level order matches DendriNet ``level_idx`` (leaves at 0).
    A final soma level is always included (root aux or empty list when absent).

    When ``include_internal_latents=False``, non-leaf compartments receive the
    union of descendant leaf blocks (subtree leaf cover). Soma stays empty
    unless root aux is effective.
    """
    model_bf = _as_bf_tuple(model_branch_factors)
    task_bf = _as_bf_tuple(task_branch_factors)
    include_internal_latents = bool(include_internal_latents)
    use_root_aux = effective_auxiliary_root(
        include_internal_latents=include_internal_latents,
        auxiliary_root=auxiliary_root,
    )
    if any(b < 1 for b in model_bf):
        raise ValueError(f"model branch_factors must be >= 1, got {model_bf}")
    if any(b < 1 for b in task_bf):
        raise ValueError(f"task branch_factors must be >= 1, got {task_bf}")

    # Flat / single-factor network: cover all task blocks on the one compartment layer.
    if len(model_bf) == 1:
        n_comps = int(model_bf[0])
        n_blocks = n_blocks_from_branch_factors(
            task_bf,
            auxiliary_root=use_root_aux,
            include_internal_latents=include_internal_latents,
        )
        level_assign: list[list[int]] = [[] for _ in range(n_comps)]
        if n_comps >= n_blocks:
            for c in range(n_comps):
                level_assign[c] = [c % n_blocks]
        else:
            for b in range(n_blocks):
                level_assign[b % n_comps].append(b)
        root_block = task_block_index_for_node(
            task_bf,
            depth_from_root=0,
            index_in_layer=0,
            auxiliary_root=use_root_aux,
            include_internal_latents=include_internal_latents,
        )
        soma_assign = [[root_block] if root_block is not None else []]
        return [level_assign, soma_assign]

    if len(model_bf) < len(task_bf):
        raise ValueError(
            f"model branch_factors {model_bf} are shallower than task "
            f"{task_bf}; nested_multixor_blocks does not support this case"
        )

    if len(model_bf) > len(task_bf):
        prefix = model_bf[: len(task_bf)]
        if prefix != task_bf:
            raise ValueError(
                f"model branch_factors {model_bf} are deeper than task {task_bf} "
                f"but proximal prefix {prefix} does not match; unsupported"
            )
        return _assign_deeper_prefix(
            model_bf=model_bf,
            task_bf=task_bf,
            auxiliary_root=use_root_aux,
            include_internal_latents=include_internal_latents,
        )

    if model_bf == task_bf:
        return _assign_exact_match(
            branch_factors=model_bf,
            auxiliary_root=use_root_aux,
            include_internal_latents=include_internal_latents,
        )

    return _assign_same_depth_mismatch(
        model_bf=model_bf,
        task_bf=task_bf,
        auxiliary_root=use_root_aux,
        include_internal_latents=include_internal_latents,
    )


def _assign_exact_match(
    *,
    branch_factors: tuple[int, ...],
    auxiliary_root: bool,
    include_internal_latents: bool,
) -> list[list[list[int]]]:
    sizes = dendritic_layer_sizes(branch_factors)
    # root→distal sizes; distal-first levels = reversed non-root + soma
    levels: list[list[list[int]]] = []
    for depth in range(len(sizes) - 1, 0, -1):
        level_blocks: list[list[int]] = []
        for i in range(sizes[depth]):
            level_blocks.append(
                _blocks_for_task_node(
                    branch_factors,
                    depth_from_root=depth,
                    index_in_layer=i,
                    auxiliary_root=auxiliary_root,
                    include_internal_latents=include_internal_latents,
                )
            )
        levels.append(level_blocks)

    root_block = task_block_index_for_node(
        branch_factors,
        depth_from_root=0,
        index_in_layer=0,
        auxiliary_root=auxiliary_root,
        include_internal_latents=include_internal_latents,
    )
    levels.append([[root_block] if root_block is not None else []])
    return levels


def _assign_deeper_prefix(
    *,
    model_bf: tuple[int, ...],
    task_bf: tuple[int, ...],
    auxiliary_root: bool,
    include_internal_latents: bool,
) -> list[list[list[int]]]:
    """Extra distal model levels share leaf blocks; prefix levels match 1:1."""
    model_sizes = dendritic_layer_sizes(model_bf)
    task_sizes = dendritic_layer_sizes(task_bf)

    levels: list[list[list[int]]] = []

    # Extra distal model depths: model_depth in (task_leaf_depth, model_leaf_depth]
    task_leaf_depth = len(task_sizes) - 1
    model_leaf_depth = len(model_sizes) - 1
    for model_depth in range(model_leaf_depth, task_leaf_depth, -1):
        level_blocks: list[list[int]] = []
        comps_per_task_leaf = model_sizes[model_depth] // task_sizes[task_leaf_depth]
        for i in range(model_sizes[model_depth]):
            task_leaf = i // comps_per_task_leaf
            level_blocks.append([task_leaf])
        levels.append(level_blocks)

    # Matched depths: model_depth == task_depth for 1..task_leaf_depth
    for depth in range(task_leaf_depth, 0, -1):
        level_blocks = []
        for i in range(task_sizes[depth]):
            level_blocks.append(
                _blocks_for_task_node(
                    task_bf,
                    depth_from_root=depth,
                    index_in_layer=i,
                    auxiliary_root=auxiliary_root,
                    include_internal_latents=include_internal_latents,
                )
            )
        # Model may have same count at matched depths because prefix equals task_bf.
        if model_sizes[depth] != task_sizes[depth]:
            raise RuntimeError(
                f"prefix match invariant failed at depth {depth}: "
                f"model {model_sizes[depth]} vs task {task_sizes[depth]}"
            )
        levels.append(level_blocks)

    root_block = task_block_index_for_node(
        task_bf,
        depth_from_root=0,
        index_in_layer=0,
        auxiliary_root=auxiliary_root,
        include_internal_latents=include_internal_latents,
    )
    levels.append([[root_block] if root_block is not None else []])
    return levels


def _assign_same_depth_mismatch(
    *,
    model_bf: tuple[int, ...],
    task_bf: tuple[int, ...],
    auxiliary_root: bool,
    include_internal_latents: bool,
) -> list[list[list[int]]]:
    """Parent–child coupled assignment with per-layer block coverage."""
    model_sizes = dendritic_layer_sizes(model_bf)
    task_sizes = dendritic_layer_sizes(task_bf)
    depth_max = len(model_sizes) - 1

    # node_task_index[depth][model_index] -> task index in that layer
    node_task_index: list[list[int]] = [[0]]  # root
    for depth in range(1, depth_max + 1):
        model_parent_bf = model_bf[depth - 1]
        task_parent_bf = task_bf[depth - 1]
        parent_depth = depth - 1
        child_assign: list[int] = [-1] * model_sizes[depth]
        for p in range(model_sizes[parent_depth]):
            task_parent = node_task_index[parent_depth][p]
            model_child_start = p * model_parent_bf
            task_child_start = task_parent * task_parent_bf
            mapping = _distribute_children(model_parent_bf, task_parent_bf)
            for local_i, task_local in enumerate(mapping):
                child_assign[model_child_start + local_i] = (
                    task_child_start + task_local
                )
        node_task_index.append(child_assign)

    # Build distal-first block lists; enforce coverage under each parent.
    levels: list[list[list[int]]] = []
    for depth in range(depth_max, 0, -1):
        level_blocks: list[list[int]] = [[] for _ in range(model_sizes[depth])]
        model_parent_bf = model_bf[depth - 1]
        task_parent_bf = task_bf[depth - 1]
        parent_depth = depth - 1
        for p in range(model_sizes[parent_depth]):
            task_parent = node_task_index[parent_depth][p]
            model_child_start = p * model_parent_bf
            task_child_start = task_parent * task_parent_bf
            mapping = _distribute_children(model_parent_bf, task_parent_bf)
            extras = _extra_task_children_for_coverage(
                model_parent_bf, task_parent_bf
            )
            for local_i, task_local in enumerate(mapping):
                task_nodes = [task_child_start + task_local]
                for extra_local in extras[local_i]:
                    task_nodes.append(task_child_start + extra_local)
                block_ids: list[int] = []
                for local_task in task_nodes:
                    block_ids.extend(
                        _blocks_for_task_node(
                            task_bf,
                            depth_from_root=depth,
                            index_in_layer=local_task,
                            auxiliary_root=auxiliary_root,
                            include_internal_latents=include_internal_latents,
                        )
                    )
                # Deduplicate while preserving order.
                seen_local: set[int] = set()
                unique_blocks: list[int] = []
                for b in block_ids:
                    if b not in seen_local:
                        seen_local.add(b)
                        unique_blocks.append(b)
                level_blocks[model_child_start + local_i] = unique_blocks

        if include_internal_latents:
            expected = set(
                blocks_at_depth(
                    task_bf,
                    depth,
                    auxiliary_root=auxiliary_root,
                    include_internal_latents=True,
                )
            )
        else:
            # Leaves-only: every leaf under this depth's task nodes must appear.
            expected = set()
            for task_i in range(task_sizes[depth]):
                expected.update(
                    descendant_leaf_indices(
                        task_bf,
                        depth_from_root=depth,
                        index_in_layer=task_i,
                    )
                )
        seen: set[int] = set()
        for bl in level_blocks:
            seen.update(bl)
        if seen != expected:
            raise RuntimeError(
                f"same-depth mismatch coverage failed at depth {depth}: "
                f"seen={sorted(seen)} expected={sorted(expected)}"
            )
        levels.append(level_blocks)

    root_block = task_block_index_for_node(
        task_bf,
        depth_from_root=0,
        index_in_layer=0,
        auxiliary_root=auxiliary_root,
        include_internal_latents=include_internal_latents,
    )
    levels.append([[root_block] if root_block is not None else []])
    return levels


def build_nested_multixor_connection_mask(
    *,
    out_features: int,
    in_features: int,
    level_idx: int,
    n_soma: int,
    branch_factors: tuple[int, ...] | list[int],
    task_branch_factors: tuple[int, ...] | list[int],
    dims_per_leaf: int,
    auxiliary_root: bool = False,
    include_internal_latents: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a ``[out_features, in_features]`` bool mask for one DendriNet level."""
    model_bf = _as_bf_tuple(branch_factors)
    task_bf = _as_bf_tuple(task_branch_factors)
    dims = int(dims_per_leaf)
    n_soma = int(n_soma)
    include_internal_latents = bool(include_internal_latents)
    use_root_aux = effective_auxiliary_root(
        include_internal_latents=include_internal_latents,
        auxiliary_root=auxiliary_root,
    )
    if n_soma < 1:
        raise ValueError(f"n_soma must be >= 1, got {n_soma}")
    if dims < 1:
        raise ValueError(f"dims_per_leaf must be >= 1, got {dims_per_leaf}")

    expected_in = n_blocks_from_branch_factors(
        task_bf,
        auxiliary_root=use_root_aux,
        include_internal_latents=include_internal_latents,
    ) * dims
    if int(in_features) != expected_in:
        raise ValueError(
            f"in_features={in_features} does not match nested MultiXOR layout "
            f"n_blocks*dims_per_leaf={expected_in} for task_branch_factors={task_bf}, "
            f"dims_per_leaf={dims}, "
            f"include_internal_latents={include_internal_latents}, "
            f"auxiliary_root={auxiliary_root}"
        )

    per_soma_levels = assign_compartment_blocks_per_soma(
        model_branch_factors=model_bf,
        task_branch_factors=task_bf,
        auxiliary_root=use_root_aux,
        include_internal_latents=include_internal_latents,
    )
    if level_idx < 0 or level_idx >= len(per_soma_levels):
        raise ValueError(
            f"level_idx={level_idx} out of range for model_bf={model_bf} "
            f"(n_levels including soma placeholder={len(per_soma_levels)})"
        )

    per_soma_assign = per_soma_levels[level_idx]
    comps_per_soma = len(per_soma_assign)
    expected_out = n_soma * comps_per_soma
    if int(out_features) != expected_out:
        raise ValueError(
            f"out_features={out_features} != n_soma*comps_per_soma="
            f"{expected_out} at level_idx={level_idx} for model_bf={model_bf}"
        )

    ranges = task_block_ranges(
        task_bf,
        dims,
        auxiliary_root=use_root_aux,
        include_internal_latents=include_internal_latents,
    )
    mask = torch.zeros(out_features, in_features, dtype=torch.bool, device=device)
    for soma in range(n_soma):
        row_base = soma * comps_per_soma
        for local_row, block_ids in enumerate(per_soma_assign):
            if not block_ids:
                raise ValueError(
                    f"compartment level_idx={level_idx} local_row={local_row} "
                    f"has no assigned task blocks "
                    f"(include_internal_latents={include_internal_latents}, "
                    f"auxiliary_root={auxiliary_root}). "
                    "Disable somatic synapses when the root has no aux block."
                )
            row = row_base + local_row
            for block_id in block_ids:
                sl = ranges[block_id]
                mask[row, sl] = True

    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError(
            f"nested_multixor_blocks produced an empty row at level_idx={level_idx}"
        )
    return mask


__all__ = [
    "assign_compartment_blocks_per_soma",
    "build_nested_multixor_connection_mask",
]
