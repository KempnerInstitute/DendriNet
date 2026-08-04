"""Tests for nested MultiXOR block-aligned connectivity masks."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNet,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.nested_multixor_mask import (
    assign_compartment_blocks_per_soma,
    build_nested_multixor_connection_mask,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.utils.nested_tree_geometry import (
    n_blocks_from_branch_factors,
    task_block_index_for_node,
)


def _allowed_blocks(mask_row: torch.Tensor, dims: int) -> list[int]:
    allowed = mask_row.nonzero(as_tuple=False).flatten().tolist()
    return sorted({idx // dims for idx in allowed})


def test_task_block_layout_aux_order():
    bf = (2, 2)
    assert task_block_index_for_node(
        bf, depth_from_root=2, index_in_layer=0, auxiliary_root=False
    ) == 0
    assert task_block_index_for_node(
        bf, depth_from_root=1, index_in_layer=0, auxiliary_root=False
    ) == 4
    assert task_block_index_for_node(
        bf, depth_from_root=1, index_in_layer=1, auxiliary_root=False
    ) == 5
    assert (
        task_block_index_for_node(
            bf, depth_from_root=0, index_in_layer=0, auxiliary_root=False
        )
        is None
    )
    assert task_block_index_for_node(
        bf, depth_from_root=0, index_in_layer=0, auxiliary_root=True
    ) == 6


def test_exact_match_assignment_and_mask_n_soma2():
    model_bf = (2, 2)
    task_bf = (2, 2)
    dims = 50
    n_soma = 2
    in_features = n_blocks_from_branch_factors(task_bf, auxiliary_root=False) * dims

    per_soma = assign_compartment_blocks_per_soma(
        model_branch_factors=model_bf,
        task_branch_factors=task_bf,
        auxiliary_root=False,
    )
    # distal, proximal, soma
    assert per_soma[0] == [[0], [1], [2], [3]]
    assert per_soma[1] == [[4], [5]]
    assert per_soma[2] == [[]]

    distal = build_nested_multixor_connection_mask(
        out_features=n_soma * 4,
        in_features=in_features,
        level_idx=0,
        n_soma=n_soma,
        branch_factors=model_bf,
        task_branch_factors=task_bf,
        dims_per_leaf=dims,
        auxiliary_root=False,
    )
    proximal = build_nested_multixor_connection_mask(
        out_features=n_soma * 2,
        in_features=in_features,
        level_idx=1,
        n_soma=n_soma,
        branch_factors=model_bf,
        task_branch_factors=task_bf,
        dims_per_leaf=dims,
        auxiliary_root=False,
    )

    for soma in range(n_soma):
        for local in range(4):
            row = soma * 4 + local
            assert _allowed_blocks(distal[row], dims) == [local]
        for local in range(2):
            row = soma * 2 + local
            assert _allowed_blocks(proximal[row], dims) == [4 + local]


def test_deeper_prefix_extra_distal_leaf_only():
    model_bf = (2, 2, 2)
    task_bf = (2, 2)
    dims = 10
    in_features = n_blocks_from_branch_factors(task_bf, auxiliary_root=False) * dims

    per_soma = assign_compartment_blocks_per_soma(
        model_branch_factors=model_bf,
        task_branch_factors=task_bf,
        auxiliary_root=False,
    )
    # Extra distal (8): pairs share leaf blocks 0..3
    assert per_soma[0] == [[0], [0], [1], [1], [2], [2], [3], [3]]
    # Matched task-leaf depth (4)
    assert per_soma[1] == [[0], [1], [2], [3]]
    # Proximal aux
    assert per_soma[2] == [[4], [5]]

    finest = build_nested_multixor_connection_mask(
        out_features=8,
        in_features=in_features,
        level_idx=0,
        n_soma=1,
        branch_factors=model_bf,
        task_branch_factors=task_bf,
        dims_per_leaf=dims,
        auxiliary_root=False,
    )
    for row in range(8):
        blocks = _allowed_blocks(finest[row], dims)
        assert len(blocks) == 1
        assert blocks[0] < 4  # leaf only


def test_same_depth_mismatch_parent_child_and_coverage():
    model_bf = (2, 3)
    task_bf = (2, 2)
    dims = 8
    in_features = n_blocks_from_branch_factors(task_bf, auxiliary_root=False) * dims

    per_soma = assign_compartment_blocks_per_soma(
        model_branch_factors=model_bf,
        task_branch_factors=task_bf,
        auxiliary_root=False,
    )
    distal = per_soma[0]
    proximal = per_soma[1]
    assert len(distal) == 6
    assert len(proximal) == 2

    # Every leaf and aux block appears at its layer.
    distal_blocks = {b for row in distal for b in row}
    assert distal_blocks == {0, 1, 2, 3}
    proximal_blocks = {b for row in proximal for b in row}
    assert proximal_blocks == {4, 5}

    # Parent-child coupling: children of proximal local 0 only see leaves of task prox 0.
    # With 1:1 proximal mapping, prox0→block4 (task node0), children leaves 0,1.
    child_blocks_p0 = {b for row in distal[0:3] for b in row}
    child_blocks_p1 = {b for row in distal[3:6] for b in row}
    assert child_blocks_p0 <= {0, 1}
    assert child_blocks_p1 <= {2, 3}
    assert child_blocks_p0 == {0, 1}
    assert child_blocks_p1 == {2, 3}

    mask = build_nested_multixor_connection_mask(
        out_features=6,
        in_features=in_features,
        level_idx=0,
        n_soma=1,
        branch_factors=model_bf,
        task_branch_factors=task_bf,
        dims_per_leaf=dims,
        auxiliary_root=False,
    )
    for row in range(6):
        assert len(_allowed_blocks(mask[row], dims)) >= 1


def test_flat_cover_all_blocks():
    model_bf = (6,)  # flattened [2,2] non-soma count
    task_bf = (2, 2)
    dims = 5
    n_blocks = n_blocks_from_branch_factors(task_bf, auxiliary_root=False)
    in_features = n_blocks * dims

    per_soma = assign_compartment_blocks_per_soma(
        model_branch_factors=model_bf,
        task_branch_factors=task_bf,
        auxiliary_root=False,
    )
    flat = per_soma[0]
    assert len(flat) == 6
    seen = {b for row in flat for b in row}
    assert seen == set(range(n_blocks))
    assert all(len(row) == 1 for row in flat)

    mask = build_nested_multixor_connection_mask(
        out_features=12,
        in_features=in_features,
        level_idx=0,
        n_soma=2,
        branch_factors=model_bf,
        task_branch_factors=task_bf,
        dims_per_leaf=dims,
        auxiliary_root=False,
    )
    for row in range(12):
        assert mask[row].sum() == dims


def test_topk_respects_block_mask_with_k_ne_dims():
    dims = 50
    k = 5
    task_bf = (2, 2)
    in_features = n_blocks_from_branch_factors(task_bf, auxiliary_root=False) * dims
    mask = build_nested_multixor_connection_mask(
        out_features=4,
        in_features=in_features,
        level_idx=0,
        n_soma=1,
        branch_factors=(2, 2),
        task_branch_factors=task_bf,
        dims_per_leaf=dims,
        auxiliary_root=False,
    )
    layer = TopKLinear(
        in_features=in_features,
        out_features=4,
        K=k,
        connection_mask=mask,
    )
    layer.initialize()
    weight_mask = layer.weight_mask()
    for row in range(4):
        active = weight_mask[row].nonzero(as_tuple=False).flatten()
        assert active.numel() == k
        assert bool(mask[row, active].all())
        assert _allowed_blocks(mask[row], dims) == [row]


def test_in_features_mismatch_raises():
    with pytest.raises(ValueError, match="in_features"):
        build_nested_multixor_connection_mask(
            out_features=4,
            in_features=100,
            level_idx=0,
            n_soma=1,
            branch_factors=(2, 2),
            task_branch_factors=(2, 2),
            dims_per_leaf=50,
            auxiliary_root=False,
        )


def test_unsupported_deeper_prefix_raises():
    with pytest.raises(ValueError, match="proximal prefix"):
        assign_compartment_blocks_per_soma(
            model_branch_factors=(3, 2, 2),
            task_branch_factors=(2, 2),
            auxiliary_root=False,
        )


def test_shallower_model_raises():
    with pytest.raises(ValueError, match="shallower"):
        assign_compartment_blocks_per_soma(
            model_branch_factors=(2, 2),
            task_branch_factors=(2, 2, 2),
            auxiliary_root=False,
        )


def test_dendrinet_applies_nested_multixor_ee_mask():
    dims = 10
    task_bf = [2, 2]
    in_dim = n_blocks_from_branch_factors(tuple(task_bf), auxiliary_root=False) * dims
    structured = {
        "enabled": True,
        "pathways": {
            "ee": {
                "method": "nested_multixor_blocks",
                "task_branch_factors": task_bf,
                "dims_per_leaf": dims,
                "auxiliary_root": False,
            }
        },
    }
    net = DendriNet(
        n_soma=2,
        branch_factors=[2, 2],
        excitatory_input_dim=in_dim,
        excitatory_synapses_per_branch=5,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        somatic_synapses=False,
        structured_connectivity=structured,
        weight_transform="softplus",
        reactivate=False,
    )
    distal = net.branch_layers[0]
    assert distal.branch_excitation is not None
    conn = distal.branch_excitation.connection_mask
    assert conn.numel() > 0
    assert tuple(conn.shape) == (8, in_dim)
    for soma in range(2):
        for local in range(4):
            row = soma * 4 + local
            assert _allowed_blocks(conn[row], dims) == [local]


def test_dataset_geometry_helpers_still_work():
    from dendritic_modeling.datasets.synthetic_datasets import (
        create_nested_multixor_dataset,
    )

    train, _, _ = create_nested_multixor_dataset(
        n_samples=32,
        branch_factors=[2, 2],
        dims_per_leaf=8,
        distractor_scale=0.0,
        auxiliary_root=False,
        seed=0,
    )
    x, _ = train.tensors
    assert x.shape[1] == 6 * 8
