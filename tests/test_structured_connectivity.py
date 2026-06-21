"""Tests for hard structured synaptic connectivity masks."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DeepstLinear,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    StochasticTopKLinear,
    TopKLinear,
    distance_kernel_probabilities,
    sample_fixed_indegree_mask,
    sample_probability_mask,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


def test_topk_connection_mask_limits_active_synapses():
    connection_mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, True, False, True, False],
            [False, False, False, True, True],
        ]
    )
    layer = TopKLinear(
        in_features=5,
        out_features=3,
        K=5,
        connection_mask=connection_mask,
    )
    layer.initialize()

    active = layer.weight_mask().bool()

    assert torch.equal(active, connection_mask)
    assert not bool((active & ~connection_mask).any())


def test_topk_connection_mask_blocks_disallowed_gradients():
    connection_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    layer = TopKLinear(
        in_features=4,
        out_features=2,
        K=4,
        connection_mask=connection_mask,
    )
    layer.initialize()

    x = torch.randn(3, 4)
    layer(x).sum().backward()

    assert layer.pre_w.grad is not None
    assert torch.count_nonzero(layer.pre_w.grad[~connection_mask]) == 0
    assert torch.count_nonzero(layer.pre_w.grad[connection_mask]) > 0


def test_topk_unmasked_state_dict_stays_legacy_compatible():
    unmasked = TopKLinear(in_features=4, out_features=2, K=2)
    masked = TopKLinear(
        in_features=4,
        out_features=2,
        K=2,
        connection_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    assert "connection_mask" not in unmasked.state_dict()
    assert "connection_mask" in masked.state_dict()


def test_stochastic_topk_connection_mask_is_enforced_in_both_modes():
    connection_mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, True, False, True, False],
        ]
    )
    for ultrafast in (False, True):
        layer = StochasticTopKLinear(
            in_features=5,
            out_features=2,
            K=5,
            connection_mask=connection_mask,
            ultrafast=ultrafast,
            temperature=0.5,
        )
        layer.initialize()

        active = layer.weight_mask().bool()

        assert not bool((active & ~connection_mask).any())
        assert bool(active.any())


def test_connection_mask_rejects_empty_output_rows():
    connection_mask = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="at least one input"):
        TopKLinear(
            in_features=2,
            out_features=2,
            K=1,
            connection_mask=connection_mask,
        )


def test_deepst_initializes_only_allowed_connections():
    connection_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    layer = DeepstLinear(
        in_features=4,
        out_features=2,
        target_density=1.0,
        connection_mask=connection_mask,
    )

    active = layer.weight_mask().bool()
    stats = layer.get_connectivity_stats()

    assert torch.equal(active, connection_mask)
    assert stats["allowed_connections"] == int(connection_mask.sum())
    assert stats["allowed_density"] == pytest.approx(1.0)


def test_deepst_linear_decays_active_and_boosts_inactive_allowed_weights():
    layer = DeepstLinear(
        in_features=4,
        out_features=2,
        target_density=0.0,
        weight_transform="exp",
        weight_init_range=(0.0, 0.0),
        connection_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    with torch.no_grad():
        layer.mask.zero_()
        layer.mask[0, 0] = True
        layer.mask[1, 2] = True

    before = layer.weight().clone()
    active = layer.weight_mask().bool()
    inactive = ~active

    layer.decay_weights(weight_decay=0.25, weight_boosting=True)
    after = layer.weight()

    assert torch.allclose(after[active], before[active] * 0.75)
    assert torch.allclose(after[inactive], before[inactive] * 1.25)


@pytest.mark.parametrize(
    "layer",
    [
        TopKLinear(
            in_features=4,
            out_features=2,
            K=2,
            weight_transform="softplus",
        ),
        IndexedSparseLinear(
            in_features=4,
            out_features=2,
            K=2,
            weight_transform="softplus",
            seed=11,
        ),
        IndexedDynamicTopKLinear(
            in_features=4,
            out_features=2,
            K=2,
            candidate_size=3,
            weight_transform="softplus",
            seed=13,
        ),
        DeepstLinear(
            in_features=4,
            out_features=2,
            target_density=1.0,
            weight_transform="softplus",
            weight_init_range=(0.0, 0.0),
        ),
    ],
)
def test_softplus_decay_inverse_is_stable_for_large_weights(layer):
    with torch.no_grad():
        layer.pre_w.fill_(1000.0)

    layer.decay_weights(weight_decay=0.1, weight_boosting=True)

    assert torch.isfinite(layer.pre_w).all()


def test_indexed_sparse_linear_seed_none_follows_global_rng():
    torch.manual_seed(123)
    first = IndexedSparseLinear(in_features=7, out_features=5, K=3, seed=None)
    torch.manual_seed(123)
    second = IndexedSparseLinear(in_features=7, out_features=5, K=3, seed=None)
    torch.manual_seed(124)
    third = IndexedSparseLinear(in_features=7, out_features=5, K=3, seed=None)

    assert torch.equal(first.connection_indices, second.connection_indices)
    assert not torch.equal(first.connection_indices, third.connection_indices)


def test_indexed_dynamic_auto_candidate_size_respects_connection_mask():
    connection_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
            [False, False, True, True, True, False],
        ]
    )

    layer = IndexedDynamicTopKLinear(
        in_features=6,
        out_features=3,
        K=2,
        candidate_size=None,
        connection_mask=connection_mask,
        seed=5,
    )

    row = torch.arange(layer.out_features)[:, None]
    assert layer.candidate_size == 2
    assert bool(connection_mask[row, layer.connection_indices].all())


def test_indexed_dynamic_explicit_candidate_size_remains_strict_with_mask():
    connection_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least K allowed inputs"):
        IndexedDynamicTopKLinear(
            in_features=4,
            out_features=2,
            K=2,
            candidate_size=3,
            connection_mask=connection_mask,
            seed=5,
        )


def test_indexed_sparse_linear_forward_param_gradient_scale():
    layer = IndexedSparseLinear(
        in_features=4,
        out_features=3,
        K=2,
        seed=0,
        weight_transform="identity",
    )
    x = torch.ones(2, 4)
    out = layer(x).sum()
    layer.register_forward_param_gradient_scale(torch.zeros(3, 1))
    out.backward()

    assert layer.pre_w.grad is not None
    assert torch.allclose(layer.pre_w.grad, torch.zeros_like(layer.pre_w.grad))


def test_dendritic_branch_layer_passes_connection_mask_to_topk():
    connection_mask = sample_fixed_indegree_mask(
        out_features=3,
        in_features=5,
        indegree=2,
        generator=torch.Generator().manual_seed(7),
    )
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=5,
        excitatory_synapses_per_branch=5,
        excitatory_connection_mask=connection_mask,
        input_branch_factor=None,
        reactivate=False,
        dbl_init_method="default",
    )

    active = layer.branch_excitation.weight_mask().bool()

    assert torch.equal(active, connection_mask)


def test_sample_probability_mask_is_reproducible_and_calibrated():
    generator_a = torch.Generator().manual_seed(11)
    generator_b = torch.Generator().manual_seed(11)

    mask_a = sample_probability_mask(200, 100, 0.25, generator=generator_a)
    mask_b = sample_probability_mask(200, 100, 0.25, generator=generator_b)

    assert torch.equal(mask_a, mask_b)
    assert mask_a.float().mean().item() == pytest.approx(0.25, abs=0.02)


def test_sample_probability_mask_repairs_empty_rows_when_possible():
    generator = torch.Generator().manual_seed(23)

    mask = sample_probability_mask(50, 4, 0.05, generator=generator)

    assert bool((mask.sum(dim=1) > 0).all())


def test_sample_fixed_indegree_mask_sets_exact_row_counts():
    mask = sample_fixed_indegree_mask(
        out_features=8,
        in_features=20,
        indegree=4,
        generator=torch.Generator().manual_seed(13),
    )

    assert mask.dtype == torch.bool
    assert torch.equal(mask.sum(dim=1), torch.full((8,), 4))


def test_distance_kernel_probabilities_decay_with_distance():
    pre_positions = torch.tensor([[0.0], [1.0], [2.0]])
    post_positions = torch.tensor([[0.0]])

    probs = distance_kernel_probabilities(
        pre_positions,
        post_positions,
        kernel="gaussian",
        sigma=1.0,
    )

    assert probs.shape == (1, 3)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]


def test_factory_applies_structured_connectivity_to_feedforward_einet():
    config = {
        "type": "dendritic_shunting",
        "architecture": {
            "excitatory_layer_sizes": [4],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [5],
            "ie_synapses_per_branch_per_layer": [0],
            "structured": {
                "enabled": True,
                "seed": 17,
                "pathways": {
                    "ee": {"method": "fixed_indegree", "indegree": 2},
                },
            },
        },
        "transfer": {"input_mode": 1},
        "morphology": {
            "somatic_synapses": True,
            "use_shunting": True,
            "weight_transform": "exp",
            "dbl_init_method": "default",
        },
        "sparsity": {"type": "standard", "init_method": "xavier_normal"},
        "reactivation": {"enabled": False},
        "blocklinear": {},
        "implementation": {"compile_forward": False},
    }

    net = get_architecture("dendritic_shunting", config, input_dim=6)
    branch_layers = net.layers[0].excitatory_cells.branch_layers

    assert branch_layers
    for branch_layer in branch_layers:
        mask = branch_layer.branch_excitation.weight_mask()
        assert torch.equal(mask.sum(dim=1), torch.full((mask.shape[0],), 2.0))


def test_factory_applies_structured_connectivity_to_recurrent_einet():
    config = {
        "type": "dendritic_shunting",
        "architecture": {
            "excitatory_layer_sizes": [5],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [3],
            "ie_synapses_per_branch_per_layer": [0],
            "rec_ee_synapses_per_branch": [4],
            "structured": {
                "enabled": True,
                "seed": 19,
                "pathways": {
                    "rec_ee": {"method": "fixed_indegree", "indegree": 1},
                },
            },
        },
        "transfer": {"input_mode": 1},
        "morphology": {
            "somatic_synapses": True,
            "use_shunting": True,
            "allow_self_recurrence": True,
            "weight_transform": "exp",
            "dbl_init_method": "default",
        },
        "sparsity": {"type": "standard", "init_method": "xavier_normal"},
        "reactivation": {"enabled": False},
        "blocklinear": {},
        "implementation": {"compile_forward": False},
        "recurrent_ei": {"output_mode": "last"},
    }

    net = get_architecture("dendritic_shunting", config, input_dim=6)
    branch_layers = net.layers[0].e_population.branch_layers

    assert branch_layers
    for branch_layer in branch_layers:
        mask = branch_layer.branch_recurrent.weight_mask()
        assert torch.equal(mask.sum(dim=1), torch.ones(mask.shape[0]))
