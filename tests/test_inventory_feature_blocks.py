import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_configured_mask,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)


def _inventory_config(owner_count: int) -> dict:
    pathway = {
        "method": "inventory_feature_blocks",
        "owner_count": owner_count,
        "inventory_counts": [4, 2, 2],
        "feature_ranges": [[0, 4], [4, 8], [8, 12]],
    }
    return {
        "enabled": True,
        "pathways": {"ee": dict(pathway), "ie": dict(pathway)},
    }


def _row_range(mask_row: torch.Tensor) -> tuple[int, int]:
    allowed = mask_row.nonzero(as_tuple=False).flatten()
    return int(allowed[0].item()), int(allowed[-1].item()) + 1


def test_inventory_feature_blocks_realizes_exact_4_2_2_tree_placement() -> None:
    expected = {
        (2, 0): [(0, 4)] * 4 + [(4, 8)] * 2 + [(8, 12)] * 2,
        (3, 0): [(0, 4)] * 4 + [(4, 8)] * 2,
        (3, 1): [(8, 12)] * 2,
        (4, 0): [(0, 4)] * 4,
        (4, 1): [(4, 8)] * 2,
        (4, 2): [(8, 12)] * 2,
    }
    config = _inventory_config(owner_count=1)

    for (n_levels, level_idx), expected_ranges in expected.items():
        mask = sample_configured_mask(
            config,
            pathway="ee",
            out_features=len(expected_ranges),
            in_features=12,
            level_idx=level_idx,
            n_levels=n_levels,
        )
        assert mask is not None
        assert [_row_range(row) for row in mask] == expected_ranges


def test_inventory_feature_blocks_accepts_explicit_four_to_three_grouping() -> None:
    pathway = {
        "method": "inventory_feature_blocks",
        "owner_count": 1,
        "inventory_counts": [4, 2, 1, 1],
        "feature_ranges": [[0, 4], [4, 8], [8, 10], [10, 12]],
        "tier_groups": [[0], [1], [2, 3]],
    }
    config = {
        "enabled": True,
        "pathways": {"ee": pathway},
    }
    expected = {
        0: [(0, 4)] * 4,
        1: [(4, 8)] * 2,
        2: [(8, 10), (10, 12)],
    }
    for level_idx, expected_ranges in expected.items():
        mask = sample_configured_mask(
            config,
            pathway="ee",
            out_features=len(expected_ranges),
            in_features=12,
            level_idx=level_idx,
            n_levels=4,
        )
        assert mask is not None
        assert [_row_range(row) for row in mask] == expected_ranges


def test_inventory_feature_blocks_rejects_invalid_tier_grouping() -> None:
    pathway = {
        "method": "inventory_feature_blocks",
        "owner_count": 1,
        "inventory_counts": [4, 2, 1, 1],
        "feature_ranges": [[0, 4], [4, 8], [8, 10], [10, 12]],
        "tier_groups": [[0], [2], [1, 3]],
    }
    config = {"enabled": True, "pathways": {"ee": pathway}}
    try:
        sample_configured_mask(
            config,
            pathway="ee",
            out_features=4,
            in_features=12,
            level_idx=0,
            n_levels=4,
        )
    except ValueError as error:
        assert "ordered partition" in str(error)
    else:
        raise AssertionError("Expected invalid tier_groups to be rejected.")


def test_inventory_feature_blocks_flows_through_all_supported_morphologies() -> None:
    common = {
        "n_neurons": 2,
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 2,
        "somatic_synapses": False,
        "reactivate": False,
        "weight_transform": "softplus",
        "structured_connectivity": _inventory_config(owner_count=2),
    }
    expected_ranges_by_morphology = {
        (8,): [[(0, 4)] * 4 + [(4, 8)] * 2 + [(8, 12)] * 2],
        (2, 3): [[(0, 4)] * 4 + [(4, 8)] * 2, [(8, 12)] * 2],
        (2, 1, 2): [[(0, 4)] * 4, [(4, 8)] * 2, [(8, 12)] * 2],
    }

    for morphology, level_ranges in expected_ranges_by_morphology.items():
        network = StatefulDendriNet(
            pop_config=PopulationConfig(
                branch_factors=list(morphology),
                **common,
            ),
            excitatory_input_dim=12,
            inhibitory_input_dim=12,
        )
        active_layers = network.branch_layers[:-1]
        assert len(active_layers) == len(level_ranges)
        for layer, expected_ranges in zip(active_layers, level_ranges):
            exc_mask = layer.branch_config.excitatory_connection_mask
            inh_mask = layer.branch_config.inhibitory_connection_mask
            assert exc_mask is not None
            assert inh_mask is not None
            # Check one owner; the mask pattern repeats identically per soma.
            assert [_row_range(row) for row in exc_mask[: len(expected_ranges)]] == (
                expected_ranges
            )
            assert torch.equal(exc_mask, inh_mask)


def test_two_level_inventory_matches_d1_and_d2_resource_layouts() -> None:
    pathway = {
        "method": "inventory_feature_blocks",
        "owner_count": 2,
        "inventory_counts": [4, 4],
        "feature_ranges": [[0, 6], [6, 12]],
    }
    structured = {
        "enabled": True,
        "pathways": {"ee": dict(pathway), "ie": dict(pathway)},
    }
    common = {
        "n_neurons": 2,
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 2,
        "somatic_synapses": False,
        "reactivate": False,
        "weight_transform": "softplus",
        "structured_connectivity": structured,
    }
    expected = {
        (8,): [[(0, 6)] * 4 + [(6, 12)] * 4],
        (4, 1): [[(0, 6)] * 4, [(6, 12)] * 4],
    }
    parameter_counts = []
    state_widths = []
    for morphology, expected_levels in expected.items():
        network = StatefulDendriNet(
            pop_config=PopulationConfig(branch_factors=list(morphology), **common),
            excitatory_input_dim=12,
            inhibitory_input_dim=12,
        )
        parameter_counts.append(sum(p.numel() for p in network.parameters()))
        state_widths.append(sum(network.level_dims))
        for layer, expected_ranges in zip(
            network.branch_layers[:-1], expected_levels, strict=True
        ):
            mask = layer.branch_config.excitatory_connection_mask
            assert mask is not None
            assert [_row_range(row) for row in mask[: len(expected_ranges)]] == (
                expected_ranges
            )
    assert parameter_counts[0] == parameter_counts[1]
    assert state_widths[0] == state_widths[1]


def test_inventory_feature_blocks_rejects_mismatched_morphology() -> None:
    config = _inventory_config(owner_count=2)
    try:
        sample_configured_mask(
            config,
            pathway="ee",
            out_features=10,
            in_features=12,
            level_idx=0,
            n_levels=2,
        )
    except ValueError as error:
        assert "morphology does not match" in str(error)
    else:
        raise AssertionError("Expected a mismatched morphology to be rejected.")


def test_inventory_network_child_conductance_is_mechanism_neutral_at_every_depth() -> (
    None
):
    """The production hierarchy receives one shared axial initialization."""

    common = {
        "n_neurons": 2,
        "branch_factors": [2, 1, 2],
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 2,
        "somatic_synapses": False,
        "reactivate": False,
        "weight_transform": "softplus",
        "dbl_init_method": "mechanism_neutral",
        "adaptive_initialization": False,
        "initial_child_conductance": 16.0,
    }

    def build(use_shunting: bool) -> StatefulDendriNet:
        torch.manual_seed(123)
        return StatefulDendriNet(
            pop_config=PopulationConfig(use_shunting=use_shunting, **common),
            excitatory_input_dim=12,
            inhibitory_input_dim=12,
        )

    shunting = build(True)
    additive = build(False)
    assert shunting.state_dict().keys() == additive.state_dict().keys()
    for key, value in shunting.state_dict().items():
        torch.testing.assert_close(
            value, additive.state_dict()[key], rtol=0.0, atol=0.0
        )

    child_layers = [layer for layer in shunting.branch_layers if layer.input_branches]
    assert len(child_layers) == 3
    for layer in child_layers:
        torch.testing.assert_close(
            layer.branches_to_output.weight(),
            torch.full_like(layer.branches_to_output.weight(), 16.0),
            rtol=0.0,
            atol=0.0,
        )
