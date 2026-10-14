"""Tests for direct morphology-aligned spatial sparse indices."""

import math

import torch

from dendritic_modeling.config import StructuredConnectivityConfig
from dendritic_modeling.networks import DendriNet
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.spatial_morphology import (
    sample_configured_indices,
    sample_spatial_morphology_indices,
    spatial_morphology_rectangles,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_configured_mask,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)


def _spatial_config(seed: int = 17) -> dict:
    return {
        "enabled": True,
        "method": "spatial_morphology",
        "seed": seed,
        "spatial": {
            "input_shape": [1, 8, 8],
            "split_axes": ["height", "width"],
        },
    }


def _assert_indices_in_rectangle(
    indices: torch.Tensor,
    *,
    y_bounds: tuple[int, int],
    x_bounds: tuple[int, int],
    width: int = 8,
) -> None:
    y = indices // width
    x = indices % width
    assert bool(((y >= y_bounds[0]) & (y < y_bounds[1])).all())
    assert bool(((x >= x_bounds[0]) & (x < x_bounds[1])).all())


def test_spatial_indices_follow_root_to_leaf_rectangles() -> None:
    distal = sample_spatial_morphology_indices(
        input_shape=[1, 8, 8],
        out_features=12,
        in_features=64,
        synapses_per_branch=3,
        owner_count=3,
        branch_factors=[2, 2],
        level_idx=0,
        split_axes=["height", "width"],
        seed=11,
    )
    assert distal.shape == (12, 3)

    expected_quadrants = [
        ((0, 4), (0, 4)),
        ((0, 4), (4, 8)),
        ((4, 8), (0, 4)),
        ((4, 8), (4, 8)),
    ]
    for owner in range(3):
        for local_branch, (y_bounds, x_bounds) in enumerate(expected_quadrants):
            row = distal[owner * 4 + local_branch]
            _assert_indices_in_rectangle(
                row,
                y_bounds=y_bounds,
                x_bounds=x_bounds,
            )
            assert row.unique().numel() == row.numel()

    proximal = sample_spatial_morphology_indices(
        input_shape=[1, 8, 8],
        out_features=6,
        in_features=64,
        synapses_per_branch=4,
        owner_count=3,
        branch_factors=[2, 2],
        level_idx=1,
        split_axes=["height", "width"],
        seed=11,
    )
    for owner in range(3):
        _assert_indices_in_rectangle(
            proximal[owner * 2],
            y_bounds=(0, 4),
            x_bounds=(0, 8),
        )
        _assert_indices_in_rectangle(
            proximal[owner * 2 + 1],
            y_bounds=(4, 8),
            x_bounds=(0, 8),
        )


def test_configured_spatial_indices_are_deterministic_and_seeded() -> None:
    kwargs = {
        "pathway": "ee",
        "out_features": 8,
        "in_features": 64,
        "synapses_per_branch": 3,
        "owner_count": 2,
        "branch_factors": [2, 2],
        "layer_idx": 0,
        "level_idx": 0,
    }
    first = sample_configured_indices(_spatial_config(17), **kwargs)
    repeated = sample_configured_indices(_spatial_config(17), **kwargs)
    changed = sample_configured_indices(_spatial_config(18), **kwargs)
    assert first is not None
    assert repeated is not None
    assert changed is not None
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)


def test_empty_typed_spatial_default_does_not_create_a_root_rule() -> None:
    config = StructuredConnectivityConfig(
        enabled=True,
        pathways={"ee": {"method": "fixed_indegree", "indegree": 2}},
    )

    assert (
        sample_configured_mask(
            config,
            pathway="ie",
            out_features=3,
            in_features=5,
        )
        is None
    )


def test_dendrinet_uses_spatial_indices_without_dense_masks() -> None:
    model = DendriNet(
        n_soma=2,
        branch_factors=[2, 2],
        excitatory_input_dim=64,
        excitatory_synapses_per_branch=3,
        inhibitory_input_dim=64,
        inhibitory_synapses_per_branch=2,
        somatic_synapses=False,
        topk_type="indexed",
        indexed_index_dtype="int32",
        indexed_persistent_indices=False,
        reactivate=True,
        reactivation_type="param_tanh",
        weight_transform="softplus",
        structured_connectivity=_spatial_config(),
    )

    active_layers = model.branch_layers[:-1]
    assert len(active_layers) == 2
    for layer in active_layers:
        assert layer.branch_config.excitatory_connection_mask is None
        assert layer.branch_config.inhibitory_connection_mask is None
        assert layer.branch_config.excitatory_connection_indices is not None
        assert layer.branch_config.inhibitory_connection_indices is not None
        assert layer.branch_excitation.connection_indices.dtype == torch.int32
        assert layer.branch_inhibition.connection_indices.dtype == torch.int32
    assert not any(key.endswith("connection_indices") for key in model.state_dict())

    x = torch.rand(4, 64, requires_grad=True)
    output = model(x, x)
    assert output.shape == (4, 2)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_dendrinet_overlapping_grid_runs_panel_k_morphologies() -> None:
    config = {
        "enabled": True,
        "method": "spatial_morphology",
        "seed": 23,
        "spatial": {
            "input_shape": [3, 32, 32],
            "region_mode": "overlapping_grid",
            "minimum_window_shape": [4, 4],
        },
    }
    for branch_factors in ([120], [10, 11], [8, 2, 6], [3, 3, 3, 3]):
        model = DendriNet(
            n_soma=2,
            branch_factors=branch_factors,
            excitatory_input_dim=3072,
            excitatory_synapses_per_branch=40,
            inhibitory_input_dim=3072,
            inhibitory_synapses_per_branch=10,
            somatic_synapses=False,
            topk_type="indexed",
            indexed_index_dtype="int32",
            indexed_persistent_indices=False,
            reactivate=True,
            reactivation_type="param_tanh",
            weight_transform="softplus",
            structured_connectivity=config,
        )
        image = torch.rand(2, 3072, requires_grad=True)
        output = model(image, image)
        assert output.shape == (2, 2)
        output.square().mean().backward()
        assert image.grad is not None
        assert torch.isfinite(image.grad).all()


def _stateful_spatial_model(seed: int = 17) -> StatefulDendriNet:
    return StatefulDendriNet(
        pop_config=PopulationConfig(
            n_neurons=2,
            branch_factors=[2, 2],
            ff_excitatory_synapses=3,
            ff_inhibitory_synapses=2,
            rec_excitatory_synapses=0,
            rec_inhibitory_synapses=0,
            somatic_synapses=False,
            topk_type="indexed",
            indexed_seed=seed,
            indexed_index_dtype="int32",
            indexed_persistent_indices=False,
            structured_connectivity=_spatial_config(seed),
            reactivate=True,
            reactivation_type="param_tanh",
            weight_transform="softplus",
        ),
        excitatory_input_dim=64,
        inhibitory_input_dim=64,
    )


def test_stateful_dendrinet_uses_spatial_indices_without_dense_masks() -> None:
    model = _stateful_spatial_model()

    active_layers = model.branch_layers[:-1]
    assert len(active_layers) == 2
    for layer in active_layers:
        assert layer.branch_config.excitatory_connection_mask is None
        assert layer.branch_config.inhibitory_connection_mask is None
        assert layer.branch_config.excitatory_connection_indices is not None
        assert layer.branch_config.inhibitory_connection_indices is not None
        assert layer.branch_excitation.connection_indices.dtype == torch.int32
        assert layer.branch_inhibition.connection_indices.dtype == torch.int32
    assert not any(key.endswith("connection_indices") for key in model.state_dict())

    x = torch.rand(4, 64, requires_grad=True)
    output = model(x, x)
    assert output.shape == (4, 2)
    output.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_stateful_spatial_indices_are_deterministic_and_seeded() -> None:
    first = _stateful_spatial_model(17)
    repeated = _stateful_spatial_model(17)
    changed = _stateful_spatial_model(18)

    first_indices = first.branch_layers[0].branch_excitation.connection_indices
    repeated_indices = repeated.branch_layers[0].branch_excitation.connection_indices
    changed_indices = changed.branch_layers[0].branch_excitation.connection_indices
    assert torch.equal(first_indices, repeated_indices)
    assert not torch.equal(first_indices, changed_indices)


def test_stateful_spatial_indices_reject_nonindexed_sparsity() -> None:
    try:
        StatefulDendriNet(
            pop_config=PopulationConfig(
                n_neurons=1,
                branch_factors=[2, 2],
                ff_excitatory_synapses=3,
                ff_inhibitory_synapses=0,
                somatic_synapses=False,
                topk_type="indexed_rewire",
                structured_connectivity=_spatial_config(),
            ),
            excitatory_input_dim=64,
        )
    except ValueError as error:
        assert "require sparsity.type='indexed'" in str(error)
    else:
        raise AssertionError("Expected stateful spatial indices to require indexed")


def test_stateful_spatial_indices_reject_active_recurrent_pathways() -> None:
    try:
        StatefulDendriNet(
            pop_config=PopulationConfig(
                n_neurons=1,
                branch_factors=[2, 2],
                ff_excitatory_synapses=3,
                ff_inhibitory_synapses=0,
                rec_excitatory_synapses=2,
                rec_inhibitory_synapses=0,
                somatic_synapses=False,
                topk_type="indexed",
                structured_connectivity=_spatial_config(),
            ),
            excitatory_input_dim=64,
            recurrent_excitatory_input_dim=64,
        )
    except ValueError as error:
        assert "feedforward image pathways only" in str(error)
        assert "rec_ee" in str(error)
    else:
        raise AssertionError("Expected recurrent spatial image routing to fail")


def test_spatial_indices_reject_adaptive_global_rewiring() -> None:
    try:
        DendriNet(
            n_soma=1,
            branch_factors=[2, 2],
            excitatory_input_dim=64,
            excitatory_synapses_per_branch=3,
            somatic_synapses=False,
            topk_type="indexed_rewire",
            structured_connectivity=_spatial_config(),
        )
    except ValueError as error:
        assert "require sparsity.type='indexed'" in str(error)
    else:
        raise AssertionError("Expected spatial indices with global rewiring to fail")


def test_spatial_indices_validate_input_shape_and_region_capacity() -> None:
    try:
        sample_spatial_morphology_indices(
            input_shape=[1, 7, 8],
            out_features=4,
            in_features=64,
            synapses_per_branch=2,
            owner_count=1,
            branch_factors=[2, 2],
            level_idx=0,
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Expected a mismatched input shape to fail")

    try:
        sample_spatial_morphology_indices(
            input_shape=[1, 4, 4],
            out_features=4,
            in_features=16,
            synapses_per_branch=5,
            owner_count=1,
            branch_factors=[2, 2],
            level_idx=0,
        )
    except ValueError as error:
        assert "containing only 4 input features" in str(error)
    else:
        raise AssertionError("Expected an overfull branch region to fail")


def test_overlapping_grid_supports_resource_matched_morphologies() -> None:
    morphologies = ([120], [10, 11], [8, 2, 6], [3, 3, 3, 3])
    for branch_factors in morphologies:
        for level_idx in range(len(branch_factors)):
            active_factors = branch_factors[: len(branch_factors) - level_idx]
            branches = math.prod(active_factors)
            indices = sample_spatial_morphology_indices(
                input_shape=[3, 32, 32],
                out_features=branches,
                in_features=3072,
                synapses_per_branch=40,
                owner_count=1,
                branch_factors=branch_factors,
                level_idx=level_idx,
                region_mode="overlapping_grid",
                minimum_window_shape=[4, 4],
                seed=23,
            )
            assert indices.shape == (branches, 40)
            assert bool((indices >= 0).all())
            assert bool((indices < 3072).all())
            assert all(row.unique().numel() == 40 for row in indices)


def test_overlapping_grid_is_deterministic_and_spatially_local() -> None:
    options = {
        "input_shape": [3, 32, 32],
        "out_features": 120,
        "in_features": 3072,
        "synapses_per_branch": 40,
        "owner_count": 1,
        "branch_factors": [120],
        "level_idx": 0,
        "region_mode": "overlapping_grid",
        "minimum_window_shape": [4, 4],
    }
    first = sample_spatial_morphology_indices(**options, seed=23)
    repeated = sample_spatial_morphology_indices(**options, seed=23)
    changed = sample_spatial_morphology_indices(**options, seed=24)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)

    # Forty of the 48 available RGB coordinates force every row to occupy its
    # local 4x4 window while remaining unique within the branch.
    spatial = first.remainder(32 * 32)
    y = spatial // 32
    x = spatial % 32
    assert bool(((y.max(dim=1).values - y.min(dim=1).values) < 4).all())
    assert bool(((x.max(dim=1).values - x.min(dim=1).values) < 4).all())


def test_configured_overlapping_grid_options_are_forwarded() -> None:
    config = _spatial_config(23)
    config["spatial"].update(
        {
            "input_shape": [3, 32, 32],
            "region_mode": "overlapping_grid",
            "minimum_window_shape": [4, 4],
        }
    )
    indices = sample_configured_indices(
        config,
        pathway="ee",
        out_features=120,
        in_features=3072,
        synapses_per_branch=40,
        owner_count=1,
        branch_factors=[120],
        level_idx=0,
    )
    assert indices is not None
    assert indices.shape == (120, 40)


def test_random_windows_preserve_aligned_window_shape_distribution() -> None:
    for branch_factors in ([120], [10, 11], [8, 2, 6], [3, 3, 3, 3]):
        for level_idx in range(len(branch_factors)):
            options = {
                "input_shape": [3, 32, 32],
                "branch_factors": branch_factors,
                "level_idx": level_idx,
                "minimum_window_shape": [4, 4],
            }
            aligned = spatial_morphology_rectangles(
                **options,
                region_mode="overlapping_grid",
                seed=23,
            )
            randomized = spatial_morphology_rectangles(
                **options,
                region_mode="random_windows",
                seed=23,
            )
            aligned_shapes = sorted((y1 - y0, x1 - x0) for y0, y1, x0, x1 in aligned)
            randomized_shapes = sorted(
                (y1 - y0, x1 - x0) for y0, y1, x0, x1 in randomized
            )
            assert randomized_shapes == aligned_shapes


def test_random_windows_are_deterministic_seeded_and_not_tree_aligned() -> None:
    options = {
        "input_shape": [3, 32, 32],
        "branch_factors": [3, 3, 3, 3],
        "level_idx": 0,
        "region_mode": "random_windows",
        "minimum_window_shape": [4, 4],
    }
    first = spatial_morphology_rectangles(**options, seed=23)
    repeated = spatial_morphology_rectangles(**options, seed=23)
    changed = spatial_morphology_rectangles(**options, seed=24)
    aligned = spatial_morphology_rectangles(
        **{**options, "region_mode": "overlapping_grid"},
        seed=23,
    )
    assert first == repeated
    assert first != changed
    assert first != aligned


def test_configured_random_windows_share_e_i_regions_but_not_indices() -> None:
    config = {
        "enabled": True,
        "method": "spatial_morphology",
        "seed": 23,
        "spatial": {
            "input_shape": [3, 32, 32],
            "region_mode": "random_windows",
            "minimum_window_shape": [4, 4],
            "region_seed": 101,
        },
    }
    options = {
        "out_features": 120,
        "in_features": 3072,
        "synapses_per_branch": 48,
        "owner_count": 1,
        "branch_factors": [120],
        "level_idx": 0,
    }
    excitatory = sample_configured_indices(config, pathway="ee", **options)
    inhibitory = sample_configured_indices(config, pathway="ie", **options)
    assert excitatory is not None
    assert inhibitory is not None
    assert not torch.equal(excitatory, inhibitory)
    # With all 48 RGB coordinates selected from each 4x4 window, sorting
    # removes the independent pathway-specific traversal and exposes the
    # shared spatial candidate set.
    assert torch.equal(
        excitatory.sort(dim=1).values,
        inhibitory.sort(dim=1).values,
    )


def _spatial_population_network(connection: dict, structured: dict):
    from dendritic_modeling.networks.architectures.factory import get_architecture

    layer = {
        "name": "layer0",
        "readout_population": "e",
        "population_defaults": {
            "ff_excitatory_synapses": 3,
            "ff_inhibitory_synapses": 0,
            "somatic_synapses": False,
            "topk_type": "indexed",
            "indexed_seed": 5,
            "structured_connectivity": structured,
        },
        "populations": [
            {
                "name": "e",
                "polarity": "excitatory",
                "n_neurons": 2,
                "branch_factors": [2, 2],
            }
        ],
        "connections": [{"source": "input", "target": "e", **connection}],
    }
    return get_architecture(
        "population_network", {"population_network": {"layers": [layer]}}, input_dim=64
    )


def test_exact_contact_rules_cannot_follow_connection_masks() -> None:
    import pytest

    model = _spatial_population_network({}, _spatial_config())
    indexed = [
        layer.branch_excitation
        for layer in model.modules()
        if getattr(layer, "branch_excitation", None) is not None
    ]
    assert indexed and all(
        synapses.connection_indices is not None for synapses in indexed
    )
    for structured in (
        _spatial_config(),
        {
            "enabled": True,
            "pathways": {"ee": {"method": "index_manifest", "manifest_path": "x.pt"}},
        },
    ):
        with pytest.raises(ValueError, match=r"places exact contacts.*use one of them"):
            _spatial_population_network({"probability": 0.5}, structured)
