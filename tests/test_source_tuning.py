"""Tests for split-safe source-tuning/support alignment."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.source_tuning import (
    SourceTuningOptions,
    SourceTuningSupportAnalyzer,
    StreamingClassFeatureStatistics,
    centered_class_profile_alignment,
    inventory_population_network_source_support,
    summarize_support_alignment,
)
from dendritic_modeling.networks.architectures.recurrent import (
    PopulationLayerConfig,
    PopulationNetwork,
    PopulationNetworkConfig,
)


def _feedforward_layer(
    name: str,
    source_dim: int,
    *,
    branch_factors: list[int] | None = None,
) -> PopulationLayerConfig:
    population_name = f"e{name[-1]}"
    return PopulationLayerConfig(
        name=name,
        recurrent=False,
        readout_population=population_name,
        population_defaults={
            "ff_excitatory_synapses": min(2, source_dim),
            "ff_inhibitory_synapses": min(2, source_dim),
            "somatic_synapses": True,
            "reactivate": False,
            "dbl_init_method": "default",
            "weight_transform": "exp",
            "topk_type": "standard",
        },
        populations=[
            {
                "name": population_name,
                "polarity": "excitatory",
                "n_neurons": 2,
                "branch_factors": branch_factors or [1],
            }
        ],
        connections=[
            {
                "source": "input_e",
                "target": population_name,
                "pathway": "ff_excitatory",
            },
            {
                "source": "input_i",
                "target": population_name,
                "pathway": "ff_inhibitory",
            },
        ],
    )


def _two_layer_network() -> PopulationNetwork:
    torch.manual_seed(11)
    return PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=4,
            layers=[
                _feedforward_layer("layer0", source_dim=4),
                _feedforward_layer("layer1", source_dim=2),
            ],
        )
    )


def _dataset(offset: float) -> TensorDataset:
    inputs = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.8, 0.0, 0.9, 0.1],
            [0.0, 1.0, 0.0, 1.0],
            [0.1, 0.9, 0.2, 0.8],
        ]
    )
    inputs = inputs + offset
    return TensorDataset(inputs, torch.tensor([0, 0, 1, 1]))


def test_streaming_class_feature_statistics_matches_direct_group_reductions():
    stats = StreamingClassFeatureStatistics(feature_dim=2, activity_threshold=1.0)
    values = torch.tensor([[1.0, 0.0], [3.0, 2.0], [0.0, 2.0], [2.0, 4.0]])
    labels = torch.tensor([0, 0, 1, 1])

    stats.update(values[:2], labels[:2])
    stats.update(values[2:], labels[2:])
    result = stats.preferred_vs_rest(0)

    torch.testing.assert_close(
        result["preferred_mean"], torch.tensor([2.0, 1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["rest_mean"], torch.tensor([1.0, 3.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["mean_contrast"], torch.tensor([1.0, -2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["preferred_activity_probability"],
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result["rest_activity_probability"],
        torch.tensor([0.5, 1.0], dtype=torch.float64),
    )
    assert stats.total_count == 4
    assert stats.classes == (0, 1)
    assert all(value.ndim == 1 for value in stats.sums.values())


def test_class_conditional_means_do_not_weight_classes_by_sample_count():
    stats = StreamingClassFeatureStatistics(feature_dim=1, activity_threshold=0.0)
    stats.update(
        torch.tensor([[1.0], [3.0], [3.0], [3.0], [5.0], [7.0]]),
        torch.tensor([0, 1, 1, 1, 2, 2]),
    )

    class_ids, means = stats.class_conditional_means()

    assert class_ids == (0, 1, 2)
    torch.testing.assert_close(
        means[:, 0],
        torch.tensor([1.0, 3.0, 6.0], dtype=torch.float64),
    )


def test_support_alignment_separates_uniform_contact_and_conductance_weighting():
    tuning = {
        "n_preferred": 3,
        "n_rest": 4,
        "preferred_mean": torch.tensor([3.0, 2.0, 1.0]),
        "rest_mean": torch.tensor([1.0, 1.0, 1.0]),
        "mean_contrast": torch.tensor([2.0, 1.0, 0.0]),
        "preferred_activity_probability": torch.tensor([1.0, 0.5, 0.0]),
        "rest_activity_probability": torch.tensor([0.0, 0.5, 0.0]),
        "activity_probability_contrast": torch.tensor([1.0, 0.0, 0.0]),
        "standardized_mean_difference": torch.tensor([4.0, 2.0, 0.0]),
        "class_profile_alignment": torch.tensor([1.0, 0.0, -1.0]),
    }

    result = summarize_support_alignment(
        tuning=tuning,
        candidate_frequency=torch.ones(3),
        contact_frequency=torch.tensor([1.0, 0.0, 0.0]),
        effective_conductance=torch.tensor([0.0, 2.0, 0.0]),
        epsilon=1e-12,
    )

    assert result["source_uniform_mean_contrast"] == 1.0
    assert result["contact_weighted_mean_contrast"] == 2.0
    assert result["conductance_weighted_mean_contrast"] == 1.0
    assert result["contact_weighted_mean_contrast_excess_over_uniform"] == 1.0
    assert result["conductance_weighted_mean_contrast_excess_over_uniform"] == 0.0
    assert result["source_uniform_class_profile_alignment"] == 0.0
    assert result["contact_weighted_class_profile_alignment"] == 1.0
    assert result["conductance_weighted_class_profile_alignment"] == 0.0
    assert result["contact_weighted_class_profile_alignment_excess_over_uniform"] == 1.0


def test_centered_class_profile_alignment_is_class_balanced_and_degeneracy_safe():
    target = torch.tensor([0.0, 1.0, 2.0])
    sources = torch.tensor(
        [
            [0.0, 2.0, 4.0],
            [1.0, 1.0, 4.0],
            [2.0, 0.0, 4.0],
        ]
    )

    alignment, target_degenerate, source_degenerate = centered_class_profile_alignment(
        target, sources, epsilon=1e-12
    )

    torch.testing.assert_close(
        alignment,
        torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64),
    )
    assert target_degenerate is False
    assert source_degenerate.tolist() == [False, False, True]

    zero_alignment, target_degenerate, source_degenerate = (
        centered_class_profile_alignment(
            torch.ones(3),
            sources,
            epsilon=1e-12,
        )
    )
    assert target_degenerate is True
    assert source_degenerate.tolist() == [False, False, True]
    torch.testing.assert_close(zero_alignment, torch.zeros(3, dtype=torch.float64))


def test_inventory_labels_only_declared_first_layer_sources_as_pixels():
    network = _two_layer_network()
    options = SourceTuningOptions(
        image_shape=(2, 2),
        image_source_layers=(0,),
    )

    inventory = inventory_population_network_source_support(network, options)

    first_layer = [
        block for block in inventory.source_blocks if block.network_layer_index == 0
    ]
    second_layer = [
        block for block in inventory.source_blocks if block.network_layer_index == 1
    ]
    assert first_layer and second_layer
    assert {block.coordinate_space for block in first_layer} == {"image_pixels"}
    assert {block.coordinate_space for block in second_layer} == {
        "native_source_neurons"
    }
    assert {block.source_kind for block in second_layer} == {
        "previous_layer_readout_population"
    }
    assert {block.source_population_name for block in second_layer} == {"e0"}
    assert {block.source_dim for block in first_layer} == {4}
    assert {block.source_dim for block in second_layer} == {2}
    assert {block.image_shape for block in first_layer} == {(2, 2)}
    assert {block.image_shape for block in second_layer} == {None}
    assert {tuple(block.metadata()["image_shape"]) for block in first_layer} == {(2, 2)}


def test_inventory_accepts_channel_aware_image_shape_and_checks_product():
    network = PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=12,
            layers=[_feedforward_layer("layer0", source_dim=12)],
        )
    )
    options = SourceTuningOptions(
        image_shape=[3, 2, 2],  # type: ignore[arg-type]
        image_source_layers=(0,),
    )

    inventory = inventory_population_network_source_support(network, options)

    assert options.image_shape == (3, 2, 2)
    assert {block.source_dim for block in inventory.source_blocks} == {12}
    assert {block.image_shape for block in inventory.source_blocks} == {(3, 2, 2)}
    assert {
        tuple(block.metadata()["image_shape"]) for block in inventory.source_blocks
    } == {(3, 2, 2)}

    mismatched = SourceTuningOptions(
        image_shape=(3, 2, 3),
        image_source_layers=(0,),
    )
    with pytest.raises(ValueError, match="has 18 pixels"):
        inventory_population_network_source_support(network, mismatched)


def test_analyzer_streams_independent_splits_and_retains_only_coordinate_profiles():
    network = _two_layer_network()
    analyzer = SourceTuningSupportAnalyzer(
        SourceTuningOptions(
            image_shape=(2, 2),
            image_source_layers=(0,),
            image_binarization_threshold=0.5,
            continuous_activity_threshold=0.0,
            retain_coordinate_profiles=True,
        )
    )

    results = analyzer.analyze(
        network,
        _dataset(0.02),
        reference_data=_dataset(0.0),
        reference_split_name="validation",
        evaluation_split_name="test",
    )

    assert results["reference"]["split"] == "validation"
    assert results["reference"]["n_samples"] == 4
    assert results["evaluation"]["split"] == "test"
    assert results["evaluation"]["n_samples"] == 4
    assert results["n_alignment_records"] > 0
    assert results["require_independent_splits"] is True
    assert "class_profile_alignment" in results["quantity_contract"]
    assert "class_profile_degeneracy" in results["quantity_contract"]
    for row in results["alignment_records"]:
        assert row["n_class_profile_classes"] == 2
        assert row["n_degenerate_target_class_profiles"] in {0, 1}
        assert 0 <= row["n_degenerate_source_class_profiles"] <= row["source_dim"]
        assert (
            0
            <= row["n_eligible_degenerate_source_class_profiles"]
            <= row["n_eligible_source_coordinates"]
        )
        value = row["source_uniform_class_profile_alignment"]
        assert value is None or -1.0 <= value <= 1.0
    assert results["coordinate_profiles"]
    layer_spaces = {
        (row["network_layer_index"], row["coordinate_space"])
        for row in results["alignment_records"]
    }
    assert (0, "image_pixels") in layer_spaces
    assert (1, "native_source_neurons") in layer_spaces
    assert {
        tuple(row["image_shape"])
        for row in results["alignment_records"]
        if row["network_layer_index"] == 0
    } == {(2, 2)}
    assert all(
        row["image_shape"] is None
        for row in results["alignment_records"]
        if row["network_layer_index"] == 1
    )
    for profile in results["coordinate_profiles"].values():
        assert all(
            not isinstance(value, torch.Tensor) or value.ndim == 1
            for value in profile.values()
        )


def test_analyzer_rejects_same_object_for_reference_and_evaluation():
    network = _two_layer_network()
    dataset = _dataset(0.0)
    analyzer = SourceTuningSupportAnalyzer(
        SourceTuningOptions(image_shape=(2, 2), image_source_layers=(0,))
    )

    with pytest.raises(ValueError, match="independent objects"):
        analyzer.analyze(network, dataset, reference_data=dataset)


def test_analyzer_allows_same_split_when_independence_policy_is_disabled():
    network = _two_layer_network()
    dataset = _dataset(0.0)
    analyzer = SourceTuningSupportAnalyzer(
        SourceTuningOptions(
            image_shape=(2, 2),
            image_source_layers=(0,),
            require_independent_splits=False,
        )
    )

    result = analyzer.analyze(
        network,
        dataset,
        reference_data=dataset,
        reference_split_name="validation",
        evaluation_split_name="validation",
    )

    assert result["require_independent_splits"] is False
    assert result["reference"]["split"] == result["evaluation"]["split"]


def test_source_statistics_are_streamed_once_and_reused_across_tree_depths():
    network = PopulationNetwork(
        PopulationNetworkConfig(
            input_dim=4,
            layers=[
                _feedforward_layer(
                    "layer0",
                    source_dim=4,
                    branch_factors=[2, 2],
                )
            ],
        )
    )
    analyzer = SourceTuningSupportAnalyzer(
        SourceTuningOptions(image_shape=(2, 2), image_source_layers=(0,))
    )
    inventory = inventory_population_network_source_support(network, analyzer.options)

    stats = analyzer._collect_source_tuning(
        network,
        _dataset(0.0),
        inventory,
        torch.device("cpu"),
        analyzer.runtime,
    )

    assert len(inventory.source_blocks) > len({id(value) for value in stats.values()})
    by_pathway: dict[str, list[int]] = {}
    for block in inventory.source_blocks:
        by_pathway.setdefault(block.pathway, []).append(id(stats[block.key]))
    assert all(len(set(stat_ids)) == 1 for stat_ids in by_pathway.values())
    assert {value.total_count for value in stats.values()} == {4}
