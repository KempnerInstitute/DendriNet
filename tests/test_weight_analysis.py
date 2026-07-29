from types import SimpleNamespace

import torch

from dendritic_modeling.analysis.tools.weight import (
    WeightAnalyzer,
    _branch_synapse_statistics,
    _mean_pairwise_jaccard,
    _source_support_statistics,
    _weight_tensor_statistics,
)
from dendritic_modeling.config.analysis import WeightAnalysisParams
from dendritic_modeling.networks import DendriticBranchLayer


def test_weight_tensor_statistics_helper_matches_current_reductions():
    weights = torch.tensor([1.0, 3.0, 5.0])

    assert _weight_tensor_statistics(
        weights,
        compute_mean=True,
        compute_variance=True,
        compute_min_max=True,
        compute_percentiles=True,
    ) == {
        "mean": weights.mean().item(),
        "variance": weights.var().item(),
        "std": weights.std().item(),
        "min": weights.min().item(),
        "max": weights.max().item(),
        "median": weights.median().item(),
        "q1": weights.quantile(0.25).item(),
        "q3": weights.quantile(0.75).item(),
    }


def test_weight_tensor_statistics_helper_respects_disabled_fields():
    weights = torch.tensor([1.0, 3.0, 5.0])

    assert _weight_tensor_statistics(
        weights,
        compute_mean=False,
        compute_variance=False,
        compute_min_max=True,
        compute_percentiles=False,
    ) == {
        "min": weights.min().item(),
        "max": weights.max().item(),
    }


def test_branch_synapse_statistics_helper_matches_active_weight_reductions():
    active = torch.tensor([0.2, 0.5])

    assert _branch_synapse_statistics(
        "exc",
        active,
        compute_mean=True,
        compute_variance=True,
        compute_min_max=True,
        compute_percentiles=True,
    ) == {
        "exc_n_active_synapses": len(active),
        "exc_weight_mean": active.mean().item(),
        "exc_weight_var": active.var().item(),
        "exc_weight_min": active.min().item(),
        "exc_weight_max": active.max().item(),
        "exc_weight_median": active.median().item(),
    }


def test_branch_synapse_statistics_helper_records_count_only_for_empty_or_disabled():
    assert _branch_synapse_statistics(
        "inh",
        torch.tensor([]),
        compute_mean=True,
        compute_variance=True,
        compute_min_max=True,
        compute_percentiles=True,
    ) == {"inh_n_active_synapses": 0}
    assert _branch_synapse_statistics(
        "exc",
        torch.tensor([0.2, 0.5]),
        compute_mean=False,
        compute_variance=False,
        compute_min_max=False,
        compute_percentiles=False,
    ) == {"exc_n_active_synapses": 2}


def test_weight_branch_synapse_statistics_match_tensor_reductions():
    analyzer = WeightAnalyzer(WeightAnalysisParams())
    weights = torch.tensor([0.0, 0.2, 0.5, 1e-7])
    active = weights[weights > analyzer.weight_threshold]
    branch_stats = {}

    analyzer._record_branch_synapse_statistics(branch_stats, "exc", weights)

    assert branch_stats["exc_n_active_synapses"] == len(active)
    assert branch_stats["exc_weight_mean"] == active.mean().item()
    assert branch_stats["exc_weight_var"] == active.var().item()
    assert branch_stats["exc_weight_min"] == active.min().item()
    assert branch_stats["exc_weight_max"] == active.max().item()
    assert branch_stats["exc_weight_median"] == active.median().item()


def test_weight_branch_synapse_statistics_records_zero_active_count():
    analyzer = WeightAnalyzer(WeightAnalysisParams(weight_threshold=1.0))
    branch_stats = {"branch_idx": 0}

    analyzer._record_branch_synapse_statistics(
        branch_stats, "inh", torch.tensor([0.0, 0.2, 0.5])
    )

    assert branch_stats == {"branch_idx": 0, "inh_n_active_synapses": 0}


def test_weight_branch_synapse_statistics_respects_disabled_metrics():
    analyzer = WeightAnalyzer(
        WeightAnalysisParams(
            compute_mean=False,
            compute_variance=False,
            compute_percentiles=False,
            compute_min_max=False,
        )
    )
    branch_stats = {}

    analyzer._record_branch_synapse_statistics(
        branch_stats, "exc", torch.tensor([0.2, 0.5])
    )

    assert branch_stats == {"exc_n_active_synapses": 2}


def test_weight_active_weights_uses_configured_threshold():
    analyzer = WeightAnalyzer(WeightAnalysisParams(weight_threshold=0.25))

    active = analyzer._active_weights(torch.tensor([0.0, 0.25, 0.3, 0.7]))

    torch.testing.assert_close(active, torch.tensor([0.3, 0.7]))


def test_source_support_helpers_preserve_branch_level_support():
    mask = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    assert _mean_pairwise_jaccard(mask) == 1.0 / 3.0
    stats = _source_support_statistics(mask, torch.ones_like(mask))

    assert stats["source_coverage_fraction"] == 0.75
    assert stats["eligible_source_fraction"] == 1.0
    assert stats["source_coverage_of_eligible"] == 0.75
    assert stats["mean_pairwise_branch_jaccard"] == 1.0 / 3.0
    assert 0.0 < stats["normalized_source_entropy"] <= 1.0


def test_weight_source_support_emits_exact_profiles_and_audit_plot(tmp_path):
    analyzer = WeightAnalyzer(
        WeightAnalysisParams(
            compute_source_support=True,
            source_support_image_shape=(2, 2),
            source_support_plot_soma_indices=[0, 1],
            source_support_fig_save_format="png",
        )
    )
    branch_layer = DendriticBranchLayer(
        output_dim=4,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=1,
        inhibitory_input_dim=4,
        inhibitory_synapses_per_branch=1,
        reactivate=False,
        topk_noise_level=0.0,
        layer_idx=1,
    )
    population = SimpleNamespace(n_soma=2, branch_layers=[branch_layer])
    population_definition = SimpleNamespace(name="e0", polarity="excitatory")
    model = SimpleNamespace(
        core_network=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    config=SimpleNamespace(name="layer0"),
                    populations={"e0": population},
                    population_definitions=[population_definition],
                )
            ]
        )
    )

    summary, profiles, profile_metadata = analyzer._compute_source_support(model)

    assert summary["n_scalar_records"] == 8
    assert summary["n_profile_arrays"] == 8
    assert {row["depth_label"] for row in summary["records"]} == {"1", "all"}
    assert {row["pathway"] for row in summary["records"]} == {
        "ff_excitatory",
        "ff_inhibitory",
    }
    for metadata in profile_metadata:
        frequency = profiles[metadata["contact_frequency_key"]]
        assert frequency.shape == (2, 4)
        torch.testing.assert_close(
            torch.from_numpy(frequency.sum(axis=1)),
            torch.ones(2, dtype=torch.float64),
        )

    analyzer._save_source_support_outputs(
        summary,
        profiles,
        profile_metadata,
        save_path=str(tmp_path),
        filename="final",
        plot=True,
    )

    assert (tmp_path / "final_source_support_profiles.npz").exists()
    assert (tmp_path / "final_source_support_records.csv").exists()
    assert len(summary["figure_files"]) == 2
    assert all((tmp_path / name).exists() for name in summary["figure_files"])


def test_weight_branch_layer_iterator_preserves_layer_idx_rules():
    analyzer = WeightAnalyzer(WeightAnalysisParams())
    branch_with_idx = SimpleNamespace(layer_idx=7)
    branch_without_idx = SimpleNamespace()
    dendrinet = SimpleNamespace(branch_layers=[branch_without_idx, branch_with_idx])

    fallback_layers = list(
        analyzer._iter_dendrinet_branch_layers(dendrinet, require_layer_idx=False)
    )
    required_layers = list(
        analyzer._iter_dendrinet_branch_layers(dendrinet, require_layer_idx=True)
    )

    assert fallback_layers == [(0, branch_without_idx), (7, branch_with_idx)]
    assert required_layers == [(7, branch_with_idx)]


def test_weight_enhanced_branch_accumulator_matches_active_weight_reductions():
    analyzer = WeightAnalyzer(WeightAnalysisParams(weight_threshold=0.1))
    branch_layer = SimpleNamespace(
        branch_excitation=SimpleNamespace(
            pruned_weight=lambda: torch.tensor([[0.0, 0.2], [0.3, 0.05]])
        ),
        branch_inhibition=SimpleNamespace(
            pruned_weight=lambda: torch.tensor([[0.4, 0.0], [0.2, 0.1]])
        ),
    )
    depth_to_strength = {}
    depth_to_weights = {}

    analyzer._accumulate_enhanced_branch_statistics(
        branch_layer, 3, depth_to_strength, depth_to_weights
    )

    expected_active = torch.tensor([0.2, 0.3, 0.4, 0.2])
    assert depth_to_strength == {3: expected_active.sum().item()}
    assert len(depth_to_weights[3]) == 2
    torch.testing.assert_close(torch.cat(depth_to_weights[3]), expected_active)


def test_weight_branch_level_statistics_include_inhibitory_populations():
    analyzer = WeightAnalyzer(WeightAnalysisParams())
    excitatory = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=1,
        inhibitory_input_dim=3,
        inhibitory_synapses_per_branch=1,
        reactivate=False,
        topk_noise_level=0.0,
        layer_idx=1,
    )
    inhibitory = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=3,
        excitatory_synapses_per_branch=1,
        inhibitory_input_dim=3,
        inhibitory_synapses_per_branch=1,
        reactivate=False,
        topk_noise_level=0.0,
        layer_idx=1,
    )
    model = SimpleNamespace(
        core_network=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    e_population=SimpleNamespace(branch_layers=[excitatory]),
                    i_population=SimpleNamespace(branch_layers=[inhibitory]),
                )
            ]
        )
    )

    stats = analyzer._compute_branch_level_statistics(model)
    branches = stats["branch_layers.1"]["branches"]

    assert len(branches) == excitatory.n_branches + inhibitory.n_branches
