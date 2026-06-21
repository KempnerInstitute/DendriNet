from types import SimpleNamespace

import torch

from dendritic_modeling.analysis.tools.weight import WeightAnalyzer
from dendritic_modeling.config.analysis import WeightAnalysisParams


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
