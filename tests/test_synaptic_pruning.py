"""Focused tests for synaptic pruning analysis helpers."""

import contextlib
import io
from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.analysis.tools.synaptic_pruning import (
    SynapticPruningAnalyzer,
    _format_pruning_summary_text,
)
from dendritic_modeling.config import TransferFunctionConfig
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
    Identity,
)


def _make_tiny_ei_model() -> Classifier:
    transfer_params = TransferFunctionConfig().asdict()
    return Classifier(
        encoder_network=Identity(input_dim=5),
        core_network=ExcitationInhibitionNetwork(
            input_dim=5,
            transfer_params=transfer_params,
            excitatory_layer_sizes=[3],
            inhibitory_layer_sizes=[2],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            ee_synapses_per_branch_per_layer=[2],
            ei_synapses_per_branch_per_layer=[1],
            ie_synapses_per_branch_per_layer=[1],
            ii_synapses_per_branch_per_layer=[1],
        ),
        decoder_network=Identity(input_dim=3),
    )


def _first_branch_layer(model: Classifier) -> DendriticBranchLayer:
    for module in model.core_network.modules():
        if isinstance(module, DendriticBranchLayer):
            return module
    raise AssertionError("Expected tiny EI model to contain a DendriticBranchLayer")


def _capture_pruning_summary(analyzer: SynapticPruningAnalyzer) -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        analyzer.print_pruning_summary()
    return stream.getvalue()


def test_print_pruning_summary_preserves_empty_event_output():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    analyzer.pruning_events = []
    analyzer.initial_connectivity_stats = {}
    analyzer.data_dict_list = []

    expected = _format_pruning_summary_text(
        analyzer.pruning_events,
        analyzer.initial_connectivity_stats,
        analyzer.data_dict_list,
    )

    assert _capture_pruning_summary(analyzer) == f"{expected}\n"


def test_print_pruning_summary_formats_events_and_initial_connectivity():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    analyzer.pruning_events = [
        {
            "epoch": 2,
            "pruning_stats": {"pruned_params": 1000, "pruning_ratio": 0.25},
            "connectivity_changes": {"sparsity_increase": 0.125},
        },
        {
            "epoch": 4,
            "pruning_stats": {"pruned_params": 500, "pruning_ratio": 0.1},
            "connectivity_changes": {},
        },
    ]
    analyzer.initial_connectivity_stats = {
        "active_synapses": 9000,
        "total_synapses": 12000,
        "overall_sparsity": 0.25,
    }
    analyzer.data_dict_list = [{"epoch": 0}]

    expected = "\n".join(
        [
            "",
            "=" * 60,
            "SYNAPTIC PRUNING ANALYSIS SUMMARY",
            "=" * 60,
            "Total pruning events: 2",
            "Total synapses pruned: 1,500",
            "Average pruning ratio per event: 0.1750",
            "",
            "Initial connectivity:",
            "  Active synapses: 9,000 / 12,000",
            "  Initial sparsity: 0.2500",
            "",
            "Pruning events:",
            f"{'Epoch':<8} {'Pruned':<12} {'Ratio':<8} {'Sparsity Inc':<12}",
            "-" * 45,
            f"{2:<8} {1000:<12,} {0.25:<8.4f} {0.125:<12.4f}",
            f"{4:<8} {500:<12,} {0.1:<8.4f} {0:<12.4f}",
            "=" * 60,
            "",
        ]
    )

    assert _format_pruning_summary_text(
        analyzer.pruning_events,
        analyzer.initial_connectivity_stats,
        analyzer.data_dict_list,
    ) == expected.rstrip("\n")
    assert _capture_pruning_summary(analyzer) == expected


def test_latest_post_pruning_stats_handles_empty_events():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    analyzer.pruning_events = []

    assert analyzer._latest_post_pruning_stats() is None


def test_latest_post_pruning_stats_handles_missing_post_stats():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    analyzer.pruning_events = [{"epoch": 1, "pruning_stats": {}}]

    assert analyzer._latest_post_pruning_stats() is None


def test_latest_post_pruning_stats_returns_latest_event_stats():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    older_stats = {"active_synapses": 10}
    latest_stats = {"active_synapses": 7}
    analyzer.pruning_events = [
        {"epoch": 1, "post_pruning_stats": older_stats},
        {"epoch": 2, "post_pruning_stats": latest_stats},
    ]

    assert analyzer._latest_post_pruning_stats() is latest_stats


def test_compute_pruning_changes_preserves_connectivity_arithmetic():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    pre_stats = {
        "active_synapses": 10,
        "overall_sparsity": 0.2,
        "layer_stats": {
            "layer_0": {"active_synapses": 8, "total_synapses": 10},
            "layer_1": {"active_synapses": 0, "total_synapses": 0},
        },
        "branch_stats": {
            "layer_0_branch_0": {"active": 6, "sparsity": 0.25},
        },
        "connection_type_stats": {
            "excitatory": {"active": 6},
            "inhibitory": {"active": 4},
        },
    }
    post_stats = {
        "active_synapses": 6,
        "overall_sparsity": 0.5,
        "layer_stats": {
            "layer_0": {"active_synapses": 4, "total_synapses": 10},
            "layer_1": {"active_synapses": 0, "total_synapses": 0},
        },
        "branch_stats": {
            "layer_0_branch_0": {"active": 3, "sparsity": 0.625},
        },
        "connection_type_stats": {
            "excitatory": {"active": 3},
            "inhibitory": {"active": 3},
        },
    }

    changes = analyzer._compute_pruning_changes(pre_stats, post_stats)

    assert changes["total_synapses_lost"] == 4
    assert changes["sparsity_increase"] == pytest.approx(0.3)
    assert changes["layer_changes"]["layer_0"] == {
        "synapses_lost": 4,
        "sparsity_increase": pytest.approx(0.4),
    }
    assert changes["layer_changes"]["layer_1"] == {
        "synapses_lost": 0,
        "sparsity_increase": 0,
    }
    assert changes["branch_changes"]["layer_0_branch_0"] == {
        "synapses_lost": 3,
        "sparsity_increase": pytest.approx(0.375),
    }
    assert changes["connection_type_changes"]["excitatory"] == {
        "synapses_lost": 3,
        "pruning_ratio": pytest.approx(0.5),
    }
    assert changes["connection_type_changes"]["inhibitory"] == {
        "synapses_lost": 1,
        "pruning_ratio": pytest.approx(0.25),
    }


def test_connectivity_stats_use_real_branch_weight_masks():
    model = _make_tiny_ei_model()
    analyzer = object.__new__(SynapticPruningAnalyzer)
    branch_layer = _first_branch_layer(model)
    exc_mask = branch_layer.branch_excitation.weight_mask()[0]
    inh_mask = branch_layer.branch_inhibition.weight_mask()[0]

    stats = analyzer._compute_connectivity_stats(model)

    branch_stats = stats["layer_stats"]["layer_0"]["branch_stats"]["layer_0_branch_0"]
    assert branch_stats["excitatory"] == {
        "total": exc_mask.numel(),
        "active": exc_mask.sum().item(),
    }
    assert branch_stats["inhibitory"] == {
        "total": inh_mask.numel(),
        "active": inh_mask.sum().item(),
    }
    assert (
        stats["branch_stats"]["layer_0_branch_0"]["excitatory"]
        == branch_stats["excitatory"]
    )
    assert (
        stats["branch_stats"]["layer_0_branch_0"]["inhibitory"]
        == branch_stats["inhibitory"]
    )
    assert stats["total_synapses"] > 0
    assert 0 <= stats["active_synapses"] <= stats["total_synapses"]
    assert 0 <= stats["overall_sparsity"] <= 1


def test_extract_synaptic_masks_returns_cloned_branch_masks():
    model = _make_tiny_ei_model()
    analyzer = object.__new__(SynapticPruningAnalyzer)
    branch_layer = _first_branch_layer(model)
    original_exc_mask = branch_layer.branch_excitation.weight_mask()[0].clone()

    masks = analyzer._extract_synaptic_masks(model)

    assert torch.equal(masks["layer_0_branch_0"]["exc_mask"], original_exc_mask)
    masks["layer_0_branch_0"]["exc_mask"].zero_()
    assert torch.equal(
        branch_layer.branch_excitation.weight_mask()[0], original_exc_mask
    )


def test_sparsity_patterns_use_real_branch_weight_masks():
    model = _make_tiny_ei_model()
    analyzer = object.__new__(SynapticPruningAnalyzer)
    stats = analyzer._compute_connectivity_stats(model)

    patterns = analyzer._analyze_sparsity_patterns(model)

    expected_sparsities = [
        branch_stats["sparsity"] for branch_stats in stats["branch_stats"].values()
    ]
    assert patterns["branch_sparsities"] == pytest.approx(expected_sparsities)
    assert patterns["layer_sparsities"]
    assert patterns["sparsity_distribution"]["mean"] == pytest.approx(
        sum(expected_sparsities) / len(expected_sparsities)
    )


def test_non_ei_models_keep_empty_pruning_connectivity_outputs():
    model = SimpleNamespace(core_network=object())
    analyzer = object.__new__(SynapticPruningAnalyzer)

    assert analyzer._extract_synaptic_masks(model) == {}
    assert analyzer._compute_connectivity_stats(model) == {
        "total_synapses": 0,
        "active_synapses": 0,
        "layer_stats": {},
        "branch_stats": {},
        "connection_type_stats": {
            "excitatory": {"total": 0, "active": 0},
            "inhibitory": {"total": 0, "active": 0},
        },
    }
    assert analyzer._analyze_sparsity_patterns(model) == {
        "layer_sparsities": [],
        "branch_sparsities": [],
        "sparsity_distribution": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    }
