"""Focused tests for synaptic turnover analysis helpers."""

from types import SimpleNamespace

import torch

from dendritic_modeling.analysis.tools.synaptic_turnover import (
    SynapticTurnoverAnalyzer,
    _update_turnover_counts,
)
from dendritic_modeling.networks import ExcitationInhibitionNetwork


def test_update_turnover_counts_matches_mask_semantics():
    prev_mask = torch.tensor([[1, 0, 1], [0, 0, 1]])
    curr_mask = torch.tensor([[1, 1, 0], [0, 1, 1]])

    active, stable, transient = _update_turnover_counts(
        prev_mask,
        curr_mask,
        total_active=10,
        total_stable=20,
        total_transient=30,
    )

    assert active == 14
    assert stable == 22
    assert transient == 32


def test_synaptic_turnover_analysis_counts_two_snapshots():
    analyzer = object.__new__(SynapticTurnoverAnalyzer)
    analyzer.enabled = True
    analyzer.synapse_config = {"ee": [3], "ei": [2], "ie": [], "ii": []}
    analyzer.data_dict_list = [
        {
            "layer0": {
                "exc_mask": torch.tensor([1, 0, 1]),
                "inh_mask": torch.tensor([0, 1]),
            }
        },
        {
            "layer0": {
                "exc_mask": torch.tensor([1, 1, 0]),
                "inh_mask": torch.tensor([1, 1]),
            }
        },
    ]
    model = SimpleNamespace(core_network=object.__new__(ExcitationInhibitionNetwork))

    results = analyzer.analyze(model, training=False)

    assert results["n_snapshots"] == 2
    assert results["incremental"]["exc"] == {
        "active": [2],
        "stable": [1],
        "transient": [1],
    }
    assert results["incremental"]["inh"] == {
        "active": [2],
        "stable": [1],
        "transient": [1],
    }
    assert results["incremental"]["both"] == {
        "active": [4],
        "stable": [2],
        "transient": [2],
    }
    assert results["cumulative"] == results["incremental"]
    assert results["layer_statistics"]["incremental"]["per_layer_avg"] == {
        "ee": [2.0],
        "ei": [2.0],
        "ie": [0.0],
        "ii": [0.0],
    }
    assert (
        results["layer_statistics"]["cumulative"]
        == results["layer_statistics"]["incremental"]
    )
