"""The registered node-perturbation credit analyzer and its typed configuration."""

from __future__ import annotations

import json

import pytest
import torch
import yaml
from test_branch_local_sensitivity import _two_depth_classifier

from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.node_perturbation_credit import (
    NodePerturbationCreditAnalyzer,
    agreement,
)
from dendritic_modeling.config import NodePerturbationCreditAnalysisParams, load_config

PREFIX = "core_network.layers.0.excitatory_cells.branch_layers"


def _dataset(n=24, seed=0):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.rand(n, 10, generator=generator)
    labels = torch.randint(0, 4, (n,), generator=generator)
    return torch.utils.data.TensorDataset(inputs, labels)


def test_the_section_loads_from_a_typed_configuration(tmp_path):
    payload = {
        "data": {"dataset_name": "mnist"},
        "analysis": {
            "node_perturbation_credit": {
                "enabled": True,
                "params": {
                    "depths": [2, 1],
                    "coordinate": "voltage",
                    "dictionaries": ["independent", "siblings"],
                    "relative_scales": [0.01, 0.1],
                    "draws": [16, 1],
                    "match_perturbation_energy": True,
                },
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    section = load_config(str(path)).analysis.node_perturbation_credit
    assert section.enabled and not section.training
    assert section.params.depths == [1, 2]
    assert section.params.draws == [1, 16]
    assert section.params.coordinate == "voltage"


def test_defaults_and_invalid_settings():
    params = NodePerturbationCreditAnalysisParams()
    assert (params.depths, params.coordinate, params.dictionaries) == (
        [1],
        "output",
        ["independent"],
    )
    assert params.match_perturbation_energy is False and params.loss == "cross_entropy"
    for settings, message in (
        ({"coordinate": "conductance"}, "coordinate"),
        ({"dictionaries": ["siblings"], "depths": [0]}, "sibling"),
        ({"relative_scales": [0.0]}, "relative_scales"),
        ({"draws": [0]}, "draws"),
        ({"loss": "hinge"}, "loss"),
        ({"depths": [-1]}, "soma-relative"),
        ({"chunk": 0}, "chunk"),
    ):
        with pytest.raises(ValueError, match=message):
            NodePerturbationCreditAnalysisParams(**settings)


def test_the_analyzer_is_registered_as_final_only():
    spec = get_analyzer_spec("node_perturbation_credit")
    assert spec is not None and spec.config_field == "node_perturbation_credit"
    assert spec.supports_training is False
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(module_name_prefix=PREFIX)
    )
    with pytest.raises(ValueError, match="final-only"):
        analyzer.analyze(_two_depth_classifier(), _dataset(), training=True)


@pytest.mark.parametrize("coordinate", ["output", "voltage", "current"])
def test_the_analyzer_estimates_every_configured_case(tmp_path, coordinate):
    model = _two_depth_classifier().double()
    dataset = _dataset()
    dataset = torch.utils.data.TensorDataset(
        dataset.tensors[0].double(), dataset.tensors[1]
    )
    params = NodePerturbationCreditAnalysisParams(
        module_name_prefix=PREFIX,
        depths=[1, 2],
        coordinate=coordinate,
        dictionaries=["independent", "siblings", "neuron"],
        relative_scales=[0.01],
        draws=[4, 512],
        match_perturbation_energy=True,
        chunk=128,
    )
    result = NodePerturbationCreditAnalyzer(params).analyze(
        model, dataset, save_path=str(tmp_path), filename="credit"
    )
    saved = json.loads((tmp_path / "credit.json").read_text())
    assert saved["rows"] == result["rows"]
    assert (result["n_somas"], result["branch_factors"]) == (4, [2, 2])
    rows = result["rows"]
    # Per depth: independent once, siblings and neuron at both energies, two draw counts.
    assert len(rows) == 2 * (1 + 2 + 2) * 2
    for row in rows:
        assert row["evaluations_per_example"] == 2 * row["draws"]
        assert row["seconds_per_draw"] >= 0
        if row["dictionary"] != "independent":
            assert 0 <= row["projection_energy"] <= 1 + 1e-12
    best = [r for r in rows if r["dictionary"] == "independent" and r["draws"] == 512]
    assert all(r["cosine"] > 0.8 for r in best), best
    assert all(r["rank"] == 4 * 2 * (1 if r["depth"] == 1 else 2) for r in best)


def test_agreement_metrics_of_known_fields():
    exact = torch.tensor([[1.0, 0.0, 0.0, 2.0]], dtype=torch.float64)
    assert agreement(exact, exact, 2)["cosine"] == pytest.approx(1.0)
    flipped = agreement(-exact, exact, 2)
    assert flipped["cosine"] == pytest.approx(-1.0) and flipped["sign_agreement"] == 0.0
    projection = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    found = agreement(projection, exact, 2, projection)
    assert found["projection_energy"] == pytest.approx(0.2)
    assert found["cosine_to_projection"] == pytest.approx(1.0)
