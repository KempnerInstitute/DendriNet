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
from dendritic_modeling.config.analysis import (
    AnalysisConfig,
    EvaluationRuntimeConfig,
    fill_analysis_seeds,
)

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
    assert (params.max_samples, params.chunk, params.seed) == (256, 16, None)
    for settings, message in (
        ({"coordinate": "conductance"}, "coordinate"),
        ({"dictionaries": ["siblings"], "depths": [0]}, "sibling"),
        ({"dictionaries": ["neuron", "neuron"]}, "repeated"),
        ({"relative_scales": [0.0]}, "relative_scales"),
        ({"relative_scales": [0.1, 0.1]}, "repeated"),
        ({"draws": [0]}, "draws"),
        ({"draws": [4, 4]}, "repeated"),
        ({"loss": "hinge"}, "loss"),
        ({"depths": [-1]}, "depths"),
        ({"depths": []}, "at least one value"),
        ({"max_samples": 0}, "max_samples"),
        ({"chunk": 0}, "chunk"),
    ):
        with pytest.raises(ValueError, match=message):
            NodePerturbationCreditAnalysisParams(**settings)
    for settings, message in (
        ({"draws": 16}, "expected a list"),
        ({"draws": [1.5]}, "expected an integer"),
        ({"draws": [True]}, "expected an integer"),
        ({"relative_scales": [True]}, "numbers"),
        ({"match_perturbation_energy": "yes"}, "true or false"),
        ({"seed": True}, "seed"),
    ):
        with pytest.raises(TypeError, match=message):
            NodePerturbationCreditAnalysisParams(**settings)


def test_an_unset_seed_follows_the_analysis_seed_policy():
    config = AnalysisConfig()
    filled = fill_analysis_seeds(config, evaluation_seed=None, probe_seed=11)
    path = "node_perturbation_credit.params.seed"
    assert config.node_perturbation_credit.params.seed == filled[path]


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
    assert (result["seed"], result["device"], result["dtype"]) == (0, "cpu", "float64")
    assert (result["chunk"], result["examples_per_forward"]) == (128, 256)
    assert len({row["seed"] for row in result["rows"]}) == 2 * (1 + 2 + 2)
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


def _double_dataset():
    dataset = _dataset()
    return torch.utils.data.TensorDataset(
        dataset.tensors[0].double(), dataset.tensors[1]
    )


def test_batches_of_examples_give_the_same_estimates():
    model = _two_depth_classifier().double()
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(
            module_name_prefix=PREFIX,
            dictionaries=["independent", "siblings"],
            draws=[4, 16],
            seed=5,
        )
    )
    whole = analyzer.analyze(model, _double_dataset())
    parts = analyzer.analyze(
        model, _double_dataset(), runtime=EvaluationRuntimeConfig(batch_size=5)
    )
    assert parts["examples_per_forward"] == 5 and parts["n_examples"] == 24
    for a, b in zip(whole["rows"], parts["rows"], strict=True):
        for key, value in a.items():
            if key == "seconds_per_draw":
                continue
            if isinstance(value, float):
                assert b[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
            else:
                assert b[key] == value, key


class _Twice(torch.nn.Module):
    """Runs its model twice per forward, so every level runs twice."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return 0.5 * (self.inner(x) + self.inner(x))


def test_surrogate_credit_and_repeated_levels_are_refused():
    model = _two_depth_classifier()
    model.core_network.autograd_credit_mode = "soma_broadcast"
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(module_name_prefix=PREFIX)
    )
    with pytest.raises(ValueError, match="autograd credit"):
        analyzer.analyze(model, _dataset())
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(module_name_prefix=f"inner.{PREFIX}")
    )
    with pytest.raises(ValueError, match="ran 2 times"):
        analyzer.analyze(_Twice(_two_depth_classifier()), _dataset())


@pytest.mark.requires_gpu
def test_structured_dictionaries_run_on_an_unindexed_cuda_device():
    model = _two_depth_classifier()
    params = NodePerturbationCreditAnalysisParams(
        module_name_prefix=PREFIX,
        depths=[1, 2],
        dictionaries=["independent", "siblings", "neuron"],
        draws=[4],
    )
    result = NodePerturbationCreditAnalyzer(params).analyze(
        model, _dataset(), device="cuda"
    )
    assert result["device"].startswith("cuda:")
    assert len(result["rows"]) == 2 * 3
    assert next(model.parameters()).device.type == "cpu"
