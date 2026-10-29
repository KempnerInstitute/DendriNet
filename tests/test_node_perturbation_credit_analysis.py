"""The registered node-perturbation credit analyzer and its typed configuration."""

from __future__ import annotations

import json
import math

import pytest
import torch
import yaml
from test_branch_local_sensitivity import _two_depth_classifier

from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.node_perturbation_credit import (
    NodePerturbationCreditAnalyzer,
    agreement,
)
from dendritic_modeling.config import (
    NodePerturbationCreditAnalysisParams,
    TransferFunctionConfig,
    load_config,
)
from dendritic_modeling.config.analysis import (
    AnalysisConfig,
    EvaluationRuntimeConfig,
    fill_analysis_seeds,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
    Identity,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_rule_override import (
    BranchRuleOverride,
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
    assert (result["chunk"], result["rows_per_forward"]) == (128, 256)
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
    assert parts["rows_per_forward"] == 5 and parts["n_examples"] == 24
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


def test_forward_passes_stay_within_the_batch_size():
    model = _two_depth_classifier().double()
    sizes = []
    handle = model.register_forward_pre_hook(
        lambda _module, args, sizes=sizes: sizes.append(int(args[0].shape[0]))
    )
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(
            module_name_prefix=PREFIX, draws=[4, 8], chunk=4
        )
    )
    try:
        result = analyzer.analyze(
            model, _double_dataset(), runtime=EvaluationRuntimeConfig(batch_size=5)
        )
    finally:
        handle.remove()
    assert result["rows_per_forward"] == 5
    assert sizes and max(sizes) == 5


class _Column(torch.nn.Module):
    """The first output of a model, as one value per example or one column."""

    def __init__(self, inner, squeeze):
        super().__init__()
        self.inner, self.squeeze = inner, squeeze

    def forward(self, x):
        value = self.inner(x)[:, :1]
        return value[:, 0] if self.squeeze else value


def test_one_output_per_example_gives_the_estimates_of_one_column():
    inner = _two_depth_classifier().double()
    found = {}
    for squeeze in (True, False):
        params = NodePerturbationCreditAnalysisParams(
            module_name_prefix=f"inner.{PREFIX}", loss="mse", draws=[4, 16], seed=3
        )
        found[squeeze] = NodePerturbationCreditAnalyzer(params).analyze(
            _Column(inner, squeeze), _double_dataset()
        )["rows"]
    for a, b in zip(found[True], found[False], strict=True):
        assert {k: v for k, v in a.items() if k != "seconds_per_draw"} == {
            k: v for k, v in b.items() if k != "seconds_per_draw"
        }


def test_signal_rounded_models_are_refused():
    model = _two_depth_classifier()
    layer = next(m for m in model.modules() if isinstance(m, DendriticBranchLayer))
    # The guard reads the public option, which every rounding module carries.
    layer.signal_rounding = "bfloat16"
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(module_name_prefix=PREFIX)
    )
    with pytest.raises(ValueError, match="rounds its signals"):
        analyzer.analyze(model, _dataset())


@pytest.mark.parametrize(
    "rule, fields",
    [
        ("sample_rolled_denominator", {}),
        (
            "sample_rolled_tangent",
            {
                "anchor_numerator": torch.tensor(1.0),
                "anchor_conductance": torch.tensor(1.0),
            },
        ),
        (
            "permuted_current_denominator",
            {"denominator_permutation": torch.tensor([1, 0])},
        ),
    ],
)
@pytest.mark.parametrize("tree", [0, 1])
def test_batch_coupled_overrides_are_refused_anywhere_before_forward(
    rule, fields, tree
):
    torch.manual_seed(17)
    model = _two_depth_classifier()
    # Analyze tree 0, depth 1; the override can be on a different depth or tree.
    layer = model.core_network.layers[tree].excitatory_cells.branch_layers[-1]
    calls = []
    handle = model.register_forward_pre_hook(lambda *_: calls.append(1))
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(
            module_name_prefix=PREFIX, depths=[1], chunk=2, draws=[2]
        )
    )
    try:
        with layer.override_branch_rule(BranchRuleOverride(rule, **fields)):
            with pytest.raises(ValueError, match="batch-dependent.*" + rule):
                analyzer.analyze(
                    model, _dataset(n=2), runtime=EvaluationRuntimeConfig(batch_size=3)
                )
    finally:
        handle.remove()
    assert not calls


def test_batch_independent_override_keeps_estimates_when_rebatched():
    torch.manual_seed(23)
    model = _two_depth_classifier().double()
    layer = model.core_network.layers[1].excitatory_cells.branch_layers[-1]
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(
            module_name_prefix=PREFIX, depths=[1], chunk=2, draws=[4], seed=5
        )
    )
    data = _dataset(n=2)
    data = torch.utils.data.TensorDataset(data.tensors[0].double(), data.tensors[1])
    override = BranchRuleOverride(
        "frozen_denominator", anchor_conductance=torch.tensor(1.0)
    )
    with layer.override_branch_rule(override):
        rows = [
            analyzer.analyze(
                model, data, runtime=EvaluationRuntimeConfig(batch_size=size)
            )["rows"][0]
            for size in (2, 3)
        ]
    for key, value in rows[0].items():
        if key == "seconds_per_draw":
            continue
        if isinstance(value, float):
            assert rows[1][key] == pytest.approx(value, rel=1e-9, abs=1e-12), key
        else:
            assert rows[1][key] == value, key


def _unused_inhibitory_classifier():
    torch.manual_seed(19)
    return Classifier(
        encoder_network=Identity(input_dim=10),
        core_network=ExcitationInhibitionNetwork(
            input_dim=10,
            transfer_params=TransferFunctionConfig().asdict(),
            excitatory_layer_sizes=[4, 4],
            inhibitory_layer_sizes=[2, 2],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            ee_synapses_per_branch_per_layer=[1, 1],
            ei_synapses_per_branch_per_layer=[1, 1],
            ie_synapses_per_branch_per_layer=[0, 0],
            ii_synapses_per_branch_per_layer=[0],
        ),
        decoder_network=Identity(input_dim=4),
    ).double()


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("coordinate", ["output", "voltage", "current"])
def test_unused_inhibitory_offsets_have_zero_credit(frozen, coordinate):
    model = _unused_inhibitory_classifier()
    model.requires_grad_(not frozen)
    prefix = "core_network.layers.1.inhibitory_cells.branch_layers"
    analyzer = NodePerturbationCreditAnalyzer(
        NodePerturbationCreditAnalysisParams(
            module_name_prefix=prefix,
            coordinate=coordinate,
            depths=[1],
            dictionaries=["independent", "siblings"],
            draws=[2],
            chunk=2,
        )
    )
    data = _dataset(n=2)
    x, y = data.tensors[0].double(), data.tensors[1]
    layer = next(
        m
        for n, m in model.named_modules()
        if n.startswith(prefix)
        and isinstance(m, DendriticBranchLayer)
        and m.layer_idx == 1
    )
    errors = analyzer._exact_errors(
        model,
        layer,
        prefix,
        x,
        y,
        layer.branch_config.output_dim,
        torch.float64,
        torch.device("cpu"),
        3,
    )
    assert torch.equal(errors, torch.zeros_like(errors))
    assert not errors.requires_grad
    result = analyzer.analyze(
        model,
        torch.utils.data.TensorDataset(x, y),
        runtime=EvaluationRuntimeConfig(batch_size=3),
    )
    assert len(result["rows"]) == 2
    # Zero-vs-zero alignment and relative magnitude are undefined, not successes.
    for row in result["rows"]:
        for key in (
            "cosine",
            "within_neuron_cosine",
            "sign_agreement",
            "relative_error",
            "norm_ratio",
        ):
            assert math.isnan(row[key]), (key, row)
    assert all(p.requires_grad == (not frozen) for p in model.parameters())
