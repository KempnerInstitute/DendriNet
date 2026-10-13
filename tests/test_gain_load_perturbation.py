from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.tools.gain_load_perturbation import (
    GainLoadPerturbationAnalyzer,
    gain_load_draw_seed,
)
from dendritic_modeling.config import (
    EvaluationRuntimeConfig,
    GainLoadPerturbationAnalysisParams,
    TransferFunctionConfig,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity


def _analyzer(**kwargs) -> GainLoadPerturbationAnalyzer:
    return GainLoadPerturbationAnalyzer(GainLoadPerturbationAnalysisParams(**kwargs))


def test_zero_gain_and_load_is_identity() -> None:
    analyzer = _analyzer(clamp_min=None)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    observed = analyzer._perturb(
        inputs,
        gain_log_sd=0.0,
        load_level=0.0,
        gain_draw_seed=7,
        load_draw_seed=8,
    )
    assert torch.equal(observed, inputs)


def test_shared_gain_is_one_scalar_per_trial() -> None:
    analyzer = _analyzer(clamp_min=None)
    inputs = torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
    observed = analyzer._perturb(
        inputs,
        gain_log_sd=0.5,
        load_level=0.0,
        gain_draw_seed=31415,
        load_draw_seed=31416,
    )
    ratios = observed / inputs
    assert torch.allclose(ratios, ratios[:, :1].expand_as(ratios))


def test_positive_load_changes_only_declared_inhibitory_stream() -> None:
    analyzer = _analyzer(
        load_mode="positive_inhibitory",
        load_log_sd=0.0,
        inhibitory_start=2,
        clamp_min=None,
    )
    inputs = torch.ones(3, 5)
    observed = analyzer._perturb(
        inputs,
        gain_log_sd=0.0,
        load_level=2.0,
        gain_draw_seed=11,
        load_draw_seed=12,
    )
    assert torch.equal(observed[:, :2], inputs[:, :2])
    assert torch.equal(observed[:, 2:], torch.full((3, 3), 3.0))


def test_condition_seed_is_grid_order_independent() -> None:
    seed = gain_load_draw_seed(9, 0.5, 0.3, 2)
    assert seed == gain_load_draw_seed(9, 0.5, 0.3, 2)
    assert seed != gain_load_draw_seed(9, 0.5, 0.3, 3)
    assert seed != gain_load_draw_seed(9, 0.2, 0.3, 2)


def test_analysis_saves_standard_final_json(tmp_path) -> None:
    analyzer = _analyzer(
        gain_log_sds=[0.0],
        load_levels=[0.0],
        draws_per_condition=1,
        clamp_min=None,
        categorical_loglikelihood=False,
    )
    model = Classifier(
        Identity(2),
        Identity(2),
        Identity(2),
        output_scale_mode="fixed",
    )
    dataset = TensorDataset(
        torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
        torch.tensor([0, 1]),
    )
    result = analyzer.analyze(
        model,
        dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="final",
        runtime=EvaluationRuntimeConfig(mode="materialize", batch_size=2),
    )
    assert result["records"][0]["accuracy"] == 1.0
    assert (tmp_path / "final.json").is_file()


def test_analysis_can_record_input_statistics_and_branch_sensitivity() -> None:
    analyzer = _analyzer(
        gain_log_sds=[0.0],
        load_levels=[0.0],
        draws_per_condition=1,
        clamp_min=None,
        categorical_loglikelihood=False,
        record_input_statistics=True,
        record_branch_local_sensitivity=True,
        branch_sensitivity_module_name_prefix=(
            "core_network.layers.0.excitatory_cells.branch_layers"
        ),
        branch_sensitivity_expected_n_somas=4,
        branch_sensitivity_max_samples=4,
    )
    model = Classifier(
        encoder_network=Identity(input_dim=10),
        core_network=ExcitationInhibitionNetwork(
            input_dim=10,
            transfer_params=TransferFunctionConfig().asdict(),
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[0],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            ee_synapses_per_branch_per_layer=[1, 1],
            ei_synapses_per_branch_per_layer=[0, 0],
            ie_synapses_per_branch_per_layer=[1, 1],
            ii_synapses_per_branch_per_layer=[0],
        ),
        decoder_network=Identity(input_dim=4),
    )
    dataset = TensorDataset(
        torch.rand(4, 10),
        torch.tensor([0, 1, 2, 3]),
    )

    result = analyzer.analyze(
        model,
        dataset,
        device="cpu",
        runtime=EvaluationRuntimeConfig(
            mode="stream", batch_size=2, num_workers=0, pin_memory=False
        ),
    )

    record = result["records"][0]
    assert record["input_statistics"]["scalar_count"] == 40
    assert record["input_statistics"]["zero_fraction"] == 0.0
    sensitivity = record["branch_local_sensitivity"]
    assert sensitivity["analysis_type"] == "branch_local_sensitivity"
    assert {entry["soma_relative_depth"] for entry in sensitivity["records"]} == {
        1,
        2,
    }


def test_clamping_is_reported_per_record_and_in_the_load_definition():
    model = Classifier(Identity(3), Identity(3), Identity(3))
    inputs = torch.tensor([[0.0, 1.0, 2.0], [0.0, 0.5, 3.0]] * 4)
    analyzer = _analyzer(
        gain_log_sds=[0.0], load_levels=[0.0, 0.5], draws_per_condition=1
    )
    result = analyzer.analyze(
        model=model, test_dataset=TensorDataset(inputs, torch.tensor([2, 2] * 4))
    )
    clean, loaded = result["records"]
    assert clean["clamped_fraction_low"] == 0.0
    assert loaded["clamped_fraction_low"] > 0.0
    assert "rectifies the load" in result["load_definition"]
    assert result["baseline_clamped_fractions"]["clamped_fraction_low"] == 0.0


def test_inputs_already_below_the_clamp_warn(caplog):
    import logging

    model = Classifier(Identity(2), Identity(2), Identity(2))
    inputs = torch.tensor([[-1.0, 1.0], [0.5, -0.2]] * 4)
    analyzer = _analyzer(gain_log_sds=[0.0], load_levels=[0.0], draws_per_condition=1)
    with caplog.at_level(logging.WARNING):
        result = analyzer.analyze(
            model=model, test_dataset=TensorDataset(inputs, torch.tensor([1, 0] * 4))
        )
    assert result["baseline_clamped_fractions"]["clamped_fraction_low"] == 0.5
    assert "not the clean test set" in caplog.text
