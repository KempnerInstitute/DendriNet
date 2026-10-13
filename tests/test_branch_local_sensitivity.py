from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.branch_local_sensitivity import (
    BranchLocalSensitivityAnalyzer,
    _local_voltage_partials,
)
from dendritic_modeling.config import (
    BranchLocalSensitivityAnalysisParams,
    EvaluationRuntimeConfig,
    TransferFunctionConfig,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity


def _diagnostics(mode: str) -> dict[str, torch.Tensor | str]:
    excitation = torch.tensor([[2.0, 1.0]])
    inhibition = torch.tensor([[3.0, 0.5]])
    inherited_current = torch.tensor([[1.0, 2.0]])
    conductance = torch.tensor([[0.5, 1.5]])
    return {
        "E": excitation,
        "I": inhibition,
        "C": inherited_current,
        "G": conductance,
        "N": excitation + inherited_current,
        "T": excitation + inhibition + conductance,
        "V": torch.zeros_like(excitation),
        "mode": mode,
    }


@pytest.mark.parametrize("leak", [1.0, 2.0, torch.tensor([0.5, 3.0])])
def test_shunting_local_partials_include_excitation_in_denominator(leak) -> None:
    module = SimpleNamespace(epsilon=0.0, effective_leak=leak)
    diagnostics = _diagnostics("shunting")
    partials = _local_voltage_partials(module, diagnostics)
    denominator = leak + diagnostics["T"]
    numerator = diagnostics["N"]

    assert torch.allclose(
        partials["E"], (denominator - numerator) / denominator.square()
    )
    assert torch.allclose(partials["I"], -numerator / denominator.square())
    assert torch.allclose(partials["C"], 1 / denominator)
    assert torch.allclose(partials["G"], -numerator / denominator.square())


def test_raw_and_conductance_normalized_controls_have_declared_partials() -> None:
    raw_module = SimpleNamespace(epsilon=0.0, use_additive_normalization=False)
    raw = _local_voltage_partials(raw_module, _diagnostics("raw"))
    assert torch.equal(raw["E"], torch.ones(1, 2))
    assert torch.equal(raw["I"], -torch.ones(1, 2))
    assert torch.equal(raw["C"], torch.ones(1, 2))
    assert torch.equal(raw["G"], torch.zeros(1, 2))

    diagnostics = _diagnostics("conductance_normalized")
    normalized = _local_voltage_partials(raw_module, diagnostics)
    denominator = 1 + diagnostics["T"]
    numerator = diagnostics["E"] - diagnostics["I"] + diagnostics["C"]
    assert torch.allclose(
        normalized["I"], (-denominator - numerator) / denominator.square()
    )


def _two_depth_classifier() -> Classifier:
    return Classifier(
        encoder_network=Identity(input_dim=10),
        core_network=ExcitationInhibitionNetwork(
            input_dim=10,
            transfer_params=TransferFunctionConfig().asdict(),
            excitatory_layer_sizes=[4, 4],
            inhibitory_layer_sizes=[0, 0],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            ee_synapses_per_branch_per_layer=[1, 1],
            ei_synapses_per_branch_per_layer=[0, 0],
            ie_synapses_per_branch_per_layer=[1, 0],
            ii_synapses_per_branch_per_layer=[0],
        ),
        decoder_network=Identity(input_dim=4),
    )


def test_branch_local_sensitivity_analyzer_runs_on_fitted_tree(tmp_path) -> None:
    model = _two_depth_classifier()
    inputs = torch.rand(12, 10)
    labels = torch.zeros(12, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    analyzer = BranchLocalSensitivityAnalyzer(
        BranchLocalSensitivityAnalysisParams(
            depths=[1, 2],
            module_name_prefix=("core_network.layers.0.excitatory_cells.branch_layers"),
            expected_n_somas=4,
            expected_branch_factors=[2, 2],
        )
    )

    result = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="sensitivity",
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert result["analysis_type"] == "branch_local_sensitivity"
    assert len(result["records"]) == 10
    assert {record["soma_relative_depth"] for record in result["records"]} == {
        1,
        2,
    }
    assert {record["component"] for record in result["records"]} == {
        "E",
        "I",
        "C",
        "G",
        "V",
    }
    assert all(
        record["mean_abs_doutput_dcomponent"] >= 0.0 for record in result["records"]
    )
    assert (tmp_path / "sensitivity.json").is_file()


def test_branch_local_sensitivity_is_registered_and_rejects_soma_depth() -> None:
    spec = get_analyzer_spec("branch_local_sensitivity")
    assert spec is not None
    assert spec.config_field == "branch_local_sensitivity"
    assert spec.supports_training is False
    with pytest.raises(ValueError, match="dendritic"):
        BranchLocalSensitivityAnalysisParams(depths=[0])


def test_branch_local_sensitivity_can_include_soma_explicitly() -> None:
    model = _two_depth_classifier()
    dataset = torch.utils.data.TensorDataset(
        torch.rand(8, 10), torch.zeros(8, dtype=torch.long)
    )
    analyzer = BranchLocalSensitivityAnalyzer(
        BranchLocalSensitivityAnalysisParams(
            depths=[0],
            include_soma=True,
            module_name_prefix=("core_network.layers.0.excitatory_cells.branch_layers"),
            expected_n_somas=4,
            expected_branch_factors=[2, 2],
        )
    )

    result = analyzer.analyze(
        model=model,
        test_dataset=dataset,
        device="cpu",
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert {record["soma_relative_depth"] for record in result["records"]} == {0}
