from __future__ import annotations

import numpy as np
import pytest
import torch

from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.path_matched_intervention import (
    PathMatchedInterventionAnalyzer,
    _replacement_hook,
    _signal_sites,
    build_nested_path_supports,
    inventory_nested_tree,
)
from dendritic_modeling.config import (
    EvaluationRuntimeConfig,
    PathMatchedInterventionAnalysisParams,
    TransferFunctionConfig,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity

FROZEN_FIG3_SELECTED_INDEX_SUPPORTS_SHA256 = (
    "80835a67bf25118689351083e20c35437818a781b80dafa231e60b15d60a2286"
)


def _two_depth_classifier() -> Classifier:
    return Classifier(
        encoder_network=Identity(input_dim=10),
        core_network=ExcitationInhibitionNetwork(
            input_dim=10,
            transfer_params=TransferFunctionConfig().asdict(),
            excitatory_layer_sizes=[4, 4],
            inhibitory_layer_sizes=[4, 4],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            ee_synapses_per_branch_per_layer=[1, 1],
            ei_synapses_per_branch_per_layer=[1, 1],
            ie_synapses_per_branch_per_layer=[1, 1],
            ii_synapses_per_branch_per_layer=[1],
        ),
        decoder_network=Identity(input_dim=4),
    )


def test_nested_supports_are_fixed_count_anatomical_path_prefixes():
    support = build_nested_path_supports(
        n_somas=5,
        branch_factors=[3, 2, 2],
        draws=4,
        seed=271828,
        namespace="unit-test",
    )

    assert support["coordinates_per_depth_per_draw"] == 5
    for draw in range(4):
        paths_by_depth = {}
        for depth in range(1, 4):
            entry = support["depths"][str(depth)][draw]
            assert len(entry["selected_indices"]) == 5
            assert len(set(entry["selected_indices"])) == 5
            paths_by_depth[depth] = {
                record["soma_owner"]: record["path_from_soma"]
                for record in entry["coordinates"]
            }
        for soma in range(5):
            assert paths_by_depth[2][soma][:1] == paths_by_depth[1][soma]
            assert paths_by_depth[3][soma][:2] == paths_by_depth[2][soma]

    repeated = build_nested_path_supports(
        n_somas=5,
        branch_factors=[3, 2, 2],
        draws=4,
        seed=271828,
        namespace="unit-test",
    )
    assert repeated == support


def test_fig3_support_indices_match_the_frozen_information_analysis() -> None:
    support = build_nested_path_supports(
        n_somas=20,
        branch_factors=[3, 3, 3, 3],
        draws=5,
        seed=271828,
        namespace="fig3-v4",
    )

    assert (
        support["selected_index_supports_sha256"]
        == FROZEN_FIG3_SELECTED_INDEX_SUPPORTS_SHA256
    )


def test_replacement_hook_changes_only_selected_coordinates_and_audits_cursor():
    output = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    selected = np.asarray([1, 6], dtype=np.int64)
    union_positions = np.asarray([0, 2], dtype=np.int64)
    fixed = np.asarray([10.0, 11.0, 12.0], dtype=np.float32)
    shuffled = np.asarray(
        [[20.0, 21.0, 22.0], [30.0, 31.0, 32.0], [40.0, 41.0, 42.0]],
        dtype=np.float32,
    )
    hook, cursor = _replacement_hook(
        method="global_test_shuffle",
        selected_indices=selected,
        selected_union_positions=union_positions,
        fixed_reference_values=fixed,
        shuffled_evaluation_values=shuffled,
    )

    replaced = hook(None, (), output)
    untouched = [0, 2, 3, 4, 5, 7]
    assert torch.equal(replaced[:, untouched], output[:, untouched])
    assert torch.equal(
        replaced[:, selected],
        torch.tensor([[20.0, 22.0], [30.0, 32.0], [40.0, 42.0]]),
    )
    assert cursor() == 3
    assert torch.equal(output, torch.arange(24, dtype=torch.float32).reshape(3, 8))


def test_path_matched_config_rejects_non_dendritic_depth_and_unknown_target():
    with pytest.raises(ValueError, match="dendritic"):
        PathMatchedInterventionAnalysisParams(depths=[0])
    with pytest.raises(ValueError, match="unknown path-matched targets"):
        PathMatchedInterventionAnalysisParams(targets=["branch_output"])


def test_post_gate_target_is_opt_in_and_maps_to_branch_layer_output():
    default_params = PathMatchedInterventionAnalysisParams()
    assert "post_gate_output" not in default_params.targets
    assert PathMatchedInterventionAnalysisParams(
        targets=["post_gate_output"]
    ).targets == ["post_gate_output"]

    model = _two_depth_classifier()
    modules, _n_somas, _branch_factors = inventory_nested_tree(
        model,
        module_name_prefix=("core_network.layers.0.excitatory_cells.branch_layers"),
    )
    sites = _signal_sites(modules)
    assert sites[(1, "post_gate_output")].producer is modules[1][1]
    assert sites[(2, "post_gate_output")].producer is modules[2][1]


def test_path_matched_analyzer_is_registered_final_only():
    spec = get_analyzer_spec("path_matched_intervention")
    assert spec is not None
    assert spec.config_field == "path_matched_intervention_analysis"
    assert spec.supports_training is False


def test_analyzer_runs_equal_coordinate_interventions_on_one_selected_tree(tmp_path):
    model = _two_depth_classifier()
    inputs = torch.rand(16, 10)
    with torch.no_grad():
        labels = model(inputs).argmax(dim=1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    params = PathMatchedInterventionAnalysisParams(
        targets=["joint_EI_currents"],
        methods=["zero"],
        depths=[1, 2],
        support_draws=2,
        support_seed=17,
        module_name_prefix=("core_network.layers.0.excitatory_cells.branch_layers"),
        expected_n_somas=4,
        expected_branch_factors=[2, 2],
        chance_accuracy=0.0,
    )
    analyzer = PathMatchedInterventionAnalyzer(params)
    result = analyzer.analyze(
        model=model,
        reference_dataset=dataset,
        test_dataset=dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="control",
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert result["baseline"]["accuracy"] == 1.0
    assert len(result["records"]) == 4
    assert {record["soma_relative_depth"] for record in result["records"]} == {
        1,
        2,
    }
    assert {
        record["selected_compartment_coordinate_count"] for record in result["records"]
    } == {4}
    assert {
        record["replaced_tensor_coordinate_count"] for record in result["records"]
    } == {8}
    assert (tmp_path / "control.json").is_file()


def test_analyzer_intervenes_on_mixed_current_and_post_gate_targets():
    model = _two_depth_classifier()
    inputs = torch.rand(12, 10)
    with torch.no_grad():
        labels = model(inputs).argmax(dim=1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    analyzer = PathMatchedInterventionAnalyzer(
        PathMatchedInterventionAnalysisParams(
            targets=["joint_EI_currents", "post_gate_output"],
            methods=["zero", "fixed_train_mean", "global_test_shuffle"],
            depths=[1, 2],
            support_draws=1,
            support_seed=19,
            module_name_prefix=("core_network.layers.0.excitatory_cells.branch_layers"),
            expected_n_somas=4,
            expected_branch_factors=[2, 2],
            chance_accuracy=0.0,
        )
    )

    result = analyzer.analyze(
        model=model,
        reference_dataset=dataset,
        test_dataset=dataset,
        device="cpu",
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert len(result["records"]) == 12
    assert {record["target"] for record in result["records"]} == {
        "joint_EI_currents",
        "post_gate_output",
    }
    signals_by_target = {
        target: {
            tuple(record["target_signals"])
            for record in result["records"]
            if record["target"] == target
        }
        for target in ("joint_EI_currents", "post_gate_output")
    }
    assert signals_by_target == {
        "joint_EI_currents": {("excitation_current", "inhibition_current")},
        "post_gate_output": {("post_gate_output",)},
    }
    assert {record["soma_relative_depth"] for record in result["records"]} == {
        1,
        2,
    }
    assert {
        record["selected_compartment_coordinate_count"] for record in result["records"]
    } == {4}
    replaced_counts_by_target = {
        target: {
            record["replaced_tensor_coordinate_count"]
            for record in result["records"]
            if record["target"] == target
        }
        for target in ("joint_EI_currents", "post_gate_output")
    }
    assert replaced_counts_by_target == {
        "joint_EI_currents": {8},
        "post_gate_output": {4},
    }
    assert (
        "depth 0 is the soma"
        in result["post_gate_output_semantics"]["depth_convention"]
    )
