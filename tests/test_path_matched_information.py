from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.path_matched_information import (
    PathMatchedInformationAnalyzer,
)
from dendritic_modeling.analysis.tools.path_matched_intervention import (
    build_nested_path_supports,
)
from dendritic_modeling.config import (
    EvaluationRuntimeConfig,
    PathMatchedInformationAnalysisParams,
    PathMatchedInterventionAnalysisParams,
    TransferFunctionConfig,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity

FROZEN_FIG3_SELECTED_INDEX_SUPPORTS_SHA256 = (
    "80835a67bf25118689351083e20c35437818a781b80dafa231e60b15d60a2286"
)


def test_path_information_uses_frozen_fig3_selection_indices() -> None:
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


def test_path_information_config_validates_signals_and_estimator() -> None:
    with pytest.raises(ValueError, match="unknown path-matched information signals"):
        PathMatchedInformationAnalysisParams(signals=["branchwise_mi"])
    with pytest.raises(ValueError, match="n_label_shuffles"):
        PathMatchedInformationAnalysisParams(n_label_shuffles=0)
    with pytest.raises(ValueError, match="non-negative"):
        PathMatchedInformationAnalysisParams(depths=[-1])


def test_path_information_and_intervention_defaults_select_the_same_supports() -> None:
    information = PathMatchedInformationAnalysisParams()
    intervention = PathMatchedInterventionAnalysisParams()
    assert information.support_seed == intervention.support_seed
    assert information.support_draws == intervention.support_draws
    assert information.support_namespace == intervention.support_namespace


def test_path_information_is_registered_as_final_only() -> None:
    spec = get_analyzer_spec("path_matched_information")
    assert spec is not None
    assert spec.config_field == "path_matched_information_analysis"
    assert spec.supports_training is False


def test_path_information_captures_stage_explicit_signals_on_validation_only(
    tmp_path,
) -> None:
    model = Classifier(
        encoder_network=Identity(input_dim=10),
        core_network=ExcitationInhibitionNetwork(
            input_dim=10,
            transfer_params=TransferFunctionConfig().asdict(),
            somatic_synapses=False,
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
    generator = torch.Generator().manual_seed(71)
    inputs = torch.rand(40, 10, generator=generator)
    labels = torch.arange(40) % 2
    validation_dataset = torch.utils.data.TensorDataset(inputs, labels)
    analyzer = PathMatchedInformationAnalyzer(
        PathMatchedInformationAnalysisParams(
            signals=[
                "joint_EI_currents",
                "upstream_child_current",
                "post_gate_output",
            ],
            depths=[0, 1, 2],
            support_draws=2,
            support_seed=17,
            support_namespace="unit-path-information",
            information_seed=23,
            n_neighbors=3,
            n_label_shuffles=2,
            module_name_prefix=("core_network.layers.0.excitatory_cells.branch_layers"),
            expected_n_somas=4,
            expected_branch_factors=[2, 2],
        )
    )

    result = analyzer.analyze(
        model=model,
        validation_dataset=validation_dataset,
        device="cpu",
        save_path=str(tmp_path),
        filename="information",
        seed_offset=5,
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=8,
            num_workers=0,
            pin_memory=False,
        ),
    )

    assert result["analysis_split"] == "validation"
    assert result["test_examples_or_labels_used"] is False
    assert result["sample_count"] == 40
    assert result["somatic_support_draw_count"] == 1
    assert result["non_somatic_support_draw_count"] == 2
    assert result["information_seed_base"] == 23
    assert result["information_seed_offset"] == 5
    assert result["information_seed_effective"] == 28
    assert result["available_depths_by_signal"] == {
        "joint_EI_currents": [1, 2],
        "upstream_child_current": [0, 1],
        "post_gate_output": [0, 1, 2],
    }
    assert len(result["records"]) == 12
    assert {
        record["selected_coordinate_count_per_component"]
        for record in result["records"]
    } == {4}
    assert all(record["information_units"] == "bits" for record in result["records"])
    assert all(
        record["standardized_per_coordinate"] is True for record in result["records"]
    )
    assert all(
        len(record["label_shuffle_mi_bits"]) == 2 for record in result["records"]
    )
    assert (tmp_path / "information.json").is_file()
