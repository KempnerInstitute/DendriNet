from __future__ import annotations

import pytest
import torch

from dendritic_modeling.analysis.tools.input_region_intervention import (
    ImageRegionSpec,
    InputRegionInterventionAnalyzer,
    InputRegionInterventionSettings,
    apply_input_region_intervention,
    build_input_region_mask,
    estimate_featurewise_reference_mean,
)
from dendritic_modeling.config import EvaluationRuntimeConfig
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture


def _population_layer(index: int, source: str) -> dict:
    return {
        "name": f"layer{index}",
        "readout_population": f"e{index}",
        "population_defaults": {
            "ff_excitatory_synapses": 2,
            "ff_inhibitory_synapses": 1,
            "rec_excitatory_synapses": 0,
            "rec_inhibitory_synapses": 0,
            "somatic_synapses": True,
            "reactivate": True,
            "dbl_init_method": "default",
            "weight_transform": "exp",
            "use_shunting": True,
        },
        "populations": [
            {
                "name": f"i{index}",
                "polarity": "inhibitory",
                "n_neurons": 2,
                "branch_factors": [1],
                "population": {"ff_inhibitory_synapses": 0},
            },
            {
                "name": f"e{index}",
                "polarity": "excitatory",
                "n_neurons": 4,
                "branch_factors": [1],
            },
        ],
        "connections": [
            {
                "source": source,
                "target": f"i{index}",
                "pathway": "ff_excitatory",
                "timing": "same_step",
            },
            {
                "source": source,
                "target": f"e{index}",
                "pathway": "ff_excitatory",
                "timing": "same_step",
            },
            {
                "source": f"i{index}",
                "target": f"e{index}",
                "pathway": "ff_inhibitory",
                "timing": "same_step",
            },
        ],
    }


def _model() -> Classifier:
    layers = [_population_layer(0, "input"), _population_layer(1, "layer0.e0")]
    core = get_architecture(
        "population_network",
        {"population_network": {"layers": layers}},
        input_dim=6,
    )
    return Classifier(Identity(input_dim=6), core, Identity(input_dim=4))


def _direct_ei_model(*, use_shunting: bool) -> Classifier:
    layer = {
        "name": "layer0",
        "readout_population": "e0",
        "population_defaults": {
            "ff_excitatory_synapses": 2,
            "ff_inhibitory_synapses": 1,
            "rec_excitatory_synapses": 0,
            "rec_inhibitory_synapses": 0,
            "somatic_synapses": True,
            "reactivate": True,
            "dbl_init_method": "default",
            "weight_transform": "exp",
            "use_shunting": use_shunting,
        },
        "populations": [
            {
                "name": "e0",
                "polarity": "excitatory",
                "n_neurons": 4,
                "branch_factors": [1],
            }
        ],
        "connections": [
            {
                "source": "input_e",
                "target": "e0",
                "pathway": "ff_excitatory",
                "timing": "same_step",
            },
            {
                "source": "input_i",
                "target": "e0",
                "pathway": "ff_inhibitory",
                "timing": "same_step",
            },
        ],
    }
    core = get_architecture(
        "population_network",
        {
            "population_network": {
                "use_transfer": True,
                "transfer_params": {
                    "independent_pathways": False,
                    "output_activation": "relu",
                },
                "layers": [layer],
            }
        },
        input_dim=6,
    )
    return Classifier(Identity(input_dim=6), core, Identity(input_dim=4))


def test_fixed_and_sample_dependent_region_masks_are_distinct() -> None:
    inputs = torch.tensor(
        [
            [0.0, 1.0, 0.6, 0.1, 0.7, 0.2, 0.9, 0.3, 0.8, 0.4, 0.0, 1.0],
            [1.0, 0.0, 0.4, 0.9, 0.3, 0.8, 0.1, 0.7, 0.2, 0.6, 1.0, 0.0],
        ]
    )
    center = build_input_region_mask(
        inputs,
        image_shape=(1, 3, 4),
        region=ImageRegionSpec("center", "central_square", center_fraction=0.5),
    )
    surround = build_input_region_mask(
        inputs,
        image_shape=(1, 3, 4),
        region=ImageRegionSpec("surround", "surround", center_fraction=0.5),
    )
    foreground = build_input_region_mask(
        inputs,
        image_shape=(1, 3, 4),
        region=ImageRegionSpec("foreground", "foreground", foreground_threshold=0.5),
    )
    background = build_input_region_mask(
        inputs,
        image_shape=(1, 3, 4),
        region=ImageRegionSpec("background", "background", foreground_threshold=0.5),
    )

    assert torch.equal(center, center[0].expand_as(center))
    assert torch.equal(surround, ~center)
    assert not torch.equal(foreground[0], foreground[1])
    assert torch.equal(background, ~foreground)
    assert center[0].sum().item() == 4


def test_remove_and_retain_only_replace_complementary_features() -> None:
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    region = torch.tensor([[True, False, True, False], [False, True, False, True]])
    replacement = torch.full((1, 4), -1.0)
    removed, removed_mask = apply_input_region_intervention(
        inputs,
        region_mask=region,
        operation="remove_only",
        replacement=replacement,
    )
    retained, retained_mask = apply_input_region_intervention(
        inputs,
        region_mask=region,
        operation="retain_only",
        replacement=replacement,
    )

    assert torch.equal(removed_mask, region)
    assert torch.equal(retained_mask, ~region)
    assert torch.equal(removed[region], torch.full((4,), -1.0))
    assert torch.equal(retained[~region], torch.full((4,), -1.0))
    assert torch.equal(removed[~region], inputs[~region])
    assert torch.equal(retained[region], inputs[region])


def test_settings_reject_statically_duplicate_complement_interventions() -> None:
    with pytest.raises(ValueError, match="statically duplicate"):
        InputRegionInterventionSettings(
            image_shape=(1, 2, 3),
            regions=(
                ImageRegionSpec("foreground", "foreground"),
                ImageRegionSpec("background", "background"),
            ),
            operations=("remove_only", "retain_only"),
            replacement_methods=("zero",),
        )


def test_reference_mean_is_streamed_featurewise_and_deterministic() -> None:
    inputs = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    labels = torch.zeros(5, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    runtime = EvaluationRuntimeConfig(
        mode="stream",
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=11,
    )

    first = estimate_featurewise_reference_mean(
        dataset,
        image_shape=(1, 2, 3),
        runtime=runtime,
    )
    second = estimate_featurewise_reference_mean(
        dataset,
        image_shape=(1, 2, 3),
        runtime=runtime,
    )

    assert first.sample_count == 5
    assert torch.equal(first.values, inputs.mean(dim=0).reshape(1, 2, 3))
    assert torch.equal(first.values, second.values)


def test_analyzer_reports_paired_prediction_and_layer_depth_responses(tmp_path) -> None:
    torch.manual_seed(17)
    model = _model()
    reference_inputs = torch.rand(18, 6)
    test_inputs = torch.rand(13, 6)
    with torch.no_grad():
        labels = model(test_inputs).argmax(dim=1)
        original_logits = model(test_inputs).clone()
    reference_dataset = torch.utils.data.TensorDataset(
        reference_inputs,
        torch.zeros(reference_inputs.shape[0], dtype=torch.long),
    )
    test_dataset = torch.utils.data.TensorDataset(test_inputs, labels)
    settings = InputRegionInterventionSettings(
        image_shape=(1, 2, 3),
        regions=(
            ImageRegionSpec("center", "central_square", center_fraction=0.5),
            ImageRegionSpec("foreground", "foreground", foreground_threshold=0.5),
        ),
        operations=("remove_only",),
        replacement_methods=("zero", "reference_mean"),
        reference_split="train",
        evaluation_split="test",
    )
    analyzer = InputRegionInterventionAnalyzer(settings)
    result = analyzer.analyze(
        model=model,
        reference_dataset=reference_dataset,
        test_dataset=test_dataset,
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            seed=19,
        ),
        device="cpu",
        save_path=str(tmp_path),
    )

    assert result["analysis_type"] == "input_region_intervention"
    assert result["baseline"]["accuracy"] == 1.0
    assert result["reference"]["split"] == "train"
    assert result["reference"]["sample_count"] == 18
    assert len(result["interventions"]) == 4
    assert (tmp_path / "final.json").is_file()

    response_records = result["interventions"][0]["response_records"]
    soma_records = [
        record
        for record in response_records
        if record["response_type"] == "soma_population_output"
    ]
    current_records = [
        record
        for record in response_records
        if record["response_type"] == "branch_current"
    ]
    assert {record["population_polarity"] for record in soma_records} == {
        "excitatory",
        "inhibitory",
    }
    assert {record["network_layer_number"] for record in soma_records} == {1, 2}
    assert {record["current_polarity"] for record in current_records} == {
        "excitatory",
        "inhibitory",
    }
    assert {record["target_population_polarity"] for record in current_records} == {
        "excitatory"
    }
    assert {record["response_type"] for record in response_records} >= {
        "branch_voltage",
        "branch_output",
    }
    assert all(
        record["dendritic_depth_reference"] == "soma_relative"
        for record in current_records
    )
    assert all(record["sample_count"] == 13 for record in response_records)
    assert "not literal pixel receptive fields" in result["response_interpretation"]

    with torch.no_grad():
        assert torch.equal(model(test_inputs), original_logits)


@pytest.mark.parametrize(
    ("use_shunting", "expected_mechanism"),
    [(True, "shunting"), (False, "additive")],
)
def test_direct_ei_currents_and_identity_soma_aggregates_are_numeric(
    use_shunting: bool,
    expected_mechanism: str,
) -> None:
    torch.manual_seed(31)
    model = _direct_ei_model(use_shunting=use_shunting)
    inputs = 0.1 + torch.rand(9, 6)
    with torch.no_grad():
        labels = model(inputs).argmax(dim=1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    settings = InputRegionInterventionSettings(
        image_shape=(1, 2, 3),
        regions=(ImageRegionSpec("all", "central_square", center_fraction=1.0),),
        operations=("remove_only",),
        replacement_methods=("zero",),
        identity_soma_class_mapping=(0, 1, 2, 3),
        identity_soma_population_name="e0",
        identity_soma_network_layer_index=0,
    )
    result = InputRegionInterventionAnalyzer(settings).analyze(
        model=model,
        test_dataset=dataset,
        runtime=EvaluationRuntimeConfig(
            mode="stream",
            batch_size=5,
            num_workers=0,
            pin_memory=False,
            seed=7,
        ),
        device="cpu",
    )

    intervention = result["interventions"][0]
    currents = [
        record
        for record in intervention["response_records"]
        if record["response_type"] == "branch_current"
    ]
    assert {record["source_current_polarity"] for record in currents} == {
        "excitatory",
        "inhibitory",
    }
    assert all(record["target_population_name"] == "e0" for record in currents)
    assert all(
        record["integration_mechanism"] == expected_mechanism for record in currents
    )
    assert all(record["baseline_mean"] > 0.0 for record in currents)
    assert all(record["intervention_mean"] == pytest.approx(0.0) for record in currents)
    assert all(
        record["mean_signed_change"] == pytest.approx(-record["baseline_mean"])
        for record in currents
    )
    assert intervention["response_class_records"]
    assert len(intervention["assigned_soma_response_records"]) == 1
    assert intervention["assigned_soma_class_response_records"]


def test_stateful_branch_paths_fail_closed_before_analysis() -> None:
    model = _model()
    model.core_network.layers[0].populations["e0"].soma_feedback_enabled = True
    settings = InputRegionInterventionSettings(
        image_shape=(1, 2, 3),
        regions=(ImageRegionSpec("center", "central_square"),),
        operations=("remove_only",),
        replacement_methods=("zero",),
    )
    dataset = torch.utils.data.TensorDataset(
        torch.rand(2, 6), torch.zeros(2, dtype=torch.long)
    )
    with pytest.raises(ValueError, match="soma feedback"):
        InputRegionInterventionAnalyzer(settings).analyze(
            model=model,
            test_dataset=dataset,
            device="cpu",
        )
