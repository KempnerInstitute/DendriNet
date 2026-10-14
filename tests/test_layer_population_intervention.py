from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from dendritic_modeling.analysis.registry import get_analyzer_spec
from dendritic_modeling.analysis.tools.layer_population_intervention import (
    LocalInhibitoryPopulationInterventionAnalyzer,
    _local_i_inventory,
)
from dendritic_modeling.config import (
    EvaluationRuntimeConfig,
    LocalInhibitoryPopulationInterventionAnalysisParams,
)
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.analysis.collect_layer_population_interventions import (
    collect,
)


def _layer(index: int, source: str) -> dict:
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


def _model(*, cross_layer_i_consumer: bool = False) -> Classifier:
    layers = [
        _layer(0, "input"),
        _layer(1, "layer0.e0"),
    ]
    if cross_layer_i_consumer:
        layers[1]["connections"].append(
            {
                "source": "layer0.i0",
                "target": "e1",
                "pathway": "ff_excitatory",
                "timing": "same_step",
            }
        )
    core = get_architecture(
        "population_network",
        {"population_network": {"layers": layers}},
        input_dim=6,
    )
    return Classifier(Identity(input_dim=6), core, Identity(input_dim=4))


def test_local_i_analyzer_is_registered_and_config_validates() -> None:
    spec = get_analyzer_spec("local_inhibitory_population_intervention")
    assert spec is not None
    assert spec.config_field == "local_inhibitory_population_intervention"
    assert spec.supports_training is False

    params = LocalInhibitoryPopulationInterventionAnalysisParams(
        methods=["global_test_shuffle", "within_class_test_shuffle", "zero"],
        permutation_draws=2,
        expected_network_layers=2,
        expected_inhibitory_width=2,
    )
    assert params.permutation_draws == 2


def test_local_i_inventory_rejects_qualified_cross_layer_i_consumers() -> None:
    model = _model(cross_layer_i_consumer=True)
    with pytest.raises(ValueError, match="cross-layer consumers"):
        _local_i_inventory(
            model,
            expected_network_layers=2,
            expected_inhibitory_width=2,
        )


def test_local_i_analyzer_replaces_complete_population_across_network_position(
    tmp_path,
) -> None:
    torch.manual_seed(11)
    model = _model()
    inputs = torch.rand(30, 6)
    with torch.no_grad():
        labels = model(inputs).argmax(dim=1)
        baseline_logits = model(inputs).clone()
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    params = LocalInhibitoryPopulationInterventionAnalysisParams(
        methods=["global_test_shuffle", "within_class_test_shuffle", "zero"],
        permutation_draws=2,
        shuffle_seed=17,
        expected_network_layers=2,
        expected_inhibitory_width=2,
    )
    result = LocalInhibitoryPopulationInterventionAnalyzer(params).analyze(
        model=model,
        test_dataset=dataset,
        runtime=EvaluationRuntimeConfig(mode="stream", batch_size=7),
        save_path=str(tmp_path),
        device="cpu",
    )

    assert result["analysis_type"] == "local_inhibitory_population_intervention"
    assert len(result["records"]) == 10
    assert len(result["paired_global_minus_within_class_records"]) == 4
    assert {record["network_layer_number"] for record in result["records"]} == {
        1,
        2,
    }
    assert {record["replacement_coordinate_count"] for record in result["records"]} == {
        2
    }
    assert {record["route"] for record in result["records"]} == {
        "same_step_ff_inhibitory_local_I_to_E"
    }
    assert (tmp_path / "final.json").is_file()

    with torch.no_grad():
        assert torch.equal(model(inputs), baseline_logits)


def test_local_i_collector_is_fail_closed_and_averages_technical_draws(
    tmp_path,
) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    analysis_config = tmp_path / "analysis.yaml"
    config.write_text("model: test\n")
    checkpoint.write_bytes(b"checkpoint")
    analysis_config.write_text("analysis: test\n")
    output = tmp_path / "run"
    result_dir = output / "local_inhibitory_population_intervention"
    result_dir.mkdir(parents=True)
    records = []
    paired = []
    for layer in (1, 2):
        for method in (
            "global_test_shuffle",
            "within_class_test_shuffle",
        ):
            for draw in (0, 1):
                records.append(
                    {
                        "network_layer_index": layer - 1,
                        "network_layer_number": layer,
                        "source_population": f"i{layer - 1}",
                        "target_population": f"e{layer - 1}",
                        "route": "same_step_ff_inhibitory_local_I_to_E",
                        "method": method,
                        "permutation_draw": draw,
                        "baseline_accuracy": 0.8,
                        "accuracy": 0.7,
                        "accuracy_drop": 0.1,
                        "baseline_mean_cross_entropy": 0.6,
                        "mean_cross_entropy": 0.8,
                        "mean_cross_entropy_increase": 0.2,
                        "normalized_above_chance_accuracy_loss": 0.3,
                    }
                )
                if method == "global_test_shuffle":
                    paired.append(
                        {
                            "network_layer_index": layer - 1,
                            "network_layer_number": layer,
                            "permutation_draw": draw,
                            "class_aligned_mean_cross_entropy_excess": 0.1,
                            "class_aligned_accuracy_drop_excess": 0.05,
                        }
                    )
        records.append(
            {
                "network_layer_index": layer - 1,
                "network_layer_number": layer,
                "source_population": f"i{layer - 1}",
                "target_population": f"e{layer - 1}",
                "route": "same_step_ff_inhibitory_local_I_to_E",
                "method": "zero",
                "permutation_draw": None,
                "baseline_accuracy": 0.8,
                "accuracy": 0.7,
                "accuracy_drop": 0.1,
                "baseline_mean_cross_entropy": 0.6,
                "mean_cross_entropy": 0.8,
                "mean_cross_entropy_increase": 0.2,
                "normalized_above_chance_accuracy_loss": 0.3,
            }
        )
    result = {
        "schema_version": 1,
        "analysis_type": "local_inhibitory_population_intervention",
        "permutation_draws": 2,
        "baseline": {
            "accuracy": 0.8,
            "mean_cross_entropy": 0.6,
            "sample_count": 20,
        },
        "records": records,
        "paired_global_minus_within_class_records": paired,
    }
    (result_dir / "final.json").write_text(json.dumps(result))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "enabled_analyzers": ["local_inhibitory_population_intervention"],
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "output_dir": str(output),
        "config_path": str(config),
        "config_sha256": sha(config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "analysis_config_path": str(analysis_config),
        "analysis_config_sha256": sha(analysis_config),
    }
    (output / "checkpoint_analysis_provenance.json").write_text(json.dumps(provenance))
    job = {
        "lineage": "test",
        "mechanism": "shunting",
        "config_id": 0,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "analysis_config": str(analysis_config),
        "output_dir": str(output),
    }
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(job) + "\n")

    manifest = collect(
        jobs_paths=[jobs],
        output_dir=tmp_path / "collected",
        expected_count=1,
    )
    assert manifest["checkpoint_count"] == 1
    assert manifest["output_tables"]["intervention_records"]["row_count"] == 10
    assert manifest["output_tables"]["paired_donor_technical_summary"]["row_count"] == 2
    source_identity = manifest["collector_source_identity"]
    assert (
        source_identity["repository_name"] == Path(__file__).resolve().parents[1].name
    )
    assert source_identity["files"][0]["path"] == (
        "repo:src/dendritic_modeling/scripts/analysis/"
        "collect_layer_population_interventions.py"
    )

    duplicate_result = {**result, "records": [*records[:-1], records[0]]}
    (result_dir / "final.json").write_text(json.dumps(duplicate_result))
    with pytest.raises(ValueError, match="duplicate local-I record identities"):
        collect(
            jobs_paths=[jobs],
            output_dir=tmp_path / "duplicate_collected",
            expected_count=1,
        )

    changed_grid_records = [dict(record) for record in records]
    changed_grid_records[0]["permutation_draw"] = 99
    (result_dir / "final.json").write_text(
        json.dumps({**result, "records": changed_grid_records})
    )
    with pytest.raises(ValueError, match="layer/method/draw grid changed"):
        collect(
            jobs_paths=[jobs],
            output_dir=tmp_path / "changed_grid_collected",
            expected_count=1,
        )

    (result_dir / "final.json").write_text(json.dumps(result))
    invalid_provenance = {**provenance, "enabled_analyzers": []}
    (output / "checkpoint_analysis_provenance.json").write_text(
        json.dumps(invalid_provenance)
    )
    with pytest.raises(ValueError, match="analyzer is absent"):
        collect(
            jobs_paths=[jobs],
            output_dir=tmp_path / "disabled_analyzer_collected",
            expected_count=1,
        )

    invalid_provenance = {**provenance, "schema_version": "old_schema"}
    (output / "checkpoint_analysis_provenance.json").write_text(
        json.dumps(invalid_provenance)
    )
    with pytest.raises(ValueError, match="provenance schema"):
        collect(
            jobs_paths=[jobs],
            output_dir=tmp_path / "old_schema_collected",
            expected_count=1,
        )

    (output / "checkpoint_analysis_provenance.json").write_text(json.dumps(provenance))
    nonfinite_records = [dict(record) for record in records]
    nonfinite_records[0]["accuracy"] = float("nan")
    (result_dir / "final.json").write_text(
        json.dumps({**result, "records": nonfinite_records})
    )
    with pytest.raises(ValueError, match="non-finite local-I intervention metric"):
        collect(
            jobs_paths=[jobs],
            output_dir=tmp_path / "nonfinite_collected",
            expected_count=1,
        )
