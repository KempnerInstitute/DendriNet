"""Tests for the package-native input-region intervention collector."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import _sha256
from dendritic_modeling.scripts.analysis.collect_input_region_intervention import (
    ANALYZER_NAME,
    CONFIG_SCHEMA_VERSION,
    RESULT_SUBDIRECTORY,
    _load_collection_config,
    _validate_result,
    collect_input_region_intervention,
)


def _mechanism() -> dict:
    return {
        "integration_mechanism": "shunting",
        "use_shunting": True,
        "additive_mode": "raw",
        "reactivation_enabled": True,
        "reactivation_type": "ParametricTanh",
        "inhibitory_drive_semantics": ("positive_drive_added_to_shunting_denominator"),
        "pre_gate_voltage_semantics": (
            "after_configured_EI_integration_before_branch_reactivation"
        ),
        "post_gate_output_semantics": "after_configured_branch_reactivation",
    }


def _response_record(*, class_label: int | None = None) -> dict:
    record = {
        "response_type": "branch_current",
        "response_stage": "pre_gate_synaptic_current",
        "signal": "pre_gate_inhibition_current",
        "module_name": "core_network.layers.0.populations.e0.branch_layers.0",
        "population_name": "e0",
        "population_polarity": "excitatory",
        "target_population_name": "e0",
        "target_population_polarity": "excitatory",
        "soma_relative_dendritic_depth": 1,
        "dendritic_depth_reference": "soma_relative",
        "current_polarity": "inhibitory",
        "source_current_polarity": "inhibitory",
        "network_layer_index": 0,
        "network_layer_number": 1,
        "sample_count": 4 if class_label is None else 2,
        "value_count": 8 if class_label is None else 4,
        "baseline_mean": 2.0,
        "intervention_mean": 1.0,
        "mean_signed_change": -1.0,
        "mean_absolute_change": 1.0,
        "root_mean_square_change": 1.0,
        "relative_l2_change": 0.5,
        "interpretation": "response propagation, not a pixel receptive field",
        **_mechanism(),
    }
    if class_label is not None:
        record.update(
            {
                "class_label": class_label,
                "aggregation_scope": "within_true_class",
            }
        )
    return record


def _assigned_soma_record(*, class_label: int | None = None) -> dict:
    record = {
        "response_type": "true_class_assigned_soma_output",
        "signal": "post_gate_soma_output",
        "module_name": "core_network.layers.0",
        "network_layer_index": 0,
        "network_layer_number": 1,
        "network_layer_name": "layer0",
        "population_name": "e0",
        "population_polarity": "excitatory",
        "interpretation": "network-layer response propagation",
        "aggregation_scope": (
            "true_class_assigned_soma_within_class"
            if class_label is not None
            else "true_class_assigned_soma"
        ),
        "identity_soma_class_mapping": [0, 1],
        "sample_count": 4 if class_label is None else 2,
        "value_count": 4 if class_label is None else 2,
        "baseline_mean": 2.0,
        "intervention_mean": 1.0,
        "mean_signed_change": -1.0,
        "mean_absolute_change": 1.0,
        "root_mean_square_change": 1.0,
        "relative_l2_change": 0.5,
        **_mechanism(),
    }
    if class_label is not None:
        record["class_label"] = class_label
    return record


def _result() -> dict:
    region = {
        "region_name": "center",
        "region_kind": "central_square",
        "mask_scope": "fixed_spatial",
        "sample_dependent": False,
        "center_fraction": 0.5,
    }
    fraction = {
        "sample_count": 4,
        "mean": 0.25,
        "standard_deviation": 0.0,
        "minimum": 0.25,
        "maximum": 0.25,
    }
    return {
        "schema_version": 1,
        "analysis_type": "input_region_intervention",
        "analysis_contract": "paired frozen-checkpoint input intervention",
        "image_shape_channels_height_width": [1, 2, 2],
        "evaluation_split": "test",
        "evaluation_max_samples": None,
        "region_definitions": [region],
        "operations": ["remove_only"],
        "replacement_methods": ["zero"],
        "branch_target_population_polarities": ["excitatory"],
        "replacement_semantics": {"zero": "numerical feature-space zero"},
        "baseline": {
            "sample_count": 4,
            "accuracy": 0.75,
            "mean_negative_log_likelihood": 1.0,
        },
        "interventions": [
            {
                **region,
                "operation": "remove_only",
                "operation_semantics": "replace_named_region",
                "replacement_method": "zero",
                "classification": {
                    "accuracy": 0.5,
                    "accuracy_change": -0.25,
                    "accuracy_drop": 0.25,
                    "mean_negative_log_likelihood": 1.2,
                    "mean_negative_log_likelihood_change": 0.2,
                },
                "named_region_feature_fraction": fraction,
                "replaced_feature_fraction": fraction,
                "response_records": [_response_record()],
                "response_class_records": [
                    _response_record(class_label=0),
                    _response_record(class_label=1),
                ],
                "assigned_soma_response_records": [_assigned_soma_record()],
                "assigned_soma_class_response_records": [
                    _assigned_soma_record(class_label=0),
                    _assigned_soma_record(class_label=1),
                ],
            }
        ],
        "response_interpretation": (
            "later-layer records are not literal pixel receptive fields"
        ),
        "serialization_policy": (
            "aggregate paired metrics only; no per-sample activation or "
            "sample-by-synapse arrays"
        ),
        "identity_soma_mapping": {
            "network_layer_index": 0,
            "population_name": "e0",
            "soma_index_to_class": [0, 1],
        },
    }


def _write_run(tmp_path: Path) -> tuple[Path, Path, dict]:
    run_dir = tmp_path / "run"
    result_dir = run_dir / RESULT_SUBDIRECTORY
    result_dir.mkdir(parents=True)
    inputs: dict[str, Path] = {}
    for field in ("config", "checkpoint", "analysis_config"):
        path = tmp_path / f"{field}.dat"
        path.write_text(f"{field} bytes\n", encoding="utf-8")
        inputs[field] = path.resolve()
    result_path = result_dir / "final.json"
    result_path.write_text(
        json.dumps(_result(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "output_dir": str(run_dir.resolve()),
        "enabled_analyzers": [ANALYZER_NAME],
        "code_identity": {"git_commit": "abc123"},
    }
    for field, path in inputs.items():
        provenance[f"{field}_path"] = str(path)
        provenance[f"{field}_sha256"] = _sha256(path)
    provenance_path = run_dir / "checkpoint_analysis_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    job = {
        "output_dir": str(run_dir.resolve()),
        "config": str(inputs["config"]),
        "checkpoint": str(inputs["checkpoint"]),
        "analysis_config": str(inputs["analysis_config"]),
        "seed": 23,
    }
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job, sort_keys=True) + "\n", encoding="utf-8")
    return jobs_path, result_path, provenance


def test_validate_result_accepts_all_five_aggregate_record_levels(tmp_path):
    result = _result()

    validated = _validate_result(result, result_path=tmp_path / "final.json")

    intervention = validated["interventions"][0]
    assert len(intervention["response_records"]) == 1
    assert len(intervention["response_class_records"]) == 2
    assert len(intervention["assigned_soma_response_records"]) == 1
    assert len(intervention["assigned_soma_class_response_records"]) == 2


def test_validate_result_accepts_fraction_summary_roundoff(tmp_path):
    result = _result()
    fraction = result["interventions"][0]["named_region_feature_fraction"]
    fraction["mean"] = 0.24999999999999994
    fraction["minimum"] = 0.24999999999999997
    fraction["maximum"] = 0.24999999999999997

    _validate_result(result, result_path=tmp_path / "final.json")


def test_validate_result_rejects_material_fraction_ordering_error(tmp_path):
    result = _result()
    fraction = result["interventions"][0]["named_region_feature_fraction"]
    fraction["mean"] = 0.2
    fraction["minimum"] = 0.25

    with pytest.raises(ValueError, match="minimum/mean/maximum ordering"):
        _validate_result(result, result_path=tmp_path / "final.json")


def test_collector_writes_separate_tables_and_preserves_scientific_coordinates(
    tmp_path,
):
    jobs_path, result_path, provenance = _write_run(tmp_path)
    output_dir = tmp_path / "collection"

    release = collect_input_region_intervention(
        jobs_paths=[jobs_path], output_dir=output_dir, expected_count=1
    )

    expected_counts = {
        "checkpoint_classification_records.csv": 1,
        "checkpoint_aggregate_response_records.csv": 1,
        "checkpoint_per_class_response_records.csv": 2,
        "checkpoint_assigned_soma_response_records.csv": 1,
        "checkpoint_assigned_soma_per_class_response_records.csv": 2,
        "source_artifacts.csv": 1,
    }
    tables = {}
    for filename, expected_count in expected_counts.items():
        with (output_dir / filename).open(encoding="utf-8") as handle:
            tables[filename] = list(csv.DictReader(handle))
        assert len(tables[filename]) == expected_count

    row = tables["checkpoint_aggregate_response_records.csv"][0]
    assert row["job_seed"] == "23"
    assert row["region_name"] == "center"
    assert row["mask_scope"] == "fixed_spatial"
    assert row["replacement_method"] == "zero"
    assert row["integration_mechanism"] == "shunting"
    assert row["source_current_polarity"] == "inhibitory"
    assert row["target_population_name"] == "e0"
    assert row["network_layer_index"] == "0"
    assert row["soma_relative_dendritic_depth"] == "1"
    assigned = tables["checkpoint_assigned_soma_response_records.csv"][0]
    assert assigned["identity_soma_class_mapping"] == "[0,1]"
    artifact = tables["source_artifacts.csv"][0]
    assert artifact["result_sha256"] == _sha256(result_path)
    assert artifact["checkpoint_sha256"] == provenance["checkpoint_sha256"]
    assert artifact["sample_level_records_present"] == "False"
    assert release["aggregation_policy"] == "none"
    assert release["trained_seed_aggregation_performed"] is False
    assert release["sample_level_records_collected"] is False


def test_validate_result_rejects_sample_level_payloads(tmp_path):
    result = _result()
    result["interventions"][0]["response_records"][0]["activation_samples"] = [
        0.1,
        0.2,
    ]

    with pytest.raises(ValueError, match="sample-level payload"):
        _validate_result(result, result_path=tmp_path / "final.json")


def test_validate_result_rejects_incomplete_per_class_partition(tmp_path):
    result = _result()
    result["interventions"][0]["response_class_records"].pop()

    with pytest.raises(ValueError, match="do not partition"):
        _validate_result(result, result_path=tmp_path / "final.json")


def test_collection_manifest_resolves_paths_and_hash_policy(tmp_path):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            (
                f"schema_version: {CONFIG_SCHEMA_VERSION}",
                "jobs:",
                "  - jobs.jsonl",
                "output_dir: outputs",
                "expected_count: 5",
                "input_hash_validation: recorded_hash_chain",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_collection_config(config)

    assert loaded["jobs_paths"] == [jobs.resolve()]
    assert loaded["output_dir"] == (tmp_path / "outputs").resolve()
    assert loaded["expected_count"] == 5
    assert loaded["input_hash_validation"] == "recorded_hash_chain"
