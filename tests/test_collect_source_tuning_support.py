"""Tests for the source-tuning/support checkpoint collector."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import _sha256
from dendritic_modeling.scripts.analysis.collect_source_tuning_support import (
    ANALYZER_NAME,
    CONFIG_SCHEMA_VERSION,
    RESULT_SUBDIRECTORY,
    _checkpoint_alignment_summary_rows,
    _load_collection_config,
    _validate_image_shape,
    _validate_result,
    collect_source_tuning_support,
)


def _preference_record() -> dict:
    return {
        "target_key": "layers.0.layer0.e0",
        "network_layer_index": 0,
        "network_layer_name": "layer0",
        "target_population": "e0",
        "target_polarity": "excitatory",
        "target_soma_index": 0,
        "preferred_class": 0,
        "preferred_reference_mean": 0.8,
        "runner_up_class": 1,
        "runner_up_reference_mean": 0.3,
        "preference_margin": 0.5,
    }


def _alignment_record() -> dict:
    record = {
        "source_block_key": (
            "layers.0.layer0.e0.depth_1.ff_inhibitory.source_0_input_i"
        ),
        "target_key": "layers.0.layer0.e0",
        "network_layer_index": 0,
        "network_layer_name": "layer0",
        "target_population": "e0",
        "target_polarity": "excitatory",
        "n_soma": 1,
        "dendritic_depth": 1,
        "pathway": "ff_inhibitory",
        "source_name": "input_i",
        "source_polarity": "inhibitory",
        "source_kind": "external_input",
        "source_layer_name": None,
        "source_population_name": None,
        "coordinate_space": "image_pixels",
        "image_shape": [1, 2],
        "source_start": 0,
        "source_stop": 2,
        "source_dim": 2,
        "target_soma_index": 0,
        "preferred_class": 0,
        "activity_threshold": 0.5,
        "activity_threshold_kind": "image_binarization",
        "n_branches": 2,
        "n_preferred_evaluation_samples": 2,
        "n_rest_evaluation_samples": 2,
        "candidate_contact_count": 4,
        "active_contact_count": 2,
        "active_fraction_of_candidates": 0.5,
        "total_effective_conductance": 1.2,
        "profile_key": (
            "layers.0.layer0.e0.depth_1.ff_inhibitory.source_0_input_i.soma_0"
        ),
        "n_source_coordinates": 2,
        "n_eligible_source_coordinates": 2,
        "n_class_profile_classes": 2,
        "n_degenerate_target_class_profiles": 0,
        "n_degenerate_source_class_profiles": 0,
        "n_eligible_degenerate_source_class_profiles": 0,
    }
    tuning_metrics = (
        "preferred_mean",
        "rest_mean",
        "mean_contrast",
        "preferred_activity_probability",
        "rest_activity_probability",
        "activity_probability_contrast",
        "standardized_mean_difference",
        "class_profile_alignment",
    )
    for weighting in (
        "source_uniform",
        "contact_weighted",
        "conductance_weighted",
    ):
        for metric in tuning_metrics:
            record[f"{weighting}_{metric}"] = 0.2
    for weighting in ("contact_weighted", "conductance_weighted"):
        for metric in (
            "mean_contrast",
            "activity_probability_contrast",
            "standardized_mean_difference",
            "class_profile_alignment",
        ):
            record[f"{weighting}_{metric}_excess_over_uniform"] = 0.0
    return record


def _result(*, retain_profile: bool = True) -> dict:
    alignment = _alignment_record()
    profile = {
        "coordinate_axis": "image_pixels",
        "source_dim": 2,
        "candidate_frequency": [1.0, 1.0],
        "exact_contact_frequency": [1.0, 0.0],
        "mean_effective_conductance": [0.6, 0.0],
        "preferred_mean": [0.8, 0.2],
        "rest_mean": [0.1, 0.3],
        "mean_contrast": [0.7, -0.1],
        "preferred_activity_probability": [1.0, 0.0],
        "rest_activity_probability": [0.0, 0.5],
        "activity_probability_contrast": [1.0, -0.5],
        "standardized_mean_difference": [1.2, -0.2],
        "class_profile_alignment": [1.0, -1.0],
    }
    return {
        "schema_version": 1,
        "analysis_type": ANALYZER_NAME,
        "quantity_contract": {
            "preferred_class": "reference",
            "source_tuning": "evaluation",
            "class_profile_alignment": "centered class means",
            "class_profile_degeneracy": "zero with counts",
            "source_uniform": "uniform",
            "contact_weighted": "contacts",
            "conductance_weighted": "conductance",
            "image_coordinates": "declared",
            "hidden_coordinates": "native neurons",
            "stored_state": "sufficient statistics only",
        },
        "require_independent_splits": True,
        "reference": {
            "split": "validation",
            "n_samples": 4,
            "class_counts": {"0": 2, "1": 2},
            "preferred_class_selection": "maximum class-conditional soma mean",
            "preference_records": [_preference_record()],
        },
        "evaluation": {
            "split": "test",
            "n_samples": 4,
            "class_counts": {"0": 2, "1": 2},
        },
        "n_alignment_records": 1,
        "alignment_records": [alignment],
        "coordinate_profiles": (
            {alignment["profile_key"]: profile} if retain_profile else {}
        ),
    }


def _write_run(tmp_path: Path) -> tuple[Path, Path, dict]:
    run_dir = tmp_path / "run"
    result_dir = run_dir / RESULT_SUBDIRECTORY
    result_dir.mkdir(parents=True)
    input_paths = {}
    for field, content in (
        ("config", "model: test\n"),
        ("checkpoint", "checkpoint bytes\n"),
        ("analysis_config", "analysis: test\n"),
    ):
        path = tmp_path / f"{field}.dat"
        path.write_text(content, encoding="utf-8")
        input_paths[field] = path.resolve()

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
    for field, path in input_paths.items():
        provenance[f"{field}_path"] = str(path)
        provenance[f"{field}_sha256"] = _sha256(path)
    provenance_path = run_dir / "checkpoint_analysis_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    job = {
        "output_dir": str(run_dir.resolve()),
        "config": str(input_paths["config"]),
        "checkpoint": str(input_paths["checkpoint"]),
        "analysis_config": str(input_paths["analysis_config"]),
        "seed": 17,
    }
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job, sort_keys=True) + "\n", encoding="utf-8")
    return jobs_path, result_path, provenance


def test_validate_result_accepts_split_safe_scalar_and_coordinate_records(tmp_path):
    result = _result()

    preferences, alignments, profiles = _validate_result(
        result,
        result_path=tmp_path / "final.json",
    )

    assert len(preferences) == 1
    assert len(alignments) == 1
    assert list(profiles) == [alignments[0]["profile_key"]]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", 999, "result schema"),
        ("analysis_type", "information", "analysis type"),
    ],
)
def test_validate_result_rejects_wrong_producer_contract(
    tmp_path,
    field,
    value,
    match,
):
    result = _result(retain_profile=False)
    result[field] = value

    with pytest.raises(ValueError, match=match):
        _validate_result(result, result_path=tmp_path / "final.json")


def test_collector_preserves_job_checkpoint_and_artifact_provenance(tmp_path):
    jobs_path, result_path, provenance = _write_run(tmp_path)
    output_dir = tmp_path / "collection"

    release = collect_source_tuning_support(
        jobs_paths=[jobs_path],
        output_dir=output_dir,
        expected_count=1,
    )

    with (output_dir / "checkpoint_source_tuning_alignment_records.csv").open(
        encoding="utf-8"
    ) as handle:
        alignment_rows = list(csv.DictReader(handle))
    with (output_dir / "checkpoint_source_tuning_preference_records.csv").open(
        encoding="utf-8"
    ) as handle:
        preference_rows = list(csv.DictReader(handle))
    with (output_dir / "checkpoint_source_tuning_alignment_summary.csv").open(
        encoding="utf-8"
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    with (output_dir / "source_artifacts.csv").open(encoding="utf-8") as handle:
        artifact_rows = list(csv.DictReader(handle))

    assert (
        len(alignment_rows)
        == len(preference_rows)
        == len(summary_rows)
        == len(artifact_rows)
        == 1
    )
    assert alignment_rows[0]["job_seed"] == "17"
    assert alignment_rows[0]["coordinate_space"] == "image_pixels"
    assert alignment_rows[0]["image_shape"] == "[1,2]"
    assert artifact_rows[0]["result_sha256"] == _sha256(result_path)
    assert artifact_rows[0]["checkpoint_sha256"] == provenance["checkpoint_sha256"]
    assert summary_rows[0]["technical_soma_count"] == "1"
    assert summary_rows[0]["soma_weighting"] == "equal"
    assert release["aggregation_policy"] == "equal_soma_mean_within_checkpoint_only"
    assert release["technical_soma_aggregation_performed"] is True
    assert release["trained_seed_aggregation_performed"] is False
    assert release["collected_checkpoint_count"] == 1
    assert set(release["outputs"]) == {
        "checkpoint_source_tuning_alignment_records.csv",
        "checkpoint_source_tuning_alignment_summary.csv",
        "checkpoint_source_tuning_preference_records.csv",
        "source_artifacts.csv",
    }


def test_checkpoint_summary_uses_equal_soma_weighting_and_population_sd():
    first = _alignment_record()
    second = copy.deepcopy(first)
    for record, soma_index, value, conductance in (
        (first, 0, 0.2, 1.0),
        (second, 1, 0.6, 9.0),
    ):
        record["n_soma"] = 2
        record["target_soma_index"] = soma_index
        record["preferred_class"] = soma_index
        record["profile_key"] = f"profile.soma_{soma_index}"
        record["conductance_weighted_class_profile_alignment"] = value
        record["total_effective_conductance"] = conductance
        record.update(
            {
                "job_index": 0,
                "job_id": "checkpoint_0",
                "job_training_seed": 17,
                "technical_alignment_record_is_independent_replicate": False,
                "trained_checkpoint_is_inference_unit": True,
            }
        )

    summaries = _checkpoint_alignment_summary_rows([first, second])

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["job_training_seed"] == 17
    assert summary["source_block_key"] == first["source_block_key"]
    assert summary["technical_soma_count"] == 2
    assert summary["soma_weighting"] == "equal"
    assert summary["technical_somas_are_independent_replicates"] is False
    assert summary["trained_checkpoint_is_inference_unit"] is True
    assert summary["conductance_weighted_class_profile_alignment_defined_count"] == 2
    assert summary[
        "conductance_weighted_class_profile_alignment_mean"
    ] == pytest.approx(0.4)
    assert summary["conductance_weighted_class_profile_alignment_sd"] == pytest.approx(
        0.2
    )
    # The nine-fold larger conductance of soma 1 does not weight the soma mean.
    assert summary["total_effective_conductance_mean"] == pytest.approx(5.0)


def test_checkpoint_summary_rejects_missing_or_duplicate_soma_records():
    first = _alignment_record()
    first["n_soma"] = 2
    first.update({"job_index": 0, "job_id": "checkpoint_0"})

    with pytest.raises(ValueError, match="exactly one record for every target soma"):
        _checkpoint_alignment_summary_rows([first])

    duplicate = copy.deepcopy(first)
    duplicate["profile_key"] = "duplicate-profile"
    with pytest.raises(ValueError, match="duplicate target somas"):
        _checkpoint_alignment_summary_rows([first, duplicate])


def test_validate_result_rejects_later_layer_coordinates_labeled_as_pixels(tmp_path):
    result = _result(retain_profile=False)
    result["alignment_records"][0]["network_layer_index"] = 1

    with pytest.raises(ValueError, match="later-layer native source"):
        _validate_result(result, result_path=tmp_path / "final.json")


def test_validate_result_rejects_alignment_preference_mismatch(tmp_path):
    result = _result(retain_profile=False)
    result["alignment_records"][0]["preferred_class"] = 1

    with pytest.raises(ValueError, match="reference preference"):
        _validate_result(result, result_path=tmp_path / "final.json")


def test_validate_result_checks_class_profile_fields_and_degeneracy(tmp_path):
    result = _result(retain_profile=False)
    del result["alignment_records"][0]["source_uniform_class_profile_alignment"]
    with pytest.raises(ValueError, match="class_profile_alignment"):
        _validate_result(result, result_path=tmp_path / "final.json")

    result = _result(retain_profile=False)
    result["alignment_records"][0]["n_degenerate_source_class_profiles"] = 3
    with pytest.raises(ValueError, match="degeneracy count exceeds source_dim"):
        _validate_result(result, result_path=tmp_path / "final.json")

    result = _result(retain_profile=False)
    result["alignment_records"][0]["n_degenerate_target_class_profiles"] = 1
    with pytest.raises(ValueError, match="must have zero alignment"):
        _validate_result(result, result_path=tmp_path / "final.json")


def test_validate_result_honors_declared_split_independence_policy(tmp_path):
    result = _result(retain_profile=False)
    result["evaluation"]["split"] = result["reference"]["split"]
    with pytest.raises(ValueError, match="independence policy"):
        _validate_result(result, result_path=tmp_path / "final.json")

    result["require_independent_splits"] = False
    preferences, alignments, profiles = _validate_result(
        result,
        result_path=tmp_path / "final.json",
    )
    assert len(preferences) == len(alignments) == 1
    assert profiles == {}


def test_collector_accepts_chw_image_metadata_and_rejects_it_for_hidden_sources():
    _validate_image_shape(
        [3, 32, 32],
        coordinate_space="image_pixels",
        source_dim=3072,
        context="record.image_shape",
    )

    with pytest.raises(ValueError, match="null for non-image"):
        _validate_image_shape(
            [3, 32, 32],
            coordinate_space="native_source_neurons",
            source_dim=3072,
            context="record.image_shape",
        )
    with pytest.raises(ValueError, match="product"):
        _validate_image_shape(
            [3, 32, 32],
            coordinate_space="image_pixels",
            source_dim=1024,
            context="record.image_shape",
        )


def test_collection_manifest_resolves_paths_and_result_override(tmp_path):
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
                "expected_count: 3",
                "result_subdirectory: custom_source_tuning",
                "result_filename: checkpoint_final",
                "input_hash_validation: recorded_hash_chain",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_collection_config(config)

    assert loaded["jobs_paths"] == [jobs.resolve()]
    assert loaded["output_dir"] == (tmp_path / "outputs").resolve()
    assert loaded["expected_count"] == 3
    assert loaded["result_subdirectory"] == "custom_source_tuning"
    assert loaded["result_filename"] == "checkpoint_final"
    assert loaded["input_hash_validation"] == "recorded_hash_chain"


def test_checkpoint_summary_retains_optional_operator_semantics():
    record = _alignment_record()
    metadata = {
        "selection_policy": "indexedreleaselinear",
        "transmission_mode": "mean_release",
        "signal_rounding": "float16",
    }
    record.update(metadata)
    summary = _checkpoint_alignment_summary_rows([record])[0]
    assert all(summary[key] == value for key, value in metadata.items())
    # Existing checkpoint artifacts remain readable without the new fields.
    assert _checkpoint_alignment_summary_rows([_alignment_record()])
