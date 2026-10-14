"""Tests for validated representation-analysis collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dendritic_modeling.scripts.analysis.collect_representation_accessibility import (
    _load_collection_config,
    collect_representation_accessibility,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(root: Path, config_id: int) -> dict[str, object]:
    run_dir = root / f"run_{config_id}"
    result_dir = run_dir / "representation_accessibility"
    result_dir.mkdir(parents=True)
    config = root / f"config_{config_id}.yaml"
    checkpoint = root / f"model_{config_id}.pt"
    analysis = root / "analysis.yaml"
    config.write_text(f"seed: {config_id}\n")
    checkpoint.write_bytes(f"checkpoint-{config_id}".encode())
    analysis.write_text("analysis: {}\n")

    result = {
        "schema_version": 1,
        "analysis_type": "representation_accessibility",
        "fit_split": "valid",
        "evaluation_split": "test",
        "test_used_for_probe_selection": False,
        "inventory": [{"network_layer_number": 1, "representation": "excitatory"}],
        "missing_representations": [],
        "validation_summary": {"model_accuracy": 0.6},
        "test_summary": {"model_accuracy": 0.5 + config_id / 100},
        "probe_records": [
            {
                "network_layer_index": 0,
                "network_layer_number": 1,
                "representation": "excitatory",
                "population_names": ["e0"],
                "native_feature_count": 64,
                "projection_name": "matched16",
                "decoded_feature_count": 16,
                "projection_draw": draw,
                "probe_name": "linear",
                "probe_type": "ridge",
                "probe_family": "linear",
                "probe_draw": 0,
                "class_accessibility_above_null": 0.2 + 0.1 * draw,
                "heldout_test_balanced_accuracy": 0.3 + 0.1 * draw,
                "heldout_test_label_shuffle_balanced_accuracy": 0.1,
            }
            for draw in range(2)
        ],
        "joint_information_records": [],
        "per_soma_information_records": [],
        "per_soma_information_summary": [],
    }
    (result_dir / "final.json").write_text(json.dumps(result))
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "enabled_analyzers": ["representation_accessibility"],
        "config_path": str(config.resolve()),
        "config_sha256": _sha256(config),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "analysis_config_path": str(analysis.resolve()),
        "analysis_config_sha256": _sha256(analysis),
        "output_dir": str(run_dir.resolve()),
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "code_identity": {"git": {"commit": "abc"}},
    }
    (run_dir / "checkpoint_analysis_provenance.json").write_text(json.dumps(provenance))
    return {
        "config_id": config_id,
        "mechanism": "shunting",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "analysis_config": str(analysis),
        "output_dir": str(run_dir),
    }


def test_collection_validates_and_summarizes_technical_draws(tmp_path: Path) -> None:
    jobs = [_write_run(tmp_path, config_id) for config_id in range(2)]
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
    output_dir = tmp_path / "collected"

    provenance = collect_representation_accessibility(
        jobs_paths=[jobs_path],
        output_dir=output_dir,
        expected_count=2,
    )

    assert provenance["collected_job_count"] == 2
    checkpoints = pd.read_csv(output_dir / "checkpoint_summary.csv")
    summaries = pd.read_csv(output_dir / "probe_technical_summary.csv")
    assert len(checkpoints) == 2
    assert len(summaries) == 2
    assert summaries["technical_record_count"].tolist() == [2, 2]
    assert summaries["class_accessibility_above_null_mean"].tolist() == pytest.approx(
        [0.25, 0.25]
    )
    assert (output_dir / "collection_provenance.json").is_file()


def test_collection_rejects_incomplete_or_modified_inputs(tmp_path: Path) -> None:
    job = _write_run(tmp_path, 0)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    Path(str(job["checkpoint"])).write_bytes(b"modified")

    with pytest.raises(ValueError, match="checkpoint hash"):
        collect_representation_accessibility(
            jobs_paths=[jobs_path],
            output_dir=tmp_path / "collected",
            expected_count=1,
        )

    Path(str(job["output_dir"]), "representation_accessibility", "final.json").unlink()
    provenance = collect_representation_accessibility(
        jobs_paths=[jobs_path],
        output_dir=tmp_path / "partial",
        expected_count=1,
        allow_incomplete=True,
    )
    assert provenance["collected_job_count"] == 0
    assert len(provenance["skipped_jobs"]) == 1


def test_collection_requires_validation_information_metadata_but_allows_probe_only(
    tmp_path: Path,
) -> None:
    job = _write_run(tmp_path, 0)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    output = Path(str(job["output_dir"]))
    result_path = output / "representation_accessibility" / "final.json"

    # Probe-only outputs written before the information metadata legitimately predate
    # the explicit information-split fields.
    collect_representation_accessibility(
        jobs_paths=[jobs_path],
        output_dir=tmp_path / "probe_only",
        expected_count=1,
    )

    result = json.loads(result_path.read_text())
    result["joint_information_records"] = [
        {
            "network_layer_index": 0,
            "information_split": "test",
            "observed_mi_bits": 0.2,
        }
    ]
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="validation-only"):
        collect_representation_accessibility(
            jobs_paths=[jobs_path],
            output_dir=tmp_path / "invalid_information",
            expected_count=1,
        )

    result["information_split"] = "valid"
    result["test_used_for_information"] = False
    result["joint_information_records"][0]["information_split"] = "valid"
    result_path.write_text(json.dumps(result))
    release = collect_representation_accessibility(
        jobs_paths=[jobs_path],
        output_dir=tmp_path / "valid_information",
        expected_count=1,
    )
    assert release["collected_job_count"] == 1


def test_collection_requires_registered_analyzer_and_run_schema(tmp_path: Path) -> None:
    job = _write_run(tmp_path, 0)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    provenance_path = Path(str(job["output_dir"])) / (
        "checkpoint_analysis_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text())

    provenance["enabled_analyzers"] = []
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="analyzer is absent"):
        collect_representation_accessibility(
            jobs_paths=[jobs_path],
            output_dir=tmp_path / "disabled",
            expected_count=1,
        )

    provenance["enabled_analyzers"] = ["representation_accessibility"]
    provenance["schema_version"] = "old"
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="checkpoint-analysis schema"):
        collect_representation_accessibility(
            jobs_paths=[jobs_path],
            output_dir=tmp_path / "old_schema",
            expected_count=1,
        )


def test_representation_collection_config_resolves_paths(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("{}\n")
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: representation_accessibility_collection_config_v1",
                "jobs: [jobs.jsonl]",
                "output_dir: collected",
                "expected_count: 7",
                "allow_incomplete: false",
            ]
        )
        + "\n"
    )

    loaded = _load_collection_config(config)

    assert loaded["jobs_paths"] == [jobs.resolve()]
    assert loaded["output_dir"] == (tmp_path / "collected").resolve()
    assert loaded["expected_count"] == 7
    assert loaded["allow_incomplete"] is False
