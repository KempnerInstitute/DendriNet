"""Tests for the generic frozen-checkpoint analysis collector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import (
    _load_collection_config,
    _result_path,
    collect_checkpoint_analysis,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(root: Path, config_id: int) -> dict[str, object]:
    run_dir = root / f"run_{config_id}"
    result_dir = run_dir / "information_analysis"
    result_dir.mkdir(parents=True)
    config = root / f"config_{config_id}.yaml"
    checkpoint = root / f"model_{config_id}.pt"
    analysis = root / "analysis.yaml"
    config.write_text(f"seed: {config_id}\n")
    checkpoint.write_bytes(f"checkpoint-{config_id}".encode())
    analysis.write_text("analysis: {}\n")
    (result_dir / "final.json").write_text(
        json.dumps(
            {
                "analysis_split": "validation",
                "basic_mi": {"I(E;C)": 0.25 + config_id},
                "branch_sampling": {
                    "seed": 17,
                    "selected_indices": [1, 4],
                    "selection_depends_on_data_or_labels": False,
                },
            }
        )
    )
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "enabled_analyzers": ["information"],
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
    }
    (run_dir / "checkpoint_analysis_provenance.json").write_text(json.dumps(provenance))
    return {
        "config_id": config_id,
        "mechanism": "shunting",
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "analysis_config": str(analysis),
        "analysis_config_sha256": _sha256(analysis),
        "output_dir": str(run_dir),
    }


def test_generic_collector_preserves_results_and_numeric_leaves(tmp_path: Path) -> None:
    jobs = [_write_run(tmp_path, config_id) for config_id in range(2)]
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs))

    release = collect_checkpoint_analysis(
        jobs_paths=[jobs_path],
        output_dir=tmp_path / "collection",
        required_analyzers=["information"],
        expected_count=2,
    )

    assert release["collected_job_count"] == 2
    assert release["collected_result_count"] == 2
    metrics = pd.read_csv(tmp_path / "collection/checkpoint_scalar_metrics.csv")
    observed = metrics[metrics["metric_path"] == "basic_mi.I(E;C)"]
    assert observed["value"].tolist() == pytest.approx([0.25, 1.25])
    assert not metrics["independent_replicate"].any()
    records = (tmp_path / "collection/checkpoint_analysis_records.jsonl").read_text()
    assert len(records.strip().splitlines()) == 2


def test_generic_collector_rejects_modified_checkpoint_and_missing_result(
    tmp_path: Path,
) -> None:
    job = _write_run(tmp_path, 0)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    Path(str(job["checkpoint"])).write_bytes(b"modified")

    with pytest.raises(ValueError, match="checkpoint hash"):
        collect_checkpoint_analysis(
            jobs_paths=[jobs_path],
            output_dir=tmp_path / "modified",
            required_analyzers=["information"],
        )

    job = _write_run(tmp_path / "fresh", 0)
    jobs_path = tmp_path / "fresh_jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    Path(str(job["output_dir"]), "information_analysis", "final.json").unlink()
    with pytest.raises(FileNotFoundError):
        collect_checkpoint_analysis(
            jobs_paths=[jobs_path],
            output_dir=tmp_path / "missing",
            required_analyzers=["information"],
        )


def test_generic_collector_can_validate_recorded_input_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _write_run(tmp_path, 0)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    checkpoint = Path(str(job["checkpoint"])).resolve()

    from dendritic_modeling.scripts.analysis import (
        collect_checkpoint_analysis as module,
    )

    original_sha256 = module._sha256

    def fail_if_checkpoint_is_rehashed(path: Path, *, chunk_size: int = 2**20) -> str:
        if path.resolve() == checkpoint:
            raise AssertionError("recorded_hash_chain re-hashed the checkpoint")
        return original_sha256(path, chunk_size=chunk_size)

    monkeypatch.setattr(module, "_sha256", fail_if_checkpoint_is_rehashed)
    release = collect_checkpoint_analysis(
        jobs_paths=[jobs_path],
        output_dir=tmp_path / "recorded_chain",
        required_analyzers=["information"],
        input_hash_validation="recorded_hash_chain",
    )

    assert release["input_hash_validation"] == "recorded_hash_chain"
    assert "were not re-hashed" in release["input_hash_validation_semantics"]


def test_recorded_hash_chain_requires_job_and_provenance_agreement(
    tmp_path: Path,
) -> None:
    job = _write_run(tmp_path, 0)
    job["checkpoint_sha256"] = "0" * 64
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")

    with pytest.raises(ValueError, match="checkpoint hash differs"):
        collect_checkpoint_analysis(
            jobs_paths=[jobs_path],
            output_dir=tmp_path / "mismatch",
            required_analyzers=["information"],
            input_hash_validation="recorded_hash_chain",
        )


def test_generic_collector_accepts_extensionless_standard_result(
    tmp_path: Path,
) -> None:
    job = _write_run(tmp_path, 0)
    result = Path(str(job["output_dir"]), "information_analysis", "final.json")
    result.rename(result.with_suffix(""))
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")

    release = collect_checkpoint_analysis(
        jobs_paths=[jobs_path],
        output_dir=tmp_path / "collected",
        required_analyzers=["information"],
    )
    assert release["collected_result_count"] == 1


def test_generic_collector_resolves_reactivation_final_snapshot(tmp_path: Path) -> None:
    result = tmp_path / "reactivation_dynamics/epochs/final.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}\n")

    assert _result_path(tmp_path, "reactivation_dynamics") == result


def test_generic_collector_selects_declared_metrics_without_copying_results(
    tmp_path: Path,
) -> None:
    job = _write_run(tmp_path, 0)
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")

    collect_checkpoint_analysis(
        jobs_paths=[jobs_path],
        output_dir=tmp_path / "selected",
        required_analyzers=["information"],
        metric_patterns=["basic_mi.*"],
        include_full_results=False,
    )

    metrics = pd.read_csv(tmp_path / "selected/checkpoint_scalar_metrics.csv")
    assert metrics["metric_path"].tolist() == ["basic_mi.I(E;C)"]
    record = json.loads(
        (tmp_path / "selected/checkpoint_analysis_records.jsonl").read_text()
    )
    assert record["selected_metrics"] == {"basic_mi.I(E;C)": 0.25}
    assert "result" not in record


def test_generic_collector_loads_relative_yaml_config(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("{}\n")
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: checkpoint_analysis_collection_config_v1",
                "jobs: [jobs.jsonl]",
                "output_dir: collected",
                "required_analyzers: [information]",
                "expected_count: 1",
                "metric_patterns: ['basic_mi.*']",
                "include_full_results: false",
            ]
        )
        + "\n"
    )

    loaded = _load_collection_config(config)

    assert loaded["jobs_paths"] == [jobs.resolve()]
    assert loaded["output_dir"] == (tmp_path / "collected").resolve()
    assert loaded["required_analyzers"] == ["information"]
    assert loaded["metric_patterns"] == ["basic_mi.*"]
    assert loaded["include_full_results"] is False
