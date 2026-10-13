from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from dendritic_modeling.scripts.analysis.collect_branch_local_sensitivity import (
    _load_collection_config,
    collect_branch_local_sensitivities,
    main,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, include_soma: bool = False) -> Path:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pt"
    analysis = tmp_path / "analysis.yaml"
    for path, content in (
        (config, "model: {}\n"),
        (checkpoint, "frozen checkpoint\n"),
        (analysis, "analysis: {}\n"),
    ):
        path.write_text(content, encoding="utf-8")
    output = tmp_path / "run"
    result_dir = output / "branch_local_sensitivity"
    result_dir.mkdir(parents=True)
    records = []
    modules = {"1": "tree.1", "2": "tree.0"}
    if include_soma:
        modules["0"] = "tree.2"
    for depth in range(0 if include_soma else 1, 3):
        for component in ("E", "I", "C", "G", "V"):
            records.append(
                {
                    "module_name": modules[str(depth)],
                    "soma_relative_depth": depth,
                    "integration_mode": "shunting",
                    "component": component,
                    "coordinate_count": 4 * (2**depth),
                    "sample_coordinate_count": 400 * (2**depth),
                    "component_mean": 1.0,
                    "component_std": 0.1,
                    "mean_signed_doutput_dcomponent": -0.2 if component == "I" else 0.2,
                    "mean_abs_doutput_dcomponent": 0.2,
                    "rms_doutput_dcomponent": 0.21,
                    "mean_abs_first_order_contribution": 0.3,
                }
            )
    result = {
        "schema_version": 1,
        "analysis_type": "branch_local_sensitivity",
        "estimand": "local diagonal derivative",
        "evaluation_split": "test",
        "depth_reference": "soma_relative",
        "module_name_prefix": "tree",
        "tree": {
            "n_somas": 4,
            "branch_factors": [2, 2],
            "module_names_by_depth": modules,
        },
        "records": records,
    }
    (result_dir / "final.json").write_text(json.dumps(result), encoding="utf-8")
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "output_dir": str(output.resolve()),
        "enabled_analyzers": ["branch_local_sensitivity"],
    }
    for name, path in (
        ("config", config),
        ("checkpoint", checkpoint),
        ("analysis_config", analysis),
    ):
        provenance[f"{name}_path"] = str(path.resolve())
        provenance[f"{name}_sha256"] = _sha256(path)
    (output / "checkpoint_analysis_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(
        json.dumps(
            {
                "mechanism": "shunting",
                "training_seed": 3,
                "config": str(config),
                "checkpoint": str(checkpoint),
                "analysis_config": str(analysis),
                "output_dir": str(output),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return jobs


def test_collector_preserves_scientific_coordinates(tmp_path):
    jobs = _fixture(tmp_path)
    result = collect_branch_local_sensitivities(
        jobs_paths=[jobs], output_dir=tmp_path / "collected", expected_count=1
    )
    assert result["collected_job_count"] == 1
    with (tmp_path / "collected/checkpoint_records.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {row["soma_relative_depth"] for row in rows} == {"1", "2"}
    assert {row["component"] for row in rows} == {"E", "I", "C", "G", "V"}
    assert {row["integration_mode"] for row in rows} == {"shunting"}


def test_collector_accepts_complete_soma_and_branch_inventory(tmp_path):
    jobs = _fixture(tmp_path, include_soma=True)
    result = collect_branch_local_sensitivities(
        jobs_paths=[jobs], output_dir=tmp_path / "collected", expected_count=1
    )
    assert result["collected_job_count"] == 1
    with (tmp_path / "collected/checkpoint_records.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert {row["soma_relative_depth"] for row in rows} == {"0", "1", "2"}


def test_collector_rejects_another_provenance_schema(tmp_path):
    jobs = _fixture(tmp_path)
    job = json.loads(jobs.read_text(encoding="utf-8"))
    provenance_path = Path(job["output_dir"]) / "checkpoint_analysis_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    del provenance["schema_version"]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance schema"):
        collect_branch_local_sensitivities(
            jobs_paths=[jobs], output_dir=tmp_path / "collected"
        )


def test_collector_fails_closed_on_incomplete_component_grid(tmp_path):
    jobs = _fixture(tmp_path)
    job = json.loads(jobs.read_text(encoding="utf-8"))
    result_path = Path(job["output_dir"]) / "branch_local_sensitivity" / "final.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["records"].pop()
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="component coverage"):
        collect_branch_local_sensitivities(
            jobs_paths=[jobs], output_dir=tmp_path / "collected"
        )


def test_collection_config_resolves_relative_paths(tmp_path):
    jobs = _fixture(tmp_path)
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            (
                "schema_version: branch_local_sensitivity_collection_config_v1",
                f"jobs: [{jobs.name}]",
                "output_dir: collected",
                "expected_count: 1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = _load_collection_config(config)
    assert loaded["jobs_paths"] == [jobs.resolve()]
    assert loaded["output_dir"] == (tmp_path / "collected").resolve()


def test_branch_sensitivity_collector_cli_returns_success(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    jobs = _fixture(tmp_path)
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            (
                "schema_version: branch_local_sensitivity_collection_config_v1",
                f"jobs: [{jobs.name}]",
                "output_dir: collected",
                "expected_count: 1",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["--config", str(config)]) == 0
    assert "Collected 1 branch-sensitivity jobs" in capsys.readouterr().out
