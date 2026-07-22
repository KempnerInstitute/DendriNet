"""Tests for the frozen-checkpoint reactivation collector."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dendritic_modeling.scripts.analysis.collect_reactivation_dynamics import (
    _module_coordinates,
    collect_reactivation_dynamics,
)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(*, active: float = 0.7) -> dict:
    return {
        "V": {
            "mean": 0.5,
            "std": 0.1,
            "median": 0.5,
            "mad_std": 0.1,
            "q01": 0.2,
            "q10": 0.3,
            "q50": 0.5,
            "q90": 0.7,
            "q99": 0.8,
            "min": 0.1,
            "max": 0.9,
            "n": 128,
        },
        "params": {
            "log_m_mean": 0.0,
            "log_m_std": 0.1,
            "m_mean": 1.0,
            "m_std": 0.1,
            "m_min": 0.8,
            "m_max": 1.2,
            "b_mean": 0.5,
            "b_std": 0.1,
            "b_min": 0.3,
            "b_max": 0.7,
        },
        "reactivation_output": {
            "mean": 0.5,
            "std": 0.2,
            "sat_lo": 0.1,
            "sat_hi": 0.2,
            "active_frac": active,
        },
    }


def _write_run(tmp_path: Path, *, module_name: str) -> tuple[Path, Path]:
    config = tmp_path / "training.yaml"
    checkpoint = tmp_path / "model.pt"
    analysis = tmp_path / "analysis.yaml"
    config.write_text("seed: 1\n")
    checkpoint.write_bytes(b"checkpoint")
    analysis.write_text("analysis: {}\n")
    output = tmp_path / "run"
    result_path = output / "reactivation_dynamics/epochs/final.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "filename": "final",
                "training": False,
                "n_layers": 1,
                "layers": {module_name: _entry()},
            }
        )
    )
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "enabled_analyzers": ["reactivation_dynamics"],
        "output_dir": str(output),
    }
    for field, path in (
        ("config", config),
        ("checkpoint", checkpoint),
        ("analysis_config", analysis),
    ):
        provenance[f"{field}_path"] = str(path)
        provenance[f"{field}_sha256"] = _sha256(path)
    provenance_path = output / "checkpoint_analysis_provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    job = {
        "id": "condition_seed_1",
        "output_dir": str(output),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "analysis_config": str(analysis),
        "lineage": "shunting",
        "training_seed": 1,
    }
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(job) + "\n")
    return jobs, result_path


def test_module_coordinates_parse_population_network_path() -> None:
    coordinates = _module_coordinates(
        "core_network.layers.4.populations.e0.branch_layers.3.reactivation"
    )

    assert coordinates == {
        "structured_population_path": True,
        "network_layer_index_zero_based": 4,
        "network_layer_position": 5,
        "network_layer_coordinate_inferred_single_layer": False,
        "population_name": "e0",
        "population_type": "excitatory",
        "soma_relative_dendritic_depth": 3,
    }


def test_collector_emits_structured_module_and_metric_tables(tmp_path: Path) -> None:
    jobs, _result = _write_run(
        tmp_path,
        module_name="core_network.layers.1.populations.i0.branch_layers.2.reactivation",
    )
    output = tmp_path / "collection"

    release = collect_reactivation_dynamics(
        jobs_paths=[jobs],
        output_dir=output,
        expected_count=1,
        require_structured_coordinates=True,
    )

    assert release["collected_checkpoint_count"] == 1
    assert release["module_record_count"] == 1
    with (output / "checkpoint_modules.csv").open() as handle:
        modules = list(csv.DictReader(handle))
    assert modules[0]["network_layer_position"] == "2"
    assert modules[0]["population_type"] == "inhibitory"
    assert modules[0]["soma_relative_dendritic_depth"] == "2"
    assert modules[0]["output_active_frac"] == "0.7"
    with (output / "checkpoint_metric_records.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert {row["metric_group"] for row in metrics} == {
        "pre_gate_voltage",
        "reactivation_parameter",
        "post_gate_output",
    }
    assert all(row["independent_replicate"] == "False" for row in metrics)


def test_collector_rejects_unstructured_path_when_required(tmp_path: Path) -> None:
    jobs, _result = _write_run(tmp_path, module_name="r1")

    with pytest.raises(ValueError, match="standard population coordinates"):
        collect_reactivation_dynamics(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collection",
            expected_count=1,
            require_structured_coordinates=True,
        )


def test_collector_rejects_invalid_occupancy_partition(tmp_path: Path) -> None:
    jobs, result_path = _write_run(
        tmp_path,
        module_name="core_network.layers.0.populations.e0.branch_layers.0.reactivation",
    )
    payload = json.loads(result_path.read_text())
    next(iter(payload["layers"].values()))["reactivation_output"]["active_frac"] = 0.5
    result_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fractions do not sum to one"):
        collect_reactivation_dynamics(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collection",
            expected_count=1,
        )
