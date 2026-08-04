"""Tests for clean-aligned frozen-checkpoint perturbation collection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from dendritic_modeling.scripts.analysis.collect_perturbation_response import (
    collect_perturbation_response,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "training.yaml"
    checkpoint = tmp_path / "model.pt"
    analysis = tmp_path / "analysis.yaml"
    config.write_text("seed: 7\n")
    checkpoint.write_bytes(b"checkpoint")
    analysis.write_text(
        "analysis:\n" "  noise_perturbation:\n" "    params:\n" "      n_samples: 5\n"
    )
    output = tmp_path / "run"
    results = {
        "performance/final.json": {"accuracy": {"test": 0.5}},
        "noise_perturbation_analysis/final": {
            "gaussian": {"0.2": {"accuracy": 0.3}},
            "uniform": {"0.1": {"accuracy": 0.6}},
        },
        "multiplicative_gain_analysis/final.json": {
            "gain_factors": [0.5, 1.0, 2.0],
            "raw_scores": {"accuracy": [0.4, 0.5, 0.2]},
        },
    }
    for relative, payload in results.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "enabled_analyzers": [
            "performance",
            "noise_perturbation",
            "multiplicative_gain",
        ],
        "output_dir": str(output),
    }
    for field, path in (
        ("config", config),
        ("checkpoint", checkpoint),
        ("analysis_config", analysis),
    ):
        provenance[f"{field}_path"] = str(path)
        provenance[f"{field}_sha256"] = _sha256(path)
    (output / "checkpoint_analysis_provenance.json").write_text(json.dumps(provenance))
    job = {
        "id": "shunting_seed_7",
        "output_dir": str(output),
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "analysis_config": str(analysis),
        "analysis_config_sha256": _sha256(analysis),
        "integration_rule": "shunting",
        "training_seed": 7,
    }
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text(json.dumps(job) + "\n")
    return jobs, output / "multiplicative_gain_analysis/final.json"


def test_collector_aligns_clean_and_perturbed_accuracy(tmp_path: Path) -> None:
    jobs, _gain_path = _write_run(tmp_path)
    output = tmp_path / "collection"

    release = collect_perturbation_response(
        jobs_paths=[jobs],
        output_dir=output,
        chance_accuracy=0.1,
        expected_count=1,
        input_hash_validation="recorded_hash_chain",
    )

    assert release["collected_checkpoint_count"] == 1
    assert release["response_record_count"] == 5
    with (output / "checkpoint_perturbation_response.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    gaussian = next(row for row in rows if row["perturbation"] == "gaussian")
    uniform = next(row for row in rows if row["perturbation"] == "uniform")
    gain_one = next(
        row
        for row in rows
        if row["perturbation"] == "multiplicative_gain"
        and row["perturbation_level"] == "1.0"
    )
    assert float(gaussian["retention_above_chance"]) == pytest.approx(0.5)
    assert float(uniform["retention_above_chance"]) == pytest.approx(1.25)
    assert float(gain_one["retention_above_chance"]) == pytest.approx(1.0)
    assert gaussian["technical_noise_draws"] == "5"
    assert gain_one["technical_noise_draws"] == "1"
    assert all(row["job_training_seed"] == "7" for row in rows)


def test_collector_rejects_misaligned_gain_vectors(tmp_path: Path) -> None:
    jobs, gain_path = _write_run(tmp_path)
    payload = json.loads(gain_path.read_text())
    payload["raw_scores"]["accuracy"].pop()
    gain_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different lengths"):
        collect_perturbation_response(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collection",
            chance_accuracy=0.1,
            expected_count=1,
            input_hash_validation="recorded_hash_chain",
        )
