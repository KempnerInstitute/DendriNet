from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from dendritic_modeling.scripts.analysis.collect_gain_load_response import (
    collect_gain_load_response,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "training.yaml"
    checkpoint = tmp_path / "model.pt"
    analysis = tmp_path / "analysis.yaml"
    config.write_text("seed: 7\n")
    checkpoint.write_bytes(b"checkpoint")
    analysis.write_text("analysis:\n  gain_load_perturbation:\n    enabled: true\n")
    output = tmp_path / "run"
    result_path = output / "gain_load_perturbation/final.json"
    result_path.parent.mkdir(parents=True)
    records = []
    for gain in (0.0, 0.5):
        for load in (0.0, 0.3):
            for draw in (0, 1):
                accuracy = 0.5 - 0.1 * gain - 0.2 * load + 0.01 * draw
                if gain == 0 and load == 0:
                    accuracy = 0.5
                records.append(
                    {
                        "gain_log_sd": gain,
                        "load_level": load,
                        "draw": draw,
                        "draw_seed": 1000 + draw,
                        "gain_draw_seed": int(100 * gain) + draw,
                        "load_draw_seed": int(100 * load) + draw,
                        "n_samples": 100,
                        "accuracy": accuracy,
                    }
                )
    result_path.write_text(
        json.dumps(
            {
                "gain_model": "one mean-one lognormal scalar per trial",
                "load_mode": "gaussian_input",
                "gain_log_sds": [0.0, 0.5],
                "load_levels": [0.0, 0.3],
                "draws_per_condition": 2,
                "common_random_numbers": True,
                "technical_draws_are_inference_units": False,
                "records": records,
            }
        )
    )
    provenance = {
        "schema_version": "checkpoint_analysis_run_v1",
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "enabled_analyzers": ["gain_load_perturbation"],
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
    return jobs, result_path


def test_collector_reduces_draws_within_checkpoint(tmp_path: Path) -> None:
    jobs, _result = _write_run(tmp_path)
    output = tmp_path / "collection"
    release = collect_gain_load_response(
        jobs_paths=[jobs],
        output_dir=output,
        chance_accuracy=0.1,
        expected_count=1,
        input_hash_validation="recorded_hash_chain",
    )
    assert release["draw_record_count"] == 8
    assert release["checkpoint_condition_count"] == 4
    assert release["input_statistic_record_count"] == 0
    assert release["branch_sensitivity_record_count"] == 0
    assert (output / "gain_load_input_statistics.csv").is_file()
    assert (output / "gain_load_branch_sensitivity.csv").is_file()
    with (output / "gain_load_checkpoint_response.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    target = next(
        row
        for row in rows
        if row["gain_log_sd"] == "0.5" and row["load_level"] == "0.3"
    )
    expected_accuracy = (0.39 + 0.40) / 2
    assert float(target["perturbed_accuracy"]) == pytest.approx(expected_accuracy)
    assert float(target["retention_above_chance"]) == pytest.approx(
        (expected_accuracy - 0.1) / 0.4
    )
    assert target["technical_draws"] == "2"
    assert target["job_training_seed"] == "7"


def test_collector_rejects_incomplete_factorial(tmp_path: Path) -> None:
    jobs, result_path = _write_run(tmp_path)
    payload = json.loads(result_path.read_text())
    payload["records"].pop()
    result_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete gain/load factorial"):
        collect_gain_load_response(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collection",
            chance_accuracy=0.1,
            input_hash_validation="recorded_hash_chain",
        )
