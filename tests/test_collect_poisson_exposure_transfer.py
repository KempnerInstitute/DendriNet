"""Tests for Poisson exposure-transfer collection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dendritic_modeling.analysis.tools.poisson_exposure_transfer import (
    poisson_exposure_draw_seed,
)
from dendritic_modeling.scripts.analysis.collect_poisson_exposure_transfer import (
    collect_poisson_exposure_transfer,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_job(root: Path, seed: int) -> dict[str, object]:
    output = root / f"run_{seed}"
    output.mkdir(parents=True)
    config = root / f"config_{seed}.yaml"
    checkpoint = root / f"checkpoint_{seed}.pt"
    transfer = root / "transfer.yaml"
    config.write_text(f"seed: {seed}\n")
    checkpoint.write_bytes(f"model-{seed}".encode())
    transfer.write_text("draws_per_duration: 2\n")
    records = []
    for duration in (0.1, 1.0):
        for draw, accuracy in enumerate((0.4 + seed / 10, 0.6 + seed / 10)):
            n_samples = 10
            n_correct = round(accuracy * n_samples)
            records.append(
                {
                    "train_duration": 0.1,
                    "test_duration": duration,
                    "draw": draw,
                    "draw_seed": poisson_exposure_draw_seed(73000, duration, draw),
                    "n_correct": n_correct,
                    "n_samples": n_samples,
                    "accuracy": n_correct / n_samples,
                }
            )
    result = {
        "schema_version": "poisson_exposure_transfer_run_v1",
        "training_performed": False,
        "base_dataset": "cifar10",
        "experiment_seed": seed,
        "test_durations": [0.1, 1.0],
        "draws_per_duration": 2,
        "base_draw_seed": 73000,
        "common_random_numbers": True,
        "records": records,
    }
    result_path = output / "poisson_exposure_transfer.json"
    result_path.write_text(json.dumps(result))
    provenance = {
        "schema_version": "poisson_exposure_transfer_run_v1",
        "training_performed": False,
        "strict_checkpoint_loading": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "config_path": str(config.resolve()),
        "config_sha256": _sha256(config),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "transfer_config_path": str(transfer.resolve()),
        "transfer_config_sha256": _sha256(transfer),
        "output_dir": str(output.resolve()),
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
    }
    (output / "poisson_exposure_transfer_provenance.json").write_text(
        json.dumps(provenance)
    )
    return {
        "seed": seed,
        "mechanism": "shunting",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "transfer_config": str(transfer),
        "output_dir": str(output),
    }


def test_poisson_transfer_collector_reduces_draws_within_checkpoint(
    tmp_path: Path,
) -> None:
    jobs = [_write_job(tmp_path, seed) for seed in range(2)]
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs))
    release = collect_poisson_exposure_transfer(
        jobs_paths=[jobs_path], output_dir=tmp_path / "collection", expected_count=2
    )

    assert release["collected_checkpoint_count"] == 2
    records = pd.read_csv(tmp_path / "collection/checkpoint_records.csv")
    assert len(records) == 4
    assert records.loc[records.job_seed == 0, "accuracy"].tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert not records["technical_draws_are_model_replicates"].any()


def test_poisson_transfer_collector_rejects_modified_result(tmp_path: Path) -> None:
    job = _write_job(tmp_path, 0)
    result = Path(str(job["output_dir"])) / "poisson_exposure_transfer.json"
    result.write_text("{}")
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps(job) + "\n")
    with pytest.raises(ValueError, match="result hash"):
        collect_poisson_exposure_transfer(
            jobs_paths=[jobs_path], output_dir=tmp_path / "collection"
        )
