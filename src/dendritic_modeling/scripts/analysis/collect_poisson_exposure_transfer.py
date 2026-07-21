"""Collect Poisson exposure-transfer outputs at the trained-checkpoint level."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dendritic_modeling.analysis.tools.poisson_exposure_transfer import (
    poisson_exposure_draw_seed,
)
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "poisson_exposure_transfer_collection_v1"
RUN_SCHEMA_VERSION = "poisson_exposure_transfer_run_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _job_fields(job_index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_index": int(job_index),
        **{f"job_{key}": _csv_cell(value) for key, value in sorted(job.items())},
    }


def _read_jobs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    required = {"config", "checkpoint", "transfer_config", "output_dir"}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                job = json.loads(line)
                if not isinstance(job, dict) or not required.issubset(job):
                    raise ValueError(f"invalid job at {path}:{line_number}")
                jobs.append(job)
    outputs = [str(Path(str(job["output_dir"])).resolve()) for job in jobs]
    if not jobs or len(outputs) != len(set(outputs)):
        raise ValueError("jobs must be nonempty with unique output directories")
    return jobs


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if "job_index" in fields:
        fields.remove("job_index")
        fields.insert(0, "job_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["job_index"])
        writer.writeheader()
        writer.writerows(rows)


def _validate_run(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(str(job["output_dir"])).resolve()
    provenance_path = output / "poisson_exposure_transfer_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unexpected Poisson transfer provenance schema")
    if provenance.get("training_performed") is not False:
        raise ValueError("Poisson transfer unexpectedly performed training")
    if provenance.get("strict_checkpoint_loading") is not True:
        raise ValueError("Poisson transfer checkpoint loading was not strict")
    if provenance.get("missing_keys") or provenance.get("unexpected_keys"):
        raise ValueError("Poisson transfer checkpoint incompatibilities recorded")
    if Path(str(provenance.get("output_dir", ""))).resolve() != output:
        raise ValueError("Poisson transfer output directory differs from job")

    for job_key, provenance_key in (
        ("config", "config"),
        ("checkpoint", "checkpoint"),
        ("transfer_config", "transfer_config"),
    ):
        path = Path(str(job[job_key])).resolve()
        if Path(str(provenance[f"{provenance_key}_path"])).resolve() != path:
            raise ValueError(f"{job_key} path differs from provenance")
        if _sha256(path) != provenance[f"{provenance_key}_sha256"]:
            raise ValueError(f"{job_key} hash differs from provenance")

    result_path = Path(str(provenance["result_path"])).resolve()
    if (
        result_path.parent != output
        or _sha256(result_path) != provenance["result_sha256"]
    ):
        raise ValueError("Poisson transfer result hash differs from provenance")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unexpected Poisson transfer result schema")
    if result.get("training_performed") is not False:
        raise ValueError("Poisson transfer result has invalid training flag")
    if result.get("common_random_numbers") is not True:
        raise ValueError("Poisson transfer does not declare common random numbers")
    return provenance, result


def collect_poisson_exposure_transfer(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate draw records and average them within trained checkpoints."""

    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        provenance, result = _validate_run(job)
        base = _job_fields(job_index, job)
        records = result.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("Poisson transfer contains no draw records")
        expected_draws = int(result["draws_per_duration"])
        base_seed = int(result["base_draw_seed"])
        groups: dict[float, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[float, int]] = set()
        for record in records:
            duration = float(record["test_duration"])
            draw = int(record["draw"])
            identity = (duration, draw)
            if identity in seen:
                raise ValueError("duplicate Poisson duration/draw record")
            seen.add(identity)
            n_correct = int(record["n_correct"])
            n_samples = int(record["n_samples"])
            accuracy = float(record["accuracy"])
            if (
                n_samples < 1
                or n_correct < 0
                or n_correct > n_samples
                or not math.isfinite(accuracy)
                or not math.isclose(accuracy, n_correct / n_samples, abs_tol=1e-12)
            ):
                raise ValueError("invalid Poisson accuracy record")
            if int(record["draw_seed"]) != poisson_exposure_draw_seed(
                base_seed, duration, draw
            ):
                raise ValueError("Poisson draw seed differs from declared rule")
            groups[duration].append(record)
            draw_rows.append(
                {
                    **base,
                    **record,
                    "technical_draw_is_model_replicate": False,
                }
            )

        declared_durations = {float(value) for value in result["test_durations"]}
        if set(groups) != declared_durations:
            raise ValueError("Poisson test-duration coverage is incomplete")
        for duration, group in sorted(groups.items()):
            if len(group) != expected_draws:
                raise ValueError("Poisson technical draw count is incomplete")
            accuracies = np.asarray([float(record["accuracy"]) for record in group])
            checkpoint_rows.append(
                {
                    **base,
                    "base_dataset": result["base_dataset"],
                    "experiment_seed": int(result["experiment_seed"]),
                    "train_duration": float(group[0]["train_duration"]),
                    "test_duration": duration,
                    "accuracy": float(accuracies.mean()),
                    "accuracy_technical_std": (
                        float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0
                    ),
                    "technical_draw_count": len(group),
                    "technical_draws_are_model_replicates": False,
                }
            )
        inventory_rows.append(
            {
                **base,
                "result_sha256": provenance["result_sha256"],
                "record_count": len(records),
                "test_duration_count": len(groups),
                "draws_per_duration": expected_draws,
            }
        )

    outputs = {
        "draw_records.csv": draw_rows,
        "checkpoint_records.csv": checkpoint_rows,
        "checkpoint_inventory.csv": inventory_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_dir / name, rows)
    release = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_paths": [str(path.resolve()) for path in jobs_paths],
        "jobs_sha256": [_sha256(path.resolve()) for path in jobs_paths],
        "collected_checkpoint_count": len(jobs),
        "technical_draws_are_model_replicates": False,
        "checkpoint_records_are_inferential_units": True,
        "source_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
        "outputs": {name: _sha256(output_dir / name) for name in outputs},
    }
    (output_dir / "collection_provenance.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return collect_poisson_exposure_transfer(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
