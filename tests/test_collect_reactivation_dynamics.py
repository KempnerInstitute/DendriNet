"""Tests for the frozen-checkpoint reactivation collector."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dendritic_modeling.scripts.analysis.collect_reactivation_dynamics import (
    _load_collection_config,
    _module_coordinates,
    collect_reactivation_dynamics,
    main,
)

_MODULE = "core_network.layers.0.populations.e0.branch_layers.0.reactivation"
_COMMIT_A = "a" * 40
_COMMIT_B = "b" * 40


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(*, active: float = 0.7) -> dict:
    return {
        "soma_relative_dendritic_depth": 0,
        "population_polarity": "inhibitory",
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
        "core_network.layers.4.populations.e0.branch_layers.3.reactivation",
        {"soma_relative_dendritic_depth": 1, "population_polarity": "excitatory"},
    )

    assert coordinates == {
        "structured_population_path": True,
        "network_layer_index_zero_based": 4,
        "network_layer_position": 5,
        "network_layer_coordinate_inferred_single_layer": False,
        "population_name": "e0",
        "population_type": "excitatory",
        "branch_layer_list_index": 3,
        "soma_relative_dendritic_depth": 1,
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
    assert modules[0]["soma_relative_dendritic_depth"] == "0"
    assert modules[0]["branch_layer_list_index"] == "2"
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


def test_collection_config_resolves_relative_paths(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("{}\n")
    config = tmp_path / "collection.yaml"
    config.write_text(
        "\n".join(
            [
                "schema_version: reactivation_dynamics_collection_config_v1",
                "jobs: [jobs.jsonl]",
                "output_dir: collected",
                "expected_count: 3",
                "require_structured_coordinates: true",
            ]
        )
        + "\n"
    )

    loaded = _load_collection_config(config)

    assert loaded["jobs_paths"] == [jobs.resolve()]
    assert loaded["output_dir"] == (tmp_path / "collected").resolve()
    assert loaded["expected_count"] == 3
    assert loaded["require_structured_coordinates"] is True


def test_collector_rejects_results_without_analyzer_depth(tmp_path: Path) -> None:
    jobs, result_path = _write_run(
        tmp_path,
        module_name="core_network.layers.0.populations.e0.branch_layers.0.reactivation",
    )
    payload = json.loads(result_path.read_text())
    next(iter(payload["layers"].values())).pop("soma_relative_dendritic_depth")
    result_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="re-run the reactivation analysis"):
        collect_reactivation_dynamics(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collection",
            expected_count=1,
        )


def _set_analysis_source(jobs: Path, edit=None, **git_fields) -> None:
    """Record the Git identity of the analysis source in a run's provenance."""
    job = json.loads(jobs.read_text().splitlines()[0])
    path = Path(job["output_dir"]) / "checkpoint_analysis_provenance.json"
    provenance = json.loads(path.read_text())
    git = {
        "available": True,
        "commit": _COMMIT_A,
        "tracked_worktree_dirty": False,
        "untracked_src_files": [],
        **git_fields,
    }
    if edit is not None:
        edit(git)
    provenance["code_identity"] = {"repository_name": "dendritic-modeling", "git": git}
    path.write_text(json.dumps(provenance))


def test_the_analysis_source_check_is_off_by_default(tmp_path: Path) -> None:
    """Runs without a recorded source identity still collect, and the release
    record is unchanged."""
    jobs, _result = _write_run(tmp_path, module_name=_MODULE)

    release = collect_reactivation_dynamics(
        jobs_paths=[jobs], output_dir=tmp_path / "collection", expected_count=1
    )

    assert release["collected_checkpoint_count"] == 1
    assert "require_clean_analysis_source" not in release
    assert "analysis_source_commit" not in release


def test_a_clean_analysis_source_is_recorded(tmp_path: Path) -> None:
    jobs, _result = _write_run(tmp_path, module_name=_MODULE)
    _set_analysis_source(jobs)

    release = collect_reactivation_dynamics(
        jobs_paths=[jobs],
        output_dir=tmp_path / "collection",
        expected_count=1,
        require_clean_analysis_source=True,
    )

    assert release["require_clean_analysis_source"] is True
    assert release["analysis_source_commit"] == _COMMIT_A
    recorded = json.loads(
        (tmp_path / "collection/collection_provenance.json").read_text()
    )
    assert recorded["analysis_source_commit"] == _COMMIT_A


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (None, "records no Git source identity"),
        ({"available": False}, "records no Git source identity"),
        ({"commit": "abc123"}, "records no valid commit"),
        ({"tracked_worktree_dirty": True}, "ran with tracked changes"),
        ({"untracked_src_files": ["src/extra.py"]}, "untracked files under src/"),
        ("drop untracked list", "untracked files under src/"),
    ],
)
def test_an_unclean_analysis_source_is_rejected(
    tmp_path: Path, change, message: str
) -> None:
    jobs, _result = _write_run(tmp_path, module_name=_MODULE)
    if isinstance(change, dict):
        _set_analysis_source(jobs, **change)
    elif change == "drop untracked list":
        _set_analysis_source(jobs, edit=lambda git: git.pop("untracked_src_files"))

    with pytest.raises(ValueError, match=message):
        collect_reactivation_dynamics(
            jobs_paths=[jobs],
            output_dir=tmp_path / "collection",
            expected_count=1,
            require_clean_analysis_source=True,
        )


def test_analyses_from_different_commits_are_rejected(tmp_path: Path) -> None:
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
    first, _ = _write_run(tmp_path / "first", module_name=_MODULE)
    second, _ = _write_run(tmp_path / "second", module_name=_MODULE)
    _set_analysis_source(first, commit=_COMMIT_A)
    _set_analysis_source(second, commit=_COMMIT_B)
    kwargs = {"jobs_paths": [first, second], "output_dir": tmp_path / "collection"}

    collect_reactivation_dynamics(**kwargs)  # accepted without the check
    with pytest.raises(ValueError, match="ran from different commits"):
        collect_reactivation_dynamics(**kwargs, require_clean_analysis_source=True)


def test_the_check_is_set_by_the_config_or_the_command_line(tmp_path: Path) -> None:
    jobs, _result = _write_run(tmp_path, module_name=_MODULE)
    config = tmp_path / "collection.yaml"
    config.write_text(
        "schema_version: reactivation_dynamics_collection_config_v1\n"
        "jobs: [jobs.jsonl]\n"
        "output_dir: collected\n"
        "require_clean_analysis_source: true\n"
    )

    assert _load_collection_config(config)["require_clean_analysis_source"] is True
    with pytest.raises(ValueError, match="records no Git source identity"):
        main(["--config", str(config)])
    with pytest.raises(ValueError, match="records no Git source identity"):
        main(
            [
                "--jobs",
                str(jobs),
                "--output-dir",
                str(tmp_path / "cli"),
                "--require-clean-analysis-source",
            ]
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        main(["--config", str(config), "--require-clean-analysis-source"])
