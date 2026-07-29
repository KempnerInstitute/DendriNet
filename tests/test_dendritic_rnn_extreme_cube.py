from __future__ import annotations

import json

from dendritic_modeling.scripts.analysis import (
    analyze_dendritic_rnn_extreme_cube as analysis,
)
from dendritic_modeling.scripts.experiments import dendritic_rnn_extreme_cube as cube


def test_materialize_and_validate_exact_twenty(tmp_path):
    generated = tmp_path / "generated"
    results = tmp_path / "results"
    manifest_path = cube.materialize(
        generated_dir=generated,
        results_root=results,
    )

    summary = cube.validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())

    assert summary["run_count"] == 20
    assert summary["pair_count"] == 10
    assert summary["reference_parity"] is True
    assert len(list(generated.glob("*.yaml"))) == 20
    assert all(row["output_dir"].startswith(str(results)) for row in manifest["runs"])
    assert all("/drafts/" not in row["config"] for row in manifest["runs"])


def test_h200_route_requires_explicit_opt_in(tmp_path):
    manifest_path = cube.materialize(
        generated_dir=tmp_path / "generated",
        results_root=tmp_path / "results",
    )
    manifest = json.loads(manifest_path.read_text())

    try:
        cube._route(manifest, "h200_opt_in", allow_h200=False)
    except ValueError as exc:
        assert "--allow-h200" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("H200 route was accepted without explicit opt-in")


def test_post_analysis_submission_is_cpu_only_and_depends_on_all_evaluations(
    tmp_path,
    monkeypatch,
):
    manifest_path = cube.materialize(
        generated_dir=tmp_path / "generated",
        results_root=tmp_path / "results",
    )
    reference_path = tmp_path / "reference.json"
    primary_path = tmp_path / "primary.json"
    reference_path.write_text("{}\n")
    primary_path.write_text("{}\n")
    snapshot_root = tmp_path / "source_snapshot"
    analysis_script = (
        snapshot_root / "src/dendritic_modeling/scripts/analysis/"
        "analyze_dendritic_rnn_extreme_cube.py"
    )
    analysis_script.parent.mkdir(parents=True)
    analysis_script.write_text("# frozen analyzer\n")
    source_snapshot = {
        "root": str(snapshot_root),
        "package_sha256": cube._package_tree_sha256(snapshot_root),
        "training_runtime_source_sha256": cube._tree_sha256(snapshot_root),
    }
    monkeypatch.setattr(
        cube, "_materialize_source_snapshot", lambda _output_root: source_snapshot
    )
    monkeypatch.setattr(
        cube,
        "_materialize_overlay_source_snapshot",
        lambda _output_root, **_kwargs: source_snapshot,
    )
    runtime_attestation_path = tmp_path / "runtime_attestation.json"
    runtime_attestation_path.write_text(
        json.dumps(
            {
                "training_runtime_source_sha256": "training-source-hash",
                "evaluation_runtime_source_sha256": cube._tree_sha256(snapshot_root),
            }
        )
        + "\n"
    )
    receipt_path = tmp_path / "evaluation_receipt.json"
    run_ids = [row["run_id"] for row in json.loads(manifest_path.read_text())["runs"]]
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "dendritic-rnn-standard-evaluation-submission-receipt-v1"
                ),
                "dry_run": False,
                "manifest": str(manifest_path),
                "manifest_sha256": cube._sha256(manifest_path),
                "reference_manifest_sha256": cube._sha256(reference_path),
                "training_runtime_source_sha256": "training-source-hash",
                "evaluation_runtime_source_sha256": cube._tree_sha256(snapshot_root),
                "source_snapshot": source_snapshot,
                "jobs": [
                    {"run_id": run_id, "job_id": str(10_000 + index)}
                    for index, run_id in enumerate(run_ids)
                ],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        cube.subprocess,
        "run",
        lambda *_args, **_kwargs: type("Result", (), {"stdout": "RUNNING|0:0\n"})(),
    )

    result = cube.submit_analysis(
        evaluation_receipt_path=receipt_path,
        primary_manifest_path=primary_path,
        extension_reference_manifest_path=reference_path,
        runtime_attestation_path=runtime_attestation_path,
        output_dir=None,
        dry_run=True,
    )

    assert result["resources"]["gpus"] == 0
    assert result["resources"]["cpus"] == 2
    assert result["resources"]["memory"] == "4G"
    assert result["resources"]["partition"] == "shared"
    assert len(result["dependency_job_ids"]) == 20
    assert any(item.startswith("--dependency=afterok:") for item in result["command"])


def test_analysis_retains_prespecified_route_sensitivity_as_descriptive_only():
    rows = []
    for replicate in range(analysis.N_REPLICATES):
        for tau, ff, rec in analysis.CELL_ORDER:
            source = (
                "extension" if (tau, ff, rec) in analysis.EXTENSION_CELLS else "primary"
            )
            partition = (
                analysis.PRIORITY_PARTITION
                if source == "extension" or replicate >= 8
                else analysis.BASE_PARTITION
            )
            rows.append(
                {
                    "replicate": f"r{replicate:03d}",
                    "tau": tau,
                    "ff": ff,
                    "rec": rec,
                    "source": source,
                    "training_partition": partition,
                    "accuracy": 1.0,
                }
            )
    route_audit = {
        "status": "PASSED",
        "observed_counts": {
            analysis.BASE_PARTITION: 48,
            analysis.PRIORITY_PARTITION: 32,
        },
    }
    result = analysis.analyze_collection(
        {
            "schema_version": analysis.COLLECTION_SCHEMA,
            "design": {"route_audit": route_audit},
            "runs": rows,
        },
        bootstrap_draws=100,
    )

    sensitivity = result["route_sensitivity"]
    assert sensitivity["role"] == "descriptive_noninferential_nuisance_check"
    assert sensitivity["can_rescue_or_modify_any_claim"] is False
    assert sensitivity["route_calendar_alias_remains"] is True
    assert sensitivity["route_audit"] == route_audit
    assert sensitivity["groups"]["mixed_route"]["n_units"] == 8
    assert sensitivity["groups"]["all_priority"]["n_units"] == 2
    assert all(
        row["mean"] == 0.0
        for group in sensitivity["groups"].values()
        for row in group["contrasts"].values()
    )
