from __future__ import annotations

import json
from pathlib import Path

from dendritic_modeling.scripts.experiments.dendritic_rnn_mechanism_pilot import (
    EXPECTED_RECURRENT_CONTACTS,
    LEVEL_SIZES,
    materialize,
    submit,
    validate_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX = (
    REPO_ROOT / "configs/dendritic_rnn/mechanism_identification_pilot_v1/matrix.yaml"
)
MATRIX_V2 = (
    REPO_ROOT / "configs/dendritic_rnn/mechanism_identification_pilot_v2/matrix.yaml"
)


def test_mechanism_pilot_materializes_standard_paired_quartets(tmp_path: Path) -> None:
    manifest_path = materialize(
        matrix_path=MATRIX,
        generated_dir=tmp_path / "generated",
        results_root=tmp_path / "results",
    )
    result = validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert result["run_count"] == 12
    assert result["pairing_groups"] == 3
    assert result["contact_matching"] == "PASSED_EXACT"
    assert {row["pairing_group"] for row in manifest["runs"]} == {
        "r000",
        "r001",
        "r002",
    }
    assert all("/drafts/" not in row["config"] for row in manifest["runs"])


def test_recurrent_profiles_have_equal_active_contacts() -> None:
    proximal = (1, 1, 55)
    distal = (7, 1, 1)
    assert sum(a * b for a, b in zip(LEVEL_SIZES, proximal)) == (
        EXPECTED_RECURRENT_CONTACTS
    )
    assert sum(a * b for a, b in zip(LEVEL_SIZES, distal)) == (
        EXPECTED_RECURRENT_CONTACTS
    )


def test_preflight_dry_run_selects_one_complete_quartet(tmp_path: Path) -> None:
    manifest_path = materialize(
        matrix_path=MATRIX,
        generated_dir=tmp_path / "generated",
        results_root=tmp_path / "results",
    )
    receipt = submit(
        manifest_path=manifest_path,
        route_name="primary",
        allow_h200=False,
        dry_run=True,
        preflight=True,
    )
    assert receipt["preflight"] is True
    assert len(receipt["jobs"]) == 4
    assert {row["condition"] for row in receipt["jobs"]} == {
        "tau_distal_slow__rec_proximal_concentrated",
        "tau_distal_slow__rec_distal_concentrated",
        "tau_distal_fast__rec_proximal_concentrated",
        "tau_distal_fast__rec_distal_concentrated",
    }
    assert all("--cpus-per-task=3" in row["command"] for row in receipt["jobs"])
    assert all("--mem=6G" in row["command"] for row in receipt["jobs"])


def test_v2_dry_run_is_bound_to_external_snapshot_and_configs(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "protected-results"
    manifest_path = materialize(
        matrix_path=MATRIX_V2,
        generated_dir=results_root / "_protocol/generated",
        results_root=results_root,
    )
    validation = validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    receipt = submit(
        manifest_path=manifest_path,
        route_name="primary",
        allow_h200=False,
        dry_run=True,
        preflight=True,
    )

    snapshot_root = manifest["source_snapshot"]["root"]
    assert validation["source_snapshot"] == snapshot_root
    assert not Path(manifest_path).stat().st_mode & 0o222
    assert all(
        not Path(row["config"]).stat().st_mode & 0o222 for row in manifest["runs"]
    )
    assert all(f"--chdir={snapshot_root}" in row["command"] for row in receipt["jobs"])
    assert all("--cpus-per-task=3" in row["command"] for row in receipt["jobs"])
    assert all("--mem=4G" in row["command"] for row in receipt["jobs"])
    assert all(
        "--expected-config-sha256" in row["command"][-1]
        and "PYTHONDONTWRITEBYTECODE=1" in row["command"][-1]
        and str(results_root) in row["command"][-1]
        for row in receipt["jobs"]
    )
