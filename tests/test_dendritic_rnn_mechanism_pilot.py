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


def test_mechanism_pilot_materializes_standard_paired_quartets(tmp_path: Path) -> None:
    manifest_path = materialize(
        matrix_path=MATRIX, generated_dir=tmp_path / "generated"
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
        matrix_path=MATRIX, generated_dir=tmp_path / "generated"
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
