"""Evidence checks must detect mutations without claiming scientific validation."""

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/dendritic_modeling/utils/project_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("project_evidence_under_test", SOURCE)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


@pytest.fixture
def paper(tmp_path):
    root = tmp_path / "paper"
    root.mkdir()
    figure = root / "figure.csv"
    figure.write_text("seed,value\n1,0.5\n")
    data = {
        "schema_version": 1,
        "project": "fixture",
        "baseline_paper_commit": "a" * 40,
        "files": [
            {
                "id": "figure",
                "root": "paper",
                "path": "figure.csv",
                "sha256": evidence.sha256_file(figure),
            }
        ],
        "results": [
            {
                "id": "Figure 1",
                "artifact": "figure",
                "evidence": [],
                "workflow_kind": "standalone",
                "qualification": "pending",
            }
        ],
    }
    manifest = root / "current.json"
    manifest.write_text(json.dumps(data))
    return root, manifest, data


def test_integrity_does_not_imply_qualification(paper, tmp_path):
    root, manifest, _ = paper
    report = tmp_path / "report.json"
    assert (
        evidence.main(
            [
                "--manifest",
                str(manifest),
                "--root",
                f"paper={root}",
                "--report",
                str(report),
                "--require-qualified",
            ]
        )
        == 1
    )
    result = json.loads(report.read_text())
    assert result["integrity"] == "passed"
    assert result["unqualified_results"] == ["Figure 1"]


@pytest.mark.parametrize("change", ["modify", "remove"])
def test_existing_scientific_evidence_cannot_silently_change(paper, change):
    root, manifest, _ = paper
    if change == "modify":
        (root / "figure.csv").write_text("seed,value\n1,0.6\n")
    else:
        (root / "figure.csv").unlink()
    result = evidence.check_evidence(manifest, {"paper": root})
    assert result["integrity"] == "failed"


@pytest.mark.parametrize(
    "mutation", ["escape", "unbound", "duplicate", "unsupported_claim"]
)
def test_manifest_cannot_hide_unbound_or_escaping_evidence(paper, mutation):
    root, manifest, data = paper
    if mutation == "escape":
        data["files"][0]["path"] = "../outside.csv"
    elif mutation == "unbound":
        data["results"][0]["evidence"] = ["missing"]
    elif mutation == "duplicate":
        data["files"].append(data["files"][0])
    else:
        data["results"][0].update(
            qualification="numerically_validated", validation_evidence=["missing"]
        )
    manifest.write_text(json.dumps(data))
    with pytest.raises(evidence.EvidenceError):
        evidence.check_evidence(manifest, {"paper": root})


def test_external_symlink_is_not_an_implicit_artifact_root(paper, tmp_path):
    root, manifest, _ = paper
    external = tmp_path / "outside.csv"
    external.write_bytes((root / "figure.csv").read_bytes())
    (root / "figure.csv").unlink()
    (root / "figure.csv").symlink_to(external)
    with pytest.raises(evidence.EvidenceError):
        evidence.check_evidence(manifest, {"paper": root})


def test_report_cannot_overwrite_original_evidence(paper):
    root, manifest, _ = paper
    original = (root / "figure.csv").read_bytes()
    with pytest.raises(SystemExit):
        evidence.main(
            [
                "--manifest",
                str(manifest),
                "--root",
                f"paper={root}",
                "--report",
                str(root / "figure.csv"),
            ]
        )
    assert (root / "figure.csv").read_bytes() == original
