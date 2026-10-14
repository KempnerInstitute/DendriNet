"""Publication boundaries: committed inputs only, no source leak or overwrite."""

import importlib.util
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "project_export_under_test",
    Path(__file__).resolve().parents[1]
    / "src/dendritic_modeling/utils/project_export.py",
)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


@pytest.fixture
def paper(tmp_path):
    root = tmp_path / "paper"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Test User")
    git("config", "user.email", "test@example.com")
    (root / "main.tex").write_text("Committed manuscript\n")
    (root / "private.py").write_text("private_value = 1\n")
    manifest = root / "manuscript_export.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "kind": "manuscript", "files": ["main.tex"]})
    )
    git("add", "--", "main.tex", "private.py", manifest.name)
    git(
        "commit",
        "-qm",
        "Record export fixture",
        "--",
        "main.tex",
        "private.py",
        manifest.name,
    )
    return root, manifest, git


def test_exact_committed_inputs_determinism_and_no_overwrite(paper, tmp_path):
    root, manifest, git = paper
    commit = git("rev-parse", "HEAD")
    (root / "main.tex").write_text("Uncommitted manuscript edits\n")
    outputs = [tmp_path / "first.zip", tmp_path / "second.zip"]
    for path in outputs:
        export.export_project(manifest, commit, path)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    with zipfile.ZipFile(outputs[0]) as archive:
        assert set(archive.namelist()) == {"main.tex", "EXPORT_SOURCE.json"}
        assert archive.read("main.tex") == b"Committed manuscript\n"
        assert json.loads(archive.read("EXPORT_SOURCE.json"))["paper_commit"] == commit
    with pytest.raises(FileExistsError):
        export.export_project(manifest, commit, outputs[0])


@pytest.mark.parametrize(
    "bad", ["private.py", "../outside.tex", "absent.tex", "/absolute.tex"]
)
def test_rejects_nonmanuscript_or_missing_inputs(paper, tmp_path, bad):
    _, manifest, git = paper
    manifest.write_text(
        json.dumps({"schema_version": 1, "kind": "manuscript", "files": [bad]})
    )
    git("commit", "-qm", "Record invalid export fixture", "--", manifest.name)
    with pytest.raises(ValueError):
        export.export_project(manifest, git("rev-parse", "HEAD"), tmp_path / "bad.zip")
    assert not (tmp_path / "bad.zip").exists()


def test_rejects_artifact_inside_paper(paper):
    root, manifest, git = paper
    with pytest.raises(ValueError, match="outside"):
        export.export_project(manifest, git("rev-parse", "HEAD"), root / "result.zip")


def test_paper_code_records_dependency_without_vendoring(paper, tmp_path):
    _, manifest, git = paper
    dependency = {"package": "dendritic_modeling", "validated_public_release": None}
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "paper-code",
                "publication_status": "internal_candidate",
                "library_dependency": dependency,
                "files": ["private.py"],
            }
        )
    )
    git("commit", "-qm", "Record paper code candidate", "--", manifest.name)
    output = tmp_path / "code.zip"
    receipt = export.export_project(manifest, git("rev-parse", "HEAD"), output)
    assert receipt["library_dependency"] == dependency
    assert receipt["library_source_included"] is False
    assert receipt["publication_status"] == "internal_candidate"
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"private.py", "EXPORT_SOURCE.json"}


def test_refuses_copied_library_even_when_allowlisted(paper, tmp_path):
    root, manifest, git = paper
    name = "code/runtime/src/dendritic_modeling/copied.py"
    copied = root / name
    copied.parent.mkdir(parents=True)
    copied.write_text("value = 1\n")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "paper-code",
                "publication_status": "internal_candidate",
                "library_dependency": {},
                "files": [name],
            }
        )
    )
    git("add", "--", name)
    git("commit", "-qm", "Record rejected copy fixture", "--", name, manifest.name)
    with pytest.raises(ValueError, match="Not an allowed"):
        export.export_project(manifest, git("rev-parse", "HEAD"), tmp_path / "bad.zip")
