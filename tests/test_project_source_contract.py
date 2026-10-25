"""Source identity failures and an unchanged numerical entry-point execution."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = (
    Path(__file__).resolve().parents[1]
    / "src/dendritic_modeling/utils/project_source.py"
)
SPEC = importlib.util.spec_from_file_location("project_source_under_test", TOOL)
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        text=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
        stderr=subprocess.DEVNULL,
    ).strip()


def initialize(root):
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test User")
    git(root, "config", "user.email", "test@example.com")


def commit(root, *paths):
    git(root, "add", "--", *paths)
    git(root, "commit", "-qm", "Record test source", "--", *paths)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def project(tmp_path):
    library = tmp_path / "library"
    initialize(library)
    # Like real checkouts, ignore bytecode: running an entry imports the package
    # and writes __pycache__, which a later run must not see as uncommitted.
    (library / ".gitignore").write_text("__pycache__/\n")
    package = library / "src/dendritic_modeling"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "def response(x):\n    return (x * 0.125 + 2.0) / 3.0\n"
    )
    (library / "configs").mkdir()
    (library / "configs/default.json").write_text('{"gain": 0.125}\n')
    sha = commit(library, ".gitignore", "src", "configs")
    git(library, "tag", "paper/test/v1")
    git(library, "branch", "develop")
    paper = tmp_path / "paper"
    initialize(paper)
    (paper / ".gitignore").write_text("__pycache__/\n")
    (paper / "scripts").mkdir()
    (paper / "scripts/evaluate.py").write_text(
        "import json, random\n"
        "from dendritic_modeling import response\n"
        "random.seed(17)\n"
        "print(json.dumps([response(random.random()) for _ in range(20)]))\n"
    )
    contract = paper / "dendrinet.project.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project": "test-project",
                "library": {
                    "line": "core",
                    "commit": sha,
                    "ref": "refs/tags/paper/test/v1",
                    "maintained_ref": "refs/heads/develop",
                },
                "source_paths": ["scripts"],
                "entry_points": {"evaluate": "scripts/evaluate.py"},
            }
        )
    )
    commit(paper, ".gitignore", "scripts", contract.name)
    return library, paper, contract


def test_verifies_exact_library_and_paper_source(project):
    library, paper, contract = project
    receipt = source.verify_contract(contract, library)
    assert receipt["library_commit"] == git(library, "rev-parse", "HEAD")
    assert receipt["paper_commit"] == git(paper, "rev-parse", "HEAD")
    assert receipt["scientific_equivalence"] == "not assessed by source verification"


def test_explicit_root_preserves_default_receipt(project):
    library, paper, contract = project
    assert source.verify_contract(contract, library) == source.verify_contract(
        contract, library, paper_root=paper
    )


def test_nested_contract_checks_sources_and_its_own_cleanliness(project):
    library, paper, contract = project
    nested = paper / "reproducibility" / contract.name
    nested.parent.mkdir()
    nested.write_bytes(contract.read_bytes())
    commit(paper, str(nested.relative_to(paper)))
    old = source.verify_contract(contract, library)
    new = source.verify_contract(nested, library, paper_root=paper)
    assert {k: v for k, v in old.items() if k != "contract"} == {
        k: v for k, v in new.items() if k != "contract"
    }
    nested.write_text(nested.read_text() + "\n")
    with pytest.raises(source.SourceContractError, match="Uncommitted source"):
        source.verify_contract(nested, library, paper_root=paper)


def test_nested_contract_cli_preserves_numerical_execution(project, tmp_path):
    library, paper, contract = project
    nested = paper / "reproducibility" / contract.name
    nested.parent.mkdir()
    nested.write_bytes(contract.read_bytes())
    commit(paper, str(nested.relative_to(paper)))
    cmd = [
        sys.executable,
        str(TOOL),
        "run",
        "--library-root",
        str(library),
        "--entry",
        "evaluate",
    ]
    outputs = []
    for name, path, extra in (
        ("root", contract, []),
        ("nested", nested, ["--paper-root", str(paper)]),
    ):
        outputs.append(
            subprocess.check_output(
                [
                    *cmd,
                    "--contract",
                    str(path),
                    "--receipt",
                    str(tmp_path / (name + ".json")),
                    *extra,
                ],
                text=True,
            )
        )
    assert outputs[0] == outputs[1]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_output(
            [
                *cmd,
                "--contract",
                str(nested),
                "--paper-root",
                str(paper),
                "--receipt",
                str(paper / "forbidden.json"),
            ]
        )
    assert not (paper / "forbidden.json").exists()


def test_nested_contract_rejects_symlink_escape(project, tmp_path):
    library, paper, contract = project
    outside = tmp_path / "outside.json"
    outside.write_bytes(contract.read_bytes())
    alias = paper / "alias.json"
    alias.symlink_to(outside)
    with pytest.raises(source.SourceContractError, match="inside"):
        source.verify_contract(alias, library, paper_root=paper)


@pytest.mark.parametrize("location", ["library", "paper"])
def test_rejects_modified_source(project, location):
    library, paper, contract = project
    path = (
        library / "src/dendritic_modeling/__init__.py"
        if location == "library"
        else paper / "scripts/evaluate.py"
    )
    path.write_text(path.read_text() + "# uncommitted change\n")
    with pytest.raises(source.SourceContractError, match="Uncommitted source"):
        source.verify_contract(contract, library)


@pytest.mark.parametrize(
    "suffix", [".py", ".jsonl", ".sh", ".sbatch", ".slurm", ".ipynb"]
)
def test_rejects_ignored_source_shadow(project, suffix):
    library, _, contract = project
    ignored = "src/ignored" + suffix
    (library / ".gitignore").write_text(ignored + "\n")
    commit(library, ".gitignore")
    # Restore the test's expected exact pin to the new clean commit first.
    data = json.loads(contract.read_text())
    data["library"]["commit"] = git(library, "rev-parse", "HEAD")
    data["library"]["ref"] = "refs/tags/paper/test/v2"
    data["library"]["line"] = "historical"
    del data["library"]["maintained_ref"]
    git(library, "tag", "paper/test/v2")
    contract.write_text(json.dumps(data))
    commit(contract.parent, contract.name)
    (library / ignored).write_text("value = 1\n")
    with pytest.raises(source.SourceContractError, match="Ignored source"):
        source.verify_contract(contract, library)


def test_rejects_wrong_commit(project):
    library, _, contract = project
    (library / "src/dendritic_modeling/__init__.py").write_text("value = 2\n")
    commit(library, "src")
    with pytest.raises(source.SourceContractError, match="HEAD/tag mismatch"):
        source.verify_contract(contract, library)


def test_rejects_plain_copy_inside_unrelated_repository(project):
    _, paper, contract = project
    copied = paper / "copy"
    copied.mkdir()
    with pytest.raises(source.SourceContractError, match="No Git marker"):
        source.verify_contract(contract, copied)


def test_rejects_entry_outside_declared_source(project):
    library, paper, contract = project
    (paper / "outside.py").write_text("pass\n")
    data = json.loads(contract.read_text())
    data["entry_points"]["evaluate"] = "outside.py"
    contract.write_text(json.dumps(data))
    commit(paper, "outside.py", contract.name)
    with pytest.raises(
        source.SourceContractError, match="outside checked source_paths"
    ):
        source.verify_contract(contract, library)


@pytest.mark.parametrize("line", ["core", "llm"])
def test_a_maintained_line_must_name_its_maintained_ref(project, line):
    library, _, contract = project
    data = json.loads(contract.read_text())
    data["library"]["line"] = line
    del data["library"]["maintained_ref"]
    contract.write_text(json.dumps(data))
    commit(contract.parent, contract.name)
    with pytest.raises(
        source.SourceContractError, match=r"must name library\.maintained_ref"
    ):
        source.verify_contract(contract, library)


def test_rejects_a_maintained_ref_missing_from_the_checkout(project):
    library, _, contract = project
    data = json.loads(contract.read_text())
    data["library"].update(line="llm", maintained_ref="refs/heads/absent")
    contract.write_text(json.dumps(data))
    commit(contract.parent, contract.name)
    with pytest.raises(source.SourceContractError):
        source.verify_contract(contract, library)


def test_rejects_previously_imported_wrong_package(project, monkeypatch):
    library, _, contract = project
    receipt = source.verify_contract(contract, library)
    wrong = type("Module", (), {"__file__": "/wrong/package/__init__.py"})()
    monkeypatch.setitem(sys.modules, "dendritic_modeling", wrong)
    with pytest.raises(source.SourceContractError, match="already imported"):
        source.activate_source(receipt)


@pytest.mark.parametrize(
    "ref", ["refs/heads/public_line", "refs/remotes/origin/develop"]
)
def test_public_contract_uses_explicit_maintained_ref(project, ref):
    library, paper, contract = project
    sha = git(library, "rev-parse", "HEAD")
    git(library, "update-ref", ref, sha)
    data = json.loads(contract.read_text())
    data["library"]["maintained_ref"] = ref
    contract.write_text(json.dumps(data))
    commit(paper, contract.name)
    receipt = source.verify_contract(contract, library)
    assert receipt["maintained_ref"] == ref
    assert receipt["maintained_commit"] == sha
    assert receipt["library_commit"] == sha


def test_explicit_maintained_ref_still_requires_ancestry(project):
    library, paper, contract = project
    (library / "src/dendritic_modeling/__init__.py").write_text("value = 2\n")
    sha = commit(library, "src")
    git(library, "tag", "paper/test/v2")
    data = json.loads(contract.read_text())
    data["library"].update(
        commit=sha, ref="refs/tags/paper/test/v2", maintained_ref="refs/heads/develop"
    )
    contract.write_text(json.dumps(data))
    commit(paper, contract.name)
    with pytest.raises(source.SourceContractError):
        source.verify_contract(contract, library)


@pytest.mark.parametrize(
    "ref",
    [
        None,
        "HEAD",
        "refs/tags/paper/test/v1",
        "refs/heads/develop^{commit}",
        "refs/heads/../develop",
    ],
)
def test_rejects_non_branch_maintained_ref(project, ref):
    library, paper, contract = project
    data = json.loads(contract.read_text())
    data["library"]["maintained_ref"] = ref
    contract.write_text(json.dumps(data))
    commit(paper, contract.name)
    with pytest.raises(source.SourceContractError):
        source.verify_contract(contract, library)


def test_historical_contract_rejects_ignored_maintained_ref(project):
    _, _, contract = project
    data = json.loads(contract.read_text())
    data["library"].update(line="historical", maintained_ref="refs/heads/develop")
    contract.write_text(json.dumps(data))
    with pytest.raises(source.SourceContractError, match="Historical sources"):
        source.read_contract(contract)


def test_runner_preserves_output_and_protects_old_receipt(project, tmp_path):
    library, paper, contract = project
    (paper / "helpers").mkdir()
    (paper / "helpers/__init__.py").write_text("offset = 0.0\n")
    script = paper / "scripts/evaluate.py"
    script.write_text("from helpers import offset\n" + script.read_text())
    data = json.loads(contract.read_text())
    data["source_paths"].append("helpers")
    contract.write_text(json.dumps(data))
    commit(paper, "helpers", "scripts/evaluate.py", contract.name)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(library / "src"), str(paper))),
        PYTHONDONTWRITEBYTECODE="1",
    )
    reference = subprocess.check_output(
        [sys.executable, str(paper / "scripts/evaluate.py")], env=env, text=True
    )
    receipt = tmp_path / "run.json"
    args = [
        sys.executable,
        str(TOOL),
        "run",
        "--contract",
        str(contract),
        "--library-root",
        str(library),
        "--entry",
        "evaluate",
        "--receipt",
        str(receipt),
    ]
    # The runner must make paper packages available without inherited paths.
    actual = subprocess.check_output(args, env=dict(env, PYTHONPATH=""), text=True)
    assert actual == reference
    result = json.loads(receipt.read_text())
    assert result["status"] == "completed"
    assert result["imported_package_file"] == str(
        library / "src/dendritic_modeling/__init__.py"
    )
    before = receipt.read_bytes()
    repeated = subprocess.run(
        args, env=env, text=True, capture_output=True, check=False
    )
    assert repeated.returncode != 0
    assert receipt.read_bytes() == before


def test_failed_entry_is_recorded_as_failed(project, tmp_path):
    library, paper, contract = project
    (paper / "scripts/evaluate.py").write_text(
        "raise RuntimeError('intentional failure')\n"
    )
    commit(paper, "scripts/evaluate.py")
    receipt = tmp_path / "failure.json"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "run",
            "--contract",
            str(contract),
            "--library-root",
            str(library),
            "--entry",
            "evaluate",
            "--receipt",
            str(receipt),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert json.loads(receipt.read_text())["status"] == "failed"


# ---------------------------------------------------------------------------
# Releases built by squashing: the explicit, verified release bridge.
# ---------------------------------------------------------------------------


@pytest.fixture
def squash_release(project):
    """A core line (pin, then source) and a squash release with the source tree."""
    library, paper, contract = project
    pinned = git(library, "rev-parse", "HEAD")
    (library / "src/dendritic_modeling/extra.py").write_text("extra = 1\n")
    source_commit = commit(library, "src")
    tree = git(library, "rev-parse", f"{source_commit}^{{tree}}")
    release = git(library, "commit-tree", tree, "-m", "Release 1.0")
    follow_up = git(library, "commit-tree", tree, "-p", release, "-m", "Release 1.0.1")
    git(library, "update-ref", "refs/heads/release", follow_up)
    git(library, "checkout", "-q", "paper/test/v1")
    data = json.loads(contract.read_text())
    data["library"]["maintained_ref"] = "refs/heads/release"
    contract.write_text(json.dumps(data))
    commit(paper, contract.name)
    return {
        "library": library,
        "paper": paper,
        "contract": contract,
        "pinned": pinned,
        "source": source_commit,
        "tree": tree,
        "release": release,
        "maintained": follow_up,
    }


def _declare(contract, **library_fields):
    data = json.loads(contract.read_text())
    data["library"].update(library_fields)
    contract.write_text(json.dumps(data))
    commit(contract.parent, contract.name)


def test_a_squash_release_fails_the_ancestry_check_without_a_bridge(squash_release):
    with pytest.raises(source.SourceContractError):
        source.verify_contract(squash_release["contract"], squash_release["library"])


def test_a_verified_bridge_admits_the_squash_release(squash_release):
    release = squash_release
    _declare(
        release["contract"],
        release_bridge={
            "source_commit": release["source"],
            "release_commit": release["release"],
        },
    )
    receipt = source.verify_contract(release["contract"], release["library"])
    assert receipt["library_commit"] == release["pinned"]
    assert receipt["maintained_ref"] == "refs/heads/release"
    assert receipt["maintained_commit"] == release["maintained"]
    assert receipt["release_bridge"] == {
        "source_commit": release["source"],
        "release_commit": release["release"],
        "tree": release["tree"],
    }


def test_a_receipt_without_a_bridge_records_no_bridge(project):
    library, _, contract = project
    assert "release_bridge" not in source.verify_contract(contract, library)


def test_a_bridge_between_different_trees_is_rejected(squash_release):
    release = squash_release
    # The pin is the source here, but its tree differs from the release tree.
    _declare(
        release["contract"],
        release_bridge={
            "source_commit": release["pinned"],
            "release_commit": release["release"],
        },
    )
    with pytest.raises(source.SourceContractError, match="trees differ"):
        source.verify_contract(release["contract"], release["library"])


def test_a_bridge_whose_source_does_not_contain_the_pin_is_rejected(squash_release):
    release = squash_release
    library = release["library"]
    git(library, "checkout", "-q", "-b", "side", "paper/test/v1")
    (library / "src/dendritic_modeling/side.py").write_text("side = 1\n")
    side = commit(library, "src")
    git(library, "tag", "paper/test/side")
    _declare(
        release["contract"],
        commit=side,
        ref="refs/tags/paper/test/side",
        release_bridge={
            "source_commit": release["source"],
            "release_commit": release["release"],
        },
    )
    with pytest.raises(source.SourceContractError, match="does not contain it"):
        source.verify_contract(release["contract"], library)


def test_a_bridge_to_a_release_outside_the_maintained_ref_is_rejected(
    squash_release,
):
    release = squash_release
    other = git(
        release["library"], "commit-tree", release["tree"], "-m", "Unmaintained"
    )
    _declare(
        release["contract"],
        release_bridge={"source_commit": release["source"], "release_commit": other},
    )
    with pytest.raises(source.SourceContractError, match="not in the history"):
        source.verify_contract(release["contract"], release["library"])


def test_a_bridge_commit_absent_from_the_checkout_is_rejected(squash_release):
    release = squash_release
    _declare(
        release["contract"],
        release_bridge={
            "source_commit": release["source"],
            "release_commit": "1" * 40,
        },
    )
    with pytest.raises(source.SourceContractError, match="is not a commit"):
        source.verify_contract(release["contract"], release["library"])


@pytest.mark.parametrize(
    "bridge",
    [
        "not-a-mapping",
        {"source_commit": "a" * 40},
        {"source_commit": "a" * 40, "release_commit": "b" * 40, "note": "x"},
        {"source_commit": "a" * 39, "release_commit": "b" * 40},
        {"source_commit": "A" * 40, "release_commit": "b" * 40},
    ],
)
def test_malformed_bridges_are_rejected(project, bridge):
    _, _, contract = project
    data = json.loads(contract.read_text())
    data["library"]["release_bridge"] = bridge
    contract.write_text(json.dumps(data))
    with pytest.raises(source.SourceContractError, match="release_bridge"):
        source.read_contract(contract)


def test_a_historical_contract_cannot_declare_a_bridge(project):
    _, _, contract = project
    data = json.loads(contract.read_text())
    del data["library"]["maintained_ref"]
    data["library"].update(
        line="historical",
        release_bridge={"source_commit": "a" * 40, "release_commit": "b" * 40},
    )
    contract.write_text(json.dumps(data))
    with pytest.raises(source.SourceContractError, match="Historical sources"):
        source.read_contract(contract)
