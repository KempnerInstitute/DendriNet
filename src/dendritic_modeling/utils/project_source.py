"""Verify and run paper entry points against an explicit immutable library pin.

A paper repository declares the library it runs on in ``dendrinet.project.json``:
the exact commit and tag, the maintained branch whose history must contain that
commit, and optionally a release bridge for a release built by squashing.
``check`` verifies the declaration against a library checkout and prints a
receipt; ``run`` then executes a declared entry point with the verified package
and writes the receipt to a new file.

This module deliberately uses only the standard library. It can be executed
by filename before importing DendriNet, including when the selected historical
library predates this utility. It never edits a checkout or updates Git refs.
"""

from __future__ import annotations

# Direct execution otherwise exposes utils/math as the stdlib math module.
import sys

if __name__ == "__main__" and not __package__:
    sys.path.pop(0)

import argparse
import hashlib
import importlib
import json
import os
import re
import runpy
import subprocess
from collections import deque
from pathlib import Path
from typing import Any


class SourceContractError(RuntimeError):
    """The selected source does not satisfy a project's declared dependency."""


def _project_path(
    path: Path, paper_root: Path | None, *, allow_symlinks: bool = False
) -> tuple[Path, Path]:
    """Validate traversal and retain the requested path within a canonical root.

    Root inference must not normalize the request: doing so could hide an
    internal symlink. Walk from the filesystem anchor, allowing directory
    aliases before entering the repository. After first entry, never leave it;
    contracts reject all symlinks and exports permit only contained hops.
    The recorded Git path precedes any internal symlink expansion.
    """
    requested = path.absolute()
    kind = "manifest" if allow_symlinks else "contract"
    boundary_error = f"The {kind} must stay inside the paper root"
    try:
        root = (requested.parent if paper_root is None else paper_root).resolve()
    except RuntimeError as error:
        raise SourceContractError(f"Symlink loop in {kind} path: {path}") from error
    current = Path(requested.anchor)
    entered = False
    relative_parts: tuple[str, ...] = ()
    active: set[Path] = set()

    def walk(parts: tuple[str, ...]) -> None:
        nonlocal current, entered, relative_parts
        # Path tokens mark the end of a symlink expansion. Removing a link
        # then permits later non-cyclic reuse, e.g. link/../link/file.
        pending: deque[str | Path] = deque(parts)
        while True:
            if not entered and current == root:
                entered = True
                relative_parts = (
                    *(part for part in pending if isinstance(part, str)),
                    requested.name,
                )
            if not pending:
                return
            part = pending.popleft()
            if isinstance(part, Path):
                active.remove(part)
                continue
            if part == "..":
                if entered and current == root:
                    raise SourceContractError(boundary_error)
                current = current.parent
                continue
            current /= part
            if not current.is_symlink():
                continue
            link = current
            if entered and not allow_symlinks:
                raise SourceContractError(
                    f"Source symlink requires explicit resolution: {link}"
                )
            if link in active:
                raise SourceContractError(f"Symlink loop in {kind} path: {link}")
            active.add(link)
            target = link.readlink()
            if target.is_absolute():
                if entered:
                    if not target.is_relative_to(root):
                        raise SourceContractError(boundary_error)
                    current = root
                    target_parts = target.relative_to(root).parts
                else:
                    current = Path(target.anchor)
                    target_parts = target.parts[1:]
            else:
                current = link.parent
                target_parts = target.parts
            pending.extendleft(reversed((*target_parts, link)))

    walk(requested.parent.parts[1:])
    # Only directory aliases may lead into the root; an outside file symlink
    # pointing at a contract/manifest is not a repository alias.
    if not entered:
        raise SourceContractError(boundary_error)
    walk((requested.name,))
    # Normalize safe parent components only after their traversal was checked.
    # This is a Git path, so internal symlink targets never select other blobs.
    normalized: list[str] = []
    for part in relative_parts:
        if part == "..":
            if not normalized:
                raise SourceContractError(boundary_error)
            normalized.pop()
        else:
            normalized.append(part)
    return root, Path(*normalized)


def _git(root: Path, *args: str) -> str:
    if not (root / ".git").exists():
        raise SourceContractError(f"No Git marker at declared source root: {root}")
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        env=dict(
            os.environ,
            GIT_OPTIONAL_LOCKS="0",
            GIT_LITERAL_PATHSPECS="0",
            GIT_GLOB_PATHSPECS="0",
            GIT_NOGLOB_PATHSPECS="0",
            GIT_ICASE_PATHSPECS="0",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise SourceContractError(result.stderr.strip() or "Git source check failed")
    return result.stdout.rstrip("\n")


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    """Whether commit ``ancestor`` is ``descendant`` or one of its ancestors."""
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise SourceContractError(result.stderr.strip() or "Git ancestry check failed")
    return result.returncode == 0


def _inside(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise SourceContractError(f"Expected a contained relative path: {relative}")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise SourceContractError(f"Path escapes declared root: {relative}")
    component = root
    for part in value.parts:
        component /= part
        if component.is_symlink():
            raise SourceContractError(
                f"Source symlink requires explicit resolution: {component}"
            )
    return resolved


def _clean(root: Path, paths: list[str]) -> None:
    for path in paths:
        _inside(root, path)
    pathspecs = [f":(literal){path}" for path in paths]
    status = _git(
        root, "status", "--porcelain=v1", "--untracked-files=all", "--", *pathspecs
    )
    if status:
        raise SourceContractError(f"Uncommitted source in {root}:\n{status}")
    ignored = _git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        *pathspecs,
    )
    source_suffixes = {
        ".py",
        ".so",
        ".pyd",
        ".yaml",
        ".yml",
        ".json",
        ".jsonl",
        ".toml",
        ".sh",
        ".sbatch",
        ".slurm",
        ".ipynb",
    }
    shadows = [p for p in ignored.splitlines() if Path(p).suffix in source_suffixes]
    if shadows:
        raise SourceContractError(f"Ignored source could affect execution: {shadows}")
    for row in _git(root, "ls-files", "--stage", "--", *pathspecs).splitlines():
        if row.startswith("120000 "):
            path = row.split("\t", 1)[1]
            raise SourceContractError(
                f"Source symlink requires explicit resolution: {path}"
            )


_SHA = re.compile(r"[0-9a-f]{40}")
_BRIDGE_KEYS = ("source_commit", "release_commit")


def read_contract(path: Path) -> dict[str, Any]:
    """Load the small, versioned source contract stored with a paper's code.

    ``library.line`` is ``core``, ``llm`` or ``historical``. A ``core`` or
    ``llm`` contract names ``library.maintained_ref``, the full branch ref
    (``refs/heads/...`` or ``refs/remotes/...``) whose history must contain the
    pinned commit; a ``historical`` contract records an original source and
    names no maintained branch. ``library.release_bridge`` optionally names a
    ``source_commit`` and a ``release_commit`` for a release built by
    squashing (see :func:`verify_contract`). Raises
    :class:`SourceContractError` for a malformed contract.
    """
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("project"), str):
        raise SourceContractError("Expected schema_version=1 and a project name")
    library = data.get("library", {})
    if library.get("line") not in {"core", "llm", "historical"}:
        raise SourceContractError("Library line must be core, llm, or historical")
    maintained_ref = library.get("maintained_ref")
    if library["line"] == "historical":
        for key in ("maintained_ref", "release_bridge"):
            if key in library:
                raise SourceContractError(f"Historical sources cannot declare {key}")
    elif "maintained_ref" not in library:
        raise SourceContractError(
            f"A {library['line']!r} library contract must name library."
            "maintained_ref: the full branch ref (refs/heads/... or "
            "refs/remotes/...) whose history must contain the pinned commit"
        )
    if "maintained_ref" in library:
        if not isinstance(maintained_ref, str) or not maintained_ref.startswith(
            ("refs/heads/", "refs/remotes/")
        ):
            raise SourceContractError(
                "maintained_ref must name a full local or remote-tracking branch ref"
            )
    if "release_bridge" in library:
        bridge = library["release_bridge"]
        if not isinstance(bridge, dict) or sorted(bridge) != sorted(_BRIDGE_KEYS):
            raise SourceContractError(
                "library.release_bridge must hold exactly source_commit and "
                "release_commit"
            )
        for key in _BRIDGE_KEYS:
            if not isinstance(bridge[key], str) or not _SHA.fullmatch(bridge[key]):
                raise SourceContractError(
                    f"library.release_bridge.{key} must be an exact 40-character SHA"
                )
    if not _SHA.fullmatch(str(library.get("commit", ""))):
        raise SourceContractError("Library commit must be an exact 40-character SHA")
    if not isinstance(library.get("ref"), str) or not library["ref"].startswith(
        "refs/tags/"
    ):
        raise SourceContractError("Library ref must name a durable refs/tags/... pin")
    source_paths = data.get("source_paths")
    if (
        not isinstance(source_paths, list)
        or not source_paths
        or not all(isinstance(x, str) for x in source_paths)
    ):
        raise SourceContractError("Declare the paper's nonempty source_paths list")
    entries = data.get("entry_points")
    if not isinstance(entries, dict) or not entries:
        raise SourceContractError("Declare named Python entry_points")
    return data


def _verify_release_bridge(
    root: Path,
    bridge: dict[str, str],
    *,
    pinned: str,
    maintained_ref: str,
    maintained_commit: str,
) -> dict[str, str]:
    """Check that a squash release carries the pinned commit; return its record."""
    source = bridge["source_commit"]
    release = bridge["release_commit"]
    for key, sha in (("source_commit", source), ("release_commit", release)):
        try:
            found = _git(root, "rev-parse", "--verify", "--quiet", sha + "^{commit}")
        except SourceContractError as error:
            raise SourceContractError(
                f"library.release_bridge.{key} {sha} is not a commit of {root}; "
                "fetch it before verification"
            ) from error
        if found != sha:
            raise SourceContractError(
                f"library.release_bridge.{key} {sha} resolves to {found}"
            )
    if not _is_ancestor(root, pinned, source):
        raise SourceContractError(
            f"The pinned commit {pinned} is not library.release_bridge."
            f"source_commit {source} or one of its ancestors, so the release "
            "does not contain it"
        )
    if not _is_ancestor(root, release, maintained_commit):
        raise SourceContractError(
            f"library.release_bridge.release_commit {release} is not in the "
            f"history of {maintained_ref} ({maintained_commit})"
        )
    source_tree = _git(root, "rev-parse", "--verify", source + "^{tree}")
    release_tree = _git(root, "rev-parse", "--verify", release + "^{tree}")
    if source_tree != release_tree:
        raise SourceContractError(
            f"library.release_bridge trees differ: source_commit {source} has "
            f"tree {source_tree}, release_commit {release} has tree "
            f"{release_tree}, so the release is not a squash of the source"
        )
    return {"source_commit": source, "release_commit": release, "tree": source_tree}


def verify_contract(
    contract_path: Path, library_root: Path, *, paper_root: Path | None = None
) -> dict[str, Any]:
    """Check exact source identity and cleanliness without running paper code.

    The library checkout must be clean, its ``HEAD`` and ``library.ref`` must
    both be ``library.commit``, and for a ``core`` or ``llm`` contract that
    commit must be ``library.maintained_ref`` or one of its ancestors. A release
    built by squashing has none of the source commits as ancestors; for it the
    contract declares ``library.release_bridge`` instead, and verification
    requires that the pinned commit is ``source_commit`` or one of its
    ancestors, that ``release_commit`` is the maintained ref or one of its
    ancestors, and that both commits have the same tree. The paper checkout
    must be clean and hold every declared source path and entry point.

    ``paper_root`` permits a contract in a subdirectory of that explicit Git
    root. Omitting it retains the contract-directory default. All declared
    source paths remain relative to the paper root.

    Returns the receipt: contract, paper and library identities, the
    maintained ref and its commit, the verified bridge with its tree (only
    when one is declared), the package and configuration roots and the
    interpreter. Raises :class:`SourceContractError` when a check fails.
    """
    paper, relative = _project_path(contract_path, paper_root)
    if not contract_path.is_file():
        raise SourceContractError(f"Contract is not a regular file: {contract_path}")
    contract_relative = relative.as_posix()
    contract_path = _inside(paper, contract_relative)
    root = library_root.resolve()
    contract = read_contract(contract_path)
    for checkout in (paper, root):
        if Path(_git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout:
            raise SourceContractError(
                f"Declared path is not a repository root: {checkout}"
            )
    expected = contract["library"]["commit"]
    head = _git(root, "rev-parse", "HEAD")
    pinned = _git(
        root, "rev-parse", "--verify", contract["library"]["ref"] + "^{commit}"
    )
    if head != expected or pinned != expected:
        raise SourceContractError(
            f"Library HEAD/tag mismatch: expected {expected}; HEAD={head}; tag={pinned}"
        )
    line = contract["library"]["line"]
    maintained_ref = None
    maintained_commit = None
    bridge_record = None
    if line != "historical":
        maintained_ref = contract["library"]["maintained_ref"]
        _git(root, "check-ref-format", maintained_ref)
        maintained_commit = _git(
            root, "rev-parse", "--verify", maintained_ref + "^{commit}"
        )
        bridge = contract["library"].get("release_bridge")
        if bridge is None:
            _git(root, "merge-base", "--is-ancestor", expected, maintained_commit)
        else:
            bridge_record = _verify_release_bridge(
                root,
                bridge,
                pinned=expected,
                maintained_ref=maintained_ref,
                maintained_commit=maintained_commit,
            )
    package = root / "src" / "dendritic_modeling" / "__init__.py"
    if not package.is_file():
        raise SourceContractError(f"No maintained package at {package}")
    _clean(root, ["src", "configs", "pyproject.toml", "setup.py", "requirements.txt"])
    paper_paths = [*contract["source_paths"], contract_relative]
    _clean(paper, paper_paths)
    # Status cannot see into an embedded repository, so require the contract to
    # be a file the paper repository itself has committed.
    try:
        _git(
            paper, "ls-files", "--error-unmatch", "--", f":(literal){contract_relative}"
        )
    except SourceContractError as error:
        raise SourceContractError(
            f"The contract is not a committed file of {paper}: {contract_relative}"
        ) from error
    for relative in paper_paths:
        if not _inside(paper, relative).exists():
            raise SourceContractError(f"Declared paper source is missing: {relative}")
    for name, relative in contract["entry_points"].items():
        if not isinstance(name, str) or not isinstance(relative, str):
            raise SourceContractError(
                "Entry point names and script paths must be strings"
            )
        entry = _inside(paper, relative)
        if entry.suffix != ".py" or not entry.is_file():
            raise SourceContractError(f"Missing Python entry point: {relative}")
        if not any(
            entry.is_relative_to(_inside(paper, scope))
            for scope in contract["source_paths"]
        ):
            raise SourceContractError(
                f"Entry point is outside checked source_paths: {relative}"
            )
        _git(paper, "ls-files", "--error-unmatch", "--", f":(literal){relative}")
    return {
        "schema_version": 1,
        "project": contract["project"],
        "contract": str(contract_path),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "paper_commit": _git(paper, "rev-parse", "HEAD"),
        "library_commit": head,
        "library_ref": contract["library"]["ref"],
        "library_line": contract["library"]["line"],
        "maintained_ref": maintained_ref,
        "maintained_commit": maintained_commit,
        **({} if bridge_record is None else {"release_bridge": bridge_record}),
        "library_root": str(root),
        "configuration_root": str(root / "configs"),
        "package_file": str(package.resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "status": "source_verified",
        "scientific_equivalence": "not assessed by source verification",
    }


def activate_source(receipt: dict[str, Any]) -> None:
    """Select the verified package and reject an already imported wrong copy."""
    expected = Path(receipt["package_file"]).resolve()
    existing = sys.modules.get("dendritic_modeling")
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise SourceContractError(
            f"DendriNet was already imported from {existing.__file__}"
        )
    src = str(Path(receipt["library_root"]) / "src")
    sys.path.insert(0, src)
    importlib.invalidate_caches()
    module = importlib.import_module("dendritic_modeling")
    if Path(module.__file__).resolve() != expected:
        raise SourceContractError(f"Unexpected imported package: {module.__file__}")
    os.environ["DENDRINET_LIBRARY_ROOT"] = receipt["library_root"]
    os.environ["DENDRINET_CONFIG_ROOT"] = receipt["configuration_root"]
    # Child Python processes inherit the same first-choice package location.
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = src + (os.pathsep + previous if previous else "")
    receipt["imported_package_file"] = str(Path(module.__file__).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "action",
        choices=["check", "run"],
        help="check: verify and print the receipt; run: verify, then run --entry",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="the paper's source contract (root-level unless --paper-root is set)",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        help="explicit Git repository root for a contract stored in a subdirectory",
    )
    parser.add_argument(
        "--library-root",
        type=Path,
        default=os.environ.get("DENDRINET_LIBRARY_ROOT"),
        help="root of the library checkout to verify "
        "(default: $DENDRINET_LIBRARY_ROOT)",
    )
    parser.add_argument("--entry", help="run: name of a declared entry point")
    parser.add_argument(
        "--receipt",
        type=Path,
        help="run: new receipt file outside the paper and library checkouts",
    )
    raw = list(sys.argv[1:] if argv is None else argv)
    split = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:split])
    extra = raw[split + 1 :] if split < len(raw) else []
    if args.library_root is None:
        parser.error(
            "Set --library-root or DENDRINET_LIBRARY_ROOT; no editable-install fallback"
        )
    receipt = verify_contract(
        args.contract, args.library_root, paper_root=args.paper_root
    )
    if args.action == "check":
        print(json.dumps(receipt, indent=2))
        return 0
    contract = read_contract(args.contract)
    if args.entry not in contract["entry_points"] or args.receipt is None:
        parser.error(
            "run requires a declared --entry and a new external --receipt path"
        )
    paper = (
        args.contract.resolve().parent
        if args.paper_root is None
        else args.paper_root.resolve()
    )
    receipt_path = args.receipt.resolve()
    if receipt_path.is_relative_to(paper) or receipt_path.is_relative_to(
        Path(receipt["library_root"])
    ):
        raise SourceContractError(
            "Run receipts belong outside paper and library checkouts"
        )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an earlier run's evidence even on failure.
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    entry = _inside(paper, contract["entry_points"][args.entry])
    receipt.update(
        entry_point=args.entry, script=str(entry), arguments=extra, status="started"
    )
    try:
        activate_source(receipt)
        os.chdir(paper)
        sys.path.insert(1, str(entry.parent))
        sys.path.insert(2, str(paper))
        sys.argv = [str(entry), *extra]
        runpy.run_path(str(entry), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        receipt.update(status="completed" if code == 0 else "failed", exit_code=code)
        return code
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    else:
        receipt.update(status="completed", exit_code=0)
        return 0
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
