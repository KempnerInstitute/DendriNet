"""Identify the library source a program runs: Git checkout or installed package.

Used by run records, sweep manifests and analysis collectors. In a Git
checkout the identity is the commit plus digests of the tracked changes and of
the untracked files under ``src/``; git failures raise rather than leave a run
unidentified. An installed package is identified by a digest of its files.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GIT_TIMEOUT_SECONDS = 60
UNTRACKED_SOURCE_ROOT = "src"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str, allow_failure: bool = False) -> bytes | None:
    """Run ``git args`` in ``repo_root`` and return its stdout.

    A timeout, a missing ``git`` or a failing command raises, so a slow
    network filesystem can never turn a guarded checkout into an unguarded
    one; ``allow_failure`` returns ``None`` for commands whose failure is an
    answer (``symbolic-ref`` on a detached HEAD).
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            # Read-only: never take the index lock other users of the checkout need.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS} s in "
            f"{repo_root}; the source identity cannot be established"
        ) from error
    except OSError as error:
        raise RuntimeError(f"git could not run in {repo_root}: {error}") from error
    if completed.returncode != 0:
        if allow_failure:
            return None
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo_root}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


def is_git_checkout(repo_root: str | Path) -> bool:
    """Whether ``repo_root`` itself is a Git checkout (main tree or worktree).

    Only the root is checked, so an installed package inside some unrelated
    enclosing repository is not mistaken for a checkout of this library.
    """
    return (Path(repo_root) / ".git").exists()


def untracked_source_digest(repo_root: Path, names: list[bytes]) -> str:
    """Digest of untracked source files: ``name NUL sha256 LF`` per file, sorted.

    The job-side source guard recomputes the same digest in shell.
    """
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(
            name + b"\0" + sha256_file(repo_root / name.decode()).encode() + b"\n"
        )
    return digest.hexdigest()


def package_tree_digest(package_dir: str | Path) -> str:
    """Digest of every file of an installed or checked-out package directory."""
    root = Path(package_dir).resolve()
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        digest.update(
            path.relative_to(root).as_posix().encode()
            + b"\0"
            + sha256_file(path).encode()
            + b"\n"
        )
    return digest.hexdigest()


def _package_identity() -> dict[str, Any]:
    from importlib import metadata

    import dendritic_modeling

    package_dir = Path(dendritic_modeling.__file__).resolve().parent
    try:
        version = metadata.version("dendritic_modeling")
    except metadata.PackageNotFoundError:
        version = None
    return {
        "path": str(package_dir),
        "imported_version": dendritic_modeling.__version__,
        # Keep the legacy distribution-metadata field for existing consumers.
        "version": version,
        "tree_sha256": package_tree_digest(package_dir),
    }


def discover_source_identity(
    repo_root: str | Path,
    *,
    source_files: Iterable[str | Path] = (),
    include_package: bool = False,
) -> dict[str, Any]:
    """Collect the source and Git identity of ``repo_root``.

    In a Git checkout this records the commit, branch, the digest of the
    tracked changes (``git diff --binary HEAD``) and the digest of untracked
    files under ``src/``; git failures raise. Outside a checkout (an installed
    package) ``git.available`` is false. ``include_package`` adds the digest
    of the imported package's files, which identifies an installed package.
    Its ``imported_version`` comes from the loaded module; the legacy
    ``version`` field remains the installed distribution's metadata version.
    They can differ when ``PYTHONPATH`` selects a different checkout.
    """
    root = Path(repo_root).resolve()
    files = []
    for source_file in source_files:
        path = Path(source_file).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"source file to hash does not exist: {path}")
        try:
            display_path = f"repo:{path.relative_to(root).as_posix()}"
        except ValueError:
            display_path = str(path)
        files.append({"path": display_path, "sha256": sha256_file(path)})

    git_identity: dict[str, Any] = {"available": is_git_checkout(root)}
    if git_identity["available"]:
        commit = _git(root, "rev-parse", "HEAD").decode().strip()
        branch = _git(root, "symbolic-ref", "--short", "-q", "HEAD", allow_failure=True)
        tracked_status = _git(root, "status", "--porcelain", "--untracked-files=no")
        tracked_diff = _git(root, "diff", "--binary", "HEAD")
        untracked = [
            name
            for name in _git(
                root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                UNTRACKED_SOURCE_ROOT,
            ).split(b"\0")
            if name
        ]
        git_identity.update(
            commit=commit,
            branch=(branch or b"").decode().strip() or None,
            tracked_worktree_dirty=bool(tracked_status.strip()),
            tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(),
            untracked_src_files=sorted(name.decode() for name in untracked),
            untracked_src_sha256=untracked_source_digest(root, untracked),
        )

    identity = {
        "repository_name": root.name,
        "git": git_identity,
        "python_version": platform.python_version(),
        "files": sorted(files, key=lambda item: item["path"]),
    }
    if include_package:
        identity["package"] = _package_identity()
    return identity


__all__ = [
    "GIT_TIMEOUT_SECONDS",
    "UNTRACKED_SOURCE_ROOT",
    "discover_source_identity",
    "is_git_checkout",
    "package_tree_digest",
    "sha256_file",
    "untracked_source_digest",
]
