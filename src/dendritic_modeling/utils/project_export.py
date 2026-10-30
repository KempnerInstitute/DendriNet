"""Export committed manuscript inputs or an explicitly scoped paper-code candidate."""

from __future__ import annotations

import sys

if __name__ == "__main__" and not __package__:
    sys.path.pop(0)

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

MANUSCRIPT_SUFFIXES = {
    ".tex",
    ".bib",
    ".bst",
    ".cls",
    ".sty",
    ".pdf",
    ".eps",
    ".png",
    ".jpg",
}
CODE_SUFFIXES = {
    ".py",
    ".sh",
    ".sbatch",
    ".slurm",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".tsv",
    ".csv",
    ".toml",
    ".ini",
    ".cpp",
    ".h",
}
CODE_FILENAMES = {"LICENSE", "Makefile"}
FORBIDDEN_CODE_PARTS = {
    "src",
    "dendritic_modeling",
    "runtime",
    "historical_runtimes",
    "source_snapshots",
    "frozen_sources",
    ".git",
    "outputs",
    "results",
    "logs",
}


def _git(root: Path, *args: str) -> bytes:
    if not (root / ".git").exists():
        raise ValueError(f"Expected a Git repository root: {root}")
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    )


def export_project(
    manifest: Path, commit: str, output: Path, *, paper_root: Path | None = None
) -> dict:
    """Write committed Git blobs; ``paper_root`` opts into a nested manifest."""
    requested = manifest.absolute()
    root = (requested.parent if paper_root is None else paper_root).resolve()
    # Resolve aliases to the repository root, but retain every component below
    # it: a working-tree symlink must not select a different committed manifest.
    for parent in reversed(requested.parents):
        if parent.resolve() == root:
            relative = requested.relative_to(parent)
            break
    else:
        raise ValueError("The manifest must be inside the paper root")
    if ".." in relative.parts or not requested.resolve().is_relative_to(root):
        raise ValueError("The manifest must be inside the paper root")
    output = output.resolve()
    if (
        Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
        != root
    ):
        raise ValueError("The declared paper root must be a Git repository root")
    if output.is_relative_to(root) or output.is_relative_to(
        Path(__file__).resolve().parents[3]
    ):
        raise ValueError(
            "Export archives belong outside paper and library repositories"
        )
    sha = _git(root, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    entries = {}
    for row in _git(root, "ls-tree", "-rz", sha).split(b"\0"):
        if row:
            meta, name = row.split(b"\t", 1)
            entries[name.decode()] = meta.split()[0].decode()
    manifest_name = relative.as_posix()
    if entries.get(manifest_name) not in {"100644", "100755"}:
        raise ValueError(f"Not a regular committed manifest: {manifest_name}")
    raw_manifest = _git(root, "show", f"{sha}:{manifest_name}")
    data = json.loads(raw_manifest)
    kind = data.get("kind")
    if data.get("schema_version") != 1 or kind not in {"manuscript", "paper-code"}:
        raise ValueError(
            "Expected a version 1 manuscript or paper-code export manifest"
        )
    if kind == "paper-code" and (
        data.get("publication_status") != "internal_candidate"
        or not isinstance(data.get("library_dependency"), dict)
    ):
        raise ValueError(
            "Paper code requires an internal-candidate status and explicit library dependency record"
        )
    paths = data.get("files")
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(p, str) for p in paths)
        or len(paths) != len(set(paths))
    ):
        raise ValueError("Expected a nonempty, unique file allow-list")
    for name in paths:
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or (
                path.suffix
                not in (MANUSCRIPT_SUFFIXES if kind == "manuscript" else CODE_SUFFIXES)
                and not (kind == "paper-code" and path.name in CODE_FILENAMES)
            )
            or (kind == "paper-code" and bool(set(path.parts) & FORBIDDEN_CODE_PARTS))
        ):
            raise ValueError(f"Not an allowed {kind} input: {name}")
        if entries.get(name) not in {"100644", "100755"}:
            raise ValueError(f"Not a regular committed file: {name}")
    receipt = {
        "schema_version": 1,
        "kind": kind,
        "paper_commit": sha,
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": [],
    }
    if kind == "paper-code":
        receipt.update(
            publication_status=data["publication_status"],
            reproduction_status=data.get("reproduction_status", "not validated"),
            library_dependency=data["library_dependency"],
            library_source_included=False,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves older exports. Fixed ZIP metadata makes the
    # same commit/manifest export byte-identical across runs and hosts.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(paths):
            payload = _git(root, "show", f"{sha}:{name}")
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = int(entries[name], 8) << 16
            archive.writestr(info, payload)
            receipt["files"].append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
        info = zipfile.ZipInfo("EXPORT_SOURCE.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--paper-root",
        type=Path,
        help="explicit Git repository root for a manifest stored in a subdirectory",
    )
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = export_project(
        args.manifest, args.commit, args.output, paper_root=args.paper_root
    )
    print(
        json.dumps(
            {
                "paper_commit": receipt["paper_commit"],
                "files": len(receipt["files"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
