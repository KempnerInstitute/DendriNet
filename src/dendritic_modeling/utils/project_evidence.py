"""Verify declared current-paper evidence without importing scientific code.

An integrity pass proves preservation of declared bytes only. It does not prove
that a figure's scientific workflow, source identity or numerical result is valid.
"""

from __future__ import annotations

import sys

if __name__ == "__main__" and not __package__:
    sys.path.pop(0)

import argparse
import hashlib
import json
import re
from pathlib import Path


class EvidenceError(ValueError):
    """An evidence manifest is invalid or does not identify preserved files."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_evidence(manifest: Path, roots: dict[str, Path]) -> dict:
    """Check explicit files and result mappings; never regenerate a baseline."""
    data = json.loads(manifest.read_text())
    if data.get("schema_version") != 1 or not data.get("project"):
        raise EvidenceError("Expected schema_version=1 and a project name")
    if not re.fullmatch(r"[0-9a-f]{40}", data.get("baseline_paper_commit", "")):
        raise EvidenceError("Record the original paper's full commit identity")
    files = data.get("files")
    results = data.get("results")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(results, list)
        or not results
    ):
        raise EvidenceError("Declare nonempty files and results")
    indexed = {}
    checks = []
    for row in files:
        name = row.get("id")
        if not isinstance(name, str) or not name or name in indexed:
            raise EvidenceError("File identifiers must be nonempty and unique")
        indexed[name] = row
        root_name = row.get("root")
        if root_name not in roots:
            raise EvidenceError(f"Missing explicit root: {root_name}")
        root = Path(roots[root_name]).resolve()
        relative = Path(row.get("path", ""))
        if not row.get("path") or relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError(f"Expected contained relative file path: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise EvidenceError(f"Evidence path escapes its root: {relative}")
        expected = row.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise EvidenceError(f"Invalid SHA-256 for {name}")
        actual = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "id": name,
                "path": str(path),
                "expected": expected,
                "actual": actual,
                "status": (
                    "match"
                    if actual == expected
                    else "missing" if actual is None else "changed"
                ),
            }
        )
    identifiers = set()
    incomplete = []
    for result in results:
        name = result.get("id")
        if not isinstance(name, str) or not name or name in identifiers:
            raise EvidenceError("Result identifiers must be nonempty and unique")
        identifiers.add(name)
        refs = [result.get("artifact"), *result.get("evidence", [])]
        if any(ref not in indexed for ref in refs):
            raise EvidenceError(f"Unbound file reference in result {name}")
        if not result.get("workflow_kind"):
            raise EvidenceError(f"Classify the computation for {name}")
        # Qualification is a recorded review decision with linked evidence.
        # This checker does not evaluate that evidence's scientific validity.
        if result.get("qualification") != "numerically_validated" or not result.get(
            "validation_evidence"
        ):
            incomplete.append(name)
        else:
            if any(ref not in indexed for ref in result["validation_evidence"]):
                raise EvidenceError(f"Unbound validation evidence for {name}")
    failures = [row for row in checks if row["status"] != "match"]
    return {
        "schema_version": 1,
        "project": data["project"],
        "manifest_sha256": sha256_file(manifest),
        "baseline_paper_commit": data["baseline_paper_commit"],
        "integrity": "failed" if failures else "passed",
        "checked_files": len(checks),
        "declared_results": len(results),
        "unqualified_results": incomplete,
        "unresolved_dependencies": data.get("missing_evidence", []),
        "checks": checks,
        "scientific_equivalence": "not assessed by file integrity verification",
        "coverage": "declared files/results only; not inferred dependency closure",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-qualified", action="store_true")
    args = parser.parse_args(argv)
    roots = {}
    for value in args.root:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in roots:
            parser.error("Use unique --root NAME=PATH arguments")
        roots[name] = Path(path).resolve()
    report = args.report.resolve()
    if any(report.is_relative_to(root) for root in roots.values()):
        parser.error("Write the report outside all evidence roots")
    if report.exists():
        parser.error("Refusing to overwrite an existing report")
    receipt = check_evidence(args.manifest, roots)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "checks"}, indent=2))
    return int(
        receipt["integrity"] != "passed"
        or (
            args.require_qualified
            and bool(
                receipt["unqualified_results"] or receipt["unresolved_dependencies"]
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
