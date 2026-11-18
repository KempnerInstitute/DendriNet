"""Check a release tag and the exact distributions being promoted from TestPyPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

from packaging.version import Version


def project(distribution: str) -> tuple[str, str, str]:
    import tomllib  # The release workflow runs on Python 3.11.

    root = Path(__file__).resolve().parents[1]
    directory, prefix = (
        (root, "v") if distribution == "core" else (root / "apps/studio", "studio-v")
    )
    metadata = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
    version = metadata["version"]
    return metadata["name"], version, prefix + version


def verify_files(directory: Path, name: str, version: str, urls: list[dict]) -> None:
    from packaging.utils import (
        canonicalize_name,
        parse_sdist_filename,
        parse_wheel_filename,
    )

    files = sorted(directory.iterdir())
    if (
        len(files) != 2
        or len(list(directory.glob("*.whl"))) != 1
        or len(list(directory.glob("*.tar.gz"))) != 1
    ):
        raise ValueError("Expected one wheel and one source distribution")
    published = {item["filename"]: item["digests"]["sha256"] for item in urls}
    for path in files:
        parse = parse_wheel_filename if path.suffix == ".whl" else parse_sdist_filename
        artifact_name, artifact_version, *_ = parse(path.name)
        if artifact_name != canonicalize_name(name) or artifact_version != Version(
            version
        ):
            raise ValueError(f"Unexpected distribution name or version: {path.name}")
    local = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if local != published:
        raise ValueError(
            "The local files and TestPyPI release differ in names or SHA-256 hashes"
        )
    print("Verified TestPyPI files:", json.dumps(local, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution", choices=("core", "studio"))
    parser.add_argument("tag")
    parser.add_argument(
        "--staged", type=Path, help="Check these files against the version on TestPyPI"
    )
    args = parser.parse_args()
    name, version, tag = project(args.distribution)
    if args.tag != "refs/tags/" + tag:
        parser.error(f"The release tag must be {tag}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "origin/develop"], check=True
    )
    if args.staged:
        with urllib.request.urlopen(
            f"https://test.pypi.org/pypi/{name}/{version}/json", timeout=30
        ) as response:
            verify_files(args.staged, name, version, json.load(response)["urls"])
