"""Check a release tag and compare its distribution files with a package index."""

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

    metadata_path, prefix = (
        ("pyproject.toml", "v")
        if distribution == "core"
        else ("apps/studio/pyproject.toml", "studio-v")
    )
    source = subprocess.check_output(
        ["git", "show", f"HEAD:{metadata_path}"], text=True
    )
    metadata = tomllib.loads(source)["project"]
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
    if len(urls) != 2 or any(item.get("yanked") for item in urls):
        raise ValueError("Expected two active files on the package index")
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
            "The local files and package index differ in names or SHA-256 hashes"
        )
    print("Verified release files:", json.dumps(local, sort_keys=True))


def verify_checkout(tag: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tagged = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{tag}^{{commit}}"], text=True
    ).strip()
    if head != tagged:
        raise ValueError("The checkout must be at the selected release tag")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "origin/develop"], check=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution", choices=("core", "studio"))
    parser.add_argument("tag")
    parser.add_argument(
        "--staged", type=Path, help="Directory containing the wheel and source archive"
    )
    parser.add_argument(
        "--index",
        choices=("testpypi", "pypi"),
        default="testpypi",
        help="Package index to compare with --staged (default: testpypi)",
    )
    args = parser.parse_args()
    name, version, tag = project(args.distribution)
    if args.tag != "refs/tags/" + tag:
        parser.error(f"The release tag must be {tag}")
    verify_checkout(args.tag)
    if args.staged:
        host = "pypi.org" if args.index == "pypi" else "test.pypi.org"
        with urllib.request.urlopen(
            f"https://{host}/pypi/{name}/{version}/json", timeout=30
        ) as response:
            verify_files(args.staged, name, version, json.load(response)["urls"])
