"""Check the contents of a built core wheel and source distribution."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def check(dist: Path, source: Path) -> dict[str, int]:
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Use a clean directory with exactly one wheel and one sdist")
    package = source / "src" / "dendritic_modeling"
    expected = {
        p.relative_to(package.parent).as_posix()
        for p in package.rglob("*")
        if p.is_file() and p.suffix in {".py", ".yaml"}
    }
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_files = set(archive.namelist())
    missing = expected - wheel_files
    if missing:
        raise ValueError(f"Files missing from wheel: {sorted(missing)}")
    unexpected = {
        p for p in wheel_files if p.startswith("dendritic_modeling/")
    } - expected
    if unexpected:
        raise ValueError(f"Unexpected runtime files in wheel: {sorted(unexpected)}")
    for name in wheel_files:
        parts = PurePosixPath(name).parts
        if (
            ".." in parts
            or name.startswith("/")
            or not (
                parts[0] == "dendritic_modeling"
                or (
                    parts[0].startswith("dendritic_modeling-")
                    and parts[0].endswith(".dist-info")
                )
            )
        ):
            raise ValueError(f"Unexpected wheel path: {name}")
    with tarfile.open(sdists[0]) as archive:
        members = archive.getmembers()
    files = set()
    allowed = {"src", "tests", "configs", "setup", "docs"}
    root_files = {
        "LICENSE",
        "README.md",
        "README_PYPI.md",
        "pyproject.toml",
        "setup.cfg",
        "MANIFEST.in",
        "PKG-INFO",
        "pytest.ini",
    }
    for member in members:
        parts = PurePosixPath(member.name).parts
        if (
            ".." in parts
            or member.name.startswith("/")
            or member.issym()
            or member.islnk()
        ):
            raise ValueError(f"Unsafe source archive path: {member.name}")
        if len(parts) < 2:
            continue
        relative = PurePosixPath(*parts[1:])
        if relative.parts[0] not in allowed and str(relative) not in root_files:
            raise ValueError(f"Unexpected source archive path: {relative}")
        if member.isfile():
            files.add(str(relative))
    missing = {"src/" + name for name in expected} - files
    if missing:
        raise ValueError(f"Files missing from source distribution: {sorted(missing)}")
    unexpected = {
        p[4:] for p in files if p.startswith("src/dendritic_modeling/")
    } - expected
    if unexpected:
        raise ValueError(
            f"Unexpected runtime files in source distribution: {sorted(unexpected)}"
        )
    return {"wheel_files": len(wheel_files), "source_files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    print(check(args.dist, args.source))
