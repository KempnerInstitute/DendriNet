"""Files promoted to the public index must match the staged pair exactly."""

import hashlib
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

release = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "setup" / "verify_release.py")
)
verify_files = release["verify_files"]


def test_staged_release_identity(tmp_path):
    names = ["dendrinet_studio-0.1.0-py3-none-any.whl", "dendrinet_studio-0.1.0.tar.gz"]
    urls = []
    for name in names:
        data = name.encode()
        (tmp_path / name).write_bytes(data)
        urls.append(
            {"filename": name, "digests": {"sha256": hashlib.sha256(data).hexdigest()}}
        )
    verify_files(tmp_path, "dendrinet-studio", "0.1.0", urls)
    for name, version in [
        ("dendritic_modeling", "0.1.0"),
        ("dendrinet-studio", "0.2.0"),
    ]:
        with pytest.raises(ValueError, match="name or version"):
            verify_files(tmp_path, name, version, urls)
    (tmp_path / names[0]).write_bytes(b"rebuilt")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_files(tmp_path, "dendrinet-studio", "0.1.0", urls)

    (tmp_path / names[0]).write_bytes(names[0].encode())
    with pytest.raises(ValueError, match="two active files"):
        verify_files(tmp_path, "dendrinet-studio", "0.1.0", [*urls, urls[0]])
    urls[1]["yanked"] = True
    with pytest.raises(ValueError, match="two active files"):
        verify_files(tmp_path, "dendrinet-studio", "0.1.0", urls)


def test_release_uses_the_tagged_commit_and_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.org", *args],
            check=True,
        )

    git("init", "-q")
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text('[project]\nname="core-example"\nversion="1.0"\n')
    git("add", "--", "pyproject.toml")
    git("commit", "-qm", "First release", "--", "pyproject.toml")
    git("tag", "-a", "v1.0", "-m", "First release")
    git("update-ref", "refs/remotes/origin/develop", "HEAD")
    release["verify_checkout"]("refs/tags/v1.0")

    metadata.write_text('[project]\nname="core-example"\nversion="2.0"\n')
    git("commit", "-qm", "Next release", "--", "pyproject.toml")
    git("update-ref", "refs/remotes/origin/develop", "HEAD")
    with pytest.raises(ValueError, match="at the selected release tag"):
        release["verify_checkout"]("refs/tags/v1.0")

    if sys.version_info >= (3, 11):  # Release metadata uses the standard TOML parser.
        metadata.write_text('[project]\nname="uncommitted"\nversion="3.0"\n')
        assert release["project"]("core") == ("core-example", "2.0", "v2.0")
