"""Files promoted to the public index must match the staged pair exactly."""

import hashlib
import runpy
from pathlib import Path

import pytest

verify_files = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "setup" / "verify_release.py")
)["verify_files"]


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
