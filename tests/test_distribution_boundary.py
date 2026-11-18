"""Release archives must retain tutorials and exclude research applications."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[1] / "setup" / "check_distribution.py"
_spec = importlib.util.spec_from_file_location("check_distribution", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _archives(tmp_path, *, extra=None, missing=False):
    source = tmp_path / "source"
    package = source / "src" / "dendritic_modeling" / "examples"
    package.mkdir(parents=True)
    (package / "demo.yaml").write_text("config_version: 4\n")
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "core.whl", "w") as wheel:
        if not missing:
            wheel.writestr(
                "dendritic_modeling/examples/demo.yaml", "config_version: 4\n"
            )
        if extra:
            wheel.writestr(extra, "unintended content")
    with tarfile.open(dist / "core.tar.gz", "w:gz") as sdist:
        content = b"config_version: 4\n"
        entry = tarfile.TarInfo("core/src/dendritic_modeling/examples/demo.yaml")
        entry.size = len(content)
        sdist.addfile(entry, io.BytesIO(content))
    return dist, source


def test_installed_tutorial_is_required(tmp_path):
    dist, source = _archives(tmp_path, missing=True)
    with pytest.raises(ValueError, match="missing from wheel"):
        _module.check(dist, source)


@pytest.mark.parametrize(
    "path", ["projects/study.py", "apps/editor.py", "llm/private.py"]
)
def test_nonlibrary_wheel_paths_are_rejected(tmp_path, path):
    dist, source = _archives(tmp_path, extra=path)
    with pytest.raises(ValueError, match="Unexpected wheel path"):
        _module.check(dist, source)


def test_valid_minimal_package(tmp_path):
    dist, source = _archives(tmp_path)
    assert _module.check(dist, source) == {"wheel_files": 1, "source_files": 1}
