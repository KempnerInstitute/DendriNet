"""Run records identify the library source and the numerical stack."""

from __future__ import annotations

from importlib import metadata

import pytest
import torch

import dendritic_modeling
from dendritic_modeling.utils import runtime_environment, source_identity


@pytest.mark.parametrize("installed_version", ["0.2.0", "0.4.0", None])
def test_records_distinguish_imported_source_from_distribution_metadata(
    tmp_path, monkeypatch, installed_version
):
    package = tmp_path / "dendritic_modeling"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text('__version__ = "0.4.0"\n')
    monkeypatch.setattr(dendritic_modeling, "__file__", str(init))
    monkeypatch.setattr(dendritic_modeling, "__version__", "0.4.0")
    original_version = metadata.version

    def distribution_version(name):
        if name != "dendritic_modeling":
            return original_version(name)
        if installed_version is None:
            raise metadata.PackageNotFoundError(name)
        return installed_version

    monkeypatch.setattr(metadata, "version", distribution_version)
    monkeypatch.setattr(runtime_environment, "version", distribution_version)
    library = runtime_environment.runtime_environment()["library"]
    identity = source_identity.discover_source_identity(tmp_path, include_package=True)

    assert library["imported_version"] == "0.4.0"
    assert library["distribution_version"] == installed_version
    assert library["package_path"] == str(package)
    assert identity["package"]["imported_version"] == "0.4.0"
    assert identity["package"]["version"] == installed_version
    assert identity["package"]["path"] == str(package)
    assert len(identity["package"]["tree_sha256"]) == 64


def test_an_installed_package_is_not_attributed_to_an_enclosing_repository(
    tmp_path, monkeypatch
):
    package = tmp_path / "installed" / "src" / "dendritic_modeling"
    package.mkdir(parents=True)
    (tmp_path / ".git").mkdir()  # an unrelated repository above the install
    monkeypatch.setattr(dendritic_modeling, "__file__", str(package / "__init__.py"))

    run = source_identity.subprocess.run

    def no_git(command, *args, **kwargs):
        if command[0] == "git":
            pytest.fail("must not invoke git for a package outside a checkout")
        return run(command, *args, **kwargs)

    monkeypatch.setattr(source_identity.subprocess, "run", no_git)
    assert runtime_environment.library_checkout() is None
    assert runtime_environment.source_revision() is None
    record = runtime_environment.runtime_environment()
    assert record["library"]["git"] == {"available": False}
    assert record["library"]["package_path"] == str(package)


@pytest.mark.requires_git_checkout
def test_the_record_names_the_checkout_and_the_numerical_stack():
    record = runtime_environment.runtime_environment()
    checkout = runtime_environment.library_checkout()
    assert checkout is not None and record["library"]["checkout"] == str(checkout)
    assert len(record["library"]["git"]["commit"]) == 40
    assert record["library"]["git"]["commit"] == runtime_environment.source_revision()
    assert record["torch"]["version"] == torch.__version__
    assert set(record["dependencies"]) == set(
        runtime_environment.NUMERICAL_DEPENDENCIES
    )
    assert len(record["devices"]) == torch.cuda.device_count()
    assert record["process_group"] == {"initialized": False, "world_size": 1}


def test_the_record_names_an_initialized_process_group(tmp_path):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        group = runtime_environment.runtime_environment()["process_group"]
    finally:
        dist.destroy_process_group()
    assert group == {"initialized": True, "backend": "gloo", "world_size": 1}


def test_the_record_includes_the_kernel_cache_limits(monkeypatch):
    monkeypatch.setenv("DENDRITIC_GX_SETUP_CACHE_MAX", "8")
    monkeypatch.delenv("DENDRITIC_GX_SETUP_CACHE_MAX_BYTES", raising=False)
    variables = runtime_environment.runtime_environment()["environment_variables"]
    assert variables["DENDRITIC_GX_SETUP_CACHE_MAX"] == "8"
    assert variables["DENDRITIC_GX_SETUP_CACHE_MAX_BYTES"] is None
