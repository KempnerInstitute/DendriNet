"""Run records identify the library source and the numerical stack."""

from __future__ import annotations

import pytest
import torch

import dendritic_modeling
from dendritic_modeling.utils import runtime_environment, source_identity


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
