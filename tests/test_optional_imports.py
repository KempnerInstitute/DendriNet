"""Optional dependencies: absence is reported, other import errors propagate."""

from __future__ import annotations

import pytest

from dendritic_modeling.utils.optional_imports import is_missing_module


def _missing(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_is_missing_module_matches_the_module_and_its_parent_packages():
    target = "dendritic_modeling.kernels.triton_ell"
    assert is_missing_module(_missing(target), target)
    assert is_missing_module(_missing("dendritic_modeling.kernels"), target)
    assert not is_missing_module(_missing("triton"), target)
    assert not is_missing_module(_missing("dendritic_modeling.kernels.other"), target)
    assert not is_missing_module(ModuleNotFoundError("no name"), target)


def _kernel_backend(monkeypatch, missing_error):
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
        backend,
    )

    real_import = backend.importlib.import_module

    def fake_import(name, *args, **kwargs):
        if "triton" in name:
            raise missing_error(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(backend.importlib, "import_module", fake_import)
    monkeypatch.setattr(backend.torch.cuda, "is_available", lambda: True)
    return backend


def test_absent_triton_kernels_report_unavailable_and_auto_falls_back(monkeypatch):
    backend = _kernel_backend(monkeypatch, _missing)
    for name in ("triton_transposed", "triton_fused", "triton_ell"):
        assert not backend.indexed_projection_backend_available(name, device="cuda")
    resolve = backend.resolve_indexed_projection_backend
    assert resolve("auto", device="cuda") == "recompute"
    assert resolve("auto", device="cuda", in_features=200_000) == "recompute"
    assert resolve("eager", device="cuda") == "eager"
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve("triton_transposed", device="cuda")


def test_kernel_probe_propagates_unrelated_missing_modules(monkeypatch):
    backend = _kernel_backend(monkeypatch, lambda _name: _missing("triton"))
    with pytest.raises(ModuleNotFoundError):
        backend.indexed_projection_backend_available("triton_fused", device="cuda")
