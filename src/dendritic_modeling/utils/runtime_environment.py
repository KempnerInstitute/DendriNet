"""Record the library source and numerical stack a run executes with.

``runtime_environment()`` is written to ``runtime_environment.json`` in every
training run directory. With ``resolved_seeds.json`` (seeds and the numerics
flags in effect) it identifies what produced a run: the imported package and
its checkout state, the loaded plugins and their checkouts, the Python and
PyTorch builds, the versions of the numerical dependencies, the devices, the
process group and the thread settings.
"""

from __future__ import annotations

import os
import platform
import site
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dendritic_modeling.utils.source_identity import discover_source_identity

RUNTIME_ENVIRONMENT_FILE = "runtime_environment.json"
NUMERICAL_DEPENDENCIES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "triton",
    "torchvision",
    "omegaconf",
)
_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTHONNOUSERSITE",
    # Data root used when data.base_dir is empty.
    "DENDRITIC_DATA_DIR",
    # Site scheduler policy for sweep and analysis submission.
    "DENDRITIC_SCHEDULER_PROFILES",
    # Memory limits of the indexed-synapse input-gradient setup cache.
    "DENDRITIC_GX_SETUP_CACHE_MAX",
    "DENDRITIC_GX_SETUP_CACHE_MAX_BYTES",
)


def dependency_version(name: str) -> str | None:
    """Installed version of distribution ``name``, or None when absent."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def package_checkout(package_file: str | Path) -> Path | None:
    """The Git checkout whose ``src/`` holds the package of ``package_file``.

    ``package_file`` is a file of a top-level package (its ``__init__.py`` or a
    module). Only ``<root>/src/<package>/...`` with ``<root>/.git`` counts, so an
    unrelated enclosing repository is never reported; None otherwise.
    """
    path = Path(package_file).resolve()
    for package in path.parents:
        # The top-level package: a package directory directly under ``src/``.
        if package.parent.name == "src" and (package / "__init__.py").exists():
            root = package.parent.parent
            return root if (root / ".git").exists() else None
        if not (package / "__init__.py").exists():
            return None
    return None


def library_checkout() -> Path | None:
    """The Git checkout whose ``src/`` the imported package comes from.

    Returns None for an installed package.
    """
    import dendritic_modeling

    return package_checkout(dendritic_modeling.__file__)


def source_revision() -> str | None:
    """Commit of the library checkout, or None for an installed package."""
    checkout = library_checkout()
    if checkout is None:
        return None
    return discover_source_identity(checkout)["git"]["commit"]


def _devices() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        return []
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    return devices


def _process_group() -> dict[str, Any]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return {"initialized": False, "world_size": 1}
    return {
        "initialized": True,
        "backend": str(dist.get_backend()),
        "world_size": dist.get_world_size(),
    }


def _plugin_provenance() -> list[dict[str, Any]]:
    from dendritic_modeling.plugins import loaded_plugins

    records = []
    for plugin in loaded_plugins():
        record = plugin.as_dict()
        checkout = (
            None if plugin.module_path is None else package_checkout(plugin.module_path)
        )
        record["checkout"] = None if checkout is None else str(checkout)
        record["git"] = (
            discover_source_identity(checkout)["git"]
            if checkout is not None
            else {"available": False}
        )
        records.append(record)
    return records


def runtime_environment() -> dict[str, Any]:
    """Describe the library source, plugins, Python, PyTorch, dependencies and devices."""
    import torch

    import dendritic_modeling

    checkout = library_checkout()
    git = (
        discover_source_identity(checkout)["git"]
        if checkout is not None
        else {"available": False}
    )
    cudnn = (
        torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    )
    return {
        "library": {
            "package_path": str(Path(dendritic_modeling.__file__).resolve().parent),
            "checkout": None if checkout is None else str(checkout),
            "git": git,
            "distribution_version": dependency_version("dendritic_modeling"),
        },
        "plugins": _plugin_provenance(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "user_site_enabled": bool(site.ENABLE_USER_SITE),
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": cudnn,
            "num_threads": torch.get_num_threads(),
            "default_dtype": str(torch.get_default_dtype()),
        },
        "dependencies": {
            name: dependency_version(name) for name in NUMERICAL_DEPENDENCIES
        },
        "devices": _devices(),
        "process_group": _process_group(),
        "environment_variables": {
            name: os.environ.get(name) for name in _ENVIRONMENT_VARIABLES
        },
        "platform": platform.platform(),
        "host": platform.node(),
    }


__all__ = [
    "NUMERICAL_DEPENDENCIES",
    "RUNTIME_ENVIRONMENT_FILE",
    "dependency_version",
    "library_checkout",
    "runtime_environment",
    "source_revision",
]
