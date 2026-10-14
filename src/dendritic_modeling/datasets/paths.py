"""Shared dataset path and split-seed helpers."""

import os
from pathlib import Path

import torch

#: Environment variable naming the shared data root used when
#: ``data.base_dir`` is empty.
DATA_DIR_ENV = "DENDRITIC_DATA_DIR"


def resolve_data_root(base_dir: str | None = None) -> str:
    """The data root: ``base_dir``, else ``$DENDRITIC_DATA_DIR``, else the cache.

    The cache is ``$XDG_CACHE_HOME/dendritic_modeling`` (default
    ``~/.cache/dendritic_modeling``), so runs share downloads instead of
    keeping a copy in each results directory.
    """
    if base_dir and str(base_dir).strip():
        return str(base_dir)
    configured = os.environ.get(DATA_DIR_ENV, "").strip()
    if configured:
        return os.path.expanduser(configured)
    cache_home = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return str(Path(cache_home) / "dendritic_modeling")


def get_data_directory(data_type, task_data_path=None):
    """
    Get or create a data directory for the specified data type.

    Args:
        data_type (str): Type of data (e.g., 'mnist', 'cifar')
        task_data_path (str, optional): Data root (``data.base_dir``); when
            empty, :func:`resolve_data_root` chooses it.

    Returns:
        str: ``<root>/datasets/<data_type>``
    """
    data_path = os.path.join(resolve_data_root(task_data_path), "datasets", data_type)
    os.makedirs(data_path, exist_ok=True)
    return data_path


def split_generator(seed: int | None = 0) -> torch.Generator:
    """Return a CPU generator for reproducible train/validation splits."""
    return torch.Generator().manual_seed(int(seed or 0))


__all__ = [
    "DATA_DIR_ENV",
    "get_data_directory",
    "resolve_data_root",
    "split_generator",
]
