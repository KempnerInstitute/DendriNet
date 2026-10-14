"""
Shared pytest fixtures for dendritic modeling tests.

This module contains common fixtures used across multiple test files.
"""

import logging
import os
from pathlib import Path

import pytest

#: The repository these tests belong to. Tests marked ``requires_git_checkout``
#: read its Git checkout (commit, tracked files), which an archive lacks.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# The core suite tests built-in components; installed plugins stay unloaded
# unless DENDRITIC_MODELING_PLUGINS is set (for example to 1).
os.environ.setdefault("DENDRITIC_MODELING_PLUGINS", "0")


def pytest_collection_modifyitems(config, items):
    """Mark every test parametrized with a CUDA ``device`` as ``requires_gpu``."""
    for item in items:
        callspec = getattr(item, "callspec", None)
        device = None if callspec is None else callspec.params.get("device")
        if str(device or "").startswith("cuda"):
            item.add_marker(pytest.mark.requires_gpu)


def pytest_runtest_setup(item):
    """Skip ``requires_gpu`` tests without CUDA and ``requires_git_checkout``
    tests outside a Git checkout of the library."""
    if item.get_closest_marker("requires_gpu") is not None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA GPU")
    if item.get_closest_marker("requires_git_checkout") is not None:
        if not (REPOSITORY_ROOT / ".git").exists():
            pytest.skip("needs the library's Git checkout, not a source archive")


@pytest.fixture(autouse=True)
def _hermetic_site_settings(monkeypatch, tmp_path_factory):
    """Tests see no site scheduler policy and keep data under a temporary root."""
    monkeypatch.setenv("DENDRITIC_SCHEDULER_PROFILES", "none")
    monkeypatch.setenv(
        "DENDRITIC_DATA_DIR", str(tmp_path_factory.getbasetemp() / "data_root")
    )


@pytest.fixture(autouse=True)
def _restore_program_logging():
    """Remove the logging handlers a test added to the package and ``__main__``
    loggers. A program run in-process calls ``setup_logging``, whose console
    handler writes to the test's captured stdout; once that stream closes, a
    later test's log records fail with a logging error on stderr."""
    loggers = [logging.getLogger("dendritic_modeling"), logging.getLogger("__main__")]
    saved = [(logger, list(logger.handlers), logger.level) for logger in loggers]
    yield
    for logger, handlers, level in saved:
        for handler in list(logger.handlers):
            if handler not in handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(level)


@pytest.fixture
def simple_model():
    """Create a simple classifier model for testing."""
    from dendritic_modeling.models import Classifier
    from dendritic_modeling.networks import Identity
    from dendritic_modeling.networks.architectures.factory import get_architecture

    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [5], "output_dim": 2}
    )
    decoder = Identity(2)
    return Classifier(encoder, core, decoder)


@pytest.fixture
def sample_data():
    """Create sample training data for testing."""
    import torch
    from torch.utils.data import TensorDataset

    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    return TensorDataset(x, y)


@pytest.fixture
def sample_analyzer():
    """Create a sample analyzer configuration for testing."""
    return {
        "selection": {
            "levels": ["layer"],
            "targets": ["all_synapses"],
            "metrics": ["accuracy"],
        },
        "methods": {
            "ablation_methods": ["lesion"],
        },
    }
