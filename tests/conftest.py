"""
Shared pytest fixtures for dendritic modeling tests.

This module contains common fixtures used across multiple test files.
"""

import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture


@pytest.fixture
def simple_model():
    """Create a simple classifier model for testing."""
    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [5], "output_dim": 2}
    )
    decoder = Identity(2)
    return Classifier(encoder, core, decoder)


@pytest.fixture
def sample_data():
    """Create sample training data for testing."""
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    return TensorDataset(x, y)


@pytest.fixture
def sample_analyzer():
    """Create a sample analyzer configuration for testing."""
    return {
        "layer_ablation": True,
        "compartment_ablation": False,
        "branch_ablation": False,
    }
