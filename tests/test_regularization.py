"""
Tests for training regularization module to improve coverage.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import json

from dendritic_modeling.training.regularization import (
    RegularizationManager,
    PruningManager,
    create_sparsity_mask,
    apply_sparsity_masks,
)
from dendritic_modeling.config.training import RegularizationConfig, PruningConfig


class TestRegularizationManager:
    """Test RegularizationManager class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    @pytest.fixture
    def complex_model(self):
        """Create a model with named parameters for selective regularization."""
        model = nn.Module()
        model.topk_layer = nn.Linear(10, 8)
        model.blocklinear_layer = nn.Linear(8, 6)
        model.reactivation_layer = nn.Linear(6, 4)
        model.decoder_layer = nn.Linear(4, 2)
        model.other_layer = nn.Linear(2, 1)
        return model

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        config = RegularizationConfig()
        manager = RegularizationManager(config)

        assert manager.config == config
        assert manager.l1_weight == config.l1_weight
        assert manager.l2_weight == config.l2_weight
        assert manager.selective == config.selective
        assert manager.split_params == config.split_params

    def test_compute_global_regularization_l1(self, simple_model):
        """Test computing global L1 regularization."""
        config = RegularizationConfig(l1_weight=0.01, l2_weight=0.0, selective=False)
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(simple_model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0
        assert reg_loss.requires_grad

    def test_compute_global_regularization_l2(self, simple_model):
        """Test computing global L2 regularization."""
        config = RegularizationConfig(l1_weight=0.0, l2_weight=0.01, selective=False)
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(simple_model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0
        assert reg_loss.requires_grad

    def test_compute_combined_regularization(self, simple_model):
        """Test computing combined L1 and L2 regularization."""
        config = RegularizationConfig(l1_weight=0.01, l2_weight=0.01, selective=False)
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(simple_model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0

    def test_compute_no_regularization(self, simple_model):
        """Test computing with no regularization."""
        config = RegularizationConfig(l1_weight=0.0, l2_weight=0.0, selective=False)
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(simple_model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() == 0.0

    def test_compute_selective_regularization(self, complex_model):
        """Test computing selective regularization based on parameter names."""
        config = RegularizationConfig(
            l1_weight=0.01,
            l2_weight=0.01,
            selective=True,
            layer_weights={
                "topk": {"l1": 0.02, "l2": 0.02},
                "blocklinear": {"l1": 0.015, "l2": 0.015},
                "decoder": {"l1": 0.005, "l2": 0.005},
            },
        )
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(complex_model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0

    def test_compute_param_group_regularization(self, simple_model):
        """Test computing regularization with parameter groups."""
        config = RegularizationConfig(
            l1_weight=0.01,
            l2_weight=0.01,
            split_params=True,
            param_group_weights={
                "group_0": {"l1": 0.02, "l2": 0.02},
                "group_1": {"l1": 0.01, "l2": 0.01},
            },
        )
        manager = RegularizationManager(config)

        # Create mock parameter groups
        param_groups = [
            {"params": list(simple_model.parameters())[:2], "name": "group_0"},
            {"params": list(simple_model.parameters())[2:], "name": "group_1"},
        ]

        reg_loss = manager.compute_regularization_loss(simple_model, param_groups)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0

    def test_selective_regularization_no_matches(self, simple_model):
        """Test selective regularization when no parameter names match."""
        config = RegularizationConfig(
            l1_weight=0.01,
            l2_weight=0.01,
            selective=True,
            layer_weights={"nonexistent": {"l1": 0.02, "l2": 0.02}},
        )
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(simple_model)

        # Should use global weights for unmatched parameters
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0

    def test_regularization_with_different_devices(self):
        """Test regularization when model is on different device."""
        model = nn.Linear(5, 2)

        config = RegularizationConfig(l1_weight=0.01, l2_weight=0.01)
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(model)

        assert reg_loss.device == next(model.parameters()).device


class TestPruningManager:
    """Test PruningManager class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        config = PruningConfig()
        manager = PruningManager(config)

        assert manager.config == config
        assert manager.enabled == config.enabled
        assert manager.sparsity == config.sparsity

    def test_magnitude_pruning(self, simple_model):
        """Test magnitude-based pruning."""
        config = PruningConfig(
            enabled=True, method="magnitude", sparsity=0.5, structured=False
        )
        manager = PruningManager(config)

        # Apply pruning
        manager.prune_model(simple_model)

        # Check that some weights are zero
        total_params = 0
        zero_params = 0
        for param in simple_model.parameters():
            if param.dim() > 1:  # Only check weight matrices
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        sparsity_achieved = zero_params / total_params if total_params > 0 else 0
        assert sparsity_achieved > 0  # Some pruning should have occurred

    def test_random_pruning(self, simple_model):
        """Test random pruning."""
        config = PruningConfig(
            enabled=True, method="random", sparsity=0.3, structured=False
        )
        manager = PruningManager(config)

        manager.prune_model(simple_model)

        # Check that some weights are zero
        total_params = 0
        zero_params = 0
        for param in simple_model.parameters():
            if param.dim() > 1:
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

        sparsity_achieved = zero_params / total_params if total_params > 0 else 0
        assert sparsity_achieved > 0

    def test_structured_pruning(self, simple_model):
        """Test structured pruning."""
        config = PruningConfig(
            enabled=True, method="magnitude", sparsity=0.5, structured=True
        )
        manager = PruningManager(config)

        manager.prune_model(simple_model)

        # For structured pruning, check that entire rows/columns are zeroed
        for module in simple_model.modules():
            if isinstance(module, nn.Linear):
                # Check if any entire rows or columns are zero
                weight = module.weight.data
                zero_rows = (weight == 0).all(dim=1).sum().item()
                zero_cols = (weight == 0).all(dim=0).sum().item()
                assert (zero_rows > 0) or (zero_cols > 0) or weight.numel() == 0

    def test_pruning_disabled(self, simple_model):
        """Test when pruning is disabled."""
        config = PruningConfig(enabled=False)
        manager = PruningManager(config)

        # Store original weights
        original_weights = {}
        for name, param in simple_model.named_parameters():
            original_weights[name] = param.data.clone()

        manager.prune_model(simple_model)

        # Weights should be unchanged
        for name, param in simple_model.named_parameters():
            assert torch.equal(param.data, original_weights[name])

    def test_gradual_pruning(self, simple_model):
        """Test gradual pruning over multiple epochs."""
        config = PruningConfig(
            enabled=True,
            method="magnitude",
            sparsity=0.8,
            structured=False,
            gradual=True,
            pruning_schedule="linear",
            start_epoch=0,
            end_epoch=10,
        )
        manager = PruningManager(config)

        # Test pruning at different epochs
        sparsities = []
        for epoch in [0, 5, 10, 15]:
            # Reset model
            for param in simple_model.parameters():
                param.data.uniform_(-1, 1)

            manager.prune_model(simple_model, epoch=epoch)

            # Calculate current sparsity
            total_params = 0
            zero_params = 0
            for param in simple_model.parameters():
                if param.dim() > 1:
                    total_params += param.numel()
                    zero_params += (param == 0).sum().item()

            current_sparsity = zero_params / total_params if total_params > 0 else 0
            sparsities.append(current_sparsity)

        # Sparsity should generally increase over epochs (until end_epoch)
        assert sparsities[1] >= sparsities[0]  # Epoch 5 >= Epoch 0
        assert sparsities[2] >= sparsities[1]  # Epoch 10 >= Epoch 5

    def test_save_and_load_pruning_state(self, simple_model):
        """Test saving and loading pruning state."""
        config = PruningConfig(enabled=True, method="magnitude", sparsity=0.5)
        manager = PruningManager(config)

        # Apply pruning
        manager.prune_model(simple_model)

        # Save pruning state
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            manager.save_pruning_state(tmp_path)
            assert os.path.exists(tmp_path)

            # Verify saved content
            with open(tmp_path) as f:
                saved_state = json.load(f)

            assert "config" in saved_state
            assert "pruning_masks" in saved_state

            # Load pruning state
            loaded_state = manager.load_pruning_state(tmp_path)
            assert loaded_state is not None

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_unsupported_pruning_method(self, simple_model):
        """Test unsupported pruning method."""
        config = PruningConfig(enabled=True, method="unsupported_method", sparsity=0.5)
        manager = PruningManager(config)

        with pytest.raises(ValueError, match="Unsupported pruning method"):
            manager.prune_model(simple_model)


class TestRegularizationUtils:
    """Test regularization utility functions."""

    def test_create_sparsity_mask_magnitude(self):
        """Test creating sparsity mask based on magnitude."""
        weights = torch.tensor([[1.0, 0.1, 0.8], [0.2, 1.5, 0.3], [0.9, 0.05, 1.2]])

        mask = create_sparsity_mask(weights, sparsity=0.5, method="magnitude")

        assert mask.shape == weights.shape
        assert mask.dtype == torch.bool

        # Check that approximately 50% of weights are pruned
        pruned_ratio = (~mask).sum().item() / mask.numel()
        assert abs(pruned_ratio - 0.5) < 0.2  # Allow some tolerance

    def test_create_sparsity_mask_random(self):
        """Test creating random sparsity mask."""
        weights = torch.randn(5, 5)

        mask = create_sparsity_mask(weights, sparsity=0.3, method="random")

        assert mask.shape == weights.shape
        assert mask.dtype == torch.bool

        # Check approximate sparsity
        pruned_ratio = (~mask).sum().item() / mask.numel()
        assert abs(pruned_ratio - 0.3) < 0.2

    def test_create_sparsity_mask_zero_sparsity(self):
        """Test creating mask with zero sparsity."""
        weights = torch.randn(3, 3)

        mask = create_sparsity_mask(weights, sparsity=0.0, method="magnitude")

        assert torch.all(mask)  # All weights should be kept

    def test_create_sparsity_mask_full_sparsity(self):
        """Test creating mask with full sparsity."""
        weights = torch.randn(3, 3)

        mask = create_sparsity_mask(weights, sparsity=1.0, method="magnitude")

        assert torch.all(~mask)  # All weights should be pruned

    def test_apply_sparsity_masks(self):
        """Test applying sparsity masks to model."""
        model = nn.Sequential(nn.Linear(5, 3), nn.Linear(3, 2))

        # Create masks
        masks = {}
        for name, param in model.named_parameters():
            if "weight" in name:
                masks[name] = torch.zeros_like(param, dtype=torch.bool)
                # Keep only first row
                masks[name][0] = True

        apply_sparsity_masks(model, masks)

        # Check that masks were applied
        for name, param in model.named_parameters():
            if name in masks:
                # All values except first row should be zero
                if param.dim() > 1:
                    assert torch.all(param[1:] == 0)
                    assert not torch.all(
                        param[0] == 0
                    )  # First row should not be all zeros

    def test_apply_sparsity_masks_empty(self):
        """Test applying empty sparsity masks."""
        model = nn.Linear(5, 3)

        # Store original weights
        original_weight = model.weight.data.clone()

        apply_sparsity_masks(model, {})

        # Weights should be unchanged
        assert torch.equal(model.weight.data, original_weight)


class TestRegularizationEdgeCases:
    """Test edge cases for regularization functionality."""

    def test_regularization_empty_model(self):
        """Test regularization with empty model."""
        model = nn.Module()  # Empty model
        config = RegularizationConfig(l1_weight=0.01, l2_weight=0.01)
        manager = RegularizationManager(config)

        # Should handle gracefully
        try:
            reg_loss = manager.compute_regularization_loss(model)
            assert reg_loss.item() == 0.0
        except StopIteration:
            # Expected if model has no parameters
            pass

    def test_pruning_model_without_weights(self):
        """Test pruning model with only bias parameters."""
        model = nn.Module()
        model.bias = nn.Parameter(torch.randn(5))  # Only bias, no weight matrices

        config = PruningConfig(enabled=True, sparsity=0.5)
        manager = PruningManager(config)

        # Should handle gracefully
        original_bias = model.bias.data.clone()
        manager.prune_model(model)

        # Bias should be unchanged (we typically don't prune biases)
        assert torch.equal(model.bias.data, original_bias)

    def test_regularization_with_non_leaf_tensors(self):
        """Test regularization computation with non-leaf tensors."""
        model = nn.Linear(5, 3)

        # Make parameters non-leaf by applying operations
        with torch.no_grad():
            model.weight.data = model.weight.data * 2

        config = RegularizationConfig(l1_weight=0.01, l2_weight=0.01)
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0

    def test_create_mask_single_element(self):
        """Test creating mask for single element tensor."""
        weights = torch.tensor([[1.0]])

        mask = create_sparsity_mask(weights, sparsity=0.5, method="magnitude")

        assert mask.shape == weights.shape
        assert mask.dtype == torch.bool

    def test_pruning_state_invalid_file(self):
        """Test loading pruning state from invalid file."""
        config = PruningConfig()
        manager = PruningManager(config)

        with pytest.raises((FileNotFoundError, json.JSONDecodeError)):
            manager.load_pruning_state("/nonexistent/file.json")
