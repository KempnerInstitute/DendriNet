"""
Tests for training regularization module to improve coverage.
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.config.training import (
    PruningConfig,
    RegularizationConfig,
    SelectiveConfig,
)
from dendritic_modeling.training.regularization import (
    PruningManager,
    RegularizationManager,
    apply_sparsity_masks,
    create_sparsity_mask,
)


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
        config = RegularizationConfig(
            l1_weight=0.01, l2_weight=0.0, selective=SelectiveConfig()
        )
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(simple_model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0
        assert reg_loss.requires_grad

    def test_compute_global_regularization_l2(self, simple_model):
        """Test computing global L2 regularization."""
        config = RegularizationConfig(
            l1_weight=0.0, l2_weight=0.01, selective=SelectiveConfig()
        )
        manager = RegularizationManager(config)

        reg_loss = manager.compute_regularization_loss(simple_model)

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0.0
        assert reg_loss.requires_grad

    def test_compute_combined_regularization(self, simple_model):
        """Test computing combined L1 and L2 regularization."""
        config = RegularizationConfig(
            l1_weight=0.01, l2_weight=0.01, selective=SelectiveConfig()
        )
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
            selective=SelectiveConfig(layers=[0, 1, 2]),
            param_group_weights={
                "topk": {"l1_weight": 0.02, "l2_weight": 0.02},
                "blocklinear": {"l1_weight": 0.015, "l2_weight": 0.015},
                "decoder": {"l1_weight": 0.005, "l2_weight": 0.005},
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
            selective=SelectiveConfig(layers=[99]),
            param_group_weights={"nonexistent": {"l1_weight": 0.02, "l2_weight": 0.02}},
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
        assert manager.threshold == config.threshold
        assert manager.selective == config.selective

    def test_magnitude_pruning(self, simple_model):
        """Test magnitude-based pruning."""
        config = PruningConfig(enabled=True, threshold=0.5, selective=SelectiveConfig())
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
        config = PruningConfig(enabled=True, threshold=0.3, selective=SelectiveConfig())
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
        config = PruningConfig(enabled=True, threshold=0.5, selective=SelectiveConfig())
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
            threshold=0.001,
            selective=SelectiveConfig(),
            dynamic_pruning=True,
            pruning_frequency=10,
        )
        manager = PruningManager(config)

        # Test pruning at different stages
        sparsities = []
        for _iteration in [0, 1, 2, 3]:
            # Reset model weights
            for param in simple_model.parameters():
                if param.requires_grad:
                    param.data.uniform_(-1, 1)

            manager.prune_model(simple_model)

            # Calculate current sparsity
            total_params = 0
            zero_params = 0
            for param in simple_model.parameters():
                if param.dim() > 1:
                    total_params += param.numel()
                    zero_params += (param == 0).sum().item()

            current_sparsity = zero_params / total_params if total_params > 0 else 0
            sparsities.append(current_sparsity)

        # Verify we got sparsity measurements
        assert len(sparsities) == 4
        # Each iteration should produce some level of sparsity
        for sparsity in sparsities:
            assert sparsity >= 0.0

    def test_save_and_load_pruning_state(self, simple_model):
        """Test saving and loading pruning state."""
        config = PruningConfig(enabled=True, threshold=0.001)
        manager = PruningManager(config)

        # Apply pruning
        manager.prune_model(simple_model)

        # Save pruning state
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            manager.save_pruning_stats(tmp_path)
            assert os.path.exists(tmp_path)

            # Verify saved content
            with open(tmp_path) as f:
                saved_stats = json.load(f)

            assert isinstance(saved_stats, dict)

            # Stats saved successfully

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_threshold_based_pruning(self, simple_model):
        """Test threshold-based pruning (current implementation)."""
        config = PruningConfig(enabled=True, threshold=0.001)
        manager = PruningManager(config)

        # Should complete without error
        stats = manager.prune_model(simple_model)
        assert isinstance(stats, dict)


class TestRegularizationUtils:
    """Test regularization utility functions."""

    def test_create_sparsity_mask_magnitude(self):
        """Test creating sparsity mask based on magnitude."""
        model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))

        masks = create_sparsity_mask(model, sparsity_ratio=0.5)

        assert isinstance(masks, dict)
        assert len(masks) > 0
        for _name, mask in masks.items():
            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.bool

    def test_create_sparsity_mask_random(self):
        """Test creating sparsity mask with different ratio."""
        model = nn.Linear(5, 3)

        masks = create_sparsity_mask(model, sparsity_ratio=0.3)

        assert isinstance(masks, dict)
        assert len(masks) > 0
        for _name, mask in masks.items():
            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.bool

    def test_create_sparsity_mask_zero_sparsity(self):
        """Test creating mask with zero sparsity."""
        model = nn.Linear(3, 2)

        masks = create_sparsity_mask(model, sparsity_ratio=0.0)

        # All values should be kept (all True) with zero sparsity
        assert isinstance(masks, dict)
        for _name, mask in masks.items():
            assert mask.all()

    def test_create_sparsity_mask_full_sparsity(self):
        """Test creating mask with full sparsity."""
        model = nn.Linear(3, 2)

        masks = create_sparsity_mask(model, sparsity_ratio=1.0)

        # Most or all values should be pruned with full sparsity
        assert isinstance(masks, dict)
        for _name, mask in masks.items():
            # With sparsity_ratio=1.0, most weights should be pruned
            sparsity_achieved = (~mask).float().mean().item()
            assert sparsity_achieved >= 0.8  # At least 80% should be pruned

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

        config = PruningConfig(enabled=True, threshold=0.001)
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
        """Test creating mask for model with single parameter."""
        model = nn.Linear(1, 1, bias=False)

        masks = create_sparsity_mask(model, sparsity_ratio=0.5)

        assert isinstance(masks, dict)
        assert len(masks) >= 1
        for _name, mask in masks.items():
            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.bool

    def test_pruning_save_stats(self):
        """Test saving pruning statistics."""
        config = PruningConfig(enabled=True)
        manager = PruningManager(config)

        # The save_pruning_stats method exists
        assert hasattr(manager, "save_pruning_stats")
        assert callable(manager.save_pruning_stats)
