"""
Tests for training optimizer factory to improve coverage.
"""

import pytest
import torch.nn as nn
import torch.optim as optim

from dendritic_modeling.training.optimizers.factory import (
    create_optimizer,
    create_parameter_groups,
    validate_optimizer_config,
)
from dendritic_modeling.config.training import OptimizerConfig


class TestCreateOptimizer:
    """Test create_optimizer function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    def test_create_adam_optimizer(self, simple_model):
        """Test creating Adam optimizer."""
        config = OptimizerConfig(
            name="adam", lr=0.001, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8
        )

        optimizer = create_optimizer(simple_model.parameters(), config)

        assert isinstance(optimizer, optim.Adam)
        assert optimizer.defaults["lr"] == 0.001
        assert optimizer.defaults["weight_decay"] == 1e-5
        assert optimizer.defaults["betas"] == (0.9, 0.999)
        assert optimizer.defaults["eps"] == 1e-8

    def test_create_adamw_optimizer(self, simple_model):
        """Test creating AdamW optimizer."""
        config = OptimizerConfig(
            name="adamw", lr=0.01, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-7
        )

        optimizer = create_optimizer(simple_model.parameters(), config)

        assert isinstance(optimizer, optim.AdamW)
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["weight_decay"] == 0.1
        assert optimizer.defaults["betas"] == (0.9, 0.95)
        assert optimizer.defaults["eps"] == 1e-7

    def test_create_sgd_optimizer(self, simple_model):
        """Test creating SGD optimizer."""
        config = OptimizerConfig(name="sgd", lr=0.1, momentum=0.9, weight_decay=1e-4)

        optimizer = create_optimizer(simple_model.parameters(), config)

        assert isinstance(optimizer, optim.SGD)
        assert optimizer.defaults["lr"] == 0.1
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 1e-4

    def test_create_rmsprop_optimizer(self, simple_model):
        """Test creating RMSprop optimizer."""
        config = OptimizerConfig(
            name="rmsprop", lr=0.01, eps=1e-8, momentum=0.9, weight_decay=1e-5
        )

        optimizer = create_optimizer(simple_model.parameters(), config)

        assert isinstance(optimizer, optim.RMSprop)
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["eps"] == 1e-8
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 1e-5

    def test_create_optimizer_case_insensitive(self, simple_model):
        """Test optimizer creation with different cases."""
        # Test uppercase
        config = OptimizerConfig(name="ADAM", lr=0.001)
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert isinstance(optimizer, optim.Adam)

        # Test mixed case
        config = OptimizerConfig(name="AdamW", lr=0.001)
        optimizer = create_optimizer(simple_model.parameters(), config)
        assert isinstance(optimizer, optim.AdamW)

    def test_create_optimizer_unsupported(self, simple_model):
        """Test creating unsupported optimizer."""
        config = OptimizerConfig(name="unsupported", lr=0.001)

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            create_optimizer(simple_model.parameters(), config)

    def test_create_optimizer_with_defaults(self, simple_model):
        """Test creating optimizer with default config values."""
        config = OptimizerConfig(name="adam")  # Only specify required field

        optimizer = create_optimizer(simple_model.parameters(), config)

        assert isinstance(optimizer, optim.Adam)
        # Should use default values from OptimizerConfig


class TestCreateParameterGroups:
    """Test create_parameter_groups function."""

    def test_create_parameter_groups_no_split(self):
        """Test creating parameter groups without splitting."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        config = OptimizerConfig(name="adam", lr=0.001)

        param_groups = create_parameter_groups(model, config)

        assert len(param_groups) == 1
        assert param_groups[0]["lr"] == 0.001
        assert len(list(param_groups[0]["params"])) > 0

    def test_create_parameter_groups_with_split(self):
        """Test creating parameter groups with splitting."""
        # Create a model with named parameters that match the patterns
        model = nn.Module()
        model.topk_layer = nn.Linear(10, 5)
        model.blocklinear_layer = nn.Linear(5, 3)
        model.reactivation_layer = nn.Linear(3, 2)
        model.decoder_layer = nn.Linear(2, 1)
        model.other_layer = nn.Linear(1, 1)

        config = OptimizerConfig(name="adam", lr=0.001)
        param_groups_config = {
            "split_params": True,
            "topk_lr": 0.01,
            "blocklinear_lr": 0.005,
            "reactivation_lr": 0.002,
            "decoder_lr": 0.001,
        }

        param_groups = create_parameter_groups(model, config, param_groups_config)

        # Should have multiple groups
        assert len(param_groups) > 1

        # Check that different learning rates are assigned
        lrs = [group["lr"] for group in param_groups]
        assert len(set(lrs)) > 1  # Should have different learning rates

    def test_create_parameter_groups_empty_categories(self):
        """Test creating parameter groups when some categories are empty."""
        model = nn.Sequential(nn.Linear(10, 5))  # No special parameter names
        config = OptimizerConfig(name="adam", lr=0.001)
        param_groups_config = {"split_params": True}

        param_groups = create_parameter_groups(model, config, param_groups_config)

        # Should still create at least the default group
        assert len(param_groups) >= 1

    def test_create_parameter_groups_none_config(self):
        """Test creating parameter groups with None config."""
        model = nn.Linear(10, 5)
        config = OptimizerConfig(name="adam", lr=0.001)

        param_groups = create_parameter_groups(model, config, None)

        assert len(param_groups) == 1
        assert param_groups[0]["lr"] == 0.001

    def test_create_parameter_groups_no_split_flag(self):
        """Test parameter groups when split_params is False."""
        model = nn.Linear(10, 5)
        config = OptimizerConfig(name="adam", lr=0.001)
        param_groups_config = {"split_params": False}

        param_groups = create_parameter_groups(model, config, param_groups_config)

        assert len(param_groups) == 1

    def test_parameter_groups_with_complex_model(self):
        """Test parameter groups with a more complex model structure."""

        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_topk = nn.Linear(10, 8)
                self.branch_blocklinear = nn.Linear(8, 6)
                self.soma_reactivation = nn.Linear(6, 4)
                self.final_decoder = nn.Linear(4, 2)
                self.auxiliary = nn.Linear(2, 1)

        model = ComplexModel()
        config = OptimizerConfig(name="adam", lr=0.001)
        param_groups_config = {
            "split_params": True,
            "topk_lr": 0.01,
            "blocklinear_lr": 0.005,
            "reactivation_lr": 0.002,
            "decoder_lr": 0.001,
        }

        param_groups = create_parameter_groups(model, config, param_groups_config)

        # Should categorize parameters appropriately
        assert len(param_groups) > 1

        # Verify that parameters are actually assigned to groups
        total_params_in_groups = sum(
            len(list(group["params"])) for group in param_groups
        )
        total_model_params = len(list(model.parameters()))
        assert total_params_in_groups == total_model_params


class TestValidateOptimizerConfig:
    """Test validate_optimizer_config function."""

    def test_validate_valid_adam_config(self):
        """Test validating valid Adam configuration."""
        config = OptimizerConfig(
            name="adam", lr=0.001, weight_decay=1e-5, betas=(0.9, 0.999), eps=1e-8
        )

        # Should not raise any exception
        validate_optimizer_config(config)

    def test_validate_valid_sgd_config(self):
        """Test validating valid SGD configuration."""
        config = OptimizerConfig(name="sgd", lr=0.1, momentum=0.9, weight_decay=1e-4)

        # Should not raise any exception
        validate_optimizer_config(config)

    def test_validate_invalid_optimizer_name(self):
        """Test validating invalid optimizer name."""
        config = OptimizerConfig(name="invalid_optimizer", lr=0.001)

        with pytest.raises(ValueError, match="Invalid optimizer"):
            validate_optimizer_config(config)

    def test_validate_negative_learning_rate(self):
        """Test validating negative learning rate."""
        config = OptimizerConfig(name="adam", lr=-0.001)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            validate_optimizer_config(config)

    def test_validate_zero_learning_rate(self):
        """Test validating zero learning rate."""
        config = OptimizerConfig(name="adam", lr=0.0)

        with pytest.raises(ValueError, match="Learning rate must be positive"):
            validate_optimizer_config(config)

    def test_validate_negative_weight_decay(self):
        """Test validating negative weight decay."""
        config = OptimizerConfig(name="adam", lr=0.001, weight_decay=-1e-5)

        with pytest.raises(ValueError, match="Weight decay must be non-negative"):
            validate_optimizer_config(config)

    def test_validate_invalid_betas_adam(self):
        """Test validating invalid betas for Adam."""
        # Test with wrong number of values
        config = OptimizerConfig(name="adam", lr=0.001, betas=(0.9,))

        with pytest.raises(ValueError, match="Invalid betas"):
            validate_optimizer_config(config)

        # Test with out-of-range values
        config = OptimizerConfig(name="adam", lr=0.001, betas=(1.1, 0.999))

        with pytest.raises(ValueError, match="Invalid betas"):
            validate_optimizer_config(config)

    def test_validate_invalid_betas_adamw(self):
        """Test validating invalid betas for AdamW."""
        config = OptimizerConfig(name="adamw", lr=0.001, betas=(-0.1, 0.999))

        with pytest.raises(ValueError, match="Invalid betas"):
            validate_optimizer_config(config)

    def test_validate_invalid_eps_adam(self):
        """Test validating invalid epsilon for Adam."""
        config = OptimizerConfig(name="adam", lr=0.001, eps=-1e-8)

        with pytest.raises(ValueError, match="Epsilon must be positive"):
            validate_optimizer_config(config)

        config = OptimizerConfig(name="adam", lr=0.001, eps=0.0)

        with pytest.raises(ValueError, match="Epsilon must be positive"):
            validate_optimizer_config(config)

    def test_validate_invalid_momentum_sgd(self):
        """Test validating invalid momentum for SGD."""
        config = OptimizerConfig(name="sgd", lr=0.1, momentum=-0.1)

        with pytest.raises(ValueError, match="Momentum must be non-negative"):
            validate_optimizer_config(config)

    def test_validate_invalid_momentum_rmsprop(self):
        """Test validating invalid momentum for RMSprop."""
        config = OptimizerConfig(name="rmsprop", lr=0.01, momentum=-0.1)

        with pytest.raises(ValueError, match="Momentum must be non-negative"):
            validate_optimizer_config(config)

    def test_validate_case_insensitive_names(self):
        """Test validating with case-insensitive optimizer names."""
        # Test uppercase
        config = OptimizerConfig(name="ADAM", lr=0.001)
        validate_optimizer_config(config)  # Should not raise

        # Test mixed case
        config = OptimizerConfig(name="AdamW", lr=0.001)
        validate_optimizer_config(config)  # Should not raise


class TestOptimizerFactoryEdgeCases:
    """Test edge cases for optimizer factory."""

    def test_create_optimizer_empty_parameters(self):
        """Test creating optimizer with empty parameters."""
        empty_params = iter([])  # Empty iterator
        config = OptimizerConfig(name="adam", lr=0.001)

        # PyTorch optimizers raise ValueError for empty parameter lists
        with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
            create_optimizer(empty_params, config)

    def test_parameter_groups_model_without_parameters(self):
        """Test parameter groups with model that has no parameters."""
        model = nn.Module()  # Empty module
        config = OptimizerConfig(name="adam", lr=0.001)

        param_groups = create_parameter_groups(model, config)

        assert len(param_groups) == 1
        assert len(list(param_groups[0]["params"])) == 0

    def test_validate_config_boundary_values(self):
        """Test validation with boundary values."""
        # Test minimum positive learning rate
        config = OptimizerConfig(name="adam", lr=1e-10)
        validate_optimizer_config(config)  # Should not raise

        # Test zero weight decay (should be allowed)
        config = OptimizerConfig(name="adam", lr=0.001, weight_decay=0.0)
        validate_optimizer_config(config)  # Should not raise

        # Test boundary beta values
        config = OptimizerConfig(name="adam", lr=0.001, betas=(0.0, 0.999))
        validate_optimizer_config(config)  # Should not raise

        config = OptimizerConfig(name="adam", lr=0.001, betas=(0.9, 0.999999))
        validate_optimizer_config(config)  # Should not raise

    def test_parameter_groups_with_partial_config(self):
        """Test parameter groups with partial configuration."""
        model = nn.Module()
        model.topk_layer = nn.Linear(5, 3)
        model.other_layer = nn.Linear(3, 1)

        config = OptimizerConfig(name="adam", lr=0.001)
        param_groups_config = {
            "split_params": True,
            "topk_lr": 0.01,
            # Missing other learning rates - should use defaults
        }

        param_groups = create_parameter_groups(model, config, param_groups_config)

        # Should handle missing lr configs by using default
        assert len(param_groups) >= 1
        for group in param_groups:
            assert "lr" in group
            assert group["lr"] > 0
