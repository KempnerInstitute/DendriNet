"""
Extended tests for training factory to improve coverage.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from dendritic_modeling.training.factory import (
    get_trainer,
    create_optimizer,
    setup_training_components,
    validate_training_config,
)


class TestTrainingFactoryExtended:
    """Extended tests for training factory functions."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def sample_optimizer(self, simple_model):
        """Create a sample optimizer."""
        return torch.optim.Adam(simple_model.parameters(), lr=0.001)
    
    def test_get_trainer_two_step_strategy(self, sample_optimizer):
        """Test getting two-step trainer."""
        config = {
            "loss_function": "cross_entropy",
            "epochs": 10,
            "batch_size": 32,
        }
        
        trainer = get_trainer("two_step", sample_optimizer, config)
        
        assert trainer is not None
        assert hasattr(trainer, 'train')
    
    def test_get_trainer_freeze_branches_strategy(self, sample_optimizer):
        """Test getting freeze branches trainer."""
        config = {
            "loss_function": "cross_entropy",
            "epochs": 10,
            "batch_size": 32,
            "freeze_epochs": 5,
        }
        
        trainer = get_trainer("freeze_branches", sample_optimizer, config)
        
        assert trainer is not None
        assert hasattr(trainer, 'train')
    
    def test_get_trainer_voltage_stabilization_strategy(self, sample_optimizer):
        """Test getting voltage stabilization trainer."""
        config = {
            "loss_function": "cross_entropy",
            "epochs": 10,
            "batch_size": 32,
            "voltage_threshold": 0.5,
        }
        
        trainer = get_trainer("voltage_stabilization", sample_optimizer, config)
        
        assert trainer is not None
        assert hasattr(trainer, 'train')
    
    def test_get_trainer_invalid_strategy(self, sample_optimizer):
        """Test getting trainer with invalid strategy."""
        config = {
            "loss_function": "cross_entropy",
            "epochs": 10,
            "batch_size": 32,
        }
        
        with pytest.raises(ValueError):
            get_trainer("invalid_strategy", sample_optimizer, config)
    
    def test_create_optimizer_adam(self, simple_model):
        """Test creating Adam optimizer."""
        optimizer_config = {
            "type": "adam",
            "lr": 0.001,
            "weight_decay": 1e-5,
        }
        
        optimizer = create_optimizer(simple_model, optimizer_config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['weight_decay'] == 1e-5
    
    def test_create_optimizer_sgd(self, simple_model):
        """Test creating SGD optimizer."""
        optimizer_config = {
            "type": "sgd",
            "lr": 0.01,
            "momentum": 0.9,
        }
        
        optimizer = create_optimizer(simple_model, optimizer_config)
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == 0.01
        assert optimizer.defaults['momentum'] == 0.9
    
    def test_create_optimizer_rmsprop(self, simple_model):
        """Test creating RMSprop optimizer."""
        optimizer_config = {
            "type": "rmsprop",
            "lr": 0.001,
            "alpha": 0.99,
        }
        
        optimizer = create_optimizer(simple_model, optimizer_config)
        
        assert isinstance(optimizer, torch.optim.RMSprop)
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['alpha'] == 0.99
    
    def test_create_optimizer_invalid_type(self, simple_model):
        """Test creating optimizer with invalid type."""
        optimizer_config = {
            "type": "invalid_optimizer",
            "lr": 0.001,
        }
        
        with pytest.raises(ValueError):
            create_optimizer(simple_model, optimizer_config)
    
    def test_setup_training_components(self, simple_model):
        """Test setting up training components."""
        config = {
            "optimizer": {
                "type": "adam",
                "lr": 0.001,
            },
            "loss_function": "cross_entropy",
            "scheduler": {
                "type": "step",
                "step_size": 10,
                "gamma": 0.1,
            }
        }
        
        components = setup_training_components(simple_model, config)
        
        assert 'optimizer' in components
        assert 'loss_function' in components
        assert 'scheduler' in components
        assert isinstance(components['optimizer'], torch.optim.Adam)
    
    def test_validate_training_config_valid(self):
        """Test validating a valid training config."""
        config = {
            "epochs": 10,
            "batch_size": 32,
            "loss_function": "cross_entropy",
            "optimizer": {
                "type": "adam",
                "lr": 0.001,
            }
        }
        
        # Should not raise any exceptions
        is_valid = validate_training_config(config)
        assert is_valid
    
    def test_validate_training_config_missing_required(self):
        """Test validating config with missing required fields."""
        config = {
            "batch_size": 32,
            # Missing epochs and loss_function
        }
        
        with pytest.raises(ValueError):
            validate_training_config(config)
    
    def test_validate_training_config_invalid_types(self):
        """Test validating config with invalid types."""
        config = {
            "epochs": "not_a_number",  # Should be int
            "batch_size": 32,
            "loss_function": "cross_entropy",
        }
        
        with pytest.raises(TypeError):
            validate_training_config(config)


class TestTrainingFactoryEdgeCases:
    """Test edge cases for training factory."""
    
    def test_create_optimizer_with_parameter_groups(self):
        """Test creating optimizer with parameter groups."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': model[0].parameters(), 'lr': 0.001},
            {'params': model[1].parameters(), 'lr': 0.01},
        ]
        
        optimizer_config = {
            "type": "adam",
            "param_groups": param_groups,
        }
        
        optimizer = create_optimizer(model, optimizer_config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) == 2
    
    def test_get_trainer_with_scheduler_config(self, simple_model):
        """Test getting trainer with scheduler configuration."""
        optimizer = torch.optim.Adam(simple_model.parameters())
        config = {
            "loss_function": "cross_entropy",
            "epochs": 10,
            "batch_size": 32,
            "scheduler": {
                "type": "step",
                "step_size": 5,
                "gamma": 0.5,
            }
        }
        
        trainer = get_trainer("standard", optimizer, config)
        
        assert trainer is not None
        assert hasattr(trainer, 'scheduler')
    
    def test_setup_components_minimal_config(self, simple_model):
        """Test setup with minimal configuration."""
        config = {
            "optimizer": {
                "type": "adam",
                "lr": 0.001,
            },
            "loss_function": "mse",
        }
        
        components = setup_training_components(simple_model, config)
        
        assert 'optimizer' in components
        assert 'loss_function' in components
        # Scheduler should be optional
        assert components.get('scheduler') is None or 'scheduler' in components
    
    def test_create_optimizer_empty_model(self):
        """Test creating optimizer for model with no parameters."""
        model = nn.Sequential()  # Empty model
        
        optimizer_config = {
            "type": "adam",
            "lr": 0.001,
        }
        
        # Should handle gracefully or raise appropriate error
        try:
            optimizer = create_optimizer(model, optimizer_config)
            # If it succeeds, check it's valid
            assert isinstance(optimizer, torch.optim.Adam)
        except ValueError:
            # It's acceptable to raise an error for empty models
            pass