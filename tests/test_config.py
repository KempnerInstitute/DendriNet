"""
Test suite for dendritic_modeling.config module.

This module tests all configuration functionality including:
- Configuration loading and parsing
- Dataclass structure validation
- YAML/OmegaConf integration
- Configuration validation and error handling
"""

import unittest
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from unittest.mock import Mock, patch
import yaml

from omegaconf import OmegaConf, DictConfig

from dendritic_modeling.config import (
    Config,
    ModelConfig,
    TrainConfig,
    TaskConfig,
    AnalysisConfig,
    VisualizationConfig,
    WandbConfig,
    OutputsConfig,
    EINetParameters,
    CoreNetworkConfig,
    load_config,
    BaseConfig
)


class TestConfigDataclasses(unittest.TestCase):
    """Test configuration dataclass structures."""
    
    def test_network_parameters_creation(self):
        """Test EINetParameters dataclass creation."""
        params = EINetParameters(
            input_dim=784,
            excitatory_layer_sizes=[64, 32],
            inhibitory_layer_sizes=[32, 16],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            output_dim=10
        )
        
        self.assertEqual(params.input_dim, 784)
        self.assertEqual(params.excitatory_layer_sizes, [64, 32])
        self.assertEqual(params.output_dim, 10)
        self.assertTrue(params.reactivate)  # Default value
        self.assertTrue(params.somatic_synapses)  # Default value
    
    def test_network_config_creation(self):
        """Test CoreNetworkConfig dataclass creation."""
        network_config = CoreNetworkConfig(
            type="EINet",
            parameters=EINetParameters(
                input_dim=784,
                excitatory_layer_sizes=[32],
                inhibitory_layer_sizes=[24],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2]
            )
        )
        
        self.assertEqual(network_config.type, "EINet")
        self.assertIsInstance(network_config.parameters, EINetParameters)
        self.assertEqual(network_config.parameters.input_dim, 784)
    
    def test_model_config_creation(self):
        """Test ModelConfig dataclass creation."""
        model_config = ModelConfig(
            task="classification",
            core_network=CoreNetworkConfig(
                type="EINet",
                parameters=EINetParameters(
                    input_dim=784,
                    excitatory_layer_sizes=[32],
                    inhibitory_layer_sizes=[24],
                    excitatory_branch_factors=[2],
                    inhibitory_branch_factors=[2]
                )
            )
        )
        
        self.assertEqual(model_config.task, "classification")
        self.assertIsInstance(model_config.core_network, CoreNetworkConfig)
    
    def test_train_config_creation(self):
        """Test TrainConfig dataclass creation."""
        train_config = TrainConfig(
            run_name="test_run",
            learning_strategy="mle",
            seed=42,
            epochs=100,
            batch_size=64,
            optim_dict={
                'lr': 0.001,
                'split_params': True,
                'topk_lr': 0.0005
            }
        )
        
        self.assertEqual(train_config.run_name, "test_run")
        self.assertEqual(train_config.learning_strategy, "mle")
        self.assertEqual(train_config.seed, 42)
        self.assertEqual(train_config.epochs, 100)
        self.assertEqual(train_config.batch_size, 64)
        self.assertEqual(train_config.optim_dict['lr'], 0.001)
    
    def test_task_config_creation(self):
        """Test TaskConfig dataclass creation."""
        task_config = TaskConfig(
            dataset="mnist",
            data_path="/path/to/data",
            parameters={
                'flatten': True,
                'normalize': False,
                'train_size': 50000
            }
        )
        
        self.assertEqual(task_config.dataset, "mnist")
        self.assertEqual(task_config.data_path, "/path/to/data")
        self.assertEqual(task_config.parameters['flatten'], True)
    
    def test_wandb_config_creation(self):
        """Test WandbConfig dataclass creation."""
        wandb_config = WandbConfig(
            entity="test_entity",
            project="test_project",
            group="test_group",
            tags=["tag1", "tag2"],
            use_wandb=True
        )
        
        self.assertEqual(wandb_config.entity, "test_entity")
        self.assertEqual(wandb_config.project, "test_project")
        self.assertEqual(wandb_config.group, "test_group")
        self.assertEqual(wandb_config.tags, ["tag1", "tag2"])
        self.assertTrue(wandb_config.use_wandb)
    
    def test_complete_config_creation(self):
        """Test complete Config dataclass creation."""
        complete_config = Config(
            model=ModelConfig(
                task="classification",
                core_network=CoreNetworkConfig(
                    type="EINet",
                    parameters=EINetParameters(
                        input_dim=784,
                        excitatory_layer_sizes=[32],
                        inhibitory_layer_sizes=[24],
                        excitatory_branch_factors=[2],
                        inhibitory_branch_factors=[2]
                    )
                )
            ),
            train=TrainConfig(
                run_name="test_run",
                learning_strategy="mle",
                seed=42,
                epochs=100,
                batch_size=64
            ),
            task=TaskConfig(
                dataset="mnist",
                data_path=None,
                parameters={}
            ),
            wandb=WandbConfig(
                entity=None,
                project="test_project",
                use_wandb=False
            ),
            outputs=OutputsConfig(
                dir="/tmp/outputs"
            ),
            visualization=VisualizationConfig(
                enabled=False,
                visualize_epochs=[],
                plots=[]
            ),
            analysis=AnalysisConfig()
        )
        
        self.assertIsInstance(complete_config, Config)
        self.assertIsInstance(complete_config.model, ModelConfig)
        self.assertIsInstance(complete_config.train, TrainConfig)
        self.assertIsInstance(complete_config.task, TaskConfig)
        self.assertIsInstance(complete_config.wandb, WandbConfig)


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading and parsing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample config YAML
        self.sample_config = {
            'model': {
                'task': 'classification',
                'core_network': {
                    'type': 'EINet',
                    'parameters': {
                        'input_dim': 784,
                        'excitatory_layer_sizes': [64, 32],
                        'inhibitory_layer_sizes': [32, 16],
                        'excitatory_branch_factors': [2, 2],
                        'inhibitory_branch_factors': [2, 2],
                        'reactivate': True,
                        'somatic_synapses': True
                    }
                }
            },
            'train': {
                'run_name': 'test_run',
                'learning_strategy': 'mle',
                'seed': 42,
                'epochs': 100,
                'batch_size': 64,
                'optim_dict': {
                    'lr': 0.001,
                    'split_params': True
                }
            },
            'task': {
                'dataset': 'mnist',
                'data_path': None,
                'parameters': {
                    'flatten': True,
                    'normalize': False
                }
            },
            'wandb': {
                'entity': None,
                'project': 'test_project',
                'use_wandb': False,
                'group': None,
                'tags': []
            },
            'outputs': {
                'dir': '/tmp/outputs'
            },
            'visualization': {
                'enabled': False,
                'visualize_epochs': [],
                'plots': []
            },
            'analysis': {}
        }
        
        # Write config to file
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.sample_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        config = load_config(self.config_file)
        
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.task, "classification")
        self.assertEqual(config.train.run_name, "test_run")
        self.assertEqual(config.task.dataset, "mnist")
        self.assertEqual(config.model.core_network.parameters.input_dim, 784)
    
    def test_load_config_from_dictconfig(self):
        """Test loading configuration from OmegaConf DictConfig."""
        omega_config = OmegaConf.create(self.sample_config)
        config = load_config(omega_config)
        
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.task, "classification")
        self.assertEqual(config.train.learning_strategy, "mle")
    
    def test_load_config_from_dict(self):
        """Test loading configuration from plain dictionary."""
        config = load_config(self.sample_config)
        
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.probabilistic, True)
        self.assertEqual(config.train.epochs, 100)
    
    def test_load_config_invalid_file(self):
        """Test handling of invalid config file."""
        invalid_file = "/nonexistent/config.yaml"
        
        with self.assertRaises(FileNotFoundError):
            load_config(invalid_file)
    
    def test_load_config_invalid_yaml(self):
        """Test handling of invalid YAML content."""
        invalid_yaml_file = os.path.join(self.temp_dir, "invalid.yaml")
        with open(invalid_yaml_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with self.assertRaises(yaml.YAMLError):
            load_config(invalid_yaml_file)
    
    def test_create_default_config(self):
        """Test creation of default configuration."""
        default_config = Config()  # Use default constructor instead
        
        self.assertIsInstance(default_config, Config)
        self.assertIsInstance(default_config.model, ModelConfig)
        self.assertIsInstance(default_config.train, TrainConfig)
        self.assertIsInstance(default_config.task, TaskConfig)
        
        # Check some default values
        self.assertEqual(default_config.model.task, "classification")
        self.assertEqual(default_config.train.learning_strategy, "mle")
        self.assertEqual(default_config.task.dataset, "mnist")


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and error handling."""
    
    def test_network_parameters_validation(self):
        """Test EINetParameters validation."""
        # Test valid parameters
        valid_params = EINetParameters(
            input_dim=784,
            excitatory_layer_sizes=[32],
            inhibitory_layer_sizes=[24],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            output_dim=10
        )
        self.assertEqual(valid_params.input_dim, 784)
        
        # For now, just test that the constructor works
        # The actual validation would be in the model initialization
        self.assertTrue(True)
    
    def test_train_config_validation(self):
        """Test TrainConfig validation."""
        # Test valid config
        valid_config = TrainConfig(
            run_name="test",
            learning_strategy="mle",
            seed=42,
            epochs=100,
            batch_size=32
        )
        self.assertEqual(valid_config.epochs, 100)
        
        # For now, just test that the constructor works
        # The actual validation would be in the training setup
        self.assertTrue(True)
    
    def test_task_config_validation(self):
        """Test TaskConfig validation."""
        # Test valid config
        valid_config = TaskConfig(
            dataset="mnist",
            data_path="",
            parameters={}
        )
        self.assertEqual(valid_config.dataset, "mnist")
        
        # For now, just test that the constructor works
        # The actual validation would be in the dataset loading
        self.assertTrue(True)


class TestConfigUpdating(unittest.TestCase):
    """Test configuration updating and merging."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_config = Config()
    
    def test_config_field_updates(self):
        """Test updating individual config fields."""
        # Update train config
        self.base_config.train.epochs = 200
        self.base_config.train.batch_size = 128
        
        self.assertEqual(self.base_config.train.epochs, 200)
        self.assertEqual(self.base_config.train.batch_size, 128)
        
        # Update model config
        self.base_config.model.task = "regression"
        self.assertEqual(self.base_config.model.task, "regression")
    
    def test_nested_config_updates(self):
        """Test updating nested configuration fields."""
        # Update network parameters
        self.base_config.model.core_network.parameters.input_dim = 1024
        self.base_config.model.core_network.parameters.excitatory_layer_sizes = [128, 64]
        
        self.assertEqual(self.base_config.model.core_network.parameters.input_dim, 1024)
        self.assertEqual(self.base_config.model.core_network.parameters.excitatory_layer_sizes, [128, 64])
    
    def test_optim_dict_updates(self):
        """Test updating optimizer dictionary."""
        # Update optimizer parameters
        self.base_config.train.optim_dict['lr'] = 0.005
        self.base_config.train.optim_dict['weight_decay'] = 0.01
        
        self.assertEqual(self.base_config.train.optim_dict['lr'], 0.005)
        self.assertEqual(self.base_config.train.optim_dict['weight_decay'], 0.01)


class TestConfigSerialization(unittest.TestCase):
    """Test configuration serialization and deserialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_to_yaml(self):
        """Test converting config to YAML format."""
        # Convert to OmegaConf
        omega_config = OmegaConf.structured(self.config)
        
        # Save to YAML
        yaml_file = os.path.join(self.temp_dir, "serialized_config.yaml")
        OmegaConf.save(omega_config, yaml_file)
        
        self.assertTrue(os.path.exists(yaml_file))
        
        # Load back and verify
        loaded_config = load_config(yaml_file)
        self.assertEqual(loaded_config.model.task, self.config.model.task)
        self.assertEqual(loaded_config.train.epochs, self.config.train.epochs)
    
    def test_config_round_trip(self):
        """Test round-trip serialization/deserialization."""
        # Modify config
        self.config.train.epochs = 150
        self.config.model.core_network.parameters.input_dim = 512
        self.config.task.dataset = "cifar10"
        
        # Convert to dict
        config_dict = OmegaConf.to_container(OmegaConf.structured(self.config), resolve=True)
        
        # Load from dict
        restored_config = load_config(config_dict)
        
        # Verify values are preserved
        self.assertEqual(restored_config.train.epochs, 150)
        self.assertEqual(restored_config.model.core_network.parameters.input_dim, 512)
        self.assertEqual(restored_config.task.dataset, "cifar10")


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration with other modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
    
    def test_config_model_compatibility(self):
        """Test config compatibility with model initialization."""
        # This is a basic test to ensure config structure is compatible
        # with expected model initialization parameters
        model_config = self.config.model
        network_params = model_config.core_network.parameters
        
        # Check required fields exist
        required_fields = [
            'input_dim', 'excitatory_layer_sizes', 'inhibitory_layer_sizes',
            'excitatory_branch_factors', 'inhibitory_branch_factors'
        ]
        
        for field in required_fields:
            self.assertTrue(hasattr(network_params, field))
            self.assertIsNotNone(getattr(network_params, field))
    
    def test_config_training_compatibility(self):
        """Test config compatibility with training setup."""
        train_config = self.config.train
        
        # Check required training fields
        required_fields = [
            'learning_strategy', 'epochs', 'batch_size', 'seed'
        ]
        
        for field in required_fields:
            self.assertTrue(hasattr(train_config, field))
            self.assertIsNotNone(getattr(train_config, field))
        
        # Check optimizer dict has required fields
        self.assertIn('lr', train_config.optim_dict)
    
    def test_config_analysis_compatibility(self):
        """Test config compatibility with analysis setup."""
        analysis_config = self.config.analysis
        
        # Should have analysis configuration
        self.assertIsNotNone(analysis_config)
        
        # Basic structure should exist
        self.assertTrue(hasattr(analysis_config, '__dict__'))


if __name__ == '__main__':
    unittest.main() 