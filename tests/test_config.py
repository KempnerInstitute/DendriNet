import unittest
import os
import tempfile
import shutil
import yaml
from pathlib import Path

from dendritic_modeling.config import (
    load_config, 
    save_config, 
    build_model_from_config,
    build_dataset_from_config,
    setup_experiment_from_config
)

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for configs
        self.temp_dir = tempfile.mkdtemp()
        
        # Define a sample configuration
        self.sample_config = {
            'experiment': {
                'name': 'test_experiment',
                'seed': 42,
                'save_path': self.temp_dir
            },
            'model': {
                'type': 'ExcitationInhibitionNetwork',
                'params': {
                    'input_dim': 4,
                    'hidden_layer_sizes': [16, 16],
                    'branch_factors': [2, 2],
                    'output_dim': 2,
                    'use_simple_branches': False,
                    'activation': 'relu',
                    'dropout': 0.0
                }
            },
            'dataset': {
                'type': 'MultiXORDataset',
                'params': {
                    'n_samples': 100,
                    'n_bits': 4,
                    'seed': 42
                }
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001,
                'epochs': 10,
                'optimizer': 'adam',
                'scheduler': None,
                'train_split': 0.8,
                'device': 'cpu'
            }
        }
        
        # Save the config to a temp file
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.sample_config, f)
    
    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_load_config(self):
        """Test loading a configuration from a file."""
        # Load the config
        config = load_config(self.config_path)
        
        # Check that the config matches the original
        self.assertEqual(config['experiment']['name'], 'test_experiment')
        self.assertEqual(config['model']['type'], 'ExcitationInhibitionNetwork')
        self.assertEqual(config['dataset']['type'], 'MultiXORDataset')
        self.assertEqual(config['training']['batch_size'], 16)
    
    def test_save_config(self):
        """Test saving a configuration to a file."""
        # Modify the config
        modified_config = self.sample_config.copy()
        modified_config['experiment']['name'] = 'modified_experiment'
        
        # Save the modified config
        modified_path = os.path.join(self.temp_dir, 'modified_config.yaml')
        save_config(modified_config, modified_path)
        
        # Load it back
        loaded_config = load_config(modified_path)
        
        # Check that the modifications were saved
        self.assertEqual(loaded_config['experiment']['name'], 'modified_experiment')
    
    def test_build_model_from_config(self):
        """Test building a model from configuration."""
        # Build the model
        model = build_model_from_config(self.sample_config)
        
        # Check that the model has the expected class and parameters
        self.assertEqual(model.__class__.__name__, 'ExcitationInhibitionNetwork')
        self.assertEqual(model.input_dim, 4)
        self.assertEqual(model.output_dim, 2)
        self.assertEqual(len(model.excitatory_layers), 2)  # 2 hidden layers as specified
    
    def test_build_dataset_from_config(self):
        """Test building a dataset from configuration."""
        # Build the dataset
        dataset = build_dataset_from_config(self.sample_config)
        
        # Check that the dataset has the expected class and parameters
        self.assertEqual(dataset.__class__.__name__, 'MultiXORDataset')
        self.assertEqual(len(dataset), 100)  # n_samples = 100
        self.assertEqual(dataset[0][0].shape[0], 4)  # n_bits = 4
    
    def test_setup_experiment_from_config(self):
        """Test setting up a complete experiment from configuration."""
        # Set up the experiment
        result = setup_experiment_from_config(self.config_path)
        
        # Check that all expected components are returned
        self.assertIn('model', result)
        self.assertIn('dataset', result)
        self.assertIn('trainer', result)
        self.assertIn('config', result)
        
        # Check that components have expected types
        self.assertEqual(result['model'].__class__.__name__, 'ExcitationInhibitionNetwork')
        self.assertEqual(result['dataset'].__class__.__name__, 'MultiXORDataset')
        self.assertEqual(result['trainer'].__class__.__name__, 'ClassificationTrainer')
    
    def test_config_with_missing_fields(self):
        """Test handling of configs with missing fields."""
        # Create a config with missing fields
        incomplete_config = {
            'model': self.sample_config['model'],
            'dataset': self.sample_config['dataset']
            # Missing 'experiment' and 'training' sections
        }
        
        # Save the incomplete config
        incomplete_path = os.path.join(self.temp_dir, 'incomplete_config.yaml')
        save_config(incomplete_config, incomplete_path)
        
        # Build model (should work with defaults)
        model = build_model_from_config(incomplete_config)
        self.assertIsNotNone(model)
        
        # Build dataset (should work)
        dataset = build_dataset_from_config(incomplete_config)
        self.assertIsNotNone(dataset)
        
        # Full setup should raise an error for missing required fields
        with self.assertRaises(KeyError):
            setup_experiment_from_config(incomplete_path)
    
    def test_config_with_unknown_model(self):
        """Test handling of configs with unknown model types."""
        # Create a config with an unknown model type
        invalid_config = self.sample_config.copy()
        invalid_config['model']['type'] = 'NonExistentModel'
        
        # Save the invalid config
        invalid_path = os.path.join(self.temp_dir, 'invalid_config.yaml')
        save_config(invalid_config, invalid_path)
        
        # Building the model should raise an error
        with self.assertRaises(ValueError):
            build_model_from_config(invalid_config)
    
    def test_config_with_unknown_dataset(self):
        """Test handling of configs with unknown dataset types."""
        # Create a config with an unknown dataset type
        invalid_config = self.sample_config.copy()
        invalid_config['dataset']['type'] = 'NonExistentDataset'
        
        # Save the invalid config
        invalid_path = os.path.join(self.temp_dir, 'invalid_config.yaml')
        save_config(invalid_config, invalid_path)
        
        # Building the dataset should raise an error
        with self.assertRaises(ValueError):
            build_dataset_from_config(invalid_config)

if __name__ == '__main__':
    unittest.main() 