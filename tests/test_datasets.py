"""
Test suite for dendritic_modeling.datasets module.

This module tests all dataset functionality including:
- Synthetic datasets
- Real dataset loading (MNIST, CIFAR-10)
- Dataset utilities and transformations
- Data path handling
"""

import unittest
import tempfile
import torch
import numpy as np
import os
from unittest.mock import Mock, patch

from dendritic_modeling.datasets import (
    get_unified_datasets,
    load_mnist_as_datasets,
    load_cifar10_as_datasets,
    load_MNIST_modulo10,
    LineDataset,
    CenterSurroundDataset,
    OrientationBarsDataset,
    MultiXORDataset,
    InfoShuntingDataset,
    NoisyLineDataset,
    SparseMNISTDataset,
    CompositionalMNIST,
    OrthoNetDataset,
    RandomFlipMNISTDataset,
    PoissonGeneratorDataset,
    get_data_directory,
    split_MNIST_inputs
)


class TestSyntheticDatasets(unittest.TestCase):
    """Test synthetic dataset classes."""
    
    def test_line_dataset(self):
        """Test LineDataset creation and properties."""
        dataset = LineDataset(n_samples=100, image_size=10)
        
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.image_size, 10)
        
        # Test data access
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (100,))  # 10x10 flattened
        self.assertIn(label.item(), [0, 1, 2])
        
        # Test that data types are correct
        self.assertIsInstance(sample, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
    
    def test_center_surround_dataset(self):
        """Test CenterSurroundDataset creation and properties."""
        dataset = CenterSurroundDataset(n_samples=50, image_size=15)
        
        self.assertEqual(len(dataset), 50)
        self.assertEqual(dataset.image_size, 15)
        
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (225,))  # 15x15 flattened
        self.assertIn(label.item(), [0, 1, 2])
    
    def test_orientation_bars_dataset(self):
        """Test OrientationBarsDataset creation and properties."""
        n_classes = 4
        dataset = OrientationBarsDataset(
            n_samples=40, 
            image_size=16,
            n_orient_classes=n_classes
        )
        
        self.assertEqual(len(dataset), 40)
        self.assertEqual(dataset.n_orient_classes, n_classes)
        
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (256,))  # 16x16 flattened
        self.assertGreaterEqual(label.item(), 0)
        self.assertLess(label.item(), n_classes)
    
    def test_multi_xor_dataset(self):
        """Test MultiXORDataset creation and properties."""
        n_bits = 6
        dataset = MultiXORDataset(n_samples=30, n_bits=n_bits)
        
        self.assertEqual(len(dataset), 30)
        self.assertEqual(dataset.n_bits, n_bits)
        
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (n_bits,))
        
        # Check that all bits are binary
        self.assertTrue(torch.all((sample == 0) | (sample == 1)))
        
        # Check label range (should be 0-3 for standard XOR)
        self.assertGreaterEqual(label.item(), 0)
        self.assertLess(label.item(), 4)
    
    def test_info_shunting_dataset(self):
        """Test InfoShuntingDataset creation and properties."""
        dataset = InfoShuntingDataset(n_samples=100, input_dim=8)
        
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.input_dim, 8)
        
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (8,))
        self.assertIn(label.item(), [0, 1])
    
    def test_noisy_line_dataset(self):
        """Test NoisyLineDataset creation and properties."""
        dataset = NoisyLineDataset(n_samples=50, image_size=8, noise_level=0.1)
        
        self.assertEqual(len(dataset), 50)
        self.assertEqual(dataset.image_size, 8)
        self.assertEqual(dataset.noise_level, 0.1)
        
        sample, label = dataset[0]
        self.assertEqual(sample.shape, (64,))  # 8x8 flattened
        self.assertIn(label.item(), [0, 1, 2])
    
    def test_orthonet_dataset(self):
        """Test OrthoNetDataset creation and properties."""
        dataset = OrthoNetDataset(
            n_samples=50,
            input_dim=10,
            n_layers=3,
            nonlinearity='sigmoid',
            seed=42
        )
        
        self.assertEqual(len(dataset), 50)
        self.assertEqual(dataset.input_dim, 10)
        self.assertEqual(dataset.n_layers, 3)
        
        # Test deterministic behavior with seed
        sample1, label1 = dataset[0]
        dataset2 = OrthoNetDataset(
            n_samples=50,
            input_dim=10,
            n_layers=3,
            nonlinearity='sigmoid',
            seed=42
        )
        sample2, label2 = dataset2[0]
        
        self.assertTrue(torch.allclose(sample1, sample2))
        self.assertTrue(torch.allclose(label1, label2))


class TestRealDatasets(unittest.TestCase):
    """Test real dataset loading functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('dendritic_modeling.datasets.synthetic_datasets.datasets.MNIST')
    def test_load_mnist_as_datasets(self, mock_mnist):
        """Test MNIST dataset loading."""
        # Mock MNIST dataset
        mock_data = [(torch.randn(28, 28), i % 10) for i in range(100)]
        mock_mnist.return_value = mock_data
        
        train_ds, valid_ds, test_ds = load_mnist_as_datasets(
            train_valid_split=0.8,
            flatten=True,
            normalize=False,
            output_dir=self.temp_dir
        )
        
        # Check that datasets were created
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(valid_ds)
        self.assertIsNotNone(test_ds)
        
        # Check that the mock was called
        self.assertTrue(mock_mnist.called)
    
    @patch('dendritic_modeling.datasets.synthetic_datasets.datasets.CIFAR10')
    def test_load_cifar10_as_datasets(self, mock_cifar):
        """Test CIFAR-10 dataset loading."""
        # Mock CIFAR-10 dataset
        mock_data = [(torch.randn(3, 32, 32), i % 10) for i in range(100)]
        mock_cifar.return_value = mock_data
        
        train_ds, valid_ds, test_ds = load_cifar10_as_datasets(
            train_valid_split=0.8,
            flatten=False,
            normalize=True,
            output_dir=self.temp_dir
        )
        
        # Check that datasets were created
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(valid_ds)
        self.assertIsNotNone(test_ds)
        
        # Check that the mock was called
        self.assertTrue(mock_cifar.called)
    
    def test_load_mnist_modulo10(self):
        """Test MNIST modulo 10 dataset loading."""
        # Create mock datasets
        mock_train = torch.utils.data.TensorDataset(
            torch.randn(60, 784),
            torch.randint(0, 10, (60,))
        )
        mock_valid = torch.utils.data.TensorDataset(
            torch.randn(20, 784),
            torch.randint(0, 10, (20,))
        )
        mock_test = torch.utils.data.TensorDataset(
            torch.randn(20, 784),
            torch.randint(0, 10, (20,))
        )
        
        with patch('dendritic_modeling.datasets.synthetic_datasets.load_mnist_as_datasets') as mock_load:
            mock_load.return_value = (mock_train, mock_valid, mock_test)
            
            data = load_MNIST_modulo10(
                shuffle_iterations=1,
                train_valid_split=0.8,
                flatten=True,
                output_dir=self.temp_dir
            )
            
            # Check that data dictionary was created
            self.assertIn('train', data)
            self.assertIn('valid', data)
            self.assertIn('test', data)
            
            # Check that mock was called with correct parameters
            mock_load.assert_called_once()


class TestDatasetUtilities(unittest.TestCase):
    """Test dataset utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_data_directory(self):
        """Test data directory creation and path handling."""
        # Test with custom path
        data_dir = get_data_directory("mnist", custom_path=self.temp_dir)
        expected_path = os.path.join(self.temp_dir, "mnist")
        self.assertEqual(data_dir, expected_path)
        self.assertTrue(os.path.exists(data_dir))
        
        # Test with output directory
        data_dir2 = get_data_directory("cifar", output_dir=self.temp_dir)
        expected_path2 = os.path.join(self.temp_dir, "datasets", "cifar")
        self.assertEqual(data_dir2, expected_path2)
        self.assertTrue(os.path.exists(data_dir2))
    
    def test_split_mnist_inputs(self):
        """Test MNIST input splitting by labels."""
        # Create test data with multiple classes
        inputs = torch.randn(100, 784)
        labels = torch.cat([
            torch.zeros(30, dtype=torch.long),  # 30 samples of class 0
            torch.ones(40, dtype=torch.long),   # 40 samples of class 1
            torch.full((30,), 2, dtype=torch.long)  # 30 samples of class 2
        ])
        
        input_list = split_MNIST_inputs(inputs, labels)
        
        # Check that we got the right number of classes
        self.assertEqual(len(input_list), 3)
        
        # Check shapes
        self.assertEqual(input_list[0].shape, (30, 784))  # Class 0
        self.assertEqual(input_list[1].shape, (40, 784))  # Class 1
        self.assertEqual(input_list[2].shape, (30, 784))  # Class 2


class TestGetUnifiedDatasets(unittest.TestCase):
    """Test the unified dataset loading function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock configs
        self.task_cfg = Mock()
        self.train_cfg = Mock()
        self.train_cfg.train_valid_split = 0.8
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_synthetic_datasets(self):
        """Test loading of synthetic datasets."""
        synthetic_datasets = [
            "line", "center_surround", "orientation_bars", 
            "multi_xor", "info_shunting"
        ]
        
        for dataset_name in synthetic_datasets:
            with self.subTest(dataset=dataset_name):
                self.task_cfg.dataset = dataset_name
                self.task_cfg.data_path = None
                self.task_cfg.parameters = {
                    'train_size': 30,
                    'valid_size': 10,
                    'test_size': 10,
                    'image_size': 8,
                    'n_orient_classes': 4,
                    'n_bits': 6
                }
                
                train_ds, valid_ds, test_ds = get_unified_datasets(
                    self.task_cfg, 
                    self.train_cfg,
                    output_dir=self.temp_dir
                )
                
                # Check that datasets were created
                self.assertIsNotNone(train_ds)
                self.assertIsNotNone(valid_ds)
                self.assertIsNotNone(test_ds)
                
                # Check sizes
                self.assertEqual(len(train_ds), 30)
                self.assertEqual(len(valid_ds), 10)
                self.assertEqual(len(test_ds), 10)
    
    @patch('dendritic_modeling.datasets.synthetic_datasets.load_mnist_as_datasets')
    def test_mnist_dataset(self, mock_load_mnist):
        """Test MNIST dataset loading through unified interface."""
        # Setup mock
        mock_train = torch.utils.data.TensorDataset(torch.randn(48, 784), torch.randint(0, 10, (48,)))
        mock_valid = torch.utils.data.TensorDataset(torch.randn(12, 784), torch.randint(0, 10, (12,)))
        mock_test = torch.utils.data.TensorDataset(torch.randn(10, 784), torch.randint(0, 10, (10,)))
        mock_load_mnist.return_value = (mock_train, mock_valid, mock_test)
        
        # Configure for MNIST
        self.task_cfg.dataset = "mnist"
        self.task_cfg.data_path = None
        self.task_cfg.parameters = {'flatten': True, 'normalize': False}
        
        train_ds, valid_ds, test_ds = get_unified_datasets(
            self.task_cfg,
            self.train_cfg,
            output_dir=self.temp_dir
        )
        
        # Verify results
        self.assertEqual(len(train_ds), 48)
        self.assertEqual(len(valid_ds), 12)
        self.assertEqual(len(test_ds), 10)
        
        # Verify the function was called with correct parameters
        mock_load_mnist.assert_called_once()
        args, kwargs = mock_load_mnist.call_args
        self.assertEqual(kwargs['output_dir'], self.temp_dir)
    
    def test_invalid_dataset(self):
        """Test handling of invalid dataset names."""
        self.task_cfg.dataset = "invalid_dataset_name"
        self.task_cfg.data_path = None
        self.task_cfg.parameters = {}
        
        with self.assertRaises(ValueError):
            get_unified_datasets(self.task_cfg, self.train_cfg, output_dir=self.temp_dir)


class TestSpecializedDatasets(unittest.TestCase):
    """Test specialized dataset classes with complex behavior."""
    
    def test_poisson_generator_dataset(self):
        """Test PoissonGeneratorDataset functionality."""
        # Create base data
        base_inputs = torch.rand(50, 10) * 2  # Scale for better Poisson behavior
        base_labels = torch.randint(0, 2, (50,))
        
        dataset = PoissonGeneratorDataset(
            base_inputs=base_inputs,
            base_labels=base_labels,
            rate_multiplier=1.5
        )
        
        self.assertEqual(len(dataset), 50)
        
        # Test that output has Poisson noise
        sample, label = dataset[0]
        self.assertEqual(sample.shape, base_inputs[0].shape)
        self.assertEqual(label, base_labels[0])
        
        # The output should be different from input due to Poisson noise
        # (though this is probabilistic, so we test multiple samples)
        differences = []
        for i in range(10):
            sample, _ = dataset[i]
            diff = torch.sum(torch.abs(sample - base_inputs[i])).item()
            differences.append(diff)
        
        # At least some samples should be different
        self.assertTrue(any(diff > 0 for diff in differences))
    
    def test_compositional_mnist(self):
        """Test CompositionalMNIST dataset."""
        # Create mock MNIST data
        mock_mnist_data = [(torch.randn(28, 28), i % 10) for i in range(20)]
        
        dataset = CompositionalMNIST(mock_mnist_data, n_samples=10)
        
        self.assertEqual(len(dataset), 10)
        
        sample, label = dataset[0]
        # Should be horizontally concatenated images
        self.assertEqual(sample.shape, (28 * 28 * 2,))  # Two 28x28 images flattened
        
        # Label should be in range 0-99 (digit1*10 + digit2)
        self.assertGreaterEqual(label.item(), 0)
        self.assertLessEqual(label.item(), 99)
    
    def test_random_flip_mnist_dataset(self):
        """Test RandomFlipMNISTDataset functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the MNIST loading
            with patch('dendritic_modeling.datasets.synthetic_datasets.datasets.MNIST') as mock_mnist:
                mock_data = [(torch.randn(28, 28), i % 10) for i in range(100)]
                mock_mnist.return_value = mock_data
                
                # Test with random flipping
                dataset = RandomFlipMNISTDataset(
                    split="train",
                    output_dir=temp_dir,
                    random_flip=True
                )
                
                self.assertGreater(len(dataset), 0)
                
                sample, label = dataset[0]
                self.assertEqual(sample.shape, (784,))  # 28x28 flattened
                
                # Test with no flipping
                dataset_no_flip = RandomFlipMNISTDataset(
                    split="train",
                    output_dir=temp_dir,
                    random_flip=False
                )
                
                sample_no_flip, label_no_flip = dataset_no_flip[0]
                self.assertEqual(sample_no_flip.shape, (784,))


if __name__ == '__main__':
    unittest.main() 