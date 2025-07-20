"""Tests for TopK utilities module."""

import unittest
import torch
import torch.nn as nn

from dendritic_modeling.utils.topk import (
    get_topk_layer,
    get_available_topk_types,
    calculate_effective_sparsity,
    topk_mask_from_weights,
    apply_topk_sparsity,
    count_active_parameters,
    estimate_topk_memory_usage,
    register_topk_layer
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import TopKLinear


class TestTopKUtils(unittest.TestCase):
    """Test TopK utility functions."""
    
    def test_get_available_topk_types(self):
        """Test getting available TopK types."""
        types = get_available_topk_types()
        self.assertIn("standard", types)
        self.assertIn("stochastic", types)
        self.assertIn("variance", types)
        self.assertIsInstance(types, list)
    
    def test_get_topk_layer_standard(self):
        """Test creating standard TopK layer."""
        layer = get_topk_layer("standard", 10, 5, 3)
        self.assertIsInstance(layer, TopKLinear)
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.K, 3)
    
    def test_get_topk_layer_stochastic(self):
        """Test creating stochastic TopK layer."""
        layer = get_topk_layer("stochastic", 10, 5, 3, temperature=0.5)
        self.assertIsInstance(layer, TopKLinear)
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.K, 3)
    
    def test_get_topk_layer_invalid_type(self):
        """Test error for invalid TopK type."""
        with self.assertRaises(ValueError):
            get_topk_layer("invalid_type", 10, 5, 3)
    
    def test_calculate_effective_sparsity(self):
        """Test sparsity calculation."""
        weights = torch.randn(5, 10)
        sparsity = calculate_effective_sparsity(weights, K=3)
        expected_sparsity = 1.0 - (5 * 3) / (5 * 10)
        self.assertAlmostEqual(sparsity, expected_sparsity, places=6)
    
    def test_topk_mask_from_weights(self):
        """Test TopK mask creation."""
        weights = torch.tensor([[1.0, 3.0, 2.0], [0.5, 2.5, 1.5]])
        mask = topk_mask_from_weights(weights, K=2)
        
        # Should select top 2 elements per row
        self.assertEqual(mask.sum(dim=1).tolist(), [2.0, 2.0])
        
        # Check specific selections
        self.assertEqual(mask[0, 1].item(), 1.0)  # 3.0 selected
        self.assertEqual(mask[0, 2].item(), 1.0)  # 2.0 selected
        self.assertEqual(mask[0, 0].item(), 0.0)  # 1.0 not selected
    
    def test_apply_topk_sparsity(self):
        """Test applying TopK sparsity to weights."""
        weights = torch.tensor([[1.0, 3.0, 2.0, 0.5], [0.5, 2.5, 1.5, 3.0]])
        sparse_weights = apply_topk_sparsity(weights, K=2)
        
        # Should have only top 2 elements per row non-zero
        self.assertEqual((sparse_weights != 0).sum(dim=1).tolist(), [2, 2])
        
        # Check that largest elements are preserved
        self.assertEqual(sparse_weights[0, 1].item(), 3.0)  # Largest
        self.assertEqual(sparse_weights[0, 2].item(), 2.0)  # Second largest
        self.assertEqual(sparse_weights[0, 0].item(), 0.0)  # Zeroed out
        self.assertEqual(sparse_weights[0, 3].item(), 0.0)  # Zeroed out
    
    def test_count_active_parameters(self):
        """Test counting active parameters in TopK layers."""
        layer1 = get_topk_layer("standard", 10, 5, 3)
        layer2 = get_topk_layer("standard", 8, 4, 2)
        
        total_active = count_active_parameters([layer1, layer2])
        expected = (5 * 3) + (4 * 2)  # 15 + 8 = 23
        self.assertEqual(total_active, expected)
    
    def test_estimate_topk_memory_usage(self):
        """Test memory usage estimation."""
        memory_usage = estimate_topk_memory_usage(
            in_features=100,
            out_features=50,
            K=10,
            batch_size=32,
            dtype_bytes=4
        )
        
        self.assertIn("parameters", memory_usage)
        self.assertIn("indices", memory_usage)
        self.assertIn("activations", memory_usage)
        self.assertIn("total", memory_usage)
        
        # Check parameter memory calculation
        expected_param_memory = 50 * 100 * 4  # out * in * bytes
        self.assertEqual(memory_usage["parameters"], expected_param_memory)
        
        # Check total is sum of components
        total = (memory_usage["parameters"] + 
                memory_usage["indices"] + 
                memory_usage["activations"])
        self.assertEqual(memory_usage["total"], total)
    
    def test_register_topk_layer(self):
        """Test registering a new TopK layer type."""
        class CustomTopK(TopKLinear):
            pass
        
        # Register the new type
        register_topk_layer("custom", CustomTopK)
        
        # Check it's available
        types = get_available_topk_types()
        self.assertIn("custom", types)
        
        # Test creating it
        layer = get_topk_layer("custom", 5, 3, 2)
        self.assertIsInstance(layer, CustomTopK)
        
        # Clean up (remove from registry)
        from dendritic_modeling.utils.topk import TOPK_REGISTRY
        del TOPK_REGISTRY["custom"]


class TestAbstractAnalyzer(unittest.TestCase):
    """Test the AbstractAnalyzer base class functionality."""
    
    def test_abstract_analyzer_import(self):
        """Test that AbstractAnalyzer can be imported."""
        from dendritic_modeling.analysis.core.base import AbstractAnalyzer
        self.assertTrue(hasattr(AbstractAnalyzer, 'analyze'))
        self.assertTrue(hasattr(AbstractAnalyzer, 'save_results'))
    
    def test_concrete_analyzer_inheritance(self):
        """Test that concrete analyzers inherit from AbstractAnalyzer."""
        from dendritic_modeling.analysis.core.ablation import AblationAnalyzer
        from dendritic_modeling.analysis.core.information import InformationAnalyzer
        from dendritic_modeling.analysis.core.base import AbstractAnalyzer
        from dendritic_modeling.config.analysis import AblationAnalysisParams, InformationAnalysisParams
        
        # Test AblationAnalyzer
        ablation_params = AblationAnalysisParams()
        ablation_analyzer = AblationAnalyzer(ablation_params)
        self.assertIsInstance(ablation_analyzer, AbstractAnalyzer)
        self.assertTrue(hasattr(ablation_analyzer, 'save_results'))
        self.assertTrue(hasattr(ablation_analyzer, 'logger'))
        
        # Test InformationAnalyzer  
        info_params = InformationAnalysisParams()
        info_analyzer = InformationAnalyzer(info_params)
        self.assertIsInstance(info_analyzer, AbstractAnalyzer)
        self.assertTrue(hasattr(info_analyzer, 'save_results'))
        self.assertTrue(hasattr(info_analyzer, 'logger'))


class TestConfigCaching(unittest.TestCase):
    """Test configuration caching functionality."""
    
    def test_config_load_cache(self):
        """Test that config loading is cached."""
        import tempfile
        import os
        from dendritic_modeling.config import load_config
        
        # Create a temporary config file
        config_content = """
model:
  task: classification
train:
  seed: 42
task:
  dataset: mnist
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name
        
        try:
            # Load config twice
            config1 = load_config(temp_config_path)
            config2 = load_config(temp_config_path)
            
            # Should be the same object (cached)
            self.assertIs(config1, config2)
            self.assertEqual(config1.train.seed, 42)
            self.assertEqual(config1.task.dataset, "mnist")
            
        finally:
            # Clean up temp file
            os.unlink(temp_config_path)


if __name__ == '__main__':
    unittest.main() 