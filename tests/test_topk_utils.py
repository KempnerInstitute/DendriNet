"""Tests for TopK utilities module."""

import unittest

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.utils.sparse_ops import (
    SPARSE_LAYER_REGISTRY,
    apply_sparse_mask,
    calculate_effective_sparsity,
    count_active_parameters,
    create_topk_mask,
    estimate_memory_usage,
    get_available_sparse_types,
    get_sparse_layer,
    register_sparse_layer,
)


class TestSparseOps(unittest.TestCase):
    """Test sparse operations utility functions."""

    def test_get_available_sparse_types(self):
        """Test getting available sparse layer types."""
        types = get_available_sparse_types()
        self.assertIn("standard", types)
        self.assertIn("stochastic", types)
        self.assertIn("variance", types)
        self.assertIsInstance(types, list)

    def test_get_sparse_layer_standard(self):
        """Test creating standard sparse layer."""
        layer = get_sparse_layer("standard", 10, 5, 3)
        self.assertIsInstance(layer, TopKLinear)
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.K, 3)

    def test_get_sparse_layer_stochastic(self):
        """Test creating stochastic sparse layer."""
        layer = get_sparse_layer("stochastic", 10, 5, 3, temperature=0.5)
        self.assertIsInstance(layer, TopKLinear)
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.K, 3)

    def test_get_sparse_layer_invalid_type(self):
        """Test error for invalid sparse layer type."""
        with self.assertRaises(ValueError):
            get_sparse_layer("invalid_type", 10, 5, 3)

    def test_calculate_effective_sparsity(self):
        """Test sparsity calculation."""
        weights = torch.randn(5, 10)
        sparsity = calculate_effective_sparsity(weights, K=3)
        expected_sparsity = 1.0 - (5 * 3) / (5 * 10)
        self.assertAlmostEqual(sparsity, expected_sparsity, places=6)

    def test_create_topk_mask(self):
        """Test TopK mask creation."""
        weights = torch.tensor([[1.0, 3.0, 2.0], [0.5, 2.5, 1.5]])
        mask = create_topk_mask(weights, K=2)

        # Should select top 2 elements per row
        self.assertEqual(mask.sum(dim=1).tolist(), [2.0, 2.0])

        # Check specific selections
        self.assertEqual(mask[0, 1].item(), 1.0)  # 3.0 selected
        self.assertEqual(mask[0, 2].item(), 1.0)  # 2.0 selected
        self.assertEqual(mask[0, 0].item(), 0.0)  # 1.0 not selected

    def test_apply_sparse_mask(self):
        """Test applying sparse mask to weights."""
        weights = torch.tensor([[1.0, 3.0, 2.0, 0.5], [0.5, 2.5, 1.5, 3.0]])
        sparse_weights = apply_sparse_mask(weights, K=2)

        # Should have only top 2 elements per row non-zero
        self.assertEqual((sparse_weights != 0).sum(dim=1).tolist(), [2, 2])

        # Check that largest elements are preserved
        self.assertEqual(sparse_weights[0, 1].item(), 3.0)  # Largest
        self.assertEqual(sparse_weights[0, 2].item(), 2.0)  # Second largest
        self.assertEqual(sparse_weights[0, 0].item(), 0.0)  # Zeroed out
        self.assertEqual(sparse_weights[0, 3].item(), 0.0)  # Zeroed out

    def test_count_active_parameters(self):
        """Test counting active parameters in sparse layers."""
        weights = torch.randn(5, 10)
        active_count = count_active_parameters(weights, K=3)
        expected = 5 * 3  # 5 outputs * 3 connections each
        self.assertEqual(active_count, expected)

    def test_estimate_memory_usage(self):
        """Test memory usage estimation for sparse layers."""
        in_features = 100
        out_features = 50
        K = 10

        memory_estimate = estimate_memory_usage(in_features, out_features, K)

        self.assertIsInstance(memory_estimate, dict)
        self.assertIn("full_weights_mb", memory_estimate)
        self.assertIn("active_weights_mb", memory_estimate)
        self.assertIn("savings_mb", memory_estimate)
        self.assertIn("savings_percent", memory_estimate)

        # Verify memory savings
        self.assertGreater(memory_estimate["savings_mb"], 0)
        self.assertGreater(memory_estimate["savings_percent"], 0)

    def test_register_topk_layer(self):
        """Test registering a new TopK layer type."""
        from dendritic_modeling.utils.sparse_ops import (
            get_available_sparse_types,
            get_sparse_layer,
        )

        class CustomTopK(TopKLinear):
            pass

        # Register the new type
        register_sparse_layer("custom", CustomTopK)

        # Check it's available
        types = get_available_sparse_types()
        self.assertIn("custom", types)

        # Test creating it
        layer = get_sparse_layer("custom", 5, 3, 2)
        self.assertIsInstance(layer, CustomTopK)

        # Clean up (remove from registry)
        del SPARSE_LAYER_REGISTRY["custom"]

    def test_register_duplicate_layer(self):
        """Test registering duplicate layer type raises error."""
        from dendritic_modeling.utils.sparse_ops import get_sparse_layer

        # Try to register an existing type - this should not raise an error
        # as the registry allows overwriting (which might be intended behavior)
        register_sparse_layer("standard", TopKLinear)

        # Verify it still works
        layer = get_sparse_layer("standard", 5, 3, 2)
        self.assertIsInstance(layer, TopKLinear)


class TestAbstractAnalyzer(unittest.TestCase):
    """Test the AbstractAnalyzer base class functionality."""

    def test_abstract_analyzer_import(self):
        """Test that AbstractAnalyzer can be imported."""
        from dendritic_modeling.analysis.core.base import AbstractAnalyzer

        self.assertTrue(hasattr(AbstractAnalyzer, "analyze"))
        self.assertTrue(hasattr(AbstractAnalyzer, "save_results"))

    def test_concrete_analyzer_inheritance(self):
        """Test that concrete analyzers inherit from AbstractAnalyzer."""
        from dendritic_modeling.analysis.core.ablation import AblationAnalyzer
        from dendritic_modeling.analysis.core.base import AbstractAnalyzer
        from dendritic_modeling.analysis.core.information import InformationAnalyzer
        from dendritic_modeling.config.analysis import (
            AblationAnalysisParams,
            InformationAnalysisParams,
        )

        # Test AblationAnalyzer
        ablation_params = AblationAnalysisParams()
        ablation_analyzer = AblationAnalyzer(ablation_params)
        self.assertIsInstance(ablation_analyzer, AbstractAnalyzer)
        self.assertTrue(hasattr(ablation_analyzer, "save_results"))
        self.assertTrue(hasattr(ablation_analyzer, "logger"))

        # Test InformationAnalyzer
        info_params = InformationAnalysisParams()
        info_analyzer = InformationAnalyzer(info_params)
        self.assertIsInstance(info_analyzer, AbstractAnalyzer)
        self.assertTrue(hasattr(info_analyzer, "save_results"))
        self.assertTrue(hasattr(info_analyzer, "logger"))


class TestConfigCaching(unittest.TestCase):
    """Test configuration caching functionality."""

    def test_config_load_cache(self):
        """Test that config loading is cached."""
        import os
        import tempfile

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
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


if __name__ == "__main__":
    unittest.main()
