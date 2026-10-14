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
    normalize_sparse_layer_type,
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

    def test_sparse_ops_preserves_layer_class_reexports(self):
        """Legacy notebooks import sparse layer classes from sparse_ops."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
            DenseToSparseLinear as SynapseDenseToSparseLinear,
            IndexedDynamicTopKLinear as SynapseIndexedDynamicTopKLinear,
            IndexedRewireLinear as SynapseIndexedRewireLinear,
            IndexedSparseLinear as SynapseIndexedSparseLinear,
            TopKLinear as SynapseTopKLinear,
        )
        from dendritic_modeling.utils.sparse_ops import (
            DenseToSparseLinear,
            IndexedDynamicTopKLinear,
            IndexedRewireLinear,
            IndexedSparseLinear,
            TopKLinear as SparseOpsTopKLinear,
        )

        self.assertIs(IndexedSparseLinear, SynapseIndexedSparseLinear)
        self.assertIs(IndexedDynamicTopKLinear, SynapseIndexedDynamicTopKLinear)
        self.assertIs(IndexedRewireLinear, SynapseIndexedRewireLinear)
        self.assertIs(DenseToSparseLinear, SynapseDenseToSparseLinear)
        self.assertIs(SparseOpsTopKLinear, SynapseTopKLinear)

    def test_get_sparse_layer_standard(self):
        """Test creating standard sparse layer."""
        layer = get_sparse_layer("standard", 10, 5, 3)
        self.assertIsInstance(layer, TopKLinear)
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.K, 3)

    def test_sparse_layer_aliases_normalize_to_registry_keys(self):
        """Public alias spellings should resolve before registry lookup."""
        aliases = {
            None: "standard",
            "topk": "standard",
            "indexed-dynamic": "indexed_dynamic",
            "indexed_dynamic_topk": "indexed_dynamic",
            "indexed-rewire": "indexed_rewire",
            "indexed_rewire_topk": "indexed_rewire",
            "dense-to-sparse": "dense_to_sparse",
            "dense_to_sparse_topk": "dense_to_sparse",
            "annealed-topk": "dense_to_sparse",
        }

        for alias, expected in aliases.items():
            self.assertEqual(normalize_sparse_layer_type(alias), expected)

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
        """Replacing an existing sparse type needs allow_override=True."""
        from dendritic_modeling.utils.sparse_ops import get_sparse_layer

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_sparse_layer("standard", TopKLinear)
        register_sparse_layer("standard", TopKLinear, allow_override=True)

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
        from dendritic_modeling.analysis.core.branch_activation import (
            BranchActivationAnalyzer,
        )
        from dendritic_modeling.analysis.core.information import InformationAnalyzer
        from dendritic_modeling.analysis.metrics.adversarial_robustness import (
            AdversarialRobustnessAnalyzer,
        )
        from dendritic_modeling.analysis.metrics.layer_contribution import (
            SingleLayerContributionAnalyzer,
        )
        from dendritic_modeling.analysis.metrics.noise_perturbation import (
            NoisePerturbationAnalyzer,
        )
        from dendritic_modeling.analysis.metrics.performance import PerformanceAnalyzer
        from dendritic_modeling.analysis.tools.synaptic_activation import (
            SynapticActivationAnalyzer,
        )
        from dendritic_modeling.analysis.tools.synaptic_turnover import (
            SynapticTurnoverAnalyzer,
        )
        from dendritic_modeling.config.analysis import (
            AblationAnalysisParams,
            AdversarialRobustnessAnalysisParams,
            BranchActivationAnalysisParams,
            InformationAnalysisParams,
            NoisePerturbationAnalysisParams,
            PerformanceAnalysisParams,
            SingleLayerContributionAnalysisParams,
            SynapticActivationAnalysisParams,
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

        # Test hook-based analyzers
        branch_analyzer = BranchActivationAnalyzer(BranchActivationAnalysisParams())
        self.assertIsInstance(branch_analyzer, AbstractAnalyzer)
        self.assertTrue(hasattr(branch_analyzer, "save_results"))
        self.assertTrue(hasattr(branch_analyzer, "logger"))

        synaptic_analyzer = SynapticActivationAnalyzer(
            SynapticActivationAnalysisParams()
        )
        self.assertIsInstance(synaptic_analyzer, AbstractAnalyzer)
        self.assertTrue(hasattr(synaptic_analyzer, "save_results"))
        self.assertTrue(hasattr(synaptic_analyzer, "logger"))

        # Test metric analyzers
        metric_analyzers = [
            PerformanceAnalyzer(PerformanceAnalysisParams()),
            NoisePerturbationAnalyzer(NoisePerturbationAnalysisParams()),
            AdversarialRobustnessAnalyzer(AdversarialRobustnessAnalysisParams()),
            SingleLayerContributionAnalyzer(SingleLayerContributionAnalysisParams()),
        ]
        for analyzer in metric_analyzers:
            self.assertIsInstance(analyzer, AbstractAnalyzer)
            self.assertTrue(hasattr(analyzer, "save_results"))
            self.assertTrue(hasattr(analyzer, "logger"))

        # The training-state analyzer snapshots a model with branch layers.
        from types import SimpleNamespace

        from dendritic_modeling.config import TransferFunctionConfig
        from dendritic_modeling.models import Classifier
        from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity

        model = Classifier(
            encoder_network=Identity(input_dim=5),
            core_network=ExcitationInhibitionNetwork(
                input_dim=5,
                transfer_params=TransferFunctionConfig().asdict(),
                excitatory_layer_sizes=[3],
                inhibitory_layer_sizes=[0],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[2],
                ei_synapses_per_branch_per_layer=[0],
                ie_synapses_per_branch_per_layer=[0],
                ii_synapses_per_branch_per_layer=[0],
            ),
            decoder_network=Identity(input_dim=3),
        )
        turnover_analyzer = SynapticTurnoverAnalyzer(
            model, SimpleNamespace(snapshot_interval=1)
        )
        self.assertIsInstance(turnover_analyzer, AbstractAnalyzer)
        self.assertTrue(hasattr(turnover_analyzer, "save_results"))
        self.assertTrue(hasattr(turnover_analyzer, "logger"))


class TestConfigLoading(unittest.TestCase):
    """Test that configuration loading always reflects the file on disk."""

    def test_load_config_returns_fresh_independent_configs(self):
        """Each load parses the file again and returns an independent object."""
        import os
        import tempfile

        from dendritic_modeling.config import load_config

        config_content = """
experiment:
  seed: 42
data:
  dataset_name: mnist
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name

        try:
            config1 = load_config(temp_config_path)
            config2 = load_config(temp_config_path)
            self.assertIsNot(config1, config2)
            self.assertEqual(config1.experiment.seed, 42)
            self.assertEqual(config1.data.dataset_name, "mnist")
            config1.experiment.seed = 99
            self.assertEqual(load_config(temp_config_path).experiment.seed, 42)

            # An edited file is picked up by the next load.
            with open(temp_config_path, "w") as handle:
                handle.write(config_content.replace("seed: 42", "seed: 7"))
            self.assertEqual(load_config(temp_config_path).experiment.seed, 7)
        finally:
            os.unlink(temp_config_path)


if __name__ == "__main__":
    unittest.main()
