"""
Test suite for dendritic_modeling.analysis module.

This module tests all analysis functionality including:
- Performance analysis
- Ablation studies  
- Information analysis
- Branch analysis
- Activation analysis
"""

import unittest
import tempfile
import torch
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock

from dendritic_modeling.analysis import (
    PerformanceAnalyzer,
    AblationAnalyzer,
    EIInformationAnalyzer,
    BranchInformationAnalyzer,
    SynapticActivationAnalyzer,
    BranchActivationAnalyzer,
    AnalysisManager
)
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.models import ProbabilisticClassifier


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test the PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        self.analyzer = PerformanceAnalyzer(device=self.device)
        
        # Create a simple test model
        self.model = ProbabilisticClassifier(
            net=ExcitationInhibitionNetwork(
                input_dim=10,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                output_dim=2
            ),
            output_dim=2
        )
        
        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 2, (100,))
        )
    
    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        self.assertEqual(self.analyzer.device, self.device)
        self.assertIsInstance(self.analyzer, PerformanceAnalyzer)
    
    def test_compute_accuracy(self):
        """Test accuracy computation."""
        # Create simple test predictions and targets
        predictions = torch.tensor([0, 1, 0, 1, 0])
        targets = torch.tensor([0, 1, 1, 1, 0])
        
        accuracy = self.analyzer._compute_accuracy(predictions, targets)
        expected_accuracy = 4.0 / 5.0  # 4 correct out of 5
        self.assertAlmostEqual(accuracy, expected_accuracy, places=4)
    
    def test_compute_categorical_loglikelihood(self):
        """Test categorical log-likelihood computation."""
        # Create simple test logits and targets
        logits = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        targets = torch.tensor([0, 1, 0])
        
        loglik = self.analyzer._compute_categorical_loglikelihood(logits, targets)
        self.assertIsInstance(loglik, float)
        self.assertLess(loglik, 0)  # Should be negative log-likelihood
    
    def test_evaluate_model(self):
        """Test model evaluation on dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.analyzer.analyze_performance(
                model=self.model,
                test_data=self.test_data,
                save_path=tmpdir
            )
            
            # Check that results contain expected keys
            self.assertIn('accuracy', results)
            self.assertIn('categorical_loglikelihood', results)
            
            # Check that results are reasonable
            self.assertGreaterEqual(results['accuracy'], 0.0)
            self.assertLessEqual(results['accuracy'], 1.0)
            self.assertLess(results['categorical_loglikelihood'], 0.0)


class TestAblationAnalyzer(unittest.TestCase):
    """Test the AblationAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = AblationAnalyzer(device="cpu")
        
        # Create test model with multiple layers
        self.model = ProbabilisticClassifier(
            net=ExcitationInhibitionNetwork(
                input_dim=10,
                excitatory_layer_sizes=[4, 4],
                inhibitory_layer_sizes=[4, 4],
                excitatory_branch_factors=[2, 2],
                inhibitory_branch_factors=[2, 2],
                output_dim=2
            ),
            output_dim=2
        )
        
        # Create test dataset
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, AblationAnalyzer)
        self.assertEqual(self.analyzer.device, "cpu")
    
    def test_branch_ablation(self):
        """Test branch ablation functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.analyzer.analyze_ablation(
                model=self.model,
                test_data=self.test_data,
                save_path=tmpdir,
                ablate_synapses=True,
                ablate_branches=True
            )
            
            # Check results structure
            self.assertIsInstance(results, dict)
            self.assertIn('baseline_accuracy', results)
    
    def test_layer_ablation(self):
        """Test layer-wise ablation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.analyzer.analyze_ablation(
                model=self.model,
                test_data=self.test_data,
                save_path=tmpdir,
                ablate_synapses=False,
                ablate_inhibition=True
            )
            
            self.assertIsInstance(results, dict)


class TestEIInformationAnalyzer(unittest.TestCase):
    """Test the EI Information Analyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = EIInformationAnalyzer(device="cpu")
        
        # Create test model
        self.model = ProbabilisticClassifier(
            net=ExcitationInhibitionNetwork(
                input_dim=8,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                output_dim=2
            ),
            output_dim=2
        )
        
        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(30, 8),
            torch.randint(0, 2, (30,))
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, EIInformationAnalyzer)
    
    def test_mutual_information_analysis(self):
        """Test mutual information computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the MI computation since it requires complex dependencies
            with patch.object(self.analyzer, '_compute_mutual_information') as mock_mi:
                mock_mi.return_value = 0.5
                
                results = self.analyzer.analyze_ei_information(
                    model=self.model,
                    test_data=self.test_data,
                    save_path=tmpdir
                )
                
                self.assertIsInstance(results, dict)


class TestBranchInformationAnalyzer(unittest.TestCase):
    """Test the Branch Information Analyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BranchInformationAnalyzer(device="cpu")
        
        # Create test model
        self.model = ProbabilisticClassifier(
            net=ExcitationInhibitionNetwork(
                input_dim=8,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                output_dim=2
            ),
            output_dim=2
        )
        
        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(30, 8),
            torch.randint(0, 2, (30,))
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, BranchInformationAnalyzer)
    
    def test_branch_information_analysis(self):
        """Test branch information analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data for branch analysis
            results = self.analyzer.analyze_branch_information(
                model=self.model,
                test_data=self.test_data,
                save_path=tmpdir
            )
            
            self.assertIsInstance(results, dict)


class TestActivationAnalyzers(unittest.TestCase):
    """Test the activation analyzer classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synaptic_analyzer = SynapticActivationAnalyzer(device="cpu")
        self.branch_analyzer = BranchActivationAnalyzer(device="cpu")
        
        # Create test model
        self.model = ProbabilisticClassifier(
            net=ExcitationInhibitionNetwork(
                input_dim=8,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                output_dim=2
            ),
            output_dim=2
        )
        
        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(30, 8),
            torch.randint(0, 2, (30,))
        )
    
    def test_synaptic_activation_analyzer(self):
        """Test synaptic activation analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.synaptic_analyzer.analyze_synaptic_activations(
                model=self.model,
                test_data=self.test_data,
                save_path=tmpdir
            )
            
            self.assertIsInstance(results, dict)
    
    def test_branch_activation_analyzer(self):
        """Test branch activation analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.branch_analyzer.analyze_branch_activations(
                model=self.model,
                test_data=self.test_data,
                save_path=tmpdir
            )
            
            self.assertIsInstance(results, dict)


class TestAnalysisManager(unittest.TestCase):
    """Test the Analysis Manager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test model
        self.model = ProbabilisticClassifier(
            net=ExcitationInhibitionNetwork(
                input_dim=8,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                output_dim=2
            ),
            output_dim=2
        )
        
        # Create test data
        self.train_data = torch.utils.data.TensorDataset(
            torch.randn(30, 8),
            torch.randint(0, 2, (30,))
        )
        self.valid_data = torch.utils.data.TensorDataset(
            torch.randn(20, 8),
            torch.randint(0, 2, (20,))
        )
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(25, 8),
            torch.randint(0, 2, (25,))
        )
        
        # Create mock config
        self.mock_config = Mock()
        self.mock_config.performance_analysis = Mock()
        self.mock_config.performance_analysis.enabled = True
        self.mock_config.performance_analysis.params = {
            'accuracy': True,
            'categorical_loglikelihood': True
        }
        
        # Initialize manager
        self.manager = AnalysisManager(
            model=self.model,
            train_data=self.train_data,
            valid_data=self.valid_data,
            test_data=self.test_data,
            config=self.mock_config,
            save_path="/tmp/test_analysis"
        )
    
    def test_manager_initialization(self):
        """Test that the analysis manager initializes correctly."""
        self.assertIsInstance(self.manager, AnalysisManager)
        self.assertEqual(self.manager.model, self.model)
        self.assertEqual(self.manager.train_data, self.train_data)
    
    def test_run_performance_analysis(self):
        """Test running performance analysis through manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.manager.save_path = tmpdir
            
            # Mock the performance analyzer
            with patch.object(self.manager, 'performance_analyzer') as mock_analyzer:
                mock_analyzer.analyze_performance.return_value = {
                    'accuracy': 0.85,
                    'categorical_loglikelihood': -0.5
                }
                
                results = self.manager.run_performance_analysis()
                
                self.assertIsInstance(results, dict)
                mock_analyzer.analyze_performance.assert_called_once()


if __name__ == '__main__':
    unittest.main() 