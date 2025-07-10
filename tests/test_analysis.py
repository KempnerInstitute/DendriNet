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
    InformationAnalyzer,
    BranchActivationAnalyzer,
    AnalysisManager
)
from dendritic_modeling.analysis.tools import (
    SynapticActivationAnalyzer
)
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity
from dendritic_modeling.models import Classifier
from dendritic_modeling.config import (
    TransferFunctionParams, 
    PerformanceAnalysisParams,
    AblationAnalysisParams,
    BranchActivationAnalysisParams,
    SynapticActivationAnalysisParams,
    AnalysisConfig
)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test the PerformanceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cpu"
        params = PerformanceAnalysisParams()
        self.analyzer = PerformanceAnalyzer(params)
        
        # Create a simple test model
        transfer_params = TransferFunctionParams().asdict()
        self.model = Classifier(
            encoder_network=Identity(),
            core_network=ExcitationInhibitionNetwork(
                input_dim=10,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[]
            ),
            decoder_network=Identity()
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
            results = self.analyzer.analyze(
                model=self.model,
                test_ds=self.test_data,
                device=self.device,
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
        params = AblationAnalysisParams()
        self.analyzer = AblationAnalyzer(params)
        
        # Create test model with multiple layers
        transfer_params = TransferFunctionParams().asdict()
        self.model = Classifier(
            encoder_network=Identity(),
            core_network=ExcitationInhibitionNetwork(
                input_dim=10,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4, 4],
                inhibitory_layer_sizes=[4, 4],
                excitatory_branch_factors=[2, 2],
                inhibitory_branch_factors=[2, 2],
                ee_synapses_per_branch_per_layer=[1, 1],
                ei_synapses_per_branch_per_layer=[1, 1],
                ie_synapses_per_branch_per_layer=[1, 1],
                ii_synapses_per_branch_per_layer=[]
            ),
            decoder_network=Identity()
        )
        
        # Create test dataset
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(50, 10),
            torch.randint(0, 2, (50,))
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, AblationAnalyzer)
    
    def test_branch_ablation(self):
        """Test branch ablation functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir
            )
            
            # Check results structure
            self.assertIsInstance(results, dict)
            self.assertIn('baseline_accuracy', results)
    
    def test_layer_ablation(self):
        """Test layer-wise ablation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir
            )
            
            self.assertIsInstance(results, dict)


class TestInformationAnalyzer(unittest.TestCase):
    """Test the Information Analyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        from dendritic_modeling.config import InformationAnalysisParams
        params = InformationAnalysisParams()
        self.analyzer = InformationAnalyzer(params)
        
        # Create test model
        transfer_params = TransferFunctionParams().asdict()
        self.model = Classifier(
            encoder_network=Identity(),
            core_network=ExcitationInhibitionNetwork(
                input_dim=8,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[]
            ),
            decoder_network=Identity()
        )
        
        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(30, 8),
            torch.randint(0, 2, (30,))
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, InformationAnalyzer)
    
    def test_mutual_information_analysis(self):
        """Test mutual information computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the MI computation since it requires complex dependencies
            with patch.object(self.analyzer, '_compute_mutual_information') as mock_mi:
                mock_mi.return_value = 0.5
                
                results = self.analyzer.analyze(
                    model=self.model,
                    test_dataset=self.test_data,
                    device="cpu",
                    save_path=tmpdir
                )
                
                self.assertIsInstance(results, dict)



class TestActivationAnalyzers(unittest.TestCase):
    """Test the activation analyzer classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        synaptic_params = SynapticActivationAnalysisParams()
        self.synaptic_analyzer = SynapticActivationAnalyzer(synaptic_params)
        branch_params = BranchActivationAnalysisParams()
        self.branch_analyzer = BranchActivationAnalyzer(branch_params)
        
        # Create test model
        transfer_params = TransferFunctionParams().asdict()
        self.model = Classifier(
            encoder_network=Identity(),
            core_network=ExcitationInhibitionNetwork(
                input_dim=8,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[]
            ),
            decoder_network=Identity()
        )
        
        # Create test data
        self.test_data = torch.utils.data.TensorDataset(
            torch.randn(30, 8),
            torch.randint(0, 2, (30,))
        )
    
    def test_synaptic_activation_analyzer(self):
        """Test synaptic activation analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.synaptic_analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir
            )
            
            self.assertIsInstance(results, dict)
    
    def test_branch_activation_analyzer(self):
        """Test branch activation analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = self.branch_analyzer.analyze(
                model=self.model,
                test_dataset=self.test_data,
                device="cpu",
                save_path=tmpdir
            )
            
            self.assertIsInstance(results, dict)


class TestAnalysisManager(unittest.TestCase):
    """Test the Analysis Manager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test model
        transfer_params = TransferFunctionParams().asdict()
        self.model = Classifier(
            encoder_network=Identity(),
            core_network=ExcitationInhibitionNetwork(
                input_dim=8,
                transfer_params=transfer_params,
                excitatory_layer_sizes=[4],
                inhibitory_layer_sizes=[4],
                excitatory_branch_factors=[2],
                inhibitory_branch_factors=[2],
                ee_synapses_per_branch_per_layer=[1],
                ei_synapses_per_branch_per_layer=[1],
                ie_synapses_per_branch_per_layer=[1],
                ii_synapses_per_branch_per_layer=[]
            ),
            decoder_network=Identity()
        )
        
        # Create test data
        train_data = torch.utils.data.TensorDataset(
            torch.randn(30, 8),
            torch.randint(0, 2, (30,))
        )
        valid_data = torch.utils.data.TensorDataset(
            torch.randn(20, 8),
            torch.randint(0, 2, (20,))
        )
        test_data = torch.utils.data.TensorDataset(
            torch.randn(25, 8),
            torch.randint(0, 2, (25,))
        )
        
        self.data = {
            "train": train_data,
            "valid": valid_data,
            "test": test_data
        }
        
        # Create analysis config
        self.analysis_config = AnalysisConfig()
        
        # Initialize manager
        self.manager = AnalysisManager(
            model=self.model,
            data=self.data,
            analysis_config=self.analysis_config,
            save_root="/tmp/test_analysis"
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