"""
Test suite for dendritic_modeling.training module.

This module tests all training functionality including:
- Different training strategies (MLE, two-step, voltage stabilization, etc.)
- Trainer factory functions
- Optimization and gradient handling
- Training workflows
"""

import unittest
import tempfile
import torch
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock

from dendritic_modeling.training import (
    get_trainer,
    CustomWeightDecayOptimizer,
    Trainer
)
from dendritic_modeling.training.strategies import (
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
    LocalCreditAssignment
)
from dendritic_modeling.networks import ExcitationInhibitionNetwork, Identity
from dendritic_modeling.models import Classifier
from dendritic_modeling.config import TransferFunctionParams


class TestTrainerFactory(unittest.TestCase):
    """Test the trainer factory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple optimizer for testing
        transfer_params = TransferFunctionParams().asdict()
        model = Classifier(
            encoder_network=Identity(input_dim=10),
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
            decoder_network=Identity(input_dim=4)
        )
        self.optimizer = torch.optim.Adam(model.parameters())
    
    def test_get_trainer_standard(self):
        """Test getting standard trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("standard", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, Trainer)
    
    def test_get_trainer_two_step(self):
        """Test getting two-step trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("two_step", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, TwoStepTrainer)
    
    def test_get_trainer_two_step_kl(self):
        """Test getting two-step KL trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("two_step_kl", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, TwoStepTrainerWithKL)
    
    def test_get_trainer_voltage_stabilization(self):
        """Test getting voltage stabilization trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("voltage_stabilization", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, VoltageStabilizationTrainer)
    
    def test_get_trainer_freeze_branches(self):
        """Test getting local credit assignment trainer."""
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("freeze_branches", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, LocalCreditAssignment)
    
    def test_get_trainer_invalid_strategy(self):
        """Test handling of invalid training strategy."""
        # Should fall back to standard trainer and log warning
        trainer_config = {"loss_function": "cross_entropy", "epochs": 10}
        trainer = get_trainer("invalid_strategy", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, Trainer)
    
    def test_get_trainer_with_options(self):
        """Test getting trainer with additional options."""
        trainer_config = {
            "loss_function": "cross_entropy", 
            "epochs": 10,
            "suppress_prints": True, 
            "print_every": 20
        }
        trainer = get_trainer("standard", self.optimizer, trainer_config)
        self.assertIsInstance(trainer, Trainer)
        self.assertTrue(trainer.suppress_prints)
        self.assertEqual(trainer.print_every, 20)


class TestTrainer(unittest.TestCase):
    """Test the standard trainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test model
        transfer_params = TransferFunctionParams().asdict()
        self.model = Classifier(
            encoder_network=Identity(input_dim=8),
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
            decoder_network=Identity(input_dim=4)
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create trainer
        self.trainer = Trainer(
            optimizer=self.optimizer,
            loss_function="cross_entropy",
            suppress_prints=True,
            print_every=10
        )
        
        # Create test data
        self.train_data = torch.utils.data.TensorDataset(
            torch.randn(50, 8),
            torch.randint(0, 2, (50,))
        )
        self.valid_data = torch.utils.data.TensorDataset(
            torch.randn(20, 8),
            torch.randint(0, 2, (20,))
        )
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, Trainer)
        self.assertEqual(self.trainer.optimizer, self.optimizer)
        self.assertTrue(self.trainer.suppress_prints)
        self.assertEqual(self.trainer.print_every, 10)
    
    def test_single_epoch_training(self):
        """Test single epoch of training."""
        # Train for one epoch
        results = self.trainer.train(
            model=self.model,
            train_data=self.train_data,
            valid_data=self.valid_data
        )
        
        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn('train_losses', results)
        self.assertIn('valid_losses', results)
        
        # Check that we have at least one loss value
        self.assertGreater(len(results['train_losses']), 0)
        self.assertGreater(len(results['valid_losses']), 0)


if __name__ == '__main__':
    unittest.main() 