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
    GradientScaler
)
from dendritic_modeling.training.strategies import (
    TrainerMLE,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
    LocalCreditAssignment,
    TrainOnlyMReactivation
)
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.models import ProbabilisticClassifier


class TestTrainerFactory(unittest.TestCase):
    """Test the trainer factory function."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple optimizer for testing
        model = ProbabilisticClassifier(
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
        self.optimizer = torch.optim.Adam(model.parameters())
    
    def test_get_trainer_mle(self):
        """Test getting MLE trainer."""
        trainer = get_trainer("mle", self.optimizer)
        self.assertIsInstance(trainer, TrainerMLE)
    
    def test_get_trainer_two_step(self):
        """Test getting two-step trainer."""
        trainer = get_trainer("two_step", self.optimizer)
        self.assertIsInstance(trainer, TwoStepTrainer)
    
    def test_get_trainer_two_step_kl(self):
        """Test getting two-step KL trainer."""
        trainer = get_trainer("two_step_kl", self.optimizer)
        self.assertIsInstance(trainer, TwoStepTrainerWithKL)
    
    def test_get_trainer_voltage_stabilization(self):
        """Test getting voltage stabilization trainer."""
        trainer = get_trainer("voltage_stabilization", self.optimizer)
        self.assertIsInstance(trainer, VoltageStabilizationTrainer)
    
    def test_get_trainer_local_credit_assignment(self):
        """Test getting local credit assignment trainer."""
        trainer = get_trainer("local_credit_assignment", self.optimizer)
        self.assertIsInstance(trainer, LocalCreditAssignment)
    
    def test_get_trainer_invalid_strategy(self):
        """Test handling of invalid training strategy."""
        # Should fall back to MLE trainer and log warning
        trainer = get_trainer("invalid_strategy", self.optimizer)
        self.assertIsInstance(trainer, TrainerMLE)
    
    def test_get_trainer_with_options(self):
        """Test getting trainer with additional options."""
        trainer = get_trainer(
            "mle", 
            self.optimizer, 
            suppress_prints=True, 
            print_every=20
        )
        self.assertIsInstance(trainer, TrainerMLE)
        self.assertTrue(trainer.suppress_prints)
        self.assertEqual(trainer.print_every, 20)


class TestTrainerMLE(unittest.TestCase):
    """Test the MLE trainer class."""
    
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
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Create trainer
        self.trainer = TrainerMLE(
            optimizer=self.optimizer,
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
        self.assertIsInstance(self.trainer, TrainerMLE)
        self.assertEqual(self.trainer.optimizer, self.optimizer)
        self.assertTrue(self.trainer.suppress_prints)
        self.assertEqual(self.trainer.print_every, 10)
    
    def test_single_epoch_training(self):
        """Test single epoch of training."""
        initial_loss = self.trainer._compute_loss_on_dataset(self.model, self.train_data)
        
        # Train for one epoch
        results = self.trainer.train(
            model=self.model,
            train_data=self.train_data,
            valid_data=self.valid_data,
            epochs=1,
            batch_size=16,
            plot_losses=False
        )
        
        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn('train_losses', results)
        self.assertIn('valid_losses', results)
        
        # Check that training happened (loss should change)
        final_loss = self.trainer._compute_loss_on_dataset(self.model, self.train_data)
        # Note: Loss might increase or decrease, we just check it changed
        self.assertNotAlmostEqual(initial_loss, final_loss, places=6)


if __name__ == '__main__':
    unittest.main() 