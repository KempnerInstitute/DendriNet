import logging
import os
import tempfile
import unittest

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset

from dendritic_modeling import logger
from dendritic_modeling.learning_strategies import (
    BaseTrainer, LocalCreditAssignment, TrainerMLE, TrainOnlyMReactivation,
    TwoStepTrainer, TwoStepTrainerWithKL, VoltageStabilizationTrainer,
    get_trainer)
from dendritic_modeling.models import (Classifier, ProbabilisticClassifier,
                                       SimpleClassifier)
from dendritic_modeling.networks import (ExcitationInhibitionNetwork,
                                         MLPExcInhNetwork)


class TestTrainers(unittest.TestCase):
    def setUp(self):
        # Create a simple dataset for testing
        self.n_samples = 100
        self.input_dim = 10
        self.output_dim = 5

        # Generate random data
        X = torch.randn(self.n_samples, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.n_samples,))
        self.dataset = TensorDataset(X, y)

        # Split into train/valid
        train_size = int(0.8 * self.n_samples)
        valid_size = self.n_samples - train_size
        self.train_data, self.valid_data = torch.utils.data.random_split(
            self.dataset, [train_size, valid_size]
        )

        # Create a simple model
        self.net = MLPExcInhNetwork(
            input_dim=self.input_dim,
            hidden_dims=[20],
            output_dim=self.output_dim,
        )
        self.model = Classifier(self.net)

        # Create optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

        # Create a dendritic model for warmup tests
        self.dendritic_net = ExcitationInhibitionNetwork(
            input_dim=self.input_dim,
            excitatory_layer_sizes=[20],
            inhibitory_layer_sizes=[10],
            excitatory_branch_factors=[3],
            inhibitory_branch_factors=[],
            output_dim=self.output_dim,
            enable_branch_outputs=True,
        )
        self.dendritic_model = Classifier(self.dendritic_net)
        self.dendritic_optimizer = Adam(self.dendritic_model.parameters(), lr=0.01)

        # Create a temporary directory for saving models
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_trainer_mle_init(self):
        """Test TrainerMLE initialization"""
        trainer = TrainerMLE(self.optimizer)
        self.assertIsInstance(trainer, TrainerMLE)
        self.assertIsInstance(trainer, BaseTrainer)
        self.assertEqual(trainer.optimizer, self.optimizer)

    def test_trainer_mle_train(self):
        """Test TrainerMLE training for a few epochs"""
        trainer = TrainerMLE(self.optimizer, suppress_prints=True)

        # Train for just a few epochs to test functionality
        results = trainer.train(
            model=self.model,
            train_data=self.train_data,
            valid_data=self.valid_data,
            epochs=2,
            batch_size=32,
            grad_clip_value=1.0,
            plot_losses=False,
            save_path=self.temp_dir,
        )

        # Check results structure
        self.assertIn("best epoch", results)
        self.assertIn("best loss", results)
        self.assertIn("best state dict", results)
        self.assertIn("train losses", results)
        self.assertIn("valid losses", results)

        # Check that losses were recorded
        self.assertEqual(len(results["train losses"]), 2)
        self.assertEqual(len(results["valid losses"]), 2)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "best_model.pt")))

    def test_eval_dataset(self):
        """Test the _eval_dataset method"""
        trainer = TrainerMLE(self.optimizer)

        # Test evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        loss, acc = trainer._eval_dataset(
            model=self.model, dataset=self.valid_data, device=device, batch_size=32
        )

        # Check that loss and accuracy are valid values
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_voltage_stabilization_trainer(self):
        """Test VoltageStabilizationTrainer initialization and training"""
        trainer = VoltageStabilizationTrainer(
            optimizer=self.dendritic_optimizer,
            suppress_prints=True,
        )

        self.assertIsInstance(trainer, VoltageStabilizationTrainer)
        self.assertIsInstance(trainer, BaseTrainer)

        train_losses, valid_losses = trainer.train(
            model=self.dendritic_model,
            train_data=self.train_data,
            valid_data=self.valid_data,
            epochs=2,
            batch_size=32,
            plot_losses=False,
            save_path=os.path.join(self.temp_dir, "vs_best_model.pt"),
        )

        self.assertIsInstance(train_losses, list)
        self.assertIsInstance(valid_losses, list)

        self.assertEqual(len(train_losses), 2)
        self.assertEqual(len(valid_losses), 2)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "vs_best_model.pt")))

    def test_get_trainer(self):
        """Test the get_trainer factory function for all strategies"""
        strategies = {
            "mle": TrainerMLE,
            "standard": TrainerMLE,
            "freeze_layers": TrainerMLE,
            "local_credit_assignment": LocalCreditAssignment,
            "train_only_m_reactivation": TrainOnlyMReactivation,
            "two_step": TwoStepTrainer,
            "two_step_kl": TwoStepTrainerWithKL,
            "voltage_stabilization": VoltageStabilizationTrainer,
        }

        for name, expected_type in strategies.items():
            trainer = get_trainer(strategy=name, optimizer=self.optimizer)
            self.assertIsInstance(
                trainer, expected_type, f"Failed for strategy '{name}'"
            )

        with self.assertLogs(logger, level="WARNING") as log:
            fallback_trainer = get_trainer(
                strategy="invalid_strategy", optimizer=self.optimizer
            )
            self.assertIsInstance(fallback_trainer, TrainerMLE)
            self.assertTrue(
                any(
                    "Unknown strategy 'invalid_strategy'" in message
                    for message in log.output
                )
            )


if __name__ == "__main__":
    unittest.main()
