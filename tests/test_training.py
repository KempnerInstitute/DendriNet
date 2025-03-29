# import unittest
# import torch
# from torch.utils.data import TensorDataset
# from torch.optim import Adam

# from dendritic_modeling.training import TrainerMLE

import unittest

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset

from dendritic_modeling.learning_strategies import BaseTrainer, TrainerMLE
from dendritic_modeling.models import Classifier, SimpleClassifier
from dendritic_modeling.networks import MLPExcInhNetwork


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
            activation="relu",
        )
        self.model = Classifier(self.net)

        # Create optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

    def test_trainer_mle_init(self):
        """Test TrainerMLE initialization"""
        trainer = TrainerMLE(self.optimizer)
        self.assertIsInstance(trainer, TrainerMLE)
        self.assertIsInstance(trainer, BaseTrainer)
        self.assertEqual(trainer.optimizer, self.optimizer)

    def test_trainer_mle_train(self):
        """Test TrainerMLE training for a few epochs"""
        trainer = TrainerMLE(self.optimizer, print_every=5)

        # Train for just a few epochs to test functionality
        results = trainer.train(
            model=self.model,
            train_data=self.train_data,
            valid_data=self.valid_data,
            epochs=2,
            batch_size=32,
            grad_clip_value=1.0,
            plot_losses=False,
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


if __name__ == "__main__":
    unittest.main()
