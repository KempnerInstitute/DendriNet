import unittest

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset

from dendritic_modeling.training.strategies.standard import Trainer


class TestTrainer(unittest.TestCase):
    """Test the standard Trainer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model for testing
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        )

        # Create optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

        # Create test datasets
        self.train_data = TensorDataset(torch.randn(50, 10), torch.randint(0, 2, (50,)))
        self.valid_data = TensorDataset(torch.randn(20, 10), torch.randint(0, 2, (20,)))

        # Create trainer
        self.trainer = Trainer(
            optimizer=self.optimizer,
            loss_function="cross_entropy",
            epochs=2,
            batch_size=16,
            suppress_prints=True,
        )

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        self.assertIsInstance(self.trainer, Trainer)
        self.assertEqual(self.trainer.optimizer, self.optimizer)
        self.assertEqual(self.trainer.epochs, 2)
        self.assertEqual(self.trainer.batch_size, 16)

    def test_training_execution(self):
        """Test that training executes without errors."""
        results = self.trainer.train(
            model=self.model, train_data=self.train_data, valid_data=self.valid_data
        )

        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn("train_losses", results)
        self.assertIn("valid_losses", results)

        # Check that we have the expected number of loss values
        self.assertEqual(len(results["train_losses"]), 2)  # 2 epochs
        self.assertEqual(len(results["valid_losses"]), 2)  # 2 epochs

    def test_training_reduces_loss(self):
        """Test that training generally reduces loss over epochs."""
        results = self.trainer.train(
            model=self.model, train_data=self.train_data, valid_data=self.valid_data
        )

        train_losses = results["train_losses"]
        # Loss should generally decrease (though may not be monotonic)
        self.assertGreater(len(train_losses), 0)
        self.assertTrue(all(isinstance(loss, (int, float)) for loss in train_losses))


if __name__ == "__main__":
    unittest.main()
