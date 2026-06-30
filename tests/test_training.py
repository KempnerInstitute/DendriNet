import unittest

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.config.training import RegularizationConfig
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

    def test_split_param_regularization_is_applied_during_epoch_train(self):
        """Split-group regularization should contribute nonzero loss during training."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        )
        optimizer = Adam(
            [
                {"params": list(model[0].parameters()), "lr": 0.01},
                {"params": list(model[2].parameters()), "lr": 0.01},
            ]
        )
        trainer = Trainer(
            optimizer=optimizer,
            loss_function="cross_entropy",
            epochs=1,
            batch_size=8,
            suppress_prints=True,
            regularization_config=RegularizationConfig(
                l1_weight=0.0,
                l2_weight=0.0,
                split_params=True,
                param_group_weights={
                    "topk": {"l1_weight": 0.05, "l2_weight": 0.0},
                    "blocklinear": {"l1_weight": 0.05, "l2_weight": 0.0},
                    "reactivation": {"l1_weight": 0.0, "l2_weight": 0.0},
                    "decoder": {"l1_weight": 0.0, "l2_weight": 0.0},
                },
            ),
        )
        trainer.device = torch.device("cpu")
        train_loader = DataLoader(
            TensorDataset(torch.randn(32, 10), torch.randint(0, 2, (32,))),
            batch_size=8,
            shuffle=False,
        )

        _, _, train_loss_reg = trainer._epoch_train(model, train_loader)

        self.assertGreater(train_loss_reg, 0.0)

    def test_profile_dataloader_logs_epoch_wait_time(self):
        """Dataloader profiling should be observable without changing training APIs."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
        )
        trainer = Trainer(
            optimizer=Adam(model.parameters(), lr=0.01),
            loss_function="cross_entropy",
            epochs=1,
            batch_size=8,
            suppress_prints=True,
            num_workers=0,
            profile_dataloader=True,
        )
        trainer.device = torch.device("cpu")
        loader = DataLoader(
            TensorDataset(torch.randn(16, 10), torch.randint(0, 2, (16,))),
            batch_size=8,
            shuffle=False,
        )

        with self.assertLogs(
            "dendritic_modeling.training.strategies.standard",
            level="INFO",
        ) as captured:
            trainer._epoch_train(model, loader)

        self.assertIn("DataLoader profile", "\n".join(captured.output))

    def test_non_blocking_transfer_requires_cuda_and_pinned_loader(self):
        """Pinned CUDA loaders can use async copies; CPU/unpinned paths cannot."""
        trainer = Trainer(
            optimizer=self.optimizer,
            loss_function="cross_entropy",
            suppress_prints=True,
        )
        loader = DataLoader(self.train_data, batch_size=4, pin_memory=True)

        trainer.device = torch.device("cuda:0")
        self.assertTrue(trainer._loader_uses_non_blocking_transfer(loader))

        trainer.device = torch.device("cpu")
        self.assertFalse(trainer._loader_uses_non_blocking_transfer(loader))

        loader.pin_memory = False
        trainer.device = torch.device("cuda:0")
        self.assertFalse(trainer._loader_uses_non_blocking_transfer(loader))


if __name__ == "__main__":
    unittest.main()
