import os
import tempfile
import unittest
from pathlib import Path

import yaml

from dendritic_modeling.config import (Config, ModelConfig, OutputsConfig,
                                       TaskConfig, TrainConfig,
                                       VisualizationConfig, WandbConfig,
                                       load_config)


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for configs
        self.temp_dir = tempfile.mkdtemp()

        # Define a sample configuration
        self.sample_config = {
            "model": {
                "task": "classification",
                "probabilistic": True,
                "network": {
                    "type": "EINet",
                    "parameters": {
                        "input_dim": 784,
                        "excitatory_layer_sizes": [128, 64],
                        "inhibitory_layer_sizes": [64, 32],
                        "excitatory_branch_factors": [2, 2],
                        "inhibitory_branch_factors": [2, 2],
                        "output_dim": 10,
                    },
                },
            },
            "train": {
                "learning_strategy": "mle",
                "run_name": "test_run",
                "seed": 42,
                "lr": 0.001,
                "epochs": 10,
                "batch_size": 64,
            },
            "task": {
                "dataset": "mnist",
                "data_path": "/tmp/data",
                "parameters": {"train_valid_split": 0.8},
            },
            "wandb": {
                "entity": "test_entity",
                "project": "test_project",
                "use_wandb": False,
            },
            "outputs": {"dir": "test_outputs"},
        }

        # Save the config to a temp file
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        with open(self.config_path, "w") as f:
            yaml.dump(self.sample_config, f)

    def tearDown(self):
        # Remove the temporary directory
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_load_config(self):
        """Test loading a configuration from a file."""
        # Load the config
        config = load_config(self.config_path)

        # Check that config is a Config object
        self.assertIsInstance(config, Config)

        # Check model config
        self.assertEqual(config.model.task, "classification")
        self.assertEqual(config.model.network.type, "EINet")
        self.assertEqual(config.model.network.parameters.input_dim, 784)
        self.assertEqual(config.model.network.parameters.output_dim, 10)

        # Check train config
        self.assertEqual(config.train.run_name, "test_run")
        self.assertEqual(config.train.seed, 42)
        self.assertEqual(config.train.batch_size, 64)

        # Check task config
        self.assertEqual(config.task.dataset, "mnist")

        # Check wandb config
        self.assertEqual(config.wandb.entity, "test_entity")
        self.assertEqual(config.wandb.project, "test_project")
        self.assertFalse(config.wandb.use_wandb)

        # Check outputs config
        self.assertEqual(config.outputs.dir, "test_outputs")

    def test_default_config(self):
        """Test loading default config when no file is provided."""
        # Load default config
        config = load_config()

        # Check that config is a Config object with default values
        self.assertIsInstance(config, Config)
        self.assertEqual(config.model.task, "classification")
        self.assertEqual(config.model.network.type, "EINet")
        self.assertEqual(config.train.run_name, "mnist_einet")
        self.assertEqual(config.task.dataset, "mnist")

    def test_load_partial_config(self):
        """Test loading a configuration with only some sections."""
        # Create a partial config
        partial_config = {
            "model": {"task": "segmentation", "probabilistic": False},
            "train": {"lr": 0.005},
        }

        # Save the partial config
        partial_path = os.path.join(self.temp_dir, "partial_config.yaml")
        with open(partial_path, "w") as f:
            yaml.dump(partial_config, f)

        # Load the partial config
        config = load_config(partial_path)

        # Check that the specified values are loaded
        self.assertEqual(config.model.task, "segmentation")
        self.assertFalse(config.model.probabilistic)
        self.assertEqual(config.train.lr, 0.005)

        # Check that unspecified values use defaults
        self.assertEqual(config.train.run_name, "mnist_einet")  # Default value
        self.assertEqual(config.task.dataset, "mnist")  # Default value

    def test_config_dataclasses(self):
        """Test that config dataclasses work as expected."""
        # Create config objects
        model_config = ModelConfig(task="regression", probabilistic=False)
        train_config = TrainConfig(lr=0.01, batch_size=32)
        task_config = TaskConfig(dataset="cifar10")

        # Create a full config
        config = Config(model=model_config, train=train_config, task=task_config)

        # Check values
        self.assertEqual(config.model.task, "regression")
        self.assertFalse(config.model.probabilistic)
        self.assertEqual(config.train.lr, 0.01)
        self.assertEqual(config.train.batch_size, 32)
        self.assertEqual(config.task.dataset, "cifar10")

        # Check that unspecified configs use defaults
        self.assertEqual(config.wandb.use_wandb, False)
        self.assertEqual(config.outputs.dir, "outputs")


if __name__ == "__main__":
    unittest.main()
