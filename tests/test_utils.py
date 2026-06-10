import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from dendritic_modeling.datasets import load_mnist_modulo10

# Import functions from their actual locations
from dendritic_modeling.training.utils.evaluation import accuracy_score
from dendritic_modeling.utils import Shaper
from dendritic_modeling.utils.gpu_estimation import GPUMemoryEstimator
from dendritic_modeling.utils.information.estimators import (
    EstimatorConfig,
    InformationEstimatorFactory,
)


class TestShaper(unittest.TestCase):
    def test_initialization(self):
        shape = (2, 3)
        shaper = Shaper(shape)
        self.assertEqual(shaper.shape, shape)

    def test_reshape_valid_shape(self):
        shaper = Shaper((2, 3))
        x = torch.randn(6)
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 3]))

    def test_reshape_invalid_shape(self):
        shaper = Shaper((2, 4))
        x = torch.randn(6)
        with self.assertRaises(RuntimeError):
            shaper.reshape(x)

    def test_reshape_higher_dimensions(self):
        shaper = Shaper((2, 2, 3))
        x = torch.randn(12)
        reshaped_x = shaper.reshape(x)
        self.assertEqual(reshaped_x.shape, torch.Size([2, 2, 3]))

    def test_reshape_invalid_input(self):
        shaper = Shaper((2, 3))
        x = torch.randn(7)  # Invalid size
        with self.assertRaises(RuntimeError):
            shaper.reshape(x)


class TestAccuracyScore(unittest.TestCase):
    def setUp(self):
        # Create a dummy classifier that always predicts class 1
        class DummyClassifier(nn.Module):
            def forward(self, inputs):
                batch_size = inputs.shape[0]
                logits = torch.zeros(batch_size, 2)
                logits[:, 1] = 10.0  # High logit for class 1
                return logits

        self.classifier = DummyClassifier()

    def test_accuracy_all_correct(self):
        inputs = torch.randn(10, 784)
        labels = torch.ones(10, dtype=torch.long)
        accuracy = accuracy_score(self.classifier, inputs, labels, move_device=False)
        self.assertEqual(accuracy.item(), 1.0)

    def test_accuracy_all_incorrect(self):
        inputs = torch.randn(10, 784)
        labels = torch.zeros(10, dtype=torch.long)
        accuracy = accuracy_score(self.classifier, inputs, labels, move_device=False)
        self.assertEqual(accuracy.item(), 0.0)

    def test_accuracy_half_correct(self):
        inputs = torch.randn(10, 784)
        labels = torch.cat(
            [torch.ones(5, dtype=torch.long), torch.zeros(5, dtype=torch.long)]
        )
        accuracy = accuracy_score(self.classifier, inputs, labels, move_device=False)
        self.assertAlmostEqual(accuracy.item(), 0.5)  # 50% accuracy


class TestLoadMNISTModulo10(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for caching during tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self._mnist_patch = patch(
            "dendritic_modeling.datasets.standard_datasets.load_mnist_as_datasets",
            side_effect=self._mock_mnist_as_datasets,
        )
        self._mnist_patch.start()

    def tearDown(self):
        self._mnist_patch.stop()
        # Clean up the temporary directory after tests
        self.temp_dir.cleanup()

    @staticmethod
    def _make_dataset(n_samples: int, offset: int):
        x = torch.arange(
            offset,
            offset + n_samples * 28 * 28,
            dtype=torch.float32,
        ).view(n_samples, 1, 28, 28)
        y = (torch.arange(n_samples, dtype=torch.long) + offset) % 10
        return torch.utils.data.TensorDataset(x, y)

    def _mock_mnist_as_datasets(
        self,
        train_valid_split=0.8,
        flatten=False,
        normalize=False,
        task_data_path=None,
        split_seed: int = 0,
    ):
        del normalize, task_data_path, split_seed
        total_train = 100
        train_size = int(total_train * train_valid_split)
        valid_size = total_train - train_size
        train = self._make_dataset(train_size, 0)
        valid = self._make_dataset(valid_size, 10_000)
        test = self._make_dataset(40, 20_000)
        if flatten:
            train = torch.utils.data.TensorDataset(
                train.tensors[0].view(train_size, -1), train.tensors[1]
            )
            valid = torch.utils.data.TensorDataset(
                valid.tensors[0].view(valid_size, -1), valid.tensors[1]
            )
            test = torch.utils.data.TensorDataset(
                test.tensors[0].view(40, -1), test.tensors[1]
            )
        return train, valid, test

    def test_data_loading_no_split(self):
        """Test data loading with no train/validation split."""
        data = load_mnist_modulo10(
            train_valid_split=1, task_data_path=self.temp_dir.name
        )
        self.assertIn("train", data)
        self.assertIn("test", data)
        # Check that datasets are not None and have length
        self.assertIsNotNone(data["train"])
        self.assertIsNotNone(data["test"])
        self.assertGreater(len(data["train"]), 0)
        self.assertGreater(len(data["test"]), 0)

    def test_data_loading_with_split(self):
        """Test data loading with train/validation split."""
        data = load_mnist_modulo10(
            train_valid_split=0.8, task_data_path=self.temp_dir.name
        )
        self.assertIn("train", data)
        self.assertIn("valid", data)
        self.assertIn("test", data)
        # Check that datasets are not None
        self.assertIsNotNone(data["train"])
        self.assertIsNotNone(data["valid"])
        self.assertIsNotNone(data["test"])

        # Check that train/validation split ratio is respected approximately
        total_train_valid = len(data["train"]) + len(data["valid"])
        self.assertAlmostEqual(len(data["train"]) / total_train_valid, 0.8, delta=0.05)
        self.assertGreater(len(data["test"]), 0)

    def test_shuffle_iterations_effect(self):
        """Test that different shuffle iterations produce different results."""
        # Load data with 1 shuffle iteration
        data1 = load_mnist_modulo10(
            shuffle_iterations=1,
            train_valid_split=1,
            task_data_path=self.temp_dir.name,
        )
        # Load data with 2 shuffle iterations
        data2 = load_mnist_modulo10(
            shuffle_iterations=2,
            train_valid_split=1,
            task_data_path=self.temp_dir.name,
        )

        # Check that different shuffle iterations yield different input tensors
        # For example, the 10th input tensor should be different
        self.assertFalse(torch.equal(data1["train"][9][0], data2["train"][9][0]))

    def test_modulo10_labels(self):
        """Test that labels are correctly computed with modulo 10."""
        data = load_mnist_modulo10(
            shuffle_iterations=1,
            train_valid_split=1,
            task_data_path=self.temp_dir.name,
        )
        labels = [item[1].numpy().tolist() for item in data["train"]]
        mod_status = [(item >= 0) and (item < 10) for item in labels]
        self.assertTrue(all(mod_status))


def test_mutual_information():
    config = EstimatorConfig(method="kraskov", n_neighbors=3)
    estimator = InformationEstimatorFactory.create(config)
    x = np.random.rand(100)
    y = np.random.rand(100)  # Uncorrelated
    mi = estimator.mutual_information(x.reshape(-1, 1), y.reshape(-1, 1))
    assert (
        isinstance(mi, (int, float)) and mi >= 0
    )  # MI should be non-negative, can be int or float


def test_conditional_mi():
    config = EstimatorConfig(method="kraskov", n_neighbors=3)
    estimator = InformationEstimatorFactory.create(config)
    x = np.random.rand(100)
    y = x + np.random.rand(100) * 0.1
    z = np.random.rand(100)
    cmi = estimator.conditional_mutual_information(
        x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)
    )
    assert isinstance(cmi, float)


def test_gpu_estimator():
    estimator = GPUMemoryEstimator(dtype="fp16", optimizer="adam")
    assert estimator.bytes_per_param == 2


if __name__ == "__main__":
    unittest.main()
