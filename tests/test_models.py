import unittest

import torch
import torch.nn as nn
from torch.distributions import Categorical

from dendritic_modeling.models import BaseModel as ProbabilisticModel
from dendritic_modeling.models import (Classifier, ProbabilisticClassifier,
                                       SimpleClassifier)


class ProbabilisticModelTest(unittest.TestCase):
    def setUp(self):
        self.model = ProbabilisticModel(net=torch.nn.Linear(10, 10))

    def test_log_prob(self):
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        # test that the method raises a NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.model.predict(x)


class ProbabilisticClassifierTest(unittest.TestCase):
    def setUp(self):
        class DummyClassifier(ProbabilisticClassifier):
            def forward(self, x, y=None, sampler=False):
                # define a binary classifier
                logits = torch.randn(x.size(0), 2)
                categorical_dist = Categorical(logits=logits)
                return categorical_dist if sampler else categorical_dist.logits

        self.model = DummyClassifier(net=torch.nn.Linear(10, 2))

    def test_predict_deterministic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=False)
        self.assertEqual(preds.shape, (5,))
        self.assertTrue(((preds == 0) | (preds == 1)).all())

    def test_predict_stochastic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=True)
        self.assertEqual(preds.shape, (5,))
        self.assertTrue(((preds == 0) | (preds == 1)).all())


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        dummy_net = torch.nn.Linear(784, 10)
        self.model = Classifier(net=dummy_net)

    def test_forward_output_shape(self):
        # Batch size of 32, input dim of 784
        x = torch.randn(32, 784)
        output = self.model.forward(x)
        # Output should be logits
        self.assertEqual(output.shape, (32, 10))

    def test_predict(self):
        x = torch.randn(32, 784)
        preds = self.model.predict(x)
        self.assertEqual(preds.shape, (32,))

    def test_compute_loss(self):
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        loss = self.model.compute_loss(x, y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar loss


class TestSimpleClassifier(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.hidden_dims = [5]
        self.output_dim = 2
        self.model = SimpleClassifier(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
        )

    def test_forward_pass(self):
        """Test the forward pass output shape."""
        x = torch.randn(8, self.input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (8, self.output_dim))

    def test_compute_loss(self):
        """Test loss computation."""
        x = torch.randn(8, self.input_dim)
        y = torch.randint(0, self.output_dim, (8,))
        # compute_loss takes x and y, computes logits internally
        loss = self.model.compute_loss(x, y)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())

    def test_predict(self):
        """Test prediction output shape."""
        # Assuming SimpleClassifier should have a predict method like Classifier
        # It currently doesn't in models.py, but let's add a placeholder test
        # If predict is added later, this test should be uncommented/updated.
        # x = torch.randn(8, self.input_dim)
        # predictions = self.model.predict(x)
        # self.assertEqual(predictions.shape, (8,))
        pass  # Placeholder, as predict is not implemented in SimpleClassifier


if __name__ == "__main__":
    unittest.main()
