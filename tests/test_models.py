import unittest

import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn

from dendritic_modeling.models import Classifier
from dendritic_modeling.models.base import BaseModel

# Mock probabilistic classes for testing
class ProbabilisticModel(BaseModel):
    """Mock probabilistic model for testing."""

    def __init__(self):
        super().__init__(nn.Identity(), nn.Identity(), nn.Identity())

    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement log_prob method.")

class ProbabilisticClassifier(ProbabilisticModel):
    """Mock probabilistic classifier for testing."""

    def forward(self, x):
        # Return a categorical distribution
        return Categorical(logits=torch.randn(x.size(0), 2))

    def predict(self, x: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            dist = self.forward(x)
            if stochastic:
                return dist.sample()
            else:
                return dist.logits.argmax(dim=-1)

class ProbabilisticRegressor(ProbabilisticModel):
    """Mock probabilistic regressor for testing."""

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        return Normal(loc=mean, scale=torch.ones_like(mean))

    def predict(self, x: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            dist = self.forward(x)
            if stochastic:
                return dist.sample()
            else:
                return dist.mean

class EINetClassifier(ProbabilisticClassifier):
    """Mock EINet classifier for testing."""

    def __init__(
        self,
        input_dim: int,
        excitatory_layer_sizes: list,
        inhibitory_layer_sizes: list,
        excitatory_branch_factors: list,
        inhibitory_branch_factors: list,
        ee_synapses_per_branch_per_layer: list = None,
        ei_synapses_per_branch_per_layer: list = None,
        ie_synapses_per_branch_per_layer: list = None,
        ii_synapses_per_branch_per_layer: list = None,
        reactivate: bool = True,
        somatic_synapses: bool = True,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = excitatory_layer_sizes[-1] if excitatory_layer_sizes else 10

class ProbabilisticModelTest(unittest.TestCase):
    def setUp(self):
        self.model = ProbabilisticModel()

    def test_log_prob(self):
        x = torch.randn(5, 10)
        y = torch.randint(0, 2, (5,))
        # test that the method raises a NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.model.log_prob(x, y)

class ProbabilisticClassifierTest(unittest.TestCase):
    def setUp(self):
        self.model = ProbabilisticClassifier()

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

class ProbabilisticRegressorTest(unittest.TestCase):
    def setUp(self):
        self.model = ProbabilisticRegressor()

    def test_predict_deterministic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=False)
        self.assertEqual(preds.shape, (5, 1))

    def test_predict_stochastic(self):
        x = torch.randn(5, 10)
        preds = self.model.predict(x, stochastic=True)
        self.assertEqual(preds.shape, (5, 1))

class EINetClassifierTest(unittest.TestCase):
    def setUp(self):
        # Set up the EINetClassifier with example layer sizes
        self.model = EINetClassifier(
            input_dim=784,
            excitatory_layer_sizes=[10],
            inhibitory_layer_sizes=[20],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[1],
            ee_synapses_per_branch_per_layer=[3],
            ei_synapses_per_branch_per_layer=[3],
            ie_synapses_per_branch_per_layer=[2],
            ii_synapses_per_branch_per_layer=[],
            reactivate=True,
            somatic_synapses=True,
        )

    def test_forward_output_shape(self):
        # Batch size of 32, input dim of 784
        x = torch.randn(32, 784)
        output = self.model.forward(x)
        self.assertIsInstance(output, Categorical)
        # Batch size should be preserved
        self.assertEqual(output.logits.shape[0], 32)

    def test_predict_deterministic(self):
        x = torch.randn(32, 784)
        preds = self.model.predict(x, stochastic=False)
        self.assertEqual(preds.shape, (32,))
