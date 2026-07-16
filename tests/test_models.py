import unittest
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture


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
        ee_synapses_per_branch_per_layer: Optional[list] = None,
        ei_synapses_per_branch_per_layer: Optional[list] = None,
        ie_synapses_per_branch_per_layer: Optional[list] = None,
        ii_synapses_per_branch_per_layer: Optional[list] = None,
        reactivate: bool = True,
        somatic_synapses: bool = True,
        **kwargs,
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


def test_classifier_init():
    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 5}
    )
    decoder = Identity(5)
    model = Classifier(encoder, core, decoder)
    assert model is not None
    assert isinstance(model, Classifier)


def test_classifier_forward():
    model = Classifier(
        Identity(10),
        get_architecture(
            "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 5}
        ),
        Identity(5),
    )
    x = torch.randn(2, 10)
    output = model(x)
    assert output.shape == (2, 5)


def test_classifier_can_use_literal_identity_logits():
    model = Classifier(
        Identity(5),
        Identity(5),
        Identity(5),
        learned_output_scale=False,
    )
    x = torch.randn(3, 5)

    assert not hasattr(model, "log_output_scale")
    assert torch.equal(model(x), x)


def test_classifier_identity_scale_remains_backward_compatible():
    model = Classifier(Identity(5), Identity(5), Identity(5))

    assert hasattr(model, "log_output_scale")
    assert model.log_output_scale.requires_grad


def test_classifier_fixed_global_scale_preserves_identity_and_argmax():
    model = Classifier(
        Identity(5),
        Identity(5),
        Identity(5),
        learned_output_scale=False,
        fixed_output_scale=10.0,
    )
    x = torch.randn(3, 5)

    assert torch.equal(model(x), 10.0 * x)
    assert torch.equal(model(x).argmax(dim=-1), x.argmax(dim=-1))
    assert not hasattr(model, "log_output_scale")


def test_classifier_rejects_invalid_fixed_global_scale():
    with pytest.raises(ValueError, match="fixed_output_scale"):
        Classifier(
            Identity(5),
            Identity(5),
            Identity(5),
            learned_output_scale=False,
            fixed_output_scale=0.0,
        )


def test_classifier_single_global_temperature_preserves_argmax():
    model = Classifier(
        Identity(5),
        Identity(5),
        Identity(5),
        learned_output_scale=False,
        output_scale_mode="global",
    )
    x = torch.randn(3, 5)

    assert not hasattr(model, "log_output_scale")
    assert model.log_global_output_scale.shape == ()
    assert model.log_global_output_scale.requires_grad
    assert torch.equal(model(x), x)
    with torch.no_grad():
        model.log_global_output_scale.fill_(torch.log(torch.tensor(7.0)))
    assert torch.allclose(model(x), 7.0 * x)
    assert torch.equal(model(x).argmax(dim=-1), x.argmax(dim=-1))


def test_regressor_init():
    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 3}
    )
    decoder = Identity(3)
    model = Regressor(encoder, core, decoder)
    assert model is not None
    assert isinstance(model, Regressor)


def test_regressor_forward():
    model = Regressor(
        Identity(10),
        get_architecture(
            "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 3}
        ),
        Identity(3),
    )
    x = torch.randn(2, 10)
    output = model(x)
    assert output.shape == (2, 3)


def test_regressor_predict():
    model = Regressor(
        Identity(10),
        get_architecture(
            "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 3}
        ),
        Identity(3),
    )
    x = torch.randn(2, 10)
    preds = model.predict(x)
    assert preds.shape == (2, 3)


def test_base_model_non_recurrent_path_unchanged():
    """Non-recurrent forward should be identical to encoder->core->decoder composition."""
    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 5}
    )
    decoder = Identity(5)
    model = BaseModel(encoder, core, decoder)
    assert model._is_recurrent_core is False

    x = torch.randn(4, 10)
    with torch.no_grad():
        expected = decoder(core(encoder(x)))
        actual = model(x)
    assert torch.allclose(actual, expected)
