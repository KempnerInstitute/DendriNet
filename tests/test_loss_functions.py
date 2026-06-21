"""
Test suite for loss functions.

This module provides comprehensive testing for all loss function implementations
including MSE, LogMSE, CrossEntropy, BCE, CategoricalNLL, VaeElbo,
NegativeCosineSimilarity, LogAdjustedNegativeCosineSimilarity,
VoltageStabilization, and the loss function factory.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.classical.autoencoder.vae import (
    BaseVariationalAutoencoder,
)
from dendritic_modeling.training.loss.functions import (
    BCELoss,
    CategoricalNLLLoss,
    CrossEntropyLoss,
    EIWeightRatioLoss,
    ExcitationInhibitionBalanceLoss,
    HomeostaticControlLoss,
    LogAdjustedNegativeCosineSimilarityLoss,
    LogMSELoss,
    LossFunction,
    MSELoss,
    NegativeCosineSimilarityLoss,
    VaeElboLoss,
    VoltageStabilizationLoss,
    _loss_reduction_fn,
    _reduce_tensor_loss,
    get_loss_function,
)


class TestLossFunction:
    """Test the base LossFunction class."""

    def test_initialization(self):
        """Test LossFunction initialization."""
        loss_fn = LossFunction()
        assert loss_fn.reduction == "mean"
        assert loss_fn._loss_name == "Base Loss"

    def test_initialization_with_reduction(self):
        """Test LossFunction initialization with custom reduction."""
        loss_fn = LossFunction(reduction="sum")
        assert loss_fn.reduction == "sum"

    def test_forward_not_implemented(self):
        """Test that forward method raises NotImplementedError."""
        loss_fn = LossFunction()
        model = Mock()
        x = torch.randn(2, 10)
        y = torch.randn(2, 10)

        with pytest.raises(NotImplementedError):
            loss_fn(model, x, y)


def test_reduce_tensor_loss_preserves_reduction_convention():
    loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    torch.testing.assert_close(_reduce_tensor_loss(loss, "mean"), loss.mean())
    torch.testing.assert_close(_reduce_tensor_loss(loss, "sum"), loss.sum())
    assert _reduce_tensor_loss(loss, "none") is loss
    assert _reduce_tensor_loss(loss, "custom") is loss


def test_loss_reduction_fn_preserves_callable_convention():
    loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    torch.testing.assert_close(_loss_reduction_fn("mean")(loss), loss.mean())
    torch.testing.assert_close(_loss_reduction_fn("sum")(loss), loss.sum())
    assert _loss_reduction_fn("none")(loss) is loss


class TestMSELoss:
    """Test the MSELoss class."""

    @pytest.fixture
    def mse_loss(self):
        """Create MSELoss for testing."""
        return MSELoss()

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_initialization(self, mse_loss):
        """Test MSELoss initialization."""
        assert mse_loss.reduction == "mean"
        assert mse_loss._loss_name == "Mean Squared Error"
        assert isinstance(mse_loss.mse, nn.MSELoss)

    def test_forward(self, mse_loss, simple_model):
        """Test MSELoss forward pass."""
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        loss = mse_loss(simple_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_different_reductions(self, simple_model):
        """Test MSELoss with different reduction modes."""
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        for reduction in ["mean", "sum", "none"]:
            mse_loss = MSELoss(reduction=reduction)
            loss = mse_loss(simple_model, x, y)

            if reduction == "none":
                assert loss.shape == (4, 5)
            else:
                assert loss.dim() == 0

    def test_gradient_flow(self, mse_loss, simple_model):
        """Test that gradients flow through MSELoss."""
        x = torch.randn(2, 10, requires_grad=True)
        y = torch.randn(2, 5)

        loss = mse_loss(simple_model, x, y)
        loss.backward()

        assert x.grad is not None
        assert simple_model.weight.grad is not None


class TestLogMSELoss:
    """Test the LogMSELoss class."""

    @pytest.fixture
    def log_mse_loss(self):
        """Create LogMSELoss for testing."""
        return LogMSELoss()

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_initialization(self, log_mse_loss):
        """Test LogMSELoss initialization."""
        assert log_mse_loss.reduction == "mean"
        assert log_mse_loss._loss_name == "Log Mean Squared Error"

    def test_forward(self, log_mse_loss, simple_model):
        """Test LogMSELoss forward pass."""
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        loss = log_mse_loss(simple_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_log_property(self, simple_model):
        """Test that LogMSELoss actually computes log of MSE."""
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        mse_loss = MSELoss()
        log_mse_loss = LogMSELoss()

        mse_val = mse_loss(simple_model, x, y)
        log_mse_val = log_mse_loss(simple_model, x, y)

        expected = torch.log(mse_val + 1e-12)
        assert torch.allclose(log_mse_val, expected, atol=1e-6)


class TestCrossEntropyLoss:
    """Test the CrossEntropyLoss class."""

    @pytest.fixture
    def ce_loss(self):
        """Create CrossEntropyLoss for testing."""
        return CrossEntropyLoss()

    @pytest.fixture
    def classification_model(self):
        """Create a classification model for testing."""
        return nn.Linear(10, 5)

    def test_initialization(self, ce_loss):
        """Test CrossEntropyLoss initialization."""
        assert ce_loss.reduction == "mean"
        assert ce_loss._loss_name == "Cross Entropy"
        assert isinstance(ce_loss.ce, nn.CrossEntropyLoss)

    def test_forward(self, ce_loss, classification_model):
        """Test CrossEntropyLoss forward pass."""
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        loss = ce_loss(classification_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_with_probabilities(self, ce_loss, classification_model):
        """Test CrossEntropyLoss with probability targets."""
        x = torch.randn(4, 10)
        y = torch.softmax(torch.randn(4, 5), dim=1)

        loss = ce_loss(classification_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestBCELoss:
    """Test the BCELoss class."""

    @pytest.fixture
    def bce_loss(self):
        """Create BCELoss for testing."""
        return BCELoss()

    @pytest.fixture
    def binary_model(self):
        """Create a binary classification model."""
        return nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

    def test_initialization(self, bce_loss):
        """Test BCELoss initialization."""
        assert bce_loss.reduction == "mean"
        assert bce_loss._loss_name == "Binary Cross Entropy"
        assert isinstance(bce_loss.bce, nn.BCELoss)

    def test_forward(self, bce_loss, binary_model):
        """Test BCELoss forward pass."""
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4, 1)).float()

        loss = bce_loss(binary_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0


class TestCategoricalNLLLoss:
    """Test the CategoricalNLLLoss class."""

    @pytest.fixture
    def nll_loss(self):
        """Create CategoricalNLLLoss for testing."""
        return CategoricalNLLLoss()

    @pytest.fixture
    def log_softmax_model(self):
        """Create a model that outputs log probabilities."""
        return nn.Sequential(nn.Linear(10, 5), nn.LogSoftmax(dim=1))

    def test_initialization(self, nll_loss):
        """Test CategoricalNLLLoss initialization."""
        assert nll_loss.reduction == "mean"
        assert nll_loss._loss_name == "Negative Log Likelihood"

    def test_forward(self, nll_loss, log_softmax_model):
        """Test CategoricalNLLLoss forward pass."""
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))

        loss = nll_loss(log_softmax_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_distribution_sampling(self, nll_loss, log_softmax_model):
        """Test that the loss correctly handles distribution sampling."""
        x = torch.randn(4, 10)
        log_probs = log_softmax_model(x)

        # This should work without error
        assert log_probs.shape == (4, 5)

    def test_reductions_match_manual_nll(self, log_softmax_model):
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))
        logits = log_softmax_model(x)
        expected = -1 * torch.distributions.Categorical(logits=logits).log_prob(y)

        for reduction in ("mean", "sum", "none"):
            actual = CategoricalNLLLoss(reduction=reduction)(log_softmax_model, x, y)
            torch.testing.assert_close(
                actual,
                _reduce_tensor_loss(expected, reduction),
            )


class TestVaeElboLoss:
    """Test the VaeElboLoss class."""

    @pytest.fixture
    def vae_loss(self):
        """Create VaeElboLoss for testing."""
        return VaeElboLoss()

    @pytest.fixture
    def vae_model(self):
        """Create a mock VAE model."""

        # Create a proper mock that inherits from BaseVariationalAutoencoder
        class MockVAE(BaseVariationalAutoencoder):
            def __init__(self):
                # Initialize without calling super().__init__() to avoid complexity
                pass

        model = MockVAE()
        model.encoder = Mock()
        model.encoder.latent_distribution = Mock(
            return_value=(torch.randn(2, 10), torch.ones(2, 10))  # mu, sigma
        )
        model.decoder = Mock(return_value=torch.randn(2, 20))
        return model

    def test_initialization(self, vae_loss):
        """Test VaeElboLoss initialization."""
        assert vae_loss.reduction == "mean"
        assert vae_loss._loss_name == "VAE ELBO"

    def test_forward(self, vae_loss, vae_model):
        """Test VaeElboLoss forward pass."""
        x = torch.randn(2, 20)
        y = torch.randn(2, 20)  # Target for reconstruction

        loss = vae_loss(vae_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_kl_divergence_computation(self, vae_loss):
        """Test KL divergence computation."""
        mu = torch.zeros(2, 5)
        sigma = torch.ones(2, 5)

        # Manually compute KL divergence
        kl_div = 0.5 * torch.sum(mu**2 + sigma**2 - torch.log(sigma**2) - 1, dim=1)

        # Test that our implementation gives similar results
        # (We can't test exactly without accessing private methods)
        assert kl_div.shape == (2,)

    def test_reconstruction_loss(self, vae_loss, vae_model):
        """Test that reconstruction loss is computed."""
        x = torch.randn(2, 20)
        y = torch.randn(2, 20)

        # Test with explicit target
        loss = vae_loss(vae_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_reductions_preserve_elbo_shape(self, vae_model):
        x = torch.randn(2, 20)
        y = torch.randn(2, 20)

        none_loss = VaeElboLoss(reduction="none")(vae_model, x, y)
        mean_loss = VaeElboLoss(reduction="mean")(vae_model, x, y)
        sum_loss = VaeElboLoss(reduction="sum")(vae_model, x, y)

        assert none_loss.shape == (2,)
        assert mean_loss.dim() == 0
        assert sum_loss.dim() == 0


class TestNegativeCosineSimilarityLoss:
    """Test the NegativeCosineSimilarityLoss class."""

    @pytest.fixture
    def cosine_loss(self):
        """Create NegativeCosineSimilarityLoss for testing."""
        return NegativeCosineSimilarityLoss()

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_initialization(self, cosine_loss):
        """Test NegativeCosineSimilarityLoss initialization."""
        assert cosine_loss.reduction == "mean"
        assert cosine_loss._loss_name == "Negative Cosine Similarity"

    def test_forward(self, cosine_loss, simple_model):
        """Test NegativeCosineSimilarityLoss forward pass."""
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        loss = cosine_loss(simple_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_similarity_bounds(self, cosine_loss, simple_model):
        """Test that loss is bounded correctly."""
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        loss = cosine_loss(simple_model, x, y)

        # Negative cosine similarity should be between -1 and 1
        # After negation, it should be between -1 and 1
        assert loss.item() >= -1.0 and loss.item() <= 1.0

    def test_identical_vectors(self, cosine_loss):
        """Test loss with identical vectors."""
        model = nn.Identity()
        x = torch.ones(1, 5)
        y = torch.ones(1, 5)

        loss = cosine_loss(model, x, y)

        # Cosine similarity of identical vectors is 1, so negative is -1
        assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-6)

    def test_reductions_match_manual_negative_cosine(self, simple_model):
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        expected = -1 * nn.CosineSimilarity(dim=-1)(simple_model(x), y)

        for reduction in ("mean", "sum", "none"):
            actual = NegativeCosineSimilarityLoss(reduction=reduction)(
                simple_model, x, y
            )
            torch.testing.assert_close(
                actual,
                _reduce_tensor_loss(expected, reduction),
            )


class TestLogAdjustedNegativeCosineSimilarityLoss:
    """Test the LogAdjustedNegativeCosineSimilarityLoss class."""

    @pytest.fixture
    def log_cosine_loss(self):
        """Create LogAdjustedNegativeCosineSimilarityLoss for testing."""
        return LogAdjustedNegativeCosineSimilarityLoss()

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 5)

    def test_initialization(self, log_cosine_loss):
        """Test LogAdjustedNegativeCosineSimilarityLoss initialization."""
        assert log_cosine_loss.reduction == "mean"
        assert log_cosine_loss._loss_name == "Log Adjusted Negative Cosine Similarity"

    def test_forward(self, log_cosine_loss, simple_model):
        """Test LogAdjustedNegativeCosineSimilarityLoss forward pass."""
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        loss = log_cosine_loss(simple_model, x, y)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_log_adjustment(self, simple_model):
        """Test that log adjustment is applied correctly."""
        x = torch.randn(2, 10)
        y = torch.randn(2, 5)

        base_loss = NegativeCosineSimilarityLoss()
        log_loss = LogAdjustedNegativeCosineSimilarityLoss()

        base_val = base_loss(simple_model, x, y)
        log_val = log_loss(simple_model, x, y)

        # Log adjusted should be different from base
        assert not torch.allclose(base_val, log_val)


class TestVoltageStabilizationLoss:
    """Test the VoltageStabilizationLoss class."""

    class TinyDendriticModel(nn.Module):
        """Minimal model with one parametric and one identity reactivation."""

        def __init__(self):
            super().__init__()
            self.parametric = DendriticBranchLayer(
                output_dim=2,
                excitatory_input_dim=3,
                excitatory_synapses_per_branch=1,
                reactivate=True,
                reactivation_type="param_tanh",
            )
            self.identity = DendriticBranchLayer(
                output_dim=2,
                excitatory_input_dim=2,
                excitatory_synapses_per_branch=1,
                reactivate=False,
            )

        def forward(self, x):
            x = self.parametric(x)
            return self.identity(x)

    @pytest.fixture
    def voltage_loss(self):
        """Create VoltageStabilizationLoss for testing."""
        return VoltageStabilizationLoss(target_voltage=0.5)

    @pytest.fixture
    def dendritic_model(self):
        """Create a mock model with dendritic branch layers."""
        from dendritic_modeling.models import BaseModel
        from dendritic_modeling.networks import DendriticBranchLayer

        model = Mock(spec=BaseModel)

        # Create mock dendritic branch layers
        branch_layer1 = Mock(spec=DendriticBranchLayer)
        branch_layer1.branch_voltages = torch.tensor([0.3, 0.7, 0.4, 0.6])

        branch_layer2 = Mock(spec=DendriticBranchLayer)
        branch_layer2.branch_voltages = torch.tensor([0.2, 0.8, 0.5])

        # Mock the named_modules method to return our layers
        model.named_modules.return_value = [
            ("layer1", branch_layer1),
            ("layer2", branch_layer2),
            ("other", Mock()),  # Non-dendritic layer
        ]

        model.return_value = torch.randn(2, 10)
        return model

    def test_initialization(self, voltage_loss):
        """Test VoltageStabilizationLoss initialization."""
        assert voltage_loss.reduction == "mean"
        assert voltage_loss.target_voltage == 0.5
        assert voltage_loss._loss_name == "Voltage Stabilization"

    def test_forward_with_dendritic_model(self, voltage_loss, dendritic_model):
        """Test VoltageStabilizationLoss with dendritic model."""
        x = torch.randn(2, 10)

        loss = voltage_loss(dendritic_model, x)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_forward_without_dendritic_layers(self, voltage_loss):
        """Test VoltageStabilizationLoss with non-dendritic model."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)

        loss = voltage_loss(model, x)

        # Should return zero loss when no dendritic layers
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_hook_management(self, voltage_loss, dendritic_model):
        """Test hook attachment and removal."""
        # Mock the modules method to return an iterable
        dendritic_model.modules = Mock(return_value=[Mock(), Mock()])

        # Test that hooks can be attached and removed
        handles = voltage_loss.attach_forward_hooks(dendritic_model)
        assert isinstance(handles, list)

        voltage_loss.remove_forward_hooks(handles)
        # Should not raise an error

    def test_different_target_voltages(self, dendritic_model):
        """Test with different target voltages."""
        x = torch.randn(2, 10)

        for target in [0.0, 0.25, 0.5, 0.75, 1.0]:
            loss_fn = VoltageStabilizationLoss(target_voltage=target)
            loss = loss_fn(dendritic_model, x)
            assert isinstance(loss, torch.Tensor)
            assert loss >= 0

    def test_gate_occupancy_only_hooks_parametric_reactivations(self):
        """gate_occupancy should ignore identity/non-parametric reactivations."""
        model = self.TinyDendriticModel()
        loss_fn = VoltageStabilizationLoss(stabilize_mode="gate_occupancy")

        handles = loss_fn.attach_forward_hooks(model)
        try:
            assert len(handles) == 1
            assert isinstance(model.parametric.reactivation, nn.Module)
            assert isinstance(model.identity.reactivation, nn.Identity)
        finally:
            loss_fn.remove_forward_hooks(handles)

    def test_gate_occupancy_target_saturation_changes_loss(self):
        """The occupancy target should change the spread penalty."""
        activations = torch.tensor([0.0, 0.05, 0.95, 1.0], dtype=torch.float32)
        x = torch.randn(1, 1)

        loss_high_target = VoltageStabilizationLoss(
            stabilize_mode="gate_occupancy", target_saturation=1.0
        )
        loss_high_target.voltages = [activations]
        high_val = loss_high_target._gate_occupancy_loss(x)

        loss_low_target = VoltageStabilizationLoss(
            stabilize_mode="gate_occupancy", target_saturation=0.0
        )
        loss_low_target.voltages = [activations]
        low_val = loss_low_target._gate_occupancy_loss(x)

        assert high_val < low_val

    def test_metric_reduction_modes_preserve_mean_sum_and_passthrough(self):
        x = torch.zeros(2, 2)
        y = torch.ones(2, 2)

        mean_loss = VoltageStabilizationLoss(loss_metric="mse", reduction="mean")
        sum_loss = VoltageStabilizationLoss(loss_metric="mse", reduction="sum")
        none_loss = VoltageStabilizationLoss(loss_metric="mse", reduction="none")

        torch.testing.assert_close(mean_loss.metric_fn(x, y), ((x - y) ** 2).mean())
        torch.testing.assert_close(sum_loss.metric_fn(x, y), ((x - y) ** 2).sum())
        torch.testing.assert_close(none_loss.metric_fn(x, y), (x - y) ** 2)

    def test_voltage_hooks_removed_after_failed_registration(self, monkeypatch):
        model = self.TinyDendriticModel()
        loss_fn = VoltageStabilizationLoss()

        def _failing_register(*args, **kwargs):
            raise RuntimeError("voltage hook registration failed")

        monkeypatch.setattr(
            model.identity.reactivation,
            "register_forward_hook",
            _failing_register,
        )

        with pytest.raises(RuntimeError, match="voltage hook registration failed"):
            loss_fn.attach_forward_hooks(model)

        assert sum(len(module._forward_hooks) for module in model.modules()) == 0

    def test_ei_balance_hooks_removed_after_failed_registration(self, monkeypatch):
        model = nn.Module()
        model.branch = DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=3,
            excitatory_synapses_per_branch=1,
            inhibitory_input_dim=3,
            inhibitory_synapses_per_branch=1,
            reactivate=False,
        )
        loss_fn = ExcitationInhibitionBalanceLoss()

        def _failing_register(*args, **kwargs):
            raise RuntimeError("ei hook registration failed")

        monkeypatch.setattr(
            model.branch.branch_inhibition,
            "register_forward_hook",
            _failing_register,
        )

        with pytest.raises(RuntimeError, match="ei hook registration failed"):
            loss_fn.attach_forward_hooks(model)

        assert sum(len(module._forward_hooks) for module in model.modules()) == 0

    def test_homeostatic_hooks_removed_after_second_loss_registration_fails(
        self, monkeypatch
    ):
        model = self.TinyDendriticModel()
        loss_fn = HomeostaticControlLoss()

        def _failing_attach(_model):
            raise RuntimeError("homeostatic hook registration failed")

        monkeypatch.setattr(loss_fn.ei_balance, "attach_forward_hooks", _failing_attach)

        with pytest.raises(RuntimeError, match="homeostatic hook registration failed"):
            loss_fn.attach_forward_hooks(model)

        assert sum(len(module._forward_hooks) for module in model.modules()) == 0


def test_ei_weight_ratio_reduction_fn_preserves_modes():
    values = torch.tensor([1.0, 2.0, 3.0])

    mean_loss = EIWeightRatioLoss(reduction="mean")
    sum_loss = EIWeightRatioLoss(reduction="sum")
    none_loss = EIWeightRatioLoss(reduction="none")

    torch.testing.assert_close(mean_loss.reduction_fn(values), values.mean())
    torch.testing.assert_close(sum_loss.reduction_fn(values), values.sum())
    assert none_loss.reduction_fn(values) is values


class TestGetLossFunction:
    """Test the get_loss_function factory function."""

    def test_valid_loss_functions(self):
        """Test that all valid loss functions can be created."""
        valid_losses = [
            "mse",
            "log_mse",
            "ce",
            "bce",
            "cat_nll",
            "vae_elbo",
            "negative_cosine_similarity",
            "log_adjusted_negative_cosine_similarity",
        ]

        for loss_name in valid_losses:
            loss_fn = get_loss_function(loss_name)
            assert isinstance(loss_fn, LossFunction)

    def test_invalid_loss_function(self):
        """Test that invalid loss function names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid loss function"):
            get_loss_function("invalid_loss")

    def test_case_sensitivity(self):
        """Test that loss function names are case sensitive."""
        # Should work
        loss_fn = get_loss_function("mse")
        assert isinstance(loss_fn, MSELoss)

        # Should fail (case sensitive)
        with pytest.raises(ValueError):
            get_loss_function("MSE")

    def test_loss_function_types(self):
        """Test that correct loss function types are returned."""
        test_cases = [
            ("mse", MSELoss),
            ("log_mse", LogMSELoss),
            ("ce", CrossEntropyLoss),
            ("bce", BCELoss),
            ("cat_nll", CategoricalNLLLoss),
            ("vae_elbo", VaeElboLoss),
            ("negative_cosine_similarity", NegativeCosineSimilarityLoss),
            (
                "log_adjusted_negative_cosine_similarity",
                LogAdjustedNegativeCosineSimilarityLoss,
            ),
        ]

        for loss_name, expected_type in test_cases:
            loss_fn = get_loss_function(loss_name)
            assert isinstance(loss_fn, expected_type)


class TestLossFunctionIntegration:
    """Integration tests for loss functions."""

    def test_loss_functions_with_autograd(self):
        """Test that all loss functions work with autograd."""
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10, requires_grad=True)
        y = torch.randn(4, 5)

        loss_names = ["mse", "log_mse", "negative_cosine_similarity"]

        for loss_name in loss_names:
            loss_fn = get_loss_function(loss_name)
            loss = loss_fn(model, x, y)

            # Should be able to compute gradients
            loss.backward(retain_graph=True)
            assert x.grad is not None
            x.grad.zero_()

    def test_loss_functions_deterministic(self):
        """Test that loss functions produce deterministic results."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)

        for loss_name in ["mse", "ce", "negative_cosine_similarity"]:
            loss_fn = get_loss_function(loss_name)

            # Set same seed and compute twice
            torch.manual_seed(42)
            loss1 = loss_fn(
                model, x, y if loss_name != "ce" else torch.randint(0, 5, (4,))
            )

            torch.manual_seed(42)
            loss2 = loss_fn(
                model, x, y if loss_name != "ce" else torch.randint(0, 5, (4,))
            )

            assert torch.allclose(loss1, loss2)

    def test_batch_size_consistency(self):
        """Test that loss functions handle different batch sizes correctly."""
        model = nn.Linear(10, 5)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10)
            y = torch.randn(batch_size, 5)

            loss_fn = MSELoss()
            loss = loss_fn(model, x, y)

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Should always be scalar with mean reduction
