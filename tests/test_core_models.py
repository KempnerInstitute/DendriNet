"""
Test suite for core dendritic modeling functionality.

This module tests the core models and networks to ensure they work correctly
with the refactored codebase structure.
"""

import unittest

import torch
import torch.nn as nn

from dendritic_modeling.config import TransferFunctionConfig
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import MLP, ExcitationInhibitionNetwork


class TestCoreModels(unittest.TestCase):
    """Test the core model functionality."""

    def test_classifier_creation(self):
        """Test that we can create a basic classifier."""
        # Create simple MLP network
        core_network = MLP(
            input_dim=10, hidden_dims=[5], output_dim=2, activation="relu"
        )

        classifier = Classifier(
            encoder_network=nn.Identity(),
            core_network=core_network,
            decoder_network=nn.Identity(),
        )

        # Test forward pass
        x = torch.randn(32, 10)
        output = classifier(x)

        self.assertEqual(output.shape, (32, 2))
        self.assertIsInstance(output, torch.Tensor)

    def test_classifier_prediction(self):
        """Test classifier prediction functionality."""
        core_network = MLP(
            input_dim=10, hidden_dims=[5], output_dim=2, activation="relu"
        )

        classifier = Classifier(
            encoder_network=nn.Identity(),
            core_network=core_network,
            decoder_network=nn.Identity(),
        )

        x = torch.randn(32, 10)
        predictions = classifier.predict(x)

        self.assertEqual(predictions.shape, (32,))
        self.assertTrue(torch.all((predictions >= 0) & (predictions < 2)))


class TestEINetworkIntegration(unittest.TestCase):
    """Test EI Network integration with models."""

    def test_ei_network_creation(self):
        """Test creating an EI network."""
        transfer_params = TransferFunctionConfig().asdict()

        ei_network = ExcitationInhibitionNetwork(
            input_dim=8,
            transfer_params=transfer_params,
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[4],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            ee_synapses_per_branch_per_layer=[1],
            ei_synapses_per_branch_per_layer=[1],
            ie_synapses_per_branch_per_layer=[1],
            ii_synapses_per_branch_per_layer=[1],
        )

        self.assertIsInstance(ei_network, ExcitationInhibitionNetwork)

    def test_ei_classifier_integration(self):
        """Test integrating EI network with classifier."""
        transfer_params = TransferFunctionConfig().asdict()

        ei_network = ExcitationInhibitionNetwork(
            input_dim=8,
            transfer_params=transfer_params,
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[4],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            ee_synapses_per_branch_per_layer=[1],
            ei_synapses_per_branch_per_layer=[1],
            ie_synapses_per_branch_per_layer=[1],
            ii_synapses_per_branch_per_layer=[1],
        )

        classifier = Classifier(
            encoder_network=nn.Identity(),
            core_network=ei_network,
            decoder_network=nn.Identity(),
        )

        # Test forward pass
        x = torch.randn(16, 8)
        output = classifier(x)

        # Output should have correct shape
        self.assertEqual(output.shape, (16, 4))  # Last layer size is 4

        # Test prediction
        predictions = classifier.predict(x)
        self.assertEqual(predictions.shape, (16,))


class TestModelParameterGroups(unittest.TestCase):
    """Test model parameter grouping functionality."""

    def test_parameter_groups(self):
        """Test parameter group creation for optimizers."""
        from dendritic_modeling.config import ParamGroupsConfig

        # Create EI network
        transfer_params = TransferFunctionConfig().asdict()
        ei_network = ExcitationInhibitionNetwork(
            input_dim=8,
            transfer_params=transfer_params,
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[4],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            ee_synapses_per_branch_per_layer=[1],
            ei_synapses_per_branch_per_layer=[1],
            ie_synapses_per_branch_per_layer=[1],
            ii_synapses_per_branch_per_layer=[1],
        )

        classifier = Classifier(
            encoder_network=nn.Identity(),
            core_network=ei_network,
            decoder_network=nn.Identity(),
        )

        # Create parameter groups config
        param_config = ParamGroupsConfig(
            split_params=True,
            topk_lr=0.001,
            blocklinear_lr=0.001,
            reactivation_lr=0.001,
            decoder_lr=0.001,
        )

        param_groups = classifier.get_param_groups(param_config)

        # Should get multiple parameter groups when split_params=True
        self.assertGreater(len(param_groups), 0)
        self.assertIsInstance(param_groups, list)

        # Each group should have 'params' and 'lr' keys
        for group in param_groups:
            self.assertIn("params", group)
            self.assertIn("lr", group)


class TestModelSaving(unittest.TestCase):
    """Test model saving and loading functionality."""

    def test_model_state_dict(self):
        """Test model state dict operations."""
        core_network = MLP(
            input_dim=10, hidden_dims=[5], output_dim=2, activation="relu"
        )

        classifier = Classifier(
            encoder_network=nn.Identity(),
            core_network=core_network,
            decoder_network=nn.Identity(),
        )

        # Get state dict
        state_dict = classifier.state_dict()
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)

        # Create new model and load state dict
        new_classifier = Classifier(
            encoder_network=nn.Identity(),
            core_network=MLP(
                input_dim=10, hidden_dims=[5], output_dim=2, activation="relu"
            ),
            decoder_network=nn.Identity(),
        )

        new_classifier.load_state_dict(state_dict)

        # Test that they produce the same output
        x = torch.randn(16, 10)
        with torch.no_grad():
            output1 = classifier(x)
            output2 = new_classifier(x)

        self.assertTrue(torch.allclose(output1, output2))


if __name__ == "__main__":
    unittest.main()
