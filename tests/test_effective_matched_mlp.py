"""
Test suite for EffectiveMatchedParamMLP.

This module tests the EffectiveMatchedParamMLP functionality including:
- Parameter matching with EINet
- Forward pass functionality
- Factory creation
"""

import unittest

import torch

from dendritic_modeling.networks.architectures.classical.mlp import (
    EffectiveMatchedParamMLP,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


class TestEffectiveMatchedParamMLP(unittest.TestCase):
    """Test the EffectiveMatchedParamMLP class."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = {
            "input_dim": 784,
            "excitatory_layer_sizes": [256, 128],
            "inhibitory_layer_sizes": [128, 64],
            "excitatory_branch_factors": [4, 2],
            "inhibitory_branch_factors": [2, 1],
            "ee_synapses_per_branch_per_layer": [200, 100],
            "ei_synapses_per_branch_per_layer": [100, 50],
            "ie_synapses_per_branch_per_layer": [50, 25],
            "ii_synapses_per_branch_per_layer": [1],
            "reactivate": True,
            "somatic_synapses": True,
        }

    def test_effective_param_matching(self):
        """Test that EffectiveMatchedParamMLP matches EINet's effective parameters."""
        # Create EINet with default transfer_params
        transfer_params = {
            "input_mode": 0,
            "independent_pathways": False,
            "excitatory_dim": [],
            "inhibitory_dim": [],
            "output_activation": "none",
        }

        einet = ExcitationInhibitionNetwork(
            transfer_params=transfer_params, **self.params
        )
        effective_params = einet.get_effective_params()

        # Create EffectiveMatchedParamMLP
        mlp = EffectiveMatchedParamMLP(**self.params)
        mlp_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)

        # Check that MLP has at least as many params as EINet's effective count
        self.assertGreaterEqual(
            mlp_params,
            effective_params,
            f"MLP params ({mlp_params}) should be >= EINet effective params ({effective_params})",
        )

        # Check that param counts are reasonably close (within 20%)
        ratio = abs(mlp_params - effective_params) / max(effective_params, 1)
        self.assertLess(
            ratio,
            0.2,
            f"Param counts should be close: MLP={mlp_params}, EINet effective={effective_params}, ratio={ratio:.3f}",
        )

    def test_forward_pass(self):
        """Test that EffectiveMatchedParamMLP forward pass works correctly."""
        mlp = EffectiveMatchedParamMLP(**self.params)

        # Test forward pass with batch of inputs
        batch_size = 32
        input_dim = self.params["input_dim"]
        x = torch.randn(batch_size, input_dim)

        output = mlp(x)

        # Check output shape
        expected_output_shape = (batch_size, self.params["excitatory_layer_sizes"][-1])
        self.assertEqual(output.shape, expected_output_shape)

        # Check that output is finite
        self.assertTrue(torch.isfinite(output).all())

    def test_factory_creation(self):
        """Test that EffectiveMatchedParamMLP can be created via factory."""
        net = get_architecture("effectivematchedparammlp", self.params)
        self.assertIsInstance(net, EffectiveMatchedParamMLP)

    def test_comparison_with_einet_total_params(self):
        """Test that EffectiveMatchedParamMLP uses fewer params than dense EINet total."""
        transfer_params = {
            "input_mode": 0,
            "independent_pathways": False,
            "excitatory_dim": [],
            "inhibitory_dim": [],
            "output_activation": "none",
        }

        einet = ExcitationInhibitionNetwork(
            transfer_params=transfer_params, **self.params
        )

        # Create effective MLP
        effective_mlp = EffectiveMatchedParamMLP(**self.params)

        # Count parameters
        effective_params = sum(
            p.numel() for p in effective_mlp.parameters() if p.requires_grad
        )
        einet_total_params = sum(
            p.numel() for p in einet.parameters() if p.requires_grad
        )

        # Effective-matched baseline should be smaller than dense EINet total parameters.
        self.assertLess(
            effective_params,
            einet_total_params,
            f"EffectiveMatchedParamMLP ({effective_params}) should have fewer params than EINet total ({einet_total_params})",
        )

    def test_small_network(self):
        """Test with a small network configuration."""
        small_params = {
            "input_dim": 10,
            "excitatory_layer_sizes": [5],
            "inhibitory_layer_sizes": [3],
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [],
            "ee_synapses_per_branch_per_layer": [3],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [1],
            "reactivate": False,
            "somatic_synapses": True,
        }

        # Should not raise any errors
        mlp = EffectiveMatchedParamMLP(**small_params)

        # Test forward pass
        x = torch.randn(5, 10)
        output = mlp(x)
        self.assertEqual(output.shape, (5, 5))


if __name__ == "__main__":
    unittest.main()
