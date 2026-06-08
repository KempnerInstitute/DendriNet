import unittest

import pytest
import torch

from dendritic_modeling.networks import (
    BlockLinear,
    DendriticBranchLayer,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    TopKLinear,
)


class TestDendriticBranchLayer(unittest.TestCase):
    def setUp(self):
        # Set dimensions and parameters for test cases
        self.output_dim = 5
        self.excitatory_input_dim = 10
        self.excitatory_synapses_per_branch = 3
        self.inhibitory_input_dim = 10
        self.inhibitory_synapses_per_branch = 3
        self.input_branch_factor = 2

    def test_initialization_no_inhibition_no_branch(self):
        # Test DendriticBranchLayer without inhibitory input and branch input
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            reactivate=False,
        )
        self.assertTrue(isinstance(layer.branch_excitation, TopKLinear))
        self.assertIsNone(layer.branch_inhibition)
        self.assertFalse(hasattr(layer, "branches_to_output"))
        self.assertFalse(layer.reactivate)

    def test_initialization_with_inhibition_and_branch(self):
        # Test DendriticBranchLayer with inhibitory and branch inputs
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            self.inhibitory_input_dim,
            self.inhibitory_synapses_per_branch,
            self.input_branch_factor,
            reactivate=True,
        )

        self.assertTrue(isinstance(layer.branch_excitation, TopKLinear))
        self.assertTrue(isinstance(layer.branch_inhibition, TopKLinear))
        self.assertTrue(isinstance(layer.branches_to_output, BlockLinear))
        self.assertTrue(layer.reactivate)

    def test_forward_no_inhibition_no_branch(self):
        # Forward pass without inhibition and branch inputs
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
        )
        x = torch.rand(2, self.excitatory_input_dim)  # Batch size of 2
        output = layer(x)
        self.assertEqual(output.shape, (2, self.output_dim))

    def test_forward_with_inhibition_and_branch(self):
        # Forward pass with inhibition and branch inputs
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            self.inhibitory_input_dim,
            self.inhibitory_synapses_per_branch,
            self.input_branch_factor,
        )
        x = torch.rand(2, self.excitatory_input_dim)  # Batch size of 2
        inhibitory_input = torch.rand(2, self.inhibitory_input_dim)
        branch_input = torch.rand(2, self.output_dim * self.input_branch_factor)
        output = layer(x, inhibitory_input, branch_input)
        self.assertEqual(output.shape, (2, self.output_dim))

    def test_forward_with_reactivation(self):
        # Test reactivation in forward pass
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            reactivate=True,
        )
        x = torch.rand(2, self.excitatory_input_dim)  # Batch size of 2
        output = layer(x)
        self.assertEqual(output.shape, (2, self.output_dim))
        # Check if reactivation threshold applied
        # Should be non-negative due to reactivation
        self.assertTrue(torch.all(output >= 0))

    def test_dendritic_activation_alias_enables_tanh(self):
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            reactivate=False,
            dendritic_activation="tanh",
        )
        self.assertTrue(layer.reactivate)
        self.assertEqual(layer.dendritic_activation, "tanh")
        self.assertIsInstance(layer.reactivation, torch.nn.Tanh)

    def test_dendritic_activation_none_disables_reactivation(self):
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            reactivate=True,
            dendritic_activation="none",
        )
        self.assertFalse(layer.reactivate)
        self.assertEqual(layer.dendritic_activation, "none")
        self.assertIsInstance(layer.reactivation, torch.nn.Identity)

    def test_get_weights(self):
        # Check weights retrieval
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            self.inhibitory_input_dim,
            self.inhibitory_synapses_per_branch,
        )
        weights = layer.get_weights()
        # Returns dict with excitatory and inhibitory weights
        self.assertIsInstance(weights, dict)
        self.assertIn("exc", weights)
        self.assertIn("inh", weights)
        self.assertEqual(
            weights["exc"].shape, (self.output_dim, self.excitatory_input_dim)
        )
        self.assertEqual(
            weights["inh"].shape, (self.output_dim, self.inhibitory_input_dim)
        )

    def test_get_mask(self):
        # Check weight mask retrieval
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            self.inhibitory_input_dim,
            self.inhibitory_synapses_per_branch,
        )
        masks = layer.get_mask()
        # Returns dict with excitatory and inhibitory masks
        self.assertIsInstance(masks, dict)
        self.assertIn("exc", masks)
        self.assertIn("inh", masks)
        self.assertEqual(
            masks["exc"].shape, (self.output_dim, self.excitatory_input_dim)
        )
        self.assertEqual(
            masks["inh"].shape, (self.output_dim, self.inhibitory_input_dim)
        )

    def test_get_pruned_weights(self):
        # Check pruned weights retrieval
        layer = DendriticBranchLayer(
            self.output_dim,
            self.excitatory_input_dim,
            self.excitatory_synapses_per_branch,
            self.inhibitory_input_dim,
            self.inhibitory_synapses_per_branch,
        )
        pruned_weights = layer.get_pruned_weights()
        # Returns dict with excitatory and inhibitory pruned weights
        self.assertIsInstance(pruned_weights, dict)
        self.assertIn("exc", pruned_weights)
        self.assertIn("inh", pruned_weights)
        self.assertEqual(
            pruned_weights["exc"].shape, (self.output_dim, self.excitatory_input_dim)
        )
        self.assertEqual(
            pruned_weights["inh"].shape, (self.output_dim, self.inhibitory_input_dim)
        )


@pytest.fixture
def sample_inputs():
    batch_size = 2
    in_features = 4
    x = torch.rand(batch_size, in_features)
    return x


@pytest.fixture
def sample_branch_layer():
    output_dim = 5
    excitatory_input_dim = 10
    excitatory_synapses_per_branch = 3
    return DendriticBranchLayer(
        output_dim,
        excitatory_input_dim,
        excitatory_synapses_per_branch,
        reactivate=False,
    )


class TestDendriticBranchLayerAdditional:
    """Additional tests for DendriticBranchLayer to improve coverage."""

    def test_forward_with_zero_input(self, sample_branch_layer):
        """Test forward pass with zero input."""
        batch_size = 2
        excitatory_input_dim = 10
        x = torch.zeros(batch_size, excitatory_input_dim)
        output = sample_branch_layer(x)
        # output_dim is stored as an instance variable during init
        expected_output_dim = 5  # From our fixture
        assert output.shape == (batch_size, expected_output_dim)
        assert output.dtype == torch.float32

    def test_forward_with_negative_input(self, sample_branch_layer):
        """DendriticBranchLayer accepts negative inputs directly.

        The nonneg-input validation for shunting lives at the network
        boundary (``ExcitationInhibitionNetwork``), not inside the branch
        layer itself. A raw DBL call with signed input does not raise —
        although the shunting forward may produce unbounded voltages, the
        layer itself does not guard against this when instantiated directly.
        """
        batch_size = 2
        excitatory_input_dim = 10
        x = torch.full((batch_size, excitatory_input_dim), -1.0)
        output = sample_branch_layer(x)
        expected_output_dim = 5
        assert output.shape == (batch_size, expected_output_dim)
        assert output.dtype == torch.float32

    def test_get_mask_no_threshold(self, sample_branch_layer):
        """Test get_mask method (no threshold parameter)."""
        mask = sample_branch_layer.get_mask()
        # For excitatory-only layer, returns dict with "exc" key
        assert isinstance(mask, dict)
        assert "exc" in mask
        assert torch.all((mask["exc"] == 0) | (mask["exc"] == 1))

    def test_get_pruned_weights_basic(self, sample_branch_layer):
        """Test get_pruned_weights method."""
        pruned_weights = sample_branch_layer.get_pruned_weights()
        # For excitatory-only layer, returns dict with "exc" key
        assert isinstance(pruned_weights, dict)
        assert pruned_weights["exc"].dtype == torch.float32

    def test_get_weights_basic(self, sample_branch_layer):
        """Test get_weights returns correct type."""
        weights = sample_branch_layer.get_weights()
        # For excitatory-only layer, returns dict with "exc" key
        assert isinstance(weights, dict)
        assert weights["exc"].dtype == torch.float32

    def test_get_log_weights(self, sample_branch_layer):
        """Test get_log_weights method."""
        log_weights = sample_branch_layer.get_log_weights()
        assert isinstance(log_weights, dict)
        assert log_weights["exc"].dtype == torch.float32

    def test_layer_with_both_excitatory_and_inhibitory(self):
        """Test layer with both excitatory and inhibitory inputs."""
        layer = DendriticBranchLayer(
            output_dim=3,
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=2,
            inhibitory_input_dim=6,
            inhibitory_synapses_per_branch=2,
        )

        # Test forward pass
        batch_size = 2
        x_exc = torch.rand(batch_size, 8)
        x_inh = torch.rand(batch_size, 6)
        output = layer(x_exc, x_inh)
        assert output.shape == (batch_size, 3)

        # Test get_weights returns dict for both exc and inh
        weights = layer.get_weights()
        assert isinstance(weights, dict)
        assert "exc" in weights and "inh" in weights

    def test_parameter_count(self, sample_branch_layer):
        """Test the number of trainable parameters."""
        total_params = sum(
            p.numel() for p in sample_branch_layer.parameters() if p.requires_grad
        )
        assert total_params > 0  # Should have some trainable parameters

    def test_unknown_topk_type_raises(self):
        with pytest.raises(ValueError, match="Unknown sparse layer type"):
            DendriticBranchLayer(
                output_dim=3,
                excitatory_input_dim=8,
                excitatory_synapses_per_branch=2,
                topk_type="not_a_sparse_strategy",
            )

    @pytest.mark.parametrize(
        ("topk_type", "expected_cls", "extra_kwargs"),
        [
            ("indexed", IndexedSparseLinear, {}),
            (
                "indexed_dynamic",
                IndexedDynamicTopKLinear,
                {"indexed_candidate_size": 3},
            ),
        ],
    )
    def test_indexed_recurrent_synapses_respect_forbidden_indices(
        self, topk_type, expected_cls, extra_kwargs
    ):
        forbidden = torch.tensor([0, 1, 2])
        layer = DendriticBranchLayer(
            output_dim=3,
            excitatory_input_dim=5,
            excitatory_synapses_per_branch=2,
            recurrent_input_dim=4,
            recurrent_synapses_per_branch=2,
            recurrent_forbidden_input_index_per_output=forbidden,
            topk_type=topk_type,
            indexed_seed=17,
            reactivate=False,
            input_branch_factor=None,
            **extra_kwargs,
        )

        assert isinstance(layer.branch_recurrent, expected_cls)
        if hasattr(layer.branch_recurrent, "candidate_mask"):
            mask = layer.branch_recurrent.candidate_mask().bool()
        else:
            mask = layer.branch_recurrent.weight_mask().bool()
        rows = torch.arange(forbidden.numel())
        assert not bool(mask[rows, forbidden].any())


if __name__ == "__main__":
    unittest.main()
