import unittest

import torch

from dendritic_modeling.dendrinet import TopKLinear


class TestTopKLinear(unittest.TestCase):
    def test_basic_functionality(self):
        """Test the basic functionality with standard input data."""
        in_features = 10
        out_features = 5
        K = 3
        batch_size = 4

        model = TopKLinear(in_features, out_features, K)
        model.initialize()  # Initialize weights
        x = torch.randn(batch_size, in_features)
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Check that weights are positive (assuming default 'log' param_space)
        weights = model.weight()
        self.assertTrue(torch.all(weights >= 0))

        # Check that only K weights are non-zero per output neuron after pruning
        pruned_weights = model.pruned_weight()
        non_zero_counts = (pruned_weights > 1e-9).sum(
            dim=1
        )  # Use tolerance for float comparison
        self.assertTrue(torch.all(non_zero_counts == K))

    def test_edge_cases_small_input(self):
        """Test with very small input data."""
        in_features = 1
        out_features = 1
        K = 1
        batch_size = 1

        model = TopKLinear(in_features, out_features, K)
        model.initialize()
        x = torch.randn(batch_size, in_features)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_edge_cases_large_input(self):
        """Test with very large input data."""
        in_features = 10000
        out_features = 5000
        K = 50
        batch_size = 64

        model = TopKLinear(in_features, out_features, K)
        model.initialize()
        x = torch.randn(batch_size, in_features)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertIsInstance(output, torch.Tensor)

        # Check that only K weights are non-zero per output neuron after pruning
        pruned_weights = model.pruned_weight()
        non_zero_counts = (pruned_weights > 1e-9).sum(dim=1)
        self.assertTrue(torch.all(non_zero_counts == K))

    def test_edge_cases_empty_input(self):
        """Test with empty input data."""
        in_features = 10  # K must be <= in_features
        out_features = 5
        K = 0

        # Should raise ValueError because K must be >= 1
        with self.assertRaises(ValueError) as context:
            model = TopKLinear(in_features, out_features, K)
            # The error occurs during initialization, no need for forward pass here
        self.assertIn("K must be >= 1", str(context.exception))

    def test_error_handling_negative_K(self):
        """Test handling of negative K value."""
        in_features = 10
        out_features = 5
        K = -1
        with self.assertRaises(ValueError):
            model = TopKLinear(in_features, out_features, K)

    def test_error_handling_K_greater_than_in_features(self):
        """Test handling when K is greater than in_features."""
        in_features = 5
        out_features = 3
        K = 10
        with self.assertRaises(ValueError):
            model = TopKLinear(in_features, out_features, K)
            # The error occurs during initialization, no need for forward pass here

    def test_error_handling_non_integer_K(self):
        """Test handling when K is non-integer."""
        in_features = 10
        out_features = 5
        K = 2.5  # Non-integer K
        with self.assertRaises(TypeError):
            model = TopKLinear(in_features, out_features, K)

    def test_error_handling_incorrect_input_shape(self):
        """Test handling of incorrect input shapes."""
        in_features = 10
        out_features = 5
        K = 3
        model = TopKLinear(in_features, out_features, K)
        model.initialize()
        x = torch.randn(4, in_features + 1)  # Incorrect input size

        with self.assertRaises(RuntimeError):
            output = model(x)

    def test_param_space_presigmoid(self):
        """Test functionality with param_space set to 'presigmoid'."""
        in_features = 10
        out_features = 5
        K = 3
        batch_size = 4

        model = TopKLinear(in_features, out_features, K, param_space="presigmoid")
        model.initialize()
        x = torch.randn(batch_size, in_features)
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

        # Ensure weights are between 0 and 1
        weights = model.weight()
        self.assertTrue(torch.all((weights >= 0) & (weights <= 1)))

    def test_weight_gradients(self):
        """Test that gradients flow correctly through the pruned weights."""
        in_features = 10
        out_features = 5
        K = 3
        batch_size = 2

        model = TopKLinear(in_features, out_features, K)
        model.initialize()
        x = torch.randn(batch_size, in_features, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for the underlying parameter.
        self.assertIsNotNone(model.pre_w.grad)
        self.assertFalse(torch.all(model.pre_w.grad == 0))

    def test_forward_pass_consistency(self):
        """Test that forward pass outputs are consistent with manual computation."""

        torch.manual_seed(0)

        in_features = 5
        out_features = 2
        K = 2
        batch_size = 1

        model = TopKLinear(in_features, out_features, K)
        model.initialize()
        x = torch.randn(batch_size, in_features)

        # Perform forward pass
        output = model(x)

        # Manually compute expected output
        pruned_weight = model.pruned_weight()
        expected_output = torch.mm(x, pruned_weight.t())

        # Check that outputs match
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_weighted_synapses_no_prune(self):
        # Test without pruning (prune=False)
        in_features = 10
        out_features = 5
        K = 3

        model = TopKLinear(in_features, out_features, K)
        model.initialize()

        torch.manual_seed(0)

        cell_weights = torch.rand(out_features)
        model.pre_w.data.uniform_(-2.0, -1.5)

        result = model.weighted_synapses(cell_weights, prune=False)

        expected_weights = model.weight()
        expected_weighted_synapses = (cell_weights[:, None] * expected_weights).sum(
            dim=0
        )

        torch.testing.assert_close(
            result, expected_weighted_synapses, rtol=1e-5, atol=1e-5
        )

    def test_weighted_synapses_with_prune(self):
        # Test with pruning (prune=True)
        in_features = 10
        out_features = 5
        K = 3

        model = TopKLinear(in_features, out_features, K)
        model.initialize()

        torch.manual_seed(0)
        cell_weights = torch.rand(out_features)
        model.pre_w.data.uniform_(-2.0, -1.5)

        result = model.weighted_synapses(cell_weights, prune=True)

        expected_pruned_weights = model.pruned_weight()
        expected_weighted_synapses = (
            cell_weights[:, None] * expected_pruned_weights
        ).sum(dim=0)

        torch.testing.assert_close(
            result, expected_weighted_synapses, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
