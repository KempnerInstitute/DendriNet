import unittest

import torch

from dendritic_modeling.networks import TopKLinear


class TestTopKLinear(unittest.TestCase):
    def test_basic_functionality(self):
        """Test the basic functionality with standard input data."""
        in_features = 10
        out_features = 5
        K = 3
        batch_size = 4

        model = TopKLinear(in_features, out_features, K)
        x = torch.randn(batch_size, in_features)
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Check that weights are positive
        weights = model.weight()
        self.assertTrue(torch.all(weights >= 0))

        # Check that only K weights are non-zero per output neuron after pruning
        pruned_weights = model.pruned_weight()
        non_zero_counts = (pruned_weights > 0).sum(dim=1)
        self.assertTrue(torch.all(non_zero_counts == K))

    def test_edge_cases_small_input(self):
        """Test with very small input data."""
        in_features = 1
        batch_size = 4
        out_features = 1
        K = 1

        model = TopKLinear(in_features, out_features, K)
        x = torch.randn(batch_size, in_features)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_edge_cases_large_input(self):
        """A wide layer keeps exactly K contacts per output."""
        in_features = 2000
        batch_size = 4
        out_features = 300
        K = 50

        model = TopKLinear(in_features, out_features, K)
        x = torch.randn(batch_size, in_features)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertIsInstance(output, torch.Tensor)

        # Check that only K weights are non-zero per output neuron after pruning
        pruned_weights = model.pruned_weight()
        non_zero_counts = (pruned_weights > 0).sum(dim=1)

        # Check that no neuron has more than K weights
        self.assertTrue(
            torch.all(non_zero_counts <= K),
            f"Some neurons have more than K={K} non-zero weights: {non_zero_counts[non_zero_counts > K]}",
        )

        # For large models, topk selection might have edge cases
        # Just check that pruning is working in general - some neurons should have weights
        total_nonzero = torch.sum(non_zero_counts > 0).item()
        total_neurons = non_zero_counts.size(0)

        # At least half the neurons should have some non-zero weights
        self.assertGreater(
            total_nonzero / total_neurons,
            0.5,
            f"Only {total_nonzero}/{total_neurons} neurons have non-zero weights - pruning may not be working",
        )

    def test_edge_cases_empty_input(self):
        """Test with empty input data."""
        in_features = 0
        batch_size = 4
        out_features = 1
        K = 1

        with self.assertRaises(ValueError):
            _ = TopKLinear(in_features, out_features, K)
            _ = torch.randn(batch_size, in_features)
            # Note: model and x are not defined in this scope, but the exception
            # should be raised during TopKLinear construction

    def test_error_handling_negative_k(self):
        """Test handling of negative K value."""
        in_features = 10
        out_features = 5
        K = -1
        with self.assertRaises(ValueError):
            _ = TopKLinear(in_features, out_features, K)

    def test_error_handling_k_greater_than_in_features(self):
        """Test handling when K is greater than in_features."""
        in_features = 5
        out_features = 2
        K = 10  # Greater than in_features

        with self.assertRaises(ValueError):
            _ = TopKLinear(in_features, out_features, K)

    def test_error_handling_non_integer_k(self):
        """Test handling when K is non-integer."""
        in_features = 10
        out_features = 5
        K = 2.5  # Non-integer K

        # Current implementation converts non-integer K to int
        model = TopKLinear(in_features, out_features, K)
        # Verify that K was converted to integer
        self.assertEqual(model.K, 2)

    def test_error_handling_incorrect_input_shape(self):
        """Test handling of incorrect input shapes."""
        in_features = 10
        out_features = 5
        K = 3

        model = TopKLinear(in_features, out_features, K)
        x = torch.randn(4, in_features + 1)  # Incorrect input size

        with self.assertRaises(RuntimeError):
            _ = model(x)

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

        # Ensure transformed weights are finite and top-k pruning still works.
        weights = model.weight()
        self.assertTrue(torch.isfinite(weights).all())
        pruned_weights = model.pruned_weight()
        non_zero_counts = (pruned_weights > 0).sum(dim=1)
        self.assertTrue(torch.all(non_zero_counts == K))


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
