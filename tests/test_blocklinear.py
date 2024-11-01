import unittest

import torch

from dendritic_modeling.dendrinet import BlockLinear


class TestBlockLinear(unittest.TestCase):

    def test_basic_functionality(self):
        """Test the basic functionality with standard input data."""
        in_features = 8
        out_features = 2
        batch_size = 4

        model = BlockLinear(in_features, out_features)
        x = torch.randn(batch_size, in_features)
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Manually compute expected output
        block_size = model.block_size
        expected_output = torch.zeros(batch_size, out_features)
        for i in range(out_features):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            expected_output[:, i] = x[:, start_idx:end_idx].sum(dim=1)

        # Check that the model output matches the expected output
        self.assertTrue(torch.allclose(output, expected_output))

    def test_edge_case_small_input(self):
        """Test with minimal input sizes."""
        in_features = 2
        out_features = 1
        batch_size = 1

        model = BlockLinear(in_features, out_features)
        x = torch.randn(batch_size, in_features)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_edge_case_large_input(self):
        """Test with large input sizes."""
        in_features = 10000
        out_features = 100
        batch_size = 16

        model = BlockLinear(in_features, out_features)
        x = torch.randn(batch_size, in_features)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_error_handling_in_features_not_divisible(self):
        """Test error handling when in_features is not divisible by out_features."""
        in_features = 7
        out_features = 2

        with self.assertRaises(AssertionError) as context:
            model = BlockLinear(in_features, out_features)
        self.assertIn("must be divisible",
                      str(context.exception))

    def test_error_handling_negative_in_features(self):
        """Test error handling with negative in_features."""
        in_features = -8
        out_features = 2

        with self.assertRaises(AssertionError):
            model = BlockLinear(in_features, out_features)

    def test_sum_conductances(self):
        """Test the sum_conductances method."""
        in_features = 8
        out_features = 2

        model = BlockLinear(in_features, out_features)
        sum_g = model.sum_conductances()

        # Since weights are initialized to ones, sum_g should equal block_size for each output
        expected_sum_g = torch.tensor([model.block_size] * out_features, dtype=torch.float32)
        self.assertTrue(torch.allclose(sum_g, expected_sum_g))

    def test_input_shape_mismatch(self):
        """Test error handling when input shape does not match in_features."""
        in_features = 8
        out_features = 2
        batch_size = 1

        model = BlockLinear(in_features, out_features)
        x = torch.randn(batch_size, in_features + 1)  # Incorrect input size

        with self.assertRaises(RuntimeError) as context:
            output = model(x)
        self.assertIn("cannot be multiplied", str(context.exception))



if __name__ == '__main__':
    unittest.main()