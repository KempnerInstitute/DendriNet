import unittest

import torch

from dendritic_modeling.networks import ExcitationInhibitionNetwork


class TestExcitationInhibitionNetwork(unittest.TestCase):
    def test_initialization(self):
        """Test that the network initializes correctly."""
        model = ExcitationInhibitionNetwork(
            input_dim=784,
            excitatory_layer_sizes=[512, 256],
            inhibitory_layer_sizes=[200, 100],
            excitatory_branch_factors=[4, 4],
            inhibitory_branch_factors=[2, 2],
            reactivate=True,
            output_dim=10,
        )

        # Check attributes
        self.assertEqual(model.input_dim, 784)
        self.assertEqual(model.excitatory_layer_sizes, [512, 256])
        self.assertEqual(model.inhibitory_layer_sizes, [200, 100])
        self.assertEqual(model.excitatory_branch_factors, [4, 4])
        self.assertEqual(model.inhibitory_branch_factors, [2, 2])
        self.assertEqual(model.output_dim, 10)
        self.assertTrue(model.reactivate)

        # Check layer structure
        self.assertEqual(len(model.layers), 2)  # Two EI layers

    def test_forward_pass(self):
        """Test the forward pass."""
        model = ExcitationInhibitionNetwork(
            input_dim=784,
            excitatory_layer_sizes=[512, 256],
            inhibitory_layer_sizes=[200, 100],
            excitatory_branch_factors=[4, 4],
            inhibitory_branch_factors=[2, 2],
            reactivate=True,
            output_dim=10,
        )

        # Create dummy input
        x = torch.randn(2, 784)

        # Check forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))

        # Check that output is not all zeros or NaNs
        self.assertFalse(torch.all(output == 0))
        self.assertFalse(torch.isnan(output).any())

    def test_input_transformation(self):
        """Test that input transformation works correctly."""
        model = ExcitationInhibitionNetwork(
            input_dim=784,
            excitatory_layer_sizes=[512, 256],
            inhibitory_layer_sizes=[200, 100],
            excitatory_branch_factors=[4, 4],
            inhibitory_branch_factors=[2, 2],
            input_mode=2,  # Use input transformation
            input_net_dict={
                "type": "MLPInputTransform",
                "hidden_dims": [512],
                "output_dim": 128,
            },
            reactivate=True,
            output_dim=10,
        )

        # Create dummy input
        x = torch.randn(2, 784)

        # Check forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))

        # Check that output is not all zeros or NaNs
        self.assertFalse(torch.all(output == 0))
        self.assertFalse(torch.isnan(output).any())

    def test_branch_layer_outputs(self):
        """Test the forward pass with branch_layer_outputs=True."""
        model = ExcitationInhibitionNetwork(
            input_dim=784,
            excitatory_layer_sizes=[512, 256],
            inhibitory_layer_sizes=[200, 100],
            excitatory_branch_factors=[4, 4],
            inhibitory_branch_factors=[2, 2],
            reactivate=True,
            enable_branch_outputs=True,
            output_dim=10,
        )

        # Create dummy input
        x = torch.randn(2, 784)

        # Check forward pass
        output, branch_outputs = model(x, return_branch_outputs=True)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))

        # Check that branch_outputs is a dictionary
        self.assertIsInstance(branch_outputs, dict)

        # Check that branch_outputs contains entries for each layer
        self.assertEqual(len(branch_outputs), 2)  # Two layers

        # Check structure of branch_outputs for the first layer
        layer_output = branch_outputs[0]
        self.assertIn("excitatory", layer_output)
        self.assertIn("inhibitory", layer_output)

    def test_different_dimensions(self):
        """Test with different input, hidden, and output dimensions."""
        model = ExcitationInhibitionNetwork(
            input_dim=100,
            excitatory_layer_sizes=[32, 16],
            inhibitory_layer_sizes=[16, 8],
            excitatory_branch_factors=[2, 2],
            inhibitory_branch_factors=[2, 2],
            reactivate=True,
            output_dim=5,
        )

        # Create dummy input
        x = torch.randn(3, 100)

        # Check forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (3, 5))

    def test_no_inhibitory_pathway(self):
        """Test with no inhibitory pathway."""
        model = ExcitationInhibitionNetwork(
            input_dim=784,
            excitatory_layer_sizes=[512, 256],
            inhibitory_layer_sizes=[],  # No inhibitory layers
            excitatory_branch_factors=[4, 4],
            inhibitory_branch_factors=[],  # No inhibitory branches
            reactivate=True,
            output_dim=10,
        )

        # Create dummy input
        x = torch.randn(2, 784)

        # Check forward pass
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (2, 10))

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the network."""
        model = ExcitationInhibitionNetwork(
            input_dim=100,
            excitatory_layer_sizes=[32],
            inhibitory_layer_sizes=[16],
            excitatory_branch_factors=[2],
            inhibitory_branch_factors=[2],
            reactivate=True,
            output_dim=5,
        )

        # Create dummy input and target
        x = torch.randn(3, 100, requires_grad=True)

        # Forward pass
        output = model(x)

        # Create a dummy loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients are computed for model parameters
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")

        # Check that gradients are computed for the input
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.all(x.grad == 0))


if __name__ == "__main__":
    unittest.main()
