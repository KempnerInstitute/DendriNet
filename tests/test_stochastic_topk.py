import unittest

import torch

from dendritic_modeling.networks import StochasticTopKLinear, VarianceTopKLinear


class TestStochasticTopKLinear(unittest.TestCase):
    """Test cases for the StochasticTopKLinear class."""

    def test_basic_functionality(self):
        """Test the basic functionality with standard input data."""
        torch.manual_seed(0)
        in_features = 10
        out_features = 5
        K = 3
        batch_size = 4

        model = StochasticTopKLinear(in_features, out_features, K, temperature=1.0)
        model.train()  # Set to training mode to enable stochastic behavior
        x = torch.randn(batch_size, in_features)
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Verify that weights are positive
        weights = model.weight()
        self.assertTrue(torch.all(weights >= 0))

        # Training selects exactly K contacts per row, and the selected set
        # varies between forward passes; evaluation selects one fixed set.
        torch.manual_seed(0)
        masks = []
        for _ in range(10):
            model(x)
            masks.append(model.pruned_weight() > 0)
        self.assertTrue(all(torch.all(mask.sum(dim=1) == K) for mask in masks))
        self.assertTrue(any(not torch.equal(masks[0], mask) for mask in masks[1:]))
        model.eval()
        evaluated = []
        for _ in range(3):
            model(x)
            evaluated.append(model.pruned_weight() > 0)
        self.assertTrue(all(torch.equal(evaluated[0], mask) for mask in evaluated))

    def test_ultrafast_implementation(self):
        """Test the ultrafast implementation of stochastic TopK."""
        in_features = 100
        batch_size = 4
        out_features = 10
        K = 5

        # Create model with ultrafast enabled
        model = StochasticTopKLinear(
            in_features=in_features,
            out_features=out_features,
            K=K,
            temperature=0.5,
            ultrafast=True,
        )
        model.initialize()

        # Test forward pass
        x = torch.randn(batch_size, in_features)
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

        # Verify that the mask is created properly
        mask = model.weight_mask()
        self.assertEqual(mask.shape, (out_features, in_features))

        # Check that the mask is binary (0 or 1)
        unique_values = torch.unique(mask)
        self.assertTrue(len(unique_values) <= 2)
        self.assertTrue(torch.all((unique_values == 0) | (unique_values == 1)))

        # In ultrafast mode, exactly K connections should be active per neuron
        connections_per_neuron = mask.sum(dim=1)
        self.assertTrue(torch.all(connections_per_neuron == K))

    def test_temperature_effect(self):
        """Test that temperature affects the stochasticity of selection."""
        torch.manual_seed(0)
        in_features = 20
        out_features = 10
        K = 5

        # Create models with different temperatures
        model_low_temp = StochasticTopKLinear(
            in_features=in_features,
            out_features=out_features,
            K=K,
            temperature=0.1,  # Low temperature (more deterministic)
            ultrafast=False,
        )

        model_high_temp = StochasticTopKLinear(
            in_features=in_features,
            out_features=out_features,
            K=K,
            temperature=2.0,  # High temperature (more random)
            ultrafast=False,
        )

        # Initialize both models with the same weights
        model_low_temp.initialize()
        with torch.no_grad():
            model_high_temp.pre_w.copy_(model_low_temp.pre_w)

        # Set to training mode for stochastic behavior
        model_low_temp.train()
        model_high_temp.train()

        # Generate multiple samples to compare stochasticity
        x = torch.randn(1, in_features)

        low_temp_outputs = []
        high_temp_outputs = []

        # Run multiple forward passes
        for _ in range(10):
            low_temp_outputs.append(model_low_temp(x))
            high_temp_outputs.append(model_high_temp(x))

        # Calculate variance in outputs
        low_temp_stack = torch.stack(low_temp_outputs)
        high_temp_stack = torch.stack(high_temp_outputs)

        low_temp_variance = low_temp_stack.var(dim=0).mean().item()
        high_temp_variance = high_temp_stack.var(dim=0).mean().item()

        # Higher temperature varies the output more.
        self.assertGreater(low_temp_variance, 0.0)
        self.assertGreater(high_temp_variance, low_temp_variance)

    def test_zero_temperature(self):
        """Test deterministic behavior with zero temperature."""
        model = StochasticTopKLinear(10, 5, 3, temperature=0.0)
        model.initialize()
        x = torch.randn(4, 10)
        output = model(x)
        self.assertEqual(output.shape, (4, 5))
        # Check exactly K non-zero per row
        pruned_weights = model.pruned_weight()
        non_zero_counts = (pruned_weights > 0).sum(dim=1)
        self.assertTrue(torch.all(non_zero_counts == 3))

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow properly through the stochastic selection."""
        in_features = 50
        batch_size = 4
        out_features = 5
        K = 10

        # Create model
        model = StochasticTopKLinear(
            in_features=in_features,
            out_features=out_features,
            K=K,
            temperature=0.5,
            ultrafast=False,
        )
        model.initialize()

        # Prepare input and ensure requires_grad
        x = torch.randn(batch_size, in_features, requires_grad=True)

        # Forward pass
        output = model(x)

        # Create a loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        self.assertIsNotNone(model.pre_w.grad)
        self.assertIsNotNone(x.grad)

        # At least some gradients should be non-zero
        self.assertTrue(torch.any(model.pre_w.grad != 0))
        self.assertTrue(torch.any(x.grad != 0))


class TestVarianceTopKLinear(unittest.TestCase):
    """Test cases for the VarianceTopKLinear class."""

    def test_basic_functionality(self):
        """Test the basic functionality with standard input data."""
        torch.manual_seed(0)
        in_features = 100
        batch_size = 4
        out_features = 10
        K = 5

        # Create model
        model = VarianceTopKLinear(
            in_features=in_features, out_features=out_features, K=K, momentum=0.9
        )
        model.initialize()

        # Test forward pass with multiple batches to build statistics
        for _ in range(5):
            x = torch.randn(batch_size, in_features)
            output = model(x)

        # Check output shape of the last run
        self.assertEqual(output.shape, (batch_size, out_features))

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Verify that weights are positive
        weights = model.weight()
        self.assertTrue(torch.all(weights >= 0))

        # Check that activation statistics have been updated
        self.assertTrue(torch.any(model.activation_var != 1.0))  # Default is 1.0

    def test_transformer_shaped_inputs_reduce_all_leading_axes(self):
        model = VarianceTopKLinear(
            in_features=7,
            out_features=3,
            K=2,
            momentum=0.9,
        )
        inputs = torch.randn(4, 5, 7)

        output = model(inputs)

        self.assertEqual(output.shape, (4, 5, 3))
        self.assertEqual(model.activation_mean.shape, (7,))
        self.assertTrue(torch.allclose(model.activation_mean, inputs.mean(dim=(0, 1))))

    def test_train_eval_modes(self):
        """Test that train/eval modes properly control statistics tracking."""
        in_features = 100
        batch_size = 4
        out_features = 10
        K = 5

        # Create model
        model = VarianceTopKLinear(
            in_features=in_features, out_features=out_features, K=K, momentum=0.9
        )
        model.initialize()

        # Run in train mode
        model.train()
        x1 = torch.randn(batch_size, in_features)
        model(x1)
        var_after_train = model.activation_var.clone()

        # Switch to eval mode
        model.eval()
        x2 = torch.randn(batch_size, in_features) * 10  # Very different data
        model(x2)
        var_after_eval = model.activation_var.clone()

        # Stats should not change in eval mode
        self.assertTrue(torch.allclose(var_after_train, var_after_eval))

    def test_importance_score_calculation(self):
        """Test that importance scores combine weight and variance information."""
        in_features = 20
        batch_size = 4
        out_features = 5
        K = 4

        # Set seed for reproducibility
        torch.manual_seed(42)

        # Create model
        model = VarianceTopKLinear(
            in_features=in_features, out_features=out_features, K=K, momentum=0.9
        )
        model.initialize()

        # Create data with known variance properties
        # First half of features has high variance, second half has low
        high_var_features = torch.randn(batch_size, in_features // 2) * 5.0
        low_var_features = torch.randn(batch_size, in_features // 2) * 0.1
        x = torch.cat([high_var_features, low_var_features], dim=1)

        # Verify the data has the expected variance properties
        high_var_actual = high_var_features.var(dim=0).mean()
        low_var_actual = low_var_features.var(dim=0).mean()
        self.assertGreater(high_var_actual, low_var_actual)

        # Update model with this data several times
        for _ in range(10):
            model(x)

        # Get the weight mask
        mask = model.weight_mask()

        # Count connections to high-variance vs low-variance inputs
        high_var_connections = mask[:, : in_features // 2].sum()
        low_var_connections = mask[:, in_features // 2 :].sum()

        # More connections should go to high-variance inputs
        # Use a more lenient assertion in case of edge cases
        total_connections = high_var_connections + low_var_connections
        high_var_ratio = high_var_connections.float() / total_connections.float()

        # Expect at least 55% of connections to go to high-variance inputs
        # (more lenient than strict greater-than comparison)
        self.assertGreater(
            high_var_ratio,
            0.55,
            f"Expected >55% connections to high-var inputs, got {high_var_ratio:.2%}",
        )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
