import unittest

import torch

from dendritic_modeling.dendrinet import StochasticTopKLinear, VarianceTopKLinear


class TestStochasticTopKLinear(unittest.TestCase):
    """Test cases for the StochasticTopKLinear class."""

    def test_basic_functionality(self):
        """Test the basic functionality with standard input data."""
        in_features = 100
        out_features = 10
        K = 5
        batch_size = 8

        # Create model with moderate temperature
        model = StochasticTopKLinear(
            in_features=in_features,
            out_features=out_features,
            K=K,
            temperature=0.5,
            ultrafast=False,
        )
        model.initialize()

        # Test forward pass
        x = torch.randn(batch_size, in_features)
        output = model(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, out_features))

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Verify that weights are positive
        weights = model.weight()
        self.assertTrue(torch.all(weights >= 0))

        # Check that selection is probabilistic (not exactly K per row)
        pruned_weights = model.pruned_weight()
        non_zero_counts = (pruned_weights > 0).sum(dim=1)

        # Since it's stochastic, we expect some variation in the counts
        self.assertTrue(torch.any(non_zero_counts != K))

        # But on average, it should be close to K
        self.assertTrue(abs(non_zero_counts.float().mean().item() - K) < K)

    def test_ultrafast_implementation(self):
        """Test the ultrafast implementation of stochastic TopK."""
        in_features = 100
        out_features = 10
        K = 5
        batch_size = 8

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
        in_features = 100
        out_features = 10
        K = 5
        batch_size = 8
        torch.manual_seed(42)  # For reproducibility

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
            temperature=1.0,  # High temperature (more random)
            ultrafast=False,
        )

        # Initialize both models with the same weights
        model_low_temp.initialize()
        with torch.no_grad():
            model_high_temp.pre_w.copy_(model_low_temp.pre_w)

        # Get masks from both models
        low_temp_mask = model_low_temp.weight_mask()
        high_temp_mask = model_high_temp.weight_mask()

        # Calculate diversity in neuron connections
        low_temp_connections = low_temp_mask.sum(dim=1)
        high_temp_connections = high_temp_mask.sum(dim=1)

        # Higher temperature should lead to more variation in connection counts
        self.assertGreater(
            high_temp_connections.float().std().item(),
            low_temp_connections.float().std().item(),
        )

    def test_forward_pass_gradient_flow(self):
        """Test that gradients flow properly through the stochastic selection."""
        in_features = 50
        out_features = 5
        K = 10
        batch_size = 4

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
        in_features = 100
        out_features = 10
        K = 5
        batch_size = 8

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

    def test_train_eval_modes(self):
        """Test that train/eval modes properly control statistics tracking."""
        in_features = 100
        out_features = 10
        K = 5
        batch_size = 8

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
        out_features = 5
        K = 4
        batch_size = 10

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

        # Update model with this data several times
        for _ in range(10):
            model(x)

        # Get the weight mask
        mask = model.weight_mask()

        # Count connections to high-variance vs low-variance inputs
        high_var_connections = mask[:, : in_features // 2].sum()
        low_var_connections = mask[:, in_features // 2 :].sum()

        # More connections should go to high-variance inputs
        self.assertGreater(high_var_connections, low_var_connections)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
