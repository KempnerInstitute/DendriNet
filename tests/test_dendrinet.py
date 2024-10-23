import unittest
import torch

from dendritic_modeling.dendrinet import (DendriNet,
                                          DendriticBranchLayer, 
                                          BlockLinear)


class TestDendriNet(unittest.TestCase):

    def test_initialization_valid_parameters(self):
        """Test that DendriNet initializes correctly with valid parameters."""
        n_soma = 2
        branch_factors = [2, 3]
        excitatory_input_dim = 100
        excitatory_synapses_per_branch = 5

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch
        )

        # Check that the model has the correct number of branch layers
        self.assertEqual(len(model.branch_layers), len(branch_factors))

        # Check that the soma layer is correctly initialized
        self.assertIsInstance(model.soma_layer, DendriticBranchLayer)

    def test_forward_pass(self):
        """Test that the forward pass works with valid input data."""
        n_soma = 1
        branch_factors = [2, 2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3
        batch_size = 4

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        output = model(x)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (batch_size, n_soma))

        # Ensure output is a tensor
        self.assertIsInstance(output, torch.Tensor)

    def test_inhibitory_input(self):
        """Test that the network handles inhibitory inputs correctly."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3
        inhibitory_input_dim = 5
        inhibitory_synapses_per_branch = 2
        batch_size = 4

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            reactivate=True
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        inhibitory_input = torch.randn(batch_size, inhibitory_input_dim)
        output = model(x, inhibitory_input)

        # Check that the output has the correct shape
        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_error_handling_invalid_branch_factors(self):
        """Test that the network raises an error with invalid branch factors."""
        n_soma = 1
        branch_factors = [0, -2]  # Invalid branch factors
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        with self.assertRaises(ValueError):
            model = DendriNet(
                n_soma=n_soma,
                branch_factors=branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch
            )

    def test_error_handling_invalid_synapses_per_branch(self):
        """Test that the network raises an error with invalid synapses per branch."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = -1  # Invalid value

        with self.assertRaises(ValueError):
            model = DendriNet(
                n_soma=n_soma,
                branch_factors=branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch
            )

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the network."""
        n_soma = 1
        branch_factors = [2, 2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3
        batch_size = 2

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for trainable parameters
        parameters_with_grad = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(parameters_with_grad) > 0)

    def test_variable_branch_factors(self):
        """Test the network with variable branch factors."""
        n_soma = 1
        branch_factors = [3, 1, 4]
        excitatory_input_dim = 20
        excitatory_synapses_per_branch = 5
        batch_size = 3

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_edge_case_zero_soma(self):
        """Test the network initialization with zero soma neurons."""
        n_soma = 0
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        with self.assertRaises(ValueError):
            model = DendriNet(
                n_soma=n_soma,
                branch_factors=branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch
            )

    def test_edge_case_large_branch_factors(self):
        """Test the network with large branch factors."""
        n_soma = 1
        branch_factors = [10, 10]
        excitatory_input_dim = 100
        excitatory_synapses_per_branch = 5
        batch_size = 1

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_different_input_dimensions(self):
        """Test the network with different excitatory and inhibitory input dimensions."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 15
        inhibitory_input_dim = 5
        excitatory_synapses_per_branch = 3
        inhibitory_synapses_per_branch = 2
        batch_size = 2

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            inhibitory_input_dim=inhibitory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        inhibitory_input = torch.randn(batch_size, inhibitory_input_dim)
        output = model(x, inhibitory_input)

        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_activation_function(self):
        """Test that the reactivation function works when reactivate is True."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3
        batch_size = 2

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        output = model(x)

        # Ensure output is non-negative due to the activation function
        self.assertTrue(torch.all(output >= 0))

    def test_no_activation_function(self):
        """Test the network when reactivate is False."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3
        batch_size = 2

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=False
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        output = model(x)

        # Output can be negative or positive
        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_large_network(self):
        """Test the network with a larger configuration."""
        n_soma = 10
        branch_factors = [5, 4, 3]
        excitatory_input_dim = 100
        excitatory_synapses_per_branch = 10
        batch_size = 5

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True
        )

        x = torch.randn(batch_size, excitatory_input_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_weight_initialization(self):
        """Test that weights are initialized correctly."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch
        )

        # Check that weights in TopKLinear are initialized
        for layer in model.branch_layers:
            self.assertIsNotNone(layer.branch_excitation.pre_w)
            self.assertEqual(layer.branch_excitation.pre_w.shape,
                             (layer.branch_excitation.weight().shape))


    def test_blocklinear_aggregation(self):
        """Test that BlockLinear aggregates inputs correctly."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 5
        excitatory_synapses_per_branch = 2

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch
        )

        # Simulate branch input
        branch_input = torch.randn(1, 2)  # Two branches from previous layer

        # Get the soma layer
        soma_layer = model.soma_layer

        # Check that BlockLinear is used
        self.assertIsInstance(soma_layer.branches_to_output, BlockLinear)

        # Perform forward pass through BlockLinear
        output = soma_layer.branches_to_output(branch_input)

        # Output should have shape (1, n_soma)
        self.assertEqual(output.shape, (1, n_soma))