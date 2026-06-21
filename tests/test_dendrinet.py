import unittest
from unittest.mock import call, patch

import torch

from dendritic_modeling.networks import (
    BlockLinear,
    DendriNet,
    DendriNetConfig,
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    _build_dendrinet_branch_layers,
    _plan_dendrinet_branch_layout,
    _sample_dendrinet_structured_masks,
    _validate_dendrinet_dimensions,
)


class TestDendriNet(unittest.TestCase):
    def assert_state_dicts_equal(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_branch_layout_plan_matches_construction_order(self):
        """Test DendriNet branch layout planning."""
        layer_sizes, input_branch_factors = _plan_dendrinet_branch_layout(
            n_soma=2,
            branch_factors=[2, 3],
        )

        self.assertEqual(layer_sizes, [12, 4, 2])
        self.assertEqual(input_branch_factors, [None, 3, 2])

        soma_only_sizes, soma_only_factors = _plan_dendrinet_branch_layout(
            n_soma=2,
            branch_factors=[],
        )
        self.assertEqual(soma_only_sizes, [2])
        self.assertEqual(soma_only_factors, [None])

    def test_structured_mask_pair_samples_excitatory_then_inhibitory(self):
        """Test DendriNet structured mask pair sampling."""
        structured_connectivity = object()
        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition."
            "dendritic.dendrinet.sample_configured_mask"
        ) as sample_mask:
            sample_mask.side_effect = ["exc-mask", "inh-mask"]

            masks = _sample_dendrinet_structured_masks(
                structured_connectivity,
                structured_excitatory_pathway="ee",
                structured_inhibitory_pathway="ie",
                out_dim=4,
                excitatory_input_dim=7,
                excitatory_synapses_per_branch=3,
                inhibitory_input_dim=5,
                inhibitory_synapses_per_branch=2,
                structured_layer_idx=6,
                level_idx=1,
            )

        self.assertEqual(masks, ("exc-mask", "inh-mask"))
        sample_mask.assert_has_calls(
            [
                call(
                    structured_connectivity,
                    pathway="ee",
                    out_features=4,
                    in_features=7,
                    layer_idx=6,
                    level_idx=1,
                ),
                call(
                    structured_connectivity,
                    pathway="ie",
                    out_features=4,
                    in_features=5,
                    layer_idx=6,
                    level_idx=1,
                ),
            ]
        )

    def test_dimension_validation_preserves_public_errors(self):
        """Test DendriNet dimension validation errors."""
        valid_kwargs = {
            "n_soma": 1,
            "branch_factors": [2],
            "excitatory_input_dim": 10,
            "excitatory_synapses_per_branch": 3,
            "inhibitory_input_dim": 5,
            "inhibitory_synapses_per_branch": 2,
        }

        with self.assertRaisesRegex(TypeError, "n_soma must be an integer"):
            _validate_dendrinet_dimensions(**{**valid_kwargs, "n_soma": 1.5})
        with self.assertRaisesRegex(TypeError, "branch_factors must be integers"):
            _validate_dendrinet_dimensions(**{**valid_kwargs, "branch_factors": [2.0]})
        with self.assertRaisesRegex(
            TypeError,
            "inhibitory_synapses_per_branch must be int",
        ):
            _validate_dendrinet_dimensions(
                **{**valid_kwargs, "inhibitory_synapses_per_branch": 1.0}
            )
        with self.assertRaisesRegex(
            TypeError,
            "excitatory_synapses_per_branch must be int",
        ):
            _validate_dendrinet_dimensions(
                **{**valid_kwargs, "excitatory_synapses_per_branch": 1.0}
            )

    def test_branch_layer_builder_preserves_layer_and_soma_order(self):
        """Test DendriNet branch layer construction order."""
        synapse_config = object()
        extra_kwargs = {"extra": "value"}
        with (
            patch(
                "dendritic_modeling.networks.architectures.excitation_inhibition."
                "dendritic.dendrinet._sample_dendrinet_structured_masks"
            ) as sample_masks,
            patch(
                "dendritic_modeling.networks.architectures.excitation_inhibition."
                "dendritic.dendrinet._make_dendrinet_branch_layer"
            ) as make_layer,
        ):
            sample_masks.side_effect = [("exc-0", "inh-0"), ("exc-1", "inh-1")]
            make_layer.side_effect = ["distal", "soma"]

            layers = _build_dendrinet_branch_layers(
                n_soma=2,
                n_branch_layers=1,
                layer_sizes=[4, 2],
                input_branch_factors=[None, 2],
                excitatory_input_dim=7,
                excitatory_synapses_per_branch=3,
                inhibitory_input_dim=5,
                inhibitory_synapses_per_branch=2,
                somatic_synapses=True,
                structured_connectivity="structured",
                structured_layer_idx=6,
                structured_excitatory_pathway="ee",
                structured_inhibitory_pathway="ie",
                synapse_config=synapse_config,
                branch_extra_kwargs=extra_kwargs,
            )

        self.assertEqual(layers, ["distal", "soma"])
        sample_masks.assert_has_calls(
            [
                call(
                    "structured",
                    structured_excitatory_pathway="ee",
                    structured_inhibitory_pathway="ie",
                    out_dim=4,
                    excitatory_input_dim=7,
                    excitatory_synapses_per_branch=3,
                    inhibitory_input_dim=5,
                    inhibitory_synapses_per_branch=2,
                    structured_layer_idx=6,
                    level_idx=0,
                ),
                call(
                    "structured",
                    structured_excitatory_pathway="ee",
                    structured_inhibitory_pathway="ie",
                    out_dim=2,
                    excitatory_input_dim=7,
                    excitatory_synapses_per_branch=3,
                    inhibitory_input_dim=5,
                    inhibitory_synapses_per_branch=2,
                    structured_layer_idx=6,
                    level_idx=1,
                ),
            ]
        )
        make_layer.assert_has_calls(
            [
                call(
                    output_dim=4,
                    excitatory_input_dim=7,
                    excitatory_synapses_per_branch=3,
                    inhibitory_input_dim=5,
                    inhibitory_synapses_per_branch=2,
                    excitatory_connection_mask="exc-0",
                    inhibitory_connection_mask="inh-0",
                    input_branch_factor=None,
                    layer_idx=1,
                    synapse_config=synapse_config,
                    branch_extra_kwargs=extra_kwargs,
                ),
                call(
                    output_dim=2,
                    excitatory_input_dim=7,
                    excitatory_synapses_per_branch=3,
                    inhibitory_input_dim=5,
                    inhibitory_synapses_per_branch=2,
                    excitatory_connection_mask="exc-1",
                    inhibitory_connection_mask="inh-1",
                    input_branch_factor=2,
                    layer_idx=0,
                    synapse_config=synapse_config,
                    branch_extra_kwargs=extra_kwargs,
                ),
            ]
        )

    def test_branch_layer_builder_preserves_soma_without_external_synapses(self):
        """Test DendriNet soma construction when somatic synapses are disabled."""
        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition."
            "dendritic.dendrinet._make_dendrinet_branch_layer"
        ) as make_layer:
            make_layer.return_value = "soma"

            layers = _build_dendrinet_branch_layers(
                n_soma=2,
                n_branch_layers=0,
                layer_sizes=[2],
                input_branch_factors=[None],
                excitatory_input_dim=7,
                excitatory_synapses_per_branch=3,
                inhibitory_input_dim=5,
                inhibitory_synapses_per_branch=2,
                somatic_synapses=False,
                structured_connectivity="structured",
                structured_layer_idx=6,
                structured_excitatory_pathway="ee",
                structured_inhibitory_pathway="ie",
                synapse_config="synapse",
                branch_extra_kwargs={},
            )

        self.assertEqual(layers, ["soma"])
        make_layer.assert_called_once_with(
            output_dim=2,
            excitatory_input_dim=None,
            excitatory_synapses_per_branch=None,
            inhibitory_input_dim=None,
            inhibitory_synapses_per_branch=None,
            input_branch_factor=None,
            layer_idx=0,
            synapse_config="synapse",
            branch_extra_kwargs={},
        )

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
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        )

        # Check that the model has the correct number of branch layers (including soma)
        self.assertEqual(len(model.branch_layers), len(branch_factors) + 1)

        # Check that the soma layer is correctly initialized (last layer)
        self.assertIsInstance(model.branch_layers[-1], DendriticBranchLayer)

    def test_config_path_matches_legacy_constructor_numerically(self):
        """Typed DendriNetConfig construction should preserve legacy behavior."""
        kwargs = {
            "n_soma": 2,
            "branch_factors": [2, 1],
            "excitatory_input_dim": 8,
            "excitatory_synapses_per_branch": 3,
            "inhibitory_input_dim": 5,
            "inhibitory_synapses_per_branch": 2,
            "reactivate": True,
            "reactivation_type": "param_tanh",
            "reactivation_init_m": 0.75,
            "reactivation_init_b": 0.25,
            "topk_type": "standard",
            "topk_init_method": "xavier_normal",
            "use_shunting": False,
            "weight_transform": "softplus",
            "efficient_blocklinear": True,
        }

        torch.manual_seed(123)
        legacy = DendriNet(**kwargs)
        torch.manual_seed(123)
        configured = DendriNet(config=DendriNetConfig.from_kwargs(kwargs))

        self.assertEqual(configured.branch_factors, legacy.branch_factors)
        self.assertEqual(configured.branch_configs, legacy.branch_configs)
        self.assertEqual(configured.synapse_config, legacy.synapse_config)
        self.assertEqual(configured.efficient_blocklinear, legacy.efficient_blocklinear)
        self.assert_state_dicts_equal(configured.state_dict(), legacy.state_dict())

        x = torch.rand(4, kwargs["excitatory_input_dim"])
        inhibitory = torch.rand(4, kwargs["inhibitory_input_dim"])
        torch.testing.assert_close(
            configured(x, inhibitory),
            legacy(x, inhibitory),
            rtol=0,
            atol=0,
        )

    def test_config_mapping_accepts_nested_synapse_sections(self):
        """Nested synapse sections should feed the same typed synapse config."""
        config = DendriNetConfig.from_config(
            {
                "n_soma": 1,
                "branch_factors": [2],
                "excitatory_input_dim": 6,
                "excitatory_synapses_per_branch": 2,
                "synapse": {
                    "topk": {"type": "indexed_rewire"},
                    "morphology": {
                        "weight_transform": "softplus",
                        "efficient_blocklinear": True,
                    },
                    "reactivation": {
                        "enabled": True,
                        "type": "relu",
                    },
                },
            }
        )

        model = DendriNet(config=config)

        self.assertEqual(model.synapse_config.topk.type, "indexed_rewire")
        self.assertEqual(model.synapse_config.morphology.weight_transform, "softplus")
        self.assertTrue(model.synapse_config.morphology.efficient_blocklinear)
        self.assertEqual(model.dendritic_activation, "relu")
        self.assertTrue(model.efficient_blocklinear)

    @property
    def soma_layer(self):
        """Property to access soma layer for backward compatibility."""
        return self.branch_layers[-1]

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
            reactivate=True,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
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
            reactivate=True,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        inhibitory_input = torch.rand(batch_size, inhibitory_input_dim)
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
            _ = DendriNet(
                n_soma=n_soma,
                branch_factors=branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            )

    def test_error_handling_invalid_synapses_per_branch(self):
        """Test that the network raises an error with invalid synapses per branch."""
        n_soma = 1
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = -1  # Invalid value

        with self.assertRaises(ValueError):
            _ = DendriNet(
                n_soma=n_soma,
                branch_factors=branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            )

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the network."""
        n_soma = 1
        batch_size = 4
        branch_factors = [2, 2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for trainable parameters
        parameters_with_grad = [p for p in model.parameters() if p.grad is not None]
        self.assertTrue(len(parameters_with_grad) > 0)

    def test_variable_branch_factors(self):
        """Test the network with variable branch factors."""
        n_soma = 1
        batch_size = 4
        branch_factors = [3, 1, 4]
        excitatory_input_dim = 20
        excitatory_synapses_per_branch = 5

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_edge_case_zero_soma(self):
        """Test the network initialization with zero soma neurons."""
        n_soma = 0
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        with self.assertRaises(ValueError):
            _ = DendriNet(
                n_soma=n_soma,
                branch_factors=branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            )

    def test_edge_case_large_branch_factors(self):
        """Test the network with large branch factors."""
        n_soma = 1
        batch_size = 4
        branch_factors = [10, 10]
        excitatory_input_dim = 100
        excitatory_synapses_per_branch = 5

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_different_input_dimensions(self):
        """Test the network with different excitatory and inhibitory input dimensions."""
        n_soma = 1
        batch_size = 4
        branch_factors = [2]
        excitatory_input_dim = 15
        inhibitory_input_dim = 5
        excitatory_synapses_per_branch = 3
        inhibitory_synapses_per_branch = 2

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            inhibitory_input_dim=inhibitory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        inhibitory_input = torch.rand(batch_size, inhibitory_input_dim)
        output = model(x, inhibitory_input)

        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_activation_function(self):
        """Test that the reactivation function works when reactivate is True."""
        n_soma = 1
        batch_size = 4
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        output = model(x)

        # Ensure output is non-negative due to the activation function
        self.assertTrue(torch.all(output >= 0))

    def test_dendritic_activation_alias_applies_to_all_layers(self):
        """Test public dendritic_activation alias for tree-level activation."""
        model = DendriNet(
            n_soma=1,
            branch_factors=[2, 2],
            excitatory_input_dim=10,
            excitatory_synapses_per_branch=3,
            reactivate=False,
            dendritic_activation="tanh",
        )

        self.assertEqual(model.dendritic_activation, "tanh")
        for layer in model.branch_layers:
            self.assertTrue(layer.reactivate)
            self.assertEqual(layer.dendritic_activation, "tanh")
            self.assertIsInstance(layer.reactivation, torch.nn.Tanh)

    def test_no_activation_function(self):
        """Test the network when reactivate is False."""
        n_soma = 1
        batch_size = 4
        branch_factors = [2]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=False,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        output = model(x)

        # Output can be negative or positive
        self.assertEqual(output.shape, (batch_size, n_soma))

    def test_large_network(self):
        """Test the network with a larger configuration."""
        n_soma = 10
        batch_size = 4
        branch_factors = [5, 4, 3]
        excitatory_input_dim = 100
        excitatory_synapses_per_branch = 10

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            reactivate=True,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
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
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        )

        # Check that weights in TopKLinear are initialized
        for layer in model.branch_layers:
            if layer.branch_excitation is not None:
                self.assertIsNotNone(layer.branch_excitation.pre_w)
                self.assertEqual(
                    layer.branch_excitation.pre_w.shape,
                    (layer.branch_excitation.weight().shape),
                )

    def test_blocklinear_aggregation(self):
        """Test that BlockLinear aggregates inputs correctly."""
        n_soma = 2  # Use 2 soma to ensure proper block_size
        branch_factors = [2]
        excitatory_input_dim = 5
        excitatory_synapses_per_branch = 2

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        )

        # Simulate branch input with correct size (4 branches for 2 soma)
        branch_input = torch.rand(1, 4)  # 2 branches * 2 soma = 4 inputs

        # Get the soma layer (last layer)
        soma_layer = model.branch_layers[-1]

        # Check that BlockLinear is used
        self.assertIsInstance(soma_layer.branches_to_output, BlockLinear)

        # Perform forward pass through BlockLinear
        output = soma_layer.branches_to_output(branch_input)

        # Output should have shape (1, n_soma)
        self.assertEqual(output.shape, (1, n_soma))

    def test_soma_layer_forward(self):
        """Test that the soma layer processes aggregated output from
        previous layers."""
        n_soma = 1
        batch_size = 4
        # Multiple branch layers leading to soma
        branch_factors = [2, 3]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 3

        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        )

        x = torch.rand(batch_size, excitatory_input_dim)
        output = model(x)

        # Check output shape and ensure the forward pass reaches soma layer
        self.assertEqual(output.shape, (batch_size, n_soma))
        # Confirm that the soma layer was involved in processing (last layer)
        self.assertIsInstance(model.branch_layers[-1], DendriticBranchLayer)

    def test_sum_weights_no_inhibitory(self):
        """Test sum_weights without inhibitory inputs."""
        n_soma = 2
        branch_factors = [2, 3]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 4

        # Initialize model without inhibitory input
        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        )

        # Calculate summed weights
        exc_total = model.sum_weights(pruned=False)

        self.assertIsInstance(exc_total, torch.Tensor)

        # Verify that the sum is positive
        self.assertTrue(exc_total.sum().item() > 0)

    def test_sum_weights_with_inhibitory(self):
        """Test sum_weights with inhibitory inputs."""
        n_soma = 2
        branch_factors = [2, 3]
        excitatory_input_dim = 10
        excitatory_synapses_per_branch = 4
        inhibitory_input_dim = 5
        inhibitory_synapses_per_branch = 3

        # Initialize model with inhibitory input
        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
        )

        # Calculate summed weights
        exc_total, inh_total = model.sum_weights(pruned=False)

        # Verify exc_total and inh_total are tensors
        self.assertIsInstance(exc_total, torch.Tensor)
        self.assertIsInstance(inh_total, torch.Tensor)

        # Verify that the sums are positive
        self.assertTrue(exc_total.sum().item() > 0)
        self.assertTrue(inh_total.sum().item() > 0)

    def test_sum_weights_with_pruned(self):
        """Test sum_weights with pruned weights."""
        n_soma = 2
        branch_factors = [1]
        excitatory_input_dim = 6
        excitatory_synapses_per_branch = 3

        # Initialize model with small dimensions for simple testing
        model = DendriNet(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        )

        # Calculate summed pruned weights
        exc_total = model.sum_weights(pruned=True)

        # Verify exc_total is a tensor
        self.assertIsInstance(exc_total, torch.Tensor)

        # Pruned weights should also yield a positive total
        self.assertTrue(exc_total.sum().item() > 0)


if __name__ == "__main__":
    unittest.main()
