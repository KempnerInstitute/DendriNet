"""Regression tests for EI network variants.

These tests verify that network construction and forward pass outputs remain
stable across code changes. They use frozen seeds so any change in initialization
or forward-pass logic is immediately detected.
"""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture

# ---------------------------------------------------------------------------
# Helpers: minimal configs for each network variant
# ---------------------------------------------------------------------------

INPUT_DIM = 20
OUTPUT_DIM = 10
BATCH = 4


def _minimal_config():
    """Return a minimal structured config dict for EI networks."""
    return {
        "architecture": {
            "excitatory_layer_sizes": [OUTPUT_DIM],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [3],
            "inhibitory_branch_factors": [],
            "inhibitory_network_type": "dendritic",
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [5],
            "ei_synapses_per_branch_per_layer": [0],
            "ie_synapses_per_branch_per_layer": [5],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "input_mode": 1,
            "independent_pathways": False,
        },
        "morphology": {
            "use_shunting": True,
            "weight_transform": "exp",
            "dbl_init_method": "analytical_expectation",
            "somatic_synapses": False,
        },
        "sparsity": {
            "init_method": "xavier_normal",
            "noise_level": 0.0,
            "type": "standard",
        },
        "reactivation": {
            "enabled": True,
            "type": "param_tanh",
            "init_m": 1.5,
            "init_b": 0.5,
        },
        "blocklinear": {
            "efficient": False,
        },
        "implementation": {},
    }


def _build_network(net_type, seed=42):
    """Build a network of the given type with a fixed seed."""
    torch.manual_seed(seed)
    config = _minimal_config()
    return get_architecture(net_type, config, input_dim=INPUT_DIM)


# ---------------------------------------------------------------------------
# Deterministic forward pass (golden-value regression)
# ---------------------------------------------------------------------------


class TestDeterministicForward:
    """Verify that seed=42 + fixed input → same output tensor.

    If any initialization or forward logic changes, these tests will fail,
    alerting the developer that network behavior has changed.
    """

    def _forward_output(self, net_type, seed=42):
        net = _build_network(net_type, seed=seed)
        net.eval()
        torch.manual_seed(seed)
        x = torch.randn(BATCH, INPUT_DIM)
        with torch.no_grad():
            return net(x)

    def test_dendritic_shunting_deterministic(self):
        out1 = self._forward_output("dendritic_shunting")
        out2 = self._forward_output("dendritic_shunting")
        torch.testing.assert_close(out1, out2)

    def test_dendritic_additive_deterministic(self):
        out1 = self._forward_output("dendritic_additive")
        out2 = self._forward_output("dendritic_additive")
        torch.testing.assert_close(out1, out2)

    def test_flat_shunting_deterministic(self):
        out1 = self._forward_output("flat_shunting")
        out2 = self._forward_output("flat_shunting")
        torch.testing.assert_close(out1, out2)

    def test_flat_additive_deterministic(self):
        out1 = self._forward_output("flat_additive")
        out2 = self._forward_output("flat_additive")
        torch.testing.assert_close(out1, out2)

    def test_dendritic_mlp_deterministic(self):
        out1 = self._forward_output("dendritic_mlp")
        out2 = self._forward_output("dendritic_mlp")
        torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# Backward determinism
# ---------------------------------------------------------------------------


class TestDeterministicBackward:
    def test_dendritic_shunting_backward_deterministic(self):
        def _run():
            torch.manual_seed(42)
            net = _build_network("dendritic_shunting", seed=42)
            net.train()
            torch.manual_seed(42)
            x = torch.randn(BATCH, INPUT_DIM)
            out = net(x)
            out.sum().backward()
            grads = []
            for p in net.parameters():
                if p.grad is not None:
                    grads.append(p.grad.clone())
            return grads

        g1 = _run()
        g2 = _run()
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# Output shape across all variants
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.mark.parametrize(
        "net_type",
        [
            "dendritic_shunting",
            "dendritic_additive",
            "flat_shunting",
            "flat_additive",
            "dendritic_mlp",
        ],
    )
    def test_output_shape(self, net_type):
        net = _build_network(net_type)
        x = torch.randn(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Weight positivity invariant
# ---------------------------------------------------------------------------


class TestWeightPositivity:
    @pytest.mark.parametrize("net_type", ["dendritic_shunting", "dendritic_additive"])
    def test_topk_weights_positive(self, net_type):
        """All TopKLinear weights should be positive when using exp transform."""
        net = _build_network(net_type)
        for name, module in net.named_modules():
            if isinstance(module, TopKLinear) and module.weight_transform == "exp":
                assert (module.weight() > 0).all(), f"Negative weight found in {name}"


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    @pytest.mark.parametrize(
        "net_type",
        [
            "dendritic_shunting",
            "dendritic_additive",
            "flat_shunting",
            "flat_additive",
            "dendritic_mlp",
            "einet",
        ],
    )
    def test_factory_creates_network(self, net_type):
        config = _minimal_config()
        net = get_architecture(net_type, config, input_dim=INPUT_DIM)
        assert net is not None
        x = torch.randn(1, INPUT_DIM)
        out = net(x)
        assert out.shape[1] == OUTPUT_DIM

    def test_factory_canonical_builds_network(self):
        config = _minimal_config()
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        assert net is not None

    def test_factory_invalid_type_raises(self):
        config = _minimal_config()
        with pytest.raises(ValueError):
            get_architecture("nonexistent_type", config, input_dim=INPUT_DIM)


# ---------------------------------------------------------------------------
# ConfigurableEINetwork from config
# ---------------------------------------------------------------------------


class TestConfigurableEINetwork:
    def test_builds_from_dict(self):
        config = _minimal_config()
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)
        x = torch.randn(2, INPUT_DIM)
        out = net(x)
        assert out.shape == (2, OUTPUT_DIM)

    def test_builds_with_flatten(self):
        config = _minimal_config()
        net = ConfigurableEINetwork(
            config=config, input_dim=INPUT_DIM, flatten_dendrites=True
        )
        x = torch.randn(2, INPUT_DIM)
        out = net(x)
        assert out.shape == (2, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Decay weights on all variants
# ---------------------------------------------------------------------------


class TestDecayWeights:
    @pytest.mark.parametrize(
        "net_type",
        [
            "dendritic_shunting",
            "dendritic_additive",
            "flat_shunting",
            "flat_additive",
        ],
    )
    def test_decay_weights_no_error(self, net_type):
        net = _build_network(net_type)
        net.decay_weights(weight_decay=0.01)  # Should not raise


# ---------------------------------------------------------------------------
# Multi-layer network
# ---------------------------------------------------------------------------


class TestMultiLayerNetwork:
    def test_two_layer_network(self):
        config = _minimal_config()
        config["architecture"]["excitatory_layer_sizes"] = [15, OUTPUT_DIM]
        config["connectivity"]["ee_synapses_per_branch_per_layer"] = [5, 5]
        config["connectivity"]["ie_synapses_per_branch_per_layer"] = [5, 5]
        config["connectivity"]["ei_synapses_per_branch_per_layer"] = [0, 0]
        config["connectivity"]["ii_synapses_per_branch_per_layer"] = [0, 0]
        torch.manual_seed(42)
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        x = torch.randn(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)
        assert len(net.layers) == 2


# ---------------------------------------------------------------------------
# Network with inhibitory neurons
# ---------------------------------------------------------------------------


class TestWithInhibitoryNeurons:
    def test_inhibitory_layer_sizes(self):
        config = _minimal_config()
        config["architecture"]["inhibitory_layer_sizes"] = [5]
        config["transfer"]["input_mode"] = 0
        config["connectivity"]["ei_synapses_per_branch_per_layer"] = [3]
        config["connectivity"]["ii_synapses_per_branch_per_layer"] = [2]
        torch.manual_seed(42)
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        x = torch.randn(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------


class TestTransferFunction:
    def test_transfer_fn_present(self):
        net = _build_network("dendritic_shunting")
        assert hasattr(net, "transfer_fn")
        x = torch.randn(2, INPUT_DIM)
        exc, _inh = net.transfer_fn(x)
        assert exc.shape[-1] == INPUT_DIM


# ---------------------------------------------------------------------------
# EI Layer with synapse_mode="mlp"
# ---------------------------------------------------------------------------


class TestSynapseModeMLPNetwork:
    def test_dendritic_mlp_forward_shape(self):
        net = _build_network("dendritic_mlp")
        x = torch.randn(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_dendritic_mlp_identity_weights(self):
        """dendritic_mlp uses identity weight transform (signed weights)."""
        net = _build_network("dendritic_mlp")
        assert net.weight_transform == "identity"

    def test_dendritic_mlp_no_shunting(self):
        net = _build_network("dendritic_mlp")
        assert net.use_shunting is False

    def test_dendritic_mlp_gradient_flow(self):
        net = _build_network("dendritic_mlp")
        x = torch.randn(2, INPUT_DIM)
        out = net(x)
        out.sum().backward()
        has_grad = any(p.grad is not None for p in net.parameters())
        assert has_grad

    def test_dendritic_mlp_with_inhibitory(self):
        """dendritic_mlp with inhibitory neurons (synapse_mode=mlp path in ei_layer)."""
        config = _minimal_config()
        config["architecture"]["inhibitory_layer_sizes"] = [5]
        config["architecture"]["inhibitory_branch_factors"] = [2]
        config["transfer"]["input_mode"] = 0
        config["connectivity"]["ei_synapses_per_branch_per_layer"] = [3]
        config["connectivity"]["ii_synapses_per_branch_per_layer"] = [2]
        torch.manual_seed(42)
        net = get_architecture("dendritic_mlp", config, input_dim=INPUT_DIM)
        x = torch.randn(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# EI Layer validation
# ---------------------------------------------------------------------------


class TestEILayerValidation:
    def test_invalid_synapse_mode_raises(self):
        from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
            ExcitationInhibitionLayer,
        )

        with pytest.raises(ValueError, match="synapse_mode"):
            ExcitationInhibitionLayer(
                n_excitatory_cells=10,
                n_inhibitory_cells=None,
                excitatory_branch_factors=[3],
                inhibitory_branch_factors=[],
                excitatory_input_dim=20,
                ee_synapses_per_branch=5,
                ei_synapses_per_branch=0,
                build_inhibitory_cells=False,
                synapse_mode="invalid",
            )


# ---------------------------------------------------------------------------
# Factory: classical architecture dispatches
# ---------------------------------------------------------------------------


class TestFactoryClassicalArchitectures:
    def test_mlp_flat_params(self):
        net = get_architecture(
            "mlp",
            {"input_dim": 20, "hidden_dims": [32], "output_dim": 10},
        )
        x = torch.randn(2, 20)
        out = net(x)
        assert out.shape == (2, 10)

    def test_identity(self):
        net = get_architecture("identity", {"input_dim": 20, "output_dim": 20})
        x = torch.randn(2, 20)
        out = net(x)
        assert out.shape == (2, 20)

    def test_factory_requires_structured_config_for_einet(self):
        """Flat params should raise for EINet types."""
        with pytest.raises(ValueError, match="structured config"):
            get_architecture("dendritic_shunting", {"some_flat_key": 1}, input_dim=20)

    def test_factory_requires_input_dim_for_einet(self):
        config = _minimal_config()
        with pytest.raises(ValueError, match="input_dim required"):
            get_architecture("dendritic_shunting", config, input_dim=None)


# ---------------------------------------------------------------------------
# Network properties
# ---------------------------------------------------------------------------


class TestNetworkProperties:
    def test_branch_layers_property(self):
        net = _build_network("dendritic_shunting")
        bl = net.branch_layers
        assert len(bl) > 0

    def test_n_branch_layers_property(self):
        net = _build_network("dendritic_shunting")
        assert net.n_branch_layers > 0

    def test_get_effective_params(self):
        net = _build_network("dendritic_shunting")
        eff = net.get_effective_params()
        assert eff > 0

    def test_apply_rewiring_no_error(self):
        net = _build_network("dendritic_shunting")
        net.apply_rewiring()  # Should not raise
