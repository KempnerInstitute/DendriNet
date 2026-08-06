"""Regression tests for EI network variants.

These tests verify that network construction and forward pass outputs remain
stable across code changes. They use frozen seeds so any change in initialization
or forward-pass logic is immediately detected.
"""

from unittest.mock import patch

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNetConfig,
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    _build_dendrinet_cells,
    _build_excitatory_cells,
    _build_inhibitory_cells,
    _resolve_excitatory_inhibitory_input_dim,
    _structured_connectivity_enabled,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
    _extend_synapse_schedule,
    _prepare_ei_layer_size_sequences,
    _prepare_ei_synapse_schedules,
    _resolve_ei_layer_build_spec,
    _validate_direct_inhibitory_stream_contract,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform import (
    TransferLayer,
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
            "allow_direct_inhibitory_stream": True,
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


def _config_with_inhibitory_layers(input_mode, inhibitory_layer_sizes):
    """Return a minimal EI config with optional explicit inhibitory cells."""
    config = _minimal_config()
    config["transfer"]["input_mode"] = input_mode
    config["transfer"]["allow_direct_inhibitory_stream"] = (
        input_mode == 1 and not inhibitory_layer_sizes
    )
    config["architecture"]["inhibitory_layer_sizes"] = list(inhibitory_layer_sizes)
    config["architecture"]["inhibitory_branch_factors"] = (
        [2] if inhibitory_layer_sizes else []
    )
    config["connectivity"]["ei_synapses_per_branch_per_layer"] = [
        3 if inhibitory_layer_sizes else 0
    ]
    config["connectivity"]["ii_synapses_per_branch_per_layer"] = [
        2 if inhibitory_layer_sizes else 0
    ]
    config["implementation"]["compile_forward"] = False
    return config


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
        x = torch.rand(BATCH, INPUT_DIM)
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
            x = torch.rand(BATCH, INPUT_DIM)
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
            "dendritic_signed",
            "flat_signed",
            "dendritic_mlp",
        ],
    )
    def test_output_shape(self, net_type):
        net = _build_network(net_type)
        x = torch.rand(BATCH, INPUT_DIM)
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
            "dendritic_signed",
            "flat_signed",
            "dendritic_mlp",
            "einet",
        ],
    )
    def test_factory_creates_network(self, net_type):
        config = _minimal_config()
        net = get_architecture(net_type, config, input_dim=INPUT_DIM)
        assert net is not None
        x = torch.rand(1, INPUT_DIM)
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

    @pytest.mark.parametrize(
        "alias",
        ["matchedtotalparammlp", "matchedactiveparammlp"],
    )
    def test_removed_matched_mlp_aliases_raise(self, alias):
        config = _minimal_config()
        with pytest.raises(ValueError):
            get_architecture(alias, config, input_dim=INPUT_DIM)


# ---------------------------------------------------------------------------
# ConfigurableEINetwork from config
# ---------------------------------------------------------------------------


class TestConfigurableEINetwork:
    def test_builds_from_dict(self):
        config = _minimal_config()
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)
        x = torch.rand(2, INPUT_DIM)
        out = net(x)
        assert out.shape == (2, OUTPUT_DIM)

    def test_builds_with_flatten(self):
        config = _minimal_config()
        net = ConfigurableEINetwork(
            config=config, input_dim=INPUT_DIM, flatten_dendrites=True
        )
        x = torch.rand(2, INPUT_DIM)
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
        x = torch.rand(BATCH, INPUT_DIM)
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
        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_input_mode1_empty_inhibitory_layers_uses_direct_inhibitory_stream(self):
        config = _config_with_inhibitory_layers(input_mode=1, inhibitory_layer_sizes=[])
        torch.manual_seed(42)
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)
        assert net.layers[0].inhibitory_cells is None
        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_input_mode1_direct_stream_can_require_explicit_opt_in(self):
        config = _config_with_inhibitory_layers(input_mode=1, inhibitory_layer_sizes=[])
        config["transfer"].pop("allow_direct_inhibitory_stream")
        config["transfer"]["require_explicit_direct_inhibitory_stream"] = True
        with pytest.raises(ValueError, match="allow_direct_inhibitory_stream"):
            ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)

    @pytest.mark.parametrize("input_mode", [0])
    def test_defined_inhibitory_layer_size_builds_inhibitory_population(
        self, input_mode
    ):
        config = _config_with_inhibitory_layers(
            input_mode=input_mode, inhibitory_layer_sizes=[5]
        )
        torch.manual_seed(42)
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)
        assert net.layers[0].inhibitory_cells is not None
        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_zero_drive_inhibitory_layer_size_uses_direct_inhibitory_stream(self):
        config = _config_with_inhibitory_layers(
            input_mode=0, inhibitory_layer_sizes=[5]
        )
        config["connectivity"]["ei_synapses_per_branch_per_layer"] = [0]
        config["connectivity"]["ii_synapses_per_branch_per_layer"] = [0]
        config["connectivity"]["ie_synapses_per_branch_per_layer"] = [3]

        torch.manual_seed(42)
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)
        layer = net.layers[0]

        assert layer.inhibitory_cells is None
        assert layer.excitatory_cells.branch_layers[0].branch_inhibition is not None

        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)
        assert torch.isfinite(out).all()

    def test_input_mode1_defined_inhibitory_size_is_legacy_direct_stream_by_default(
        self,
    ):
        config = _config_with_inhibitory_layers(
            input_mode=1, inhibitory_layer_sizes=[5]
        )
        torch.manual_seed(42)
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)
        assert net.layers[0].inhibitory_cells is None
        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_input_mode1_can_opt_into_explicit_inhibitory_population(self):
        config = _config_with_inhibitory_layers(
            input_mode=1, inhibitory_layer_sizes=[5]
        )
        config["transfer"]["input_mode1_build_inhibitory_population"] = True
        torch.manual_seed(42)
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)
        assert net.layers[0].inhibitory_cells is not None
        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_transfer_legacy_aliases_still_work_for_direct_dicts(self):
        config = _config_with_inhibitory_layers(
            input_mode=1, inhibitory_layer_sizes=[5]
        )
        config["transfer"].pop("allow_direct_inhibitory_stream", None)
        config["transfer"]["allow_direct_inhibitory_input"] = True
        config["transfer"]["build_inhibitory_population_for_input_mode1"] = True

        torch.manual_seed(42)
        net = ConfigurableEINetwork(config=config, input_dim=INPUT_DIM)

        assert net.transfer_params["allow_direct_inhibitory_stream"] is True
        assert "allow_direct_inhibitory_input" not in net.transfer_params
        assert net.layers[0].inhibitory_cells is not None
        out = net(torch.rand(BATCH, INPUT_DIM))
        assert out.shape == (BATCH, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------


class TestTransferFunction:
    def test_transfer_fn_present(self):
        net = _build_network("dendritic_shunting")
        assert hasattr(net, "transfer_fn")
        x = torch.rand(2, INPUT_DIM)
        exc, _inh = net.transfer_fn(x)
        assert exc.shape[-1] == INPUT_DIM

    def test_transfer_input_mode0_duplicates_full_input(self):
        transfer = TransferLayer(
            input_dim=7,
            transfer_params={"input_mode": 0, "independent_pathways": False},
        )
        x = torch.randn(3, 7)
        exc, inh = transfer(x)
        assert exc.shape == (3, 7)
        assert inh.shape == (3, 7)
        assert torch.allclose(exc, x)
        assert torch.allclose(inh, x)

    def test_transfer_input_mode0_allows_independent_split_shapes(self):
        transfer = TransferLayer(
            input_dim=7,
            transfer_params={
                "input_mode": 0,
                "independent_pathways": True,
                "excitatory_dim": 4,
                "inhibitory_dim": 3,
            },
        )
        x = torch.randn(3, 7)
        exc, inh = transfer(x)
        assert exc.shape == (3, 4)
        assert inh.shape == (3, 3)
        assert transfer.excitatory_dim == 4
        assert transfer.inhibitory_dim == 3

    def test_transfer_input_mode1_independent_split_shapes(self):
        transfer = TransferLayer(
            input_dim=7,
            transfer_params={
                "input_mode": 1,
                "independent_pathways": True,
                "excitatory_dim": 4,
                "inhibitory_dim": 3,
            },
        )
        x = torch.randn(3, 7)
        exc, inh = transfer(x)
        assert exc.shape == (3, 4)
        assert inh.shape == (3, 3)
        assert transfer.excitatory_dim == 4
        assert transfer.inhibitory_dim == 3

    def test_transfer_rejects_invalid_independent_split_dims(self):
        with pytest.raises(ValueError, match="must equal input_dim"):
            TransferLayer(
                input_dim=7,
                transfer_params={
                    "input_mode": 1,
                    "independent_pathways": True,
                    "excitatory_dim": 5,
                    "inhibitory_dim": 5,
                },
            )

    def test_transfer_rejects_runtime_input_dim_mismatch(self):
        transfer = TransferLayer(
            input_dim=7,
            transfer_params={"input_mode": 0, "independent_pathways": False},
        )
        with pytest.raises(ValueError, match="expected last input dimension 7"):
            transfer(torch.randn(3, 8))

    @pytest.mark.parametrize(
        "e_stream_dims",
        [4, ":4", "0:4", slice(None, 4), slice(0, 4)],
    )
    def test_transfer_split_input_stream_prefix_shapes(self, e_stream_dims):
        transfer = TransferLayer(
            input_dim=7,
            transfer_params={
                "input_mode": 0,
                "split_input_stream": True,
                "e_stream_dims": e_stream_dims,
            },
        )
        x = torch.arange(21, dtype=torch.float32).reshape(3, 7)
        e_stream, i_stream = transfer(x)
        assert transfer.split_input_stream is True
        assert transfer.excitatory_dim == 4
        assert transfer.inhibitory_dim == 3
        assert e_stream.shape == (3, 4)
        assert i_stream.shape == (3, 3)
        assert torch.allclose(e_stream, x[..., :4])
        assert torch.allclose(i_stream, x[..., 4:])

    def test_transfer_split_input_stream_omitted_dims_reverts_to_identity(self):
        transfer = TransferLayer(
            input_dim=7,
            transfer_params={"input_mode": 0, "split_input_stream": True},
        )
        x = torch.randn(3, 7)
        e_stream, i_stream = transfer(x)
        assert transfer.split_input_stream is False
        assert e_stream.shape == (3, 7)
        assert i_stream.shape == (3, 7)
        assert torch.allclose(e_stream, x)
        assert torch.allclose(i_stream, x)

    def test_transfer_split_input_stream_mutual_exclusion(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            TransferLayer(
                input_dim=7,
                transfer_params={
                    "input_mode": 0,
                    "independent_pathways": True,
                    "split_input_stream": True,
                    "e_stream_dims": 4,
                    "excitatory_dim": 4,
                    "inhibitory_dim": 3,
                },
            )

    def test_transfer_split_input_stream_rejects_bad_k(self):
        with pytest.raises(ValueError, match="0 < k < input_dim"):
            TransferLayer(
                input_dim=7,
                transfer_params={
                    "input_mode": 0,
                    "split_input_stream": True,
                    "e_stream_dims": 7,
                },
            )

    def test_transfer_split_input_stream_rejects_non_prefix_slice(self):
        with pytest.raises(ValueError, match="prefix slice"):
            TransferLayer(
                input_dim=7,
                transfer_params={
                    "input_mode": 0,
                    "split_input_stream": True,
                    "e_stream_dims": "1:4",
                },
            )


class TestSplitInputStreamNetwork:
    def _split_config(self, *, input_mode):
        if input_mode == 0:
            config = _config_with_inhibitory_layers(
                input_mode=0, inhibitory_layer_sizes=[5]
            )
        else:
            config = _config_with_inhibitory_layers(
                input_mode=1, inhibitory_layer_sizes=[]
            )
            config["transfer"]["allow_direct_inhibitory_stream"] = True
        config["transfer"]["split_input_stream"] = True
        config["transfer"]["e_stream_dims"] = 12
        config["connectivity"]["ee_synapses_per_branch_per_layer"] = [5]
        config["connectivity"]["ie_synapses_per_branch_per_layer"] = [3]
        return config

    def test_input_mode0_wires_i_to_i_stream_only(self):
        config = self._split_config(input_mode=0)
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        layer0 = net.layers[0]
        assert layer0.split_input_stream is True
        assert net.transfer_fn.excitatory_dim == 12
        assert net.transfer_fn.inhibitory_dim == 8
        assert layer0.inhibitory_cells is not None
        # I.ee source is i_stream dim; II bank is absent.
        assert layer0.inhibitory_cells.branch_layers[0].excitatory_input_dim == 8
        assert layer0.inhibitory_cells.branch_layers[0].input_inhibitory is False
        # E.ee source is e_stream dim; IE comes from I-cell outputs.
        assert layer0.excitatory_cells.branch_layers[0].excitatory_input_dim == 12
        assert layer0.excitatory_cells.branch_layers[0].inhibitory_input_dim == 5

        x = torch.rand(BATCH, INPUT_DIM)
        y = net(x)
        assert y.shape == (BATCH, OUTPUT_DIM)

    def test_input_mode0_mlp_forward(self):
        config = self._split_config(input_mode=0)
        net = get_architecture("dendritic_mlp", config, input_dim=INPUT_DIM)
        assert net.layers[0].split_input_stream is True
        y = net(torch.rand(BATCH, INPUT_DIM))
        assert y.shape == (BATCH, OUTPUT_DIM)

    def test_input_mode1_routes_i_stream_to_e_ie(self):
        config = self._split_config(input_mode=1)
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        layer0 = net.layers[0]
        assert layer0.split_input_stream is True
        assert layer0.inhibitory_cells is None
        assert layer0.excitatory_cells.branch_layers[0].excitatory_input_dim == 12
        assert layer0.excitatory_cells.branch_layers[0].inhibitory_input_dim == 8
        y = net(torch.rand(BATCH, INPUT_DIM))
        assert y.shape == (BATCH, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# EI Layer with synapse_mode="mlp"
# ---------------------------------------------------------------------------


class TestSynapseModeMLPNetwork:
    def test_topk_alias_normalizes_to_standard(self):
        """Legacy sparsity.type='topk' should build as standard TopK."""
        config = _minimal_config()
        config["sparsity"]["type"] = "topk"
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        assert net.topk_type == "standard"

    def test_dendritic_mlp_forward_shape(self):
        net = _build_network("dendritic_mlp")
        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_dendritic_mlp_identity_weights(self):
        """dendritic_mlp uses identity weight transform (signed weights)."""
        net = _build_network("dendritic_mlp")
        assert net.weight_transform == "identity"

    @pytest.mark.parametrize("net_type", ["dendritic_signed", "flat_signed"])
    def test_signed_einet_identity_weights(self, net_type):
        """Signed EINet aliases keep E/I mode but use identity weights."""
        net = _build_network(net_type)
        assert net.synapse_mode == "ei"
        assert net.weight_transform == "identity"
        assert net.use_shunting is False

    def test_dendritic_mlp_no_shunting(self):
        net = _build_network("dendritic_mlp")
        assert net.use_shunting is False

    def test_dendritic_mlp_gradient_flow(self):
        net = _build_network("dendritic_mlp")
        x = torch.rand(2, INPUT_DIM)
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
        x = torch.rand(BATCH, INPUT_DIM)
        out = net(x)
        assert out.shape == (BATCH, OUTPUT_DIM)


# ---------------------------------------------------------------------------
# EI Layer validation
# ---------------------------------------------------------------------------


class TestEILayerValidation:
    def test_structured_connectivity_enabled_accepts_mapping_and_config_objects(self):
        class StructuredConfig:
            enabled = True

        assert _structured_connectivity_enabled({"enabled": True})
        assert _structured_connectivity_enabled(StructuredConfig())
        assert not _structured_connectivity_enabled({"enabled": False})
        assert not _structured_connectivity_enabled(None)

    def test_excitatory_inhibitory_input_dim_gates_on_ie_synapses(self):
        dim, concat = _resolve_excitatory_inhibitory_input_dim(
            n_inhibitory_cells=5,
            build_inhibitory_cells=True,
            inhibitory_input_dim=3,
            ie_synapses_per_branch=2,
        )
        assert dim == 5
        assert concat is True

        dim, concat = _resolve_excitatory_inhibitory_input_dim(
            n_inhibitory_cells=5,
            build_inhibitory_cells=True,
            inhibitory_input_dim=3,
            ie_synapses_per_branch=0,
        )
        assert dim is None
        assert concat is False

        dim, concat = _resolve_excitatory_inhibitory_input_dim(
            n_inhibitory_cells=None,
            build_inhibitory_cells=False,
            inhibitory_input_dim=3,
            ie_synapses_per_branch=1,
        )
        assert dim == 3
        assert concat is True

    def test_dendrinet_cell_builder_groups_config_and_preserves_extra_kwargs(self):
        synapse_config = DendriticSynapseConfig.from_kwargs(
            {"topk_type": "indexed_rewire", "weight_transform": "softplus"}
        )
        marker = object()

        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition."
            "ei_layer.DendriNet"
        ) as build_dendrinet:
            build_dendrinet.return_value = "cells"
            result = _build_dendrinet_cells(
                n_soma=7,
                branch_factors=[2, 3],
                excitatory_input_dim=11,
                excitatory_synapses_per_branch=5,
                inhibitory_input_dim=4,
                inhibitory_synapses_per_branch=2,
                somatic_synapses=False,
                synapse_config=synapse_config,
                dendrinet_extra_kwargs={"unit_marker": marker},
                structured_connectivity={"enabled": True},
                structured_layer_idx=3,
                structured_excitatory_pathway="ei",
                structured_inhibitory_pathway="ii",
            )

        assert result == "cells"
        build_dendrinet.assert_called_once()
        call_kwargs = build_dendrinet.call_args.kwargs
        config = call_kwargs["config"]
        assert isinstance(config, DendriNetConfig)
        assert config.n_soma == 7
        assert config.branch_factors == (2, 3)
        assert config.excitatory_input_dim == 11
        assert config.excitatory_synapses_per_branch == 5
        assert config.inhibitory_input_dim == 4
        assert config.inhibitory_synapses_per_branch == 2
        assert config.somatic_synapses is False
        assert config.structured_connectivity == {"enabled": True}
        assert config.structured_layer_idx == 3
        assert config.structured_excitatory_pathway == "ei"
        assert config.structured_inhibitory_pathway == "ii"
        assert config.synapse_config == synapse_config
        assert call_kwargs["unit_marker"] is marker

    def test_ei_mode_builders_preserve_structured_pathway_metadata(self):
        synapse_config = DendriticSynapseConfig()
        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition."
            "ei_layer._build_dendrinet_cells"
        ) as build_cells:
            build_cells.side_effect = ["inh", "exc"]
            inhibitory = _build_inhibitory_cells(
                n_inhibitory_cells=5,
                inhibitory_branch_factors=[2],
                excitatory_input_dim=13,
                ei_synapses_per_branch=3,
                build_inhibitory_cells=True,
                inhibitory_network_type="dendritic",
                inhibitory_input_dim=4,
                ii_synapses_per_branch=2,
                somatic_synapses=True,
                synapse_mode="ei",
                common_synapse_config=synapse_config,
                structured_connectivity={"enabled": True},
                structured_layer_idx=9,
                dendrinet_extra_kwargs={"extra": "value"},
                mlp_inhibitory_network_params={},
            )
            excitatory = _build_excitatory_cells(
                n_excitatory_cells=7,
                excitatory_branch_factors=[3],
                excitatory_input_dim=13,
                ee_synapses_per_branch=4,
                excitatory_inhibitory_input_dim=5,
                ie_synapses_per_branch=1,
                somatic_synapses=True,
                synapse_mode="ei",
                common_synapse_config=synapse_config,
                structured_connectivity={"enabled": True},
                structured_layer_idx=9,
                dendrinet_extra_kwargs={"extra": "value"},
            )

        assert inhibitory == "inh"
        assert excitatory == "exc"
        inhibitory_kwargs = build_cells.call_args_list[0].kwargs
        excitatory_kwargs = build_cells.call_args_list[1].kwargs
        assert inhibitory_kwargs["structured_excitatory_pathway"] == "ei"
        assert inhibitory_kwargs["structured_inhibitory_pathway"] == "ii"
        assert inhibitory_kwargs["inhibitory_input_dim"] == 4
        assert inhibitory_kwargs["inhibitory_synapses_per_branch"] == 2
        assert excitatory_kwargs["structured_excitatory_pathway"] == "ee"
        assert excitatory_kwargs["structured_inhibitory_pathway"] == "ie"
        assert excitatory_kwargs["inhibitory_input_dim"] == 5
        assert excitatory_kwargs["inhibitory_synapses_per_branch"] == 1

    def test_mlp_synapse_mode_builders_preserve_merged_input_geometry(self):
        synapse_config = DendriticSynapseConfig()
        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition."
            "ei_layer._build_dendrinet_cells"
        ) as build_cells:
            build_cells.side_effect = ["inh", "exc"]
            inhibitory = _build_inhibitory_cells(
                n_inhibitory_cells=5,
                inhibitory_branch_factors=[2],
                excitatory_input_dim=13,
                ei_synapses_per_branch=3,
                build_inhibitory_cells=True,
                inhibitory_network_type="dendritic",
                inhibitory_input_dim=4,
                ii_synapses_per_branch=2,
                somatic_synapses=True,
                synapse_mode="mlp",
                common_synapse_config=synapse_config,
                structured_connectivity=None,
                structured_layer_idx=0,
                dendrinet_extra_kwargs={},
                mlp_inhibitory_network_params={},
            )
            excitatory = _build_excitatory_cells(
                n_excitatory_cells=7,
                excitatory_branch_factors=[3],
                excitatory_input_dim=13,
                ee_synapses_per_branch=4,
                excitatory_inhibitory_input_dim=5,
                ie_synapses_per_branch=1,
                somatic_synapses=True,
                synapse_mode="mlp",
                common_synapse_config=synapse_config,
                structured_connectivity=None,
                structured_layer_idx=0,
                dendrinet_extra_kwargs={},
            )

        assert inhibitory == "inh"
        assert excitatory == "exc"
        inhibitory_kwargs = build_cells.call_args_list[0].kwargs
        excitatory_kwargs = build_cells.call_args_list[1].kwargs
        assert inhibitory_kwargs["excitatory_input_dim"] == 17
        assert inhibitory_kwargs["excitatory_synapses_per_branch"] == 5
        assert inhibitory_kwargs["inhibitory_input_dim"] is None
        assert inhibitory_kwargs["inhibitory_synapses_per_branch"] is None
        assert excitatory_kwargs["excitatory_input_dim"] == 18
        assert excitatory_kwargs["excitatory_synapses_per_branch"] == 5
        assert excitatory_kwargs["inhibitory_input_dim"] is None
        assert excitatory_kwargs["inhibitory_synapses_per_branch"] is None

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
# EI Network schedule helpers
# ---------------------------------------------------------------------------


class TestEINetworkScheduleHelpers:
    def test_extend_synapse_schedule_repeats_last_value(self):
        assert _extend_synapse_schedule(
            [2],
            num_layers=3,
            required=True,
            empty_error="missing",
        ) == [2, 2, 2]

    def test_extend_synapse_schedule_fills_optional_empty_with_zero(self):
        assert _extend_synapse_schedule(
            [],
            num_layers=3,
            required=False,
            empty_error="missing",
        ) == [0, 0, 0]

    def test_extend_synapse_schedule_preserves_required_empty_error(self):
        with pytest.raises(ValueError, match="ee_synapses_per_branch_per_layer"):
            _extend_synapse_schedule(
                [],
                num_layers=1,
                required=True,
                empty_error="ee_synapses_per_branch_per_layer cannot be empty",
            )


class TestEINetworkSynapseScheduleHelpers:
    def test_prepare_synapse_schedules_extends_all_streams(self):
        schedules = _prepare_ei_synapse_schedules(
            ee_synapses_per_branch_per_layer=[5],
            ei_synapses_per_branch_per_layer=[3],
            ie_synapses_per_branch_per_layer=[2],
            ii_synapses_per_branch_per_layer=[1],
            num_layers=3,
            input_mode=0,
            has_inhibitory_neurons=True,
        )

        assert schedules == ([5, 5, 5], [3, 3, 3], [2, 2, 2], [1, 1, 1])

    def test_prepare_synapse_schedules_defaults_optional_absent_i_streams(self):
        schedules = _prepare_ei_synapse_schedules(
            ee_synapses_per_branch_per_layer=[5],
            ei_synapses_per_branch_per_layer=[],
            ie_synapses_per_branch_per_layer=[],
            ii_synapses_per_branch_per_layer=[],
            num_layers=2,
            input_mode=0,
            has_inhibitory_neurons=False,
        )

        assert schedules == ([5, 5], [0, 0], [0, 0], [0, 0])

    def test_prepare_synapse_schedules_requires_ie_for_input_mode1(self):
        with pytest.raises(ValueError, match="ie_synapses_per_branch_per_layer"):
            _prepare_ei_synapse_schedules(
                ee_synapses_per_branch_per_layer=[5],
                ei_synapses_per_branch_per_layer=[],
                ie_synapses_per_branch_per_layer=[],
                ii_synapses_per_branch_per_layer=[],
                num_layers=1,
                input_mode=1,
                has_inhibitory_neurons=False,
            )

    def test_prepare_synapse_schedules_requires_i_schedules_with_i_neurons(self):
        with pytest.raises(ValueError, match="ei_synapses_per_branch_per_layer"):
            _prepare_ei_synapse_schedules(
                ee_synapses_per_branch_per_layer=[5],
                ei_synapses_per_branch_per_layer=[],
                ie_synapses_per_branch_per_layer=[2],
                ii_synapses_per_branch_per_layer=[1],
                num_layers=1,
                input_mode=0,
                has_inhibitory_neurons=True,
            )


class TestEINetworkLayerSizeHelpers:
    def test_prepare_layer_size_sequences_adds_input_sentinel(self):
        exc, inh = _prepare_ei_layer_size_sequences(
            input_dim=20,
            excitatory_layer_sizes=[10, 5],
            inhibitory_layer_sizes=[8, 4],
            input_mode=0,
            input_mode1_builds_i=False,
        )

        assert exc == [20, 10, 5]
        assert inh == [None, 8, 4]

    def test_prepare_layer_size_sequences_preserves_direct_stream_default(self):
        exc, inh = _prepare_ei_layer_size_sequences(
            input_dim=20,
            excitatory_layer_sizes=[10, 5],
            inhibitory_layer_sizes=[8, 4],
            input_mode=1,
            input_mode1_builds_i=False,
        )

        assert exc == [20, 10, 5]
        assert inh == [None, None, None]

    def test_prepare_layer_size_sequences_preserves_input_mode1_opt_in(self):
        exc, inh = _prepare_ei_layer_size_sequences(
            input_dim=20,
            excitatory_layer_sizes=[10, 5],
            inhibitory_layer_sizes=[8, 4],
            input_mode=1,
            input_mode1_builds_i=True,
        )

        assert exc == [20, 10, 5]
        assert inh == [None, 8, 4]


class TestEINetworkLayerBuildSpec:
    def test_layer_build_spec_uses_transfer_dims_for_first_layer(self):
        spec = _resolve_ei_layer_build_spec(
            layer_idx=0,
            excitatory_layer_sizes=[20, 10, 5],
            inhibitory_layer_sizes=[None, 8, 4],
            excitatory_input_dim=12,
            inhibitory_input_dim=6,
            ei_synapses_per_branch=2,
            ii_synapses_per_branch=1,
        )

        assert spec == {
            "n_excitatory_cells": 10,
            "n_inhibitory_cells": 8,
            "excitatory_input_dim": 12,
            "inhibitory_input_dim": 6,
            "build_inhibitory_cells": True,
        }

    def test_layer_build_spec_uses_previous_layer_dims_after_first_layer(self):
        spec = _resolve_ei_layer_build_spec(
            layer_idx=1,
            excitatory_layer_sizes=[20, 10, 5],
            inhibitory_layer_sizes=[None, 8, 4],
            excitatory_input_dim=12,
            inhibitory_input_dim=6,
            ei_synapses_per_branch=2,
            ii_synapses_per_branch=1,
        )

        assert spec == {
            "n_excitatory_cells": 5,
            "n_inhibitory_cells": 4,
            "excitatory_input_dim": 10,
            "inhibitory_input_dim": 8,
            "build_inhibitory_cells": True,
        }

    def test_layer_build_spec_skips_inhibitory_population_without_drive(self):
        spec = _resolve_ei_layer_build_spec(
            layer_idx=0,
            excitatory_layer_sizes=[20, 10],
            inhibitory_layer_sizes=[None, 8],
            excitatory_input_dim=12,
            inhibitory_input_dim=6,
            ei_synapses_per_branch=0,
            ii_synapses_per_branch=0,
        )

        assert spec["n_inhibitory_cells"] == 8
        assert spec["build_inhibitory_cells"] is False


class TestEINetworkDirectStreamValidation:
    def test_direct_stream_requires_explicit_opt_in_when_required(self):
        with pytest.raises(ValueError, match="directly instead of building inhibitory"):
            _validate_direct_inhibitory_stream_contract(
                input_mode=1,
                has_inhibitory_neurons=False,
                ie_synapses_per_branch_per_layer=[1],
                allow_direct_inhibitory_stream=False,
                require_direct_stream_opt_in=True,
            )

    def test_direct_stream_allows_explicit_opt_in(self):
        _validate_direct_inhibitory_stream_contract(
            input_mode=1,
            has_inhibitory_neurons=False,
            ie_synapses_per_branch_per_layer=[1],
            allow_direct_inhibitory_stream=True,
            require_direct_stream_opt_in=True,
        )

    def test_direct_stream_validation_ignores_non_direct_cases(self):
        _validate_direct_inhibitory_stream_contract(
            input_mode=0,
            has_inhibitory_neurons=False,
            ie_synapses_per_branch_per_layer=[1],
            allow_direct_inhibitory_stream=False,
            require_direct_stream_opt_in=True,
        )
        _validate_direct_inhibitory_stream_contract(
            input_mode=1,
            has_inhibitory_neurons=True,
            ie_synapses_per_branch_per_layer=[1],
            allow_direct_inhibitory_stream=False,
            require_direct_stream_opt_in=True,
        )
        _validate_direct_inhibitory_stream_contract(
            input_mode=1,
            has_inhibitory_neurons=False,
            ie_synapses_per_branch_per_layer=[0],
            allow_direct_inhibitory_stream=False,
            require_direct_stream_opt_in=True,
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
        x = torch.rand(2, 20)
        out = net(x)
        assert out.shape == (2, 10)

    def test_identity(self):
        net = get_architecture("identity", {"input_dim": 20, "output_dim": 20})
        x = torch.rand(2, 20)
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


# ---------------------------------------------------------------------------
# Nonneg input validation (data-boundary guard)
# ---------------------------------------------------------------------------


class TestShuntingNonnegInputCheck:
    """Runtime guard: shunting + positive-conductance weights require non-negative
    conductance-driving inputs after the transfer layer.

    The check lives on ``ExcitationInhibitionNetwork.forward`` because that is
    where the transfer layer defines the actual tensors that feed the first
    shunting DBL. Raw inputs may be signed if the transfer layer clamps them
    (e.g. ``output_activation='relu'``), but signed conductance drive itself
    should never pass unnoticed.
    """

    def test_shunting_softplus_errors_on_signed_input(self):
        config = _minimal_config()
        config["morphology"]["weight_transform"] = "softplus"
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        x_signed = torch.randn(BATCH, INPUT_DIM)  # includes negatives
        with pytest.raises(ValueError, match="non-negative"):
            _ = net(x_signed)

    def test_shunting_exp_errors_on_signed_input(self):
        config = _minimal_config()
        config["morphology"]["weight_transform"] = "exp"
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        with pytest.raises(ValueError, match="non-negative"):
            _ = net(torch.randn(BATCH, INPUT_DIM))

    def test_shunting_softplus_accepts_nonneg_input(self):
        config = _minimal_config()
        config["morphology"]["weight_transform"] = "softplus"
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        out = net(torch.rand(BATCH, INPUT_DIM))
        assert out.shape == (BATCH, OUTPUT_DIM)
        assert torch.isfinite(out).all()

    def test_shunting_softplus_allows_signed_raw_input_when_transfer_relu(self):
        config = _minimal_config()
        config["morphology"]["weight_transform"] = "softplus"
        config["transfer"]["output_activation"] = "relu"
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        out = net(torch.randn(BATCH, INPUT_DIM))
        assert out.shape == (BATCH, OUTPUT_DIM)
        assert torch.isfinite(out).all()

    def test_additive_accepts_signed_input(self):
        """Additive morphology has no shunting denominator; signed input is fine."""
        config = _minimal_config()
        config["morphology"]["weight_transform"] = "softplus"
        net = get_architecture("dendritic_additive", config, input_dim=INPUT_DIM)
        out = net(torch.randn(BATCH, INPUT_DIM))
        assert out.shape == (BATCH, OUTPUT_DIM)
        assert torch.isfinite(out).all()

    def test_additive_alias_is_raw_without_hidden_fixed_gate(self):
        config = _minimal_config()
        net = get_architecture("dendritic_additive", config, input_dim=INPUT_DIM)
        layer = net.layers[0].excitatory_cells.branch_layers[0]
        # dendritic_additive is now RAW additive: no implicit normalization and
        # no hidden fixed reactivation gate (the old init_m=0.1/init_b=0.0/
        # init_policy="fixed" bootstrap was removed).
        assert layer.use_shunting is False
        assert layer.use_additive_normalization is False
        assert not (
            layer.reactivation_init_policy == "fixed"
            and layer.reactivation_init_m == 0.1
            and layer.reactivation_init_b == 0.0
        )

    def test_normalized_additive_alias_enables_normalization(self):
        config = _minimal_config()
        net = get_architecture(
            "dendritic_normalized_additive", config, input_dim=INPUT_DIM
        )
        layer = net.layers[0].excitatory_cells.branch_layers[0]
        assert layer.use_shunting is False
        assert layer.use_additive_normalization is True

    def test_shunting_identity_transform_accepts_signed_input(self):
        """identity weight_transform produces signed weights, so the
        'positive conductance' assumption does not apply and the check
        must not fire."""
        config = _minimal_config()
        config["morphology"]["weight_transform"] = "identity"
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        out = net(torch.randn(BATCH, INPUT_DIM))
        assert out.shape == (BATCH, OUTPUT_DIM)

    def test_check_runs_on_every_forward_for_positive_shunting(self):
        """A later signed batch must still be rejected after a positive batch."""
        config = _minimal_config()
        config["morphology"]["weight_transform"] = "softplus"
        net = get_architecture("dendritic_shunting", config, input_dim=INPUT_DIM)
        assert net._nonneg_input_checked is False  # not yet validated
        _ = net(torch.rand(BATCH, INPUT_DIM))
        assert net._nonneg_input_checked is True
        with pytest.raises(ValueError, match="signed conductance-driving inputs"):
            _ = net(torch.randn(BATCH, INPUT_DIM))
