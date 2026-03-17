"""Tests for HierarchicalDendriticConv — spatially local dendritic core."""

import pytest
import torch

from dendritic_modeling.networks.architectures.factory import get_architecture


# ---------------------------------------------------------------------------
# Shared EINet config fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def einet_config():
    """Minimal EINet config with a small spatial block for testing."""
    return {
        "spatial": {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        "architecture": {
            "excitatory_layer_sizes": [32],
            "inhibitory_layer_sizes": [8],
            "excitatory_branch_factors": [2, 2],
            "inhibitory_branch_factors": [2, 2],
            "inhibitory_network_type": "dendritic",
            "mlp_inhibitory_network_params": {
                "hidden_dims": [],
                "activation": "relu",
            },
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [8],
            "ei_synapses_per_branch_per_layer": [4],
            "ie_synapses_per_branch_per_layer": [4],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "input_mode": 0,
            "independent_pathways": False,
            "excitatory_dim": None,
            "inhibitory_dim": None,
            "output_activation": None,
        },
        "morphology": {
            "somatic_synapses": False,
            "use_shunting": True,
            "weight_transform": "softplus",
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {
            "init_method": "xavier_normal",
            "noise_level": 0.0,
            "type": "standard",
            "weight_norm_order": None,
            "gamma": 1.0,
            "temperature": 0.0,
            "gradient_scaling": "none",
        },
        "reactivation": {
            "enabled": True,
            "type": "param_tanh",
            "init_m": 1.5,
            "init_b": 0.5,
            "gradient_scaling": "none",
        },
        "blocklinear": {
            "gradient_scaling": "none",
            "efficient": False,
        },
        "implementation": {
            "adaptive_initialization": False,
            "print_hooks": False,
        },
    }


# ---------------------------------------------------------------------------
# Core forward pass tests
# ---------------------------------------------------------------------------
class TestHierarchicalDendriticConv:
    def test_basic_forward(self, einet_config):
        """Spatial conv should produce (B, C_out, H_out, W_out) output."""
        # in_channels=16, kernel_size=3, stride=1, padding=1 → same spatial size
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.randn(2, 16, 8, 8)
        out = core(x)
        assert out.shape == (2, 32, 8, 8)  # C_out = excitatory_layer_sizes[-1]

    def test_output_channels_match_config(self, einet_config):
        """out_channels should equal excitatory_layer_sizes[-1]."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert core.out_channels == 32
        assert core.output_dim == 32

    def test_stride_reduces_spatial(self, einet_config):
        """Stride > 1 should reduce spatial dimensions."""
        einet_config["spatial"]["stride"] = 2
        einet_config["spatial"]["padding"] = 1

        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.randn(2, 16, 8, 8)
        out = core(x)
        # H_out = (8 + 2*1 - 3) // 2 + 1 = 4
        assert out.shape == (2, 32, 4, 4)

    def test_no_padding(self, einet_config):
        """Without padding, spatial dims shrink by kernel_size - 1."""
        einet_config["spatial"]["padding"] = 0

        core = get_architecture(
            "spatial_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.randn(2, 16, 8, 8)
        out = core(x)
        # H_out = (8 + 0 - 3) // 1 + 1 = 6
        assert out.shape == (2, 32, 6, 6)

    def test_compute_output_shape(self, einet_config):
        """compute_output_shape should match actual forward output."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
            HierarchicalDendriticConv,
        )

        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert isinstance(core, HierarchicalDendriticConv)

        H_out, W_out = core.compute_output_shape(8, 8)
        x = torch.randn(1, 16, 8, 8)
        out = core(x)
        assert out.shape[2] == H_out
        assert out.shape[3] == W_out

    def test_rectangular_input(self, einet_config):
        """Should handle non-square spatial dimensions."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.randn(2, 16, 6, 10)
        out = core(x)
        assert out.shape == (2, 32, 6, 10)  # padding=1, kernel=3 → same size


# ---------------------------------------------------------------------------
# Factory type aliases
# ---------------------------------------------------------------------------
class TestFactoryAliases:
    def test_all_aliases_create_correct_type(self, einet_config):
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
            HierarchicalDendriticConv,
        )

        for alias in [
            "hierarchical_dendritic_conv",
            "spatial_dendritic_conv",
        ]:
            core = get_architecture(alias, einet_config, input_dim=16)
            assert isinstance(
                core, HierarchicalDendriticConv
            ), f"Failed for alias: {alias}"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def test_gradients_flow_through(self, einet_config):
        """Gradients should flow from output back to input via the core."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.randn(2, 16, 4, 4)
        out = core(x)
        loss = out.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in core.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# Training hooks delegation
# ---------------------------------------------------------------------------
class TestTrainingHooks:
    def test_decay_weights(self, einet_config):
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert hasattr(core, "decay_weights")
        core.decay_weights(weight_decay=0.01, weight_boosting=False)

    def test_apply_rewiring(self, einet_config):
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert hasattr(core, "apply_rewiring")
        core.apply_rewiring()

    def test_model_base_class_hooks_discoverable(self, einet_config):
        """models/base.py dispatches via hasattr — ensure hooks are visible."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert callable(core.decay_weights)
        assert callable(core.apply_rewiring)


# ---------------------------------------------------------------------------
# Introspection delegation
# ---------------------------------------------------------------------------
class TestIntrospection:
    def test_branch_layers(self, einet_config):
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert len(core.branch_layers) > 0
        # branch_layers builds a new list each call, so compare contents not identity
        assert len(core.branch_layers) == len(core.einet.branch_layers)

    def test_n_branch_layers(self, einet_config):
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert core.n_branch_layers == core.einet.n_branch_layers
        assert core.n_branch_layers > 0

    def test_layers(self, einet_config):
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        assert core.layers is core.einet.layers
        assert len(core.layers) > 0

    def test_get_effective_params(self, einet_config):
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        eff = core.get_effective_params()
        assert eff > 0
        assert eff == core.einet.get_effective_params()


# ---------------------------------------------------------------------------
# Spatial decoder support (pretrained.py integration)
# ---------------------------------------------------------------------------
@pytest.fixture()
def _skip_if_no_torchvision():
    try:
        from torchvision.models import get_model

        get_model("alexnet", weights=None)
    except Exception:
        pytest.skip("torchvision alexnet not available")


@pytest.mark.usefixtures("_skip_if_no_torchvision")
class TestSpatialDecoderSupport:
    def test_conv_suffix_split_no_longer_raises(self):
        """Splitting mid-conv should now build a spatial decoder, not raise."""
        from dendritic_modeling.networks.architectures.classical.pretrained import (
            build_encoder_decoder_from_plan,
            build_split_plan,
        )

        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="features.6",
            replace_end="features.8",
            spatial=True,
        )

        if plan.suffix_input_spec is not None:
            # Should NOT raise anymore
            encoder, decoder = build_encoder_decoder_from_plan(plan)
            assert encoder.output_spec is not None
            assert decoder.output_dim > 0

    def test_spatial_decoder_input_shape(self):
        """Spatial decoder should accept 4D input matching suffix_input_spec."""
        from dendritic_modeling.networks.architectures.classical.pretrained import (
            build_encoder_decoder_from_plan,
            build_split_plan,
        )

        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="features.6",
            replace_end="features.8",
            spatial=True,
        )

        if plan.suffix_input_spec is not None:
            _encoder, decoder = build_encoder_decoder_from_plan(plan)
            spec = plan.suffix_input_spec

            # Decoder should work with spatial input
            x = torch.randn(2, spec.channels, spec.height, spec.width)
            with torch.no_grad():
                out = decoder(x)
            assert out.dim() == 2  # should flatten to (B, num_classes)


# ---------------------------------------------------------------------------
# 4D input guard
# ---------------------------------------------------------------------------
class TestInputGuard:
    def test_rejects_2d_input(self, einet_config):
        """Flat (B, D) input should raise a clear ValueError."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.randn(2, 144)  # patch_dim = 16*3*3 = 144
        with pytest.raises(ValueError, match="requires 4D input"):
            core(x)

    def test_rejects_3d_input(self, einet_config):
        """3D input should also raise."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.randn(2, 16, 8)
        with pytest.raises(ValueError, match="requires 4D input"):
            core(x)


# ---------------------------------------------------------------------------
# Config-path tests: load_config → verify spatial settings survive
# ---------------------------------------------------------------------------
class TestConfigPath:
    def test_spatial_config_survives_load(self):
        """core.spatial.{kernel_size, stride, padding} must survive load_config."""
        from dendritic_modeling.config import load_config

        cfg = load_config("configs/vision/alexnet_conv_dendritic_conv.yaml")

        # SpatialConfig should be populated from the YAML
        assert cfg.model.core.spatial.kernel_size == 3
        assert cfg.model.core.spatial.stride == 1
        assert cfg.model.core.spatial.padding == 1

    def test_spatial_config_in_plain_dict(self):
        """_config_to_plain_dict should include core.spatial as a dict."""
        from dendritic_modeling.config import load_config

        cfg = load_config("configs/vision/alexnet_conv_dendritic_conv.yaml")

        from dataclasses import asdict

        core_dict = asdict(cfg.model.core)
        assert "spatial" in core_dict
        assert core_dict["spatial"]["kernel_size"] == 3
        assert core_dict["spatial"]["padding"] == 1


# ---------------------------------------------------------------------------
# Spatial contract validation tests
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("_skip_if_no_torchvision")
class TestSpatialContractValidation:
    def test_flat_suffix_with_spatial_conv_raises(self):
        """Using spatial conv core on a flat-suffix split should raise ValueError."""
        from dendritic_modeling.config import load_config

        # Load the tensor_map config (flat suffix) and switch type to spatial conv.
        # Force weights=None so this test works offline without cached weights.
        cfg = load_config("configs/vision/alexnet_k2_tensor_map.yaml")
        cfg.model.pretrained_replacement.weights = None
        cfg.model.core.type = "hierarchical_dendritic_conv"
        cfg.model.core.spatial.kernel_size = 3
        cfg.model.core.spatial.stride = 1
        cfg.model.core.spatial.padding = 1

        from dendritic_modeling.scripts.script_utils.setup_utils import (
            _build_from_pretrained_replacement,
        )

        with pytest.raises(ValueError, match="suffix expects flat input"):
            _build_from_pretrained_replacement(cfg.model)

    def test_valid_spatial_split_builds_cleanly(self, einet_config):
        """A correctly configured spatial conv split should build without error."""
        from dendritic_modeling.networks.architectures.classical.pretrained import (
            build_split_plan,
        )

        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="features.6",
            replace_end="features.8",
            spatial=True,
        )

        if plan.prefix_output_spec is not None and plan.suffix_input_spec is not None:
            # Build a core that matches the suffix contract
            spec = plan.suffix_input_spec
            config = dict(einet_config)
            config["spatial"] = {
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            }
            config["architecture"] = dict(config["architecture"])
            config["architecture"]["excitatory_layer_sizes"] = [spec.channels]

            core = get_architecture(
                "hierarchical_dendritic_conv",
                config,
                input_dim=plan.prefix_output_spec.channels,
            )

            assert core.out_channels == spec.channels
            H_out, W_out = core.compute_output_shape(
                plan.prefix_output_spec.height,
                plan.prefix_output_spec.width,
            )
            assert H_out == spec.height
            assert W_out == spec.width

    def test_end_to_end_spatial_compatibility(self):
        """Full pipeline: spatial encoder → dendritic conv → spatial decoder on AlexNet."""
        from dendritic_modeling.networks.architectures.classical.pretrained import (
            build_encoder_decoder_from_plan,
            build_split_plan,
        )

        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="features.6",
            replace_end="features.8",
            spatial=True,
        )

        if plan.prefix_output_spec is None or plan.suffix_input_spec is None:
            pytest.skip("Split did not produce spatial specs")

        encoder, decoder = build_encoder_decoder_from_plan(plan, prefix_freeze=True)

        # Build a matching core: C_out = suffix channels, same spatial size
        einet_config = {
            "spatial": {"kernel_size": 3, "stride": 1, "padding": 1},
            "architecture": {
                "excitatory_layer_sizes": [plan.suffix_input_spec.channels],
                "inhibitory_layer_sizes": [16],
                "excitatory_branch_factors": [2],
                "inhibitory_branch_factors": [2],
                "inhibitory_network_type": "dendritic",
                "mlp_inhibitory_network_params": {
                    "hidden_dims": [],
                    "activation": "relu",
                },
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [8],
                "ei_synapses_per_branch_per_layer": [4],
                "ie_synapses_per_branch_per_layer": [4],
                "ii_synapses_per_branch_per_layer": [0],
            },
            "transfer": {
                "input_mode": 0,
                "independent_pathways": False,
                "excitatory_dim": None,
                "inhibitory_dim": None,
                "output_activation": None,
            },
            "morphology": {
                "somatic_synapses": False,
                "use_shunting": True,
                "weight_transform": "softplus",
                "dbl_init_method": "analytical_expectation",
            },
            "sparsity": {
                "init_method": "xavier_normal",
                "noise_level": 0.0,
                "type": "standard",
                "weight_norm_order": None,
                "gamma": 1.0,
                "temperature": 0.0,
                "gradient_scaling": "none",
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.5,
                "init_b": 0.5,
                "gradient_scaling": "none",
            },
            "blocklinear": {"gradient_scaling": "none", "efficient": False},
            "implementation": {
                "adaptive_initialization": False,
                "print_hooks": False,
            },
        }

        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=plan.prefix_output_spec.channels,
        )

        # Verify dimensions match
        assert core.out_channels == plan.suffix_input_spec.channels
        H_out, W_out = core.compute_output_shape(
            plan.prefix_output_spec.height,
            plan.prefix_output_spec.width,
        )
        assert H_out == plan.suffix_input_spec.height
        assert W_out == plan.suffix_input_spec.width

        # Full forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            encoded = encoder(x)
        assert encoded.dim() == 4
        assert encoded.shape[1] == plan.prefix_output_spec.channels

        core_out = core(encoded)
        assert core_out.shape == (
            1,
            plan.suffix_input_spec.channels,
            plan.suffix_input_spec.height,
            plan.suffix_input_spec.width,
        )

        logits = decoder(core_out)
        assert logits.shape == (1, 1000)
