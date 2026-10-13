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

        x = torch.rand(2, 16, 8, 8)
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

    def test_output_contract_is_channel_aligned(self, einet_config):
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        assert core.spatial_output_contract() == {
            "input_layout": "BCHW",
            "patch_layout": "channel_y_x_flat",
            "output_layout": "BCHW",
            "spatial_weight_sharing": True,
            "input_transform": "identity",
            "input_scale": 1.0,
            "adapted_input_channels": 16,
            "output_channels": 32,
            "excitatory_somas_per_location": 32,
            "channel_mapping": "one_excitatory_soma_per_output_channel",
            "propagated_population": "excitatory",
            "inhibitory_population_scope": "local_per_output_position",
            "output_adapter": "identity",
        }

    def test_input_scale_rescales_patch_drive(self, einet_config):
        einet_config["spatial"]["input_scale"] = 0.25
        scaled_core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )
        reference_config = {
            **einet_config,
            "spatial": {**einet_config["spatial"], "input_scale": 1.0},
        }
        reference_core = get_architecture(
            "hierarchical_dendritic_conv",
            reference_config,
            input_dim=16,
        )
        reference_core.load_state_dict(scaled_core.state_dict())

        x = torch.rand(2, 16, 8, 8)
        torch.testing.assert_close(scaled_core(x), reference_core(0.25 * x))
        assert scaled_core.spatial_output_contract()["input_scale"] == 0.25
        assert (
            scaled_core.get_effective_params() == reference_core.get_effective_params()
        )

    def test_input_scale_must_be_positive(self, einet_config):
        einet_config["spatial"]["input_scale"] = 0.0
        with pytest.raises(ValueError, match="input_scale"):
            get_architecture(
                "hierarchical_dendritic_conv",
                einet_config,
                input_dim=16,
            )

    def test_threshold_relu_output_adapter_is_config_selectable(self, einet_config):
        einet_config["spatial"][
            "output_adapter_mode"
        ] = "learned_per_channel_threshold_relu"
        einet_config["spatial"]["output_initial_threshold"] = 10.0
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        output = core(torch.rand(2, 16, 8, 8))

        assert torch.count_nonzero(output) == 0
        assert core.spatial_output_contract()["output_adapter"] == (
            "learned_per_channel_threshold_relu"
        )
        assert core.output_adapter.log_scale.numel() == core.out_channels
        identity_config = {
            **einet_config,
            "spatial": {
                **einet_config["spatial"],
                "output_adapter_mode": "identity",
            },
        }
        identity_core = get_architecture(
            "hierarchical_dendritic_conv",
            identity_config,
            input_dim=16,
        )
        assert core.get_effective_params() == (
            identity_core.get_effective_params() + 2 * core.out_channels
        )

    def test_signed_split_preserves_signed_spatial_input(self, einet_config):
        einet_config["spatial"]["input_transform"] = "signed_split"
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=3,
        )
        inputs = torch.randn(2, 3, 8, 8)

        adapted = core.input_adapter(inputs)

        assert adapted.shape == (2, 6, 8, 8)
        torch.testing.assert_close(adapted[:, :3] - adapted[:, 3:], inputs)
        assert core.einet.input_dim == 6 * 3 * 3

    def test_identical_patches_emit_identical_channel_vectors(self, einet_config):
        einet_config["spatial"]["padding"] = 0
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        output = core(torch.ones(1, 16, 5, 5))
        reference = output[:, :, :1, :1].expand_as(output)
        torch.testing.assert_close(output, reference)

    def test_stride_reduces_spatial(self, einet_config):
        """Stride > 1 should reduce spatial dimensions."""
        einet_config["spatial"]["stride"] = 2
        einet_config["spatial"]["padding"] = 1

        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.rand(2, 16, 8, 8)
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

        x = torch.rand(2, 16, 8, 8)
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
        x = torch.rand(1, 16, 8, 8)
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

        x = torch.rand(2, 16, 6, 10)
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

    def test_sparse_active_matched_point_conv_matches_reference(self, einet_config):
        """The spatial point control shares patches and matches active scalars."""

        einet_config["sparsity"] = {
            **einet_config["sparsity"],
            "type": "indexed",
            "indexed": {"seed": 17, "output_chunk_size": 32},
        }
        reference = get_architecture(
            "hierarchical_dendritic_conv", einet_config, input_dim=16
        )
        point = get_architecture(
            "sparse_active_matched_point_conv", einet_config, input_dim=16
        )
        x = torch.rand(2, 16, 8, 8)
        assert point(x).shape == reference(x).shape == (2, 32, 8, 8)
        assert point.get_effective_params() == reference.get_effective_params()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------
class TestGradientFlow:
    def test_indexed_backend_config_reaches_shared_patch_einet(self, einet_config):
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
            IndexedSparseLinear,
        )

        einet_config["sparsity"] = {
            **einet_config["sparsity"],
            "type": "indexed",
            "indexed": {
                "seed": 17,
                "index_dtype": "int32",
                "projection_backend": "auto",
            },
        }
        einet_config["reactivation"] = {
            **einet_config["reactivation"],
            "memory_efficient": True,
        }
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=4,
        )
        value = torch.rand(2, 4, 5, 5, requires_grad=True)

        core(value).square().mean().backward()

        indexed_layers = [
            module
            for module in core.modules()
            if isinstance(module, IndexedSparseLinear)
        ]
        assert indexed_layers
        assert all(layer.projection_backend == "auto" for layer in indexed_layers)
        assert all(
            layer._last_resolved_projection_backend == "recompute"
            for layer in indexed_layers
        )
        assert value.grad is not None

    def test_gradients_flow_through(self, einet_config):
        """Gradients should flow from output back to input via the core."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.rand(2, 16, 4, 4)
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
            x = torch.rand(2, spec.channels, spec.height, spec.width)
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

        x = torch.rand(2, 144)  # patch_dim = 16*3*3 = 144
        with pytest.raises(ValueError, match="requires 4D input"):
            core(x)

    def test_rejects_3d_input(self, einet_config):
        """3D input should also raise."""
        core = get_architecture(
            "hierarchical_dendritic_conv",
            einet_config,
            input_dim=16,
        )

        x = torch.rand(2, 16, 8)
        with pytest.raises(ValueError, match="requires 4D input"):
            core(x)


# ---------------------------------------------------------------------------
# Config-path tests: load_config → verify spatial settings survive
# ---------------------------------------------------------------------------
class TestConfigPath:
    def test_spatial_config_survives_load(self, tmp_path):
        """core.spatial.{kernel_size, stride, padding} must survive load_config."""
        import yaml

        from dendritic_modeling.config import load_config

        # Create a minimal config with spatial settings inline
        cfg_dict = {
            "experiment": {"seed": 42, "train_valid_split": 0.8},
            "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
            "model": {
                "task": "classification",
                "encoder": {"type": "identity"},
                "core": {
                    "type": "hierarchical_dendritic_conv",
                    "spatial": {"kernel_size": 3, "stride": 1, "padding": 1},
                    "architecture": {"excitatory_layer_sizes": [32]},
                },
                "decoder": {"type": "MLP", "params": {"output_dim": 10}},
            },
            "training": {"main": {"strategy": "standard", "common": {"epochs": 1}}},
            "outputs": {"run_name": "test", "results_dir": str(tmp_path)},
        }
        cfg_path = tmp_path / "spatial_test.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict))

        cfg = load_config(str(cfg_path))

        assert cfg.model.core.spatial.kernel_size == 3
        assert cfg.model.core.spatial.stride == 1
        assert cfg.model.core.spatial.padding == 1

    def test_spatial_config_in_plain_dict(self, tmp_path):
        """_config_to_plain_dict should include core.spatial as a dict."""
        from dataclasses import asdict

        import yaml

        from dendritic_modeling.config import load_config

        cfg_dict = {
            "experiment": {"seed": 42, "train_valid_split": 0.8},
            "data": {"dataset_name": "mnist", "processing": {"flatten": True}},
            "model": {
                "task": "classification",
                "encoder": {"type": "identity"},
                "core": {
                    "type": "hierarchical_dendritic_conv",
                    "spatial": {"kernel_size": 3, "stride": 1, "padding": 1},
                    "architecture": {"excitatory_layer_sizes": [32]},
                },
                "decoder": {"type": "MLP", "params": {"output_dim": 10}},
            },
            "training": {"main": {"strategy": "standard", "common": {"epochs": 1}}},
            "outputs": {"run_name": "test", "results_dir": str(tmp_path)},
        }
        cfg_path = tmp_path / "spatial_test.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict))

        cfg = load_config(str(cfg_path))

        core_dict = asdict(cfg.model.core)
        assert "spatial" in core_dict
        assert core_dict["spatial"]["kernel_size"] == 3
        assert core_dict["spatial"]["padding"] == 1


# ---------------------------------------------------------------------------
# Spatial contract validation tests
# ---------------------------------------------------------------------------
class TestSpatialContractValidation:
    def test_enabled_extension_section_is_rejected_before_its_split(self, tmp_path):
        """An enabled extension section fails before its split is validated."""
        import yaml

        from dendritic_modeling.config import load_config

        # Standalone spatial construction is tested below.
        cfg_dict = {
            "experiment": {"seed": 42, "train_valid_split": 0.8},
            "data": {
                "dataset_name": "imagenet",
                "processing": {"flatten": False, "normalize": True},
            },
            "model": {
                "task": "classification",
                "pretrained_replacement": {"enabled": True},
                "encoder": {"type": "identity"},
                "core": {
                    "type": "hierarchical_dendritic_conv",
                    "spatial": {"kernel_size": 3, "stride": 1, "padding": 1},
                    "architecture": {"excitatory_layer_sizes": [32]},
                },
                "decoder": {"type": "MLP", "params": {"output_dim": 1000}},
            },
            "training": {"main": {"strategy": "standard", "common": {"epochs": 1}}},
            "outputs": {"run_name": "test", "results_dir": str(tmp_path)},
        }
        cfg_path = tmp_path / "flat_suffix_test.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict))

        with pytest.raises(ValueError, match="DendriNet plugin"):
            load_config(str(cfg_path))

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

        assert plan.prefix_output_spec is not None
        assert plan.suffix_input_spec is not None

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
        x = torch.rand(1, 3, 224, 224)
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
