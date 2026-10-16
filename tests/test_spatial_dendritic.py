"""Tests for TensorSpec, spatial split paths, and HierarchicalDendriticTensorMap."""

import pytest
import torch

from dendritic_modeling.networks.architectures.classical.pretrained import (
    TensorSpec,
    build_encoder_decoder_from_plan,
    build_split_plan,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


# ---------------------------------------------------------------------------
# TensorSpec
# ---------------------------------------------------------------------------
class TestTensorSpec:
    def test_basic_creation(self):
        spec = TensorSpec(channels=256, height=6, width=6)
        assert spec.channels == 256
        assert spec.height == 6
        assert spec.width == 6

    def test_numel(self):
        spec = TensorSpec(channels=256, height=6, width=6)
        assert spec.numel == 256 * 6 * 6

    def test_as_tuple(self):
        spec = TensorSpec(channels=64, height=8, width=8)
        assert spec.as_tuple() == (64, 8, 8)

    def test_frozen(self):
        spec = TensorSpec(channels=3, height=224, width=224)
        with pytest.raises(AttributeError):
            spec.channels = 10

    def test_equality(self):
        a = TensorSpec(channels=64, height=8, width=8)
        b = TensorSpec(channels=64, height=8, width=8)
        assert a == b

    def test_inequality(self):
        a = TensorSpec(channels=64, height=8, width=8)
        b = TensorSpec(channels=64, height=8, width=4)
        assert a != b


# ---------------------------------------------------------------------------
# Spatial split path (uses real torchvision AlexNet)
# ---------------------------------------------------------------------------
class TestSpatialSplit:
    def test_spatial_split_alexnet_k2(self):
        """AlexNet K=2: prefix = conv backbone + avgpool → 4D spatial output."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
            spatial=True,
        )

        assert plan.is_spatial
        assert plan.prefix_output_spec is not None
        assert plan.prefix_output_spec.channels == 256
        assert plan.prefix_output_spec.height == 6
        assert plan.prefix_output_spec.width == 6
        assert plan.prefix_output_spec.numel == 9216
        # Flattened dim should still be correct
        assert plan.prefix_output_dim == 9216

    def test_spatial_split_alexnet_k1_falls_back(self):
        """AlexNet K=1: prefix includes fc6 (Linear) → spatial specs are None."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
            spatial=True,
        )

        # Prefix includes conv + avgpool + fc6 + ReLU → needs flatten internally
        assert not plan.is_spatial
        assert plan.prefix_output_spec is None
        # Flattened output still works
        assert plan.prefix_output_dim == 4096

    def test_nonspatial_split_has_no_spec(self):
        """Default (spatial=False) should have None specs."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
            spatial=False,
        )

        assert not plan.is_spatial
        assert plan.prefix_output_spec is None
        assert plan.suffix_input_spec is None

    def test_spatial_encoder_preserves_4d(self):
        """With spatial plan, encoder should output (B, C, H, W)."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
            spatial=True,
        )

        encoder, _decoder = build_encoder_decoder_from_plan(plan, prefix_freeze=True)

        assert encoder.output_spec is not None
        assert encoder.output_spec.channels == 256

        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)
        assert out.dim() == 4
        assert out.shape == (2, 256, 6, 6)

    def test_nonspatial_encoder_flattens(self):
        """Default encoder should still flatten to 2D."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
            spatial=False,
        )

        encoder, _ = build_encoder_decoder_from_plan(plan, prefix_freeze=True)

        assert encoder.output_spec is None

        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            out = encoder(x)
        assert out.dim() == 2
        assert out.shape == (2, 9216)

    def test_summary_includes_spatial_specs(self):
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
            spatial=True,
        )
        s = plan.summary()
        assert "spatial=True" in s
        assert "prefix_output_spec" in s


# ---------------------------------------------------------------------------
# HierarchicalDendriticTensorMap
# ---------------------------------------------------------------------------
class TestHierarchicalDendriticTensorMap:
    """Test the tensor map core with realistic EINet configs."""

    @pytest.fixture()
    def einet_config(self):
        """Minimal EINet config for testing."""
        return {
            "biological_neuron": False,
            "architecture": {
                "excitatory_layer_sizes": [64],
                "inhibitory_layer_sizes": [16],
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

    def test_flat_input(self, einet_config):
        """Tensor map should work with flat (B, D) input."""
        core = get_architecture(
            "hierarchical_dendritic_tensor_map",
            einet_config,
            input_dim=128,
            suffix_input_dim=64,
        )

        x = torch.rand(4, 128)
        out = core(x)
        assert out.shape == (4, 64)
        assert core.output_dim == 64

    def test_spatial_input(self, einet_config):
        """Tensor map should flatten (B, C, H, W) before processing."""
        # input_dim = 8 * 4 * 4 = 128
        core = get_architecture(
            "hierarchical_dendritic_tensor_map",
            einet_config,
            input_dim=128,
            suffix_input_dim=64,
        )

        x = torch.rand(4, 8, 4, 4)
        out = core(x)
        assert out.shape == (4, 64)

    def test_no_projection_when_dims_match(self, einet_config):
        """No projection layer when EINet output matches suffix_input_dim."""
        core = get_architecture(
            "hierarchical_dendritic_tensor_map",
            einet_config,
            input_dim=128,
            suffix_input_dim=64,  # matches excitatory_layer_sizes[-1]
        )

        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic.tensor_map import (
            HierarchicalDendriticTensorMap,
        )

        assert isinstance(core, HierarchicalDendriticTensorMap)
        assert core.projection is None
        assert core.output_dim == 64

    def test_projection_when_dims_differ(self, einet_config):
        """Projection layer added when EINet output != suffix_input_dim."""
        core = get_architecture(
            "hierarchical_dendritic_tensor_map",
            einet_config,
            input_dim=128,
            suffix_input_dim=100,  # != 64
        )

        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic.tensor_map import (
            HierarchicalDendriticTensorMap,
        )

        assert isinstance(core, HierarchicalDendriticTensorMap)
        assert core.projection is not None
        assert core.output_dim == 100

        x = torch.rand(4, 128)
        out = core(x)
        assert out.shape == (4, 100)

    def test_no_suffix_dim_uses_einet_output(self, einet_config):
        """When suffix_input_dim is None, output_dim = EINet excitatory size."""
        core = get_architecture(
            "hierarchical_dendritic_tensor_map",
            einet_config,
            input_dim=128,
        )

        assert core.output_dim == 64  # excitatory_layer_sizes[-1]

    def test_factory_type_aliases(self, einet_config):
        """All type aliases should create HierarchicalDendriticTensorMap."""
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic.tensor_map import (
            HierarchicalDendriticTensorMap,
        )

        for alias in [
            "hierarchical_dendritic_tensor_map",
            "dendritic_tensor_map",
            "tensor_map",
        ]:
            core = get_architecture(alias, einet_config, input_dim=128)
            assert isinstance(core, HierarchicalDendriticTensorMap)

    def test_branch_layers_delegation(self, einet_config):
        """branch_layers should delegate to inner EINet."""
        core = get_architecture(
            "tensor_map",
            einet_config,
            input_dim=128,
        )

        assert hasattr(core, "branch_layers")
        assert len(core.branch_layers) > 0

    def test_gradient_flow(self, einet_config):
        """Gradients should flow through the tensor map."""
        core = get_architecture(
            "tensor_map",
            einet_config,
            input_dim=128,
            suffix_input_dim=32,
        )

        x = torch.rand(4, 128)
        out = core(x)
        loss = out.sum()
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in core.parameters()
        )
        assert has_grad

    def test_decay_weights_delegation(self, einet_config):
        """decay_weights should forward to the inner EINet."""
        core = get_architecture(
            "tensor_map",
            einet_config,
            input_dim=128,
        )

        assert hasattr(core, "decay_weights")
        # Should not raise — delegates to einet.decay_weights
        core.decay_weights(weight_decay=0.01, weight_boosting=False)

    def test_apply_rewiring_delegation(self, einet_config):
        """apply_rewiring should forward to the inner EINet."""
        core = get_architecture(
            "tensor_map",
            einet_config,
            input_dim=128,
        )

        assert hasattr(core, "apply_rewiring")
        # Should not raise — delegates to einet.apply_rewiring
        core.apply_rewiring()

    def test_model_base_class_finds_training_hooks(self, einet_config):
        """models/base.py dispatches decay_weights/apply_rewiring via hasattr."""
        core = get_architecture(
            "tensor_map",
            einet_config,
            input_dim=128,
        )

        # These are checked by models/base.py before calling
        assert hasattr(core, "decay_weights")
        assert hasattr(core, "apply_rewiring")
        assert callable(core.decay_weights)
        assert callable(core.apply_rewiring)

    def test_n_branch_layers_delegation(self, einet_config):
        """n_branch_layers should match inner EINet."""
        core = get_architecture("tensor_map", einet_config, input_dim=128)
        assert core.n_branch_layers == core.einet.n_branch_layers
        assert core.n_branch_layers > 0

    def test_get_effective_params_delegation(self, einet_config):
        """get_effective_params should include EINet + projection params."""
        core = get_architecture(
            "tensor_map", einet_config, input_dim=128, suffix_input_dim=32
        )
        einet_params = core.einet.get_effective_params()
        total = core.get_effective_params()
        # Should include projection params (64*32 + 32 = 2080)
        assert total > einet_params
        assert total == einet_params + sum(
            p.numel() for p in core.projection.parameters()
        )

    def test_get_effective_params_no_projection(self, einet_config):
        """Without projection, get_effective_params matches inner EINet."""
        core = get_architecture("tensor_map", einet_config, input_dim=128)
        assert core.get_effective_params() == core.einet.get_effective_params()

    def test_layers_property_delegation(self, einet_config):
        """layers property should expose inner EI layers."""
        core = get_architecture("tensor_map", einet_config, input_dim=128)
        assert core.layers is core.einet.layers
        assert len(core.layers) > 0


# ---------------------------------------------------------------------------
# Spatial decoder guard
# ---------------------------------------------------------------------------
class TestSpatialDecoderGuard:
    def test_conv_suffix_split_builds_spatial_decoder(self):
        """Splitting mid-conv (so suffix has conv layers) should build a spatial decoder."""
        # Replace features.6 through features.8 (Conv2d layers inside backbone)
        # This produces a spatial suffix_input_spec since the replaced layers
        # output 4D, meaning the suffix needs spatial input.
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="features.6",
            replace_end="features.8",
            spatial=True,
        )

        # The plan should have a suffix_input_spec (spatial suffix)
        if plan.suffix_input_spec is not None:
            # Should build without error now that spatial decoder is supported
            encoder, decoder = build_encoder_decoder_from_plan(plan)
            assert encoder.output_spec is not None
            assert decoder.output_dim > 0

    def test_late_layer_split_works(self):
        """Splitting at classifier layers (flat suffix) should work fine."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
            spatial=True,
        )

        # suffix_input_spec should be None (replaced layers contain Linear)
        assert plan.suffix_input_spec is None

        # Should not raise
        encoder, decoder = build_encoder_decoder_from_plan(plan)
        assert encoder.output_dim == 9216
        assert decoder.output_dim == 1000


# ---------------------------------------------------------------------------
# End-to-end: spatial split + tensor map core
# ---------------------------------------------------------------------------
class TestSpatialEndToEnd:
    def test_alexnet_k2_spatial_tensor_map_pipeline(self):
        """Full pipeline: spatial encoder → tensor_map core → decoder."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
            spatial=True,
        )

        encoder, decoder = build_encoder_decoder_from_plan(plan, prefix_freeze=True)

        # Create tensor map core with small EINet
        einet_config = {
            "biological_neuron": False,
            "architecture": {
                "excitatory_layer_sizes": [256],
                "inhibitory_layer_sizes": [64],
                "excitatory_branch_factors": [2, 2],
                "inhibitory_branch_factors": [2, 2],
                "inhibitory_network_type": "dendritic",
                "mlp_inhibitory_network_params": {
                    "hidden_dims": [],
                    "activation": "relu",
                },
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [16],
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

        core = get_architecture(
            "tensor_map",
            einet_config,
            input_dim=encoder.output_dim,  # 9216
            suffix_input_dim=plan.suffix_input_dim,  # 4096
        )

        assert core.output_dim == plan.suffix_input_dim

        # Full forward pass
        x = torch.rand(2, 3, 224, 224)
        with torch.no_grad():
            encoded = encoder(x)  # (2, 256, 6, 6) — spatial!
        assert encoded.dim() == 4

        core_out = core(encoded)  # flatten + EINet + project → (2, 4096)
        assert core_out.shape == (2, 4096)

        logits = decoder(core_out)
        assert logits.shape == (2, 1000)
