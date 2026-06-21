"""Tests for the pretrained backbone splitter (build_split_plan API)."""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.pretrained import (
    BackboneSegment,
    _extract_layers,
    _filter_omitted_layers,
    _flatten_model_layers,
    _has_linear_module,
    _has_spatial_module,
    _insert_flatten,
    _load_backbone,
    _names_and_modules,
    _needs_flatten,
    _prepare_segment_input,
    _require_layer_index,
    _require_split_index,
    _resolve_target_module_span,
    _restore_segment_output,
    _run_modules_no_grad,
    _segment_output_metadata,
    _select_layers_for_boundary,
    _slice_layers_by_span,
    _tensor_spec_from_spatial_output,
    build_encoder_decoder_from_plan,
    build_split_plan,
    list_backbone_layers,
)
from dendritic_modeling.networks.base import BaseNetwork


# ---------------------------------------------------------------------------
# Helper: build a minimal AlexNet-like model without downloading weights
# ---------------------------------------------------------------------------
class _MiniAlexNet(nn.Module):
    """Tiny model with the same named structure as torchvision AlexNet."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),  # 0
            nn.ReLU(),  # 1
            nn.MaxPool2d(2),  # 2
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(),  # 0
            nn.Linear(32, 16),  # 1  (fc6)
            nn.ReLU(),  # 2
            nn.Dropout(),  # 3
            nn.Linear(16, 16),  # 4  (fc7)
            nn.ReLU(),  # 5
            nn.Linear(16, 10),  # 6  (fc8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class _SimpleCore(BaseNetwork):
    """Minimal core network stand-in for testing the pipeline."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x)

    def get_output_dim(self):
        return self.output_dim


# ---------------------------------------------------------------------------
# Tests for _flatten_model_layers
# ---------------------------------------------------------------------------
class TestFlattenModelLayers:
    def test_expands_sequential(self):
        model = _MiniAlexNet()
        layers = _flatten_model_layers(model)
        names = [n for n, _ in layers]

        # Sequential children should be expanded
        assert "features.0" in names
        assert "features.1" in names
        assert "features.2" in names
        assert "avgpool" in names
        assert "classifier.0" in names
        assert "classifier.6" in names

        # Top-level Sequential names should NOT appear
        assert "features" not in names
        assert "classifier" not in names

    def test_preserves_order(self):
        model = _MiniAlexNet()
        layers = _flatten_model_layers(model)
        names = [n for n, _ in layers]

        # features before avgpool before classifier
        assert names.index("features.0") < names.index("avgpool")
        assert names.index("avgpool") < names.index("classifier.0")


# ---------------------------------------------------------------------------
# Tests for _extract_layers
# ---------------------------------------------------------------------------
class TestExtractLayers:
    def test_split_after(self):
        model = _MiniAlexNet()
        _modules, names = _extract_layers(model, split_after="classifier.2")

        assert "classifier.2" in names
        assert "classifier.4" not in names
        assert "classifier.6" not in names

    def test_split_from(self):
        model = _MiniAlexNet()
        modules, names = _extract_layers(model, split_from="classifier.6")

        assert names == ["classifier.6"]
        assert isinstance(modules[0], nn.Linear)

    def test_drop_layers(self):
        model = _MiniAlexNet()
        _modules, names = _extract_layers(
            model,
            split_after="classifier.2",
            drop_layers=["classifier.0"],
        )

        assert "classifier.0" not in names
        # fc6 and ReLU should still be there
        assert "classifier.1" in names
        assert "classifier.2" in names

    def test_invalid_split_after_raises(self):
        model = _MiniAlexNet()
        with pytest.raises(ValueError, match="not found"):
            _extract_layers(model, split_after="nonexistent_layer")

    def test_invalid_split_from_raises(self):
        model = _MiniAlexNet()
        with pytest.raises(ValueError, match="not found"):
            _extract_layers(model, split_from="nonexistent_layer")

    def test_neither_split_raises(self):
        model = _MiniAlexNet()
        with pytest.raises(ValueError, match="Either split_after or split_from"):
            _extract_layers(model)

    def test_split_after_avgpool(self):
        model = _MiniAlexNet()
        _modules, names = _extract_layers(model, split_after="avgpool")

        assert names[-1] == "avgpool"
        assert "classifier.0" not in names

    def test_select_layers_for_boundary_uses_split_after(self):
        layers = [(f"layer.{idx}", nn.Identity()) for idx in range(4)]
        names = [name for name, _ in layers]

        selected = _select_layers_for_boundary(layers, names, split_after="layer.1")

        assert [name for name, _ in selected] == ["layer.0", "layer.1"]

    def test_select_layers_for_boundary_uses_split_from(self):
        layers = [(f"layer.{idx}", nn.Identity()) for idx in range(4)]
        names = [name for name, _ in layers]

        selected = _select_layers_for_boundary(layers, names, split_from="layer.2")

        assert [name for name, _ in selected] == ["layer.2", "layer.3"]

    def test_select_layers_for_boundary_preserves_split_after_precedence(self):
        layers = [(f"layer.{idx}", nn.Identity()) for idx in range(4)]
        names = [name for name, _ in layers]

        selected = _select_layers_for_boundary(
            layers,
            names,
            split_after="layer.1",
            split_from="layer.3",
        )

        assert [name for name, _ in selected] == ["layer.0", "layer.1"]

    def test_require_split_index_returns_index_or_legacy_error(self):
        names = ["layer.0", "layer.1"]

        assert _require_split_index(names, "layer.1", arg_name="split_after") == 1
        with pytest.raises(ValueError, match="split_from='missing' not found"):
            _require_split_index(names, "missing", arg_name="split_from")


# ---------------------------------------------------------------------------
# Tests for flatten insertion helpers
# ---------------------------------------------------------------------------
class TestFlattenHelpers:
    def test_module_type_predicates_detect_spatial_and_linear_layers(self):
        spatial_modules = [nn.Conv2d(3, 4, 1), nn.ReLU()]
        linear_modules = [nn.ReLU(), nn.Linear(4, 2)]

        assert _has_spatial_module(spatial_modules)
        assert not _has_spatial_module(linear_modules)
        assert _has_linear_module(linear_modules)
        assert not _has_linear_module(spatial_modules)

    def test_needs_flatten_only_for_mixed_spatial_and_linear_stacks(self):
        assert _needs_flatten([nn.Conv2d(3, 4, 1), nn.Flatten(), nn.Linear(4, 2)])
        assert not _needs_flatten([nn.Conv2d(3, 4, 1), nn.ReLU()])
        assert not _needs_flatten([nn.Linear(4, 2), nn.ReLU()])

    def test_insert_flatten_before_first_linear_only(self):
        conv = nn.Conv2d(3, 4, 1)
        linear1 = nn.Linear(4, 3)
        linear2 = nn.Linear(3, 2)

        modules = _insert_flatten([conv, linear1, nn.ReLU(), linear2])

        assert modules[0] is conv
        assert isinstance(modules[1], nn.Flatten)
        assert modules[2] is linear1
        assert modules[4] is linear2


# ---------------------------------------------------------------------------
# Tests for _load_backbone
# ---------------------------------------------------------------------------
class TestLoadBackbone:
    def test_none_weights(self):
        """weights=None creates model with random init."""
        model = _load_backbone("alexnet", weights=None)
        assert model is not None
        assert hasattr(model, "features")
        assert hasattr(model, "classifier")

    def test_local_file_path(self, tmp_path):
        """Loading from a local .pt file."""
        model = _load_backbone("alexnet", weights=None)
        path = str(tmp_path / "model.pt")
        torch.save(model.state_dict(), path)

        loaded = _load_backbone("alexnet", weights=path)
        assert loaded is not None

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------
# Tests for list_backbone_layers
# ---------------------------------------------------------------------------
class TestListBackboneLayers:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_alexnet_layers(self):
        names = list_backbone_layers("alexnet")
        assert "features.0" in names
        assert "avgpool" in names
        assert "classifier.6" in names
        assert "features" not in names
        assert "classifier" not in names

    def test_resnet18_layers(self):
        try:
            from torchvision.models import get_model

            get_model("resnet18", weights=None)
        except Exception:
            pytest.skip("resnet18 not available")

        names = list_backbone_layers("resnet18")
        assert "conv1" in names
        assert "avgpool" in names
        assert "fc" in names


# ---------------------------------------------------------------------------
# Tests for build_split_plan
# ---------------------------------------------------------------------------
class TestBuildSplitPlan:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_alexnet_replace_fc7(self):
        """Replace fc7 (classifier.4 + classifier.5) in AlexNet."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
        )

        assert "classifier.4" in plan.replaced_names
        assert "classifier.5" in plan.replaced_names
        assert "classifier.6" in plan.suffix_names
        assert plan.prefix_output_dim == 4096
        assert plan.suffix_input_dim == 4096
        assert plan.suffix_output_dim == 1000

    def test_alexnet_replace_fc6_fc7(self):
        """Replace fc6+fc7 (classifier.1 through classifier.5)."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
        )

        assert plan.prefix_output_dim == 9216
        assert plan.suffix_output_dim == 1000
        assert "classifier.1" in plan.replaced_names
        assert "classifier.5" in plan.replaced_names

    def test_target_modules_resolve_contiguous_replacement_span(self):
        with patch(
            "dendritic_modeling.networks.architectures.classical.pretrained._load_backbone",
            return_value=_MiniAlexNet(),
        ):
            plan = build_split_plan(
                "mini_alexnet",
                weights=None,
                target_modules=["classifier.5", "classifier.4"],
                input_shape=[3, 8, 8],
            )

        assert plan.replaced_names == ["classifier.4", "classifier.5"]
        assert plan.prefix_output_dim == 16
        assert plan.suffix_input_dim == 16

    def test_target_modules_must_be_contiguous(self):
        with patch(
            "dendritic_modeling.networks.architectures.classical.pretrained._load_backbone",
            return_value=_MiniAlexNet(),
        ):
            with pytest.raises(ValueError, match="contiguous span"):
                build_split_plan(
                    "mini_alexnet",
                    weights=None,
                    target_modules=["classifier.1", "classifier.5"],
                    input_shape=[3, 8, 8],
                )

    def test_invalid_start_layer_raises(self):
        with pytest.raises(ValueError, match="not found"):
            build_split_plan(
                "alexnet",
                weights=None,
                replace_start="nonexistent",
                replace_end="classifier.5",
            )

    def test_invalid_end_layer_raises(self):
        with pytest.raises(ValueError, match="not found"):
            build_split_plan(
                "alexnet",
                weights=None,
                replace_start="classifier.4",
                replace_end="nonexistent",
            )

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="comes after"):
            build_split_plan(
                "alexnet",
                weights=None,
                replace_start="classifier.6",
                replace_end="classifier.4",
            )

    def test_omit_layers_filtered(self):
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
        )

        assert "classifier.0" not in plan.prefix_names
        assert "classifier.3" not in plan.prefix_names

    def test_summary_string(self):
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
        )
        s = plan.summary()
        assert "BackboneSplitPlan" in s
        assert "prefix" in s
        assert "replace" in s
        assert "suffix" in s


class TestReplacementSpanHelpers:
    def test_resolve_target_module_span_accepts_unordered_contiguous_targets(self):
        all_names = ["features.0", "features.1", "classifier.1", "classifier.2"]

        assert _resolve_target_module_span(
            all_names, ["classifier.2", "classifier.1"]
        ) == (2, 3)

    def test_resolve_target_module_span_returns_none_without_targets(self):
        assert _resolve_target_module_span(["features.0"], None) is None
        assert _resolve_target_module_span(["features.0"], []) is None

    def test_resolve_target_module_span_rejects_unknown_target(self):
        with pytest.raises(ValueError, match="unknown layer"):
            _resolve_target_module_span(["features.0"], ["classifier.1"])

    def test_resolve_target_module_span_rejects_non_contiguous_targets(self):
        all_names = ["features.0", "features.1", "classifier.1"]

        with pytest.raises(ValueError, match="contiguous span"):
            _resolve_target_module_span(all_names, ["features.0", "classifier.1"])

    def test_require_layer_index_returns_index_or_raises_with_backbone_context(self):
        all_names = ["features.0", "classifier.1"]

        assert _require_layer_index(all_names, "classifier.1", backbone="mini") == 1
        with pytest.raises(ValueError, match="Layer 'missing' not found in mini"):
            _require_layer_index(all_names, "missing", backbone="mini")


class TestLayerSplitHelpers:
    def test_slice_layers_by_span_returns_prefix_replaced_and_suffix(self):
        layers = [(f"layer.{idx}", nn.Identity()) for idx in range(5)]

        prefix, replaced, suffix = _slice_layers_by_span(layers, 1, 3)

        assert [name for name, _ in prefix] == ["layer.0"]
        assert [name for name, _ in replaced] == ["layer.1", "layer.2", "layer.3"]
        assert [name for name, _ in suffix] == ["layer.4"]

    def test_names_and_modules_preserves_order_and_identity(self):
        layers = [("a", nn.Identity()), ("b", nn.ReLU())]

        names, modules = _names_and_modules(layers)

        assert names == ["a", "b"]
        assert modules == [layers[0][1], layers[1][1]]

    def test_omit_filtering_is_applied_outside_replacement_span_only(self):
        layers = [(f"layer.{idx}", nn.Identity()) for idx in range(5)]
        prefix, replaced, suffix = _slice_layers_by_span(layers, 1, 3)

        prefix_filtered = _filter_omitted_layers(prefix, ["layer.0", "layer.2"])
        suffix_filtered = _filter_omitted_layers(suffix, ["layer.4", "layer.2"])

        assert prefix_filtered == []
        assert [name for name, _ in replaced] == ["layer.1", "layer.2", "layer.3"]
        assert suffix_filtered == []


# ---------------------------------------------------------------------------
# Tests for BackboneSegment
# ---------------------------------------------------------------------------
class TestBackboneSegment:
    def test_run_modules_no_grad_preserves_order_without_tracking_grad(self):
        modules = [nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)]
        x = torch.randn(5, 3, requires_grad=True)

        expected = nn.Sequential(*modules)(x)
        actual = _run_modules_no_grad(modules, x)

        assert torch.allclose(actual, expected)
        assert actual.requires_grad is False

    def test_tensor_spec_from_spatial_output_uses_channel_height_width(self):
        spec = _tensor_spec_from_spatial_output(torch.zeros(2, 3, 4, 5))

        assert spec.channels == 3
        assert spec.height == 4
        assert spec.width == 5
        assert spec.numel == 60

    def test_segment_output_metadata_for_flat_output(self):
        output_spec, output_dim = _segment_output_metadata(torch.zeros(2, 10))

        assert output_spec is None
        assert output_dim == 10

    def test_segment_output_metadata_for_spatial_output(self):
        output_spec, output_dim = _segment_output_metadata(torch.zeros(2, 3, 4, 5))

        assert output_spec is not None
        assert output_spec.channels == 3
        assert output_spec.height == 4
        assert output_spec.width == 5
        assert output_dim == 60

    def test_prepare_segment_input_preserves_flat_input_batch_shape(self):
        x = torch.zeros(2, 8)

        prepared, batch_shape, squeezed = _prepare_segment_input(x, auto_flatten=True)

        assert prepared.shape == (2, 8)
        assert batch_shape == torch.Size([2])
        assert squeezed is False

    def test_prepare_segment_input_batches_single_spatial_input(self):
        x = torch.zeros(3, 4, 5)

        prepared, batch_shape, squeezed = _prepare_segment_input(x, auto_flatten=True)

        assert prepared.shape == (1, 3, 4, 5)
        assert batch_shape == torch.Size([1])
        assert squeezed is True

    def test_prepare_segment_input_flattens_leading_spatial_batches(self):
        x = torch.zeros(2, 4, 3, 5, 7)

        prepared, batch_shape, squeezed = _prepare_segment_input(x, auto_flatten=True)

        assert prepared.shape == (8, 3, 5, 7)
        assert batch_shape == torch.Size([2, 4])
        assert squeezed is False

    def test_prepare_segment_input_is_noop_without_auto_flatten(self):
        x = torch.zeros(3, 4, 5)

        prepared, batch_shape, squeezed = _prepare_segment_input(x, auto_flatten=False)

        assert prepared is x
        assert batch_shape == torch.Size()
        assert squeezed is False

    def test_restore_segment_output_restores_batch_shape_and_squeeze(self):
        restored = _restore_segment_output(
            torch.zeros(8, 6),
            auto_flatten=True,
            batch_shape=torch.Size([2, 4]),
            squeezed=False,
        )
        single = _restore_segment_output(
            torch.zeros(1, 6),
            auto_flatten=True,
            batch_shape=torch.Size([1]),
            squeezed=True,
        )
        unchanged = _restore_segment_output(
            torch.zeros(3, 6),
            auto_flatten=False,
            batch_shape=torch.Size(),
            squeezed=False,
        )

        assert restored.shape == (2, 4, 6)
        assert single.shape == (6,)
        assert unchanged.shape == (3, 6)

    def test_linear_segment(self):
        """Simple linear segment with known dims."""
        mods = [nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 10)]
        seg = BackboneSegment(
            mods,
            ["fc1", "relu", "fc2"],
            input_size=(1, 32),
            auto_flatten=False,
        )

        assert seg.output_dim == 10

        x = torch.randn(4, 32)
        out = seg(x)
        assert out.shape == (4, 10)

    def test_frozen_segment(self):
        mods = [nn.Linear(32, 16)]
        seg = BackboneSegment(
            mods,
            ["fc1"],
            freeze=True,
            input_size=(1, 32),
            auto_flatten=False,
        )

        assert seg.frozen is True
        for p in seg.parameters():
            assert not p.requires_grad

    def test_unfrozen_segment(self):
        mods = [nn.Linear(32, 16)]
        seg = BackboneSegment(
            mods,
            ["fc1"],
            freeze=False,
            input_size=(1, 32),
            auto_flatten=False,
        )

        assert seg.frozen is False
        for p in seg.parameters():
            assert p.requires_grad

    def test_is_base_network(self):
        mods = [nn.Linear(32, 16)]
        seg = BackboneSegment(
            mods,
            ["fc1"],
            input_size=(1, 32),
            auto_flatten=False,
        )
        assert isinstance(seg, BaseNetwork)
        assert seg.get_output_dim() == 16


# ---------------------------------------------------------------------------
# Tests for build_encoder_decoder_from_plan
# ---------------------------------------------------------------------------
class TestBuildEncoderDecoderFromPlan:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_alexnet_encoder_decoder_dims(self):
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
        )

        encoder, decoder = build_encoder_decoder_from_plan(
            plan,
            prefix_freeze=True,
            suffix_freeze=False,
        )

        assert encoder.output_dim == 4096
        assert decoder.output_dim == 1000
        assert encoder.frozen is True
        assert decoder.frozen is False

    def test_encoder_decoder_forward(self):
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
        )

        encoder, decoder = build_encoder_decoder_from_plan(plan)

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            encoded = encoder(x)
        assert encoded.shape == (2, 4096)

        core = _SimpleCore(4096, 4096)
        core_out = core(encoded)
        logits = decoder(core_out)
        assert logits.shape == (2, 1000)

    def test_full_pipeline_with_plan(self):
        """End-to-end: plan -> encoder -> core -> decoder."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.1",
            replace_end="classifier.5",
        )

        encoder, decoder = build_encoder_decoder_from_plan(
            plan,
            prefix_freeze=True,
            suffix_freeze=False,
        )

        assert encoder.output_dim == 9216
        assert decoder.output_dim == 1000

        core = _SimpleCore(encoder.output_dim, plan.suffix_input_dim)

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            encoded = encoder(x)
        core_out = core(encoded)
        logits = decoder(core_out)
        assert logits.shape == (2, 1000)

    def test_frozen_encoder_no_grad(self):
        """Frozen encoder should not accumulate gradients during pipeline."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
            omit_layers=["classifier.0", "classifier.3"],
        )

        encoder, _decoder = build_encoder_decoder_from_plan(
            plan,
            prefix_freeze=True,
            suffix_freeze=False,
        )

        core = _SimpleCore(encoder.output_dim, 100)

        x = torch.randn(2, 3, 224, 224)
        encoded = encoder(x)
        out = core(encoded)
        loss = out.sum()
        loss.backward()

        for p in encoder.parameters():
            assert p.grad is None
        for p in core.parameters():
            assert p.grad is not None

    def test_3d_input_adds_batch_dim(self):
        """Single image (C, H, W) should work."""
        plan = build_split_plan(
            "alexnet",
            weights=None,
            replace_start="classifier.4",
            replace_end="classifier.5",
            omit_layers=["classifier.0"],
        )

        encoder, _ = build_encoder_decoder_from_plan(plan, prefix_freeze=True)

        x = torch.randn(3, 224, 224)
        out = encoder(x)
        assert out.shape == (4096,)


# ---------------------------------------------------------------------------
# Integration test: load_config -> initialize_model
# ---------------------------------------------------------------------------
class TestConfigIntegration:
    @pytest.fixture(autouse=True)
    def skip_if_no_torchvision(self):
        try:
            from torchvision.models import get_model

            get_model("alexnet", weights=None)
        except Exception:
            pytest.skip("torchvision alexnet not available")

    def test_load_config_pretrained_replacement_enabled(self, tmp_path):
        """Loading a YAML with pretrained_replacement.enabled=true should
        populate ModelConfig.pretrained_replacement correctly."""
        from dendritic_modeling.config import load_config
        from dendritic_modeling.config.training import (
            CommonTrainerConfig,
            ParamGroupsConfig,
        )

        yaml_content = """\
experiment:
  seed: 42
  train_valid_split: 0.8

data:
  dataset_name: "imagenet"
  base_dir: "/tmp/fake"
  processing:
    flatten: false
    normalize: true

model:
  task: "classification"
  pretrained_replacement:
    enabled: true
    backbone: "alexnet"
    weights: null
    input_shape: [3, 224, 224]
    replace:
      start: "classifier.4"
      end: "classifier.5"
    omit_layers: ["classifier.0", "classifier.3"]
    trainability:
      encoder:
        mode: "frozen"
      decoder:
        mode: "trainable"

  core:
    type: "einet"
    architecture:
      excitatory_layer_sizes: [4096]
      inhibitory_layer_sizes: [820]
      excitatory_branch_factors: [2, 2]
      inhibitory_branch_factors: [2, 2]
    connectivity:
      ee_synapses_per_branch_per_layer: [40]
      ei_synapses_per_branch_per_layer: [10]
      ie_synapses_per_branch_per_layer: [10]
      ii_synapses_per_branch_per_layer: [0]

training:
  encoder:
    strategy: "standard"
    common:
      lr: 0.0
      epochs: 0
      batch_size: 64
  main:
    strategy: "standard"
    common:
      epochs: 1
      batch_size: 64
      loss_function: "cat_nll"
      param_groups:
        lr: 0.001
        split_params: true
        encoder_lr: 0.0

outputs:
  run_name: "test_integration"
  results_dir: "/tmp/test_outputs"
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)

        config = load_config(str(config_path))
        mc = config.model

        # pretrained_replacement should be enabled
        assert mc.pretrained_replacement.enabled is True
        assert mc.pretrained_replacement.backbone == "alexnet"
        assert mc.pretrained_replacement.replace.start == "classifier.4"
        assert mc.pretrained_replacement.replace.end == "classifier.5"
        assert "classifier.0" in mc.pretrained_replacement.omit_layers
        assert isinstance(config.training.main.common, CommonTrainerConfig)
        assert isinstance(config.training.encoder.common, CommonTrainerConfig)
        assert isinstance(config.training.main.common.param_groups, ParamGroupsConfig)

    def test_initialize_model_from_pretrained_replacement(self, tmp_path):
        """initialize_model should produce BackboneSegment encoder/decoder
        when pretrained_replacement is enabled."""
        from dendritic_modeling.config import load_config
        from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

        yaml_content = """\
experiment:
  seed: 42
  train_valid_split: 0.8

data:
  dataset_name: "imagenet"
  base_dir: "/tmp/fake"
  processing:
    flatten: false
    normalize: true

model:
  task: "classification"
  pretrained_replacement:
    enabled: true
    backbone: "alexnet"
    weights: null
    input_shape: [3, 224, 224]
    replace:
      start: "classifier.4"
      end: "classifier.5"
    omit_layers: ["classifier.0", "classifier.3"]
    trainability:
      encoder:
        mode: "frozen"
      decoder:
        mode: "trainable"

  core:
    type: "einet"
    architecture:
      excitatory_layer_sizes: [4096]
      inhibitory_layer_sizes: [820]
      excitatory_branch_factors: [2, 2]
      inhibitory_branch_factors: [2, 2]
    connectivity:
      ee_synapses_per_branch_per_layer: [40]
      ei_synapses_per_branch_per_layer: [10]
      ie_synapses_per_branch_per_layer: [10]
      ii_synapses_per_branch_per_layer: [0]

training:
  encoder:
    strategy: "standard"
    common:
      lr: 0.0
      epochs: 0
      batch_size: 64
  main:
    strategy: "standard"
    common:
      epochs: 1
      batch_size: 64
      loss_function: "cat_nll"
      param_groups:
        lr: 0.001
        split_params: true
        encoder_lr: 0.0

outputs:
  run_name: "test_integration"
  results_dir: "/tmp/test_outputs"
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(yaml_content)

        config = load_config(str(config_path))
        model, encoder_network = initialize_model(config.model)

        # Encoder and decoder should be BackboneSegment instances
        assert isinstance(model.encoder_network, BackboneSegment)
        assert isinstance(model.decoder_network, BackboneSegment)
        assert isinstance(encoder_network, BackboneSegment)

        # Dims should match the split plan
        assert model.encoder_network.output_dim == 4096
        assert model.decoder_network.output_dim == 1000

        # Core input/output dims should be consistent
        assert model.core_network.output_dim == 4096

        # Forward pass should work
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1000)

    def test_initialize_model_from_vision_replacement_target_modules(self, tmp_path):
        """vision_replacement mirrors transformer-style target-module naming."""
        from dendritic_modeling.config import load_config
        from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

        yaml_content = """\
experiment:
  seed: 42

data:
  dataset_name: "imagenet"
  base_dir: "/tmp/fake"
  processing:
    flatten: false

model:
  task: "classification"
  vision_replacement:
    enabled: true
    backbone: "mini_alexnet"
    weights: null
    input_shape: [3, 8, 8]
    target_modules: ["classifier.4", "classifier.5"]
    omit_layers: ["classifier.0", "classifier.3"]
    trainability:
      encoder:
        mode: "frozen"
      decoder:
        mode: "trainable"
  core:
    type: "einet"
    architecture:
      excitatory_layer_sizes: [16]
      inhibitory_layer_sizes: [4]
      excitatory_branch_factors: [1]
      inhibitory_branch_factors: [1]
    connectivity:
      ee_synapses_per_branch_per_layer: [2]
      ei_synapses_per_branch_per_layer: [2]
      ie_synapses_per_branch_per_layer: [1]
      ii_synapses_per_branch_per_layer: [0]

training:
  main:
    common:
      param_groups:
        lr: 0.001
"""
        config_path = tmp_path / "vision_replacement.yaml"
        config_path.write_text(yaml_content)

        with patch(
            "dendritic_modeling.networks.architectures.classical.pretrained._load_backbone",
            return_value=_MiniAlexNet(),
        ):
            config = load_config(str(config_path))
            model, _encoder_network = initialize_model(config.model)

        assert config.model.vision_replacement.enabled is True
        assert config.model.vision_replacement.target_modules == [
            "classifier.4",
            "classifier.5",
        ]
        assert isinstance(model.encoder_network, BackboneSegment)
        assert isinstance(model.decoder_network, BackboneSegment)
        assert model.encoder_network.output_dim == 16
        assert model.decoder_network.output_dim == 10

        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_initialize_model_warns_for_ignored_legacy_sections(self, tmp_path):
        """Legacy encoder/decoder config should warn when replacement is enabled."""
        from dendritic_modeling.config import load_config
        from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

        yaml_content = """\
experiment:
  seed: 42

data:
  dataset_name: "imagenet"
  base_dir: "/tmp/fake"
  processing:
    flatten: false

model:
  task: "classification"
  encoder:
    type: "pathway_router"
  decoder:
    type: "identity"
  pretrained_replacement:
    enabled: true
    backbone: "alexnet"
    weights: null
    input_shape: [3, 224, 224]
    replace:
      start: "classifier.4"
      end: "classifier.5"
    omit_layers: ["classifier.0", "classifier.3"]
  core:
    type: "einet"
    architecture:
      excitatory_layer_sizes: [4096]
      inhibitory_layer_sizes: [820]
      excitatory_branch_factors: [2, 2]
      inhibitory_branch_factors: [2, 2]
    connectivity:
      ee_synapses_per_branch_per_layer: [40]
      ei_synapses_per_branch_per_layer: [10]
      ie_synapses_per_branch_per_layer: [10]
      ii_synapses_per_branch_per_layer: [0]
"""
        config_path = tmp_path / "warn_config.yaml"
        config_path.write_text(yaml_content)

        config = load_config(str(config_path))

        with patch(
            "dendritic_modeling.scripts.script_utils.setup_utils.logger.warning"
        ) as warn_mock:
            initialize_model(config.model)

        warn_mock.assert_called_once()
        assert "ignored because model.pretrained_replacement.enabled=true" in (
            warn_mock.call_args.args[0]
        )
        assert "model.encoder, model.decoder" in warn_mock.call_args.args[1]

    def test_fsdp_create_large_model_from_pretrained_replacement(self, tmp_path):
        """FSDP model creation should honor pretrained_replacement configs."""
        from dendritic_modeling.config import load_config
        from dendritic_modeling.scripts.training.train_experiments_fsdp import (
            create_large_model,
        )

        yaml_content = """\
experiment:
  seed: 42

data:
  dataset_name: "imagenet"
  base_dir: "/tmp/fake"
  processing:
    flatten: false

model:
  task: "classification"
  pretrained_replacement:
    enabled: true
    backbone: "alexnet"
    weights: null
    input_shape: [3, 224, 224]
    replace:
      start: "classifier.4"
      end: "classifier.5"
    omit_layers: ["classifier.0", "classifier.3"]
    trainability:
      encoder:
        mode: "frozen"
      decoder:
        mode: "trainable"
  core:
    type: "einet"
    architecture:
      excitatory_layer_sizes: [4096]
      inhibitory_layer_sizes: [820]
      excitatory_branch_factors: [2, 2]
      inhibitory_branch_factors: [2, 2]
    connectivity:
      ee_synapses_per_branch_per_layer: [40]
      ei_synapses_per_branch_per_layer: [10]
      ie_synapses_per_branch_per_layer: [10]
      ii_synapses_per_branch_per_layer: [0]

training:
  main:
    common:
      param_groups:
        lr: 0.001
"""
        config_path = tmp_path / "fsdp_config.yaml"
        config_path.write_text(yaml_content)

        config = load_config(str(config_path))
        model = create_large_model(config)

        assert isinstance(model.encoder_network, BackboneSegment)
        assert isinstance(model.decoder_network, BackboneSegment)
