from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DeepstLinear,
    IndexedDynamicTopKLinear,
    IndexedRewireLinear,
    IndexedSparseLinear,
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
)
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
    EIStackMLPSlot,
    apply_transformer_replacement_config,
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
    estimate_deepseek_dense_ffn_params,
    estimate_dendritic_ffn_params,
    replace_transformer_mlp_layers,
    replace_transformer_mlp_layers_with_ei_stack,
)
from dendritic_modeling.networks.architectures.transformer.utils import _infer_mlp_dims
from dendritic_modeling.training.transformer_replacement import _MLPIOCapture

TRANSFORMER_BENCHMARK_CONFIGS = sorted(
    Path("configs/transformer_replacement_benchmarks").rglob("*.yaml")
)


class TinyGatedMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


class TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyGatedMLP()

    def forward(self, x):
        return x + self.mlp(x)


class TinyNestedDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Module()
        self.block.inner = TinyGatedMLP()

    def forward(self, x):
        return x + self.block.inner(x)


class TinyHFLikeModel(nn.Module):
    def __init__(
        self, n_layers: int = 3, layer_cls: type[nn.Module] = TinyDecoderLayer
    ):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer_cls() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def _count_forward_hooks(model: nn.Module) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def test_transformer_refactor_preserves_public_imports():
    from dendritic_modeling.networks.architectures.transformer import (
        config_translation,
        dendritic_ffn,
        modules,
        patching,
    )

    assert dendritic_ffn.DendriticFFNReplacement is modules.DendriticFFNReplacement
    assert (
        dendritic_ffn.LayerwiseEIStackReplacement is modules.LayerwiseEIStackReplacement
    )
    assert dendritic_ffn.apply_transformer_replacement_config is (
        patching.apply_transformer_replacement_config
    )
    assert dendritic_ffn.build_dendritic_ffn_kwargs_from_core_config is (
        config_translation.build_dendritic_ffn_kwargs_from_core_config
    )


def test_infer_mlp_dims_fallback_preserves_recursive_linear_order():
    mlp = nn.Sequential(
        nn.Sequential(nn.Linear(8, 16, bias=False)),
        nn.GELU(),
        nn.Sequential(nn.Linear(16, 8, bias=False)),
    )

    assert _infer_mlp_dims(mlp) == (8, 16)


def test_mlp_io_capture_removes_hooks_after_partial_init_failure():
    model = TinyHFLikeModel(n_layers=2)
    del model.model.layers[1].mlp

    assert _count_forward_hooks(model) == 0
    with pytest.raises(AttributeError):
        _MLPIOCapture(
            model,
            layer_indices=[0, 1],
            layers_attr="model.layers",
            mlp_attr="mlp",
        )

    assert _count_forward_hooks(model) == 0


def test_mlp_io_capture_removes_pre_hook_when_forward_hook_registration_fails(
    monkeypatch,
):
    model = TinyHFLikeModel(n_layers=1)
    mlp = model.model.layers[0].mlp

    def _raise_forward_hook_failure(*args, **kwargs):
        raise RuntimeError("forward hook registration failed")

    monkeypatch.setattr(mlp, "register_forward_hook", _raise_forward_hook_failure)

    assert _count_forward_hooks(model) == 0
    with pytest.raises(RuntimeError, match="forward hook registration failed"):
        _MLPIOCapture(
            model,
            layer_indices=[0],
            layers_attr="model.layers",
            mlp_attr="mlp",
        )

    assert _count_forward_hooks(model) == 0


def test_mlp_io_capture_close_removes_hooks_and_clears_handles():
    model = TinyHFLikeModel(n_layers=1)
    capture = _MLPIOCapture(
        model,
        layer_indices=[0],
        layers_attr="model.layers",
        mlp_attr="mlp",
    )

    assert _count_forward_hooks(model) == 2
    assert len(capture.handles) == 2

    capture.close()

    assert _count_forward_hooks(model) == 0
    assert capture.handles == []


@pytest.mark.parametrize(
    "config_path",
    TRANSFORMER_BENCHMARK_CONFIGS,
    ids=[p.name for p in TRANSFORMER_BENCHMARK_CONFIGS],
)
def test_transformer_benchmark_job_config_loads(config_path):
    """Benchmark configs are source configs, independent of local job scripts."""
    load_config.cache_clear()
    cfg = load_config(str(config_path))

    assert cfg.model.transformer_replacement.enabled


@pytest.mark.parametrize(
    "config_path",
    TRANSFORMER_BENCHMARK_CONFIGS,
    ids=[p.name for p in TRANSFORMER_BENCHMARK_CONFIGS],
)
def test_transformer_benchmark_configs_use_population_network_schema(config_path):
    raw = yaml.safe_load(config_path.read_text())
    core = raw["model"]["core"]

    assert "population_network" in core
    for old_key in (
        "architecture",
        "connectivity",
        "transfer",
        "morphology",
        "sparsity",
        "reactivation",
    ):
        assert old_key not in core

    layer = core["population_network"]["layers"][0]
    assert layer["readout_population"]
    assert layer["population_defaults"]["ff_excitatory_synapses"] > 0
    assert layer["populations"]


@pytest.mark.parametrize(
    ("topk_type", "expected_cls", "extra_kwargs"),
    [
        ("standard", TopKLinear, {}),
        ("stochastic", StochasticTopKLinear, {"topk_temperature": 0.5}),
        ("variance", VarianceTopKLinear, {}),
        (
            "deepst",
            DeepstLinear,
            {
                "rewiring_mode": "constant_branch",
                "synapses_per_branch": 2,
                "excitatory_target_density": 0.5,
            },
        ),
        ("indexed", IndexedSparseLinear, {"indexed_seed": 3}),
        (
            "indexed_rewire",
            IndexedRewireLinear,
            {
                "indexed_seed": 3,
                "indexed_rewire_frequency": 1,
                "indexed_rewire_quantile": 0.5,
            },
        ),
        (
            "indexed_dynamic",
            IndexedDynamicTopKLinear,
            {
                "indexed_seed": 3,
                "indexed_candidate_size": 4,
                "indexed_selection": "stochastic",
            },
        ),
    ],
)
def test_dendritic_branch_layer_builds_all_topk_variants(
    topk_type, expected_cls, extra_kwargs
):
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=6,
        excitatory_synapses_per_branch=2,
        topk_type=topk_type,
        weight_transform="softplus",
        use_shunting=False,
        reactivate=False,
        input_branch_factor=None,
        **extra_kwargs,
    )

    assert isinstance(layer.branch_excitation, expected_cls)
    y = layer(torch.rand(2, 6))
    assert y.shape == (2, 3)


def test_indexed_sparse_linear_stores_only_active_weights_and_backprops():
    layer = IndexedSparseLinear(
        in_features=7,
        out_features=5,
        K=3,
        weight_transform="softplus",
        seed=123,
    )

    assert layer.pre_w.shape == (5, 3)
    assert layer.connection_indices.shape == (5, 3)

    x = torch.randn(2, 4, 7, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 4, 5)
    y.square().mean().backward()
    assert x.grad is not None
    assert layer.pre_w.grad is not None

    dense = layer.pruned_weight()
    assert dense.shape == (5, 7)
    assert layer.weight().shape == (5, 7)
    assert layer.sparse_weight().shape == (5, 3)
    assert int(layer.weight_mask().sum().item()) == 15


def test_dendritic_ffn_replacement_preserves_transformer_mlp_shape():
    replacement = DendriticFFNReplacement(
        hidden_size=8,
        dendritic_units=4,
        branch_factors=[2],
        synapses_per_branch=2,
        input_transform="signed_split",
        topk_type="indexed",
        indexed_seed=11,
        use_shunting=True,
    )

    x = torch.randn(2, 3, 8, requires_grad=True)
    y = replacement(x)

    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert replacement.output_projection.weight.grad is not None


def test_indexed_sparse_linear_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="unique"):
        IndexedSparseLinear(
            in_features=4,
            out_features=1,
            K=2,
            connection_indices=torch.tensor([[1, 1]]),
        )


def test_indexed_dynamic_topk_allows_candidate_synapses_to_enter_topk():
    layer = IndexedDynamicTopKLinear(
        in_features=5,
        out_features=1,
        K=2,
        candidate_size=4,
        connection_indices=torch.tensor([[0, 1, 2, 3]]),
        weight_transform="identity",
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[0.0, 1.0, 2.0, 3.0]]))

    mask_before = layer.weight_mask()
    assert mask_before[0, 0].item() == 0
    assert mask_before[0, 2].item() == 1
    assert mask_before[0, 3].item() == 1

    with torch.no_grad():
        layer.pre_w[0, 0] = 10.0

    mask_after = layer.weight_mask()
    assert mask_after[0, 0].item() == 1
    assert mask_after[0, 1].item() == 0


def test_indexed_dynamic_topk_supports_stochastic_candidate_selection():
    layer = IndexedDynamicTopKLinear(
        in_features=6,
        out_features=3,
        K=2,
        candidate_size=5,
        selection="stochastic",
        temperature=0.5,
        seed=17,
    )

    x = torch.randn(4, 6, requires_grad=True)
    y = layer(x)
    assert y.shape == (4, 3)
    y.sum().backward()
    assert layer.pre_w.grad is not None


def test_replace_transformer_mlp_layers_patches_hf_like_model():
    model = TinyHFLikeModel(n_layers=3)
    records = replace_transformer_mlp_layers(
        model,
        [0, 2],
        replacement_kwargs={
            "dendritic_units": 4,
            "branch_factors": [2],
            "synapses_per_branch": 2,
            "topk_type": "indexed",
            "indexed_seed": 7,
        },
    )

    assert len(records) == 2
    assert isinstance(model.model.layers[0].mlp, DendriticFFNReplacement)
    assert isinstance(model.model.layers[1].mlp, TinyGatedMLP)
    assert isinstance(model.model.layers[2].mlp, DendriticFFNReplacement)
    assert records[0].replacement.hidden_size == 8
    assert records[0].replacement.reference_intermediate_size == 16

    x = torch.randn(2, 5, 8)
    y = model(x)
    assert y.shape == x.shape


def test_replace_transformer_layers_accepts_nested_target_module_path():
    model = TinyHFLikeModel(n_layers=2, layer_cls=TinyNestedDecoderLayer)
    records = replace_transformer_mlp_layers(
        model,
        [1],
        mlp_attr="block.inner",
        replacement_kwargs={
            "dendritic_units": 4,
            "branch_factors": [2],
            "synapses_per_branch": 2,
            "input_transform": "signed_split",
        },
    )

    assert len(records) == 1
    assert records[0].mlp_attr == "block.inner"
    assert isinstance(model.model.layers[0].block.inner, TinyGatedMLP)
    assert isinstance(model.model.layers[1].block.inner, DendriticFFNReplacement)
    x = torch.randn(2, 5, 8)
    assert model(x).shape == x.shape


def test_replace_transformer_mlp_layers_rejects_bad_index():
    model = TinyHFLikeModel(n_layers=1)
    with pytest.raises(IndexError):
        replace_transformer_mlp_layers(
            model,
            [3],
            replacement_kwargs={
                "dendritic_units": 4,
                "branch_factors": [2],
                "synapses_per_branch": 2,
            },
        )


def test_transformer_replacement_rejects_non_module_target_consistently():
    model = TinyHFLikeModel(n_layers=2)
    model.model.layers[0].not_module = object()
    model.model.layers[1].not_module = object()

    with pytest.raises(TypeError, match=r"not an nn\.Module"):
        replace_transformer_mlp_layers(
            model,
            [0],
            mlp_attr="not_module",
            replacement_kwargs={
                "dendritic_units": 4,
                "branch_factors": [2],
                "synapses_per_branch": 2,
            },
        )

    with pytest.raises(TypeError, match=r"not an nn\.Module"):
        replace_transformer_mlp_layers_with_ei_stack(
            model,
            [0, 1],
            mlp_attr="not_module",
            replacement_kwargs={
                "excitatory_cells": 4,
                "inhibitory_cells": 2,
                "excitatory_branch_factors": [1],
                "inhibitory_branch_factors": [1],
                "ee_synapses_per_branch": 2,
                "ei_synapses_per_branch": 1,
                "ie_synapses_per_branch": 1,
                "ii_synapses_per_branch": 1,
                "use_shunting": False,
            },
        )


def test_deepseek_parameter_estimates_show_sparse_vs_dense_difference():
    dense = estimate_deepseek_dense_ffn_params(
        hidden_size=7168, intermediate_size=18432
    )
    indexed = estimate_dendritic_ffn_params(
        hidden_size=7168,
        dendritic_units=512,
        branch_factors=[3, 3],
        synapses_per_branch=40,
        input_transform="signed_split",
        topk_type="indexed",
    )
    indexed_dynamic = estimate_dendritic_ffn_params(
        hidden_size=7168,
        dendritic_units=512,
        branch_factors=[3, 3],
        synapses_per_branch=40,
        candidate_size=256,
        input_transform="signed_split",
        topk_type="indexed_dynamic",
    )
    dynamic = estimate_dendritic_ffn_params(
        hidden_size=7168,
        dendritic_units=512,
        branch_factors=[3, 3],
        synapses_per_branch=40,
        input_transform="signed_split",
        topk_type="standard",
    )

    assert dense == 396_361_728
    assert indexed["stored_total"] < dense // 50
    assert (
        indexed["stored_total"]
        < indexed_dynamic["stored_total"]
        < dynamic["stored_total"]
    )
    assert dynamic["stored_total"] > indexed["stored_total"]


def test_transformer_replacement_config_maps_shared_core_like_alexnet():
    core_config = {
        "type": "dendritic_ffn",
        "architecture": {
            "excitatory_layer_sizes": [4],
            "excitatory_branch_factors": [2],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
        },
        "transfer": {
            "input_transform": "signed_split",
        },
        "morphology": {
            "use_shunting": True,
            "weight_transform": "softplus",
        },
        "blocklinear": {
            "efficient": True,
        },
        "sparsity": {
            "type": "indexed_rewire",
            "indexed": {
                "seed": 5,
                "rewire_frequency": 10,
                "rewire_quantile": 0.1,
                "rewire_until_step": 200,
            },
        },
        "reactivation": {
            "enabled": True,
            "type": "param_tanh",
        },
    }
    transformer_replacement = {
        "enabled": True,
        "layers": [1],
        "target_module": "mlp",
        "input_transform": "signed_split",
    }

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=transformer_replacement,
    )
    assert kwargs["dendritic_units"] == 4
    assert kwargs["topk_type"] == "indexed_rewire"
    assert kwargs["indexed_rewire_frequency"] == 10
    assert kwargs["indexed_rewire_quantile"] == 0.1
    assert kwargs["indexed_rewire_until_step"] == 200
    assert kwargs["efficient_blocklinear"] is True

    model = TinyHFLikeModel(n_layers=2)
    records = apply_transformer_replacement_config(
        model,
        transformer_replacement,
        core_config,
    )
    assert len(records) == 1
    replacement = model.model.layers[1].mlp
    assert isinstance(replacement, DendriticFFNReplacement)
    assert replacement.topk_type == "indexed_rewire"
    assert replacement.indexed_rewire_frequency == 10
    assert replacement.efficient_blocklinear is True


def test_transformer_replacement_maps_population_network_core():
    core_config = {
        "type": "dendritic_ffn",
        "population_network": {
            "input_transform": "signed_split",
            "layers": [
                {
                    "name": "transformer_mlp",
                    "readout_population": "ffn",
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "somatic_synapses": True,
                        "use_shunting": False,
                        "reactivate": True,
                        "reactivation_type": "param_tanh",
                        "reactivation_init_policy": "occupancy_quantile",
                        "weight_transform": "softplus",
                        "topk_type": "indexed_rewire",
                        "indexed_seed": 5,
                        "indexed_rewire_frequency": 10,
                        "indexed_rewire_quantile": 0.1,
                        "indexed_rewire_until_step": 200,
                        "efficient_blocklinear": True,
                    },
                    "populations": [
                        {
                            "name": "ffn",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [2],
                        }
                    ],
                    "connections": [{"source": "input", "target": "ffn"}],
                }
            ],
        },
    }
    transformer_replacement = {
        "enabled": True,
        "layers": [1],
        "target_module": "mlp",
    }

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=transformer_replacement,
    )
    assert kwargs["dendritic_units"] == 4
    assert kwargs["branch_factors"] == [2]
    assert kwargs["synapses_per_branch"] == 2
    assert kwargs["topk_type"] == "indexed_rewire"
    assert kwargs["indexed_rewire_frequency"] == 10
    assert kwargs["indexed_rewire_quantile"] == 0.1
    assert kwargs["indexed_rewire_until_step"] == 200
    assert kwargs["reactivation_init_policy"] == "occupancy_quantile"
    assert kwargs["use_shunting"] is False
    assert kwargs["efficient_blocklinear"] is True

    model = TinyHFLikeModel(n_layers=2)
    records = apply_transformer_replacement_config(
        model,
        transformer_replacement,
        core_config,
    )
    assert len(records) == 1
    replacement = model.model.layers[1].mlp
    assert isinstance(replacement, DendriticFFNReplacement)
    assert replacement.dendritic_units == 4
    assert replacement.indexed_rewire_frequency == 10
    assert replacement.efficient_blocklinear is True


def test_transformer_replacement_core_alias_overrides_morphology_shunting():
    core_config = {
        "type": "dendritic_additive",
        "architecture": {
            "excitatory_layer_sizes": [4],
            "excitatory_branch_factors": [1],
        },
        "connectivity": {"ee_synapses_per_branch_per_layer": [2]},
        "morphology": {
            "use_shunting": True,
            "use_additive_normalization": False,
        },
    }

    with pytest.warns(UserWarning, match="dendritic_additive"):
        kwargs = build_dendritic_ffn_kwargs_from_core_config(core_config)

    # dendritic_additive overrides use_shunting -> False but is now RAW additive
    # (normalization is opt-in via dendritic_normalized_additive).
    assert kwargs["use_shunting"] is False
    assert kwargs["use_additive_normalization"] is False

    replacement = DendriticFFNReplacement(hidden_size=8, **kwargs)
    assert replacement.core.branch_layers[0].use_shunting is False
    assert replacement.core.branch_layers[0].use_additive_normalization is False


def test_transformer_ei_stack_core_alias_overrides_morphology_shunting():
    core_config = {
        "type": "dendritic_shunting",
        "architecture": {
            "excitatory_layer_sizes": [5],
            "inhibitory_layer_sizes": [2],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [1],
        },
        "morphology": {"use_shunting": False},
    }

    with pytest.warns(UserWarning, match="dendritic_shunting"):
        kwargs = build_ei_stack_kwargs_from_core_config(core_config, num_layers=1)

    assert kwargs["use_shunting"] is True


def test_transformer_structured_sparsity_translation_is_shared_across_replacements():
    core_config = {
        "architecture": {
            "excitatory_layer_sizes": [5],
            "inhibitory_layer_sizes": [2],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [1],
        },
        "sparsity": {
            "type": "indexed_rewire",
            "init_method": "xavier_uniform",
            "noise_level": 0.03,
            "temperature": 0.4,
            "ultrafast": True,
            "weight_norm_order": 2,
            "gamma": 0.9,
            "dense_to_sparse": {
                "initial_density": 0.8,
                "initial_k": 3,
                "start_step": 4,
                "end_step": 40,
                "update_interval": 5,
                "schedule": "linear",
                "freeze_on_end": True,
                "advance_on_forward": False,
                "prune_metric": "magnitude",
            },
            "indexed": {
                "seed": 13,
                "candidate_size": 7,
                "selection": "stochastic",
                "output_chunk_size": 128,
                "rewire_frequency": 6,
                "rewire_quantile": 0.2,
                "rewire_until_step": 60,
            },
        },
    }
    transformer_replacement = {
        "input_transform": "signed_split",
        "replacement_kwargs": {
            "kind": "ei_stack",
            "type": "selector_only",
            "output_scale": 0.75,
        },
    }

    ffn_kwargs = build_dendritic_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=transformer_replacement,
    )
    stack_kwargs = build_ei_stack_kwargs_from_core_config(
        core_config,
        transformer_replacement=transformer_replacement,
        num_layers=2,
    )

    shared_keys = [
        "topk_type",
        "topk_init_method",
        "topk_noise_level",
        "topk_temperature",
        "topk_weight_norm_order",
        "topk_gamma",
        "dense_to_sparse_initial_density",
        "dense_to_sparse_initial_k",
        "dense_to_sparse_start_step",
        "dense_to_sparse_end_step",
        "dense_to_sparse_update_interval",
        "dense_to_sparse_schedule",
        "dense_to_sparse_freeze_on_end",
        "dense_to_sparse_advance_on_forward",
        "dense_to_sparse_prune_metric",
        "indexed_seed",
        "indexed_candidate_size",
        "indexed_selection",
        "indexed_output_chunk_size",
        "indexed_rewire_frequency",
        "indexed_rewire_quantile",
        "indexed_rewire_until_step",
    ]
    for key in shared_keys:
        assert ffn_kwargs[key] == stack_kwargs[key]

    assert ffn_kwargs["topk_ultrafast"] is True
    assert "topk_ultrafast" not in stack_kwargs
    assert ffn_kwargs["output_scale"] == 0.75
    assert stack_kwargs["output_scale"] == 0.75
    assert "kind" not in stack_kwargs
    assert "type" not in stack_kwargs


def test_transformer_replacement_can_patch_shared_ei_stack():
    core_config = {
        "architecture": {
            "excitatory_layer_sizes": [5],
            "inhibitory_layer_sizes": [2],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [1],
        },
        "transfer": {"input_transform": "signed_split"},
        "morphology": {"use_shunting": False, "weight_transform": "softplus"},
        "blocklinear": {"efficient": True},
        "sparsity": {"type": "standard", "init_method": "xavier_normal"},
        "reactivation": {"enabled": True, "type": "param_tanh"},
    }
    transformer_replacement = {
        "enabled": True,
        "layers": [0, 1, 2],
        "target_module": "mlp",
        "input_transform": "signed_split",
        "replacement_kwargs": {
            "kind": "ei_stack",
            "output_init_std": 0.05,
        },
    }

    model = TinyHFLikeModel(n_layers=3)
    records = apply_transformer_replacement_config(
        model,
        transformer_replacement,
        core_config,
    )

    assert len(records) == 3
    assert all(isinstance(layer.mlp, EIStackMLPSlot) for layer in model.model.layers)
    assert model.model.layers[0].mlp.stack is model.model.layers[1].mlp.stack
    assert model.model.layers[0].mlp.stack.efficient_blocklinear is True
    x = torch.randn(2, 4, 8)
    y = model(x)
    assert y.shape == x.shape
    estimate = model.model.layers[0].mlp.parameter_estimate()
    assert estimate["stored_total"] > 0
    assert estimate["active_total"] > 0


def test_transformer_ei_stack_maps_population_network_core():
    core_config = {
        "type": "population_network",
        "population_network": {
            "input_transform": "signed_split",
            "layers": [
                {
                    "name": "transformer_mlp",
                    "readout_population": "e",
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 1,
                        "somatic_synapses": True,
                        "use_shunting": True,
                        "reactivate": True,
                        "reactivation_type": "param_tanh",
                        "reactivation_init_policy": "occupancy_quantile",
                        "weight_transform": "softplus",
                        "topk_type": "standard",
                        "blocklinear": {"efficient": True},
                    },
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [1],
                            "population": {
                                "ff_excitatory_synapses": 3,
                                "ff_inhibitory_synapses": 1,
                            },
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ],
        },
    }
    transformer_replacement = {
        "enabled": True,
        "layers": [0, 1],
        "target_module": "mlp",
        "replacement_kwargs": {"kind": "ei_stack"},
    }

    kwargs = build_ei_stack_kwargs_from_core_config(
        core_config,
        transformer_replacement=transformer_replacement,
        num_layers=2,
    )
    assert kwargs["excitatory_cells"] == 5
    assert kwargs["inhibitory_cells"] == 2
    assert kwargs["ee_synapses_per_branch"] == 2
    assert kwargs["ei_synapses_per_branch"] == 3
    assert kwargs["ie_synapses_per_branch"] == 1
    assert kwargs["ii_synapses_per_branch"] == 1
    assert kwargs["efficient_blocklinear"] is True

    model = TinyHFLikeModel(n_layers=2)
    records = apply_transformer_replacement_config(
        model,
        transformer_replacement,
        core_config,
    )
    assert len(records) == 2
    assert isinstance(model.model.layers[0].mlp, EIStackMLPSlot)
