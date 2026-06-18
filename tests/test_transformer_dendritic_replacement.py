from pathlib import Path

import pytest
import torch
import torch.nn as nn

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
)

TRANSFORMER_BENCHMARK_CONFIGS = sorted(
    Path("jobs/transformer_replacement_benchmarks/configs").rglob("*.yaml")
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


class TinyHFLikeModel(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyDecoderLayer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


@pytest.mark.parametrize(
    "config_path",
    TRANSFORMER_BENCHMARK_CONFIGS,
    ids=[p.name for p in TRANSFORMER_BENCHMARK_CONFIGS],
)
def test_transformer_benchmark_job_config_loads(config_path):
    """Benchmark job configs live with the job package, but remain normal configs."""
    load_config.cache_clear()
    cfg = load_config(str(config_path))

    assert cfg.model.transformer_replacement.enabled


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
        "sparsity": {
            "type": "indexed_dynamic",
            "temperature": 0.25,
            "indexed": {
                "candidate_size": 4,
                "selection": "stochastic",
                "seed": 5,
                "rewire_frequency": 10,
                "rewire_quantile": 0.1,
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
    assert kwargs["topk_type"] == "indexed_dynamic"
    assert kwargs["indexed_candidate_size"] == 4
    assert kwargs["indexed_selection"] == "stochastic"
    assert kwargs["indexed_rewire_frequency"] == 10
    assert kwargs["indexed_rewire_quantile"] == 0.1

    model = TinyHFLikeModel(n_layers=2)
    records = apply_transformer_replacement_config(
        model,
        transformer_replacement,
        core_config,
    )
    assert len(records) == 1
    replacement = model.model.layers[1].mlp
    assert isinstance(replacement, DendriticFFNReplacement)
    assert replacement.topk_type == "indexed_dynamic"
    assert replacement.indexed_candidate_size == 4


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
    x = torch.randn(2, 4, 8)
    y = model(x)
    assert y.shape == x.shape
    estimate = model.model.layers[0].mlp.parameter_estimate()
    assert estimate["stored_total"] > 0
    assert estimate["active_total"] > 0
