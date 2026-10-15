import torch

from dendritic_modeling.networks import DendriticBranchLayer, DenseToSparseLinear
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.structured_einet_factory import (
    _structured_population_sparsity_kwargs,
)
from dendritic_modeling.utils.sparse_ops import get_available_sparse_types


def test_dense_to_sparse_registered_aliases():
    available = get_available_sparse_types()
    assert "dense_to_sparse" in available
    assert "annealed_topk" in available


def test_dense_to_sparse_schedule_reduces_to_target_k():
    layer = DenseToSparseLinear(
        in_features=6,
        out_features=2,
        K=2,
        initial_density=1.0,
        start_step=0,
        end_step=4,
        schedule="linear",
        advance_on_forward=False,
        weight_transform="identity",
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.arange(12, dtype=torch.float32).view(2, 6))

    assert layer.initial_k == 6
    assert layer.target_k == 2
    assert layer.current_k == 6
    assert torch.all(layer.weight_mask().sum(dim=1) == 6)

    layer.set_sparsity_step(2)
    assert layer.current_k == 4
    assert torch.all(layer.weight_mask().sum(dim=1) == 4)

    layer.set_sparsity_step(4)
    assert layer.current_k == 2
    assert torch.all(layer.weight_mask().sum(dim=1) == 2)


def test_dense_phase_gives_all_allowed_synapses_gradient():
    layer = DenseToSparseLinear(
        in_features=5,
        out_features=3,
        K=2,
        initial_density=1.0,
        end_step=10,
        advance_on_forward=False,
        weight_transform="softplus",
    )
    with torch.no_grad():
        layer.pre_w.fill_(0.1)

    out = layer(torch.ones(4, 5)).sum()
    out.backward()

    assert layer.pre_w.grad is not None
    assert torch.all(layer.pre_w.grad != 0)


def test_dense_to_sparse_freeze_on_end_keeps_final_mask_fixed():
    layer = DenseToSparseLinear(
        in_features=5,
        out_features=1,
        K=2,
        initial_density=1.0,
        end_step=1,
        freeze_on_end=True,
        advance_on_forward=False,
        weight_transform="identity",
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))

    layer.set_sparsity_step(1)
    frozen_mask = layer.weight_mask().clone()
    assert torch.equal(frozen_mask, torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]))

    with torch.no_grad():
        layer.pre_w[0, 0] = 100.0
    assert torch.equal(layer.weight_mask(), frozen_mask)


def test_branch_layer_uses_dense_to_sparse_and_advances_on_forward():
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=6,
        excitatory_synapses_per_branch=2,
        topk_type="dense_to_sparse",
        dense_to_sparse_initial_density=1.0,
        dense_to_sparse_end_step=2,
        dense_to_sparse_schedule="linear",
        reactivate=False,
    )

    assert isinstance(layer.branch_excitation, DenseToSparseLinear)
    assert layer.branch_excitation.current_k == 6
    layer.train()
    _ = layer(torch.ones(2, 6))
    assert int(layer.branch_excitation.sparsity_step.item()) == 1


def test_structured_einet_config_propagates_dense_to_sparse():
    core = {
        "biological_neuron": False,
        "architecture": {
            "excitatory_layer_sizes": [4],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ei_synapses_per_branch_per_layer": [0],
            "ie_synapses_per_branch_per_layer": [0],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {"output_activation": "none"},
        "morphology": {"use_shunting": False, "weight_transform": "identity"},
        "sparsity": {
            "type": "annealed_topk",
            "dense_to_sparse": {
                "initial_density": 1.0,
                "end_step": 3,
                "schedule": "linear",
            },
        },
        "reactivation": {"enabled": False},
    }

    net = get_architecture("einet", core, input_dim=5)
    dense_layers = [m for m in net.modules() if isinstance(m, DenseToSparseLinear)]
    assert dense_layers
    assert all(layer.initial_k == layer.in_features for layer in dense_layers)
    assert all(layer.target_k == 2 for layer in dense_layers)


def test_structured_population_sparsity_kwargs_preserve_all_sparse_fields():
    kwargs = _structured_population_sparsity_kwargs(
        sparsity={
            "type": "annealed-topk",
            "init_method": "xavier_uniform",
            "noise_level": 0.2,
            "weight_norm_order": 2,
            "gamma": 0.8,
            "temperature": 0.3,
            "ultrafast": True,
            "gradient_scaling": "conductance_dynamic",
            "seed": 11,
        },
        dense_to_sparse={
            "initial_density": 0.6,
            "initial_k": 3,
            "start_step": 2,
            "end_step": 20,
            "update_interval": 4,
            "schedule": "linear",
            "freeze_on_end": True,
            "advance_on_forward": False,
            "prune_metric": "magnitude",
        },
        indexed={
            "seed": 13,
            "candidate_size": 7,
            "selection": "stochastic",
            "output_chunk_size": 128,
            "index_dtype": "int32",
            "workspace_mb": 64,
            "cache_transformed_weights": True,
            "projection_backend": "auto",
            "persistent_indices": False,
            "init_mode": "rank0_broadcast",
            "rewire_frequency": 5,
            "rewire_quantile": 0.15,
            "rewire_until_step": 30,
        },
    )

    assert kwargs == {
        "topk_init_method": "xavier_uniform",
        "topk_noise_level": 0.2,
        "topk_type": "dense_to_sparse",
        "topk_weight_norm_order": 2,
        "topk_gamma": 0.8,
        "topk_temperature": 0.3,
        "topk_ultrafast": True,
        "topk_gradient_scaling": "conductance_dynamic",
        "topk_variance_score": "absolute",
        "dense_to_sparse_initial_density": 0.6,
        "dense_to_sparse_initial_k": 3,
        "dense_to_sparse_start_step": 2,
        "dense_to_sparse_end_step": 20,
        "dense_to_sparse_update_interval": 4,
        "dense_to_sparse_schedule": "linear",
        "dense_to_sparse_freeze_on_end": True,
        "dense_to_sparse_advance_on_forward": False,
        "dense_to_sparse_prune_metric": "magnitude",
        "release_p_min": 0.5,
        "release_p_max": 1.0,
        "release_weight_scale": None,
        "release_impact_decay": 0.99,
        "release_impact_metric": "drive",
        "release_max_conductance": None,
        "release_track_source_activity": False,
        "indexed_seed": 13,
        "indexed_candidate_size": 7,
        "indexed_selection": "stochastic",
        "indexed_output_chunk_size": 128,
        "indexed_index_dtype": "int32",
        "indexed_workspace_mb": 64,
        "indexed_cache_transformed_weights": True,
        "indexed_recompute_backward": False,
        "indexed_projection_backend": "auto",
        "indexed_persistent_indices": False,
        "indexed_init_mode": "rank0_broadcast",
        "indexed_rewire_frequency": 5,
        "indexed_rewire_quantile": 0.15,
        "indexed_rewire_until_step": 30,
    }
