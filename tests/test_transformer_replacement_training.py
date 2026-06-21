from types import SimpleNamespace

import torch

from dendritic_modeling.config import Config
from dendritic_modeling.config.training import TransformerReplacementTrainingConfig
from dendritic_modeling.networks.architectures.transformer import (
    EIStackMLPSlot,
    LayerwiseEIStackReplacement,
    ReplacementRecord,
)
from dendritic_modeling.training import (
    benchmark_saved_transformer_replacement,
    run_transformer_replacement_training,
)
from dendritic_modeling.training._transformer_replacement.common import (
    _make_tensor_loader,
)
from dendritic_modeling.training._transformer_replacement.loops import (
    _infinite_loader,
)
from dendritic_modeling.training.transformer_replacement import (
    _load_replacement_record_checkpoints,
    _make_replacement,
    _select_joint_trainable_parameters,
)


def _small_distillation_config(tmp_path):
    cfg = Config()
    cfg.experiment.seed = 123
    cfg.model.transformer_replacement.enabled = True
    cfg.model.transformer_replacement.layers = [0]
    cfg.model.transformer_replacement.input_transform = "signed_split"
    cfg.model.transformer_replacement.replacement_kwargs = {
        "output_init_std": 0.05,
    }

    cfg.model.core.type = "dendritic_ffn"
    cfg.model.core.architecture.excitatory_layer_sizes = [16]
    cfg.model.core.architecture.excitatory_branch_factors = [2]
    cfg.model.core.connectivity.ee_synapses_per_branch_per_layer = [3]
    cfg.model.core.morphology.use_shunting = False
    cfg.model.core.morphology.weight_transform = "softplus"
    cfg.model.core.sparsity.type = "indexed_rewire"
    cfg.model.core.sparsity.indexed = {
        "seed": 19,
        "rewire_frequency": 100,
        "rewire_quantile": 0.05,
        "rewire_until_step": None,
        "output_chunk_size": 128,
    }
    cfg.model.core.reactivation.enabled = True
    cfg.model.core.reactivation.type = "param_tanh"

    cfg.training.main.optimizer.name = "adamw"
    cfg.training.main.optimizer.lr = 0.01
    cfg.training.main.optimizer.weight_decay = 0.0
    cfg.training.transformer_replacement = TransformerReplacementTrainingConfig(
        enabled=True,
        mode="layerwise_distillation",
        teacher_source="synthetic_mlp",
        train_target="replacement_only",
        layers=[0],
        device="cpu",
        dtype="float32",
        seed=123,
        hidden_size=4,
        intermediate_size=16,
        sequence_length=3,
        train_samples=128,
        valid_samples=64,
        batch_size=32,
        max_steps=40,
        eval_every=10,
        log_every=100,
        save_dir=str(tmp_path),
        save_replacements=True,
        synthetic_teacher={
            "kind": "linear",
            "activation": "identity",
            "bias": False,
            "input_std": 1.0,
        },
    )
    return cfg


def test_synthetic_transformer_replacement_distillation_reduces_loss(tmp_path):
    torch.set_num_threads(1)
    cfg = _small_distillation_config(tmp_path)

    result = run_transformer_replacement_training(cfg)

    assert result.final_valid_loss < result.initial_valid_loss
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "layer_0_replacement.pt").exists()

    benchmark = benchmark_saved_transformer_replacement(cfg)
    assert benchmark.trained["mse"] < benchmark.initial["mse"]
    assert (tmp_path / "benchmark_metrics.json").exists()


def test_transformer_replacement_infinite_loader_restarts_without_cycle_cache():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(3).unsqueeze(1), torch.arange(3).unsqueeze(1)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    iterator = _infinite_loader(loader)
    batches = [next(iterator)[0].squeeze(1).tolist() for _ in range(4)]

    assert batches == [[0, 1], [2], [0, 1], [2]]


def test_transformer_replacement_tensor_loader_preserves_tail_batch():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(3).unsqueeze(1), torch.arange(3).unsqueeze(1)
    )

    loader = _make_tensor_loader(dataset, batch_size=2, shuffle=False)
    batches = [batch[0].squeeze(1).tolist() for batch in loader]

    assert batches == [[0, 1], [2]]


def test_transformer_replacement_tensor_loader_honors_explicit_tuning():
    dataset = torch.utils.data.TensorDataset(
        torch.arange(3).unsqueeze(1), torch.arange(3).unsqueeze(1)
    )
    train_cfg = SimpleNamespace(
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=3,
    )

    loader = _make_tensor_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        train_cfg=train_cfg,
        device="cuda",
    )

    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.persistent_workers is False
    assert loader.prefetch_factor == 3
    assert loader.drop_last is False


def test_layerwise_replacement_factory_honors_ei_stack_kind(tmp_path):
    cfg = _small_distillation_config(tmp_path)
    cfg.model.transformer_replacement.replacement_kwargs = {
        "kind": "ei_stack",
        "excitatory_cells": 5,
        "inhibitory_cells": 2,
        "excitatory_branch_factors": [1],
        "inhibitory_branch_factors": [1],
        "ee_synapses_per_branch": 2,
        "ei_synapses_per_branch": 1,
        "ie_synapses_per_branch": 1,
        "ii_synapses_per_branch": 1,
        "topk_type": "standard",
    }

    replacement = _make_replacement(
        hidden_size=4,
        config=cfg,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert isinstance(replacement, EIStackMLPSlot)
    x = torch.randn(2, 3, 4)
    assert replacement(x).shape == x.shape


def test_ei_stack_checkpoint_loader_maps_single_layer_slots(tmp_path):
    replacement_kwargs = {
        "kind": "ei_stack",
        "excitatory_cells": 5,
        "inhibitory_cells": 2,
        "excitatory_branch_factors": [1],
        "inhibitory_branch_factors": [1],
        "ee_synapses_per_branch": 2,
        "ei_synapses_per_branch": 1,
        "ie_synapses_per_branch": 1,
        "ii_synapses_per_branch": 1,
        "topk_type": "standard",
    }
    cfg = _small_distillation_config(tmp_path)
    cfg.model.transformer_replacement.replacement_kwargs = replacement_kwargs

    for layer_idx, value in enumerate((1.0, 2.0)):
        replacement = _make_replacement(
            hidden_size=4,
            config=cfg,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            projection = replacement.stack.output_projections[0]
            projection.weight.fill_(value)
        torch.save(
            {"state_dict": replacement.state_dict()},
            tmp_path / f"layer_{layer_idx}_replacement.pt",
        )

    shared_stack = LayerwiseEIStackReplacement(
        hidden_size=4,
        num_layers=2,
        excitatory_cells=5,
        inhibitory_cells=2,
        excitatory_branch_factors=[1],
        inhibitory_branch_factors=[1],
        ee_synapses_per_branch=2,
        ei_synapses_per_branch=1,
        ie_synapses_per_branch=1,
        ii_synapses_per_branch=1,
        topk_type="standard",
    )
    records = [
        ReplacementRecord(
            layer_index=0,
            mlp_attr="mlp",
            original_mlp=torch.nn.Identity(),
            replacement=EIStackMLPSlot(shared_stack, 0),
        ),
        ReplacementRecord(
            layer_index=1,
            mlp_attr="mlp",
            original_mlp=torch.nn.Identity(),
            replacement=EIStackMLPSlot(shared_stack, 1),
        ),
    ]

    _load_replacement_record_checkpoints(
        records,
        str(tmp_path),
        device=torch.device("cpu"),
    )

    assert torch.allclose(
        shared_stack.output_projections[0].weight,
        torch.ones_like(shared_stack.output_projections[0].weight),
    )
    assert torch.allclose(
        shared_stack.output_projections[1].weight,
        torch.full_like(shared_stack.output_projections[1].weight, 2.0),
    )


def test_joint_trainable_parameter_selection_supports_full_model():
    original = torch.nn.Linear(4, 4)
    replacement = torch.nn.Linear(4, 4)
    model = torch.nn.Sequential(original, replacement)
    records = [
        ReplacementRecord(
            layer_index=0,
            mlp_attr="mlp",
            original_mlp=original,
            replacement=replacement,
        )
    ]

    replacement_params = _select_joint_trainable_parameters(
        model,
        records,
        "replacement_only",
    )
    assert {id(param) for param in replacement_params} == {
        id(param) for param in replacement.parameters()
    }
    assert all(not param.requires_grad for param in original.parameters())
    assert all(param.requires_grad for param in replacement.parameters())

    full_model_params = _select_joint_trainable_parameters(
        model,
        records,
        "full_model",
    )
    assert {id(param) for param in full_model_params} == {
        id(param) for param in model.parameters()
    }
    assert all(param.requires_grad for param in model.parameters())
