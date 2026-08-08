"""Configuration-contract tests for the large indexed-backend examples."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.activations.parametric import ParametricTanh
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.sweeps.unified.sweep_types.unified_sweep import (
    UnifiedSweepGenerator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "configs" / "examples" / "indexed_backends"
BACKENDS = {"eager", "recompute", "triton", "cuda", "auto"}
EXAMPLES = {
    "feedforward": (
        EXAMPLE_DIR / "feedforward_large.yaml",
        ["model.core.sparsity.indexed.projection_backend"],
    ),
    "rnn": (
        EXAMPLE_DIR / "rnn_large.yaml",
        [
            "model.core.unified_ei.layers.0.excitatory.indexed_projection_backend",
            "model.core.unified_ei.layers.0.inhibitory.indexed_projection_backend",
        ],
    ),
    "alexnet": (
        EXAMPLE_DIR / "alexnet_fc6_large.yaml",
        ["model.core.sparsity.indexed.projection_backend"],
    ),
}


def _generated_configs(path: Path):
    request = OmegaConf.load(path)
    generator = UnifiedSweepGenerator()
    assert generator.validate_config(request)
    return request, generator.generate_configs(request)


@pytest.mark.parametrize("family", EXAMPLES)
def test_large_backend_example_expands_to_every_mode_and_loads(family, tmp_path):
    path, selector_paths = EXAMPLES[family]
    request, generated = _generated_configs(path)

    assert request.sweep_contract.expected_config_count == len(BACKENDS)
    assert {config._sweep_variant for config in generated} == BACKENDS
    assert len(generated) == len(BACKENDS)

    for config in generated:
        selected = config._sweep_variant
        assert all(OmegaConf.select(config, key) == selected for key in selector_paths)
        generated_path = tmp_path / f"{family}_{selected}.yaml"
        OmegaConf.save(config, generated_path)
        load_config.cache_clear()
        loaded = load_config(str(generated_path))
        assert loaded.model.core.type == config.model.core.type


def test_large_backend_examples_preserve_architecture_contracts():
    feedforward = OmegaConf.load(EXAMPLES["feedforward"][0]).base_config
    recurrent = OmegaConf.load(EXAMPLES["rnn"][0]).base_config
    alexnet = OmegaConf.load(EXAMPLES["alexnet"][0]).base_config

    assert feedforward.model.core.architecture.excitatory_layer_sizes == [
        8192,
        8192,
        8192,
    ]
    assert feedforward.model.core.reactivation.memory_efficient is True

    layer = recurrent.model.core.unified_ei.layers[0]
    assert layer.recurrent is True
    assert layer.excitatory.n_neurons + layer.inhibitory.n_neurons == 5120
    assert layer.excitatory.reactivation_memory_efficient is True
    assert layer.inhibitory.reactivation_memory_efficient is True

    replacement = alexnet.model.vision_replacement
    assert replacement.enabled is True
    assert replacement.backbone == "alexnet"
    assert replacement.core_source == "configured"
    assert replacement.target_modules == ["classifier.1", "classifier.2"]
    assert alexnet.model.core.architecture.excitatory_layer_sizes == [4096]
    assert alexnet.model.core.sparsity.type == "indexed"
    assert alexnet.model.core.reactivation.memory_efficient is True


def test_reference_configs_expose_both_indexed_option_schemas():
    template = OmegaConf.load(
        REPO_ROOT / "configs" / "templates" / "indexed_runtime_template.yaml"
    )
    structured = template.model.core.sparsity.indexed
    assert structured.projection_backend == "eager"
    assert structured.index_dtype == "int64"
    assert structured.workspace_mb is None
    assert structured.cache_transformed_weights is False
    assert structured.persistent_indices is True
    assert structured.init_mode == "per_rank"
    assert template.model.core.reactivation.memory_efficient is False

    reference = OmegaConf.load(REPO_ROOT / "configs" / "config_exp.yaml")
    defaults = reference.model.core.population_network.layers[0].population_defaults
    assert defaults.indexed_projection_backend == "eager"
    assert defaults.indexed_index_dtype == "int64"
    assert defaults.indexed_workspace_mb is None
    assert defaults.indexed_cache_transformed_weights is False
    assert defaults.indexed_persistent_indices is True
    assert defaults.indexed_init_mode == "per_rank"
    assert defaults.reactivation_memory_efficient is False


def _write_reduced_config(config, tmp_path: Path, name: str):
    path = tmp_path / name
    OmegaConf.save(config, path)
    load_config.cache_clear()
    return load_config(str(path))


@pytest.mark.parametrize("backend", ["eager", "recompute", "auto"])
def test_feedforward_example_portable_variants_forward_and_backward(backend, tmp_path):
    _, variants = _generated_configs(EXAMPLES["feedforward"][0])
    config = next(item for item in variants if item._sweep_variant == backend)
    config.model.encoder.params.input_dim = 32
    architecture = config.model.core.architecture
    architecture.excitatory_layer_sizes = [16, 16]
    architecture.inhibitory_layer_sizes = [4, 4]
    architecture.excitatory_branch_factors = [2]
    architecture.inhibitory_branch_factors = [1]
    connectivity = config.model.core.connectivity
    connectivity.ee_synapses_per_branch_per_layer = [4, 4]
    connectivity.ei_synapses_per_branch_per_layer = [2, 2]
    connectivity.ie_synapses_per_branch_per_layer = [2, 2]
    connectivity.ii_synapses_per_branch_per_layer = [0, 0]
    config.model.decoder.params.output_dim = 3

    loaded = _write_reduced_config(config, tmp_path, f"feedforward_{backend}.yaml")
    model, _ = initialize_model(loaded.model)
    value = torch.rand(2, 32, requires_grad=True)
    output = model(value)
    output.square().mean().backward()

    indexed = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
    activations = [m for m in model.modules() if isinstance(m, ParametricTanh)]
    assert indexed and all(m.projection_backend == backend for m in indexed)
    assert activations and all(m.memory_efficient for m in activations)
    assert value.grad is not None and torch.isfinite(output).all()


@pytest.mark.parametrize("backend", ["eager", "recompute", "auto"])
def test_rnn_example_portable_variants_forward_and_backward(backend, tmp_path):
    _, variants = _generated_configs(EXAMPLES["rnn"][0])
    config = next(item for item in variants if item._sweep_variant == backend)
    layer = config.model.core.unified_ei.layers[0]
    excitatory = layer.excitatory
    excitatory.n_neurons = 16
    excitatory.branch_factors = [2]
    excitatory.ff_excitatory_synapses_by_level = [0, 4]
    excitatory.ff_inhibitory_synapses_by_level = [0, 0]
    excitatory.rec_excitatory_synapses_by_level = [2, 4]
    excitatory.rec_inhibitory_synapses_by_level = [1, 2]
    excitatory.level_taus = [20.0, 5.0]
    inhibitory = layer.inhibitory
    inhibitory.n_neurons = 4
    inhibitory.ff_excitatory_synapses_by_level = [4]
    inhibitory.rec_excitatory_synapses_by_level = [4]
    inhibitory.rec_inhibitory_synapses_by_level = [2]
    config.model.core.unified_ei.input_projection_dims = [8]

    loaded = _write_reduced_config(config, tmp_path, f"rnn_{backend}.yaml")
    model, _ = initialize_model(loaded.model)
    value = torch.rand(2, 3, 10, requires_grad=True)
    output = model(value)
    output.square().mean().backward()

    indexed = [m for m in model.modules() if isinstance(m, IndexedSparseLinear)]
    activations = [m for m in model.modules() if isinstance(m, ParametricTanh)]
    assert indexed and all(m.projection_backend == backend for m in indexed)
    assert activations and all(m.memory_efficient for m in activations)
    assert value.grad is not None and torch.isfinite(output).all()
