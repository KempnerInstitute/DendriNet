"""Plugin sparse layer types: fixed-index subclasses build like ``indexed`` and take their own options."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.config.model import SparsityConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    TopKConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
    BUILTIN_SPARSE_LAYER_TYPES,
    get_sparse_layer_class,
    register_sparse_layer,
    unregister_sparse_layer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.structured_population_kwargs import (
    _structured_population_sparsity_kwargs,
)
from dendritic_modeling.networks.architectures.transformer.dendritic_ffn import (
    DendriticFFNReplacement,
    LayerwiseEIStackReplacement,
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
)


class ScaledIndexed(IndexedSparseLinear):
    """A fixed-index layer whose weights its ``scale`` option multiplies."""

    def __init__(self, *args, scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = float(scale)

    def sparse_weight(self) -> torch.Tensor:
        return super().sparse_weight() * self.scale


class TaggedTopK(TopKLinear):
    """A dense Top-K layer that records its ``tag`` option."""

    def __init__(self, *args, tag: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.tag = tag


@pytest.fixture
def plugin_types():
    register_sparse_layer("test_scaled_indexed", ScaledIndexed)
    register_sparse_layer("test_tagged_topk", TaggedTopK)
    yield
    unregister_sparse_layer("test_scaled_indexed")
    unregister_sparse_layer("test_tagged_topk")


def _network(topk_type: str, **population) -> torch.nn.Module:
    defaults = {
        "ff_excitatory_synapses": 3,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "somatic_synapses": True,
        "reactivate": False,
        "use_shunting": False,
        "dbl_init_method": "default",
        "weight_transform": "identity",
        "topk_type": topk_type,
        "indexed_seed": 17,
        "indexed_projection_backend": "recompute",
        **population,
    }
    config = {
        "biological_neuron": False,
        "population_network": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": defaults,
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ]
        },
    }
    torch.manual_seed(0)
    return get_architecture("population_network", config, input_dim=8)


def _bank(network: torch.nn.Module, cls: type) -> torch.nn.Module:
    banks = [m for m in network.modules() if isinstance(m, cls)]
    assert len(banks) == 1
    return banks[0]


def test_the_registry_tells_built_in_from_plugin_types(plugin_types):
    assert "indexed" in BUILTIN_SPARSE_LAYER_TYPES
    assert "test_scaled_indexed" not in BUILTIN_SPARSE_LAYER_TYPES
    assert get_sparse_layer_class("test_scaled_indexed") is ScaledIndexed
    assert get_sparse_layer_class("indexed") is IndexedSparseLinear
    with pytest.raises(ValueError, match="Unknown sparse layer type"):
        get_sparse_layer_class("no_such_type")


def test_a_plugin_indexed_type_builds_like_indexed_and_takes_its_options(
    plugin_types,
):
    reference = _network("indexed")
    plugin = _network("test_scaled_indexed", topk_options={"scale": 2.0})
    built = _bank(plugin, ScaledIndexed)
    twin = _bank(reference, IndexedSparseLinear)
    assert built.scale == 2.0
    assert built.projection_backend == twin.projection_backend == "recompute"
    assert built.output_chunk_size == twin.output_chunk_size
    assert torch.equal(built.connection_indices, twin.connection_indices)
    assert torch.equal(built.pre_w, twin.pre_w)
    x = torch.randn(4, 8, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(plugin(x), 2 * reference(x))


def test_options_reach_a_plugin_type_that_is_not_indexed(plugin_types):
    network = _network(
        "test_tagged_topk",
        topk_options={"tag": "kept"},
        indexed_seed=None,
        indexed_projection_backend="eager",
    )
    assert _bank(network, TaggedTopK).tag == "kept"


def test_options_on_a_built_in_type_are_rejected():
    with pytest.raises(ValueError, match="only plugin sparse layer types"):
        _network("indexed", topk_options={"scale": 2.0})


def test_a_plugin_indexed_type_rejects_options_indexed_does_not_read(plugin_types):
    with pytest.raises(ValueError, match="'test_scaled_indexed' does not apply"):
        _network("test_scaled_indexed", indexed_rewire_frequency=7)


def test_structured_sparsity_options_reach_the_population_options():
    kwargs = _structured_population_sparsity_kwargs(
        sparsity={"type": "test_scaled_indexed", "options": {"scale": 3.0}},
        dense_to_sparse={},
        indexed={},
    )
    assert kwargs["topk_options"] == {"scale": 3.0}
    assert (
        _structured_population_sparsity_kwargs(
            sparsity={}, dense_to_sparse={}, indexed={}
        )["topk_options"]
        == {}
    )


def _transformer_core(**sparsity) -> dict:
    return {
        "architecture": {
            "dendritic_units": 6,
            "branch_factors": [2],
            "excitatory_layer_sizes": [4],
            "inhibitory_layer_sizes": [2],
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [2],
        },
        "connectivity": {
            "synapses_per_branch": 3,
            "ee_synapses_per_branch_per_layer": [3],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [2],
            "ii_synapses_per_branch_per_layer": [2],
        },
        "sparsity": {"indexed": {"seed": 3}, **sparsity},
        "morphology": {"use_shunting": False, "weight_transform": "identity"},
        "reactivation": {"enabled": False},
    }


def _scales(module: torch.nn.Module) -> set[float]:
    return {b.scale for b in module.modules() if isinstance(b, ScaledIndexed)}


def test_the_transformer_ffn_builder_passes_plugin_options(plugin_types):
    core = _transformer_core(type="test_scaled_indexed", options={"scale": 2.0})
    kwargs = build_dendritic_ffn_kwargs_from_core_config(core)
    assert kwargs["topk_options"] == {"scale": 2.0}
    assert _scales(DendriticFFNReplacement(hidden_size=8, **kwargs)) == {2.0}
    # Without options the key stays out, so replacement classes that do not
    # take it build every configuration they built before.
    assert "topk_options" not in build_dendritic_ffn_kwargs_from_core_config(
        _transformer_core(type="indexed")
    )


def test_the_ei_stack_builder_passes_plugin_options(plugin_types):
    core = _transformer_core(type="test_scaled_indexed", options={"scale": 3.0})
    kwargs = build_ei_stack_kwargs_from_core_config(core, num_layers=2)
    assert kwargs["topk_options"] == {"scale": 3.0}
    assert _scales(LayerwiseEIStackReplacement(hidden_size=8, **kwargs)) == {3.0}
    assert "topk_options" not in build_ei_stack_kwargs_from_core_config(
        _transformer_core(type="indexed"), num_layers=1
    )


@pytest.mark.parametrize("value", [False, 0, [], ""])
def test_options_that_are_not_a_mapping_are_rejected(plugin_types, value):
    with pytest.raises(
        ValueError, match=r"model\.core\.sparsity\.options must be a mapping"
    ):
        SparsityConfig(options=value)
    with pytest.raises(ValueError, match="population topk_options must be a mapping"):
        PopulationConfig(topk_options=value)
    with pytest.raises(ValueError, match="must be a mapping"):
        TopKConfig(options=value)
    with pytest.raises(ValueError, match=r"sparsity\.options must be a mapping"):
        _structured_population_sparsity_kwargs(
            sparsity={"options": value}, dense_to_sparse={}, indexed={}
        )
    core = _transformer_core(type="test_scaled_indexed", options=value)
    for build in (
        build_dendritic_ffn_kwargs_from_core_config,
        lambda core: build_ei_stack_kwargs_from_core_config(core, num_layers=1),
    ):
        with pytest.raises(ValueError, match=r"model\.core\.sparsity\.options"):
            build(core)
    with pytest.raises(ValueError, match="topk_options must be a mapping"):
        _network("test_scaled_indexed", topk_options=value)


def test_omitted_or_null_options_mean_none():
    assert SparsityConfig().options == {}
    assert SparsityConfig(options=None).options == {}
    assert PopulationConfig(topk_options=None).topk_options == {}
    assert TopKConfig(options=None).options == {}
