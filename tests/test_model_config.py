"""Tests for config/model.py dataclasses."""

import pytest

from dendritic_modeling.config.extensions import DisabledExtensionConfig
from dendritic_modeling.config.model import (
    ArchitectureConfig,
    BlockLinearConfig,
    ConnectivityConfig,
    CoreConfig,
    DecoderConfig,
    DecoderParamsConfig,
    EncoderConfig,
    EncoderParamsConfig,
    ImplementationConfig,
    ModelConfig,
    MorphologyConfig,
    ReactivationConfig,
    SparsityConfig,
    StructuredConnectivityConfig,
    TransferFunctionConfig,
    TransformerReplacementConfig,
)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.task == "classification"
    assert cfg.fixed_output_scale == 1.0
    assert cfg.output_scale_mode is None
    assert isinstance(cfg.encoder, EncoderConfig)
    assert isinstance(cfg.core, CoreConfig)
    assert isinstance(cfg.decoder, DecoderConfig)
    assert isinstance(cfg.vision_replacement, DisabledExtensionConfig)
    assert isinstance(cfg.pretrained_replacement, DisabledExtensionConfig)
    assert cfg.vision_replacement.enabled is False
    assert cfg.pretrained_replacement.enabled is False
    assert isinstance(cfg.transformer_replacement, TransformerReplacementConfig)
    assert cfg.core.vision == {}


def test_architecture_config_defaults():
    cfg = ArchitectureConfig()
    assert cfg.excitatory_layer_sizes == [20]
    assert cfg.inhibitory_layer_sizes == []
    assert cfg.excitatory_branch_factors == [3, 3, 3, 3]
    assert cfg.inhibitory_network_type == "dendritic"


def test_connectivity_config_defaults():
    cfg = ConnectivityConfig()
    assert cfg.ee_synapses_per_branch_per_layer == [40]
    assert cfg.ei_synapses_per_branch_per_layer == [0]
    assert cfg.ie_synapses_per_branch_per_layer == [10]
    assert cfg.ii_synapses_per_branch_per_layer == [0]
    assert isinstance(cfg.structured, StructuredConnectivityConfig)
    assert cfg.structured.enabled is False


def test_morphology_config_defaults():
    cfg = MorphologyConfig()
    assert cfg.use_shunting is True
    assert cfg.allow_self_recurrence is True
    assert cfg.weight_transform == "softplus"
    assert cfg.dbl_init_method == "analytical_expectation"
    assert cfg.somatic_synapses is False


def test_sparsity_config_defaults():
    cfg = SparsityConfig()
    assert cfg.init_method == "xavier_normal"
    assert cfg.noise_level == 0.0
    assert cfg.type == "standard"
    # The stochastic defaults match StochasticTopKLinear and PopulationConfig.
    assert cfg.temperature == 0.5
    assert cfg.ultrafast is True
    assert isinstance(cfg.deepst, dict)
    assert isinstance(cfg.indexed, dict)
    assert "excitatory_target_density" in cfg.deepst
    assert cfg.indexed["selection"] == "standard"


def test_transformer_replacement_config_defaults():
    cfg = TransformerReplacementConfig()
    assert cfg.enabled is False
    assert cfg.layers == []
    assert cfg.target_module == "mlp"
    assert cfg.input_transform == "signed_split"


def test_reactivation_config_defaults():
    cfg = ReactivationConfig()
    assert cfg.enabled is True
    assert cfg.type == "param_tanh"
    assert cfg.init_m == 1.5
    assert cfg.init_b == 0.5
    assert cfg.init_policy == "analytical"
    assert cfg.occupancy_quantile_low is None
    assert cfg.occupancy_quantile_high is None
    assert cfg.occupancy_target_low is None
    assert cfg.occupancy_target_high is None
    assert cfg.sigma_aware_k == 0.25


def test_transfer_function_config_defaults():
    cfg = TransferFunctionConfig()
    assert cfg.input_transform is None
    assert cfg.independent_pathways is False
    assert cfg.output_activation is None


def test_implementation_config_defaults():
    cfg = ImplementationConfig()
    assert cfg.adaptive_initialization is True
    assert cfg.adaptive_initialization_policy == "preserve_shunting_center"
    assert cfg.compile_forward is False


def test_encoder_config_defaults():
    cfg = EncoderConfig()
    assert cfg.type == "identity"
    assert isinstance(cfg.params, EncoderParamsConfig)


def test_decoder_config_defaults():
    cfg = DecoderConfig()
    assert cfg.type == "MLP"
    assert isinstance(cfg.params, DecoderParamsConfig)
    assert cfg.params.output_dim == 10


# ---------------------------------------------------------------------------
# Nested structure
# ---------------------------------------------------------------------------


def test_core_config_nested():
    cfg = CoreConfig()
    assert cfg.type == "einet"
    assert isinstance(cfg.architecture, ArchitectureConfig)
    assert isinstance(cfg.connectivity, ConnectivityConfig)
    assert isinstance(cfg.transfer, TransferFunctionConfig)
    assert isinstance(cfg.morphology, MorphologyConfig)
    assert isinstance(cfg.sparsity, SparsityConfig)
    assert isinstance(cfg.reactivation, ReactivationConfig)
    assert isinstance(cfg.blocklinear, BlockLinearConfig)
    assert isinstance(cfg.implementation, ImplementationConfig)


# ---------------------------------------------------------------------------
# asdict roundtrip
# ---------------------------------------------------------------------------


def test_model_config_asdict():
    cfg = ModelConfig()
    d = cfg.asdict()
    assert isinstance(d, dict)
    assert "task" in d
    assert "encoder" in d
    assert "core" in d
    assert "decoder" in d
    # Nested configs should be dicts too
    assert isinstance(d["core"], dict)
    assert "architecture" in d["core"]


def test_core_config_asdict_keys():
    cfg = CoreConfig()
    d = cfg.asdict()
    expected_keys = {
        "type",
        "target_active_parameters",
        "biological_neuron",
        "affine_mlp",
        "shared_projection_shunting",
        "routed_ei",
        "initialization_seed",
        "architecture",
        "connectivity",
        "transfer",
        "morphology",
        "sparsity",
        "reactivation",
        "blocklinear",
        "implementation",
        "spatial",
        "synapse_types",
        "dynamics",
        "dendritic_spikes",
        "soma_feedback",
        "unified_ei",
        "recurrent_ei",
        "baseline_rnn",
        "heterogeneous_leak_ctrnn",
        "legendre_memory",
        "population_network",
        "vision",
    }
    assert expected_keys == set(d.keys())
    assert d["shared_projection_shunting"] is None
    assert d["routed_ei"] is None


# ---------------------------------------------------------------------------
# Custom values
# ---------------------------------------------------------------------------


def test_model_config_custom_task():
    cfg = ModelConfig(task="regression")
    assert cfg.task == "regression"


def test_architecture_custom_layers():
    cfg = ArchitectureConfig(
        excitatory_layer_sizes=[10, 20],
        excitatory_branch_factors=[2, 2],
    )
    assert cfg.excitatory_layer_sizes == [10, 20]
    assert cfg.excitatory_branch_factors == [2, 2]


def test_sparsity_deepst_dict():
    cfg = SparsityConfig()
    deepst = cfg.deepst
    assert deepst["excitatory_target_density"] == 0.1
    assert deepst["inhibitory_target_density"] == 0.1
    assert deepst["use_noise"] is True
    assert deepst["sigma"] == 0.05
    assert deepst["rewiring_mode"] == "global"


def test_sparsity_indexed_dict():
    cfg = SparsityConfig()
    indexed = cfg.indexed
    assert indexed["candidate_size"] is None
    assert indexed["selection"] == "standard"
    assert indexed["output_chunk_size"] == 2048
    assert indexed["rewire_frequency"] == 100
    assert indexed["rewire_quantile"] == 0.05
    assert indexed["rewire_until_step"] is None


@pytest.mark.parametrize(
    "learned, mode, expected",
    [
        (False, "global", None),
        (False, "per_class", None),
        (True, "fixed", None),
        (False, "fixed", "fixed"),
        (True, "global", "global"),
        (None, "global", "global"),
        (False, None, "fixed"),
        (True, None, None),
    ],
)
def test_retired_learned_output_scale_folds_into_the_mode(learned, mode, expected):
    from dendritic_modeling.config.model import fold_learned_output_scale

    if expected is None and mode is not None:
        with pytest.raises(ValueError, match=r"contradicts model\.output_scale_mode"):
            fold_learned_output_scale(learned, mode)
    else:
        assert fold_learned_output_scale(learned, mode) == expected


def test_output_scale_mode_is_validated_when_the_config_is_built():
    with pytest.raises(ValueError, match=r"output_scale_mode must be one of"):
        ModelConfig(output_scale_mode="learned")


@pytest.mark.parametrize(
    "mode, expected",
    [
        (None, "per_class"),
        ("Global", "global"),
        ("per_class", "per_class"),
        ("fixed", "fixed"),
    ],
)
def test_resolve_output_scale_mode(mode, expected):
    from dendritic_modeling.config.model import resolve_output_scale_mode

    assert resolve_output_scale_mode(mode) == expected


def test_loader_folds_the_retired_learned_output_scale(tmp_path, caplog):
    import logging

    import yaml

    from dendritic_modeling.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"learned_output_scale": False}}))
    with caplog.at_level(logging.WARNING):
        config = load_config(str(path))
    assert config.model.output_scale_mode == "fixed"
    assert "learned_output_scale was retired" in caplog.text


def test_alexnet_rejects_written_block_sizes(tmp_path):
    import yaml

    from dendritic_modeling.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {"dataset_name": "imagenet"},
                "model": {
                    "core": {
                        "type": "dendritic_alexnet",
                        "architecture": {"excitatory_layer_sizes": [64]},
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="sets each block's population sizes"):
        load_config(str(path))


def test_stochastic_defaults_agree_across_entry_points_and_never_sample_nothing():
    import pytest

    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
        StochasticTopKLinear,
    )
    from dendritic_modeling.networks.architectures.recurrent import PopulationConfig
    from dendritic_modeling.networks.architectures.recurrent.structured_population_kwargs import (
        _structured_population_sparsity_kwargs,
    )

    typed = SparsityConfig()
    population = PopulationConfig(n_neurons=2, branch_factors=[1])
    translated = _structured_population_sparsity_kwargs(
        sparsity={}, dense_to_sparse={}, indexed={}
    )
    layer = StochasticTopKLinear(4, 2, 2)
    for temperature, ultrafast in (
        (typed.temperature, typed.ultrafast),
        (population.topk_temperature, population.topk_ultrafast),
        (translated["topk_temperature"], translated["topk_ultrafast"]),
    ):
        assert (temperature, ultrafast) == (layer.temperature, layer.ultrafast)
    with pytest.raises(ValueError, match="selects exactly the standard Top-K"):
        StochasticTopKLinear(4, 2, 2, temperature=0.0, ultrafast=False)
