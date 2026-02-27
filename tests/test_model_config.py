"""Tests for config/model.py dataclasses."""

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
    TransferFunctionConfig,
)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.task == "classification"
    assert isinstance(cfg.encoder, EncoderConfig)
    assert isinstance(cfg.core, CoreConfig)
    assert isinstance(cfg.decoder, DecoderConfig)


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


def test_morphology_config_defaults():
    cfg = MorphologyConfig()
    assert cfg.use_shunting is True
    assert cfg.weight_transform == "softplus"
    assert cfg.dbl_init_method == "analytical_expectation"
    assert cfg.somatic_synapses is False


def test_sparsity_config_defaults():
    cfg = SparsityConfig()
    assert cfg.init_method == "xavier_normal"
    assert cfg.noise_level == 0.0
    assert cfg.type == "standard"
    assert isinstance(cfg.deepst, dict)
    assert "excitatory_target_density" in cfg.deepst


def test_reactivation_config_defaults():
    cfg = ReactivationConfig()
    assert cfg.enabled is True
    assert cfg.type == "param_tanh"
    assert cfg.init_m == 1.5
    assert cfg.init_b == 0.5


def test_transfer_function_config_defaults():
    cfg = TransferFunctionConfig()
    assert cfg.input_mode == 1
    assert cfg.independent_pathways is False


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
    assert cfg.type == "EINet"
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
        "architecture",
        "connectivity",
        "transfer",
        "morphology",
        "sparsity",
        "reactivation",
        "blocklinear",
        "implementation",
        "recurrent",
        "unified_ei",
    }
    assert expected_keys == set(d.keys())


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
