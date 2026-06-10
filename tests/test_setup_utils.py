import pytest
import torch
from omegaconf import OmegaConf

from dendritic_modeling.config import DataConfig, DataProcessingConfig, load_config
from dendritic_modeling.config.model import (
    ArchitectureConfig,
    ConnectivityConfig,
    CoreConfig,
    DecoderConfig,
    DecoderParamsConfig,
    EncoderConfig,
    EncoderParamsConfig,
    ModelConfig,
    MorphologyConfig,
    ReactivationConfig,
    TransferFunctionConfig,
)
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils.config_utils import (
    canonicalize_model_core_flags,
    canonicalize_rnn_core_flags,
)
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _config_to_plain_dict,
    _get_dataset_specific_params,
    _is_recurrent_core_config,
    initialize_model,
)

# Inline model config used by tests that previously loaded from drafts/
_SHUNTING_PILOT_MODEL_CFG = OmegaConf.create(
    {
        "task": "classification",
        "encoder": {"type": "identity", "params": {"input_dim": None}},
        "core": {
            "type": "dendritic_shunting",
            "architecture": {
                "excitatory_layer_sizes": [64, 64],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [2, 2],
                "inhibitory_branch_factors": [1],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [18],
                "ei_synapses_per_branch_per_layer": [0],
                "ie_synapses_per_branch_per_layer": [6],
                "ii_synapses_per_branch_per_layer": [0],
            },
            "transfer": {
                "input_mode": 1,
                "independent_pathways": False,
                "output_activation": None,
                "allow_direct_inhibitory_stream": True,
            },
            "morphology": {
                "somatic_synapses": False,
                "weight_transform": "softplus",
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.5,
                "init_b": 0.5,
            },
        },
        "decoder": {
            "type": "MLP",
            "params": {"hidden_dims": [], "activation": "relu", "output_dim": 2},
        },
    }
)


def test_config_to_plain_dict_handles_dictconfig() -> None:
    section = OmegaConf.create({"flatten": False, "normalize": True})

    converted = _config_to_plain_dict(section)

    assert converted == {"flatten": False, "normalize": True}


def test_canonicalize_rnn_additive_alias_saved_config() -> None:
    payload = {
        "model": {
            "core": {
                "type": "rnn_dendritic_additive",
                "unified_ei": {
                    "layers": [
                        {
                            "excitatory": {
                                "n_neurons": 6,
                                "branch_factors": [2],
                                "use_shunting": True,
                                "reactivation_init_policy": "quantile",
                            }
                        }
                    ]
                },
            }
        }
    }

    with pytest.warns(UserWarning, match="rnn_dendritic_additive"):
        out = canonicalize_rnn_core_flags(payload)
    exc = out["model"]["core"]["unified_ei"]["layers"][0]["excitatory"]

    # rnn_dendritic_additive is now RAW additive (no implicit normalization) and
    # no longer injects a hidden fixed reactivation gate. The explicit
    # Legacy quantile alias is normalized to the canonical occupancy policy.
    assert exc["use_shunting"] is False
    assert exc["use_additive_normalization"] is False
    assert exc["reactivation_init_policy"] == "occupancy_quantile"
    assert "reactivation_init_m" not in exc
    assert "reactivation_init_b" not in exc


def test_canonicalize_rnn_normalized_additive_alias_sets_normalization() -> None:
    payload = {
        "model": {
            "core": {
                "type": "rnn_dendritic_normalized_additive",
                "unified_ei": {
                    "layers": [
                        {
                            "excitatory": {
                                "n_neurons": 6,
                                "branch_factors": [2],
                            }
                        }
                    ]
                },
            }
        }
    }

    out = canonicalize_rnn_core_flags(payload)
    exc = out["model"]["core"]["unified_ei"]["layers"][0]["excitatory"]

    assert exc["use_shunting"] is False
    assert exc["use_additive_normalization"] is True
    # No hidden fixed reactivation gate is injected by the alias.
    assert "reactivation_init_policy" not in exc


def test_canonicalize_preserves_point_mlp_reference_use_shunting() -> None:
    payload = {
        "model": {
            "core": {
                "type": "point_mlp",
                "morphology": {"use_shunting": True},
            }
        }
    }

    out = canonicalize_model_core_flags(payload)

    assert out["model"]["core"]["morphology"]["use_shunting"] is True


def test_canonicalize_structured_recurrent_additive_saved_config() -> None:
    payload = {
        "model": {
            "core": {
                "type": "dendritic_additive",
                "connectivity": {"rec_ee_synapses_per_branch": [1]},
                "morphology": {
                    "use_shunting": True,
                    "use_additive_normalization": False,
                },
                "reactivation": {
                    "init_policy": "analytical",
                    "init_m": 1.5,
                    "init_b": 0.5,
                },
            }
        }
    }

    with pytest.warns(UserWarning, match="dendritic_additive"):
        out = canonicalize_model_core_flags(payload)
    core = out["model"]["core"]

    # dendritic_additive is now RAW additive even with recurrent connectivity:
    # no forced normalization and no hidden fixed reactivation gate. The
    # explicitly configured reactivation is preserved.
    assert core["morphology"]["use_shunting"] is False
    assert core["morphology"]["use_additive_normalization"] is False
    assert core["reactivation"]["init_policy"] == "analytical"
    assert core["reactivation"]["init_m"] == 1.5
    assert core["reactivation"]["init_b"] == 0.5


def test_canonicalize_rnn_flat_alias_overrides_branch_factors() -> None:
    payload = {
        "model": {
            "core": {
                "type": "rnn_flat_shunting",
                "unified_ei": {
                    "layers": [
                        {
                            "excitatory": {
                                "n_neurons": 6,
                                "branch_factors": [2, 2],
                            }
                        }
                    ]
                },
            }
        }
    }

    with pytest.warns(UserWarning, match="rnn_flat_shunting"):
        out = canonicalize_rnn_core_flags(payload)
    exc = out["model"]["core"]["unified_ei"]["layers"][0]["excitatory"]

    assert exc["use_shunting"] is True
    assert exc["branch_factors"] == [1]


def test_get_dataset_specific_params_handles_custom_dict_sections() -> None:
    data_config = DataConfig(
        dataset_name="cue_integration",
        processing=DataProcessingConfig(flatten=True, normalize=False),
        dataset_params={
            "cue_integration": {
                "cue_dim": 8,
                "reliable_noise": 0.05,
                "conflict_fraction": 0.4,
            }
        },
    )

    params = _get_dataset_specific_params(data_config)

    assert params["cue_dim"] == 8
    assert params["reliable_noise"] == 0.05
    assert params["conflict_fraction"] == 0.4


def test_initialize_model_accepts_structured_core_config() -> None:
    cfg_model = OmegaConf.structured(OmegaConf.to_container(_SHUNTING_PILOT_MODEL_CFG))
    cfg_model.encoder.params.input_dim = 18

    model, _ = initialize_model(cfg_model)

    assert model is not None
    logits = model(torch.rand(8, 18))
    assert torch.isfinite(logits).all()


def test_initialize_model_accepts_default_model_config() -> None:
    cfg_model = ModelConfig()
    cfg_model.encoder.params.input_dim = 18

    model, _ = initialize_model(cfg_model)

    logits = model(torch.rand(4, 18))
    assert torch.isfinite(logits).all()


def test_initialize_model_accepts_typed_point_mlp_config() -> None:
    cfg_model = ModelConfig()
    cfg_model.encoder.params.input_dim = 18
    cfg_model.core.type = "point_mlp"
    cfg_model.core.architecture.excitatory_layer_sizes = [32]
    cfg_model.core.architecture.inhibitory_layer_sizes = []
    cfg_model.core.architecture.excitatory_branch_factors = [2]
    cfg_model.core.architecture.inhibitory_branch_factors = [1]
    cfg_model.core.connectivity.ee_synapses_per_branch_per_layer = [4]
    cfg_model.core.connectivity.ei_synapses_per_branch_per_layer = [0]
    cfg_model.core.connectivity.ie_synapses_per_branch_per_layer = [2]
    cfg_model.core.connectivity.ii_synapses_per_branch_per_layer = [0]

    model, _ = initialize_model(cfg_model)

    assert model is not None
    assert model.core_network.output_dim == 32


def test_standard_mnist_uses_develop_scale_shunting_init_by_default() -> None:
    cfg = load_config("configs/examples/standard_mnist.yaml")

    torch.manual_seed(0)
    model, _ = initialize_model(cfg.model)

    first_branch = next(
        module
        for module in model.core_network.modules()
        if isinstance(module, DendriticBranchLayer)
        and getattr(module, "branch_excitation", None) is not None
    )
    assert first_branch.adaptive_initialization is False
    assert first_branch.branch_excitation.weight().mean().item() == pytest.approx(
        0.693, rel=0.05
    )
    assert first_branch.reactivation.b.mean().item() == pytest.approx(0.478, rel=0.05)


def test_initialize_model_rejects_transformer_replacement_with_runner_hint() -> None:
    cfg_model = ModelConfig()
    cfg_model.transformer_replacement.enabled = True
    cfg_model.core.type = "dendritic_ffn"

    with pytest.raises(ValueError, match="train_transformer_replacement"):
        initialize_model(cfg_model)


def test_initialize_model_accepts_population_graph_recurrent_core() -> None:
    cfg_model = OmegaConf.create(
        {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": 5}},
            "core": {
                "type": "population_graph",
                "population_graph": {
                    "readout_population": "pyr",
                    "output_mode": "last",
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "rec_excitatory_synapses": 1,
                        "rec_inhibitory_synapses": 1,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                    },
                    "cell_types": [
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        },
                        {
                            "name": "pv",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [1],
                        },
                    ],
                },
            },
            "decoder": {
                "type": "MLP",
                "params": {"hidden_dims": [], "activation": "relu", "output_dim": 2},
            },
        }
    )

    model, _ = initialize_model(cfg_model)

    logits = model(torch.rand(3, 4, 5))
    assert torch.isfinite(logits).all()
    assert model._is_recurrent_core


def test_initialize_model_accepts_population_graph_feedforward_core() -> None:
    cfg_model = OmegaConf.create(
        {
            "task": "classification",
            "encoder": {"type": "identity", "params": {"input_dim": 5}},
            "core": {
                "type": "population_graph",
                "population_graph": {
                    "readout_population": "pyr",
                    "output_mode": "last",
                    "population_defaults": {
                        "ff_excitatory_synapses": 2,
                        "ff_inhibitory_synapses": 0,
                        "rec_excitatory_synapses": 0,
                        "rec_inhibitory_synapses": 0,
                        "somatic_synapses": True,
                        "reactivate": False,
                        "dbl_init_method": "default",
                    },
                    "cell_types": [
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        },
                    ],
                },
            },
            "decoder": {
                "type": "MLP",
                "params": {"hidden_dims": [], "activation": "relu", "output_dim": 2},
            },
        }
    )

    assert _is_recurrent_core_config(cfg_model.core) is False
    model, _ = initialize_model(cfg_model)

    logits = model(torch.rand(3, 5))
    assert torch.isfinite(logits).all()
    assert not model._is_recurrent_core


def _typed_soma_feedback_core() -> CoreConfig:
    return CoreConfig(
        type="dendritic_shunting",
        architecture=ArchitectureConfig(
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[],
            excitatory_branch_factors=[1],
        ),
        connectivity=ConnectivityConfig(
            ee_synapses_per_branch_per_layer=[2],
            ie_synapses_per_branch_per_layer=[0],
            ei_synapses_per_branch_per_layer=[0],
            ii_synapses_per_branch_per_layer=[0],
            rec_ee_synapses_per_branch=[0],
            rec_ie_synapses_per_branch=[0],
            rec_ei_synapses_per_branch=[0],
            rec_ii_synapses_per_branch=[0],
        ),
        transfer=TransferFunctionConfig(input_mode=0),
        morphology=MorphologyConfig(use_shunting=True, somatic_synapses=True),
        reactivation=ReactivationConfig(enabled=False),
        soma_feedback={"enabled": True, "levels": "distal"},
    )


def _typed_stateful_feature_core(feature_name: str, payload: dict) -> CoreConfig:
    core = CoreConfig(
        type="dendritic_shunting",
        architecture=ArchitectureConfig(
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[],
            excitatory_branch_factors=[1],
        ),
        connectivity=ConnectivityConfig(
            ee_synapses_per_branch_per_layer=[2],
            ie_synapses_per_branch_per_layer=[0],
            ei_synapses_per_branch_per_layer=[0],
            ii_synapses_per_branch_per_layer=[0],
            rec_ee_synapses_per_branch=[0],
            rec_ie_synapses_per_branch=[0],
            rec_ei_synapses_per_branch=[0],
            rec_ii_synapses_per_branch=[0],
        ),
        transfer=TransferFunctionConfig(input_mode=0),
        morphology=MorphologyConfig(use_shunting=True, somatic_synapses=True),
        reactivation=ReactivationConfig(enabled=False),
    )
    setattr(core, feature_name, payload)
    return core


@pytest.mark.parametrize(
    ("feature_name", "payload"),
    [
        ("dynamics", {"mode": "spike", "readout": "rate"}),
        ("dendritic_spikes", {"enabled": True, "levels": "distal"}),
        (
            "synapse_types",
            {
                "enabled": True,
                "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            },
        ),
    ],
)
def test_initialize_model_stateful_einet_features_use_recurrent_wrapper(
    feature_name: str, payload: dict
) -> None:
    cfg_model = ModelConfig(
        task="classification",
        encoder=EncoderConfig(
            type="identity",
            params=EncoderParamsConfig(input_dim=5),
        ),
        core=_typed_stateful_feature_core(feature_name, payload),
        decoder=DecoderConfig(
            type="MLP",
            params=DecoderParamsConfig(
                hidden_dims=[],
                activation="relu",
                output_dim=2,
            ),
        ),
    )

    assert _is_recurrent_core_config(cfg_model.core) is True
    model, _ = initialize_model(cfg_model)

    assert model._is_recurrent_core
    logits = model(torch.rand(3, 5))
    assert torch.isfinite(logits).all()


def test_factory_accepts_typed_core_config_for_structured_einet() -> None:
    core = _typed_soma_feedback_core()

    net = get_architecture(core.type, core, input_dim=5)

    assert net.is_recurrent is True
    assert net.output_dim == 4


def test_initialize_model_typed_soma_feedback_core_uses_recurrent_wrapper() -> None:
    cfg_model = ModelConfig(
        task="classification",
        encoder=EncoderConfig(
            type="identity",
            params=EncoderParamsConfig(input_dim=5),
        ),
        core=_typed_soma_feedback_core(),
        decoder=DecoderConfig(
            type="MLP",
            params=DecoderParamsConfig(
                hidden_dims=[],
                activation="relu",
                output_dim=2,
            ),
        ),
    )

    assert _is_recurrent_core_config(cfg_model.core) is True
    model, _ = initialize_model(cfg_model)

    logits = model(torch.rand(3, 4, 5))
    assert torch.isfinite(logits).all()
    assert model._is_recurrent_core


def test_population_graph_soma_feedback_core_requires_recurrent_wrapper() -> None:
    core = CoreConfig(
        type="population_graph",
        population_graph={
            "output_mode": "last",
            "population_defaults": {
                "ff_excitatory_synapses": 1,
                "soma_feedback_enabled": True,
                "soma_feedback_levels": "distal",
            },
            "cell_types": [
                {
                    "name": "pyr",
                    "polarity": "excitatory",
                    "n_neurons": 4,
                    "branch_factors": [1],
                }
            ],
        },
    )

    assert _is_recurrent_core_config(core) is True


def _temporal_population_feature_cases() -> list[dict]:
    return [
        {"dynamics_mode": "spike", "spike_readout": "rate"},
        {
            "synapse_types": {
                "enabled": True,
                "excitatory": [{"name": "AMPA", "fraction": 1.0}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0}],
            }
        },
        {"dendritic_spikes_enabled": True, "dendritic_spike_levels": "distal"},
        {"soma_feedback_enabled": True, "soma_feedback_levels": "distal"},
    ]


def _population_network_temporal_core(population_overrides: dict) -> CoreConfig:
    population_defaults = {
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 1,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "somatic_synapses": True,
        "reactivate": True,
        "weight_transform": "softplus",
        "use_shunting": True,
        **population_overrides,
    }
    return CoreConfig(
        type="population_network",
        population_network={
            "output_mode": "last",
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": population_defaults,
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                            "population": {"ff_inhibitory_synapses": 0},
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [2],
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
    )


@pytest.mark.parametrize("population_overrides", _temporal_population_feature_cases())
def test_population_network_temporal_features_use_recurrent_wrapper(
    population_overrides: dict,
) -> None:
    core = _population_network_temporal_core(population_overrides)
    cfg_model = ModelConfig(
        task="classification",
        encoder=EncoderConfig(
            type="identity",
            params=EncoderParamsConfig(input_dim=5),
        ),
        core=core,
        decoder=DecoderConfig(
            type="MLP",
            params=DecoderParamsConfig(
                hidden_dims=[],
                activation="relu",
                output_dim=2,
            ),
        ),
    )

    assert _is_recurrent_core_config(core) is True
    model, _ = initialize_model(cfg_model)

    assert model._is_recurrent_core
    assert model.core_network.is_recurrent is True
    logits = model(torch.rand(3, 4, 5))
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("population_overrides", _temporal_population_feature_cases())
def test_population_graph_temporal_features_require_recurrent_wrapper(
    population_overrides: dict,
) -> None:
    core = CoreConfig(
        type="population_graph",
        population_graph={
            "output_mode": "last",
            "population_defaults": {
                "ff_excitatory_synapses": 2,
                "rec_excitatory_synapses": 0,
                "rec_inhibitory_synapses": 0,
                "use_shunting": True,
                **population_overrides,
            },
            "cell_types": [
                {
                    "name": "pyr",
                    "polarity": "excitatory",
                    "n_neurons": 4,
                    "branch_factors": [1],
                }
            ],
        },
    )

    assert _is_recurrent_core_config(core) is True
    net = get_architecture(core.type, core, input_dim=5)
    assert net.is_recurrent is True


def test_pathway_router_fixed_and_learned_modes() -> None:
    fixed_router = PathwayRouter(
        input_dim=18,
        router_mode="fixed",
        shared_indices=[0, 1],
        pathway_groups=[list(range(2, 10)), list(range(10, 18))],
        pathway_dim=10,
    )
    learned_router = PathwayRouter(
        input_dim=18,
        router_mode="learned",
        shared_indices=[0, 1],
        pathway_groups=[list(range(2, 10)), list(range(10, 18))],
        pathway_dim=10,
        learned_router_temperature=0.5,
    )

    x = torch.rand(4, 18)
    fixed_y = fixed_router(x)
    learned_y = learned_router(x)

    assert fixed_y.shape == (4, 20)
    assert learned_y.shape == (4, 20)
    learned_assign = learned_router.get_assignment_matrix()
    output_roles = learned_router.get_output_role_matrix()
    assert learned_assign.shape == (18, 2)
    assert output_roles.shape == (20, 2)
    assert torch.all(output_roles[:10, 0] == 1.0)
    assert torch.all(output_roles[:10, 1] == 0.0)
    assert torch.all(output_roles[10:, 1] == 1.0)
    assert torch.allclose(learned_assign[[0, 1]], torch.ones(2, 2))


def test_initialize_model_accepts_pathway_router_encoder() -> None:
    cfg_model = OmegaConf.structured(OmegaConf.to_container(_SHUNTING_PILOT_MODEL_CFG))
    cfg_model.encoder.type = "pathway_router"
    cfg_model.encoder.params.input_dim = 18
    cfg_model.encoder.params.router_mode = "fixed"
    cfg_model.encoder.params.shared_indices = [0, 1]
    cfg_model.encoder.params.pathway_groups = [list(range(2, 10)), list(range(10, 18))]
    cfg_model.encoder.params.pathway_dim = 10
    cfg_model.core.connectivity.ee_synapses_per_branch_per_layer = [10]

    model, _ = initialize_model(cfg_model)

    logits = model(torch.rand(8, 18))
    assert torch.isfinite(logits).all()
