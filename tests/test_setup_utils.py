import torch
from omegaconf import OmegaConf

from dendritic_modeling.config import DataConfig, DataProcessingConfig
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _config_to_plain_dict,
    _get_dataset_specific_params,
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
