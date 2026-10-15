"""Positive afferents also cover the public unified recurrent builder."""

import copy

import pytest
import torch

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.config.biological_adoption import prepare_biological_config
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.utils.nonnegative_linear import NonnegativeLinear


def recipe(recurrent):
    return {
        "config_version": 3,
        "model": {
            "core": {
                "type": "ei_unified",
                "unified_ei": {
                    "input_projection_dims": [5],
                    "layers": [
                        {
                            "recurrent": recurrent,
                            "inhibitory": None,
                            "excitatory": {
                                "n_neurons": 4,
                                "branch_factors": [2],
                                "ff_excitatory_synapses": 2,
                                "ff_inhibitory_synapses": 0,
                                "rec_excitatory_synapses": 1 if recurrent else 0,
                                "rec_inhibitory_synapses": 0,
                                "use_shunting": True,
                                "weight_transform": "softplus",
                                "reactivation_type": "param_tanh",
                            },
                        }
                    ],
                },
            }
        },
    }


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_typed_unified_signed_split_positive_projection_gradients_and_roundtrip(
    recurrent, dtype
):
    data = prepare_biological_config(
        recipe(recurrent),
        input_activation="signed_split",
        positive_input_projection=True,
    )
    config = config_from_mapping(data)
    model = get_architecture(config.model.core.type, config.model.core, input_dim=3).to(
        dtype
    )
    assert isinstance(model.input_projection[0], NonnegativeLinear)
    x = torch.tensor([-0.3, 0.0, 0.7], dtype=dtype)
    x = x.expand(2, 3, 3).clone() if recurrent else x.expand(2, 3).clone()
    x.requires_grad_()
    y = model(x)
    assert torch.isfinite(y).all() and (y >= 0).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.input_projection[0].raw_weight.grad).all()
    restored = get_architecture(
        config.model.core.type, config.model.core, input_dim=3
    ).to(dtype)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(model(x), restored(x))


@pytest.mark.parametrize("recurrent", [False, True])
def test_explicit_identity_keeps_unified_defaults_exact(recurrent):
    data = recipe(recurrent)["model"]["core"]
    data["biological_neuron"] = False
    old = copy.deepcopy(data)
    data["unified_ei"].update(
        input_transform="identity", input_projection_weight_transform="identity"
    )
    models = []
    for config in (old, data):
        torch.manual_seed(71)
        models.append(get_architecture("ei_unified", config, input_dim=3))
    for k, v in models[0].state_dict().items():
        assert torch.equal(v, models[1].state_dict()[k])
    x = torch.rand(2, 3, 3) if recurrent else torch.rand(2, 3)
    assert torch.equal(models[0](x), models[1](x))


def test_unified_local_rule_rejects_untrained_positive_projection():
    from test_trainer_configuration import IN_DIM, _data, _make_trainer, _wrap

    payload = prepare_biological_config(
        recipe(False), input_activation="nonnegative", positive_input_projection=True
    )
    config = config_from_mapping(payload)
    model = _wrap(
        get_architecture(config.model.core.type, config.model.core, input_dim=IN_DIM)
    )
    trainer, _ = _make_trainer("local_ca", model, local_rule_config={})
    with pytest.raises(ValueError, match="do not train the positive input projection"):
        trainer.train(model, _data(), _data(seed=1))


def test_old_pickled_unified_shape_keeps_identity_input():
    data = recipe(False)["model"]["core"]
    data["biological_neuron"] = False
    model = get_architecture("ei_unified", data, input_dim=3)
    x = torch.rand(2, 3)
    expected = model(x)
    del model.input_adapter
    del model.input_transform
    assert torch.equal(model(x), expected)
