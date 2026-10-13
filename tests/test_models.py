import math

import pytest
import torch

from dendritic_modeling.models import Classifier, Regressor
from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture


def test_classifier_init():
    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 5}
    )
    decoder = Identity(5)
    model = Classifier(encoder, core, decoder)
    assert model is not None
    assert isinstance(model, Classifier)


def test_classifier_forward():
    model = Classifier(
        Identity(10),
        get_architecture(
            "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 5}
        ),
        Identity(5),
    )
    x = torch.randn(2, 10)
    output = model(x)
    assert output.shape == (2, 5)


def test_classifier_can_use_literal_identity_logits():
    model = Classifier(
        Identity(5),
        Identity(5),
        Identity(5),
        learned_output_scale=False,
    )
    x = torch.randn(3, 5)

    assert not hasattr(model, "log_output_scale")
    assert torch.equal(model(x), x)


def test_classifier_identity_scale_remains_backward_compatible():
    model = Classifier(Identity(5), Identity(5), Identity(5))

    assert hasattr(model, "log_output_scale")
    assert model.log_output_scale.requires_grad


def test_classifier_fixed_global_scale_preserves_identity_and_argmax():
    model = Classifier(
        Identity(5),
        Identity(5),
        Identity(5),
        learned_output_scale=False,
        fixed_output_scale=10.0,
    )
    x = torch.randn(3, 5)

    assert torch.equal(model(x), 10.0 * x)
    assert torch.equal(model(x).argmax(dim=-1), x.argmax(dim=-1))
    assert not hasattr(model, "log_output_scale")


def test_classifier_rejects_invalid_fixed_global_scale():
    with pytest.raises(ValueError, match="fixed_output_scale"):
        Classifier(
            Identity(5),
            Identity(5),
            Identity(5),
            learned_output_scale=False,
            fixed_output_scale=0.0,
        )


def test_classifier_single_global_temperature_preserves_argmax():
    model = Classifier(
        Identity(5),
        Identity(5),
        Identity(5),
        output_scale_mode="global",
    )
    x = torch.randn(3, 5)

    assert not hasattr(model, "log_output_scale")
    assert model.log_global_output_scale.shape == (1,)
    assert model.log_global_output_scale.requires_grad
    assert torch.equal(model(x), x)
    with torch.no_grad():
        model.log_global_output_scale.fill_(torch.log(torch.tensor(7.0)))
    assert torch.allclose(model(x), 7.0 * x)
    assert torch.equal(model(x).argmax(dim=-1), x.argmax(dim=-1))


def test_a_global_scale_saved_as_a_0d_tensor_still_loads():
    model = Classifier(
        Identity(5), Identity(5), Identity(5), output_scale_mode="global"
    )
    state = model.state_dict()
    state["log_global_output_scale"] = torch.tensor(math.log(3.0))  # pre-2026-10-10
    model.load_state_dict(state)
    x = torch.randn(2, 5)
    assert model.log_global_output_scale.shape == (1,)
    assert torch.allclose(model(x), 3.0 * x)


def _mlp_decoder():
    return get_architecture("MLP", {"input_dim": 5, "hidden_dims": [], "output_dim": 3})


@pytest.mark.parametrize("mode", ["per_class", "global"])
def test_classifier_rejects_a_learned_mode_its_decoder_cannot_scale(mode):
    with pytest.raises(ValueError, match="only for an identity decoder"):
        Classifier(Identity(5), Identity(5), _mlp_decoder(), output_scale_mode=mode)


def test_classifier_default_scale_learns_nothing_without_identity_decoder():
    model = Classifier(Identity(5), Identity(5), _mlp_decoder())
    assert not any("output_scale" in name for name, _ in model.named_parameters())
    with pytest.warns(DeprecationWarning, match="learned_output_scale"):
        model = Classifier(
            Identity(5), Identity(5), _mlp_decoder(), learned_output_scale=True
        )
    assert not any("output_scale" in name for name, _ in model.named_parameters())


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_classifier_rejects_contradictory_output_scale_settings():
    with pytest.raises(ValueError, match="contradicts"):
        Classifier(
            Identity(5),
            Identity(5),
            Identity(5),
            learned_output_scale=False,
            output_scale_mode="global",
        )


def test_regressor_init():
    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 3}
    )
    decoder = Identity(3)
    model = Regressor(encoder, core, decoder)
    assert model is not None
    assert isinstance(model, Regressor)


def test_regressor_forward():
    model = Regressor(
        Identity(10),
        get_architecture(
            "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 3}
        ),
        Identity(3),
    )
    x = torch.randn(2, 10)
    output = model(x)
    assert output.shape == (2, 3)


def test_regressor_predict():
    model = Regressor(
        Identity(10),
        get_architecture(
            "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 3}
        ),
        Identity(3),
    )
    x = torch.randn(2, 10)
    preds = model.predict(x)
    assert preds.shape == (2, 3)


def test_base_model_non_recurrent_path_unchanged():
    """Non-recurrent forward should be identical to encoder->core->decoder composition."""
    encoder = Identity(10)
    core = get_architecture(
        "MLP", {"input_dim": 10, "hidden_dims": [20], "output_dim": 5}
    )
    decoder = Identity(5)
    model = BaseModel(encoder, core, decoder)
    assert model._is_recurrent_core is False

    x = torch.randn(4, 10)
    with torch.no_grad():
        expected = decoder(core(encoder(x)))
        actual = model(x)
    assert torch.allclose(actual, expected)
