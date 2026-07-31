"""Tests for the soma-level DFA baseline."""

import torch
from torch import nn

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.soma_dfa import SomaDFATrainer


def _model() -> BaseModel:
    torch.manual_seed(3)
    return BaseModel(
        encoder_network=nn.Identity(),
        core_network=nn.Sequential(nn.Linear(4, 5), nn.Tanh()),
        decoder_network=nn.Sequential(nn.Linear(5, 3)),
    )


def _trainer(model: BaseModel) -> SomaDFATrainer:
    return SomaDFATrainer(
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        loss_function="cross_entropy",
        feedback_seed=11,
        device="cpu",
    )


def test_soma_dfa_keeps_decoder_gradient_exact_and_replaces_core_direction():
    model = _model()
    trainer = _trainer(model)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 2, 0, 1, 2])

    reference = _model()
    reference.load_state_dict(model.state_dict())
    reference_loss = nn.functional.cross_entropy(reference(x), y)
    reference_loss.backward()

    trainer.optimizer.zero_grad()
    base_loss, surrogate = trainer._dfa_surrogate(
        model=model,
        x_batch=x,
        y_batch=y,
    )
    surrogate.backward()

    torch.testing.assert_close(base_loss, reference_loss)
    torch.testing.assert_close(
        model.decoder_network[0].weight.grad,
        reference.decoder_network[0].weight.grad,
    )
    torch.testing.assert_close(
        model.decoder_network[0].bias.grad,
        reference.decoder_network[0].bias.grad,
    )
    assert not torch.allclose(
        model.core_network[0].weight.grad,
        reference.core_network[0].weight.grad,
    )
    assert trainer._feedback_matrix is not None
    assert tuple(trainer._feedback_matrix.shape) == (5, 3)


def test_soma_dfa_feedback_is_seed_reproducible():
    first_model = _model()
    second_model = _model()
    first = _trainer(first_model)
    second = _trainer(second_model)

    kwargs = {
        "soma_dim": 5,
        "output_dim": 3,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    torch.testing.assert_close(
        first._feedback_for(**kwargs), second._feedback_for(**kwargs)
    )
