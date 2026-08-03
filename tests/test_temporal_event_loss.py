import pytest
import torch

from dendritic_modeling.training.loss.functions import get_loss_function


def test_temporal_event_cross_entropy_accepts_indices_and_soft_targets():
    loss = get_loss_function("temporal_event_ce")
    logits = torch.tensor(
        [
            [[0.0], [1.0], [3.0], [0.5]],
            [[2.0], [0.0], [0.0], [0.0]],
        ]
    )
    indices = torch.tensor([2, 0])
    hard = loss.from_predictions(logits, indices)

    soft = torch.zeros(2, 4)
    soft[0, 2] = 1.0
    soft[1, 0] = 1.0
    smooth = loss.from_predictions(logits, soft)
    assert hard == pytest.approx(float(smooth))


def test_temporal_event_cross_entropy_rejects_invalid_shapes_and_mass():
    loss = get_loss_function("temporal_event_ce")
    with pytest.raises(ValueError, match="predictions"):
        loss.from_predictions(torch.zeros(2, 4, 2), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="positive mass"):
        loss.from_predictions(torch.zeros(2, 4, 1), torch.zeros(2, 4))


@pytest.mark.parametrize(
    "targets",
    (
        torch.tensor([2.0, 0.0]),
        torch.tensor([2.5, 0.0]),
        torch.tensor([True, False]),
        torch.tensor([2.0 + 0.0j, 0.0 + 0.0j]),
    ),
)
def test_temporal_event_cross_entropy_rejects_non_integer_hard_targets(targets):
    loss = get_loss_function("temporal_event_ce")
    with pytest.raises(TypeError, match="integer dtype"):
        loss.from_predictions(torch.zeros(2, 4), targets)


def test_temporal_event_cross_entropy_accepts_non_long_integer_targets():
    loss = get_loss_function("temporal_event_ce")
    logits = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.0, 0.0]])
    int32_targets = torch.tensor([2, 0], dtype=torch.int32)

    assert torch.equal(
        loss.from_predictions(logits, int32_targets),
        loss.from_predictions(logits, int32_targets.to(torch.long)),
    )


def test_temporal_event_cross_entropy_has_uniform_baseline_and_gradients():
    loss = get_loss_function("temporal_event_ce")
    logits = torch.zeros(3, 11, 1, requires_grad=True)
    targets = torch.zeros(3, 11)
    targets[0, 2] = 1.0
    targets[1, 5] = 1.0
    targets[2, 8] = 1.0

    value = loss.from_predictions(logits, targets)
    assert float(value.detach()) == pytest.approx(torch.log(torch.tensor(11.0)).item())
    value.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0
