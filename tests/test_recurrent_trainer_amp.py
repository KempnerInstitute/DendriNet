import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies.recurrent_trainer import RecurrentTrainer


class _DummyRecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.decay_calls = 0
        self.rewire_calls = 0

    def forward(self, x: torch.Tensor, seq_lengths=None) -> torch.Tensor:
        # [B, T, D] -> [B, C] many-to-one
        return self.fc(x.mean(dim=1))

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False):
        self.decay_calls += 1

    def apply_rewiring(self):
        self.rewire_calls += 1


class _FakeScaledLoss:
    def __init__(self, loss: torch.Tensor):
        self.loss = loss

    def backward(self):
        self.loss.backward()


class _FakeGradScaler:
    """Minimal GradScaler stub to force step-or-skip behavior in tests."""

    def __init__(self, should_step: bool):
        self.should_step = should_step
        self.step_optimizer = None
        self.unscale_optimizer = None

    def scale(self, loss: torch.Tensor):
        return _FakeScaledLoss(loss)

    def unscale_(self, optimizer):
        self.unscale_optimizer = optimizer

    def step(self, optimizer):
        self.step_optimizer = optimizer
        if self.should_step:
            optimizer.step()

    def update(self):
        return None


def _make_loader():
    x = torch.randn(8, 3, 4)  # [B, T, D]
    y = torch.randint(low=0, high=3, size=(8,), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def _make_trainer(model: _DummyRecurrentModel, should_step: bool) -> RecurrentTrainer:
    base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    wrapped_opt = CustomWeightDecayOptimizer(
        model=model,
        optimizer=base_opt,
        weight_decay=0.1,
        weight_boosting=False,
    )
    trainer = RecurrentTrainer(
        optimizer=wrapped_opt,
        task="classification",
        grad_clip_value=1.0,
        use_amp=False,
    )
    trainer.device = "cpu"
    trainer.recurrent_amp = True
    trainer.recurrent_scaler = _FakeGradScaler(should_step=should_step)
    return trainer


def test_amp_skip_does_not_apply_custom_decay_or_rewire():
    model = _DummyRecurrentModel()
    trainer = _make_trainer(model, should_step=False)
    loader = _make_loader()
    params_before = [p.detach().clone() for p in model.parameters()]

    trainer._epoch_train(model, loader)

    # AMP scaler must operate on wrapped optimizer for correct skip semantics.
    assert trainer.recurrent_scaler.step_optimizer is trainer.optimizer
    assert trainer.recurrent_scaler.unscale_optimizer is trainer.optimizer
    # When scaler skips the step, wrapper step() must not run.
    assert model.decay_calls == 0
    assert model.rewire_calls == 0
    for before, after in zip(params_before, model.parameters()):
        assert torch.allclose(before, after.detach())


def test_amp_step_applies_custom_decay_and_rewire_once():
    model = _DummyRecurrentModel()
    trainer = _make_trainer(model, should_step=True)
    loader = _make_loader()
    params_before = [p.detach().clone() for p in model.parameters()]

    trainer._epoch_train(model, loader)

    assert trainer.recurrent_scaler.step_optimizer is trainer.optimizer
    assert trainer.recurrent_scaler.unscale_optimizer is trainer.optimizer
    assert model.decay_calls == 1
    assert model.rewire_calls == 1
    assert any(
        not torch.allclose(before, after.detach())
        for before, after in zip(params_before, model.parameters())
    )
