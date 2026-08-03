import math
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.config.model import (
    CoreConfig,
    DecoderConfig,
    DecoderParamsConfig,
    EncoderConfig,
    EncoderParamsConfig,
    ModelConfig,
)
from dendritic_modeling.datasets.sequence.multiscale_working_memory import (
    SpeededDistractorDMSDataset,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.loss import (
    MaskedDualTaskTemporalLoss,
    get_available_loss_functions,
    get_loss_function,
)
from dendritic_modeling.training.strategies.recurrent_trainer import RecurrentTrainer


def _packed_targets() -> torch.Tensor:
    targets = torch.zeros(2, 4, 4)
    targets[0, 0, 0:2] = torch.tensor([1.0, 1.0])
    targets[0, 1, 0:2] = torch.tensor([0.0, 1.0])
    targets[1, 0, 0:2] = torch.tensor([1.0, 1.0])
    targets[1, 1, 0:2] = torch.tensor([0.0, 1.0])
    targets[0, 2, 2:4] = torch.tensor([1.0, 1.0])
    targets[1, 2, 2:4] = torch.tensor([0.0, 1.0])
    return targets


def _binary_logit_loss(logit: float, label: float) -> float:
    return math.log1p(math.exp(logit)) - label * logit


def test_masked_dual_task_loss_matches_hand_calculation_and_gradients():
    logits = torch.tensor(
        [
            [[0.0, 4.0], [math.log(3.0), -2.0], [9.0, -math.log(2.0)], [7.0, 8.0]],
            [
                [-math.log(3.0), -4.0],
                [math.log(2.0), 3.0],
                [6.0, math.log(4.0)],
                [5.0, 2.0],
            ],
        ],
        requires_grad=True,
    )
    targets = _packed_targets()
    loss = MaskedDualTaskTemporalLoss().from_predictions(logits, targets)

    fast_positive = 0.5 * (
        _binary_logit_loss(0.0, 1.0) + _binary_logit_loss(-math.log(3.0), 1.0)
    )
    fast_negative = 0.5 * (
        _binary_logit_loss(math.log(3.0), 0.0) + _binary_logit_loss(math.log(2.0), 0.0)
    )
    memory_positive = _binary_logit_loss(-math.log(2.0), 1.0)
    memory_negative = _binary_logit_loss(math.log(4.0), 0.0)
    expected = 0.25 * (
        fast_positive + fast_negative + memory_positive + memory_negative
    )
    assert loss.item() == pytest.approx(expected)

    loss.backward()
    expected_gradient = torch.zeros_like(logits)
    expected_gradient[0, 0, 0] = 0.125 * (torch.sigmoid(logits[0, 0, 0]) - 1)
    expected_gradient[1, 0, 0] = 0.125 * (torch.sigmoid(logits[1, 0, 0]) - 1)
    expected_gradient[0, 1, 0] = 0.125 * torch.sigmoid(logits[0, 1, 0])
    expected_gradient[1, 1, 0] = 0.125 * torch.sigmoid(logits[1, 1, 0])
    expected_gradient[0, 2, 1] = 0.25 * (torch.sigmoid(logits[0, 2, 1]) - 1)
    expected_gradient[1, 2, 1] = 0.25 * torch.sigmoid(logits[1, 2, 1])
    assert torch.allclose(logits.grad, expected_gradient)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda _pred, target: target.__setitem__((0, 0, 1), 0.5), "mask"),
        (lambda _pred, target: target.__setitem__((0, 0, 0), float("nan")), "finite"),
        (
            lambda _pred, target: target.__setitem__((slice(None), slice(None), 0), 0),
            "fast-positive",
        ),
        (lambda pred, _target: pred.__setitem__((0, 0, 0), float("inf")), "finite"),
    ],
)
def test_masked_dual_task_loss_fails_closed(mutation, message):
    predictions = torch.zeros(2, 4, 2)
    targets = _packed_targets()
    mutation(predictions, targets)

    with pytest.raises(ValueError, match=message):
        MaskedDualTaskTemporalLoss().from_predictions(predictions, targets)


def test_masked_dual_task_loss_rejects_shape_and_dtype_changes():
    loss = MaskedDualTaskTemporalLoss()
    targets = _packed_targets()

    with pytest.raises(ValueError, match="shape"):
        loss.from_predictions(torch.zeros(2, 4), targets)
    with pytest.raises(ValueError, match="shape"):
        loss.from_predictions(torch.zeros(2, 4, 2), targets[..., :3])
    with pytest.raises(TypeError, match="floating"):
        loss.from_predictions(torch.zeros(2, 4, 2), targets.long())


def test_masked_dual_task_loss_is_registered():
    assert "masked_dual_task_temporal" in get_available_loss_functions()
    assert isinstance(
        get_loss_function("masked_dual_task_temporal"), MaskedDualTaskTemporalLoss
    )
    assert isinstance(get_loss_function("masked_dual_task"), MaskedDualTaskTemporalLoss)


def test_validation_statistics_are_additive_with_locally_empty_strata():
    loss_function = MaskedDualTaskTemporalLoss()
    predictions = torch.zeros(2, 4, 2)
    targets = _packed_targets()

    positive_sums, positive_counts = loss_function.validation_sums_and_counts(
        predictions[:1],
        targets[:1],
    )
    negative_sums, negative_counts = loss_function.validation_sums_and_counts(
        predictions[1:],
        targets[1:],
    )
    assert positive_counts.tolist() == [1, 1, 1, 0]
    assert negative_counts.tolist() == [1, 1, 0, 1]
    with pytest.raises(ValueError, match="global memory-negative"):
        loss_function.reduce_validation_sums_and_counts(
            positive_sums,
            positive_counts,
        )

    accumulated = loss_function.reduce_validation_sums_and_counts(
        positive_sums + negative_sums,
        positive_counts + negative_counts,
    )
    direct = loss_function.from_predictions(predictions, targets)
    torch.testing.assert_close(accumulated, direct.to(torch.float64))


def test_distributed_training_reducer_scales_local_differentiable_sums():
    loss_function = MaskedDualTaskTemporalLoss()
    local_sums = torch.tensor([2.0, 0.0, 6.0, 0.0], requires_grad=True)
    global_counts = torch.tensor([4, 8, 3, 9])

    actual = loss_function.reduce_distributed_training_sums_and_counts(
        local_sums,
        global_counts,
        world_size=2,
    )
    expected = 2.0 * torch.mean(local_sums / global_counts)
    torch.testing.assert_close(actual, expected)

    actual.backward()
    torch.testing.assert_close(
        local_sums.grad,
        2.0 / (4.0 * global_counts.to(torch.float32)),
    )


def test_distributed_training_reducer_fails_on_globally_empty_stratum():
    loss_function = MaskedDualTaskTemporalLoss()
    with pytest.raises(ValueError, match="global memory-positive"):
        loss_function.reduce_distributed_training_sums_and_counts(
            torch.tensor([1.0, 2.0, 0.0, 3.0], requires_grad=True),
            torch.tensor([2, 2, 0, 2]),
            world_size=2,
        )


class _SequenceModel(nn.Module):
    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        self.predictions = nn.Parameter(predictions)
        self.received_lengths = None

    def forward(self, inputs, *, seq_lengths=None):
        self.received_lengths = seq_lengths
        return self.predictions


class _IdentitySequenceModel(nn.Module):
    def forward(self, inputs, *, seq_lengths=None):
        del seq_lengths
        return inputs


class _OffsetSequenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs, *, seq_lengths=None):
        del seq_lengths
        return inputs + self.offset


class _DistributedSequenceLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [[0.30, -0.20, 0.10], [-0.15, 0.25, 0.35]],
                    dtype=torch.float64,
                )
            )
            self.linear.bias.copy_(torch.tensor([0.05, -0.10], dtype=torch.float64))

    def forward(self, inputs, *, seq_lengths=None):
        del seq_lengths
        return self.linear(inputs)


def _imbalanced_ddp_rank_batch(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rank == 0:
        inputs = torch.tensor(
            [
                [[0.2, -0.5, 0.7], [1.0, 0.3, -0.2], [-0.4, 0.8, 0.1]],
                [[0.6, -0.1, -0.9], [-0.7, 0.4, 0.5], [0.9, -0.6, 0.2]],
            ],
            dtype=torch.float64,
        )
        fast_labels = [1, 1, 1, 0, 0, 0]
        fast_masks = [1, 1, 1, 1, 0, 0]
        memory_labels = [0, 0, 0, 0, 0, 0]
        memory_masks = [1, 1, 0, 0, 0, 0]
    elif rank == 1:
        inputs = torch.tensor(
            [
                [[-0.8, 0.2, 0.4], [0.5, 0.9, -0.3], [0.1, -0.7, 0.6]],
                [[-0.2, -0.4, 1.1], [0.7, 0.5, -0.8], [-0.6, 0.3, -0.1]],
            ],
            dtype=torch.float64,
        )
        fast_labels = [0, 0, 0, 0, 0, 0]
        fast_masks = [1, 1, 1, 1, 0, 0]
        memory_labels = [1, 1, 1, 1, 1, 0]
        memory_masks = [1, 1, 1, 1, 1, 1]
    else:
        raise ValueError(f"unexpected rank {rank}")

    targets = torch.zeros(2, 3, 4, dtype=torch.float64)
    targets[..., 0] = torch.tensor(fast_labels, dtype=torch.float64).reshape(2, 3)
    targets[..., 1] = torch.tensor(fast_masks, dtype=torch.float64).reshape(2, 3)
    targets[..., 2] = torch.tensor(memory_labels, dtype=torch.float64).reshape(2, 3)
    targets[..., 3] = torch.tensor(memory_masks, dtype=torch.float64).reshape(2, 3)
    return inputs, targets


def _flatten_linear_state(model: _DistributedSequenceLinear) -> torch.Tensor:
    return torch.cat(
        (
            model.linear.weight.detach().reshape(-1),
            model.linear.bias.detach().reshape(-1),
        )
    )


def _ddp_balanced_loss_step_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    result_path: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        model = _DistributedSequenceLinear()
        ddp_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.075)
        trainer = object.__new__(RecurrentTrainer)
        trainer.loss_function = MaskedDualTaskTemporalLoss()
        trainer.task_loss = nn.MSELoss()

        inputs, targets = _imbalanced_ddp_rank_batch(rank)
        optimizer.zero_grad()
        loss = trainer._compute_sequence_loss(ddp_model, inputs, targets)
        gathered_losses = [torch.empty_like(loss.detach()) for _ in range(world_size)]
        dist.all_gather(gathered_losses, loss.detach())
        loss.backward()
        local_gradient = torch.cat(
            (
                model.linear.weight.grad.detach().reshape(-1),
                model.linear.bias.grad.detach().reshape(-1),
            )
        )
        gathered_gradients = [
            torch.empty_like(local_gradient) for _ in range(world_size)
        ]
        dist.all_gather(gathered_gradients, local_gradient)

        optimizer.step()
        local_state = _flatten_linear_state(model)
        gathered_states = [torch.empty_like(local_state) for _ in range(world_size)]
        dist.all_gather(gathered_states, local_state)
        if rank == 0:
            torch.save(
                {
                    "losses": torch.stack(gathered_losses),
                    "gradients": torch.stack(gathered_gradients),
                    "states": torch.stack(gathered_states),
                },
                result_path,
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="CPU gloo distributed backend is unavailable",
)
def test_ddp_step_matches_global_equal_stratum_reference(tmp_path: Path):
    world_size = 2
    result_path = tmp_path / "ddp_result.pt"
    mp.spawn(
        _ddp_balanced_loss_step_worker,
        args=(world_size, str(tmp_path / "gloo_rendezvous"), str(result_path)),
        nprocs=world_size,
        join=True,
    )

    reference_model = _DistributedSequenceLinear()
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.075)
    rank_batches = [_imbalanced_ddp_rank_batch(rank) for rank in range(world_size)]
    global_inputs = torch.cat([batch[0] for batch in rank_batches])
    global_targets = torch.cat([batch[1] for batch in rank_batches])
    reference_optimizer.zero_grad()
    reference_loss = MaskedDualTaskTemporalLoss().from_predictions(
        reference_model(global_inputs),
        global_targets,
    )
    reference_loss.backward()
    reference_gradient = torch.cat(
        (
            reference_model.linear.weight.grad.detach().reshape(-1),
            reference_model.linear.bias.grad.detach().reshape(-1),
        )
    )
    reference_optimizer.step()
    reference_state = _flatten_linear_state(reference_model)

    distributed_result = torch.load(result_path, weights_only=True)
    for loss in distributed_result["losses"]:
        torch.testing.assert_close(
            loss, reference_loss.detach(), rtol=1e-10, atol=1e-12
        )
    for gradient in distributed_result["gradients"]:
        torch.testing.assert_close(gradient, reference_gradient, rtol=1e-10, atol=1e-12)
    for state in distributed_result["states"]:
        torch.testing.assert_close(state, reference_state, rtol=1e-10, atol=1e-12)


def test_recurrent_trainer_dispatches_generic_sequence_prediction_loss():
    trainer = object.__new__(RecurrentTrainer)
    trainer.loss_function = MaskedDualTaskTemporalLoss()
    trainer.task_loss = nn.Identity()
    predictions = torch.zeros(2, 4, 2)
    model = _SequenceModel(predictions)
    targets = _packed_targets()
    lengths = torch.tensor([4, 3])

    actual = RecurrentTrainer._compute_sequence_loss(
        trainer,
        model,
        torch.zeros(2, 4, 1),
        targets,
        lengths,
    )
    expected = trainer.loss_function.from_predictions(predictions, targets)

    assert torch.equal(model.received_lengths, lengths)
    assert torch.equal(actual, expected)


def test_validation_statistics_capability_does_not_select_ddp_training(monkeypatch):
    trainer = object.__new__(RecurrentTrainer)
    trainer.loss_function = MaskedDualTaskTemporalLoss()
    trainer.loss_function.exact_distributed_training_statistics = False
    trainer.task_loss = nn.Identity()
    predictions = torch.zeros(2, 4, 2)
    targets = _packed_targets()

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        dist,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail(
            "validation-only statistics must not select the DDP training path"
        ),
    )
    actual = trainer._compute_sequence_loss(
        _SequenceModel(predictions),
        torch.zeros(2, 4, 1),
        targets,
    )

    expected = trainer.loss_function.from_predictions(predictions, targets)
    torch.testing.assert_close(actual, expected)


def test_recurrent_ddp_epoch_reports_rank_averaged_training_loss(monkeypatch):
    model = _OffsetSequenceModel()
    trainer = object.__new__(RecurrentTrainer)
    trainer.loss_function = nn.Identity()
    trainer.task_loss = nn.MSELoss()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer.recurrent_amp = False
    trainer.grad_clip = 0.0
    trainer.scheduler = None
    trainer.device = "cpu"
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.0]]), torch.tensor([[2.0]])),
        batch_size=1,
    )
    calls = []

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    def _add_remote_epoch_statistics(statistics, op=None):
        calls.append(op)
        statistics.add_(torch.tensor([9.0, 1.0], dtype=statistics.dtype))

    monkeypatch.setattr(dist, "all_reduce", _add_remote_epoch_statistics)
    total, base, regularization = trainer._epoch_train(model, loader)

    assert calls == [dist.ReduceOp.SUM]
    assert total == pytest.approx(6.5)
    assert base == pytest.approx(6.5)
    assert regularization == 0.0


def _exact_validation_trainer() -> RecurrentTrainer:
    trainer = object.__new__(RecurrentTrainer)
    trainer.loss_function = MaskedDualTaskTemporalLoss()
    trainer.task_loss = nn.MSELoss()
    trainer.device = "cpu"
    return trainer


def test_recurrent_validation_is_exact_and_invariant_to_batch_boundaries():
    predictions = torch.linspace(-2.0, 2.0, 48).reshape(6, 4, 2)
    targets = _packed_targets().repeat(3, 1, 1)
    lengths = torch.full((6,), 4, dtype=torch.long)
    dataset = TensorDataset(predictions, targets, lengths)
    trainer = _exact_validation_trainer()
    model = _IdentitySequenceModel()

    expected = trainer.loss_function.from_predictions(predictions, targets).item()
    batch_one = trainer._non_distributed_validation_loss(
        model,
        DataLoader(dataset, batch_size=1, shuffle=False),
    )
    uneven_batches = trainer._non_distributed_validation_loss(
        model,
        DataLoader(dataset, batch_size=4, shuffle=False),
    )

    assert batch_one == pytest.approx(expected)
    assert uneven_batches == pytest.approx(expected)
    assert batch_one == pytest.approx(uneven_batches)


def _single_class_rank_target(*, positive: bool) -> torch.Tensor:
    target = torch.zeros(1, 2, 4)
    target[0, 0, 0] = float(positive)
    target[0, 0, 1] = 1.0
    target[0, 1, 2] = float(positive)
    target[0, 1, 3] = 1.0
    return target


def test_recurrent_ddp_validation_reduces_before_global_nonempty_check(monkeypatch):
    trainer = _exact_validation_trainer()
    model = _IdentitySequenceModel()
    local_predictions = torch.tensor([[[1.5, -0.5], [0.2, 0.8]]])
    remote_predictions = torch.tensor([[[-1.0, 0.3], [0.7, -1.2]]])
    local_targets = _single_class_rank_target(positive=True)
    remote_targets = _single_class_rank_target(positive=False)
    local_dataset = TensorDataset(
        local_predictions,
        local_targets,
        torch.tensor([2]),
    )

    remote_sums, remote_counts = trainer.loss_function.validation_sums_and_counts(
        remote_predictions,
        remote_targets,
    )
    remote_statistics = torch.cat((remote_sums, remote_counts.to(dtype=torch.float64)))
    calls = []

    def _fake_all_reduce(statistics, op=None):
        calls.append(op)
        statistics.add_(remote_statistics)

    monkeypatch.setattr(torch.distributed, "all_reduce", _fake_all_reduce)
    actual = trainer._epoch_valid_exact_distributed(
        model,
        DataLoader(local_dataset, batch_size=1, shuffle=False),
    )

    local_sums, local_counts = trainer.loss_function.validation_sums_and_counts(
        local_predictions,
        local_targets,
    )
    expected = trainer.loss_function.reduce_validation_sums_and_counts(
        local_sums + remote_sums,
        local_counts + remote_counts,
    )
    assert calls == [torch.distributed.ReduceOp.SUM]
    assert actual == pytest.approx(expected.item())


def test_real_recurrent_config_emits_two_independent_binary_logits():
    dataset = SpeededDistractorDMSDataset(
        n_pairs=4,
        delay_support=(96,),
        n_distractors=4,
        lure_fraction=0.25,
        sequence_duration=140,
        positive_input_encoding=True,
        seed=811,
        split_name="integration",
    )
    config = ModelConfig(
        task="regression",
        encoder=EncoderConfig(
            type="identity",
            params=EncoderParamsConfig(input_dim=dataset.input_dim),
        ),
        core=CoreConfig(
            type="gru",
            baseline_rnn={
                "hidden_dim": 8,
                "num_layers": 1,
                "output_mode": "all",
            },
        ),
        decoder=DecoderConfig(
            type="MLP",
            params=DecoderParamsConfig(
                hidden_dims=[],
                output_dim=2,
            ),
        ),
    )
    model, _encoder = initialize_model(config)
    logits = model(dataset.inputs, seq_lengths=dataset.seq_lengths)

    assert logits.shape == (dataset.n_samples, dataset.sequence_duration, 2)
    loss = MaskedDualTaskTemporalLoss().from_predictions(logits, dataset.targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    independent_logits = torch.tensor([[[2.0, 2.0], [-2.0, -2.0]]])
    independent_decisions = torch.sigmoid(independent_logits) >= 0.5
    assert independent_decisions[0, 0].tolist() == [True, True]
    assert independent_decisions[0, 1].tolist() == [False, False]
