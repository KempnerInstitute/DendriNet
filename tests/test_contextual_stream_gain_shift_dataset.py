import torch

from dendritic_modeling.datasets.synthetic_datasets import (
    create_contextual_stream_gain_shift_dataset,
)


def _irrelevant_stream_deviation(inputs: torch.Tensor, stream_dim: int) -> torch.Tensor:
    stream_a = inputs[:, :stream_dim]
    stream_b = inputs[:, stream_dim : 2 * stream_dim]
    context = inputs[:, 2 * stream_dim :].argmax(dim=1)
    irrelevant = torch.where(
        (context == 0).unsqueeze(1),
        stream_b,
        stream_a,
    )
    return (irrelevant - 0.5).abs().mean()


def _contextual_class_means(
    inputs: torch.Tensor, labels: torch.Tensor, stream_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stream_a = inputs[:, :stream_dim]
    stream_b = inputs[:, stream_dim : 2 * stream_dim]
    context = inputs[:, 2 * stream_dim :].argmax(dim=1)

    a0 = stream_a[(context == 0) & (labels == 0)].mean(dim=0)
    a1 = stream_a[(context == 0) & (labels == 1)].mean(dim=0)
    b0 = stream_b[(context == 1) & (labels == 0)].mean(dim=0)
    b1 = stream_b[(context == 1) & (labels == 1)].mean(dim=0)
    return a0, a1, b0, b1


def test_contextual_stream_gain_shift_shapes_and_balance() -> None:
    train_ds, valid_ds, test_ds = create_contextual_stream_gain_shift_dataset(
        n_samples=400,
        stream_dim=8,
        seed=0,
    )

    for ds in (train_ds, valid_ds, test_ds):
        x, y = ds.tensors
        assert x.shape[1] == 18  # A + B + 2-d context
        assert set(y.tolist()) == {0, 1}
        counts = torch.bincount(y, minlength=2)
        assert counts[0] == counts[1]


def test_contextual_stream_gain_shift_ood_increases_irrelevant_stream_contrast() -> (
    None
):
    stream_dim = 8
    train_ds, _, test_ds = create_contextual_stream_gain_shift_dataset(
        n_samples=600,
        stream_dim=stream_dim,
        test_gain_irrelevant=5.0,
        ood_mode="irrelevant",
        seed=1,
    )

    train_x, _ = train_ds.tensors
    test_x, _ = test_ds.tensors

    train_dev = _irrelevant_stream_deviation(train_x, stream_dim)
    test_dev = _irrelevant_stream_deviation(test_x, stream_dim)

    assert test_dev > train_dev


def test_contextual_stream_gain_shift_valid_and_test_share_stream_prototypes() -> None:
    _, valid_ds, test_ds = create_contextual_stream_gain_shift_dataset(
        n_samples=1000,
        stream_dim=12,
        relevant_noise_std=0.02,
        irrelevant_noise_std=0.02,
        train_gain_relevant_min=1.0,
        train_gain_relevant_max=1.0,
        train_gain_irrelevant_min=1.0,
        train_gain_irrelevant_max=1.0,
        test_gain_relevant=1.0,
        test_gain_irrelevant=1.0,
        seed=7,
    )

    valid_x, valid_y = valid_ds.tensors
    test_x, test_y = test_ds.tensors

    valid_means = _contextual_class_means(valid_x, valid_y, stream_dim=12)
    test_means = _contextual_class_means(test_x, test_y, stream_dim=12)

    for vm, tm in zip(valid_means, test_means):
        assert torch.max(torch.abs(vm - tm)) < 0.12
