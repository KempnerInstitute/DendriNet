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


def _signal_directions(
    inputs: torch.Tensor, labels: torch.Tensor, stream_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    a0, a1, b0, b1 = _contextual_class_means(inputs, labels, stream_dim)
    dir_a = a1 - a0
    dir_b = b1 - b0
    dir_a = dir_a / torch.clamp(torch.linalg.norm(dir_a), min=1e-8)
    dir_b = dir_b / torch.clamp(torch.linalg.norm(dir_b), min=1e-8)
    return dir_a, dir_b


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


def test_contextual_stream_alignment_alpha_rotates_irrelevant_test_shift() -> None:
    stream_dim = 10
    common_kwargs = {
        "n_samples": 1200,
        "stream_dim": stream_dim,
        "relevant_noise_std": 0.0,
        "irrelevant_noise_std": 0.0,
        "train_gain_relevant_min": 1.0,
        "train_gain_relevant_max": 1.0,
        "train_gain_irrelevant_min": 1.0,
        "train_gain_irrelevant_max": 1.0,
        "test_gain_relevant": 1.0,
        "ood_mode": "irrelevant",
        "seed": 11,
    }

    _, _, clean_ds = create_contextual_stream_gain_shift_dataset(
        **common_kwargs,
        test_gain_irrelevant=1.0,
        test_irrelevant_alignment_alpha=1.0,
    )
    _, _, aligned_ds = create_contextual_stream_gain_shift_dataset(
        **common_kwargs,
        test_gain_irrelevant=4.0,
        test_irrelevant_alignment_alpha=1.0,
    )
    _, _, orth_ds = create_contextual_stream_gain_shift_dataset(
        **common_kwargs,
        test_gain_irrelevant=4.0,
        test_irrelevant_alignment_alpha=0.0,
    )

    clean_x, clean_y = clean_ds.tensors
    aligned_x, aligned_y = aligned_ds.tensors
    orth_x, orth_y = orth_ds.tensors

    assert torch.equal(clean_y, aligned_y)
    assert torch.equal(clean_y, orth_y)

    dir_a, dir_b = _signal_directions(clean_x, clean_y, stream_dim)
    context = clean_x[:, 2 * stream_dim :].argmax(dim=1)

    clean_a = clean_x[:, :stream_dim]
    clean_b = clean_x[:, stream_dim : 2 * stream_dim]
    aligned_a = aligned_x[:, :stream_dim]
    aligned_b = aligned_x[:, stream_dim : 2 * stream_dim]
    orth_a = orth_x[:, :stream_dim]
    orth_b = orth_x[:, stream_dim : 2 * stream_dim]

    clean_irr = torch.where((context == 0).unsqueeze(1), clean_b, clean_a)
    aligned_irr = torch.where((context == 0).unsqueeze(1), aligned_b, aligned_a)
    orth_irr = torch.where((context == 0).unsqueeze(1), orth_b, orth_a)

    delta_aligned = aligned_irr - clean_irr
    delta_orth = orth_irr - clean_irr

    irrelevant_signal_dirs = torch.where(
        (context == 0).unsqueeze(1),
        dir_b.unsqueeze(0),
        dir_a.unsqueeze(0),
    )

    aligned_cos = torch.nn.functional.cosine_similarity(
        delta_aligned, irrelevant_signal_dirs, dim=1
    ).abs()
    orth_cos = torch.nn.functional.cosine_similarity(
        delta_orth, irrelevant_signal_dirs, dim=1
    ).abs()

    assert aligned_cos.mean() > orth_cos.mean() + 0.2

    aligned_norm = torch.linalg.norm(delta_aligned, dim=1).mean()
    orth_norm = torch.linalg.norm(delta_orth, dim=1).mean()
    ratio = aligned_norm / torch.clamp(orth_norm, min=1e-8)
    assert 0.8 < ratio < 1.25


def test_contextual_stream_valid_split_mode_train_is_alpha_invariant() -> None:
    common_kwargs = {
        "n_samples": 800,
        "stream_dim": 10,
        "relevant_noise_std": 0.0,
        "irrelevant_noise_std": 0.0,
        "train_gain_relevant_min": 1.0,
        "train_gain_relevant_max": 1.0,
        "train_gain_irrelevant_min": 1.0,
        "train_gain_irrelevant_max": 1.0,
        "test_gain_relevant": 1.0,
        "test_gain_irrelevant": 4.0,
        "ood_mode": "irrelevant",
        "valid_split_mode": "train",
        "seed": 17,
    }

    _, valid_a, test_a = create_contextual_stream_gain_shift_dataset(
        **common_kwargs,
        test_irrelevant_alignment_alpha=0.0,
    )
    _, valid_b, test_b = create_contextual_stream_gain_shift_dataset(
        **common_kwargs,
        test_irrelevant_alignment_alpha=1.0,
    )

    valid_x_a, valid_y_a = valid_a.tensors
    valid_x_b, valid_y_b = valid_b.tensors
    test_x_a, test_y_a = test_a.tensors
    test_x_b, test_y_b = test_b.tensors

    assert torch.equal(valid_y_a, valid_y_b)
    assert torch.equal(valid_x_a, valid_x_b)

    assert torch.equal(test_y_a, test_y_b)
    assert not torch.equal(test_x_a, test_x_b)
