import torch

from dendritic_modeling.datasets.synthetic_datasets import (
    create_branch_local_gain_load_dataset,
    create_contextual_stream_gain_shift_dataset,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
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


def test_ordered_independent_transfer_preserves_ei_halves() -> None:
    transfer = TransferLayer(
        input_dim=8,
        transfer_params={
            "input_mode": 1,
            "independent_pathways": True,
            "excitatory_dim": 4,
            "inhibitory_dim": 4,
            "split_strategy": "ordered",
        },
    )
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    x_e, x_i = transfer(x)
    assert torch.equal(x_e, x[:, :4])
    assert torch.equal(x_i, x[:, 4:])


def test_branch_local_gain_load_has_shunting_favorable_proxy_regime() -> None:
    train_ds, _, test_ds = create_branch_local_gain_load_dataset(
        n_samples=3000,
        stream_dim=32,
        signal_fraction=0.5,
        train_gain_sigma=0.8,
        test_gain_sigma=0.8,
        gain_alignment_alpha=1.0,
        load_mean=0.0,
        independent_noise_std=0.02,
        e_baseline=5.0,
        i_baseline=2.0,
        e_signal_delta=1.0,
        seed=123,
    )
    train_x, train_y = train_ds.tensors
    test_x, test_y = test_ds.tensors
    stream_dim = train_x.shape[1] // 2
    n_signal = stream_dim // 2

    def _scores(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e = x[:, :stream_dim][:, :n_signal].mean(dim=1)
        i = x[:, stream_dim:][:, :n_signal].mean(dim=1)
        additive = e - i
        shunting = e / (1.0 + e + i)
        return additive, shunting

    def _threshold_accuracy(
        train_scores: torch.Tensor,
        train_labels: torch.Tensor,
        test_scores: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> float:
        m0 = train_scores[train_labels == 0].mean()
        m1 = train_scores[train_labels == 1].mean()
        threshold = 0.5 * (m0 + m1)
        if m1 >= m0:
            pred = test_scores >= threshold
        else:
            pred = test_scores < threshold
        return float((pred.long() == test_labels).float().mean().item())

    add_train, sh_train = _scores(train_x)
    add_test, sh_test = _scores(test_x)
    add_acc = _threshold_accuracy(add_train, train_y, add_test, test_y)
    sh_acc = _threshold_accuracy(sh_train, train_y, sh_test, test_y)

    assert sh_acc > add_acc + 0.08
    assert sh_acc > 0.68


def test_branch_local_gain_load_balanced_ratio_proxy_has_load_boundary() -> None:
    def _proxy_accuracy(load_mean: float) -> tuple[float, float]:
        train_ds, _, test_ds = create_branch_local_gain_load_dataset(
            n_samples=4000,
            stream_dim=32,
            signal_fraction=0.5,
            train_gain_sigma=0.05,
            test_gain_sigma=1.8,
            gain_alignment_alpha=1.0,
            load_mean=load_mean,
            load_noise_sigma=0.2 if load_mean > 0 else 0.0,
            independent_noise_std=0.08,
            e_baseline=5.0,
            i_baseline=2.0,
            e_signal_delta=1.0,
            i_signal_delta=1.0,
            signal_mode="balanced_ratio",
            seed=321,
        )
        train_x, train_y = train_ds.tensors
        test_x, test_y = test_ds.tensors
        stream_dim = train_x.shape[1] // 2
        n_signal = stream_dim // 2

        def _scores(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            e = x[:, :stream_dim][:, :n_signal].mean(dim=1)
            i = x[:, stream_dim:][:, :n_signal].mean(dim=1)
            additive = e - i
            shunting = e / (1.0 + e + i)
            return additive, shunting

        def _threshold_accuracy(
            train_scores: torch.Tensor,
            train_labels: torch.Tensor,
            test_scores: torch.Tensor,
            test_labels: torch.Tensor,
        ) -> float:
            m0 = train_scores[train_labels == 0].mean()
            m1 = train_scores[train_labels == 1].mean()
            threshold = 0.5 * (m0 + m1)
            pred = test_scores >= threshold if m1 >= m0 else test_scores < threshold
            return float((pred.long() == test_labels).float().mean().item())

        add_train, sh_train = _scores(train_x)
        add_test, sh_test = _scores(test_x)
        return (
            _threshold_accuracy(add_train, train_y, add_test, test_y),
            _threshold_accuracy(sh_train, train_y, sh_test, test_y),
        )

    add_low_load, sh_low_load = _proxy_accuracy(load_mean=0.0)
    add_high_load, sh_high_load = _proxy_accuracy(load_mean=3.0)

    assert sh_low_load > add_low_load + 0.10
    assert sh_low_load > 0.75
    assert sh_high_load < add_high_load
    assert (sh_high_load - add_high_load) < (sh_low_load - add_low_load)


def test_branch_local_gain_load_sampled_support_is_sparse_and_reproducible() -> None:
    kwargs = {
        "n_samples": 1200,
        "stream_dim": 40,
        "signal_fraction": 0.25,
        "support_mode": "sampled",
        "gain_fraction": -1.0,
        "load_fraction": -1.0,
        "train_gain_sigma": 0.2,
        "test_gain_sigma": 0.2,
        "gain_alignment_alpha": 1.0,
        "independent_noise_std": 0.02,
        "e_baseline": 5.0,
        "i_baseline": 2.0,
        "e_signal_delta": 0.7,
        "i_signal_delta": 0.7,
        "signal_mode": "balanced_ratio",
        "seed": 777,
    }
    train_a, _, test_a = create_branch_local_gain_load_dataset(**kwargs)
    train_b, _, test_b = create_branch_local_gain_load_dataset(**kwargs)

    train_x, train_y = train_a.tensors
    repeat_x, repeat_y = train_b.tensors
    assert torch.equal(train_x, repeat_x)
    assert torch.equal(train_y, repeat_y)
    assert torch.equal(test_a.tensors[0], test_b.tensors[0])

    stream_dim = train_x.shape[1] // 2
    e = train_x[:, :stream_dim]
    i = train_x[:, stream_dim:]
    class_diff = (e[train_y == 1].mean(dim=0) - e[train_y == 0].mean(dim=0)).abs() + (
        i[train_y == 1].mean(dim=0) - i[train_y == 0].mean(dim=0)
    ).abs()
    active = class_diff > 0.5

    assert 4 <= int(active.sum().item()) <= 16
    assert bool(active[stream_dim // 4 :].any())


def test_branch_local_gain_load_sampled_matched_support_uses_same_gain_mask() -> None:
    base_kwargs = {
        "n_samples": 2400,
        "stream_dim": 40,
        "signal_fraction": 0.25,
        "support_mode": "sampled",
        "gain_fraction": -1.0,
        "load_fraction": -1.0,
        "train_gain_sigma": 1.0,
        "test_gain_sigma": 1.0,
        "gain_alignment_alpha": 0.0,
        "load_mean": 0.0,
        "independent_noise_std": 0.0,
        "e_baseline": 5.0,
        "i_baseline": 2.0,
        "e_signal_delta": 0.0,
        "i_signal_delta": 0.0,
        "signal_mode": "balanced_ratio",
        "seed": 779,
    }

    shifted_train, _, shifted_test = create_branch_local_gain_load_dataset(
        **base_kwargs,
        train_gain_support_mode="signal",
    )
    matched_train, _, matched_test = create_branch_local_gain_load_dataset(
        **base_kwargs,
        train_gain_support_mode="matched",
    )

    def _active_gain_dims(dataset: torch.utils.data.TensorDataset) -> torch.Tensor:
        x, _ = dataset.tensors
        stream_dim = x.shape[1] // 2
        variance = x[:, :stream_dim].var(dim=0)
        return variance > 1.0

    shifted_train_mask = _active_gain_dims(shifted_train)
    shifted_test_mask = _active_gain_dims(shifted_test)
    matched_train_mask = _active_gain_dims(matched_train)
    matched_test_mask = _active_gain_dims(matched_test)

    assert bool((shifted_train_mask & shifted_test_mask).sum() == 0)
    assert torch.equal(matched_train_mask, matched_test_mask)


def test_branch_local_gain_load_sampled_proxy_has_load_boundary() -> None:
    def _proxy_gap(load_mean: float) -> float:
        train_ds, _, test_ds = create_branch_local_gain_load_dataset(
            n_samples=4000,
            stream_dim=40,
            signal_fraction=0.25,
            support_mode="sampled",
            gain_fraction=-1.0,
            load_fraction=-1.0,
            train_gain_sigma=0.05,
            test_gain_sigma=1.8,
            gain_alignment_alpha=1.0,
            load_mean=load_mean,
            load_noise_sigma=0.2 if load_mean > 0 else 0.0,
            independent_noise_std=0.06,
            e_baseline=5.0,
            i_baseline=2.0,
            e_signal_delta=0.7,
            i_signal_delta=0.7,
            signal_mode="balanced_ratio",
            seed=778,
        )
        train_x, train_y = train_ds.tensors
        test_x, test_y = test_ds.tensors
        stream_dim = train_x.shape[1] // 2

        class_diff = (
            train_x[train_y == 1, :stream_dim].mean(dim=0)
            - train_x[train_y == 0, :stream_dim].mean(dim=0)
        ).abs() + (
            train_x[train_y == 1, stream_dim:].mean(dim=0)
            - train_x[train_y == 0, stream_dim:].mean(dim=0)
        ).abs()
        support = class_diff > 0.4

        def _scores(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            e = x[:, :stream_dim][:, support].mean(dim=1)
            i = x[:, stream_dim:][:, support].mean(dim=1)
            additive = e - i
            shunting = e / (1.0 + e + i)
            return additive, shunting

        def _threshold_accuracy(
            train_scores: torch.Tensor,
            train_labels: torch.Tensor,
            test_scores: torch.Tensor,
            test_labels: torch.Tensor,
        ) -> float:
            m0 = train_scores[train_labels == 0].mean()
            m1 = train_scores[train_labels == 1].mean()
            threshold = 0.5 * (m0 + m1)
            pred = test_scores >= threshold if m1 >= m0 else test_scores < threshold
            return float((pred.long() == test_labels).float().mean().item())

        add_train, sh_train = _scores(train_x)
        add_test, sh_test = _scores(test_x)
        add_acc = _threshold_accuracy(add_train, train_y, add_test, test_y)
        sh_acc = _threshold_accuracy(sh_train, train_y, sh_test, test_y)
        return sh_acc - add_acc

    low_load_gap = _proxy_gap(load_mean=0.0)
    high_load_gap = _proxy_gap(load_mean=3.0)

    assert low_load_gap > 0.08
    assert high_load_gap < low_load_gap - 0.08
