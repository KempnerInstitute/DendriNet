import torch

from dendritic_modeling.datasets.synthetic_datasets import (
    create_hierarchical_gain_load_dataset,
    get_synthetic_datasets,
)


def _log_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    values = torch.corrcoef(torch.stack([left.flatten(), right.flatten()]))
    return float(values[0, 1].item())


def test_hierarchical_gain_load_is_registered_and_reproducible() -> None:
    config = {
        "type": "hierarchical_gain_load",
        "n_samples": 600,
        "stream_dim": 30,
        "n_levels": 3,
        "seed": 71,
    }
    datasets_a = get_synthetic_datasets(config)
    datasets_b = get_synthetic_datasets(config)

    for dataset_a, dataset_b in zip(datasets_a, datasets_b):
        inputs_a, labels_a = dataset_a.tensors
        inputs_b, labels_b = dataset_b.tensors
        assert inputs_a.shape[1] == 60
        assert torch.equal(inputs_a, inputs_b)
        assert torch.equal(labels_a, labels_b)
        assert set(labels_a.unique().tolist()) == {0, 1}


def test_factorized_sensors_exactly_report_nested_scalar_gain_product() -> None:
    train, _, _ = create_hierarchical_gain_load_dataset(
        n_samples=1800,
        stream_dim=60,
        n_levels=3,
        hierarchy_branching=1,
        e_baseline=1.0,
        i_baseline=1.0,
        sensor_e_baseline=0.0,
        e_signal_delta=0.0,
        train_gain_sigma=0.8,
        test_gain_sigma=0.8,
        sensor_alignment_alpha=1.0,
        sensor_support_mode="matched",
        independent_noise_std=0.0,
        private_gain_sigma=0.0,
        load_mean=0.0,
        seed=72,
    )
    inputs, _ = train.tensors
    excitation = inputs[:, :60]
    inhibition = inputs[:, 60:]

    signal_log_gain = excitation[:, :20].log().mean(dim=1)
    sensor_log_product = torch.stack(
        [
            inhibition[:, 0].log(),
            inhibition[:, 20].log(),
            inhibition[:, 40].log(),
        ]
    ).sum(dim=0)

    assert torch.allclose(signal_log_gain, sensor_log_product, atol=1e-5, rtol=1e-5)


def test_alignment_and_shuffle_preserve_sensor_marginal_but_remove_matching() -> None:
    common = {
        "n_samples": 2400,
        "stream_dim": 60,
        "n_levels": 3,
        "hierarchy_branching": 1,
        "e_baseline": 1.0,
        "i_baseline": 1.0,
        "sensor_e_baseline": 0.0,
        "e_signal_delta": 0.0,
        "train_gain_sigma": 0.9,
        "test_gain_sigma": 0.9,
        "sensor_alignment_alpha": 1.0,
        "independent_noise_std": 0.0,
        "private_gain_sigma": 0.0,
        "seed": 73,
    }
    matched, _, _ = create_hierarchical_gain_load_dataset(
        **common,
        sensor_support_mode="matched",
    )
    shuffled, _, _ = create_hierarchical_gain_load_dataset(
        **common,
        sensor_support_mode="shuffled",
    )

    def _signal_and_sensor(dataset):
        inputs, _ = dataset.tensors
        excitation = inputs[:, :60]
        inhibition = inputs[:, 60:]
        signal = excitation[:, :20].log().mean(dim=1)
        sensor = torch.stack(
            [
                inhibition[:, 0].log(),
                inhibition[:, 20].log(),
                inhibition[:, 40].log(),
            ]
        ).sum(dim=0)
        return signal, sensor

    matched_signal, matched_sensor = _signal_and_sensor(matched)
    shuffled_signal, shuffled_sensor = _signal_and_sensor(shuffled)

    assert _log_correlation(matched_signal, matched_sensor) > 0.999
    assert abs(_log_correlation(shuffled_signal, shuffled_sensor)) < 0.1
    # A cyclic trial shuffle changes pairing, not the exact sensor marginal.
    assert torch.allclose(
        matched_sensor.sort().values,
        shuffled_sensor.sort().values,
        atol=1e-5,
        rtol=1e-5,
    )


def test_paired_cumulative_layout_has_a_shunting_favorable_gain_regime() -> None:
    train, _, test = create_hierarchical_gain_load_dataset(
        n_samples=6000,
        stream_dim=60,
        n_levels=3,
        nuisance_layout="paired_cumulative",
        signal_mode="balanced_ratio",
        e_baseline=5.0,
        i_baseline=2.0,
        e_signal_delta=0.5,
        i_signal_delta=0.5,
        train_gain_sigma=0.2,
        test_gain_sigma=1.2,
        sensor_alignment_alpha=1.0,
        independent_noise_std=0.08,
        seed=74,
    )

    def _scores(dataset):
        inputs, labels = dataset.tensors
        excitation = inputs[:, :60].mean(dim=1)
        inhibition = inputs[:, 60:].mean(dim=1)
        additive = excitation - inhibition
        shunting = excitation / (1.0 + excitation + inhibition)
        return additive, shunting, labels

    def _threshold_accuracy(train_score, train_labels, test_score, test_labels):
        mean_0 = train_score[train_labels == 0].mean()
        mean_1 = train_score[train_labels == 1].mean()
        threshold = 0.5 * (mean_0 + mean_1)
        prediction = (
            test_score >= threshold if mean_1 >= mean_0 else test_score < threshold
        )
        return float((prediction.long() == test_labels).float().mean().item())

    add_train, shunt_train, train_labels = _scores(train)
    add_test, shunt_test, test_labels = _scores(test)
    additive_accuracy = _threshold_accuracy(
        add_train, train_labels, add_test, test_labels
    )
    shunting_accuracy = _threshold_accuracy(
        shunt_train, train_labels, shunt_test, test_labels
    )

    assert shunting_accuracy > additive_accuracy + 0.07
    assert shunting_accuracy > 0.65


def test_private_gain_and_load_are_genuine_unmatched_controls() -> None:
    common = {
        "n_samples": 2000,
        "stream_dim": 60,
        "n_levels": 3,
        "e_signal_delta": 0.0,
        "train_gain_sigma": 0.8,
        "test_gain_sigma": 0.8,
        "sensor_alignment_alpha": 1.0,
        "independent_noise_std": 0.0,
        "seed": 75,
    }
    clean, _, _ = create_hierarchical_gain_load_dataset(**common)
    adverse, _, _ = create_hierarchical_gain_load_dataset(
        **common,
        private_gain_sigma=0.7,
        load_mean=2.0,
        load_noise_sigma=0.2,
    )
    clean_inputs, _ = clean.tensors
    adverse_inputs, _ = adverse.tensors

    clean_e = clean_inputs[:, :60]
    clean_i = clean_inputs[:, 60:]
    adverse_e = adverse_inputs[:, :60]
    adverse_i = adverse_inputs[:, 60:]

    assert adverse_i.mean() > clean_i.mean() + 1.0
    assert _log_correlation(adverse_e[:, :20].log(), adverse_i[:, :20].log()) < (
        _log_correlation(clean_e[:, :20].log(), clean_i[:, :20].log()) - 0.1
    )
