import sys
import types

import numpy as np
import pytest
import torch

from dendritic_modeling.datasets.sequence.builders.common import (
    build_deferred_train_test_split,
    build_explicit_or_deferred_split,
    explicit_split,
)
from dendritic_modeling.datasets.sequence.builders.registry import (
    SEQUENCE_DATASET_BUILDERS,
    get_registered_sequence_dataset_names,
    get_sequence_dataset_builder,
    has_sequence_dataset_builder,
    merge_sequence_dataset_builders,
    register_sequence_dataset_builder,
    unregister_sequence_dataset_builder,
)
from dendritic_modeling.datasets.sequence.factory import (
    _registered_sequence_dataset_task_config,
)
from dendritic_modeling.datasets.sequence_datasets import (
    AddingProblemDataset,
    ContextDependentDecisionDataset,
    CopyTaskDataset,
    GainModulatedContextualComparisonDataset,
    HierarchicalTemporalDataset,
    LorenzSequenceDataset,
    MultiFrequencyDataset,
    MultiSineForecastDataset,
    OculomotorDelayedResponseDataset,
    RomoDelayComparisonDataset,
    SequentialMNISTDataset,
    StringerV1Dataset,
    SwitchingContextDecisionDataset,
    SwitchingLDSDataset,
    TimescaleGeneralizationDataset,
    VariableDelayDMSDataset,
    get_sequence_datasets,
)


@pytest.fixture(autouse=True)
def _seed_rngs():
    torch.manual_seed(0)
    np.random.seed(0)


def _write_tiny_stringer(
    tmp_path,
    *,
    n_neurons=4,
    n_trials=20,
    grating_file=StringerV1Dataset._GRATING_FILE,
    offset=0.0,
):
    responses = np.stack(
        [
            np.linspace(0.0, 1.0 + neuron, n_trials)
            + (10.0 * (np.arange(n_trials) >= n_trials // 2))
            + offset
            for neuron in range(n_neurons)
        ],
        axis=0,
    ).astype(np.float32)
    orientations = np.linspace(0.0, 2 * np.pi, n_trials, endpoint=False).astype(
        np.float32
    )
    np.save(
        tmp_path / grating_file,
        {"sresp": responses, "istim": orientations},
        allow_pickle=True,
    )
    all_running = np.empty(1, dtype=object)
    all_running[0] = np.linspace(0.0, 1.0, n_trials).astype(np.float32)
    np.save(tmp_path / StringerV1Dataset._RUNNING_FILE, all_running, allow_pickle=True)


def test_sequence_builder_registry_rejects_duplicate_names():
    def builder(_dataset_name, _data_path, _kwargs):
        raise AssertionError("builder should not be called")

    with pytest.raises(ValueError, match="duplicate"):
        merge_sequence_dataset_builders({"task": builder}, {"task": builder})

    assert SEQUENCE_DATASET_BUILDERS["copy_task"] is not None


def test_sequence_builder_registry_exposes_builtin_names():
    names = get_registered_sequence_dataset_names()

    assert "copy_task" in names
    assert "multi_sine_forecast" in names
    assert (
        get_sequence_dataset_builder("copy_task")
        is SEQUENCE_DATASET_BUILDERS["copy_task"]
    )
    assert has_sequence_dataset_builder("copy_task")
    assert not has_sequence_dataset_builder("not_a_sequence_dataset")


def test_sequence_builder_registry_adds_custom_builder():
    def builder(dataset_name, data_path, kwargs):
        assert dataset_name == "unit_test_sequence"
        assert data_path == "memory://unit-test"
        assert kwargs["marker"] == 7
        inputs = torch.zeros(3, 2)
        targets = torch.zeros(3, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        return explicit_split(dataset, dataset, dataset)

    register_sequence_dataset_builder("unit_test_sequence", builder)
    try:
        train_ds, valid_ds, test_ds = get_sequence_datasets(
            "unit_test_sequence",
            data_path="memory://unit-test",
            marker=7,
        )
        assert len(train_ds) == len(valid_ds) == len(test_ds) == 3
        assert "unit_test_sequence" in get_registered_sequence_dataset_names()

        with pytest.raises(ValueError, match="already registered"):
            register_sequence_dataset_builder("unit_test_sequence", builder)
    finally:
        unregister_sequence_dataset_builder("unit_test_sequence")


def test_registered_sequence_dataset_task_config_copies_parameters():
    kwargs = {"seed": 3, "length": 5}

    task_config = _registered_sequence_dataset_task_config(
        dataset_name="unit_sequence",
        data_path="memory://unit",
        train_valid_split=0.7,
        kwargs=kwargs,
    )
    kwargs["seed"] = 99

    assert task_config == {
        "dataset": "unit_sequence",
        "data_path": "memory://unit",
        "train_valid_split": 0.7,
        "parameters": {"seed": 3, "length": 5},
    }


def test_sequence_builder_explicit_split_merges_common_and_split_kwargs():
    created = []

    class RecordingDataset:
        def __init__(self, *, n_samples, **params):
            self.n_samples = n_samples
            self.params = params
            created.append(self)

    common_defaults = {"noise_std": 0.1, "seed": 1}
    build = build_explicit_or_deferred_split(
        dataset_cls=RecordingDataset,
        kwargs={
            "train": {"n_samples": 5, "seed": 10},
            "valid": {"noise_std": 0.3},
            "test": {"n_samples": 7},
        },
        common_defaults=common_defaults,
        n_train=20,
        min_test_size=3,
    )

    assert build.split_datasets == tuple(created)
    assert [dataset.n_samples for dataset in created] == [5, 4, 7]
    assert [dataset.params for dataset in created] == [
        {"noise_std": 0.1, "seed": 10},
        {"noise_std": 0.3, "seed": 1},
        {"noise_std": 0.1, "seed": 1},
    ]
    assert common_defaults == {"noise_std": 0.1, "seed": 1}


def test_sequence_builder_deferred_split_uses_derived_test_size():
    created = []

    class RecordingDataset:
        def __init__(self, *, n_samples, **params):
            self.n_samples = n_samples
            self.params = params
            created.append(self)

    common_defaults = {"seq_len": 8, "seed": 4}
    build = build_deferred_train_test_split(
        dataset_cls=RecordingDataset,
        n_train=30,
        common_defaults=common_defaults,
        min_test_size=5,
    )

    assert build.train_full is created[0]
    assert build.test_ds is created[1]
    assert [dataset.n_samples for dataset in created] == [30, 6]
    assert [dataset.params for dataset in created] == [
        {"seq_len": 8, "seed": 4},
        {"seq_len": 8, "seed": 4},
    ]


def test_sequence_builder_registry_rejects_non_callable_builder():
    with pytest.raises(TypeError, match="must be callable"):
        register_sequence_dataset_builder(
            "unit_test_bad",
            object(),  # type: ignore[arg-type]
        )


def test_sequential_mnist_modes_and_invalid(monkeypatch):
    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            del root, train, download, transform
            values = torch.arange(2 * 28 * 28, dtype=torch.float32) % 256
            self.data = values.reshape(2, 28, 28).to(torch.uint8)
            self.targets = torch.tensor([3, 7], dtype=torch.long)

    fake_transforms = types.SimpleNamespace(
        ToTensor=object,
        Compose=lambda transforms: transforms,
        Lambda=lambda fn: fn,
    )
    fake_torchvision = types.SimpleNamespace(
        datasets=types.SimpleNamespace(MNIST=FakeMNIST),
        transforms=fake_transforms,
    )
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)

    pixel_ds = SequentialMNISTDataset(mode="pixel", normalize=True)
    pixel_input, pixel_target = pixel_ds[0]
    assert pixel_input.shape == (784, 1)
    assert pixel_target.item() == 3
    assert float(pixel_ds.sequences.max()) <= 1.0

    row_ds = SequentialMNISTDataset(mode="row", normalize=False)
    row_input, row_target = row_ds[1]
    assert row_input.shape == (28, 28)
    assert row_target.item() == 7
    assert float(row_ds.sequences.max()) > 1.0

    with pytest.raises(ValueError, match="pixel' or 'row"):
        SequentialMNISTDataset(mode="bad", normalize=True)


def test_copy_and_adding_problem_datasets():
    copy_ds = CopyTaskDataset(n_samples=2, seq_len=3, delay=2, n_symbols=4)
    copy_inputs, copy_targets = copy_ds[0]
    output_start = copy_ds.seq_len + copy_ds.delay
    assert copy_inputs.shape == (8, 6)
    assert copy_inputs[output_start, copy_ds.n_symbols + 1].item() == 1.0
    assert torch.all(copy_targets[:output_start] == CopyTaskDataset.IGNORE_INDEX)
    assert torch.all(copy_targets[output_start : output_start + copy_ds.seq_len] >= 0)

    adding_ds = AddingProblemDataset(n_samples=3, seq_len=12)
    adding_inputs, adding_target = adding_ds[0]
    marked_positions = torch.nonzero(adding_inputs[:, 1], as_tuple=False).flatten()
    assert len(marked_positions) == 2
    expected = adding_inputs[marked_positions, 0].sum()
    assert torch.isclose(adding_target.squeeze(), expected)


def test_variable_delay_and_oculomotor_datasets():
    dms_ds = VariableDelayDMSDataset(
        n_samples=4,
        stimulus_dim=5,
        stimulus_duration=3,
        min_delay=2,
        max_delay=4,
        probe_duration=2,
    )
    sample_inputs, sample_target, sample_seq_len = dms_ds[0]
    batch_inputs, batch_target, batch_seq_len = dms_ds[[0, 1]]
    assert sample_inputs.shape == (dms_ds.max_total_len, dms_ds.input_dim)
    assert sample_target.item() in {0, 1}
    assert isinstance(sample_seq_len, int)
    assert batch_inputs.shape[0] == 2
    assert batch_target.shape == (2,)
    assert batch_seq_len.shape == (2,)
    assert torch.any(sample_inputs[:, -1] > 0.0)

    odr_ds = OculomotorDelayedResponseDataset(
        n_samples=2,
        n_targets=4,
        input_tuning_dim=6,
        fixation_duration=2,
        cue_duration=3,
        min_delay=2,
        max_delay=2,
        go_duration=2,
        cue_noise_std=0.0,
        positive_input_encoding=True,
    )
    odr_inputs, odr_target, odr_seq_len = odr_ds[0]
    cue_slice = slice(
        odr_ds.fixation_duration, odr_ds.fixation_duration + odr_ds.cue_duration
    )
    go_start = odr_ds.fixation_duration + odr_ds.cue_duration + odr_ds.delays[0].item()
    assert odr_inputs.shape == (odr_ds.max_total_len, odr_ds.input_dim)
    assert odr_target.item() in range(odr_ds.n_targets)
    assert isinstance(odr_seq_len, int)
    assert torch.all(odr_inputs[cue_slice, : odr_ds.input_tuning_dim] >= 0.0)
    assert torch.all(
        odr_inputs[
            : odr_ds.fixation_duration + odr_ds.cue_duration + odr_ds.delays[0],
            odr_ds.input_tuning_dim,
        ]
        == 1.0
    )
    assert torch.all(
        odr_inputs[go_start:odr_seq_len, odr_ds.input_tuning_dim + 1] == 1.0
    )


def test_romo_and_gain_modulated_comparison_datasets():
    signed_ds = RomoDelayComparisonDataset(
        n_samples=2,
        frequency_values=(2.0, 4.0, 6.0),
        stimulus_duration=5,
        min_delay=2,
        max_delay=2,
        response_duration=2,
        noise_std=0.0,
        positive_input_encoding=False,
    )
    signed_inputs, _, signed_seq_len = signed_ds[0]
    assert signed_inputs.shape == (signed_ds.max_total_len, signed_ds.input_dim)
    assert signed_seq_len == (
        2 * signed_ds.stimulus_duration + 2 + signed_ds.response_duration
    )
    assert signed_ds.input_dim == 4
    assert float(signed_inputs[: signed_ds.stimulus_duration, 0].min()) < 0.0
    assert torch.all(signed_ds.frequency1_indices != signed_ds.frequency2_indices)
    assert torch.equal(
        signed_ds.targets,
        (signed_ds.frequency2_values > signed_ds.frequency1_values).long(),
    )
    assert torch.all(
        (signed_ds.frequency2_values - signed_ds.frequency1_values).abs() > 0
    )

    positive_ds = RomoDelayComparisonDataset(
        n_samples=2,
        frequency_values=(2.0, 4.0, 6.0),
        stimulus_duration=5,
        min_delay=2,
        max_delay=2,
        response_duration=2,
        noise_std=0.0,
        positive_input_encoding=True,
        tuning_dim=5,
    )
    positive_inputs, _, positive_seq_len = positive_ds[0]
    positive_code = positive_ds._positive_frequency_code(4.0, noise_std=0.1)
    assert positive_ds.input_dim == positive_ds.tuning_dim + 3
    assert positive_inputs.shape == (positive_ds.max_total_len, positive_ds.input_dim)
    assert positive_seq_len == (
        2 * positive_ds.stimulus_duration + 2 + positive_ds.response_duration
    )
    assert torch.all(positive_inputs >= 0.0)
    assert torch.all(positive_code >= 0.0)

    with pytest.raises(ValueError, match="n_pathways must be >= 2"):
        GainModulatedContextualComparisonDataset(n_pathways=1)
    with pytest.raises(ValueError, match="pathway_dim must be >= 1"):
        GainModulatedContextualComparisonDataset(pathway_dim=0)
    with pytest.raises(ValueError, match="min_delay"):
        GainModulatedContextualComparisonDataset(min_delay=5, max_delay=3)
    with pytest.raises(ValueError, match="at least two entries"):
        GainModulatedContextualComparisonDataset(amplitude_values=(0.5,))
    with pytest.raises(ValueError, match="positive"):
        GainModulatedContextualComparisonDataset(global_gain_min=0.0)

    gain_ds = GainModulatedContextualComparisonDataset(
        n_samples=4,
        n_pathways=3,
        pathway_dim=2,
        cue_duration=2,
        stimulus_duration=4,
        min_delay=3,
        max_delay=3,
        response_duration=2,
        amplitude_values=(0.4, 0.7, 1.1),
        comparison_margin=0.2,
        gain_drift_std=0.0,
        noise_std=0.0,
        cue_noise_std=0.0,
        distractor_burst_prob=1.0,
        distractor_burst_duration=2,
        positive_input_encoding=True,
    )
    gain_inputs, gain_target, gain_seq_len = gain_ds[0]
    assert gain_inputs.shape == (gain_ds.max_total_len, gain_ds.input_dim)
    assert gain_target.item() in {0, 1}
    assert isinstance(gain_seq_len, int)
    assert torch.all(gain_inputs >= 0.0)
    assert gain_ds.pathway_templates.shape == (
        gain_ds.n_pathways,
        gain_ds.stimulus_duration,
        gain_ds.pathway_dim,
    )
    assert gain_ds.delay_burst_flags.any()
    constant_trace = gain_ds._build_gain_trace(torch.tensor(1.1), length=1)
    assert torch.equal(constant_trace, torch.tensor([1.1]))

    with pytest.raises(ValueError, match="No valid amplitude pairs"):
        GainModulatedContextualComparisonDataset(
            n_samples=1,
            amplitude_values=(0.5, 0.6),
            comparison_margin=1.0,
        )


def test_romo_balanced_second_stimulus_pairs_are_exact_and_shortcut_free():
    dataset = RomoDelayComparisonDataset(
        n_samples=96,
        frequency_values=(2.0, 2.8, 3.6, 4.4, 5.2, 6.0),
        stimulus_duration=5,
        min_delay=2,
        max_delay=5,
        response_duration=2,
        noise_std=0.05,
        positive_input_encoding=True,
        tuning_dim=5,
        pair_sampling="balanced_second_stimulus_v1",
    )

    assert dataset.targets.bincount(minlength=2).tolist() == [48, 48]
    assert sorted(dataset.pair_ids.unique().tolist()) == list(range(48))
    stratum_counts = {}
    for pair_id in range(48):
        positions = torch.nonzero(dataset.pair_ids == pair_id, as_tuple=False).flatten()
        assert positions.numel() == 2
        first, second = (int(value) for value in positions.tolist())
        assert {int(dataset.targets[first]), int(dataset.targets[second])} == {0, 1}
        assert dataset.frequency2_indices[first] == dataset.frequency2_indices[second]
        assert dataset.delays[first] == dataset.delays[second]
        assert (
            dataset.comparison_distance_indices[first]
            == dataset.comparison_distance_indices[second]
        )
        # Everything after the first stimulus is byte-identical across the
        # opposite-label pair, including the silent delay, noisy f2 trace,
        # phase markers, response window, and padding.
        assert torch.equal(
            dataset.inputs[first, dataset.stimulus_duration :],
            dataset.inputs[second, dataset.stimulus_duration :],
        )
        for position in (first, second):
            target = int(dataset.targets[position])
            f1_index = int(dataset.frequency1_indices[position])
            f2_index = int(dataset.frequency2_indices[position])
            assert (f1_index < f2_index) is bool(target)
            key = (
                f2_index,
                int(dataset.comparison_distance_indices[position]),
                int(dataset.delays[position]),
            )
            counts = stratum_counts.setdefault(key, [0, 0])
            counts[target] += 1

    assert all(counts[0] == counts[1] for counts in stratum_counts.values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_samples": 3}, "positive even"),
        ({"frequency_values": (2.0, 3.0)}, "at least three"),
        (
            {"frequency_values": (2.0, 4.0, 3.0)},
            "strictly increasing",
        ),
    ],
)
def test_romo_balanced_second_stimulus_rejects_invalid_designs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RomoDelayComparisonDataset(
            pair_sampling="balanced_second_stimulus_v1",
            **kwargs,
        )


def test_romo_rejects_unknown_pair_sampling_policy():
    with pytest.raises(ValueError, match="pair_sampling"):
        RomoDelayComparisonDataset(pair_sampling="unknown")


def test_frequency_forecast_and_dynamical_sequence_datasets(tmp_path):
    freq_ds = MultiFrequencyDataset(
        n_samples=3, seq_len=16, n_frequencies=4, base_freq=0.05
    )
    freq_inputs, freq_target = freq_ds[0]
    assert freq_inputs.shape == (16, 1)
    assert freq_target.item() in range(freq_ds.n_frequencies)

    with pytest.raises(ValueError, match="prediction_mode"):
        MultiSineForecastDataset(prediction_mode="bad")
    with pytest.raises(ValueError, match="n_components"):
        MultiSineForecastDataset(n_components=0)

    sine_all = MultiSineForecastDataset(
        n_samples=3,
        seq_len=12,
        rollout_horizon=5,
        prediction_mode="all",
        n_components=2,
        observation_noise_std=0.0,
    )
    sine_last = MultiSineForecastDataset(
        n_samples=3,
        seq_len=12,
        rollout_horizon=5,
        prediction_mode="last",
        n_components=2,
        observation_noise_std=0.0,
    )
    assert sine_all[0][0].shape == (12, 1)
    assert sine_all[0][1].shape == (12, 1)
    assert sine_last[0][1].shape == (1,)
    recovered = sine_all.denormalize_state_tensor(sine_all.full_states[:1])
    assert torch.allclose(recovered, sine_all.full_states_raw[:1], atol=1e-5)
    obs = sine_all.states_to_observations(sine_all.full_states_raw[:1], add_noise=False)
    assert obs.shape == sine_all.full_states[:1].shape

    with pytest.raises(ValueError, match="prediction_mode"):
        LorenzSequenceDataset(prediction_mode="bad")
    with pytest.raises(ValueError, match="observation_mode"):
        LorenzSequenceDataset(observation_mode="bad")
    with pytest.raises(ValueError, match="integration_method"):
        LorenzSequenceDataset(integration_method="bad")

    lorenz_ds = LorenzSequenceDataset(
        n_samples=2,
        seq_len=8,
        rollout_horizon=3,
        dt=0.02,
        prediction_mode="all",
        observation_mode="partial",
        observation_dim=2,
        observation_noise_std=0.0,
        burn_in_steps=5,
        normalization_steps=40,
        integration_method="rk4",
    )
    lorenz_inputs, lorenz_target = lorenz_ds[0]
    assert lorenz_inputs.shape == (8, 2)
    assert lorenz_target.shape == (8, 3)
    assert lorenz_ds.observation_matrix.shape == (3, 2)
    restored = lorenz_ds.denormalize_state_tensor(lorenz_ds.full_states[:1])
    assert torch.allclose(restored, lorenz_ds.full_states_raw[:1], atol=1e-4)
    obs = lorenz_ds.states_to_observations(
        lorenz_ds.full_states_raw[:1], add_noise=False
    )
    assert obs.shape[-1] == 2

    lorenz_train = LorenzSequenceDataset(
        n_samples=2,
        seq_len=6,
        rollout_horizon=2,
        observation_mode="linear",
        observation_dim=2,
        burn_in_steps=3,
        normalization_steps=20,
        seed=10,
        observation_seed=123,
    )
    lorenz_test = LorenzSequenceDataset(
        n_samples=2,
        seq_len=6,
        rollout_horizon=2,
        observation_mode="linear",
        observation_dim=2,
        burn_in_steps=3,
        normalization_steps=20,
        seed=11,
        observation_seed=123,
    )
    assert torch.allclose(
        lorenz_train.observation_matrix, lorenz_test.observation_matrix
    )

    with pytest.raises(ValueError, match="latent_dim=4"):
        SwitchingLDSDataset(latent_dim=3)
    with pytest.raises(ValueError, match="n_modes must be >= 2"):
        SwitchingLDSDataset(n_modes=1)

    slds_ds = SwitchingLDSDataset(
        n_samples=2,
        seq_len=10,
        latent_dim=4,
        observation_dim=2,
        n_modes=3,
        rollout_horizon=4,
        prediction_mode="last",
        observation_noise_std=0.0,
        transition_noise_std=0.01,
        min_mode_duration=3,
        max_mode_duration=5,
        burn_in_steps=2,
        normalization_samples=12,
    )
    slds_inputs, slds_target = slds_ds[0]
    assert slds_inputs.shape == (10, 2)
    assert slds_target.shape == (4,)
    assert slds_ds.switch_mask.shape[0] == len(slds_ds)
    assert torch.all(slds_ds.time_since_switch[:, 0] == 0)
    restored_obs = slds_ds.denormalize_observation_tensor(slds_ds.inputs[:1])
    assert restored_obs.shape == slds_ds.inputs[:1].shape

    with pytest.raises(ValueError, match="split must be"):
        StringerV1Dataset(data_dir=str(tmp_path), split="dev")
    with pytest.raises(ValueError, match="running_state"):
        StringerV1Dataset(data_dir=str(tmp_path), running_state="medium")
    with pytest.raises(ValueError, match="running_quantile"):
        StringerV1Dataset(data_dir=str(tmp_path), running_quantile=0.75)
    with pytest.raises(ValueError, match="gain_group_count"):
        StringerV1Dataset(data_dir=str(tmp_path), gain_group_count=0)
    with pytest.raises(ValueError, match="poisson_rate_scale"):
        StringerV1Dataset(data_dir=str(tmp_path), poisson_rate_scale=0.0)
    with pytest.raises(ValueError, match="neuron_selection_scope"):
        StringerV1Dataset(data_dir=str(tmp_path), neuron_selection_scope="future")
    with pytest.raises(FileNotFoundError, match="Stringer V1 data not found"):
        StringerV1Dataset(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="grating_file must be a filename"):
        StringerV1Dataset(
            data_dir=str(tmp_path),
            grating_file="../outside.npy",
        )


def test_stringer_v1_selects_requested_grating_file(tmp_path):
    _write_tiny_stringer(tmp_path, grating_file=StringerV1Dataset._GRATING_FILE)
    _write_tiny_stringer(tmp_path, grating_file="alternate_session.npy", offset=5.0)

    default_ds = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=4,
        n_classes=4,
        split="all",
        normalize=False,
        session_idx=0,
    )
    alternate_ds = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=4,
        n_classes=4,
        split="all",
        normalize=False,
        session_idx=0,
        grating_file="alternate_session.npy",
    )

    assert default_ds.grating_file == StringerV1Dataset._GRATING_FILE
    assert alternate_ds.grating_file == "alternate_session.npy"
    assert torch.allclose(alternate_ds.responses, default_ds.responses + 5.0)


def test_stringer_v1_running_residualization_removes_classwise_running_gain(tmp_path):
    n_trials = 48
    n_neurons = 4
    running = np.linspace(-1.0, 1.0, n_trials).astype(np.float32)
    labels = np.arange(n_trials) % 4
    orientations = (labels * (2 * np.pi / 4)).astype(np.float32)
    responses = np.stack(
        [3.0 + labels + (neuron + 1) * running for neuron in range(n_neurons)],
        axis=0,
    ).astype(np.float32)
    responses -= responses.min() - 0.25
    np.save(
        tmp_path / StringerV1Dataset._GRATING_FILE,
        {"sresp": responses, "istim": orientations},
        allow_pickle=True,
    )
    all_running = np.empty(1, dtype=object)
    all_running[0] = running
    np.save(tmp_path / StringerV1Dataset._RUNNING_FILE, all_running, allow_pickle=True)

    raw_ds = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=n_neurons,
        n_classes=4,
        split="all",
        normalize=False,
        session_idx=0,
    )
    resid_ds = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=n_neurons,
        n_classes=4,
        split="all",
        normalize=False,
        session_idx=0,
        running_residualize=True,
    )

    assert resid_ds.running_residualized is True
    assert torch.all(resid_ds.responses >= 0.0)
    raw_np = raw_ds.responses.numpy()
    resid_np = resid_ds.responses.numpy()
    for class_idx in range(4):
        mask = labels == class_idx
        r = running[mask] - running[mask].mean()
        raw_slope = float(np.dot(r, raw_np[mask, 0]) / np.dot(r, r))
        resid_slope = float(np.dot(r, resid_np[mask, 0]) / np.dot(r, r))
        assert abs(resid_slope) < 1e-5
        assert abs(raw_slope) > 0.1


def test_stringer_v1_session_id_resolves_file_and_running_index(tmp_path):
    session_file = "gratings_drifting_GT1_2019_04_12_1.npy"
    _write_tiny_stringer(tmp_path, grating_file=session_file)
    np.save(
        tmp_path / "database.npy",
        np.array(
            [
                {
                    "expt": "gratings_drifting",
                    "mouse_name": "GT1",
                    "date": "2019_04_12",
                    "block": "1",
                }
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )

    train_ds, _, test_ds = get_sequence_datasets(
        "stringer_v1",
        data_path=str(tmp_path),
        n_neurons=4,
        n_classes=4,
        normalize=False,
        session_id="GT1_2019_04_12",
        neuron_selection_scope="train_split",
        train_valid_split=0.8,
    )

    assert train_ds.dataset.grating_file == session_file
    assert test_ds.grating_file == session_file
    assert train_ds.dataset.neuron_selection_scope == "train_split"
    assert test_ds.neuron_selection_scope == "train_split"
    assert np.array_equal(train_ds.dataset._neuron_indices, test_ds._neuron_indices)


def test_stringer_v1_uses_train_split_normalization_stats(tmp_path):
    _write_tiny_stringer(tmp_path)
    train_ds = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=4,
        n_classes=4,
        split="train",
        train_fraction=0.5,
        normalize=True,
        seed=3,
        session_idx=0,
    )
    test_ds = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=4,
        n_classes=4,
        split="test",
        train_fraction=0.5,
        normalize=True,
        seed=3,
        session_idx=0,
    )

    assert torch.allclose(train_ds.responses.mean(dim=0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(
        train_ds.responses.std(dim=0, unbiased=False), torch.ones(4), atol=1e-5
    )
    train_mu, train_std = train_ds.normalization_stats
    test_mu, test_std = test_ds.normalization_stats
    assert np.allclose(train_mu, test_mu)
    assert np.allclose(train_std, test_std)

    raw = (
        np.load(tmp_path / StringerV1Dataset._GRATING_FILE, allow_pickle=True)
        .item()["sresp"][train_ds._neuron_indices]
        .T
    )
    assert not np.allclose(train_mu, raw.mean(axis=0, keepdims=True))


def test_stringer_v1_training_only_neuron_selection_excludes_test_trials(tmp_path):
    n_trials = 20
    seed = 3
    train_fraction = 0.5
    perm = np.random.RandomState(seed).permutation(n_trials)
    train_idx = perm[: int(n_trials * train_fraction)]
    test_idx = perm[int(n_trials * train_fraction) :]

    responses = np.zeros((4, n_trials), dtype=np.float32)
    # Neuron 0 varies only in the held-out trials, whereas neuron 1 has the
    # largest variance in the training partition.
    responses[0, test_idx] = np.linspace(0.0, 100.0, len(test_idx))
    responses[1, train_idx] = np.linspace(0.0, 10.0, len(train_idx))
    responses[2, train_idx] = np.linspace(0.0, 2.0, len(train_idx))
    orientations = np.linspace(0.0, 2 * np.pi, n_trials, endpoint=False).astype(
        np.float32
    )
    np.save(
        tmp_path / StringerV1Dataset._GRATING_FILE,
        {"sresp": responses, "istim": orientations},
        allow_pickle=True,
    )
    all_running = np.empty(1, dtype=object)
    all_running[0] = np.linspace(0.0, 1.0, n_trials).astype(np.float32)
    np.save(tmp_path / StringerV1Dataset._RUNNING_FILE, all_running, allow_pickle=True)

    legacy = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=1,
        split="train",
        train_fraction=train_fraction,
        normalize=False,
        seed=seed,
        session_idx=0,
    )
    prospective_train = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=1,
        neuron_selection_scope="train_split",
        split="train",
        train_fraction=train_fraction,
        normalize=False,
        seed=seed,
        session_idx=0,
    )
    prospective_test = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=1,
        neuron_selection_scope="train_split",
        split="test",
        train_fraction=train_fraction,
        normalize=False,
        seed=seed,
        session_idx=0,
    )

    assert legacy._neuron_indices.tolist() == [0]
    assert prospective_train._neuron_indices.tolist() == [1]
    assert np.array_equal(
        prospective_train._neuron_indices, prospective_test._neuron_indices
    )
    assert set(prospective_train._split_indices).isdisjoint(
        set(prospective_test._split_indices)
    )

    prospective_top_three = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=3,
        neuron_selection_scope="train_split",
        split="train",
        train_fraction=train_fraction,
        normalize=False,
        seed=seed,
        session_idx=0,
    )
    assert (
        prospective_top_three._neuron_indices[0] == prospective_train._neuron_indices[0]
    )


def test_stringer_v1_state_specific_overlap_is_rejected(tmp_path):
    _write_tiny_stringer(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        get_sequence_datasets(
            "stringer_v1",
            data_path=str(tmp_path),
            n_neurons=4,
            n_classes=4,
            normalize=False,
            session_idx=0,
            train_running_state="all",
            test_running_state="high",
        )


def test_stringer_v1_state_transfer_reuses_training_state_neurons(tmp_path):
    _write_tiny_stringer(tmp_path, n_neurons=6, n_trials=40)
    train_ds, _, test_ds = get_sequence_datasets(
        "stringer_v1",
        data_path=str(tmp_path),
        n_neurons=3,
        n_classes=4,
        normalize=False,
        session_idx=0,
        neuron_selection_scope="train_split",
        train_running_state="low",
        test_running_state="high",
        running_quantile=0.25,
        train_valid_split=0.8,
    )

    assert train_ds.dataset.neuron_selection_scope == "train_split"
    assert test_ds.neuron_selection_scope == "train_split"
    assert np.array_equal(train_ds.dataset._neuron_indices, test_ds._neuron_indices)
    assert set(train_ds.dataset._split_indices).isdisjoint(test_ds._split_indices)


def test_stringer_v1_state_transfer_reserves_untouched_same_state_holdout(tmp_path):
    _write_tiny_stringer(tmp_path, n_neurons=6, n_trials=80)
    train_ds, valid_ds, high_ds = get_sequence_datasets(
        "stringer_v1",
        data_path=str(tmp_path),
        n_neurons=3,
        n_classes=4,
        normalize=False,
        session_idx=0,
        neuron_selection_scope="train_split",
        train_running_state="low",
        test_running_state="high",
        running_quantile=0.25,
        state_holdout_fraction=0.2,
        train_valid_split=0.8,
        label_noise_seed=7,
    )
    train_full = train_ds.dataset
    low_holdout = StringerV1Dataset(
        data_dir=str(tmp_path),
        n_neurons=3,
        n_classes=4,
        normalize=False,
        session_idx=0,
        neuron_selection_scope="train_split",
        neuron_selection_indices=train_full._neuron_indices,
        split="test",
        train_fraction=0.8,
        seed=7,
        running_state="low",
        running_quantile=0.25,
    )

    used_low = set(train_full._split_indices.tolist())
    heldout_low = set(low_holdout._split_indices.tolist())
    assert used_low.isdisjoint(heldout_low)
    assert set(high_ds._split_indices.tolist()).isdisjoint(used_low | heldout_low)
    assert np.array_equal(train_full._neuron_indices, low_holdout._neuron_indices)
    assert np.array_equal(train_full._neuron_indices, high_ds._neuron_indices)
    assert len(train_ds) + len(valid_ds) == len(train_full)


def test_stringer_v1_state_holdout_fraction_is_validated(tmp_path):
    _write_tiny_stringer(tmp_path)
    with pytest.raises(ValueError, match="state_holdout_fraction"):
        get_sequence_datasets(
            "stringer_v1",
            data_path=str(tmp_path),
            state_holdout_fraction=1.0,
        )


def test_context_and_timescale_datasets():
    hierarchical_ds = HierarchicalTemporalDataset(
        n_samples=4,
        n_contexts=2,
        n_block_types=3,
        n_blocks=5,
        context_duration=3,
        block_duration=4,
        signal_dim=6,
        noise_std=0.1,
    )
    assert hierarchical_ds[0][0].shape == (
        hierarchical_ds.total_len,
        hierarchical_ds.signal_dim,
    )
    assert hierarchical_ds.targets.tolist()[:4] == [0, 1, 0, 1]
    sample_idx = 1
    ctx = hierarchical_ds.contexts[sample_idx].item()
    target_type = hierarchical_ds.context_to_target[ctx].item()
    target_count = int((hierarchical_ds.block_types[sample_idx] == target_type).sum())
    if hierarchical_ds.targets[sample_idx].item() == 1:
        assert target_count >= hierarchical_ds.threshold
    else:
        assert target_count < hierarchical_ds.threshold

    cdd_ds = ContextDependentDecisionDataset(
        n_samples=3,
        signal_dim=4,
        context_duration=3,
        integration_duration=6,
        variable_delay=True,
        min_integration=5,
        max_integration=5,
        min_delay=2,
        max_delay=2,
        conflict_fraction=1.0,
        distractor_burst_prob=1.0,
        distractor_burst_duration=2,
        cue_dropout_prob=1.0,
        cue_keep_duration=1,
        late_context_switch_prob=1.0,
        return_seq_lengths=True,
        positive_input_encoding=True,
    )
    cdd_inputs, cdd_target, cdd_seq_len = cdd_ds[0]
    assert cdd_inputs.shape == (cdd_ds.seq_len, cdd_ds.input_dim)
    assert cdd_target.item() in {0, 1}
    assert isinstance(cdd_seq_len, int)
    assert torch.all(cdd_inputs >= 0.0)
    assert cdd_ds.return_seq_lengths
    assert torch.any(cdd_ds.switch_steps >= 0)

    scd_ds = SwitchingContextDecisionDataset(
        n_samples=2,
        signal_dim=4,
        seq_len=48,
        trial_duration=6,
        min_context_duration=12,
        max_context_duration=18,
        distractor_burst_prob=1.0,
        distractor_burst_duration=2,
        positive_input_encoding=True,
    )
    scd_inputs, scd_targets = scd_ds[0]
    assert scd_inputs.shape == (scd_ds.seq_len, scd_ds.input_dim)
    assert scd_targets.shape == (scd_ds.seq_len,)
    assert torch.all(scd_inputs >= 0.0)
    assert torch.all(scd_ds.switch_mask[:, 0])
    assert torch.any(scd_ds.cue_mask)

    tg_ds = TimescaleGeneralizationDataset(
        n_samples=3,
        stimulus_dim=5,
        stimulus_duration=3,
        min_delay=2,
        max_delay=4,
        probe_duration=2,
        positive_input_encoding=True,
        noise_std=0.0,
    )
    tg_inputs, tg_target, tg_seq_len = tg_ds[0]
    assert tg_inputs.shape == (tg_ds.max_seq_len, 5)
    assert tg_target.item() in {0, 1}
    assert int(tg_seq_len) >= 3 + 2 + 2
    assert int(tg_seq_len) <= 3 + 4 + 2
    assert torch.all(tg_inputs >= 0.0)
