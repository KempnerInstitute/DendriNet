import sys
import types

import numpy as np
import pytest
import torch

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
)


@pytest.fixture(autouse=True)
def _seed_rngs():
    torch.manual_seed(0)
    np.random.seed(0)


def test_sequential_mnist_modes_and_invalid(monkeypatch):
    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            del root, train, download, transform
            values = torch.arange(2 * 28 * 28, dtype=torch.float32) % 256
            self.data = values.reshape(2, 28, 28).to(torch.uint8)
            self.targets = torch.tensor([3, 7], dtype=torch.long)

    fake_transforms = types.SimpleNamespace(
        ToTensor=lambda: object(),
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
    with pytest.raises(FileNotFoundError, match="Stringer V1 data not found"):
        StringerV1Dataset(data_dir=str(tmp_path))


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
