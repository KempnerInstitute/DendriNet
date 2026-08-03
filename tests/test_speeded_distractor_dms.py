import math

import pytest
import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets.sequence.builders.registry import (
    get_registered_sequence_dataset_names,
)
from dendritic_modeling.datasets.sequence.multiscale_working_memory import (
    SpeededDistractorDMSDataset,
)
from dendritic_modeling.datasets.sequence_datasets import get_sequence_datasets
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.training.loss import (
    MaskedDualTaskTemporalLoss,
    get_loss_function,
)


def _small_dataset(**overrides):
    parameters = {
        "n_samples": 16,
        "delay_support": (96, 112),
        "n_distractors": 4,
        "lure_fraction": 0.25,
        "salience_mu": 0.8,
        "circular_noise_std": 0.04,
        "sequence_duration": 160,
        "positive_input_encoding": True,
        "seed": 17,
        "intervention_seed": 271,
        "split_name": "unit",
    }
    parameters.update(overrides)
    return SpeededDistractorDMSDataset(**parameters)


_FROZEN_REGIMES = (
    ("c0", (128, 160, 192, 224), 4, 0.25, 0.80, 0.04),
    ("c1", (160, 192, 224, 256), 6, 0.40, 0.70, 0.06),
    ("c2", (192, 224, 256, 288), 8, 0.50, 0.60, 0.08),
    ("c3", (224, 256, 288, 320), 10, 0.60, 0.50, 0.10),
    ("c4", (192, 224, 256, 288), 8, 0.50, 0.45, 0.08),
    ("c5", (224, 256, 288, 320), 10, 0.70, 0.60, 0.12),
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and set(value) <= set("0123456789abcdef")


def test_speeded_distractor_dms_is_deterministic_without_ambient_rng_drift():
    torch.manual_seed(9876)
    before = torch.random.get_rng_state().clone()
    first = _small_dataset()
    after_first = torch.random.get_rng_state().clone()
    same = _small_dataset()
    after_second = torch.random.get_rng_state().clone()
    changed = _small_dataset(seed=18)
    relabeled = _small_dataset(split_name="relabeled")

    assert torch.equal(before, after_first)
    assert torch.equal(before, after_second)
    assert torch.equal(first.inputs, same.inputs)
    assert torch.equal(first.targets, same.targets)
    assert torch.equal(first.distractor_onsets, same.distractor_onsets)
    assert not torch.equal(first.inputs, changed.inputs)
    assert first.content_sha256 == same.content_sha256
    assert first.content_sha256 == relabeled.content_sha256
    assert first.content_sha256 != changed.content_sha256
    assert first.split_receipt_sha256 == same.split_receipt_sha256
    assert first.split_receipt_sha256 != relabeled.split_receipt_sha256
    assert first.rng_namespace_seeds == relabeled.rng_namespace_seeds
    assert first.rng_namespace_seeds != changed.rng_namespace_seeds
    assert set(first.rng_namespace_seeds) == set(first.RNG_NAMESPACES)
    assert all(_is_sha256(value) for value in first.sensory_noise_sha256_by_pair)
    assert _is_sha256(first.content_sha256)
    assert _is_sha256(first.counterbalance_sha256)
    assert _is_sha256(first.input_intervention_receipt_sha256)
    assert _is_sha256(first.split_receipt_sha256)
    assert first.split_metadata["seed"] == 17
    assert first.split_metadata["split_name"] == "unit"
    assert first.split_metadata["n_samples"] == 16
    assert first.split_metadata["n_pairs"] == 8


def test_speeded_distractor_dms_exact_pairs_timeline_and_packed_targets():
    dataset = _small_dataset()

    assert dataset.inputs.shape == (16, 160, 13)
    assert dataset.targets.shape == (16, 160, 4)
    assert dataset.target_plane_names == (
        "fast_label",
        "fast_mask",
        "memory_label",
        "memory_mask",
    )
    assert torch.all(dataset.inputs >= 0)
    assert torch.equal(dataset.pair_ids, torch.arange(8).repeat_interleave(2))
    assert torch.equal(dataset.sample_identities[::2], dataset.sample_identities[1::2])
    assert torch.equal(dataset.delays[::2], dataset.delays[1::2])
    assert torch.equal(dataset.distractor_onsets[::2], dataset.distractor_onsets[1::2])
    assert torch.equal(
        dataset.distractor_target_flags[::2],
        dataset.distractor_target_flags[1::2],
    )

    for pair_start in range(0, len(dataset), 2):
        probe_onset = int(dataset.probe_onsets[pair_start])
        assert torch.equal(
            dataset.inputs[pair_start, :probe_onset],
            dataset.inputs[pair_start + 1, :probe_onset],
        )
        assert sorted(dataset.match_labels[pair_start : pair_start + 2].tolist()) == [
            0,
            1,
        ]
        assert torch.equal(
            dataset.targets[pair_start, :, :2],
            dataset.targets[pair_start + 1, :, :2],
        )

    pair_delays = dataset.delays[::2]
    assert [int((pair_delays == delay).sum()) for delay in (96, 112)] == [4, 4]
    pair_targets = dataset.distractor_target_flags[::2]
    assert int(pair_targets.sum()) * 2 == pair_targets.numel()
    assert torch.unique(pair_targets.sum(dim=1)).numel() > 1
    assert torch.all(torch.diff(dataset.distractor_onsets, dim=1) >= 16)
    assert torch.all(dataset.distractor_onsets[:, 0] >= 32)
    assert torch.all(dataset.distractor_onsets[:, -1] <= dataset.probe_onsets - 20)

    for sample_index in range(len(dataset)):
        probe = int(dataset.probe_onsets[sample_index])
        memory_start = probe + 5
        memory_end = memory_start + 12
        delay = int(dataset.delays[sample_index])
        assert probe == 20 + delay
        assert int(dataset.seq_lengths[sample_index]) == memory_end
        assert torch.all(dataset.targets[sample_index, 20:probe, 1] == 1)
        assert not bool(dataset.targets[sample_index, :20, 1].any())
        assert not bool(dataset.targets[sample_index, probe:, 1].any())
        assert torch.all(dataset.targets[sample_index, memory_start:memory_end, 3] == 1)
        assert torch.all(
            dataset.targets[sample_index, memory_start:memory_end, 2]
            == dataset.match_labels[sample_index]
        )
        assert not bool(dataset.targets[sample_index, :memory_start, 3].any())
        assert not bool(dataset.targets[sample_index, memory_end:].any())
        for event_index, onset in enumerate(
            dataset.distractor_onsets[sample_index].tolist()
        ):
            expected = bool(dataset.distractor_target_flags[sample_index, event_index])
            response = dataset.targets[sample_index, onset + 2 : onset + 13, 0]
            assert torch.all(response == float(expected))

        seq_length = int(dataset.seq_lengths[sample_index])
        assert not bool(dataset.inputs[sample_index, seq_length:].any())
        assert not bool(dataset.targets[sample_index, seq_length:].any())

    for name in (
        "pair_ids",
        "sample_identities",
        "delays",
        "distractor_onsets",
        "distractor_identities",
        "distractor_target_flags",
        "lure_flags",
        "lure_types",
        "sensory_noise_realization_ids",
        "probe_onsets",
        "seq_lengths",
        "fast_response_starts",
        "fast_response_ends",
        "memory_response_starts",
        "memory_response_ends",
    ):
        value = getattr(dataset, name)
        assert torch.equal(value[::2], value[1::2])
    assert dataset.pair_sample_sensory_noise.shape == (8, 10, 8)
    assert dataset.pair_distractor_sensory_noise.shape == (8, 4, 5, 8)
    assert dataset.pair_probe_sensory_noise.shape == (8, 5, 8)
    assert dataset.pair_salience_evidence.shape == (8, 4, 5)
    assert len(dataset.sensory_noise_sha256_by_pair) == dataset.n_pairs


def test_speeded_distractor_dms_metadata_is_not_an_input_shortcut():
    dataset = _small_dataset()

    # All label-varying metadata are paired behind the dataset boundary. The
    # role and salience channels are exactly label-invariant before the probe.
    for pair_start in range(0, len(dataset), 2):
        probe = int(dataset.probe_onsets[pair_start])
        assert torch.equal(
            dataset.inputs[pair_start, :probe, 8:],
            dataset.inputs[pair_start + 1, :probe, 8:],
        )
        assert (
            dataset.sample_identities[pair_start]
            == dataset.sample_identities[pair_start + 1]
        )
        assert dataset.sensory_noise_realization_ids[pair_start] == (
            dataset.sensory_noise_realization_ids[pair_start + 1]
        )


def test_speeded_distractor_dms_builder_registers_both_encodings_and_splits():
    names = get_registered_sequence_dataset_names()
    assert "speeded_distractor_dms" in names
    assert "speeded_distractor_dms_positive" in names

    common = {
        "n_samples": 8,
        "delay_support": (96,),
        "n_distractors": 4,
        "sequence_duration": 140,
        "seed": 31,
        "valid": {"n_samples": 8, "seed": 41},
        "test": {"n_samples": 8, "seed": 51},
    }
    signed_splits = get_sequence_datasets("speeded_distractor_dms", **common)
    positive_splits = get_sequence_datasets("speeded_distractor_dms_positive", **common)

    assert [dataset.split_name for dataset in signed_splits] == [
        "train",
        "valid",
        "test",
    ]
    assert [dataset.seed for dataset in signed_splits] == [31, 41, 51]
    assert [dataset.input_dim for dataset in signed_splits] == [12, 12, 12]
    assert [dataset.input_dim for dataset in positive_splits] == [13, 13, 13]
    assert all(dataset.positive_input_encoding for dataset in positive_splits)
    assert torch.any(signed_splits[0].inputs[..., 8] < 0)
    assert all(torch.all(dataset.inputs >= 0) for dataset in positive_splits)
    assert not torch.equal(signed_splits[0].inputs, signed_splits[1].inputs)
    assert len({dataset.seed for dataset in signed_splits}) == 3
    assert len({dataset.content_sha256 for dataset in signed_splits}) == 3
    assert len({dataset.split_receipt_sha256 for dataset in signed_splits}) == 3


def test_speeded_distractor_dms_n_pairs_is_explicit_and_strict_in_builder():
    direct = _small_dataset(n_samples=None, n_pairs=7)
    assert direct.n_pairs == 7
    assert direct.n_samples == 14
    assert len(direct) == 14

    with pytest.raises(ValueError, match="exactly one"):
        _small_dataset(n_pairs=8)

    common = {
        "n_pairs": 4,
        "delay_support": (96,),
        "n_distractors": 4,
        "sequence_duration": 140,
        "seed": 101,
        "valid": {"n_pairs": 3, "seed": 103},
        "test": {"n_pairs": 5, "seed": 107},
    }
    splits = get_sequence_datasets("speeded_distractor_dms_positive", **common)
    assert [dataset.n_pairs for dataset in splits] == [4, 3, 5]
    assert [dataset.n_samples for dataset in splits] == [8, 6, 10]

    with pytest.raises(ValueError, match="exactly one"):
        get_sequence_datasets(
            "speeded_distractor_dms_positive",
            **{**common, "n_samples": 8},
        )
    with pytest.raises(ValueError, match="pairwise distinct"):
        get_sequence_datasets(
            "speeded_distractor_dms_positive",
            **{
                **common,
                "valid": {"n_pairs": 3, "seed": 101},
            },
        )
    with pytest.raises(ValueError, match="unsupported"):
        get_sequence_datasets(
            "speeded_distractor_dms_positive",
            **{**common, "ignored_scientific_parameter": 1},
        )


def test_standard_loader_plumbing_is_validated_but_does_not_change_content():
    common = {
        "n_pairs": 4,
        "delay_support": (96,),
        "n_distractors": 4,
        "sequence_duration": 140,
        "seed": 131,
        "valid": {"n_pairs": 3, "seed": 137},
        "test": {"n_pairs": 3, "seed": 139},
    }
    plain = get_sequence_datasets("speeded_distractor_dms_positive", **common)
    through_standard_loader = get_sequence_datasets(
        "speeded_distractor_dms_positive",
        **{
            **common,
            "flatten": False,
            "normalize": False,
            "label_noise_rate": 0.0,
            "label_noise_seed": 991,
            "split_seed": 997,
        },
    )

    assert [dataset.content_sha256 for dataset in through_standard_loader] == [
        dataset.content_sha256 for dataset in plain
    ]
    assert [dataset.split_receipt_sha256 for dataset in through_standard_loader] == [
        dataset.split_receipt_sha256 for dataset in plain
    ]


@pytest.mark.parametrize(
    ("field_name", "value", "error_type", "message"),
    (
        ("flatten", True, ValueError, "flatten=false"),
        ("normalize", True, ValueError, "normalize=false"),
        ("flatten", 0, TypeError, "flatten must be a bool"),
        ("normalize", 0, TypeError, "normalize must be a bool"),
        ("label_noise_rate", 0.1, ValueError, "label_noise_rate=0"),
        ("label_noise_rate", float("nan"), ValueError, "must be finite"),
        ("label_noise_seed", True, TypeError, "must be an integer"),
        ("label_noise_seed", 1.0, TypeError, "must be an integer"),
        ("split_seed", False, TypeError, "must be an integer"),
        ("split_seed", 1.0, TypeError, "must be an integer"),
    ),
)
def test_standard_loader_plumbing_fails_closed(
    field_name,
    value,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        get_sequence_datasets(
            "speeded_distractor_dms_positive",
            n_pairs=4,
            delay_support=(96,),
            n_distractors=4,
            sequence_duration=140,
            seed=151,
            valid={"n_pairs": 3, "seed": 157},
            test={"n_pairs": 3, "seed": 163},
            **{field_name: value},
        )


def test_unified_validate_only_setup_constructs_sd_dms_model_and_loss(tmp_path):
    """Exercise the same unified setup path used before validate-only exits."""

    config_path = tmp_path / "sd_dms_validate_only.yaml"
    payload = {
        "experiment": {
            "seed": 199,
            "dataset_seed": 211,
            # Generic split plumbing is intentionally independent of the
            # explicit SD-DMS train/valid/test content seeds.
            "split_seed": 223,
            "model_seed": 227,
            "topology_seed": 229,
            "loader_seed": 233,
            "evaluation_seed": 239,
            "probe_seed": 241,
            "train_valid_split": 1.0,
            "record_dataset_fingerprints": False,
        },
        "data": {
            "dataset_name": "speeded_distractor_dms_positive",
            "base_dir": "",
            "processing": {
                "flatten": False,
                "normalize": False,
                "label_noise_rate": 0.0,
            },
            "dataset_params": {
                "speeded_distractor_dms_positive": {
                    "n_pairs": 4,
                    "seed": 211,
                    "delay_support": [96],
                    "n_distractors": 4,
                    "lure_fraction": 0.25,
                    "salience_mu": 0.8,
                    "circular_noise_std": 0.04,
                    "sequence_duration": 140,
                    "positive_input_encoding": True,
                    "valid": {"n_pairs": 4, "seed": 251},
                    "test": {"n_pairs": 4, "seed": 257},
                }
            },
        },
        "model": {
            "task": "regression",
            "encoder": {
                "type": "identity",
                "params": {"input_dim": 13, "input_shape": None},
            },
            "core": {
                "type": "gru",
                "baseline_rnn": {
                    "hidden_dim": 8,
                    "num_layers": 1,
                    "output_mode": "all",
                },
            },
            "decoder": {
                "type": "MLP",
                "params": {"hidden_dims": [], "output_dim": 2},
            },
        },
        "training": {
            "main": {
                "strategy": "recurrent",
                "optimizer": {"name": "adam", "lr": 0.001},
                "common": {
                    "epochs": 1,
                    "batch_size": 8,
                    "loss_function": "masked_dual_task_temporal",
                    "num_workers": 0,
                },
            }
        },
        "analysis": {},
        "wandb": {"use_wandb": False},
        "outputs": {
            "run_name": "validate-only",
            "results_dir": str(tmp_path / "results"),
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    load_config.cache_clear()
    config = load_config(str(config_path))

    (
        _run_path,
        train_dataset,
        valid_dataset,
        test_dataset,
        model,
        _encoder,
    ) = setup_environment(
        model_config=config.model,
        training_config=config.training,
        data_config=config.data,
        wandb_config=config.wandb,
        outputs_config=config.outputs,
        experiment_config=config.experiment,
        is_main=False,
    )

    assert [
        dataset.n_pairs for dataset in (train_dataset, valid_dataset, test_dataset)
    ] == [
        4,
        4,
        4,
    ]
    assert [
        dataset.seed for dataset in (train_dataset, valid_dataset, test_dataset)
    ] == [
        211,
        251,
        257,
    ]
    assert model._is_recurrent_core

    configured_loss = get_loss_function(config.training.main.common.loss_function)
    assert isinstance(configured_loss, MaskedDualTaskTemporalLoss)
    logits = model(train_dataset.inputs, seq_lengths=train_dataset.seq_lengths)
    assert logits.shape == (8, 140, 2)
    loss = configured_loss.from_predictions(logits, train_dataset.targets)
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    (
        "regime_id",
        "delay_support",
        "n_distractors",
        "lure_fraction",
        "salience_mu",
        "noise_std",
    ),
    _FROZEN_REGIMES,
)
def test_all_frozen_regimes_realize_exact_counterbalance_contract(
    regime_id,
    delay_support,
    n_distractors,
    lure_fraction,
    salience_mu,
    noise_std,
):
    dataset = SpeededDistractorDMSDataset(
        n_pairs=20,
        delay_support=delay_support,
        n_distractors=n_distractors,
        lure_fraction=lure_fraction,
        salience_mu=salience_mu,
        circular_noise_std=noise_std,
        sequence_duration=360,
        positive_input_encoding=True,
        seed=104729 + int(regime_id[1:]),
        split_name=regime_id,
    )
    receipt = dataset.counterbalance_receipt
    total_events = dataset.n_pairs * n_distractors
    expected_lures = math.floor(lure_fraction * total_events + 0.5)

    assert receipt["n_pair_events"] == total_events
    assert receipt["target_events"] * 2 == total_events
    assert receipt["realized_lure_count"] == expected_lures
    assert receipt["realized_lure_fraction"] == expected_lures / total_events
    assert (
        receipt["lure_type_counts"]["exact"] + receipt["lure_type_counts"]["adjacent"]
        == expected_lures
    )
    assert (
        abs(
            receipt["lure_type_counts"]["exact"]
            - receipt["lure_type_counts"]["adjacent"]
        )
        <= 1
    )
    for subtype in ("non_lure", "exact", "adjacent"):
        count = receipt["lure_type_counts"][subtype]
        targets = receipt["lure_type_target_counts"][subtype]
        assert abs(targets - (count - targets)) <= 1
    assert len(receipt["per_trial_target_count_support"]) > 1
    assert receipt["maximum_factor_imbalance"] <= 1


@pytest.mark.parametrize("condition", SpeededDistractorDMSDataset.POINT_INTERVENTIONS)
def test_indexed_point_interventions_are_exact_and_isolated(condition):
    dataset = _small_dataset()
    original = dataset.inputs.clone()
    indices = torch.tensor([0, 3, 7, 11], dtype=torch.long)
    transformed_batch = dataset.intervention_transform_batch(indices, condition)

    for batch_index, sample_index_tensor in enumerate(indices):
        sample_index = int(sample_index_tensor)
        allowed = torch.zeros_like(dataset.inputs[sample_index], dtype=torch.bool)
        if condition == "initial_sample_cue_erased":
            allowed[dataset.SAMPLE_START : dataset.SAMPLE_END, :8] = True
        elif condition == "probe_identity_erased":
            probe_start = int(dataset.probe_onsets[sample_index])
            probe_end = probe_start + dataset.PROBE_DURATION
            allowed[probe_start:probe_end, :8] = True
        else:
            allowed[:, dataset.salience_slice] = True

        transformed = dataset.intervention_transform(sample_index, condition)
        assert torch.equal(transformed_batch[batch_index], transformed)
        assert torch.equal(transformed[~allowed], original[sample_index][~allowed])
        assert not bool(transformed[allowed].any())

    assert torch.equal(dataset.inputs, original)


def test_indexed_point_interventions_reject_ambiguous_requests():
    dataset = _small_dataset()
    with pytest.raises(ValueError, match="point intervention"):
        dataset.intervention_transform(0, "full")
    with pytest.raises(IndexError, match="outside"):
        dataset.intervention_transform(len(dataset), "initial_sample_cue_erased")
    with pytest.raises(TypeError, match="integer"):
        dataset.intervention_transform(True, "initial_sample_cue_erased")
    with pytest.raises(ValueError, match="one-dimensional"):
        dataset.intervention_transform_batch(
            torch.tensor([[0]]), "initial_sample_cue_erased"
        )


def test_identity_history_randomization_is_exhaustive_pair_shared_and_isolated():
    dataset = _small_dataset(n_samples=32)
    original = dataset.inputs.clone()
    assert dataset.input_intervention_receipt["exhaustive_per_pair"] is True
    assert torch.equal(
        dataset.sample_identity_rotation_order.sort(dim=1).values,
        torch.arange(dataset.N_IDENTITIES).expand(dataset.n_pairs, -1),
    )

    for pair_index in range(dataset.n_pairs):
        assert torch.equal(
            dataset.sample_identity_rotation_order[pair_index],
            dataset.sample_identity_rotation_order[
                int(dataset.pair_ids[2 * pair_index + 1])
            ],
        )
    indices = torch.tensor([0, 1, 6, 7], dtype=torch.long)
    for dose in range(dataset.N_IDENTITIES):
        batch = dataset.sample_identity_history_randomized_batch(indices, dose)
        for batch_index, sample_index_tensor in enumerate(indices):
            sample_index = int(sample_index_tensor)
            transformed = batch[batch_index]
            probe_start = int(dataset.probe_onsets[sample_index])
            probe_end = probe_start + dataset.PROBE_DURATION
            assert torch.equal(
                transformed[probe_start:probe_end],
                original[sample_index, probe_start:probe_end],
            )
            assert torch.equal(
                transformed[:, dataset.salience_slice],
                original[sample_index, :, dataset.salience_slice],
            )
            assert torch.equal(
                transformed[:, dataset.sample_role_channel :],
                original[sample_index, :, dataset.sample_role_channel :],
            )

    with pytest.raises(IndexError, match="outside"):
        dataset.sample_identity_history_randomized(0, dataset.N_IDENTITIES)
    without_seed = _small_dataset(intervention_seed=None)
    with pytest.raises(RuntimeError, match="intervention_seed"):
        without_seed.sample_identity_history_randomized(0, 0)


def test_independent_distractor_mode_is_balanced_and_rejects_lures():
    dataset = _small_dataset(
        n_samples=64,
        lure_fraction=0.0,
        distractor_identity_mode="independent_uniform",
    )
    counts = torch.bincount(
        dataset.distractor_identities[::2].reshape(-1),
        minlength=dataset.N_IDENTITIES,
    )
    assert int(counts.max() - counts.min()) <= 1
    assert dataset.counterbalance_receipt["distractor_identity_mode"] == (
        "independent_uniform"
    )
    assert dataset.counterbalance_receipt["distractor_identity_counts"] == (
        counts.tolist()
    )
    assert (
        dataset.input_intervention_receipt["initial_sample_cue_erasure"][
            "eligibility_gate"
        ]
        is True
    )

    with pytest.raises(ValueError, match="require lure_fraction=0"):
        _small_dataset(distractor_identity_mode="independent_uniform")


def test_intervention_seed_changes_only_the_bound_rotation_bank():
    first = _small_dataset(intervention_seed=271)
    changed = _small_dataset(intervention_seed=272)
    assert torch.equal(first.inputs, changed.inputs)
    assert torch.equal(first.targets, changed.targets)
    assert not torch.equal(
        first.sample_identity_rotation_order,
        changed.sample_identity_rotation_order,
    )
    assert first.input_intervention_receipt_sha256 != (
        changed.input_intervention_receipt_sha256
    )


def test_probe_identity_erasure_exactly_collapses_counterfactual_pairs():
    dataset = _small_dataset()
    for pair_index in range(dataset.n_pairs):
        first = dataset.intervention_transform(2 * pair_index, "probe_identity_erased")
        second = dataset.intervention_transform(
            2 * pair_index + 1, "probe_identity_erased"
        )
        assert torch.equal(first, second)


def test_speeded_distractor_dms_rejects_infeasible_or_unpaired_geometry():
    for parameters in (
        {"n_samples": 3},
        {"n_distractors": 5},
        {"delay_support": (64,)},
        {"sequence_duration": 100},
    ):
        try:
            _small_dataset(**parameters)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {parameters}")
