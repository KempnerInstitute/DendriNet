from __future__ import annotations

import pytest
import torch

from dendritic_modeling.datasets.sequence.neuro_tasks import RomoDelayComparisonDataset
from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_mechanism_pilot import (
    EXPECTED_DELAY_STRATA,
    FAST_DIST,
    FAST_PROX,
    METRICS,
    SLOW_DIST,
    SLOW_PROX,
    _load_analysis_contract,
    _summarize_delay_strata,
    _summarize_metric,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_mechanism_pilot import (
    DEFAULT_CONFIG,
    MechanismPilotEvaluationError,
    _pair_audit,
    load_config,
)


def test_mechanism_pilot_evaluation_contract_loads() -> None:
    config, path = load_config(DEFAULT_CONFIG)
    assert path == DEFAULT_CONFIG.resolve()
    assert config["bank"]["n_trials"] == 4096
    assert config["bank"]["min_delay"] == 240
    assert config["resources"]["checkpoints_per_job"] == 4


def test_mechanism_pilot_analysis_contract_is_hash_bound() -> None:
    contract = _load_analysis_contract()
    assert contract["frozen_before_any_training_complete"] is True
    assert contract["delay_strata_in_aggregate_gate"] is False


def test_pair_audit_detects_post_stimulus_mismatch() -> None:
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(91)
        dataset = RomoDelayComparisonDataset(
            n_samples=8,
            frequency_values=(2.0, 2.8, 3.6, 4.4, 5.2, 6.0),
            stimulus_duration=20,
            response_duration=8,
            noise_std=0.15,
            positive_input_encoding=True,
            tuning_dim=8,
            pair_sampling="balanced_second_stimulus_v1",
            min_delay=240,
            max_delay=320,
        )
    _pair_audit(dataset)
    pair_id = int(dataset.pair_ids[0])
    mate = int((dataset.pair_ids == pair_id).nonzero(as_tuple=False)[1].item())
    dataset.inputs[mate, dataset.stimulus_duration, 0] += 1.0
    with pytest.raises(MechanismPilotEvaluationError, match="differ after delay onset"):
        _pair_audit(dataset)


def test_primary_contrast_is_paired_difference_of_differences() -> None:
    nested = {}
    for seed_index, seed in enumerate(("r000", "r001", "r002")):
        offset = 0.01 * seed_index
        nested[seed] = {
            SLOW_PROX: dict.fromkeys(METRICS, 0.60 + offset),
            SLOW_DIST: dict.fromkeys(METRICS, 0.75 + offset),
            FAST_PROX: dict.fromkeys(METRICS, 0.65 + offset),
            FAST_DIST: dict.fromkeys(METRICS, 0.67 + offset),
        }
    summary = _summarize_metric(nested, "accuracy")
    assert summary["primary_contrast_seed_values"] == pytest.approx([0.13] * 3)
    assert summary["primary_contrast_mean"] == pytest.approx(0.13)
    assert summary["primary_contrast_positive_seed_count"] == 3


def test_delay_strata_preserve_paired_interaction() -> None:
    rows = []
    for seed_index, seed in enumerate(("r000", "r001", "r002")):
        for condition, base in (
            (SLOW_PROX, 0.60),
            (SLOW_DIST, 0.70),
            (FAST_PROX, 0.62),
            (FAST_DIST, 0.64),
        ):
            rows.append(
                {
                    "pairing_group": seed,
                    "condition": condition,
                    "delay_strata": [
                        {
                            "delay_low": low,
                            "delay_high": high,
                            "n_trials": 100 + stratum_index,
                            **dict.fromkeys(
                                METRICS,
                                base + 0.01 * seed_index + 0.02 * stratum_index,
                            ),
                        }
                        for stratum_index, (low, high) in enumerate(
                            EXPECTED_DELAY_STRATA
                        )
                    ],
                }
            )
    summary = _summarize_delay_strata(rows)
    assert [row["primary_contrast_mean"] for row in summary["strata"]] == pytest.approx(
        [0.08] * 4
    )
    assert summary["longest_minus_shortest_mean"] == pytest.approx(0.0)
