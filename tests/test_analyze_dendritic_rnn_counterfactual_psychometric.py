from __future__ import annotations

import math

import numpy as np

from dendritic_modeling.scripts.analysis import (
    analyze_dendritic_rnn_counterfactual_psychometric as analysis,
)


def _synthetic_run(offset: float = 0.0) -> dict:
    surface = [
        {
            "receiver_first_rank": first,
            "target_second_rank": second,
            "support_class": "synthetic",
            "n_rows": 512,
            "mean_class1_probability": 0.1 * second,
        }
        for first in (2, 3)
        for second in (1, 2, 3, 4)
    ]
    return {
        "mean_joint_supported_central_difference": 1.0 + offset,
        "joint_supported_unequal_accuracy": 1.0,
        "full_unequal_accuracy": 1.0,
        "psychometric_surface": surface,
    }


def test_run_summary_uses_frozen_supported_rank_pairs() -> None:
    receiver = np.repeat(np.arange(1024), 4)
    first_by_receiver = np.repeat([2, 3], 512)
    first = np.repeat(first_by_receiver, 4)
    second = np.tile(np.arange(1, 5), 1024)
    logits = np.zeros((4096, 2), dtype=np.float32)
    logits[:, 1] = second.astype(np.float32) * 2.0
    arrays = {
        "receiver_first_ranks": first,
        "target_second_ranks": second,
        "targets": (second > first).astype(np.int64),
        "comparison_defined": second != first,
        "receiver_indices": receiver,
    }

    result = analysis.summarize_run(logits, arrays)

    assert result["receiver_count"] == 1024
    assert result["mean_joint_supported_central_difference"] == 2.0
    assert result["fraction_receivers_positive_c2"] == 1.0


def test_seed_analysis_treats_ten_models_as_units() -> None:
    summaries = {
        role: [_synthetic_run(0.01 * seed) for seed in range(10)]
        for role in analysis.ROLE_CELLS
    }

    result = analysis.analyze_seed_summaries(
        summaries,
        analysis_seed=7,
        bootstrap_draws=100,
    )

    assert result["analysis_plan"]["n_independent_units_per_cell"] == 10
    for row in result["roles"].values():
        assert math.isclose(row["mean"], 1.045)
        assert row["exact_two_sided_sign_flip"]["p_value"] == 2 / 1024
        assert row["support_gate"]["unadjusted_positive_support"] is True
        assert row["support_gate"]["task_language_authorized_by_c2_alone"] is False
    assert math.isclose(result["descriptive_c2_crossover"]["mean"], 0.0)
