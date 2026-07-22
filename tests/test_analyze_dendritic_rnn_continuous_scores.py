from __future__ import annotations

import math
from pathlib import Path

from dendritic_modeling.scripts.analysis import (
    analyze_dendritic_rnn_continuous_scores as continuous,
)


def _synthetic_scores() -> dict:
    scores = {}
    for replicate in range(continuous.N_REPLICATES):
        for cell_index, cell in enumerate(continuous.PRIMARY_CROSSOVER_CELLS):
            scores[replicate, cell] = {
                "accuracy": 0.70 + 0.001 * replicate + 0.01 * cell_index,
                "target_log_score": -0.4 + 0.01 * replicate + 0.02 * cell_index,
                "target_margin": 2.0 + 0.1 * replicate + 0.3 * cell_index,
            }
    return scores


def test_prediction_reduction_uses_target_oriented_binary_margin() -> None:
    payload = {
        "n_trials": 2,
        "predictions": [
            {
                "target": 0,
                "prediction": 0,
                "correct": True,
                "logits": [2.0, -1.0],
            },
            {
                "target": 1,
                "prediction": 0,
                "correct": False,
                "logits": [0.5, -0.5],
            },
        ],
    }

    summary = continuous.summarize_predictions(payload)

    expected_log_score = (-math.log1p(math.exp(-3.0)) - math.log1p(math.exp(1.0))) / 2
    assert summary["correct_count"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["target_margin"] == 1.0
    assert math.isclose(summary["target_log_score"], expected_log_score)


def test_analysis_forms_same_four_cell_contrast_without_a_null_test() -> None:
    result = continuous.analyze_scores(
        _synthetic_scores(),
        analysis_seed=11,
        bootstrap_draws=100,
    )

    assert result["analysis_plan"]["n_independent_units"] == 10
    assert result["analysis_plan"]["null_tests"] == []
    assert (
        result["analysis_plan"]["claim_role"] == "descriptive_non_rescuing_diagnostic"
    )
    assert math.isclose(result["metrics"]["accuracy"]["mean"], 0.0, abs_tol=1e-15)
    assert math.isclose(
        result["metrics"]["target_log_score"]["mean"], 0.0, abs_tol=1e-15
    )
    assert math.isclose(result["metrics"]["target_margin"]["mean"], 0.0, abs_tol=1e-15)


def test_renderer_writes_pdf_and_png(tmp_path) -> None:
    result = continuous.analyze_scores(
        _synthetic_scores(),
        analysis_seed=11,
        bootstrap_draws=20,
    )

    paths = continuous.render_figure(result, tmp_path)

    assert set(paths) == {"pdf", "png"}
    assert all(
        tmp_path.joinpath(Path(path).name).stat().st_size > 0 for path in paths.values()
    )
