from __future__ import annotations

import math

from dendritic_modeling.scripts.analysis import (
    analyze_dendritic_rnn_extreme_cube as cube,
)


def _synthetic_collection() -> dict:
    rows = []
    for replicate in range(cube.N_REPLICATES):
        for tau, ff, rec in cube.CELL_ORDER:
            source = (
                "extension" if (tau, ff, rec) in cube.EXTENSION_CELLS else "primary"
            )
            tau_sign = 1.0 if tau == "hierarchical" else -1.0
            ff_sign = 1.0 if ff == "proximal" else -1.0
            rec_sign = 1.0 if rec == "proximal" else -1.0
            accuracy = (
                0.60
                + 0.001 * replicate
                + 0.01 * tau_sign * ff_sign
                + 0.02 * tau_sign * rec_sign
                + 0.003 * tau_sign * ff_sign * rec_sign
            )
            rows.append(
                {
                    "replicate": f"r{replicate:03d}",
                    "tau": tau,
                    "ff": ff,
                    "rec": rec,
                    "source": source,
                    "accuracy": accuracy,
                    "training_partition": (
                        cube.PRIORITY_PARTITION
                        if source == "extension" or replicate >= 8
                        else cube.BASE_PARTITION
                    ),
                }
            )
    return {
        "schema_version": cube.COLLECTION_SCHEMA,
        "design": {
            "route_audit": {
                "status": "PASSED",
                "observed_counts": {
                    cube.BASE_PARTITION: 48,
                    cube.PRIORITY_PARTITION: 32,
                },
            }
        },
        "runs": rows,
    }


def test_contrast_coefficients_have_expected_orthogonal_norms() -> None:
    coefficients = {
        name: [cube.contrast_coefficients(cell)[name] for cell in cube.CELL_ORDER]
        for name in cube.CONTRAST_ORDER
    }

    assert sum(coefficients["ff_by_tau"]) == 0.0
    assert sum(coefficients["rec_by_tau"]) == 0.0
    assert sum(coefficients["ff_by_rec_by_tau"]) == 0.0
    assert sum(value * value for value in coefficients["ff_by_tau"]) == 2.0
    assert sum(value * value for value in coefficients["rec_by_tau"]) == 2.0
    assert sum(value * value for value in coefficients["ff_by_rec_by_tau"]) == 8.0


def test_exact_sign_flip_enumerates_all_assignments() -> None:
    result = cube.exact_two_sided_sign_flip([1.0, 1.0, 1.0])

    assert result["assignment_count"] == 8
    assert result["extreme_assignment_count"] == 2
    assert result["p_value"] == 0.25


def test_holm_adjustment_is_step_down_and_monotone() -> None:
    result = cube.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.05})

    assert result["a"]["adjusted_p_value"] == 0.03
    assert result["b"]["adjusted_p_value"] == 0.08
    assert result["c"]["adjusted_p_value"] == 0.08
    assert result["a"]["reject_at_family_alpha"] is True
    assert result["b"]["reject_at_family_alpha"] is False


def test_analysis_uses_ten_seed_units_and_passes_algebraic_bridge() -> None:
    result = cube.analyze_collection(
        _synthetic_collection(),
        analysis_seed=17,
        bootstrap_draws=200,
    )

    assert result["analysis_plan"]["independent_unit"] == "paired_trained_model_seed"
    assert result["analysis_plan"]["n_independent_units"] == 10
    assert result["algebraic_bridge"]["status"] == "PASSED"
    assert (
        result["algebraic_bridge"]["max_absolute_residual"]
        <= result["algebraic_bridge"]["tolerance"]
    )
    assert math.isclose(result["contrasts"]["ff_by_tau"]["mean"], 0.04)
    assert math.isclose(result["contrasts"]["rec_by_tau"]["mean"], 0.08)
    assert math.isclose(result["contrasts"]["ff_by_rec_by_tau"]["mean"], 0.024)


def test_renderer_writes_pdf_and_png(tmp_path) -> None:
    statistics = cube.analyze_collection(
        _synthetic_collection(),
        analysis_seed=17,
        bootstrap_draws=20,
    )

    paths = cube.render_figure(statistics, tmp_path)

    assert set(paths) == {"pdf", "png"}
    assert all(
        tmp_path.joinpath(path.split("/")[-1]).stat().st_size > 0
        for path in paths.values()
    )
