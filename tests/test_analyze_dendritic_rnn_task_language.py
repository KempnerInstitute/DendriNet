from __future__ import annotations

from dendritic_modeling.scripts.analysis import (
    analyze_dendritic_rnn_task_language as task,
)


def test_global_gate_uses_one_eight_member_holm_family() -> None:
    rows = {
        role: [{"C1": 0.4, "C2": 2.0, "C3": 0.1, "C4": 0.5} for _ in range(10)]
        for role in task.ROLE_CELLS
    }

    result = task.analyze_components(
        rows,
        analysis_seed=7,
        bootstrap_draws=100,
        family_alpha=0.05,
    )

    assert len(result["holm_8_family"]["members"]) == 8
    for role in task.ROLE_CELLS:
        assert (
            result["roles"][role]["language_decisions"]["both_stimulus_identities"][
                "authorized"
            ]
            is True
        )
        assert (
            result["roles"][role]["language_decisions"]["two_stimulus_comparison"][
                "authorized"
            ]
            is True
        )
        assert result["roles"][role]["C4_role"].startswith("out_of_distribution")


def test_failed_c3_blocks_only_comparison_language() -> None:
    rows = {
        role: [{"C1": 0.4, "C2": 2.0, "C3": -0.1, "C4": 0.5} for _ in range(10)]
        for role in task.ROLE_CELLS
    }

    result = task.analyze_components(
        rows,
        analysis_seed=7,
        bootstrap_draws=100,
        family_alpha=0.05,
    )

    for role in task.ROLE_CELLS:
        assert (
            result["roles"][role]["language_decisions"]["both_stimulus_identities"][
                "authorized"
            ]
            is True
        )
        assert (
            result["roles"][role]["language_decisions"]["two_stimulus_comparison"][
                "authorized"
            ]
            is False
        )
