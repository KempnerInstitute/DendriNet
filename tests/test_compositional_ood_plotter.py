"""Smoke tests for compositional-OOD sweep plotting."""

from pathlib import Path

import pandas as pd

from dendritic_modeling.scripts.sweeps.plotters.compositional_ood_plotter import (
    CompositionalOODPlotter,
)


def test_compositional_ood_plotter_writes_1d_stimulus_duration_plot(tmp_path: Path):
    data = pd.DataFrame(
        {
            "network_category": [
                "dendritic_shunting",
                "dendritic_shunting",
                "dendritic_additive",
                "dendritic_additive",
            ],
            "stimulus_duration": [1.0, 2.0, 1.0, 2.0],
            "max_gain_factor": [1.0, 1.0, 1.0, 1.0],
            "comp_ood_accuracy_mean": [0.8, 0.7, 0.75, 0.65],
        }
    )

    paths = CompositionalOODPlotter().generate_all(data, tmp_path)
    names = {path.name for path in paths}

    assert "comp_ood_accuracy_vs_stimulus_duration.png" in names
    assert (
        tmp_path / "compositional_ood" / "comp_ood_accuracy_vs_stimulus_duration.png"
    ).exists()


def test_compositional_ood_plotter_skips_when_columns_missing(tmp_path: Path):
    data = pd.DataFrame(
        {
            "network_category": ["dendritic_shunting", "dendritic_additive"],
            "stimulus_duration": [1.0, 2.0],
            "max_gain_factor": [1.0, 1.0],
        }
    )

    paths = CompositionalOODPlotter().generate_all(data, tmp_path)

    assert paths == []
    assert not (tmp_path / "compositional_ood").exists()
