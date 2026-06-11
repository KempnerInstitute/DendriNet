"""Smoke tests for multiplicative-gain sweep plotting."""

from pathlib import Path

import pandas as pd

from dendritic_modeling.scripts.sweeps.plotters.multiplicative_gain_plotter import (
    MultiplicativeGainPlotter,
)


def test_multiplicative_gain_plotter_handles_generalization_scores(tmp_path: Path):
    data = pd.DataFrame(
        {
            "network_category": ["dendritic_shunting", "dendritic_additive"],
            "ei_ratio": [1.0, 2.0],
            "mult_gain_accuracy_gf1.0_mean": [0.8, 0.7],
            "mult_gain_accuracy_gf2.0_mean": [0.75, 0.65],
            "mult_gain_accuracy_id_mean": [0.8, 0.7],
            "mult_gain_accuracy_ood_lower_mean": [0.6, 0.5],
            "mult_gain_accuracy_ood_upper_mean": [0.7, 0.6],
        }
    )

    paths = MultiplicativeGainPlotter().generate_all(data, tmp_path)
    names = {path.name for path in paths}

    assert "gain_accuracy_vs_ei_ratio_generalization_scores.png" in names
