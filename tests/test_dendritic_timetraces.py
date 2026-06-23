"""Unit tests for the dendritic-time-trace simulator."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from dendritic_modeling.analysis.tools.dendritic_timetraces import (
    DendriticTimetracesAnalyzer,
)
from dendritic_modeling.config.analysis import DendriticTimetracesAnalysisParams


class _FakeReactivation(nn.Module):
    def __init__(self, m: float, b: float, n_units: int = 4) -> None:
        super().__init__()
        self.log_m = nn.Parameter(torch.full((n_units,), math.log(m)))
        self.b = nn.Parameter(torch.full((n_units,), b))


class _FakeBranchLayer(nn.Module):
    def __init__(self, reactivation: _FakeReactivation) -> None:
        super().__init__()
        self.reactivation = reactivation


class _FakePopulation(nn.Module):
    def __init__(
        self,
        branch_layers: list[_FakeBranchLayer],
        log_tau: torch.Tensor,
    ) -> None:
        super().__init__()
        self.branch_layers = nn.ModuleList(branch_layers)
        self.register_buffer("log_tau", log_tau)


class _FakeLayer(nn.Module):
    def __init__(self, e_population: nn.Module) -> None:
        super().__init__()
        self.e_population = e_population
        self.i_population = None


class _FakeCore(nn.Module):
    def __init__(self, layers: list[_FakeLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _FakeModel(nn.Module):
    def __init__(self, core: _FakeCore) -> None:
        super().__init__()
        self.core_network = core


def _build_model(tau: list[float], m_b: list[tuple[float, float]]) -> _FakeModel:
    log_tau = torch.log(torch.tensor(tau, dtype=torch.float32))
    branch_layers = [_FakeBranchLayer(_FakeReactivation(m, b)) for m, b in m_b]
    return _FakeModel(_FakeCore([_FakeLayer(_FakePopulation(branch_layers, log_tau))]))


class TestDendriticTimetracesAnalyzer(unittest.TestCase):
    def test_shapes_and_bounds(self) -> None:
        model = _build_model(tau=[20.0, 5.0, 1.5], m_b=[(1.0, 0.0)] * 3)
        params = DendriticTimetracesAnalysisParams(
            n_timesteps=40,
            stim1_window=(2, 10),
            stim2_window=(20, 30),
            response_window=(30, 40),
            noise_std=0.0,
            save_figures=False,
        )
        results = DendriticTimetracesAnalyzer(params).analyze(model)
        entry = results["layers"]["layer_0"]

        self.assertEqual(results["stim1_window"], [2, 10])
        self.assertEqual(results["stim2_window"], [20, 30])
        self.assertEqual(results["response_window"], [30, 40])
        self.assertEqual(entry["population"], "excitatory")
        self.assertEqual(entry["n_levels"], 3)
        self.assertEqual(entry["n_timesteps"], 40)
        self.assertEqual(len(entry["V"]), 3)
        self.assertEqual(len(entry["V"][0]), 40)
        # hat V is 0.5 * (1 + tanh(...)) so strictly in [0, 1]
        for lvl in range(3):
            for t in range(40):
                self.assertGreaterEqual(entry["V_hat"][lvl][t], 0.0)
                self.assertLessEqual(entry["V_hat"][lvl][t], 1.0)

    def test_constant_input_converges_to_input(self) -> None:
        """With constant u=1 and enough time, V_l → 1 at every level.

        The cascade is a low-pass with gain 1 at DC, so the fixed point
        is V_l = u for all l regardless of τ.
        """
        model = _build_model(tau=[3.0, 3.0], m_b=[(1.0, 0.0), (1.0, 0.0)])
        params = DendriticTimetracesAnalysisParams(
            n_timesteps=200,
            stim1_window=(0, 200),
            stim1_amp=1.0,
            stim2_window=None,
            response_window=None,
            noise_std=0.0,
            save_figures=False,
        )
        results = DendriticTimetracesAnalyzer(params).analyze(model)
        entry = results["layers"]["layer_0"]
        # By t=199 both levels should be very close to 1
        self.assertAlmostEqual(entry["V"][0][-1], 1.0, places=3)
        self.assertAlmostEqual(entry["V"][1][-1], 1.0, places=3)

    def test_deterministic_under_seed(self) -> None:
        """With noise_std > 0, fixing the seed reproduces the same trace."""
        model = _build_model(tau=[5.0, 5.0], m_b=[(1.0, 0.0), (1.0, 0.0)])
        params = DendriticTimetracesAnalysisParams(
            n_timesteps=30, noise_std=0.1, seed=7, save_figures=False
        )
        a = DendriticTimetracesAnalyzer(params).analyze(model)
        b = DendriticTimetracesAnalyzer(params).analyze(model)
        self.assertEqual(a["layers"]["layer_0"]["u"], b["layers"]["layer_0"]["u"])

    def test_baseline_rnn_returns_empty(self) -> None:
        class _Core(nn.Module):
            pass

        self.assertEqual(DendriticTimetracesAnalyzer().analyze(_FakeModel(_Core())), {})

    def test_saves_json_and_pdf(self) -> None:
        model = _build_model(tau=[5.0, 2.0], m_b=[(1.0, 0.0), (1.0, 0.0)])
        with tempfile.TemporaryDirectory() as td:
            DendriticTimetracesAnalyzer().analyze(model, save_path=td, filename="final")
            files = {p.name for p in Path(td).iterdir()}
            self.assertIn("final.json", files)
            self.assertIn("final_layer_0.pdf", files)


if __name__ == "__main__":
    unittest.main()
