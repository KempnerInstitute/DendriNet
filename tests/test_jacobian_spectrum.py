"""Unit tests for the Jacobian / recurrent spectrum analyzer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from dendritic_modeling.analysis.tools.jacobian_spectrum import JacobianSpectrumAnalyzer
from dendritic_modeling.config.analysis import JacobianSpectrumAnalysisParams


class _FakeTopK(nn.Module):
    def __init__(self, w: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self._w = w
        self._mask = torch.ones_like(w) if mask is None else mask

    def weight(self) -> torch.Tensor:
        return self._w

    def weight_mask(self) -> torch.Tensor:
        return self._mask


class _FakeBranchLayer(nn.Module):
    def __init__(
        self,
        rec: _FakeTopK | None = None,
        inh: _FakeTopK | None = None,
    ) -> None:
        super().__init__()
        self.branch_recurrent = rec
        self.branch_rec_inhibition = inh


class _FakePopulation(nn.Module):
    def __init__(
        self,
        branch_layers: list[_FakeBranchLayer],
        log_tau: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.branch_layers = nn.ModuleList(branch_layers)
        if log_tau is not None:
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


class TestJacobianSpectrumAnalyzer(unittest.TestCase):
    def test_diagonal_weight_identity_spectrum(self) -> None:
        """Diagonal weight with entries 3,2,1 gives singular values [3, 2, 1]."""
        w_rec = torch.diag(torch.tensor([3.0, 2.0, 1.0]))
        layer = _FakeLayer(
            _FakePopulation(
                [_FakeBranchLayer(rec=_FakeTopK(w_rec), inh=None)],
                log_tau=torch.log(torch.tensor([5.0])),
            )
        )
        model = _FakeModel(_FakeCore([layer]))
        analyzer = JacobianSpectrumAnalyzer(
            JacobianSpectrumAnalysisParams(top_k=3, save_figures=False)
        )
        results = analyzer.analyze(model)
        entry = results["layers"]["layer_0"]["levels"][0]

        self.assertEqual(results["top_k"], 3)
        self.assertFalse(results["is_state_transition_jacobian"])
        self.assertEqual(set(results["layers"].keys()), {"layer_0"})
        self.assertEqual(entry["level"], 0)
        self.assertAlmostEqual(entry["tau"], 5.0, places=4)
        self.assertAlmostEqual(entry["sigma_rec"][0], 3.0, places=4)
        self.assertAlmostEqual(entry["sigma_rec"][1], 2.0, places=4)
        self.assertAlmostEqual(entry["sigma_rec"][2], 1.0, places=4)
        self.assertAlmostEqual(entry["spectral_radius_rec"], 3.0, places=4)
        self.assertNotIn("sigma_inh", entry)

    def test_inactive_candidate_weights_do_not_enter_spectrum(self) -> None:
        candidate = torch.diag(torch.tensor([1000.0, 2.0, 1.0]))
        mask = torch.diag(torch.tensor([0.0, 1.0, 1.0]))
        layer = _FakeLayer(
            _FakePopulation(
                [_FakeBranchLayer(rec=_FakeTopK(candidate, mask))],
                log_tau=torch.log(torch.tensor([5.0])),
            )
        )

        results = JacobianSpectrumAnalyzer(
            JacobianSpectrumAnalysisParams(top_k=3, save_figures=False)
        ).analyze(_FakeModel(_FakeCore([layer])))

        spectrum = results["layers"]["layer_0"]["levels"][0]["sigma_rec"]
        self.assertEqual(spectrum, [2.0, 1.0, 0.0])

    def test_multiple_levels_both_rec_and_inh(self) -> None:
        """Both rec and inh present at each level; per-level entries populated."""
        w_rec_0 = torch.eye(3) * 2.0
        w_rec_1 = torch.eye(3) * 1.0
        w_inh_0 = torch.eye(3) * 0.5
        w_inh_1 = torch.eye(3) * 0.3
        branch_layers = [
            _FakeBranchLayer(rec=_FakeTopK(w_rec_0), inh=_FakeTopK(w_inh_0)),
            _FakeBranchLayer(rec=_FakeTopK(w_rec_1), inh=_FakeTopK(w_inh_1)),
        ]
        layer = _FakeLayer(
            _FakePopulation(branch_layers, log_tau=torch.log(torch.tensor([10.0, 2.0])))
        )
        model = _FakeModel(_FakeCore([layer]))
        analyzer = JacobianSpectrumAnalyzer(
            JacobianSpectrumAnalysisParams(save_figures=False)
        )
        results = analyzer.analyze(model)
        levels = results["layers"]["layer_0"]["levels"]

        self.assertEqual(len(levels), 2)
        self.assertAlmostEqual(levels[0]["spectral_radius_rec"], 2.0, places=4)
        self.assertAlmostEqual(levels[0]["spectral_radius_inh"], 0.5, places=4)
        self.assertAlmostEqual(levels[1]["tau"], 2.0, places=4)
        self.assertAlmostEqual(levels[1]["spectral_radius_rec"], 1.0, places=4)

    def test_baseline_rnn_returns_empty(self) -> None:
        class _Core(nn.Module):
            pass

        self.assertEqual(JacobianSpectrumAnalyzer().analyze(_FakeModel(_Core())), {})

    def test_saves_json_and_pdf(self) -> None:
        w = torch.randn(4, 4)
        layer = _FakeLayer(
            _FakePopulation(
                [_FakeBranchLayer(rec=_FakeTopK(w), inh=_FakeTopK(w * 0.5))],
                log_tau=torch.log(torch.tensor([3.0])),
            )
        )
        model = _FakeModel(_FakeCore([layer]))
        with tempfile.TemporaryDirectory() as td:
            JacobianSpectrumAnalyzer().analyze(model, save_path=td, filename="final")
            files = {p.name for p in Path(td).iterdir()}
            self.assertIn("final.json", files)
            self.assertIn("final.pdf", files)


if __name__ == "__main__":
    unittest.main()
