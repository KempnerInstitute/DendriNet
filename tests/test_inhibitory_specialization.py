"""Unit tests for the inhibitory-specialization analyzer.

We avoid loading a full trained checkpoint by constructing lightweight
fakes that mimic the minimum contract the analyzer reads from a
recurrent E-I model:

    model.core_network.layers[idx].e_population.branch_layers[l].branch_rec_inhibition.weight()
    model.core_network.layers[idx].e_population.branch_layers[l].branch_rec_inhibition.weight_mask()
    model.core_network.layers[idx].e_population.n_soma
    model.core_network.layers[idx].i_population.n_soma

Each test constructs a hand-crafted weight profile so we can check the
gate-index computation, per-I classification, and per-pair dominance
percentages against known ground-truth values.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from dendritic_modeling.analysis.tools.inhibitory_specialization import (
    InhibitorySpecializationAnalyzer,
    extract_recurrent_inhibitory_compartment_weights_from_state_dict,
    pair_stats_by_class,
    summarize_inhibitory_specialization_from_compartment_weights,
)
from dendritic_modeling.config.analysis import InhibitorySpecializationAnalysisParams


class _FakeTopK(nn.Module):
    """Minimal deterministic Top-K stand-in with candidate and active views."""

    def __init__(self, w: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self._w = w
        self._mask = torch.ones_like(w) if mask is None else mask

    def weight(self) -> torch.Tensor:
        return self._w

    def weight_mask(self) -> torch.Tensor:
        return self._mask


class _FakeBranchLayer(nn.Module):
    def __init__(self, rec_inh: _FakeTopK | None) -> None:
        super().__init__()
        self.branch_rec_inhibition = rec_inh


class _FakePopulation(nn.Module):
    def __init__(self, n_soma: int, branch_layers: list[_FakeBranchLayer]) -> None:
        super().__init__()
        self.n_soma = n_soma
        self.branch_layers = nn.ModuleList(branch_layers)


class _FakeLayer(nn.Module):
    def __init__(self, e_population: nn.Module, i_population: nn.Module) -> None:
        super().__init__()
        self.e_population = e_population
        self.i_population = i_population


class _FakeCore(nn.Module):
    def __init__(self, layers: list[_FakeLayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _FakeModel(nn.Module):
    def __init__(self, core: _FakeCore) -> None:
        super().__init__()
        self.core_network = core


def _make_layer(
    n_E: int,
    n_I: int,
    branch_factors: list[int],
    level_weights: list[torch.Tensor],
) -> _FakeLayer:
    """Build a fake E-I layer with per-level inhibition weights.

    ``level_weights[l]`` has shape ``(n_E * branch_factors[l], n_I)``.
    """
    assert len(branch_factors) == len(level_weights)
    branch_layers: list[_FakeBranchLayer] = []
    for f, w in zip(branch_factors, level_weights):
        assert w.shape == (n_E * f, n_I)
        branch_layers.append(_FakeBranchLayer(_FakeTopK(w)))
    e_pop = _FakePopulation(n_soma=n_E, branch_layers=branch_layers)
    i_pop = _FakePopulation(n_soma=n_I, branch_layers=[])
    return _FakeLayer(e_pop, i_pop)


class TestInhibitorySpecializationAnalyzer(unittest.TestCase):
    def test_state_dict_helper_matches_expected_gate_index(self) -> None:
        """Shared checkpoint helper should reproduce the same G algebra."""
        n_E, n_I = 4, 2
        sd = {
            "core.branch_layers.0.branch_rec_inhibition.pre_w": torch.log(
                torch.ones(n_E * 2, n_I) * 0.25
            ),
            "core.branch_layers.1.branch_rec_inhibition.pre_w": torch.log(
                torch.ones(n_E * 1, n_I) * 1.0
            ),
        }
        comp = extract_recurrent_inhibitory_compartment_weights_from_state_dict(
            sd,
            prefix="core",
            n_E=n_E,
            weight_transform="exp",
            active_masks=[torch.ones(n_E * 2, n_I), torch.ones(n_E, n_I)],
        )
        summary = summarize_inhibitory_specialization_from_compartment_weights(comp)
        G = summary["G"]
        # Realized mass sums over the two distal compartments: distal=0.5,
        # soma=1.0, hence G = 2/3 - 1/3 = 1/3.
        self.assertTrue(torch.allclose(G, torch.full_like(G, 1 / 3), atol=1e-6))
        self.assertEqual(summary["class_counts"]["gate-specialist"], n_I)

    def test_state_dict_helper_rejects_dense_candidates_without_masks(self) -> None:
        sd = {"core.branch_layers.0.branch_rec_inhibition.pre_w": torch.zeros(2, 2)}
        with self.assertRaisesRegex(ValueError, "active_masks are required"):
            extract_recurrent_inhibitory_compartment_weights_from_state_dict(
                sd, prefix="core", n_E=1
            )

    def test_inactive_large_weights_do_not_change_specialization(self) -> None:
        n_E, n_I = 2, 1
        distal_candidates = torch.tensor([[1000.0], [1000.0], [1000.0], [1000.0]])
        distal_mask = torch.zeros_like(distal_candidates)
        soma_candidates = torch.ones(n_E, n_I)
        layer = _FakeLayer(
            _FakePopulation(
                n_soma=n_E,
                branch_layers=[
                    _FakeBranchLayer(_FakeTopK(distal_candidates, distal_mask)),
                    _FakeBranchLayer(_FakeTopK(soma_candidates)),
                ],
            ),
            _FakePopulation(n_soma=n_I, branch_layers=[]),
        )
        results = InhibitorySpecializationAnalyzer().analyze(
            _FakeModel(_FakeCore([layer])), device="cpu"
        )
        entry = results["layers"]["layer_0"]
        self.assertEqual(entry["class_counts"]["gate-specialist"], 1)
        self.assertEqual(entry["pair_level"]["strong_soma_pct"], 100.0)

    def test_disconnected_pairs_are_not_counted_as_balanced(self) -> None:
        weights = [
            torch.tensor([[[0.0], [0.0]], [[0.0], [0.0]]]),
            torch.tensor([[[1.0]], [[0.0]]]),
        ]
        summary = summarize_inhibitory_specialization_from_compartment_weights(weights)
        self.assertEqual(summary["pair_level"]["n_pairs"], 1)
        self.assertEqual(summary["pair_level"]["n_disconnected_pairs"], 1)
        self.assertEqual(summary["pair_level"]["balanced_pct"], 0.0)

    def test_pair_stats_by_class_reports_bimodality(self) -> None:
        """Mixed-conditional rows should show non-zero bimodality."""
        G = torch.tensor(
            [
                [1.0, 1.0, -1.0, -1.0],  # mixed-conditional
                [-0.4, -0.3, -0.5, -0.2],  # context-specialist
            ]
        )
        classes = ["mixed-conditional", "context-specialist"]
        stats = pair_stats_by_class(G, classes, hi=0.1)
        self.assertGreater(stats["mixed-conditional"]["bimodality"], 0.9)
        self.assertEqual(stats["context-specialist"]["strong_distal_pct"], 100.0)

    def test_all_soma_pair_dominant(self) -> None:
        """All inhibition on the soma → every pair is strong-soma, G=+1.

        The distal-level weights are zero, so
        ``f_soma = 1``, ``f_distal = 0``, ``G = +1`` for every (i, e).
        """
        n_E, n_I = 4, 3
        # Two levels: distal (factor=2) and soma (factor=1)
        W_distal = torch.zeros(n_E * 2, n_I)
        W_soma = torch.ones(n_E * 1, n_I)
        layer = _make_layer(n_E, n_I, [2, 1], [W_distal, W_soma])
        model = _FakeModel(_FakeCore([layer]))

        params = InhibitorySpecializationAnalysisParams(
            gate_threshold=0.1, std_threshold=0.15, save_figures=False
        )
        analyzer = InhibitorySpecializationAnalyzer(params)
        results = analyzer.analyze(model, device="cpu")

        self.assertEqual(list(results["layers"].keys()), ["layer_0"])
        entry = results["layers"]["layer_0"]
        self.assertEqual(entry["n_E"], n_E)
        self.assertEqual(entry["n_I"], n_I)
        self.assertEqual(entry["n_levels"], 2)

        G = torch.as_tensor(entry["G"])
        self.assertEqual(tuple(G.shape), (n_I, n_E))
        self.assertTrue(torch.allclose(G, torch.ones_like(G), atol=1e-6))

        self.assertEqual(entry["class_counts"]["gate-specialist"], n_I)
        self.assertEqual(entry["class_counts"]["context-specialist"], 0)
        self.assertAlmostEqual(entry["pair_level"]["strong_soma_pct"], 100.0)
        self.assertAlmostEqual(entry["pair_level"]["strong_distal_pct"], 0.0)

    def test_all_distal_pair_dominant(self) -> None:
        """All inhibition on distal → every pair is strong-distal, G=-1."""
        n_E, n_I = 4, 3
        W_distal = torch.ones(n_E * 2, n_I)
        W_soma = torch.zeros(n_E * 1, n_I)
        layer = _make_layer(n_E, n_I, [2, 1], [W_distal, W_soma])
        model = _FakeModel(_FakeCore([layer]))
        analyzer = InhibitorySpecializationAnalyzer()
        results = analyzer.analyze(model, device="cpu")
        entry = results["layers"]["layer_0"]

        G = torch.as_tensor(entry["G"])
        self.assertTrue(torch.allclose(G, -torch.ones_like(G), atol=1e-6))
        self.assertEqual(entry["class_counts"]["context-specialist"], n_I)
        self.assertAlmostEqual(entry["pair_level"]["strong_distal_pct"], 100.0)

    def test_mixed_conditional_when_half_and_half(self) -> None:
        """Half soma-like E-partners, half distal-like → I is mixed-conditional.

        We build a single I-neuron that targets the soma of half the E
        population and the distal compartments of the other half. The
        per-pair G takes both signs, the std across E is large, and the
        per-I classification should be "mixed-conditional".
        """
        n_E, n_I = 6, 1
        W_distal = torch.zeros(n_E * 2, n_I)
        W_soma = torch.zeros(n_E * 1, n_I)
        # First half of E get all-soma inhibition, second half all-distal
        W_soma[: n_E // 2, :] = 1.0
        for e in range(n_E // 2, n_E):
            W_distal[e * 2, 0] = 1.0
            W_distal[e * 2 + 1, 0] = 1.0

        layer = _make_layer(n_E, n_I, [2, 1], [W_distal, W_soma])
        model = _FakeModel(_FakeCore([layer]))
        analyzer = InhibitorySpecializationAnalyzer()
        results = analyzer.analyze(model, device="cpu")
        entry = results["layers"]["layer_0"]

        self.assertEqual(entry["class_counts"]["mixed-conditional"], 1)
        self.assertEqual(entry["class_counts"]["gate-specialist"], 0)
        self.assertEqual(entry["class_counts"]["context-specialist"], 0)

    def test_baseline_rnn_returns_empty(self) -> None:
        """No E-I layers (baseline RNN) → analyzer returns an empty dict."""

        class _Core(nn.Module):
            pass

        model = _FakeModel(_Core())
        analyzer = InhibitorySpecializationAnalyzer()
        self.assertEqual(analyzer.analyze(model, device="cpu"), {})

    def test_three_level_network_middle_reported(self) -> None:
        """With L=3, the analyzer reports ``frac_middle_pop`` and keeps G = soma - distal."""
        n_E, n_I = 4, 2
        W_distal = torch.ones(n_E * 3, n_I) * 0.1
        W_middle = torch.ones(n_E * 2, n_I) * 0.5
        W_soma = torch.ones(n_E * 1, n_I) * 0.4
        layer = _make_layer(n_E, n_I, [3, 2, 1], [W_distal, W_middle, W_soma])
        model = _FakeModel(_FakeCore([layer]))
        analyzer = InhibitorySpecializationAnalyzer(
            InhibitorySpecializationAnalysisParams(save_figures=False)
        )
        results = analyzer.analyze(model, device="cpu")
        entry = results["layers"]["layer_0"]

        self.assertEqual(entry["n_levels"], 3)
        self.assertIn("frac_middle_pop", entry)
        # f_distal = 0.1/(0.1+0.5+0.4) = 0.1, f_soma = 0.4 → G = 0.3 for all pairs
        G = torch.as_tensor(entry["G"])
        self.assertTrue(torch.allclose(G, torch.full_like(G, 0.3), atol=1e-4))

    def test_saves_json_and_pdf(self) -> None:
        n_E, n_I = 4, 3
        W_distal = torch.rand(n_E * 2, n_I)
        W_soma = torch.rand(n_E * 1, n_I)
        layer = _make_layer(n_E, n_I, [2, 1], [W_distal, W_soma])
        model = _FakeModel(_FakeCore([layer]))
        analyzer = InhibitorySpecializationAnalyzer(
            InhibitorySpecializationAnalysisParams(save_figures=True)
        )
        with tempfile.TemporaryDirectory() as td:
            analyzer.analyze(model, device="cpu", save_path=td, filename="final")
            files = {p.name for p in Path(td).iterdir()}
            self.assertIn("final.json", files)
            self.assertIn("final.pdf", files)
            with open(Path(td) / "final.json") as f:
                payload = json.load(f)
            self.assertIn("layers", payload)
            self.assertEqual(payload["thresholds"]["gate_threshold"], 0.1)


if __name__ == "__main__":
    unittest.main()
