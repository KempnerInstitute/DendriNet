"""Tests for generic E/I connectivity motif summaries."""

from __future__ import annotations

import unittest

import torch

from dendritic_modeling.analysis.tools.ei_connectivity_motifs import (
    LOCAL_RECURRENT_ROUTE_KEYS,
    analyze_ei_connectivity_motifs_from_state_dict,
    extract_compartment_weights_from_state_dict,
    pair_stats_by_class,
    summarize_compartment_targeting_from_compartment_weights,
)


class TestEIConnectivityMotifs(unittest.TestCase):
    def test_extract_compartment_weights_for_different_routes(self) -> None:
        n_E, n_I = 4, 2
        sd = {
            "core.e_population.branch_layers.0.branch_rec_inhibition.pre_w": torch.log(
                torch.ones(n_E * 2, n_I) * 0.5
            ),
            "core.e_population.branch_layers.1.branch_rec_inhibition.pre_w": torch.log(
                torch.ones(n_E * 1, n_I) * 1.0
            ),
            "core.i_population.branch_layers.0.branch_recurrent.pre_w": torch.log(
                torch.ones(n_I * 3, n_E) * 0.25
            ),
            "core.i_population.branch_layers.1.branch_recurrent.pre_w": torch.log(
                torch.ones(n_I * 1, n_E) * 0.75
            ),
        }

        ie = extract_compartment_weights_from_state_dict(
            sd,
            prefix="core.e_population",
            synapse_attr="branch_rec_inhibition",
            n_target=n_E,
            weight_transform="exp",
            active_masks=[torch.ones(n_E * 2, n_I), torch.ones(n_E, n_I)],
        )
        ei = extract_compartment_weights_from_state_dict(
            sd,
            prefix="core.i_population",
            synapse_attr="branch_recurrent",
            n_target=n_I,
            weight_transform="exp",
            active_masks=[torch.ones(n_I * 3, n_E), torch.ones(n_I, n_E)],
        )

        self.assertEqual([tuple(w.shape) for w in ie], [(n_E, 2, n_I), (n_E, 1, n_I)])
        self.assertEqual([tuple(w.shape) for w in ei], [(n_I, 3, n_E), (n_I, 1, n_E)])

    def test_summary_reports_expected_bias_and_classes(self) -> None:
        n_target, n_source = 4, 2
        distal = torch.ones(n_target, 2, n_source) * 0.25
        soma = torch.ones(n_target, 1, n_source) * 1.0
        summary = summarize_compartment_targeting_from_compartment_weights(
            [distal, soma]
        )

        bias = summary["targeting_index"]
        self.assertTrue(torch.allclose(bias, torch.full_like(bias, 0.6), atol=1e-6))
        self.assertEqual(summary["class_counts"]["soma-biased"], n_source)
        self.assertEqual(summary["pair_level"]["strong_soma_pct"], 100.0)

    def test_pair_stats_by_class_reports_expected_balance(self) -> None:
        bias = torch.tensor(
            [
                [1.0, 1.0, -1.0, -1.0],  # mixed
                [-0.4, -0.3, -0.5, -0.2],  # distal
            ]
        )
        classes = ["mixed-targeting", "distal-biased"]
        stats = pair_stats_by_class(bias, classes, hi=0.1)

        self.assertEqual(stats["mixed-targeting"]["n_pairs"], 4)
        self.assertEqual(stats["distal-biased"]["strong_distal_pct"], 100.0)

    def test_route_analysis_skips_missing_and_keeps_present(self) -> None:
        n_E, n_I = 3, 2
        sd = {
            "core_network.layers.0.e_population.branch_layers.0.branch_recurrent.pre_w": torch.log(
                torch.ones(n_E * 2, n_E) * 0.2
            ),
            "core_network.layers.0.e_population.branch_layers.1.branch_recurrent.pre_w": torch.log(
                torch.ones(n_E * 1, n_E) * 0.8
            ),
            "core_network.layers.0.i_population.branch_layers.0.branch_rec_inhibition.pre_w": torch.log(
                torch.ones(n_I * 2, n_I) * 0.7
            ),
            "core_network.layers.0.i_population.branch_layers.1.branch_rec_inhibition.pre_w": torch.log(
                torch.ones(n_I * 1, n_I) * 0.1
            ),
        }

        out = analyze_ei_connectivity_motifs_from_state_dict(
            sd,
            layer_prefix="core_network.layers.0",
            n_E=n_E,
            n_I=n_I,
            weight_transform="exp",
            route_keys=LOCAL_RECURRENT_ROUTE_KEYS,
            active_masks_by_route={
                "e_rec_exc": [
                    torch.ones(n_E * 2, n_E),
                    torch.ones(n_E, n_E),
                ],
                "i_rec_inh": [
                    torch.ones(n_I * 2, n_I),
                    torch.ones(n_I, n_I),
                ],
            },
        )

        self.assertIn("e_rec_exc", out)
        self.assertIn("i_rec_inh", out)
        self.assertNotIn("e_rec_inh", out)
        self.assertNotIn("i_rec_exc", out)
        self.assertEqual(out["e_rec_exc"]["target_label"], "E")
        self.assertEqual(out["i_rec_inh"]["source_label"], "I")


if __name__ == "__main__":
    unittest.main()
