"""Focused tests for synaptic pruning analysis helpers."""

import contextlib
import io

from dendritic_modeling.analysis.tools.synaptic_pruning import SynapticPruningAnalyzer


def _capture_pruning_summary(analyzer: SynapticPruningAnalyzer) -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        analyzer.print_pruning_summary()
    return stream.getvalue()


def test_print_pruning_summary_preserves_empty_event_output():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    analyzer.pruning_events = []

    assert _capture_pruning_summary(analyzer) == "No pruning events recorded.\n"


def test_print_pruning_summary_formats_events_and_initial_connectivity():
    analyzer = object.__new__(SynapticPruningAnalyzer)
    analyzer.pruning_events = [
        {
            "epoch": 2,
            "pruning_stats": {"pruned_params": 1000, "pruning_ratio": 0.25},
            "connectivity_changes": {"sparsity_increase": 0.125},
        },
        {
            "epoch": 4,
            "pruning_stats": {"pruned_params": 500, "pruning_ratio": 0.1},
            "connectivity_changes": {},
        },
    ]
    analyzer.initial_connectivity_stats = {
        "active_synapses": 9000,
        "total_synapses": 12000,
        "overall_sparsity": 0.25,
    }
    analyzer.data_dict_list = [{"epoch": 0}]

    assert _capture_pruning_summary(analyzer) == "\n".join(
        [
            "",
            "=" * 60,
            "SYNAPTIC PRUNING ANALYSIS SUMMARY",
            "=" * 60,
            "Total pruning events: 2",
            "Total synapses pruned: 1,500",
            "Average pruning ratio per event: 0.1750",
            "",
            "Initial connectivity:",
            "  Active synapses: 9,000 / 12,000",
            "  Initial sparsity: 0.2500",
            "",
            "Pruning events:",
            f"{'Epoch':<8} {'Pruned':<12} {'Ratio':<8} {'Sparsity Inc':<12}",
            "-" * 45,
            f"{2:<8} {1000:<12,} {0.25:<8.4f} {0.125:<12.4f}",
            f"{4:<8} {500:<12,} {0.1:<8.4f} {0:<12.4f}",
            "=" * 60,
            "",
        ]
    )
