"""Synaptic pruning analysis: realized connectivity and what pruning removed.

The analyzer takes a census of the model's realized contacts (see
:mod:`dendritic_modeling.analysis.utils.connectivity_census`) at the final
analysis. When the run prunes weights, the training script records a census
immediately before and after each pruning step through
:meth:`AnalysisManager.record_pruning_start` and
:meth:`AnalysisManager.record_pruning_end`; each step becomes a pruning event
with the pruning statistics, the pre/post-pruning losses and the contacts
removed per branch layer and pathway.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.connectivity_census import (
    census_changes,
    connectivity_census,
)
from dendritic_modeling.config.analysis import SynapsePruningAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_active_synapses_per_branch,
    plot_active_synapses_summary_by_layer,
)
from dendritic_modeling.plotting.visualizations.plotting_utils import save_plot
from dendritic_modeling.utils import save_dict


def _format_pruning_summary_text(results: dict[str, Any]) -> str:
    """Return a display-ready summary of the connectivity and pruning events."""
    totals = results["connectivity"]["totals"]
    lines = [
        "=" * 60,
        "SYNAPTIC PRUNING ANALYSIS",
        "=" * 60,
        "Realized contacts of the final model:",
    ]
    for group, counts in totals.items():
        lines.append(
            f"  {group:<8} {counts['realized']:>12,} realized of "
            f"{counts['selected']:,} selected"
        )
    events = results["pruning_events"]
    if not events:
        lines.append("No pruning step was recorded.")
    for index, event in enumerate(events):
        total = event["changes"]["totals"]["all"]
        lines.append(
            f"Pruning step {index + 1}: removed {total['removed']:,} of "
            f"{total['realized_before']:,} realized contacts "
            f"({total['removed_fraction']:.2%})"
        )
        for phase in ("before", "after"):
            performance = event["performance"].get(phase)
            if performance:
                lines.append(
                    f"  {phase:<6} train loss {performance['train_loss']:.4f}, "
                    f"validation loss {performance['valid_loss']:.4f}"
                )
    lines.append("=" * 60)
    return "\n".join(lines)


class SynapticPruningAnalyzer(AbstractAnalyzer):
    """Realized connectivity of the final model and the effect of each pruning step."""

    def __init__(self, params: SynapsePruningAnalysisParams | None = None):
        super().__init__(type(self).__name__)
        self.params = params if params is not None else SynapsePruningAnalysisParams()
        self.pruning_events: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None

    def record_pruning_start(self, model: BaseModel) -> None:
        """Take the census immediately before a pruning step."""
        if self._pending is not None:
            raise RuntimeError("a pruning step was started twice without ending")
        self._pending = {"connectivity_before": connectivity_census(model)}

    def record_pruning_end(
        self, model: BaseModel, pruning_results: dict[str, Any]
    ) -> None:
        """Take the census after the step and record it as a pruning event."""
        if self._pending is None:
            raise RuntimeError("record_pruning_end called without record_pruning_start")
        before = self._pending.pop("connectivity_before")
        self._pending = None
        after = connectivity_census(model)
        self.pruning_events.append(
            {
                "pruning_stats": dict(pruning_results.get("pruning_stats") or {}),
                "performance": {
                    "before": pruning_results.get("pre_pruning_performance"),
                    "after": pruning_results.get("post_pruning_performance"),
                },
                "connectivity_before": before,
                "connectivity_after": after,
                "changes": census_changes(before, after),
            }
        )

    def analyze(
        self,
        model: BaseModel,
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
    ) -> dict[str, Any]:
        """Return the final census and the recorded pruning events; save them."""
        results = {
            "connectivity": connectivity_census(model),
            "pruning_events": list(self.pruning_events),
        }
        if save_path is not None:
            self._save_results(results, save_path, filename)
        return results

    def summary_text(self, results: dict[str, Any]) -> str:
        """Return a human-readable summary of ``analyze`` results."""
        return _format_pruning_summary_text(results)

    def _save_results(
        self, results: dict[str, Any], save_path: str, filename: str
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        save_dict(results, save_path, filename)
        with open(os.path.join(save_path, f"{filename}_summary.txt"), "w") as handle:
            handle.write(self.summary_text(results) + "\n")
        if not self.params.save_figures:
            return
        before = (
            results["pruning_events"][0]["connectivity_before"]
            if results["pruning_events"]
            else None
        )
        save_plot(
            plot_active_synapses_per_branch(results["connectivity"], before=before),
            save_path,
            f"{filename}_active_synapses_per_branch",
        )
        save_plot(
            plot_active_synapses_summary_by_layer(
                results["connectivity"], before=before
            ),
            save_path,
            f"{filename}_active_synapses_by_layer",
        )


__all__ = ["SynapticPruningAnalyzer"]
