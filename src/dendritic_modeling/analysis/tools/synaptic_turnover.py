"""Synapse turnover: the contacts that persist, form and disappear in training.

The analyzer snapshots the active-contact mask of every synapse bank of every
branch layer when it is built (the model before training) and every
``snapshot_interval`` training epochs. Each snapshot is compared with the
previous one (``incremental``) and with the first (``cumulative``). Only those
two reference masks are kept, so memory does not grow with training length.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    get_model_device,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.structure import inhibitory_branch_layer_ids
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_layer_synapse_statistics,
    plot_synapse_turnover,
)
from dendritic_modeling.plotting.visualizations.plotting_utils import save_plot
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import iter_named_modules_of_type

logger = logging.getLogger(__name__)

# Synapse banks of a branch layer, by pathway label.
SYNAPSE_BANKS = {
    "exc": "branch_excitation",
    "inh": "branch_inhibition",
    "rec_exc": "branch_recurrent",
    "rec_inh": "branch_rec_inhibition",
}
COUNTS = ("active", "stable", "formed", "eliminated")


def turnover_counts(reference: torch.Tensor, current: torch.Tensor) -> dict[str, int]:
    """Count active, stable, newly formed and eliminated contacts.

    ``stable`` contacts are active in both masks, ``formed`` only in
    ``current`` and ``eliminated`` only in ``reference``.
    """
    reference = reference.bool()
    current = current.bool()
    formed = int((current & ~reference).sum())
    active = int(current.sum())
    return {
        "active": active,
        "stable": active - formed,
        "formed": formed,
        "eliminated": int((reference & ~current).sum()),
    }


def _empty_series() -> dict[str, list[int]]:
    return {count: [] for count in COUNTS}


class SynapticTurnoverAnalyzer(AbstractAnalyzer):
    """Track synapse formation and elimination during training.

    Counts are reported per synapse bank of every branch layer and summed per
    pathway (``exc``, ``inh``, ``rec_exc``, ``rec_inh``) and over all banks
    (``all``). Every list has one entry per snapshot after the first, at the
    epochs in ``snapshot_epochs[1:]``.
    """

    def __init__(self, model: BaseModel, params: Any):
        super().__init__(type(self).__name__)
        self.snapshot_interval = int(getattr(params, "snapshot_interval", 1))
        if self.snapshot_interval < 1:
            raise ValueError("synapse_turnover snapshot_interval must be at least 1")
        inhibitory = inhibitory_branch_layer_ids(model)
        self.populations = {
            name: "explicit_inhibitory" if id(module) in inhibitory else "excitatory"
            for name, module in iter_named_modules_of_type(model, DendriticBranchLayer)
        }
        self.epoch = 0
        self.snapshot_epochs = [0]
        self._initial = self._masks(model)
        if not self._initial:
            raise ValueError(
                "synapse turnover needs branch layers with synapse banks; the "
                "model has none"
            )
        self._previous = self._initial
        self._series = {
            mode: {key: _empty_series() for key in self._initial}
            for mode in ("incremental", "cumulative")
        }

    def _masks(self, model: BaseModel) -> dict[tuple[str, str], torch.Tensor]:
        masks = {}
        # Eval mode: some banks perturb their selection while training.
        with analysis_device_context(model, get_model_device(model)), torch.no_grad():
            for name, module in iter_named_modules_of_type(model, DendriticBranchLayer):
                for pathway, attribute in SYNAPSE_BANKS.items():
                    bank = getattr(module, attribute, None)
                    if bank is None:
                        continue
                    if not callable(getattr(bank, "weight_mask", None)):
                        raise ValueError(
                            f"{name}.{attribute} is a {type(bank).__name__}, which "
                            "exposes no weight_mask(); synapse turnover cannot "
                            "read its active contacts"
                        )
                    # A copy: some banks return their live mask buffer.
                    mask = bank.weight_mask().detach().to("cpu", copy=True).bool()
                    masks[(name, pathway)] = mask
        return masks

    def analyze(
        self,
        model: BaseModel,
        training: bool = True,
        save_path: Optional[str] = None,
        filename: str = "final",
    ) -> Optional[dict[str, Any]]:
        """Snapshot during training; report and save the statistics after it."""
        if training:
            self.epoch += 1
            if self.epoch % self.snapshot_interval == 0:
                self._record_snapshot(model)
            self.report_deferred_output(
                "training snapshots are kept in memory; the final analysis "
                "writes the turnover statistics"
            )
            return None
        results = self.results()
        if save_path is not None:
            save_dict(results, save_path, filename)
            save_plot(plot_synapse_turnover(results), save_path, f"{filename}_turnover")
            save_plot(
                plot_layer_synapse_statistics(results),
                save_path,
                f"{filename}_turnover_by_layer",
            )
        return results

    def _record_snapshot(self, model: BaseModel) -> None:
        current = self._masks(model)
        if current.keys() != self._initial.keys():
            raise ValueError(
                "the model's synapse banks changed during training; synapse "
                "turnover compares the same banks across snapshots"
            )
        for key, mask in current.items():
            for mode, reference in (
                ("incremental", self._previous[key]),
                ("cumulative", self._initial[key]),
            ):
                for count, value in turnover_counts(reference, mask).items():
                    self._series[mode][key][count].append(value)
        self._previous = current
        self.snapshot_epochs.append(self.epoch)

    def results(self) -> dict[str, Any]:
        """Return the turnover statistics of the snapshots taken so far."""
        if len(self.snapshot_epochs) < 2:
            raise ValueError(
                "synapse turnover has no training snapshot to compare; it runs "
                "with training: true and snapshots every snapshot_interval epochs"
            )
        results: dict[str, Any] = {
            "snapshot_epochs": list(self.snapshot_epochs),
            "n_snapshots": len(self.snapshot_epochs),
        }
        modules: dict[str, dict[str, Any]] = {}
        for mode, series in self._series.items():
            totals: dict[str, dict[str, list[int]]] = {}
            for (name, pathway), counts in series.items():
                module = modules.setdefault(
                    name, {"population": self.populations[name]}
                )
                module.setdefault(mode, {})[pathway] = counts
                for group in (pathway, "all"):
                    total = totals.setdefault(group, _empty_series())
                    for count, values in counts.items():
                        if not total[count]:
                            total[count] = [0] * len(values)
                        total[count] = [a + b for a, b in zip(total[count], values)]
            results[mode] = totals
        results["modules"] = modules
        return results


__all__ = [
    "COUNTS",
    "SYNAPSE_BANKS",
    "SynapticTurnoverAnalyzer",
    "turnover_counts",
]
