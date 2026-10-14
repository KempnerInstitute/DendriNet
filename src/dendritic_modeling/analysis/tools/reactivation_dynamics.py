"""
Reactivation dynamics analyzer.

Tracks per-epoch trajectories of every parametric reactivation's gate
parameters (``log_m`` and the midpoint ``b`` or ``fixed_b``, and any other
gate parameter) alongside statistics of the pre-reactivation voltage
distribution ``V`` over every call in the forward pass (each timestep of a
recurrent unroll) and of the gate output ``r``. Rail occupancy (``r < 0.05``,
``r > 0.95``) is reported for the positive-tanh gates, whose outputs lie in
``(0, 1)``.

Purpose
-------
Diagnose whether the reactivation gate drifts during training (e.g., in
the "empirical calibration over-sharpens shunting" hypothesis): a
progressively steeper ``log_m`` paired with a shift in ``V`` quantiles
toward the saturation regions ``r < 0.05`` or ``r > 0.95`` indicates the
gate is memorizing the training signal by rail-saturating its output.

Each call emits one JSON snapshot per reactivation layer for that epoch;
a post-hoc tool (or a notebook) can concatenate all epoch files under
``<save_root>/reactivation_dynamics/epochs/epoch*.json`` into full
trajectories.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.dendritic_depth import (
    soma_relative_dendritic_depth,
)
from dendritic_modeling.analysis.utils.runtime import analysis_device_context
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    ReactivationDynamicsAnalysisParams,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.structure import inhibitory_branch_layer_ids
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import ForwardHookRemovalMixin, register_hook_groups


def _quantile_stats(v: torch.Tensor) -> dict[str, float]:
    """Compute the diagnostic statistics for a 1-D voltage tensor ``v``.

    Returns both moment-based (mean/std) and robust (median/MAD) estimators
    so we can cross-check them — when the distribution is approximately
    symmetric the two agree, and when they diverge the robust estimators
    are the ones actually used by ``calibrate_reactivation_empirical``.
    """
    v = v.reshape(-1).float()
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "mad_std": 0.0,
            "q01": 0.0,
            "q10": 0.0,
            "q50": 0.0,
            "q90": 0.0,
            "q99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
        }
    mean = float(v.mean().item())
    std = float(v.std(unbiased=False).item())
    median = float(torch.median(v).item())
    mad = float(torch.median(torch.abs(v - median)).item())
    mad_std = 1.4826 * mad
    # Percentiles used by the quantile-occupancy calibration proposal.
    q = torch.quantile(
        v, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], dtype=v.dtype, device=v.device)
    )
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "mad_std": mad_std,
        "q01": float(q[0].item()),
        "q10": float(q[1].item()),
        "q50": float(q[2].item()),
        "q90": float(q[3].item()),
        "q99": float(q[4].item()),
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "n": int(v.numel()),
    }


# Gates whose outputs lie in (0, 1), so rail occupancy is defined.
UNIT_RANGE_GATES = (ParametricTanh, ParametricTanhOnlyM)


def _is_tracked_gate(module: torch.nn.Module) -> bool:
    return isinstance(module, (ParametricActivation, EndpointNormalizedTanh))


def _reactivation_output_stats(
    reactivation: torch.nn.Module, v: torch.Tensor
) -> dict[str, float]:
    """Return summary stats of ``r = reactivation(v)`` for the given voltage.

    For positive-tanh gates (outputs in ``(0, 1)``) this includes the
    fraction of outputs piled at the rails (``< 0.05`` or ``> 0.95``) as a
    direct saturation indicator; other gates report the output moments.
    """
    with torch.no_grad():
        r = reactivation(v)
    r_flat = r.reshape(-1).float()
    r_flat = r_flat[torch.isfinite(r_flat)]
    unit_range = isinstance(reactivation, UNIT_RANGE_GATES)
    if r_flat.numel() == 0:
        stats = {"mean": 0.0, "std": 0.0}
        if unit_range:
            stats.update(sat_lo=0.0, sat_hi=0.0, active_frac=0.0)
        return stats
    stats = {
        "mean": float(r_flat.mean().item()),
        "std": float(r_flat.std(unbiased=False).item()),
    }
    if unit_range:
        sat_lo = float((r_flat < 0.05).float().mean().item())
        sat_hi = float((r_flat > 0.95).float().mean().item())
        stats.update(sat_lo=sat_lo, sat_hi=sat_hi, active_frac=1.0 - sat_lo - sat_hi)
    else:
        stats.update(min=float(r_flat.min().item()), max=float(r_flat.max().item()))
    return stats


def _summary(prefix: str, values: torch.Tensor) -> dict[str, float]:
    values = values.detach().reshape(-1).float()
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_std": float(values.std(unbiased=False).item()),
        f"{prefix}_min": float(values.min().item()),
        f"{prefix}_max": float(values.max().item()),
    }


def _gate_params(reactivation: torch.nn.Module) -> dict[str, float]:
    """Snapshot a gate's parameters as per-layer summaries across units.

    ``log_m`` is also reported as the slope ``m = exp(log_m)``; the midpoint
    is the learnable ``b`` or the calibrated ``fixed_b``. Every other gate
    parameter is summarized under its own name.
    """
    out: dict[str, float] = {}
    with torch.no_grad():
        log_m = getattr(reactivation, "log_m", None)
        if torch.is_tensor(log_m):
            out["log_m_mean"] = float(log_m.mean().item())
            out["log_m_std"] = float(log_m.std(unbiased=False).item())
            m = torch.exp(log_m)
            out["m_mean"] = float(m.mean().item())
            out["m_std"] = float(m.std(unbiased=False).item())
            out["m_min"] = float(m.min().item())
            out["m_max"] = float(m.max().item())
        midpoint = getattr(reactivation, "b", None)
        if midpoint is None:
            midpoint = getattr(reactivation, "fixed_b", None)
        if midpoint is not None:
            out.update(_summary("b", torch.as_tensor(midpoint)))
        for name, parameter in reactivation.named_parameters(recurse=False):
            if name not in {"log_m", "b"}:
                out.update(_summary(name, parameter))
    return out


class ReactivationDynamicsAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """Per-epoch trajectory tracker for parametric reactivation gates.

    For every parametric gate in the model, on each ``analyze()`` call
    the analyzer:

    1. Runs a fixed-size sample batch through the model with forward
       pre-hooks on the reactivation modules, capturing the
       pre-reactivation voltage ``V``.
    2. Computes moment-based and robust-statistic summaries of ``V``
       (``mean/std`` and ``median/MAD*1.4826``), quantiles
       (``q01, q10, q50, q90, q99``), and extremes.
    3. Snapshots the current learnable ``(log_m, b)`` parameter values.
    4. Applies the reactivation to ``V`` once to measure the post-gate
       output's mean, std, and the saturation fractions
       ``sat_lo = P(r < 0.05)``, ``sat_hi = P(r > 0.95)``.
    5. Saves one JSON file per epoch containing one entry per layer.

    The analyzer has no internal epoch counter; the trainer passes
    ``filename=f"epoch{N}"`` into ``analyze()`` on each call. After
    training, the set of files under
    ``<save_root>/reactivation_dynamics/epochs/`` concatenates into
    per-layer trajectories.
    """

    def __init__(self, params: ReactivationDynamicsAnalysisParams):
        super().__init__("ReactivationDynamicsAnalyzer")
        self.params = params
        self.max_samples = int(getattr(params, "max_samples", 1024))

    def _list_gate_layers(
        self, model: torch.nn.Module
    ) -> list[tuple[str, torch.nn.Module]]:
        """Return ``(dotted_name, module)`` pairs for every parametric gate."""
        return [
            (name, module)
            for name, module in model.named_modules()
            if _is_tracked_gate(module)
        ]

    def _sample_inputs(
        self,
        data: torch.utils.data.Dataset,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Pull up to ``self.max_samples`` inputs from the dataset as a single tensor."""
        if data is None:
            return None
        loader = DataLoader(
            data,
            batch_size=min(self.max_samples, max(1, len(data))),
            shuffle=False,
        )
        batch = next(iter(loader))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if x is None or not torch.is_tensor(x):
            return None
        return x[: self.max_samples].to(device)

    def analyze(
        self,
        model: BaseModel,
        train_ds: torch.utils.data.Dataset | None = None,
        valid_ds: torch.utils.data.Dataset | None = None,
        test_ds: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture reactivation dynamics for one epoch snapshot.

        The canonical training-time call path from
        ``AnalysisManager.run_analysis`` provides ``filename=f"epoch{N}"``
        and ``training=True`` each epoch. On the final call
        (``training=False``), we use the test split so the last snapshot
        reflects held-out voltages.
        """
        layers = self._list_gate_layers(model)
        if not layers:
            self.logger.debug("No parametric reactivations found; skipping")
            self.report_skip("the model has no parametric reactivation gates")
            return {}

        dataset = self._select_dataset(training, train_ds, valid_ds, test_ds)
        if dataset is None:
            self.logger.debug("No dataset available; skipping dynamics snapshot")
            self.report_skip("no dataset was available")
            return {}

        with analysis_device_context(model, device) as current_device:
            x = self._sample_inputs(dataset, current_device)
            if x is None:
                return {}

            x = self._prepare_inputs(model, x)
            captured = self._capture_reactivation_inputs(model, layers, x)

            snapshot = self._build_snapshot(
                layers,
                captured,
                filename,
                training,
                coordinates=self._gate_coordinates(model, layers),
            )

        if save_path is not None:
            self._save_snapshot(snapshot, save_path, filename)

        return snapshot

    def _select_dataset(
        self,
        training: bool,
        train_ds: torch.utils.data.Dataset | None,
        valid_ds: torch.utils.data.Dataset | None,
        test_ds: torch.utils.data.Dataset | None,
    ) -> torch.utils.data.Dataset | None:
        dataset = train_ds if training else test_ds
        if dataset is None:
            dataset = valid_ds or test_ds
        return dataset

    def _prepare_inputs(
        self,
        model: BaseModel,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(model, "_is_recurrent_core", False) and x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def _capture_reactivation_inputs(
        self,
        model: BaseModel,
        layers: list[tuple[str, torch.nn.Module]],
        x: torch.Tensor,
    ) -> dict[str, list[torch.Tensor]]:
        """Inputs of every gate call in one forward pass (each timestep)."""
        captured: dict[str, list[torch.Tensor]] = {}
        handles = self._attach_capture_hooks(layers, captured)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                _ = model(x)
        finally:
            self.remove_forward_hooks(handles)
            if was_training:
                model.train()
        return captured

    def _attach_capture_hooks(
        self,
        layers: list[tuple[str, torch.nn.Module]],
        captured: dict[str, list[torch.Tensor]],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        return register_hook_groups(
            layers,
            lambda layer_entry: [self._register_capture_hook(layer_entry, captured)],
        )

    def _register_capture_hook(
        self,
        layer_entry: tuple[str, torch.nn.Module],
        captured: dict[str, list[torch.Tensor]],
    ) -> torch.utils.hooks.RemovableHandle:
        name, layer = layer_entry
        return layer.register_forward_pre_hook(self._make_hook(name, captured))

    @staticmethod
    def _make_hook(layer_name: str, captured: dict[str, list[torch.Tensor]]):
        def hook(_mod, inputs):
            captured.setdefault(layer_name, []).append(inputs[0].detach())

        return hook

    @staticmethod
    def _gate_coordinates(
        model: torch.nn.Module, layers: list[tuple[str, torch.nn.Module]]
    ) -> dict[str, dict[str, Any]]:
        """Soma-relative depth and population polarity of each gate's branch layer."""
        inhibitory = inhibitory_branch_layer_ids(model)
        coordinates: dict[str, dict[str, Any]] = {}
        for name, _gate in layers:
            parent_name, _, _ = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else None
            if not isinstance(parent, DendriticBranchLayer):
                continue
            coordinates[name] = {
                "soma_relative_dendritic_depth": soma_relative_dendritic_depth(parent),
                "population_polarity": (
                    "inhibitory" if id(parent) in inhibitory else "excitatory"
                ),
            }
        return coordinates

    def _build_snapshot(
        self,
        layers: list[tuple[str, torch.nn.Module]],
        captured: dict[str, list[torch.Tensor]],
        filename: str,
        training: bool,
        coordinates: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        per_layer: dict[str, dict[str, Any]] = {}
        for name, layer in layers:
            calls = captured.get(name)
            if not calls:
                per_layer[name] = {"skipped": True}
                continue
            v = torch.cat([call.reshape(-1, call.shape[-1]) for call in calls])
            per_layer[name] = {
                **(coordinates or {}).get(name, {}),
                "gate": type(layer).__name__,
                "n_calls": len(calls),
                "V": _quantile_stats(v),
                "params": _gate_params(layer),
                "reactivation_output": _reactivation_output_stats(layer, v),
            }

        return {
            "filename": filename,
            "training": bool(training),
            "n_layers": len(per_layer),
            "layers": per_layer,
        }

    def _save_snapshot(
        self,
        snapshot: dict[str, Any],
        save_path: str,
        filename: str,
    ) -> None:
        epochs_dir = os.path.join(save_path, "epochs")
        os.makedirs(epochs_dir, exist_ok=True)
        save_dict(snapshot, epochs_dir, f"{filename}.json")


__all__ = ["ReactivationDynamicsAnalyzer"]
