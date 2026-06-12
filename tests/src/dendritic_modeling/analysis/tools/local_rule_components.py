"""Local-rule component analyzer for dendritic LocalCA models.

This analyzer records the fast factors that enter the local conductance
updates:

* branch voltage ``V_b`` and post-reactivation activity,
* total conductance and input resistance ``R_tot``,
* E/I synaptic driving forces ``E_rev - V_b``,
* active presynaptic drives for E and I synapses,
* rule-correct eligibility factors for E, I, and dendritic conductances,
* optional low-bandwidth broadcast-error and local-update products.

It is designed to run both at final evaluation and during training.  When
called with ``training=True`` it writes one CSV snapshot per epoch plus an
append-only trajectory CSV.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    LocalRuleComponentAnalysisParams,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.activations.parametric import (
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.utils import save_dict


@dataclass(frozen=True)
class _ComponentKey:
    """Stable grouping key for one recorded component."""

    module_name: str
    population: str
    branch_module_index: int
    dendritic_depth: int
    depth_label: str
    synapse_type: str
    component: str
    used_in_rule: bool


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _flatten_finite(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().reshape(-1).float().cpu()
    return values[torch.isfinite(values)]


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    values = _flatten_finite(values)
    if values.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "mean_abs": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q05": float("nan"),
            "q50": float("nan"),
            "q95": float("nan"),
            "n": 0,
        }

    quantile_values = values
    max_quantile_values = 1_000_000
    if quantile_values.numel() > max_quantile_values:
        # torch.quantile can fail on very large tensors; strided subsampling
        # keeps epoch-level diagnostics deterministic and inexpensive.
        stride = int(
            torch.ceil(
                torch.tensor(quantile_values.numel() / max_quantile_values)
            ).item()
        )
        quantile_values = quantile_values[::stride]

    q = torch.quantile(
        quantile_values,
        torch.tensor([0.05, 0.50, 0.95], dtype=quantile_values.dtype),
    )
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "mean_abs": float(values.abs().mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "q05": float(q[0].item()),
        "q50": float(q[1].item()),
        "q95": float(q[2].item()),
        "n": int(values.numel()),
    }


def _reduce_error_to_scalar(delta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match LocalCA's robust per-example scalar broadcast reduction."""

    if delta.dim() == 1:
        return delta.unsqueeze(-1)
    if delta.dim() != 2:
        delta = delta.view(delta.size(0), -1)

    delta_scalar = delta.mean(dim=1, keepdim=True)
    near_zero = delta_scalar.abs() <= eps
    if near_zero.any():
        max_idx = delta.abs().argmax(dim=1, keepdim=True)
        fallback = delta.gather(dim=1, index=max_idx)
        delta_scalar = torch.where(near_zero, fallback, delta_scalar)
    return delta_scalar


def _parse_branch_module_index(module_name: str) -> int:
    match = re.search(r"branch_layers\.(\d+)", module_name)
    if match is None:
        return -1
    return int(match.group(1))


class LocalRuleComponentAnalyzer(AbstractAnalyzer):
    """Record LocalCA eligibility factors by E/I type and dendritic depth."""

    def __init__(self, params: LocalRuleComponentAnalysisParams):
        super().__init__("LocalRuleComponentAnalyzer")
        self.params = params

    def _depth_labels(self, records: list[dict[str, Any]]) -> dict[int, str]:
        depths = sorted(
            {
                int(getattr(rec["layer"], "layer_idx", -1))
                for rec in records
                if isinstance(rec.get("layer"), DendriticBranchLayer)
            }
        )
        positive = [d for d in depths if d > 0]
        min_positive = min(positive) if positive else None
        max_positive = max(positive) if positive else None
        labels: dict[int, str] = {}
        for depth in depths:
            if depth == 0:
                labels[depth] = "soma"
            elif min_positive is not None and max_positive is not None:
                if depth == max_positive:
                    labels[depth] = "distal"
                elif depth == min_positive:
                    labels[depth] = "proximal"
                else:
                    labels[depth] = "intermediate"
            else:
                labels[depth] = "dendritic"
        return labels

    def _attach_recorders(self, model: BaseModel) -> tuple[
        list[dict[str, Any]],
        list[torch.utils.hooks.RemovableHandle],
        dict[str, torch.Tensor],
    ]:
        records: list[dict[str, Any]] = []
        handles: list[torch.utils.hooks.RemovableHandle] = []
        decoder_cache: dict[str, torch.Tensor] = {}

        if getattr(self.params, "include_broadcast", True) and hasattr(
            model, "decoder_network"
        ):

            def _decoder_pre_hook(_mod, inputs):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    decoder_cache["decoder_input"] = inputs[0]

            handles.append(
                model.decoder_network.register_forward_pre_hook(_decoder_pre_hook)
            )

        for module_name, module in model.named_modules():
            if not isinstance(module, DendriticBranchLayer):
                continue
            population = (
                "explicit_inhibitory"
                if ".inhibitory_cells." in module_name
                else "excitatory"
            )
            rec: dict[str, Any] = {
                "layer": module,
                "module_name": module_name,
                "population": population,
                "branch_module_index": _parse_branch_module_index(module_name),
                "dendritic_depth": int(getattr(module, "layer_idx", -1)),
            }

            def _react_hook(_m, inputs, outputs, rec=rec):
                rec["v_n"] = inputs[0].detach() if inputs else None
                rec["v_out"] = outputs.detach() if torch.is_tensor(outputs) else None

            handles.append(module.reactivation.register_forward_hook(_react_hook))

            exc_layer = getattr(module, "branch_excitation", None)
            if exc_layer is not None:

                def _exc_hook(_m, inputs, outputs, rec=rec, layer=exc_layer):
                    rec["x_exc"] = inputs[0].detach() if inputs else None
                    rec["exc_out"] = outputs.detach()
                    rec["exc_module"] = layer
                    cached = getattr(layer, "_last_forward_weight_mask", None)
                    rec["exc_mask"] = (
                        cached.detach()
                        if isinstance(cached, torch.Tensor)
                        else layer.weight_mask().detach()
                    )

                handles.append(exc_layer.register_forward_hook(_exc_hook))

            inh_layer = getattr(module, "branch_inhibition", None)
            if inh_layer is not None:

                def _inh_hook(_m, inputs, outputs, rec=rec, layer=inh_layer):
                    rec["x_inh"] = inputs[0].detach() if inputs else None
                    rec["inh_out"] = outputs.detach()
                    rec["inh_module"] = layer
                    cached = getattr(layer, "_last_forward_weight_mask", None)
                    rec["inh_mask"] = (
                        cached.detach()
                        if isinstance(cached, torch.Tensor)
                        else layer.weight_mask().detach()
                    )

                handles.append(inh_layer.register_forward_hook(_inh_hook))

            blk_layer = getattr(module, "branches_to_output", None)
            if blk_layer is not None:

                def _blk_hook(_m, inputs, outputs, rec=rec, layer=blk_layer):
                    rec["x_blk_raw"] = inputs[0].detach() if inputs else None
                    rec["blk_out"] = outputs.detach()
                    rec["blk_module"] = layer

                handles.append(blk_layer.register_forward_hook(_blk_hook))

            records.append(rec)

        return records, handles, decoder_cache

    def _activation_derivative(self, rec: dict[str, Any]) -> torch.Tensor:
        v_n = rec.get("v_n")
        if not isinstance(v_n, torch.Tensor):
            return torch.empty(0)
        layer = rec["layer"]
        reactivation = getattr(layer, "reactivation", None)
        v_out = rec.get("v_out")
        if reactivation is None or isinstance(reactivation, nn.Identity):
            return torch.ones_like(v_n)
        if isinstance(reactivation, nn.ReLU):
            return (v_n > 0).to(dtype=v_n.dtype)
        if isinstance(reactivation, nn.Sigmoid):
            y = v_out if isinstance(v_out, torch.Tensor) else torch.sigmoid(v_n)
            return y * (1.0 - y)
        if isinstance(reactivation, nn.Tanh):
            y = v_out if isinstance(v_out, torch.Tensor) else torch.tanh(v_n)
            return 1.0 - y.square()
        if isinstance(reactivation, (ParametricTanh, ParametricTanhOnlyM)):
            y = v_out if isinstance(v_out, torch.Tensor) else reactivation(v_n)
            m = reactivation.log_m.detach().exp().to(device=v_n.device, dtype=v_n.dtype)
            return 0.5 * m.unsqueeze(0) * (1.0 - (2.0 * y - 1.0).square())
        return torch.ones_like(v_n)

    def _total_conductance(
        self, rec: dict[str, Any], v_n: torch.Tensor
    ) -> torch.Tensor:
        layer = rec["layer"]
        if not bool(getattr(layer, "use_shunting", False)):
            return torch.ones_like(v_n)

        g_tot = torch.ones_like(v_n)
        exc_out = rec.get("exc_out")
        if isinstance(exc_out, torch.Tensor):
            g_tot = g_tot + functional.relu(exc_out)
        inh_out = rec.get("inh_out")
        if isinstance(inh_out, torch.Tensor):
            g_tot = g_tot + functional.relu(inh_out)
        blk_module = rec.get("blk_module")
        if blk_module is not None and hasattr(blk_module, "sum_conductances"):
            g_blk = blk_module.sum_conductances().detach()[None, :].expand_as(v_n)
            g_tot = g_tot + functional.relu(g_blk)
        return g_tot

    def _active_presynaptic_values(
        self,
        x: torch.Tensor,
        layer: TopKLinear,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if x.dim() != 2:
            x = x.reshape(x.size(0), -1)
        if x.size(1) != layer.in_features:
            return None, None
        if mask is None:
            mask = layer.weight_mask().detach()
        mask = mask.to(device=x.device)
        active_counts = mask.gt(0).sum(dim=1)
        if int(active_counts.max().item()) <= 0:
            return None, None
        k = int(active_counts.max().item())
        active_idx = torch.topk(mask, k=k, dim=1).indices
        active_mask = mask.gather(dim=1, index=active_idx).gt(0)
        active_values = x[:, active_idx]
        return active_values, active_mask

    def _add_component(
        self,
        accum: dict[_ComponentKey, list[torch.Tensor]],
        key: _ComponentKey,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(values, torch.Tensor) or values.numel() == 0:
            return
        if mask is not None:
            if mask.dim() == values.dim() - 1:
                mask = mask.unsqueeze(0).expand_as(values)
            elif mask.shape != values.shape:
                mask = torch.broadcast_to(mask, values.shape)
            values = values[mask.to(device=values.device)]
        if values.numel() == 0:
            return
        accum.setdefault(key, []).append(values.detach().cpu())

    def _base_key(
        self,
        rec: dict[str, Any],
        depth_labels: dict[int, str],
        synapse_type: str,
        component: str,
        used_in_rule: bool,
    ) -> _ComponentKey:
        depth = int(rec.get("dendritic_depth", -1))
        return _ComponentKey(
            module_name=str(rec.get("module_name", "")),
            population=str(rec.get("population", "unknown")),
            branch_module_index=int(rec.get("branch_module_index", -1)),
            dendritic_depth=depth,
            depth_label=depth_labels.get(depth, "unknown"),
            synapse_type=synapse_type,
            component=component,
            used_in_rule=bool(used_in_rule),
        )

    def _compute_delta_out(
        self, y_hat: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor | None:
        mode = str(getattr(self.params, "error_mode", "auto")).lower()
        if mode == "none":
            return None
        if mode == "mse" or (
            mode == "auto" and y.shape == y_hat.shape and y_hat.dtype.is_floating_point
        ):
            return y_hat - y.to(device=y_hat.device, dtype=y_hat.dtype)
        if mode == "ce" and y.shape == y_hat.shape:
            return functional.softmax(y_hat, dim=-1) - y.to(
                device=y_hat.device, dtype=y_hat.dtype
            )
        if y_hat.dim() == 2 and (y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1)):
            y_idx = y.view(-1).to(device=y_hat.device, dtype=torch.long)
            if y_idx.numel() == y_hat.size(0) and int(y_idx.max().item()) < y_hat.size(
                1
            ):
                y_onehot = functional.one_hot(y_idx, num_classes=y_hat.size(1)).to(
                    y_hat.dtype
                )
                return functional.softmax(y_hat, dim=-1) - y_onehot
        if y.shape == y_hat.shape:
            return y_hat - y.to(device=y_hat.device, dtype=y_hat.dtype)
        return None

    def _compute_soma_error(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        decoder_cache: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        delta_out = self._compute_delta_out(y_hat, y)
        if delta_out is None:
            return None

        decoder_input = decoder_cache.get("decoder_input")
        if isinstance(decoder_input, torch.Tensor) and decoder_input.requires_grad:
            try:
                grad = torch.autograd.grad(
                    outputs=y_hat,
                    inputs=decoder_input,
                    grad_outputs=delta_out,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if isinstance(grad, torch.Tensor):
                    return grad.detach()
            except Exception as exc:
                self.logger.debug("Could not map output error to soma space: %s", exc)
        return delta_out.detach()

    def _layer_broadcast(
        self,
        delta0: torch.Tensor | None,
        out_features: int,
    ) -> torch.Tensor | None:
        if delta0 is None:
            return None
        scalar = _reduce_error_to_scalar(delta0)
        mode = str(getattr(self.params, "broadcast_mode", "scalar")).lower()
        if (
            mode in {"per_soma", "neuron_wise", "neuron-wise"}
            and delta0.dim() == 2
            and delta0.size(1) == out_features
        ):
            return delta0
        return scalar.expand(-1, out_features)

    def _record_common_components(
        self,
        accum: dict[_ComponentKey, list[torch.Tensor]],
        rec: dict[str, Any],
        depth_labels: dict[int, str],
        v_n: torch.Tensor,
        v_out: torch.Tensor | None,
        g_tot: torch.Tensor,
        r_tot: torch.Tensor,
        activation_derivative: torch.Tensor,
        e_n: torch.Tensor | None,
    ) -> None:
        for component, values in [
            ("branch_voltage", v_n),
            ("total_conductance", g_tot),
            ("input_resistance", r_tot),
            ("activation_derivative", activation_derivative),
        ]:
            self._add_component(
                accum,
                self._base_key(rec, depth_labels, "branch", component, True),
                values,
            )
        if isinstance(v_out, torch.Tensor):
            self._add_component(
                accum,
                self._base_key(
                    rec, depth_labels, "branch", "post_reactivation_activity", True
                ),
                v_out,
            )
        if isinstance(e_n, torch.Tensor):
            self._add_component(
                accum,
                self._base_key(rec, depth_labels, "branch", "broadcast_error", True),
                e_n,
            )

    def _record_synaptic_components(
        self,
        accum: dict[_ComponentKey, list[torch.Tensor]],
        rec: dict[str, Any],
        depth_labels: dict[int, str],
        synapse_type: str,
        x_key: str,
        module_key: str,
        mask_key: str,
        current_key: str,
        e_rev: float,
        sign_additive: float,
        v_n: torch.Tensor,
        r_tot: torch.Tensor,
        activation_derivative: torch.Tensor,
        e_n: torch.Tensor | None,
    ) -> None:
        layer = rec["layer"]
        x = rec.get(x_key)
        syn_layer = rec.get(module_key)
        if not isinstance(x, torch.Tensor) or not isinstance(syn_layer, TopKLinear):
            return
        active_x, active_mask = self._active_presynaptic_values(
            x, syn_layer, rec.get(mask_key)
        )
        if active_x is None or active_mask is None:
            return

        driving = float(e_rev) - v_n
        conductance_eligibility = active_x * r_tot.unsqueeze(-1) * driving.unsqueeze(-1)
        rule_eligibility = (
            conductance_eligibility
            if bool(getattr(layer, "use_shunting", False))
            else float(sign_additive) * active_x
        )
        current = rec.get(current_key)

        component_items: list[tuple[str, torch.Tensor, bool]] = [
            ("presynaptic_drive", active_x, True),
            ("driving_force", driving, bool(getattr(layer, "use_shunting", False))),
            ("driving_force_abs", driving.abs(), False),
            ("conductance_form_eligibility", conductance_eligibility, True),
            ("rule_eligibility", rule_eligibility, True),
        ]
        if isinstance(current, torch.Tensor):
            component_items.append(("synaptic_conductance_current", current, False))
        if isinstance(e_n, torch.Tensor):
            e_v = e_n * activation_derivative
            local_update = rule_eligibility * e_v.unsqueeze(-1)
            component_items.append(("local_update_factor", local_update, True))

        for component, values, used in component_items:
            mask = active_mask if values.dim() == 3 else None
            self._add_component(
                accum,
                self._base_key(rec, depth_labels, synapse_type, component, used),
                values,
                mask=mask,
            )

    def _record_dendritic_components(
        self,
        accum: dict[_ComponentKey, list[torch.Tensor]],
        rec: dict[str, Any],
        depth_labels: dict[int, str],
        v_n: torch.Tensor,
        r_tot: torch.Tensor,
        activation_derivative: torch.Tensor,
        e_n: torch.Tensor | None,
    ) -> None:
        if not bool(getattr(self.params, "include_dendritic", True)):
            return
        blk_layer = rec.get("blk_module")
        x_blk_raw = rec.get("x_blk_raw")
        if blk_layer is None or not isinstance(x_blk_raw, torch.Tensor):
            return
        if not hasattr(blk_layer, "block_size") or not hasattr(
            blk_layer, "out_features"
        ):
            return
        block_size = int(blk_layer.block_size)
        out_features = int(blk_layer.out_features)
        try:
            x_blk = x_blk_raw.reshape(x_blk_raw.size(0), out_features, block_size)
        except Exception:
            return

        layer = rec["layer"]
        diff = x_blk - v_n.unsqueeze(-1)
        conductance_eligibility = r_tot.unsqueeze(-1) * diff
        rule_eligibility = (
            conductance_eligibility
            if bool(getattr(layer, "use_shunting", False))
            else x_blk
        )
        component_items: list[tuple[str, torch.Tensor, bool]] = [
            ("child_branch_activity", x_blk, True),
            (
                "dendritic_driving_force",
                diff,
                bool(getattr(layer, "use_shunting", False)),
            ),
            ("conductance_form_eligibility", conductance_eligibility, True),
            ("rule_eligibility", rule_eligibility, True),
        ]
        if isinstance(e_n, torch.Tensor):
            e_v = e_n * activation_derivative
            component_items.append(
                ("local_update_factor", rule_eligibility * e_v.unsqueeze(-1), True)
            )
        for component, values, used in component_items:
            self._add_component(
                accum,
                self._base_key(rec, depth_labels, "dendritic", component, used),
                values,
            )

    def _records_to_rows(
        self,
        accum: dict[_ComponentKey, list[torch.Tensor]],
        filename: str,
        training: bool,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, chunks in sorted(
            accum.items(),
            key=lambda kv: (
                kv[0].module_name,
                kv[0].population,
                kv[0].branch_module_index,
                kv[0].dendritic_depth,
                kv[0].synapse_type,
                kv[0].component,
            ),
        ):
            if not chunks:
                continue
            values = torch.cat([chunk.reshape(-1) for chunk in chunks], dim=0)
            stats = _summary(values)
            row = {
                "filename": filename,
                "training": bool(training),
                "module_name": key.module_name,
                "population": key.population,
                "branch_module_index": key.branch_module_index,
                "dendritic_depth": key.dendritic_depth,
                "depth_label": key.depth_label,
                "synapse_type": key.synapse_type,
                "component": key.component,
                "used_in_rule": key.used_in_rule,
                **stats,
            }
            rows.append(row)
        return rows

    def _write_csv(self, rows: list[dict[str, Any]], path: str) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _append_csv(self, rows: list[dict[str, Any]], path: str) -> None:
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        fieldnames = list(rows[0].keys())
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset | None = None,
        train_ds: torch.utils.data.Dataset | None = None,
        valid_ds: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        dataset = train_ds if training else test_dataset
        if dataset is None:
            dataset = valid_ds or test_dataset or train_ds
        if dataset is None:
            self.logger.debug(
                "No dataset available; skipping LocalRuleComponentAnalyzer"
            )
            return {}

        accum: dict[_ComponentKey, list[torch.Tensor]] = {}
        with analysis_device_context(model, device) as current_device:
            topk_modules = [m for m in model.modules() if isinstance(m, TopKLinear)]
            for module in topk_modules:
                module.cache_mask = True
            try:
                for batch in iter_analysis_batches(
                    dataset,
                    runtime,
                    explicit_max_samples=getattr(self.params, "max_samples", 1024),
                    device=current_device,
                ):
                    if not batch:
                        continue
                    x = batch[0].to(current_device)
                    y = batch[1].to(current_device) if len(batch) > 1 else None

                    records, handles, decoder_cache = self._attach_recorders(model)
                    try:
                        if (
                            getattr(self.params, "include_broadcast", True)
                            and y is not None
                        ):
                            with torch.enable_grad():
                                y_hat = model(x)
                                delta0 = self._compute_soma_error(
                                    y_hat, y, decoder_cache
                                )
                        else:
                            with torch.no_grad():
                                _ = model(x)
                            delta0 = None
                    finally:
                        for handle in handles:
                            handle.remove()

                    depth_labels = self._depth_labels(records)
                    for rec in records:
                        v_n = rec.get("v_n")
                        if not isinstance(v_n, torch.Tensor):
                            continue
                        v_out = rec.get("v_out")
                        g_tot = self._total_conductance(rec, v_n)
                        r_tot = 1.0 / (g_tot + 1e-8)
                        activation_derivative = self._activation_derivative(rec)
                        e_n = self._layer_broadcast(delta0, v_n.size(1))
                        if isinstance(e_n, torch.Tensor):
                            e_n = e_n.to(device=v_n.device, dtype=v_n.dtype)

                        self._record_common_components(
                            accum,
                            rec,
                            depth_labels,
                            v_n,
                            v_out if isinstance(v_out, torch.Tensor) else None,
                            g_tot,
                            r_tot,
                            activation_derivative,
                            e_n,
                        )
                        self._record_synaptic_components(
                            accum,
                            rec,
                            depth_labels,
                            synapse_type="excitatory",
                            x_key="x_exc",
                            module_key="exc_module",
                            mask_key="exc_mask",
                            current_key="exc_out",
                            e_rev=_safe_float(getattr(self.params, "e_rev_exc", 1.0)),
                            sign_additive=1.0,
                            v_n=v_n,
                            r_tot=r_tot,
                            activation_derivative=activation_derivative,
                            e_n=e_n,
                        )
                        self._record_synaptic_components(
                            accum,
                            rec,
                            depth_labels,
                            synapse_type="inhibitory",
                            x_key="x_inh",
                            module_key="inh_module",
                            mask_key="inh_mask",
                            current_key="inh_out",
                            e_rev=_safe_float(getattr(self.params, "e_rev_inh", 0.0)),
                            sign_additive=-1.0,
                            v_n=v_n,
                            r_tot=r_tot,
                            activation_derivative=activation_derivative,
                            e_n=e_n,
                        )
                        self._record_dendritic_components(
                            accum,
                            rec,
                            depth_labels,
                            v_n,
                            r_tot,
                            activation_derivative,
                            e_n,
                        )
            finally:
                for module in topk_modules:
                    module.cache_mask = False
                    module._last_forward_weight_mask = None

        rows = self._records_to_rows(accum, filename=filename, training=training)
        snapshot = {
            "filename": filename,
            "training": bool(training),
            "n_rows": len(rows),
            "rows": rows,
        }

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            if bool(getattr(self.params, "save_json", True)):
                json_dir = os.path.join(save_path, "epochs" if training else "final")
                save_dict(snapshot, json_dir, f"{filename}.json")
            if bool(getattr(self.params, "save_csv", True)):
                csv_dir = os.path.join(save_path, "epochs" if training else "final")
                self._write_csv(rows, os.path.join(csv_dir, f"{filename}.csv"))
            if training and bool(getattr(self.params, "append_training_csv", True)):
                self._append_csv(rows, os.path.join(save_path, "trajectory.csv"))

        return snapshot


__all__ = ["LocalRuleComponentAnalyzer"]
