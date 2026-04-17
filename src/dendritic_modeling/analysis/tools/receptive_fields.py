import os
import re
import tqdm
from functools import partial
from typing import Optional, Any
from copy import deepcopy

import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import ReceptiveFieldAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    DendriNet,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class ReceptiveFieldAnalyzer(AbstractAnalyzer):
    """
    Analyzer for visualizing receptive fields of the model.
    """

    def __init__(self, params: ReceptiveFieldAnalysisParams):
        super().__init__("ReceptiveFieldAnalyzer")

        exc_rf_shape: Optional[tuple[int, int]] = getattr(params, "rf_shape", None)
        if exc_rf_shape is not None:
            exc_rf_shape = tuple(exc_rf_shape)
        self.exc_rf_shape = exc_rf_shape

    @staticmethod
    def _max_rq_optimization(
        cov: torch.Tensor,
        pc_1: torch.Tensor,
        lr: float = 1.0,
        max_iter: int = 1000,
        patience: int = 10,
    ):
        """Enforce positivity constrain, maximize RQ, and return the optimized vector."""
        with torch.enable_grad():
            v_init = pc_1.clone()[:,None]
            v_init = torch.where(v_init < 1e-2, 0.1, v_init)
            pre_v_init = torch.log(v_init.exp() - 1)
            pre_v_init = torch.zeros(
                (cov.shape[0], 1), 
                device=cov.device, 
                dtype=cov.dtype
            )

            pre_v = torch.nn.Parameter(pre_v_init, requires_grad=True)
            optimizer = torch.optim.Adam([pre_v], lr=lr)

            best_pre_v = deepcopy(pre_v.data)
            best_rq = -float("inf")
            patience_counter = 0

            for _ in range(max_iter):
                optimizer.zero_grad()
                v = F.softplus(pre_v)
                rq = (v.T @ cov @ v) / (v.T @ v)
                rq_loss = -1 * rq
                rq_loss.backward()
                torch.nn.utils.clip_grad_value_(pre_v, 5.0)
                optimizer.step()

                if rq > best_rq:
                    best_rq = rq
                    best_pre_v = deepcopy(pre_v.data)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            return F.softplus(best_pre_v).squeeze(1)
    
    @staticmethod
    def _max_rq_power_iteration(
        cov: torch.Tensor,
        tol: float = 1e-6,
        max_iter: int = 1000,
        patience: int = 10,
    ):
        """Enforce positivity constrain, maximize RQ, and return the optimized vector."""
        w_prev = torch.randn(
            cov.shape[0], 1, 
            device=cov.device, 
            dtype=cov.dtype
        )
        v_prev = w_prev / w_prev.norm(p=2)

        patience_counter = 0

        for _ in range(max_iter):
            w_next = torch.relu(cov @ v_prev)
            v_next = w_next / w_next.norm(p=2)

            if (v_next - v_prev).abs().max() > tol:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            v_prev = v_next
        
        return v_prev.squeeze(1)

    def _compute_receptive_field_metrics(
        self,
        weights: torch.Tensor,
        weight_mask: torch.Tensor,
        inputs: torch.Tensor,
        input_mask: torch.Tensor,
        branch_data: dict[str, float | tuple[int, int, int]],
        prefix: str = "exc_pref"
    ):
        # syn = weights[weight_mask]
        # x = inputs[input_mask][:,weight_mask]
        syn = weights
        x = inputs[input_mask]

        x_mean = x.mean(dim=0)
        x_sim_mean = F.cosine_similarity(x_mean, syn, dim=0)

        cov = torch.cov(x.T, correction=0)
        L, V = torch.linalg.eigh(cov)
        eig_mask = L > 1e-6
        L : torch.Tensor = L[eig_mask]
        V : torch.Tensor = V[:,eig_mask]
        
        cos_sim = F.cosine_similarity(syn[None,:], V.T, dim=1)
        
        # max_rq_pc = self._max_rq_optimization(cov, V[:,-1])
        max_rq_pc = self._max_rq_power_iteration(cov)
        max_rq_sim = F.cosine_similarity(max_rq_pc, syn, dim=0)

        branch_data[f"{prefix}_sim_mean"] = x_sim_mean.item()
        branch_data[f"{prefix}_sim_max_rq"] = max_rq_sim.item()
        branch_data[f"{prefix}_sim_pc_1"] = cos_sim[-1].item()
        branch_data[f"{prefix}_sim_pc_all"] = ((L / L.sum()) * cos_sim).sum().item()
        # branch_data[f"{prefix}_active"] = (x_mean > 0).float().mean().item()
        branch_data[f"{prefix}_active"] = (
            (x_mean > 0).float() * weight_mask.float()
        ).sum() / weight_mask.float().sum().item()
        return branch_data

    def forward_hook(
        self, 
        module: DendriticBranchLayer, 
        input: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], 
        output: torch.Tensor,
        labels: torch.Tensor,
        data_dict: dict[int, dict[str, float | tuple[int, int, int]]],
    ):
        """Forward hook to collect receptive fields."""
        excitatory_input, inhibitory_input, _ = input

        exc_weight = module.branch_excitation.pruned_weight()
        exc_mask = module.branch_excitation.weight_mask().bool()

        if module.branch_inhibition is not None:
            inh_weight = module.branch_inhibition.pruned_weight()
            inh_mask = module.branch_inhibition.weight_mask().bool()

        branches_per_soma = int(module.n_branches // self.n_soma)

        depth = module.layer_idx
        for b in range(module.n_branches):
            soma_idx = int(b // branches_per_soma)
            branch_idx = int(b % branches_per_soma)
            branch_location = (soma_idx, depth, branch_idx)
            
            branch_data = {"location": branch_location}
            input_mask = labels == soma_idx

            branch_data = self._compute_receptive_field_metrics(
                weights=exc_weight[b],
                weight_mask=exc_mask[b],
                inputs=excitatory_input,
                input_mask=input_mask,
                branch_data=branch_data,
                prefix="exc_pref"
            )
            branch_data = self._compute_receptive_field_metrics(
                weights=exc_weight[b],
                weight_mask=exc_mask[b],
                inputs=excitatory_input,
                input_mask=~input_mask,
                branch_data=branch_data,
                prefix="exc_unpref"
            )

            if module.branch_inhibition is not None:
                branch_data = self._compute_receptive_field_metrics(
                    weights=inh_weight[b],
                    weight_mask=inh_mask[b],
                    inputs=inhibitory_input,
                    input_mask=input_mask,
                    branch_data=branch_data,
                    prefix="inh_pref"
                )
                branch_data = self._compute_receptive_field_metrics(
                    weights=inh_weight[b],
                    weight_mask=inh_mask[b],
                    inputs=inhibitory_input,
                    input_mask=~input_mask,
                    branch_data=branch_data,
                    prefix="inh_unpref"
                )

            data_dict[len(data_dict)] = branch_data

            self._pbar.update(1)

    def attach_forward_hooks(
        self, 
        model: BaseModel, 
        labels: torch.Tensor
    ):
        """Attach forward hooks to the model."""
        core_network: ExcitationInhibitionNetwork = model.core_network
        exc_dendrinet: DendriNet = core_network.layers[0].excitatory_cells
        self.n_soma = exc_dendrinet.n_soma

        exc_cell_data: dict[int, dict[str, float | tuple[int, int, int]]] = {}
        handles = []
        for dbl in exc_dendrinet.branch_layers:
            if dbl.branch_excitation is not None and dbl.branch_inhibition is not None:
                handle = dbl.register_forward_hook(
                    partial(self.forward_hook, labels=labels, data_dict=exc_cell_data)
                )
                handles.append(handle)
        
        # inh_dendrinet: DendriNet = core_network.layers[0].inhibitory_cells
        # inh_cell_data: dict[int, dict[str, float | tuple[int, int, int]]] = {}
        # if isinstance(inh_dendrinet, DendriNet):
        #     for dbl in inh_dendrinet.branch_layers:
        #         if dbl.branch_excitation is not None:
        #             handle = dbl.register_forward_hook(
        #                 partial(self.forward_hook, labels=labels, data_dict=inh_cell_data)
        #             )
        #             handles.append(handle)

        # return handles, exc_cell_data, inh_cell_data
        return handles, exc_cell_data

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def collect_receptive_fields(self, ei_layer: ExcitationInhibitionLayer):
        """Collect receptive fields from the model."""
        rfs_dict: dict[str, dict[int, dict[str, Any]]] = {"exc_cells": {}}

        def _rfs_from_dendrinet(dendrinet: DendriNet, cell_type: str):
            branch_counter = 0
            for dbl in dendrinet.branch_layers:
                dbl: DendriticBranchLayer
                if dbl.branch_excitation is not None or dbl.branch_inhibition is not None:
                    if dbl.branch_excitation is not None:
                        exc_weight = dbl.branch_excitation.pruned_weight()
                    if dbl.branch_inhibition is not None:
                        inh_weight = dbl.branch_inhibition.pruned_weight()
            
                    branches_per_soma = int(dbl.n_branches // self.n_soma)

                    depth = dbl.layer_idx
                    for b in range(dbl.n_branches):
                        soma_idx = int(b // branches_per_soma)
                        branch_idx = int(b % branches_per_soma)
                        branch_location = (soma_idx, depth, branch_idx)

                        branch_rfs = {"location": branch_location}
                        
                        if dbl.branch_excitation is not None:
                            if self.exc_rf_shape is not None:
                                exc_rf = exc_weight[b].reshape(self.exc_rf_shape)
                            else:
                                exc_rf = exc_weight[b][:,None]
                            branch_rfs["exc_rf"] = exc_rf.detach().cpu().numpy()
                        
                        if dbl.branch_inhibition is not None:
                            if self.inh_rf_shape is not None:
                                inh_rf = inh_weight[b].reshape(self.inh_rf_shape)
                            else:
                                inh_rf = inh_weight[b][:,None]
                            branch_rfs["inh_rf"] = inh_rf.detach().cpu().numpy()

                        rfs_dict[cell_type][branch_counter] = branch_rfs
                        branch_counter += 1


        exc_dendrinet: DendriNet = ei_layer.excitatory_cells
        _rfs_from_dendrinet(exc_dendrinet, "exc_cells")

        inh_dendrinet: DendriNet = ei_layer.inhibitory_cells
        if isinstance(inh_dendrinet, DendriNet):
            rfs_dict["inh_cells"] = {}
            _rfs_from_dendrinet(inh_dendrinet, "inh_cells")

        return rfs_dict

    @staticmethod
    def _layer_aggregate_branch_data(
        data_dict: dict[int, dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Aggregate branch data to layer level."""
        layer_data: dict[str, dict[str, list[float]]] = {}

        max_depth = data_dict[0]["location"][1]
        keys = list(data_dict[0].keys())
        keys.remove("location")

        for depth in range(max_depth, -1, -1):
            layer_data[f"depth_{depth}"] = {}
            for key in keys:
                layer_data[f"depth_{depth}"][key] = []

        for branch_data in data_dict.values():
            depth = branch_data["location"][1]
            for key in keys:
                layer_data[f"depth_{depth}"][key].append(branch_data[key])
        
        return layer_data

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
    ):
        """
        Analyze receptive fields in the model.
        """
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return
        
        ei_layer = model.core_network.layers[0]
        
        if ei_layer.inhibitory_cells is not None:
            self.inh_rf_shape = None
        else:
            self.inh_rf_shape = self.exc_rf_shape
        
        n_total = 0
        # for module in ei_layer.modules():
        #     if (
        #         isinstance(module, DendriticBranchLayer)
        #         and (module.branch_excitation is not None
        #         or module.branch_inhibition is not None)
        #     ):
        #         n_total += module.n_branches
        for dbl in ei_layer.excitatory_cells.branch_layers:
            dbl: DendriticBranchLayer
            if (
                dbl.branch_excitation is not None
                and dbl.branch_inhibition is not None
            ):
                n_total += dbl.n_branches

        model.to(device)
        x, labels = test_dataset[:]
        x: torch.Tensor = x.to(device)
        labels: torch.Tensor = labels.to(device)

        # handles, exc_cell_data, inh_cell_data = self.attach_forward_hooks(model, labels)
        handles, exc_cell_data = self.attach_forward_hooks(model, labels)

        self._pbar = tqdm.tqdm(
            total=n_total, 
            desc="Computing receptive field metrics...", 
            unit=" branches"
        )
        with torch.no_grad():
            _ = model(x)

        self.remove_forward_hooks(handles)
        self._pbar.close()
        del self._pbar

        layer_data = {}
        layer_data["exc_cells"] = self._layer_aggregate_branch_data(exc_cell_data)
        # if inh_cell_data is not None:
        #     layer_data["inh_cells"] = self._layer_aggregate_branch_data(inh_cell_data)

        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")
         
            save_dict(layer_data, save_path, f"{filename}.json")

        if not training:
            rfs_dict = self.collect_receptive_fields(ei_layer)
            _plot_receptive_fields(rfs_dict, "exc", "exc", save_path)
            _plot_receptive_fields(rfs_dict, "exc", "inh", save_path)
            _plot_receptive_fields(rfs_dict, "inh", "exc", save_path)

            # for mkey in (
            #     "sim_mean", "sim_max_rq", "sim_pc_1", "sim_pc_all", "silent"
            # ):
            #     _plot_receptive_fields_layer_metric_quantiles(
            #         layer_data["exc_cells"], "exc", "exc", mkey, save_path
            #     )
            #     _plot_receptive_fields_layer_metric_quantiles(
            #         layer_data["exc_cells"], "exc", "inh", mkey, save_path
            #     )
                # if inh_cell_data is not None:
                #     _plot_receptive_fields_layer_metric_quantiles(
                #         layer_data["inh_cells"], "inh", "exc", mkey, save_path
                #     )
                ### DELETE INH CELLS FROM METRIC ANALYSIS, THEY DO NOT HAVE PREFERRED/UNPREFERRED

            _plot_receptive_fields_layer_metric_ratios(
                layer_data["exc_cells"], "exc", save_path
            )


def _plot_receptive_fields(
    rfs_dict: dict[str, dict[int, dict[str, tuple[int, int, int] | torch.Tensor]]], 
    cell_type: str,
    synapse_type: str,
    save_path: str
):
    """Plot receptive fields."""
    if f"{cell_type}_cells" not in rfs_dict:
        return
    
    cell_rfs_dict = rfs_dict[f"{cell_type}_cells"]
    n_total_branches = len(cell_rfs_dict)
    
    n_rows = 1
    n_cols = 1

    while n_rows * n_cols < n_total_branches:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for i, branch_rfs in enumerate(cell_rfs_dict.values()):
        ax = axes[i]
        
        soma_idx, depth, branch_idx = branch_rfs["location"]
        rf = branch_rfs[f"{synapse_type}_rf"]

        sns.heatmap(rf, ax=ax, cmap="viridis")
        ax.set_title(f"Soma {soma_idx}, Depth {depth}, Branch {branch_idx}")
    
    fig.suptitle(f"{synapse_type.title()} Synapse Receptive Fields on {cell_type.title()} Cells")
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"{cell_type}_cell_{synapse_type}_syn_rfs.png"))
    plt.close(fig)


def _plot_receptive_fields_layer_metric_quantiles(
    layer_data: dict[str, dict[str, list[float]]],
    cell_type: str,
    synapse_type: str,
    metric_key: str,
    save_path: str,
) -> None:
    """
    One figure: preferred vs unpreferred distributions by layer (quantile/box view).

    All layers share one axes; per depth, preferred and unpreferred appear side-by-side
    (hue) with distinct colors. ``metric_key`` is the suffix after the pref/unpref prefix,
    e.g. ``sim_mean``, ``sim_max_rq``, ``active`` → keys ``exc_pref_sim_mean``, etc.
    """
    if cell_type not in ("exc", "inh"):
        logger.warning("cell_type must be 'exc' or 'inh', got %s", cell_type)
        return
    if synapse_type not in ("exc", "inh"):
        logger.warning("synapse_type must be 'exc' or 'inh', got %s", synapse_type)
        return

    pref_key = f"{synapse_type}_pref_{metric_key}"
    unpref_key = f"{synapse_type}_unpref_{metric_key}"

    def _depth_sort_key(k: str) -> tuple[int, str]:
        m = re.match(r"depth_(\d+)$", k)
        if m:
            return (int(m.group(1)), k)
        return (10**9, k)

    sorted_layer_keys = sorted(layer_data.keys(), key=_depth_sort_key)

    rows: list[dict[str, Any]] = []
    for layer_key in sorted_layer_keys:
        m = re.match(r"depth_(\d+)$", layer_key)
        if not m:
            continue
        depth = int(m.group(1))
        ld = layer_data[layer_key]
        if pref_key not in ld or unpref_key not in ld:
            logger.warning(
                "Missing keys %s or %s in %s; skipping layer",
                pref_key,
                unpref_key,
                layer_key,
            )
            continue
        pref_vals = ld[pref_key]
        unpref_vals = ld[unpref_key]
        if not pref_vals and not unpref_vals:
            continue
        for v in pref_vals:
            rows.append(
                {
                    "depth": depth,
                    "depth_label": f"Layer {depth}",
                    "condition": "Preferred",
                    "value": float(v),
                }
            )
        for v in unpref_vals:
            rows.append(
                {
                    "depth": depth,
                    "depth_label": f"Layer {depth}",
                    "condition": "Unpreferred",
                    "value": float(v),
                }
            )

    if not rows:
        logger.warning(
            "No data for quantile plot (cell=%s, syn=%s, metric=%s)",
            cell_type,
            synapse_type,
            metric_key,
        )
        return

    df = pd.DataFrame(rows)
    depth_order = [f"Layer {d}" for d in sorted(df["depth"].unique())]
    df["depth_label"] = pd.Categorical(
        df["depth_label"], categories=depth_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(depth_order)), 6))
    sns.boxplot(
        data=df,
        x="depth_label",
        y="value",
        hue="condition",
        ax=ax,
        palette={"Preferred": "#2E7D9A", "Unpreferred": "#C97C2E"},
        dodge=True,
        linewidth=1.0,
        showfliers=False,
    )
    ax.set_xlabel("Layer (depth)")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    syn_label = "Excitatory" if synapse_type == "exc" else "Inhibitory"
    cell_label = "Excitatory" if cell_type == "exc" else "Inhibitory"
    ax.set_title(
        f"{cell_label} cells, {syn_label} synapses — {metric_key.replace('_', ' ')} "
        "(preferred vs unpreferred)"
    )
    ax.legend(title=None, loc="best")
    fig.tight_layout()

    safe_metric = metric_key.replace("/", "_")
    out_name = (
        f"{cell_type}_cell_{synapse_type}_syn_{safe_metric}_by_layer_quantiles.png"
    )
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, out_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_receptive_fields_layer_metric_ratios(
    layer_data: dict[str, dict[str, list[float]]],
    cell_type: str,
    save_path: str,
) -> None:
    """
    2×2 figure: top row — preferred-only similarities (max RQ, mean, active); bottom row —
    preferred/unpreferred ratios (same hues), with a grey line at y=1 on the bottom row only.
    """
    if cell_type not in ("exc", "inh"):
        logger.warning("cell_type must be 'exc' or 'inh', got %s", cell_type)
        return

    metrics = ("sim_max_rq", "sim_mean", "active")
    # Display labels and colors must match what we store in the ``metric`` column (for seaborn palette).
    _metric_plot_meta: dict[str, tuple[str, str]] = {
        "sim_max_rq": ("Max RQ similarity", "#2E7D9A"),
        "sim_mean": ("Mean similarity", "#6A4C93"),
        "active": ("Active", "#C97C2E"),
    }
    synapse_types = ("exc", "inh")

    def _depth_sort_key(k: str) -> tuple[int, str]:
        m = re.match(r"depth_(\d+)$", k)
        if m:
            return (int(m.group(1)), k)
        return (10**9, k)

    sorted_layer_keys = sorted(layer_data.keys(), key=_depth_sort_key)

    def _pref_rows_for_synapse(synapse_type: str) -> list[dict[str, Any]]:
        rows_local: list[dict[str, Any]] = []
        for layer_key in sorted_layer_keys:
            m = re.match(r"depth_(\d+)$", layer_key)
            if not m:
                continue
            depth = int(m.group(1))
            ld = layer_data[layer_key]
            for metric_key in metrics:
                pref_key = f"{synapse_type}_pref_{metric_key}"
                if pref_key not in ld:
                    logger.warning(
                        "Missing key %s in %s; skipping", pref_key, layer_key
                    )
                    continue
                pref_vals = ld[pref_key]
                if not pref_vals:
                    continue
                metric_label = _metric_plot_meta[metric_key][0]
                for v in pref_vals:
                    rows_local.append(
                        {
                            "depth": depth,
                            "depth_label": f"Layer {depth}",
                            "metric": metric_label,
                            "value": float(v),
                        }
                    )
        return rows_local

    def _ratio_rows_for_synapse(synapse_type: str) -> list[dict[str, Any]]:
        rows_local: list[dict[str, Any]] = []
        for layer_key in sorted_layer_keys:
            m = re.match(r"depth_(\d+)$", layer_key)
            if not m:
                continue
            depth = int(m.group(1))
            ld = layer_data[layer_key]
            for metric_key in metrics:
                pref_key = f"{synapse_type}_pref_{metric_key}"
                unpref_key = f"{synapse_type}_unpref_{metric_key}"
                if pref_key not in ld or unpref_key not in ld:
                    logger.warning(
                        "Missing keys %s or %s in %s; skipping",
                        pref_key,
                        unpref_key,
                        layer_key,
                    )
                    continue
                pref_vals = ld[pref_key]
                unpref_vals = ld[unpref_key]
                if not pref_vals or not unpref_vals:
                    continue
                pref_t = torch.as_tensor(pref_vals, dtype=torch.float32)
                unpref_t = torch.as_tensor(unpref_vals, dtype=torch.float32)
                n = int(min(pref_t.numel(), unpref_t.numel()))
                if n == 0:
                    continue
                pref_t = pref_t[:n]
                unpref_t = unpref_t[:n]
                ratio_t = torch.where(
                    unpref_t.abs() > 1e-8,
                    pref_t / unpref_t,
                    torch.tensor(float("nan"), dtype=torch.float32),
                )
                metric_label = _metric_plot_meta[metric_key][0]
                for v in ratio_t.tolist():
                    if v != v:  # skip NaN from ratio
                        continue
                    rows_local.append(
                        {
                            "depth": depth,
                            "depth_label": f"Layer {depth}",
                            "metric": metric_label,
                            "value": float(v),
                        }
                    )
        return rows_local

    pref_rows_by_syn: dict[str, list[dict[str, Any]]] = {
        st: _pref_rows_for_synapse(st) for st in synapse_types
    }
    ratio_rows_by_syn: dict[str, list[dict[str, Any]]] = {
        st: _ratio_rows_for_synapse(st) for st in synapse_types
    }
    if not any(ratio_rows_by_syn.values()) and not any(pref_rows_by_syn.values()):
        logger.warning(
            "No data for pref similarity / pref-unpref ratio plot (cell=%s)", cell_type
        )
        return

    all_depths: set[int] = set()
    for rows in (*pref_rows_by_syn.values(), *ratio_rows_by_syn.values()):
        for r in rows:
            all_depths.add(int(r["depth"]))
    depth_order = [f"Layer {d}" for d in sorted(all_depths)]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(max(10, 1.4 * len(depth_order) * 2), 10),
        sharex=True,
        sharey="row",
    )
    palette = {
        _metric_plot_meta[m][0]: _metric_plot_meta[m][1] for m in metrics
    }
    row0_titles = ("Excitatory Synapses", "Inhibitory Synapses")

    for col, syn_type in enumerate(synapse_types):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]

        pref_rows = pref_rows_by_syn[syn_type]
        if pref_rows:
            df_pref = pd.DataFrame(pref_rows)
            df_pref["depth_label"] = pd.Categorical(
                df_pref["depth_label"], categories=depth_order, ordered=True
            )
            sns.boxplot(
                data=df_pref,
                x="depth_label",
                y="value",
                hue="metric",
                ax=ax_top,
                palette=palette,
                dodge=True,
                linewidth=1.0,
                showfliers=False,
            )
            ax_top.legend(title=None, loc="best")
        else:
            ax_top.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_top.transAxes)

        ax_top.set_title(row0_titles[col])
        if col == 0:
            ax_top.set_ylabel("Preferred")
        ax_top.set_xlabel("")

        ratio_rows = ratio_rows_by_syn[syn_type]
        if ratio_rows:
            df_ratio = pd.DataFrame(ratio_rows)
            df_ratio["depth_label"] = pd.Categorical(
                df_ratio["depth_label"], categories=depth_order, ordered=True
            )
            sns.boxplot(
                data=df_ratio,
                x="depth_label",
                y="value",
                hue="metric",
                ax=ax_bot,
                palette=palette,
                dodge=True,
                linewidth=1.0,
                showfliers=False,
            )
            ax_bot.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, zorder=0)
            ax_bot.legend(title=None, loc="best")
        else:
            ax_bot.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_bot.transAxes)

        ax_bot.set_xlabel("Layer (depth)")
        if col == 0:
            ax_bot.set_ylabel("Preferred / unpreferred ratio")

    cell_label = "Excitatory" if cell_type == "exc" else "Inhibitory"
    fig.suptitle(
        f"{cell_label} cells — preferred (top) and pref/unpref ratio "
        f"(max RQ sim, mean sim, & active; bottom), exc vs inh synapses",
        y=1.01,
    )
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out_name = f"{cell_type}_cell_pref_unpref_ratio_by_layer_exc_inh_metrics.png"
    fig.savefig(os.path.join(save_path, out_name), dpi=200, bbox_inches="tight")
    plt.close(fig)


