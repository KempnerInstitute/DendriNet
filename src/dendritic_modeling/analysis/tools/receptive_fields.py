import math
import os
import re
import tqdm
from functools import partial
from typing import Optional, Any
from copy import deepcopy

import logging
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import NMF

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
    
    @staticmethod
    def _nmf(
        inputs: torch.Tensor,
        n_components: int,
        max_iter: int = 1000,
        tol: float = 1e-6
    ):
        """Extract NMF components from the input data."""
        device = inputs.device
        dtype = inputs.dtype
        
        X = inputs.detach().cpu().numpy()

        nmf = NMF(n_components=n_components, init="nndsvd", max_iter=max_iter, tol=tol)
        nmf.fit(X)

        components = torch.from_numpy(nmf.components_).to(device, dtype)
        return components
    
    def _compute_receptive_field_metrics(
        self,
        soma_idx: int,
        weights: torch.Tensor,
        weight_mask: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        branch_data: dict[str, float | tuple[int, int, int]],
        prefix: str = "exc"
    ):
        syn = weights[weight_mask]

        unique_classes = torch.unique(labels)
        n_classes = len(unique_classes)

        class_dict: dict[str, dict[str, float]] = {}

        for label in unique_classes:
            input_mask = labels == label
            x = inputs[input_mask][:,weight_mask]

            x_mean = x.mean(dim=0)
            x_sim_mean = F.cosine_similarity(x_mean, syn, dim=0).item()

            cov = torch.cov(x.T, correction=0)

            # max_rq_pc = self._max_rq_optimization(cov, V[:,-1])
            max_rq_pc = self._max_rq_power_iteration(cov)
            max_rq_sim = F.cosine_similarity(max_rq_pc, syn, dim=0).item()

            nmf_components = self._nmf(x, n_components=5, max_iter=5000, tol=1e-3)
            nmf_sim = F.cosine_similarity(nmf_components, syn[None,:], dim=1)
            nmf_sim_mean = nmf_sim.mean().item()
            nmf_sim_max = nmf_sim.max().item()
            nmf_sum_sim = F.cosine_similarity(nmf_components.sum(dim=0), syn, dim=0).item()

            active = (x_mean > 0).float().mean().item()

            class_dict[f"class_{label}"] = {
                "sim_mean": x_sim_mean,
                "sim_max_rq": max_rq_sim,
                "sim_nmf_mean": nmf_sim_mean,
                "sim_nmf_max": nmf_sim_max,
                "sim_sum_nmf": nmf_sum_sim,
                "active": active
            }

        branch_data[prefix] = class_dict
        return branch_data

    # def forward_hook(
    #     self, 
    #     module: DendriticBranchLayer, 
    #     input: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], 
    #     output: torch.Tensor,
    #     labels: torch.Tensor,
    #     data_dict: dict[int, dict[str, float | tuple[int, int, int]]],
    # ):
    #     """Forward hook to collect receptive fields."""
    #     excitatory_input, inhibitory_input, _ = input

    #     exc_weight = module.branch_excitation.pruned_weight()
    #     exc_mask = module.branch_excitation.weight_mask().bool()

    #     if module.branch_inhibition is not None:
    #         inh_weight = module.branch_inhibition.pruned_weight()
    #         inh_mask = module.branch_inhibition.weight_mask().bool()

    #     branches_per_soma = int(module.n_branches // self.n_soma)

    #     depth = module.layer_idx
    #     for b in range(module.n_branches):
    #         soma_idx = int(b // branches_per_soma)
    #         branch_idx = int(b % branches_per_soma)
    #         branch_location = (soma_idx, depth, branch_idx)
            
    #         branch_data = {"location": branch_location}

    #         branch_data = self._compute_receptive_field_metrics(
    #             soma_idx=soma_idx,
    #             weights=exc_weight[b],
    #             weight_mask=exc_mask[b],
    #             inputs=excitatory_input,
    #             labels=labels,
    #             branch_data=branch_data,
    #             prefix="exc"
    #         )

    #         if module.branch_inhibition is not None:
    #             branch_data = self._compute_receptive_field_metrics(
    #                 soma_idx=soma_idx,
    #                 weights=inh_weight[b],
    #                 weight_mask=inh_mask[b],
    #                 inputs=inhibitory_input,
    #                 labels=labels,
    #                 branch_data=branch_data,
    #                 prefix="inh"
    #             )

    #         data_dict[len(data_dict)] = branch_data

    #         self._pbar.update(1)

    def forward_hook(
        self, 
        module: DendriticBranchLayer, 
        input: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], 
        output: torch.Tensor,
        labels: torch.Tensor,
        data_dict: dict[str, dict[str, dict[str, list[float]]]],
    ):
        """Forward hook to collect receptive fields."""
        excitatory_input, inhibitory_input, _ = input

        if self.exc_inputs is None:
            self.exc_inputs = excitatory_input
        if self.inh_inputs is None:
            self.inh_inputs = inhibitory_input

        excitation: torch.Tensor = module.branch_excitation(excitatory_input)
        inhibition: torch.Tensor = module.branch_inhibition(inhibitory_input)

        branches_per_soma = int(module.n_branches // self.n_exc_soma)

        depth = module.layer_idx
        for b in range(module.n_branches):
            soma_idx = int(b // branches_per_soma)
            branch_idx = int(b % branches_per_soma)
            
            # branch_location = (soma_idx, depth, branch_idx)
            location_key = f"soma {soma_idx} - depth {depth} - branch {branch_idx}"

            
            # branch_data = {"location": branch_location}
            branch_data = {}

            for class_idx in range(self.n_classes):
                class_key = f"class_{class_idx}"
                class_mask = labels == class_idx

                exc = excitation[class_mask][:,b]
                inh = inhibition[class_mask][:,b]
                vout = output[class_mask][:,b]

                branch_data[class_key] = {
                    "exc": exc.detach().cpu().tolist(),
                    "inh": inh.detach().cpu().tolist(),
                    "ei_ratio": (exc / (inh + 1e-6)).detach().cpu().tolist(),
                    "vout": vout.detach().cpu().tolist()
                }
            
            # data_dict[len(data_dict)] = branch_data
            data_dict[location_key] = branch_data
            self._pbar.update(1)

    def attach_forward_hooks(
        self, 
        model: BaseModel, 
        labels: torch.Tensor
    ):
        """Attach forward hooks to the model."""
        core_network: ExcitationInhibitionNetwork = model.core_network
        exc_dendrinet: DendriNet = core_network.layers[0].excitatory_cells
        self.n_exc_soma = exc_dendrinet.n_soma

        inh_dendrinet = core_network.layers[0].inhibitory_cells
        self.n_inh_soma = None
        if isinstance(inh_dendrinet, DendriNet):
            self.n_inh_soma = inh_dendrinet.n_soma

        exc_cell_data: dict[int, dict[str, float | tuple[int, int, int]]] = {}
        handles = []
        for dbl in exc_dendrinet.branch_layers:
            if dbl.branch_excitation is not None and dbl.branch_inhibition is not None:
                handle = dbl.register_forward_hook(
                    partial(self.forward_hook, labels=labels, data_dict=exc_cell_data)
                )
                handles.append(handle)
        
        return handles, exc_cell_data

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def collect_receptive_fields(self, ei_layer: ExcitationInhibitionLayer):
        """Collect receptive fields from the model."""

        def _rfs_from_dendrinet(
            dendrinet: DendriNet, 
            cell_type: str, 
            rfs_dict: dict[str, dict[str, Any]]
        ):
            branch_counter = 0
            for dbl in dendrinet.branch_layers:
                dbl: DendriticBranchLayer
                if dbl.branch_excitation is not None or dbl.branch_inhibition is not None:
                    if dbl.branch_excitation is not None:
                        exc_weight = dbl.branch_excitation.pruned_weight()
                    if dbl.branch_inhibition is not None:
                        inh_weight = dbl.branch_inhibition.pruned_weight()

                    if cell_type == "exc":
                        n_soma = self.n_exc_soma
                    elif cell_type == "inh":
                        n_soma = self.n_inh_soma
                    else:
                        raise ValueError(f"Invalid cell type: {cell_type}")

                    branches_per_soma = int(dbl.n_branches // n_soma)

                    depth = dbl.layer_idx
                    for b in range(dbl.n_branches):
                        soma_idx = int(b // branches_per_soma)
                        branch_idx = int(b % branches_per_soma)
                        
                        # branch_location = (soma_idx, depth, branch_idx)
                        location_key = f"soma {soma_idx} - depth {depth} - branch {branch_idx}"

                        # branch_rfs = {"location": branch_location}
                        branch_rfs = {}
                        
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

                        # rfs_dict[branch_counter] = branch_rfs
                        rfs_dict[location_key] = branch_rfs
                        branch_counter += 1


        exc_dendrinet: DendriNet = ei_layer.excitatory_cells
        exc_rfs_dict: dict[str, dict[str, Any]] = {}
        _rfs_from_dendrinet(
            dendrinet=exc_dendrinet,
            cell_type="exc",
            rfs_dict=exc_rfs_dict
        )

        inh_dendrinet: DendriNet = ei_layer.inhibitory_cells
        inh_rfs_dict: Optional[dict[str, dict[str, Any]]] = None
        if isinstance(inh_dendrinet, DendriNet):
            inh_rfs_dict = {}
            _rfs_from_dendrinet(
                dendrinet=inh_dendrinet,
                cell_type="inh",
                rfs_dict=inh_rfs_dict
            )

        return exc_rfs_dict, inh_rfs_dict
    
    @staticmethod
    def _aggregate_rfs_by_soma(
        rfs_dict: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Aggregate receptive fields by soma."""
        agg_rfs_dict: dict[str, dict[str, np.ndarray]] = {}
        
        for location_key, branch_data in rfs_dict.items():
            soma_idx = _location_key_to_tuple(location_key)[0]
            soma_key = f"soma {soma_idx}"
            
            if soma_key not in agg_rfs_dict:
                agg_rfs_dict[soma_key] = {}
            
            if "exc_rf" in branch_data and "exc_rf" not in agg_rfs_dict[soma_key]:
                agg_rfs_dict[soma_key]["exc_rf"] = branch_data["exc_rf"]
            elif "exc_rf" in branch_data:
                agg_rfs_dict[soma_key]["exc_rf"] += branch_data["exc_rf"]

            if "inh_rf" in branch_data and "inh_rf" not in agg_rfs_dict[soma_key]:
                agg_rfs_dict[soma_key]["inh_rf"] = branch_data["inh_rf"]
            elif "inh_rf" in branch_data:
                agg_rfs_dict[soma_key]["inh_rf"] += branch_data["inh_rf"]
        
        n_soma = len(agg_rfs_dict)
        n_branches = len(rfs_dict)
        branches_per_soma = n_branches / n_soma

        for soma_data in agg_rfs_dict.values():
            if "exc_rf" in soma_data:
                soma_data["exc_rf"] = soma_data["exc_rf"] / branches_per_soma
            if "inh_rf" in soma_data:
                soma_data["inh_rf"] = soma_data["inh_rf"] / branches_per_soma
                
        return agg_rfs_dict
    
    @staticmethod
    def _prod_agg_i_cell_e_rf_by_e_cell_i_ref(
        agg_inh_rfs_dict: dict[str, dict[str, np.ndarray]],
        exc_rfs_dict: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Product of aggregated inh cell exc syn RFs by exc cell inh syn RFs."""
        prod_rfs_dict: dict[str, dict[str, np.ndarray]] = {}

        for location_key, branch_rfs in exc_rfs_dict.items():
            e_cell_inh_rf = branch_rfs["inh_rf"]
            i_cell_ixs = e_cell_inh_rf.nonzero()[0].tolist()

            prod_branch_rf = {}
            for i_cell_idx in i_cell_ixs:
                i_cell_key = f"soma {i_cell_idx}"
                i_cell_exc_rf = agg_inh_rfs_dict[i_cell_key]["exc_rf"]

                if "inh_rf" not in prod_branch_rf:
                    prod_branch_rf["inh_rf"] = i_cell_exc_rf
                else:
                    prod_branch_rf["inh_rf"] += i_cell_exc_rf
            
            prod_branch_rf["inh_rf"] = prod_branch_rf["inh_rf"] / len(i_cell_ixs)
            prod_rfs_dict[location_key] = prod_branch_rf
        return prod_rfs_dict
    
    @staticmethod
    def _compute_mean_vector_dict(
        excitatory_inputs: torch.Tensor,
        inhibitory_inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, dict[str, np.ndarray]]:
        """Compute mean vector dictionary."""
        mean_vector_dict: dict[str, dict[str, np.ndarray]] = {}

        exc_mean_vector = excitatory_inputs.mean(dim=0)
        inh_mean_vector = inhibitory_inputs.mean(dim=0)
        mean_vector_dict["total"] = {
            "exc": exc_mean_vector.detach().cpu().numpy(),
            "inh": inh_mean_vector.detach().cpu().numpy()
        }

        for class_idx in torch.unique(labels):
            class_mask = labels == class_idx

            exc_mean_vector = excitatory_inputs[class_mask].mean(dim=0)
            inh_mean_vector = inhibitory_inputs[class_mask].mean(dim=0)
            mean_vector_dict[f"class_{class_idx}"] = {
                "exc": exc_mean_vector.detach().cpu().numpy(),
                "inh": inh_mean_vector.detach().cpu().numpy()
            }

        return mean_vector_dict

    @staticmethod
    def _branch_data_to_preferred(
        data_dict: dict[str, dict[str, dict[str, list[float]]]]
    ) -> dict[str, dict[str, dict[str, list[float]]]]:
        """Convert branch data to preferred class only."""
        pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]] = {}
        for location_key, branch_data in data_dict.items():
            pref_unpref_data[location_key] = {}

            pref_dict: dict[str, list[float]] = {}
            unpref_dict: dict[str, list[float]] = {}

            metrics = list(list(branch_data.values())[-1].keys())
            for metric in metrics:
                pref_dict[metric] = []
                unpref_dict[metric] = []
            
            loc_tuple = _location_key_to_tuple(location_key)
            pref_class_key = f"class_{loc_tuple[0]}"
            for class_key, class_data in branch_data.items():
                if isinstance(class_data, dict):
                    for metric, value_list in class_data.items():
                        if class_key == pref_class_key:
                            pref_dict[metric].extend(value_list)
                        else:
                            unpref_dict[metric].extend(value_list)
            
            pref_unpref_data[location_key]["preferred"] = pref_dict
            pref_unpref_data[location_key]["unpreferred"] = unpref_dict

        return pref_unpref_data
    
    # @staticmethod
    # def _branch_data_to_preferred(
    #     data_dict: dict[int, dict[str, tuple | dict[str, list[float]]]]
    # ) -> dict[int, dict[str, tuple | dict[str, list[float]]]]:
    #     """Convert branch data to preferred class only."""
    #     pref_unpref_data: dict[int, dict[str, tuple | dict[str, list[float]]]] = {}
    #     for branch_idx, branch_data in data_dict.items():
    #         location = branch_data["location"]
    #         pref_unpref_data[branch_idx] = {"location": location}

    #         pref_dict: dict[str, list[float]] = {}
    #         unpref_dict: dict[str, list[float]] = {}

    #         metrics = list(list(branch_data.values())[-1].keys())
    #         for metric in metrics:
    #             pref_dict[metric] = []
    #             unpref_dict[metric] = []
            
    #         pref_class_key = f"class_{location[0]}"
    #         for class_key, class_data in branch_data.items():
    #             if isinstance(class_data, dict):
    #                 for metric, value_list in class_data.items():
    #                     if class_key == pref_class_key:
    #                         pref_dict[metric].extend(value_list)
    #                     else:
    #                         unpref_dict[metric].extend(value_list)
            
    #         pref_unpref_data[branch_idx]["preferred"] = pref_dict
    #         pref_unpref_data[branch_idx]["unpreferred"] = unpref_dict

    #     return pref_unpref_data

    # @staticmethod
    # def _branch_data_to_ratios(
    #     data_dict: dict[int, dict[str, tuple | dict[str, dict[str, float]]]]
    # ) -> dict[int, dict[str, tuple | dict[str, dict[str, float]]]]:
    #     # data_dict[branch_idx][location | synapse_type][class_key][metric]
    #     ratios_dict = deepcopy(data_dict)

    #     for branch_idx, branch_data in data_dict.items():
    #         location = branch_data["location"]
    #         soma_str = f"{location[0]}"

    #         for synapse_type in ("exc", "inh"):
    #             if synapse_type in branch_data:
    #                 synapse_data = branch_data[synapse_type]

    #                 pref_data = synapse_data[f"class_{soma_str}"]
    #                 for class_key, class_data in synapse_data.items():
    #                     for metric, value in class_data.items():
    #                         ratio = pref_data[metric] / value
    #                         ratios_dict[branch_idx][synapse_type][class_key][metric] = ratio

    #     return ratios_dict
    
    # @staticmethod
    # def _layer_aggregate_ratios_data(
    #     ratios_dict: dict[int, dict[str, dict[str, dict[str, float]]]]
    # ) -> dict[str, dict[str, dict[str, list[float]]]]:
    #     """Aggregate ratios data to layer level."""
    #     # ratios_dict[branch_idx][location | synapse_type][class_key][metric]
    #     # layer_data[depth][synapse_type][metric]: list[float]
    #     layer_data: dict[str, dict[str, dict[str, list[float]]]] = {}

    #     max_depth = ratios_dict[0]["location"][1]
    #     synapse_types = list(ratios_dict[0].keys())
    #     synapse_types.remove("location")
    #     metrics = list(ratios_dict[0][synapse_types[0]]["class_0"].keys())

    #     for depth in range(max_depth, 0, -1):
    #         layer_data[f"depth_{depth}"] = {}
    #         for synapse_type in synapse_types:
    #             layer_data[f"depth_{depth}"][synapse_type] = {}
    #             for metric in metrics:
    #                 layer_data[f"depth_{depth}"][synapse_type][metric] = []
        
    #     for branch_data in ratios_dict.values():
    #         location = branch_data["location"]
    #         soma_str = f"{location[0]}"
    #         depth_key = f"depth_{location[1]}"
    #         for synapse_type in synapse_types:
    #             synapse_data = branch_data[synapse_type]
    #             for class_key, class_data in synapse_data.items():
    #                 if class_key != f"class_{soma_str}":
    #                     for metric, value in class_data.items():
    #                         layer_data[depth_key][synapse_type][metric].append(value)
        
    #     return layer_data
    
    # @staticmethod
    # def _layer_soma_aggregate_ratios_data(
    #     ratios_dict: dict[int, dict[str, dict[str, dict[str, float]]]]
    # ) -> dict[str, dict[str, dict[str, dict[str, dict[str, list[float]]]]]]:
    #     """Aggregate ratios data by layer and soma."""
    #     # ratios_dict[branch_idx][location | synapse_type][class_key][metric]
    #     # layer_soma_data[synapse_type][depth_key][soma_key][class_key][metric]: list[float]
    #     layer_soma_data: dict[str, dict[str, dict[str, dict[str, dict[str, list[float]]]]]] = {}

    #     max_depth = ratios_dict[0]["location"][1]
    #     synapse_types = list(ratios_dict[0].keys())
    #     synapse_types.remove("location")
    #     metrics = list(ratios_dict[0][synapse_types[0]]["class_0"].keys())
    #     n_classes = len(ratios_dict[0][synapse_types[0]].keys())

    #     for synapse_type in synapse_types:
    #         layer_soma_data[synapse_type] = {}
    #         for depth in range(max_depth, 0, -1):
    #             depth_key = f"depth_{depth}"
    #             layer_soma_data[synapse_type][depth_key] = {}
    #             for soma_idx in range(n_classes):
    #                 soma_key = f"soma_{soma_idx}"
    #                 layer_soma_data[synapse_type][depth_key][soma_key] = {}
    #                 for class_idx in range(n_classes):
    #                     class_key = f"class_{class_idx}"
    #                     layer_soma_data[synapse_type][depth_key][soma_key][class_key] = {}
    #                     for metric in metrics:
    #                         layer_soma_data[synapse_type][depth_key][soma_key][class_key][metric] = []
        
    #     for branch_data in ratios_dict.values():
    #         location = branch_data["location"]
    #         soma_key = f"soma_{location[0]}"
    #         depth_key = f"depth_{location[1]}"
    #         for synapse_type in synapse_types:
    #             synapse_data = branch_data[synapse_type]
    #             for class_key, class_data in synapse_data.items():
    #                 for metric, value in class_data.items():
    #                     layer_soma_data[synapse_type][depth_key][soma_key][
    #                         class_key
    #                     ][metric].append(value)

    #     return layer_soma_data

    # @staticmethod
    # def _layer_aggregate_branch_data_preferred(
    #     data_dict: dict[int, dict[str, tuple | dict[str, dict[str, float]]]]
    # ) -> dict[str, dict[str, dict[str, list[float]]]]:
    #     """Aggregate branch data to layer level (preferred class only)."""
    #     # data_dict[branch_idx][location | synapse_type][class_key][metric]
    #     # layer_data[depth][synapse_type][metric]: list[float]
    #     layer_data: dict[str, dict[str, dict[str, list[float]]]] = {}

    #     max_depth = data_dict[0]["location"][1]
    #     synapse_types = list(data_dict[0].keys())
    #     synapse_types.remove("location")
    #     metrics = list(data_dict[0][synapse_types[0]]["class_0"].keys())

    #     for depth in range(max_depth, 0, -1):
    #         layer_data[f"depth_{depth}"] = {}
    #         for synapse_type in synapse_types:
    #             layer_data[f"depth_{depth}"][synapse_type] = {}
    #             for metric in metrics:
    #                 layer_data[f"depth_{depth}"][synapse_type][metric] = []
        
    #     for branch_data in data_dict.values():
    #         location = branch_data["location"]
    #         soma_str = f"{location[0]}"
    #         depth_key = f"depth_{location[1]}"
    #         for synapse_type in synapse_types:
    #             synapse_data = branch_data[synapse_type]
    #             for class_key, class_data in synapse_data.items():
    #                 if class_key == f"class_{soma_str}":
    #                     for metric, value in class_data.items():
    #                         layer_data[depth_key][synapse_type][metric].append(value)
        
    #     return layer_data

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

        self.exc_inputs = None
        self.inh_inputs = None

        unique_classes = torch.unique(labels)
        self.n_classes = len(unique_classes)

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

        pref_unpref_data = self._branch_data_to_preferred(exc_cell_data)
        
        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")
         
        if not training:
            
            if save_path is not None:
                activation_plots_path = os.path.join(save_path, "activation_plots")
                
                _plot_exc_vs_inh_by_compartment(
                    pref_unpref_data, activation_plots_path
                )
                _plot_vout_vs_eiratio_by_compartment(
                    pref_unpref_data, activation_plots_path
                )
                _plot_vout_activation_boxplots(
                    exc_cell_data, activation_plots_path
                )

                weight_rfs_path = os.path.join(save_path, "synapse_weight_rfs")
                exc_rfs_dict, inh_rfs_dict = self.collect_receptive_fields(ei_layer)
                _plot_receptive_fields(
                    save_path=weight_rfs_path,
                    rfs_dict=exc_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="exc"
                )
                _plot_receptive_fields(
                    save_path=weight_rfs_path,
                    rfs_dict=exc_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="inh"
                )
                if inh_rfs_dict is not None:
                    _plot_receptive_fields(
                        save_path=weight_rfs_path,
                        rfs_dict=inh_rfs_dict, 
                        cell_type="inh", 
                        synapse_type="exc"
                    )
                
                agg_weight_rfs_path = os.path.join(save_path, "agg_synapse_weight_rfs")
                exc_agg_rfs_dict = self._aggregate_rfs_by_soma(exc_rfs_dict)
                _plot_receptive_fields(
                    save_path=agg_weight_rfs_path,
                    rfs_dict=exc_agg_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="exc"
                )
                _plot_receptive_fields(
                    save_path=agg_weight_rfs_path,
                    rfs_dict=exc_agg_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="inh"
                )
                if inh_rfs_dict is not None:
                    inh_agg_rfs_dict = self._aggregate_rfs_by_soma(inh_rfs_dict)
                    _plot_receptive_fields(
                        save_path=agg_weight_rfs_path,
                        rfs_dict=inh_agg_rfs_dict, 
                        cell_type="inh", 
                        synapse_type="exc"
                    )
                
                mean_vector_dict = self._compute_mean_vector_dict(
                    self.exc_inputs,
                    self.inh_inputs,
                    labels
                )
                syn_act_rfs_path = os.path.join(save_path, "synapse_activation_rfs")
                _plot_receptive_fields(
                    save_path=syn_act_rfs_path,
                    rfs_dict=exc_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="exc",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=False,
                    mean_vector_type="exc"
                )
                _plot_receptive_fields(
                    save_path=syn_act_rfs_path,
                    rfs_dict=exc_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="inh",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=False,
                    mean_vector_type="inh"
                )
                if inh_rfs_dict is not None:
                    _plot_receptive_fields(
                        save_path=syn_act_rfs_path,
                        rfs_dict=inh_rfs_dict, 
                        cell_type="inh", 
                        synapse_type="exc",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=True,
                        mean_vector_type="exc"
                    )
                
                agg_syn_act_rfs_path = os.path.join(save_path, "agg_synapse_activation_rfs")
                _plot_receptive_fields(
                    save_path=agg_syn_act_rfs_path,
                    rfs_dict=exc_agg_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="exc",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=False,
                    mean_vector_type="exc"
                )
                _plot_receptive_fields(
                    save_path=agg_syn_act_rfs_path,
                    rfs_dict=exc_agg_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="inh",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=False,
                    mean_vector_type="inh"
                )
                if inh_rfs_dict is not None:
                    _plot_receptive_fields(
                        save_path=agg_syn_act_rfs_path,
                        rfs_dict=inh_agg_rfs_dict, 
                        cell_type="inh", 
                        synapse_type="exc",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=True,
                        mean_vector_type="exc"
                    )
                
                prod_rfs_dict = self._prod_agg_i_cell_e_rf_by_e_cell_i_ref(
                    agg_inh_rfs_dict=inh_agg_rfs_dict,
                    exc_rfs_dict=exc_rfs_dict
                )
                agg_prod_rfs_dict = self._aggregate_rfs_by_soma(prod_rfs_dict)

                agg_prod_rfs_path = os.path.join(save_path, "agg_prod_synapse_activation_rfs")
                _plot_receptive_fields(
                    save_path=agg_prod_rfs_path,
                    rfs_dict=agg_prod_rfs_dict, 
                    cell_type="inh", 
                    synapse_type="inh",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=True,
                    mean_vector_type="exc"
                )
                _plot_receptive_fields(
                    save_path=agg_prod_rfs_path,
                    rfs_dict=prod_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="inh",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=True,
                    mean_vector_type="exc"
                )

                agg_prod_rfs_path = os.path.join(save_path, "agg_prod_synapse_weight_rfs")
                _plot_receptive_fields(
                    save_path=agg_prod_rfs_path,
                    rfs_dict=agg_prod_rfs_dict, 
                    cell_type="inh", 
                    synapse_type="inh"
                )
                _plot_receptive_fields(
                    save_path=agg_prod_rfs_path,
                    rfs_dict=prod_rfs_dict, 
                    cell_type="exc", 
                    synapse_type="inh"
                )


### NOTES AND TODOs
###   (DONE) Show preferred / unpreferred (or class-conditioned) E vs I plots for each compartment on exc cells.
###   Show class-conditioned vout activation boxplots for each compartment (including soma) on exc and inh cells.
###   Confusion matrix
###   Reliability diagram? (entropy of readout space versus some notion of error, architecture-agnostic visual for "how the model decides")
###   Mean MNIST digit for each class next to aggregated exc receptive field for each readout neuron
###   Sum of exc syn receptive fields for each interneuron
###   Linear weighted sum of inh syn receptive fields for each readout neuron (multiply aggregated interneuron exc RFs by inh syn weights on readout neurons)
###   Aggregate above^ by soma (total interneuron exc RF for each readout neuron)
###   Multiply all receptive fields by class-conditioned mean vectors of inputs (to kill silent synapses)
###   Some visual (for each compartment) showing why some preferred stimuli have E/I ratio < 1 and nonpreferred stimuli have E/I ratio > 1 (think about this HARD AHHHH)


def _location_key_to_tuple(location_key: str) -> tuple[int, int, int]:
    """Convert location key to tuple."""
    loc = location_key.replace("soma", "").replace("depth", "").replace("branch", "")
    loc = loc.replace(" ", "")
    return tuple(int(x) for x in loc.split("-"))


def _plot_receptive_fields(
    save_path: str,
    rfs_dict: dict[str, dict[str, np.ndarray]],
    cell_type: str,
    synapse_type: str,
    mean_vector_dict: Optional[dict[str, dict[str, np.ndarray]]] = None,
    total_mean_vector: Optional[bool] = False,
    mean_vector_type: Optional[str] = None,
):
    """Plot receptive fields."""
    n_total_branches = len(rfs_dict)
    
    n_rows = 1
    n_cols = 1

    while n_rows * n_cols < n_total_branches:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for i, (location_key, branch_rfs) in enumerate(rfs_dict.items()):
        ax = axes[i]
        
        if "depth" in location_key and "branch" in location_key:
            soma_idx = _location_key_to_tuple(location_key)[0]
        elif "soma " in location_key:
            soma_idx = location_key.replace("soma ", "")
        else:
            raise ValueError(f"Invalid location key: {location_key}")
        
        
        
        # soma_idx, depth, branch_idx = branch_rfs["location"]
        rf = branch_rfs[f"{synapse_type}_rf"]

        if mean_vector_dict is not None:
            if total_mean_vector:
                class_key = "total"
            else:
                class_key = f"class_{soma_idx}"
            input_mean_vec = mean_vector_dict[class_key][mean_vector_type]
            shaped_vec = input_mean_vec.reshape(*rf.shape)
            rf = rf * shaped_vec

        sns.heatmap(rf, ax=ax, cmap="viridis")
        # ax.set_title(f"Soma {soma_idx}, Depth {depth}, Branch {branch_idx}")
        ax.set_title(location_key.title())
    
    for j in range(n_total_branches, len(axes)):
        axes[j].set_visible(False)
    
    # fig.suptitle(f"{synapse_type.title()} Synapse Receptive Fields on {cell_type.title()} Cells")
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"{cell_type}_cell_{synapse_type}_syn_rfs.png"))
    plt.close(fig)


def _plot_vout_activation_boxplots(
    data_dict: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
) -> None:
    """Per-branch boxplots of compartment output ``vout`` vs. input class label."""
    if not data_dict:
        return

    branch_keys = list(data_dict.keys())
    n_total_branches = len(branch_keys)

    n_rows = 1
    n_cols = 1
    while n_rows * n_cols < n_total_branches:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    first_bd = data_dict[branch_keys[0]]
    class_order = sorted(
        int(k.split("_")[1])
        for k in first_bd
        if isinstance(first_bd.get(k), dict) and k.startswith("class_")
    )
    if not class_order:
        plt.close(fig)
        return

    for i, bk in enumerate(branch_keys):
        ax = axes_flat[i]
        bd = data_dict[bk]
        # soma_idx, depth, branch_idx = _location_key_to_tuple(bk)

        rows: list[dict[str, int | float]] = []
        for ci in class_order:
            ck = f"class_{ci}"
            if ck not in bd or not isinstance(bd[ck], dict):
                continue
            vals = bd[ck].get("vout", [])
            for v in vals:
                rows.append({"Class": ci, "vout": float(v)})

        if rows:
            df = pd.DataFrame(rows)
            sns.boxplot(
                data=df,
                x="Class",
                y="vout",
                ax=ax,
                order=class_order,
                color="0.88",
            )
        ax.set_xlabel("Class")
        ax.set_ylabel(r"$V_{\mathrm{out}}$")
        ax.set_ylim(0.0, 1.0)
        xticks = [str(c) for c in class_order]
        ax.set_xticks(range(len(class_order)))
        ax.set_xticklabels(xticks)
        # ax.set_title(f"Soma {soma_idx}, Depth {depth}, Branch {branch_idx}")
        ax.set_title(bk.title())

    for j in range(n_total_branches, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        r"$V_{\mathrm{out}}$ by input class (excitatory cell compartments)",
        y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, "exc_vout_activation_by_class_boxplots.png"))
    plt.close(fig)


def _plot_pref_unpref_scatter_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    xvar: str,
    yvar: str,
    x_logscale: bool,
    save_path: str,
    fig_basename: str,
    reference_line: Optional[str] = None,
) -> None:
    """
    One subplot per branch/compartment, same grid heuristic as _plot_receptive_fields.

    ``pref_unpref_data`` is the output of ``_branch_data_to_preferred``: for each
    branch, ``preferred`` / ``unpreferred`` map metric names (``exc``, ``inh``,
    ``ei_ratio``, ``vout``) to per-sample lists. Scatter ``yvar`` vs ``xvar``;
    preferred points in red, unpreferred in black.

    ``reference_line``: optional grey dotted guide — ``"y=x"`` (identity) or
    ``"x=1"`` (vertical).
    """
    if not pref_unpref_data:
        return

    branch_keys = list(pref_unpref_data.keys())
    n_total = len(branch_keys)

    n_rows = 1
    n_cols = 1
    while n_rows * n_cols < n_total:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows)
    )
    axes_flat = np.atleast_1d(axes).ravel()

    x_label = xvar.replace("_", " ").title()
    y_label = yvar.replace("_", " ").title()

    for i, bk in enumerate(branch_keys):
        ax = axes_flat[i]
        bd = pref_unpref_data[bk]
        # soma_idx, depth, branch_idx = _location_key_to_tuple(bk)
        pref = bd["preferred"]
        unpref = bd["unpreferred"]

        xp = np.asarray(pref.get(xvar, []), dtype=float)
        yp = np.asarray(pref.get(yvar, []), dtype=float)
        xu = np.asarray(unpref.get(xvar, []), dtype=float)
        yu = np.asarray(unpref.get(yvar, []), dtype=float)

        mp = np.isfinite(xp) & np.isfinite(yp)
        mu = np.isfinite(xu) & np.isfinite(yu)
        if x_logscale:
            mp = mp & (xp > 0)
            mu = mu & (xu > 0)

        if mu.any():
            ax.scatter(
                xu[mu],
                yu[mu],
                c="black",
                s=12,
                alpha=0.45,
                edgecolors="none",
                label="unpreferred",
            )
        if mp.any():
            ax.scatter(
                xp[mp],
                yp[mp],
                c="red",
                s=12,
                alpha=0.45,
                edgecolors="none",
                label="preferred",
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_logscale:
            ax.set_xscale("log")

        _ref_kw = {"color": "green", "linestyle": "--", "linewidth": 5}
        if reference_line == "y=x":
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            lo = max(xmin, ymin)
            hi = min(xmax, ymax)
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [lo, hi], **_ref_kw)
        elif reference_line == "x=1":
            ax.axvline(1.0, **_ref_kw)

        # ax.set_title(f"Soma {soma_idx}, Depth {depth}, Branch {branch_idx}")
        ax.set_title(bk.title())
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    for j in range(n_total, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, fig_basename))
    plt.close(fig)


def _plot_exc_vs_inh_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
) -> None:
    _plot_pref_unpref_scatter_by_compartment(
        pref_unpref_data,
        xvar="inh",
        yvar="exc",
        x_logscale=False,
        save_path=save_path,
        fig_basename="exc_vs_inh_by_compartment_pref_unpref.png",
        reference_line="y=x",
    )


def _plot_vout_vs_eiratio_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
) -> None:
    _plot_pref_unpref_scatter_by_compartment(
        pref_unpref_data,
        xvar="ei_ratio",
        yvar="vout",
        x_logscale=True,
        save_path=save_path,
        fig_basename="vout_vs_eiratio_by_compartment_pref_unpref.png",
        reference_line="x=1",
    )



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
    layer_pref_data: dict[str, dict[str, dict[str, list[float]]]],
    layer_ratios_data: dict[str, dict[str, dict[str, list[float]]]],
    cell_type: str,
    save_path: str,
) -> None:
    """
    2×2 figure: top row — preferred-class similarities from ``layer_pref_data`` (y in [0, 1]);
    bottom row — pre-aggregated ratios from ``layer_ratios_data`` (same hues), grey line at y=1
    on the bottom row only.

    Expected layout: ``layer_key -> synapse_type ("exc"|"inh") -> metric -> list[float]``.
    """
    if cell_type not in ("exc", "inh"):
        logger.warning("cell_type must be 'exc' or 'inh', got %s", cell_type)
        return

    metrics = (
        "sim_max_rq",
        "sim_nmf_mean",
        "sim_nmf_max",
        "sim_sum_nmf",
        "sim_mean",
        "active",
    )
    # Display labels and colors must match the ``metric`` column (palette + hue_order).
    _metric_plot_meta: dict[str, tuple[str, str]] = {
        "sim_max_rq": ("Max RQ similarity", "#2E7D9A"),
        "sim_nmf_mean": ("NMF mean sim", "#348A98"),
        "sim_nmf_max": ("NMF max sim", "#4A7F90"),
        "sim_sum_nmf": ("NMF sum sim", "#587489"),
        "sim_mean": ("Mean similarity", "#6A4C93"),
        "active": ("Active", "#C97C2E"),
    }
    synapse_types = ("exc", "inh")
    hue_order = [_metric_plot_meta[m][0] for m in metrics]

    def _depth_sort_key(k: str) -> tuple[int, str]:
        m = re.match(r"depth_(\d+)$", k)
        if m:
            return (int(m.group(1)), k)
        return (10**9, k)

    layer_keys = set(layer_pref_data.keys()) | set(layer_ratios_data.keys())
    sorted_layer_keys = sorted(layer_keys, key=_depth_sort_key)

    def _pref_rows_for_synapse(synapse_type: str) -> list[dict[str, Any]]:
        rows_local: list[dict[str, Any]] = []
        for layer_key in sorted_layer_keys:
            m = re.match(r"depth_(\d+)$", layer_key)
            if not m:
                continue
            depth = int(m.group(1))
            by_syn = layer_pref_data.get(layer_key, {})
            syn_metrics = by_syn.get(synapse_type)
            if not syn_metrics:
                continue
            for metric_key in metrics:
                pref_vals = syn_metrics.get(metric_key)
                if not pref_vals:
                    continue
                metric_label = _metric_plot_meta[metric_key][0]
                for v in pref_vals:
                    fv = float(v)
                    if not math.isfinite(fv):
                        continue
                    rows_local.append(
                        {
                            "depth": depth,
                            "depth_label": f"Layer {depth}",
                            "metric": metric_label,
                            "value": fv,
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
            by_syn = layer_ratios_data.get(layer_key, {})
            syn_metrics = by_syn.get(synapse_type)
            if not syn_metrics:
                continue
            for metric_key in metrics:
                ratio_vals = syn_metrics.get(metric_key)
                if not ratio_vals:
                    continue
                metric_label = _metric_plot_meta[metric_key][0]
                for v in ratio_vals:
                    fv = float(v)
                    if not math.isfinite(fv):
                        continue
                    rows_local.append(
                        {
                            "depth": depth,
                            "depth_label": f"Layer {depth}",
                            "metric": metric_label,
                            "value": fv,
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
        figsize=(max(14, 1.6 * len(depth_order) * 2), 10),
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
                hue_order=hue_order,
                ax=ax_top,
                palette=palette,
                dodge=True,
                linewidth=1.0,
                showfliers=False,
            )
            ax_top.set_ylim(0.0, 1.0)
            ax_top.legend(title=None, loc="best")
        else:
            ax_top.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax_top.transAxes)

        ax_top.set_ylim(0.0, 1.0)
        ax_top.set_title(row0_titles[col])
        if col == 0:
            ax_top.set_ylabel("Preferred Similarity")
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
                hue_order=hue_order,
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
        f"{cell_label} cells — preferred similarity [0,1] (top) and pref/unpref ratio "
        f"(max RQ, NMF, mean, active; bottom), exc vs inh synapses",
        y=1.01,
    )
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    out_name = f"{cell_type}_cell_pref_unpref_ratio_by_layer_exc_inh_metrics.png"
    fig.savefig(os.path.join(save_path, out_name), dpi=200, bbox_inches="tight")
    plt.close(fig)



def _plot_receptive_fields_layer_soma_metric_ratios_heatmaps(
    layer_soma_data: dict[str, dict[str, dict[str, dict[str, dict[str, list[float]]]]]],
    cell_type: str,
    save_path: str,
) -> None:
    """Plot receptive fields layer soma metric ratios heatmaps."""
    # layer_soma_data[synapse_type][depth_key][soma_key][class_key][metric]: list[float]
    # heatmap_data[synapse_type][depth_key][metric]: np.ndarray
    if not layer_soma_data:
        logger.warning("empty layer_soma_data; skipping soma heatmaps")
        return

    first_syn = next(iter(layer_soma_data.keys()))
    if not layer_soma_data[first_syn]:
        logger.warning("no depth keys in layer_soma_data; skipping soma heatmaps")
        return

    heatmap_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    synapse_types = list(layer_soma_data.keys())
    depth_keys = list(layer_soma_data[synapse_types[0]].keys())
    soma_keys = list(layer_soma_data[synapse_types[0]][depth_keys[0]].keys())
    class_keys = list(
        layer_soma_data[synapse_types[0]][depth_keys[0]][soma_keys[0]].keys()
    )
    metrics = list(
        layer_soma_data[synapse_types[0]][depth_keys[0]][soma_keys[0]][
            class_keys[0]
        ].keys()
    )

    for synapse_type in synapse_types:
        heatmap_data[synapse_type] = {}
        for depth_key in depth_keys:
            heatmap_data[synapse_type][depth_key] = {}
            for metric in metrics:
                heatmap_data[synapse_type][depth_key][metric] = np.zeros(
                    (len(soma_keys), len(class_keys))
                )

    for synapse_type in synapse_types:
        for depth_key in depth_keys:
            for metric in metrics:
                for soma_key in soma_keys:
                    soma_idx = int(soma_key.split("_")[-1])
                    for class_key in class_keys:
                        class_idx = int(class_key.split("_")[-1])
                        vals = layer_soma_data[synapse_type][depth_key][soma_key][
                            class_key
                        ][metric]
                        if vals:
                            heatmap_data[synapse_type][depth_key][metric][
                                soma_idx, class_idx
                            ] = float(np.mean(vals))
                        else:
                            heatmap_data[synapse_type][depth_key][metric][
                                soma_idx, class_idx
                            ] = float("nan")

    def _depth_key_sort_key(dk: str) -> tuple[int, str]:
        m = re.match(r"depth_(\d+)$", dk)
        if m:
            return (int(m.group(1)), dk)
        return (10**9, dk)

    def _parse_depth_index(dk: str) -> int:
        m = re.match(r"depth_(\d+)$", dk)
        if m:
            return int(m.group(1))
        return 0

    depth_keys_sorted = sorted(depth_keys, key=_depth_key_sort_key)
    # depth_keys_sorted = depth_keys
    n_layers = len(depth_keys_sorted)
    if n_layers < 1:
        logger.warning("no depth layers; skipping soma heatmaps")
        return
    n_soma = len(soma_keys)
    n_class = len(class_keys)
    # Row i matches soma_keys[i]; label ticks with the corresponding soma index
    soma_index_labels: list[str] = [
        str(int(sk.split("_")[-1])) for sk in soma_keys
    ]
    class_tick = np.arange(n_class, dtype=int)
    _syn_figure_titles: dict[str, str] = {
        "exc": "Excitatory Synapses",
        "inh": "Inhibitory Synapses",
    }

    os.makedirs(save_path, exist_ok=True)

    def _bracket_vcenter(dmin: float, dmax: float, vcenter: float) -> tuple[float, float]:
        """Return (lo, hi) with lo < vcenter < hi for TwoSlopeNorm."""
        dmin, dmax = float(dmin), float(dmax)
        if not (math.isfinite(dmin) and math.isfinite(dmax)):
            return 0.0, 2.0
        if dmax < dmin:
            dmin, dmax = dmax, dmin
        if abs(dmax - dmin) < 1e-12:
            d = dmin
            if d < vcenter:
                return d, 2.0 * vcenter - d
            if d > vcenter:
                return 2.0 * vcenter - d, d
            return vcenter - 1.0, vcenter + 1.0
        if dmin < vcenter < dmax:
            return dmin, dmax
        if dmax <= vcenter:
            return dmin, vcenter + max(vcenter - dmin, 1e-3)
        return vcenter - max(dmax - vcenter, 1e-3), dmax

    for metric in metrics:
        blocks: list[np.ndarray] = []
        for syn in ("exc", "inh"):
            if syn not in heatmap_data:
                continue
            for depth_key in depth_keys_sorted:
                mdict = heatmap_data[syn].get(depth_key, {})
                if metric in mdict:
                    h = np.asarray(mdict[metric], dtype=np.float64)
                    if h.size:
                        blocks.append(h)
        if not blocks:
            continue
        mat_stack = np.concatenate([b.ravel() for b in blocks])
        v0 = float(np.nanmin(mat_stack))
        vm = float(np.nanmax(mat_stack))
        lo, hi = _bracket_vcenter(v0, vm, 1.0)
        norm = TwoSlopeNorm(vmin=lo, vcenter=1.0, vmax=hi)

        per_panel_w = 2.0
        per_panel_h = 2.6
        w_in = max(6.0, per_panel_w * n_layers + 1.4)
        h_in = 2 * per_panel_h + 1.1
        fig, axes = plt.subplots(
            2,
            n_layers,
            figsize=(w_in, h_in),
            squeeze=False,
            layout="constrained",
        )

        mappable: Any = None
        for col, depth_key in enumerate(depth_keys_sorted):
            layer_idx = _parse_depth_index(depth_key)
            for row, syn in enumerate(("exc", "inh")):
                ax = axes[row, col]
                if syn not in heatmap_data or depth_key not in heatmap_data[syn]:
                    ax.text(
                        0.5, 0.5, "no data",
                        ha="center", va="center", transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    M = np.asarray(
                        heatmap_data[syn][depth_key][metric], dtype=np.float64
                    )
                    im = ax.imshow(
                        M,
                        cmap="PRGn",
                        norm=norm,
                        aspect="equal",
                        origin="upper",
                        interpolation="nearest",
                    )
                    mappable = im

                    ax.set_xticks(np.arange(n_class))
                    if row == 1:
                        ax.set_xticklabels([str(int(t)) for t in class_tick], fontsize=8)
                    else:
                        ax.set_xticklabels([])

                    ax.set_yticks(np.arange(n_soma))
                    if col == 0:
                        ax.set_yticklabels(soma_index_labels, fontsize=8)
                    else:
                        ax.set_yticklabels([])

                    ax.tick_params(axis="both", length=3, pad=2)

                ax.set_title(
                    f"{_syn_figure_titles[syn]} Layer {layer_idx}",
                    fontsize=9,
                    pad=3,
                )
                if row == 1:
                    ax.set_xlabel("Class", fontsize=9, labelpad=2)
                if col == 0:
                    ax.set_ylabel("Soma", fontsize=9, labelpad=2)

        if mappable is not None:
            fig.colorbar(
                mappable,
                ax=axes,
                orientation="vertical",
                shrink=0.85,
                aspect=28,
                pad=0.02,
            )

        title_metric = str(metric).replace("_", " ").title()
        cell_label = "Excitatory" if cell_type == "exc" else "Inhibitory"
        fig.suptitle(
            f"{cell_label} cells — {title_metric} (soma vs class, pref ratio)",
            fontsize=11,
        )

        safe_metric = re.sub(r"[^a-zA-Z0-9_]+", "_", str(metric)).strip("_")
        out = os.path.join(
            save_path, f"{cell_type}_soma_{safe_metric}_ratio_heatmaps.png"
        )
        fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
        plt.close(fig)





