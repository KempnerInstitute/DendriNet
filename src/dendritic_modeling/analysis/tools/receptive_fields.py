import math
import os
import re
import tqdm
from functools import partial
from typing import Optional, Any
from copy import deepcopy

import logging
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from sklearn.decomposition import NMF
from sklearn import metrics

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
        self.fig_save_format = getattr(params, "fig_save_format", "png")

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

    def forward_hook(
        self, 
        module: DendriticBranchLayer, 
        input: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], 
        output: torch.Tensor,
        labels: torch.Tensor,
        data_dict: dict[str, dict[str, dict[str, list[float]]]],
        cell_type: str
    ):
        """Forward hook to collect receptive fields."""
        excitatory_input, inhibitory_input, _ = input

        if self.exc_inputs is None:
            self.exc_inputs = excitatory_input
        if self.inh_inputs is None:
            self.inh_inputs = inhibitory_input
        
        if module.branch_excitation is not None:
            excitation: torch.Tensor = module.branch_excitation(excitatory_input)
        if module.branch_inhibition is not None:
            inhibition: torch.Tensor = module.branch_inhibition(inhibitory_input)

        n_soma = self.n_exc_soma if cell_type == "exc" else self.n_inh_soma
        branches_per_soma = int(module.n_branches // n_soma)

        depth = module.layer_idx
        for b in range(module.n_branches):
            soma_idx = int(b // branches_per_soma)
            branch_idx = int(b % branches_per_soma)            
            location_key = f"soma {soma_idx} - depth {depth} - branch {branch_idx}"

            branch_data = {}

            for class_idx in range(self.n_classes):
                class_key = f"class_{class_idx}"
                class_mask = labels == class_idx

                class_data = {}

                if module.branch_excitation is not None:
                    exc = excitation[class_mask][:,b]
                    class_data["exc"] = exc.detach().cpu().tolist()
                
                if module.branch_inhibition is not None:
                    inh = inhibition[class_mask][:,b]
                    class_data["inh"] = inh.detach().cpu().tolist()
                
                if (
                    module.branch_excitation is not None 
                    and module.branch_inhibition is not None
                ):
                    ei_ratio = exc / (inh + 1e-6)
                    class_data["ei_ratio"] = ei_ratio.detach().cpu().tolist()
                
                vout = output[class_mask][:,b]
                class_data["vout"] = vout.detach().cpu().tolist()

                branch_data[class_key] = class_data
            
            data_dict[location_key] = branch_data
            
            if hasattr(self, "_pbar"):
                self._pbar.update(1)

    def attach_forward_hooks(
        self, 
        model: BaseModel, 
        labels: torch.Tensor
    ):
        """Attach forward hooks to the model."""
        core_network: ExcitationInhibitionNetwork = model.core_network
        handles = []
        
        exc_dendrinet: DendriNet = core_network.layers[0].excitatory_cells
        self.n_exc_soma = exc_dendrinet.n_soma
        exc_cell_data: dict[str, dict[str, dict[str, list[float]]]] = {}
        for dbl in exc_dendrinet.branch_layers:
            handle = dbl.register_forward_hook(
                partial(
                    self.forward_hook, 
                    labels=labels, 
                    data_dict=exc_cell_data, 
                    cell_type="exc"
                )
            )
            handles.append(handle)
        
        inh_dendrinet = core_network.layers[0].inhibitory_cells
        self.n_inh_soma = None
        if isinstance(inh_dendrinet, DendriNet):
            self.n_inh_soma = inh_dendrinet.n_soma
        
            inh_cell_data: dict[str, dict[str, dict[str, list[float]]]] = {}
            for dbl in inh_dendrinet.branch_layers:
                handle = dbl.register_forward_hook(
                    partial(
                        self.forward_hook, 
                        labels=labels, 
                        data_dict=inh_cell_data, 
                        cell_type="inh"
                    )
                )
                handles.append(handle)
        else:
            inh_cell_data = None
        
        return handles, exc_cell_data, inh_cell_data

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()
    
    @staticmethod
    def _compute_point_biserial_correlation(
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        rpb_dict: dict[str, np.ndarray] = {}

        n_classes = len(torch.unique(labels))
        n_samples = inputs.shape[0]
        total_std = inputs.std(dim=0)
        for class_idx in range(n_classes):
            class_mask = labels == class_idx

            pref_inputs = inputs[class_mask]
            pref_mean = pref_inputs.mean(dim=0)
            pref_prob = pref_inputs.shape[0] / n_samples

            unpref_inputs = inputs[~class_mask]
            unpref_mean = unpref_inputs.mean(dim=0)
            unpref_prob = unpref_inputs.shape[0] / n_samples

            numer = pref_prob * unpref_prob * (pref_mean - unpref_mean)

            rpb = numer / (total_std + 1e-9)
            rpb_dict[f"class_{class_idx}"] = rpb.detach().cpu().numpy()

        return rpb_dict
    
    def compute_confusion_entropy(
        self,
        model: BaseModel,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """Compute confusion matrix and entropy dictionary."""
        with torch.no_grad():
            logits: torch.Tensor = model(x)
        
        y_pred = logits.argmax(dim=-1)

        confusion_matrix: np.ndarray = metrics.confusion_matrix(
            y_true=labels.detach().cpu().numpy(),
            y_pred=y_pred.detach().cpu().numpy()
        )
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

        entropies = Categorical(logits=logits).entropy()
        correct_mask = labels == y_pred

        entropy_dict = {
            "correct": entropies[correct_mask].detach().cpu().numpy(),
            "incorrect": entropies[~correct_mask].detach().cpu().numpy()
        }
        
        return confusion_matrix, entropy_dict
    
    def agg_rf_tuning(
        self, 
        agg_dict: dict[str, dict[str, np.ndarray]], 
        synapse_type: str
    ) -> dict[str, dict[str, float]]:
        agg_tuning_dict: dict[str, dict[str, float]] = {}

        n_soma = len(agg_dict)

        for soma_idx in range(n_soma):
            class_key = f"class_{soma_idx}"
            rpb = self.rpb_dict[class_key]

            soma_key = f"soma {soma_idx}"
            w = agg_dict[soma_key][f"{synapse_type}_rf"].reshape(-1)

            pos_mask = rpb > 0
            rpb_pos = rpb[pos_mask]
            w_pos = w[pos_mask]

            cs_pos = np.dot(rpb_pos, w_pos) / (
                np.linalg.norm(rpb_pos) * np.linalg.norm(w_pos) + 1e-9
            )

            neg_mask = rpb < 0
            rpb_neg = np.abs(rpb[neg_mask])
            w_neg = w[neg_mask]

            cs_neg = np.dot(rpb_neg, w_neg) / (
                np.linalg.norm(rpb_neg) * np.linalg.norm(w_neg) + 1e-9
            )

            agg_tuning_dict[soma_key] = {
                "pos": float(cs_pos),
                "neg": float(cs_neg)
            }

        return agg_tuning_dict

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
                agg_rfs_dict[soma_key]["exc_rf"] = deepcopy(branch_data["exc_rf"])
            elif "exc_rf" in branch_data:
                agg_rfs_dict[soma_key]["exc_rf"] = (
                    agg_rfs_dict[soma_key]["exc_rf"] + branch_data["exc_rf"]
                )

            if "inh_rf" in branch_data and "inh_rf" not in agg_rfs_dict[soma_key]:
                agg_rfs_dict[soma_key]["inh_rf"] = deepcopy(branch_data["inh_rf"])
            elif "inh_rf" in branch_data:
                agg_rfs_dict[soma_key]["inh_rf"] = (
                    agg_rfs_dict[soma_key]["inh_rf"] + branch_data["inh_rf"]
                )
        
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
                    prod_branch_rf["inh_rf"] = deepcopy(i_cell_exc_rf)
                else:
                    prod_branch_rf["inh_rf"] = (
                        prod_branch_rf["inh_rf"] + i_cell_exc_rf
                    )
            
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

        if not hasattr(self, "rpb_dict"):
            self.rpb_dict = self._compute_point_biserial_correlation(x, labels)

        self.exc_inputs = None
        self.inh_inputs = None

        unique_classes = torch.unique(labels)
        self.n_classes = len(unique_classes)

        handles, exc_cell_data, inh_cell_data = self.attach_forward_hooks(model, labels)

        if not training:
            self._pbar = tqdm.tqdm(
                total=n_total, 
                desc="Computing receptive field metrics...", 
                unit=" branches"
            )
        
        (
            confusion_matrix, 
            entropy_dict
        ) = self.compute_confusion_entropy(model, x, labels)
        
        self.remove_forward_hooks(handles)

        if hasattr(self, "_pbar"):
            self._pbar.close()
            del self._pbar

        pref_unpref_data = self._branch_data_to_preferred(exc_cell_data)

        exc_rfs_dict, inh_rfs_dict = self.collect_receptive_fields(ei_layer)
        exc_agg_rfs_dict = self._aggregate_rfs_by_soma(exc_rfs_dict)

        exc_agg_tuning_dict = self.agg_rf_tuning(
            agg_dict=exc_agg_rfs_dict,
            synapse_type="exc"
        )
        
        if inh_rfs_dict is not None:
            inh_agg_rfs_dict = self._aggregate_rfs_by_soma(inh_rfs_dict)
            prod_rfs_dict = self._prod_agg_i_cell_e_rf_by_e_cell_i_ref(
                agg_inh_rfs_dict=inh_agg_rfs_dict,
                exc_rfs_dict=exc_rfs_dict
            )
            agg_prod_rfs_dict = self._aggregate_rfs_by_soma(prod_rfs_dict)

            inh_agg_tuning_dict = self.agg_rf_tuning(
                agg_dict=agg_prod_rfs_dict,
                synapse_type="inh"
            )
        else:
            inh_agg_tuning_dict = self.agg_rf_tuning(
                agg_dict=exc_agg_rfs_dict,
                synapse_type="inh"
            )
        
        agg_tuning_dict = {
            "exc": exc_agg_tuning_dict,
            "inh": inh_agg_tuning_dict
        }

        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")

            save_dict(agg_tuning_dict, save_path, f"{filename}.json")

            if not training:
                mean_vector_dict = self._compute_mean_vector_dict(
                    self.exc_inputs,
                    self.inh_inputs,
                    labels
                )
                
                _pbar = tqdm.tqdm(
                    total=22 if inh_rfs_dict else 18, 
                    desc="Generating plots...", 
                    unit=" plots"
                )

                _plot_class_discriminability_heatmaps(
                    save_path=save_path,
                    fig_format=self.fig_save_format,
                    inputs=x,
                    labels=labels,
                    shape=self.exc_rf_shape
                )
                _pbar.update(1)
                _plot_agg_tuning_final(
                    save_path=save_path,
                    fig_format=self.fig_save_format,
                    agg_tuning_dict=agg_tuning_dict
                )
                _pbar.update(1)
                _plot_class_examples(
                    save_path=save_path,
                    fig_format=self.fig_save_format,
                    inputs=x,
                    labels=labels,
                    shape=self.exc_rf_shape,
                    n_examples=5
                )
                _pbar.update(1)
                _plot_confusion_and_entropies(
                    confusion_matrix=confusion_matrix,
                    entropy_dict=entropy_dict,
                    save_path=save_path,
                    fig_format=self.fig_save_format
                )
                _pbar.update(1)

                activation_plots_path = os.path.join(save_path, "activation_plots")
                _plot_exc_vs_inh_by_compartment(
                    pref_unpref_data=pref_unpref_data,
                    save_path=activation_plots_path,
                    fig_format=self.fig_save_format
                )
                _pbar.update(1)
                _plot_vout_vs_eiratio_by_compartment(
                    pref_unpref_data=pref_unpref_data,
                    save_path=activation_plots_path,
                    fig_format=self.fig_save_format
                )
                _pbar.update(1)
                _plot_vout_activation_boxplots(
                    data_dict=exc_cell_data,
                    save_path=activation_plots_path,
                    fig_format=self.fig_save_format,
                    cell_type="exc"
                )
                _pbar.update(1)
                _plot_vout_activation_boxplots(
                    data_dict=inh_cell_data,
                    save_path=activation_plots_path,
                    fig_format=self.fig_save_format,
                    cell_type="inh"
                )
                _pbar.update(1)

                weight_rfs_path = os.path.join(save_path, "synapse_weight_rfs")                
                # _plot_receptive_fields(
                #     save_path=weight_rfs_path,
                #     title="exc_cell_exc_syn_weight_rfs",
                #     rfs_dict=exc_rfs_dict, 
                #     synapse_type="exc"
                # )
                # _pbar.update(1)
                
                agg_weight_rfs_path = os.path.join(save_path, "agg_synapse_weight_rfs")
                # _plot_receptive_fields(
                #     save_path=agg_weight_rfs_path,
                #     title="agg_exc_cell_exc_syn_weight_rfs",
                #     rfs_dict=exc_agg_rfs_dict, 
                #     synapse_type="exc"
                # )
                # _pbar.update(1)
                
                syn_act_rfs_path = os.path.join(save_path, "synapse_activation_rfs")
                _plot_receptive_fields(
                    save_path=syn_act_rfs_path,
                    fig_format=self.fig_save_format,
                    title="exc_cell_exc_syn_pref_act_rfs",
                    rfs_dict=exc_rfs_dict, 
                    synapse_type="exc",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=False,
                    mean_vector_type="exc"
                )
                # _pbar.update(1)
                # _plot_receptive_fields(
                #     save_path=syn_act_rfs_path,
                #     title="exc_cell_exc_syn_total_act_rfs",
                #     rfs_dict=exc_rfs_dict, 
                #     synapse_type="exc",
                #     mean_vector_dict=mean_vector_dict,
                #     total_mean_vector=True,
                #     mean_vector_type="exc"
                # )
                # _pbar.update(1)
                
                agg_syn_act_rfs_path = os.path.join(save_path, "agg_synapse_activation_rfs")
                
                _plot_receptive_fields(
                    save_path=agg_syn_act_rfs_path,
                    fig_format=self.fig_save_format,
                    title="agg_exc_cell_exc_syn_pref_act_rfs",
                    rfs_dict=exc_agg_rfs_dict, 
                    synapse_type="exc",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=False,
                    mean_vector_type="exc",
                    flatten_axes=True
                )
                _pbar.update(1)
                _plot_receptive_fields(
                    save_path=agg_syn_act_rfs_path,
                    fig_format=self.fig_save_format,
                    title="agg_exc_cell_exc_syn_total_act_rfs",
                    rfs_dict=exc_agg_rfs_dict, 
                    synapse_type="exc",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector=True,
                    mean_vector_type="exc",
                    flatten_axes=True
                )
                _pbar.update(1)
                
                if inh_rfs_dict is not None:
                    # _plot_receptive_fields(
                    #     save_path=weight_rfs_path,
                    #     title="inh_cell_exc_syn_weight_rfs",
                    #     rfs_dict=inh_rfs_dict, 
                    #     synapse_type="exc"
                    # )
                    # _pbar.update(1)
                    # _plot_receptive_fields(
                    #     save_path=agg_weight_rfs_path,
                    #     title="agg_inh_cell_exc_syn_weight_rfs",
                    #     rfs_dict=inh_agg_rfs_dict, 
                    #     synapse_type="exc"
                    # )
                    # _pbar.update(1)
                    _plot_receptive_fields(
                        save_path=syn_act_rfs_path,
                        fig_format=self.fig_save_format,
                        title="inh_cell_exc_syn_total_act_rfs",
                        rfs_dict=inh_rfs_dict, 
                        synapse_type="exc",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=True,
                        mean_vector_type="exc"
                    )
                    _pbar.update(1)
                    _plot_receptive_fields(
                        save_path=agg_syn_act_rfs_path,
                        fig_format=self.fig_save_format,
                        title="agg_inh_cell_exc_syn_total_act_rfs",
                        rfs_dict=inh_agg_rfs_dict, 
                        synapse_type="exc",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=True,
                        mean_vector_type="exc"
                    )
                    _pbar.update(1)

                    agg_prod_rfs_path = os.path.join(save_path, "inh_agg_prod_synapse_activation_rfs")
                    _plot_receptive_fields(
                        save_path=agg_prod_rfs_path,
                        fig_format=self.fig_save_format,
                        title="inh_agg_prod_ie_syn_pref_act_rfs",
                        rfs_dict=prod_rfs_dict, 
                        synapse_type="inh",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=False,
                        mean_vector_type="exc"
                    )
                    _pbar.update(1)
                    _plot_receptive_fields(
                        save_path=agg_prod_rfs_path,
                        fig_format=self.fig_save_format,
                        title="inh_agg_prod_ie_syn_total_act_rfs",
                        rfs_dict=prod_rfs_dict, 
                        synapse_type="inh",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=True,
                        mean_vector_type="exc"
                    )
                    _pbar.update(1)
                    _plot_receptive_fields(
                        save_path=agg_prod_rfs_path,
                        fig_format=self.fig_save_format,
                        title="agg_inh_agg_prod_ie_syn_pref_act_rfs",
                        rfs_dict=agg_prod_rfs_dict, 
                        synapse_type="inh",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=False,
                        mean_vector_type="exc",
                        flatten_axes=True
                    )
                    _pbar.update(1)
                    _plot_receptive_fields(
                        save_path=agg_prod_rfs_path,
                        fig_format=self.fig_save_format,
                        title="agg_inh_agg_prod_ie_syn_total_act_rfs",
                        rfs_dict=agg_prod_rfs_dict, 
                        synapse_type="inh",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=True,
                        mean_vector_type="exc",
                        flatten_axes=True
                    )
                    _pbar.update(1)
                    agg_prod_rfs_path = os.path.join(save_path, "inh_agg_prod_synapse_weight_rfs")
                    # _plot_receptive_fields(
                    #     save_path=agg_prod_rfs_path,
                    #     title="agg_inh_agg_prod_ie_syn_weight_rfs",
                    #     rfs_dict=agg_prod_rfs_dict, 
                    #     synapse_type="inh"
                    # )
                    # _pbar.update(1)
                    # _plot_receptive_fields(
                    #     save_path=agg_prod_rfs_path,
                    #     title="inh_agg_prod_ie_syn_weight_rfs",
                    #     rfs_dict=prod_rfs_dict, 
                    #     synapse_type="inh"
                    # )
                    # _pbar.update(1)
                else:
                    # _plot_receptive_fields(
                    #     save_path=weight_rfs_path,
                    #     title="exc_cell_inh_syn_weight_rfs",
                    #     rfs_dict=exc_rfs_dict, 
                    #     synapse_type="inh"
                    # )
                    # _pbar.update(1)
                    # _plot_receptive_fields(
                    #     save_path=agg_weight_rfs_path,
                    #     title="agg_exc_cell_inh_syn_weight_rfs",
                    #     rfs_dict=exc_agg_rfs_dict, 
                    #     synapse_type="inh"
                    # )
                    # _pbar.update(1)
                    _plot_receptive_fields(
                        save_path=syn_act_rfs_path,
                        fig_format=self.fig_save_format,
                        title="exc_cell_inh_syn_pref_act_rfs",
                        rfs_dict=exc_rfs_dict, 
                        synapse_type="inh",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=False,
                        mean_vector_type="inh"
                    )
                    _pbar.update(1)
                    _plot_receptive_fields(
                        save_path=agg_syn_act_rfs_path,
                        fig_format=self.fig_save_format,
                        title="agg_exc_cell_inh_syn_pref_act_rfs",
                        rfs_dict=exc_agg_rfs_dict, 
                        synapse_type="inh",
                        mean_vector_dict=mean_vector_dict,
                        total_mean_vector=False,
                        mean_vector_type="inh",
                        flatten_axes=True
                    )
                    _pbar.update(1)

                _pbar.close()
                del _pbar


### NOTES AND TODOs
###   (DONE) Show preferred / unpreferred (or class-conditioned) E vs I plots for each compartment on exc cells.
###   (DONE) Show class-conditioned vout activation boxplots for each compartment (including soma) on exc and inh cells.
###   (DONE) Confusion matrix
###   (DONE) Reliability diagram? (entropy of readout space versus some notion of error, architecture-agnostic visual for "how the model decides")
###   (DONE) Mean MNIST digit for each class next to aggregated exc receptive field for each readout neuron
###   (DONE) Sum of exc syn receptive fields for each interneuron
###   (DONE) Linear weighted sum of inh syn receptive fields for each readout neuron (multiply aggregated interneuron exc RFs by inh syn weights on readout neurons)
###   (DONE) Aggregate above^ by soma (total interneuron exc RF for each readout neuron)
###   (DONE) Multiply all receptive fields by class-conditioned mean vectors of inputs (to kill silent synapses)
###   Some visual (for each compartment) showing why some preferred stimuli have E/I ratio < 1 and nonpreferred stimuli have E/I ratio > 1 (think about this HARD AHHHH)


def _location_key_to_tuple(location_key: str) -> tuple[int, int, int]:
    """Convert location key to tuple."""
    loc = location_key.replace("soma", "").replace("depth", "").replace("branch", "")
    loc = loc.replace(" ", "")
    return tuple(int(x) for x in loc.split("-"))


def _plot_agg_tuning_final(
    save_path: str,
    fig_format: str,
    agg_tuning_dict: dict[str, dict[str, dict[str, float]]]
):
    """Grouped bars of cosine similarity between agg RF weights and RPB (sign-split)."""
    exc_data = agg_tuning_dict.get("exc") or {}
    inh_data = agg_tuning_dict.get("inh") or {}
    
    if not exc_data and not inh_data:
        return
    
    def _soma_idx(key: str) -> int:
        return int(key.replace("soma", "").strip())
    
    soma_keys = sorted(exc_data.keys(), key=_soma_idx)
    if not soma_keys:
        soma_keys = sorted(inh_data.keys(), key=_soma_idx)
    if not soma_keys:
        return
    
    n = len(soma_keys)
    x = np.arange(n, dtype=float)
    width = 0.35
    rpb_str = r"$\rho_{PB}$"
    
    fig, axes = plt.subplots(2, 1, figsize=(max(7.0, 1.1 * n), 8), sharex=True)
    title_suffix = "Feature Tuning to Input Point-Biserial Correlation"
    row_specs = (
        ("exc", "Excitatory " + title_suffix),
        ("inh", "Inhibitory " + title_suffix),
    )
    
    for ax, (syn_key, row_title) in zip(axes, row_specs):
        syn_dict = agg_tuning_dict.get(syn_key) or {}
        pos_vals: list[float] = []
        neg_vals: list[float] = []
        
        for sk in soma_keys:
            cell = syn_dict.get(sk, {})
            pos_vals.append(float(cell.get("pos", float("nan"))))
            neg_vals.append(float(cell.get("neg", float("nan"))))
        
        ax.bar(
            x - width / 2,
            pos_vals,
            width,
            label=f"pos {rpb_str}",
            color="#2E7FD9",
            edgecolor="white",
            linewidth=0.6,
        )
        ax.bar(
            x + width / 2,
            neg_vals,
            width,
            label=f"neg {rpb_str}",
            color="#D93E3E",
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(row_title)
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="-", zorder=0)
        ax.legend(loc="best", framealpha=0.92)
        ax.set_ylim(0.0, 1.0)

    axes[-1].set_xlabel("Soma")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([str(_soma_idx(sk)) for sk in soma_keys])
    
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"agg_feature_tuning.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_confusion_and_entropies(
    confusion_matrix: np.ndarray,
    entropy_dict: dict[str, np.ndarray],
    save_path: str,
    fig_format: str,
) -> None:
    """Plot confusion matrix (left) and entropy boxplots for correct vs incorrect (right)."""
    fig, (ax_cm, ax_ent) = plt.subplots(
        1, 2, figsize=(10.5, 5), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    cm = np.asarray(confusion_matrix, dtype=float)
    if cm.size == 0:
        ax_cm.text(
            0.5,
            0.5,
            "empty confusion matrix",
            ha="center",
            va="center",
            transform=ax_cm.transAxes,
        )
    else:
        n = cm.shape[0]
        tick_labels = [str(i) for i in range(n)]
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cmap="inferno",
            ax=ax_cm,
            vmin=0,
            cbar_kws={"label": "Ratio of Samples"},
        )
        ax_cm.set_xlabel("Predicted class")
        ax_cm.set_ylabel("True class")
        ax_cm.set_xticklabels(tick_labels, rotation=0)
        ax_cm.set_yticklabels(tick_labels, rotation=0)

    correct = np.asarray(entropy_dict.get("correct", []), dtype=float).ravel()
    incorrect = np.asarray(entropy_dict.get("incorrect", []), dtype=float).ravel()

    rows: list[dict[str, str | float]] = []
    for v in correct:
        if np.isfinite(v):
            rows.append({"outcome": "correct", "entropy": float(v)})
    for v in incorrect:
        if np.isfinite(v):
            rows.append({"outcome": "incorrect", "entropy": float(v)})

    if rows:
        df = pd.DataFrame(rows)
        sns.boxplot(
            data=df,
            x="outcome",
            y="entropy",
            order=["correct", "incorrect"],
            ax=ax_ent,
            width=0.55,
        )
    else:
        ax_ent.text(
            0.5,
            0.5,
            "no entropy data",
            ha="center",
            va="center",
            transform=ax_ent.transAxes,
        )

    ax_ent.set_ylabel("Entropy")
    ax_ent.set_xlabel("Classification outcome")

    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"confusion_and_entropies.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_class_examples(
    save_path: str,
    fig_format: str,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    shape: tuple[int, int],
    n_examples: int = 5,
) -> None:
    """Plot class discriminability heatmaps."""
    n_classes = len(torch.unique(labels))
    
    fig, axes = plt.subplots(
        n_examples, n_classes,
        figsize=(6 * n_classes, 6 * n_examples)
    )

    for class_idx in range(n_classes):
        class_mask = labels == class_idx
        class_inputs = inputs[class_mask]
        ex_ixs = torch.randperm(class_inputs.shape[0])[:n_examples]
        ex_inputs = class_inputs[ex_ixs]
        
        for ex_idx in range(n_examples):
            ax = axes[ex_idx, class_idx]
            ex_input = ex_inputs[ex_idx]
            shaped_input = ex_input.reshape(*shape).detach().cpu().numpy()
            sns.heatmap(
                shaped_input,
                ax=ax,
                cmap="Greys",
                vmin=0,
                vmax=1,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            
            if ex_idx == 0:
                ax.set_title(f"Class {class_idx}")
    
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"class_examples.{fig_format}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_dprime(
    inputs: torch.Tensor,
    labels: torch.Tensor,
):
    dprime_dict: dict[str, np.ndarray] = {}

    n_classes = len(torch.unique(labels))
    n_samples = inputs.shape[0]
    total_mean = inputs.mean(dim=0)
    total_var = inputs.var(dim=0)
    for class_idx in range(n_classes):
        class_mask = labels == class_idx

        pref_inputs = inputs[class_mask]
        pref_mean = pref_inputs.mean(dim=0)
        pref_prob = pref_inputs.shape[0] / n_samples

        numer = pref_prob * (pref_mean - total_mean) ** 2

        unpref_inputs = inputs[~class_mask]
        unpref_mean = unpref_inputs.mean(dim=0)
        unpref_prob = unpref_inputs.shape[0] / n_samples

        numer = numer + unpref_prob * (unpref_mean - total_mean) ** 2

        dprime = (numer / (total_var + 1e-9)).sqrt()
        dprime_dict[f"class_{class_idx}"] = dprime.detach().cpu().numpy()

    return dprime_dict


def _compute_point_biserial_correlation(
    inputs: torch.Tensor,
    labels: torch.Tensor,
):
    rpb_dict: dict[str, np.ndarray] = {}

    n_classes = len(torch.unique(labels))
    n_samples = inputs.shape[0]
    total_std = inputs.std(dim=0)
    for class_idx in range(n_classes):
        class_mask = labels == class_idx

        pref_inputs = inputs[class_mask]
        pref_mean = pref_inputs.mean(dim=0)
        pref_prob = pref_inputs.shape[0] / n_samples

        unpref_inputs = inputs[~class_mask]
        unpref_mean = unpref_inputs.mean(dim=0)
        unpref_prob = unpref_inputs.shape[0] / n_samples

        numer = pref_prob * unpref_prob * (pref_mean - unpref_mean)

        rpb = numer / (total_std + 1e-9)
        rpb_dict[f"class_{class_idx}"] = rpb.detach().cpu().numpy()

    return rpb_dict


def _plot_discrim_metric_heatmaps(
    save_path: str,
    fig_format: str,
    n_rows: int,
    n_cols: int,
    metric_dict: dict[str, np.ndarray],
    metric_name: str,
    shape: tuple[int, int],
    cmap: str,
    vmin: float,
    vmax: float,
    center: Optional[float] = None,
):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    for i, (class_key, value_vec) in enumerate(metric_dict.items()):
        ax = axes[i]
        shaped_vec = value_vec.reshape(*shape)
        use_cbar = i == len(metric_dict) - 1
        sns.heatmap(
            shaped_vec, 
            ax=ax, 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax, 
            center=center,
            cbar=use_cbar,
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(class_key.replace("_", " ").title())

    for j in range(len(metric_dict), len(axes)):
        axes[j].set_visible(False)
    
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"class_{metric_name}_heatmaps.{fig_format}"), 
        dpi=200, 
        bbox_inches="tight"
    )
    plt.close(fig)


def _plot_class_discriminability_heatmaps(
    save_path: str,
    fig_format: str,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    shape: tuple[int, int],
) -> None:
    """Plot class discriminability heatmaps."""
    dprime_dict = _compute_dprime(inputs, labels)
    rpb_dict = _compute_point_biserial_correlation(inputs, labels)

    n_classes = len(dprime_dict)

    # n_rows = 1
    # n_cols = 1

    # while n_rows * n_cols < n_classes:
    #     if n_rows < n_cols:
    #         n_rows += 1
    #     else:
            # n_cols += 1
    
    _plot_discrim_metric_heatmaps(
        save_path=save_path,
        fig_format=fig_format,
        # n_rows=n_rows,
        # n_cols=n_cols,
        n_rows=1,
        n_cols=n_classes,
        metric_dict=dprime_dict,
        metric_name="dprime",
        shape=shape,
        cmap="binary",
        vmin=0,
        vmax=1,
        center=None
    )

    # cmap = LinearSegmentedColormap.from_list(
    #     "magenta_purple_black_blue_azure",
    #     [
    #         (0.00, "#E5D4FA"),   # light purple bridge
    #         (0.46, "#FF10F0"),
    #         (0.50, "#000000"),
    #         (0.54, "#2FA3E6"),
    #         (1.00, "#C8E9FA"),   # light blue bridge
    #     ],
    # )
    _plot_discrim_metric_heatmaps(
        save_path=save_path,
        fig_format=fig_format,
        # n_rows=n_rows,
        # n_cols=n_cols,
        n_rows=1,
        n_cols=n_classes,
        metric_dict=rpb_dict,
        metric_name="rpb",
        shape=shape,
        cmap="seismic_r",
        # cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0
    )

def _plot_receptive_fields(
    save_path: str,
    fig_format: str,
    title: str,
    rfs_dict: dict[str, dict[str, np.ndarray]],
    synapse_type: str,
    mean_vector_dict: Optional[dict[str, dict[str, np.ndarray]]] = None,
    total_mean_vector: Optional[bool] = False,
    mean_vector_type: Optional[str] = None,
    flatten_axes: bool = False
):
    """Plot receptive fields."""
    n_total_branches = len(rfs_dict)

    if flatten_axes:
        n_rows = 1
        n_cols = n_total_branches
    else:
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
        
                
        rf = branch_rfs[f"{synapse_type}_rf"]

        if mean_vector_dict is not None:
            if total_mean_vector:
                class_key = "total"
            else:
                class_key = f"class_{soma_idx}"
            input_mean_vec = mean_vector_dict[class_key][mean_vector_type]
            shaped_vec = input_mean_vec.reshape(*rf.shape)
            
            rf = rf * shaped_vec

        sns.heatmap(
            rf, 
            ax=ax, 
            cmap="inferno"
            # cmap="gnuplot2"
        )
        ax.set_title(location_key.title())
    
    for j in range(n_total_branches, len(axes)):
        axes[j].set_visible(False)
    
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"{title}.{fig_format}"), 
        dpi=200, 
        bbox_inches="tight"
    )
    plt.close(fig)


def _plot_vout_activation_boxplots(
    data_dict: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
    fig_format: str,
    cell_type: str
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
    fig.savefig(
        os.path.join(save_path, f"{cell_type}_vout_activation_by_class_boxplots.{fig_format}"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)


def _plot_pref_unpref_scatter_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    xvar: str,
    yvar: str,
    x_logscale: bool,
    save_path: str,
    fig_format: str,
    fig_basename: str,
    reference_line: Optional[str] = None,
) -> None:
    """
    One subplot per branch/compartment, same grid heuristic as _plot_receptive_fields.

    ``pref_unpref_data`` is the output of ``_branch_data_to_preferred``: for each
    branch, ``preferred`` / ``unpreferred`` map metric names (``exc``, ``inh``,
    ``ei_ratio``, ``vout``) to per-sample lists. Scatter ``yvar`` vs ``xvar``;
    preferred points in red, unpreferred in black. KDE marginals above (for x)
    and to the right (for y) match scatter colors. Compartment title is placed on
    the top marginal axis (above the x-density strip).

    ``reference_line``: optional grey dotted guide — ``"y=x"`` (identity) or
    ``"x=1"`` (vertical).
    """
    def _linear_pad(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            return 0.0, 1.0
        if hi == lo:
            m = lo if lo != 0 else 1.0
            return lo - 0.05 * abs(m), hi + 0.05 * abs(m)
        pad = (hi - lo) * frac
        return lo - pad, hi + pad

    def _log_pad(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi < lo:
            return 1e-6, 1.0
        if hi / lo < 1 + 1e-12:
            return lo * 0.9, hi * 1.1
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        span = log_hi - log_lo
        pad = frac * span
        return 10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad)

    def _kde_curve_x(
        ax_marg: Any, xv: np.ndarray, color: str, x_lo: float, x_hi: float
    ) -> None:
        if xv.size < 2:
            return
        try:
            sns.kdeplot(
                x=xv,
                ax=ax_marg,
                color=color,
                fill=True,
                linewidth=3.0,
                alpha=0.35,
                clip=(x_lo, x_hi),
            )
        except Exception:
            nb = min(35, max(8, xv.size // 4))
            sns.histplot(
                x=xv,
                ax=ax_marg,
                color=color,
                stat="density",
                bins=nb,
                alpha=0.35,
                element="step",
                binrange=(x_lo, x_hi),
            )

    def _kde_curve_y(
        ax_marg: Any, yv: np.ndarray, color: str, y_lo: float, y_hi: float
    ) -> None:
        if yv.size < 2:
            return
        try:
            sns.kdeplot(
                y=yv,
                ax=ax_marg,
                color=color,
                fill=True,
                linewidth=3.0,
                alpha=0.35,
                clip=(y_lo, y_hi),
            )
        except Exception:
            nb = min(35, max(8, yv.size // 4))
            sns.histplot(
                y=yv,
                ax=ax_marg,
                color=color,
                stat="density",
                bins=nb,
                alpha=0.35,
                element="step",
                binrange=(y_lo, y_hi),
            )

    if not pref_unpref_data:
        return
    
    branch_keys = []
    for bk, bd in pref_unpref_data.items():
        if xvar in bd["preferred"] and yvar in bd["preferred"]:
            branch_keys.append(bk)

    n_total = len(branch_keys)

    n_rows = 1
    n_cols = 1
    while n_rows * n_cols < n_total:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows)
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

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        if mp.any():
            x_parts.append(xp[mp])
            y_parts.append(yp[mp])
        if mu.any():
            x_parts.append(xu[mu])
            y_parts.append(yu[mu])
        if not x_parts:
            ax.set_visible(False)
            ax.set_title(bk.title())
            continue

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="20%", pad=0.06, sharex=ax)
        ax_right = divider.append_axes("right", size="20%", pad=0.06, sharey=ax)

        ax_top.set_axis_off()
        ax_right.set_axis_off()

        xs = np.concatenate(x_parts)
        ys = np.concatenate(y_parts)

        if x_logscale:
            ax.set_xscale("log")
            x_lo, x_hi = _log_pad(float(xs.min()), float(xs.max()))
        else:
            x_lo, x_hi = _linear_pad(float(xs.min()), float(xs.max()))

        y_lo, y_hi = _linear_pad(float(ys.min()), float(ys.max()))

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
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        _ref_kw = {"color": "green", "linestyle": "--", "linewidth": 5}
        if reference_line == "y=x":
            lo = max(x_lo, y_lo)
            hi = min(x_hi, y_hi)
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                ax.plot([lo, hi], [lo, hi], **_ref_kw)
        elif reference_line == "x=1":
            ax.axvline(1.0, **_ref_kw)

        ax_top.set_title(bk.title(), fontsize=10, pad=4)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

        if mu.any():
            _kde_curve_x(ax_top, xu[mu], "black", x_lo, x_hi)
            _kde_curve_y(ax_right, yu[mu], "black", y_lo, y_hi)
        if mp.any():
            _kde_curve_x(ax_top, xp[mp], "red", x_lo, x_hi)
            _kde_curve_y(ax_right, yp[mp], "red", y_lo, y_hi)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax_top.set_ylim(bottom=0)
        ax_right.set_xlim(left=0)

    for j in range(n_total, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.subplots_adjust(
        left=0.06, right=0.96, top=0.94, bottom=0.06, wspace=0.28, hspace=0.36
    )
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"{fig_basename}.{fig_format}"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)


def _plot_exc_vs_inh_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
    fig_format: str,
) -> None:
    _plot_pref_unpref_scatter_by_compartment(
        pref_unpref_data,
        xvar="inh",
        yvar="exc",
        x_logscale=False,
        save_path=save_path,
        fig_format=fig_format,
        fig_basename="exc_vs_inh_by_compartment_pref_unpref",
        reference_line="y=x",
    )


def _plot_vout_vs_eiratio_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
    fig_format: str,
) -> None:
    _plot_pref_unpref_scatter_by_compartment(
        pref_unpref_data,
        xvar="ei_ratio",
        yvar="vout",
        x_logscale=True,
        save_path=save_path,
        fig_format=fig_format,
        fig_basename="vout_vs_eiratio_by_compartment_pref_unpref",
        reference_line="x=1",
    )
