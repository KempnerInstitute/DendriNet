import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from torch.utils.data import TensorDataset

from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.dendrinet import DendriticBranchLayer
from dendritic_modeling.information import (
    compute_mi_cc,
    compute_mi_cd,
    compute_pairwise_sr,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.utils import (
    add_gaussian_noise,
    add_uniform_noise,
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    save_dict,
)


class EIInformationAnalyzer:
    def __init__(
        self,
        compute_mi_E_I: bool = False,
        compute_mi_EI_Vout: bool = False,
        compute_sr_EI_Vout: bool = False,
        compute_mi_EI_label: bool = False,
        n_neighbors: int = 10,
    ):
        self.mi_E_I = compute_mi_E_I
        self.mi_EI_Vout = compute_mi_EI_Vout
        self.sr_EI_Vout = compute_sr_EI_Vout
        self.mi_EI_label = compute_mi_EI_label
        self.data_dict = {}
        self.n_neighbors = n_neighbors

    def attach_forward_hooks(self, model: BaseModel):
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    if name not in self.data_dict.keys():
                        self.data_dict[name] = {}
                        module._name = name
                        handle = module.register_forward_hook(self.forward_hook)
                        handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: List[torch.utils.hooks.RemovableHandle]):
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        output: torch.Tensor,
    ):
        excitatory_input, inhibitory_input, _ = input
        key = module._name
        self.data_dict[key]["excitation"] = module.branch_excitation(excitatory_input)
        self.data_dict[key]["inhibition"] = module.branch_inhibition(inhibitory_input)
        if self.mi_EI_Vout:
            self.data_dict[key]["Vout"] = output

    def compute_mi_E_I(self, excitation: np.ndarray, inhibition: np.ndarray) -> float:
        return compute_mi_cc(excitation, inhibition, self.n_neighbors)

    def compute_mi_EI_Vout(self, EI: np.ndarray, Vout: np.ndarray) -> float:
        return compute_mi_cc(EI, Vout, self.n_neighbors)

    def compute_sr_EI_Vout(
        self,
        excitation: np.ndarray,
        inhibition: np.ndarray,
        Vout: np.ndarray,
        mi_EI_Vout: Optional[float] = None,
    ) -> float:
        return compute_pairwise_sr(
            excitation, inhibition, Vout, self.n_neighbors, mi_EI_Vout
        )

    def compute_mi_EI_label(self, EI: np.ndarray, labels: np.ndarray) -> float:
        return compute_mi_cd(EI, labels, self.n_neighbors)

    def analyze(
        self,
        model: BaseModel,
        dataset: torch.utils.data.Dataset,
        device: str = "cpu",  # 'cuda' if torch.cuda.is_available() else 'cpu'
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork), (
                "model.net must be an instance of ExcitationInhibitionNetwork. "
                + f"Got model.net type: {type(model.net)}"
            )
        except AssertionError:
            return

        model = model.to(device)
        model.eval()

        inputs, labels = dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor

        handles = self.attach_forward_hooks(model)

        _ = model(inputs)

        excitation = []
        inhibition = []
        if self.mi_EI_Vout:
            Vout = []
        for branch_dict in self.data_dict.values():
            excitation.append(branch_dict["excitation"])
            inhibition.append(branch_dict["inhibition"])
            if self.mi_EI_Vout:
                Vout.append(branch_dict["Vout"])

        excitation = torch.cat(excitation, dim=-1).detach().cpu().numpy()
        inhibition = torch.cat(inhibition, dim=-1).detach().cpu().numpy()

        if self.mi_EI_label:
            labels = labels.detach().cpu().numpy()
            label_len = len(labels)
            for i, dim_size in enumerate(excitation.shape):
                if dim_size == label_len:
                    break
            new_shape = [1] * excitation.ndim
            new_shape[i] = label_len
            labels = labels.reshape(new_shape)
            labels = np.broadcast_to(labels, excitation.shape).flatten()

        excitation = excitation.flatten()[:, None]
        inhibition = inhibition.flatten()[:, None]

        if self.mi_EI_Vout or self.mi_EI_label:
            EI = np.concatenate([excitation, inhibition], axis=1)

        info_dict = {}

        if self.mi_E_I:
            info_dict["mi_E_I"] = self.compute_mi_E_I(excitation, inhibition)

        if self.mi_EI_Vout:
            Vout = torch.cat(Vout, dim=-1).detach().cpu().flatten().numpy()[:, None]
            info_dict["mi_EI_Vout"] = self.compute_mi_EI_Vout(EI, Vout)

        if self.sr_EI_Vout:
            mi_EI_Vout = None
            if self.mi_EI_Vout:
                mi_EI_Vout = info_dict["mi_EI_Vout"]
            (
                si_EI_Vout,
                ri_EI_Vout,
                mi_EI_Vout,
                mi_E_Vout,
                mi_I_Vout,
            ) = self.compute_sr_EI_Vout(excitation, inhibition, Vout, mi_EI_Vout)
            info_dict["si_EI_Vout"] = si_EI_Vout
            info_dict["ri_EI_Vout"] = ri_EI_Vout
            info_dict["mi_EI_Vout"] = mi_EI_Vout
            info_dict["mi_E_Vout"] = mi_E_Vout
            info_dict["mi_I_Vout"] = mi_I_Vout

        if self.mi_EI_label:
            info_dict["mi_EI_label"] = self.compute_mi_EI_label(EI, labels)

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(info_dict, save_path, filename)
        else:
            return info_dict


class SynapticTurnoverAnalyzer:
    def __init__(self, model: BaseModel):
        assert isinstance(model.net, ExcitationInhibitionNetwork), (
            "model.net must be an instance of ExcitationInhibitionNetwork. "
            + f"Got model.net type: {type(model.net)}"
        )

        self.data_dict_list = []
        self.analyze(model=model, training=True)

    def analyze(
        self,
        model: BaseModel,
        training: bool = True,
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        if training:
            data_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, DendriticBranchLayer):
                    data_dict[name] = {}
                    if module.branch_excitation is not None:
                        data_dict[name]["exc_mask"] = (
                            deepcopy(module.branch_excitation.weight_mask())
                            .detach()
                            .cpu()
                        )
                    if module.branch_inhibition is not None:
                        data_dict[name]["inh_mask"] = (
                            deepcopy(module.branch_inhibition.weight_mask())
                            .detach()
                            .cpu()
                        )
            self.data_dict_list.append(data_dict)
        else:

            def turnover_helper(
                prev_mask: torch.Tensor,
                curr_mask: torch.Tensor,
                total_active: int,
                total_stable: int,
                total_transient: int,
            ) -> Tuple[int, int, int]:
                mask_diff = curr_mask - prev_mask
                curr_active = torch.sum(curr_mask).item()
                curr_transient = torch.sum(mask_diff > 1e-3).item()
                curr_stable = curr_active - curr_transient
                total_active += curr_active
                total_stable += curr_stable
                total_transient += curr_transient
                return total_active, total_stable, total_transient

            turnover_dict = {}
            for i in range(1, len(self.data_dict_list)):
                prev_data_dict: dict = self.data_dict_list[i - 1]
                curr_data_dict: dict = self.data_dict_list[i]

                exc_total_active = 0
                exc_total_stable = 0
                exc_total_transient = 0

                inh_total_active = 0
                inh_total_stable = 0
                inh_total_transient = 0

                for name in curr_data_dict.keys():
                    prev_module_dict: dict = prev_data_dict[name]
                    curr_module_dict: dict = curr_data_dict[name]

                    if "exc_mask" in curr_module_dict.keys():
                        prev_exc_mask = prev_module_dict["exc_mask"]
                        curr_exc_mask = curr_module_dict["exc_mask"]
                        (
                            exc_total_active,
                            exc_total_stable,
                            exc_total_transient,
                        ) = turnover_helper(
                            prev_exc_mask,
                            curr_exc_mask,
                            exc_total_active,
                            exc_total_stable,
                            exc_total_transient,
                        )

                    if "inh_mask" in curr_module_dict.keys():
                        prev_inh_mask = prev_module_dict["inh_mask"]
                        curr_inh_mask = curr_module_dict["inh_mask"]
                        (
                            inh_total_active,
                            inh_total_stable,
                            inh_total_transient,
                        ) = turnover_helper(
                            prev_inh_mask,
                            curr_inh_mask,
                            inh_total_active,
                            inh_total_stable,
                            inh_total_transient,
                        )

                turnover_dict[i] = {}
                turnover_dict[i]["exc"] = {
                    "active": exc_total_active,
                    "stable": exc_total_stable,
                    "transient": exc_total_transient,
                }
                turnover_dict[i]["inh"] = {
                    "active": inh_total_active,
                    "stable": inh_total_stable,
                    "transient": inh_total_transient,
                }
                turnover_dict[i]["both"] = {
                    "active": exc_total_active + inh_total_active,
                    "stable": exc_total_stable + inh_total_stable,
                    "transient": exc_total_transient + inh_total_transient,
                }

            if save_path is not None:
                save_dict(turnover_dict, save_path, filename)
            else:
                return turnover_dict


class NoisePerturbationAnalyzer:
    def __init__(
        self,
        uniform_noise: bool = False,
        uniform_magnitudes: List[float] = [0.1],
        gaussian_noise: bool = False,
        gaussian_sdevs: List[float] = [0.1],
        n_samples: int = 10,
        accuracy: bool = True,
        categorical_loglikelihood: bool = True,
    ):
        self.uniform_noise = uniform_noise
        self.uniform_magnitudes = uniform_magnitudes
        self.gaussian_noise = gaussian_noise
        self.gaussian_sdevs = gaussian_sdevs
        self.n_samples = n_samples
        self.accuracy = accuracy
        self.categorical_ll = categorical_loglikelihood

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        model = model.to(device)
        x, y = test_dataset[:]
        x: torch.Tensor = x.to(device)
        y: torch.Tensor = y.to(device)
        y = y[None, ...].expand(self.n_samples, *y.shape)

        noise_results = {}

        if self.uniform_noise:
            noise_results["uniform"] = {}
            for mag in self.uniform_magnitudes:
                noise_results["uniform"][f"{mag}"] = {}
                noisy_x = add_uniform_noise(x, spread=mag, n_samples=self.n_samples)
                if self.accuracy:
                    accs = evaluate_accuracy(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["uniform"][f"{mag}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    cat_lls = cat_lls.cpu().tolist()
                    noise_results["uniform"][f"{mag}"]["categorical_ll"] = cat_lls

        if self.gaussian_noise:
            noise_results["gaussian"] = {}
            for sdev in self.gaussian_sdevs:
                noise_results["gaussian"][f"{sdev}"] = {}
                noisy_x = add_gaussian_noise(x, std=sdev, n_samples=self.n_samples)
                if self.accuracy:
                    accs = evaluate_accuracy(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    accs = accs.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["accuracy"] = accs
                if self.categorical_ll:
                    cat_lls = evaluate_categorical_loglikelihood(
                        classifier=model,
                        test_ds=TensorDataset(noisy_x, y),
                        reduce_dim=1,
                        move_device=False,
                    )[-1]
                    cat_lls = cat_lls.cpu().tolist()
                    noise_results["gaussian"][f"{sdev}"]["categorical_ll"] = cat_lls

        if save_path is not None:
            save_dict(noise_results, save_path, filename)
        else:
            return noise_results


class SingleLayerContributionAnalyzer:
    """
    Analyzes the contribution of individual layers by isolating them.
    This is the renamed version of "layer_ablation" - it removes all other layers
    to test what each layer can do on its own.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = "cuda",
        save_plots: bool = True,
        save_csv: bool = True,
    ):
        self.batch_size = batch_size
        self.device = device
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        save_path: Optional[str] = None,
        filename: str = "single_layer_contribution",
    ):
        """Analyze each layer's contribution in isolation"""
        import matplotlib.pyplot as plt
        import pandas as pd

        from dendritic_modeling.utils import measure_performance

        self.logger.info("Starting single layer contribution analysis...")

        # Ensure model is on the correct device
        model = model.to(self.device)
        # Get the underlying network
        if hasattr(model, "net"):
            net = model.net
        else:
            net = model

        # Debug logging
        self.logger.info(f"Model type: {type(model)}")
        self.logger.info(f"Net type: {type(net)}")
        self.logger.info(
            f"Net attributes: {[attr for attr in dir(net) if 'branch' in attr]}"
        )

        # Check layers structure
        if hasattr(net, "layers"):
            self.logger.info(f"Number of EI layers: {len(net.layers)}")
            for i, layer in enumerate(net.layers):
                self.logger.info(
                    f"Layer {i}: has excitatory_cells={hasattr(layer, 'excitatory_cells')}, has inhibitory_cells={hasattr(layer, 'inhibitory_cells')}"
                )
                if (
                    hasattr(layer, "excitatory_cells")
                    and layer.excitatory_cells is not None
                ):
                    self.logger.info(
                        f"  Excitatory cells type: {type(layer.excitatory_cells)}"
                    )
                    self.logger.info(
                        f"  Excitatory has branch_layers: {hasattr(layer.excitatory_cells, 'branch_layers')}"
                    )
                if (
                    hasattr(layer, "inhibitory_cells")
                    and layer.inhibitory_cells is not None
                ):
                    self.logger.info(
                        f"  Inhibitory cells type: {type(layer.inhibitory_cells)}"
                    )
                    self.logger.info(
                        f"  Inhibitory has branch_layers: {hasattr(layer.inhibitory_cells, 'branch_layers')}"
                    )

        # Get branch layers
        try:
            branch_layers = net.branch_layers
            self.logger.info(
                f"Successfully got branch_layers: {len(branch_layers)} layers"
            )
        except Exception as e:
            self.logger.error(f"Error accessing branch_layers: {type(e).__name__}: {e}")
            return

        # Get baseline performance with all layers active
        baseline_acc = measure_performance(net, test_dataset, device=self.device)
        self.logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # Log layer structure for debugging
        self.logger.info(f"Total branch layers: {len(branch_layers)}")
        for i, layer in enumerate(branch_layers):
            has_exc = (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            )
            has_inh = (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            )
            self.logger.info(
                f"Layer {i}: has_excitation={has_exc}, has_inhibition={has_inh}"
            )

        # Store original parameters
        original_params = {}
        for n, p in net.named_parameters():
            original_params[n] = p.detach().clone()

        # Initialize results
        results = {
            "layer_idx": [],
            "isolated_baseline_acc": [],
            "isolated_exc_only_acc": [],
            "isolated_inh_only_acc": [],
            "isolated_neither_acc": [],
        }

        # Analyze each layer in isolation
        for layer_idx in range(len(branch_layers)):
            self.logger.info(f"Analyzing layer {layer_idx} in isolation...")

            # First, ablate all OTHER layers (set to very negative values)
            for i in range(len(branch_layers)):
                if i != layer_idx:
                    if (
                        hasattr(branch_layers[i], "branch_excitation")
                        and branch_layers[i].branch_excitation is not None
                    ):
                        branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)
                    if (
                        hasattr(branch_layers[i], "branch_inhibition")
                        and branch_layers[i].branch_inhibition is not None
                    ):
                        branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)

            # Measure with only this layer active (both E and I)
            isolated_baseline = measure_performance(
                net, test_dataset, device=self.device
            )

            # Test with only excitatory active
            if (
                hasattr(branch_layers[layer_idx], "branch_inhibition")
                and branch_layers[layer_idx].branch_inhibition is not None
            ):
                branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
                exc_only_acc = measure_performance(
                    net, test_dataset, device=self.device
                )
                # Restore inhibitory
                for n, p in net.named_parameters():
                    if f"branch_layers.{layer_idx}.branch_inhibition" in n:
                        p.data.copy_(original_params[n])
            else:
                exc_only_acc = isolated_baseline

            # Test with only inhibitory active
            if (
                hasattr(branch_layers[layer_idx], "branch_excitation")
                and branch_layers[layer_idx].branch_excitation is not None
            ):
                branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
                inh_only_acc = measure_performance(
                    net, test_dataset, device=self.device
                )
                # Restore excitatory
                for n, p in net.named_parameters():
                    if f"branch_layers.{layer_idx}.branch_excitation" in n:
                        p.data.copy_(original_params[n])
            else:
                inh_only_acc = isolated_baseline

            # Test with neither (both ablated)
            if (
                hasattr(branch_layers[layer_idx], "branch_excitation")
                and branch_layers[layer_idx].branch_excitation is not None
            ):
                branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
            if (
                hasattr(branch_layers[layer_idx], "branch_inhibition")
                and branch_layers[layer_idx].branch_inhibition is not None
            ):
                branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
            neither_acc = measure_performance(net, test_dataset, device=self.device)

            # Restore all parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Store results
            results["layer_idx"].append(layer_idx)
            results["isolated_baseline_acc"].append(isolated_baseline)
            results["isolated_exc_only_acc"].append(exc_only_acc)
            results["isolated_inh_only_acc"].append(inh_only_acc)
            results["isolated_neither_acc"].append(neither_acc)

            self.logger.info(
                f"Layer {layer_idx}: baseline={isolated_baseline:.4f}, "
                f"exc_only={exc_only_acc:.4f}, inh_only={inh_only_acc:.4f}, "
                f"neither={neither_acc:.4f}"
            )

        # Save results
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            if self.save_csv:
                df = pd.DataFrame(results)
                df["full_network_baseline"] = baseline_acc
                csv_path = os.path.join(save_path, f"{filename}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved results to {csv_path}")

            if self.save_plots:
                # Reverse layer indices so soma (last layer) becomes 0
                num_layers = len(results["layer_idx"])
                reversed_layer_idx = [
                    num_layers - 1 - idx for idx in results["layer_idx"]
                ]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Absolute accuracy values
                x = np.array(reversed_layer_idx)
                ax1.plot(
                    x,
                    results["isolated_baseline_acc"],
                    "o-",
                    label="Both E+I",
                    linewidth=3,
                    markersize=10,
                    color="purple",
                )
                ax1.plot(
                    x,
                    results["isolated_exc_only_acc"],
                    "s-",
                    label="Excitatory only",
                    linewidth=3,
                    markersize=10,
                    color="red",
                )
                ax1.plot(
                    x,
                    results["isolated_inh_only_acc"],
                    "^-",
                    label="Inhibitory only",
                    linewidth=3,
                    markersize=10,
                    color="blue",
                )
                ax1.plot(
                    x,
                    results["isolated_neither_acc"],
                    "v-",
                    label="Neither",
                    linewidth=3,
                    markersize=10,
                    color="gray",
                )
                ax1.axhline(
                    y=baseline_acc,
                    color="k",
                    linestyle="--",
                    linewidth=2,
                    label=f"Full network baseline ({baseline_acc:.3f})",
                )

                # Set x-axis ticks and labels
                ax1.set_xticks(x)
                # Create labels where 0=soma and higher numbers=more distal
                x_labels = ["Soma" if pos == 0 else f"{pos}" for pos in x]
                ax1.set_xticklabels(x_labels)

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Classification Accuracy", fontsize=14)
                ax1.set_title(
                    "Single Layer Contribution Analysis\n(Each layer tested in isolation)",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(
                    [0, max(max(results["isolated_baseline_acc"]), baseline_acc) * 1.1]
                )

                # Right plot: Contribution breakdown (bar chart)
                width = 0.25

                # Calculate contributions
                exc_contrib = [
                    b - n
                    for b, n in zip(
                        results["isolated_exc_only_acc"],
                        results["isolated_neither_acc"],
                    )
                ]
                inh_contrib = [
                    b - n
                    for b, n in zip(
                        results["isolated_inh_only_acc"],
                        results["isolated_neither_acc"],
                    )
                ]
                synergy = [
                    full - max(e, i)
                    for full, e, i in zip(
                        results["isolated_baseline_acc"],
                        results["isolated_exc_only_acc"],
                        results["isolated_inh_only_acc"],
                    )
                ]

                # Debug: Print contributions before reversal
                self.logger.info(
                    f"Before reversal - Excitatory contributions: {exc_contrib}"
                )
                self.logger.info(
                    f"Before reversal - Inhibitory contributions: {inh_contrib}"
                )
                self.logger.info(f"Before reversal - Synergy: {synergy}")

                # Reverse the contributions too
                exc_contrib = [exc_contrib[i] for i in range(num_layers - 1, -1, -1)]
                inh_contrib = [inh_contrib[i] for i in range(num_layers - 1, -1, -1)]
                synergy = [synergy[i] for i in range(num_layers - 1, -1, -1)]

                # Debug: Print contributions after reversal
                self.logger.info(
                    f"After reversal - Excitatory contributions: {exc_contrib}"
                )
                self.logger.info(
                    f"After reversal - Inhibitory contributions: {inh_contrib}"
                )
                self.logger.info(f"After reversal - Synergy: {synergy}")
                self.logger.info(f"Plotting x positions: {x.tolist()}")

                ax2.bar(
                    x - width,
                    exc_contrib,
                    width,
                    label="Excitatory contribution",
                    color="red",
                    alpha=0.7,
                )
                ax2.bar(
                    x,
                    inh_contrib,
                    width,
                    label="Inhibitory contribution",
                    color="blue",
                    alpha=0.7,
                )
                ax2.bar(
                    x + width,
                    synergy,
                    width,
                    label="E+I synergy",
                    color="purple",
                    alpha=0.7,
                )

                # Set x-axis ticks and labels
                ax2.set_xticks(x)
                # Create labels where 0=soma and higher numbers=more distal
                x_labels = ["Soma" if pos == 0 else f"{pos}" for pos in x]
                ax2.set_xticklabels(x_labels)

                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Accuracy Contribution", fontsize=14)
                ax2.set_title(
                    "Layer Contributions Breakdown\n(When tested in isolation)",
                    fontsize=16,
                )
                ax2.legend(fontsize=12)
                ax2.grid(True, axis="y", alpha=0.3)
                ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                self.logger.info(f"Saved plot to {plot_path}")

        return results


class SynapticAblationAnalyzer:
    """
    Performs synaptic ablation analysis - removes specific synapses while keeping
    the rest of the network intact. This is the renamed version of "branch_ablation".
    """

    def __init__(
        self,
        ablation_type: str = "whole_layer",  # "whole_layer" or "single_branch"
        layers_to_analyze: List[int] = [],  # Empty means all layers
        mode: str = "direct",  # "direct" or "full"
        batch_size: int = 32,
        device: str = "cuda",
        save_plots: bool = True,
        save_csv: bool = True,
    ):
        self.ablation_type = ablation_type
        self.layers_to_analyze = layers_to_analyze
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        save_path: Optional[str] = None,
        filename: str = "synaptic_ablation",
    ):
        """Perform synaptic ablation analysis"""
        import matplotlib.pyplot as plt
        import pandas as pd

        from dendritic_modeling.utils import ablate_branch, measure_performance

        self.logger.info(
            f"Starting synaptic ablation analysis (type: {self.ablation_type})..."
        )

        # Ensure model is on the correct device
        model = model.to(self.device)
        # Get the underlying network
        if hasattr(model, "net"):
            net = model.net
        else:
            net = model

        # Debug logging
        self.logger.info(f"Model type: {type(model)}")
        self.logger.info(f"Net type: {type(net)}")
        self.logger.info(
            f"Net attributes: {[attr for attr in dir(net) if 'branch' in attr]}"
        )

        # Check layers structure
        if hasattr(net, "layers"):
            self.logger.info(f"Number of EI layers: {len(net.layers)}")
            for i, layer in enumerate(net.layers):
                self.logger.info(
                    f"Layer {i}: has excitatory_cells={hasattr(layer, 'excitatory_cells')}, has inhibitory_cells={hasattr(layer, 'inhibitory_cells')}"
                )
                if (
                    hasattr(layer, "excitatory_cells")
                    and layer.excitatory_cells is not None
                ):
                    self.logger.info(
                        f"  Excitatory cells type: {type(layer.excitatory_cells)}"
                    )
                    self.logger.info(
                        f"  Excitatory has branch_layers: {hasattr(layer.excitatory_cells, 'branch_layers')}"
                    )
                if (
                    hasattr(layer, "inhibitory_cells")
                    and layer.inhibitory_cells is not None
                ):
                    self.logger.info(
                        f"  Inhibitory cells type: {type(layer.inhibitory_cells)}"
                    )
                    self.logger.info(
                        f"  Inhibitory has branch_layers: {hasattr(layer.inhibitory_cells, 'branch_layers')}"
                    )

        # Get branch layers
        try:
            branch_layers = net.branch_layers
            self.logger.info(
                f"Successfully got branch_layers: {len(branch_layers)} layers"
            )
        except Exception as e:
            self.logger.error(f"Error accessing branch_layers: {type(e).__name__}: {e}")
            return

        # Get baseline performance
        baseline_acc = measure_performance(net, test_dataset, device=self.device)
        self.logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # Log layer structure for debugging
        self.logger.info(f"Total branch layers: {len(branch_layers)}")
        for i, layer in enumerate(branch_layers):
            has_exc = (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            )
            has_inh = (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            )
            self.logger.info(
                f"Layer {i}: has_excitation={has_exc}, has_inhibition={has_inh}"
            )

        # Determine which layers to analyze
        if self.layers_to_analyze:
            layer_indices = self.layers_to_analyze
        else:
            layer_indices = list(range(len(branch_layers)))

        # Store original parameters
        original_params = {}
        for n, p in net.named_parameters():
            original_params[n] = p.detach().clone()

        # Initialize results based on ablation type
        if self.ablation_type == "whole_layer":
            results = self._analyze_whole_layers(
                net,
                branch_layers,
                layer_indices,
                baseline_acc,
                original_params,
                test_dataset,
            )
        else:  # single_branch
            results = self._analyze_single_branches(
                net,
                branch_layers,
                layer_indices,
                baseline_acc,
                original_params,
                test_dataset,
            )

        # Save results
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self._save_results(results, baseline_acc, save_path, filename)

        return results

    def _analyze_whole_layers(
        self,
        net,
        branch_layers,
        layer_indices,
        baseline_acc,
        original_params,
        test_dataset,
    ):
        """Analyze ablation of whole layers"""
        from dendritic_modeling.utils import measure_performance

        results = {
            "layer_idx": [],
            "both_ablated_acc": [],
            "exc_ablated_acc": [],
            "inh_ablated_acc": [],
            "both_drop": [],
            "exc_drop": [],
            "inh_drop": [],
        }

        for layer_idx in layer_indices:
            layer = branch_layers[layer_idx]
            self.logger.info(f"Ablating layer {layer_idx}...")

            # Ablate both E and I
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            ):
                layer.branch_excitation.pre_w.data.zero_()
            if (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            ):
                layer.branch_inhibition.pre_w.data.zero_()
            both_acc = measure_performance(net, test_dataset, device=self.device)

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Ablate only excitatory
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            ):
                layer.branch_excitation.pre_w.data.zero_()
                exc_acc = measure_performance(net, test_dataset, device=self.device)
            else:
                exc_acc = baseline_acc

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Ablate only inhibitory
            if (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            ):
                layer.branch_inhibition.pre_w.data.zero_()
                inh_acc = measure_performance(net, test_dataset, device=self.device)
            else:
                inh_acc = baseline_acc

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Store results
            results["layer_idx"].append(layer_idx)
            results["both_ablated_acc"].append(both_acc)
            results["exc_ablated_acc"].append(exc_acc)
            results["inh_ablated_acc"].append(inh_acc)
            results["both_drop"].append(baseline_acc - both_acc)
            results["exc_drop"].append(baseline_acc - exc_acc)
            results["inh_drop"].append(baseline_acc - inh_acc)

            self.logger.info(
                f"Layer {layer_idx}: both_drop={baseline_acc - both_acc:.4f}, "
                f"exc_drop={baseline_acc - exc_acc:.4f}, inh_drop={baseline_acc - inh_acc:.4f}"
            )

        return results

    def _analyze_single_branches(
        self,
        net,
        branch_layers,
        layer_indices,
        baseline_acc,
        original_params,
        test_dataset,
    ):
        """Analyze ablation of individual branches"""
        from dendritic_modeling.utils import ablate_branch, measure_performance

        results = {
            "branch_id": [],
            "layer_idx": [],
            "branch_idx": [],
            "both_ablated_acc": [],
            "exc_ablated_acc": [],
            "inh_ablated_acc": [],
            "both_drop": [],
            "exc_drop": [],
            "inh_drop": [],
        }

        for layer_idx in layer_indices:
            layer = branch_layers[layer_idx]

            # Determine number of branches
            n_branches = 0
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
                and hasattr(layer.branch_excitation, "pre_w")
            ):
                n_branches = layer.branch_excitation.pre_w.shape[0]
            elif (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
                and hasattr(layer.branch_inhibition, "pre_w")
            ):
                n_branches = layer.branch_inhibition.pre_w.shape[0]

            if n_branches == 0:
                continue

            self.logger.info(f"Analyzing {n_branches} branches in layer {layer_idx}...")

            for branch_idx in range(n_branches):
                # Ablate both E and I for this branch
                ablate_branch(net, layer_idx, [branch_idx], mode=self.mode)
                both_acc = measure_performance(net, test_dataset, device=self.device)

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Ablate only excitatory
                if (
                    hasattr(layer, "branch_excitation")
                    and layer.branch_excitation is not None
                ):
                    layer.branch_excitation.pre_w.data[branch_idx].zero_()
                    exc_acc = measure_performance(net, test_dataset, device=self.device)
                else:
                    exc_acc = baseline_acc

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Ablate only inhibitory
                if (
                    hasattr(layer, "branch_inhibition")
                    and layer.branch_inhibition is not None
                ):
                    layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                    inh_acc = measure_performance(net, test_dataset, device=self.device)
                else:
                    inh_acc = baseline_acc

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Store results
                branch_id = f"{layer_idx}-{branch_idx}"
                results["branch_id"].append(branch_id)
                results["layer_idx"].append(layer_idx)
                results["branch_idx"].append(branch_idx)
                results["both_ablated_acc"].append(both_acc)
                results["exc_ablated_acc"].append(exc_acc)
                results["inh_ablated_acc"].append(inh_acc)
                results["both_drop"].append(baseline_acc - both_acc)
                results["exc_drop"].append(baseline_acc - exc_acc)
                results["inh_drop"].append(baseline_acc - inh_acc)

        return results

    def _save_results(self, results, baseline_acc, save_path, filename):
        """Save ablation results"""
        import matplotlib.pyplot as plt
        import pandas as pd

        if self.save_csv:
            df = pd.DataFrame(results)
            df["baseline_acc"] = baseline_acc
            csv_path = os.path.join(save_path, f"{filename}_{self.ablation_type}.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved results to {csv_path}")

        if self.save_plots:
            if self.ablation_type == "whole_layer":
                # Reverse layer indices so soma (last layer) becomes 0
                num_layers = len(results["layer_idx"])
                reversed_layer_idx = [
                    num_layers - 1 - idx for idx in results["layer_idx"]
                ]

                # Reverse the results too
                both_drop_reversed = [
                    results["both_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]
                exc_drop_reversed = [
                    results["exc_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]
                inh_drop_reversed = [
                    results["inh_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Accuracy drop (bar chart)
                x = np.array(reversed_layer_idx)
                width = 0.25

                ax1.bar(
                    x - width,
                    both_drop_reversed,
                    width,
                    label="Both E+I ablated",
                    color="purple",
                    alpha=0.7,
                )
                ax1.bar(
                    x,
                    exc_drop_reversed,
                    width,
                    label="Excitatory ablated",
                    color="red",
                    alpha=0.7,
                )
                ax1.bar(
                    x + width,
                    inh_drop_reversed,
                    width,
                    label="Inhibitory ablated",
                    color="blue",
                    alpha=0.7,
                )

                # Set x-axis ticks and labels
                ax1.set_xticks(x)
                # Create labels where 0=soma and higher numbers=more distal
                x_labels = ["Soma" if pos == 0 else f"{pos}" for pos in x]
                ax1.set_xticklabels(x_labels)

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Accuracy Drop", fontsize=14)
                ax1.set_title(
                    f"Synaptic Ablation Analysis - Whole Layers\n(Baseline accuracy: {baseline_acc:.3f})",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, axis="y", alpha=0.3)
                ax1.set_ylim([0, max(max(both_drop_reversed), 0.1) * 1.2])

                # Right plot: Remaining accuracy after ablation
                both_acc_reversed = [
                    results["both_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_acc_reversed = [
                    results["exc_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_acc_reversed = [
                    results["inh_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]

                ax2.plot(
                    x,
                    [baseline_acc] * len(x),
                    "k--",
                    linewidth=2,
                    label=f"Baseline ({baseline_acc:.3f})",
                )
                ax2.plot(
                    x,
                    both_acc_reversed,
                    "o-",
                    label="Both E+I ablated",
                    linewidth=3,
                    markersize=10,
                    color="purple",
                )
                ax2.plot(
                    x,
                    exc_acc_reversed,
                    "s-",
                    label="Excitatory ablated",
                    linewidth=3,
                    markersize=10,
                    color="red",
                )
                ax2.plot(
                    x,
                    inh_acc_reversed,
                    "^-",
                    label="Inhibitory ablated",
                    linewidth=3,
                    markersize=10,
                    color="blue",
                )

                # Set x-axis ticks and labels
                ax2.set_xticks(x)
                # Create labels where 0=soma and higher numbers=more distal
                x_labels = ["Soma" if pos == 0 else f"{pos}" for pos in x]
                ax2.set_xticklabels(x_labels)

                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Classification Accuracy", fontsize=14)
                ax2.set_title("Remaining Accuracy After Layer Ablation", fontsize=16)
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, baseline_acc * 1.1])

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}_whole_layer.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

            else:  # single_branch
                # Calculate layer-wise statistics
                df = pd.DataFrame(results)
                layer_stats = (
                    df.groupby("layer_idx")
                    .agg(
                        {
                            "both_drop": ["mean", "std"],
                            "exc_drop": ["mean", "std"],
                            "inh_drop": ["mean", "std"],
                        }
                    )
                    .reset_index()
                )

                # Reverse layer indices
                num_layers = len(layer_stats)
                reversed_layer_idx = [num_layers - 1 - idx for idx in range(num_layers)]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Mean accuracy drop per layer
                x = np.array(reversed_layer_idx)
                width = 0.25

                # Reverse the statistics
                both_mean = [
                    layer_stats["both_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                both_std = [
                    layer_stats["both_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_mean = [
                    layer_stats["exc_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_std = [
                    layer_stats["exc_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_mean = [
                    layer_stats["inh_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_std = [
                    layer_stats["inh_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]

                ax1.bar(
                    x - width,
                    both_mean,
                    width,
                    yerr=both_std,
                    label="Both E+I",
                    color="purple",
                    alpha=0.7,
                    capsize=5,
                )
                ax1.bar(
                    x,
                    exc_mean,
                    width,
                    yerr=exc_std,
                    label="Excitatory",
                    color="red",
                    alpha=0.7,
                    capsize=5,
                )
                ax1.bar(
                    x + width,
                    inh_mean,
                    width,
                    yerr=inh_std,
                    label="Inhibitory",
                    color="blue",
                    alpha=0.7,
                    capsize=5,
                )

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Mean Accuracy Drop (± std)", fontsize=14)
                ax1.set_title(
                    f"Synaptic Ablation - Single Branches\n(Baseline: {baseline_acc:.3f})",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, axis="y", alpha=0.3)

                # Right plot: Distribution of drops across branches
                # Create box plots for each layer
                positions = []
                data_to_plot = []
                labels = []
                colors = []

                for i, layer_idx in enumerate(reversed_layer_idx):
                    original_idx = num_layers - 1 - layer_idx
                    layer_data = df[df["layer_idx"] == original_idx]

                    positions.extend([i * 3 - 0.6, i * 3, i * 3 + 0.6])
                    data_to_plot.extend(
                        [
                            layer_data["both_drop"].values,
                            layer_data["exc_drop"].values,
                            layer_data["inh_drop"].values,
                        ]
                    )
                    if i == 0:  # Only add labels once
                        labels.extend(["Both E+I", "Excitatory", "Inhibitory"])
                    colors.extend(["purple", "red", "blue"])

                bp = ax2.boxplot(
                    data_to_plot, positions=positions, widths=0.5, patch_artist=True
                )

                # Color the boxes
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Set x-axis
                ax2.set_xticks([i * 3 for i in range(len(reversed_layer_idx))])
                ax2.set_xticklabels(reversed_layer_idx)
                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Accuracy Drop Distribution", fontsize=14)
                ax2.set_title("Branch-wise Accuracy Drop Distribution", fontsize=16)
                ax2.grid(True, axis="y", alpha=0.3)

                # Add a custom legend
                legend_elements = [
                    Patch(facecolor="purple", alpha=0.7, label="Both E+I"),
                    Patch(facecolor="red", alpha=0.7, label="Excitatory"),
                    Patch(facecolor="blue", alpha=0.7, label="Inhibitory"),
                ]
                ax2.legend(handles=legend_elements, fontsize=12)

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}_single_branch.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Save summary statistics with reversed indices
                if self.save_csv:
                    summary_df = pd.DataFrame(
                        {
                            "layer_idx_reversed": reversed_layer_idx,
                            "layer_idx_original": [
                                num_layers - 1 - idx for idx in reversed_layer_idx
                            ],
                            "both_drop_mean": both_mean,
                            "both_drop_std": both_std,
                            "exc_drop_mean": exc_mean,
                            "exc_drop_std": exc_std,
                            "inh_drop_mean": inh_mean,
                            "inh_drop_std": inh_std,
                        }
                    )
                    summary_path = os.path.join(
                        save_path, f"{filename}_single_branch_summary.csv"
                    )
                    summary_df.to_csv(summary_path, index=False)

            self.logger.info(f"Saved plots to {save_path}")


class AnalysisManager:
    def __init__(
        self,
        model: BaseModel,
        data: Dict[str, torch.utils.data.Dataset],
        analysis_config: AnalysisConfig,
        save_root: Optional[str] = None,
        device: str = "cpu",
        train_config: Optional[Dict] = None,
    ):
        self.model = model
        self.data = data
        self.config = analysis_config
        self.train_config = train_config or {}
        self.save_root = save_root
        self.device = device
        os.makedirs(self.save_root, exist_ok=True)
        # Note: Logging should be configured externally at the application startup.
        self.logger = logging.getLogger(__name__)

        ei_info_config = self.config.ei_info_analysis
        self.ei_info_enabled = ei_info_config.get("enabled", False)
        self.ei_info_training = ei_info_config.get("training", False)
        if self.ei_info_enabled:
            ei_info_params: dict = ei_info_config.get("params", {})
            self.ei_info_analyzer = EIInformationAnalyzer(
                compute_mi_E_I=ei_info_params.get("compute_mi_E_I", True),
                compute_mi_EI_Vout=ei_info_params.get("compute_mi_EI_Vout", True),
                compute_sr_EI_Vout=ei_info_params.get("compute_sr_EI_Vout", True),
                compute_mi_EI_label=ei_info_params.get("compute_mi_EI_label", True),
                n_neighbors=ei_info_params.get("n_neighbors", 10),
            )

        syn_turnover_config = self.config.synapse_turnover_analysis
        self.syn_turnover_enabled = syn_turnover_config.get("enabled", False)
        self.syn_turnover_training = syn_turnover_config.get("training", False)
        if self.syn_turnover_enabled:
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(model=model)

        noise_config = self.config.noise_perturbation_analysis
        self.noise_enabled = noise_config.get("enabled", False)
        self.noise_training = noise_config.get("training", False)
        if self.noise_enabled:
            noise_params: dict = noise_config.get("params", {})
            self.noise_perturbation_analyzer = NoisePerturbationAnalyzer(
                uniform_noise=noise_params.get("uniform_noise", True),
                uniform_magnitudes=noise_params.get("uniform_magnitudes", [0.1]),
                gaussian_noise=noise_params.get("gaussian_noise", True),
                gaussian_sdevs=noise_params.get("gaussian_sdevs", [0.1]),
                n_samples=noise_params.get("n_samples", 10),
                accuracy=noise_params.get("accuracy", True),
                categorical_loglikelihood=noise_params.get(
                    "categorical_loglikelihood", True
                ),
            )

        # Single layer contribution analysis (formerly layer_ablation)
        single_layer_config = self.config.single_layer_contribution
        self.single_layer_enabled = single_layer_config.get("enabled", False)
        self.single_layer_training = single_layer_config.get("training", False)
        if self.single_layer_enabled:
            params = single_layer_config.get("params", {})
            # Get batch_size from train config
            batch_size = self.train_config.get("batch_size", 32)
            self.single_layer_analyzer = SingleLayerContributionAnalyzer(
                batch_size=batch_size,
                device=self.device,
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
            )

        # Synaptic ablation analysis (formerly branch_ablation)
        synaptic_ablation_config = self.config.synaptic_ablation
        self.synaptic_ablation_enabled = synaptic_ablation_config.get("enabled", False)
        self.synaptic_ablation_training = synaptic_ablation_config.get(
            "training", False
        )
        if self.synaptic_ablation_enabled:
            params = synaptic_ablation_config.get("params", {})
            # Get batch_size from train config
            batch_size = self.train_config.get("batch_size", 32)
            self.synaptic_ablation_analyzer = SynapticAblationAnalyzer(
                ablation_type=params.get("ablation_type", "whole_layer"),
                layers_to_analyze=params.get("layers_to_analyze", []),
                mode=params.get("mode", "direct"),
                batch_size=batch_size,
                device=self.device,
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
            )

    def run_analysis(self, filename: str = "final", training: bool = False):
        if self.ei_info_enabled and (
            (self.ei_info_training and training) or not training
        ):
            self.ei_info_analyzer.analyze(
                model=self.model,
                dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "ei_info_analysis"),
                filename=filename,
            )

        if self.syn_turnover_enabled and (
            (self.syn_turnover_training and training) or not training
        ):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                filename=filename,
            )

        if self.noise_enabled and ((self.noise_training and training) or not training):
            self.noise_perturbation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "noise_perturbation_analysis"),
                filename=filename,
            )

        if self.single_layer_enabled and (
            (self.single_layer_training and training) or not training
        ):
            self.single_layer_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "single_layer_contribution"),
                filename=filename,
            )

        if self.synaptic_ablation_enabled and (
            (self.synaptic_ablation_training and training) or not training
        ):
            self.synaptic_ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "synaptic_ablation"),
                filename=filename,
            )

        # TODO: add other analysis classes here implemented same as above
