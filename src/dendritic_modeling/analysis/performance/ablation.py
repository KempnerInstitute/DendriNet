"""
Ablation analysis module.

This module contains classes for evaluating model performance by systematically
disabling or ablating different network components to understand their contributions.
"""
import os
from copy import deepcopy
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.layers.dendritic import DendriticBranchLayer
from dendritic_modeling.utils import (
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    save_dict
)


class AblationAnalyzer:
    def __init__(
        self,
        layer_ablation: bool = True,
        compartment_ablation: bool = True,
        ablate_all_synapses: bool = True,
        ablate_excitation: bool = True,
        ablate_inhibition: bool = True,
        ablate_upstream: bool = True,
        accuracy: bool = True,
        categorical_loglikelihood: bool = True,
    ):
        self.layer_ablation = layer_ablation
        self.compartment_ablation = compartment_ablation
        self.ablate_all_synapses = ablate_all_synapses
        self.ablate_excitation = ablate_excitation
        self.ablate_inhibition = ablate_inhibition
        self.ablate_upstream = ablate_upstream
        self.accuracy = accuracy
        self.categorical_ll = categorical_loglikelihood
    
    def ablate_layer_and_evaluate(
        self,
        test_dataset: torch.utils.data.Dataset,
        model: BaseModel,
        module: DendriticBranchLayer,
        module_name: str,
        ablation_type: str,
        baseline_dict: dict,
        results_dict: dict,
        state_dict: dict
    ):
        results_dict[ablation_type][module_name] = {}
        results_dict[ablation_type][module_name]["depth"] = module.layer_idx

        if ablation_type in ["all synapses", "excitation"]:
            module.branch_excitation.pre_w.data.fill_(-1000)
        if ablation_type in ["all synapses", "inhibition"]:
            module.branch_inhibition.pre_w.data.fill_(-1000)
        if ablation_type == "upstream":
            module.branches_to_output.log_weight.data.fill_(-1000)

        if self.accuracy:
            accuracy = evaluate_accuracy(
                classifier=model, test_ds=test_dataset, move_device=False
            )[-1]
            results_dict[ablation_type][module_name]["accuracy"] = \
                baseline_dict["accuracy"] - accuracy
        if self.categorical_ll:
            loglikelihood = evaluate_categorical_loglikelihood(
                classifier=model, test_ds=test_dataset, move_device=False
            )[-1]
            results_dict[ablation_type][module_name]["log likelihood"] = \
                baseline_dict["log likelihood"] - loglikelihood
        
        model.load_state_dict(state_dict)

        return results_dict
    
    def ablate_compartment_and_evaluate(
        self,
        test_dataset: torch.utils.data.Dataset,
        model: BaseModel,
        module: DendriticBranchLayer,
        module_name: str,
        ablation_type: str,
        baseline_dict: dict,
        results_dict: dict,
        state_dict: dict
    ):
        results_dict[ablation_type][module_name] = {}
        results_dict[ablation_type][module_name]["depth"] = module.layer_idx

        if module.branch_excitation is not None:
            n_branches = module.branch_excitation.pre_w.shape[0]
        elif module.branch_inhibition is not None:
            n_branches = module.branch_inhibition.pre_w.shape[0]
        elif hasattr(module, "branches_to_output"):
            n_branches = module.branches_to_output.log_weight.shape[0]
        else:
            raise ValueError(f"Number of compartments cannot be determined for {module_name}")

        if self.accuracy:
            accuracy_list = []
        if self.categorical_ll:
            loglikelihood_list = []

        for branch_ix in range(n_branches):
            if ablation_type in ["all synapses", "excitation"]:
                module.branch_excitation.pre_w.data[branch_ix, :] = -1000
            if ablation_type in ["all synapses", "inhibition"]:
                module.branch_inhibition.pre_w.data[branch_ix, :] = -1000
            if ablation_type == "upstream":
                module.branches_to_output.log_weight.data[branch_ix, :] = -1000

            if self.accuracy:
                accuracy = evaluate_accuracy(
                    classifier=model, test_ds=test_dataset, move_device=False
                )[-1]
                accuracy_list.append(baseline_dict["accuracy"] - accuracy)
            if self.categorical_ll:
                loglikelihood = evaluate_categorical_loglikelihood(
                    classifier=model, test_ds=test_dataset, move_device=False
                )[-1]
                loglikelihood_list.append(baseline_dict["log likelihood"] - loglikelihood)
            
            model.load_state_dict(state_dict)
        
        if self.accuracy:
            results_dict[ablation_type][module_name]["accuracy"] = accuracy_list
        if self.categorical_ll:
            results_dict[ablation_type][module_name]["log likelihood"] = loglikelihood_list

        return results_dict
    
    def plot_layer_ablation(
        self,
        layer_results: dict,
        save_path: Optional[str] = None,
        filename: str = "final"
    ):
        if save_path is None:
            fig_list = []
        
        metric_mapping = {
            "accuracy": "Accuracy",
            "log likelihood": "Log Likelihood"
        }

        metrics = []
        for metric in metric_mapping.keys():
            if metric in layer_results:
                metrics.append(metric)
        
        synapse_ablation_types = []
        if "excitation" in layer_results:
            synapse_ablation_types.append("excitation")
        if "inhibition" in layer_results:
            synapse_ablation_types.append("inhibition")
        if "all synapses" in layer_results:
            synapse_ablation_types.append("all synapses")
        
        for metric in metrics:
            results = {}
            for ablation_type in synapse_ablation_types:
                data_dict = {"depth": [], "decrease": []}
                type_dict: dict = layer_results[ablation_type]
                for layer_dict in type_dict.values():
                    data_dict["depth"].append(layer_dict["depth"])
                    data_dict["decrease"].append(layer_dict[metric])
                results[ablation_type] = data_dict
            
            fig, ax = plt.subplots(figsize=(10, 8))
            n_ablation_types = len(synapse_ablation_types)
            depths = results[synapse_ablation_types[0]]["depth"]
            width = 0.25
            for i, ablation_type in enumerate(synapse_ablation_types):
                x = np.array(depths) + width * (i - (n_ablation_types - 1) / 2)
                ax.bar(
                    x,
                    results[ablation_type]["decrease"],
                    width=width,
                    label=ablation_type
                )
            ax.set_xticks(depths)
            ax.set_xlabel("Depth")
            ax.set_ylabel(f"{metric_mapping[metric]} Decrease")
            ax.set_title(f"{metric_mapping[metric]} Decrease vs. Depth")
            ax.legend()
            # ax.grid(True, axis="y", alpha=0.3)
            if save_path is not None:
                plot_path = os.path.join(
                    save_path, f"{filename}_synapse_ablation_{metric}.png"
                )
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                fig_list.append(fig)
        
        if save_path is None:
            return fig_list

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        model = model.to(device)
        model.eval()
        inputs, labels = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        test_dataset = TensorDataset(inputs, labels)

        state_dict = deepcopy(model.state_dict())

        baselines = {}
        if self.accuracy:
            baselines["accuracy"] = evaluate_accuracy(
                classifier=model, test_ds=test_dataset, move_device=False
            )[-1]
        if self.categorical_ll:
            baselines["log likelihood"] = evaluate_categorical_loglikelihood(
                classifier=model, test_ds=test_dataset, move_device=False
            )[-1]

        if self.layer_ablation:
            layer_results = {}
            if self.ablate_all_synapses:
                layer_results["all synapses"] = {}
            if self.ablate_excitation:
                layer_results["excitation"] = {}
            if self.ablate_inhibition:
                layer_results["inhibition"] = {}
            if self.ablate_upstream:
                layer_results["upstream"] = {}

        if self.compartment_ablation:
            compartment_results = {}
            if self.ablate_all_synapses:
                compartment_results["all synapses"] = {}
            if self.ablate_excitation:
                compartment_results["excitation"] = {}
            if self.ablate_inhibition:
                compartment_results["inhibition"] = {}
            if self.ablate_upstream:
                compartment_results["upstream"] = {}

        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    self.ablate_all_synapses
                    and module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="all synapses",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="all synapses",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict
                        )

                if self.ablate_excitation and module.branch_excitation is not None:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="excitation",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="excitation",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict
                        )

                if self.ablate_inhibition and module.branch_inhibition is not None:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="inhibition",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="inhibition",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict
                        )
                if self.ablate_upstream and hasattr(module, "branches_to_output"):
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="upstream",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=name,
                            ablation_type="upstream",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict
                        )

        results = {}
        if self.layer_ablation:
            results["layer"] = layer_results
        if self.compartment_ablation:
            results["compartment"] = compartment_results

        if save_path is not None:
            save_dict(results, save_path, filename)
            if self.layer_ablation:
                self.plot_layer_ablation(
                    layer_results=layer_results,
                    save_path=save_path,
                    filename=filename
                )
        else:
            return results
