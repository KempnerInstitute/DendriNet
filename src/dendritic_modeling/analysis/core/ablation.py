"""
Ablation analysis module.

This module contains classes for evaluating model performance by systematically
disabling or ablating different network components to understand their contributions.
"""
import os
from copy import deepcopy
from typing import Optional

import torch
from torch.utils.data import TensorDataset
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.architectures import DendriticBranchLayer
from dendritic_modeling.config.analysis import AblationAnalysisParams
from dendritic_modeling.utils import (
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    save_dict
)


class AblationAnalyzer:
    def __init__(self, params: AblationAnalysisParams):
        self.layer_ablation = getattr(params, "layer_ablation", True)
        self.compartment_ablation = getattr(params, "compartment_ablation", False)
        self.ablate_all_synapses = getattr(params, "ablate_all_synapses", True)
        self.ablate_excitation = getattr(params, "ablate_excitation", True)
        self.ablate_inhibition = getattr(params, "ablate_inhibition", True)
        self.ablate_upstream = getattr(params, "ablate_upstream", True)
        self.accuracy = getattr(params, "accuracy", True)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", True)
    
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
            results_dict[ablation_type][module_name]["accuracy drop"] = \
                baseline_dict["accuracy"] - accuracy
        if self.categorical_ll:
            loglikelihood = evaluate_categorical_loglikelihood(
                classifier=model, test_ds=test_dataset, move_device=False
            )[-1]
            results_dict[ablation_type][module_name]["log likelihood drop"] = \
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
            results_dict[ablation_type][module_name]["accuracy drop"] = accuracy_list
        if self.categorical_ll:
            results_dict[ablation_type][module_name]["log likelihood drop"] = loglikelihood_list

        return results_dict
    


    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
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
                # Import plotting function
                from dendritic_modeling.plotting.ablation_plots import plot_layer_ablation
                plot_layer_ablation(
                    layer_results=layer_results,
                    save_path=save_path,
                    filename=filename
                )
        else:
            return results
