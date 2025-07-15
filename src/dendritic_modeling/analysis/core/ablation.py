"""
Ablation analysis module.

This module contains classes for evaluating model performance by systematically
disabling or ablating different network components to understand their contributions.
"""

from copy import deepcopy
from typing import Callable, Optional

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.config.analysis import AblationAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.architectures import DendriticBranchLayer
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict


class AblationAnalyzer:
    def __init__(self, params: AblationAnalysisParams):
        self.layer_ablation = getattr(params, "layer_ablation", False)
        self.compartment_ablation = getattr(params, "compartment_ablation", False)
        self.ablate_all_synapses = getattr(params, "ablate_all_synapses", False)
        self.ablate_excitation = getattr(params, "ablate_excitation", False)
        self.ablate_inhibition = getattr(params, "ablate_inhibition", False)
        self.ablate_upstream = getattr(params, "ablate_upstream", False)
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", False)
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)

    def ablate_layer_and_evaluate(
        self,
        test_dataset: torch.utils.data.Dataset,
        model: BaseModel,
        module: DendriticBranchLayer,
        module_name: str,
        ablation_type: str,
        baseline_dict: dict,
        results_dict: dict,
        state_dict: dict,
    ):
        results_dict[ablation_type][module_name] = {}
        results_dict[ablation_type][module_name]["depth"] = module.layer_idx

        if ablation_type in ["all synapses", "excitation"]:
            module.branch_excitation.pre_w.data.fill_(-1000)
        if ablation_type in ["all synapses", "inhibition"]:
            module.branch_inhibition.pre_w.data.fill_(-1000)
        if ablation_type == "upstream":
            module.branches_to_output.log_weight.data.fill_(-1000)

        def _evaluate_metric(metric_func: Callable, metric_key: str):
            metric_score = metric_func(model, test_ds=test_dataset, move_device=False)[
                -1
            ]
            results_dict[ablation_type][module_name][f"{metric_key} drop"] = (
                baseline_dict[metric_key] - metric_score
            )

        if self.accuracy:
            _evaluate_metric(metric_func=evaluate_accuracy, metric_key="accuracy")
        if self.auc:
            _evaluate_metric(metric_func=evaluate_auc, metric_key="auc")
        if self.categorical_ll:
            _evaluate_metric(
                metric_func=evaluate_categorical_loglikelihood,
                metric_key="log likelihood",
            )
        if self.mse:
            _evaluate_metric(metric_func=evaluate_mse, metric_key="mse")
        if self.cosine_similarity:
            _evaluate_metric(
                metric_func=evaluate_cosine_similarity, metric_key="cosine similarity"
            )

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
        state_dict: dict,
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
            raise ValueError(
                f"Number of compartments cannot be determined for {module_name}"
            )

        if self.accuracy:
            accuracy_list = []
        if self.auc:
            auc_list = []
        if self.categorical_ll:
            loglikelihood_list = []
        if self.mse:
            mse_list = []
        if self.cosine_similarity:
            cosine_similarity_list = []

        def _evaluate_metric(metric_func: Callable, metric_key: str, metric_list: list):
            metric_score = metric_func(model, test_ds=test_dataset, move_device=False)[
                -1
            ]
            metric_list.append(baseline_dict[metric_key] - metric_score)

        for branch_ix in range(n_branches):
            if ablation_type in ["all synapses", "excitation"]:
                module.branch_excitation.pre_w.data[branch_ix, :] = -1000
            if ablation_type in ["all synapses", "inhibition"]:
                module.branch_inhibition.pre_w.data[branch_ix, :] = -1000
            if ablation_type == "upstream":
                module.branches_to_output.log_weight.data[branch_ix, :] = -1000

            if self.accuracy:
                _evaluate_metric(
                    metric_func=evaluate_accuracy,
                    metric_key="accuracy",
                    metric_list=accuracy_list,
                )
            if self.auc:
                _evaluate_metric(
                    metric_func=evaluate_auc, metric_key="auc", metric_list=auc_list
                )
            if self.categorical_ll:
                _evaluate_metric(
                    metric_func=evaluate_categorical_loglikelihood,
                    metric_key="log likelihood",
                    metric_list=loglikelihood_list,
                )
            if self.mse:
                _evaluate_metric(
                    metric_func=evaluate_mse, metric_key="mse", metric_list=mse_list
                )
            if self.cosine_similarity:
                _evaluate_metric(
                    metric_func=evaluate_cosine_similarity,
                    metric_key="cosine similarity",
                    metric_list=cosine_similarity_list,
                )

            model.load_state_dict(state_dict)

        if self.accuracy:
            results_dict[ablation_type][module_name]["accuracy drop"] = accuracy_list
        if self.auc:
            results_dict[ablation_type][module_name]["auc drop"] = auc_list
        if self.categorical_ll:
            results_dict[ablation_type][module_name][
                "log likelihood drop"
            ] = loglikelihood_list
        if self.mse:
            results_dict[ablation_type][module_name]["mse drop"] = mse_list
        if self.cosine_similarity:
            results_dict[ablation_type][module_name][
                "cosine similarity drop"
            ] = cosine_similarity_list

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
        if self.auc:
            baselines["auc"] = evaluate_auc(
                classifier=model, test_ds=test_dataset, move_device=False
            )[-1]
        if self.categorical_ll:
            baselines["log likelihood"] = evaluate_categorical_loglikelihood(
                classifier=model, test_ds=test_dataset, move_device=False
            )[-1]
        if self.mse:
            # Check if model is a regressor before using MSE
            from dendritic_modeling.models import Regressor

            if isinstance(model, Regressor):
                baselines["mse"] = evaluate_mse(
                    regressor=model, test_ds=test_dataset, move_device=False
                )[-1]
            else:
                # Skip MSE for non-regressor models
                print(
                    f"Warning: MSE evaluation skipped for {type(model).__name__} model"
                )
                self.mse = False
        if self.cosine_similarity:
            # Check if model is a regressor before using cosine similarity
            from dendritic_modeling.models import Regressor

            if isinstance(model, Regressor):
                baselines["cosine similarity"] = evaluate_cosine_similarity(
                    regressor=model, test_ds=test_dataset, move_device=False
                )[-1]
            else:
                # Skip cosine similarity for non-regressor models
                print(
                    f"Warning: Cosine similarity evaluation skipped for {type(model).__name__} model"
                )
                self.cosine_similarity = False

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

        for _name, module in model.named_modules():
            if (
                isinstance(module, DendriticBranchLayer)
                and module.branch_excitation is not None
                and module.branch_inhibition is not None
            ):
                if self.ablate_all_synapses:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="all synapses",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="all synapses",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )

                if self.ablate_excitation:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="excitation",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="excitation",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )

                if self.ablate_inhibition:
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="inhibition",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="inhibition",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )
                if self.ablate_upstream and hasattr(module, "branches_to_output"):
                    if self.layer_ablation:
                        layer_results = self.ablate_layer_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="upstream",
                            baseline_dict=baselines,
                            results_dict=layer_results,
                            state_dict=state_dict,
                        )
                    if self.compartment_ablation:
                        compartment_results = self.ablate_compartment_and_evaluate(
                            test_dataset=test_dataset,
                            model=model,
                            module=module,
                            module_name=_name,
                            ablation_type="upstream",
                            baseline_dict=baselines,
                            results_dict=compartment_results,
                            state_dict=state_dict,
                        )

        results = {}
        if self.layer_ablation:
            results["layer"] = layer_results
        if self.compartment_ablation:
            results["compartment"] = compartment_results

        if save_path is not None:
            save_dict(results, save_path, filename)
            if self.layer_ablation:
                import logging
                logging.info(f"Generating ablation plots. Save path: {save_path}, filename: {filename}")
                logging.info(f"Layer results keys: {list(layer_results.keys())}")
                # Import plotting function
                from dendritic_modeling.plotting.ablation_plots import (
                    plot_layer_ablation,
                )

                plot_layer_ablation(
                    layer_results=layer_results, save_path=save_path, filename=filename
                )
                logging.info("Ablation plot generation completed")
        else:
            return results
