"""
Layer contribution analysis module.

This module contains classes for analyzing the contribution of individual layers to overall
model performance by testing each layer in isolation and measuring its individual impact.
"""

import logging
from copy import deepcopy
from typing import Callable, Optional

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.config import SingleLayerContributionAnalysisParams
from dendritic_modeling.models import BaseModel, Classifier
from dendritic_modeling.networks import (DendriticBranchLayer,
                                         ExcitationInhibitionNetwork)
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy, evaluate_auc, evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity, evaluate_mse)
from dendritic_modeling.utils import save_dict


class SingleLayerContributionAnalyzer:
    """
    Analyzes the contribution of individual layers by isolating them.
    This is the renamed version of "layer_ablation" - it removes all other layers
    to test what each layer can do on its own.
    """

    def __init__(self, params: SingleLayerContributionAnalysisParams):
        self.excitation_contribution = getattr(params, "excitation_contribution", False)
        self.inhibition_contribution = getattr(params, "inhibition_contribution", False)
        self.both_contribution = getattr(params, "both_contribution", False)
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", False)
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)
        self.logger = logging.getLogger("dendritic_modeling.analysis")

    def _evaluate_single_layer(
        self, model: BaseModel, test_dataset: TensorDataset, **kwargs
    ) -> dict:
        """Evaluate each dendritic layer in isolation"""
        results_dict = {"excitation": {}, "inhibition": {}, "all": {}}

        # Check if model is a Classifier to skip inappropriate metrics
        is_classifier = isinstance(model, Classifier)
        if is_classifier:
            self.logger.info(
                "Detected Classifier model - skipping MSE and cosine similarity metrics"
            )

        state_dict = deepcopy(model.state_dict())

        for layer_idx, module in enumerate(model.core_network.layers):
            if not isinstance(module, DendriticBranchLayer):
                continue

            module_name = f"layer_{layer_idx}"

            # Initialize results for this layer
            for contribution_type in ["excitation", "inhibition", "all"]:
                if (
                    contribution_type == "excitation"
                    and not self.excitation_contribution
                ):
                    continue
                if (
                    contribution_type == "inhibition"
                    and not self.inhibition_contribution
                ):
                    continue
                if contribution_type == "all" and not self.both_contribution:
                    continue

                results_dict[contribution_type][module_name] = {}

        for layer_idx, module in enumerate(model.core_network.layers):
            if not isinstance(module, DendriticBranchLayer):
                continue

            module_name = f"layer_{layer_idx}"

            for contribution_type in ["excitation", "inhibition", "all"]:
                if (
                    contribution_type == "excitation"
                    and not self.excitation_contribution
                ):
                    continue
                if (
                    contribution_type == "inhibition"
                    and not self.inhibition_contribution
                ):
                    continue
                if contribution_type == "all" and not self.both_contribution:
                    continue

                # Reset model state
                model.load_state_dict(state_dict)

                # Disable all layers except the current one
                for lidx, mod in enumerate(model.core_network.layers):
                    if lidx != layer_idx and isinstance(mod, DendriticBranchLayer):
                        mod.branch_excitation.pre_w.data.fill_(-1000)
                        mod.branch_inhibition.pre_w.data.fill_(-1000)

                if contribution_type == "excitation":
                    module.branch_inhibition.pre_w.data.fill_(-1000)
                if contribution_type == "inhibition":
                    module.branch_excitation.pre_w.data.fill_(-1000)

                def _evaluate_metric(metric_func: Callable, metric_key: str):
                    metric_score = metric_func(
                        model, test_ds=test_dataset, move_device=True
                    )[-1]
                    results_dict[contribution_type][module_name][
                        metric_key
                    ] = metric_score

                if self.accuracy:
                    _evaluate_metric(
                        metric_func=evaluate_accuracy, metric_key="accuracy"
                    )
                if self.auc:
                    _evaluate_metric(metric_func=evaluate_auc, metric_key="auc")
                if self.categorical_ll:
                    _evaluate_metric(
                        metric_func=evaluate_categorical_loglikelihood,
                        metric_key="log likelihood",
                    )
                # Skip MSE and cosine similarity for Classifier models
                if self.mse and not is_classifier:
                    _evaluate_metric(metric_func=evaluate_mse, metric_key="mse")
                elif self.mse and is_classifier:
                    self.logger.info(
                        f"Skipping MSE evaluation for Classifier model at {module_name}"
                    )

                if self.cosine_similarity and not is_classifier:
                    _evaluate_metric(
                        metric_func=evaluate_cosine_similarity,
                        metric_key="cosine similarity",
                    )
                elif self.cosine_similarity and is_classifier:
                    self.logger.info(
                        f"Skipping cosine similarity evaluation for Classifier model at {module_name}"
                    )

                model.load_state_dict(state_dict)

        return results_dict

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cuda",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """Analyze each layer's contribution in isolation"""
        self.logger.info("Starting single layer contribution analysis...")

        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork), (
                "Model must be an instance of ExcitationInhibitionNetwork"
                + "but got "
                + str(type(model.core_network))
            )
        except AssertionError as e:
            self.logger.error(e)
            return

        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()
        inputs, labels = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        test_dataset = TensorDataset(inputs, labels)

        # Use the new _evaluate_single_layer method
        results = self._evaluate_single_layer(model, test_dataset)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results
