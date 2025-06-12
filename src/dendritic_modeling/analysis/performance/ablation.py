"""
Ablation analysis module.

This module contains classes for evaluating model performance by systematically
disabling or ablating different network components to understand their contributions.
"""

from copy import deepcopy
from typing import Optional
import torch
from torch.utils.data import TensorDataset
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.layers.dendritic import DendriticBranchLayer
from dendritic_modeling.utils import (
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    save_dict,
)


class AblationAnalyzer:
    """
    Analyzer for evaluating model performance through component ablation.
    
    Systematically disables different network components (synapses, excitation,
    inhibition, branches) to measure their individual contributions to performance.
    """
    
    def __init__(
        self,
        ablate_all_synapses: bool = True,
        ablate_excitation: bool = True,
        ablate_inhibition: bool = True,
        ablate_branches: bool = True,
        accuracy: bool = True,
        categorical_loglikelihood: bool = True,
    ):
        """
        Initialize the ablation analyzer.
        
        Args:
            ablate_all_synapses: Whether to test ablating all synapses simultaneously
            ablate_excitation: Whether to test ablating excitatory synapses
            ablate_inhibition: Whether to test ablating inhibitory synapses  
            ablate_branches: Whether to test ablating branch connections
            accuracy: Whether to compute accuracy after ablation
            categorical_loglikelihood: Whether to compute categorical log-likelihood after ablation
        """
        self.ablate_all_synapses = ablate_all_synapses
        self.ablate_excitation = ablate_excitation
        self.ablate_inhibition = ablate_inhibition
        self.ablate_branches = ablate_branches
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
        """
        Analyze model performance through systematic component ablation.
        
        Args:
            model: The model to analyze
            test_dataset: Test dataset for evaluation
            device: Device to run analysis on
            save_path: Path to save results (optional)
            filename: Filename for saved results
            
        Returns:
            Dictionary containing ablation results if save_path is None, otherwise None
        """
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        model = model.to(device)
        inputs, labels = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)
        test_dataset = TensorDataset(inputs, labels)

        # Save original model state for restoration after each ablation
        state_dict = deepcopy(model.state_dict())

        results = {}
        if self.ablate_all_synapses:
            results["all synapses"] = {}
        if self.ablate_excitation:
            results["excitation"] = {}
        if self.ablate_inhibition:
            results["inhibition"] = {}
        if self.ablate_branches:
            results["branches"] = {}

        # Iterate through all dendritic branch layers
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                
                # Test ablating all synapses (both excitatory and inhibitory)
                if (
                    self.ablate_all_synapses
                    and module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    results["all synapses"][name] = {}
                    results["all synapses"][name]["layer_idx"] = module.layer_idx

                    # Ablate by setting weights to very negative values
                    module.branch_excitation.pre_w.data.fill_(-100)
                    module.branch_inhibition.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["all synapses"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["all synapses"][name]["log likelihood"] = loglikelihood

                    # Restore original state
                    model.load_state_dict(state_dict)

                # Test ablating only excitatory synapses
                if self.ablate_excitation and module.branch_excitation is not None:
                    results["excitation"][name] = {}
                    results["excitation"][name]["layer_idx"] = module.layer_idx

                    module.branch_excitation.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["excitation"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["excitation"][name]["log likelihood"] = loglikelihood

                    # Restore original state
                    model.load_state_dict(state_dict)

                # Test ablating only inhibitory synapses
                if self.ablate_inhibition and module.branch_inhibition is not None:
                    results["inhibition"][name] = {}
                    results["inhibition"][name]["layer_idx"] = module.layer_idx

                    module.branch_inhibition.pre_w.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["inhibition"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["inhibition"][name]["log likelihood"] = loglikelihood

                    # Restore original state
                    model.load_state_dict(state_dict)

                # Test ablating branch-to-output connections
                if self.ablate_branches and hasattr(module, "branches_to_output"):
                    results["branches"][name] = {}
                    results["branches"][name]["layer_idx"] = module.layer_idx

                    module.branches_to_output.log_weight.data.fill_(-100)

                    if self.accuracy:
                        accuracy = evaluate_accuracy(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["branches"][name]["accuracy"] = accuracy
                    if self.categorical_ll:
                        loglikelihood = evaluate_categorical_loglikelihood(
                            classifier=model, test_ds=test_dataset, move_device=False
                        )[-1]
                        results["branches"][name]["log likelihood"] = loglikelihood

                    # Restore original state
                    model.load_state_dict(state_dict)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results 