"""
Layer contribution analysis module.

This module contains classes for analyzing the contribution of individual layers to overall
model performance by testing each layer in isolation and measuring its individual impact.
"""

import logging
from copy import deepcopy
from typing import Optional, Callable, List
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import TensorDataset

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    ExcitationInhibitionNetwork,
    DendriticBranchLayer
)
from dendritic_modeling.config import SingleLayerContributionAnalysisParams
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_mse,
    evaluate_cosine_similarity,
)
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
        self.logger = logging.getLogger(__name__)

    def evaluate_layer_contribution(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        module: DendriticBranchLayer,
        module_name: str,
        contribution_type: str,
        results_dict: dict,
        state_dict: dict
    ):
        results_dict[contribution_type][module_name] = {}
        results_dict[contribution_type][module_name]["depth"] = module.layer_idx
    
        for mod, name in model.named_modules():
            if isinstance(mod, DendriticBranchLayer):
                if (
                    name != module_name
                    and mod.branch_excitation is not None
                    and mod.branch_inhibition is not None
                ):
                    mod.branch_excitation.pre_w.data.fill_(-1000)
                    mod.branch_inhibition.pre_w.data.fill_(-1000)
        
        if contribution_type == "excitation":
            module.branch_inhibition.pre_w.data.fill_(-1000)
        if contribution_type == "inhibition":
            module.branch_excitation.pre_w.data.fill_(-1000)
        
        def _evaluate_metric(metric_func: Callable, metric_key: str):
            metric_score = metric_func(
                model, test_ds=test_dataset, move_device=False
            )[-1]
            results_dict[contribution_type][module_name][metric_key] = metric_score

        if self.accuracy:
            _evaluate_metric(
                metric_func=evaluate_accuracy,
                metric_key="accuracy"
            )
        if self.auc:
            _evaluate_metric(
                metric_func=evaluate_auc,
                metric_key="auc"
            )
        if self.categorical_ll:
            _evaluate_metric(
                metric_func=evaluate_categorical_loglikelihood,
                metric_key="log likelihood"
            )
        if self.mse:
            _evaluate_metric(
                metric_func=evaluate_mse,
                metric_key="mse"
            )
        if self.cosine_similarity:
            _evaluate_metric(
                metric_func=evaluate_cosine_similarity,
                metric_key="cosine similarity"
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
            assert isinstance(model.core_network, ExcitationInhibitionNetwork), \
                "Model must be an instance of ExcitationInhibitionNetwork" + \
                "but got " + str(type(model.core_network))
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

        state_dict = deepcopy(model.state_dict())
        
        results = {}
        if self.excitation_contribution:
            results["excitation"] = {}
        if self.inhibition_contribution:
            results["inhibition"] = {}
        if self.both_contribution:
            results["both"] = {}
        
        for name, module in model.named_modules():
            if (
                isinstance(module, DendriticBranchLayer)
                and module.branch_excitation is not None
                and module.branch_inhibition is not None
            ):
                if self.both_contribution:
                    results = self.evaluate_layer_contribution(
                        model=model,
                        test_dataset=test_dataset,
                        module=module,
                        module_name=name,
                        contribution_type="both",
                        results_dict=results,
                        state_dict=state_dict
                    )
                
                if self.excitation_contribution:
                    results = self.evaluate_layer_contribution(
                        model=model,
                        test_dataset=test_dataset,
                        module=module,
                        module_name=name,
                        contribution_type="excitation",
                        results_dict=results,
                        state_dict=state_dict
                    )

                if self.inhibition_contribution:
                    results = self.evaluate_layer_contribution(
                        model=model,
                        test_dataset=test_dataset,
                        module=module,
                        module_name=name,
                        contribution_type="inhibition",
                        results_dict=results,
                        state_dict=state_dict
                    )
        
        if save_path is not None:
            save_dict(results, save_path, filename)
            plot_layer_contribution(
                results=results,
                save_path=save_path,
                filename=filename
            )
        else:
            return results



def plot_layer_contribution(
    results: dict,
    save_path: Optional[str] = None,
    filename: str = "final"
) -> Optional[List[plt.Figure]]:
    """
    Plot layer ablation results showing accuracy/log-likelihood decrease vs depth.
    
    Args:
        results: Dictionary containing ablation results by type and layer
        save_path: Path to save plots. If None, returns list of figures
        filename: Base filename for saved plots
        
    Returns:
        List of matplotlib figures if save_path is None
    """
    logger = logging.getLogger(__name__)
    
    if not results:
        logger.warning("No results to plot")
        return None if save_path else []
    
    figures = []
    
    # Plot each contribution type
    for contrib_type, contrib_data in results.items():
        if not contrib_data:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Layer Contribution Analysis - {contrib_type.title()}', fontsize=16)
        
        # Extract layer names and depths
        layer_names = list(contrib_data.keys())
        layer_depths = [int(name.split('.')[-1]) if '.' in name else 0 for name in layer_names]
        
        # Sort by depth
        sorted_indices = np.argsort(layer_depths)
        sorted_names = [layer_names[i] for i in sorted_indices]
        sorted_depths = [layer_depths[i] for i in sorted_indices]
        
        # Extract metrics for each layer
        metrics = ['accuracy', 'log_likelihood', 'auc', 'mse']
        metric_titles = ['Accuracy Drop', 'Log-Likelihood Drop', 'AUC Drop', 'MSE Drop']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            
            values = []
            for name in sorted_names:
                if metric in contrib_data[name]:
                    values.append(contrib_data[name][metric])
                else:
                    values.append(0.0)
            
            if any(v != 0 for v in values):
                ax.plot(sorted_depths, values, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel('Layer Depth')
                ax.set_ylabel(title)
                ax.set_title(f'{title} vs Layer Depth')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on points
                for x, y in zip(sorted_depths, values):
                    ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                              xytext=(0,10), ha='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No {metric} data available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{title} vs Layer Depth')
        
        plt.tight_layout()
        
        if save_path:
            plot_filename = f"{filename}_{contrib_type}_layer_contribution.png"
            plot_path = os.path.join(save_path, plot_filename)
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved layer contribution plot: {plot_path}")
            plt.close(fig)
        else:
            figures.append(fig)
    
    return None if save_path else figures
