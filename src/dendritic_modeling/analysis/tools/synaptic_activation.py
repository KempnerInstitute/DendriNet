"""
Synaptic activation analysis module.

This module contains classes for analyzing synaptic activation patterns during forward passes,
tracking excitatory and inhibitory synaptic activations with comprehensive statistical summaries.
"""

from typing import Optional

import torch

from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config import SynapticActivationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.training.strategies.local_learning_pathways import (
    iter_topk_path_modules,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import remove_hook_handles


class SynapticActivationAnalyzer:
    """
    Analyzer for tracking synaptic activation patterns during model inference.

    Uses forward hooks to collect excitatory and inhibitory synaptic activations
    and computes detailed statistical summaries at network, layer, and branch levels.
    """

    def __init__(self, params: SynapticActivationAnalysisParams):
        """
        Initialize the synaptic activation analyzer.

        Args:
            network_summary: Whether to compute network-wide activation statistics
            layer_summary: Whether to compute per-layer activation statistics
            branch_summary: Whether to compute per-branch activation statistics
            raw_summary: Whether to include raw activation data in results
            logspace: Whether to apply log transformation to activations
            downsample: Whether to downsample input data for efficiency
            samples: Number of samples to use if downsampling
        """
        self.network_summary = getattr(params, "network_summary", False)
        self.layer_summary = getattr(params, "layer_summary", False)
        self.branch_summary = getattr(params, "branch_summary", False)
        self.raw_summary = getattr(params, "raw_summary", False)
        self.logspace = getattr(params, "logspace", False)
        self.downsample = getattr(params, "downsample", False)
        self.samples = getattr(params, "samples", 5)

    def attach_forward_hooks(self, model: BaseModel):
        """
        Attach forward hooks to collect synaptic activations.

        Args:
            model: The model to attach hooks to

        Returns:
            List of hook handles for later removal
        """
        handles = []
        for layer_name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                self.data_dict.setdefault(layer_name, {})
                for path, synapse_module in iter_topk_path_modules(module):
                    synapse_module._analysis_layer_name = layer_name
                    synapse_module._analysis_synapse_type = path.pathway
                    handle = synapse_module.register_forward_hook(self.forward_hook)
                    handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks."""
        remove_hook_handles(handles)

    def forward_hook(
        self,
        module: TopKLinear,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """
        Forward hook to collect synaptic activations.

        Args:
            module: The synaptic TopK layer.
            input: Input tuple to the synaptic layer.
            output: Branch current output.
        """
        key = module._analysis_layer_name
        synapse_type = module._analysis_synapse_type
        self.data_dict[key].setdefault(synapse_type, []).append(output.detach().cpu())

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """
        Analyze synaptic activations in the model.

        Args:
            model: The model to analyze
            test_dataset: Test dataset for analysis
            device: Device to run analysis on
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing activation analysis if save_path is None, otherwise None
        """
        core = getattr(model, "core_network", model)
        if not any(
            isinstance(module, DendriticBranchLayer) for module in core.modules()
        ):
            return

        self.data_dict = {}

        sample_cap = self.samples if self.downsample else None
        handles = []
        with analysis_device_context(model, device) as analysis_device:
            handles = self.attach_forward_hooks(model)
            try:
                with torch.no_grad():
                    for batch in iter_analysis_batches(
                        test_dataset,
                        runtime,
                        sample_cap,
                        device=analysis_device,
                    ):
                        _ = model(batch[0].to(analysis_device))
            finally:
                self.remove_forward_hooks(handles)

        for module_dict in self.data_dict.values():
            for key, value in list(module_dict.items()):
                if isinstance(value, list) and value:
                    module_dict[key] = torch.cat(value, dim=0)

        # Initialize summary structures
        if self.network_summary:
            network_values = {}

        if self.layer_summary:
            layer_summary = {}

        if self.branch_summary:
            branch_summary = {}

        # Process collected activation data
        for name, module_dict in self.data_dict.items():
            module_dict: dict
            if self.layer_summary:
                layer_summary[name] = {}
            if self.branch_summary:
                branch_summary[name] = {}

            for synapse_type, activation in list(module_dict.items()):
                if not isinstance(activation, torch.Tensor):
                    continue
                if self.logspace:
                    activation = torch.log(activation + 1e-9)

                if self.network_summary or self.layer_summary:
                    flat = activation.flatten()
                    if self.logspace:
                        flat = flat[flat > -20]
                    else:
                        flat = flat[flat > 0]
                    if flat.numel() == 0:
                        flat = torch.zeros(1)

                if self.network_summary:
                    network_values.setdefault(synapse_type, []).append(flat)

                if self.layer_summary:
                    layer_summary[name][synapse_type] = {
                        "mean": flat.mean().item(),
                        "std": flat.std(unbiased=False).item(),
                    }

                if self.branch_summary:
                    branch_summary[name][synapse_type] = {
                        "mean": activation.mean(dim=0).tolist(),
                        "std": activation.std(dim=0, unbiased=False).tolist(),
                    }

                if self.raw_summary:
                    self.data_dict[name][synapse_type] = activation.tolist()

        # Compile results
        results = {"logspace": self.logspace}

        if self.network_summary:
            results["network_summary"] = {
                synapse_type: {
                    "mean": torch.cat(values).mean().item(),
                    "std": torch.cat(values).std(unbiased=False).item(),
                }
                for synapse_type, values in network_values.items()
                if values
            }

        if self.layer_summary:
            results["layer_summary"] = layer_summary

        if self.branch_summary:
            results["branch_summary"] = branch_summary

        if self.raw_summary:
            results["raw_summary"] = self.data_dict

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results
