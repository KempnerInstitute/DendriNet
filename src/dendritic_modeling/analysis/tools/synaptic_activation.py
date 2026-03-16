"""
Synaptic activation analysis module.

This module contains classes for analyzing synaptic activation patterns during forward passes,
tracking excitatory and inhibitory synaptic activations with comprehensive statistical summaries.
"""

from typing import Optional

import torch

from dendritic_modeling.analysis.utils.runtime import iter_analysis_batches
from dendritic_modeling.config import SynapticActivationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict


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
        for _name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if _name not in self.data_dict.keys():
                    self.data_dict[_name] = {}
                    module._name = _name
                    handle = module.register_forward_hook(self.forward_hook)
                    handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks."""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        output: torch.Tensor,
    ):
        """
        Forward hook to collect synaptic activations.

        Args:
            module: The dendritic branch layer
            input: Input tuple (excitatory, inhibitory, optional)
            output: Module output
        """
        excitatory_input, inhibitory_input, _ = input
        excitatory_input = excitatory_input[..., None]
        if inhibitory_input is not None:
            inhibitory_input = inhibitory_input[..., None]

        key = module._name

        # Collect excitatory synaptic activations
        if module.branch_excitation is not None:
            pruned_exc_syn = module.branch_excitation.pruned_weight().t()
            for _ in range(excitatory_input.ndim - 2):
                pruned_exc_syn = pruned_exc_syn[None, ...]
            self.data_dict[key].setdefault("exc", []).append(
                (excitatory_input * pruned_exc_syn).detach().cpu()
            )

        # Collect inhibitory synaptic activations
        if module.branch_inhibition is not None and inhibitory_input is not None:
            pruned_inh_syn = module.branch_inhibition.pruned_weight().t()
            for _ in range(inhibitory_input.ndim - 2):
                pruned_inh_syn = pruned_inh_syn[None, ...]
            self.data_dict[key].setdefault("inh", []).append(
                (inhibitory_input * pruned_inh_syn).detach().cpu()
            )

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
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        self.data_dict = {}

        model = model.to(device)
        model.eval()

        handles = self.attach_forward_hooks(model)
        sample_cap = self.samples if self.downsample else None
        with torch.no_grad():
            for batch in iter_analysis_batches(test_dataset, runtime, sample_cap):
                _ = model(batch[0].to(device))

        for module_dict in self.data_dict.values():
            for key, value in list(module_dict.items()):
                if isinstance(value, list) and value:
                    module_dict[key] = torch.cat(value, dim=0)

        # Initialize summary structures
        if self.network_summary:
            exc_net = []
            inh_net = []

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

            # Process excitatory activations
            if "exc" in module_dict.keys():
                exc: torch.Tensor = module_dict["exc"]
                if self.logspace:
                    exc = torch.log(exc + 1e-9)

                if self.network_summary or self.layer_summary:
                    exc_flat = exc.flatten()
                    if self.logspace:
                        exc_flat = exc_flat[exc_flat > -20]
                    else:
                        exc_flat = exc_flat[exc_flat > 0]

                if self.network_summary:
                    exc_net.append(exc_flat)

                if self.layer_summary:
                    layer_summary[name]["exc"] = {}
                    layer_summary[name]["exc"]["mean"] = exc_flat.mean().item()
                    layer_summary[name]["exc"]["std"] = exc_flat.std().item()

                if self.branch_summary:
                    branch_summary[name]["exc"] = {}
                    branch_summary[name]["exc"]["mean"] = exc.mean(dim=0).tolist()
                    branch_summary[name]["exc"]["std"] = exc.std(dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["exc"] = exc.tolist()

            # Process inhibitory activations
            if "inh" in module_dict.keys():
                inh: torch.Tensor = module_dict["inh"]
                if self.logspace:
                    inh = torch.log(inh + 1e-9)

                if self.network_summary or self.layer_summary:
                    inh_flat = inh.flatten()
                    if self.logspace:
                        inh_flat = inh_flat[inh_flat > -20]
                    else:
                        inh_flat = inh_flat[inh_flat > 0]

                if self.network_summary:
                    inh_net.append(inh_flat)

                if self.layer_summary:
                    layer_summary[name]["inh"] = {}
                    layer_summary[name]["inh"]["mean"] = inh_flat.mean().item()
                    layer_summary[name]["inh"]["std"] = inh_flat.std().item()

                if self.branch_summary:
                    branch_summary[name]["inh"] = {}
                    branch_summary[name]["inh"]["mean"] = inh.mean(dim=0).tolist()
                    branch_summary[name]["inh"]["std"] = inh.std(dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["inh"] = inh.tolist()

        # Compile results
        results = {"logspace": self.logspace}

        if self.network_summary:
            network_summary = {}
            network_summary["exc"] = {}
            network_summary["inh"] = {}
            network_summary["exc"]["mean"] = torch.cat(exc_net).mean().item()
            network_summary["inh"]["mean"] = torch.cat(inh_net).mean().item()
            network_summary["exc"]["std"] = torch.cat(exc_net).std().item()
            network_summary["inh"]["std"] = torch.cat(inh_net).std().item()

            results["network_summary"] = network_summary

        if self.layer_summary:
            results["layer_summary"] = layer_summary

        if self.branch_summary:
            results["branch_summary"] = branch_summary

        if self.raw_summary:
            results["raw_summary"] = self.data_dict

        self.remove_forward_hooks(handles)

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results
