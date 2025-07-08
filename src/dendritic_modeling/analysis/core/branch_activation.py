"""
Branch activation analysis module.

This module contains classes for analyzing both synaptic and branch-level activation patterns
during forward passes, providing detailed statistical analysis at multiple levels.
"""

from typing import List, Optional, Tuple
import torch
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.config import BranchActivationAnalysisParams
from dendritic_modeling.utils import save_dict


class BranchActivationAnalyzer:
    """
    Analyzer for tracking both synaptic and branch activation patterns during model inference.
    
    Uses forward hooks to collect excitatory synaptic activations, inhibitory synaptic activations,
    and branch output activations, computing detailed statistical summaries at multiple levels.
    """
    
    def __init__(self, params: BranchActivationAnalysisParams):
        """
        Initialize the branch activation analyzer.
        
        Args:
            synapse_activations: Whether to collect excitatory and inhibitory synapse activations
            branch_activations: Whether to collect branch output activations (Vout)
            network_summary: Whether to compute network-wide activation statistics
            layer_summary: Whether to compute per-layer activation statistics
            branch_summary: Whether to compute per-branch activation statistics
            raw_summary: Whether to include raw activation data in results
            logspace: Whether to apply log transformation to activations
            downsample: Whether to downsample input data for efficiency
            samples: Number of samples to use if downsampling
        """
        self.synapse_activations = getattr(params, "synapse_activations", True)
        self.branch_activations = getattr(params, "branch_activations", True)
        self.network_summary = getattr(params, "network_summary", True)
        self.layer_summary = getattr(params, "layer_summary", True)
        self.branch_summary = getattr(params, "branch_summary", False)
        self.raw_summary = getattr(params, "raw_summary", False)
        self.logspace = getattr(params, "logspace", False)
        self.downsample = getattr(params, "downsample", True)
        self.samples = getattr(params, "samples", 5)

    def attach_forward_hooks(self, model: BaseModel):
        """
        Attach forward hooks to collect branch activations.
        
        Args:
            model: The model to attach hooks to
            
        Returns:
            List of hook handles for later removal
        """
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if name not in self.data_dict.keys():
                    self.data_dict[name] = {}
                    module._name = name
                    handle = module.register_forward_hook(self.forward_hook)
                    handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: List[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks."""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        output: torch.Tensor,
    ):
        """
        Forward hook to collect branch activations.
        
        Args:
            module: The dendritic branch layer
            input: Input tuple (excitatory, inhibitory, optional)
            output: Module output
        """
        excitatory_input, inhibitory_input, _ = input
        key = module._name

        # Collect excitatory synapse activations
        if self.synapse_activations and module.branch_excitation is not None:
            self.data_dict[key]["exc"] = module.branch_excitation(excitatory_input)

        # Collect inhibitory synapse activations
        if self.synapse_activations and module.branch_inhibition is not None:
            self.data_dict[key]["inh"] = module.branch_inhibition(inhibitory_input)

        # Collect branch output activations
        if self.branch_activations:
            self.data_dict[key]["Vout"] = output

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Analyze branch activations in the model.
        
        Args:
            model: The model to analyze
            test_dataset: Test dataset for analysis
            device: Device to run analysis on
            save_path: Path to save results (optional)
            filename: Filename for saved results
            
        Returns:
            Dictionary containing branch activation analysis if save_path is None, otherwise None
        """
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        self.data_dict = {}

        model = model.to(device)
        model.eval()

        inputs, _ = test_dataset[:]
        inputs: torch.Tensor = inputs.to(device)

        # Optionally downsample for efficiency
        if self.downsample:
            ixs = torch.randperm(inputs.shape[0])[: self.samples]
            inputs = inputs[ixs]

        handles = self.attach_forward_hooks(model)

        _ = model(inputs)

        # Initialize summary structures
        if self.network_summary:
            if self.synapse_activations:
                exc_net = []
                inh_net = []
            if self.branch_activations:
                Vout_net = []

        if self.layer_summary:
            layer_summary = {}

        if self.branch_summary:
            branch_summary = {}

        # Process collected activation data with detailed debugging
        for name, module_dict in self.data_dict.items():
            module_dict: dict
            if self.layer_summary:
                layer_summary[name] = {}

            if self.branch_summary:
                branch_summary[name] = {}

            # Process excitatory synapse activations
            if "exc" in module_dict.keys():
                exc: torch.Tensor = module_dict["exc"]

                if self.logspace:
                    exc = torch.log(exc)

                if self.network_summary:
                    exc_net.append(exc.flatten())

                if self.layer_summary:
                    layer_summary[name]["exc"] = {}
                    layer_summary[name]["exc"]["mean"] = torch.mean(exc).item()
                    layer_summary[name]["exc"]["std"] = torch.std(exc).item()

                if self.branch_summary:
                    branch_summary[name]["exc"] = {}
                    branch_summary[name]["exc"]["mean"] = torch.mean(
                        exc, dim=0
                    ).tolist()
                    branch_summary[name]["exc"]["std"] = torch.std(exc, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["exc"] = exc.tolist()

            # Process inhibitory synapse activations
            if "inh" in module_dict.keys():
                inh: torch.Tensor = module_dict["inh"]

                if self.logspace:
                    inh = torch.log(inh)

                if self.network_summary:
                    inh_net.append(inh.flatten())

                if self.layer_summary:
                    layer_summary[name]["inh"] = {}
                    layer_summary[name]["inh"]["mean"] = torch.mean(inh).item()
                    layer_summary[name]["inh"]["std"] = torch.std(inh).item()

                if self.branch_summary:
                    branch_summary[name]["inh"] = {}
                    branch_summary[name]["inh"]["mean"] = torch.mean(
                        inh, dim=0
                    ).tolist()
                    branch_summary[name]["inh"]["std"] = torch.std(inh, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["inh"] = inh.tolist()

            # Process branch output activations
            if "Vout" in module_dict.keys():
                Vout: torch.Tensor = module_dict["Vout"]

                if self.logspace:
                    Vout = torch.log(Vout)

                if self.network_summary:
                    Vout_net.append(Vout.flatten())

                if self.layer_summary:
                    layer_summary[name]["Vout"] = {}
                    layer_summary[name]["Vout"]["mean"] = torch.mean(Vout).item()
                    layer_summary[name]["Vout"]["std"] = torch.std(Vout).item()

                if self.branch_summary:
                    branch_summary[name]["Vout"] = {}
                    branch_summary[name]["Vout"]["mean"] = torch.mean(
                        Vout, dim=0
                    ).tolist()
                    branch_summary[name]["Vout"]["std"] = torch.std(
                        Vout, dim=0
                    ).tolist()

                if self.raw_summary:
                    self.data_dict[name]["Vout"] = Vout.tolist()

        # Compile results
        results = {"logspace": self.logspace}

        if self.network_summary:
            network_summary = {}

            if self.synapse_activations:
                exc_net = torch.cat(exc_net, dim=0)
                network_summary["exc"] = {}
                network_summary["exc"]["mean"] = torch.mean(exc_net).item()
                network_summary["exc"]["std"] = torch.std(exc_net).item()

                inh_net = torch.cat(inh_net, dim=0)
                network_summary["inh"] = {}
                network_summary["inh"]["mean"] = torch.mean(inh_net).item()
                network_summary["inh"]["std"] = torch.std(inh_net).item()

            if self.branch_activations:
                Vout_net = torch.cat(Vout_net, dim=0)
                network_summary["Vout"] = {}
                network_summary["Vout"]["mean"] = torch.mean(Vout_net).item()
                network_summary["Vout"]["std"] = torch.std(Vout_net).item()

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