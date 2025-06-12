"""
Mutual information analysis module.

This module contains classes for computing mutual information between excitatory/inhibitory
inputs and outputs in dendritic networks.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.layers.dendritic import DendriticBranchLayer
from dendritic_modeling.analysis.information import (
    compute_mi_cc,
    compute_mi_cd,
    compute_pairwise_sr,
)
from dendritic_modeling.utils import save_dict


class EIInformationAnalyzer:
    """
    Analyzer for computing mutual information between excitatory/inhibitory inputs and outputs.
    
    Computes various information-theoretic measures including:
    - MI(E;I): Mutual information between excitatory and inhibitory inputs
    - MI(EI;Vout): Mutual information between combined inputs and branch outputs
    - SR(E,I;Vout): Synergy and redundancy between E and I with respect to Vout
    - MI(EI;label): Mutual information between inputs and class labels
    """
    
    def __init__(
        self,
        compute_mi_E_I: bool = False,
        compute_mi_EI_Vout: bool = False,
        compute_sr_EI_Vout: bool = False,
        compute_mi_EI_label: bool = False,
        n_neighbors: int = 10,
    ):
        """
        Initialize the mutual information analyzer.
        
        Args:
            compute_mi_E_I: Whether to compute MI(E;I) between excitatory and inhibitory inputs
            compute_mi_EI_Vout: Whether to compute MI(EI;Vout) between inputs and outputs
            compute_sr_EI_Vout: Whether to compute synergy/redundancy between E,I and Vout
            compute_mi_EI_label: Whether to compute MI(EI;label) between inputs and labels
            n_neighbors: Number of neighbors for k-NN mutual information estimation
        """
        self.mi_E_I = compute_mi_E_I
        self.mi_EI_Vout = compute_mi_EI_Vout
        self.sr_EI_Vout = compute_sr_EI_Vout
        self.mi_EI_label = compute_mi_EI_label
        self.n_neighbors = n_neighbors

    def attach_forward_hooks(self, model: BaseModel):
        """
        Attach forward hooks to collect excitatory/inhibitory activations.
        
        Args:
            model: The model to attach hooks to
            
        Returns:
            List of hook handles for later removal
        """
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
        Forward hook to collect excitatory/inhibitory activations.
        
        Args:
            module: The dendritic branch layer
            input: Input tuple (excitatory, inhibitory, optional)
            output: Module output
        """
        excitatory_input, inhibitory_input, _ = input
        key = module._name
        self.data_dict[key]["excitation"] = module.branch_excitation(excitatory_input)
        self.data_dict[key]["inhibition"] = module.branch_inhibition(inhibitory_input)
        if self.mi_EI_Vout:
            self.data_dict[key]["Vout"] = output

    def compute_mi_E_I(self, excitation: np.ndarray, inhibition: np.ndarray) -> float:
        """
        Compute mutual information between excitatory and inhibitory inputs.
        
        Args:
            excitation: Excitatory activations
            inhibition: Inhibitory activations
            
        Returns:
            Mutual information value
        """
        return compute_mi_cc(excitation, inhibition, self.n_neighbors)

    def compute_mi_EI_Vout(self, EI: np.ndarray, Vout: np.ndarray) -> float:
        """
        Compute mutual information between combined E/I inputs and output.
        
        Args:
            EI: Combined excitatory/inhibitory activations
            Vout: Branch output activations
            
        Returns:
            Mutual information value
        """
        return compute_mi_cc(EI, Vout, self.n_neighbors)

    def compute_sr_EI_Vout(
        self,
        excitation: np.ndarray,
        inhibition: np.ndarray,
        Vout: np.ndarray,
        mi_EI_Vout: Optional[float] = None,
    ) -> float:
        """
        Compute synergy and redundancy between E and I with respect to Vout.
        
        Args:
            excitation: Excitatory activations
            inhibition: Inhibitory activations
            Vout: Branch output activations
            mi_EI_Vout: Pre-computed MI(EI;Vout) if available
            
        Returns:
            Tuple of (synergy, redundancy, mi_EI_Vout, mi_E_Vout, mi_I_Vout)
        """
        return compute_pairwise_sr(
            excitation, inhibition, Vout, self.n_neighbors, mi_EI_Vout
        )

    def compute_mi_EI_label(self, EI: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute mutual information between combined E/I inputs and class labels.
        
        Args:
            EI: Combined excitatory/inhibitory activations
            labels: Class labels
            
        Returns:
            Mutual information value
        """
        return compute_mi_cd(EI, labels, self.n_neighbors)

    def analyze(
        self,
        model: BaseModel,
        dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Analyze mutual information in the model.
        
        Args:
            model: The model to analyze
            dataset: Dataset for analysis
            device: Device to run analysis on
            save_path: Path to save results (optional)
            filename: Filename for saved results
            
        Returns:
            Dictionary containing mutual information results if save_path is None, otherwise None
        """
        try:
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        self.data_dict = {}

        model = model.to(device)
        model.eval()

        inputs, labels = dataset[:]
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor

        handles = self.attach_forward_hooks(model)

        _ = model(inputs)

        # Collect activations from all branches
        excitation = []
        inhibition = []
        if self.mi_EI_Vout:
            Vout = []
        for branch_dict in self.data_dict.values():
            excitation.append(branch_dict["excitation"])
            inhibition.append(branch_dict["inhibition"])
            if self.mi_EI_Vout:
                Vout.append(branch_dict["Vout"])

        # Convert to numpy arrays
        excitation = torch.cat(excitation, dim=-1).detach().cpu().numpy()
        inhibition = torch.cat(inhibition, dim=-1).detach().cpu().numpy()

        # Prepare labels if needed
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

        # Flatten for mutual information computation
        excitation = excitation.flatten()[:, None]
        inhibition = inhibition.flatten()[:, None]

        # Combine E and I for joint computations
        if self.mi_EI_Vout or self.mi_EI_label:
            EI = np.concatenate([excitation, inhibition], axis=1)

        info_dict = {}

        # Compute requested mutual information measures
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