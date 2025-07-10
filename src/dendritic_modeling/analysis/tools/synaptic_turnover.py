"""
Synaptic turnover analysis module.

This module contains classes for analyzing synaptic plasticity and turnover during training,
tracking the dynamics of active, stable, and transient synapses over time.
"""

from copy import deepcopy
from typing import Optional
import torch
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils import save_dict

class SynapticTurnoverAnalyzer:
    """
    Analyzer for tracking synaptic turnover and plasticity during training.

    Collects snapshots of synaptic weight masks over time and computes turnover
    statistics including active, stable, and transient synapses for both
    excitatory and inhibitory connections.
    """

    def __init__(self, model: BaseModel):
        """
        Initialize the synaptic turnover analyzer.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)

        Note:
            This automatically takes an initial snapshot of the model's synaptic masks.
        """
        assert isinstance(model.core_network, ExcitationInhibitionNetwork), (
            "model.core_network must be an instance of ExcitationInhibitionNetwork. "
            + f"Got model.core_network type: {type(model.core_network)}"
        )

        self.data_dict_list = []
        self.analyze(model=model, training=True)

    def analyze(
        self,
        model: BaseModel,
        training: bool = True,
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Analyze synaptic turnover by taking snapshots or computing statistics.

        Args:
            model: The model to analyze
            training: If True, takes a snapshot of current synaptic masks.
                     If False, computes turnover statistics from collected snapshots.
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing turnover statistics if training=False and save_path is None
        """
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        if training:
            # Training mode: collect a snapshot of current synaptic masks
            data_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, DendriticBranchLayer):
                    data_dict[name] = {}
                    if module.branch_excitation is not None:
                        data_dict[name]["exc_mask"] = (
                            deepcopy(module.branch_excitation.weight_mask())
                            .detach()
                            .cpu()
                        )
                    if module.branch_inhibition is not None:
                        data_dict[name]["inh_mask"] = (
                            deepcopy(module.branch_inhibition.weight_mask())
                            .detach()
                            .cpu()
                        )
            self.data_dict_list.append(data_dict)
        else:
            # Analysis mode: compute turnover statistics from collected snapshots

            def turnover_helper(
                prev_mask: torch.Tensor,
                curr_mask: torch.Tensor,
                total_active: int,
                total_stable: int,
                total_transient: int,
            ) -> tuple[int, int, int]:
                """
                Helper function to compute turnover statistics between two masks.

                Args:
                    prev_mask: Previous time step's weight mask
                    curr_mask: Current time step's weight mask
                    total_active: Running total of active synapses
                    total_stable: Running total of stable synapses
                    total_transient: Running total of transient synapses

                Returns:
                    Tuple of updated (total_active, total_stable, total_transient)
                """
                mask_diff = curr_mask - prev_mask
                curr_active = torch.sum(curr_mask).item()
                curr_transient = torch.sum(mask_diff > 1e-3).item()
                curr_stable = curr_active - curr_transient
                total_active += curr_active
                total_stable += curr_stable
                total_transient += curr_transient
                return total_active, total_stable, total_transient

            # Initialize turnover statistics dictionary
            turnover_dict = {
                "exc": {"active": [], "stable": [], "transient": []},
                "inh": {"active": [], "stable": [], "transient": []},
                "both": {"active": [], "stable": [], "transient": []},
            }

            # Compare consecutive snapshots to compute turnover
            for i in range(1, len(self.data_dict_list)):
                prev_data_dict: dict = self.data_dict_list[i - 1]
                curr_data_dict: dict = self.data_dict_list[i]

                # Initialize counters for this time step
                exc_total_active = 0
                exc_total_stable = 0
                exc_total_transient = 0

                inh_total_active = 0
                inh_total_stable = 0
                inh_total_transient = 0

                # Process each dendritic branch layer
                for name in curr_data_dict.keys():
                    prev_module_dict: dict = prev_data_dict[name]
                    curr_module_dict: dict = curr_data_dict[name]

                    # Process excitatory synapses
                    if "exc_mask" in curr_module_dict.keys():
                        prev_exc_mask = prev_module_dict["exc_mask"]
                        curr_exc_mask = curr_module_dict["exc_mask"]
                        (
                            exc_total_active,
                            exc_total_stable,
                            exc_total_transient,
                        ) = turnover_helper(
                            prev_exc_mask,
                            curr_exc_mask,
                            exc_total_active,
                            exc_total_stable,
                            exc_total_transient,
                        )

                    # Process inhibitory synapses
                    if "inh_mask" in curr_module_dict.keys():
                        prev_inh_mask = prev_module_dict["inh_mask"]
                        curr_inh_mask = curr_module_dict["inh_mask"]
                        (
                            inh_total_active,
                            inh_total_stable,
                            inh_total_transient,
                        ) = turnover_helper(
                            prev_inh_mask,
                            curr_inh_mask,
                            inh_total_active,
                            inh_total_stable,
                            inh_total_transient,
                        )

                # Store turnover statistics for this time step
                turnover_dict["exc"]["active"].append(exc_total_active)
                turnover_dict["exc"]["stable"].append(exc_total_stable)
                turnover_dict["exc"]["transient"].append(exc_total_transient)

                turnover_dict["inh"]["active"].append(inh_total_active)
                turnover_dict["inh"]["stable"].append(inh_total_stable)
                turnover_dict["inh"]["transient"].append(inh_total_transient)

                # Combined statistics (excitatory + inhibitory)
                turnover_dict["both"]["active"].append(
                    exc_total_active + inh_total_active
                )
                turnover_dict["both"]["stable"].append(
                    exc_total_stable + inh_total_stable
                )
                turnover_dict["both"]["transient"].append(
                    exc_total_transient + inh_total_transient
                )

            if save_path is not None:
                save_dict(turnover_dict, save_path, filename)
            else:
                return turnover_dict
