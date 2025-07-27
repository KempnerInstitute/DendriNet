"""
Synaptic turnover analysis module.

This module contains classes for analyzing synaptic plasticity and turnover during training,
tracking the dynamics of active, stable, and transient synapses over time.
"""

import logging
from copy import deepcopy
from typing import Optional

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class SynapticTurnoverAnalyzer:
    """
    Analyzer for tracking synaptic turnover and plasticity during training.

    Collects snapshots of synaptic weight masks over time and computes turnover
    statistics including active, stable, and transient synapses for both
    excitatory and inhibitory connections.
    """

    def __init__(self, model: BaseModel, snapshot_interval: int = 1):
        """
        Initialize the synaptic turnover analyzer.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            snapshot_interval: Collect snapshots every N epochs (default=1 for every epoch)

        Note:
            This automatically takes an initial snapshot of the model's synaptic masks.
        """
        assert isinstance(model.core_network, ExcitationInhibitionNetwork), (
            "model.core_network must be an instance of ExcitationInhibitionNetwork. "
            + f"Got model.core_network type: {type(model.core_network)}"
        )

        self.data_dict_list = []
        self.snapshot_interval = snapshot_interval
        self.epoch_counter = 0
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
            self.epoch_counter += 1
            
            # Only collect snapshot at specified intervals
            if (self.epoch_counter - 1) % self.snapshot_interval != 0:
                return
            
            data_dict = {}
            # Use torch.no_grad() to avoid storing gradients
            with torch.no_grad():
                for _name, module in model.named_modules():
                    if isinstance(module, DendriticBranchLayer):
                        data_dict[_name] = {}
                        if module.branch_excitation is not None:
                            # Get mask, convert to bool for memory efficiency (1 byte vs 4 bytes)
                            # and move to CPU. No deep copy needed - just tensor operations
                            mask = module.branch_excitation.weight_mask()
                            data_dict[_name]["exc_mask"] = mask.bool().cpu()
                        if module.branch_inhibition is not None:
                            mask = module.branch_inhibition.weight_mask()
                            data_dict[_name]["inh_mask"] = mask.bool().cpu()
            
            self.data_dict_list.append(data_dict)
        else:
            # Analysis mode: compute turnover statistics from collected snapshots
            logger.info(f"Analyzing synaptic turnover from {len(self.data_dict_list)} snapshots")

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
                # Convert to bool if not already (for older snapshots)
                if prev_mask.dtype != torch.bool:
                    prev_mask = prev_mask.bool()
                if curr_mask.dtype != torch.bool:
                    curr_mask = curr_mask.bool()
                
                # Use bitwise operations for efficiency
                curr_active = curr_mask.sum().item()
                # Transient = newly activated (curr & ~prev)
                curr_transient = (curr_mask & ~prev_mask).sum().item()
                curr_stable = curr_active - curr_transient
                
                total_active += curr_active
                total_stable += curr_stable
                total_transient += curr_transient
                return total_active, total_stable, total_transient

            # Initialize turnover statistics dictionaries
            # Incremental: comparing each timepoint with previous (t vs t-1)
            incremental_turnover = {
                "exc": {"active": [], "stable": [], "transient": []},
                "inh": {"active": [], "stable": [], "transient": []},
                "both": {"active": [], "stable": [], "transient": []},
            }
            
            # Cumulative: comparing each timepoint with initial (t vs t=0)
            cumulative_turnover = {
                "exc": {"active": [], "stable": [], "transient": []},
                "inh": {"active": [], "stable": [], "transient": []},
                "both": {"active": [], "stable": [], "transient": []},
            }

            # Get initial snapshot for cumulative comparison
            initial_data_dict = self.data_dict_list[0] if len(self.data_dict_list) > 0 else None

            # Compare consecutive snapshots for incremental turnover
            for i in range(1, len(self.data_dict_list)):
                prev_data_dict: dict = self.data_dict_list[i - 1]
                curr_data_dict: dict = self.data_dict_list[i]

                # Initialize counters for incremental comparison (t vs t-1)
                inc_exc_active = inc_exc_stable = inc_exc_transient = 0
                inc_inh_active = inc_inh_stable = inc_inh_transient = 0
                
                # Initialize counters for cumulative comparison (t vs t=0)
                cum_exc_active = cum_exc_stable = cum_exc_transient = 0
                cum_inh_active = cum_inh_stable = cum_inh_transient = 0

                # Process each dendritic branch layer
                for name in curr_data_dict.keys():
                    prev_module_dict: dict = prev_data_dict[name]
                    curr_module_dict: dict = curr_data_dict[name]
                    initial_module_dict: dict = initial_data_dict[name] if initial_data_dict else prev_module_dict

                    # Process excitatory synapses
                    if "exc_mask" in curr_module_dict.keys():
                        curr_exc_mask = curr_module_dict["exc_mask"]
                        
                        # Incremental comparison (t vs t-1)
                        prev_exc_mask = prev_module_dict["exc_mask"]
                        (
                            inc_exc_active,
                            inc_exc_stable,
                            inc_exc_transient,
                        ) = turnover_helper(
                            prev_exc_mask,
                            curr_exc_mask,
                            inc_exc_active,
                            inc_exc_stable,
                            inc_exc_transient,
                        )
                        
                        # Cumulative comparison (t vs t=0)
                        initial_exc_mask = initial_module_dict["exc_mask"]
                        (
                            cum_exc_active,
                            cum_exc_stable,
                            cum_exc_transient,
                        ) = turnover_helper(
                            initial_exc_mask,
                            curr_exc_mask,
                            cum_exc_active,
                            cum_exc_stable,
                            cum_exc_transient,
                        )

                    # Process inhibitory synapses
                    if "inh_mask" in curr_module_dict.keys():
                        curr_inh_mask = curr_module_dict["inh_mask"]
                        
                        # Incremental comparison (t vs t-1)
                        prev_inh_mask = prev_module_dict["inh_mask"]
                        (
                            inc_inh_active,
                            inc_inh_stable,
                            inc_inh_transient,
                        ) = turnover_helper(
                            prev_inh_mask,
                            curr_inh_mask,
                            inc_inh_active,
                            inc_inh_stable,
                            inc_inh_transient,
                        )
                        
                        # Cumulative comparison (t vs t=0)
                        initial_inh_mask = initial_module_dict["inh_mask"]
                        (
                            cum_inh_active,
                            cum_inh_stable,
                            cum_inh_transient,
                        ) = turnover_helper(
                            initial_inh_mask,
                            curr_inh_mask,
                            cum_inh_active,
                            cum_inh_stable,
                            cum_inh_transient,
                        )

                # Store incremental turnover statistics
                incremental_turnover["exc"]["active"].append(inc_exc_active)
                incremental_turnover["exc"]["stable"].append(inc_exc_stable)
                incremental_turnover["exc"]["transient"].append(inc_exc_transient)

                incremental_turnover["inh"]["active"].append(inc_inh_active)
                incremental_turnover["inh"]["stable"].append(inc_inh_stable)
                incremental_turnover["inh"]["transient"].append(inc_inh_transient)

                incremental_turnover["both"]["active"].append(
                    inc_exc_active + inc_inh_active
                )
                incremental_turnover["both"]["stable"].append(
                    inc_exc_stable + inc_inh_stable
                )
                incremental_turnover["both"]["transient"].append(
                    inc_exc_transient + inc_inh_transient
                )
                
                # Store cumulative turnover statistics
                cumulative_turnover["exc"]["active"].append(cum_exc_active)
                cumulative_turnover["exc"]["stable"].append(cum_exc_stable)
                cumulative_turnover["exc"]["transient"].append(cum_exc_transient)

                cumulative_turnover["inh"]["active"].append(cum_inh_active)
                cumulative_turnover["inh"]["stable"].append(cum_inh_stable)
                cumulative_turnover["inh"]["transient"].append(cum_inh_transient)

                cumulative_turnover["both"]["active"].append(
                    cum_exc_active + cum_inh_active
                )
                cumulative_turnover["both"]["stable"].append(
                    cum_exc_stable + cum_inh_stable
                )
                cumulative_turnover["both"]["transient"].append(
                    cum_exc_transient + cum_inh_transient
                )

            # Return both types of turnover statistics
            turnover_dict = {
                "incremental": incremental_turnover,
                "cumulative": cumulative_turnover,
                "n_snapshots": len(self.data_dict_list)
            }

            if save_path is not None:
                save_dict(turnover_dict, save_path, filename)
            else:
                return turnover_dict
