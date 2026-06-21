"""
Synaptic turnover analysis module.

This module contains classes for analyzing synaptic plasticity and turnover during training,
tracking the dynamics of active, stable, and transient synapses over time.
"""

import logging
from typing import Optional

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_synapse_turnover,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


def _update_turnover_counts(
    prev_mask: torch.Tensor,
    curr_mask: torch.Tensor,
    total_active: int,
    total_stable: int,
    total_transient: int,
) -> tuple[int, int, int]:
    """Update active/stable/transient counts for two consecutive masks."""
    if prev_mask.dtype != torch.bool:
        prev_mask = prev_mask.bool()
    if curr_mask.dtype != torch.bool:
        curr_mask = curr_mask.bool()

    curr_active = curr_mask.sum().item()
    curr_transient = (curr_mask & ~prev_mask).sum().item()
    curr_stable = curr_active - curr_transient

    total_active += curr_active
    total_stable += curr_stable
    total_transient += curr_transient
    return total_active, total_stable, total_transient


class SynapticTurnoverAnalyzer:
    """
    Analyzer for tracking synaptic turnover and plasticity during training.

    Collects snapshots of synaptic weight masks over time and computes turnover
    statistics including active, stable, and transient synapses for both
    excitatory and inhibitory connections.
    """

    def __init__(self, model: BaseModel, params):
        """
        Initialize the synaptic turnover analyzer.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            params: Parameters object containing snapshot_interval, synaptic_analysis, save_synaptic_report

        This automatically takes an initial snapshot of the model's synaptic masks.
        """
        self.data_dict_list = []
        self.epoch_counter = 0
        self.snapshot_interval = max(1, int(getattr(params, "snapshot_interval", 1)))

        # Store additional params for potential future use
        self.synaptic_analysis = getattr(params, "synaptic_analysis", True)
        self.save_synaptic_report = getattr(params, "save_synaptic_report", True)

        # Default to unsupported, then enable if model has EI core.
        self.enabled = isinstance(
            getattr(model, "core_network", None), ExcitationInhibitionNetwork
        )
        self.synapse_config = {"ee": [], "ei": [], "ie": [], "ii": []}
        if not self.enabled:
            return

        # Store synapse configuration for layer-wise analysis
        self.synapse_config = {
            "ee": getattr(model.core_network, "ee_synapses_per_branch_per_layer", []),
            "ei": getattr(model.core_network, "ei_synapses_per_branch_per_layer", []),
            "ie": getattr(model.core_network, "ie_synapses_per_branch_per_layer", []),
            "ii": getattr(model.core_network, "ii_synapses_per_branch_per_layer", []),
        }

        self.analyze(model=model, training=True)

    def _compute_layer_synapse_averages(
        self, total_exc_active: int, total_inh_active: int
    ) -> dict:
        """
        Compute average synapse counts per layer for each synapse type.

        Args:
            total_exc_active: Total active excitatory synapses
            total_inh_active: Total active inhibitory synapses

        Returns:
            Dictionary with average synapse counts for ee, ei, ie, ii
        """
        # Get configuration
        ee_config = self.synapse_config["ee"]
        ei_config = self.synapse_config["ei"]
        ie_config = self.synapse_config["ie"]
        ii_config = self.synapse_config["ii"]

        # Default to 0 if no configuration available
        num_layers = max(
            len(ee_config), len(ei_config), len(ie_config), len(ii_config), 1
        )

        # Estimate average synapses per layer for each type
        # This is an approximation since we don't have exact layer-wise breakdowns
        # from the mask data. We use the configuration ratios as a guide.
        total_ee_config = sum(ee_config) if ee_config else 0
        total_ei_config = sum(ei_config) if ei_config else 0
        total_ie_config = sum(ie_config) if ie_config else 0
        total_ii_config = sum(ii_config) if ii_config else 0

        total_config = total_ee_config + total_ie_config  # Excitatory connections
        total_inh_config = total_ei_config + total_ii_config  # Inhibitory connections

        # Compute averages per layer based on actual vs configured ratios
        if total_config > 0:
            exc_ratio = total_exc_active / max(total_config, 1)
            ee_avg = (total_ee_config * exc_ratio) / max(num_layers, 1)
            ie_avg = (total_ie_config * exc_ratio) / max(num_layers, 1)
        else:
            ee_avg = ie_avg = 0

        if total_inh_config > 0:
            inh_ratio = total_inh_active / max(total_inh_config, 1)
            ei_avg = (total_ei_config * inh_ratio) / max(num_layers, 1)
            ii_avg = (total_ii_config * inh_ratio) / max(num_layers, 1)
        else:
            ei_avg = ii_avg = 0

        return {"ee": ee_avg, "ei": ei_avg, "ie": ie_avg, "ii": ii_avg}

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
        if not self.enabled or not isinstance(
            getattr(model, "core_network", None), ExcitationInhibitionNetwork
        ):
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
            logger.info(
                f"Analyzing synaptic turnover from {len(self.data_dict_list)} snapshots"
            )

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

            # Initialize per-layer synapse type statistics
            # Collect layer-wise statistics for ee, ei, ie, ii synapses
            layer_stats = {
                "incremental": {
                    "per_layer_avg": {"ee": [], "ei": [], "ie": [], "ii": []}
                },
                "cumulative": {
                    "per_layer_avg": {"ee": [], "ei": [], "ie": [], "ii": []}
                },
            }

            # Get initial snapshot for cumulative comparison
            initial_data_dict = (
                self.data_dict_list[0] if len(self.data_dict_list) > 0 else None
            )

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

                # Initialize per-layer synapse type counters

                # Process each dendritic branch layer
                for name in curr_data_dict.keys():
                    prev_module_dict: dict = prev_data_dict[name]
                    curr_module_dict: dict = curr_data_dict[name]
                    initial_module_dict: dict = (
                        initial_data_dict[name]
                        if initial_data_dict
                        else prev_module_dict
                    )

                    # Process excitatory synapses
                    if "exc_mask" in curr_module_dict.keys():
                        curr_exc_mask = curr_module_dict["exc_mask"]

                        # Incremental comparison (t vs t-1)
                        prev_exc_mask = prev_module_dict["exc_mask"]
                        (
                            inc_exc_active,
                            inc_exc_stable,
                            inc_exc_transient,
                        ) = _update_turnover_counts(
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
                        ) = _update_turnover_counts(
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
                        ) = _update_turnover_counts(
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
                        ) = _update_turnover_counts(
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

                # Compute per-layer averages for synapse types
                inc_layer_avg = self._compute_layer_synapse_averages(
                    inc_exc_active, inc_inh_active
                )
                layer_stats["incremental"]["per_layer_avg"]["ee"].append(
                    inc_layer_avg["ee"]
                )
                layer_stats["incremental"]["per_layer_avg"]["ei"].append(
                    inc_layer_avg["ei"]
                )
                layer_stats["incremental"]["per_layer_avg"]["ie"].append(
                    inc_layer_avg["ie"]
                )
                layer_stats["incremental"]["per_layer_avg"]["ii"].append(
                    inc_layer_avg["ii"]
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

                # Compute cumulative per-layer averages for synapse types
                cum_layer_avg = self._compute_layer_synapse_averages(
                    cum_exc_active, cum_inh_active
                )
                layer_stats["cumulative"]["per_layer_avg"]["ee"].append(
                    cum_layer_avg["ee"]
                )
                layer_stats["cumulative"]["per_layer_avg"]["ei"].append(
                    cum_layer_avg["ei"]
                )
                layer_stats["cumulative"]["per_layer_avg"]["ie"].append(
                    cum_layer_avg["ie"]
                )
                layer_stats["cumulative"]["per_layer_avg"]["ii"].append(
                    cum_layer_avg["ii"]
                )

            # Return both types of turnover statistics
            turnover_dict = {
                "incremental": incremental_turnover,
                "cumulative": cumulative_turnover,
                "layer_statistics": layer_stats,
                "n_snapshots": len(self.data_dict_list),
            }

            if save_path is not None:
                save_dict(turnover_dict, save_path, filename)

                # Generate turnover plots
                self._plot_turnover_results(turnover_dict, save_path, filename)
            else:
                return turnover_dict

    def _plot_turnover_results(self, results: dict, save_path: str, filename: str):
        """Generate plots for synaptic turnover results."""
        try:
            # Generate turnover plots
            plot_synapse_turnover(results, save_path=save_path)
        except Exception as e:
            logger.error(f"Error generating synaptic turnover plots: {e}")
